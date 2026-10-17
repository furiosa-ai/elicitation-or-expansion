"""Answer-audit memory: the model's own track record as the retrieved context.

Every system in this line stores knowledge about the *task* - which cases look
alike, which answer strings exist, how they decompose. None stores knowledge
about the *answerer*. Yet the grader hands back a per-step decomposition of
every attempt (which parts of the answer were right, which were spurious,
which were missing) and no system has ever read it.

Offline replay of the iteration-3 train streams says that signal is sharp and
it is not recoverable from any input-side channel:

  - Realised precision per answer unit spans an order of magnitude. On the
    large label space one frequently-proposed answer belonged in the true
    answer 1 time in 15 while another managed 6 in 7. The low-precision ones
    are generic forms that the task always resolves to something more
    specific; nothing about the query predicts this, only the history of the
    model's own attempts does.
  - Some answers are never the whole story. One unit was part of the true
    answer 20 times and on all 20 the true answer named a further item.
    A global answer-cardinality hint was tried in iteration 3 and did not pay;
    this is the same quantity conditioned on the answer actually being given,
    which turns a 31% base rate into a 100% one.
  - Displacements concentrate. On the small label space, two answers were
    displaced by the same rival every single time they were wrong (4/4, 3/3)
    and a third 5 times in 7 - a directed, recurring confusion that survives
    unchanged across all four systems benchmarked so far, so it is a property
    of the answerer rather than of the memory.

The ledger is rendered as a short, query-INDEPENDENT block: a track record,
not a candidate list. That distinction matters - iteration 3 showed that
query-conditional candidate briefs drag the model toward the modal retrieved
answer and cost accuracy where the model is already better than every
retrieval channel. A track record cannot do that: it names no candidate for
the current problem, it only reports how the model's own habitual answers have
turned out.

Where the grader exposes it, the audit is kept in the grader's own units
rather than the memory's guessed delimiter, so the accounting matches what is
actually being scored.

Everything is gated on measured content: the block is suppressed unless the
answer space repeats, unless enough answers clear a minimum observation count,
and unless realised precision actually spreads across them. On open-ended
generation the ledger stays empty.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Example retrieval
NGRAM_SIZES = (2, 3)
TOP_K = 12
PER_ANSWER_CAP = 2
MAX_EXEMPLAR_CHARS = 9000
# Char budget when the example channel is measurably weak.
WEAK_EXEMPLAR_CHARS = 1800
WEAK_EXEMPLAR_K = 4
GOOD_PRECISION = 0.25

# Answer-space gating
MIN_EXAMPLES_FOR_GATE = 15
MIN_REUSE_RATE = 0.25
MAX_INVENTORY_CHARS = 4000

# Answer-inventory retrieval
SHORTLIST_N = 8
MIN_CALIB_OBS = 12
MIN_EVIDENCE_LIFT = 2.0

# Canonicalisation
SNAP_THRESHOLD = 0.65
SNAP_MARGIN = 0.04
MIN_AFFIX_OBS = 4

SPLIT_PRECEDENCE = [";", "；", "\n", "|", "，", ",", "、"]

# Answer audit
AUDIT_MIN_OBS = 4
AUDIT_MAX_ROWS = 8
AUDIT_MIN_ROWS = 3
AUDIT_MIN_PRECISION_SPREAD = 0.25
AUDIT_MIN_REUSE = 0.15
AUDIT_COMPANION_MIN = 3
AUDIT_COMPANION_RATE = 0.8
AUDIT_DISPLACE_MIN = 2
AUDIT_DISPLACE_RATE = 0.4
AUDIT_CARD_MIN_GAP = 0.15
AUDIT_NEAR_MISS_MIN = 0.35

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{shortlist_section}{inventory_section}{examples_section}{audit_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}{audit_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

INVENTORY_RULE = (
    "- If the correct answer is one of the listed answer strings, reproduce that "
    "string EXACTLY, character for character. Otherwise answer in the same style "
    "and level of detail as those strings.\n"
)

SHORTLIST_RULE = (
    "- The shortlisted answers are the ones whose own wording is echoed by this "
    "problem's text. Treat them as strong candidates, check each against the "
    "problem, and prefer one of them unless the evidence clearly points "
    "elsewhere.\n"
)

AUDIT_RULE = (
    "- The track record describes your own past answers on this task, not this "
    "problem. If the answer you are about to give appears there with a poor "
    "record, re-read the problem for the evidence that would separate it from "
    "the answer that displaced it before committing to it.\n"
)

TERSE_RETRY = """{input}

Give your answer immediately. Do not deliberate and do not explain.
Reply with only this JSON object:

{{"final_answer": "[your answer]"}}"""

_PUNCT = re.compile(r"[\s、。,;；，.\[\]/_\-()（）:：]+")


def _char_ngrams(text: str, sizes=NGRAM_SIZES) -> Counter:
    text = text.lower()
    grams: Counter = Counter()
    for n in sizes:
        for i in range(max(0, len(text) - n + 1)):
            grams[text[i : i + n]] += 1
    return grams


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def _answer_ngrams(text: str, n: int = 2) -> Counter:
    text = _normalize(text)
    grams = Counter(text[i : i + n] for i in range(max(0, len(text) - n + 1)))
    return grams or Counter([text])


def _dice(a: Counter, b: Counter) -> float:
    total = sum(a.values()) + sum(b.values())
    return 2 * sum((a & b).values()) / total if total else 0.0


def _coverage(a: Counter, b: Counter) -> float:
    total = sum(a.values())
    return sum((a & b).values()) / total if total else 0.0


class _Index:
    """Incremental TF-IDF inverted index over char n-grams (script agnostic)."""

    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.df: Counter = Counter()
        self.n_docs = 0

    def add(self, text: str) -> None:
        grams = _char_ngrams(text)
        weights = {g: 1.0 + math.log(c) for g, c in grams.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        doc_id = self.n_docs
        for g, w in weights.items():
            self.postings[g].append((doc_id, w / norm))
            self.df[g] += 1
        self.n_docs += 1

    def search(self, text: str) -> list[tuple[float, int]]:
        if not self.n_docs:
            return []
        grams = _char_ngrams(text)
        scores: dict[int, float] = defaultdict(float)
        for g, c in grams.items():
            postings = self.postings.get(g)
            if not postings:
                continue
            qw = (1.0 + math.log(c)) * math.log(1 + self.n_docs / (1 + self.df[g]))
            for doc_id, dw in postings:
                scores[doc_id] += qw * dw
        return sorted(((s, i) for i, s in scores.items()), key=lambda x: (-x[0], x[1]))


class _Affix:
    """Modal output markup, learned from (prediction, gold) pairs."""

    def __init__(self) -> None:
        self.prefixes: Counter = Counter()
        self.suffixes: Counter = Counter()
        self.n = 0

    def observe(self, prediction: str, gold: str) -> None:
        idx = prediction.find(gold)
        if idx < 0 or len(prediction) == len(gold):
            return
        self.n += 1
        self.prefixes[prediction[:idx]] += 1
        self.suffixes[prediction[idx + len(gold) :]] += 1

    def _modal(self, counts: Counter) -> str:
        if self.n < MIN_AFFIX_OBS or not counts:
            return ""
        text, hits = counts.most_common(1)[0]
        # Only trust markup the model applies most of the time.
        return text if text and hits >= 0.5 * self.n else ""

    def strip(self, text: str) -> str:
        text = text.strip()
        pre, suf = self._modal(self.prefixes), self._modal(self.suffixes)
        if pre and text.startswith(pre):
            text = text[len(pre) :]
        if suf and text.endswith(suf):
            text = text[: -len(suf)]
        return text.strip()


class _Template:
    """The boilerplate shared by every input for a task.

    Inputs arrive as a fixed instruction template wrapped around the part that
    actually varies. That template can be a large share of the characters and,
    when it contains worked format examples, it injects answer strings that have
    nothing to do with the case at hand. It is exactly the longest prefix and
    suffix common to all inputs seen so far, so it can be learned and removed.
    """

    MIN_INPUTS = 5

    def __init__(self) -> None:
        self.prefix: str | None = None
        self.suffix: str | None = None
        self.n = 0

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        i = 0
        limit = min(len(a), len(b))
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    def observe(self, text: str) -> None:
        self.n += 1
        if self.prefix is None:
            self.prefix, self.suffix = text, text
            return
        self.prefix = self._common_prefix(self.prefix, text)
        self.suffix = self._common_prefix(self.suffix[::-1], text[::-1])[::-1]

    def strip(self, text: str) -> str:
        if self.n < self.MIN_INPUTS or self.prefix is None:
            return text
        body = text
        if self.prefix and body.startswith(self.prefix):
            body = body[len(self.prefix) :]
        if self.suffix and body.endswith(self.suffix):
            body = body[: -len(self.suffix)]
        return body.strip() or text


class _Inventory:
    """Distinct atomic answer units plus the delimiter that composes them."""

    def __init__(self) -> None:
        self.units: Counter = Counter()
        self.delim_hits: Counter = Counter()
        self.n_answers = 0
        self.golds: list[str] = []
        self._grams: dict[str, Counter] = {}
        self._unigrams: dict[str, Counter] = {}
        self._built_with: str | None = None

    def delimiter(self) -> str | None:
        for d in SPLIT_PRECEDENCE:
            if self.delim_hits[d] >= max(2, 0.03 * self.n_answers):
                return d
        return None

    def split(self, text: str) -> list[str]:
        d = self.delimiter()
        if d and d in text:
            return [p.strip() for p in text.split(d) if p.strip()]
        return [text.strip()] if text.strip() else []

    def _register(self, unit: str) -> None:
        self.units[unit] += 1
        if unit not in self._grams:
            self._grams[unit] = _answer_ngrams(unit)
            self._unigrams[unit] = _answer_ngrams(unit, 1)

    def observe(self, gold: str) -> bool:
        """Store an answer; returns True if the unit inventory was rebuilt."""
        self.n_answers += 1
        self.golds.append(gold)
        for d in SPLIT_PRECEDENCE:
            if d in gold:
                self.delim_hits[d] += 1
        # The best delimiter is only identifiable once enough answers have been
        # seen; when it changes, every unit split under the old one is wrong.
        current = self.delimiter()
        if current != self._built_with:
            self._built_with = current
            self.units = Counter()
            self._grams = {}
            self._unigrams = {}
            for g in self.golds:
                for unit in self.split(g):
                    self._register(unit)
            return True
        for unit in self.split(gold):
            self._register(unit)
        return False

    def reuse_rate(self) -> float:
        total = sum(self.units.values())
        if not total:
            return 0.0
        singletons = sum(1 for c in self.units.values() if c == 1)
        return 1.0 - singletons / total


class _Calibration:
    """Online utility of each retrieval channel, measured against ground truth."""

    def __init__(self) -> None:
        self.shortlist_hits = 0
        self.neighbour_hits = 0
        self.n = 0

    def observe(self, shortlist_hit: bool, neighbour_hit: bool) -> None:
        self.shortlist_hits += int(shortlist_hit)
        self.neighbour_hits += int(neighbour_hit)
        self.n += 1

    def evidence_lift(self, inventory_size: int) -> float:
        """Shortlist hit rate divided by the rate a random shortlist would get."""
        if self.n < MIN_CALIB_OBS or inventory_size <= SHORTLIST_N:
            return 0.0
        chance = SHORTLIST_N / inventory_size
        return (self.shortlist_hits / self.n) / chance if chance else 0.0

    def neighbour_precision(self) -> float:
        return self.neighbour_hits / self.n if self.n else 1.0



class _Audit:
    """Per-answer-unit track record, accumulated from graded feedback.

    Three quantities that only the answering history can supply, all keyed on
    an answer unit the model actually proposes:

      realised precision  how often proposing it was right,
      companion rate      how often being right still left the answer
                          incomplete,
      displacement        which answer took its place when it was wrong.

    Where the grader reports its own true/false positive and negative counts
    those are used for the cardinality reconciliation, so the accounting is in
    the units that are actually scored rather than in the delimiter this
    memory happened to infer.
    """

    def __init__(self) -> None:
        self.proposed: Counter = Counter()
        self.true_pos: Counter = Counter()
        self.in_gold: Counter = Counter()
        self.gold_multi: Counter = Counter()
        self.displaced: dict[str, Counter] = defaultdict(Counter)
        self.pred_units = 0
        self.gold_units = 0
        self.graded_fp = 0
        self.graded_fn = 0
        self.graded_steps = 0
        self.near_miss_total = 0.0
        self.near_miss_n = 0
        self.n = 0

    @staticmethod
    def _partial_credit(metrics: dict[str, Any] | None) -> float | None:
        """A grader-supplied score that is neither pass nor fail.

        Graders expose several such numbers under names this system cannot
        assume (a precision and a recall bracket the combined score between
        them). The median of them is taken rather than the first, which would
        depend on dict order and could report the most flattering of them.

        Fractional scores are distinguished from the tallies that sit beside
        them in the same dict by type: a count is an integer, a score is a
        fraction of one. Zero scores are kept - dropping them would make the
        statistic read as "how close were the answers that were already close",
        which is not what it is used to say.
        """
        if not metrics:
            return None
        scores = sorted(
            v
            for v in metrics.values()
            if isinstance(v, float) and not isinstance(v, bool) and 0.0 <= v <= 1.0
        )
        if not scores:
            return None
        return scores[len(scores) // 2]

    def observe(
        self,
        pred_units: set[str],
        gold_units: set[str],
        metrics: dict[str, Any] | None,
        was_correct: bool,
    ) -> None:
        self.n += 1
        for unit in pred_units:
            self.proposed[unit] += 1
        hit = pred_units & gold_units
        for unit in hit:
            self.true_pos[unit] += 1
        for unit in gold_units:
            self.in_gold[unit] += 1
            if len(gold_units) > 1:
                self.gold_multi[unit] += 1
        # What the answer held instead, for each unit that did not belong.
        for unit in pred_units - gold_units:
            for other in gold_units - pred_units:
                self.displaced[unit][other] += 1

        self.pred_units += len(pred_units)
        self.gold_units += len(gold_units)
        fp = metrics.get("fp") if metrics else None
        fn = metrics.get("fn") if metrics else None
        if isinstance(fp, (int, float)) and isinstance(fn, (int, float)):
            self.graded_fp += int(fp)
            self.graded_fn += int(fn)
            self.graded_steps += 1
        if not was_correct:
            credit = self._partial_credit(metrics)
            if credit is not None:
                self.near_miss_total += credit
                self.near_miss_n += 1

    # ---------------- persistence ----------------

    _COUNTERS = ("proposed", "true_pos", "in_gold", "gold_multi")
    _SCALARS = (
        "pred_units",
        "gold_units",
        "graded_fp",
        "graded_fn",
        "graded_steps",
        "near_miss_total",
        "near_miss_n",
        "n",
    )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: dict(getattr(self, k)) for k in self._COUNTERS}
        out.update({k: getattr(self, k) for k in self._SCALARS})
        out["displaced"] = {k: dict(v) for k, v in self.displaced.items()}
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Audit":
        audit = cls()
        for key in cls._COUNTERS:
            setattr(audit, key, Counter(data.get(key, {})))
        for key in cls._SCALARS:
            setattr(audit, key, data.get(key, 0))
        audit.displaced = defaultdict(Counter)
        for key, value in data.get("displaced", {}).items():
            audit.displaced[key] = Counter(value)
        return audit

    # ---------------- rendering ----------------

    def _rows(self) -> list[tuple[int, str]]:
        rows = [(c, u) for u, c in self.proposed.items() if c >= AUDIT_MIN_OBS]
        rows.sort(key=lambda r: (-r[0], r[1]))
        return rows[:AUDIT_MAX_ROWS]

    def _line(self, unit: str, count: int) -> str:
        line = (
            f'- "{unit}" - proposed {count}x, belonged in the true answer '
            f"{self.true_pos[unit]}x"
        )
        seen = self.in_gold.get(unit, 0)
        if (
            seen >= AUDIT_COMPANION_MIN
            and self.gold_multi[unit] / seen >= AUDIT_COMPANION_RATE
        ):
            line += (
                f"; every time it belonged, the true answer named at least one "
                f"further item ({self.gold_multi[unit]}/{seen})"
            )
        misses = self.displaced.get(unit)
        if misses:
            total = sum(misses.values())
            rival, hits = misses.most_common(1)[0]
            if hits >= AUDIT_DISPLACE_MIN and hits / total >= AUDIT_DISPLACE_RATE:
                line += f'; when it was wrong the true answer held "{rival}" ({hits}/{total})'
        return line

    def _cardinality_line(self) -> str | None:
        """Reconciliation of answer size, in the grader's units where available."""
        if self.graded_steps >= MIN_CALIB_OBS and (self.graded_fp or self.graded_fn):
            if self.graded_fn > self.graded_fp * 1.2:
                return (
                    f"- Across {self.graded_steps} graded answers you left out "
                    f"{self.graded_fn} required items and added {self.graded_fp} "
                    f"spurious ones: you name too few items more often than too many."
                )
            if self.graded_fp > self.graded_fn * 1.2:
                return (
                    f"- Across {self.graded_steps} graded answers you added "
                    f"{self.graded_fp} spurious items and left out {self.graded_fn}: "
                    f"you name too many items more often than too few."
                )
            return None
        if not self.n:
            return None
        mean_pred = self.pred_units / self.n
        mean_gold = self.gold_units / self.n
        if abs(mean_pred - mean_gold) < AUDIT_CARD_MIN_GAP:
            return None
        direction = "fewer" if mean_pred < mean_gold else "more"
        return (
            f"- You name {mean_pred:.2f} items per answer on average; true answers "
            f"average {mean_gold:.2f} - {direction} than you give."
        )

    def _near_miss_line(self) -> str | None:
        """Only worth saying when wrong answers are usually partly right."""
        if self.near_miss_n < MIN_CALIB_OBS:
            return None
        mean = self.near_miss_total / self.near_miss_n
        if mean < AUDIT_NEAR_MISS_MIN:
            return None
        return (
            f"- Your wrong answers score {mean:.2f} against the true answer on "
            f"average: they are usually partly right, so refine rather than "
            f"replace when you are unsure."
        )

    def render(self, reuse_rate: float) -> str:
        if reuse_rate < AUDIT_MIN_REUSE:
            return ""
        rows = self._rows()
        if len(rows) < AUDIT_MIN_ROWS:
            return ""
        precisions = [self.true_pos[u] / c for c, u in rows]
        # A ledger where every answer has the same record separates nothing.
        if max(precisions) - min(precisions) < AUDIT_MIN_PRECISION_SPREAD:
            return ""
        lines = [self._line(u, c) for c, u in rows]
        for extra in (self._cardinality_line(), self._near_miss_line()):
            if extra:
                lines.append(extra)
        return (
            "**Track record of your own most frequent answers on this task:**\n"
            + "\n".join(lines)
            + "\n\n"
        )


class SelfAuditMemory(MemorySystem):
    """Task memory plus a ledger of how the model's own answers have turned out."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._template = _Template()
        self._audit = _Audit()
        self._query_df: Counter = Counter()
        self._n_queries = 0

    # ---------------- gating ----------------

    def _inventory_enabled(self) -> bool:
        return (
            len(self.examples) >= MIN_EXAMPLES_FOR_GATE
            and self._inventory.reuse_rate() >= MIN_REUSE_RATE
        )

    def _shortlist_enabled(self) -> bool:
        if not self._inventory_enabled():
            return False
        return (
            self._calib.evidence_lift(len(self._inventory.units)) >= MIN_EVIDENCE_LIFT
        )

    # ---------------- answer-inventory retrieval ----------------

    def _evidence_rank(self, query: str) -> list[tuple[float, str]]:
        """Rank known answers by how strongly their wording is attested in the query."""
        q = query.lower()
        q_bigrams = {q[i : i + 2] for i in range(len(q) - 1)}
        n = max(1, self._n_queries)
        scored: list[tuple[float, str]] = []
        for unit in self._inventory.units:
            u = unit.lower()
            bigrams = [u[i : i + 2] for i in range(len(u) - 1)]
            num = den = 0.0
            for g in bigrams:
                w = math.log(1 + n / (1 + self._query_df[g]))
                den += w
                if g in q_bigrams:
                    num += w
            attested = num / den if den else 0.0
            # Longest contiguous piece of the answer that appears verbatim.
            longest = 0
            for i in range(len(u)):
                for j in range(len(u), i + longest, -1):
                    if u[i:j] in q:
                        longest = j - i
                        break
            run = longest / len(u) if u else 0.0
            scored.append((0.55 * attested + 0.45 * run, unit))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def _shortlist(self, query: str) -> list[str]:
        if not self._inventory.units:
            return []
        return [u for _s, u in self._evidence_rank(query)[:SHORTLIST_N]]

    # ---------------- example retrieval ----------------

    def _exemplar_budget(self) -> tuple[int, int]:
        """(max examples, max chars), scaled by the example channel's measured precision.

        Only meaningful where answers repeat: on open-ended generation an exact
        answer match is near-impossible however useful the neighbours are, so the
        precision signal there says nothing and the full budget is kept.
        """
        if (
            self._inventory_enabled()
            and self._calib.n >= MIN_CALIB_OBS
            and self._calib.neighbour_precision() < GOOD_PRECISION
        ):
            return WEAK_EXEMPLAR_K, WEAK_EXEMPLAR_CHARS
        return TOP_K, MAX_EXEMPLAR_CHARS

    def _retrieve(self, query: str) -> list[dict[str, str]]:
        top_k, _ = self._exemplar_budget()
        used: Counter = Counter()
        picked: list[dict[str, str]] = []
        for _score, doc_id in self._index.search(query):
            ex = self.examples[doc_id]
            if used[ex["target"]] >= PER_ANSWER_CAP:
                continue
            picked.append(ex)
            used[ex["target"]] += 1
            if len(picked) >= top_k:
                break
        return picked

    # ---------------- prompt sections ----------------

    def _format_examples(self, picked: list[dict[str, str]]) -> str:
        if not picked:
            return ""
        _, max_chars = self._exemplar_budget()
        parts, total = [], 0
        # Most similar case last: it sits closest to the problem in the prompt.
        for ex in reversed(picked):
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total + len(part) > max_chars:
                continue
            parts.append(part)
            total += len(part) + 2
        if not parts:
            return ""
        return "**Similar solved cases:**\n\n" + "\n\n".join(parts) + "\n\n"

    def _format_inventory(self, exclude: list[str]) -> str:
        if not self._inventory_enabled():
            return ""
        skip = set(exclude)
        parts, total = [], 0
        for answer, _c in self._inventory.units.most_common():
            if answer in skip:
                continue
            line = f"- {answer}"
            if total + len(line) > MAX_INVENTORY_CHARS:
                break
            parts.append(line)
            total += len(line) + 1
        if not parts:
            return ""
        header = (
            "**Other answer strings used in previously solved cases:**\n"
            if skip
            else "**Answer strings used in previously solved cases:**\n"
        )
        return header + "\n".join(parts) + "\n\n"

    def _format_shortlist(self, shortlist: list[str]) -> str:
        if not shortlist:
            return ""
        lines = [f"- {u}" for u in shortlist]
        return (
            "**Shortlisted answers (their wording is echoed by this problem):**\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # ---------------- audit ----------------

    def _answer_units(self, text: str) -> set[str]:
        """Split an answer the way gold answers are split, markup removed."""
        body = self._affix.strip(text) or text
        return {u for u in self._inventory.split(body) if u}

    def _format_audit(self) -> str:
        return self._audit.render(self._inventory.reuse_rate())

    # ---------------- canonicalisation ----------------

    def _snap_unit(self, unit: str, targets: list[str]) -> str:
        if unit in self._inventory.units:
            return unit
        bigrams = _answer_ngrams(unit)
        unigrams = _answer_ngrams(unit, 1)
        scored = []
        for t in targets:
            t_uni = self._inventory._unigrams[t]
            # Symmetric agreement, plus asymmetric evidence that the answer is a
            # sloppy rendering of this known one.
            sym = 0.5 * _dice(unigrams, t_uni) + 0.5 * _dice(
                bigrams, self._inventory._grams[t]
            )
            scored.append((0.5 * sym + 0.5 * _coverage(unigrams, t_uni), t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored or scored[0][0] < SNAP_THRESHOLD:
            return unit
        if len(scored) > 1 and scored[0][0] - scored[1][0] < SNAP_MARGIN:
            return unit
        return scored[0][1]

    def _canonicalize(self, answer: str) -> str:
        """Strip learned markup, then snap each answer unit onto the inventory."""
        if not answer.strip() or not self._inventory_enabled():
            return answer
        body = self._affix.strip(answer)
        if not body:
            return answer
        targets = list(self._inventory.units)
        snapped = [self._snap_unit(u, targets) for u in self._inventory.split(body)]
        if not snapped:
            return answer
        joined = (self._inventory.delimiter() or ";").join(dict.fromkeys(snapped))
        if joined == body:
            return answer
        return answer.replace(body, joined) if body in answer else joined

    # ---------------- interface ----------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        query = self._template.strip(input)
        picked = self._retrieve(query)
        shortlist = self._shortlist(query) if self._shortlist_enabled() else []
        audit_section = self._format_audit()
        prompt = PROMPT.format(
            shortlist_section=self._format_shortlist(shortlist),
            inventory_section=self._format_inventory(shortlist),
            examples_section=self._format_examples(picked),
            audit_section=audit_section,
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            shortlist_rule=SHORTLIST_RULE if shortlist else "",
            audit_rule=AUDIT_RULE if audit_section else "",
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        # An unparseable / truncated response is a guaranteed miss; one terse
        # retry converts it into an actual attempt.
        retried = False
        if not answer.strip():
            retried = True
            response = self.call_llm(TERSE_RETRY.format(input=input))
            answer = extract_json_field(response, "final_answer")
            if not answer.strip():
                answer = response.strip()[:200]

        canonical = self._canonicalize(answer)
        return canonical, {
            "full_response": response,
            "num_retrieved": len(picked),
            "shortlist": shortlist,
            "inventory_on": self._inventory_enabled(),
            "snapped": canonical != answer,
            "audit_on": bool(audit_section),
            "retried": retried,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            question = r.get("raw_question", r["input"])

            # Score both retrieval channels against this answer before storing it,
            # so the calibration is always measured on unseen cases.
            if self._inventory.units:
                gold_units = self._inventory.split(target)
                shortlist = self._shortlist(question)
                neighbours = self._index.search(question)[:1]
                nearest = (
                    self.examples[neighbours[0][1]]["target"] if neighbours else None
                )
                self._calib.observe(
                    any(u in shortlist for u in gold_units), nearest == target
                )

            self._affix.observe(r.get("prediction", ""), target)

            # The audit is measured on this attempt before the case is stored,
            # so it always reports performance on cases not yet in memory.
            prediction = r.get("prediction", "")
            if prediction:
                self._audit.observe(
                    self._answer_units(prediction),
                    self._answer_units(target),
                    r.get("metrics"),
                    bool(r.get("was_correct")),
                )

            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            ex["was_correct"] = bool(r.get("was_correct"))
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._index.add(question)
            self._inventory.observe(target)
            q = question.lower()
            for g in {q[i : i + 2] for i in range(len(q) - 1)}:
                self._query_df[g] += 1
            self._n_queries += 1

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self._template.strip(self.examples[-1]["input"])
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        return (
            len(self._format_shortlist(shortlist))
            + len(self._format_inventory(shortlist))
            + len(self._format_examples(self._retrieve(probe)))
            + len(self._format_audit())
        )

    def get_state(self) -> str:
        # The ledger is carried explicitly: it is measured against the grader's
        # own report, which a replay of the stored examples cannot reconstruct.
        return json.dumps(
            {"examples": self.examples, "audit": self._audit.to_dict()}, indent=2
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._template = _Template()
        self._audit = _Audit()
        self._query_df = Counter()
        self._n_queries = 0
        for ex in data.get("examples", []):
            record = {
                "input": ex["input"],
                "ground_truth": ex["target"],
                "prediction": ex.get("prediction", ""),
                "was_correct": ex.get("was_correct", False),
            }
            if "raw_question" in ex:
                record["raw_question"] = ex["raw_question"]
            self.learn_from_batch([record])
        audit = data.get("audit")
        if audit:
            self._audit = _Audit.from_dict(audit)
