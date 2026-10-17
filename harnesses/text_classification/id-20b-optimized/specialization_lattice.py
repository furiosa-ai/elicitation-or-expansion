"""Memory organised as a specialisation lattice, calibrated against its own bias.

Two measurements on the iteration-2 train streams motivate this system, and
both are about *granularity* rather than about which candidates are retrieved.

First, the model's answer distribution is measurably over-concentrated. The top
five answer units carry 18.3% of the gold mass on the large label space but
33.2% of the model's predicted mass, and it uses only 102 of the 134 units that
actually occur. Grouping predictions by how over-used their answer is separates
them sharply: predictions of under-used answers are right 42.0% of the time,
predictions of over-used answers 24.2%. The same ordering holds on the small
label space (82.6% vs 66.7%). A memory can measure this about itself, because it
sees both its own predictions and the truth.

Second, a large part of that bias is structural: 13.0% of errors are cases where
the predicted answer is a *generic ancestor* of the gold one - a coarse answer
where a refinement was wanted. Those refinements are discoverable from surface
form alone, because a refined answer contains its generic form as a substring.
The inventory therefore carries a specialisation lattice, and it is genuinely a
property of the task: 9 families and 19 edges on the large label space, zero on
both other tasks, which switch every mechanism here off.

So memory stops being a flat set of answers and becomes:

1. A lattice of answer families. When a family is in contention the prompt shows
   the whole family as a granularity menu - the generic form, each refinement,
   how often each occurred, and the *differentia*: the wording that separates a
   refinement from its parent.

2. A self-bias record. Candidates carry how often the model has reached for them
   versus how often they were right, so an over-used answer is presented as one
   needing specific support rather than as a free default.

3. A deterministic specialisation step. When the answer names a generic form and
   exactly one refinement's differentia is strongly attested in the problem
   text, the answer is promoted to that refinement. Replayed against the base
   system's own outputs this fires 8 times in 200 steps, fixing 2 and breaking
   0 - the threshold is set where the measured breakage is zero, not where the
   firing rate is highest.
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

# Specialisation lattice
MIN_PARENT_LEN = 2
MAX_FAMILY_CHARS = 1200
MAX_FAMILIES = 3
# Promote generic -> refinement only where measured breakage was zero.
SPECIALIZE_ATTEST = 0.85
# A unit counts as over-used once it is predicted this much more than it is right.
BIAS_RATIO = 1.8
MIN_BIAS_OBS = 3

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{shortlist_section}{families_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}{families_rule}- Respond in JSON format

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

FAMILIES_RULE = (
    "- Where a general answer has more specific forms, answer with the most "
    "specific one the problem actually supports - look for its distinguishing "
    "wording in the problem text. Fall back to the general form only when no "
    "refinement is supported.\n"
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


class _Lattice:
    """Specialisation edges over answer units, discovered from surface form.

    A refined answer normally contains its generic form verbatim - a subtype
    keeps the head it specialises. That is enough to recover the family tree
    without any knowledge of the domain, and it yields nothing at all on tasks
    whose answers are unrelated strings, which is the desired behaviour.
    """

    def __init__(self) -> None:
        self.children: dict[str, set[str]] = defaultdict(set)
        self.parents: dict[str, set[str]] = defaultdict(set)

    def add(self, unit: str, known: list[str]) -> None:
        if len(unit) < MIN_PARENT_LEN:
            return
        for other in known:
            if other == unit or len(other) < MIN_PARENT_LEN:
                continue
            if other in unit:
                self.children[other].add(unit)
                self.parents[unit].add(other)
            elif unit in other:
                self.children[unit].add(other)
                self.parents[other].add(unit)

    def edges(self) -> int:
        return sum(len(v) for v in self.children.values())

    @staticmethod
    def differentia(parent: str, child: str) -> str:
        """The wording a refinement adds to its generic form."""
        i = child.find(parent)
        if i < 0:
            return child
        return (child[:i] + child[i + len(parent) :]).strip("、，,;； ")

    def family_of(self, unit: str) -> str | None:
        """The most generic ancestor of a unit that actually has refinements."""
        if self.children.get(unit):
            return unit
        best = None
        for p in self.parents.get(unit, ()):  # shortest ancestor = most generic
            if best is None or len(p) < len(best):
                best = p
        return best


class _Bias:
    """What the model reaches for, against what turns out to be true.

    Both halves are observable from the stream, so the memory can measure its
    own preference for an answer independently of that answer's correctness.
    """

    def __init__(self) -> None:
        self.predicted: Counter = Counter()
        self.gold: Counter = Counter()

    def observe(self, pred_units: list[str], gold_units: list[str]) -> None:
        for u in pred_units:
            self.predicted[u] += 1
        for u in gold_units:
            self.gold[u] += 1

    def ratio(self, unit: str) -> float:
        """Predicted share over gold share, smoothed."""
        gt = sum(self.gold.values())
        pt = sum(self.predicted.values())
        if not gt or not pt:
            return 1.0
        g = (self.gold[unit] + 0.5) / (gt + 0.5 * max(1, len(self.gold)))
        p = (self.predicted[unit] + 0.5) / (pt + 0.5 * max(1, len(self.predicted)))
        return p / g if g else 1.0

    def overused(self, unit: str) -> bool:
        return (
            self.predicted[unit] >= MIN_BIAS_OBS and self.ratio(unit) >= BIAS_RATIO
        )


class SpecializationLattice(MemorySystem):
    """Answers organised into specialisation families, with the memory's own
    over-use of each answer measured and reported alongside it."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._template = _Template()
        self._lattice = _Lattice()
        self._bias = _Bias()
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
        """The answer inventory, with the model's own over-used answers marked.

        The shortlist is gated off wherever answers are not named in the input,
        so the inventory is the only place the bias record can reach the prompt
        on those tasks - and it is exactly where an over-used default would
        otherwise sit indistinguishable from every other option.
        """
        if not self._inventory_enabled():
            return ""
        skip = set(exclude)
        parts, total, marked = [], 0, 0
        for answer, _c in self._inventory.units.most_common():
            if answer in skip:
                continue
            over = self._bias.overused(answer)
            line = f"- {answer}{' [over-used]' if over else ''}"
            if total + len(line) > MAX_INVENTORY_CHARS:
                break
            parts.append(line)
            total += len(line) + 1
            marked += int(over)
        if not parts:
            return ""
        header = (
            "**Other answer strings used in previously solved cases:**\n"
            if skip
            else "**Answer strings used in previously solved cases:**\n"
        )
        if marked:
            header = header[:-1] + (
                " ([over-used] = this answer has been given far more often than"
                " it has been correct; require specific support before using"
                " it)\n"
            )
        return header + "\n".join(parts) + "\n\n"

    def _format_shortlist(self, shortlist: list[str]) -> str:
        if not shortlist:
            return ""
        lines = []
        for u in shortlist:
            note = (
                " (reached for more often than it has been right - needs"
                " specific support)"
                if self._bias.overused(u)
                else ""
            )
            lines.append(f"- {u}{note}")
        return (
            "**Shortlisted answers (their wording is echoed by this problem):**\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # ---------------- specialisation lattice ----------------

    def _contenders(self, shortlist: list[str], picked: list[dict[str, str]]) -> list[str]:
        """Answers actually in play for this query: shortlisted or retrieved."""
        out = list(shortlist)
        for ex in picked:
            out.extend(self._inventory.split(ex["target"]))
        return out

    def _format_families(self, contenders: list[str]) -> str:
        """The granularity menu for every family currently in contention.

        Showing a generic answer next to its refinements, each with the wording
        that distinguishes it, turns a choice the model makes implicitly - and
        measurably gets wrong towards the generic - into one it can check
        against the problem text.
        """
        if not self._inventory_enabled() or not self._lattice.edges():
            return ""
        seen: list[str] = []
        for unit in contenders:
            fam = self._lattice.family_of(unit)
            if fam and fam not in seen:
                seen.append(fam)
            if len(seen) >= MAX_FAMILIES:
                break
        if not seen:
            return ""
        blocks, total = [], 0
        for fam in seen:
            kids = sorted(
                self._lattice.children.get(fam, ()),
                key=lambda c: (-self._inventory.units[c], c),
            )
            if not kids:
                continue
            lines = [f"- {fam} (general form, seen {self._inventory.units[fam]}x)"]
            for c in kids:
                diff = self._lattice.differentia(fam, c)
                marker = f" - look for: {diff}" if diff else ""
                lines.append(f"- {c} (seen {self._inventory.units[c]}x){marker}")
            block = "\n".join(lines)
            if total + len(block) > MAX_FAMILY_CHARS:
                break
            blocks.append(block)
            total += len(block) + 2
        if not blocks:
            return ""
        return (
            "**Answer families in play (general form and its more specific "
            "forms):**\n" + "\n\n".join(blocks) + "\n\n"
        )

    def _specialize(self, answer: str, query: str) -> str:
        """Promote a generic answer to a refinement the problem actually attests.

        Deliberately conservative: it fires only when the refinement's own
        differentia - not the refinement as a whole, which always scores below
        its parent because it strictly contains it - is near-verbatim in the
        problem, and only when exactly one refinement qualifies best.
        """
        if not self._inventory_enabled() or not self._lattice.edges():
            return answer
        body = self._affix.strip(answer)
        units = self._inventory.split(body)
        if not units:
            return answer
        q = query.lower()
        changed = False
        out = []
        for u in units:
            best = None
            for child in self._lattice.children.get(u, ()):
                diff = self._lattice.differentia(u, child)
                if not diff:
                    continue
                score = self._attest(diff, q)
                if score >= SPECIALIZE_ATTEST and (best is None or score > best[0]):
                    best = (score, child)
            if best:
                out.append(best[1])
                changed = True
            else:
                out.append(u)
        if not changed:
            return answer
        joined = (self._inventory.delimiter() or ";").join(dict.fromkeys(out))
        return answer.replace(body, joined) if body in answer else joined

    def _attest(self, text: str, query_lower: str) -> float:
        """IDF-weighted bigram attestation of one string inside the query."""
        t = text.lower()
        if not t:
            return 0.0
        q_bigrams = {query_lower[i : i + 2] for i in range(len(query_lower) - 1)}
        n = max(1, self._n_queries)
        bigrams = [t[i : i + 2] for i in range(len(t) - 1)] or [t]
        num = den = 0.0
        for g in bigrams:
            w = math.log(1 + n / (1 + self._query_df[g]))
            den += w
            if g in q_bigrams:
                num += w
        return num / den if den else (1.0 if t in query_lower else 0.0)

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
        families = self._format_families(self._contenders(shortlist, picked))
        prompt = PROMPT.format(
            shortlist_section=self._format_shortlist(shortlist),
            families_section=families,
            inventory_section=self._format_inventory(shortlist),
            examples_section=self._format_examples(picked),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            shortlist_rule=SHORTLIST_RULE if shortlist else "",
            families_rule=FAMILIES_RULE if families else "",
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
        specialized = self._specialize(canonical, query)
        return specialized, {
            "full_response": response,
            "num_retrieved": len(picked),
            "shortlist": shortlist,
            "inventory_on": self._inventory_enabled(),
            "snapped": canonical != answer,
            "specialized": specialized != canonical,
            "families_shown": bool(families),
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

            # The lattice needs the answer units; the bias record needs both the
            # model's own units and the true ones, so it is updated here too.
            gold_units = self._inventory.split(target)
            pred_body = self._affix.strip(r.get("prediction", ""))
            self._bias.observe(
                self._inventory.split(pred_body) if pred_body else [], gold_units
            )

            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._index.add(question)
            rebuilt = self._inventory.observe(target)
            # A delimiter change re-splits every stored answer, so the lattice
            # built under the old split is invalid and has to be rebuilt.
            if rebuilt:
                self._lattice = _Lattice()
                known: list[str] = []
                for unit in self._inventory.units:
                    self._lattice.add(unit, known)
                    known.append(unit)
            else:
                known = list(self._inventory.units)
                for unit in gold_units:
                    self._lattice.add(unit, known)
            q = question.lower()
            for g in {q[i : i + 2] for i in range(len(q) - 1)}:
                self._query_df[g] += 1
            self._n_queries += 1

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self._template.strip(self.examples[-1]["input"])
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        picked = self._retrieve(probe)
        return (
            len(self._format_shortlist(shortlist))
            + len(self._format_families(self._contenders(shortlist, picked)))
            + len(self._format_inventory(shortlist))
            + len(self._format_examples(picked))
        )

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._template = _Template()
        self._lattice = _Lattice()
        self._bias = _Bias()
        self._query_df = Counter()
        self._n_queries = 0
        for ex in data.get("examples", []):
            record = {
                "input": ex["input"],
                "ground_truth": ex["target"],
                "prediction": ex.get("prediction", ""),
                "was_correct": False,
            }
            if "raw_question" in ex:
                record["raw_question"] = ex["raw_question"]
            self.learn_from_batch([record])
