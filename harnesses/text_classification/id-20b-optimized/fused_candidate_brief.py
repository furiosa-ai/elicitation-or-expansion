"""Candidate briefs built by fusing three evidence channels with online-fit weights.

The previous system in this line ranked candidate answers by a single channel -
how strongly an answer's own wording is attested in the query - and gated that
shortlist on/off with a lift test. Replaying the iteration-2 train streams shows
the cost of a single channel: attestation puts the gold answer at rank 1 on 2.8%
of steps for a large label space and 3.9% for a small one, so the gate switches
the entire candidate section off wherever answers are not named in the input,
and the model is left with exemplars alone.

There is a second, much stronger channel that no prior system computed: the
*density of the neighbourhood*. A rank-decayed vote over the 30 nearest solved
cases puts the gold answer at rank 1 on 64.4% of steps on the small label space.
It is not a substitute - on the large label space it manages 14.4% where
attestation reaches recall@8 of 42.2% - so neither channel dominates.

This system therefore:

1. Scores every known answer unit on three channels - neighbourhood vote mass,
   query attestation, and prior frequency - instead of one.

2. Fits the channel weights ONLINE by follow-the-leader over a small grid, so
   each task discovers its own mix from arriving ground truth rather than being
   switched on or off. Offline this recovers nearly all of the oracle weighting
   (recall@1 0.633 vs 0.644 where voting should dominate; 0.183 vs 0.211 where
   the mix matters, against 0.144 and 0.028 for the best single channel).

3. Presents the result as a candidate-centric brief - each candidate carries the
   evidence that put it there - rather than a bare list, and states the ranker's
   own measured reliability so the model can weigh it instead of obeying it.

4. Reports the measured answer-cardinality profile. Where answers may list
   several units the model systematically under-fires: it emits a multi-unit
   answer on 15.5% of steps against a 31.5% base rate, at 0.81 precision when
   it does, so the miss is recall, not discrimination.

Every part is gated on measured statistics, so an open-ended generation task -
where the ranker's recall@1 is 0.000 - sees none of it.
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

# Candidate brief
BRIEF_N = 8
MIN_CALIB_OBS = 12
MIN_EVIDENCE_LIFT = 2.0
# Neighbourhood vote: wide and uncapped, so answer mass is visible.
VOTE_POOL = 30
# Channel weight grid searched online.
WEIGHT_GRID = tuple(
    (wv, wa, wp)
    for wv in (0.0, 0.25, 0.5, 0.75, 1.0)
    for wa in (0.0, 0.25, 0.5, 0.75, 1.0)
    for wp in (0.0, 0.1, 0.25)
    if wv + wa + wp > 0
)
# Credit for a rank-1 hit vs. a hit anywhere in the brief, when scoring weights.
TOP_HIT_CREDIT = 0.3
# Cardinality profile is only worth space once multi-unit answers are common.
MIN_MULTI_RATE = 0.10

# Canonicalisation
SNAP_THRESHOLD = 0.65
SNAP_MARGIN = 0.04
MIN_AFFIX_OBS = 4

SPLIT_PRECEDENCE = [";", "；", "\n", "|", "，", ",", "、"]

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{brief_section}{cardinality_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{brief_rule}{cardinality_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

INVENTORY_RULE = (
    "- If the correct answer is one of the listed answer strings, reproduce that "
    "string EXACTLY, character for character. Otherwise answer in the same style "
    "and level of detail as those strings.\n"
)

BRIEF_RULE = (
    "- The ranked candidates come from memory, not from reading this problem. "
    "Check each one against the problem's own evidence and pick the best fit; "
    "the ranking is a prior, not a verdict, so answer outside it when the "
    "problem clearly points elsewhere.\n"
)

CARDINALITY_RULE = (
    "- Answers here often list more than one item. List EVERY item the problem "
    "supports, joined by the delimiter shown above, not just the single best "
    "one.\n"
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


class _Fusion:
    """Online-fit fusion of the candidate-ranking channels.

    Each channel produces a score in [0, 1] for every known answer unit. Their
    relative usefulness is a property of the task, not something that can be
    assumed, so every weight vector on a small grid is scored against arriving
    ground truth and the current leader is the one used. This subsumes the
    binary on/off gate it replaces: a channel that carries nothing simply ends
    up with weight zero, and a task where no channel carries anything is caught
    by the lift test below.
    """

    def __init__(self) -> None:
        self.scores = [0.0] * len(WEIGHT_GRID)
        self.n = 0
        self.top_hits = 0
        self.brief_hits = 0
        self.neighbour_hits = 0

    def leader(self) -> tuple[float, float, float]:
        if not self.n:
            return (1.0, 0.0, 0.0)
        best = max(range(len(WEIGHT_GRID)), key=lambda k: (self.scores[k], -k))
        return WEIGHT_GRID[best]

    @staticmethod
    def _rank(feats: dict[str, tuple[float, float, float]], w) -> list[str]:
        wv, wa, wp = w
        fused = {
            u: wv * f[0] + wa * f[1] + wp * f[2] for u, f in feats.items()
        }
        return sorted(fused, key=lambda u: (-fused[u], u))

    def observe(
        self,
        feats: dict[str, tuple[float, float, float]],
        gold_units: set[str],
        neighbour_hit: bool,
    ) -> None:
        """Score every weight vector against one case, before it is stored."""
        if not feats:
            return
        self.n += 1
        self.neighbour_hits += int(neighbour_hit)
        for k, w in enumerate(WEIGHT_GRID):
            order = self._rank(feats, w)
            hit_top = order[0] in gold_units
            hit_brief = bool(set(order[:BRIEF_N]) & gold_units)
            self.scores[k] += float(hit_top) + TOP_HIT_CREDIT * float(hit_brief)
        order = self._rank(feats, self.leader())
        self.top_hits += int(order[0] in gold_units)
        self.brief_hits += int(bool(set(order[:BRIEF_N]) & gold_units))

    def brief_lift(self, inventory_size: int) -> float:
        """Brief hit rate over what a random brief of the same size would get."""
        if self.n < MIN_CALIB_OBS or inventory_size <= BRIEF_N:
            return 0.0
        chance = BRIEF_N / inventory_size
        return (self.brief_hits / self.n) / chance if chance else 0.0

    def top_precision(self) -> float:
        return self.top_hits / self.n if self.n else 0.0

    def brief_recall(self) -> float:
        return self.brief_hits / self.n if self.n else 0.0

    def neighbour_precision(self) -> float:
        return self.neighbour_hits / self.n if self.n else 1.0


class FusedCandidateBrief(MemorySystem):
    """Three evidence channels fused with online-fit weights into one candidate
    brief, alongside exemplars and a measured answer-cardinality profile."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._fusion = _Fusion()
        self._template = _Template()
        self._query_df: Counter = Counter()
        self._n_queries = 0
        self._cardinality: Counter = Counter()

    # ---------------- gating ----------------

    def _inventory_enabled(self) -> bool:
        return (
            len(self.examples) >= MIN_EXAMPLES_FOR_GATE
            and self._inventory.reuse_rate() >= MIN_REUSE_RATE
        )

    def _brief_enabled(self) -> bool:
        if not self._inventory_enabled():
            return False
        return (
            self._fusion.brief_lift(len(self._inventory.units)) >= MIN_EVIDENCE_LIFT
        )

    # ---------------- candidate channels ----------------

    def _attestation(self, query: str) -> dict[str, float]:
        """How strongly each known answer's own wording is attested in the query."""
        q = query.lower()
        q_bigrams = {q[i : i + 2] for i in range(len(q) - 1)}
        n = max(1, self._n_queries)
        out: dict[str, float] = {}
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
            out[unit] = 0.55 * attested + 0.45 * run
        return out

    def _vote(self, query: str) -> tuple[dict[str, float], dict[str, int]]:
        """Rank-decayed mass each answer unit carries in the query's neighbourhood.

        The pool is wide and uncapped on purpose: capping repeats per answer is
        right for exemplars, which are there to show form, but it destroys the
        very signal a vote is made of - how concentrated the neighbourhood is.
        """
        mass: dict[str, float] = defaultdict(float)
        count: dict[str, int] = defaultdict(int)
        for rank, (_score, doc_id) in enumerate(
            self._index.search(query)[:VOTE_POOL]
        ):
            for unit in self._inventory.split(self.examples[doc_id]["target"]):
                mass[unit] += 1.0 / (1 + rank)
                count[unit] += 1
        return dict(mass), dict(count)

    def _channels(
        self, query: str
    ) -> tuple[dict[str, tuple[float, float, float]], dict[str, int]]:
        """Per-unit (vote, attestation, prior) features, each normalised to [0, 1]."""
        if not self._inventory.units:
            return {}, {}
        mass, count = self._vote(query)
        att = self._attestation(query)
        vmax = max(mass.values()) if mass else 1.0
        total = sum(self._inventory.units.values()) or 1
        pmax = max(self._inventory.units.values()) / total
        feats = {
            unit: (
                (mass.get(unit, 0.0) / vmax) if vmax else 0.0,
                att.get(unit, 0.0),
                (self._inventory.units[unit] / total) / pmax if pmax else 0.0,
            )
            for unit in self._inventory.units
        }
        return feats, count

    def _brief(self, query: str) -> tuple[list[str], dict[str, int], dict[str, float]]:
        feats, count = self._channels(query)
        if not feats:
            return [], {}, {}
        order = _Fusion._rank(feats, self._fusion.leader())
        picked = order[:BRIEF_N]
        att = {u: feats[u][1] for u in picked}
        return picked, count, att

    # ---------------- example retrieval ----------------

    def _exemplar_budget(self) -> tuple[int, int]:
        """(max examples, max chars), scaled by the example channel's measured precision.

        Only meaningful where answers repeat: on open-ended generation an exact
        answer match is near-impossible however useful the neighbours are, so the
        precision signal there says nothing and the full budget is kept.
        """
        if (
            self._inventory_enabled()
            and self._fusion.n >= MIN_CALIB_OBS
            and self._fusion.neighbour_precision() < GOOD_PRECISION
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

    def _format_brief(
        self, brief: list[str], count: dict[str, int], att: dict[str, float]
    ) -> str:
        """Candidates with the evidence that put them there, not a bare list.

        A ranked list alone tells the model nothing about why an entry is there
        or how far to trust it. Each line carries its own support, and the
        header carries the ranker's measured hit rate, so a weak ranker
        advertises itself as weak instead of reading as an instruction.
        """
        if not brief:
            return ""
        lines = []
        for unit in brief:
            why = []
            n = count.get(unit, 0)
            if n:
                why.append(f"chosen in {n} of the {VOTE_POOL} most similar cases")
            if att.get(unit, 0.0) >= 0.9:
                why.append("its wording appears in this problem")
            suffix = f" - {'; '.join(why)}" if why else ""
            lines.append(f"- {unit}{suffix}")
        header = (
            "**Candidates from memory, most supported first** "
            f"(the top one has been right on {self._fusion.top_precision():.0%} "
            f"of past cases, the correct answer was somewhere in this list on "
            f"{self._fusion.brief_recall():.0%}):\n"
        )
        return header + "\n".join(lines) + "\n\n"

    def _multi_rate(self) -> float:
        total = sum(self._cardinality.values())
        if not total:
            return 0.0
        return sum(n for k, n in self._cardinality.items() if k > 1) / total

    def _format_cardinality(self) -> str:
        """How many units past answers listed.

        Where answers may name several things, a model that answers with one
        best guess is wrong by omission however good that guess is. This is the
        measured shape of the answer, and it costs one line.
        """
        if not self._inventory_enabled() or self._multi_rate() < MIN_MULTI_RATE:
            return ""
        total = sum(self._cardinality.values())
        parts = [
            f"{k} item{'s' if k > 1 else ''}: {n / total:.0%}"
            for k, n in sorted(self._cardinality.items())[:4]
        ]
        delim = self._inventory.delimiter() or ";"
        return (
            "**How many items past answers listed** (separator `"
            + delim
            + "`): "
            + ", ".join(parts)
            + "\n\n"
        )

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
        if self._brief_enabled():
            brief, count, att = self._brief(query)
        else:
            brief, count, att = [], {}, {}
        cardinality = self._format_cardinality()
        prompt = PROMPT.format(
            brief_section=self._format_brief(brief, count, att),
            cardinality_section=cardinality,
            inventory_section=self._format_inventory(brief),
            examples_section=self._format_examples(picked),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            brief_rule=BRIEF_RULE if brief else "",
            cardinality_rule=CARDINALITY_RULE if cardinality else "",
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
            "brief": brief,
            "weights": self._fusion.leader(),
            "inventory_on": self._inventory_enabled(),
            "snapped": canonical != answer,
            "retried": retried,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            question = r.get("raw_question", r["input"])

            # Score every weight vector against this answer before storing it,
            # so the fit is always measured on cases the ranker had not seen.
            if self._inventory.units:
                feats, _count = self._channels(question)
                neighbours = self._index.search(question)[:1]
                nearest = (
                    self.examples[neighbours[0][1]]["target"] if neighbours else None
                )
                self._fusion.observe(
                    feats, set(self._inventory.split(target)), nearest == target
                )

            self._affix.observe(r.get("prediction", ""), target)

            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._index.add(question)
            self._inventory.observe(target)
            self._cardinality[len(self._inventory.split(target))] += 1
            q = question.lower()
            for g in {q[i : i + 2] for i in range(len(q) - 1)}:
                self._query_df[g] += 1
            self._n_queries += 1

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self._template.strip(self.examples[-1]["input"])
        if self._brief_enabled():
            brief, count, att = self._brief(probe)
        else:
            brief, count, att = [], {}, {}
        return (
            len(self._format_brief(brief, count, att))
            + len(self._format_cardinality())
            + len(self._format_inventory(brief))
            + len(self._format_examples(self._retrieve(probe)))
        )

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._fusion = _Fusion()
        self._template = _Template()
        self._query_df = Counter()
        self._n_queries = 0
        self._cardinality = Counter()
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
