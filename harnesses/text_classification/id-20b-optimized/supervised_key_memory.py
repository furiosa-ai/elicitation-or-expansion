"""Retrieval keys that use the labels memory already owns.

Every system in this run indexes a stored case by its TEXT alone: the document
vector is a function of which n-grams the case contains and of how rare those
n-grams are in the corpus. The answer of that case - which memory is handed for
free on every step - is used to decide what to PRINT and never to decide what is
RETRIEVABLE. Rarity is a property of the corpus; association is a property of the
decision, and it is the decision that retrieval is serving.

So each stored document is re-weighted per gram by how much that gram is lifted
under the document's own answer, log P(g | answer) / P(g), and each known answer
is scored against a query directly by the same lift summed over the query's grams.
Two things keep this from overfitting itself, both of which are in the estimator
rather than in a dataset-specific gate:

  * the answer statistics are LEAVE-ONE-OUT, so a case never supplies the evidence
    for its own boost; and
  * the boost is shrunk by lambda = d / (d + C) for d the number of OTHER cases
    sharing that answer, so an answer seen once gets no supervision at all and
    falls back exactly to the unsupervised weighting.

Measured by replaying the real training streams against the shipped ranker, with
the supervised weights swapped in:

    nearest case carries the right answer   0.079 -> 0.116 and 0.621 -> 0.653
    right answer inside the top 12 cases    0.184 -> 0.211 and 0.858 -> 0.874

and where answers never repeat, lambda is zero everywhere and the ranking comes
back bit-identical to the base.

The same statistics rank ANSWERS without requiring the answer's own characters to
appear in the query, which is all the base's shortlist can key on. The two answer
rankers are strong in opposite regimes - with many cases per answer the supervised
one puts the right answer in the top 8 on 0.879 of queries against the echo
ranker's 0.374; with roughly one case per answer it is the echo ranker that leads,
0.247 to 0.168 - and blending their ranks loses to both (0.768 / 0.211 under
reciprocal-rank fusion). So the shortlist's slots are not blended and not fixed:
each channel's realised hit rate is scored against the gold on every step, and the
slots are dealt out in proportion to it. That recovers 0.853 of the 0.879 where
supervision wins and 0.232 of the 0.247 where it loses, with no constant naming
either regime.
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

# Label-supervised keys
SUP_SHRINK_C = 2.0     # support at which an answer earns half of its supervision
SUP_MAX_LIFT = 2.0     # a single gram may not dominate a document
SUP_MIN_LIFT = -1.0    # ... nor may its absence
SHARE_PRIOR = 2.0      # pseudo-observations before a channel's share is believed
SHARE_FLOOR = 1        # ... and it never loses every slot
MIN_CALIB_OBS = 12
MIN_EVIDENCE_LIFT = 2.0

# Canonicalisation
SNAP_THRESHOLD = 0.65
SNAP_MARGIN = 0.04
MIN_AFFIX_OBS = 4

SPLIT_PRECEDENCE = [";", "；", "\n", "|", "，", ",", "、"]

# Categorical query fields and their answer-component kits. The kit's own
# delimiter list is wider than SPLIT_PRECEDENCE because the base's inventory,
# track record and snapping all key off that list and must not be disturbed.
KIT_SPLIT = SPLIT_PRECEDENCE + ["."]
FIELD_MIN_ROWS = 12        # before any group may be promoted to a field
FIELD_MIN_PREFIX = 8       # a field's values must share this much literal text
FIELD_MIN_COVERAGE = 0.5   # ... and between them account for this much of the stream
FIELD_MIN_EXCLUSIVITY = 0.9  # ... and a query must land in exactly one value
FIELD_PIECE_MIN = 3
FIELD_PIECE_MAX = 120
KIT_MIN_IN_VALUE = 2       # a component must recur inside the value
KIT_MIN_PRECISION = 0.75   # ... and be near-exclusive to it
KIT_MAX_UNITS = 12
KIT_SNAP_THRESHOLD = 0.85  # a drafted component this close to a kit one is snapped

# Distinction notes
MIN_PAIR_OBS = 2          # a confusion must recur before it is worth a call
NB_NEIGHBOURS = 8         # cases voting on what is in contention
LIVE_TOP = 3              # both sides must be inside this vote
MAX_NOTES_SHOWN = 2
MAX_NOTES_TOTAL = 12      # bounds the learning-time LLM spend
NOTE_MAX_CHARS = 320
EXCERPT_CHARS = 400
EXCERPTS_PER_SIDE = 2
NOTE_MIN_OBS = 6          # before a note may be retired
NOTE_MARGIN = 0.20

# Draft track record / second opinion
MIN_SAID = 4              # observations before an answer can be called dead
DEAD_SUCCESSES = 0        # ... and it must never once have been right
MAX_GOLDS_SHOWN = 6
MIN_ARM_OBS = 8           # fired steps before the arm may switch itself off
ARM_SLACK = 1             # ... and it must be behind the draft it replaced
SECOND_OPINION_NOTES = 2

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{notes_section}{kit_section}{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{kit_rule}{inventory_rule}{shortlist_rule}{notes_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

KIT_RULE = (
    "- The recurring parts above have appeared verbatim in verified answers to "
    "other problems carrying this same context line. If one of them belongs in "
    "this answer, reproduce it EXACTLY, character for character, rather than "
    "writing your own version of it. They are parts of an answer, not the "
    "answer.\n"
)

INVENTORY_RULE = (
    "- If the correct answer is one of the listed answer strings, reproduce that "
    "string EXACTLY, character for character. Otherwise answer in the same style "
    "and level of detail as those strings.\n"
)

NOTES_RULE = (
    "- The distinctions above were written after this exact pair of answers was "
    "confused on an earlier case. Check which side of each distinction this "
    "problem falls on before answering.\n"
)

NOTE_PROMPT = """Two answers keep being confused with one another on this task. Write the
distinction that separates them, using only what the cases below show.

**Answer A:** {a}
Cases whose correct answer was A:
{a_cases}

**Answer B:** {b}
Cases whose correct answer was B:
{b_cases}

Write at most two sentences naming the features that decide between A and B.
State what must be present for A and what must be present for B. Do not restate
the answers, do not hedge, do not give background.

{{"distinction": "[your two sentences]"}}"""

SECOND_OPINION = """You have drafted an answer to the problem below. Before it is
submitted, check it against this answer's own track record on this task.

**Your draft answer:** {draft}

**Track record of that exact answer here:** given {said} times, and it was the
complete correct answer {ok} of those times.
{golds_section}{modes_section}{notes_section}{alternatives_section}
**Problem:**
{input}

**Instructions:**
- The record above is about the answer string, not about this problem. Re-read the
  problem and decide what the answer actually is.
- Keep the draft only if this problem genuinely supports it; otherwise give the
  answer the evidence points to.
{inventory_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

MODE_LABELS = {
    "incomplete": (
        "the verified answer contained your draft AND at least one further item"
    ),
    "scope": "the verified answer was a differently scoped version of your draft",
    "unrelated": "the verified answer had nothing in common with your draft",
}

SHORTLIST_RULE = (
    "- The shortlisted answers are the ones whose own wording is echoed by this "
    "problem's text. Treat them as strong candidates, check each against the "
    "problem, and prefer one of them unless the evidence clearly points "
    "elsewhere.\n"
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


class _SupervisedIndex:
    """Inverted index whose document weights are lifted by the document's answer.

    Identical to an unsupervised tf-idf index whenever an answer has no other
    case to learn from: the shrinkage term is then zero and every multiplier is
    one.  A document is re-weighted only when the support of its own answer
    changes, which is why the postings are rebuilt per answer rather than per
    step.
    """

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, float]] = defaultdict(dict)
        self.df: Counter = Counter()
        self.n_docs = 0
        self.grams: list[Counter] = []
        self.answers: list[str] = []
        self.by_answer: dict[str, list[int]] = defaultdict(list)
        self.gram_answer: dict[str, Counter] = defaultdict(Counter)
        self.gram_docs: Counter = Counter()
        self._norm: dict[int, float] = {}
        self._norm_at: dict[int, int] = {}

    # -- supervision -------------------------------------------------------
    def _lift(
        self, gram: str, answer: str, doc_id: int | None, floor: float = 0.0
    ) -> float:
        """log P(gram | answer) / P(gram), leave-one-out, clamped.

        A document is only ever boosted by the grams its own answer lifts, never
        cut by the ones it does not: measured on the real streams, letting the
        multiplier fall below one zeroes grams out of the document vector and
        costs the nearest-neighbour rate it was supposed to buy (0.116 -> 0.070
        where answers are nearly all singletons).  Ranking answers is the other
        direction - there a query gram an answer never shows is evidence against
        it - so that caller passes a negative floor.
        """
        support = len(self.by_answer[answer])
        in_answer = self.gram_answer[gram][answer]
        in_corpus = self.gram_docs[gram]
        if doc_id is not None:
            # The document never supplies the evidence for its own boost.
            support -= 1
            in_answer -= 1 if gram in self.grams[doc_id] else 0
            in_corpus -= 1 if gram in self.grams[doc_id] else 0
        if support <= 0 or in_corpus <= 0 or self.n_docs <= 1:
            return 0.0
        p_answer = in_answer / support
        p_corpus = in_corpus / max(1, self.n_docs - (1 if doc_id is not None else 0))
        if p_corpus <= 0:
            return 0.0
        raw = math.log((p_answer + 1e-6) / (p_corpus + 1e-6))
        return max(floor, min(SUP_MAX_LIFT, raw))

    def _shrink(self, answer: str, doc_id: int | None) -> float:
        d = len(self.by_answer[answer]) - (1 if doc_id is not None else 0)
        return d / (d + SUP_SHRINK_C) if d > 0 else 0.0

    def _write(self, doc_id: int) -> None:
        """Unnormalised document weights; the norm is idf-dependent and lazy."""
        grams = self.grams[doc_id]
        answer = self.answers[doc_id]
        lam = self._shrink(answer, doc_id)
        for g, c in grams.items():
            w = 1.0 + math.log(c)
            if lam > 0.0:
                w *= 1.0 + lam * self._lift(g, answer, doc_id)
            self.postings[g][doc_id] = w
        self._norm_at[doc_id] = 0

    def _idf(self, gram: str) -> float:
        return math.log(1 + self.n_docs / (1 + self.df[gram]))

    def _norm_of(self, doc_id: int) -> float:
        """Cosine norm over the idf-weighted document vector.

        The base normalises the document before idf is applied, which lets a case
        made of common grams sit as close to a query as one sharing a rare phrase
        with it.  Measured by replaying the real streams, moving the idf inside
        the norm takes the nearest case's answer from 0.075 to 0.116 where the
        documents are long and heterogeneous, and is worth nothing where they are
        short (0.608 -> 0.603) - which is the half of the key supervision cannot
        supply, since there almost every answer is a singleton.  idf drifts as
        the corpus grows, so the norm is refreshed once the corpus has grown by a
        tenth rather than on every step.
        """
        at = self._norm_at.get(doc_id, 0)
        if at and self.n_docs < at * 1.1:
            return self._norm[doc_id]
        total = 0.0
        for g in self.grams[doc_id]:
            w = self.postings[g].get(doc_id, 0.0) * self._idf(g)
            total += w * w
        self._norm[doc_id] = math.sqrt(total) or 1.0
        self._norm_at[doc_id] = self.n_docs
        return self._norm[doc_id]

    def add(self, text: str, answer: str = "") -> None:
        grams = _char_ngrams(text)
        doc_id = self.n_docs
        self.grams.append(grams)
        self.answers.append(answer)
        self.by_answer[answer].append(doc_id)
        for g in grams:
            self.df[g] += 1
            self.gram_answer[g][answer] += 1
            self.gram_docs[g] += 1
        self.n_docs += 1
        # This answer's support just moved, so every document that carries it is
        # weighted differently now.
        for other in self.by_answer[answer]:
            self._write(other)

    def search(self, text: str) -> list[tuple[float, int]]:
        if not self.n_docs:
            return []
        grams = _char_ngrams(text)
        scores: dict[int, float] = defaultdict(float)
        for g, c in grams.items():
            posting = self.postings.get(g)
            if not posting:
                continue
            idf = self._idf(g)
            qw = (1.0 + math.log(c)) * idf
            for doc_id, dw in posting.items():
                scores[doc_id] += qw * dw * idf
        ranked = [(sc / self._norm_of(i), i) for i, sc in scores.items()]
        return sorted(ranked, key=lambda x: (-x[0], x[1]))

    # -- answers ranked directly, without needing their own text in the query --
    def rank_answers(self, text: str) -> list[tuple[float, str]]:
        if not self.n_docs:
            return []
        grams = _char_ngrams(text)
        qw = {
            g: (1.0 + math.log(c)) * math.log(1 + self.n_docs / (1 + self.df[g]))
            for g, c in grams.items()
        }
        total = sum(qw.values()) or 1.0
        # A gram the query has and the answer never shows is evidence against it,
        # so the baseline is the floor and present grams buy their way back up.
        acc: dict[str, float] = defaultdict(float)
        for g, w in qw.items():
            column = self.gram_answer.get(g)
            if not column:
                continue
            for answer in column:
                acc[answer] += w * (
                    self._lift(g, answer, None, SUP_MIN_LIFT) - SUP_MIN_LIFT
                )
        out = []
        for answer in self.by_answer:
            lam = self._shrink(answer, None)
            out.append((lam * (SUP_MIN_LIFT + acc[answer] / total), answer))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out


class _ChannelShare:
    """Slots dealt to each candidate channel in proportion to its measured hits."""

    def __init__(self) -> None:
        self.hits: Counter = Counter()
        self.seen: Counter = Counter()

    def observe(self, channel: str, hit: bool) -> None:
        self.seen[channel] += 1
        self.hits[channel] += int(hit)

    def rate(self, channel: str) -> float:
        return (self.hits[channel] + 1.0) / (self.seen[channel] + SHARE_PRIOR)

    def split(self, channels: list[str], slots: int) -> dict[str, int]:
        rates = {c: self.rate(c) for c in channels}
        total = sum(rates.values()) or 1.0
        out, spent = {}, 0
        for c in channels[:-1]:
            k = int(round(slots * rates[c] / total))
            k = max(SHARE_FLOOR, min(slots - SHARE_FLOOR * (len(channels) - 1), k))
            out[c] = k
            spent += k
        out[channels[-1]] = max(SHARE_FLOOR, slots - spent)
        return out

    def dump(self) -> dict[str, Any]:
        return {"hits": dict(self.hits), "seen": dict(self.seen)}

    def load(self, data: dict[str, Any]) -> None:
        self.hits = Counter(data.get("hits", {}))
        self.seen = Counter(data.get("seen", {}))


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


def _query_pieces(text: str) -> set[str]:
    """Short line- and sentence-level phrases of a query, as candidate field values."""
    out: set[str] = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 200:
            continue
        for piece in re.split(r"(?<=[.。:：])\s*", line):
            piece = piece.strip(" .。:：")
            if FIELD_PIECE_MIN <= len(piece) <= FIELD_PIECE_MAX:
                out.add(piece)
    return out


def _shared_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


class _FieldKit:
    """A categorical field of the query, plus the answer components bound to each
    of its values.

    A field is a family of query phrases that share a long literal prefix, never
    co-occur in one query, and between them cover most of the stream. That is what
    a structured context line looks like and what free-text boilerplate does not:
    boilerplate either appears in every query (no partition) or in a handful (no
    coverage). Nothing about any particular task is assumed - the family is found
    by looking at the queries themselves.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[list[str], str]] = []
        self.delim_hits: Counter = Counter()
        self._field: tuple[str, frozenset] | None = None
        self._fitted_at = -1

    # -- answer components --------------------------------------------------

    def delimiter(self) -> str | None:
        for d in KIT_SPLIT:
            if self.delim_hits[d] >= max(2, 0.03 * len(self.rows)):
                return d
        return None

    def split(self, text: str) -> list[str]:
        d = self.delimiter()
        if d and d in text:
            return [p.strip() for p in text.split(d) if p.strip()]
        return [text.strip()] if text.strip() else []

    def observe(self, query: str, gold: str) -> None:
        self.rows.append((sorted(_query_pieces(query)), gold))
        for d in KIT_SPLIT:
            if d in gold:
                self.delim_hits[d] += 1
        self._fitted_at = -1

    # -- field discovery ----------------------------------------------------

    def field(self) -> tuple[str, frozenset] | None:
        n = len(self.rows)
        if n < FIELD_MIN_ROWS:
            return None
        if self._fitted_at == n:
            return self._field
        counts: Counter = Counter()
        for pieces, _g in self.rows:
            for piece in pieces:
                counts[piece] += 1
        # A value must repeat (or it cannot be categorical) and must not be in
        # every query (or it partitions nothing).
        cands = [p for p, c in counts.items() if 2 <= c <= 0.9 * n]
        groups: dict[str, set[str]] = defaultdict(set)
        for i, a in enumerate(cands):
            for b in cands[i + 1 :]:
                prefix = _shared_prefix(a, b)
                if len(prefix) >= FIELD_MIN_PREFIX:
                    groups[prefix].add(a)
                    groups[prefix].add(b)
        best: tuple[float, str, frozenset] | None = None
        row_sets = [set(p) for p, _g in self.rows]
        for prefix, values in groups.items():
            if not 2 <= len(values) <= max(2, n / 3):
                continue
            hit = [rs for rs in row_sets if rs & values]
            if not hit:
                continue
            coverage = len(hit) / n
            exclusive = sum(1 for rs in hit if len(rs & values) == 1) / len(hit)
            if coverage < FIELD_MIN_COVERAGE or exclusive < FIELD_MIN_EXCLUSIVITY:
                continue
            score = coverage * exclusive * len(prefix)
            if best is None or score > best[0]:
                best = (score, prefix, frozenset(values))
        self._field = (best[1], best[2]) if best else None
        self._fitted_at = n
        return self._field

    def value_of(self, query: str) -> str | None:
        field = self.field()
        if not field:
            return None
        hit = _query_pieces(query) & field[1]
        return next(iter(hit)) if len(hit) == 1 else None

    # -- the kit ------------------------------------------------------------

    def kit(self, query: str) -> list[tuple[str, int]]:
        """Answer components that recur under this query's field value and are
        near-exclusive to it, commonest first."""
        value = self.value_of(query)
        if value is None:
            return []
        overall: Counter = Counter()
        inside: Counter = Counter()
        for pieces, gold in self.rows:
            units = self.split(gold)
            for unit in units:
                overall[unit] += 1
            if value in pieces:
                for unit in units:
                    inside[unit] += 1
        out = [
            (unit, count)
            for unit, count in inside.items()
            if count >= KIT_MIN_IN_VALUE
            and count / max(1, overall[unit]) >= KIT_MIN_PRECISION
        ]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out[:KIT_MAX_UNITS]


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


class _NoteBook:
    """Confusion pairs, the notes written about them, and each note's track record."""

    def __init__(self) -> None:
        self.pairs: Counter = Counter()          # directed (given, correct)
        self.notes: dict[str, dict[str, Any]] = {}
        self.pending: list[tuple[str, str]] = []

    @staticmethod
    def key(a: str, b: str) -> str:
        lo, hi = sorted((a, b))
        return lo + "\u241f" + hi

    def observe_error(self, given: set[str], correct: set[str]) -> None:
        """Record every answer that was named instead of one that belonged."""
        for g in given - correct:
            for c in correct - given:
                self.pairs[(g, c)] += 1
                k = self.key(g, c)
                if k in self.notes:
                    continue
                both = self.pairs[(g, c)] + self.pairs[(c, g)]
                queued = (g, c) in self.pending or (c, g) in self.pending
                if both >= MIN_PAIR_OBS and not queued:
                    if len(self.notes) + len(self.pending) < MAX_NOTES_TOTAL:
                        self.pending.append((g, c))

    def add_note(self, a: str, b: str, text: str) -> None:
        self.notes[self.key(a, b)] = {
            "a": a,
            "b": b,
            "text": text,
            "n": 0,
            "ok": 0,
            "retired": False,
        }

    def live(self, contenders: list[str]) -> list[dict[str, Any]]:
        """Notes whose two sides are both in contention for this query."""
        pool = set(contenders)
        out = []
        for note in self.notes.values():
            if note["retired"]:
                continue
            if note["a"] in pool and note["b"] in pool:
                strength = self.pairs[(note["a"], note["b"])] + self.pairs[
                    (note["b"], note["a"])
                ]
                out.append((strength, note))
        out.sort(key=lambda x: (-x[0], x[1]["a"]))
        return [n for _s, n in out[:MAX_NOTES_SHOWN]]

    def for_answer(self, answer: str) -> list[dict[str, Any]]:
        """Notes that were written about this exact answer being confused."""
        out = []
        for note in self.notes.values():
            if note["retired"]:
                continue
            if note["a"] == answer or note["b"] == answer:
                other = note["b"] if note["a"] == answer else note["a"]
                strength = self.pairs[(answer, other)] + self.pairs[(other, answer)]
                out.append((strength, note))
        out.sort(key=lambda x: (-x[0], x[1]["a"]))
        return [n for _s, n in out[:SECOND_OPINION_NOTES]]

    def credit(self, keys: list[str], was_correct: bool, base_rate: float) -> None:
        """A note that does worse than the memory's own running rate is retired."""
        for k in keys:
            note = self.notes.get(k)
            if not note or note["retired"]:
                continue
            note["n"] += 1
            note["ok"] += int(was_correct)
            if note["n"] >= NOTE_MIN_OBS and note["ok"] / note["n"] < (
                base_rate - NOTE_MARGIN
            ):
                note["retired"] = True

    def dump(self) -> dict[str, Any]:
        return {
            "pairs": [[g, c, n] for (g, c), n in self.pairs.items()],
            "notes": self.notes,
            "pending": [list(p) for p in self.pending],
        }

    def load(self, data: dict[str, Any]) -> None:
        self.pairs = Counter({(g, c): n for g, c, n in data.get("pairs", [])})
        self.notes = data.get("notes", {})
        self.pending = [tuple(p) for p in data.get("pending", [])]


class _Track:
    """What actually happened the other times this exact answer was given.

    Keyed on the canonical answer string the memory shipped, so the record is
    about the answerer's habit rather than about any one query.
    """

    def __init__(self) -> None:
        self.said: Counter = Counter()
        self.ok: Counter = Counter()
        self.golds: dict[str, Counter] = defaultdict(Counter)
        self.modes: dict[str, Counter] = defaultdict(Counter)
        # Counterfactual score of the second-opinion arm.
        self.arm_n = 0
        self.arm_draft_ok = 0
        self.arm_final_ok = 0

    def observe(
        self,
        answer: str,
        gold: str,
        draft_ok: bool,
        given: list[str],
        correct: list[str],
    ) -> None:
        if not answer.strip():
            return
        self.said[answer] += 1
        self.ok[answer] += int(draft_ok)
        self.golds[answer][gold] += 1
        if draft_ok:
            return
        g, c = set(given), set(correct)
        if g and c and g < c:
            self.modes[answer]["incomplete"] += 1
        elif any(x != y and (x in y or y in x) for x in g for y in c):
            self.modes[answer]["scope"] += 1
        elif not (g & c):
            self.modes[answer]["unrelated"] += 1

    def is_dead(self, answer: str) -> bool:
        """Proposed often enough to have a record, and that record is empty."""
        return (
            bool(answer.strip())
            and self.said[answer] >= MIN_SAID
            and self.ok[answer] <= DEAD_SUCCESSES
        )

    def arm_enabled(self) -> bool:
        """A second opinion that loses to the draft it replaces switches off.

        The drafts it fires on are dead by construction, so the bar is not "does
        it help" but "does it destroy value"; a one-answer slack keeps a bad run
        from retiring an arm whose counterfactual is near zero either way.
        """
        if self.arm_n < MIN_ARM_OBS:
            return True
        return self.arm_final_ok + ARM_SLACK >= self.arm_draft_ok

    def score_arm(self, draft_ok: bool, final_ok: bool) -> None:
        self.arm_n += 1
        self.arm_draft_ok += int(draft_ok)
        self.arm_final_ok += int(final_ok)

    def dump(self) -> dict[str, Any]:
        return {
            "said": dict(self.said),
            "ok": dict(self.ok),
            "golds": {k: dict(v) for k, v in self.golds.items()},
            "modes": {k: dict(v) for k, v in self.modes.items()},
            "arm": [self.arm_n, self.arm_draft_ok, self.arm_final_ok],
        }

    def load(self, data: dict[str, Any]) -> None:
        self.said = Counter(data.get("said", {}))
        self.ok = Counter(data.get("ok", {}))
        self.golds = defaultdict(
            Counter, {k: Counter(v) for k, v in data.get("golds", {}).items()}
        )
        self.modes = defaultdict(
            Counter, {k: Counter(v) for k, v in data.get("modes", {}).items()}
        )
        self.arm_n, self.arm_draft_ok, self.arm_final_ok = data.get("arm", [0, 0, 0])


class SupervisedKeyMemory(MemorySystem):
    """Retrieval keyed on the answers memory has already been graded against, with
    the supervision shrunk to nothing exactly where no answer ever repeats."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _SupervisedIndex()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._share = _ChannelShare()
        self._template = _Template()
        self._query_df: Counter = Counter()
        self._n_queries = 0
        self._notes = _NoteBook()
        self._track = _Track()
        self._fieldkit = _FieldKit()
        self._graded = 0
        self._graded_ok = 0
        self._replaying = False

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

    def _channel_ranks(self, query: str) -> dict[str, list[str]]:
        """The two answer rankings: one keyed on the answer's own wording appearing
        in the query, one keyed on the query's grams being lifted under it."""
        echo = [u for _s, u in self._evidence_rank(query)]
        known = set(self._inventory.units)
        lifted = [a for _s, a in self._index.rank_answers(query) if a in known]
        if not lifted:
            # Answers are composite here, so score each unit by its best whole answer.
            seen: list[str] = []
            for _s, whole in self._index.rank_answers(query):
                for unit in self._inventory.split(whole):
                    if unit in known and unit not in seen:
                        seen.append(unit)
            lifted = seen
        return {"echo": echo, "lifted": lifted}

    def _shortlist(self, query: str) -> list[str]:
        if not self._inventory.units:
            return []
        ranks = self._channel_ranks(query)
        share = self._share.split(["echo", "lifted"], SHORTLIST_N)
        out: list[str] = []
        for channel, k in share.items():
            for unit in ranks[channel][:k]:
                if unit not in out:
                    out.append(unit)
        # A channel that ran out of distinct answers gives its slots back.
        for channel in ("echo", "lifted"):
            for unit in ranks[channel]:
                if len(out) >= SHORTLIST_N:
                    break
                if unit not in out:
                    out.append(unit)
        return out[:SHORTLIST_N]

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

    def _format_kit(self, kit: list[tuple[str, int]], query: str) -> str:
        if not kit:
            return ""
        value = self._fieldkit.value_of(query)
        head = (
            f"**Parts that recur in answers where {value}:**"
            if value
            else "**Parts that recur in answers to problems like this one:**"
        )
        lines = [f"- {unit}  (in {count} verified answers)" for unit, count in kit]
        return head + "\n" + "\n".join(lines) + "\n\n"

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


    # ---------------- distinction notes ----------------

    def _notes_enabled(self) -> bool:
        # Notes exist only where answers repeat; an open-ended task never gets a
        # pair to two observations, and this gate makes that explicit.
        return self._inventory_enabled()

    def _contenders(self, query: str) -> list[str]:
        """Rank-decayed vote over the nearest solved cases' answer units."""
        vote: Counter = Counter()
        for rank, (score, doc_id) in enumerate(
            self._index.search(query)[:NB_NEIGHBOURS]
        ):
            for unit in self._inventory.split(self.examples[doc_id]["target"]):
                vote[unit] += score / (1.0 + rank)
        return [u for u, _v in vote.most_common(LIVE_TOP)]

    def _live_notes(self, query: str) -> list[dict[str, Any]]:
        if not self._notes_enabled() or not self._notes.notes:
            return []
        return self._notes.live(self._contenders(query))

    def _format_notes(self, notes: list[dict[str, Any]]) -> str:
        if not notes:
            return ""
        lines = [f"- {n['a']} vs {n['b']}: {n['text']}" for n in notes]
        return (
            "**Distinctions learned from earlier mistakes on this task:**\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def _excerpts(self, unit: str) -> str:
        """The most recent cases whose correct answer contained this unit."""
        out = []
        for ex in reversed(self.examples):
            if unit not in self._inventory.split(ex["target"]):
                continue
            body = self._template.strip(ex.get("raw_question", ex["input"]))
            out.append("- " + body[:EXCERPT_CHARS].replace("\n", " ").strip())
            if len(out) >= EXCERPTS_PER_SIDE:
                break
        return "\n".join(out) if out else "- (none recorded)"

    def _author_pending(self) -> None:
        """One LLM call per newly recurring confusion, bounded in total."""
        while self._notes.pending and len(self._notes.notes) < MAX_NOTES_TOTAL:
            a, b = self._notes.pending.pop(0)
            if self._notes.key(a, b) in self._notes.notes:
                continue
            response = self.call_llm(
                NOTE_PROMPT.format(
                    a=a,
                    b=b,
                    a_cases=self._excerpts(a),
                    b_cases=self._excerpts(b),
                )
            )
            text = extract_json_field(response, "distinction").strip()
            if not text:
                text = " ".join(response.split())[:NOTE_MAX_CHARS]
            if text:
                self._notes.add_note(a, b, text[:NOTE_MAX_CHARS])

    # ---------------- draft-keyed second opinion ----------------

    def _second_opinion_due(self, draft: str) -> bool:
        return (
            self._inventory_enabled()
            and self._track.arm_enabled()
            and self._track.is_dead(draft)
        )

    def _format_golds(self, draft: str) -> str:
        """What the answer turned out to be the other times this draft was given."""
        golds = self._track.golds.get(draft)
        if not golds:
            return ""
        lines = []
        for gold, n in golds.most_common(MAX_GOLDS_SHOWN):
            times = "once" if n == 1 else f"{n} times"
            lines.append(f"- {gold}   ({times})")
        return (
            "\n**On those occasions the verified answer was:**\n"
            + "\n".join(lines)
            + "\n"
        )

    def _format_modes(self, draft: str) -> str:
        modes = self._track.modes.get(draft)
        if not modes:
            return ""
        lines = [
            f"- {MODE_LABELS[m]}: {n} of them"
            for m, n in modes.most_common()
            if m in MODE_LABELS and n
        ]
        if not lines:
            return ""
        return "\n**How it went wrong:**\n" + "\n".join(lines) + "\n"

    def _format_alternatives(self, query: str, draft: str) -> str:
        """Candidate answers for this query, with the dead draft removed."""
        pool: list[str] = []
        if self._shortlist_enabled():
            pool = [a for a in self._shortlist(query) if a != draft]
        if not pool:
            pool = [a for a in self._contenders(query) if a != draft]
        if not pool:
            return ""
        return (
            "\n**Answers this problem's own wording points at (draft removed):**\n"
            + "\n".join(f"- {a}" for a in pool)
            + "\n"
        )

    def _second_opinion_prompt(self, input: str, query: str, draft: str) -> tuple[str, list[dict[str, Any]]]:
        notes = self._notes.for_answer(draft) if self._notes_enabled() else []
        notes_section = ""
        if notes:
            notes_section = "\n" + self._format_notes(notes)
        return (
            SECOND_OPINION.format(
                draft=draft,
                said=self._track.said[draft],
                ok=self._track.ok[draft],
                golds_section=self._format_golds(draft),
                modes_section=self._format_modes(draft),
                notes_section=notes_section,
                alternatives_section=self._format_alternatives(query, draft),
                input=input,
                inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            ),
            notes,
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

    def _kit_snap(self, draft: str, kit: list[tuple[str, int]]) -> str:
        """Pull a drafted component onto a kit component it is a near-copy of.

        Restricted to the kit: those components are near-exclusive to this field
        value and have been verified several times, so a near-miss against one of
        them is a spelling of it rather than a different thing. The base's
        inventory-wide snap is a different proposition and stays gated where it is.
        """
        if not kit or not draft.strip():
            return draft
        delim = self._fieldkit.delimiter()
        parts = self._fieldkit.split(draft)
        if not parts:
            return draft
        targets = [u for u, _c in kit]
        out, changed = [], False
        for part in parts:
            if part in targets:
                out.append(part)
                continue
            grams = _answer_ngrams(part, 3)
            best_score, best = 0.0, None
            for unit in targets:
                score = _dice(grams, _answer_ngrams(unit, 3))
                if score > best_score:
                    best_score, best = score, unit
            if best is not None and best_score >= KIT_SNAP_THRESHOLD:
                out.append(best)
                changed = True
            else:
                out.append(part)
        if not changed:
            return draft
        joined = (delim or ".").join(dict.fromkeys(out))
        return joined or draft

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
        notes = self._live_notes(query)
        kit = self._fieldkit.kit(input)
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
            kit_section=self._format_kit(kit, input),
            kit_rule=KIT_RULE if kit else "",
            shortlist_section=self._format_shortlist(shortlist),
            inventory_section=self._format_inventory(shortlist),
            examples_section=self._format_examples(picked),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            shortlist_rule=SHORTLIST_RULE if shortlist else "",
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

        draft = self._kit_snap(self._canonicalize(answer), kit)
        shown = list(notes)

        # The draft is the key. Only when the memory's own record says this exact
        # answer has never worked is a second call worth its cost.
        revised = False
        final = draft
        if self._second_opinion_due(draft):
            revised = True
            prompt2, draft_notes = self._second_opinion_prompt(input, query, draft)
            shown.extend(n for n in draft_notes if n not in shown)
            response2 = self.call_llm(prompt2)
            answer2 = extract_json_field(response2, "final_answer")
            if answer2.strip():
                candidate = self._canonicalize(answer2)
                if candidate.strip():
                    final = candidate
                    response = response2

        return final, {
            "full_response": response,
            "num_retrieved": len(picked),
            "shortlist": shortlist,
            "inventory_on": self._inventory_enabled(),
            "kit": [u for u, _c in kit],
            "snapped": draft != answer,
            "retried": retried,
            "notes_shown": [self._notes.key(n["a"], n["b"]) for n in shown],
            "draft": draft,
            "revised": revised,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            question = r.get("raw_question", r["input"])
            meta = r.get("metadata") or {}
            was_correct = bool(r.get("was_correct"))

            # Score any note that was rendered for this step against the outcome,
            # before the outcome is folded into the memory's own running rate.
            self._notes.credit(
                meta.get("notes_shown") or [], was_correct, self._base_rate()
            )
            self._graded += 1
            self._graded_ok += int(was_correct)

            # The record is about what the ANSWERER proposes, so it is keyed on the
            # draft; crediting the revised answer here would resurrect a dead one.
            draft = meta.get("draft")
            if draft is None:
                draft = r.get("prediction", "")
            given = list(self._inventory.split(self._affix.strip(draft)))
            correct = list(self._inventory.split(target))
            draft_ok = (
                was_correct
                if not meta.get("revised")
                else bool(given) and set(given) == set(correct)
            )
            if not self._replaying:
                self._track.observe(draft, target, draft_ok, given, correct)
                if meta.get("revised"):
                    self._track.score_arm(draft_ok, was_correct)

            # A confusion is a unit that was named in place of one that belonged.
            if not draft_ok and self._inventory.units and given and correct:
                self._notes.observe_error(set(given), set(correct))

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
                # Each answer ranker is graded on the same unseen case, so the
                # slots follow whichever one is actually finding the answer here.
                ranks = self._channel_ranks(question)
                for channel, ranked in ranks.items():
                    self._share.observe(
                        channel, any(u in ranked[:SHORTLIST_N] for u in gold_units)
                    )

            self._affix.observe(r.get("prediction", ""), target)

            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._fieldkit.observe(r["input"], target)
            self._index.add(question, target)
            self._inventory.observe(target)
            q = question.lower()
            for g in {q[i : i + 2] for i in range(len(q) - 1)}:
                self._query_df[g] += 1
            self._n_queries += 1

        if self._notes_enabled() and not self._replaying:
            self._author_pending()

    def _base_rate(self) -> float:
        return self._graded_ok / self._graded if self._graded else 0.0

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        raw = self.examples[-1]["input"]
        probe = self._template.strip(raw)
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        return (
            len(self._format_kit(self._fieldkit.kit(raw), raw))
            + len(self._format_notes(self._live_notes(probe)))
            + len(self._format_shortlist(shortlist))
            + len(self._format_inventory(shortlist))
            + len(self._format_examples(self._retrieve(probe)))
        )

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "notebook": self._notes.dump(),
                "track": self._track.dump(),
                "share": self._share.dump(),
                "graded": [self._graded, self._graded_ok],
            },
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _SupervisedIndex()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
        self._share = _ChannelShare()
        self._template = _Template()
        self._query_df = Counter()
        self._n_queries = 0
        self._notes = _NoteBook()
        self._track = _Track()
        self._fieldkit = _FieldKit()
        self._graded = 0
        self._graded_ok = 0
        # Replaying stored cases must not re-spend learning-time LLM calls, and
        # must not write a fabricated outcome into the track record.
        self._replaying = True
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
        self._replaying = False
        self._notes.load(data.get("notebook", {}))
        self._track.load(data.get("track", {}))
        self._share.load(data.get("share", {}))
        self._graded, self._graded_ok = data.get("graded", [0, 0])
