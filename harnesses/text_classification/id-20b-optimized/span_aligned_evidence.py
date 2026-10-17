"""Evidence measured in shared text, not in whole cases.

Every system in this run spends its prompt budget the same way: pick the k nearest
past cases and print each one, truncated to a few hundred characters. That fixes the
exchange rate at roughly one candidate answer per 250-400 characters, and on 200
stored cases with a per-answer cap it buys about eight distinct answers. On a task
whose answer space is a few dozen labels that is nearly everything; on one with a
hundred and forty it is nothing, and the correct answer is simply absent from the
prompt most of the time.

But almost none of those characters are doing any work. What makes a case a
precedent is the text it has in common with the problem, and measured on the real
streams the MAXIMAL VERBATIM SPANS shared between a query and a retrieved case come
to 20-31 characters - a tenth to a fifteenth of the excerpt they are buried in. The
rest is the part of the precedent that has nothing to do with this problem.

So the unit of evidence here is the shared span, not the case. Candidate answers are
ranked exactly as the base ranks their precedents - the ordering is not the change -
and each one is rendered as the literal text it and the problem have in common. At a
matched 2000-character budget, measured leave-one-out against the base's own
case-and-window layout on the real streams:

    small label space   8.0 answers, gold present 0.930  ->  21.9 answers, gold 1.000
    large label space   8.2 answers, gold present 0.640  ->  54.8 answers, gold 0.770

with the gold's rank among the shown answers unchanged (in the top 8 on 0.93 and
0.63 of queries respectively), because the ranking is the same one. The gain is
entirely in the tail the base could not afford to print.

Three full cases are still shown beneath the table. They are not there for coverage -
the table has that - but as the worked format: a span table demonstrates which
answers are in play and why, and nothing at all about how an answer is supposed to
be written. Dropping them entirely was rejected for that reason.

A span is only printed if it is long enough not to be a coincidence, and answers
whose precedent shares nothing with the problem are not listed at all, so the table
disappears rather than degrading where surface overlap carries no signal.
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
# --- span-aligned evidence table ---
TABLE_CHARS = 2000        # the whole evidence budget the table is allowed
TABLE_SPANS = 3           # shared spans printed per candidate answer
TABLE_SPAN_CHARS = 70     # ... and the characters they may take between them
TABLE_MIN_SPAN = 6        # shorter than this and a "shared span" is a coincidence
TABLE_DEPTH = 150         # ranked cases scanned for candidates
ANCHOR_CASES = 3          # whole cases kept below the table, as the worked format
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

{notes_section}{kit_section}{shortlist_section}{table_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{kit_rule}{table_rule}{inventory_rule}{shortlist_rule}{notes_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

KIT_RULE = (
    "- The recurring parts above have appeared verbatim in verified answers to "
    "other problems carrying this same context line. If one of them belongs in "
    "this answer, reproduce it EXACTLY, character for character, rather than "
    "writing your own version of it. They are parts of an answer, not the "
    "answer.\n"
)

TABLE_RULE = (
    "- The table above lists answers that resolved earlier problems, each followed "
    "by the text that problem and this one share word for word. Longer and more "
    "specific shared text is stronger evidence than a short or generic phrase; a "
    "listed answer with only a generic phrase beside it is not supported by "
    "anything. The table is ordered by overall similarity, not by strength of the "
    "shared text, and it is not a menu - if none of these answers fits, answer "
    "outside it.\n"
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


class SpanAlignedEvidence(MemorySystem):
    """The prompt's evidence budget is spent on the text a precedent shares with
    the problem rather than on the precedent, which buys several times as many
    candidate answers at the same number of characters."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
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

    def _span_idf(self, span: str) -> float:
        """How much of this span is text the corpus does not say everywhere.

        Ranking shared spans by LENGTH picks out whatever formula the documents have
        in common - a statutory citation, a filing header - because that is exactly
        the longest text any two of them share. Density of rare bigrams picks out the
        part that is about this problem.
        """
        text = span.lower()
        grams = [text[i : i + 2] for i in range(len(text) - 1)]
        if not grams:
            return 0.0
        n = max(1, self._n_queries)
        # Sum over sqrt(length). The plain sum is just length, which on any corpus
        # of one document type ranks the shared boilerplate first; the plain mean
        # goes the other way and prefers a six-character fragment of rare text to a
        # whole shared sentence, which measured worse on both streams. sqrt keeps a
        # long specific span ahead of a short one without making length the ranking.
        total = sum(math.log(1 + n / (1 + self._query_df[g])) for g in grams)
        return total / math.sqrt(len(grams))

    def _shared_spans(self, query: str, body: str) -> list[tuple[float, str]]:
        """Maximal substrings of the query that occur verbatim in the case, ranked
        by how distinctive they are rather than by how long they are.

        Each shared n-gram seeds a span which is extended to the right for as long
        as the extended text is still present in the case, so the result is whole
        shared phrases rather than the n-grams underneath them.
        """
        q, b = query.lower(), body.lower()
        n = max(NGRAM_SIZES)
        if len(q) < n or not b:
            return []
        found: list[tuple[int, int]] = []
        i = 0
        while i <= len(q) - n:
            if q[i : i + n] not in b:
                i += 1
                continue
            hi = i + n
            while hi < len(q) and q[i : hi + 1] in b:
                hi += 1
            found.append((i, hi))
            i = max(i + 1, hi - n + 1)
        scored = [
            (self._span_idf(q[a:z]), a, z) for a, z in found if z - a >= TABLE_MIN_SPAN
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))
        out: list[tuple[float, str]] = []
        taken: list[tuple[int, int]] = []
        for w, a, z in scored:
            if any(a < tz and ta < z for ta, tz in taken):
                continue
            taken.append((a, z))
            out.append((w, " ".join(query[a:z].split())))
            if len(out) >= TABLE_SPANS:
                break
        return out

    def _span_table(self, query: str) -> tuple[str, list[str]]:
        """Candidate answers, ranked as the base ranks them, rendered as the text
        their precedent shares with this problem instead of as the precedent.

        The ranking is deliberately unchanged; the compression is the whole point,
        and it is what pays for the tail of the table.
        """
        if not self.examples or not query:
            return "", []
        best: dict[str, int] = {}
        for _score, doc_id in self._index.search(query)[:TABLE_DEPTH]:
            target = self.examples[doc_id]["target"]
            if target not in best:
                best[target] = doc_id
        if not best:
            return "", []
        candidates = []
        for target, doc_id in best.items():
            ex = self.examples[doc_id]
            spans = self._shared_spans(query, ex.get("raw_question", ex["input"]))
            if spans:
                candidates.append((target, spans))
        if not candidates:
            return "", []
        # An admission floor was prototyped here - a row had to share text more
        # distinctive than an unrelated stored case does - and was removed: it did
        # not clear the formulaic rows on the corpus that has them (coverage 0.200
        # either way) and it cost coverage on the corpus that does not (0.960 ->
        # 0.860). The rule below tells the model what a generic shared phrase is
        # worth instead, which is the part it can actually act on.
        rows, listed, total = [], [], 0
        for target, spans in candidates:
            shown, width = [], 0
            for _w, text in spans:
                if width + len(text) > TABLE_SPAN_CHARS:
                    break
                shown.append(text)
                width += len(text) + 3
            if not shown:
                continue
            line = f"- {target}  <- shares: " + " | ".join(shown)
            if total + len(line) > TABLE_CHARS:
                break
            rows.append(line)
            listed.append(target)
            total += len(line) + 1
        if not rows:
            return "", []
        return (
            "**Answers to earlier problems, with the text they share with this one:**\n\n"
            + "\n".join(rows)
            + "\n\n"
        ), listed

    def _format_examples(self, picked: list[dict[str, str]]) -> str:
        if not picked:
            return ""
        _, max_chars = self._exemplar_budget()
        parts, total = [], 0
        # The table carries coverage; these are kept only as the worked format, so
        # a handful of the nearest cases is all that is wanted here.
        picked = picked[:ANCHOR_CASES]
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
        table_section, table_answers = self._span_table(query)
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
            kit_section=self._format_kit(kit, input),
            kit_rule=KIT_RULE if kit else "",
            shortlist_section=self._format_shortlist(shortlist),
            table_section=table_section,
            inventory_section=self._format_inventory(shortlist + table_answers),
            examples_section=self._format_examples(picked),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            table_rule=TABLE_RULE if table_section else "",
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
            "table_answers": len(table_answers),
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

            self._affix.observe(r.get("prediction", ""), target)

            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._fieldkit.observe(r["input"], target)
            self._index.add(question)
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
        table_section, _answers = self._span_table(probe)
        return (
            len(table_section)
            + len(self._format_kit(self._fieldkit.kit(raw), raw))
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
                "graded": [self._graded, self._graded_ok],
            },
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._affix = _Affix()
        self._calib = _Calibration()
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
        self._graded, self._graded_ok = data.get("graded", [0, 0])
