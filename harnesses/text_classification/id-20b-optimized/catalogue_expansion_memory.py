"""Memory that elicits the parts of the answer catalogue it has never been shown.

Replaying the frontier system over the real streams with a mock answerer makes the
binding constraint measurable. On the large-label-space stream the correct answer
string appears verbatim in the rendered memory context for 65 of 200 training
queries, and accuracy there is 0.692; on the other 135 it is 0.052. Retrieval is
already close to saturated against what memory OWNS - of the 74 queries whose gold
had been observed at all, the evidence ranker surfaces it for 65 - so no further
selection mechanism can reach the remaining items. They are unreachable because the
string has never been observed: 175 distinct answer units in 200 answers, with 40%
of late items still introducing one that has never appeared.

The answer strings, however, are not invented per case. They are drawn from a fixed
catalogue of designations that the answerer already has latent knowledge of, and
which memory has only ever sampled. So memory stops waiting to be shown the
catalogue and asks for it: at learning time, a handful of calls present the verified
answer strings observed so far as a style anchor and ask for OTHER designations from
the same catalogue, written in the same style. Those entries are hypotheses, never
attestations - they are ranked against the query by the same evidence scorer used
for observed answers and rendered in their own block, marked as unverified.

Prototyped against the real streams. Ranking an oracle catalogue of all 143 distinct
answer strings by query evidence puts the gold in the top 5 for 57% of queries and
the top 8 for 61%, against today's 32.5% verbatim-in-context rate and a 19%
frequency baseline; with three synthetic decoys injected per real entry - a far
noisier catalogue than an elicitation should produce - top-8 recall is still 51.5%.
The block costs about 120 characters.

Elicitation is not attempted where the answer space is not a catalogue. Three online
statistics separate the streams cleanly: distinct units per answer (0.88-1.4 where
the catalogue is open, 0.11-0.65 where it is closed, 1.5 where answers are
constructed rather than named), the share of distinct units ever repeated (0.04
where nothing is reused), and the trailing-window rate of answers introducing a new
unit (0.40-0.96 open, 0.00-0.24 once saturated). Measured on the real streams the
gate opens on exactly one of the three and stays shut on the others, whose prompts
are byte-identical to the base. A hypothesis that a later gold confirms is promoted
and counted; if no elicited entry is ever confirmed once the block has been live for
a full window, the block retires itself.
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
EXCERPTS_PER_SIDE = 2
NOTE_MIN_OBS = 6          # before a note may be retired
NOTE_MARGIN = 0.20

# Catalogue expansion: asking for the parts of the answer space never shown
CAT_MIN_ANSWERS = 50        # before the answer space can be called a catalogue
CAT_MIN_GROWTH = 0.70       # distinct units per observed answer
CAT_MAX_GROWTH = 1.45       # above this answers are constructed, not named
CAT_MIN_REPEAT = 0.10       # some designation must actually recur
CAT_WINDOW = 25             # trailing window for the saturation test
CAT_MIN_NEWRATE = 0.35      # ... and it must still be introducing new units
CAT_MAX_UNIT_CHARS = 25     # a designation, not a constructed object
CAT_EVERY = 25              # answers between elicitation calls
CAT_MAX_CALLS = 8           # bounds the learning-time LLM spend
CAT_ANCHOR_UNITS = 60       # verified strings shown as the style anchor
CAT_ANCHOR_CHARS = 1400
CAT_ASK = 50                # entries requested per call
CAT_POOL_MAX = 400
CAT_SHOWN = 8               # entries rendered per query
CAT_MIN_SCORE = 0.08        # ... and each must be echoed by the query at all
CAT_PROBATION = 40          # graded steps with the block live before the
                            # confirmation test may retire it

# Draft track record / second opinion
MIN_SAID = 4              # observations before an answer can be called dead
DEAD_SUCCESSES = 0        # ... and it must never once have been right
MAX_GOLDS_SHOWN = 6
MIN_ARM_OBS = 8           # fired steps before the arm may switch itself off
ARM_SLACK = 1             # ... and it must be behind the draft it replaced
SECOND_OPINION_NOTES = 2

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{notes_section}{kit_section}{shortlist_section}{catalogue_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{kit_rule}{inventory_rule}{shortlist_rule}{catalogue_rule}{notes_rule}- Respond in JSON format

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

CATALOGUE_RULE = (
    "- The candidate designations above have NOT been verified on any solved case. "
    "They are names from the same answer catalogue that no solved case has used "
    "yet, offered because this problem's wording echoes them. Do not treat them as "
    "evidence and do not prefer one over an answer the facts support; but if one of "
    "them is what the facts describe, use that exact string rather than a shorter "
    "or more familiar name for the same thing.\n"
)

CATALOGUE_PROMPT = """The lines below are verified answers from previously solved items of
one task. They are drawn from a fixed catalogue of designations that this task's
answer key uses.

**Verified answer strings:**
{anchor}

List other designations belonging to the SAME catalogue that are not already listed
above. Include both:
- designations covering cases unlike any of the listed ones, and
- narrower or differently scoped variants of the listed designations, which the
  catalogue distinguishes from them.

Copy the listed entries' exact style: same language, same level of detail, same
length, same conventions about what is and is not included in a name. Give the full
official form of each designation, never an abbreviation or a shortened version.

Output at most {ask} designations, one per line, nothing else. No numbering, no
bullets, no punctuation around them, no commentary, no blank lines."""

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


class _Catalogue:
    """Designations from the task's answer catalogue that no solved case has used.

    The observed inventory is a sample of the catalogue, not the catalogue. This
    holds the rest of it: entries elicited from the answerer's own latent knowledge
    of the catalogue, anchored on the style of the verified strings, kept strictly
    apart from anything a case has attested.
    """

    def __init__(self) -> None:
        self.entries: dict[str, int] = {}     # entry -> elicitation round it came from
        self.rounds = 0
        self.confirmed = 0                    # entries a later gold turned out to be
        self.elicited = 0
        self.live_steps = 0                   # graded steps with the block rendered
        self.at_answers = 0                   # inventory size at the last elicitation
        self._grams: dict[str, Counter] = {}

    # -- structural test: is the answer space an unsaturated catalogue? --

    @staticmethod
    def stats(inventory: "_Inventory", recent_new: list[int]) -> dict[str, float]:
        units = inventory.units
        n_answers = max(1, inventory.n_answers)
        if not units:
            return {"growth": 0.0, "repeat": 0.0, "newrate": 0.0, "medlen": 0.0}
        lengths = sorted(len(u) for u in units)
        window = recent_new[-CAT_WINDOW:]
        return {
            "growth": len(units) / n_answers,
            "repeat": sum(1 for c in units.values() if c > 1) / len(units),
            "newrate": (sum(window) / len(window)) if window else 1.0,
            "medlen": lengths[len(lengths) // 2],
        }

    def open(self, inventory: "_Inventory", recent_new: list[int]) -> bool:
        if inventory.n_answers < CAT_MIN_ANSWERS:
            return False
        st = self.stats(inventory, recent_new)
        return (
            CAT_MIN_GROWTH <= st["growth"] <= CAT_MAX_GROWTH
            and st["repeat"] >= CAT_MIN_REPEAT
            and st["newrate"] >= CAT_MIN_NEWRATE
            and st["medlen"] <= CAT_MAX_UNIT_CHARS
        )

    # -- eliciting --

    def due(self, inventory: "_Inventory") -> bool:
        return (
            self.rounds < CAT_MAX_CALLS
            and inventory.n_answers - self.at_answers >= CAT_EVERY
        )

    @staticmethod
    def anchor(inventory: "_Inventory") -> str:
        """Style anchor: the commonest strings plus the most recently arrived ones."""
        common = [u for u, _c in inventory.units.most_common(CAT_ANCHOR_UNITS // 2)]
        recent: list[str] = []
        for gold in reversed(inventory.golds):
            for unit in inventory.split(gold):
                if unit not in common and unit not in recent:
                    recent.append(unit)
            if len(recent) >= CAT_ANCHOR_UNITS // 2:
                break
        lines, total = [], 0
        for unit in common + recent:
            if total + len(unit) + 1 > CAT_ANCHOR_CHARS:
                break
            lines.append(unit)
            total += len(unit) + 1
        return "\n".join(lines)

    def absorb(self, response: str, inventory: "_Inventory") -> int:
        """Parse an elicitation into catalogue entries. Returns how many were kept."""
        observed = set(inventory.units)
        if not observed:
            return 0
        lengths = sorted(len(u) for u in observed)
        lo = max(1, int(0.5 * lengths[0]))
        hi = min(CAT_MAX_UNIT_CHARS * 2, int(2.0 * lengths[-1]) + 1)
        charset: Counter = Counter()
        for u in observed:
            charset.update(u)
        known = set(charset)
        delim = inventory.delimiter()
        kept = 0
        self.rounds += 1
        for raw in response.splitlines():
            line = raw.strip()
            line = re.sub(r"^[-*\u2022\d.)\s\"'\u201c]+", "", line).strip()
            line = line.strip("\"'\u201c\u201d`").strip()
            if not line or len(line) > hi or len(line) < lo:
                continue
            # Meta-text, prose and commentary rather than a designation.
            if any(ch in line for ch in ":：=<>{}[]#") or " - " in line:
                continue
            pieces = (
                [p.strip() for p in line.split(delim) if p.strip()]
                if delim and delim in line
                else [line]
            )
            for piece in pieces:
                if not (lo <= len(piece) <= hi) or piece in observed:
                    continue
                if piece in self.entries:
                    continue
                # The catalogue is written in one script; an entry built mostly
                # from characters the verified strings never use is commentary.
                share = sum(1 for ch in piece if ch in known) / len(piece)
                if share < 0.8:
                    continue
                self.entries[piece] = self.rounds
                self._grams[piece] = _answer_ngrams(piece)
                kept += 1
                self.elicited += 1
        # Oldest hypotheses go first when the pool overflows.
        while len(self.entries) > CAT_POOL_MAX:
            oldest = min(self.entries, key=lambda e: (self.entries[e], e))
            self.entries.pop(oldest, None)
            self._grams.pop(oldest, None)
        return kept

    # -- feedback --

    def promote(self, gold_units: list[str]) -> None:
        """A hypothesis the stream has now attested stops being a hypothesis."""
        for unit in gold_units:
            if unit in self.entries:
                self.entries.pop(unit, None)
                self._grams.pop(unit, None)
                self.confirmed += 1

    def useful(self) -> bool:
        """Retire the block if a full window of live steps confirmed nothing."""
        if self.live_steps < CAT_PROBATION:
            return True
        return self.confirmed > 0

    def dump(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "rounds": self.rounds,
            "confirmed": self.confirmed,
            "elicited": self.elicited,
            "live_steps": self.live_steps,
            "at_answers": self.at_answers,
        }

    def load(self, data: dict[str, Any]) -> None:
        self.entries = {str(k): int(v) for k, v in (data.get("entries") or {}).items()}
        self._grams = {e: _answer_ngrams(e) for e in self.entries}
        self.rounds = int(data.get("rounds", 0))
        self.confirmed = int(data.get("confirmed", 0))
        self.elicited = int(data.get("elicited", 0))
        self.live_steps = int(data.get("live_steps", 0))
        self.at_answers = int(data.get("at_answers", 0))


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


class CatalogueExpansionMemory(MemorySystem):
    """Memory whose answer space is not limited to the answers it has been shown.

    Where the observed answers look like a sample of an unsaturated catalogue of
    designations, the rest of that catalogue is elicited from the answerer and held
    as ranked hypotheses, separate from anything a solved case attests.
    """

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
        self._catalogue = _Catalogue()
        self._recent_new: list[int] = []
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

    # ---------------- catalogue hypotheses ----------------

    def _catalogue_enabled(self) -> bool:
        return (
            bool(self._catalogue.entries)
            and self._catalogue.open(self._inventory, self._recent_new)
            and self._catalogue.useful()
        )

    def _hypotheses(self, query: str) -> list[str]:
        """Rank never-observed catalogue entries by the query's own wording.

        Same scorer as the observed shortlist, so an entry earns its place on the
        page for exactly the reason an attested answer would - except that it has
        nothing attesting it, and the block says so.
        """
        q = query.lower()
        q_bigrams = {q[i : i + 2] for i in range(len(q) - 1)}
        n = max(1, self._n_queries)
        scored: list[tuple[float, str]] = []
        for entry in self._catalogue.entries:
            e = entry.lower()
            bigrams = [e[i : i + 2] for i in range(len(e) - 1)]
            num = den = 0.0
            for g in bigrams:
                w = math.log(1 + n / (1 + self._query_df[g]))
                den += w
                if g in q_bigrams:
                    num += w
            attested = num / den if den else 0.0
            longest = 0
            for i in range(len(e)):
                for j in range(len(e), i + longest, -1):
                    if e[i:j] in q:
                        longest = j - i
                        break
            run = longest / len(e) if e else 0.0
            score = 0.55 * attested + 0.45 * run
            if score >= CAT_MIN_SCORE:
                scored.append((score, entry))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [e for _s, e in scored[:CAT_SHOWN]]

    def _format_catalogue(self, hypotheses: list[str], exclude: list[str]) -> str:
        skip = set(exclude)
        lines = [f"- {e}" for e in hypotheses if e not in skip]
        if not lines:
            return ""
        return (
            "**Candidate designations from the same answer catalogue "
            "(not used by any solved case - unverified):**\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def _elicit_catalogue(self) -> None:
        anchor = _Catalogue.anchor(self._inventory)
        if not anchor:
            return
        self._catalogue.at_answers = self._inventory.n_answers
        response = self.call_llm(CATALOGUE_PROMPT.format(anchor=anchor, ask=CAT_ASK))
        self._catalogue.absorb(response, self._inventory)

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
        hypotheses = self._hypotheses(query) if self._catalogue_enabled() else []
        catalogue_section = self._format_catalogue(hypotheses, shortlist)
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
            kit_section=self._format_kit(kit, input),
            kit_rule=KIT_RULE if kit else "",
            shortlist_section=self._format_shortlist(shortlist),
            catalogue_section=catalogue_section,
            catalogue_rule=CATALOGUE_RULE if catalogue_section else "",
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
            "hypotheses": hypotheses,
            "catalogue_shown": bool(catalogue_section),
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

            # Score the hypothesis block before the answer joins the inventory:
            # a hypothesis this gold confirms was a real catalogue entry, and it
            # is the only evidence there is that the elicitation was not fiction.
            gold_units = self._inventory.split(target)
            if meta.get("catalogue_shown"):
                self._catalogue.live_steps += 1
            self._catalogue.promote(gold_units)
            self._recent_new.append(
                int(any(u not in self._inventory.units for u in gold_units))
            )
            del self._recent_new[:-CAT_WINDOW]

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

        # The catalogue is asked for only while the answer space still behaves
        # like one that has not been fully sampled.
        if (
            not self._replaying
            and self._catalogue.open(self._inventory, self._recent_new)
            and self._catalogue.due(self._inventory)
        ):
            self._elicit_catalogue()

    def _base_rate(self) -> float:
        return self._graded_ok / self._graded if self._graded else 0.0

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        raw = self.examples[-1]["input"]
        probe = self._template.strip(raw)
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        hypotheses = self._hypotheses(probe) if self._catalogue_enabled() else []
        return (
            len(self._format_catalogue(hypotheses, shortlist))
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
                "catalogue": self._catalogue.dump(),
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
        self._catalogue = _Catalogue()
        self._recent_new = []
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
        self._catalogue.load(data.get("catalogue", {}))
        self._graded, self._graded_ok = data.get("graded", [0, 0])
