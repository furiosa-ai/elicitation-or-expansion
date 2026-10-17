"""Memory has learned WHICH answers exist. It has never learned HOW they are written.

Every system in this run stores answers and cases. Replaying the iteration-6 streams
shows what that leaves on the table: on the large label space 78% of all errors land
on a case whose correct answer string had never been seen before (accuracy there is
0.050 against 0.582 where it had been seen), and only 0.4% of those unseen answers
are even composable from previously seen parts. An inventory, a shortlist, a
precedent and a distinction note are all answers-that-exist mechanisms, so none of
them can reach that regime. What CAN reach it is the FORM of the answer space: the
model already knows the concept, it writes it in a shape the task never uses.

So this memory induces a specification of the answer space's surface form from the
graded answers alone - no queries, no cases - and states it as a constraint.

Three facts are induced, each validated offline against the real streams by the rate
at which the induced rule flags a VERIFIED answer (it must be ~never):

- Forbidden final characters. A character is forbidden-final only when the answer
  space uses it INSIDE answers (so it is a known character, not merely unseen) and
  never once ends one. That double condition is what makes it safe: the naive
  "character never seen" version flags 40.1% of real answers on a large-alphabet
  task, while this one flags 2.5%, 0.6% and 1.7% on the three streams. As a firing
  trigger it landed on an error 14 times out of 14.
- Item-count calibration. The share of verified answers naming more than one item
  against the share of the model's own drafts that do: +15.0 points on the large
  label space (0.315 vs 0.165), +8.0 on open-ended generation, and exactly 0.0
  where every answer is a single item, so the line writes itself out of existence
  on tasks that do not need it. Cardinality is necessary, not decorative: drafts
  whose item count disagrees with the verified answer score 0.000 on 44 cases.
- The length envelope of a single item.

Only constraints the model's own drafts have measurably violated are rendered, so
the block is empty on a task the answerer already conforms to. When a finished draft
breaks a hard constraint, one repair call names the violated rule and asks for the
answer restated in the attested form; that arm is scored against the counterfactual
and switches itself off if it ever comes out behind.
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

# Answer-form specification
FORM_MIN_UNITS = 25       # verified answer units before any rule may be asserted
FORM_MIN_DRAFTS = 10      # ... and drafts before a gap may be claimed
FORM_MIN_INTERNAL = 3     # a forbidden-final char must be attested inside answers
FORM_HOLDOUT = 40         # ... and survive this many further verified answer units
FORM_MIN_VIOLATIONS = 2   # ... and the drafts must have broken it this often
FORM_LEN_SLACK = 1.5      # length is only a violation this far past the envelope
FORM_MIN_CARD_GAP = 0.08  # item-count gap worth stating
FORM_MAX_FINALS_SHOWN = 8
FORM_MIN_ARM_OBS = 8
FORM_ARM_SLACK = 1

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{notes_section}{shortlist_section}{inventory_section}{examples_section}{form_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{form_rule}{inventory_rule}{shortlist_rule}{notes_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

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

FORM_RULE = (
    "- Your answer must satisfy every point under how answers are written. These "
    "were measured on verified answers to this task, and your own recent answers "
    "have broken them.\n"
)

FORM_REPAIR = """The answer you drafted below is written in a shape this task's verified
answers never use.

**Your draft:** {draft}

**What is wrong with its form:**
{violations}

**How answers are written on this task (measured on {n} verified answers):**
{spec}

{examples_section}**Problem:**
{input}

**Instructions:**
- The problem may well be solved correctly; it is the written form of the answer
  that is not. Give the same answer in the form this task actually uses.
- If the correct answer is genuinely a different one, give that instead.
{inventory_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

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


class _FormSpec:
    """The surface form of the answer space, induced from verified answers alone.

    Answers-that-exist mechanisms cannot reach a case whose answer string has never
    been seen. The shape of the answer space can, because it is a property of the
    space rather than of its contents, so it transfers to answers that are still
    unseen. Every rule here is stated only when the model's own drafts have broken
    it, which is what keeps the block empty on a task the answerer already fits.
    """

    def __init__(self) -> None:
        # Induced from verified answer units.
        self.n_units = 0
        self.finals: Counter = Counter()
        self.internal: Counter = Counter()
        self.lengths: list[int] = []
        self.eligible_at: dict[str, int] = {}
        self.n_answers = 0
        self.n_multi = 0
        # Measured on the model's own drafts.
        self.d_units = 0
        self.d_finals: Counter = Counter()
        self.d_answers = 0
        self.d_multi = 0
        # Counterfactual score of the repair arm.
        self.arm_n = 0
        self.arm_draft_ok = 0
        self.arm_final_ok = 0

    # ---- induction ----

    def observe_gold(self, units: list[str]) -> None:
        if not units:
            return
        self.n_answers += 1
        self.n_multi += int(len(units) > 1)
        for u in units:
            if not u:
                continue
            self.n_units += 1
            self.finals[u[-1]] += 1
            self.lengths.append(len(u))
            for ch in u[:-1]:
                self.internal[ch] += 1
        # A character becomes a rule CANDIDATE the moment it is attested inside
        # answers and has never ended one; the rule itself is withheld until it has
        # survived FORM_HOLDOUT further answers unrefuted. Refutation is permanent -
        # one verified answer ending in the character clears it for good.
        for ch, seen in self.internal.items():
            if self.finals[ch]:
                self.eligible_at.pop(ch, None)
            elif seen >= FORM_MIN_INTERNAL and ch not in self.eligible_at:
                self.eligible_at[ch] = self.n_units

    def observe_draft(self, units: list[str]) -> None:
        if not units:
            return
        self.d_answers += 1
        self.d_multi += int(len(units) > 1)
        for u in units:
            if u:
                self.d_units += 1
                self.d_finals[u[-1]] += 1

    def ready(self) -> bool:
        return self.n_units >= FORM_MIN_UNITS

    def forbidden_finals(self) -> list[str]:
        """Characters the answer space writes inside answers but never at the end.

        The "inside" condition is the whole rule. Without it, every character the
        sample happens not to contain becomes a law, and on a large alphabet that
        convicts two fifths of the verified answers themselves.
        """
        if not self.ready():
            return []
        out = [
            ch
            for ch, at in self.eligible_at.items()
            if self.finals[ch] == 0 and self.n_units - at >= FORM_HOLDOUT
        ]
        out.sort(key=lambda c: (-self.internal[c], c))
        return out

    def _broken_finals(self) -> list[tuple[str, int]]:
        """Forbidden finals the drafts have actually produced, with how often."""
        if self.d_units < FORM_MIN_DRAFTS:
            return []
        banned = set(self.forbidden_finals())
        out = [
            (ch, n)
            for ch, n in self.d_finals.items()
            if ch in banned and n >= FORM_MIN_VIOLATIONS
        ]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def _card_gap(self) -> tuple[float, float] | None:
        """Verified multi-item rate against the drafts' own, when they disagree."""
        if self.n_answers < FORM_MIN_UNITS // 2 or self.d_answers < FORM_MIN_DRAFTS:
            return None
        gold = self.n_multi / self.n_answers
        draft = self.d_multi / self.d_answers
        if abs(gold - draft) < FORM_MIN_CARD_GAP:
            return None
        return gold, draft

    # ---- rendering ----

    def lines(self) -> list[str]:
        if not self.ready():
            return []
        out: list[str] = []
        broken = self._broken_finals()
        if broken:
            chars = " ".join(repr(c) for c, _n in broken[:FORM_MAX_FINALS_SHOWN])
            total = sum(n for _c, n in broken)
            times = "once" if total == 1 else f"{total} times"
            out.append(
                f"No verified answer ends with any of: {chars}. The task writes "
                f"these characters inside answers but never at the end. Your own "
                f"recent answers have ended that way {times}."
            )
        gap = self._card_gap()
        if gap:
            gold, draft = gap
            out.append(
                f"{gold:.0%} of verified answers name more than one item; only "
                f"{draft:.0%} of your own recent answers did. Name every item the "
                f"problem supports, and no more."
            )
        if broken and self.lengths:
            out.append(
                f"A single item runs {min(self.lengths)}-{max(self.lengths)} characters."
            )
        return out

    # ---- enforcement ----

    def violations(self, units: list[str]) -> list[str]:
        """Hard breaches, phrased for the model. Only rules the drafts have broken."""
        if not self.ready() or not units:
            return []
        banned = {c for c, _n in self._broken_finals()}
        limit = max(self.lengths) * FORM_LEN_SLACK if self.lengths else 0
        out = []
        for u in units:
            if not u:
                continue
            if u[-1] in banned:
                out.append(
                    f"- {u!r} ends with {u[-1]!r}; no verified answer ends that way."
                )
            elif limit and len(u) > limit:
                out.append(
                    f"- {u!r} is {len(u)} characters; no verified answer item runs "
                    f"past {max(self.lengths)}."
                )
        return out

    def arm_enabled(self) -> bool:
        """A repair that loses to the draft it replaced switches off."""
        if self.arm_n < FORM_MIN_ARM_OBS:
            return True
        return self.arm_final_ok + FORM_ARM_SLACK >= self.arm_draft_ok

    def score_arm(self, draft_ok: bool, final_ok: bool) -> None:
        self.arm_n += 1
        self.arm_draft_ok += int(draft_ok)
        self.arm_final_ok += int(final_ok)

    def dump(self) -> dict[str, Any]:
        return {"arm": [self.arm_n, self.arm_draft_ok, self.arm_final_ok]}

    def load(self, data: dict[str, Any]) -> None:
        self.arm_n, self.arm_draft_ok, self.arm_final_ok = data.get("arm", [0, 0, 0])


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


class AnswerFormGrammar(MemorySystem):
    """Memory of how the answer space is WRITTEN, induced from verified answers and
    stated as constraints, so it reaches the cases whose answer string memory has
    never seen - where every answers-that-exist mechanism is blind."""

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
        self._form = _FormSpec()
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

    # ---------------- answer-form specification ----------------

    def _form_lines(self) -> list[str]:
        return self._form.lines()

    def _format_form(self) -> str:
        lines = self._form_lines()
        if not lines:
            return ""
        body = "\n".join(f"- {l}" for l in lines)
        return "**How answers are written on this task:**\n" + body + "\n\n"

    def _form_repair_due(self, units: list[str]) -> bool:
        return self._form.arm_enabled() and bool(self._form.violations(units))

    def _form_repair_prompt(self, input: str, query: str, draft: str,
                            units: list[str]) -> str:
        lines = self._form_lines()
        spec = "\n".join(f"- {l}" for l in lines) if lines else "- (see below)"
        return FORM_REPAIR.format(
            draft=draft,
            violations="\n".join(self._form.violations(units)),
            n=self._form.n_units,
            spec=spec,
            examples_section=self._format_examples(self._retrieve(query)[:3]),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
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
        shortlist = self._shortlist(query) if self._shortlist_enabled() else []
        notes = self._live_notes(query)
        form_section = self._format_form()
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
            form_section=form_section,
            form_rule=FORM_RULE if form_section else "",
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

        draft = self._canonicalize(answer)
        shown = list(notes)

        # A draft written in a shape the answer space never uses is wrong about its
        # form whatever it is right about, and unlike a dead answer that is legible
        # on the FIRST occurrence - no repeat count has to accumulate first.
        repaired = False
        form_draft = draft
        units = self._inventory.split(self._affix.strip(draft))
        if self._form_repair_due(units):
            repaired = True
            response3 = self.call_llm(
                self._form_repair_prompt(input, query, draft, units)
            )
            answer3 = extract_json_field(response3, "final_answer")
            if answer3.strip():
                candidate = self._canonicalize(answer3)
                if candidate.strip():
                    draft = candidate
                    response = response3

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
            "snapped": draft != answer,
            "retried": retried,
            "notes_shown": [self._notes.key(n["a"], n["b"]) for n in shown],
            "draft": draft,
            "revised": revised,
            "form_repaired": repaired,
            "form_draft": form_draft,
            "form_lines": len(self._form_lines()),
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

            # The induction must keep measuring what the ANSWERER writes, not what
            # repair rewrote, or the gap it is built on closes on its own output and
            # the block deletes itself after one success.
            form_draft = meta.get("form_draft")
            if form_draft is None:
                form_draft = draft
            raw = list(self._inventory.split(self._affix.strip(form_draft)))
            if not self._replaying:
                self._track.observe(draft, target, draft_ok, given, correct)
                if meta.get("revised"):
                    self._track.score_arm(draft_ok, was_correct)
                if meta.get("form_repaired"):
                    self._form.score_arm(
                        bool(raw) and set(raw) == set(correct), draft_ok
                    )
            self._form.observe_draft(raw)
            self._form.observe_gold(correct)

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
        probe = self._template.strip(self.examples[-1]["input"])
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        return (
            len(self._format_notes(self._live_notes(probe)))
            + len(self._format_shortlist(shortlist))
            + len(self._format_inventory(shortlist))
            + len(self._format_examples(self._retrieve(probe)))
            + len(self._format_form())
        )

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "notebook": self._notes.dump(),
                "track": self._track.dump(),
                "form": self._form.dump(),
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
        self._form = _FormSpec()
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
        self._form.load(data.get("form", {}))
        self._graded, self._graded_ok = data.get("graded", [0, 0])
