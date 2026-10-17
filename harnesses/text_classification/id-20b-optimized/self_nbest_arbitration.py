"""Memory has always been the thing that supplies answers. Make it the thing that
judges them, and let the model supply the set.

Replaying the iteration-6 streams for where the losses actually are: on the large
label space 78% of all errors land on a case whose correct answer string memory had
never seen (accuracy 0.050 there against 0.582 where it had), and 48.5% of the
verified answer units are not even composable from previously seen parts. The
inventory, the shortlist, the precedent slate and the distinction notebook are all
drawing from a pool that simply does not contain the answer. The one generator in
the loop that does contain it is the model.

So the output contract asks the answerer for its own ranked alternatives alongside
its answer, and memory's job changes from supplying candidates to scoring them.

The scoring rule was prototyped first, and the prototype returned a decisive
NEGATIVE that shaped the design. A memory-side re-ranker allowed to silently
overrule the draft was run over both real streams at every setting of draft prior
and override margin: its best result was net 0 on the large label space (17 fires,
2 conversions, 2 breakages) and net -22 on the small one (46 fires, 6 conversions,
28 breakages). Not one configuration was positive. Memory is a good WITNESS about
an answer and a bad JUDGE of one.

So this system never overrules the model. When memory's preferred alternative is
not the model's own first choice, and it wins by a margin, one arbitration call
hands the model back its OWN alternatives annotated with what memory knows about
each - how strongly the answer's wording is attested in this problem, how much of
the neighbourhood votes for it, what its track record here is, whether it has ever
been a verified answer - and the model decides. Its counterfactual is scored on
every fire and the arm switches itself off if it ever comes out behind.

Alternatives are requested only where memory can actually score them - the same
answer-space gate the inventory uses - so on a task whose answers never repeat no
alternatives are asked for and the prompt is unchanged.
"""

import ast
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

# Self-generated alternatives and arbitration
ALT_MAX_REQUESTED = 3      # alternatives the answerer is asked to rank
ALT_MAX_SCORED = 4         # alternatives carried into scoring, first choice included
ALT_MIN_EXAMPLES = 25      # graded cases before memory can witness anything
ALT_MARGIN = 0.30          # memory must beat the model's own pick by this to speak
FLOOR_MIN_OBS = 40         # graded drafts before a witness floor can be fitted
FLOOR_MIN_BUCKET = 15      # ... and observations inside the band it picks
FLOOR_RATIO = 0.5          # the band must be this much worse than the running rate
FLOOR_STEPS = 9            # candidate cut points, taken as quantiles of what is seen
ALT_MIN_ARM_OBS = 8
ALT_ARM_SLACK = 1
W_ATTEST = 1.00            # the answer's wording is echoed by this problem
W_VOTE = 0.80              # the neighbourhood's cases resolve to it
W_RECORD = 0.60            # it has worked before when this model gave it
W_ATTESTED = 0.35          # it has been a verified answer at all
RECORD_MIN_OBS = 3

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{notes_section}{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}{notes_rule}{alt_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"{alt_field}}}"""

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

ALT_RULE = (
    "- After your answer, list the next most plausible answers you considered, best "
    "first. Give complete answers in the same form, not partial ones, and leave the "
    "list empty if nothing else was close.\n"
)

ALT_FIELD = ', "alternatives": ["[next best]", "[after that]"]'

ARBITRATE = """You answered the problem below, and you also listed the other answers you
considered. The record kept on this task disagrees with your ordering.

**The answers you gave, with what the record says about each:**
{slate}

**Reading the record:**
- "echoed by this problem" is how much of that answer's own wording appears in this
  problem's text.
- "cases nearby resolved to it" is how many of the most similar solved cases had it
  as their verified answer.
- "your record with it" is how often you have given that exact answer here and how
  often it was the complete verified answer.
- An answer marked never verified here may still be right; it only means this task
  has not produced it yet.
{notes_section}
**Problem:**
{input}

**Instructions:**
- The record is evidence, not a verdict, and it is silent about answers this task
  has not yet produced. Re-read the problem and settle it yourself.
- Choose one of the answers listed above, or give a different one if the problem
  supports neither.
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


def _parse_alternatives(response: str) -> list[str]:
    """The answerer's own ranked alternatives, if it gave any."""
    raw = extract_json_field(response, "alternatives").strip()
    if not raw or raw in ("[]", "None"):
        return []
    items: list[Any] = []
    if raw.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                items = parsed
                break
    if not items:
        items = [raw]
    out: list[str] = []
    for it in items:
        text = str(it).strip()
        # Models sometimes echo the template's own placeholder rather than an answer.
        if text and not (text.startswith("[") and text.endswith("]")):
            out.append(text)
    return out[:ALT_MAX_REQUESTED]


class _Arm:
    """Counterfactual score of an arm against the draft it is allowed to replace."""

    def __init__(self, min_obs: int, slack: int) -> None:
        self.n = 0
        self.draft_ok = 0
        self.final_ok = 0
        self._min_obs = min_obs
        self._slack = slack

    def enabled(self) -> bool:
        if self.n < self._min_obs:
            return True
        return self.final_ok + self._slack >= self.draft_ok

    def score(self, draft_ok: bool, final_ok: bool) -> None:
        self.n += 1
        self.draft_ok += int(draft_ok)
        self.final_ok += int(final_ok)

    def dump(self) -> list[int]:
        return [self.n, self.draft_ok, self.final_ok]

    def load(self, data: list[int]) -> None:
        self.n, self.draft_ok, self.final_ok = data or [0, 0, 0]


class _Floor:
    """Where memory's witness of the model's own answer stops being worth listening to.

    A fixed score threshold is a guess about a task. This fits one online instead:
    it keeps the witness score of every graded draft beside whether that draft was
    right, and looks for the largest cut below which drafts are at least twice as
    likely to be wrong as the memory's own running rate. Where no such band exists
    the floor does not exist either, and everything gated on it stays silent - which
    is the behaviour wanted on a task the answerer is already good at.
    """

    def __init__(self) -> None:
        self.scores: list[float] = []
        self.oks: list[int] = []

    def observe(self, score: float, was_correct: bool) -> None:
        self.scores.append(score)
        self.oks.append(int(was_correct))

    def cut(self) -> float | None:
        n = len(self.scores)
        if n < FLOOR_MIN_OBS:
            return None
        rate = sum(self.oks) / n
        if rate <= 0.0:
            return None
        target = rate * FLOOR_RATIO
        pairs = sorted(zip(self.scores, self.oks))
        best: float | None = None
        for step in range(1, FLOOR_STEPS + 1):
            k = int(n * step / (FLOOR_STEPS + 1))
            if k < FLOOR_MIN_BUCKET or n - k < FLOOR_MIN_BUCKET:
                continue
            band = pairs[:k]
            if sum(o for _s, o in band) / k < target:
                best = band[-1][0]
        return best

    def dump(self) -> dict[str, Any]:
        return {"scores": self.scores, "oks": self.oks}

    def load(self, data: dict[str, Any]) -> None:
        self.scores = [float(x) for x in data.get("scores", [])]
        self.oks = [int(x) for x in data.get("oks", [])]


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


class SelfNbestArbitration(MemorySystem):
    """The answerer proposes a ranked set, memory witnesses each member of it, and
    the answerer - never memory - settles the disagreement. The candidate pool is
    the model's, so it reaches the answers memory has never seen."""

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
        self._arb = _Arm(ALT_MIN_ARM_OBS, ALT_ARM_SLACK)
        self._floor = _Floor()
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

    # ---------------- witnessing the model's own alternatives ----------------

    def _alternatives_enabled(self) -> bool:
        """Asking for a set is only worth it where memory can say something about it."""
        return self._inventory_enabled() and self._graded >= ALT_MIN_EXAMPLES

    def _attestation(self, answer: str, query: str) -> float:
        """How strongly this answer's own wording is echoed by the problem.

        Deliberately computed from the strings rather than looked up, so it is
        defined for an answer this task has never produced - which is the whole
        point of letting the model supply the pool.
        """
        u, q = answer.lower(), query.lower()
        if not u:
            return 0.0
        q_bigrams = {q[i : i + 2] for i in range(len(q) - 1)}
        n = max(1, self._n_queries)
        bigrams = [u[i : i + 2] for i in range(len(u) - 1)] or [u]
        num = den = 0.0
        for g in bigrams:
            w = math.log(1 + n / (1 + self._query_df[g]))
            den += w
            if g in q_bigrams:
                num += w
        attested = num / den if den else 0.0
        longest = 0
        for i in range(len(u)):
            for j in range(len(u), i + longest, -1):
                if u[i:j] in q:
                    longest = j - i
                    break
        return 0.55 * attested + 0.45 * (longest / len(u))

    def _neighbour_mass(self, query: str) -> dict[str, float]:
        """Normalised rank-decayed vote of the nearest solved cases."""
        vote: Counter = Counter()
        for rank, (score, doc_id) in enumerate(
            self._index.search(query)[:NB_NEIGHBOURS]
        ):
            for unit in self._inventory.split(self.examples[doc_id]["target"]):
                vote[unit] += score / (1.0 + rank)
        total = sum(vote.values())
        return {u: v / total for u, v in vote.items()} if total else {}

    def _witness(self, answer: str, query: str, mass: dict[str, float]) -> tuple[float, list[str]]:
        """Memory's evidence about one answer, as a score and as English."""
        units = self._inventory.split(self._affix.strip(answer)) or [answer]
        attest = sum(self._attestation(u, query) for u in units) / len(units)
        vote = sum(mass.get(u, 0.0) for u in units)
        said = self._track.said[answer]
        ok = self._track.ok[answer]
        record = ok / said if said >= RECORD_MIN_OBS else 0.0
        known = sum(1 for u in units if u in self._inventory.units) / len(units)
        score = (
            W_ATTEST * attest
            + W_VOTE * vote
            + W_RECORD * record
            + W_ATTESTED * known
        )
        notes = [f"echoed by this problem {attest:.0%}"]
        if vote:
            notes.append(f"cases nearby resolved to it {vote:.0%} of the vote")
        else:
            notes.append("no case nearby resolved to it")
        if said:
            notes.append(
                f"your record with it: given {said}, complete answer {ok}"
            )
        if known < 1.0:
            notes.append("never a verified answer here")
        return score, notes

    def _slate(self, ranked: list[tuple[float, str, list[str]]], chosen: str) -> str:
        """Best-supported first, with the learned output markup stripped so the
        answers are compared as answers rather than as envelopes."""
        lines = []
        alt = 0
        for _s, answer, notes in ranked:
            if answer == chosen:
                tag = "your answer"
            else:
                alt += 1
                tag = f"alternative {alt}"
            shown = self._affix.strip(answer) or answer
            lines.append(f"- {shown}   ({tag})\n    " + "; ".join(notes))
        return "\n".join(lines)

    def _arbitration_prompt(self, input: str, query: str, chosen: str,
                            ranked: list[tuple[float, str, list[str]]]) -> tuple[str, list[dict[str, Any]]]:
        notes = self._notes.for_answer(chosen) if self._notes_enabled() else []
        notes_section = "\n" + self._format_notes(notes) if notes else ""
        return (
            ARBITRATE.format(
                slate=self._slate(ranked, chosen),
                notes_section=notes_section,
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
        want_alts = self._alternatives_enabled()
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
            shortlist_section=self._format_shortlist(shortlist),
            inventory_section=self._format_inventory(shortlist),
            examples_section=self._format_examples(picked),
            input=input,
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
            shortlist_rule=SHORTLIST_RULE if shortlist else "",
            alt_rule=ALT_RULE if want_alts else "",
            alt_field=ALT_FIELD if want_alts else "",
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        alternatives = _parse_alternatives(response) if want_alts else []

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

        # Memory witnesses the model's own set. It never picks from it: a re-ranker
        # allowed to overrule the draft was measured over both real streams at every
        # setting of prior and margin and was never once net positive. All it may do
        # is notice that it disagrees, and hand the question back.
        arbitrated = False
        pre_arb = draft
        own_score: float | None = None
        ranked: list[tuple[float, str, list[str]]] = []
        mass: dict[str, float] = {}
        if self._alternatives_enabled():
            mass = self._neighbour_mass(query)
            own_score = self._witness(draft, query, mass)[0]
        if alternatives and self._arb.enabled():
            pool = [draft] + [
                self._canonicalize(a)
                for a in alternatives
                if self._canonicalize(a).strip() and self._canonicalize(a) != draft
            ]
            seen: set[str] = set()
            pool = [a for a in pool if not (a in seen or seen.add(a))][:ALT_MAX_SCORED]
            if len(pool) > 1:
                scored = [(*self._witness(a, query, mass), a) for a in pool]
                ranked = sorted(
                    ((s, a, n) for s, n, a in scored), key=lambda x: (-x[0], x[1])
                )
                own = next(s for s, a, _n in ranked if a == draft)
                cut = self._floor.cut()
                if (
                    ranked[0][1] != draft
                    and cut is not None
                    and own <= cut
                    and ranked[0][0] - own >= ALT_MARGIN
                ):
                    arbitrated = True
                    prompt_a, alt_notes = self._arbitration_prompt(
                        input, query, draft, ranked
                    )
                    shown.extend(n for n in alt_notes if n not in shown)
                    response_a = self.call_llm(prompt_a)
                    answer_a = extract_json_field(response_a, "final_answer")
                    if answer_a.strip():
                        candidate = self._canonicalize(answer_a)
                        if candidate.strip():
                            draft = candidate
                            response = response_a

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
            "alternatives": alternatives,
            "arbitrated": arbitrated,
            "arb_draft": pre_arb,
            "n_alternatives": len(alternatives),
            "witness": own_score,
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
            # The arbitration arm is credited against the answer the model had
            # BEFORE it was handed its own set back, so a call that talks it out of a
            # right answer is paid for at once.
            arb_draft = meta.get("arb_draft")
            if arb_draft is None:
                arb_draft = draft
            before = list(self._inventory.split(self._affix.strip(arb_draft)))
            if not self._replaying:
                self._track.observe(draft, target, draft_ok, given, correct)
                if meta.get("revised"):
                    self._track.score_arm(draft_ok, was_correct)
                if meta.get("arbitrated"):
                    self._arb.score(
                        bool(before) and set(before) == set(correct), draft_ok
                    )
                # The floor is fitted on the answer memory witnessed, and on whether
                # THAT answer was right, so an arbitrated step teaches the floor
                # about the draft it saw rather than about the revision.
                if meta.get("witness") is not None:
                    self._floor.observe(
                        float(meta["witness"]),
                        bool(before) and set(before) == set(correct)
                        if meta.get("arbitrated")
                        else draft_ok,
                    )

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
        alts = len(ALT_RULE) + len(ALT_FIELD) if self._alternatives_enabled() else 0
        return (
            alts
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
                "arb": self._arb.dump(),
                "floor": self._floor.dump(),
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
        self._arb = _Arm(ALT_MIN_ARM_OBS, ALT_ARM_SLACK)
        self._floor = _Floor()
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
        self._arb.load(data.get("arb", [0, 0, 0]))
        self._floor.load(data.get("floor", {}))
        self._graded, self._graded_ok = data.get("graded", [0, 0])
