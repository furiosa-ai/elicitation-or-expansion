"""Distinction notes: the memory writes down what it got wrong, in words.

Eight systems have now been benchmarked in this run and every one of them learns
by accumulating counters - similarity indexes, answer inventories, log-odds cues,
displacement tables, correction pools. Not one has ever spent an LLM call at
learning time. That axis is untouched, and replaying the training streams says
there is something there to spend it on.

The answerer's mistakes are not spread evenly over the label space; they collapse
onto a handful of pairs. On the small label space 44 errors involve only 25
distinct (given, correct) pairs, 8 of which recur, and the top four pairs alone
account for 19 of the 44 errors. 25% of the errors that arrive have their exact
confusion already on record from an earlier step. The large label space is the
opposite - 219 distinct pairs, 13 recurring, 2% coverage - and open-ended
generation has no repeated answers at all, so the mechanism is measured to be
inert there rather than argued to be.

So: when a confusion recurs, the memory spends one LLM call to write a short
note saying how to tell the two answers apart, grounded in the actual cases on
each side. The note is stored as text, not as a statistic.

Two things keep it honest.

Liveness. A note is shown only when both of its sides are in contention for the
current query, judged by a rank-decayed vote over the nearest solved cases - the
model's own about-to-be-given answer is not available at predict time, so the
trigger has to come from the query alone. Measured on the small label space at a
top-3 vote: notes render on 21.5% of steps and, when one renders, the correct
answer is one of its two sides 60.5% of the time. Widening the vote to top-5 or
top-8 raises coverage to 29.5%/34.5% but drops relevance to 52.5%/49.3%, so the
narrow trigger is the one shipped.

Utility. Each note carries its own realised accuracy and is retired when it
underperforms the memory's running rate by a clear margin. Simulated at 6
observations and a 0.20 margin: a clearly harmful note is retired in 92.9% of
runs, a mildly harmful one in 64.0%, while a helpful note is retired in 1.9% and
a neutral one in 24.4%. Notes are a bandit, not a monotonically growing rulebook.
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

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{notes_section}{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}{notes_rule}- Respond in JSON format

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


class ConfusionNoteMemory(MemorySystem):
    """Two retrieval channels plus LLM-authored distinction notes for the answer
    pairs this answerer has actually confused, shown only when both are live."""

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
        prompt = PROMPT.format(
            notes_section=self._format_notes(notes),
            notes_rule=NOTES_RULE if notes else "",
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

        canonical = self._canonicalize(answer)
        return canonical, {
            "full_response": response,
            "num_retrieved": len(picked),
            "shortlist": shortlist,
            "inventory_on": self._inventory_enabled(),
            "snapped": canonical != answer,
            "retried": retried,
            "notes_shown": [self._notes.key(n["a"], n["b"]) for n in notes],
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

            # A confusion is a unit that was named in place of one that belonged.
            if not was_correct and self._inventory.units:
                given = set(self._inventory.split(self._affix.strip(
                    r.get("prediction", "")
                )))
                correct = set(self._inventory.split(target))
                if given and correct:
                    self._notes.observe_error(given, correct)

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
        )

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "notebook": self._notes.dump(),
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
        self._graded = 0
        self._graded_ok = 0
        # Replaying stored cases must not re-spend learning-time LLM calls.
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
        self._graded, self._graded_ok = data.get("graded", [0, 0])
