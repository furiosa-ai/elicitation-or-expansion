"""Output-contract control: the memory learns how to ASK, not only what to show.

Every system in this line has spent its effort on what goes into the prompt and
none on the shape of the answer it demands or on what happens when no answer
comes back. Replaying the iteration-1..4 training streams says that is where the
losses are concentrated.

The base system, like all four before it, ends a step that produced no parseable
answer by re-asking with the input alone - no memory, no learned format, "give
your answer immediately". Across four systems and three tasks that fallback
fired 136 times and produced exactly one correct answer (0.7%). On open-ended
generation it fires on 42-46% of steps (21/50, 23/50, 23/50, 22/50) and is
correct 0 times out of 89, against 7-11% on the steps that never needed it. So
nearly half of that task is decided by a call that has never once succeeded,
and the memory is switched off for precisely those calls.

Two mechanisms follow.

1. A contract bandit. The output contract - the order of the JSON fields, how
   much deliberation is allowed, whether deliberation is requested at all - is
   an arm, not a constant. Arms are scored online by the rate at which they
   produce a scorable attempt, with realised accuracy breaking ties between arms
   that produce attempts equally often. Exploration is gated on the memory's own
   measured failure rate, so a task whose first call always parses never leaves
   the base contract and its prompts stay byte-identical. Simulated over 250
   steps: the picker converges to the best arm where the base contract fails 45%
   of the time, gives a harmful arm 3.4% of the traffic, and explores on exactly
   0% of steps when the base contract never fails.

2. A recovery ladder that keeps the memory. A failed extraction now re-asks with
   the same retrieved context under the strictest contract, and only falls back
   to the bare input if that also comes back empty.

Both are inert wherever the first call already parses, which is measured to be
100% of the steps on the task where this line's accuracy is highest and where
every added memory block in iterations 2-4 caused a regression.
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

# ---- output contracts (bandit arms) ----
# Arm 0 is the base contract: identical text to the system this builds on, so a
# task that never fails keeps byte-identical prompts.
CONTRACT_BASE = "base"
CONTRACT_ANSWER_FIRST = "answer_first"
CONTRACT_ANSWER_ONLY = "answer_only"
CONTRACT_ARMS = (CONTRACT_BASE, CONTRACT_ANSWER_FIRST, CONTRACT_ANSWER_ONLY)

# Exploration is gated on measured failure, not on a schedule.
MIN_ARM_OBS = 6
MIN_FAILURE_RATE = 0.20
EXPLORE_EVERY = 3
TIE_BAND = 0.05

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

# Answer-first: the answer is emitted before the deliberation, so a response that
# runs long or stops early still carries a scorable attempt.
PROMPT_ANSWER_FIRST = """Solve the problem below using the reference material learned from previously solved cases.

{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}- Decide first, then justify. Write "final_answer" BEFORE "reasoning", and keep
  "reasoning" to at most two sentences.
- Respond with this JSON object and nothing else

{{"final_answer": "[your answer]", "reasoning": "[at most two sentences]"}}"""

# Answer-only: no deliberation is requested at all. Same retrieved context.
PROMPT_ANSWER_ONLY = """Solve the problem below using the reference material learned from previously solved cases.

{shortlist_section}{inventory_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{inventory_rule}{shortlist_rule}- Answer directly. Do not deliberate, do not explain, do not show working.
- Reply with only this JSON object

{{"final_answer": "[your answer]"}}"""

CONTRACT_TEMPLATES = {
    CONTRACT_BASE: PROMPT,
    CONTRACT_ANSWER_FIRST: PROMPT_ANSWER_FIRST,
    CONTRACT_ANSWER_ONLY: PROMPT_ANSWER_ONLY,
}

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


class _ContractLedger:
    """Per-arm record of whether the contract produced a scorable attempt.

    The objective is the rate at which a contract yields an extractable answer,
    because an unparseable response is a guaranteed miss - measured at 0 correct
    out of 89 fallback firings on open-ended generation and 1 out of 47 on the
    multi-label task. Realised accuracy only breaks ties between contracts that
    produce attempts equally often, since on a hard task the accuracy signal is
    far too sparse to rank arms by itself.
    """

    def __init__(self) -> None:
        self.attempts: Counter = Counter()
        self.produced: Counter = Counter()
        self.correct: Counter = Counter()
        self.t = 0

    def observe_call(self, arm: str, produced: bool) -> None:
        self.attempts[arm] += 1
        self.produced[arm] += int(produced)

    def observe_outcome(self, arm: str, was_correct: bool) -> None:
        self.correct[arm] += int(was_correct)

    def produce_rate(self, arm: str) -> float | None:
        n = self.attempts[arm]
        return self.produced[arm] / n if n else None

    def accuracy(self, arm: str) -> float:
        n = self.attempts[arm]
        return self.correct[arm] / n if n else 0.0

    def base_is_failing(self) -> bool:
        """Only a contract that demonstrably loses answers justifies exploring."""
        n = self.attempts[CONTRACT_BASE]
        if n < MIN_ARM_OBS:
            return False
        return 1.0 - (self.produced[CONTRACT_BASE] / n) >= MIN_FAILURE_RATE

    def choose(self) -> str:
        self.t += 1
        if not self.base_is_failing():
            return CONTRACT_BASE
        unseen = [a for a in CONTRACT_ARMS if self.attempts[a] < MIN_ARM_OBS]
        if unseen and self.t % EXPLORE_EVERY == 0:
            return unseen[0]
        known = [a for a in CONTRACT_ARMS if self.attempts[a] >= MIN_ARM_OBS]
        if not known:
            return CONTRACT_BASE
        best = max(self.produce_rate(a) or 0.0 for a in known)
        tied = [a for a in known if (self.produce_rate(a) or 0.0) >= best - TIE_BAND]
        return max(tied, key=lambda a: (self.accuracy(a), a == CONTRACT_BASE))

    def recovery_arm(self, used: str) -> str | None:
        """The strictest contract that is not the one that just failed."""
        return None if used == CONTRACT_ANSWER_ONLY else CONTRACT_ANSWER_ONLY

    def dump(self) -> dict[str, Any]:
        return {
            "attempts": dict(self.attempts),
            "produced": dict(self.produced),
            "correct": dict(self.correct),
            "t": self.t,
        }

    def load(self, data: dict[str, Any]) -> None:
        self.attempts = Counter(data.get("attempts", {}))
        self.produced = Counter(data.get("produced", {}))
        self.correct = Counter(data.get("correct", {}))
        self.t = int(data.get("t", 0))


class OutputContractBandit(MemorySystem):
    """Two retrieval channels plus an online bandit over the output contract and
    a recovery ladder that keeps the retrieved context."""

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
        self._ledger = _ContractLedger()
        self._recoveries = Counter()

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

    def _sections(self, input: str) -> tuple[dict[str, str], list, list]:
        """Everything the contract templates need, built once per query."""
        query = self._template.strip(input)
        picked = self._retrieve(query)
        shortlist = self._shortlist(query) if self._shortlist_enabled() else []
        return {
            "shortlist_section": self._format_shortlist(shortlist),
            "inventory_section": self._format_inventory(shortlist),
            "examples_section": self._format_examples(picked),
            "input": input,
            "inventory_rule": INVENTORY_RULE if self._inventory_enabled() else "",
            "shortlist_rule": SHORTLIST_RULE if shortlist else "",
        }, picked, shortlist

    def _ask(self, arm: str, sections: dict[str, str]) -> tuple[str, str]:
        response = self.call_llm(CONTRACT_TEMPLATES[arm].format(**sections))
        return extract_json_field(response, "final_answer"), response

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        sections, picked, shortlist = self._sections(input)
        arm = self._ledger.choose()
        answer, response = self._ask(arm, sections)
        self._ledger.observe_call(arm, bool(answer.strip()))

        # Recovery ladder. The old behaviour dropped straight to the bare input
        # with no memory at all; measured across four systems that step produced
        # 1 correct answer in 136 firings. The first rung keeps the retrieved
        # context and only tightens the contract.
        ladder = []
        if not answer.strip():
            rung = self._ledger.recovery_arm(arm)
            if rung is not None:
                ladder.append(rung)
                answer, response = self._ask(rung, sections)
                # Not credited to the arm: a recovery call only ever runs on a
                # query that already defeated one contract, so scoring it here
                # would bias the arm downwards against first-call evidence.
                self._recoveries["rung_produced"] += int(bool(answer.strip()))
        if not answer.strip():
            ladder.append("bare")
            response = self.call_llm(TERSE_RETRY.format(input=input))
            answer = extract_json_field(response, "final_answer")
            if not answer.strip():
                answer = response.strip()[:200]
        if ladder:
            self._recoveries[ladder[-1]] += 1

        canonical = self._canonicalize(answer)
        return canonical, {
            "full_response": response,
            "num_retrieved": len(picked),
            "shortlist": shortlist,
            "inventory_on": self._inventory_enabled(),
            "snapped": canonical != answer,
            "contract": arm,
            "ladder": ladder,
            "retried": bool(ladder),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            meta = r.get("metadata") or {}
            arm = meta.get("contract")
            if arm in CONTRACT_ARMS:
                self._ledger.observe_outcome(arm, bool(r.get("was_correct")))
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

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self._template.strip(self.examples[-1]["input"])
        shortlist = self._shortlist(probe) if self._shortlist_enabled() else []
        return (
            len(self._format_shortlist(shortlist))
            + len(self._format_inventory(shortlist))
            + len(self._format_examples(self._retrieve(probe)))
        )

    def get_state(self) -> str:
        return json.dumps(
            {"examples": self.examples, "contract_ledger": self._ledger.dump()},
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
        self._ledger = _ContractLedger()
        self._recoveries = Counter()
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
        self._ledger.load(data.get("contract_ledger", {}))
