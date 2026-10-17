"""Confusion-cluster verification memory.

Instead of a flat exemplar pool consumed in one shot, memory is organised
around where the model actually goes wrong:

* an answer-cluster store  - solved cases bucketed by their answer string, so
  evidence for a specific candidate answer can be assembled on demand;
* a confusion graph        - counts of (drafted answer -> true answer) built
  from the model's own past mistakes, which yields a per-answer reliability
  estimate and a shortlist of the answers that draft is usually mistaken for.

Prediction is two-stage. A draft is produced from a lean similar-case prompt.
The confusion graph then decides whether that draft is trustworthy; only when
it is not does a second pass run, and that pass is not a re-ask but a
discrimination prompt organised by candidate answer, each candidate backed by
its own evidence cases.

The verification stage is gated on the answer space being reusable at all
(measured online). On open-ended generation tasks the confusion graph carries
no signal, verification never fires, and the retrieval budget is spent on
breadth instead of depth.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

NGRAM_SIZES = (2, 3)
K_DRAFT = 6  # lean draft prompt when a verification pass may follow
K_FULL = 12  # full breadth when verification is unavailable
PER_ANSWER_CAP = 2
MAX_EXEMPLAR_CHARS = 9000

MIN_EXAMPLES_FOR_GATE = 15
MIN_REUSE_RATE = 0.25

MAX_CANDIDATES = 6
EVIDENCE_PER_CANDIDATE = 2
MAX_EVIDENCE_CHARS = 6000
NEAR_ANSWER_THRESHOLD = 0.35
RELIABILITY_THRESHOLD = 0.9
MIN_OBSERVATIONS = 3  # a reliability estimate needs support before it can veto

DRAFT_PROMPT = """Solve the problem below using the similar solved cases as reference.

{examples_section}**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown in the cases above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

VERIFY_PROMPT = """A draft answer has been produced for the problem below. Past \
experience shows this kind of draft is sometimes confused with other answers, so \
check it against the evidence.

**Problem:**
{input}

**Draft answer:** {draft}

**Candidate answers, each with cases that were genuinely resolved that way:**

{clusters}

**Instructions:**
- Compare the problem against the evidence cases for each candidate
- The draft is usually right: keep it unless the evidence clearly favours another candidate
- Your answer must be exactly one of the candidate strings, copied character for character
- Respond in JSON format

{{"reasoning": "[which evidence decided it]", "final_answer": "[one candidate string]"}}"""

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


class ConfusionClusterVerify(MemorySystem):
    """Draft, then discriminate among the answers that draft is confused with."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.answer_counts: Counter = Counter()
        self.clusters: dict[str, list[int]] = defaultdict(list)
        # drafted answer -> Counter of what the true answer turned out to be
        self.confusion: dict[str, Counter] = defaultdict(Counter)
        self._index = _Index()
        self._answer_grams: dict[str, Counter] = {}
        self._answer_norms: dict[str, str] = {}

    # ---------------- gating ----------------

    def _reuse_rate(self) -> float:
        total = sum(self.answer_counts.values())
        if not total:
            return 0.0
        singletons = sum(1 for c in self.answer_counts.values() if c == 1)
        return 1.0 - singletons / total

    def _verify_enabled(self) -> bool:
        return (
            len(self.examples) >= MIN_EXAMPLES_FOR_GATE
            and self._reuse_rate() >= MIN_REUSE_RATE
        )

    # ---------------- retrieval ----------------

    def _retrieve(self, query: str, k: int) -> list[dict[str, str]]:
        used: Counter = Counter()
        picked: list[dict[str, str]] = []
        for _score, doc_id in self._index.search(query):
            ex = self.examples[doc_id]
            if used[ex["target"]] >= PER_ANSWER_CAP:
                continue
            picked.append(ex)
            used[ex["target"]] += 1
            if len(picked) >= k:
                break
        return picked

    def _format_examples(self, picked: list[dict[str, str]]) -> str:
        if not picked:
            return ""
        parts, total = [], 0
        for ex in reversed(picked):
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total + len(part) > MAX_EXEMPLAR_CHARS:
                continue
            parts.append(part)
            total += len(part) + 2
        return "**Similar solved cases:**\n\n" + "\n\n".join(parts) + "\n\n"

    # ---------------- confusion structure ----------------

    def _canonical_key(self, answer: str) -> str | None:
        """Map an answer onto the known answer string it actually expresses.

        Models wrap answers in the task's output markup, which would otherwise
        make every draft look like an unseen answer and destroy the confusion
        graph's statistics. Only used as a memory key - the emitted text is
        left alone unless verification replaces it.
        """
        norm = _normalize(answer)
        if not norm:
            return None
        best = None
        for known, known_norm in self._answer_norms.items():
            if known_norm and known_norm in norm:
                if best is None or len(known_norm) > len(self._answer_norms[best]):
                    best = known
        return best

    def _reliability(self, draft: str) -> tuple[float, int, int]:
        """(P(draft was the true answer), times drafted, distinct true answers)."""
        seen = self.confusion.get(draft)
        if not seen:
            return 0.0, 0, 0
        total = sum(seen.values())
        return seen[draft] / total, total, len(seen)

    def _nearest_answers(self, draft: str, k: int) -> list[tuple[float, str]]:
        grams = _answer_ngrams(draft)
        scored = sorted(
            ((_dice(grams, g), a) for a, g in self._answer_grams.items()),
            key=lambda x: (-x[0], x[1]),
        )
        return scored[:k]

    def _should_verify(self, draft: str) -> bool:
        if not self._verify_enabled():
            return False
        if not draft.strip():
            return False
        key = self._canonical_key(draft)
        rel, times, distinct = self._reliability(key or draft)
        if times >= MIN_OBSERVATIONS and distinct >= 2 and rel < RELIABILITY_THRESHOLD:
            return True  # this draft has genuinely been ambiguous before
        if key is None:
            near = self._nearest_answers(draft, 1)
            # off-vocabulary, but a known answer is close: likely a wording slip
            return bool(near) and near[0][0] >= NEAR_ANSWER_THRESHOLD
        return False

    def _candidates(self, draft: str) -> list[str]:
        """Rivals for this draft: what it has been mistaken for, or - when the
        draft is not a known answer at all - the known answers it reads like."""
        key = self._canonical_key(draft)
        ordered = [key or draft]
        for answer, _c in self.confusion.get(key or draft, Counter()).most_common(3):
            if answer not in ordered:
                ordered.append(answer)
        if key is None:
            # Surface-form similarity only means something when the draft is
            # off-vocabulary; between two known answers it is semantic noise.
            for score, answer in self._nearest_answers(draft, 4):
                if score > 0 and answer not in ordered:
                    ordered.append(answer)
        return ordered[:MAX_CANDIDATES]

    def _format_clusters(self, query: str, candidates: list[str]) -> str:
        """One block per candidate, holding its most query-relevant cases."""
        ranked = {doc_id: rank for rank, (_s, doc_id) in enumerate(self._index.search(query))}
        blocks, total = [], 0
        for answer in candidates:
            ids = sorted(
                self.clusters.get(answer, []),
                key=lambda i: ranked.get(i, 10**9),
            )[:EVIDENCE_PER_CANDIDATE]
            lines = [f"### {answer}"]
            if not ids:
                lines.append("(no recorded case with this answer)")
            for doc_id in ids:
                ex = self.examples[doc_id]
                lines.append(f"- {ex.get('raw_question', ex['input'])}")
            block = "\n".join(lines)
            if total + len(block) > MAX_EVIDENCE_CHARS:
                break
            blocks.append(block)
            total += len(block) + 2
        return "\n\n".join(blocks)

    def _snap_to_candidate(self, answer: str, candidates: list[str], draft: str) -> str:
        if answer in candidates:
            return answer
        if not answer.strip():
            return draft
        grams = _answer_ngrams(answer)
        scored = sorted(
            ((_dice(grams, _answer_ngrams(c)), c) for c in candidates),
            key=lambda x: (-x[0], x[1]),
        )
        return scored[0][1] if scored and scored[0][0] >= NEAR_ANSWER_THRESHOLD else draft

    # ---------------- interface ----------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        k = K_DRAFT if self._verify_enabled() else K_FULL
        picked = self._retrieve(input, k)
        prompt = DRAFT_PROMPT.format(
            examples_section=self._format_examples(picked), input=input
        )
        response = self.call_llm(prompt)
        draft = extract_json_field(response, "final_answer")

        retried = False
        if not draft.strip():
            retried = True
            response = self.call_llm(TERSE_RETRY.format(input=input))
            draft = extract_json_field(response, "final_answer")
            if not draft.strip():
                draft = response.strip()[:200]

        verified = False
        answer = draft
        if self._should_verify(draft):
            candidates = self._candidates(draft)
            if len(candidates) > 1:
                verified = True
                vresponse = self.call_llm(
                    VERIFY_PROMPT.format(
                        input=input,
                        draft=candidates[0],
                        clusters=self._format_clusters(input, candidates),
                    )
                )
                answer = self._snap_to_candidate(
                    extract_json_field(vresponse, "final_answer"), candidates, draft
                )
                response = vresponse

        return answer, {
            "full_response": response,
            "draft": draft,
            "verified": verified,
            "retried": retried,
            "num_retrieved": len(picked),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            doc_id = len(self.examples)
            self.examples.append(ex)
            self._index.add(ex.get("raw_question", ex["input"]))
            self.clusters[target].append(doc_id)
            self.answer_counts[target] += 1
            if target not in self._answer_grams:
                self._answer_grams[target] = _answer_ngrams(target)
                self._answer_norms[target] = _normalize(target)
            # The confusion graph is keyed on the draft: that is what the
            # trigger sees at prediction time.
            draft = (r.get("metadata") or {}).get("draft") or r.get("prediction", "")
            if draft.strip():
                self.confusion[self._canonical_key(draft) or draft][target] += 1

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self.examples[-1].get("raw_question", self.examples[-1]["input"])
        k = K_DRAFT if self._verify_enabled() else K_FULL
        return len(self._format_examples(self._retrieve(probe, k)))

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "confusion": {k: dict(v) for k, v in self.confusion.items()},
            },
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self.answer_counts = Counter()
        self.clusters = defaultdict(list)
        self.confusion = defaultdict(Counter)
        self._index = _Index()
        self._answer_grams = {}
        self._answer_norms = {}
        for ex in data.get("examples", []):
            doc_id = len(self.examples)
            self.examples.append(ex)
            self._index.add(ex.get("raw_question", ex["input"]))
            self.clusters[ex["target"]].append(doc_id)
            self.answer_counts[ex["target"]] += 1
            if ex["target"] not in self._answer_grams:
                self._answer_grams[ex["target"]] = _answer_ngrams(ex["target"])
                self._answer_norms[ex["target"]] = _normalize(ex["target"])
        for draft, counts in data.get("confusion", {}).items():
            self.confusion[draft] = Counter(counts)
