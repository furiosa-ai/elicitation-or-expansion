"""Label-grounded retrieval memory.

Three mechanisms replace the random-sampling few-shot baseline:

1. Similarity retrieval - an incremental char n-gram TF-IDF inverted index over
   solved cases, with a per-answer cap so the retrieved set stays diverse
   instead of collapsing onto near-duplicates of one answer.
2. Answer vocabulary - a separate, very compact memory structure holding the
   distinct answer strings seen so far. It costs a fraction of the characters
   an exemplar block costs while covering far more of the answer space, and it
   pins down the exact surface form the task expects.
3. Answer projection - after generation, a free-form answer is snapped onto the
   nearest vocabulary entry. This repairs answers that are semantically right
   but lexically off the canonical form.

Mechanisms 2 and 3 only switch on when the observed answer space is actually
reusable (measured online via the answer reuse rate), so open-ended generation
tasks fall back to plain similarity retrieval and are left untouched.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Retrieval
NGRAM_SIZES = (2, 3)
TOP_K = 12
PER_ANSWER_CAP = 2
MAX_EXEMPLAR_CHARS = 9000

# Answer-space gating
MIN_EXAMPLES_FOR_GATE = 15
MIN_REUSE_RATE = 0.25
MAX_VOCAB_CHARS = 4000

# Projection
PROJ_THRESHOLD = 0.40
PROJ_MARGIN = 0.05

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{vocab_section}{examples_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{vocab_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

VOCAB_RULE = (
    "- If the correct answer is one of the listed answer strings, reproduce that "
    "string EXACTLY, character for character. Otherwise answer in the same style "
    "and level of detail as those strings.\n"
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
    """Punctuation-stripped bigrams, used for answer-string similarity."""
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


class LabelGroundedRetrieval(MemorySystem):
    """Similarity retrieval grounded in a learned answer vocabulary."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.answer_counts: Counter = Counter()
        self._index = _Index()
        self._answer_grams: dict[str, Counter] = {}
        self._answer_norms: dict[str, str] = {}

    # ---------------- answer-space gating ----------------

    def _reuse_rate(self) -> float:
        """Fraction of stored cases whose answer string was seen more than once."""
        total = sum(self.answer_counts.values())
        if not total:
            return 0.0
        singletons = sum(1 for c in self.answer_counts.values() if c == 1)
        return 1.0 - singletons / total

    def _vocab_enabled(self) -> bool:
        return (
            len(self.examples) >= MIN_EXAMPLES_FOR_GATE
            and self._reuse_rate() >= MIN_REUSE_RATE
        )

    # ---------------- retrieval ----------------

    def _retrieve(self, query: str) -> list[dict[str, str]]:
        used: Counter = Counter()
        picked: list[dict[str, str]] = []
        for _score, doc_id in self._index.search(query):
            ex = self.examples[doc_id]
            if used[ex["target"]] >= PER_ANSWER_CAP:
                continue
            picked.append(ex)
            used[ex["target"]] += 1
            if len(picked) >= TOP_K:
                break
        return picked

    def _format_examples(self, picked: list[dict[str, str]]) -> str:
        if not picked:
            return ""
        parts, total = [], 0
        # Most similar case last: it sits closest to the problem in the prompt.
        for ex in reversed(picked):
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total + len(part) > MAX_EXEMPLAR_CHARS:
                continue
            parts.append(part)
            total += len(part) + 2
        return "**Similar solved cases:**\n\n" + "\n\n".join(parts) + "\n\n"

    def _format_vocab(self) -> str:
        if not self._vocab_enabled():
            return ""
        parts, total = [], 0
        for answer, _c in self.answer_counts.most_common():
            line = f"- {answer}"
            if total + len(line) > MAX_VOCAB_CHARS:
                break
            parts.append(line)
            total += len(line) + 1
        if not parts:
            return ""
        return (
            "**Answer strings used in previously solved cases:**\n"
            + "\n".join(parts)
            + "\n\n"
        )

    # ---------------- projection ----------------

    def _contained_answer(self, answer: str) -> str | None:
        """Longest known answer string appearing verbatim inside the answer.

        Catches the common case where the model emits the right answer wrapped
        in the task's own output markup.
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

    def _project(self, answer: str) -> str:
        """Snap a free-form answer onto the nearest known answer string."""
        if not answer.strip() or not self._vocab_enabled():
            return answer
        contained = self._contained_answer(answer)
        if contained is not None:
            return contained
        grams = _answer_ngrams(answer)
        scored = sorted(
            ((_dice(grams, g), a) for a, g in self._answer_grams.items()),
            key=lambda x: (-x[0], x[1]),
        )
        if not scored or scored[0][0] < PROJ_THRESHOLD:
            return answer
        if len(scored) > 1 and scored[0][0] - scored[1][0] < PROJ_MARGIN:
            return answer
        return scored[0][1]

    # ---------------- interface ----------------

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        picked = self._retrieve(input)
        prompt = PROMPT.format(
            vocab_section=self._format_vocab(),
            examples_section=self._format_examples(picked),
            input=input,
            vocab_rule=VOCAB_RULE if self._vocab_enabled() else "",
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

        projected = self._project(answer)
        return projected, {
            "full_response": response,
            "num_retrieved": len(picked),
            "vocab_on": self._vocab_enabled(),
            "projected": projected != answer,
            "retried": retried,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)
            self._index.add(ex.get("raw_question", ex["input"]))
            self.answer_counts[target] += 1
            if target not in self._answer_grams:
                self._answer_grams[target] = _answer_ngrams(target)
                self._answer_norms[target] = _normalize(target)

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        probe = self.examples[-1].get("raw_question", self.examples[-1]["input"])
        return len(self._format_vocab()) + len(
            self._format_examples(self._retrieve(probe))
        )

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self.answer_counts = Counter()
        self._index = _Index()
        self._answer_grams = {}
        self._answer_norms = {}
        for ex in data.get("examples", []):
            self.examples.append(ex)
            self._index.add(ex.get("raw_question", ex["input"]))
            self.answer_counts[ex["target"]] += 1
            if ex["target"] not in self._answer_grams:
                self._answer_grams[ex["target"]] = _answer_ngrams(ex["target"])
                self._answer_norms[ex["target"]] = _normalize(ex["target"])
