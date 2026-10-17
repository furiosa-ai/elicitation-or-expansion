"""Typed verbatim-span answering.

Retrieval is byte-identical to the seed program (2 hops, k=7, same summary/query modules), so any
delta is attributable to the ANSWER POLICY, which is rebuilt:

    seed:  answer = CoT("question, summary_1, summary_2 -> answer")     # free-form, often a sentence
    here:  evidence = deduplicated RAW passages from both hops
           (answer_type, draft) = CoT("question, summary_1, summary_2, evidence -> answer_type, answer")
           final                = Predict("question, answer_type, draft_answer, evidence -> answer")
           final                = deterministic tidy (markup, leading fillers, boolean verdicts)

Two ideas drive it: (1) the graded answer is a short span, so the answer must be *extracted* rather
than *composed*; (2) the span must be copied verbatim from a passage, because summaries paraphrase
names and values into forms that no longer match the source wording.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_LEADING_FILLER = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|short\s+|correct\s+)?answer(?:\s+is)?\s*[:\-]?\s*", re.IGNORECASE
)
_YESNO_PREFIX = re.compile(r"^\s*(yes|no)\b\s*[,.:;!\-]*", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_MARKUP = re.compile(r"[*_`]+")

EXTRACT_INSTRUCTIONS = """You extract the final short answer to a multi-step question from evidence passages.

In `reasoning`, first chain the facts across passages: resolve any intermediate entity the question
refers to only indirectly, then find the passage that states the fact actually being asked for.

Then produce:
- `answer_type`: one of `yes_no`, `person`, `place`, `organization`, `work_title`, `date`, `number`, `other`.
  Use `yes_no` only when the question asks whether something is or is not the case.
- `answer`: the answer and nothing else. Copy it verbatim from the evidence passage that states it, keeping
  exactly the form written there (for a person, the full name as the passage introduces it; for a work, its
  exact title). Never write a sentence, never restate the question, never add explanation, parentheses,
  qualifiers or trailing punctuation. If `answer_type` is `yes_no`, `answer` must be exactly `yes` or `no`."""

FINALIZE_INSTRUCTIONS = """Reduce a draft answer to the shortest string that answers the question exactly.

Return only the bare answer: the entity, name, title, date, number or verdict itself. Strip every
surrounding sentence, clause, restatement of the question, explanation, source attribution and trailing
punctuation. Keep the spelling and word form used in the evidence; do not shorten a name that the evidence
writes out in full, and do not expand one it does not. Drop descriptive words that the question did not ask
for. If `answer_type` is `yes_no`, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.summarize1 = Module(inputs=["question", "passages"], outputs=["summary"])
        self.create_query_hop2 = Module(inputs=["question", "summary_1"], outputs=["query"])
        self.summarize2 = Module(inputs=["question", "context", "passages"], outputs=["summary"])
        self.extract = Module(
            inputs=["question", "summary_1", "summary_2", "evidence"],
            outputs=["answer_type", "answer"],
            instructions=EXTRACT_INSTRUCTIONS,
            descs={
                "evidence": "raw source passages, each `Title | abstract`",
                "answer_type": "one of yes_no, person, place, organization, work_title, date, number, other",
                "answer": "the bare answer span, copied verbatim from the evidence",
            },
        )
        self.finalize = Module(
            inputs=["question", "answer_type", "draft_answer", "evidence"],
            outputs=["answer"],
            instructions=FINALIZE_INSTRUCTIONS,
            chain_of_thought=False,
            descs={"answer": "the bare answer, no sentence and no explanation"},
        )

    # -- deterministic cleanup ----------------------------------------------------------------
    @staticmethod
    def _tidy(text: str, question: str, answer_type: str) -> str:
        ans = _MARKUP.sub("", str(text or "")).strip()
        ans = ans.split("\n")[0].strip()
        ans = _LEADING_FILLER.sub("", ans).strip()
        ans = ans.strip().strip('"').strip("'").strip("«»").strip()
        atype = str(answer_type or "").strip().lower()
        boolean = "yes_no" in atype or "yes/no" in atype or "boolean" in atype
        m = _YESNO_PREFIX.match(ans)
        if m and (boolean or _BOOL_QUESTION.match(question or "") or len(ans.split()) > 1):
            return m.group(1).lower()
        while ans.endswith((".", ",", ";", ":")):
            ans = ans[:-1].strip()
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        # --- retrieval: unchanged from the seed program -----------------------------------
        hop1_docs = ctx.tools.retrieve(question, k=self.K)
        summary_1 = self.summarize1(ctx.llm, tag="summarize1", question=question, passages=hop1_docs)["summary"]
        query_2 = self.create_query_hop2(ctx.llm, tag="query_hop2", question=question, summary_1=summary_1)["query"]
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        summary_2 = self.summarize2(
            ctx.llm, tag="summarize2", question=question, context=summary_1, passages=hop2_docs
        )["summary"]

        # --- evidence pool: raw passages, deduplicated, order preserved -------------------
        evidence: list[str] = []
        seen: set[str] = set()
        for passage in list(hop1_docs) + list(hop2_docs):
            key = str(passage).split(" | ")[0].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            evidence.append(passage)

        # --- typed extraction, then compression to a bare span ---------------------------
        drafted = self.extract(
            ctx.llm,
            tag="extract",
            question=question,
            summary_1=summary_1,
            summary_2=summary_2,
            evidence=evidence,
        )
        answer_type = drafted.get("answer_type", "other")
        draft = drafted.get("answer", "")
        candidate = self._tidy(draft, question, answer_type)

        final = candidate
        # Only pay for the compression call when the draft is not already a bare span.
        if len(candidate.split()) > 5 or not candidate:
            compressed = self.finalize(
                ctx.llm,
                tag="finalize",
                question=question,
                answer_type=answer_type,
                draft_answer=candidate or draft,
                evidence=evidence,
            ).get("answer", "")
            compressed = self._tidy(compressed, question, answer_type)
            if compressed:
                final = compressed

        if not final:
            final = self._tidy(draft, question, answer_type) or str(draft).strip()
        ctx.note("typed span", answer_type=str(answer_type)[:40], draft=str(draft)[:200], final=final[:200])
        return {"answer": final, "answer_type": answer_type, "draft": draft, "query_2": query_2}
