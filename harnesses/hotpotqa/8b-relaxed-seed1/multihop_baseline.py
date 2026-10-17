"""Baseline: the GEPA HotpotQA seed program (LangProBe HoverMultiHop with an answering last hop).

Structure (4 LLM calls, 2 retrievals, k=7), all modules are DSPy ChainOfThought with DEFAULT instructions:
    hop1: passages_1 = retrieve(question, k=7)
          summary_1  = CoT("question, passages -> summary")
    hop2: query_2    = CoT("question, summary_1 -> query")
          passages_2 = retrieve(query_2, k=7)
          summary_2  = CoT("question, context, passages -> summary")     # context = summary_1
    final: answer    = CoT("question, summary_1, summary_2 -> answer")

The prompts are rendered in DSPy's ChatAdapter format (see domains/hotpotqa/dspy_format.py).
"""

from __future__ import annotations

from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.summarize1 = Module(inputs=["question", "passages"], outputs=["summary"])
        self.create_query_hop2 = Module(inputs=["question", "summary_1"], outputs=["query"])
        self.summarize2 = Module(inputs=["question", "context", "passages"], outputs=["summary"])
        self.generate_answer = Module(inputs=["question", "summary_1", "summary_2"], outputs=["answer"])

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        # HOP 1
        hop1_docs = ctx.tools.retrieve(question, k=self.K)
        summary_1 = self.summarize1(ctx.llm, tag="summarize1", question=question, passages=hop1_docs)["summary"]
        # HOP 2
        query_2 = self.create_query_hop2(ctx.llm, tag="query_hop2", question=question, summary_1=summary_1)["query"]
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        summary_2 = self.summarize2(ctx.llm, tag="summarize2", question=question, context=summary_1, passages=hop2_docs)["summary"]
        # ANSWER
        answer = self.generate_answer(ctx.llm, tag="answer", question=question, summary_1=summary_1, summary_2=summary_2)["answer"]
        return {"answer": answer, "query_2": query_2, "summary_1": summary_1, "summary_2": summary_2}
