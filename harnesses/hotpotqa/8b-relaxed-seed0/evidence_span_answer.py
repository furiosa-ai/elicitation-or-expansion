"""Evidence-grounded span answering.

Mechanism change vs the summary-chain baseline: the answerer never sees a summary. Retrieved
passages are pooled and de-duplicated by title into a numbered evidence table handed to the final
module raw, and that module is forced to (a) commit to an answer *type*, (b) quote the supporting
sentence and (c) copy the answer as a verbatim span out of that quote (or emit a literal yes/no).
A deterministic normalizer then strips any sentence scaffolding that survived, and a single
tightening call fires only when the span still looks like a sentence.

The second hop is anchored on an explicitly named bridge entity and is always spent on that
entity's own article, because the article about the intermediate entity is what states the
attribute the question actually asks for.

Rationale: a chain of free-text summaries loses the exact surface form of the answer, and an
unconstrained answer field produces explanatory sentences, which score zero under exact match.

Calls: 2-3 LLM, 2 retrievals.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+|short\s+|correct\s+)?answer\s*(?:is|:)\s*|"
    r"^(?:it|that|this)\s+(?:is|was)\s+",
    re.IGNORECASE,
)
_YES = re.compile(r"^\s*(?:yes|yeah|true|correct|both)\b", re.IGNORECASE)
_TRAIL = re.compile(r"\s*[.;,]\s*$")


class Harness(BaseHarness):
    K = 10
    MAX_ANSWER_WORDS = 12

    PLAN_INSTR = (
        "You are gathering evidence for a question that needs facts from two different articles.\n"
        "`known_facts`: the concrete facts in `passages` that bear on the question, naming entities exactly "
        "as the passages spell them.\n"
        "`bridge_entity`: the name of the entity whose own article must be read next. For a question that "
        "routes through an intermediate entity, that is the person, work, place or organisation the passages "
        "just revealed and that stands between the question and the answer. For a question comparing two "
        "named entities, it is whichever of the two the passages cover least well. Give the name only.\n"
        "`next_query`: a search query that retrieves that entity's article. It must never be empty and must "
        "never repeat the question: write `bridge_entity`'s name, optionally followed by one or two words "
        "naming the property still needed. Even when `passages` look sufficient, that entity's own article "
        "states the fact more precisely than a passing mention does, so always name it."
    )

    EXTRACT_INSTR = (
        "Answer the question using only the numbered `evidence` passages.\n"
        "`answer_type` must be exactly one of: yesno, person, date, number, other.\n"
        "`supporting_quote` must be the exact sentence fragment, copied character for character from one "
        "numbered passage, that states the answer.\n"
        "`answer` rules, applied strictly:\n"
        "- If `answer_type` is yesno, `answer` is the single word yes or no. Nothing else.\n"
        "- Otherwise `answer` is a verbatim span copied out of `supporting_quote`: the name, title, date or "
        "quantity itself. Spell it exactly as the passage spells it, neither abbreviating nor expanding it, "
        "and keep parenthetical or middle names only if the passage writes them inside that same span.\n"
        "- If `answer_type` is date, copy the whole date the passage gives, with day and month when it gives "
        "them; do not shorten a full date to its year.\n"
        "- Never write a sentence, never restate the question, never add units, reasons or hedging words.\n"
        "- If the evidence is incomplete, still give the single most likely span from the evidence. Never "
        "answer that the information is unavailable."
    )

    TIGHTEN_INSTR = (
        "`verbose_answer` states the answer inside a sentence. Return only the answer itself: the exact "
        "name, title, date or quantity, copied word for word from `supporting_quote`. If the question asks "
        "whether something is the case, return only yes or no. Output at most a few words, no final "
        "punctuation, no explanation."
    )

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question", "passages"],
            outputs=["known_facts", "bridge_entity", "next_query"],
            instructions=self.PLAN_INSTR,
        )
        self.extract = Module(
            inputs=["question", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=self.EXTRACT_INSTR,
        )
        self.tighten = Module(
            inputs=["question", "supporting_quote", "verbose_answer"],
            outputs=["answer"],
            instructions=self.TIGHTEN_INSTR,
        )

    # -- evidence pool ---------------------------------------------------------------------
    @staticmethod
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    def _pool(self, *groups: list[str]) -> list[str]:
        seen: set[str] = set()
        pool: list[str] = []
        for group in groups:
            for p in group:
                key = self._title_of(p)
                if key and key not in seen:
                    seen.add(key)
                    pool.append(p)
        return pool

    @staticmethod
    def _one_line(raw: str) -> str:
        return (raw or "").strip().split("\n")[0].strip().strip('"“”').strip()

    # -- answer normalisation --------------------------------------------------------------
    def _clean(self, raw: str, answer_type: str) -> str:
        ans = self._one_line(raw).strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if "yesno" in (answer_type or "").lower() and (_YES.match(ans) or re.match(r"^\s*(?:no|nope|false|neither|not)\b", ans, re.IGNORECASE)):
            return "yes" if _YES.match(ans) else "no"
        # a leading "Yes, ..." / "No, ..." is sentence scaffolding around a yes/no answer
        if re.match(r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE):
            return "yes" if _YES.match(ans) else "no"
        ans = _TRAIL.sub("", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        hop1 = ctx.tools.retrieve(question, k=self.K)
        plan = self.plan(ctx.llm, tag="plan", question=question, passages=hop1)

        # The second hop must land on a NEW article: fall back to the bridge entity's bare name
        # rather than re-issuing the question, which would return the same passages.
        bridge = self._one_line(plan.get("bridge_entity", ""))
        next_query = self._one_line(plan.get("next_query", ""))
        if not next_query or next_query.lower() == question.strip().lower():
            next_query = bridge or question
        hop2 = ctx.tools.retrieve(next_query, k=self.K)

        pool = self._pool(hop1, hop2)
        evidence = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        ctx.note(f"evidence pool: {len(pool)} passages", bridge_entity=bridge, next_query=next_query)

        out = self.extract(ctx.llm, tag="extract", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""), out.get("answer_type", ""))

        # A span that is still a sentence means the copy constraint was ignored: re-ask for the
        # span alone, showing the model its own quote so the second pass is pure extraction.
        if not answer or len(answer.split()) > 6:
            out2 = self.tighten(
                ctx.llm,
                tag="tighten",
                question=question,
                supporting_quote=out.get("supporting_quote", ""),
                verbose_answer=out.get("answer", ""),
            )
            tightened = self._clean(out2.get("answer", ""), out.get("answer_type", ""))
            if tightened and (not answer or len(tightened.split()) <= max(6, len(answer.split()))):
                answer = tightened

        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "supporting_quote": out.get("supporting_quote", ""),
            "bridge_entity": bridge,
            "next_query": next_query,
            "n_evidence": len(pool),
        }
