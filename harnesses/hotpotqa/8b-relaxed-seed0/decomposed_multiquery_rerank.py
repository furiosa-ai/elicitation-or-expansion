"""Decomposed multi-query retrieval with LLM reranking and an explicit sufficiency gate.

Mechanism change vs the linear summary chain: the first hop is not one query but a decomposition
of the question into several independent, entity-anchored searches issued side by side. Their
results are pooled and de-duplicated by title, the model then *reranks* that pool (naming the
passages that actually bear on the question) and states what is still missing; only if something
is missing does it spend a second retrieval round, on title-style queries anchored to the
intermediate entity it just found. The answerer sees the reranked passages, never a summary.

Rationale: a single query built from the whole question retrieves the article for whichever
entity dominates the wording, so the other required article is systematically missed; and a
second query conditioned on a lossy summary inherits that miss.

Calls: 3 LLM, 3-5 retrievals.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+|short\s+)?answer\s*(?:is|:)\s*|^(?:it|that|this)\s+(?:is|was)\s+",
    re.IGNORECASE,
)
_YES = re.compile(r"^\s*(?:yes|yeah|true|correct|both)\b", re.IGNORECASE)
_NUM = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NONANSWER = re.compile(
    r"^(?:unknown|unclear|undetermined|not\s+(?:determined|specified|available|stated|found)|n/?a|none|"
    r"cannot\b|can't\b|insufficient)\b",
    re.IGNORECASE,
)


class Harness(BaseHarness):
    K1 = 8
    K2 = 8
    MAX_FIRST_QUERIES = 3
    MAX_FOLLOWUPS = 2
    MAX_KEPT = 9
    MIN_KEPT = 5
    MAX_ANSWER_WORDS = 12

    DECOMPOSE_INSTR = (
        "A question of this kind can only be settled by combining facts from two different articles, "
        "and one search phrased from the whole question tends to find only one of them.\n"
        "Write `queries`: two or three search queries, one per line, no numbering, no explanation.\n"
        "- Give each distinct named entity, work or event mentioned in the question its own query, "
        "worded the way that entity's own article would be titled.\n"
        "- If the question describes an entity instead of naming it, write one query out of that "
        "description using the words an article about it would use.\n"
        "- Do not repeat the question verbatim as a query, and do not write questions: write titles "
        "and noun phrases."
    )

    ASSESS_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant "
        "first, at most six numbers, separated by commas. Judge relevance by whether the passage is about "
        "an entity the question actually asks about, not by superficial word overlap.\n"
        "`missing_info`: in one line, the fact the question still needs and that no listed passage states. "
        "Write exactly `none` if the passages together already settle the question.\n"
        "`followup_queries`: if something is missing, one or two queries, one per line. Each must START with the "
        "name of the intermediate entity the passages just revealed (the person, work, place or organisation "
        "that links the question to the missing fact), because that entity's own article is what must be "
        "retrieved next, and may add one or two words naming the missing property. Write titles and noun "
        "phrases, never questions. Write exactly `none` when nothing is missing."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.assess = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "missing_info", "followup_queries"],
            instructions=self.ASSESS_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”').strip()
            if not line or line.lower() in {"none", "n/a", "null"}:
                continue
            if len(line) > 300:
                line = line[:300]
            if line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    def _add(self, pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = self._title_of(p)
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip()
        ans = ans.strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # ROUND 1: decomposed, entity-anchored searches issued side by side.
        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = queries[: self.MAX_FIRST_QUERIES - 1] + [question] if queries else [question]

        pool: list[str] = []
        seen: set[str] = set()
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K1))
        ctx.note(f"round1: {len(queries)} queries -> {len(pool)} unique passages", queries=queries)

        # RERANK + SUFFICIENCY GATE
        numbered = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        judged = self.assess(ctx.llm, tag="assess", question=question, passages=numbered)
        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]

        followups = self._lines(judged.get("followup_queries", ""), self.MAX_FOLLOWUPS)
        missing = (judged.get("missing_info", "") or "").strip().lower()
        sufficient = (not followups) or missing.startswith("none")

        # ROUND 2: only when the gate says something is still missing.
        if not sufficient:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for q in followups:
                self._add(round2, seen2, ctx.tools.retrieve(q, k=self.K2))
            ctx.note(f"round2: {len(followups)} followups -> {len(round2)} new passages", followups=followups)
            kept = kept + round2
        # The reranker is aggressive and sometimes keeps a single pair; top the pool back up with the
        # best-scoring passages it passed over, so one bad selection cannot starve the answerer.
        if len(kept) < self.MIN_KEPT:
            have = {self._title_of(p) for p in kept}
            for p in pool:
                if self._title_of(p) not in have:
                    kept.append(p)
                    have.add(self._title_of(p))
                if len(kept) >= self.MIN_KEPT:
                    break

        evidence = [f"{i + 1}. {p}" for i, p in enumerate(kept)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        # Abstaining scores the same as guessing wrong, so a refusal is replaced by the subject of the
        # passage the reranker ranked first.
        if not answer or _NONANSWER.match(answer):
            answer = kept[0].split("|", 1)[0].strip() if kept else answer
        return {
            "answer": answer,
            "queries": queries,
            "followups": followups,
            "sufficient": sufficient,
            "n_evidence": len(kept),
        }
