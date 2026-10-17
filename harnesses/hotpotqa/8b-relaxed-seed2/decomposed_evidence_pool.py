"""Query decomposition into a deduplicated evidence pool with an explicit sufficiency gate.

Different retrieval architecture from the parent 2-hop pipeline. Instead of
`retrieve(question) -> summarize -> invent one follow-up query -> retrieve -> summarize`, this
harness maintains a POOL of candidate documents keyed by title:

  1. one planning call decomposes the question into several standalone search queries, one per
     fact the question needs (plus the entity that links them);
  2. every query is issued against the index and all hits are merged into a title-keyed pool, so
     the second hop no longer discards what the first hop found;
  3. a triage call reads the pool as a numbered table of titles + leads and returns (a) the ids
     that are actually relevant, (b) the linking fact it learned, (c) whether the pool answers
     the question, and (d) if not, one follow-up query anchored on the entity just discovered;
  4. the follow-up query grows the pool; triage repeats at most twice (adaptive hop count);
  5. the answer is read off the RAW selected passages, never off a lossy summary.

Rationale: with the parent pipeline both gold documents are retrieved for well under half of the
items, and accuracy collapses when they are not; summarizing hop 1 before choosing hop 2 both
loses the literal text and forces exactly one follow-up query regardless of what is missing.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip()
        if len(line) >= 3 and not line.startswith("[["):
            out.append(line[:220])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


class Harness(BaseHarness):
    K_PLAN = 8          # hits per decomposed query
    K_FOLLOW = 8        # hits per follow-up query
    MAX_QUERIES = 3     # decomposed queries issued in parallel with the raw question
    MAX_TRIAGE = 2      # adaptive hops
    POOL_CAP = 30       # candidates shown to triage
    SELECT_CAP = 10     # passages handed to the answerer

    PLAN_INSTR = (
        "Break the question into the separate facts that must be looked up in an encyclopaedia.\n"
        "Write `queries` as one short keyword search query per line (2 or 3 lines, no numbering). "
        "Each query must stand on its own: name the entity it is about explicitly instead of using "
        "a pronoun or a description that depends on another query.\n"
        "If one fact can only be looked up after another is known, still write the query for the "
        "part that is already named."
    )
    TRIAGE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia "
        "leads, shown as `id: title | beginning of the article`.\n"
        "`relevant_ids` : the ids worth reading in full, most useful first, at most 8, comma separated.\n"
        "`linking_fact` : the fact the pool establishes that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); say `unknown` if none.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`.\n"
        "`next_query`   : if not sufficient, one keyword search query for the missing fact, naming "
        "the entity from `linking_fact` explicitly; otherwise leave it empty."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = _title_of(p).lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()

        # 1. decompose -------------------------------------------------------------------------
        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )

        # 2. pooled retrieval ------------------------------------------------------------------
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K_PLAN))
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_PLAN))

        # 3-4. triage with an adaptive follow-up hop ------------------------------------------
        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_TRIAGE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.triage(ctx.llm, tag=f"triage{step + 1}", question=question, pool=table)
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
                range(1, min(len(shown), self.SELECT_CAP) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in ("unknown", "none", "n/a"):
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_TRIAGE - 1:
                break
            nq = _split_lines(out.get("next_query", ""), 1)
            if not nq:
                break
            followups.append(nq[0])
            before = len(pool)
            self._add(pool, seen, ctx.tools.retrieve(nq[0], k=self.K_FOLLOW))
            if len(pool) == before:  # follow-up added nothing new
                break
            # keep the selected evidence at the front so it survives the pool cap
            pool = [pool[i - 1] for i in ids] + [p for i, p in enumerate(pool, 1) if i not in ids]
            ctx.note("follow-up hop", query=nq[0], pool_size=len(pool))

        # 5. answer from raw passages ----------------------------------------------------------
        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]
        ans = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return {
            "answer": ans,
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "n_selected": len(evidence),
        }
