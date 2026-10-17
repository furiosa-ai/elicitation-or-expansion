"""Wide shallow scan of a deep candidate pool, narrow deep read for the answerer.

Every system so far retrieves ~7-8 hits per query and hands the same passages both to the
selector and to the answerer, so retrieval breadth and reading breadth are the same number.
That couples two requirements that pull in opposite directions: recall wants many candidates,
the answerer wants few and clean (widening what the answerer reads has already been shown to
cost accuracy on items whose evidence was complete).

This harness decouples them:

  1. one planning call decomposes the question into standalone search queries (as the parent);
  2. every query is issued at the maximum depth the index allows, and the hits are merged into a
     title-keyed pool ordered RANK-MAJOR (all rank-1 hits, then all rank-2 hits, ...) so the pool
     order reflects retrieval confidence across queries rather than the order the queries ran in;
  3. a SCAN call sees the whole deep pool as one compact line per article -- title plus the first
     clause of the lead only -- which is cheap enough that four or five times as many candidates
     fit in the same prompt, and returns the handful of articles worth opening;
  4. only those articles are expanded to their full text for the answerer, so the answering
     context stays exactly as small as the parent's;
  5. if the scan reports the evidence incomplete it issues one deep follow-up query and rescans.

Hypothesis: the articles needed for the second fact of a multi-fact question are usually retrieved
by the queries already being issued, just below the shallow cut-off, and a title-level scan is
enough to spot them. Recall should rise with the answering context size held constant.
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


def _scan_line(passage: str, chars: int) -> str:
    """One compact line: the title plus just enough of the lead to tell articles apart."""
    title, _, body = passage.partition(" | ")
    body = " ".join(body.split())
    if len(body) > chars:
        cut = body[:chars]
        sp = cut.rfind(" ")
        body = cut[: sp if sp > chars // 2 else chars]
    return f"{title.strip()} | {body}"


class Harness(BaseHarness):
    K_DEEP = 20         # depth per query: as deep as the index allows
    MAX_QUERIES = 3     # decomposed queries issued alongside the raw question
    MAX_SCANS = 2       # adaptive hops
    SCAN_CAP = 56       # articles shown to the scan (one short line each)
    SCAN_CHARS = 140    # characters of lead text per scan line
    READ_CAP = 8        # articles expanded to full text for the answerer

    PLAN_INSTR = (
        "Break the question into the separate facts that must be looked up in an encyclopaedia.\n"
        "Write `queries` as one short keyword search query per line (2 or 3 lines, no numbering). "
        "Each query must stand on its own: name the entity it is about explicitly instead of using "
        "a pronoun or a description that depends on another query.\n"
        "If one fact can only be looked up after another is known, still write the query for the "
        "part that is already named."
    )
    SCAN_INSTR = (
        "You are scanning a long list of candidate encyclopaedia articles, shown as "
        "`id: title | first words of the article`. The list is ranked by a keyword search engine, "
        "so the article you need may sit well down the list; judge every line, not just the top ones.\n"
        "`read_ids`     : the ids of the articles worth opening in full, most useful first, at most "
        "6, comma separated. Pick the article each part of the question is really about, including "
        "the one that supplies the second fact; prefer a title that matches the specific descriptors "
        "in the question (year, nationality, role, type of work) over a merely popular near-match.\n"
        "`linking_fact` : the fact these articles establish that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); `unknown` if none.\n"
        "`sufficient`   : `yes` only if the chosen articles plainly cover every fact needed to "
        "answer; otherwise `no`.\n"
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
        self.scan = Module(
            inputs=["question", "candidates"],
            outputs=["read_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.SCAN_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _merge_rank_major(pool: list[str], seen: set[str], batches: list[list[str]]) -> None:
        """Interleave several ranked hit lists so rank 1 of every query precedes rank 2 of any."""
        depth = max((len(b) for b in batches), default=0)
        for rank in range(depth):
            for batch in batches:
                if rank >= len(batch):
                    continue
                p = batch[rank]
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

        # 2. deep pooled retrieval, merged rank-major -----------------------------------------
        batches = [ctx.tools.retrieve(question, k=self.K_DEEP)]
        for q in queries:
            batches.append(ctx.tools.retrieve(q, k=self.K_DEEP))
        self._merge_rank_major(pool, seen, batches)

        # 3-5. shallow scan of the deep pool, adaptive follow-up ------------------------------
        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_SCANS):
            shown = pool[: self.SCAN_CAP]
            listing = [f"{i + 1}: {_scan_line(p, self.SCAN_CHARS)}" for i, p in enumerate(shown)]
            out = self.scan(ctx.llm, tag=f"scan{step + 1}", question=question, candidates=listing)
            ids = self._parse_ids(out.get("read_ids", ""), len(shown))[: self.READ_CAP] or list(
                range(1, min(len(shown), 5) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in ("unknown", "none", "n/a"):
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_SCANS - 1:
                break
            nq = _split_lines(out.get("next_query", ""), 1)
            if not nq:
                break
            followups.append(nq[0])
            picked = [pool[i - 1] for i in ids]
            before = len(pool)
            self._merge_rank_major(pool, seen, [ctx.tools.retrieve(nq[0], k=self.K_DEEP)])
            if len(pool) == before:  # follow-up added nothing new
                break
            # articles already chosen stay at the front so the scan cap cannot drop them
            keep = {_title_of(p).lower() for p in picked}
            pool = picked + [p for p in pool if _title_of(p).lower() not in keep]
            ctx.note("deep follow-up hop", query=nq[0], pool_size=len(pool))

        # 6. answer from the full text of the few articles that were opened -------------------
        shown = pool[: self.SCAN_CAP]
        evidence = [shown[i - 1] for i in ids[: self.READ_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[:5]
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
            "n_read": len(evidence),
        }
