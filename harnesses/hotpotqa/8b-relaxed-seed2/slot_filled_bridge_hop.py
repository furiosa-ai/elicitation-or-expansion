"""Deferred second hop: the follow-up query is a PLANNED template whose slot is filled, deterministically,
with several evidence-grounded candidate names.

Failure this attacks (measured on the parent's traces): a third of the parent's searches are wasted because
the planner fires every sub-query up front, including ones whose subject the question never names.  Those
come out as literal placeholders (`Cast of [film title]`, `Number of degrees at [University Name]`,
`career start of [Comedian's Name]`) and retrieve noise.  The single follow-up query the sufficiency gate
is allowed to invent then commits to ONE guess of the unknown entity, and when that guess is wrong the
whole hop is spent on the wrong article.

Mechanism (different from the parent's "decompose -> pool -> one invented follow-up"):

  1. the planner separates what can be looked up NOW (parts whose subject the question names) from the
     part that can only be looked up LATER, and writes the later part ONCE, as a query template with the
     literal token ``___`` where the unknown name goes;
  2. only anchored queries are ever sent to the index -- any query still containing a slot or a bracketed
     placeholder is dropped, never retrieved;
  3. a resolver reads the pool and returns the ids worth reading plus up to three CANDIDATE fillers for
     ``___``, each copied verbatim from the leads, most likely first -- candidates, not one guess;
  4. the second hop is built by deterministic string substitution of each filler into the planned template
     (plus a bare-name lookup of the filler), so the relational wording of the question is preserved and
     the query can never contain an unresolved slot;
  5. the answer is read off the raw passages, with the resolver's selection first and the newly fetched
     passages appended behind a hard cap, so widening retrieval cannot drown the evidence that was already
     good (the failure mode of unconditioned link expansion).
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

SLOT = "___"
NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\[[^\]]{2,}\]|<[^>]{2,}>|\{[^}]{2,}\}|_{2,}")
NONE_RE = re.compile(r"^\s*(?:none|n/?a|unknown|nothing|-+)\s*\.?\s*$", re.IGNORECASE)


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip("«»").strip()
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
    K_PLAN = 8          # hits per anchored query
    K_SLOT = 6          # hits per slot-filled query
    K_NAME = 4          # hits per bare-name lookup
    MAX_READY = 3       # anchored queries
    MAX_FILLERS = 3     # candidate names explored
    MAX_RETRIEVE = 11   # stay one under the domain cap
    POOL_CAP = 30
    SELECT_CAP = 6      # passages kept from the resolver's own selection
    NEW_CAP = 4         # passages the second hop may add
    MAX_RESOLVE = 2

    PLAN_INSTR = (
        "Split the question into what can be looked up straight away and what can only be looked up "
        "afterwards.\n"
        "`ready_queries`: one short search query per line (1 to 3 lines, no numbering) for every part of "
        "the question whose subject the question actually NAMES. Write each one the way the article's "
        "title reads, followed by the couple of words from the question that say what kind of thing it "
        "is. Never write a query for something the question only describes.\n"
        "`pending_fact`: if answering also needs a fact about a thing the question only DESCRIBES (a thing "
        "that must be identified first), write the search query for that fact with the exact token ___ "
        "standing in for the name you do not know yet, for example `___ founding date`. Write `none` if "
        "every part of the question is already named."
    )
    RESOLVE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia leads "
        "shown as `id: title | beginning of the article`.\n"
        "`relevant_ids`: the ids worth reading in full, most useful first, at most 6, comma separated.\n"
        "`fillers`: the names that could be the thing ___ stands for in `pending_fact`, one per line, at "
        "most 3, most likely first. Copy each name exactly as it is written in the leads (an article "
        "title, or a name mentioned inside an article). Write `none` if `pending_fact` is `none` or the "
        "leads suggest no candidate. Do not invent a name that is not written in the leads.\n"
        "`sufficient`: `yes` only if the leads already state every fact needed to answer; otherwise `no`."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question"], outputs=["ready_queries", "pending_fact"], instructions=self.PLAN_INSTR
        )
        self.resolve = Module(
            inputs=["question", "pending_fact", "pool"],
            outputs=["relevant_ids", "fillers", "sufficient"],
            instructions=self.RESOLVE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "bridge"], outputs=["answer"], instructions=self.ANSWER_INSTR
        )

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> list[str]:
        fresh: list[str] = []
        for p in passages:
            key = _title_of(p).lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)
                fresh.append(p)
        return fresh

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    @staticmethod
    def _usable(query: str) -> bool:
        """A query is only sent to the index if it has no unresolved slot left in it."""
        q = (query or "").strip()
        return bool(q) and SLOT not in q and not PLACEHOLDER.search(q) and not NONE_RE.match(q)

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        budget = self.MAX_RETRIEVE

        # 1. plan: anchored queries now, the rest as a template -----------------------------------
        plan_out = self.plan(ctx.llm, tag="plan", question=question)
        ready = [q for q in _split_lines(plan_out.get("ready_queries", ""), self.MAX_READY + 2) if self._usable(q)]
        ready = ready[: self.MAX_READY]
        pending_raw = (plan_out.get("pending_fact", "") or "").strip().splitlines()
        pending = (pending_raw[0].strip().strip('"') if pending_raw else "")[:200]
        if NONE_RE.match(pending) or PLACEHOLDER.search(pending.replace(SLOT, "")):
            pending = "" if NONE_RE.match(pending) else PLACEHOLDER.sub(SLOT, pending)
        has_slot = SLOT in pending

        # 2. anchored retrieval only --------------------------------------------------------------
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K_PLAN))
        budget -= 1
        for q in ready:
            if budget <= 3:
                break
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_PLAN))
            budget -= 1

        # 3. resolve: selection + candidate fillers, not a single guess ---------------------------
        ids: list[int] = []
        fillers: list[str] = []
        sufficient = False
        shown = pool[: self.POOL_CAP]
        for step in range(self.MAX_RESOLVE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.resolve(
                ctx.llm,
                tag=f"resolve{step + 1}",
                question=question,
                pending_fact=pending or "none",
                pool=table,
            )
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
                range(1, min(len(shown), self.SELECT_CAP) + 1)
            )
            sufficient = bool(AFFIRM.match(out.get("sufficient", "")))
            fillers = [f for f in _split_lines(out.get("fillers", ""), self.MAX_FILLERS) if not NONE_RE.match(f)]
            fillers = [f[:80] for f in fillers if 2 <= len(f) <= 80]
            break  # one resolve call; the second slot is reserved for the hop below

        selected = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not selected:
            selected = shown[: self.SELECT_CAP]

        # 4. deterministic slot filling: explore several candidate names --------------------------
        new_passages: list[str] = []
        hop_queries: list[str] = []
        n_explore = 1 if sufficient else self.MAX_FILLERS
        for filler in fillers[:n_explore]:
            if budget < 1 or len(new_passages) >= self.NEW_CAP:
                break
            if has_slot:
                q = pending.replace(SLOT, filler)
                if self._usable(q):
                    hop_queries.append(q)
                    fresh = self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_SLOT))
                    budget -= 1
                    new_passages.extend(fresh[:2])
            if budget < 1 or len(new_passages) >= self.NEW_CAP:
                break
            if self._usable(filler):
                hop_queries.append(filler)
                fresh = self._add(pool, seen, ctx.tools.retrieve(filler, k=self.K_NAME))
                budget -= 1
                # the article ABOUT the candidate is the one worth reading
                exact = [p for p in fresh if _title_of(p).lower().startswith(filler.lower()[:40])]
                new_passages.extend((exact or fresh)[:1])

        # 5. answer from raw passages, selection first --------------------------------------------
        evidence = selected + [p for p in new_passages[: self.NEW_CAP] if p not in selected]
        bridge = fillers[0] if fillers else "unknown"
        if hop_queries:
            ctx.note("slot-filled hop", queries=hop_queries, n_new=len(new_passages))
        ans = self.answer(
            ctx.llm, tag="answer", question=question, evidence=evidence, bridge=bridge
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return {
            "answer": ans,
            "ready_queries": ready,
            "pending_fact": pending,
            "fillers": fillers,
            "hop_queries": hop_queries,
            "sufficient": sufficient,
            "n_selected": len(evidence),
        }
