"""Two-resolution evidence pool: recall-wide retrieval, rank-fused triage, text-narrow answering.

The frontier router builds a NARROW pool (4 queries x k=8, ~24 unique documents) and reads it at full
resolution.  Trace analysis of that system shows the pool itself -- not the answer policy -- is now the
binding constraint: on 94/300 search-set items at least one required document never enters the pool, and
accuracy there is 0.39 against 0.82 when both are present.  Replaying the very same queries at the
retriever's maximum k recovers a required document on 28 of those items, so the missing evidence is
sitting just below the cutoff the narrow pool imposes.

The obstacle to simply widening is context: 60+ raw abstracts cannot be read at full resolution, and a
flat 60-item list has no ordering the reader can trust.  So this harness splits the pool into two
resolutions instead of one:

    WIDE  (recall): every query is issued at maximum k; documents are fused across queries with
          reciprocal rank fusion (model-free), which promotes the documents that several independent
          query formulations agree on and pushes single-query noise down.
    SHORT (triage): the fused pool is shown to the model as a ranked list of `Title | lead snippet`
          lines -- enough to identify a document, a fraction of the tokens of its full abstract.
    FULL  (reading): only the triaged documents are re-expanded to their complete abstracts for the
          answer call, so the reader's context stays as small and as verbatim as before.

Rank fusion also drives the follow-up hop: the anchor entity for the next query is taken from the
triaged documents, and every new result list is fused into the same pool, never discarded.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_BULLET = re.compile(r"^\s*(?:[-*•]|\[?\d+[.)\]]?)\s*")
_INDEX = re.compile(r"\d+")
_LEADING_FILLER = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|short\s+)?answer(?:\s+is)?\s*[:\-]?\s*", re.IGNORECASE
)
_YESNO_PREFIX = re.compile(r"^\s*(yes|no)\b\s*[,.:;!\-]*", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_MARKUP = re.compile(r"[*_`]+")
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)
# A hedge that merely *starts* like an answer ("None of the passages mention ...") also scores zero, so it has
# to reach the forced-commit stage as well -- while a bare `no` stays a legitimate verdict.
_HEDGE_PREFIX = re.compile(
    r"^(?:none\s+of\b|no\s+(?:information|passage|mention|document|evidence|such)\b|not\s+(?:enough|clear)\b"
    r"|the\s+(?:passages?|documents?|text)\b|insufficient\b|unable\s+to\b|i\s+(?:cannot|can't|do\s+not)\b)",
    re.IGNORECASE,
)
_SENT_END = re.compile(r"(?<=[.!?])\s")

DECOMPOSE_INSTRUCTIONS = """Plan a keyword search for a question that can only be answered by combining
two different source documents.

Write 3 search queries, one per line, no numbering and no explanation:
1. the most specific description in the question, phrased as the title/topic of the document that describes it;
2. a short keyword-only variant of that same target using distinctive proper nouns, dates or numbers only
   (no question words, no stop words) -- it should retrieve the same document by different wording;
3. the second document: the entity or property that the question ultimately asks about, phrased as a topic.

The retriever matches words, not meaning, so prefer the exact names and rare terms that would appear in the
document, and never write a query as a question."""

TRIAGE_INSTRUCTIONS = """You are triaging a long list of search results down to the few documents worth
reading in full.

Each line is `number. Title | opening words of the document`. The list is long and ranked by how many
independent queries agreed on the document, so useful documents can appear anywhere in it. You are seeing
only the opening of each document, so judge a line by whether the DOCUMENT it names is one the reasoning
chain has to read, not by whether the fact is visible in the snippet.

In `reasoning`, state the chain of documents the question requires: the entity the question describes, then
the entity or property that depends on it. Then scan the list for the title of each link.

Produce:
- `keep`: the numbers of the documents worth reading in full, most important first, comma separated, at most
  six. Prefer the document whose title IS the entity in question over documents that merely mention it.
  Write `none` if the list contains no document from the chain.
- `missing`: the single link of the chain that no listed document covers, named concretely (the specific
  entity whose property must still be looked up). Write `none` if every link is covered.
- `next_query`: a keyword query for that missing link, anchored on the exact entity name as it appears in the
  listed titles or snippets, plus the property being sought. Leave empty when nothing is missing. Never
  repeat a query whose results are already listed.
- `sufficient`: `yes` if the kept documents can already answer the question, otherwise `no`."""

ANSWER_INSTRUCTIONS = """Answer the question from the passages with the shortest possible answer.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 20                  # retriever maximum: recall stage is as wide as the tool allows
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 9
    RRF_C = 20.0            # reciprocal-rank-fusion damping
    TRIAGE_N = 50           # documents shown to the triage stage
    TRIAGE_SNIPPET = 170    # chars of lead text per triage line
    N_READ = 6              # documents re-expanded to full abstracts for the answer call
    READ_FALLBACK = 8

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.triage = Module(
            inputs=["question", "candidates"],
            outputs=["keep", "missing", "next_query", "sufficient"],
            instructions=TRIAGE_INSTRUCTIONS,
            descs={
                "candidates": "ranked search results, each `Title | opening words`",
                "keep": "comma separated document numbers, at most six, or `none`",
                "missing": "the chain link still uncovered, or `none`",
                "next_query": "keyword query for the missing link",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={"answer": "the bare short answer"},
        )
        self.commit = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=COMMIT_INSTRUCTIONS,
            descs={"answer": "the bare short answer, never an abstention"},
        )

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("«»").strip()
        return " ".join(q.split())[:300]

    def _parse_queries(self, raw: str, question: str) -> list[str]:
        queries: list[str] = []
        for line in str(raw or "").splitlines():
            q = self._clean_query(line)
            if len(q) < 3 or q.lower() in {"none", "queries"}:
                continue
            if q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
            if len(queries) == 3:
                break
        if question not in queries:
            queries.insert(0, question)
        return queries[:4]

    @staticmethod
    def _parse_indices(raw: str, n: int) -> list[int]:
        picks: list[int] = []
        for tok in _INDEX.findall(str(raw or "")):
            i = int(tok)
            if 1 <= i <= n and i not in picks:
                picks.append(i)
        return picks

    @staticmethod
    def _lead(text: str, budget: int) -> str:
        """`Title | first sentence(s) up to budget chars` -- enough to identify the document."""
        title, _, body = str(text).partition(" | ")
        body = " ".join(body.split())
        if len(body) > budget:
            cut = body[:budget]
            parts = _SENT_END.split(cut)
            if len(parts) > 1 and len(parts[0]) >= budget // 3:
                cut = parts[0]
            body = cut.rstrip() + " ..."
        return f"{title.strip()} | {body}"

    @staticmethod
    def _tidy(text: str, question: str) -> str:
        ans = _MARKUP.sub("", str(text or "")).strip().split("\n")[0].strip()
        ans = _LEADING_FILLER.sub("", ans).strip()
        ans = ans.strip('"').strip("'").strip("«»").strip()
        m = _YESNO_PREFIX.match(ans)
        if m and (_BOOL_QUESTION.match(question or "") or len(ans.split()) > 1):
            return m.group(1).lower()
        while ans.endswith((".", ",", ";", ":")):
            ans = ans[:-1].strip()
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        # title -> {"text": full passage, "score": fused score, "best": best rank seen}
        docs: dict[str, dict[str, Any]] = {}
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str) -> int:
            """Retrieve at maximum width and fuse the ranking into the pool (reciprocal rank fusion)."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            fresh = 0
            for rank, passage in enumerate(ctx.tools.retrieve(q, k=self.K)):
                key = str(passage).split(" | ")[0].strip()
                slot = docs.get(key.lower())
                weight = 1.0 / (self.RRF_C + rank + 1)
                if slot is None:
                    docs[key.lower()] = {"text": str(passage), "score": weight, "best": rank}
                    fresh += 1
                else:
                    slot["score"] += weight
                    slot["best"] = min(slot["best"], rank)
            return fresh

        def ranked() -> list[dict[str, Any]]:
            return sorted(docs.values(), key=lambda d: (-d["score"], d["best"]))

        # 1. wide opening sweep: decomposed queries, each at maximum k
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q)

        # 2. triage the fused pool at low resolution, follow up on the uncovered chain link
        kept: list[str] = []
        missing = ""
        n_shown = 0
        for round_i in range(self.MAX_ROUNDS):
            pool = ranked()[: self.TRIAGE_N]
            n_shown = len(pool)
            lines = [self._lead(d["text"], self.TRIAGE_SNIPPET) for d in pool]
            out = self.triage(ctx.llm, tag=f"triage{round_i + 1}", question=question, candidates=lines)
            picks = self._parse_indices(out.get("keep", ""), len(pool))
            if picks:
                kept = [pool[i - 1]["text"] for i in picks[: self.N_READ]]
            missing = str(out.get("missing", "")).strip()
            sufficient = "yes" in str(out.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(out.get("next_query", "")):
                break

        # 3. read the triaged documents at FULL resolution
        evidence = kept or [d["text"] for d in ranked()[: self.READ_FALLBACK]]
        ctx.note(
            "two-resolution pool",
            n_pool=len(docs),
            n_shown=n_shown,
            n_read=len(evidence),
            retrievals=retrievals,
            missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            wider = [d["text"] for d in ranked()[:12]]
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=wider).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(docs), "queries": queries, "missing": missing}
