"""Answer probe verification: fact-check the draft with a fresh retrieval instead of with itself.

Parent: `evidence_pool_router` (decomposition, routing and the first answer call are byte-identical).

Measured on the parent's traces, 53 of its 95 errors have the gold answer string sitting INSIDE the final
answer prompt: a competing span won. An earlier attempt asked the model to enumerate the competing spans and
choose among them in context; it fixed 1 item and broke 9, because showing a model alternatives makes it
abandon answers it had right.

This never shows the model an alternative. It takes the draft and asks the SEARCH INDEX about it: the draft
string, joined with the distinctive terms the question itself supplies, becomes one extra query. The documents
that come back were selected by a mechanism with no stake in the draft, so they are independent of the
reasoning that produced it -- if the draft really is the thing the question describes, they corroborate it,
and if a different entity holds that property, they say so in their own words.

A single verification call reads the question, the routed evidence and these check documents, and either
confirms or replaces the draft. The replacement is then filtered deterministically: it is accepted only if the
new string actually occurs in evidence the run has read. The draft therefore survives unless outside evidence
positively contradicts it, which is the asymmetry the in-context version lacked. Retrieval and model calls are
purely additive: the parent leaves 6 of 12 retrievals and 8 of 12 calls unspent.
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

DECOMPOSE_INSTRUCTIONS = """Plan a keyword search for a question that can only be answered by combining
two different source documents.

Write 3 search queries, one per line, no numbering and no explanation:
1. the most specific description in the question, phrased as the title/topic of the document that describes it;
2. a short keyword-only variant of that same target using distinctive proper nouns, dates or numbers only
   (no question words, no stop words) -- it should retrieve the same document by different wording;
3. the second document: the entity or property that the question ultimately asks about, phrased as a topic.

The retriever matches words, not meaning, so prefer the exact names and rare terms that would appear in the
document, and never write a query as a question."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. In `reasoning`, decide which chain of facts the question needs, then
check which links of that chain the passages already establish.

Produce:
- `useful`: the numbers of the passages that state a needed fact, most important first, comma separated.
  Include a passage only if it carries a fact the answer depends on; write `none` if no passage does.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property
  must still be looked up). Write `none` if every needed fact is present.
- `next_query`: a keyword query for that missing fact, anchored on the exact entity name as it appears in the
  passages, plus the property being sought. Leave it empty when nothing is missing. Never repeat a query whose
  results are already listed.
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`."""

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


VERIFY_INSTRUCTIONS = """A draft short answer to the question has already been produced from the evidence
passages. Independent check passages have since been fetched by searching for the draft itself together with
the terms the question specifies.

Use the check passages as an outside test of the draft. In `reasoning`, state the property the question demands
of its answer, then say whether the check passages show the draft holding that property, show a different
entity holding it, or say nothing either way.

Produce:
- `verdict`: `confirmed` if the draft still is the best-supported short answer, or if the check passages are
  simply uninformative. `corrected` ONLY if the passages positively establish that a different string is the
  answer. Silence is not a contradiction: when in doubt, confirm.
- `answer`: the draft copied unchanged when the verdict is `confirmed`; otherwise the corrected short answer,
  copied with the wording the passages use.

Output a bare answer either way: an entity, name, title, date, number or verdict. No sentence, no explanation.
Never abstain and never reply that the information is missing."""


_QTERM = re.compile(r'"[^"]{2,60}"|\b(?:[A-Z][\w\u00c0-\u024f\'\u2019.\-]+|\d[\d,.]*)\b')
_STOP = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is", "was", "were", "are",
         "the", "a", "an", "of", "in", "on", "and", "or", "do", "does", "did", "both", "that", "this"}


def _norm_key(text):
    """Lowercase, punctuation-free, article-free key, matching how answers are compared."""
    out = []
    for tok in re.split(r"[^\w\u00c0-\u024f]+", str(text or "").lower()):
        if tok and tok not in {"a", "an", "the"}:
            out.append(tok)
    return " ".join(out)


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3          # router passes (each may add one retrieval)
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380      # chars of each passage shown to the router
    CHECK_K = 6             # breadth of the independent verification probe
    CHECK_RETRIEVALS = 2    # retrievals reserved for verification, on top of MAX_RETRIEVALS

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.route = Module(
            inputs=["question", "passages"],
            outputs=["useful", "missing", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={"answer": "the bare short answer"},
        )
        self.verify = Module(
            inputs=["question", "draft", "passages", "checks"],
            outputs=["verdict", "answer"],
            instructions=VERIFY_INSTRUCTIONS,
            descs={
                "draft": "the short answer produced from the evidence passages",
                "passages": "the evidence the draft was produced from",
                "checks": "independently retrieved passages about the draft",
                "verdict": "confirmed or corrected",
                "answer": "the bare short answer",
            },
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
            queries.insert(0, question)      # the plain question is a reliable fallback query
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
        pool: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def retrievals_left():
            return self.MAX_RETRIEVALS + self.CHECK_RETRIEVALS - retrievals

        def bump():
            nonlocal retrievals
            retrievals += 1

        def gather(query: str) -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=self.K):
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(passage)
                added += 1
            return added

        def gather_docs(query, k):
            """Retrieve for verification: returns only passages the pool has not already seen."""
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals_left() <= 0:
                return []
            used_queries.add(q.lower())
            bump()
            fresh = []
            for passage in ctx.tools.retrieve(q, k=k):
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                fresh.append(passage)
            return fresh

        # 1. decomposed opening sweep
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q)

        # 2-4. route over the pool, following up on what is missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            shown = [f"{p[: self.POOL_SNIPPET]}" for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[:8]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            next_query = routed.get("next_query", "")
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(next_query):
                break     # follow-up query brought nothing new: stop paying for rounds

        evidence = selected or pool[:8]
        ctx.note("pool routed", n_pool=len(pool), n_selected=len(selected), retrievals=retrievals, missing=missing[:200])

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- and show the WHOLE pool, since selection may have dropped the right passage.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        # Outside test: ask the index about the draft, then let one call confirm or correct it.
        checked = self._probe_verify(ctx, question, final, evidence, gather_docs)
        if checked and checked != final:
            ctx.note("verification override", before=final[:120], after=checked[:120])
            final = checked

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "queries": queries, "missing": missing}

    # -- verification -------------------------------------------------------------------------
    def _probe_verify(self, ctx, question, draft, evidence, gather_docs):
        """Retrieve independently about the draft and let one call confirm or replace it."""
        if not draft or _NON_ANSWER.match(draft) or draft.lower() in {"yes", "no"}:
            return draft
        query = self._check_query(question, draft)
        fresh = gather_docs(query, self.CHECK_K)
        if not fresh:
            return draft
        out = self.verify(
            ctx.llm, tag="verify", question=question, draft=draft,
            passages=[p[: self.POOL_SNIPPET] for p in evidence],
            checks=[p[: self.POOL_SNIPPET] for p in fresh],
        )
        verdict = str(out.get("verdict", "")).strip().lower()
        revised = self._tidy(out.get("answer", ""), question)
        if not revised or _NON_ANSWER.match(revised):
            return draft
        if _norm_key(revised) == _norm_key(draft):
            return draft
        if not verdict.startswith("correct"):
            return draft
        # Deterministic filter: a correction must be a string this run has actually read.
        read = _norm_key(" ".join(evidence) + " " + " ".join(fresh))
        if _norm_key(revised) and _norm_key(revised) in read:
            return revised
        return draft

    @staticmethod
    def _check_query(question, draft):
        """The draft plus the distinctive terms the question supplies (proper nouns, numbers, quoted spans)."""
        terms = []
        drafted = set(_norm_key(draft).split())
        for tok in _QTERM.findall(question or ""):
            key = _norm_key(tok)
            if not key or key in _STOP or key in drafted:
                continue
            if key not in {_norm_key(t) for t in terms}:
                terms.append(tok.strip('"'))
            if len(terms) >= 8:
                break
        return " ".join([draft] + terms)[:300]
