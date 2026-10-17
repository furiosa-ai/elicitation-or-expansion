"""Answer-slot audit: catch drafts that cannot possibly answer the question, then re-derive the slot.

Diagnosis of the frontier's 20 full-coverage failures (both required documents retrieved, f1 = 0): they are not
wording problems.  Six answer the WRONG SLOT of the question -- the bridge entity instead of its property, the
film instead of the co-star asked for, the writers instead of the director, the reversed direction of "who was
replaced by X" -- and four emit `yes`/`no` for a question that asks `who`/`which`/`how`.  One cause is shared:
the drafting call reads question and passages together, so whatever the passages make salient overwrites what
the question actually asked for.

Two properties of a draft show it cannot be an answer, both decidable from the QUESTION ALONE, with no model
call and no evidence:

  1. it is `yes`/`no`, yet the question carries a wh-word and does not open with an auxiliary verb -- such a
     question asks for a value, and no polarity can fill it;
  2. it repeats, verbatim, a string the question itself already supplies -- restating the question's own words
     instead of answering it.  Suppressed for questions that offer alternatives (`or`, `between`, `either`),
     where naming one of the offered options is exactly right.

Measured over the parent's 300 search-set answers, rule 1 fires 4 times and rule 2 fires 5, and 8 of those 9
items are currently wrong.

The gate is the whole point.  Four consecutive candidates (iters 1-4) tried to improve answers by changing what
the final call reads, and every one of them lost 3-4 points on questions the parent already solved: passing a
target spec, a demo bank or a longer evidence block into the drafting call costs more collateral than it repairs.
So here the drafting call, the retrieval and the router are byte-identical to the parent and NOTHING happens on
the ~97% of items where no rule fires.  When one does fire, ONE call is spent that must first restate, from the
question, the slot the answer has to fill (`asked_for`) and only then fill it -- the slot contract is derived
inside the repair, so it costs nothing when it is not needed.  If the repair breaks the same rule again, the
original draft is kept, which protects the legitimate case where the answer really is a name the question printed.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

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

# -- audit vocabulary ----------------------------------------------------------------------------
_WH = re.compile(r"\b(who|whom|whose|what|which|where|when|why|how)\b", re.IGNORECASE)
# a question offering alternatives: repeating one of them is correct behaviour, never a violation
_CHOICE = re.compile(r"\b(or|between|either)\b", re.IGNORECASE)

_RULE_POLARITY = (
    "this question asks for a specific value, not whether something is or is not the case, so `yes`/`no` "
    "cannot be an answer to it"
)
_RULE_ECHO = (
    "this draft only repeats a name the question itself already supplies, so it restates part of the question "
    "instead of answering it"
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

RESPEC_INSTRUCTIONS = """A first attempt to answer this question produced a draft that cannot be a valid answer,
for the reason given in `problem`. Your job is to find the answer it should have given.

Work in this order:
- `asked_for`: read the QUESTION ONLY, ignoring the passages and the draft, and state in at most eight words
  the one thing the question demands, as a noun phrase ("the director of the sequel", "the country both writers
  published in", "the number of murders"). Follow the question's grammar exactly: if it asks who was replaced BY
  someone, the slot is the person replaced, not the replacement; if it asks for a property of a named entity,
  the slot is the property, not the entity. Do not put a value here, only the description of the slot.
- `answer`: now read the passages and fill exactly that slot. The rejected draft is shown so you do not repeat
  it -- it is not the answer, and neither is any name the question itself already supplied, unless the question
  explicitly offers a choice between named options.

Output the answer bare: an entity, name, title, date, number or verdict copied with the wording the passages
use, with no sentence, no explanation and no trailing punctuation. Never abstain and never reply that the
information is missing -- commit to the best-supported filler of the slot. If, and only if, the question really
does ask whether something is or is not the case, output exactly `yes` or `no`."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3          # router passes (each may add one retrieval)
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380      # chars of each passage shown to the router

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
        self.respec = Module(
            inputs=["question", "rejected_draft", "problem", "passages"],
            outputs=["asked_for", "answer"],
            instructions=RESPEC_INSTRUCTIONS,
            descs={
                "rejected_draft": "the draft that was rejected",
                "problem": "why it cannot be an answer",
                "asked_for": "the slot the answer must fill, read off the question alone",
                "answer": "the bare short answer filling that slot",
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

    # -- the audit ----------------------------------------------------------------------------
    @staticmethod
    def _audit(draft: str, question: str) -> str:
        """The violated rule, or '' when the draft could be an answer. Deterministic; costs no call."""
        norm = normalize_text(draft or "")
        if len(norm) < 2:
            return ""
        if norm in ("yes", "no"):
            if _WH.search(question or "") and not _BOOL_QUESTION.match(question or ""):
                return _RULE_POLARITY
            return ""
        if len(norm) >= 3 and not _CHOICE.search(question or ""):
            if (" " + norm + " ") in (" " + normalize_text(question or "") + " "):
                return _RULE_ECHO
        return ""

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

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

        # 5. audit: only a draft that provably cannot answer the question is touched, and only then is a call spent
        problem = self._audit(final, question)
        if problem:
            out = self.respec(
                ctx.llm, tag="respec", question=question, rejected_draft=final, problem=problem,
                passages=evidence or pool[:8],
            )
            fixed = self._tidy(out.get("answer", ""), question)
            keep = bool(fixed) and not _NON_ANSWER.match(fixed) and self._audit(fixed, question) != problem
            ctx.note(
                "slot violation", problem=problem[:120], asked_for=str(out.get("asked_for", ""))[:120],
                before=final[:120], after=fixed[:120], kept=keep,
            )
            if keep:
                final = fixed

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- and show the WHOLE pool, since selection may have dropped the right passage.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "audit": problem[:80]}
