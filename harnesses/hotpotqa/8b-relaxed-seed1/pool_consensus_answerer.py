"""Two-view consensus: answer twice from evidence chosen by two different mechanisms, arbitrate only on disagreement.

Parent: `evidence_pool_router` (decomposition and routing are byte-identical).

The parent hands the answerer one evidence set: the passages its router selected. That selection is a single
point of failure, and it is made by the same model that then answers, so a misreading of what the question
needs is never caught -- 11 of the parent's errors have the gold string in the pool but not in the selection,
and many more have a competing span winning inside the selection.

Here the pool is read through two independent views. View A is the router's selection, as before. View B is
built without the model at all: every pooled passage is scored by how much of the question's distinctive
vocabulary it carries, with a bonus for terms no other passage covers, so B is assembled to COVER the question
rather than to match one reading of it. The two views overlap only where both mechanisms agree a passage
matters.

Each view is answered independently and blind to the other. When the two agree -- the common case -- that
answer stands and nothing further is spent: agreement is a cheap correctness signal, and an answer produced
twice from differently chosen evidence is unlikely to be a stray span. Only the disagreements, which
concentrate the errors, reach a third call, and that call sees the union of both views and is asked which
reading a passage actually STATES. The arbiter's output is then filtered deterministically against the text
the run has read, so a disagreement cannot invent a third answer out of nothing.
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


CONSENSUS_INSTRUCTIONS = """Two independent readings of the evidence produced two different short answers to
this question. Exactly one reading, at most, is right.

In `reasoning`, name the property the question demands of its answer. Then find, for each candidate, the
passage sentence that would have to state that the candidate holds that property -- and check whether such a
sentence is really there. A candidate that is merely mentioned near the right topic is not supported; the
passages must state the link the question asks about. If a multi-step question needs a bridge entity, resolve
that entity first and check which candidate the second step actually yields.

Produce `answer`: the candidate the passages state, or a different short answer if the passages state one and
neither candidate is right. Output the bare answer only -- an entity, name, title, date, number or verdict,
copied with the wording the passages use. No sentence, no explanation. If the question asks whether something
is or is not the case, output exactly `yes` or `no`. Never abstain."""


_WORD = re.compile(r"[\w\u00c0-\u024f]+")
_STOP = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is", "was", "were", "are",
         "be", "been", "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "do", "does",
         "did", "both", "that", "this", "it", "its", "by", "with", "from", "as", "have", "has", "had",
         "his", "her", "their", "he", "she", "they", "you", "name", "called", "known"}


def _norm_key(text):
    """Lowercase, punctuation-free, article-free key, matching how answers are compared."""
    out = []
    for tok in _WORD.findall(str(text or "").lower()):
        if tok not in {"a", "an", "the"}:
            out.append(tok)
    return " ".join(out)


def _content_terms(text):
    return {t for t in _WORD.findall(str(text or "").lower()) if t not in _STOP and len(t) > 2}


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3          # router passes (each may add one retrieval)
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380      # chars of each passage shown to the router
    VIEW_SIZE = 8           # passages in the model-free coverage view
    MERGED_MAX = 12         # passages shown to the arbiter

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
        self.arbitrate = Module(
            inputs=["question", "candidate_a", "candidate_b", "passages"],
            outputs=["answer"],
            instructions=CONSENSUS_INSTRUCTIONS,
            descs={
                "candidate_a": "short answer from the routed evidence",
                "candidate_b": "short answer from the coverage evidence",
                "passages": "the union of both evidence views",
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

    @staticmethod
    def _coverage_view(question, pool, size):
        """Pick passages that jointly cover the question's vocabulary. No model call, no router influence.

        Greedy set cover: each pick is the passage carrying the most question terms not yet covered, so rare
        terms -- the ones that identify the second document -- pull in a passage even when common ones do not.
        """
        wanted = _content_terms(question)
        if not wanted or not pool:
            return []
        terms = [_content_terms(p) & wanted for p in pool]
        chosen, covered = [], set()
        for _ in range(min(size, len(pool))):
            best, best_gain = -1, 0
            for i, ts in enumerate(terms):
                if i in chosen:
                    continue
                gain = 2 * len(ts - covered) + len(ts & covered)
                if gain > best_gain:
                    best, best_gain = i, gain
            if best < 0:
                break
            chosen.append(best)
            covered |= terms[best]
        return [pool[i] for i in chosen]

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

        # View B: model-free coverage selection over the same pool, built to span the question's vocabulary.
        coverage = self._coverage_view(question, pool, self.VIEW_SIZE)

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if coverage and final and not _NON_ANSWER.match(final):
            alt_raw = self.answer(ctx.llm, tag="answer_coverage", question=question, passages=coverage).get("answer", "")
            alt = self._tidy(alt_raw, question)
            if alt and not _NON_ANSWER.match(alt) and _norm_key(alt) != _norm_key(final):
                merged = evidence + [p for p in coverage if p not in evidence]
                settled = self.arbitrate(
                    ctx.llm, tag="arbitrate", question=question,
                    candidate_a=final, candidate_b=alt,
                    passages=[p[: self.POOL_SNIPPET] for p in merged[: self.MERGED_MAX]],
                ).get("answer", "")
                settled = self._tidy(settled, question)
                read = _norm_key(" ".join(merged))
                if settled and not _NON_ANSWER.match(settled) and (
                    _norm_key(settled) in {_norm_key(final), _norm_key(alt)} or _norm_key(settled) in read
                ):
                    ctx.note("views disagreed", view_a=final[:80], view_b=alt[:80], settled=settled[:80])
                    final = settled
                else:
                    ctx.note("arbitration rejected", view_a=final[:80], view_b=alt[:80], raw=str(settled)[:80])
            else:
                ctx.note("views agreed", answer=final[:80])

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- and show the WHOLE pool, since selection may have dropped the right passage.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "queries": queries, "missing": missing}
