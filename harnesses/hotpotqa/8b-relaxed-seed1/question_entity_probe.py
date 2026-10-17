"""Question-side entity anchoring: guarantee one document per entity the question names.

Diagnosis of the parent router's retrieval losses (94/300 items end with only half the needed
documents, and those items convert at 46% against 82% at full coverage): the missing document is
almost always the article of an entity that the QUESTION ITSELF names by name.  The planner writes
"topic" queries that blend the entity with the property being asked about, and a word-matching
retriever then dilutes the short article titled after that entity away.  Two-entity questions are the
worst case -- one side of the pair is retrieved, the other never is.

So the mechanism here is not a smarter reformulation, it is a coverage guarantee obtained without any
model call: extract every named-entity MENTION from the question deterministically (quoted spans plus
capitalised token runs joined by name particles), and spend one bare-name retrieval per mention -- the
name alone, with no property, no question words and no other entity to compete with it.  Those probes
run FIRST, so their hits own the front of the deduplicated pool, and the planner's own queries are
still issued afterwards on top of them: coverage is only ever added, never traded away.  The retrieval
budget pays for this (the parent used under 5 of the 12 permitted searches).

Everything downstream -- the router prompt, the selection rule, the answer prompt and the forced-commit
retry -- is unchanged from the parent, so any delta is attributable to question-side entity coverage.
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

# -- deterministic mention extraction from the question -----------------------------------------
_QUOTED = re.compile(r"[\"“”«»]([^\"“”«»]{3,80})[\"“”«»]")
_TOKEN = re.compile(r"[A-Za-z0-9][\w.'&/-]*|[^\sA-Za-z0-9]")
# a token that can sit INSIDE a name without ending the capitalised run
_PARTICLE = {
    "of", "the", "in", "on", "at", "for", "to", "a", "an", "de", "del", "della", "der",
    "den", "des", "di", "do", "dos", "du", "da", "la", "le", "les", "van", "von", "y", "el",
    "bin", "ibn", "al", "ter", "op", "mac",
}
# words that are never an entity on their own even when capitalised
_STOP_ALONE = {
    "which", "what", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an",
    "this", "that", "these", "those", "both", "either", "neither", "and", "or", "but", "if",
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could",
    "will", "would", "should", "in", "on", "at", "of", "for", "to", "by", "from", "with",
    "between", "besides", "along", "after", "before", "during", "there", "it", "he", "she",
    "they", "his", "her", "their", "its", "one", "two", "name", "many", "much", "other",
    "another", "same", "different", "american", "english", "british", "born", "also", "known",
}
_ALPHA = re.compile(r"[A-Za-z]")


def _join(run: list[str]) -> str:
    return " ".join(run).replace(" ' ", " '").replace(" ’ ", " ’")


def _mentions(question: str) -> list[str]:
    """Every span of the question that looks like the title of an article, longest first."""
    text = " ".join(str(question or "").split())
    found: list[str] = []

    for m in _QUOTED.finditer(text):
        cand = m.group(1).strip(" ,;:.-")
        if len(cand) >= 3 and _ALPHA.search(cand):
            found.append(cand)

    tokens = _TOKEN.findall(text)
    run: list[str] = []
    skip = 0
    for idx, tok in enumerate(tokens):
        if skip:
            skip -= 1
            continue
        low = tok.lower().strip(".")
        is_cap = bool(tok[:1].isupper()) or (tok[:1].isdigit() and len(tok) >= 3)
        if is_cap:
            # a capitalised word right after sentence start is usually the question word
            if not run and low in _STOP_ALONE:
                continue
            run.append(tok)
            continue
        if run and tok in ("'", "’") and idx + 1 < len(tokens) and tokens[idx + 1][:1].isupper():
            run.append(tok)
            continue
        if run and low in _PARTICLE and not skip:
            # absorb a chain of particles only when a capitalised word closes it ("Back in the U.S.S.R")
            chain = [tok]
            j = idx + 1
            while j < len(tokens) and tokens[j].lower().strip(".") in _PARTICLE:
                chain.append(tokens[j])
                j += 1
            if j < len(tokens) and tokens[j][:1].isupper():
                run.extend(chain)
                skip = len(chain) - 1
                continue
        if run:
            found.append(_join(run))
            run = []
    if run:
        found.append(_join(run))

    out: list[str] = []
    for cand in found:
        cand = cand.strip(" ,;:-").strip()
        while cand.lower().split(" ")[0] in _STOP_ALONE and " " in cand:
            cand = cand.split(" ", 1)[1].strip()
        if len(cand) < 3 or not _ALPHA.search(cand):
            continue
        words = [w for w in cand.split(" ") if w]
        if len(words) == 1 and (words[0].lower().strip(".") in _STOP_ALONE or len(words[0]) < 3):
            continue
        if cand.lower() not in {o.lower() for o in out}:
            out.append(cand)

    # drop a mention fully contained in a longer one (the longer span is the better query)
    out.sort(key=lambda s: -len(s))
    kept: list[str] = []
    for cand in out:
        if not any(cand.lower() in k.lower() for k in kept):
            kept.append(cand)
    return kept


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


class Harness(BaseHarness):
    K = 8                   # k for the planner's queries
    K_PROBE = 6             # k for a bare-name probe: the article itself should rank at the top
    MAX_PROBES = 4          # bare-name probes per item
    MAX_ROUNDS = 3          # router passes (each may add one retrieval)
    MAX_RETRIEVALS = 11     # hard cap below the enforced limit of 12
    POOL_SNIPPET = 340
    SHOW_MAX = 34           # passages shown to the router

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
        anchored: list[str] = []          # probe hits whose TITLE echoes the probed name
        probe_hits: list[str] = []        # every other passage a bare-name probe returned
        planner_hits: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def ordered_pool() -> list[str]:
            out = list(anchored)
            for p in planner_hits + probe_hits:
                if p not in anchored:
                    out.append(p)
            return out

        def gather(query: str, k: int, anchor: str = "") -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=k):
                title = str(passage).split(" | ")[0].strip()
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(passage)
                added += 1
                # a probe hit whose title echoes the probed name is the article we were after
                if anchor and (anchor.lower() in key or key in anchor.lower()):
                    anchored.append(passage)
                elif anchor:
                    probe_hits.append(passage)
                else:
                    planner_hits.append(passage)
            return added

        # 0. coverage guarantee: one bare-name search per entity the question names, run FIRST
        probes = _mentions(question)[: self.MAX_PROBES]
        for name in probes:
            gather(name, self.K_PROBE, anchor=name)

        # 1. the planner's own queries, on top of the anchored pool
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q, self.K)

        # 2-4. route over the pool, following up on what is missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            view = ordered_pool()[: self.SHOW_MAX]
            shown = [p[: self.POOL_SNIPPET] for p in view]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(view))
            if picks:
                selected = [view[i - 1] for i in picks[:8]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            next_query = routed.get("next_query", "")
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(next_query, self.K):
                break

        ordered = ordered_pool()
        evidence = selected or ordered[:8]
        ctx.note(
            "pool routed",
            probes=probes,
            n_anchored=len(anchored),
            n_pool=len(pool),
            n_selected=len(selected),
            retrievals=retrievals,
            missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=ordered[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "probes": probes, "missing": missing}
