"""Self-ask chaining: resolve the intermediate entity as an explicit sub-answer, then look it up by name.

The parent gathers a deduplicated evidence pool and asks one call to do everything at once: trace the
multi-step chain across eight passages AND emit a bare short string. Measured on its own traces, that single
call is where the errors are -- on 53 of its 95 wrong items the gold answer string is literally inside the
prompt it read, and it returned a competing span instead (the person who replaced X rather than the one X
replaced, the live album rather than the studio album the song came from). The chain is never written down,
so nothing forces the two hops apart.

This writes the chain down. One call reads the pool and produces an intermediate ANSWER -- the bridge entity,
named as its own article would be titled -- plus the slot the question ultimately demands. No previous system
answered a sub-question; the parent's router only names the fact it is *missing*.

The resolved name then buys retrieval that no question-side probe can reach: 48 of the 107 gold documents the
parent never retrieves are named inside abstracts it already read, and they are second-hop articles whose
titles the question does not print (the league a footballer played in, the album a song came from). A bare-name
read of the resolved bridge, filtered to passages whose TITLE is that name, opens exactly that document and
adds at most two passages, so the pool does not bloat the way additive keyword probes do.

    1-2. decomposed opening sweep into a title-keyed pool          (unchanged from the parent)
    3.   route over the pool, follow up on what is missing         (unchanged from the parent)
    4.   resolve: intermediate sub-answer (bridge entity) + the demanded slot
    5.   lookup(bridge) -> pinned passages whose title IS the bridge name
    6.   answer the question with the chain made explicit: pinned bridge article first, then selection
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
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_DESCRIPTIVE = re.compile(
    r"\b(?:who|whom|whose|which|what|where|when|why|how|that|the\s+(?:name|property|date|year|number)\b)\b",
    re.IGNORECASE,
)
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

RESOLVE_INSTRUCTIONS = """A multi-step question is answered by a chain: first identify an intermediate
thing the question describes but does not name, then look up the property of THAT thing which the question
actually asks for.

Read the numbered passages and write the chain down explicitly.

Produce:
- `bridge`: the intermediate thing, written as the exact name its own reference article would carry -- a bare
  name or title, no describing words, no property, no dates, no article. This is the step-1 answer, never the
  final answer. If the question names every thing it needs (for example it offers two named alternatives to
  compare), write `none`.
- `demanded`: the single thing the question finally asks for, restated as a short noun phrase, together with
  the direction of the relation when the question involves one thing acting on another (for example `the
  person who held the post BEFORE him, not the one who came after`). Restate only what was asked; never
  answer it here.
"""

CHAIN_ANSWER_INSTRUCTIONS = """Answer the question from the passages with the shortest possible answer.

Step 1 of the chain has already been resolved for you: `bridge` is the intermediate thing the question
describes without naming, and `demanded` restates what the question actually asks for. The first passages are
that bridge thing's own reference article. Use them to fill exactly the demanded slot -- not the bridge name
itself, and not the neighbouring fact.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

REPAIR_INSTRUCTIONS = """The answer given for this question was just the name of the intermediate thing the
chain passes through -- the step-1 answer, not the thing the question asks for.

`bridge` is that intermediate thing and its own reference article is among the first passages. `demanded`
restates what the question actually asks for. Read the bridge's article and fill the demanded slot from it:
the answer is a DIFFERENT string from the bridge name.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no explanation, no trailing punctuation. If the passages genuinely give nothing but
the bridge name, repeat it rather than abstaining."""

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
    LOOKUPS = 1             # extra retrievals reserved for bare-name reads of the resolved bridge
    K_LOOKUP = 20           # read wide, then keep only title matches
    MAX_EVIDENCE = 8
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
        self.resolve = Module(
            inputs=["question", "passages"],
            outputs=["bridge", "demanded"],
            instructions=RESOLVE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "bridge": "the intermediate thing's bare name, or `none`",
                "demanded": "the slot the question finally asks for, with the direction of the relation",
            },
        )
        self.chain_answer = Module(
            inputs=["question", "bridge", "demanded", "passages"],
            outputs=["answer"],
            instructions=CHAIN_ANSWER_INSTRUCTIONS,
            descs={
                "bridge": "the resolved step-1 entity",
                "demanded": "what the question finally asks for",
                "answer": "the bare short answer",
            },
        )
        self.repair = Module(
            inputs=["question", "bridge", "demanded", "passages"],
            outputs=["answer"],
            instructions=REPAIR_INSTRUCTIONS,
            descs={
                "bridge": "the intermediate thing that was returned by mistake",
                "demanded": "what the question finally asks for",
                "answer": "the bare short answer, different from the bridge name",
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

    # -- bridge-name handling ------------------------------------------------------------------
    @staticmethod
    def _title_of(passage: str) -> str:
        return str(passage).split(" | ")[0].strip()

    @staticmethod
    def _norm_title(text: str) -> str:
        t = _PARENTHETICAL.sub("", str(text or "")).strip()
        t = _NONWORD.sub(" ", t.lower())
        return " ".join(t.split())

    def _clean_name(self, raw: str, question: str) -> str:
        """Reduce the model's `bridge` field to a bare lookup name, or "" if it is not usable."""
        name = _MARKUP.sub("", str(raw or "")).strip().split("\n")[0].strip()
        name = _BULLET.sub("", name).strip()
        name = _LEADING_ARTICLE.sub("", name).strip(" \"'«»,;:.")
        if not name or len(name) < 3 or len(name.split()) > 7:
            return ""
        if _NON_ANSWER.match(name) or name.lower() in {"none", "bridge", "unknown"}:
            return ""
        # a name is a name: reject fields that came back as a description or a question
        if "?" in name or _DESCRIPTIVE.search(name):
            return ""
        # reject a field that merely echoes the whole question
        if self._norm_title(name) and self._norm_title(name) in self._norm_title(question):
            if len(name.split()) > 4:
                return ""
        return name

    def _title_matched(self, docs: list[str], name: str) -> list[str]:
        """Keep only passages whose TITLE is the name (modulo parentheticals) -- 0-2 passages, not 8."""
        want = self._norm_title(name)
        if not want:
            return []
        hits = []
        for passage in docs:
            title = self._norm_title(self._title_of(passage))
            if title and (title == want or (len(want.split()) > 1 and title.startswith(want + " "))):
                hits.append(passage)
        return hits[:2]

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

        # 4. write the chain down: the intermediate sub-ANSWER plus the slot the question demands
        chain = self.resolve(ctx.llm, tag="resolve", question=question, passages=evidence)
        bridge = self._clean_name(chain.get("bridge", ""), question)
        demanded = " ".join(_MARKUP.sub("", str(chain.get("demanded", ""))).split())[:300]

        # 5. open the resolved bridge's OWN article by name; a title filter keeps this to <=2 passages
        pinned: list[str] = []
        if bridge:
            here = self._title_matched(pool, bridge)
            if here:
                pinned = here                       # already in the pool: just pin it in front, no retrieval
            elif retrievals < self.MAX_RETRIEVALS + self.LOOKUPS:
                retrievals += 1
                fetched = ctx.tools.retrieve(bridge, k=self.K_LOOKUP)
                pinned = self._title_matched(fetched, bridge)
                for passage in pinned:
                    key = self._title_of(passage).lower()
                    if key not in seen:
                        seen.add(key)
                        pool.append(passage)
        ctx.note("chain resolved", bridge=bridge[:120], demanded=demanded[:200], n_pinned=len(pinned))

        # 6. answer with the bridge article first and the chain stated explicitly
        rest = [pp for pp in evidence if pp not in pinned]
        ordered = (pinned + rest)[: self.MAX_EVIDENCE]
        if bridge:
            drafted = self.chain_answer(
                ctx.llm, tag="answer", question=question, bridge=bridge,
                demanded=demanded or "the thing the question asks for", passages=ordered,
            ).get("answer", "")
        else:
            drafted = self.answer(ctx.llm, tag="answer", question=question, passages=ordered).get("answer", "")
        final = self._tidy(drafted, question)

        # The chain names an INTERMEDIATE answer, so an answer equal to it has stopped one hop short.
        # Fires only on that exact string match, and the original is kept if the repair returns nothing better.
        if bridge and final and self._norm_title(final) == self._norm_title(bridge):
            repaired = self.repair(
                ctx.llm, tag="repair", question=question, bridge=bridge,
                demanded=demanded or "the thing the question asks for", passages=ordered,
            ).get("answer", "")
            repaired = self._tidy(repaired, question)
            if repaired and not _NON_ANSWER.match(repaired) and self._norm_title(repaired) != self._norm_title(bridge):
                ctx.note("bridge leak repaired", before=final[:120], after=repaired[:120])
                final = repaired

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- and show the WHOLE pool, since selection may have dropped the right passage.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {
            "answer": final or str(drafted).strip(),
            "n_pool": len(pool),
            "queries": queries,
            "missing": missing,
            "bridge": bridge,
            "demanded": demanded,
        }
