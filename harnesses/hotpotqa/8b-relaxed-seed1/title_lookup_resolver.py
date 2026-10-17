"""Title-anchored entity lookup as a retrieval primitive, on top of the evidence-pool router.

Diagnosis behind this harness (frontier traces, 300 items): 81 items reach exactly half the needed
documents and convert at 46%, versus 82% at full coverage -- retrieval is still the binding constraint.
But the two previous retrieval attempts failed for opposite reasons: replacing the descriptive queries
with bare-name probes destroyed first-hop coverage, and probing every name printed in the question added
almost no recall (+0.002) while flooding the answering context with distractors.

The measured cause is narrower than either attempt assumed. A word-matching index answers a bare NAME
with that name's own article, and answers the same name plus the property being asked about with nothing
useful at all -- the property words dominate the score and pull in unrelated documents. So the fix is not
more searching, it is a different KIND of call:

    search(query)  -- topical, fuzzy, what the router already does well; finds entities that the
                      question DESCRIBES but does not name
    lookup(name)   -- a dictionary read: ask for the bare name with a wide k, then keep ONLY the
                      passages whose TITLE is that name (parentheticals ignored). Returns 0-2
                      passages, never a topical spray, so the answering context does not grow.

`lookup` is what a chain needs after the first hop: the bridge entity is discovered inside a retrieved
abstract, it is a title in the corpus, and its article is one exact-title read away. The router is
therefore asked for two different things: a `next_query` for facts it must still SEARCH for, and a
`lookup_name` for an entity whose own ARTICLE it needs. Resolved articles are pinned into the evidence
handed to the answerer, so a successful lookup can never be dropped by passage selection.
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

# -- title normalisation for the lookup primitive -------------------------------------------------
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_TITLE_JUNK = re.compile(r"[^a-z0-9 ]+")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
# a property phrase attached to a name ("<name> release date"): these are what break a title read
_NAME_SPLIT = re.compile(r"\s*(?:,|;|/|\band\b|\bor\b)\s*", re.IGNORECASE)
# a proper-name run, optionally opened by an ordinal ("65th Tony Awards"), continued across name particles
_NAME_RUN = re.compile(
    r"(?:\d{1,4}(?:st|nd|rd|th)\s+)?[A-Z][\w.'\u2019-]*"
    r"(?:\s+(?:of|the|de|di|da|del|van|von|la|le|for)\s+[A-Z0-9][\w.'\u2019-]*"
    r"|\s+[A-Z0-9][\w.'\u2019-]*)*"
)
_QUOTED = re.compile(r"[\"\u201c\u2018']([^\"\u201d\u2019']{2,60})[\"\u201d\u2019']")
# words that mark a query as asking for a PROPERTY rather than naming an article
_PROPERTY_WORD = re.compile(
    r"^(?:what|which|who|whom|whose|when|where|why|how|the|a|an|is|was|were|are|did|does|do|name|date|"
    r"year|birth|born|died|death|sponsor|company|album|release|located|location|founded|number|list)$",
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

ROUTE_INSTRUCTIONS = """You are routing a search that must gather every fact needed to answer a
multi-step question. Two different kinds of search are available and they behave very differently.

The passages are numbered search results. In `reasoning`, decide which chain of facts the question needs,
then check which links of that chain the passages already establish.

Produce:
- `useful`: the numbers of the passages that state a needed fact, most important first, comma separated.
  Include a passage only if it carries a fact the answer depends on; write `none` if no passage does.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property
  must still be looked up). Write `none` if every needed fact is present.
- `lookup_name`: if the passages have revealed the NAME of an entity whose own article you still need to
  read, write that name and nothing else -- exactly as it is spelled in the passages, with no property
  words, no dates, no question words, no description, no quotation marks. This is a dictionary read: the
  article titled with that name will be fetched and added. Write `none` if no such name is known yet, and
  never write a name whose article is already listed above.
- `next_query`: a keyword query for the missing fact, for when the needed document must be found by topic
  rather than by name. Leave it empty when nothing is missing. Never repeat a query already used.
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`.

Prefer `lookup_name` whenever the entity is named: adding the property you are looking for to a search
query makes the entity's own article impossible to find, whereas the bare name finds it immediately."""

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
    K = 8
    K_LOOKUP = 20           # wide read; the title filter keeps it precise
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 10     # 12 are allowed; lookups are cheap and add ~1 passage each
    MAX_LOOKUPS = 6
    POOL_SNIPPET = 380

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.route = Module(
            inputs=["question", "passages"],
            outputs=["useful", "missing", "lookup_name", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "lookup_name": "the bare name of an entity whose own article is needed, or `none`",
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
    def _title_key(title: str) -> str:
        """Canonical form of an article title: parenthetical qualifier and punctuation dropped."""
        t = _PARENTHETICAL.sub("", str(title or "").strip())
        t = _TITLE_JUNK.sub(" ", t.lower())
        t = " ".join(t.split())
        return _LEADING_ARTICLE.sub("", t).strip()

    def _clean_name(self, raw: str) -> str:
        """A lookup name must be a bare name: one line, no markup, no quotes, no trailing property words."""
        name = _MARKUP.sub("", str(raw or "")).strip().split("\n")[0].strip()
        name = _BULLET.sub("", name).strip()
        name = name.strip('"').strip("'").strip("«»").strip()
        name = name.rstrip(".,;:").strip()
        if not name or name.lower() in {"none", "n/a", "unknown", "null", ""}:
            return ""
        # a coordination means two entities were named; read the first one
        parts = [p for p in _NAME_SPLIT.split(name) if p.strip()]
        if parts and len(parts[0].split()) >= 1:
            name = parts[0].strip()
        if len(name.split()) > 8 or len(name) > 90:
            return ""       # a description, not a title
        return name

    def _mine_names(self, text: str) -> list[str]:
        """Pull the article-title candidates out of a property-blended query.

        The router writes `next_query` as `<entity> <property>`, which is exactly the form the index
        cannot answer. The proper-name run inside it is the form the index answers at rank 1, so the
        name is recovered deterministically instead of depending on the model to also volunteer it.
        """
        raw = " ".join(str(text or "").split())
        if not raw:
            return []
        found: list[str] = []
        for m in _QUOTED.findall(raw):
            found.append(m.strip())
        for m in _NAME_RUN.finditer(raw):
            found.append(m.group(0).strip())
        names: list[str] = []
        for cand in found:
            cand = cand.strip().rstrip(".,;:'\u2019").strip()
            words = cand.split()
            # trim trailing property words ("Dan Marino Birth Date" -> "Dan Marino")
            while words and _PROPERTY_WORD.match(words[-1]):
                words.pop()
            while words and _PROPERTY_WORD.match(words[0]):
                words.pop(0)
            if not words or len(words) > 7:
                continue
            cand = " ".join(words)
            if len(cand) < 3 or (len(words) == 1 and len(cand) < 4):
                continue
            key = self._title_key(cand)
            if not key or key in {self._title_key(n) for n in names}:
                continue
            names.append(cand)
        return names

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
        pinned: list[str] = []          # articles resolved by an exact-title read
        seen: set[str] = set()
        used_queries: set[str] = set()
        looked_up: set[str] = set()
        retrievals = 0
        n_lookups = 0
        resolved: list[str] = []

        def _add(passage: str) -> bool:
            key = self._title_key(str(passage).split(" | ")[0])
            if not key or key in seen:
                return False
            seen.add(key)
            pool.append(passage)
            return True

        def search(query: str) -> int:
            """Topical BM25 search: everything it returns joins the pool."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            return sum(1 for p in ctx.tools.retrieve(q, k=self.K) if _add(p))

        def lookup(name: str) -> int:
            """Dictionary read: keep only passages actually TITLED with this name."""
            nonlocal retrievals, n_lookups
            key = self._title_key(name)
            if not key or key in looked_up or n_lookups >= self.MAX_LOOKUPS:
                return 0
            if retrievals >= self.MAX_RETRIEVALS:
                return 0
            looked_up.add(key)
            used_queries.add(name.lower())
            retrievals += 1
            n_lookups += 1
            hits = 0
            for doc in ctx.tools.retrieve_docs(name, k=self.K_LOOKUP):
                if self._title_key(doc.get("title", "")) != key:
                    continue        # not this entity's article -- do not pollute the pool
                text = doc.get("text") or ""
                if _add(text):
                    pinned.append(text)
                hits += 1
                if hits >= 2:       # the name plus at most one qualified variant
                    break
            if hits:
                resolved.append(name)
            return hits

        # 1. decomposed opening sweep (identical to the parent: descriptive search finds what is
        #    only DESCRIBED in the question, which bare-name reads structurally cannot)
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            search(q)

        # 2-4. route: read named entities' articles by title, search topically for the rest
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[:8]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            # the entity whose article is needed, from the router if it named one and otherwise
            # mined out of its own property-blended query -- then the topical search as a fallback
            candidates = [self._clean_name(routed.get("lookup_name", ""))]
            for src in (routed.get("next_query", ""), missing):
                candidates.extend(self._mine_names(src))
            gained = 0
            tried = 0
            for name in candidates:
                if not name or self._title_key(name) in seen or self._title_key(name) in looked_up:
                    continue
                gained += lookup(name)
                tried += 1
                if tried >= 2 or gained >= 2:
                    break
            gained += search(routed.get("next_query", ""))
            if not gained:
                break

        evidence = list(selected or pool[:8])
        for p in pinned:               # a resolved article is never dropped by selection
            if p not in evidence:
                evidence.append(p)
        ctx.note(
            "pool routed",
            n_pool=len(pool),
            n_selected=len(selected),
            retrievals=retrievals,
            resolved=resolved,
            missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence[:10]).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "resolved": resolved, "missing": missing}
