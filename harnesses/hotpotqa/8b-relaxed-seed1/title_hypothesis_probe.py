"""Name the documents from parametric knowledge, then verify they exist before trusting them.

Trace analysis of the frontier router shows retrieval is now the binding constraint: on 94/300 search-set
items a required document never enters the pool, and accuracy there is 0.39 against 0.82 when both are
present.  Probing the retriever offline separates two very different causes:

    * paraphrasing the question harder does NOT help -- keyword-only variants, entity fan-out and the raw
      question each surface only ~39% of the documents missing from those pools, and their union ~46%;
    * querying the ARTICLE TITLE surfaces 84% of them.

So the failure is not query wording, it is not knowing what the second document is called.  The router can
only learn that name by reading a first-hop abstract that mentions it, which fails exactly when the first hop
also misses.  This harness gets the name from the other source available for free: the solver's own
knowledge of the encyclopedia.

Instead of paraphrasing the question into search queries, the planning call NAMES the articles whose abstracts
would carry each link of the chain -- including the bridge entity the model believes connects them, recalled
from memory rather than derived from the passages.  Because a recalled name may not exist, every hypothesis is
then VERIFIED against the index: a hypothesis counts as confirmed only when the document it retrieves carries
essentially the same title, and only confirmed documents are pinned to the front of the evidence pool.
Unconfirmed hypotheses keep their results at the back of the pool (a wrong guess costs ordering, not recall)
and are reported to the router as unverified, so a hallucinated entity cannot masquerade as evidence.

The pool, router, follow-up hop and answer policy are otherwise unchanged from the frontier system, so the
whole delta is attributable to replacing paraphrase-driven retrieval with verified title hypotheses.
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
_HEDGE_PREFIX = re.compile(
    r"^(?:none\s+of\b|no\s+(?:information|passage|mention|document|evidence|such)\b|not\s+(?:enough|clear)\b"
    r"|the\s+(?:passages?|documents?|text)\b|insufficient\b|unable\s+to\b|i\s+(?:cannot|can't|do\s+not)\b)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w']+")
_PAREN = re.compile(r"\([^)]*\)")
_TITLE_STOP = frozenset(
    "a an the of in on at to for and or de la le du von van der los las el il".split()
)
_QSTOP = frozenset(
    """a an the of in on at to for from by with and or is was were are am be been being do does did done
    who what which when where how why whose whom that this these those it its their his her he she they them
    there here as than then also both same not no nor but if while during between about into over under
    after before above below out up down off again further once any some such only own so too very can could
    will would shall should may might must have has had known does you your i we our""".split()
)
_LABEL = re.compile(
    r"^\s*(?:article|title|document|doc|entity|bridge(?:\s+entity)?|first|second|third)\s*\d*\s*[:\-]\s*",
    re.IGNORECASE,
)

HYPOTHESIS_INSTRUCTIONS = """This question can only be answered by combining two different encyclopedia
articles. Name those articles.

Use your own knowledge of the encyclopedia, not just the words of the question: work out which entity the
question describes, then which entity or work its answer depends on, and recall what the article about each
one is actually called.

In `reasoning`, spell out the chain: the entity described by the question, the link that connects it to the
answer, and the article that would state the final fact. Name the connecting entity explicitly even if the
question does not, and say when you are unsure of a name.

Write 3 or 4 article titles in `titles`, one per line, best guess first, no numbering, no explanation and no
extra words -- just the titles as they would appear at the top of the article:
1. the article about the entity the question describes most specifically;
2. the article about the connecting entity or work you recalled (the bridge), if you can name it;
3. the article that would state the fact the question finally asks for;
4. optionally one alternative name for whichever of the above you are least sure about.

A title you are unsure of is still worth writing -- it is checked against the encyclopedia before it is
trusted -- but never write a question, a sentence, or a description instead of a title."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. Passages found by looking up a title that the encyclopedia
confirmed are listed first; the rest come from a plain search of the question and may be off topic. A title
that was guessed but NOT confirmed is listed under `unconfirmed`: treat those names as unreliable, and never
use one as a fact.

In `reasoning`, decide which chain of facts the question needs, then check which links of that chain the
passages already establish.

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
    K = 8
    K_TITLE = 5             # a title probe is precise; it does not need width
    MAX_HYPOTHESES = 3
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 10
    POOL_SNIPPET = 380
    N_EVIDENCE = 8

    def __init__(self) -> None:
        self.hypothesize = Module(
            inputs=["question"],
            outputs=["titles"],
            instructions=HYPOTHESIS_INSTRUCTIONS,
            descs={"titles": "3-4 encyclopedia article titles, one per line, best guess first"},
        )
        self.route = Module(
            inputs=["question", "passages", "unconfirmed"],
            outputs=["useful", "missing", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "unconfirmed": "guessed titles the encyclopedia did not confirm",
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

    def _parse_titles(self, raw: str) -> list[str]:
        """One hypothesised article title per line; drop sentences, questions and boilerplate."""
        titles: list[str] = []
        for line in str(raw or "").splitlines():
            t = _LABEL.sub("", self._clean_query(line)).strip().strip('"').strip("'")
            if len(t) < 2 or "?" in t or t.lower() in {"none", "titles", "unknown", "n/a"}:
                continue
            if len(_WORD.findall(t)) > 10:          # a description, not a title
                continue
            if t.lower() not in {x.lower() for x in titles}:
                titles.append(t)
            if len(titles) == self.MAX_HYPOTHESES:
                break
        return titles

    @staticmethod
    def _keyterms(question: str) -> str:
        """Model-free keyword-only variant: the recall floor a title hypothesis must never fall below."""
        toks = [w for w in _WORD.findall(str(question or "")) if w.lower() not in _QSTOP]
        return " ".join(toks)[:300]

    @staticmethod
    def _key_tokens(title: str) -> set[str]:
        base = _PAREN.sub(" ", str(title or "")).lower()
        return {w for w in _WORD.findall(base) if w not in _TITLE_STOP}

    def _confirms(self, hypothesis: str, found_title: str) -> bool:
        """The index confirms a hypothesised name when the retrieved article is essentially that article."""
        h = self._key_tokens(hypothesis)
        f = self._key_tokens(found_title)
        if not h or not f:
            return False
        if h <= f:                                   # retrieved title contains the whole hypothesis
            return True
        overlap = len(h & f)                         # or differs only in a qualifier / spelling variant
        return overlap >= 2 and overlap / len(h | f) >= 0.75

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
        confirmed: list[str] = []       # passages whose title the index confirmed: the pinned front of the pool
        rest: list[str] = []            # everything else, in arrival order
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str, k: int, pin_for: str | None = None) -> tuple[int, bool]:
            """Retrieve `query`; when it is a title hypothesis, pin the article only if the index confirms it."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0, False
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            verified = False
            for rank, passage in enumerate(ctx.tools.retrieve(q, k=k)):
                title = str(passage).split(" | ")[0].strip()
                key = title.lower()
                pinned = pin_for is not None and rank < 2 and self._confirms(pin_for, title)
                if pinned:
                    verified = True
                if key in seen:
                    if pinned and passage in rest:   # promote an already seen article to the pinned front
                        rest.remove(passage)
                        confirmed.append(passage)
                    continue
                seen.add(key)
                (confirmed if pinned else rest).append(passage)
                added += 1
            return added, verified

        # 1. name the articles the chain needs, from the model's own knowledge of the encyclopedia
        raw = self.hypothesize(ctx.llm, tag="hypothesize", question=question).get("titles", "")
        hypotheses = self._parse_titles(raw)

        # 2. the paraphrase sweep still runs, so a wrong guess can never cost recall: the plain question
        #    plus its model-free keyword-only variant are the floor the hypotheses are added on top of.
        gather(question, self.K)
        gather(self._keyterms(question), self.K)

        # 3. probe every hypothesis as a title lookup and check whether the index confirms the name
        unconfirmed: list[str] = []
        for h in hypotheses:
            _, ok = gather(h, self.K_TITLE, pin_for=h)
            if not ok:
                unconfirmed.append(h)

        # 4. route over the pool (pinned articles first), following up on whatever link is still missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            pool = confirmed + rest
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(
                ctx.llm,
                tag=f"route{round_i + 1}",
                question=question,
                passages=shown,
                unconfirmed=", ".join(unconfirmed) if unconfirmed else "none",
            )
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[: self.N_EVIDENCE]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(routed.get("next_query", ""), self.K)[0]:
                break

        pool = confirmed + rest
        evidence = selected or pool[: self.N_EVIDENCE]
        ctx.note(
            "verified title hypotheses",
            hypotheses=hypotheses,
            n_confirmed=len(confirmed),
            unconfirmed=unconfirmed,
            n_pool=len(pool),
            retrievals=retrievals,
            missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {
            "answer": final or str(drafted).strip(),
            "hypotheses": hypotheses,
            "n_confirmed": len(confirmed),
            "n_pool": len(pool),
            "missing": missing,
        }
