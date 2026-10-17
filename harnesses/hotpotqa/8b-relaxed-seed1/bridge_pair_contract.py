"""Commit to exactly two linked articles before answering, and answer from nothing else.

Every system evaluated so far hands the answerer a RANKED LIST of eight passages and lets it decide what to
read.  That is the wrong contract for a two-hop question, and the frontier's misses say so: the recurring
failure is not a missing document but a distractor that shares a name with the real subject and outranks it,
so the answerer reads the famous namesake sitting at position 1 and answers from it.  A ranked list gives the
model no reason to prefer position 5 over position 1; only a structural test can.

The structure a two-hop question always has is a LINK.  The article that identifies the subject names the
entity the second article is about, in its own text -- the adaptation, the performer, the parent body, the
successor.  So the pool contains a distinguished PAIR, and the pair is checkable without a model: the link
name must occur verbatim in the first article and title the second.

This harness makes that pair the unit of commitment.

    1. one call reads the pool and signs a contract instead of selecting passages: `anchor` (the passage that
       identifies the entity the question describes), `link` (the connecting entity, copied verbatim out of
       the anchor's own text) and `second` (the passage carrying the rest of the chain) -- two documents,
       always, never a ranked shortlist;
    2. the contract is then AUDITED deterministically -- the link string must really occur in the anchor's
       text, and the second article's title must really be the link -- so a hallucinated connection cannot
       be signed;
    3. when the second article is absent the link is looked up by name, which is a precise title query
       rather than another paraphrase of the question, and the contract is re-signed over the enlarged pool;
    4. an audited contract collapses the context to the two committed abstracts at full length, with the link
       named; a contract that fails its audit falls back to the eight-passage ranked list with whatever it
       did commit pinned to the front, so the mechanism can never see less evidence than it has earned.

Comparison questions have no bridge, so their contract names an empty link and the two compared articles
become the pair: the same two-document commitment, audited by the absence of a claimed connection rather
than by its presence.
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
_LABEL = re.compile(r"^\s*(?:link|entity|bridge|name)\s*\d*\s*[:\-]\s*", re.IGNORECASE)

CONTRACT_INSTRUCTIONS = """A question that needs two encyclopedia articles is being answered from a numbered
list of search results. Do not select a shortlist -- commit to the TWO articles that carry the chain, and name
the entity that connects them.

Search ranks by popularity, so a passage near the top may be a DIFFERENT thing that merely shares a name with
the subject: another film of the same title, another song, a bigger place. Check each candidate against what
the question actually says about its subject -- the medium, the year, the country, the role -- before
choosing it, and prefer a lower-ranked passage that fits over a top-ranked one that does not.

In `reasoning`, state the entity the question describes, which passage identifies it, what that passage says
that leads on to the answer, and which passage states the final fact.

Produce:
- `anchor`: the number of the passage that identifies the entity the question describes.
- `link`: the connecting entity -- the person, work, place or body that the anchor passage mentions and that
  the SECOND article is about. Copy the name EXACTLY as the anchor passage spells it: just the name, no
  description, no explanation. Write `none` only when the question simply compares two things that the
  passages already describe, so there is nothing to follow.
- `second`: the number of the passage that supplies the rest of the chain -- the article about the link, or,
  for a comparison, the other thing being compared. Two numbers are almost always needed, so give one
  wherever a listed passage carries any part of the answer. Write `none` only when no listed passage does,
  which means the article about the link has not been found yet.

Never give the same number for `anchor` and `second`."""

PAIR_ANSWER_INSTRUCTIONS = """Answer the question from these two articles with the shortest possible answer.

These two articles were committed to as the ones carrying the whole chain, and the entity connecting them is
named. They are all the evidence there is: do not wonder what other articles might say, and do not refuse
because something is missing -- read the two abstracts and combine them.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
articles use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

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
    K_LINK = 5              # looking a link up by name is precise; it does not need width
    MAX_ROUNDS = 2          # sign the contract, look the link up, sign it again
    MAX_RETRIEVALS = 10
    POOL_SNIPPET = 380
    N_EVIDENCE = 8

    def __init__(self) -> None:
        self.contract = Module(
            inputs=["question", "passages"],
            outputs=["anchor", "link", "second"],
            instructions=CONTRACT_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "anchor": "number of the passage identifying the question's entity",
                "link": "the connecting entity copied verbatim from the anchor passage, or `none`",
                "second": "number of the passage supplying the rest of the chain, or `none`",
            },
        )
        self.pair_answer = Module(
            inputs=["question", "link", "articles"],
            outputs=["answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "link": "the entity connecting the two articles",
                "articles": "the two committed articles, each `Title | abstract`",
                "answer": "the bare short answer",
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

    def _clean_link(self, raw: str) -> str:
        line = str(raw or "").strip().splitlines()
        if not line:
            return ""
        t = _LABEL.sub("", self._clean_query(line[0])).strip().strip('"').strip("'")
        if t.lower() in {"none", "n/a", "unknown", "empty"} or len(t) < 2 or "?" in t:
            return ""
        return t if len(_WORD.findall(t)) <= 8 else ""

    @staticmethod
    def _keyterms(question: str) -> str:
        toks = [w for w in _WORD.findall(str(question or "")) if w.lower() not in _QSTOP]
        return " ".join(toks)[:300]

    @staticmethod
    def _key_tokens(text: str) -> set[str]:
        base = _PAREN.sub(" ", str(text or "")).lower()
        return {w for w in _WORD.findall(base) if w not in _TITLE_STOP}

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(_WORD.findall(str(text or "").lower()))

    def _mentions(self, passage: str, link: str) -> bool:
        """Deterministic audit half one: the anchor really does name the link in its own text."""
        return bool(link) and self._norm(link) in self._norm(passage)

    def _titles(self, passage: str, link: str) -> bool:
        """Deterministic audit half two: the target article really is the article about the link."""
        title = str(passage).split(" | ")[0]
        lk = self._key_tokens(link)
        return bool(lk) and lk <= self._key_tokens(title)

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

        def gather(query: str, k: int) -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=k):
                key = str(passage).split(" | ")[0].strip().lower()
                if key not in seen:
                    seen.add(key)
                    pool.append(passage)
                    added += 1
            return added

        gather(question, self.K)
        gather(self._keyterms(question), self.K)

        pair: list[str] = []
        link = ""
        audit = "unsigned"
        for round_i in range(self.MAX_ROUNDS):
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            signed = self.contract(
                ctx.llm, tag=f"contract{round_i + 1}", question=question, passages=shown
            )
            a_idx = self._parse_indices(signed.get("anchor", ""), len(pool))
            s_idx = self._parse_indices(signed.get("second", ""), len(pool))
            link = self._clean_link(signed.get("link", ""))
            anchor = pool[a_idx[0] - 1] if a_idx else None
            second = pool[s_idx[0] - 1] if s_idx else None
            if second is anchor and len(s_idx) > 1:
                second = pool[s_idx[1] - 1]

            if anchor is not None and second is not None and second is not anchor:
                pair = [anchor, second]
                if not link:
                    # a comparison: the two articles are the compared things, there is nothing to follow
                    audit = "comparison"
                elif self._mentions(anchor, link) and self._titles(second, link):
                    # both halves of the connection check out on the raw text
                    audit = "audited"
                else:
                    audit = "weak"
                break

            if anchor is not None and second is None:
                pair = [anchor]
                audit = "half"
            if round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            # the link is named but its article is not in the pool: look it up by name, not by paraphrase
            if not gather(link, self.K_LINK):
                break

        ctx.note(
            "bridge pair contract",
            audit=audit,
            link=link[:120],
            n_pair=len(pair),
            pair_titles=[str(p).split(" | ")[0] for p in pair],
            n_pool=len(pool),
            retrievals=retrievals,
        )

        if len(pair) >= 2 and audit in ("audited", "comparison"):
            # the contract held: collapse the context to exactly the two committed abstracts
            drafted = self.pair_answer(
                ctx.llm, tag="pair_answer", question=question, link=link or "none", articles=pair
            ).get("answer", "")
        else:
            # the contract failed its audit: fall back to the ranked list, with whatever was committed
            # pinned to its front, so the answerer never sees less evidence than it has earned
            backing = pair + [p for p in pool if p not in pair]
            drafted = self.answer(
                ctx.llm, tag="answer", question=question, passages=backing[: self.N_EVIDENCE]
            ).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {
            "answer": final or str(drafted).strip(),
            "audit": audit,
            "link": link,
            "n_pair": len(pair),
            "n_pool": len(pool),
        }
