"""Treat the index as a name-to-article lookup and spend the second hop OPENING an article, not describing it.

Measured on the frontier's own traces: 87 of 300 search-set items reach the answerer with a required article
missing, and those items are where nearly all remaining loss sits.  Three measurements over the index settle
what the missing article is reachable BY.  Handing the index a required article's own title as the query
returns that article inside k=20 for 87% of the missing titles, and for a random sample of 150 training
titles the article ranks first 63% of the time and inside the top five 87% of the time.  Handing it the
question's keywords instead recovers 14% of the missing titles, and anchoring those keywords on an article
already retrieved recovers 7%.  So a lexical index over abstracts behaves like a lookup table keyed by NAME
and answers descriptions badly: every hop should be phrased as a name, and a hop phrased as a description is
close to wasted.

The name of the missing article is usually already on the page.  For 33% of the missing titles the title is
written VERBATIM somewhere in the text the frontier already retrieved, and for 60% some mention overlapping
it is (a bare surname where the article carries the full name, a short form where the article carries the
formal one).  The frontier never uses those mentions: its follow-up query is built from the question's
vocabulary, and when routing reports a fact missing it drafts a description of the unseen article.

So the follow-up hop here is a LOOKUP.  Routing gains one output field -- the exact name, copied from the
passages, of the entity whose own article must be opened next -- which costs no extra call.  That name is
snapped model-free onto a capitalised mention actually mined from the retrieved text, so a hallucinated name
cannot be searched for, and is then resolved by `_lookup`: one wide probe, keeping only the passages whose
TITLE matches the name, exact match first and a title that merely contains the name after it.  Those articles
are pinned to the front of the pool.  Ranking by title match is what makes the wide probe safe -- the index
puts the named article first only 63% of the time, so a narrow probe misses it and a wide unfiltered probe
buries the pool in twenty near-namesakes.

The hop is additive: nothing is ever removed from the pool, and when no mention can be snapped the frontier's
keyword follow-up still runs.  Name-slate disambiguation, routing and the answer policy are unchanged.
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
_MENTION = re.compile(
    r"\b[A-Z][\w'&.\-]*(?:\s+(?:of|de|del|della|di|van|von|du|da|dos|la|le)\s+[A-Z][\w'&.\-]*"
    r"|\s+[A-Z][\w'&.\-]*)*"
)
_PAREN = re.compile(r"\([^)]*\)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_COMMA_TAIL = re.compile(r",.*$")
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
    r"^\s*(?:name|title|entity|article|constraint|requirement|mention)\s*\d*\s*[:\-]\s*", re.IGNORECASE
)

ANALYZE_INSTRUCTIONS = """A multi-step question is about to be looked up in an encyclopedia whose search ranks
articles by popularity. Prepare the lookup.

Many names are shared by several articles -- a novel and the film made from it, two songs, a place and a much
bigger place with the same name, several films released in different years. The search will return the FAMOUS
one. To pick the right one instead, two things are needed.

In `reasoning`, work out which entity the question is really about and what in the question rules out the
better-known things sharing its name.

Produce:
- `names`: the proper names, work titles and place names that the question MENTIONS, one per line, most
  ambiguous first, at most three. Write each one bare, exactly as the article about it would be titled: no
  quotation marks, no year, no descriptive words the question wrapped around it, nothing invented. Write
  `none` if the question mentions no name at all.
- `constraints`: the checklist the intended article must satisfy, one short requirement per line, at most
  four. Take these from the question only -- what KIND of thing it is (film, novel, song, album, person,
  town, team), when it appeared, where it is from, what role it played. Do not guess facts the question does
  not state. Write `none` if the question states no distinguishing property."""

DISAMBIGUATE_INSTRUCTIONS = """A name in a question matches several different encyclopedia articles. Choose
the one the question is about.

You are given the checklist of properties the intended article must satisfy, and the candidate articles the
encyclopedia actually holds under each name, numbered, each shown as `Title | opening of the article`.

Judge by the checklist, not by fame. The best known article under a name is usually NOT the one an
indirect question is about: if the question says a novel, a candidate that opens by describing a film fails
the checklist however famous it is; if the question fixes a year, a nationality or a role, a candidate whose
opening contradicts it is out. A candidate that merely fails to mention a property does not violate it --
only an outright contradiction does.

In `reasoning`, go name by name and say which candidates each checklist item eliminates.

Produce `picks`: one line per name, written `<name> = <number>`, giving the number of the surviving candidate.
Write `<name> = none` when every candidate contradicts the checklist, or when two survive and nothing in the
question separates them. Guessing when nothing separates the candidates is worse than `none`."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. Passages listed first are articles that were chosen as the intended
meaning of a name in the question, in preference to better known articles sharing that name; trust them over
a passage further down that looks similar but describes a different thing. The rest come from a plain search
and may be about the wrong namesake entirely.

In `reasoning`, decide which chain of facts the question needs, then check which links of that chain the
passages already establish, and watch for passages that share a name with the subject but contradict the
question (wrong year, wrong medium, wrong country).

Produce:
- `useful`: the numbers of the passages that state a needed fact, most important first, comma separated.
  Include a passage only if it carries a fact the answer depends on; write `none` if no passage does.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property
  must still be looked up). Write `none` if every needed fact is present.
- `next_query`: a keyword query for that missing fact, anchored on the exact entity name as it appears in the
  passages, plus the property being sought. Leave it empty when nothing is missing. Never repeat a query whose
  results are already listed.
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`.
- `next_name`: the missing fact is usually held by the article of some entity that the passages NAME but do
  not describe -- a person, work, place or organisation mentioned in an abstract whose own article has not
  been opened. Write that entity's name exactly as the passages spell it, nothing else, and write `none`
  when no such entity is named or when nothing is missing. Never write a name that does not appear in the
  passages, never write a description, and never write the name of an entity whose article is already
  listed."""

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
    K_NAME = 8              # a bare-name probe must be wide enough to reach the obscure namesake
    MAX_NAMES = 3
    MAX_SLATE = 4           # candidates shown per colliding name
    MAX_ROUNDS = 3
    K_LOOKUP = 20           # the named article ranks first only 63% of the time: probe wide, then rank by title
    MAX_LOOKUPS = 3         # at most three articles are opened by name
    MAX_MENTIONS = 900      # mentions mined from the pool that a name may be snapped onto
    MAX_OPENED = 1          # one article pinned per lookup: a short name can answer to a wrong namesake
    ADMIT_LOOKUP = 5        # passages a wide lookup may add to the pool, title matches first
    MAX_RETRIEVALS = 12
    POOL_SNIPPET = 380
    SLATE_SNIPPET = 240
    N_EVIDENCE = 8

    def __init__(self) -> None:
        self.analyze = Module(
            inputs=["question"],
            outputs=["names", "constraints"],
            instructions=ANALYZE_INSTRUCTIONS,
            descs={
                "names": "bare names the question mentions, one per line, at most three, or `none`",
                "constraints": "properties the intended article must satisfy, one per line, or `none`",
            },
        )
        self.disambiguate = Module(
            inputs=["question", "constraints", "candidates"],
            outputs=["picks"],
            instructions=DISAMBIGUATE_INSTRUCTIONS,
            descs={
                "constraints": "the checklist the intended article must satisfy",
                "candidates": "candidate articles per name, numbered, `Title | opening`",
                "picks": "one `<name> = <number>` or `<name> = none` per line",
            },
        )
        self.route = Module(
            inputs=["question", "passages"],
            outputs=["useful", "missing", "next_query", "sufficient", "next_name"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
                "next_name": "the name, as the passages spell it, of the entity whose article to open, or `none`",
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


    # -- the name-lookup channel -------------------------------------------------------------
    @classmethod
    def _mine_mentions(cls, passages: list[str]) -> list[str]:
        """Every capitalised mention written in the retrieved text, in the order it is written.

        These are the only names a follow-up hop may be spent on: a name the model produces that is not
        written anywhere in the pool is a name it invented, and probing for it wastes the call.
        """
        out: list[str] = []
        seen: set[str] = set()
        for passage in passages:
            body = str(passage).split(" | ", 1)[-1]
            for sentence in _SENTENCE.split(body):   # a mention never spans a sentence boundary
              for match in _MENTION.finditer(sentence):
                  phrase = match.group(0).strip(" .,;:-'")
                  words = phrase.split()
                  if len(phrase) < 3 or len(words) > 7:
                      continue
                  if len(words) == 1 and (phrase.lower() in _QSTOP or len(phrase) < 4):
                      continue
                  key = phrase.lower()
                  if key in seen:
                      continue
                  seen.add(key)
                  out.append(phrase)
        return out[: cls.MAX_MENTIONS]

    def _snap(self, name: str, mentions: list[str], opened: set[str]) -> str:
        """Bind the routed name to a mention that really occurs in the text, or refuse the hop.

        An exact mention wins. Otherwise the shortest mention that contains every token of the name is
        taken, which is how a bare surname reaches the article that carries the full name. A name whose
        article is already open is refused so the hop is never spent twice on the same page.
        """
        want = self._key_tokens(name)
        if not want or len(want) > 7:
            return ""
        for mention in mentions:
            if self._key_tokens(mention) == want:
                if " ".join(sorted(want)) in opened:
                    return ""
                return mention
        best = ""
        for mention in mentions:
            tokens = self._key_tokens(mention)
            if want <= tokens and len(tokens) - len(want) <= 3:
                if not best or len(mention) < len(best):
                    best = mention
        if best and " ".join(sorted(self._key_tokens(best))) in opened:
            return ""
        return best

    def _title_rank(self, name: str, passage: str) -> int:
        """How well this passage's TITLE answers to `name`: 0 exact, 1 contains it, 2 not a match."""
        title = str(passage).split(" | ")[0].strip()
        want = self._key_tokens(name)
        got = self._key_tokens(title)
        if not want or not got:
            return 2
        if "disambiguation" in title.lower():
            return 1                      # a disambiguation page is a list of names, not the article
        if want == got and "(" not in title:
            return 0                      # the article titled exactly this name
        if self._key_tokens(_COMMA_TAIL.sub("", _PAREN.sub("", title))) == want:
            return 1                      # `<name> (sense)`: one of several things called this, rank by BM25
        if want <= got:
            return 1
        return 2

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("«»").strip()
        return " ".join(q.split())[:300]

    def _parse_lines(self, raw: str, limit: int, max_words: int) -> list[str]:
        out: list[str] = []
        for line in str(raw or "").splitlines():
            t = _LABEL.sub("", self._clean_query(line)).strip().strip('"').strip("'")
            if len(t) < 2 or t.lower() in {"none", "n/a", "unknown", "names", "constraints"}:
                continue
            if len(_WORD.findall(t)) > max_words:
                continue
            if t.lower() not in {x.lower() for x in out}:
                out.append(t)
            if len(out) == limit:
                break
        return out

    @classmethod
    def _content_words(cls, text: str) -> str:
        """The drafted opening stripped to its discriminating vocabulary: the second probe."""
        body = str(text or "").replace("___", " ")
        toks = [w for w in _WORD.findall(body) if w.lower() not in _QSTOP and len(w) > 1]
        out: list[str] = []
        for w in toks:                      # BM25 already weights repeats; a term set probes wider
            if w.lower() not in {x.lower() for x in out}:
                out.append(w)
        return " ".join(out)[:300]

    @staticmethod
    def _keyterms(question: str) -> str:
        """Model-free keyword-only variant: the recall floor the name probes are added on top of."""
        toks = [w for w in _WORD.findall(str(question or "")) if w.lower() not in _QSTOP]
        return " ".join(toks)[:300]

    @staticmethod
    def _key_tokens(text: str) -> set[str]:
        base = _PAREN.sub(" ", str(text or "")).lower()
        return {w for w in _WORD.findall(base) if w not in _TITLE_STOP}

    def _is_about(self, name: str, title: str) -> bool:
        """Model-free slate filter: this article really is one of the things called `name`.

        A title qualifies when it carries every content token of the name, so a bare name and the same name
        with a parenthetical sense qualifier both enter the slate, while an article that merely discusses the
        name does not, and a name the index writes with a trailing comma qualifier still matches.
        """
        n = self._key_tokens(name)
        if not n:
            return False
        t = self._key_tokens(title)
        if not n <= t:
            return False
        base = self._key_tokens(_COMMA_TAIL.sub("", _PAREN.sub("", str(title or ""))))
        return len(base - n) <= 1        # the title is the name itself, give or take one qualifier word

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
        pinned: list[str] = []          # articles chosen as the intended referent: the front of the pool
        rest: list[str] = []            # everything else, in arrival order
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0
        opened: set[str] = set()      # articles already opened by name
        lookups = 0

        def gather(query: str, k: int, name: str = "", admit: int = 0) -> list[str]:
            """Retrieve `query`, add anything new to the back of the pool, return this query's passages.

            A wide lookup probe passes `name`: its results are then admitted title-match first and capped at
            `admit`, so opening one article by name cannot bury the pool under twenty near-namesakes the way
            an unfiltered k=20 probe would.
            """
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return []
            used_queries.add(q.lower())
            retrievals += 1
            got = list(ctx.tools.retrieve(q, k=k))
            order = list(range(len(got)))
            if name:
                order.sort(key=lambda i: (self._title_rank(name, got[i]), i))
            budget = admit if admit else len(got)
            for i in order:
                key = str(got[i]).split(" | ")[0].strip().lower()
                if key in seen or budget <= 0:
                    continue
                seen.add(key)
                rest.append(got[i])
                budget -= 1
            return got

        def pin(passage: str) -> None:
            if passage in rest:
                rest.remove(passage)
            if passage not in pinned:
                pinned.append(passage)

        # 1. parse the question into the names it mentions and the checklist the intended article must satisfy
        parsed = self.analyze(ctx.llm, tag="analyze", question=question)
        names = self._parse_lines(parsed.get("names", ""), self.MAX_NAMES, 8)
        constraints = self._parse_lines(parsed.get("constraints", ""), 4, 14)

        # 2. recall floor, identical to the frontier: the plain question and its keyword-only variant
        gather(question, self.K)
        gather(self._keyterms(question), self.K)

        # 3. probe each name on its own and keep, model-free, every article the index holds under that name
        slates: list[tuple[str, list[str]]] = []
        for name in names:
            found = gather(name, self.K_NAME)
            slate: list[str] = []
            for passage in found:
                title = str(passage).split(" | ")[0].strip()
                if self._is_about(name, title) and passage not in slate:
                    slate.append(passage)
                if len(slate) == self.MAX_SLATE:
                    break
            if slate:
                slates.append((name, slate))

        # 4. a slate of one is not ambiguous -- pin it outright and spend no call on it
        colliding = [(n, s) for n, s in slates if len(s) > 1]
        for name, slate in slates:
            if len(slate) == 1:
                pin(slate[0])

        picks_raw = ""
        chosen: list[str] = []
        if colliding:
            lines: list[str] = []
            flat: list[str] = []
            for name, slate in colliding:
                lines.append(f"name: {name}")
                for passage in slate:
                    flat.append(passage)
                    lines.append(f"  [{len(flat)}] {str(passage)[: self.SLATE_SNIPPET]}")
            picks_raw = self.disambiguate(
                ctx.llm,
                tag="disambiguate",
                question=question,
                constraints="\n".join(constraints) if constraints else "none",
                candidates="\n".join(lines),
            ).get("picks", "")
            # one pick per line, `none` lines contribute no index
            for line in str(picks_raw).splitlines():
                if re.search(r"\bnone\b", line, re.IGNORECASE):
                    continue
                idx = self._parse_indices(line.split("=")[-1], len(flat))
                if idx:
                    chosen.append(flat[idx[0] - 1])
            for passage in chosen:
                pin(passage)

        # 5. route over the pool (chosen referents first), following up on whatever link is still missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            pool = pinned + rest
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[: self.N_EVIDENCE]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            progressed = bool(gather(routed.get("next_query", ""), self.K))

            # the follow-up hop that matters: OPEN the article of an entity the passages only name
            name_raw = str(routed.get("next_name", "")).strip()
            if (
                lookups < self.MAX_LOOKUPS
                and retrievals < self.MAX_RETRIEVALS
                and name_raw
                and not re.match(r"^\s*(none|n/?a|unknown)\b", name_raw, re.IGNORECASE)
            ):
                target = self._snap(
                    self._parse_lines(name_raw, 1, 7)[0] if self._parse_lines(name_raw, 1, 7) else "",
                    self._mine_mentions(pool),
                    opened,
                )
                if target:
                    lookups += 1
                    opened.add(" ".join(sorted(self._key_tokens(target))))
                    got = gather(target, self.K_LOOKUP, name=target, admit=self.ADMIT_LOOKUP)
                    # keep the wide probe from swamping the pool: only title matches are promoted
                    matched = sorted(
                        ((self._title_rank(target, p), n, p) for n, p in enumerate(got)),
                        key=lambda t: (t[0], t[1]),
                    )
                    hits = [p for rank, _, p in matched if rank < 2][: self.MAX_OPENED]
                    for passage in hits:
                        pin(passage)
                    progressed = progressed or bool(got)
                    ctx.note("opened article by name", name=target, n_hits=len(hits),
                             titles=[str(p).split(" | ")[0] for p in hits])

            if not progressed:
                break

        pool = pinned + rest
        evidence = selected or pool[: self.N_EVIDENCE]
        ctx.note(
            "referent disambiguation",
            names=names,
            constraints=constraints,
            n_colliding=len(colliding),
            n_pinned=len(pinned),
            picks=str(picks_raw)[:200],
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
            "lookups": lookups,
            "names": names,
            "constraints": constraints,
            "n_pinned": len(pinned),
            "n_pool": len(pool),
            "missing": missing,
        }
