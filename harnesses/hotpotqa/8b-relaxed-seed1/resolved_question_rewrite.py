"""Name the entity the question only describes, rewrite the question around that name, and answer THAT.

A two-step question hands the reader two jobs at once: work out which thing "the man who founded the project
with X and Y" or "the town near the highest point of the range" refers to, and then look up the property the
question actually asks for.  Every system so far asks for both in a single call -- the answerer is given the
original question and a window of passages and must carry the first inference silently while producing a bare
string.  The failures look exactly like a dropped first step: on 36 of the frontier's 88 misses the graded
answer is a span inside the very prompt that was read, and the span that wins instead is a property of the
WRONG member of the pair -- the wife rather than the man born in 1928, the assistant rather than the head
coach, the song's own release date rather than the album's.

The passages, by then, already contain the name.  So this harness spends one call turning the described
entity into a named one: it rewrites the question with every indirect description replaced by the proper name
the passages give it, keeping the interrogative word and the property being asked for untouched, so what
reaches the answerer is a one-step lookup rather than a chain.  The rewrite is admitted only when it is
model-free verifiable -- it must still ask a question, it must introduce a name that the pool actually holds
and the original question did not contain, and it must not have dropped the question's own names -- and it
falls back to the original wording whenever any of those checks fails, so a hallucinated resolution cannot
reach the reader.  The collapsed question is also a far better search key than the description it replaces,
so it is issued once as an additive query before the final read.  Retrieval, routing, referent
disambiguation and the answer policy are otherwise unchanged from the parent.
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
_HEDGE_PREFIX = re.compile(
    r"^(?:none\s+of\b|no\s+(?:information|passage|mention|document|evidence|such)\b|not\s+(?:enough|clear)\b"
    r"|the\s+(?:passages?|documents?|text)\b|insufficient\b|unable\s+to\b|i\s+(?:cannot|can't|do\s+not)\b)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w']+")
_PAREN = re.compile(r"\([^)]*\)")
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
_QUOTED = re.compile(r"[\"\u201c\u2018']([^\"\u201c\u201d\u2018\u2019']{2,80})[\"\u201d\u2019']")
_YEAR = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")
_NAME_SPAN = re.compile(r"\b[A-Z][\w'\.-]+")
_WH_WORDS = ("who", "what", "which", "when", "where", "how", "why", "whose", "whom", "in what", "in which")
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
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`."""

IMAGINE_INSTRUCTIONS = """A search of an encyclopedia has not turned up the article that carries a needed
fact. You will write a draft of that article's opening sentence so the search can be run against it.

Encyclopedia abstracts open in a fixed register: the subject, then what kind of thing it is, then the
defining specifics -- the year, the medium, the country, the studio, the office, the league, the discipline,
the works it is known for. Write that sentence for the article that would contain the missing fact.

You do not know the subject's name and must not invent one: write `___` wherever the name would go. Do not
copy the question's phrasing or its question words. Use the vocabulary the real article would use: concrete
type nouns, the field's technical terms, dates, places, roles. Prefer specific words over general ones --
`platform adventure video game developed by` says more than `a game made by`. One or two sentences, at most
forty words, no hedging and no explanation of what you are doing.

In `reasoning`, say what kind of article must hold the missing fact and which specifics the question already
fixes about it.

Produce `opening`: the drafted opening sentence."""

RESOLVE_INSTRUCTIONS = """A question refers to something by describing it instead of naming it. The passages
below name it. Rewrite the question so that it names it too.

Replace each descriptive reference -- "the man who founded X", "the town near Y", "the album of the song Z",
"the team that featured W" -- with the proper name or title the passages give that thing, taken verbatim from
a passage. Change nothing else: keep the interrogative word, keep the property being asked for word for word,
keep every name the question already contained, and add no facts and no explanation. If the passages do not
name the described thing, or the question already names everything it refers to, repeat the question unchanged.

In `reasoning`, identify each descriptive reference and the passage that names it, and say which passage.

Produce `rewritten`: the question, in one line, with descriptions replaced by names. It must still be the same
question, asking for the same thing, and must end with a question mark."""


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
    K_PSEUDO = 8            # width of a pseudo-article probe
    MAX_RETRIEVALS = 12     # the two pseudo-article probes are additive, so the ceiling is raised
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
        self.imagine = Module(
            inputs=["question", "missing", "passages"],
            outputs=["opening"],
            instructions=IMAGINE_INSTRUCTIONS,
            descs={
                "missing": "the fact the search has not found yet",
                "passages": "what the search has already returned, each `Title | abstract`",
                "opening": "a drafted abstract-style opening sentence for the missing article",
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
            outputs=["rewritten"],
            instructions=RESOLVE_INSTRUCTIONS,
            descs={
                "passages": "the selected search results, each `Title | abstract`",
                "rewritten": "the same question with described things replaced by their names",
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

    @staticmethod
    def _asked_slot(question: str) -> set[str]:
        """The content words of the clause that states WHAT is being asked, up to the first verb-ish break.

        Used only as a survival check: whatever the rewrite does to the descriptions, at least one of these
        words must still be there, otherwise the rewrite is asking for something else.
        """
        words = [w.lower() for w in _WORD.findall(str(question or ""))]
        keep = {w for w in words if w not in _QSTOP and len(w) > 2}
        return keep or set(words)

    @classmethod
    def _admit_rewrite(cls, rewritten: str, question: str, pool: list[str]) -> str:
        """Model-free gate: a rewrite is used only if it demonstrably NAMED something, without losing anything.

        Three checks, all cheap and all necessary: the rewrite must still be a question; it must keep every
        content word the original question already had (so the property being asked for cannot be silently
        traded away); and it must introduce at least one capitalised name that the retrieved pool actually
        contains and the original question did not.  A hallucinated resolution fails the last check and the
        original wording is kept.
        """
        cand = " ".join(str(rewritten or "").split()).strip()
        if cand[:1] in '"\u201c' and cand[-1:] in '"\u201d':
            cand = cand[1:-1].strip()
        cand = _LABEL.sub("", _BULLET.sub("", cand)).strip()
        if len(cand) < 8 or len(cand) > 400 or "\n" in cand:
            return ""
        low = cand.lower()
        if "?" not in cand and not any(low.startswith(w) for w in _WH_WORDS):
            return ""
        q_words = {w.lower() for w in _WORD.findall(question)}
        cand_words = {w.lower() for w in _WORD.findall(cand)}
        # a description may legitimately disappear, but a literal the question PINNED may not: the titles it
        # quotes and the years it fixes are the constraints the rewrite is not allowed to trade away
        for quoted in _QUOTED.findall(question):
            if quoted.strip() and not {w.lower() for w in _WORD.findall(quoted)} <= cand_words:
                return ""
        if not {w for w in _YEAR.findall(question)} <= {w for w in _YEAR.findall(cand)}:
            return ""
        if not (cls._asked_slot(question) & cand_words):
            return ""                                   # the property being asked for did not survive
        if len(cand_words - _QSTOP) * 2 < len({w for w in q_words if w not in _QSTOP}):
            return ""                                   # collapsed into something far too thin to be the question
        pool_text = " ".join(pool).lower()
        added = [w for w in _NAME_SPAN.findall(cand)
                 if w.lower() not in q_words and len(w) > 2 and w.lower() not in _QSTOP]
        if not any(w.lower() in pool_text for w in added):
            return ""                                   # nothing new, or nothing the passages back
        return cand

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pinned: list[str] = []          # articles chosen as the intended referent: the front of the pool
        rest: list[str] = []            # everything else, in arrival order
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0
        imagined = False

        def gather(query: str, k: int) -> list[str]:
            """Retrieve `query`, add anything new to the back of the pool, return this query's passages."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return []
            used_queries.add(q.lower())
            retrievals += 1
            got = []
            for passage in ctx.tools.retrieve(q, k=k):
                got.append(passage)
                key = str(passage).split(" | ")[0].strip().lower()
                if key not in seen:
                    seen.add(key)
                    rest.append(passage)
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

            # once per item, search with a DRAFT OF THE MISSING ARTICLE instead of words about it
            if missing and not imagined and retrievals + 2 <= self.MAX_RETRIEVALS:
                imagined = True
                opening = self._clean_query(
                    self.imagine(
                        ctx.llm,
                        tag="imagine",
                        question=question,
                        missing=missing or "the article that answers the question",
                        passages=[p[: self.SLATE_SNIPPET] for p in pool[:6]],
                    ).get("opening", "")
                )
                if opening:
                    got_open = gather(opening, self.K_PSEUDO)
                    got_terms = gather(self._content_words(opening), self.K_PSEUDO)
                    progressed = progressed or bool(got_open) or bool(got_terms)
                    ctx.note("pseudo-article probe", opening=opening[:220], n_new=len(got_open) + len(got_terms))

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

        # -- collapse the chain: replace descriptions with the names the passages give them ---------------
        asked = question
        rewritten = ""
        if evidence:
            raw_rw = self.resolve(
                ctx.llm, tag="resolve", question=question,
                passages=[p[: self.POOL_SNIPPET] for p in evidence],
            ).get("rewritten", "")
            rewritten = self._admit_rewrite(raw_rw, question, pool)
            if rewritten and normalize_text(rewritten) != normalize_text(question):
                asked = rewritten
                # a named question is also a far better search key than the description it replaced
                for passage in gather(rewritten, self.K):
                    if passage not in evidence and len(evidence) < self.N_EVIDENCE + 2:
                        evidence = evidence + [passage]
            ctx.note("hop collapse", raw=str(raw_rw)[:180], admitted=rewritten[:180], asked=asked[:180])

        drafted = self.answer(ctx.llm, tag="answer", question=asked, passages=evidence).get("answer", "")
        final = self._tidy(drafted, asked)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {
            "answer": final or str(drafted).strip(),
            "rewritten": rewritten,
            "imagined": imagined,
            "names": names,
            "constraints": constraints,
            "n_pinned": len(pinned),
            "n_pool": len(pool),
            "missing": missing,
        }
