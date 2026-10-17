"""Navigate the encyclopedia by opening articles: the only search action is `lookup(<name>)`.

Three measurements over the index, taken from the frontier's own traces, say what a lexical index over
abstracts actually is.  Given a required article's own title as the query it returns that article inside
k=20 for 87% of the titles the frontier failed to retrieve; over 150 random training titles the article
ranks first 63% of the time and inside the top five 87% of the time.  Given the question's keywords instead
it recovers 14% of those missing titles, and given keywords anchored on an already-retrieved article, 7%.
The index is therefore a lookup table keyed by NAME that answers descriptions badly -- and 9% of the missing
titles are spelled out in the question itself, so even the first hop is losing articles it was handed.

Every system so far has treated retrieval as keyword search and the reading step as a separate stage that
consumes whatever the search returned.  This one collapses the two into navigation.  There is exactly one
search action, `lookup(name)`: probe the name wide, then keep the passages whose TITLE answers to it, exact
title first and a title that merely contains the name second.  Ranking by title match is what makes a wide
probe usable -- the named article is first only 63% of the time, so a narrow probe misses it while a wide
unfiltered probe returns twenty namesakes.  The names the question spells out are opened first, which is a
model-free coverage guarantee no descriptive query provides.

Then the loop reads what is open and does one of two things per turn: answer, or name the next article to
open.  The name it gives is snapped onto a capitalised mention mined from the text that is actually open, so
an invented name is never searched for, and an article already open is never opened twice.  Hop count is
therefore set by the question rather than by the pipeline: a question whose chain is already on the page
answers on turn one, while a three-link chain spends three turns.  A keyword query still runs once as a
recall floor, and its results sit behind the opened articles as background rather than as evidence.
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

PLAN_INSTRUCTIONS = """A multi-step question must be answered out of an encyclopedia whose articles are
reached by name. Your job is to list the names to open first.

In `reasoning`, say what chain of articles the question requires and which of its links it already names.

Produce:
- `names`: the names the question spells out, one per line, at most three, each written as the encyclopedia
  would title that article -- the bare name only, with no role words, no dates and no question words around
  it. Write `none` if the question names nothing.
- `target`: what kind of thing the answer will be (a person, a year, a place, a title, a number, a verdict)."""

STEP_INSTRUCTIONS = """You are reading an encyclopedia one article at a time to answer a multi-step question.

`open` lists the articles you have opened, with their opening text. `background` lists unopened search
results, which may be about the wrong entity that happens to share a name -- treat them as hints only, never
as established fact.

In `reasoning`, state the chain of facts the question needs, then which links the open articles already
establish and which link is still unestablished.

Then do exactly one of two things.

If the open articles together establish every link, produce `answer_now`: `yes` and put the answer in
`answer`: the bare short answer only -- an entity, name, title, date, number or verdict, worded as the
articles word it, no sentence and no explanation. If the question asks whether something is or is not the
case, write exactly `yes` or `no` in `answer`.

Otherwise produce `answer_now`: `no` and put in `link` the name of the ONE article to open next. The missing
link is almost always held by the article of an entity that the open text NAMES but does not describe -- a
person, work, place or organisation it mentions in passing. Copy that name exactly as the text spells it,
nothing else: no description, no question words, no name that is not written in the text, and never the name
of an article that is already open. Leave `answer` empty."""

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
    K_LOOKUP = 20           # the named article ranks first only 63% of the time: probe wide, rank by title
    K_FLOOR = 8             # width of the single keyword query kept as a recall floor
    MAX_NAMES = 3           # names taken from the question and opened up front
    MAX_OPENED = 2          # title-matching articles kept per lookup
    ADMIT_BG = 4            # passages a lookup may add to the background pool
    MAX_TURNS = 4           # turns of read-then-open; hop count is set by the question
    MAX_RETRIEVALS = 12
    MAX_MENTIONS = 900
    OPEN_SNIPPET = 420
    BG_SNIPPET = 200
    N_BG = 8

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question"],
            outputs=["names", "target"],
            instructions=PLAN_INSTRUCTIONS,
            descs={
                "names": "names to open, one per line, at most three, or `none`",
                "target": "what kind of thing the answer is",
            },
        )
        self.step = Module(
            inputs=["question", "open", "background"],
            outputs=["answer_now", "answer", "link"],
            instructions=STEP_INSTRUCTIONS,
            descs={
                "open": "articles already opened, each `Title | opening text`",
                "background": "unopened search results, each `Title | opening text`",
                "answer_now": "yes or no",
                "answer": "the bare short answer when `answer_now` is yes, else empty",
                "link": "the name of the next article to open when `answer_now` is no, else empty",
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

    def _first_line(self, raw: str) -> str:
        lines = self._parse_lines(raw, 1, 7)
        return lines[0] if lines else ""

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
        opened_pages: list[str] = []     # articles opened by name, in the order they were opened
        background: list[str] = []       # unopened search results, hints only
        seen: set[str] = set()
        used_queries: set[str] = set()
        opened_names: set[str] = set()
        retrievals = 0

        def lookup(name: str) -> list[str]:
            """The only search action: open the article(s) the index holds under `name`.

            Wide probe, then keep the title matches -- exact title first, a title that contains the name
            second. Everything else from the probe is demoted to background, capped, so one lookup cannot
            bury the page under namesakes.
            """
            nonlocal retrievals
            q = self._clean_query(name)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return []
            used_queries.add(q.lower())
            retrievals += 1
            got = list(ctx.tools.retrieve(q, k=self.K_LOOKUP))
            ranked = sorted(range(len(got)), key=lambda i: (self._title_rank(q, got[i]), i))
            hits: list[str] = []
            budget = self.ADMIT_BG
            for i in ranked:
                passage = got[i]
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                if self._title_rank(q, passage) < 2 and len(hits) < self.MAX_OPENED:
                    hits.append(passage)
                    opened_pages.append(passage)
                elif budget > 0:
                    background.append(passage)
                    budget -= 1
            opened_names.add(" ".join(sorted(self._key_tokens(q))))
            ctx.note("opened", name=q, titles=[str(p).split(" | ")[0] for p in hits])
            return hits

        def floor(query: str) -> None:
            """One keyword query, kept as background: the recall floor under the navigation."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return
            used_queries.add(q.lower())
            retrievals += 1
            for passage in ctx.tools.retrieve(q, k=self.K_FLOOR):
                key = str(passage).split(" | ")[0].strip().lower()
                if key not in seen:
                    seen.add(key)
                    background.append(passage)

        # 1. the names the question spells out are opened first: a coverage guarantee keywords never give
        planned = self.plan(ctx.llm, tag="plan", question=question)
        names = self._parse_lines(planned.get("names", ""), self.MAX_NAMES, 8)
        for name in names:
            lookup(name)
        floor(self._keyterms(question))

        # 2. read what is open, then either answer or name the next article to open
        answer_raw = ""
        links: list[str] = []
        turns = 0
        for turn in range(self.MAX_TURNS):
            turns = turn + 1
            shown = [f"{p[: self.OPEN_SNIPPET]}" for p in opened_pages] or ["none yet"]
            hints = [p[: self.BG_SNIPPET] for p in background[: self.N_BG]] or ["none"]
            stepped = self.step(
                ctx.llm, tag=f"step{turns}", question=question, open=shown, background=hints
            )
            said = self._tidy(stepped.get("answer", ""), question)
            done = "yes" in str(stepped.get("answer_now", "")).strip().lower()[:4]
            if done and said and not _NON_ANSWER.match(said) and not _HEDGE_PREFIX.match(said):
                answer_raw = said
                break
            if turn == self.MAX_TURNS - 1 or retrievals >= self.MAX_RETRIEVALS:
                answer_raw = answer_raw or said
                break
            target = self._snap(
                self._first_line(stepped.get("link", "")),
                self._mine_mentions(opened_pages + background[: self.N_BG]),
                opened_names,
            )
            if not target:
                answer_raw = answer_raw or said
                break
            links.append(target)
            if not lookup(target):
                # nothing under that name: the page is unchanged, so a further turn would repeat itself
                answer_raw = answer_raw or said
                if retrievals >= self.MAX_RETRIEVALS:
                    break

        final = self._tidy(answer_raw, question)
        ctx.note("navigation", names=names, links=links, turns=turns,
                 n_open=len(opened_pages), retrievals=retrievals)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(
                ctx.llm, tag="commit", question=question,
                passages=[p[: self.OPEN_SNIPPET] for p in opened_pages]
                + [p[: self.BG_SNIPPET] for p in background[:6]],
            ).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", after=final[:120])

        return {
            "answer": final,
            "names": names,
            "links": links,
            "turns": turns,
            "n_open": len(opened_pages),
            "n_background": len(background),
        }
