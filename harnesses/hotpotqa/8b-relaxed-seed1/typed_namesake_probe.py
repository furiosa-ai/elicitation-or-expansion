"""Look a name up the way the index files it: with the word that says what kind of thing it is.

Measured on the current best system's own search-set run: of the 49 gold articles it never retrieved on a
question it got wrong, 20 -- 41% -- are filed under a title that carries a sense qualifier in brackets.  A
band, a playwright, a film of a given year, a song by a given artist.  These are the articles an indirect
question is usually about, because a question that has to describe an entity rather than name it is a
question about the entity that did NOT get the plain title.

A bare-name probe cannot reach them.  The index ranks by term match, so the plain-titled article and every
person who happens to share the surname outrank the qualified one; the qualified title's extra word is the
only thing that separates them, and a bare name does not carry it.  Probing the name of a band returns eight
politicians and no band.  Adding the one word that says `band` puts the band first.  The admission filter
downstream already accepts a qualified title -- it has simply never been shown one.

So this harness stops treating a name as a query.  Every lookup is typed: the name, plus the words in its
immediate context that say what kind of thing it is -- the type nouns, the nationality, the year -- taken
from the question for a name the question mentions, and from the very sentence that wrote it for a name the
evidence mentions.  The type context is read off the text, never remembered, so a typed probe can only ask
for a kind of thing something already said the entity was.  A typed probe costs no model call, and the
frontier leaves half its retrievals unspent.

Disambiguation, routing, the rival hedge, the pseudo-article probe and the answer policy are unchanged.
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
_YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
# The vocabulary an encyclopedia puts in brackets after a name when several articles share it.  General
# words for kinds of things and for where a thing is from -- nothing about any particular entity.
_TYPE_LEX = frozenset(
    """film movie films documentary short television tv series sitcom drama sequel episode season
    band group duo trio orchestra choir song songs single album albums ep soundtrack track musical opera
    novel novels book books story stories poem play plays comic manga anthology magazine newspaper journal
    game games video videogame franchise app software company corporation firm label studio publisher
    brand product aircraft car ship submarine locomotive train rocket satellite weapon rifle tank engine
    river lake mountain island peak valley bay creek town city village borough county province district
    region state country municipality neighbourhood neighborhood park stadium arena bridge tower building
    church cathedral castle palace museum library hospital school college university academy institute
    airport station railway highway road route battle war treaty election party movement award prize
    footballer cricketer basketball baseball hockey golfer boxer athlete swimmer cyclist wrestler driver
    player coach manager referee actor actress singer songwriter musician composer conductor rapper dancer
    director producer screenwriter writer author novelist poet playwright journalist editor critic
    politician senator governor mayor president minister diplomat lawyer judge general admiral soldier
    scientist physicist chemist biologist mathematician astronomer economist historian philosopher
    psychologist engineer architect painter photographer sculptor designer chef bishop priest saint
    american british english scottish welsh irish french german italian spanish portuguese dutch belgian
    swiss austrian swedish norwegian danish finnish polish czech hungarian greek turkish russian ukrainian
    canadian mexican brazilian argentine chilean colombian peruvian australian zealand indian pakistani
    bangladeshi chinese japanese korean thai vietnamese indonesian filipino iranian israeli egyptian
    nigerian kenyan ghanaian ethiopian moroccan algerian
    rock pop jazz metal punk folk hip electronic classical country blues soul reggae disco
    """.split()
)
_LABEL = re.compile(
    r"^\s*(?:name|title|entity|article|constraint|requirement|mention)\s*\d*\s*[:\-]\s*", re.IGNORECASE
)

# A capitalised run, optionally bridged by the little words real titles keep inside them.  Everything the
# sweep opens comes out of this scanner, so nothing a model remembers can be searched for.
_CAPRUN = re.compile(
    r"[A-Z][\w'’\-]*\.?"
    r"(?:\s+(?:of|the|de|del|della|di|da|van|von|der|den|und|and|for|le|la|los|las|el|al|bin)?"
    r"\s*[A-Z][\w'’\-]*\.?)*"
)
_SENT_SPLIT = re.compile(r"(?<![A-Z]\.)(?<!\sSt\.)(?<=[.!?])\s+")
_HAS_DIGIT = re.compile(r"\d")
# names that are calendar, nationality or plain scaffolding carry no article worth opening
_MENTION_STOP = frozenset(
    """january february march april may june july august september october november december monday tuesday
    wednesday thursday friday saturday sunday american british english french german italian spanish russian
    canadian australian indian chinese japanese irish scottish welsh dutch swedish polish mexican brazilian
    european african asian north south east west northern southern eastern western new old university college
    school county city town state national international world war world-war he she it they this that these
    those there here his her their its who what which when where how why the a an and or but in on at to for
    of
    from by with as is was were are be been being he's however although while during after before also
    according both during since until though united states america january's list index
    """.split()
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

RIVALS_INSTRUCTIONS = """A search is one step short of answering a multi-step question, and the step it must
still take depends on WHICH entity the question is describing. Several of the passages could be that entity.

Your job is not to decide between them. It is to list the ones that are genuinely in the running, so the
missing fact can be looked up for each of them and the evidence can settle it.

In `reasoning`, say what the question describes at that step, then go through the passages and note which
ones fit the description -- including the ones that fit less well than the obvious front-runner, and the ones
that share a name with it but differ in year, medium, country or role.

Produce `rivals`: at most three lines, best first, each written

    <entity> :: <keyword query>

where `<entity>` is the entity's name copied exactly as a passage writes it -- never a name you remember but
the passages do not contain -- and `<keyword query>` is the query that would find the missing fact for THAT
entity: its name, plus the words naming the property being sought, and nothing else.

Every line must name a DIFFERENT entity. Three rephrasings of the same lookup are worth nothing here; the
point is to try the front-runner and its rivals at once. If only one entity in the passages can possibly be
the described one, write only its line.
"""


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
    K_RIVAL = 5             # narrow: an anchored lookup puts its answer high, and three wide
                            # probes would dilute the pool the router has to read
    MAX_RIVALS = 3          # how many competing referents may have their second hop run
    RIVAL_POOL = 10         # passages the rival nomination call is shown
    RESERVE = 2             # retrievals held back so the pseudo-article probe still runs
    MAX_RETRIEVALS = 12     # the two pseudo-article probes are additive, so the ceiling is raised
    CLOSE_MAX = 4           # how many named-but-unopened pages one sweep may open
    CLOSE_K = 6             # a bare-name probe is title-filtered, so it need not be wide
    CLOSE_RESERVE = 4       # retrievals left for the rival hedge and the pseudo-article probe
    CLOSE_SOURCES = 6       # passages the scanner reads: the ones routing leaned on
    CLOSE_ADMIT = 2         # pages admitted per name (the name itself, plus one qualified variant)
    TYPE_WORDS = 2          # how many type words a typed probe may carry: enough to name a kind, not a
                            # description -- this index answers names, not descriptions
    TYPE_WINDOW = 7         # how far either side of the mention the kind may be read from
    TYPED_RESERVE = 3       # retrievals held back so routing, the hedge and the sweep still run
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
        self.rivals = Module(
            inputs=["question", "missing", "passages"],
            outputs=["rivals"],
            instructions=RIVALS_INSTRUCTIONS,
            descs={
                "missing": "the fact the search has not found yet",
                "passages": "what the search has already returned, each `Title | abstract`",
                "rivals": "at most three lines `<entity> :: <keyword query>`, best first",
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

    @classmethod
    def _type_context(cls, name: str, context: str) -> str:
        """The words NEAREST the name that say what KIND of thing it is.

        Encyclopedic prose puts the kind right against the name -- `British rock band Bush`, `the film
        Sahara`, `Bush, released in 2000` -- while the same paragraph's other type words belong to other
        entities.  So eligibility is a window around the mention, nearest first, and only lower-case words
        the general lexicon of kinds already knows: the probe can never ask for a kind nobody wrote down.
        Two words, because a third turns a lookup into a description and this index answers names.
        """
        toks = _WORD.findall(str(context or ""))
        if not toks:
            return ""
        low = [t.lower() for t in toks]
        own = [w.lower() for w in _WORD.findall(str(name or ""))]
        if not own:
            return ""
        at = -1
        for i in range(len(low) - len(own) + 1):
            if low[i : i + len(own)] == own:
                at = i
                break
        if at < 0:                       # the name is not written in this context: no window to read
            return ""
        lo, hi = max(0, at - cls.TYPE_WINDOW), min(len(toks), at + len(own) + cls.TYPE_WINDOW)
        ranked: list[tuple[int, str]] = []
        for i in range(lo, hi):
            if lo <= i < at + len(own) and at <= i:
                continue
            tok, w = toks[i], low[i]
            dist = at - i if i < at else i - (at + len(own) - 1)
            if _YEAR.fullmatch(tok):
                # a bare year is not a kind of thing, and as a query term it drags the ranking toward
                # every article that merely happens to mention that year
                continue
            if tok[:1].isupper():        # a capital is another name, not a kind of thing
                continue
            if w in _TYPE_LEX and w not in own and all(w != r for _, r in ranked):
                ranked.append((dist, w))
        ranked.sort(key=lambda e: e[0])
        return " ".join(w for _, w in ranked[: cls.TYPE_WORDS])

    def _typed_query(self, name: str, context: str) -> str:
        types = self._type_context(name, context)
        return f"{name} {types}".strip() if types else ""

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

    @classmethod
    def _written_by(cls, name: str, passages: list[str]) -> bool:
        """A rival may only be chased if the retrieved text actually writes its name.

        Rivals are the point where a model is most tempted to supply an entity from memory, and a remembered
        name sends the whole probe into the same dead end the hedge exists to avoid.
        """
        toks = {w for w in cls._key_tokens(name) if len(w) > 1}
        if not toks:
            return False
        blob = {w.lower() for w in _WORD.findall(" ".join(passages))}
        return toks <= blob

    # -- the sweep ----------------------------------------------------------------------------
    @classmethod
    def _mine_names(cls, passages: list[str]) -> list[tuple[str, str]]:
        """Every capitalised name written in `passages`, in reading order, deduplicated.

        Sentence-initial single words are skipped: a capital there is grammar, not a name.  A run that
        begins with a connector or ends with one is trimmed back to its capitalised core.
        """
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for passage in passages:
            _, _, body = str(passage).partition(" | ")
            for sentence in _SENT_SPLIT.split(body):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for m in _CAPRUN.finditer(sentence):
                    raw = m.group(0).strip().strip("'’-")
                    toks = raw.split()
                    while toks and toks[0].lower() in _MENTION_STOP:
                        toks = toks[1:]
                    while toks and toks[-1].lower() in _MENTION_STOP:
                        toks = toks[:-1]
                    # a run that swallowed the sentence's full stop, or trails a bare initial, is trimmed
                    if toks and toks[-1].endswith(".") and len(toks[-1]) > 2:
                        toks = toks[:-1] + [toks[-1][:-1]]
                    while toks and len(toks[-1].rstrip(".")) < 2:
                        toks = toks[:-1]
                    if not toks or len(toks) > 6:
                        continue
                    name = " ".join(toks)
                    # a lone capital that opens the sentence is punctuation, not a page
                    if len(toks) == 1 and (m.start() == 0 or len(toks[0]) < 4):
                        continue
                    if _HAS_DIGIT.search(name):
                        continue
                    content = [w for w in toks if w.lower() not in _MENTION_STOP]
                    if not content:
                        continue
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((name, sentence))
        # a two-or-more word name identifies a page more reliably than a lone capital, which is as often a
        # country or a common noun -- but a one-word name is exactly how the rarest pages are titled, so the
        # two kinds are interleaved two-to-one instead of one kind being starved behind the other
        compound = [n for n in out if " " in n[0]]
        single = [n for n in out if " " not in n[0]]
        order: list[tuple[str, str]] = []
        while compound or single:
            for _ in range(2):
                if compound:
                    order.append(compound.pop(0))
            if single:
                order.append(single.pop(0))
        return order

    @classmethod
    def _already_open(cls, name: str, titles: list[str]) -> bool:
        """The pool already holds a page for this name, so opening it would buy nothing."""
        n = {w for w in cls._key_tokens(name) if w not in _MENTION_STOP}
        if not n:
            return True
        for title in titles:
            if n <= cls._key_tokens(title):
                return True
        return False

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
        imagined = False
        hedged = False

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
        typed_fired: list[str] = []
        for name in names:
            found = gather(name, self.K_NAME)
            # the bare name reaches whatever is famous under it; the question's own type words reach the
            # article the index had to file under a bracketed sense, which is the one an indirect question
            # is normally about.  Both result sets feed one slate, so disambiguation still chooses.
            typed = self._typed_query(name, question)
            if typed and retrievals + self.TYPED_RESERVE < self.MAX_RETRIEVALS:
                extra = gather(typed, self.K_NAME)
                if extra:
                    typed_fired.append(typed)
                    found = found + [p for p in extra if p not in found]
            slate: list[str] = []
            for passage in found:
                title = str(passage).split(" | ")[0].strip()
                if self._is_about(name, title) and passage not in slate:
                    slate.append(passage)
                if len(slate) == self.MAX_SLATE:
                    break
            if slate:
                slates.append((name, slate))
        if typed_fired:
            ctx.note("typed name probes", queries=typed_fired)

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
        linked: list[str] = []          # pages opened purely because the evidence named them
        swept = False

        def sweep(sources: list[str]) -> None:
            """Open every page the evidence names and the pool does not already hold. No model, no call."""
            nonlocal swept
            swept = True
            titles = [str(p).split(" | ")[0].strip() for p in (pinned + rest)]
            asked = self._key_tokens(question)
            opened: list[str] = []
            skipped = 0
            typed_used: list[str] = []
            for name, sentence in self._mine_names(sources[: self.CLOSE_SOURCES]):
                if len(opened) >= self.CLOSE_MAX:
                    break
                if retrievals + self.CLOSE_RESERVE >= self.MAX_RETRIEVALS:
                    break
                toks = {w for w in self._key_tokens(name) if w not in _MENTION_STOP}
                # the question's own names were probed already, in step 3
                if not toks or toks <= asked:
                    continue
                if self._already_open(name, titles):
                    skipped += 1
                    continue
                got = gather(name, self.CLOSE_K)
                if not got:
                    continue
                opened.append(name)
                admitted = 0
                for passage in got:
                    title = str(passage).split(" | ")[0].strip()
                    if admitted >= self.CLOSE_ADMIT:
                        break
                    if self._is_about(name, title):
                        pin(passage)
                        if passage not in linked:
                            linked.append(passage)
                        titles.append(title)
                        admitted += 1
                # the bare probe brought back nothing filed under this name -- the page it wants is the
                # one the index had to qualify.  Ask again with the kind of thing the sentence that wrote
                # the name says it is.
                if admitted < self.CLOSE_ADMIT and retrievals + self.CLOSE_RESERVE < self.MAX_RETRIEVALS:
                    typed = self._typed_query(name, sentence)
                    if typed:
                        for passage in gather(typed, self.CLOSE_K):
                            title = str(passage).split(" | ")[0].strip()
                            if admitted >= self.CLOSE_ADMIT:
                                break
                            if self._is_about(name, title):
                                pin(passage)
                                if passage not in linked:
                                    linked.append(passage)
                                titles.append(title)
                                admitted += 1
                                if typed not in typed_used:
                                    typed_used.append(typed)
            ctx.note("mention closure", opened=opened, already_open=skipped,
                     typed=typed_used, n_linked=len(linked))

        for round_i in range(self.MAX_ROUNDS):
            pool = pinned + rest
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[: self.N_EVIDENCE]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]

            # the sweep reads the passages routing actually leaned on, so it runs after the first route
            # and before anything can break out: even a `sufficient` pool may be missing a named page
            if not swept and retrievals + self.CLOSE_RESERVE < self.MAX_RETRIEVALS:
                sweep(selected or pool[: self.CLOSE_SOURCES])

            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            progressed = bool(gather(routed.get("next_query", ""), self.K))

            # once per item, run the missing fact's lookup separately for every entity that could be
            # the described one, instead of only for the one the router happened to anchor on
            if missing and not hedged and retrievals + self.RESERVE < self.MAX_RETRIEVALS:
                hedged = True
                raw = self.rivals(
                    ctx.llm,
                    tag="rivals",
                    question=question,
                    missing=missing,
                    passages=[p[: self.SLATE_SNIPPET] for p in pool[: self.RIVAL_POOL]],
                ).get("rivals", "")
                fired: list[str] = []
                refused: list[str] = []
                claimed: list[frozenset[str]] = []
                for line in str(raw).splitlines():
                    if "::" not in line or len(fired) >= self.MAX_RIVALS:
                        continue
                    head, _, tail = line.partition("::")
                    name = self._clean_query(_LABEL.sub("", _BULLET.sub("", head)))
                    query = self._clean_query(tail)
                    if not name or not query or len(_WORD.findall(name)) > 8:
                        continue
                    # a hedge over one entity written three ways is three phrasings of one query, not a hedge
                    key = frozenset(self._key_tokens(name))
                    if not key or any(key <= c or c <= key for c in claimed):
                        continue
                    if not self._written_by(name, pool):
                        refused.append(name)
                        continue
                    if retrievals + self.RESERVE >= self.MAX_RETRIEVALS:
                        break
                    claimed.append(key)
                    if gather(query, self.K_RIVAL):
                        fired.append(name)
                progressed = progressed or bool(fired)
                ctx.note("rival chains", missing=missing[:120], fired=fired, refused=refused)

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
        evidence = list(selected) or pool[: self.N_EVIDENCE]
        # pages opened by the sweep were never on the router's ballot in the round that produced
        # `selected`, so append the ones it has not seen rather than let them fall off the end
        for passage in linked:
            if passage not in evidence and len(evidence) < self.N_EVIDENCE + self.CLOSE_MAX:
                evidence.append(passage)
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
            "imagined": imagined,
            "hedged": hedged,
            "n_linked": len(linked),
            "names": names,
            "constraints": constraints,
            "n_pinned": len(pinned),
            "n_pool": len(pool),
            "missing": missing,
        }
