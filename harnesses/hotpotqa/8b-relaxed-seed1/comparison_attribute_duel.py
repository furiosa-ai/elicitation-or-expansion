"""Resolve two-entity comparison questions with a branch of their own instead of one generic read.

The frontier answers every question with a single call over one ranked pool.  Measured on its own search-set
run that policy is uniform in shape but not in quality: questions that compare two named entities convert at
65% while single-chain questions convert at 74%, and the comparison losses are not retrieval losses -- both
articles are in the pool.  They are losses of the last inch.  Two families recur.

*Shared-property* questions ("A and B are located in which country", "what job do A and B both have") want a
single word that BOTH articles write.  The generic answerer paraphrases it: it returns a plural where the
articles write a singular, a compound description where they write one noun, or the name of one entity where
the wanted word sits in both leads.  Nothing in the pipeline ever compares the two articles' vocabulary, so
the one signal that identifies the answer -- the word they have in common -- is never computed.

*Either-or* questions ("who is older, A or B", "which came first") want a comparison of two values pulled
from two different articles.  One call must locate both values in an eight-passage pool and order them, and
when it slips it slips silently.

This harness routes on question shape, model-free, and gives each family a resolution path.  Shared-property
questions get a per-entity dossier split and a scanner that intersects the two dossiers' vocabulary, offering
the ranked shared phrases as the wording the answer must be copied from.  Either-or questions get one
structured call that extracts the compared value from each entity's own dossier, and the ordering is then
done in arithmetic, not in prose.  Everything else -- disambiguation, routing, the mention closure sweep, the
rival hedge, the pseudo-article probe and the answer policy -- runs exactly as before, and the branch only
replaces the default answer when its own resolution succeeds.
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

# --- question-shape router (model-free) ---------------------------------------------------------------
# a shared-property comparison: two entities and a request for the one word they have in common
_CMP_SHARED = re.compile(
    r"\b(?:both|in common|have in common|share[ds]?\b|the same\b|do they share|they share)\b", re.IGNORECASE
)
# "A and B, are located in which country" / "A and B are what type of thing" -- a coordinated subject
# followed by a request for a category, with no or-choice anywhere
_CMP_CATEGORY = re.compile(
    r"\b(?:are|were|is|was)\b[^?]*\b(?:which|what)\b[^?]*\b"
    r"(?:country|city|state|nationality|occupation|profession|job|genre|type|kind|animal|species|breed|"
    r"category|sport|language|field|industry|instrument|position|role)\b",
    re.IGNORECASE,
)
# an either-or choice between two candidates, with or without the comma the question puts before `or`
_CMP_CHOICE = re.compile(
    r"(?:,?\s+\b\w[\w.'\-]*\b[^,?]*\s+or\s+|,\s*or\s+\S)", re.IGNORECASE
)
# a shared property is only shared if the question coordinates two things in the first place: without
# this the category pattern fires on ordinary single-chain questions that merely end in `what country`
_CMP_PAIRED = re.compile(r"\b(?:both|and)\b", re.IGNORECASE)
_OR_SPLIT = re.compile(r"\s+\bor\b\s+", re.IGNORECASE)
_AND_SPLIT = re.compile(r"\s+\band\b\s+", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split()
)}
# direction words: does the question want the SMALLER value or the LARGER one?
_SMALLER = re.compile(
    r"\b(?:older|oldest|earlier|earliest|first|sooner|before|elder|born first|died first|"
    r"smaller|smallest|shorter|shortest|fewer|fewest|lower|lowest|less|lighter)\b", re.IGNORECASE
)
_LARGER = re.compile(
    r"\b(?:younger|youngest|later|latest|newer|newest|more recent|most recent|last|after|"
    r"bigger|biggest|larger|largest|taller|tallest|longer|longest|higher|highest|more|most|heavier)\b",
    re.IGNORECASE,
)
_LEAD_CUT = re.compile(r"\s+\b(?:is|was|are|were)\b\s+", re.IGNORECASE)

SHARE_INSTRUCTIONS = """Two things are being compared and the question asks for the property they have in
common. Answer with the one word or short phrase that both articles use.

You are shown the article about each of them separately, and then `shared`: the words and phrases that
appear in BOTH articles, ranked, with the question\'s own words removed. The answer is almost always one of
those shared entries, written exactly as it is listed -- singular if it is listed singular, one noun if one
noun is listed. Do not pluralise it, do not join two entries together, do not wrap it in a description, and
do not answer with the name of either thing.

If no shared entry answers the question, copy the answering word verbatim from the articles instead. Output
the bare word or phrase and nothing else."""

DUEL_INSTRUCTIONS = """Two things are being compared on one property. Find that property\'s value for each of
them, separately, before anything is compared.

Read each article on its own. In `reasoning`, say what property the question compares (a birth date, a death
date, a founding year, a release date, a height, a population, a count).

Produce:
- `value_a`: that property\'s value for the FIRST thing, copied verbatim from its own article -- the full
  date, year or number as written. Write `unknown` if its article does not state it.
- `value_b`: the same property\'s value for the SECOND thing, copied verbatim from its own article, or
  `unknown`.
- `verdict`: the bare answer to the question -- normally the name of whichever one the question asks for,
  written exactly as the question writes it, or `yes` / `no` if the question asks whether something holds."""


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
    DOSSIER = 3             # passages per entity in a comparison dossier
    SHARED_MAX = 12         # shared phrases offered as the wording of a shared-property answer
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
        self.share = Module(
            inputs=["question", "article_a", "article_b", "shared"],
            outputs=["answer"],
            instructions=SHARE_INSTRUCTIONS,
            descs={
                "article_a": "everything the search holds about the first thing",
                "article_b": "everything the search holds about the second thing",
                "shared": "words and phrases both articles use, ranked, one per line",
                "answer": "the bare shared word or phrase",
            },
        )
        self.duel = Module(
            inputs=["question", "entity_a", "article_a", "entity_b", "article_b"],
            outputs=["value_a", "value_b", "verdict"],
            instructions=DUEL_INSTRUCTIONS,
            descs={
                "value_a": "the compared property's value for the first thing, verbatim, or `unknown`",
                "value_b": "the compared property's value for the second thing, verbatim, or `unknown`",
                "verdict": "the bare answer to the question",
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
    def _mine_names(cls, passages: list[str]) -> list[str]:
        """Every capitalised name written in `passages`, in reading order, deduplicated.

        Sentence-initial single words are skipped: a capital there is grammar, not a name.  A run that
        begins with a connector or ends with one is trimmed back to its capitalised core.
        """
        out: list[str] = []
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
                    out.append(name)
        # a two-or-more word name identifies a page more reliably than a lone capital, which is as often a
        # country or a common noun -- but a one-word name is exactly how the rarest pages are titled, so the
        # two kinds are interleaved two-to-one instead of one kind being starved behind the other
        compound = [n for n in out if " " in n]
        single = [n for n in out if " " not in n]
        order: list[str] = []
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

    # -- the comparison branch ----------------------------------------------------------------
    @classmethod
    def _pair(cls, question: str, names: list[str]) -> tuple[str, str] | None:
        """The two things being compared, in the order the question writes them. Model-free."""
        cand = [n for n in names if len(n) > 1]
        if len(cand) >= 2:
            first = sorted(cand, key=lambda n: question.lower().find(n.lower()) % 10_000)
            ordered = [n for n in first if question.lower().find(n.lower()) >= 0]
            if len(ordered) >= 2:
                return ordered[0], ordered[1]
            return cand[0], cand[1]
        # fall back to the capitalised runs the question coordinates
        mined = cls._mine_names([" | " + question])
        if len(mined) >= 2:
            return mined[0], mined[1]
        return None

    def _dossier(self, name: str, pool: list[str], limit: int) -> list[str]:
        """Everything the pool holds about one entity: its own article first, then pages that write it."""
        about: list[str] = []
        mentions: list[str] = []
        toks = {w for w in self._key_tokens(name) if w not in _MENTION_STOP}
        for passage in pool:
            title = str(passage).split(" | ")[0].strip()
            if self._is_about(name, title):
                about.append(passage)
            elif toks and toks <= {w.lower() for w in _WORD.findall(str(passage))}:
                mentions.append(passage)
        return (about + mentions)[:limit]

    @classmethod
    def _vocab(cls, passages: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        """Content unigrams and adjacent bigrams of a dossier, mapped to the surface form written."""
        uni: dict[str, str] = {}
        bi: dict[str, str] = {}
        for passage in passages:
            _, _, body = str(passage).partition(" | ")
            words = _WORD.findall(body)
            keep: list[str] = []
            for w in words:
                lw = w.lower()
                if lw in _QSTOP or lw in _MENTION_STOP or len(lw) < 3 or _HAS_DIGIT.search(lw):
                    keep.append("")
                    continue
                keep.append(lw)
                uni.setdefault(lw, w)
            for i in range(len(keep) - 1):
                if keep[i] and keep[i + 1]:
                    bi.setdefault(keep[i] + " " + keep[i + 1], words[i] + " " + words[i + 1])
        return uni, bi

    @classmethod
    def _shared(cls, dos_a: list[str], dos_b: list[str], question: str, limit: int) -> list[str]:
        """The vocabulary the two dossiers have in common, question words removed, phrases first.

        This is the only signal that identifies the answer to a shared-property question, and it is
        computable without a model: the word both articles write is the word the question is asking for.
        """
        asked = {w.lower() for w in _WORD.findall(str(question or ""))}
        uni_a, bi_a = cls._vocab(dos_a)
        uni_b, bi_b = cls._vocab(dos_b)
        phrases: list[str] = []
        taken: set[str] = set()
        for key in bi_a:
            if key in bi_b and not set(key.split()) & asked:
                phrases.append(bi_a[key])
                taken.update(key.split())
        singles = [uni_a[k] for k in uni_a if k in uni_b and k not in asked and k not in taken]
        # a phrase carries more of the category than either of its words, so phrases lead
        return (phrases + singles)[:limit]

    @staticmethod
    def _value(text: str) -> tuple[float, ...] | None:
        """A comparable magnitude for one side of an either-or question: a date, a year or a number."""
        raw = str(text or "").strip()
        if not raw or raw.lower().startswith(("unknown", "none", "not ")):
            return None
        low = raw.lower()
        years = _YEAR.findall(raw)
        if years:
            year = float(years[0])
            month = 0.0
            day = 0.0
            for name, num in _MONTHS.items():
                if name in low:
                    month = float(num)
                    break
            m = re.search(r"\b([0-3]?[0-9])\b(?!\d)", low.replace(years[0], " "))
            if m and month:
                day = float(m.group(1))
            return (year, month, day)
        nums = _NUMBER.findall(raw.replace(",", ""))
        if nums:
            try:
                return (float(nums[0]),)
            except ValueError:
                return None
        return None

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
            for name in self._mine_names(sources[: self.CLOSE_SOURCES]):
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
            ctx.note("mention closure", opened=opened, already_open=skipped, n_linked=len(linked))

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

        # -- comparison branch: a question that sets two entities against each other is resolved by
        #    comparing their two articles, not by reading one ranked pool -------------------------
        kind = ""
        pair = None
        if not _BOOL_QUESTION.match(question or "") or _CMP_CHOICE.search(question or ""):
            if _CMP_CHOICE.search(question or ""):
                kind = "choice"
            elif (_CMP_SHARED.search(question or "") or _CMP_CATEGORY.search(question or "")) and (
                _CMP_PAIRED.search(question or "")
            ):
                kind = "shared"
        resolved = ""
        if kind:
            pair = self._pair(question, names)
        if kind and pair:
            ent_a, ent_b = pair
            dos_a = self._dossier(ent_a, evidence + pool, self.DOSSIER)
            dos_b = self._dossier(ent_b, evidence + pool, self.DOSSIER)
            if not dos_a or not dos_b:
                kind = ""
        if kind == "shared" and pair:
            shared = self._shared(dos_a, dos_b, question, self.SHARED_MAX)
            picked = self.share(
                ctx.llm,
                tag="share",
                question=question,
                article_a=[p[: self.POOL_SNIPPET] for p in dos_a],
                article_b=[p[: self.POOL_SNIPPET] for p in dos_b],
                shared="\n".join(shared) if shared else "none",
            ).get("answer", "")
            resolved = self._tidy(picked, question)
            ctx.note("shared property", entities=[ent_a, ent_b], shared=shared[:8], answer=resolved[:80])
        elif kind == "choice" and pair:
            duelled = self.duel(
                ctx.llm,
                tag="duel",
                question=question,
                entity_a=ent_a,
                article_a=[p[: self.POOL_SNIPPET] for p in dos_a],
                entity_b=ent_b,
                article_b=[p[: self.POOL_SNIPPET] for p in dos_b],
            )
            resolved = self._tidy(duelled.get("verdict", ""), question)
            va = self._value(duelled.get("value_a", ""))
            vb = self._value(duelled.get("value_b", ""))
            ordered = ""
            # the ordering itself is arithmetic; only the two values are read by the model
            if va and vb and len(va) == len(vb) and va != vb:
                if _SMALLER.search(question or ""):
                    ordered = ent_a if va < vb else ent_b
                elif _LARGER.search(question or ""):
                    ordered = ent_a if va > vb else ent_b
            if ordered and resolved.lower() not in ("yes", "no"):
                # only overrule the prose verdict when it named the other side of the same choice
                if self._key_tokens(resolved) <= self._key_tokens(ent_a) | self._key_tokens(ent_b):
                    resolved = ordered
            ctx.note(
                "either-or duel",
                entities=[ent_a, ent_b],
                values=[str(duelled.get("value_a", ""))[:60], str(duelled.get("value_b", ""))[:60]],
                arithmetic=ordered,
                answer=resolved[:80],
            )
        if resolved and not _NON_ANSWER.match(resolved) and not _HEDGE_PREFIX.match(resolved):
            final = resolved

        return {
            "answer": final or str(drafted).strip(),
            "comparison": kind,
            "imagined": imagined,
            "hedged": hedged,
            "n_linked": len(linked),
            "names": names,
            "constraints": constraints,
            "n_pinned": len(pinned),
            "n_pool": len(pool),
            "missing": missing,
        }
