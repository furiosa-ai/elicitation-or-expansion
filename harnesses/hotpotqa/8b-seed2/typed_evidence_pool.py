"""Label every retrieved passage with what its own abstract declares it to be, before anything reads it.

The frontier's evidence pool is 14 lines of `Title | text` in BM25 order, and the answering call has to work
out from scratch what each page even is. Its residual errors say that this is where it slips: on the search
set it answers a question asking which artist did something with the album, a question asking who co-starred
with the film, a question asking which programme an actor is known for with a page about awards, and a
question asking for the year of an event with a person. In each case a page of the right kind is in the pool
and the model mistook the role of the page it read. Those answers are lexically plausible, score f1 > 0 and
are invisible to every length, verbosity and grounding test.

The mechanism adds a label, not reasoning. An encyclopaedia abstract opens by declaring what its subject is
("... is a 1966 British adventure film", "... was a Scottish politician", "... is a genus of flowering
plants"), so parsing that opening copula out of each passage yields a kind for every pooled page that comes
from the corpus rather than from the model. The pool is presented as `Title [is a 1966 British adventure
film] | text`, newly-retrieved pages first exactly as before, and the answering instructions say what the
bracket means and that the answer must be taken from a page whose kind matches the kind the question asks for.

A replay over the frontier's own 300 outputs is why the label is presented rather than enforced: a
deterministic contradiction test between the declared kind and the kind the question demands fires on only 4
items and is right on 2, because most category pairs are not exclusive (a band answers "who", a fictional
character answers "who", a parenthetical disambiguator makes a title a different page). As information for
the reader it costs nothing and applies to every item; as a rule it would be noise. Retrieval, both queries,
the triage call, the demos, the style note, the sufficiency route, the spare call and the polisher are
byte-identical to the frontier, so the delta is exactly the corpus's own statement of what each page is.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module

# --------------------------------------------------------------------------------------------------
# answer-shape post-processing (shared contract with the typed-answer candidate)
# --------------------------------------------------------------------------------------------------
HEDGES = re.compile(
    r"(?i)\b(not (?:specified|mentioned|provided|available|determinable)|impossible to determine|"
    r"cannot be determined|no information|insufficient information|unknown|unclear)\b"
)
LEAD_IN = re.compile(
    r"(?i)^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*|^(?:answer|a)\s*:\s*|^(?:it|that)\s+(?:is|was)\s+"
)
YESNO_Q = re.compile(
    r"(?i)^(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should|both|the same)\b"
)
TRAILING_CLAUSE = re.compile(r"(?i)\s*[,;]\s*(?:while|whereas|although|though|but|because|since|so)\b.*$")
PAREN = re.compile(r"\s*\((?P<body>[a-z][^()]{0,80})\)\s*$")
QUOTED = re.compile(r"[\"“‘']([^\"”’']{2,60})[\"”’']")
CAP_SPAN = re.compile(r"[A-Z][\w.&'’-]*(?:\s+(?:of|the|de|la|von|van|and|&)?\s*[A-Z0-9][\w.&'’-]*)*")
ARTICLES = re.compile(r"(?i)^(?:the|a|an)\s+")
CHOICE_Q = re.compile(
    r"(?i)\bor\b|\bwhich (?:one|of)\b|\b(?:more|less|fewer|earlier|later|older|younger|larger|smaller|longer|shorter|higher|greater|first|closer|newer|bigger)\b"
)
NULL_QUERY = re.compile(r"(?i)^(none|n/?a|null|nil|nothing|unknown|no query|not needed|not applicable|-+)$")
STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an", "of", "in",
    "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "did", "does", "do", "has", "have",
    "had", "that", "this", "with", "by", "from", "as", "it", "its", "his", "her", "their", "both", "name",
    "known", "also", "than", "more", "most", "first", "same", "other", "one", "two", "he", "she", "they",
}


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return " ".join(s.split())


def _content_tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t not in STOP and len(t) > 2}


def _named_spans(question: str) -> list[str]:
    spans = [m.group(1).strip() for m in QUOTED.finditer(question)]
    body = question[1:] if question[:1].isupper() else question
    for m in CAP_SPAN.finditer(body):
        span = m.group(0).strip(" ,.;:")
        if len(span) > 2:
            spans.append(span)
    seen, out = set(), []
    for s in spans:
        key = _norm(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _first_sentence(a: str) -> str:
    toks = a.split()
    if len(toks) <= 8:
        return a
    for i, tok in enumerate(toks[:-1]):
        if tok.endswith(".") and len(tok.rstrip(".")) > 3 and toks[i + 1][:1].isupper():
            return " ".join(toks[: i + 1])
    return a


def polish(raw: str, answer_type: str, question: str) -> str:
    a = (raw or "").strip()
    a = re.sub(r"\[\[.*?\]\]", " ", a)
    a = a.replace("**", "").replace("*", "").replace("`", "").strip()
    for line in a.splitlines():
        if line.strip():
            a = line.strip()
            break
    a = LEAD_IN.sub("", a).strip().strip("\"“”‘’ ")

    t = (answer_type or "").strip().lower()
    head = re.match(r"(?i)^(yes|no)\b", a)
    if head and (t.startswith("yes") or YESNO_Q.match(question.strip()) or len(a.split()) > 2):
        return head.group(1).lower()

    a = TRAILING_CLAUSE.sub("", a).strip()
    a = _first_sentence(a).rstrip(".;:, ")
    if len(a.split()) > 1:
        a = PAREN.sub("", a).strip()

    if len(a.split()) > 4 and CHOICE_Q.search(question):
        na = _norm(a)
        hits = [o for o in _named_spans(question) if _norm(o) and _norm(o) in na]
        hits = [o for o in hits if not any(_norm(o) != _norm(p) and _norm(o) in _norm(p) for p in hits)]
        if len(hits) == 1 and len(hits[0].split()) <= 6:
            a = hits[0]

    return ARTICLES.sub("", a).strip() or (raw or "").strip()


def _two_options(question: str) -> list[str]:
    """The two named alternatives of an `A or B` choice question, when the question really is one."""
    parts = re.split(r"\bor\b", question, maxsplit=1)
    if len(parts) != 2:
        return []
    left = _named_spans(parts[0])
    right = _named_spans(parts[1])
    if not left or not right:
        return []
    a, b = left[-1], right[0]
    if _norm(a) == _norm(b) or _norm(a) in _norm(b) or _norm(b) in _norm(a):
        return []
    return [a, b]


def _grounded(answer: str, evidence: list[str]) -> bool:
    """Is every content word of the answer actually written in the evidence?

    Token level, not substring, so a date the model reordered or a name it re-spaced still counts as copied.
    An answer that fails this was composed from memory whatever the call claimed about its evidence.
    """
    toks = [t for t in _norm(answer).split() if t not in STOP]
    if not toks:
        return True
    have = set(_norm(" ".join(evidence)).split())
    return all(t in have for t in toks)


def _derived(answer: str, answer_type: str) -> bool:
    """Answers that are legitimately absent from the text because they are computed or chosen."""
    a = _norm(answer)
    if a in ("yes", "no") or (answer_type or "").strip().lower().startswith("yes"):
        return True
    return bool(a) and all(t.isdigit() for t in a.split())


def _status(raw: str) -> str:
    """The evidence verdict, defaulting to `complete` when the call did not give a usable one."""
    s = _norm(raw)
    for key in ("complete", "partial", "absent"):
        if key in s.split() or s.startswith(key):
            return key
    if "not " in s or "missing" in s or "insufficient" in s or "incomplete" in s:
        return "partial"
    return "complete"


def _short(raw: str) -> str:
    s = (raw or "").strip()
    for line in s.splitlines():
        if line.strip():
            s = line.strip()
            break
    s = s.strip("\"“”‘’ ").rstrip(".")
    return "" if _norm(s) in ("none", "n a", "na", "nothing", "") else s[:200]


def _needs_repair(answer: str, answer_type: str, question: str, length_budget: int = 6) -> bool:
    if not answer.strip():
        return True
    if HEDGES.search(answer):
        return True
    if len(answer.split()) > max(3, length_budget):
        return True
    if (answer_type or "").strip().lower().startswith("yes") and answer.lower() not in ("yes", "no"):
        return True
    opts = _two_options(question)
    if opts and answer.lower() not in ("yes", "no"):
        na = _norm(answer)
        # the question asked to pick one of two named things but the answer names neither of them
        if not any(_norm(o) in na or na in _norm(o) for o in opts):
            return True
    return False



# --------------------------------------------------------------------------------------------------
# type oracle: what each retrieved page declares itself to be, and what kind the question demands
# --------------------------------------------------------------------------------------------------
# the copula phrase an encyclopaedia abstract opens with: "<subject> is/was a <kind> ..."
COPULA = re.compile(r"(?i)\b(?:is|was|are|were)\s+(?:one of\s+)?(?:an?|the)?\s*([^.;:()]{3,90})")
BORN_DIED = re.compile(r"(?i)\((?:born|b\.|\d{1,2}\s+\w+\s+\d{3,4}\s*[-\u2013]|\d{4}\s*[-\u2013])")
MONTHS = re.compile(
    r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
)
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "none", "several", "many", "few", "both",
}
# a kind word -> the category it puts its subject in. Only concrete, mutually exclusive categories are
# listed: anything unlisted leaves the page uncategorised and therefore unguarded.
KIND_WORDS: dict[str, str] = {}
for _cat, _words in {
    "person": (
        "actor actress singer songwriter musician rapper composer player footballer cricketer boxer golfer "
        "politician lawyer judge author writer novelist poet playwright journalist director producer "
        "screenwriter painter sculptor architect physicist chemist mathematician biologist economist "
        "historian philosopher engineer businessman businesswoman entrepreneur executive teacher professor "
        "soldier general admiral officer pilot astronaut priest bishop pope monk rabbi king queen emperor "
        "prince princess duke earl baron president senator congressman governor mayor ambassador diplomat "
        "coach manager wrestler driver cyclist swimmer skater gymnast presenter comedian magician model "
        "dancer chef photographer entertainer broadcaster critic scientist physician doctor surgeon nurse "
        "activist preacher pastor conductor violinist pianist guitarist drummer bassist vocalist voice"
    ),
    "work": (
        "film movie album song single novel book play series sitcom soap drama documentary anime manga "
        "comic magazine newspaper journal poem painting opera musical operetta soundtrack ep mixtape "
        "videogame game show programme program episode season franchise trilogy screenplay memoir "
        "autobiography biography textbook anthology sculpture statue symphony concerto ballet track"
    ),
    "place": (
        "city town village hamlet borough suburb neighbourhood neighborhood municipality county state "
        "country province region district territory island peninsula archipelago mountain peak hill "
        "volcano river stream creek lake reservoir bay gulf sea ocean desert forest valley canyon "
        "canal road highway street bridge railway railroad park reserve beach cape strait harbour harbor"
    ),
    "structure": (
        "building skyscraper tower stadium arena hall theatre theater museum library church cathedral "
        "temple mosque synagogue castle palace fort fortress monument memorial lighthouse dam airport "
        "station terminal hospital prison hotel casino mall restaurant pub bar school"
    ),
    "org": (
        "band group duo trio quartet orchestra ensemble company corporation firm business manufacturer "
        "publisher label studio network broadcaster channel station university college academy institute "
        "society association organisation organization agency authority ministry department bureau charity "
        "foundation team club franchise league federation union party movement government council court "
        "bank airline railway operator brand chain"
    ),
    "event": (
        "war battle siege campaign revolution rebellion uprising election referendum festival ceremony "
        "tournament championship cup olympics games race match competition conference summit massacre "
        "earthquake hurricane flood eruption disaster crash accident attack bombing strike protest riot "
        "expedition voyage mission tour season"
    ),
    "species": (
        "genus species family subfamily tribe plant flower shrub tree herb grass moth butterfly beetle "
        "insect spider bird fish frog snake lizard mammal rodent bat whale dinosaur fungus algae bacterium "
        "breed cultivar"
    ),
}.items():
    for _w in _words.split():
        KIND_WORDS[_w] = _cat

# the interrogative -> the category the answer must belong to
HEAD_NOUN_Q = re.compile(r"(?i)\b(?:what|which)\s+((?:[a-z-]+\s+){0,3}?[a-z-]+)\b")


def page_kind(text: str) -> tuple[str, str]:
    """What a pooled passage declares its subject to be: (category, the declared phrase).

    Only the abstract's own opening copula is consulted, so the verdict is the corpus's, not the model's.
    """
    _, _, body = (text or "").partition(" | ")
    body = body.strip()
    if not body:
        return "", ""
    lead = re.split(r"(?<=[a-z0-9\)])\.\s+", body)[0][:400]
    m = COPULA.search(lead)
    phrase = (m.group(1).strip() if m else "")[:90]
    cat = ""
    # only the head of the declared phrase decides the category: further right it drifts into other
    # subjects ("co-creator, along with X, of a comic") and mislabels the page
    for tok in _norm(phrase).split()[:5]:
        if tok in KIND_WORDS:
            cat = KIND_WORDS[tok]
            break
    if not cat and BORN_DIED.search(lead[:200]):
        cat = "person"
    return cat, phrase


def wanted_kind(question: str) -> str:
    """The category the question demands back, read off its interrogative. Empty when unclear."""
    q = " " + " ".join((question or "").lower().split()) + " "
    if re.search(r"(?i)\b(how many|how much|how long|how old|how tall|how far|how deep|how high)\b", q):
        return "quantity"
    if re.search(r"(?i)\b(what year|which year|what date|which date|in what year|in which year)\b", q) or re.search(
        r"(?i)\bwhen (?:was|were|did|do|does|is|are|had|has)\b", q
    ):
        return "date"
    if re.search(r"(?i)\b(where (?:is|was|are|were|did|do|does)|in what (?:city|town|country|state|province|county|region)|what (?:city|town|country|state|province|county))\b", q):
        return "place"
    if re.search(r"(?i)\b(who|whom|whose)\b", q):
        return "person"
    for m in HEAD_NOUN_Q.finditer(q):
        for tok in m.group(1).split():
            tok = tok.strip(",.;:'\"")
            if tok in KIND_WORDS:
                return KIND_WORDS[tok]
            if tok.endswith("s") and tok[:-1] in KIND_WORDS:
                return KIND_WORDS[tok[:-1]]
    return ""


def _has_quantity(answer: str) -> bool:
    a = _norm(answer)
    if any(ch.isdigit() for ch in a):
        return True
    return any(t in NUMBER_WORDS for t in a.split())


def _has_date(answer: str) -> bool:
    return bool(re.search(r"\b\d{3,4}\b", answer or "")) or bool(MONTHS.search(answer or ""))


# --------------------------------------------------------------------------------------------------
# question-form index learned from the train split
# --------------------------------------------------------------------------------------------------
WH = (
    "how many", "how much", "how long", "how old", "how tall", "how far", "what year", "what nationality",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
)
AUX = ("is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "will", "would")
HEAD_SKIP = {"is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "of", "this", "that", "these"}
QUALIFIER = re.compile(r",\s*[A-Z0-9]")
JOINER = re.compile(r"(?i)\b(and|or)\b|[,;&]|/")


def form_key(question: str) -> str:
    """Shape of a question: interrogative + head noun, plus a marker for two-way comparisons."""
    q = " " + re.sub(r"\s+", " ", (question or "").lower().strip()) + " "
    q = re.sub(r"[^a-z0-9 ]", " ", q)
    q = " " + " ".join(q.split()) + " "
    sig = ""
    for w in WH:
        idx = q.find(" " + w + " ")
        if idx >= 0:
            tail = q[idx + len(w) + 2 :].split()
            head = ""
            for tok in tail[:3]:
                if tok not in HEAD_SKIP and len(tok) > 2:
                    head = tok
                    break
            sig = (w + " " + head).strip() if " " not in w else w
            break
    if not sig:
        first = q.split()[0] if q.split() else ""
        sig = "yesno " + first if first in AUX else "other"
    cmp_marker = "cmp" if (" or " in q or " both " in q or " same " in q or " in common " in q) else "brg"
    return sig + "|" + cmp_marker


def _shape(answer: str) -> dict[str, Any]:
    a = (answer or "").strip()
    toks = a.split()
    return {
        "len": len(toks),
        "yesno": a.lower() in ("yes", "no"),
        "qualifier": bool(QUALIFIER.search(a)),
        "numeric": bool(toks) and any(ch.isdigit() for ch in toks[0]),
        "proper": bool(toks) and toks[0][:1].isupper(),
        "joined": bool(len(toks) > 1 and JOINER.search(a)),
    }


def _style_note(demos: list[dict[str, Any]]) -> str:
    """A style summary derived from the selected examples only -- no hand-picked thresholds."""
    if not demos:
        return "no comparable examples were available; give exactly the span the question asks for, no more"
    shapes = [_shape(d["a"]) for d in demos]
    n = len(shapes)
    lens = sorted(s["len"] for s in shapes)
    med = lens[n // 2]
    bits = [f"reference answers to questions of this form are {min(lens)}-{max(lens)} words long (typically {med})"]
    if sum(s["yesno"] for s in shapes) * 2 > n:
        bits.append("and are usually just yes or no")
    else:
        if sum(s["qualifier"] for s in shapes):
            bits.append(
                f"{sum(s['qualifier'] for s in shapes)} of {n} keep a trailing qualifier after a comma "
                "(a state, a country or a date), so do not strip one"
            )
        if sum(s["numeric"] for s in shapes) * 2 > n:
            bits.append("and are bare numbers or dates, with no words around them")
        elif sum(s["proper"] for s in shapes) * 2 > n:
            bits.append("and are proper names or titles, capitalised as the source writes them")
        elif sum(s["proper"] or s["numeric"] for s in shapes) == 0:
            bits.append("and are lower-case common nouns, not names")
        if sum(s["joined"] for s in shapes):
            bits.append("some join two items with a word such as \"and\" rather than with a bare space")
    return "; ".join(bits)


# --------------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------------
TRIAGE_INSTRUCTIONS = """You are running the first of two searches for a question that needs facts from two
different encyclopaedia pages.

`retrieved_titles` lists the pages already retrieved, and `passages` are their abstracts.

Produce:
- `facts`: the sentences from `passages` that bear on the question, copied word for word, each on its own
  line and prefixed with the page title. Write `none` if nothing in the passages is relevant.
- `answer_type`: one of yesno, person, organisation, location, date, number, title, other.
- `next_query`: the search query for the SECOND search. Follow these rules exactly:
  * it must name the ONE page that is still missing and is needed to finish the answer -- typically the
    entity the first page points to (a bridge), or, when the question compares two named things, the one
    of them that is not in `retrieved_titles` yet;
  * write it the way that page would be titled: a bare noun phrase of at most six words, optionally with
    a short disambiguating word (for example a year, a nationality or a category);
  * it must NOT be a sentence, must NOT contain a question mark, must NOT restate the question, and must
    NOT name a page that is already in `retrieved_titles`;
  * never state the answer in it;
  * `next_query` is never empty and is never `none`, `n/a` or `unknown`. If the first search already seems
    to settle the question, use the second search to check it: name the most specific *other* page the
    relevant passage points at -- the album, film, event, organisation or person the question depends on --
    written as that page's title. If the passages were all irrelevant, name the main subject of the question
    the way an encyclopaedia would title its page."""

ANSWER_INSTRUCTIONS = """Answer `question` from `passages`, which are encyclopaedia abstracts written as
`Title | text`, and from `facts`, which were copied out of the first search.

`solved_examples` are questions of the SAME FORM as this one that have already been marked correct, each
shown with its reference answer. They are your specification for what an answer looks like: how long it is,
whether it keeps a qualifier such as a state, a country or a year, whether it is a proper name written in
full or a lower-case common noun, and whether it is simply yes or no. `style_note` summarises them.
Use them for FORM ONLY -- their subject matter is unrelated to this question and never supplies its answer.

Each passage line reads `Title [is <what that page declares itself to be>] | text`. The bracket is copied
from that page's own opening sentence, so it is the corpus's statement of what kind of thing the page is --
a person, a film, an album, a place, a band, a species, an event -- not a guess. Use it to keep roles
straight: work out what kind of thing the question asks for and take the answer from a page of that kind.
A question asking which artist, director, actor or writer did something is not answered by the work they did
it on; a question asking for a work is not answered by the person who made it; a question asking for a year
or an amount is not answered by a name. The bracket is an annotation, never part of the answer, and a page
whose bracket is missing is ordinary evidence.

First produce `supporting_titles`: the titles of the one or two passages that actually justify the answer.

Then judge the evidence honestly in `evidence_status`, exactly one of:
- `complete` -- the passages state everything the question asks for; the answer is a string you can point to
  in them;
- `partial` -- the passages identify the subjects and settle part of the chain, but the specific fact the
  question ends on (a date, a place, a count, a name, which of two things) is not written there;
- `absent` -- the passages do not even cover the subjects the question is about.
Then `missing_fact`: for `partial`/`absent`, the one thing the passages fail to state, in a short phrase
(write `none` when the status is `complete`). Judge this before you write the answer, not afterwards, and do
not talk yourself into `complete` because a string that looks like the right kind of answer is on screen.

Then `answer_type`, and then `answer`.

Rules for `answer`:
- Pitch it at the same level of detail as the reference answers in `solved_examples`. Do not compress below
  that level and do not pad above it.
- If answers of this form keep a trailing place or date qualifier, keep it; if they are bare, stay bare.
- `solved_examples` and `style_note` never license dropping part of what the question actually asks for: if
  the question asks for a date, give the date and not only its year; if it asks for a full title, give the
  full title. Form follows the examples, content follows the question.
- If `evidence_status` is `complete`, copy the answer verbatim from the passages, keeping their spelling,
  capitalisation, abbreviations and units, e.g. write a distance the way the passage writes it rather than
  expanding or converting it; and if the reference answers of this form are people's names written out in
  full, write the name the way the passage's own opening sentence writes it rather than the short form you
  would say out loud.
- If `evidence_status` is `partial` or `absent`, do NOT return the closest-looking string on screen just
  because it is on screen. Use the passages to fix WHICH entity the question is about, then answer the
  missing fact from what you know about that entity yourself, written the way a reference work would write
  it. Say what you believe is true of that entity, not what the visible text happens to say about another.
- When the answer is two things, join them the way the examples join them.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- A question that can be answered with yes or no must be answered with exactly `yes` or exactly `no`.
- A question that asks which of two named things has some property must be answered with just that one
  name; do not name both and do not say why.
- Always commit to one answer; never reply that the information is missing."""


KNOWLEDGE_INSTRUCTIONS = """`question` needs one fact that the available reference abstracts never stated.
`established` is what those abstracts did settle: treat it as true and as fixing which entities the question
is about. `missing_link` is the fact that was not written down anywhere. `draft` is what a previous attempt
answered while it could see the abstracts, so it is biased towards strings that merely appeared there.

Answer from your own knowledge of the entities named in `established`. Identify them, recall what you know
about them, and give the missing fact. If `draft` is what you would independently say about those entities,
keep it; if it is only a nearby string, replace it.

Return only `answer`: the bare fact, in the form `expected_form` describes, with no sentence around it, no
explanation and no final period. Answer a yes/no question with exactly `yes` or exactly `no`. When the
question offers two named alternatives, return exactly one of them, spelled as in the question. Commit to
your best answer; never reply that you do not know."""


REPAIR_INSTRUCTIONS = """`draft` is a verbose or evasive attempt at answering `question` from `passages`.

Return only `answer`: the bare span that answers the question, normally one to four words, copied verbatim
from `passages` or from `draft`. No sentence, no explanation, no final period. Answer a yes/no question
with exactly `yes` or exactly `no`. When the question offers two named alternatives, return exactly one of
them, spelled as in the question. Never reply that the information is missing -- commit to the best span."""


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.triage = Module(
            inputs=["question", "retrieved_titles", "passages"],
            outputs=["facts", "answer_type", "next_query"],
            instructions=TRIAGE_INSTRUCTIONS,
            descs={
                "retrieved_titles": "pages already covered by the first search",
                "facts": "verbatim sentences from the passages, one per line, each prefixed by its page title",
                "answer_type": "yesno, person, organisation, location, date, number, title or other",
                "next_query": "a title-like noun phrase of at most six words naming the page still missing",
            },
        )
        self.answerer = Module(
            inputs=["question", "solved_examples", "style_note", "facts", "passages"],
            outputs=["supporting_titles", "evidence_status", "missing_fact", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "solved_examples": "already-solved questions of the same form with their reference answers; a "
                                   "specification of answer form only, never a source of content",
                "style_note": "what the reference answers of this question form look like",
                "supporting_titles": "titles of the passages that justify the answer",
                "evidence_status": "complete, partial or absent",
                "missing_fact": "the one fact the passages do not state, or none",
                "answer": "the final answer as a bare span, no sentence, no punctuation around it",
            },
        )
        self.repair = Module(
            inputs=["question", "passages", "draft"],
            outputs=["answer"],
            instructions=REPAIR_INSTRUCTIONS,
            chain_of_thought=False,
        )
        self.knowledge = Module(
            inputs=["question", "established", "missing_link", "expected_form", "draft"],
            outputs=["answer"],
            instructions=KNOWLEDGE_INSTRUCTIONS,
            descs={
                "established": "what the retrieved abstracts did settle, including which entities are meant",
                "missing_link": "the fact the abstracts never stated",
                "expected_form": "what answers to questions of this form look like",
                "answer": "the missing fact as a bare span, no sentence, no punctuation around it",
            },
        )
        self._demos: list[dict[str, Any]] = []

    # -- prepare: index the train split by question form -------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        pool = []
        for it in train_items:
            q = (it.get("question") or "").strip()
            a = (it.get("answer") or "").strip()
            if not q or not a:
                continue
            pool.append({"q": q, "a": a, "f": form_key(q), "t": sorted(_content_tokens(q))})
        self._demos = pool
        forms: dict[str, int] = {}
        for d in pool:
            forms[d["f"]] = forms.get(d["f"], 0) + 1
        ctx.note("indexed_train_forms", n_demos=len(pool), n_forms=len(forms))

    def get_state(self) -> str:
        return json.dumps({"demos": self._demos})

    def set_state(self, state: str) -> None:
        try:
            self._demos = json.loads(state or "{}").get("demos", []) or []
        except ValueError:
            self._demos = []

    # -- demo selection (pure function of the question: thread-safe) --------------------------------
    N_DEMOS = 6
    MIN_SCORE = 2.0

    def _select_demos(self, question: str) -> list[dict[str, Any]]:
        if not self._demos:
            return []
        key = form_key(question)
        sig, _, cmp_marker = key.partition("|")
        wh = sig.split()[0] if sig.split() else ""
        qt = _content_tokens(question)
        scored = []
        for i, d in enumerate(self._demos):
            dsig, _, dcmp = d["f"].partition("|")
            dt = set(d["t"])
            score = 0.0
            if d["f"] == key:
                score += 6.0
            elif dsig == sig:
                score += 4.0
            elif dsig.split()[:1] == [wh] and wh:
                score += 2.0
            if dcmp == cmp_marker:
                score += 1.5
            if qt or dt:
                score += 2.0 * len(qt & dt) / max(1, len(qt | dt))
            scored.append((-score, i, d))
        scored.sort(key=lambda x: (x[0], x[1]))
        picked = [d for neg, _, d in scored[: self.N_DEMOS] if -neg >= self.MIN_SCORE]
        if len(picked) < 2:
            picked = [d for _, _, d in scored[: self.N_DEMOS]]
        # keep the closest example first but show a spread of answer shapes, so the model reads the
        # convention of the form rather than copying the single nearest answer
        seen_shapes, spread = set(), []
        for d in picked:
            sh = _shape(d["a"])
            k = (sh["len"], sh["yesno"], sh["qualifier"], sh["proper"])
            if k in seen_shapes and len(spread) >= 3:
                continue
            seen_shapes.add(k)
            spread.append(d)
        return spread or picked

    # -- second-hop query gate ---------------------------------------------------------------------
    def _clean_query(self, q: str) -> str:
        q = re.sub(r"\[\[.*?\]\]", " ", q or "")
        q = q.replace("**", "").replace("`", "").strip()
        for line in q.splitlines():
            if line.strip():
                q = line.strip()
                break
        q = re.sub(r"(?i)^(search|query|next query|look up|find)\s*:\s*", "", q).strip()
        return q.strip("\"“”‘’ ?.").strip()

    def _uncovered_span(self, question: str, titles: list[str]) -> str:
        """The longest entity named in the question that no retrieved title covers."""
        covered = " ".join(_norm(t) for t in titles)
        best = ""
        for span in _named_spans(question):
            ns = _norm(span)
            if not ns or ns in covered:
                continue
            if len(ns.split()) > 6:
                continue
            if len(ns) > len(best):
                best = span
        return best

    def _gate(self, raw_query: str, question: str, titles: list[str], ctx: RunContext) -> str:
        q = self._clean_query(raw_query)
        qt = _content_tokens(q)
        question_tokens = _content_tokens(question)
        overlap = len(qt & question_tokens) / max(1, len(question_tokens))
        covered = {_norm(t) for t in titles}
        degenerate = (
            not qt
            or NULL_QUERY.match(q.strip())
            or len(q.split()) > 9
            or overlap >= 0.9
            or _norm(q) in covered
        )
        if not degenerate:
            return q
        fallback = self._uncovered_span(question, titles)
        if not fallback and titles:
            # every entity named in the question is already covered, so the missing page must be one the
            # first hop points at: anchor on the best-matching page found and add the question's focus words
            anchor = titles[0]
            focus = [w for w in question.split() if w.lower().strip(",.?\"'") not in STOP and _norm(w) not in _norm(anchor)]
            fallback = (anchor + " " + " ".join(focus[:4])).strip()
        if not fallback:
            words = [w for w in q.split() if w.lower() not in STOP][:6]
            fallback = " ".join(words) or question
        ctx.note("hop2_query_gated", raw=q[:120], used=fallback[:120], overlap=round(overlap, 2))
        return fallback

    # -- main --------------------------------------------------------------------------------------
    def _repair_hint(self, question: str, note: str) -> str:
        opts = _two_options(question)
        bits = []
        if opts:
            bits.append("The answer must be exactly one of these, copied as written: " + " / ".join(opts))
        if note:
            bits.append("Expected form of the answer: " + note)
        return "\n".join(bits)

    def _established(self, supporting: str, facts: str, titles: list[str], pool: list[str]) -> str:
        """What the retrieved abstracts did settle, so the knowledge call knows which entities are meant.

        Passages are withheld: the point of the call is to stop the model reaching for a visible string.
        Instead it gets the copied fact lines and the opening sentence of each page the answering call said
        it was relying on -- enough to pin the entities down, not enough to re-read the noise."""
        bits = []
        want = [_norm(s) for s in re.split(r"[,;\n]", supporting or "") if _norm(s)]
        for text in pool:
            title, _, body = text.partition(" | ")
            nt = _norm(title)
            if want and not any(w in nt or nt in w for w in want):
                continue
            lead = re.split(r"(?<=[a-z0-9\)])\.\s+", body.strip())[0]
            bits.append(f"{title.strip()}: {lead.strip()[:300]}")
            if len(bits) >= 3:
                break
        if facts and facts.lower() != "none":
            bits.append("facts copied from the abstracts:\n" + facts[:900])
        if not bits:
            bits.append("the retrieved pages were " + ", ".join(titles[:5]) + ", none of which settled it")
        return "\n".join(bits)

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles1 = [d["title"] for d in hop1]
        t = self.triage(
            ctx.llm,
            tag="triage",
            question=question,
            retrieved_titles=titles1,
            passages=[d["text"] for d in hop1],
        )
        facts = t.get("facts", "").strip()
        answer_type = t.get("answer_type", "").strip()

        query_2 = self._gate(t.get("next_query", ""), question, titles1, ctx)
        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)

        # deduplicated verbatim evidence pool: newly found pages first, they carry the missing hop
        # Each line is also annotated with what the page's own abstract declares it to be, so the reader
        # knows the kind of every candidate before it reads any of them.
        pool, labelled, seen = [], [], set()
        for d in hop2 + hop1:
            key = d["title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(d["text"])
            title, _, body = d["text"].partition(" | ")
            _, declared = page_kind(d["text"])
            labelled.append(f"{title.strip()} [is {declared}] | {body}" if declared else d["text"])
        ctx.note("typed_pool", n=len(pool), n_declared=sum(1 for x in labelled if "[is " in x))

        demos = self._select_demos(question)
        shown = [f"question: {d['q']}\nreference answer: {d['a']}" for d in demos]
        note = _style_note(demos)
        ctx.note("answer_style", form=form_key(question), n_demos=len(demos), style=note[:200])

        out = self.answerer(
            ctx.llm,
            tag="answer",
            question=question,
            solved_examples=shown or ["none available"],
            style_note=note,
            facts=facts or "none",
            passages=labelled,
        )
        raw = out.get("answer", "")
        atype = out.get("answer_type", "").strip() or answer_type
        answer = polish(raw, atype, question)

        status = _status(out.get("evidence_status", ""))
        missing = _short(out.get("missing_fact", ""))
        # the call is systematically over-confident about completeness, so `complete` is audited rather than
        # believed: an answer whose own words are not in the evidence was not read off it, however the call
        # classified it, and that is precisely the case the knowledge branch exists for
        if status == "complete" and answer and not _derived(answer, atype) and not _grounded(answer, pool):
            status = "partial"
            missing = missing or "the exact wording the question ends on is not in the abstracts"
            ctx.note("status_overridden", answer=answer[:120])
        ctx.note("evidence_status", status=status, missing=missing[:160])

        # "too long" is measured against the reference answers of this question form, not a constant
        budget = max(len(d["a"].split()) for d in demos) + 2 if demos else 6
        verbose = _needs_repair(answer, atype, question, budget)

        knowledge_mode = status in ("partial", "absent") and atype.strip().lower()[:3] != "yes"
        if knowledge_mode:
            established = self._established(out.get("supporting_titles", ""), facts, titles1, pool)
            fixed = self.knowledge(
                ctx.llm,
                tag="knowledge",
                question=question + ("\n" + self._repair_hint(question, "") if _two_options(question) else ""),
                established=established,
                missing_link=missing or "the specific fact the question ends on",
                expected_form=note,
                draft=(raw[:300] or answer) or "none",
            )
            candidate = polish(fixed.get("answer", ""), atype, question)
            ctx.note("knowledge_answer", before=answer[:120], after=candidate[:120], status=status)
            if candidate and not HEDGES.search(candidate) and not _needs_repair(candidate, atype, question, budget):
                answer = candidate
            elif candidate and not HEDGES.search(candidate) and verbose:
                answer = candidate
        elif verbose:
            hint = self._repair_hint(question, note)
            fixed = self.repair(
                ctx.llm,
                tag="repair",
                question=(question + ("\n" + hint if hint else "")),
                passages=pool,
                draft=raw[:600] or answer,
            )
            candidate = polish(fixed.get("answer", ""), atype, question)
            ctx.note("repair", before=answer[:120], after=candidate[:120])
            if candidate and not HEDGES.search(candidate):
                answer = candidate

        return {"answer": answer, "raw_answer": raw, "query_2": query_2, "answer_type": atype,
                "evidence_status": status}
