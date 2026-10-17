"""When the passages fall short, ask the model what it already knows -- and make the corpus confirm it.

Every system on this board is strictly extractive: the answer must be copied out of a retrieved lead,
and the model's own knowledge of the world is treated as contraband.  That is the right default -- it
is what keeps the frontier at 82% on the items whose evidence is complete.  But it is exactly wrong
for the population the frontier's doubt gate isolates: 82 items answered at 50%, and in 12 of its
errors there the missing article was never returned by any query the pipeline could construct, so
there is nothing left to extract.  On those items the pipeline's only remaining source of the fact is
the solver's parametric memory, and the pipeline forbids it from using it.

So this harness gives the gated population a second, non-extractive generator.  A single closed-book
call sees the question and nothing else -- no passages, no draft -- and must commit: the value it
remembers, plus the title of the article whose opening lines would state that value.  A remembered
value is worthless on its own, so it is never adopted on the strength of the memory.  It is adopted
only after the corpus is made to confirm it, by a test with no model call in it at all: the named
witness article is fetched by title, a keyword probe joins the remembered value to a name the
question itself uses, and the value stands only if some fetched lead prints BOTH the remembered value
and a question anchor.  A memory that nothing in the corpus corroborates is discarded and the
incumbent keeps the item.

The point of the design is that the memory is used as a *generator* and the corpus as the *judge*.
Memory reaches articles no reformulation of the question's vocabulary can reach, because the article
that states the answer is usually about the answer; the co-print test is what stops a fluent
hallucination from displacing a grounded read.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
DECLINE = re.compile(
    r"^\s*(?:unknown|none|n/?a|nothing|no\s+answer|not\s+(?:given|stated|provided|mentioned|specified|found|available)"
    r"|cannot|can\s*not|can't|unable|unclear|unspecified|insufficient|there\s+is\s+no|the\s+(?:evidence|passages?|text|context|leads?)"
    r"|based\s+on\s+the)\b",
    re.IGNORECASE,
)
DATE_ASK = re.compile(
    r"\b(?:what|which|in\s+what|in\s+which)\s+year\b|\bwhat\s+date\b|\bwhen\s+(?:was|were|is|are|did|do|does|had|has)\b",
    re.IGNORECASE,
)
COUNT_ASK = re.compile(r"\bhow\s+(?:many|much)\b", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
COUNT_TOKEN = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"twenty|thirty|forty|fifty|hundred|thousand|million|billion|zero|none|no)\b",
    re.IGNORECASE,
)
MAX_ANSWER_WORDS = 8

# Language a solver uses when the passages did not settle the question.  Counted over the answering
# call's own reasoning; two or more markers is the gate.
HEDGE = re.compile(
    r"\b(?:not\s+(?:explicitly\s+)?(?:stated|mentioned|specified|given|provided|named|listed)"
    r"|does\s+not\s+(?:say|state|mention|specify|name)|doesn't\s+(?:say|state|mention|specify|name)"
    r"|do\s+not\s+(?:say|state|mention|specify)|don't\s+(?:say|state|mention|specify)"
    r"|no\s+(?:explicit|direct)\s+(?:mention|information|statement)"
    r"|is\s+not\s+(?:stated|mentioned|specified)|isn't\s+(?:stated|mentioned|specified)"
    r"|not\s+sure|unclear|uncertain|ambiguous|assum(?:e|ing|ption)|presumably|probably|perhaps"
    r"|might\s+be|may\s+be\s+the|could\s+be\s+the|best\s+guess|guess(?:ing)?|infer(?:red|ring)?"
    r"|unverified|without\s+more\s+information|if\s+I\s+had\s+to)\b",
    re.IGNORECASE,
)


def _binary(text: str) -> bool:
    """yes/no drafts are settled by reasoning, not by another article: they never buy a round."""
    return (text or "").strip().strip(".").lower() in ("yes", "no")


HEDGE_SENT = re.compile(r"[^.!?\n]*[.!?]?")


def _doubt_excerpt(reasoning: str, limit: int = 3) -> str:
    """The solver's own sentences that carry a hedge, so the lookup knows what was missing."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", reasoning or ""):
        sent = sent.strip()
        if sent and HEDGE.search(sent):
            out.append(sent[:200])
        if len(out) >= limit:
            break
    return " ".join(out) or "the evidence did not settle it"


def _hedge_count(*texts: str) -> int:
    """How many times the solver flagged, in its own words, that the evidence fell short."""
    return sum(len(HEDGE.findall(text or "")) for text in texts)


def _question_anchors(question: str, cap: int = 4) -> list[str]:
    """The names the question itself uses: multi-token capitalised runs, longest first."""
    out: list[str] = []
    for m in MENTION.finditer(question or ""):
        name = _trim_mention(m.group(0).strip())
        if len(name.split()) >= 2 and all(name.lower() != o.lower() for o in out):
            out.append(name)
    out.sort(key=lambda n: -len(n))
    return out[:cap]


def _question_terms(question: str, cap: int = 3) -> list[str]:
    """Fallback anchors for a question that names nothing: its longest content words."""
    terms = sorted({w for w in _words(question) if w not in PHRASE_STOP and len(w) > 4},
                   key=lambda w: -len(w))
    return terms[:cap]


def _prints(passage: str, value: str) -> bool:
    hay = re.sub(r"\s+", " ", passage or "").lower()
    needle = re.sub(r"\s+", " ", value or "").strip().lower()
    return bool(needle) and needle in hay


def _admissible(question: str, draft: str) -> str:
    """"" if the draft could be the answer; otherwise the name of the test it fails."""
    text = (draft or "").strip()
    if not text:
        return "empty"
    if len(text.split()) > MAX_ANSWER_WORDS:
        return "verbose"
    if DECLINE.match(text):
        return "declined"
    if DATE_ASK.search(question) and not DIGIT_RE.search(text):
        return "no_date"
    if COUNT_ASK.search(question) and not COUNT_TOKEN.search(text):
        return "no_count"
    return ""

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are", "did", "does", "do",
    "has", "have", "had", "that", "this", "with", "by", "from", "as", "it", "its", "his", "her",
    "their", "he", "she", "they", "both", "also", "than", "then", "there", "these", "those", "name",
    "named", "known", "called", "same", "other", "another", "any", "between", "about", "into",
}


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip()
        if len(line) >= 3 and not line.startswith("[["):
            out.append(line[:220])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


def _words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in STOP]


def _sentences(passage: str) -> list[str]:
    _, _, body = passage.partition(" | ")
    out = []
    for sent in SENT_SPLIT.split(body.strip()):
        sent = sent.strip()
        if 20 <= len(sent) <= 320:
            out.append(sent)
        elif len(sent) > 320:
            out.append(sent[:320].rsplit(" ", 1)[0])
    return out[:6]


def _weights(question: str, linking_fact: str, corpus: list[str]) -> dict[str, float]:
    """Rarity weight, inside the retrieved pool, of every content word the question cares about."""
    df: dict[str, int] = {}
    for passage in corpus:
        for w in set(_words(passage)):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(corpus))
    weights: dict[str, float] = {}
    for w in _words(question):
        weights[w] = math.log(1.0 + n / (1.0 + df.get(w, 0)))
    for w in _words(linking_fact):
        weights[w] = max(weights.get(w, 0.0), 0.75 * math.log(1.0 + n / (1.0 + df.get(w, 0))))
    return weights


# -- answer-form machinery ---------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9\u00c0-\u017f'\u2019\-]+")
PARTICLE = {"of", "the", "de", "del", "der", "den", "di", "da", "van", "von", "la", "le", "el"}
DETERMINER = {"the", "a", "an"}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}
NUMBER_WORD = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty",
}
QUAL_RE = re.compile(r"^\s*,\s+([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,2})")
HEAD_NOUN_RE = re.compile(r"^\s+([a-z]{3,})")
PLACE_WORDS = {"where", "city", "town", "country", "state", "county", "located", "based", "born",
               "province", "village", "headquartered"}
COUNT_WORDS = {"many", "number"}


def _ntoks(text: str) -> list[str]:
    return [normalize_text(t) for t in TOKEN_RE.findall(text or "") if normalize_text(t)]


def _name_like(text: str) -> bool:
    raw = TOKEN_RE.findall(text or "")
    if not 2 <= len(raw) <= 5 or any(ch.isdigit() for ch in text):
        return False
    return all(t[:1].isupper() or t.lower() in PARTICLE for t in raw)


def _opening_run(body: str) -> str:
    """The name an article's own lead opens with, e.g. `Thomas Lance Rentzel`."""
    first = SENT_SPLIT.split((body or "").strip())[0]
    head = re.split(r"[(\"\u201c]", first)[0]
    run: list[str] = []
    for tok in TOKEN_RE.findall(head):
        if tok[:1].isupper() or (run and tok.lower() in PARTICLE):
            run.append(tok)
        else:
            break
    return " ".join(run).strip(" ,")


def _is_expansion(short: str, long_: str) -> bool:
    """`long_` names the same thing as `short` with up to three extra name tokens threaded in."""
    a, b = _ntoks(short), _ntoks(long_)
    if not a or len(b) <= len(a) or len(b) - len(a) > 3:
        return False
    extra = [t for t in b if t not in a]
    if not extra or any(t in MONTHS or t in DETERMINER for t in extra):
        return False
    for tok in TOKEN_RE.findall(long_):
        if tok.isupper() and len(tok) > 1 and normalize_text(tok) in extra:
            return False          # an acronym gloss, not a fuller name
    i = 0
    for t in b:
        if i < len(a) and t == a[i]:
            i += 1
    return i == len(a)


def _find(text: str, phrase: str) -> int:
    """Case-insensitive whole-token occurrence of `phrase` in `text`; -1 if absent."""
    if not phrase:
        return -1
    pat = r"\b" + r"\W+".join(re.escape(t) for t in TOKEN_RE.findall(phrase)) + r"\b"
    m = re.search(pat, text, re.IGNORECASE)
    return m.start() if m else -1


def _span_end(text: str, phrase: str, start: int) -> int:
    pat = r"\b" + r"\W+".join(re.escape(t) for t in TOKEN_RE.findall(phrase)) + r"\b"
    m = re.search(pat, text[start:], re.IGNORECASE)
    return start + m.end() if m else -1


def _is_count(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\d[\d,]*", t)) or t.lower() in NUMBER_WORD


def _singular(word: str) -> str:
    w = word
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


# -- shared-property family: detection and cross-article phrase intersection --------------------
SHARED_CUE = re.compile(
    r"(have in common"
    r"|what (?:are|is|were|was) both\b"
    r"|\band\b[^?]{0,60}\b(?:known|knowns) for\b"
    r"|\bboth\b[^?]{0,70}\b(?:which|what)\b"
    r"|\b(?:which|what) (?:type|kind|sort|genre|category|nationality|profession|occupation)\b"
    r"|\bare (?:which|what)\b"
    r"|\b(?:in|part of|from|of) which\b"
    r"|how (?:is|are|was|were) .+\band\b.+ related"
    r")", re.IGNORECASE)
POLAR = re.compile(
    r"^\s*(?:are|is|was|were|do|does|did|can|could|has|have|had|will|would)\b", re.IGNORECASE)
ALTERNATIVES = re.compile(r",\s+[A-Z][^?,]*\s+or\s+[A-Z]")
MENTION = re.compile(
    r"(?:[A-Z][\w.&'\u2019\-]*|\d{2,4})(?:\s+(?:[A-Z][\w.&'\u2019\-]*|\d[\w.\-]*|of|the|de|del|la|le))*")
LEAD_WORD = {
    "what", "which", "who", "whom", "whose", "how", "where", "when", "why", "are", "is", "was",
    "were", "do", "does", "did", "in", "along", "while", "the", "a", "an", "both", "and", "of",
    "type", "kind", "sort", "besides",
}
TRAIL_WORD = {"of", "the", "a", "an", "de", "and", "in", "on", "at"}
PHRASE_STOP = STOP | {
    "is", "are", "was", "were", "be", "been", "being", "one", "two", "all", "such", "who", "which",
    "what", "when", "where", "not", "but", "may", "can", "will", "would", "also", "most", "more",
    "some", "many", "first", "second", "new", "well", "after", "before", "during", "including",
    "based", "used", "part", "since", "over", "under", "out", "up", "down", "you", "him", "them",
}
PHRASE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'\u2019\-]*")


def _trim_mention(text: str) -> str:
    toks = text.split()
    while toks and toks[0].lower() in LEAD_WORD:
        toks = toks[1:]
    while toks and toks[-1].lower() in TRAIL_WORD:
        toks = toks[:-1]
    return " ".join(toks).strip(" ,.")


def _shared_pair(question: str) -> tuple[str, str] | None:
    """The two things a shared-property question compares, or None if it is not one."""
    q = (question or "").strip()
    if POLAR.match(q) or not SHARED_CUE.search(q) or ALTERNATIVES.search(q):
        return None
    for join in re.finditer(r"\s+and\s+", q):
        left = [_trim_mention(m.group()) for m in MENTION.finditer(q[: join.start()])]
        right = [_trim_mention(m.group()) for m in MENTION.finditer(q[join.end():])]
        left = [m for m in left if len(m) >= 2]
        right = [m for m in right if len(m) >= 2]
        if left and right and normalize_text(left[-1]) != normalize_text(right[0]):
            return left[-1], right[0]
    return None


def _title_match(mention: str, passage: str) -> float:
    a = set(_ntoks(mention))
    b = set(_ntoks(_title_of(passage)))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ngrams(body: str, n_max: int = 4) -> dict[tuple[str, ...], str]:
    """Every 1..n_max word phrase of a lead, keyed by its normalised tokens, valued by its surface."""
    spans = [(m.group(), normalize_text(m.group())) for m in PHRASE_TOKEN.finditer(body or "")]
    spans = [(s, n) for s, n in spans if n]
    out: dict[tuple[str, ...], str] = {}
    for i in range(len(spans)):
        for n in range(1, n_max + 1):
            if i + n > len(spans):
                break
            key = tuple(norm for _, norm in spans[i: i + n])
            if key[0] in PHRASE_STOP or key[-1] in PHRASE_STOP:
                continue
            if all(k in PHRASE_STOP for k in key):
                continue
            surface = " ".join(surf for surf, _ in spans[i: i + n]).strip(" .,;:")
            if surface:
                out.setdefault(key, surface)
    return out


def _shared_phrases(lead_a: str, lead_b: str, question: str, background: list[str],
                    limit: int) -> list[str]:
    """Phrases printed in BOTH leads, most distinctive first. Pure string work, no model."""
    body_a = lead_a.partition(" | ")[2]
    body_b = lead_b.partition(" | ")[2]
    grams_a, grams_b = _ngrams(body_a), _ngrams(body_b)
    common = set(grams_a) & set(grams_b)
    if not common:
        return []
    corpus = background + [lead_a, lead_b]
    df: dict[str, int] = {}
    for passage in corpus:
        for w in set(_words(passage)):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(corpus))
    qtokens = set(_ntoks(question))
    scored: list[tuple[float, str]] = []
    for key in common:
        content = [k for k in key if k not in PHRASE_STOP]
        if not content:
            continue
        weight = sum(math.log(1.0 + n / (1.0 + df.get(k, 0))) for k in content)
        weight *= 1.0 + 0.12 * (len(content) - 1)          # a longer shared phrase says more
        echoed = all(k in qtokens for k in content)         # the question already said this
        scored.append((1 if echoed else 0, weight, grams_a[key]))
    scored.sort(key=lambda s: (s[0], -s[1], len(s[2])))      # a name of one of the two goes last
    out: list[str] = []
    seen: set[str] = set()
    for _, _, surface in scored:
        key = normalize_text(surface)
        if key in seen:
            continue
        seen.add(key)
        out.append(surface)
        if len(out) >= limit:
            break
    return out



# -- descriptor typing: the kind a question ascribes to each name it introduces -------------------
QUOTED = re.compile(r"[\"\u201c]([^\"\u201d]{2,60})[\"\u201d]")
CONJ = {"and", "or", "the", "a", "an", "of", "both", "by", "in", "on", "at", "to", "from", "with",
        "towards", "toward", "about", "into", "onto", "over", "under", "after", "before",
        "that", "which", "who", "is", "was", "were", "are", "as", "for", "his", "her", "their",
        "its", "this", "these", "those", "did", "does", "do", "has", "have", "had", "also"}
NOT_A_TYPE = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "name",
              "named", "called", "known", "same", "other", "another", "between", "among", "many",
              "much", "more", "most", "first", "second", "last", "next", "one", "two", "both",
              "born", "died", "released", "published", "starred", "played", "wrote", "directed",
              "founded", "located", "based", "featured", "won", "made", "used", "given", "year",
              "years", "date", "dates", "number", "numbers", "time", "times"}
MIN_DESC_LEN = 4


def _lc_words(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group().lower()) for m in TOKEN_RE.finditer(text or "")]


def _typed_mentions(question: str, limit: int = 4) -> list[tuple[str, list[str]]]:
    """Each name the question introduces, paired with the type words written just before it.

    Only an unbroken run of lower-case common nouns touching the name counts, so `the song "X"`
    and `directors A and B` are typed while `... reported on X` and `... starred in X` are not.
    """
    spans: list[tuple[int, int, str]] = []
    for m in QUOTED.finditer(question):
        spans.append((m.start(), m.end(), m.group(1).strip()))
    for m in MENTION.finditer(question):
        surface = _trim_mention(m.group())
        if len(surface) < 3 or any(a < m.end() and m.start() < b for a, b, _ in spans):
            continue
        spans.append((m.start(), m.end(), surface))
    spans.sort()
    words = [(m.start(), m.group()) for m in TOKEN_RE.finditer(question)]
    out: list[tuple[str, list[str]]] = []
    carried: list[str] = []
    for start, _end, surface in spans:
        if not any(c.isalpha() for c in surface):
            continue
        before = [w for pos, w in words if pos < start]
        own = {normalize_text(t) for t in TOKEN_RE.findall(surface)}
        descs: list[str] = []
        for raw in reversed(before[-3:]):
            w = raw.lower()
            if (raw[:1].isupper() or any(c.isdigit() for c in raw) or w in STOP or w in CONJ
                    or w in NOT_A_TYPE or w in own or len(w) < MIN_DESC_LEN
                    or w.endswith("ed") or w.endswith("ing")):
                break
            if w not in descs:
                descs.append(w)
            if len(descs) >= 2:
                break
        # `directors A and B`: a name joined on to the previous one inherits its kind
        if not descs and before and before[-1].lower() in ("and", "or") and carried:
            descs = [d for d in carried if d not in own]
        elif descs:
            carried = list(descs)
        if descs:
            out.append((surface, descs))
        if len(out) >= limit:
            break
    return out


def _states_type(passage: str, descs: list[str]) -> bool:
    """True when the lead says one of the type words, or a morphological relative of it."""
    words = set(_words(passage))
    for d in descs:
        if d in words or _singular(d) in words:
            return True
        stem = _singular(d)[:5]
        if len(stem) >= 5 and any(w.startswith(stem) for w in words):
            return True
    return False


# -- link closure: which pool passages are the partners of the ones triage picked ----------------
PAREN = re.compile(r"\s*\([^)]*\)\s*$")
GENERIC_TITLE = re.compile(r"^(?:list|index|timeline|outline|glossary|history)\b", re.IGNORECASE)


def _bare_title(passage: str) -> str:
    """The title with its disambiguating parenthesis dropped -- the form a lead actually prints."""
    return PAREN.sub("", _title_of(passage)).strip()


def _phrase_tokens(text: str) -> list[str]:
    return [t for t in normalize_text(text or "").split() if t]


def _contains_run(hay: list[str], needle: list[str]) -> bool:
    """Is `needle` a contiguous run of `hay`?  Titles are printed verbatim when they are linked."""
    n = len(needle)
    if not n or n > len(hay):
        return False
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i: i + n] == needle:
            return True
    return False


def _pool_df(pool: list[str]) -> tuple[dict[str, int], int]:
    df: dict[str, int] = {}
    for passage in pool:
        for w in set(_phrase_tokens(passage)):
            df[w] = df.get(w, 0) + 1
    return df, max(1, len(pool))


def _title_rarity(tokens: list[str], df: dict[str, int], n: int) -> float:
    """How much a title being printed somewhere else tells us: rare tokens say a lot, `2012` nothing."""
    content = [t for t in tokens if t not in PHRASE_STOP]
    if not content:
        return 0.0
    return sum(math.log(1.0 + n / (1.0 + df.get(t, 0))) for t in content)


def _link_closure(pool: list[str], selected: list[int], question: str, linking_fact: str,
                  cap: int, min_rarity: float) -> list[tuple[int, float, str]]:
    """Pool indices linked to a selected passage, strongest link first.  No model call.

    An edge exists in either direction: the candidate's title is printed inside a selected lead
    (following the link forward), or a selected title is printed inside the candidate's lead
    (following it backward).  Both are the same structural fact -- the two articles talk about each
    other -- and the rarity of the printed title is how confident the edge is.
    """
    df, n = _pool_df(pool)
    chosen = set(selected)
    sel_bodies = {i: _phrase_tokens(pool[i].partition(" | ")[2]) for i in chosen if i < len(pool)}
    sel_titles = {i: _phrase_tokens(_bare_title(pool[i])) for i in chosen if i < len(pool)}
    qwords = set(_words(question)) | set(_words(linking_fact))
    out: list[tuple[int, float, str]] = []
    for i, passage in enumerate(pool):
        if i in chosen or GENERIC_TITLE.match(_title_of(passage)):
            continue
        cand_title = _phrase_tokens(_bare_title(passage))
        cand_body = _phrase_tokens(passage.partition(" | ")[2])
        best, why = 0.0, ""
        for j, body in sel_bodies.items():
            if len(cand_title) >= 1 and _contains_run(body, cand_title):
                weight = _title_rarity(cand_title, df, n)
                if weight > best:
                    best, why = weight, f"named by {_title_of(pool[j])}"
            stitle = sel_titles.get(j) or []
            if len(stitle) >= 2 and _contains_run(cand_body, stitle):
                weight = 0.9 * _title_rarity(stitle, df, n)
                if weight > best:
                    best, why = weight, f"names {_title_of(pool[j])}"
        if best < min_rarity:
            continue
        # a partner that also echoes the question is preferred, but echoing is not required: the
        # partner of an anchor is usually the article the question describes without naming
        overlap = len(qwords & set(_words(passage)))
        out.append((i, best + 0.25 * min(overlap, 6), why))
    out.sort(key=lambda t: -t[1])
    return out[:cap]


class Harness(BaseHarness):
    K_PLAN = 8          # hits per query the parent pipeline reads (unchanged)
    K_DEEP = 20         # depth every query is now issued at; ranks K_PLAN.. feed the tail only
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    DIGEST_LINES = 16   # sentences distilled out of the deep tail
    PER_TITLE = 2       # at most this many sentences from one article
    MIN_HITS = 2        # a sentence must match at least this many distinct question words

    PLAN_INSTR = (
        "Break the question into the separate facts that must be looked up in an encyclopaedia.\n"
        "Write `queries` as one short keyword search query per line (2 or 3 lines, no numbering). "
        "Each query must stand on its own: name the entity it is about explicitly instead of using "
        "a pronoun or a description that depends on another query.\n"
        "If one fact can only be looked up after another is known, still write the query for the "
        "part that is already named."
    )
    TRIAGE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia "
        "leads, shown as `id: title | beginning of the article`.\n"
        "`relevant_ids` : the ids worth reading in full, most useful first, at most 8, comma separated.\n"
        "`linking_fact` : the fact the pool establishes that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); say `unknown` if none.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`.\n"
        "`next_query`   : if not sufficient, one keyword search query for the missing fact, naming "
        "the entity from `linking_fact` explicitly; otherwise leave it empty."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "`extra_leads` holds single sentences pulled from further down the search results, one per "
        "line as `title -- sentence`. They are unverified: use one only when the evidence passages "
        "do not settle the question, and never prefer one over a passage that does.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    # -- the shared-property route -------------------------------------------------------------
    K_ENTITY = 6         # hits per entity lookup
    K_BACKGROUND = 6     # hits of the question itself, kept only as context and as a df background
    MIN_TITLE_MATCH = 0.5
    MAX_SHARED = 14
    MIN_SHARED = 1
    SHARED_INSTR = (
        "The question names two things and asks for something they SHARE.\n"
        "`article_a` and `article_b` are the encyclopaedia leads of those two things; "
        "`other_leads` is unrelated background, use it only if the two leads say nothing.\n"
        "`shared_phrases` lists, most distinctive first, every phrase that is printed literally in "
        "BOTH leads. The shared property is almost always one of them.\n"
        "Choose the entry that answers the question -- the category, type, nationality, occupation, "
        "field, place or relation the two things have in common -- and write it as `answer`, copied "
        "character for character from `shared_phrases`, keeping its exact wording and its singular "
        "or plural form.\n"
        "Never answer with the name of either of the two things, and never write a sentence.\n"
        "Only if no entry fits, write the shortest phrase from the two leads that answers it."
    )


    # -- recovery instructions ----------------------------------------------------------------
    MISSING_INSTR = (
        "An earlier attempt to answer the question from `evidence` did not produce a usable answer; "
        "`failed_answer` is what it wrote.\n"
        "Read the question and decide what single thing it needed to know and could not find there "
        "-- usually one entity (a person, work, place, organisation or team) that the question "
        "refers to only by description, or whose article holds the attribute being asked for.\n"
        "Write that thing's most likely name, as it would title an encyclopedia article, as "
        "`missing_entity`. If the question names it outright, repeat that name.\n"
        "Write as `lookup_query` a short search query of plain keywords -- the name plus the words "
        "naming the attribute being asked for. No question words, no punctuation."
    )
    RECALL_INSTR = (
        "Answer the question from your own knowledge of the world. You are shown no passages on "
        "purpose, and saying that you lack information is always wrong here: commit to the single "
        "most likely value.\n"
        "`answer`  : the shortest phrase that answers the question -- a bare name, title, date, "
        "number or noun phrase. Never a sentence, never an explanation, never `unknown`.\n"
        "`witness` : the title of the one encyclopaedia article whose opening lines would state "
        "that fact, written exactly as that article would be titled. Usually it is the article "
        "about the answer itself or about the thing the question asks about."
    )
    REPAIR_INSTR = (
        "Answer the question from `evidence` and `new_leads` together.\n"
        "Commit to an answer. A previous attempt declined, and declining again is always wrong: if "
        "the leads do not state the fact outright, choose the single candidate they support best.\n"
        "Write only the answer: the shortest phrase that answers the question, copied from the "
        "leads where they contain it -- a name, a date, a number, a nationality or a short noun "
        "phrase. Never a sentence, never an explanation, never `unknown`.\n"
        "If the question asks in what year or on what date, answer with the date exactly as the "
        "leads print it. If it asks how many or how much, answer with the number.\n"
        "If the question asks whether something is the case, or offers two options and asks which, "
        "answer `yes` or `no` or with one option copied verbatim."
    )

    # -- anchor typing -------------------------------------------------------------------------
    K_TYPE = 8            # hits for one descriptor-qualified lookup
    MAX_RETYPE = 2        # at most this many anchors are re-resolved on one item
    MIN_ANCHOR_MATCH = 0.5

    K_REPAIR = 8         # hits per recovery retrieval

    # -- the parametric witness check ----------------------------------------------------------
    K_WITNESS = 6        # hits for the probe that joins the remembered value to the question
    MAX_WITNESS = 4      # leads the co-print test may look at
    MIN_TERMS = 2        # content words a lead must print when the question names nothing

    # -- uncertainty gate ----------------------------------------------------------------------
    MIN_HEDGES = 2       # markers in the answering reasoning before the item buys a second round
    KEEP_ORIGINAL = 3    # original evidence passages carried into the second round
    MAX_FRESH = 4        # newly fetched leads it may read, so the context never grows
    MAX_REPAIR_EVIDENCE = 6

    # -- link closure of the selected evidence -------------------------------------------------
    CLOSE_CAP = 3         # at most this many partners are admitted on one item
    MIN_LINK = 3.0        # a printed title must be at least this rare inside the pool to count
    EVIDENCE_CAP = 8      # selected passages plus partners never exceed this
    DIGEST_PER_ADMIT = 2  # tail lines retired per admitted partner, so the context stays flat

    # -- policy fitted on labelled training answers ---------------------------------------------
    MIN_OBS = 4          # a family needs this many training observations before it may fire
    MIN_SHARE = 0.6      # and this share of them must back the rewrite

    def __init__(self) -> None:
        self.policy: dict[str, bool] = {}
        self.votes: dict[str, int] = {}
        Harness._init_modules(self)

    def _vote(self, answer: str, articles: list[tuple[str, str]]) -> None:
        """Count, on ONE labelled item, which form the label took where both were available."""
        body_all = " ".join(b for _, b in articles)
        # (1) short title vs long lead: which one does a label use?
        if _name_like(answer):
            for title, body in articles:
                run = _opening_run(body)
                if normalize_text(title) == normalize_text(answer) and _is_expansion(answer, run):
                    self.votes["name_keep"] = self.votes.get("name_keep", 0) + 1
                elif normalize_text(run) == normalize_text(answer) and _is_expansion(title, answer):
                    self.votes["name_expand"] = self.votes.get("name_expand", 0) + 1
        # (2) trailing `, Qualifier`
        head = re.match(r"^(.+?), ([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,2})$", answer or "")
        pos = _find(body_all, answer)
        if head and pos >= 0:
            self.votes["qual_attach"] = self.votes.get("qual_attach", 0) + 1
        elif not head and pos >= 0 and answer[:1].isupper() and len(answer.split()) <= 4:
            end = _span_end(body_all, answer, pos)
            if end > 0 and QUAL_RE.match(body_all[end:]):
                self.votes["qual_keep"] = self.votes.get("qual_keep", 0) + 1
        # (3) bare count vs count plus its head noun
        both = re.match(r"^(\d[\d,]*|[A-Za-z]+)\s+([a-z]{3,})$", (answer or "").strip())
        if both and _is_count(both.group(1)) and _find(body_all, answer) >= 0:
            self.votes["noun_attach"] = self.votes.get("noun_attach", 0) + 1
        elif _is_count(answer):
            pos = _find(body_all, answer)
            end = _span_end(body_all, answer, pos) if pos >= 0 else -1
            if end > 0 and HEAD_NOUN_RE.match(body_all[end:]):
                self.votes["noun_keep"] = self.votes.get("noun_keep", 0) + 1
        # (4) a common noun that the source offers in both numbers
        word = (answer or "").strip()
        if word.islower() and len(word.split()) == 1 and len(word) > 3:
            if word.endswith("s") and _singular(word) != word and _find(body_all, _singular(word)) >= 0:
                self.votes["plural_keep"] = self.votes.get("plural_keep", 0) + 1
            elif not word.endswith("s") and _find(body_all, word + "s") >= 0:
                self.votes["singularise"] = self.votes.get("singularise", 0) + 1

    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        for item in train_items:
            answer = (item.get("answer") or "").strip()
            if not answer or normalize_text(answer) in ("yes", "no"):
                continue
            articles: list[tuple[str, str]] = []
            try:
                with ctx.open_tools(item) as tools:
                    for title in (item.get("gold_titles") or [])[:2]:
                        for passage in tools.retrieve(title, k=3):
                            got, _, body = passage.partition(" | ")
                            if normalize_text(got) == normalize_text(title):
                                articles.append((got.strip(), body.strip()))
                                break
            except Exception:            # a training item that cannot be fetched simply abstains
                continue
            if articles:
                self._vote(answer, articles)
        for rewrite, keep in (("name_expand", "name_keep"), ("qual_attach", "qual_keep"),
                              ("noun_attach", "noun_keep"), ("singularise", "plural_keep")):
            won = self.votes.get(rewrite, 0)
            lost = self.votes.get(keep, 0)
            total = won + lost
            self.policy[rewrite] = total >= self.MIN_OBS and won >= self.MIN_SHARE * total
        ctx.note("answer-form policy fitted", votes=self.votes, policy=self.policy)

    def get_state(self) -> str:
        return json.dumps({"policy": self.policy, "votes": self.votes})

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except ValueError:
            data = {}
        self.policy = {k: bool(v) for k, v in (data.get("policy") or {}).items()}
        self.votes = data.get("votes") or {}

    # -- run-time projection onto the source sentence -------------------------------------------
    def _host_sentence(self, question: str, draft: str, evidence: list[str],
                       digest: list[str]) -> str:
        """The sentence the draft most plausibly came from: it contains it and matches the question."""
        corpus = list(evidence) + [f"{line} " for line in digest]
        weights = _weights(question, "", evidence or corpus)
        best, best_score = "", -1.0
        for passage in corpus:
            for sent in _sentences(passage):
                if _find(sent, draft) < 0:
                    continue
                score = sum(weights.get(w, 0.0) for w in set(_words(sent)) & set(weights))
                if score > best_score:
                    best, best_score = sent, score
        return best

    def _project(self, question: str, draft: str, evidence: list[str],
                 digest: list[str]) -> tuple[str, str]:
        """Rewrite the draft into the form the training labels back. Deterministic, no LLM call."""
        if not draft or normalize_text(draft) in ("yes", "no"):
            return draft, ""
        qwords = set(_words(question)) | {w for w in re.findall(r"[a-z]+", question.lower())}
        # (1) the answer's own article opens with a fuller name
        if _name_like(draft):
            for passage in evidence:
                title, _, body = passage.partition(" | ")
                if normalize_text(title.strip()) != normalize_text(draft):
                    continue
                run = _opening_run(body)
                if run and _is_expansion(draft, run):
                    if self.policy.get("name_expand"):
                        return run, "name_expand"
                    # votes inconclusive: expand only where this article's own lead is the draft's
                    # only home in the evidence. If another article refers to the thing by the short
                    # name, the short name is how the sources themselves write it, so keep it.
                    elsewhere = any(
                        _find(other.partition(" | ")[2], draft) >= 0
                        for other in evidence
                        if normalize_text(_title_of(other)) != normalize_text(draft)
                    )
                    if not elsewhere:
                        return run, "name_expand_unique"
                break
        host = self._host_sentence(question, draft, evidence, digest)
        if not host:
            return draft, ""
        pos = _find(host, draft)
        end = _span_end(host, draft, pos) if pos >= 0 else -1
        if end > 0:
            tail = host[end:]
            # (2) the source prints a qualifier right behind it and the question asks for a place
            if self.policy.get("qual_attach") and qwords & PLACE_WORDS:
                m = QUAL_RE.match(tail)
                if m and not re.match(r"^(?:.+?), (?:[A-Z])", draft):
                    return f"{draft}, {m.group(1)}", "qual_attach"
            # (3) the source prints a head noun right behind a bare count
            if self.policy.get("noun_attach") and _is_count(draft) and qwords & COUNT_WORDS:
                m = HEAD_NOUN_RE.match(tail)
                if m:
                    return f"{draft} {m.group(1)}", "noun_attach"
        # (4) the source names the kind of thing in the singular
        if self.policy.get("singularise") and len(draft.split()) == 1 and draft.islower():
            single = _singular(draft)
            if single != draft and _find(host, single) >= 0:
                return single, "singularise"
        return draft, ""

    def _init_modules(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact", "extra_leads"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )
        self.shared = Module(
            inputs=["question", "article_a", "article_b", "shared_phrases", "other_leads"],
            outputs=["answer"],
            instructions=self.SHARED_INSTR,
        )
        self.locate = Module(
            inputs=["question", "evidence", "failed_answer"],
            outputs=["missing_entity", "lookup_query"],
            instructions=self.MISSING_INSTR,
        )
        self.recall = Module(
            inputs=["question"],
            outputs=["answer", "witness"],
            instructions=self.RECALL_INSTR,
        )
        self.repair = Module(
            inputs=["question", "evidence", "new_leads"],
            outputs=["answer"],
            instructions=self.REPAIR_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = _title_of(p).lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    def _deep(self, ctx: RunContext, query: str, pool: list[str], seen: set[str],
              tail: list[str], tail_seen: set[str], k_read: int) -> None:
        """One query, issued at full depth: the head feeds the parent's pool, the rest the tail."""
        hits = ctx.tools.retrieve(query, k=self.K_DEEP)
        self._add(pool, seen, hits[:k_read])
        for p in hits[k_read:]:
            key = _title_of(p).lower()
            if key and key not in tail_seen:
                tail_seen.add(key)
                tail.append(p)

    def _digest(self, question: str, linking_fact: str, tail: list[str],
                read: set[str]) -> list[str]:
        """Rank every tail sentence by pool-rarity overlap with the question; keep the best few."""
        corpus = [p for p in tail if _title_of(p).lower() not in read]
        if not corpus:
            return []
        weights = _weights(question, linking_fact, corpus)
        if not weights:
            return []
        qwords = set(weights)
        scored: list[tuple[float, str, str]] = []
        for passage in corpus:
            title = _title_of(passage)
            title_hit = 0.5 * sum(weights.get(w, 0.0) for w in set(_words(title)) & qwords)
            for sent in _sentences(passage):
                hit = set(_words(sent)) & qwords
                if len(hit) < self.MIN_HITS:
                    continue
                score = sum(weights[w] for w in hit) + title_hit
                score /= 1.0 + 0.004 * len(sent)
                scored.append((score, title, sent))
        scored.sort(key=lambda s: -s[0])
        per: dict[str, int] = {}
        out: list[str] = []
        for score, title, sent in scored:
            if per.get(title, 0) >= self.PER_TITLE:
                continue
            per[title] = per.get(title, 0) + 1
            out.append(f"{title} -- {sent}")
            if len(out) >= self.DIGEST_LINES:
                break
        return out

    # -- the intersective path -----------------------------------------------------------------
    def _lookup(self, ctx: RunContext, mention: str) -> str:
        """The article ABOUT `mention`, found by looking the mention up on its own."""
        best, best_score = "", 0.0
        for passage in ctx.tools.retrieve(mention, k=self.K_ENTITY):
            score = _title_match(mention, passage)
            if score > best_score:
                best, best_score = passage, score
        return best if best_score >= self.MIN_TITLE_MATCH else ""

    def _shared_route(self, question: str, pair: tuple[str, str],
                      ctx: RunContext) -> dict[str, Any] | None:
        """Answer from the literal intersection of the two articles, or None to fall through."""
        background = ctx.tools.retrieve(question, k=self.K_BACKGROUND)
        lead_a = self._lookup(ctx, pair[0])
        lead_b = self._lookup(ctx, pair[1])
        if not lead_a or not lead_b:
            return None
        if normalize_text(_title_of(lead_a)) == normalize_text(_title_of(lead_b)):
            return None
        shared = _shared_phrases(lead_a, lead_b, question, background, self.MAX_SHARED)
        if len(shared) < self.MIN_SHARED:
            return None
        ctx.note("shared-property route", pair=list(pair),
                 titles=[_title_of(lead_a), _title_of(lead_b)], shared=shared[:8])
        ans = self.shared(
            ctx.llm,
            tag="shared",
            question=question,
            article_a=lead_a,
            article_b=lead_b,
            shared_phrases="\n".join(shared),
            other_leads=[_lead(p, 200) for p in background[:4]] or ["(none)"],
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip()
        ans = ans.strip("*_`\"\u00ab\u00bb\u201c\u201d").rstrip(".,;: ").strip()
        ans = re.sub(r"^\[\d+\]\s*", "", ans).strip("\u00ab\u00bb").strip()
        if not ans:
            return None
        return {
            "answer": ans,
            "route": "shared_property",
            "pair": list(pair),
            "titles": [_title_of(lead_a), _title_of(lead_b)],
            "n_shared": len(shared),
        }

    # -- the recovery round ---------------------------------------------------------------------
    def _recover(self, question: str, draft: str, evidence: list[str], digest: list[str],
                 pool: list[str], ctx: RunContext) -> tuple[str, dict[str, Any]]:
        """One targeted round for a draft that failed the admissibility gate."""
        info: dict[str, Any] = {"missing_entity": "", "lookup_query": "", "n_new": 0}
        try:
            out = self.locate(
                ctx.llm,
                tag="locate",
                question=question,
                evidence=[_lead(p, 240) for p in evidence[: self.MAX_REPAIR_EVIDENCE]] or ["(none)"],
                failed_answer=draft or "(no answer)",
            )
        except Exception:
            return "", info
        entity = _trim_mention((out.get("missing_entity", "") or "").strip().strip('"'))[:120]
        query = ((out.get("lookup_query", "") or "").strip().strip('"').replace("\n", " "))[:200]
        info["missing_entity"], info["lookup_query"] = entity, query

        # everything already read, so the recovery only ever adds what the item has not seen
        read = {_title_of(p).lower() for p in evidence}
        fresh: list[str] = []
        fresh_seen: set[str] = set(read)
        if entity:
            article = self._lookup(ctx, entity)
            if article:
                self._add(fresh, fresh_seen, [article])
        if query and query.lower() != entity.lower():
            self._add(fresh, fresh_seen, ctx.tools.retrieve(query, k=self.K_REPAIR))
        # anything the first pass ranked but never selected is free to reuse
        for passage in pool:
            if len(fresh) >= self.SELECT_CAP:
                break
            self._add(fresh, fresh_seen, [passage])
        info["n_new"] = len(fresh)
        if not fresh:
            return "", info

        fixed = self.repair(
            ctx.llm,
            tag="repair",
            question=question,
            evidence=[_lead(p, 300) for p in evidence[: self.MAX_REPAIR_EVIDENCE]] or ["(none)"],
            new_leads=[_lead(p, 300) for p in fresh[: self.SELECT_CAP]],
        ).get("answer", "")
        fixed = (fixed or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return fixed, info

    # -- the round a hedging solve buys ------------------------------------------------------
    def _witness(self, question: str, draft: str, evidence: list[str],
                 ctx: RunContext) -> tuple[str, dict[str, Any]]:
        """Recall the answer closed-book, then require the corpus to print it beside a question name.

        No passage, no draft and no reservation is shown to the recalling call: the point is a
        candidate that owes nothing to the material the first pass already failed on.  Adoption is
        decided by a test with no model call in it -- the value and a name the question uses must
        appear in the SAME fetched lead -- so an uncorroborated memory can never take the item.
        """
        info: dict[str, Any] = {"recalled": "", "witness": "", "opened": [], "n_fresh": 0,
                                "confirmed_by": "", "grounded": False}
        try:
            out = self.recall(ctx.llm, tag="recall", question=question)
        except Exception:
            return "", info
        recalled = (out.get("answer", "") or "").strip().split("\n")[0].strip()
        recalled = recalled.strip("*_`\"").rstrip(".,;: ").strip()
        witness = _trim_mention((out.get("witness", "") or "").strip().strip('"'))[:120]
        info["recalled"], info["witness"] = recalled, witness
        if not recalled or normalize_text(recalled) == normalize_text(draft):
            return "", info          # memory agrees with the draft: nothing to decide

        # the article the memory points at, opened by title, plus one probe that joins the
        # remembered value to the question's own anchor -- a wrong memory has nowhere to hide
        fresh: list[str] = []
        fresh_seen: set[str] = set()
        if witness:
            article = self._lookup(ctx, witness)
            if article:
                self._add(fresh, fresh_seen, [article])
        anchors = _question_anchors(question)
        probe = f"{recalled} {anchors[0]}" if anchors else recalled
        self._add(fresh, fresh_seen, ctx.tools.retrieve(probe, k=self.K_WITNESS)[:3])
        fresh = fresh[: self.MAX_WITNESS]
        info["opened"] = [_title_of(p) for p in fresh]
        info["n_fresh"] = len(fresh)
        if not fresh:
            return "", info

        terms = _question_terms(question) if not anchors else []
        for passage in fresh:
            if not _prints(passage, recalled):
                continue
            if anchors:
                ok = any(_prints(passage, a) for a in anchors)
            else:
                hit = sum(1 for t in terms if t in set(_words(passage)))
                ok = hit >= self.MIN_TERMS
            if ok:
                info["confirmed_by"] = _title_of(passage)
                info["grounded"] = True
                break
        return (recalled if info["grounded"] else ""), info

    # -- typed anchor admission --------------------------------------------------------------
    def _retype(self, question: str, pool: list[str], seen: set[str],
                ctx: RunContext) -> list[dict[str, Any]]:
        """Swap out pool articles that carry an anchor's name but not the kind the question gives."""
        moves: list[dict[str, Any]] = []
        for mention, descs in _typed_mentions(question):
            if len(moves) >= self.MAX_RETYPE:
                break
            held, held_score = -1, 0.0
            for i, passage in enumerate(pool[: self.POOL_CAP]):
                score = _title_match(mention, passage)
                if score > held_score:
                    held, held_score = i, score
            if held_score < self.MIN_ANCHOR_MATCH:
                continue                      # no article claims to be this entity: not our case
            if _states_type(pool[held], descs):
                continue                      # the anchor already is the kind the question says
            impostor = _title_of(pool[held])
            better = ""
            for hit in ctx.tools.retrieve(f"{mention} {' '.join(descs)}", k=self.K_TYPE):
                if _title_of(hit).lower() == impostor.lower():
                    continue
                if _title_match(mention, hit) >= self.MIN_ANCHOR_MATCH and _states_type(hit, descs):
                    better = hit
                    break
            if not better:
                continue
            key = _title_of(better).lower()
            if key in seen:                   # already in the pool under its own name: leave it
                continue
            seen.discard(impostor.lower())
            seen.add(key)
            pool[held] = better               # substitution, not growth
            moves.append({"mention": mention, "type": descs, "was": impostor,
                          "now": _title_of(better)})
        return moves

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # 0. shared-property questions get the intersective path; everything else falls through --
        pair = _shared_pair(question)
        if pair:
            routed = self._shared_route(question, pair, ctx)
            if routed is not None:
                return routed
            ctx.note("shared-property route declined", pair=list(pair))

        pool: list[str] = []
        seen: set[str] = set()
        tail: list[str] = []
        tail_seen: set[str] = set()

        # 1. decompose (unchanged) --------------------------------------------------------------
        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )

        # 2. every query at full depth: head -> parent pool, ranks below -> deep tail -----------
        self._deep(ctx, question, pool, seen, tail, tail_seen, self.K_PLAN)
        for q in queries:
            self._deep(ctx, q, pool, seen, tail, tail_seen, self.K_PLAN)

        # 2b. an anchor whose article is not the kind the question names is replaced, in place ---
        retyped = self._retype(question, pool, seen, ctx)
        if retyped:
            ctx.note("anchor retyped", moves=retyped)

        # 3-4. triage with an adaptive follow-up hop (pool the same size as the parent's) --------
        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_TRIAGE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.triage(ctx.llm, tag=f"triage{step + 1}", question=question, pool=table)
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
                range(1, min(len(shown), self.SELECT_CAP) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in ("unknown", "none", "n/a"):
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_TRIAGE - 1:
                break
            nq = _split_lines(out.get("next_query", ""), 1)
            if not nq:
                break
            followups.append(nq[0])
            before = len(pool)
            self._deep(ctx, nq[0], pool, seen, tail, tail_seen, self.K_FOLLOW)
            if len(pool) == before:
                break
            pool = [pool[i - 1] for i in ids] + [p for i, p in enumerate(pool, 1) if i not in ids]
            ctx.note("follow-up hop", query=nq[0], pool_size=len(pool), tail=len(tail))

        # 5. the selected passages, closed under the links between pool passages ----------------
        shown = pool[: self.POOL_CAP]
        picked = [i - 1 for i in ids[: self.SELECT_CAP] if 1 <= i <= len(shown)]
        if not picked:
            picked = list(range(min(len(shown), self.SELECT_CAP)))
        room = max(0, min(self.CLOSE_CAP, self.EVIDENCE_CAP - len(picked)))
        partners = _link_closure(shown, picked, question, linking_fact, room, self.MIN_LINK) \
            if room else []
        evidence = [shown[i] for i in picked] + [shown[i] for i, _, _ in partners]
        if partners:
            ctx.note("link closure", admitted=[{"title": _title_of(shown[i]), "why": why,
                                                "weight": round(w, 2)} for i, w, why in partners])

        # each admitted partner retires distilled tail lines, so the answerer reads no more than
        # the parent pipeline handed it -- the closure trades unverified sentences for whole leads
        read = {_title_of(p).lower() for p in evidence}
        keep = max(4, self.DIGEST_LINES - self.DIGEST_PER_ADMIT * len(partners))
        digest = self._digest(question, linking_fact, tail, read)[:keep]
        ctx.note("deep tail distilled", tail=len(tail), lines=len(digest))
        # the same call the parent makes, issued directly so its reasoning can be read back
        resp = ctx.llm.chat(
            self.answer.messages(
                question=question,
                evidence=evidence,
                linking_fact=linking_fact or "unknown",
                extra_leads=digest or ["(none)"],
            ),
            tag="answer",
        )
        parsed = self.answer.parse(resp.text)
        ans = parsed.get("answer", "")
        doubt = f"{getattr(resp, 'reasoning', '') or ''}\n{parsed.get('reasoning', '')}"
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()

        # 6. admissibility gate: a draft that cannot be an answer buys one recovery round --------
        failed = _admissible(question, ans)
        repaired, info = "", {}
        if failed:
            repaired, info = self._recover(question, ans, evidence, digest, pool, ctx)
            ctx.note("admissibility gate fired", test=failed, draft=ans, repaired=repaired,
                     **{k: v for k, v in info.items()})
            # a recovered value replaces the draft only when it is itself admissible; the draft it
            # replaces was inadmissible, so this can never discard a value that could have scored
            if repaired and not _admissible(question, repaired):
                ans = repaired
            elif repaired and not ans:
                ans = repaired

        # 6b. the uncertainty gate: the solver's own doubt decides who buys a second round ------
        hedges = _hedge_count(doubt)
        resolved, rinfo = "", {}
        if not failed and hedges >= self.MIN_HEDGES and not _binary(ans):
            resolved, rinfo = self._witness(question, ans, evidence, ctx)
            adopt = bool(resolved) and not _admissible(question, resolved) and rinfo.get("grounded")
            ctx.note("witness check fired", hedges=hedges, draft=ans, resolved=resolved,
                     adopted=bool(adopt), **{k: v for k, v in rinfo.items()})
            if adopt:
                ans = resolved

        projected, family = self._project(question, ans, evidence, digest)
        if family:
            ctx.note("answer form projected", family=family, draft=ans, answer=projected)
        return {
            "answer": projected or ans,
            "draft": ans,
            "gate": failed,
            "repaired": repaired,
            "recovery": info,
            "form_family": family,
            "hedges": hedges,
            "resolved": resolved,
            "resolution": rinfo,
            "retyped": retyped,
            "partners": [_title_of(shown[i]) for i, _, _ in partners],
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "tail_size": len(tail),
            "n_digest": len(digest),
            "n_selected": len(evidence),
        }
