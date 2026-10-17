"""Find the bridge entity by JOINING two differently anchored rankings, instead of guessing its name.

The one intersective mechanism this frontier has ever run -- looking for phrases printed in two
leads at once -- is also the only recent candidate that gained, and it is confined to questions that
name both of their subjects.  Bridge questions, four fifths of the search set, are handled the
opposite way: the model reads the pool and WRITES DOWN the name it thinks connects the two halves,
and every second hop is a query built out of that guess.  When the guess is wrong or absent the
needed article is never fetched; 82 of 300 items end with a gold article that no retrieval ever
returned, and they score 35% against 82%.

But a bridge question hands over both ends of the bridge in plain sight.  One end is the proper name
it states.  The other end is everything it says WITHOUT that name -- `who did the vocals on the song
X`, `the university where Y studied law`, `the film in which Z received top billing` -- which is a
description of the article that holds the answer.  Retrieved separately, these two are two rankings
over disjoint parts of the corpus, and the bridge entity is whatever they print in COMMON: the rare
phrase that appears both in a lead the name pulled up and in a lead the description pulled up.  That
is a join, not a guess, and it is decided by string overlap rather than by the model's imagination.

So this harness issues the name-only query and the name-stripped description query, joins their two
rankings on their rarest shared phrase, and fetches the article of the winning phrase by name.  The
model is never asked what the bridge is; it only ever reads the article the join produced.  One
extra retrieval, one extra lead, and the distilled tail is shortened to pay for it.
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


# -- the bridge join: two anchored rankings, intersected on their rarest shared phrase -----------
NAME_SPAN = re.compile(
    r"(?:[\"\u201c][^\"\u201d]{2,60}[\"\u201d])"
    r"|(?:[A-Z][\w.&'\u2019\-]*(?:\s+(?:[A-Z][\w.&'\u2019\-]*|of|the|de|del|la|le|von|van|and)"
    r"(?=\s+[A-Z0-9]))*(?:\s+[A-Z][\w.&'\u2019\-]*)*)")
BRACKET = re.compile(r"\[[^\]]{0,40}\]")
DESC_STOP = STOP | {
    "was", "were", "been", "being", "does", "did", "not", "but", "for", "her", "his", "its",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "many", "much",
    "one", "two", "all", "any", "some", "more", "most", "well", "you", "him", "them", "she",
    "they", "there", "then", "than", "and", "the", "are", "had", "has", "have", "also",
}
SENT_LEAD = re.compile(r"^(?:what|which|who|whom|whose|when|where|why|how)\b\s*", re.IGNORECASE)


def _stated_names(question: str) -> list[str]:
    """The proper names the question states outright -- one end of the bridge."""
    body = BRACKET.sub(" ", question or "")
    body = SENT_LEAD.sub("", body.strip())
    out: list[str] = []
    seen: set[str] = set()
    for m in NAME_SPAN.finditer(body):
        name = _trim_mention(m.group().strip('"\u201c\u201d'))
        if len(name) < 3 or name.lower() in STOP:
            continue
        key = normalize_text(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _description(question: str, names: list[str]) -> str:
    """The question with the names it states removed, reduced to its content words.

    What is left of a question after deleting its names is mostly scaffolding -- `is a song on an
    album released on which date` -- and handing that to a term-matching index returns noise.  The
    describing end of the bridge is the content words inside it: the nouns and modifiers that say
    what KIND of article holds the answer.  Keeping only those, in their original order and casing,
    is what makes the second ranking about the target rather than about English.
    """
    body = BRACKET.sub(" ", question or "")
    for name in sorted(names, key=len, reverse=True):
        body = re.sub(re.escape(name), " ", body)
    out: list[str] = []
    for raw in re.split(r"[^\w'\u2019\-]+", body):
        token = raw.strip("'\u2019-")
        if len(token) > 2 and token.lower() not in DESC_STOP:
            out.append(token)
    return " ".join(out).strip()


def _calls_left(ctx: RunContext) -> int:
    """Retrievals still available on this item, read off the tool object the framework owns."""
    tools = ctx.tools
    return max(0, int(getattr(tools, "max_calls", 12)) - int(getattr(tools, "calls", 0)))


def _join_phrase(rank_a: list[str], rank_b: list[str], question: str, limit: int) -> list[str]:
    """The rarest phrases printed in BOTH rankings, best first.  Pure string work, no model call."""
    corpus = rank_a + rank_b
    df: dict[str, int] = {}
    for passage in corpus:
        for w in set(_words(passage)):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(corpus))
    qtokens = set(_ntoks(question))
    # a phrase is a candidate bridge only if it looks like the name of an article
    grams_a: dict[tuple[str, ...], str] = {}
    for depth, passage in enumerate(rank_a):
        for key, surface in _ngrams(passage.partition(" | ")[2], 3).items():
            if key not in grams_a and _name_like(surface):
                grams_a[key] = surface
    scored: list[tuple[float, str]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for passage in rank_b:
        for key, surface in _ngrams(passage.partition(" | ")[2], 3).items():
            if key not in grams_a or key in seen_keys:
                continue
            seen_keys.add(key)
            content = [k for k in key if k not in PHRASE_STOP]
            if len(content) < 2 or all(k in qtokens for k in content):
                continue          # the question already said it, or it is a one-word fragment
            if not _name_like(surface) or any(len(k) < 2 for k in content):
                continue          # a bridge is an article title, not a run of sentence words
            weight = sum(math.log(1.0 + n / (1.0 + df.get(k, 0))) for k in content)
            weight *= 1.0 + 0.15 * (len(content) - 1)
            scored.append((weight, grams_a[key]))
    scored.sort(key=lambda t: (-t[0], len(t[1])))
    out: list[str] = []
    seen: set[str] = set()
    for _, surface in scored:
        key = normalize_text(surface)
        if key in seen:
            continue
        seen.add(key)
        out.append(surface)
        if len(out) >= limit:
            break
    return out


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
    MAX_REPAIR_EVIDENCE = 6

    # -- the bridge join ------------------------------------------------------------------------
    K_JOIN = 10          # depth of each of the two anchored rankings
    JOIN_CONTEXT = 12    # pool passages counted as part of the name end of the bridge
    MIN_DESC_WORDS = 3   # content words the describing end needs before it is worth a query
    MAX_JOIN = 3         # joined phrases considered, best first
    K_BRIDGE = 6         # hits for the one lookup of the winning phrase
    MIN_BRIDGE_TITLE = 0.5
    JOIN_RESERVE = 7     # the join only starts while this many retrievals are still unspent
    JOIN_DIGEST = 12     # tail lines kept once the join has put whole leads in the evidence

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

    def _join(self, question: str, pool: list[str], seen: set[str],
              ctx: RunContext) -> dict[str, Any]:
        """Retrieve both ends of the bridge separately and intersect the two rankings.

        Returns what it did, for the trace.  The model is not consulted at any point: the name end
        comes from the question's own capitalisation, the description end from what is left when
        those names are deleted, and the bridge from the rarest phrase the two rankings share.
        """
        names = _stated_names(question)
        desc = _description(question, names)
        info: dict[str, Any] = {"names": names[:2], "description": desc[:120]}
        if not names or len(desc.split()) < self.MIN_DESC_WORDS \
                or _calls_left(ctx) < self.JOIN_RESERVE:
            info["fired"] = False
            return info
        rank_a = ctx.tools.retrieve(" ".join(names[:2]), k=self.K_JOIN)
        rank_b = ctx.tools.retrieve(desc[:200], k=self.K_JOIN)
        # the name end of the bridge is the name ranking TOGETHER with what the planner's queries
        # already pulled up: both were anchored on names the question states, so both are the same
        # end of the bridge, and the wider that end is the more phrases the join has to match on
        joined = _join_phrase(rank_a + pool[: self.JOIN_CONTEXT], rank_b, question, self.MAX_JOIN)
        info["fired"] = True
        info["joined"] = joined

        promoted: list[str] = []
        # the winning phrase is looked up by name: BM25 over leads answers a bare name well, and a
        # hit only counts when its own title IS the phrase, so a merely topical article is refused
        titles = {normalize_text(_title_of(p)) for p in pool}
        for cand in joined:
            if normalize_text(cand) in titles or _calls_left(ctx) < self.JOIN_RESERVE - 3:
                continue
            for hit in ctx.tools.retrieve(cand, k=self.K_BRIDGE):
                if _title_match(cand, hit) >= self.MIN_BRIDGE_TITLE:
                    info["bridge"] = _title_of(hit)
                    promoted.append(hit)
                    break
            break                          # one bridge lookup per item, whatever it returns

        # only the title-verified bridge article enters the pool, at the front so that the triage
        # call actually sees it.  The two rankings themselves are scratch work: promoting their
        # heads means promoting whatever a term index makes of a description, which is noise.
        front: list[str] = []
        for passage in promoted:
            key = _title_of(passage).lower()
            if key in seen:
                continue
            seen.add(key)
            front.append(passage)
        pool[:0] = front
        info["promoted"] = [_title_of(p) for p in front]
        return info

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

        # 2b. both ends of the bridge, retrieved separately and intersected ---------------------
        joininfo = self._join(question, pool, seen, ctx)
        ctx.note("bridge join", **joininfo)

        # 2c. an anchor whose article is not the kind the question names is replaced, in place ---
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

        # 5. answer from the raw selected passages plus the distilled deep tail -----------------
        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]
        read = {_title_of(p).lower() for p in evidence}
        keep = self.JOIN_DIGEST if joininfo.get("promoted") else self.DIGEST_LINES
        digest = self._digest(question, linking_fact, tail, read)[:keep]
        ctx.note("deep tail distilled", tail=len(tail), lines=len(digest))
        ans = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
            extra_leads=digest or ["(none)"],
        ).get("answer", "")
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
            "retyped": retyped,
            "join": joininfo,
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "tail_size": len(tail),
            "n_digest": len(digest),
            "n_selected": len(evidence),
        }
