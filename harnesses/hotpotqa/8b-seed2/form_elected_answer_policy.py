"""Choose the wording of both searches by measuring it on the train split instead of assuming it.

Every system on the board issues its first search as the raw question and its second as the bare page name
the first call asked for. Neither wording was ever measured: they are habits inherited from the seed, and
gold-page recall has been frozen at exactly the same value for six consecutive rewrites while 49 of the
frontier's 84 errors sit on items where a reference page was never retrieved.

The train split has labelled reference pages, so the wording of a search is a decision that can be settled
with data rather than taste. `prepare` therefore runs a small tournament: several deterministic rewritings of
the question compete on first-hop recall at k=7, and -- on exactly the reference pages the winning first hop
misses -- several compositions of the second query compete on whether they bring the missing page back. The
compositions differ in how the named target is weighted against the question's own content words: the name
alone, the name plus the question's content words, the name repeated so BM25 weights it above the borrowed
context, and so on. Short generic page names lose to their own context words in a bag-of-words index, and
repeating the name is the cheapest way to pay for that without spending another call.

The winners, their measured scores and their support counts are stored in the harness state and applied at
run time: the first search uses the winning rewriting, and the target proposed by the triage call is expanded
into the winning composition before the second search. Nothing else changes -- same two searches, same k,
same triage and answering calls, same evidence pool, same answer policy -- so the experiment isolates query
surface. If the measurement says the inherited wording is best, the tournament elects it and the harness is
the frontier exactly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from concurrent.futures import ThreadPoolExecutor

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import em_score

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


# --------------------------------------------------------------------------------------------------
# deterministic query compositions -- pure string functions, the alphabet the train tournament ranges over
# --------------------------------------------------------------------------------------------------
def _content_words(text: str) -> str:
    """The question stripped of function words: what a bag-of-words index actually scores on."""
    words = [w for w in re.sub(r"[^\w\s]", " ", text or "").split() if w.lower() not in STOP]
    return " ".join(words)


def _tpl_raw(question: str) -> str:
    return question


def _tpl_content(question: str) -> str:
    return _content_words(question) or question


def _tpl_spans_content(question: str) -> str:
    named = " ".join(_named_spans(question)[:4])
    return (named + " " + _content_words(question)).strip() or question


HOP1_TEMPLATES = {"raw": _tpl_raw, "content": _tpl_content, "spans_content": _tpl_spans_content}


def _c_target(target: str, question: str) -> str:
    return target


def _c_target_context(target: str, question: str) -> str:
    return (target + " " + _content_words(question)).strip() or target


def _c_target_twice(target: str, question: str) -> str:
    return (target + " " + target).strip()


def _c_target_twice_context(target: str, question: str) -> str:
    return (target + " " + target + " " + _content_words(question)).strip() or target


HOP2_TEMPLATES = {
    "target": _c_target,
    "target_context": _c_target_context,
    "target_twice": _c_target_twice,
    "target_twice_context": _c_target_twice_context,
}


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
# shared-phrase intersection for questions about what two subjects have in common
# --------------------------------------------------------------------------------------------------
COMMONALITY = re.compile(
    r"(?i)\bin common\b|\bboth\b|\bwhat (?:type|kind|sort) of\b|\bare which\b|\bis which\b|"
    r"\bhow (?:is|are) .*\brelated\b|\bshare\b"
)
EDGE_STOP = STOP | {
    "also", "both", "been", "being", "had", "have", "has", "were", "was", "are", "is", "be", "his", "her",
    "their", "he", "she", "they", "it", "who", "which", "what", "when", "where", "and", "or", "but", "as",
    "well", "known", "best", "such", "other", "than", "into", "over", "after", "before", "during", "while",
}


def _phrases(text: str, nmax: int = 6) -> set[str]:
    """Every word sequence of the passage that could stand alone as an answer."""
    body = (text or "").partition(" | ")[2] or (text or "")
    toks = _norm(body).split()
    out: set[str] = set()
    for n in range(1, nmax + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i : i + n]
            if gram[0] in EDGE_STOP or gram[-1] in EDGE_STOP:
                continue
            if any(t.isdigit() and len(t) == 4 for t in gram):
                continue
            out.add(" ".join(gram))
    return out


def _best_passage(span: str, pool: list[str]) -> int:
    """Which pooled passage is the page for this subject, by title overlap."""
    st = _content_tokens(span)
    best, best_score = -1, 0.0
    for i, text in enumerate(pool):
        title = text.partition(" | ")[0]
        tt = _content_tokens(title)
        if not tt and not st:
            continue
        inter = len(st & tt)
        score = inter / max(1, len(st | tt)) + (1.0 if _norm(span) == _norm(title) else 0.0)
        if inter and score > best_score:
            best, best_score = i, score
    return best


def shared_phrases(question: str, pool: list[str], limit: int = 12) -> list[str]:
    """Phrases written in at least two of the question's subject pages, most general first.

    Matching each named subject to exactly one passage sounds right and does not work: measured over the
    frontier's own evidence pools it identifies two distinct subject pages on only 4 of 14 items of this
    class, because a question's capitalised spans include adjectives and nationalities and because the
    second subject's page is often absent from the pool altogether. So the requirement is relaxed to
    agreement between any two of the passages the question points at: a property shared by both subjects is
    stated by at least two pages, while text peculiar to one page is stated by one. Nothing here is a
    judgement -- the menu is whatever those passages literally agree on.
    """
    spans = [s for s in _named_spans(question) if len(s.split()) <= 8]
    qt = _content_tokens(question)
    cand = [text for text in pool if _content_tokens(text.partition(" | ")[0]) & qt]
    if len(cand) < 2:
        cand = pool[:6]
    if len(cand) < 2:
        return []
    counts: dict[str, int] = {}
    for text in cand[:8]:
        for phrase in _phrases(text):
            counts[phrase] = counts.get(phrase, 0) + 1
    titles = {_norm(text.partition(" | ")[0]) for text in cand}
    named = {_norm(s) for s in spans}
    kept = []
    for phrase, seen_in in counts.items():
        if seen_in < 2:
            continue
        toks = phrase.split()
        if phrase in titles or phrase in named:
            continue
        if len(toks) == 1 and (len(phrase) <= 3 or phrase in qt):
            continue
        if all(t in qt for t in toks):
            continue
        kept.append((phrase, seen_in))
    if not kept:
        return []
    # keep maximal phrases, then show the most general first: the shared *category* is normally the short
    # phrase, while long shared strings are boilerplate the abstracts happen to have in common
    kept.sort(key=lambda x: (-len(x[0].split()), -len(x[0])))
    maximal: list[tuple[str, int]] = []
    for phrase, seen_in in kept:
        if any(phrase in longer and phrase != longer for longer, _ in maximal):
            continue
        maximal.append((phrase, seen_in))
    maximal.sort(key=lambda x: (-x[1], len(x[0].split()), len(x[0])))
    return [phrase for phrase, _ in maximal[:limit]]


def _on_menu(answer: str, menu: list[str], loose: bool = False) -> str:
    """The menu entry the answer names, or "" when it names none of them.

    Exact by default: the point of the menu is the exact wording, so a draft that merely overlaps an entry
    still needs the forced choice. `loose` is used afterwards, to snap the reply onto the entry it named.
    """
    na = _norm(answer)
    if not na:
        return ""
    for phrase in menu:
        if na == phrase:
            return phrase
    if not loose:
        return ""
    for phrase in sorted(menu, key=lambda s: -len(s)):
        if na in phrase or phrase in na:
            return phrase
    return ""


MENU_INSTRUCTIONS = """`question` asks what two subjects have in common: what they both are, what they
share, or how they are related. The reference answer to a question of this shape is the shared property
itself, worded the way the sources word it -- not a statement about the pair, not what tells them apart, and
not a synonym of your own choosing. `author` is wrong where the sources say `writer`; a clause is wrong where
they say one word.

`shared_wording` lists word sequences that occur in more than one of the retrieved abstracts, so it is a
sample of how the sources themselves word this pair. It is a hint about wording, not a menu you must obey:
several entries will be irrelevant, and the property you need may be worded in `passages` without appearing
there at all. `draft` is a previous attempt.

Answer with the most general property that is genuinely true of BOTH subjects and is of the kind the question
asks for -- a profession, a nationality, a category, a type of thing -- copied in the words the passages use
for it, singular and unqualified. Prefer a single word or a short noun phrase over a clause. If the question
can be answered with yes or no, answer exactly `yes` or exactly `no`. If `draft` is already the sources'
wording of the shared property, keep it.

Return only `answer`: the property, with no sentence around it, no explanation and no final period."""

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


_CHOICE = re.compile(r"\b(?:or)\b", re.I)
_COMMON = re.compile(r"\b(?:both|in common|the same|share[ds]?)\b", re.I)


def coarse_form(question: str) -> str:
    """A handful of buckets, coarse enough that a 150-question train split can measure each of them.

    `form_key` is deliberately fine-grained for picking look-alike demonstrations; it is far too sparse to
    estimate a policy over. These buckets are the answer's role instead of the question's wording."""
    q = " ".join((question or "").split())
    ql = q.lower()
    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would)\b", ql):
        return "yesno"
    if _CHOICE.search(ql):
        return "choice"
    if _COMMON.search(ql):
        return "common"
    if ql.startswith("how many") or ql.startswith("how much") or "the number of" in ql:
        return "count"
    if re.search(r"\b(what|which) (year|date)\b", ql) or ql.startswith("when"):
        return "date"
    if re.search(r"\b(what|which) (city|country|county|state|town|place|region)\b", ql) or ql.startswith("where"):
        return "place"
    if ql.startswith("who") or ql.startswith("whom") or ql.startswith("whose"):
        return "person"
    return "thing"


SOURCED_INSTRUCTIONS = """Answer `question` from `passages`, which are encyclopaedia abstracts written as
`Title | text`, and from `facts`, which were copied out of the first search.

Work by locating the wording rather than by composing it.

First `supporting_titles`: the titles of the one or two passages that justify the answer.

Then `source_sentence`: the ONE sentence from `passages` that states the fact the question ends on, copied
out character for character. If no sentence states it, write `none`.

Then `evidence_status`: `complete` when `source_sentence` states the asked fact, `partial` when the passages
fix which entities are meant but never state that fact, `absent` when they do not even cover the subjects.
Then `missing_fact`: for `partial`/`absent`, the one thing the passages fail to state (else `none`).

Then `answer_type`, and then `answer`.

`answer` is a CONTIGUOUS piece of `source_sentence` whenever `source_sentence` is not `none`: cut the
sentence down to exactly the part the question asks for and change nothing inside what you keep.
- Keep the boundaries the sentence itself draws. If the sentence writes a place with the region or country
  after a comma, keep that; if it writes a person's name in full where the question used a short form, keep
  the full form; if it writes a count together with the thing counted, keep both; if it puts a year or a
  qualifier in front of an event's name, keep it. Do not tidy, shorten or expand across those boundaries.
- Do not include any part of the sentence that merely repeats what the question already told you.
- `solved_examples` show the length and register of answers to questions of this form; use them to decide
  how much of the sentence to keep, never to supply content. `style_note` summarises them.
- When `source_sentence` is `none`, fix from the passages WHICH entity the question is about and answer the
  missing fact from your own knowledge of that entity, written as a reference work would write it.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- A yes/no question is answered with exactly `yes` or exactly `no`.
- A question asking which of two named things has a property is answered with just that one name.
- Always commit to one answer; never reply that the information is missing."""


CHAIN_INSTRUCTIONS = """Answer `question` from `passages`, which are encyclopaedia abstracts written as
`Title | text`, and from `facts`, which were copied out of the first search.

These questions hide a step. Make the step explicit before answering.

First `asked_for`: the thing the question actually demands, as a bare noun phrase -- the slot, not its value
(for example a role, a date, a place, a count, one of two named options). Be precise about WHICH slot: a
question about who directed something is not a question about who wrote it.

Then `link`: the entity that connects the question to the page carrying the answer, named as its page is
titled, or `direct` when the question needs no intermediate entity.

Then `supporting_titles`: the titles of the one or two passages that justify the answer, with the page that
carries `asked_for` last.

Then `evidence_status`: `complete`, `partial` or `absent` for whether those passages state `asked_for`.
Then `missing_fact`: for `partial`/`absent`, the one thing they fail to state (else `none`).

Then `answer_type`, and then `answer`.

Rules for `answer`:
- It must be a value of `asked_for` and of nothing else. Before writing it, check that it fills that slot:
  an answer of the right kind that fills a neighbouring slot is wrong.
- If `evidence_status` is `complete`, copy it verbatim from the passage that carries it, in that passage's
  own spelling, capitalisation and level of detail.
- Otherwise use the passages only to fix which entity `link` is, then answer `asked_for` from your own
  knowledge of that entity.
- `solved_examples` show how long and in what register answers of this form are written; `style_note`
  summarises them. Follow them for form only; they never supply content.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- A yes/no question is answered with exactly `yes` or exactly `no`.
- A question asking which of two named things has a property is answered with just that one name.
- Always commit to one answer; never reply that the information is missing."""


REPAIR_INSTRUCTIONS = """`draft` is a verbose or evasive attempt at answering `question` from `passages`.

Return only `answer`: the bare span that answers the question, normally one to four words, copied verbatim
from `passages` or from `draft`. No sentence, no explanation, no final period. Answer a yes/no question
with exactly `yes` or exactly `no`. When the question offers two named alternatives, return exactly one of
them, spelled as in the question. Never reply that the information is missing -- commit to the best span."""


SCAFFOLDS = ("span", "sourced", "chain")


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
        self.menu = Module(
            inputs=["question", "shared_wording", "passages", "draft"],
            outputs=["answer"],
            instructions=MENU_INSTRUCTIONS,
            chain_of_thought=False,
            descs={
                "shared_wording": "phrases occurring in more than one retrieved abstract; a hint, not a menu",
                "answer": "the shared property in the sources' own wording, bare",
            },
        )
        self.sourced = Module(
            inputs=["question", "solved_examples", "style_note", "facts", "passages"],
            outputs=["supporting_titles", "source_sentence", "evidence_status", "missing_fact",
                     "answer_type", "answer"],
            instructions=SOURCED_INSTRUCTIONS,
            descs={
                "solved_examples": "already-solved questions of the same form with their reference answers; a "
                                   "specification of answer form only, never a source of content",
                "source_sentence": "the sentence from the passages that states the asked fact, copied exactly",
                "answer": "a contiguous piece of source_sentence, bare",
            },
        )
        self.chained = Module(
            inputs=["question", "solved_examples", "style_note", "facts", "passages"],
            outputs=["asked_for", "link", "supporting_titles", "evidence_status", "missing_fact",
                     "answer_type", "answer"],
            instructions=CHAIN_INSTRUCTIONS,
            descs={
                "asked_for": "the slot the question demands, as a bare noun phrase",
                "link": "the connecting entity named as its page is titled, or direct",
                "answer": "the value of asked_for as a bare span, no sentence, no punctuation around it",
            },
        )
        self._policy: dict[str, str] = {}
        self._votes: dict[str, Any] = {}
        self._demos: list[dict[str, Any]] = []
        self._hop1 = "raw"
        self._hop2 = "target"
        self._scores: dict[str, Any] = {}

    # -- prepare: index the train split, then elect both query wordings by measurement ---------------
    MAX_PROBE_ITEMS = 110

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
        self._elect_queries(train_items, ctx)
        try:
            self._elect_scaffolds(train_items, ctx)
        except Exception as exc:  # noqa: BLE001 - the election is an add-on, never a prerequisite
            self._policy = {}
            ctx.note("election_failed", error=str(exc)[:300])

    def _titles_for(self, item: dict[str, Any], query: str, ctx: PrepareContext) -> set[str]:
        """One search for one train item, on a fresh tool handle (train-only, never at evaluation time)."""
        try:
            with ctx.open_tools(item) as tools:
                return {_norm(d.get("title", "")) for d in tools.retrieve_docs(query, k=self.K)}
        except Exception:  # noqa: BLE001 - a flaky probe must never break prepare
            return set()

    def _elect_queries(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        sample = [it for it in train_items if it.get("question") and it.get("gold_titles")][: self.MAX_PROBE_ITEMS]
        if not sample:
            return

        # round 1: which rewriting of the question finds most reference pages in the first search?
        hop1_hits = {name: 0 for name in HOP1_TEMPLATES}
        hop1_titles: dict[str, list[set[str]]] = {name: [] for name in HOP1_TEMPLATES}
        n_gold = 0
        for it in sample:
            golds = [_norm(t) for t in it["gold_titles"] if t]
            n_gold += len(golds)
            for name, fn in HOP1_TEMPLATES.items():
                titles = self._titles_for(it, fn(it["question"]), ctx)
                hop1_titles[name].append(titles)
                hop1_hits[name] += sum(1 for g in golds if g in titles)
        best1 = max(HOP1_TEMPLATES, key=lambda n: (hop1_hits[n], n == "raw"))
        if hop1_hits[best1] <= hop1_hits.get("raw", 0):
            best1 = "raw"  # ties and regressions keep the inherited wording

        # round 2: on exactly the reference pages the winning first hop misses, which composition of the
        # second query brings the page back? The target is the page's own name, i.e. a perfect proposal --
        # what is being ranked is the composition around it, not the ability to name it.
        misses: list[tuple[dict[str, Any], str]] = []
        for it, titles in zip(sample, hop1_titles[best1]):
            for t in it["gold_titles"]:
                if t and _norm(t) not in titles:
                    misses.append((it, t))
        hop2_hits = {name: 0 for name in HOP2_TEMPLATES}
        for it, target in misses:
            for name, fn in HOP2_TEMPLATES.items():
                titles = self._titles_for(it, fn(target, it["question"]), ctx)
                hop2_hits[name] += int(_norm(target) in titles)
        best2 = max(HOP2_TEMPLATES, key=lambda n: (hop2_hits[n], n == "target"))
        if not misses or hop2_hits[best2] <= hop2_hits.get("target", 0):
            best2 = "target"

        self._hop1 = best1
        self._hop2 = best2
        self._scores = {
            "hop1": {k: [v, n_gold] for k, v in hop1_hits.items()},
            "hop2": {k: [v, len(misses)] for k, v in hop2_hits.items()},
        }
        ctx.note("elected_query_templates", hop1=best1, hop2=best2, scores=self._scores)


    # -- election of the answering scaffold, per question form, by measured train accuracy -------------
    #
    # Earlier systems fixed ONE reasoning structure for the answering call and argued about its wording.
    # Different structures are measurably better at different forms of question, and which is which is not
    # something to guess: the three structures are run side by side on the same train contexts and the one
    # that is exactly right most often for a form is the one that form gets. `span` is the incumbent and
    # wins every tie, so a form only changes structure when the change is paid for by measurement.
    MAX_ELECT_ITEMS = 84
    MIN_FORM_ITEMS = 5
    ELECT_WORKERS = 4

    def _scaffold_call(self, name: str, c: dict[str, Any], llm: Any) -> dict[str, Any]:
        mod = {"span": self.answerer, "sourced": self.sourced, "chain": self.chained}[name]
        return mod(llm, tag="ans_" + name, question=c["q"], solved_examples=c["shown"],
                   style_note=c["style"], facts=c["facts"], passages=c["pool"])

    def _context_for(self, item: dict[str, Any], ctx: PrepareContext) -> dict[str, Any] | None:
        """Build the evidence the answering call would see for one TRAIN item, exactly as run() builds it."""
        question = (item.get("question") or "").strip()
        if not question:
            return None
        try:
            with ctx.open_tools(item) as tools:
                hop1 = tools.retrieve_docs(HOP1_TEMPLATES[self._hop1](question), k=self.K)
                t = self.triage(ctx.llm, tag="el_triage", question=question,
                                retrieved_titles=[d["title"] for d in hop1],
                                passages=[d["text"] for d in hop1])
                target = (t.get("next_query", "") or "").strip() or question
                hop2 = tools.retrieve_docs(
                    HOP2_TEMPLATES[self._hop2](self._clean_query(target), question), k=self.K)
        except Exception:  # noqa: BLE001 - a flaky probe must never break prepare
            return None
        pool, seen = [], set()
        for d in hop2 + hop1:
            key = d["title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(d["text"])
        demos = [d for d in self._select_demos(question) if _norm(d["q"]) != _norm(question)]
        return {
            "q": question,
            "gold": (item.get("answer") or "").strip(),
            "form": coarse_form(question),
            "pool": pool,
            "facts": (t.get("facts", "") or "").strip() or "none",
            "shown": [f"question: {d['q']}\nreference answer: {d['a']}" for d in demos] or ["none available"],
            "style": _style_note(demos),
        }

    def _elect_scaffolds(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        sample = [it for it in train_items if it.get("question") and it.get("answer")][: self.MAX_ELECT_ITEMS]
        if not sample:
            return
        with ThreadPoolExecutor(max_workers=self.ELECT_WORKERS) as ex:
            contexts = [c for c in ex.map(lambda it: self._context_for(it, ctx), sample) if c]
        if not contexts:
            return

        jobs = [(c, name) for c in contexts for name in SCAFFOLDS]

        def score(job: tuple[dict[str, Any], str]) -> int:
            c, name = job
            try:
                out = self._scaffold_call(name, c, ctx.llm)
            except Exception:  # noqa: BLE001
                return 0
            got = polish(out.get("answer", ""), (out.get("answer_type", "") or "").strip(), c["q"])
            return int(bool(got) and em_score(got, c["gold"]))

        with ThreadPoolExecutor(max_workers=self.ELECT_WORKERS) as ex:
            marks = list(ex.map(score, jobs))

        tally: dict[str, dict[str, int]] = {}
        counts: dict[str, int] = {}
        for (c, name), mark in zip(jobs, marks):
            tally.setdefault(c["form"], {n: 0 for n in SCAFFOLDS})[name] += mark
            if name == SCAFFOLDS[0]:
                counts[c["form"]] = counts.get(c["form"], 0) + 1

        policy: dict[str, str] = {}
        for form, row in tally.items():
            if counts.get(form, 0) < self.MIN_FORM_ITEMS:
                continue
            winner = max(SCAFFOLDS, key=lambda n: (row[n], n == "span"))
            if winner != "span" and row[winner] > row["span"]:
                policy[form] = winner
        self._policy = policy
        self._votes = {f: {"n": counts.get(f, 0), **row} for f, row in sorted(tally.items())}
        ctx.note("elected_scaffolds", policy=policy, votes=self._votes)

    def get_state(self) -> str:
        return json.dumps({"demos": self._demos, "hop1": self._hop1, "hop2": self._hop2,
                           "scores": self._scores, "policy": self._policy, "votes": self._votes})

    def set_state(self, state: str) -> None:
        try:
            blob = json.loads(state or "{}")
        except ValueError:
            blob = {}
        self._demos = blob.get("demos", []) or []
        self._hop1 = blob.get("hop1") if blob.get("hop1") in HOP1_TEMPLATES else "raw"
        self._hop2 = blob.get("hop2") if blob.get("hop2") in HOP2_TEMPLATES else "target"
        self._scores = blob.get("scores") or {}
        pol = blob.get("policy") or {}
        self._policy = {str(k): str(v) for k, v in pol.items() if v in SCAFFOLDS}
        self._votes = blob.get("votes") or {}

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

        hop1_query = HOP1_TEMPLATES.get(self._hop1, _tpl_raw)(question)
        hop1 = ctx.tools.retrieve_docs(hop1_query, k=self.K)
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

        target_2 = self._gate(t.get("next_query", ""), question, titles1, ctx)
        query_2 = HOP2_TEMPLATES.get(self._hop2, _c_target)(target_2, question)
        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)

        # deduplicated verbatim evidence pool: newly found pages first, they carry the missing hop
        pool, seen = [], set()
        for d in hop2 + hop1:
            key = d["title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(d["text"])

        demos = self._select_demos(question)
        shown = [f"question: {d['q']}\nreference answer: {d['a']}" for d in demos]
        note = _style_note(demos)
        ctx.note("answer_style", form=form_key(question), n_demos=len(demos), style=note[:200])

        scaffold = self._policy.get(coarse_form(question), "span")
        ctx.note("scaffold", form=coarse_form(question), used=scaffold)
        out = self._scaffold_call(
            scaffold,
            {"q": question, "shown": shown or ["none available"], "style": note,
             "facts": facts or "none", "pool": pool},
            ctx.llm,
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
        # the one spare call goes to insufficient evidence first: that is the half of the split where the
        # visible strings cannot contain the answer, so a second look at the same passages cannot help
        # ---- shared-phrase menu: for a commonality question the answer is a string both pages contain,
        # so the intersection of the two subjects' abstracts is a computed menu of candidates. The spare
        # call becomes a forced choice over it whenever the draft is not on the menu.
        menu = shared_phrases(question, pool) if COMMONALITY.search(question) else []
        if menu and _norm(answer) not in ("yes", "no") and not _on_menu(answer, menu):
            picked = self.menu(
                ctx.llm,
                tag="menu",
                question=question,
                shared_wording=menu,
                passages=pool,
                draft=answer or "none",
            )
            candidate = polish(picked.get("answer", ""), atype, question)
            snapped = _on_menu(candidate, menu, loose=True)
            if snapped and _norm(candidate) != snapped and len(candidate.split()) > len(snapped.split()):
                candidate = snapped  # the reply named a shared phrase inside a longer string: take the phrase
            ctx.note("shared_menu", n_options=len(menu), options=menu[:8], before=answer[:120],
                     after=candidate[:120])
            if candidate and not HEDGES.search(candidate):
                answer = candidate
            return {"answer": answer, "raw_answer": raw, "query_2": query_2, "answer_type": atype,
                    "evidence_status": status, "n_options": len(menu)}

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
