"""Name the still-missing page by COPYING it out of the text, instead of free-generating it.

The frontier's second search is aimed by a free-text `next_query` that the first call invents. Two rewrites
of that surface have now failed to move anything (an entity-parsed slot lost recall, a typed page descriptor
gained 0.007), so the loss is not how the name is spelled. Measured on the train split with the real
retriever: after the first search 100 reference pages are still uncovered, and 61% of them occur *verbatim
inside the abstracts already retrieved*, 32% occur in the question, 80% in one or the other -- while a query
that is exactly a page's name brings that page back 87% of the time, which puts the plan's recall ceiling at
0.91 against the 0.81 it actually reaches. So the information needed to aim the second search is already on
screen and the first call simply picks the wrong thing: identification fails, not spelling.

This harness therefore makes the first call *resolve a link* rather than *propose a query*. It must say which
retrieved page the question is anchored on, whether the missing page is reached through that abstract or is the
second thing the question names, and then copy the missing page's name word for word out of the source it just
named. A deterministic layer verifies the copy really occurs in that text (or in the question), and when it
does not it snaps the proposal onto the closest name that does occur -- the candidate names are enumerated
from the retrieved abstracts and the question by span extraction, so the snap target set is the same 80%
pool the measurement describes. Anything that still fails verification falls back to the frontier's gate.

The second search is fired with that bare name, since a bare name is what retrieves a page best. Retrieval
budget, evidence pool, train-learned answer demos, style note, answer instructions, polisher and repair
trigger are byte-identical to the frontier, so the whole delta is attributable to how the missing page is
identified. No extra LLM call and no extra retrieval.
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


LOWER_JOIN = {
    "of", "the", "de", "la", "le", "von", "van", "der", "den", "du", "di", "da", "and", "for", "in", "on",
    "at", "el", "al", "ibn", "bin", "los", "las", "dos", "des",
}
TOKENISE = re.compile(r"[A-Z0-9][\w.&'’·-]*|[a-z]+")
WEAK_NAME = {
    "he", "she", "it", "they", "i", "you", "we", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "north", "south", "east", "west", "new", "however", "after",
    "before", "during", "while", "when", "although", "this", "that", "these", "those", "his", "her", "their",
}


def _strip_paren(title: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", title or "").strip()


def _cap_runs(text: str) -> list[list[tuple[str, int, int]]]:
    """Maximal runs of capitalised tokens, allowing lower-case joiners such as `of` inside a run."""
    runs: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    for m in TOKENISE.finditer(text or ""):
        w = m.group(0)
        if w[:1].isupper() or w[:1].isdigit():
            cur.append((w, m.start(), m.end()))
        elif w.lower() in LOWER_JOIN and cur:
            cur.append((w, m.start(), m.end()))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    trimmed = []
    for run in runs:
        while run and run[-1][0].islower() and run[-1][0] in LOWER_JOIN:
            run = run[:-1]
        while run and run[0][0].islower() and run[0][0] in LOWER_JOIN:
            run = run[1:]
        if run:
            trimmed.append(run)
    return trimmed


SEGMENT = re.compile(r"\s*\|\s*|[\n\r]+|(?<=[a-z0-9\)\]\"'])\s*[.!?;:]\s+")


def _name_mentions(text: str, max_words: int = 6) -> list[str]:
    """Every plausible page name occurring in `text`: contiguous pieces of its capitalised runs.

    The text is cut at title separators and at sentence ends first, so a name can never be assembled
    across two unrelated sentences (an abbreviated forename such as `P. B.` is not a sentence end).
    """
    return [m for seg in SEGMENT.split(text or "") if seg for m in _run_mentions(seg, max_words)]


def _run_mentions(text: str, max_words: int = 6) -> list[str]:
    out: list[str] = []
    for run in _cap_runs(text):
        n = len(run)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_words) + 1):
                piece = run[i:j]
                if (piece[0][0].islower() and piece[0][0] in LOWER_JOIN) or (
                    piece[-1][0].islower() and piece[-1][0] in LOWER_JOIN
                ):
                    continue
                span = text[piece[0][1] : piece[-1][2]].strip()
                key = _norm(span)
                if len(span) < 3 or not key:
                    continue
                toks = key.split()
                if all(t in WEAK_NAME or t in STOP for t in toks):
                    continue
                if len(toks) == 1 and (key.isdigit() or len(key) < 4):
                    continue
                out.append(span)
    return out


def _mention_pool(question: str, passages: list[str], titles: list[str]) -> list[str]:
    """Names available to aim the second search: mentions in the retrieved text and in the question,
    minus the pages already retrieved. Longest first, so a full name outranks its own prefix."""
    covered = {_norm(t) for t in titles} | {_norm(_strip_paren(t)) for t in titles}
    seen, pool = set(), []
    for span in _name_mentions(question) + [m for p in passages for m in _name_mentions(p)]:
        key = _norm(span)
        if key in seen or key in covered:
            continue
        seen.add(key)
        pool.append(span)
    pool.sort(key=lambda s: -len(_norm(s).split()))
    return pool


def _snap(proposal: str, pool: list[str]) -> str:
    """Complete a partially copied name, or '' when nothing in the text is the same name.

    Only an EXTENSION of the proposal counts (every word of the proposal is in the candidate), so this
    repairs a name that was copied short -- a surname without its forenames, a work without its article --
    and can never substitute a different entity for the one the call meant.
    """
    want = [t for t in _norm(proposal).split() if t not in STOP]
    if not want:
        return ""
    best, best_len = "", 0
    for cand in pool:
        have = [t for t in _norm(cand).split() if t not in STOP]
        if not have or len(have) <= len(want):
            continue
        if not all(t in have for t in want):
            continue
        if len(have) > 2 * len(want) + 1:  # a whole clause swallowing the name is not the name
            continue
        if len(have) > best_len:
            best, best_len = cand, len(have)
    return best


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
different encyclopaedia pages. One of them is usually already in `passages` (their titles are in
`retrieved_titles`); one is still missing, and your job is to identify it by reading, not by guessing.

Produce:
- `facts`: the sentences from `passages` that bear on the question, copied word for word, each on its own
  line and prefixed with the page title. Write `none` if nothing in the passages is relevant.
- `answer_type`: one of yesno, person, organisation, location, date, number, title, other.
- `route`: how the missing page is reached.
  * `link` -- the question names one thing, one of the retrieved abstracts is about that thing, and the
    missing page is something that abstract mentions (the person behind a work, the work behind a person,
    the place, team, album, film, war or organisation the question then asks about).
  * `named` -- the question names two things and only one of them has been retrieved, so the missing page
    is the other one, exactly as the question writes it.
  * `recalled` -- neither holds: nothing on screen names the page the question needs, so you have to name
    it yourself.
- `anchor_title`: for `link`, the title from `retrieved_titles` whose abstract you are reading the missing
  page out of; for `named`, write `question`.
- `missing_page`: the name of the missing page, COPIED CHARACTER FOR CHARACTER from the source you just
  named in `anchor_title` -- from that abstract for `link`, from the question for `named`. Rules:
  * copy a name that is actually written there; do not translate it, expand it or abbreviate it;
  * copying is strongly preferred, but it is not always possible: if the page the question needs is simply
    not named anywhere in `passages` or in the question, then name it from what you know instead, as an
    encyclopaedia would title it, and write `recalled` in `route`;
  * copy the longest form the source gives (all given forenames, the full title of a work), and nothing
    else: no year you added yourself, no category word, no bracket, no quotation marks, no `and`;
  * it names ONE page. If two things are mentioned, copy the one the question still needs;
  * it is a NAME, not a description. A phrase that describes the page without naming it -- a role, a
    nationality plus an occupation, a relative clause, anything containing lower-case words other than a
    name's own joiners such as `of` or `van` -- is useless here, because the page you want is titled with
    its name. If the question only describes the page ("the comedian who ...", "the company that ..."),
    work out from the abstracts and from what you know WHICH page that description picks out and write
    that page's name;
  * it is never a page already in `retrieved_titles`, never a sentence, never a question;
  * it is never empty and never `none`, `n/a` or `unknown`. If the retrieved abstracts really are all
    about other subjects, copy from the question the main thing it asks about instead."""


ANSWER_INSTRUCTIONS = """Answer `question` from `passages`, which are encyclopaedia abstracts written as
`Title | text`, and from `facts`, which were copied out of the first search.

`solved_examples` are questions of the SAME FORM as this one that have already been marked correct, each
shown with its reference answer. They are your specification for what an answer looks like: how long it is,
whether it keeps a qualifier such as a state, a country or a year, whether it is a proper name written in
full or a lower-case common noun, and whether it is simply yes or no. `style_note` summarises them.
Use them for FORM ONLY -- their subject matter is unrelated to this question and never supplies its answer.

Produce `supporting_titles`: the titles of the one or two passages that actually justify the answer.
Then produce `answer_type`, and then `answer`.

Rules for `answer`:
- It must be the span of the passages that answers the question, pitched at the same level of detail as the
  reference answers in `solved_examples`. Do not compress below that level and do not pad above it.
- If answers of this form keep a trailing place or date qualifier, keep it; if they are bare, stay bare.
- `style_examples` and `style_note` never license dropping part of what the question actually asks for: if
  the question asks for a date, give the date and not only its year; if it asks for a full title, give the
  full title. Form follows the examples, content follows the question.
- If the reference answers of this form are people's names written out in full, write the name the way the
  passage's own opening sentence writes it rather than the short form you would say out loud.
- When the answer is two things, join them the way the examples join them.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- Copy the span verbatim from the passages, keeping the original spelling, capitalisation, abbreviations
  and units, e.g. write a distance the way the passage writes it rather than expanding or converting it.
- A question that can be answered with yes or no must be answered with exactly `yes` or exactly `no`.
- A question that asks which of two named things has some property must be answered with just that one
  name; do not name both and do not say why.
- If the passages are incomplete, combine what they do say with the most likely reading and still commit
  to a single span; never answer that the information is missing."""

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
            outputs=["facts", "answer_type", "route", "anchor_title", "missing_page"],
            instructions=TRIAGE_INSTRUCTIONS,
            descs={
                "retrieved_titles": "pages already covered by the first search",
                "facts": "verbatim sentences from the passages, one per line, each prefixed by its page title",
                "answer_type": "yesno, person, organisation, location, date, number, title or other",
                "route": "link (the missing page is mentioned in a retrieved abstract) or named (the missing "
                         "page is the second thing the question names)",
                "anchor_title": "the retrieved title you are reading the missing page out of, or `question`",
                "missing_page": "the missing page's name, copied verbatim from that source",
            },
        )
        self.answerer = Module(
            inputs=["question", "solved_examples", "style_note", "facts", "passages"],
            outputs=["supporting_titles", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "solved_examples": "already-solved questions of the same form with their reference answers; a "
                                   "specification of answer form only, never a source of content",
                "style_note": "what the reference answers of this question form look like",
                "supporting_titles": "titles of the passages that justify the answer",
                "answer": "the final answer as a bare span, no sentence, no punctuation around it",
            },
        )
        self.repair = Module(
            inputs=["question", "passages", "draft"],
            outputs=["answer"],
            instructions=REPAIR_INSTRUCTIONS,
            chain_of_thought=False,
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

    def _resolve(self, proposal: str, question: str, docs: list[dict[str, Any]], ctx: RunContext) -> str:
        """Turn the proposed missing page into a query, but only if the text really names it.

        Order: accept a proposal that occurs verbatim in the retrieved abstracts or in the question; else
        snap it onto the closest name that does occur there; else fall back to the frontier's heuristics.
        """
        titles = [d["title"] for d in docs]
        passages = [d["text"] for d in docs]
        name = self._clean_query(proposal)
        name = re.sub(r"(?i)^(the page|page|article|title)\s*:?\s*", "", name).strip()
        name = re.sub(r"\s*\((?:[^()]{0,60})\)\s*$", "", name).strip()
        pool = _mention_pool(question, passages, titles)
        covered = {_norm(t) for t in titles} | {_norm(_strip_paren(t)) for t in titles}
        key = _norm(name)
        haystack = _norm(" ".join(passages) + " " + question)

        usable = bool(key) and key not in covered and len(key.split()) <= 8 and not NULL_QUERY.match(key)
        if usable:
            # a short copy gets completed from the text; anything else is searched for as given, whether the
            # text names it or not -- a page this call knows about but the abstracts never mention is exactly
            # the 20% of cases that copying cannot reach
            snapped = _snap(name, pool)
            if snapped:
                ctx.note("hop2_snapped", proposed=name[:120], used=snapped[:120])
                return snapped
            ctx.note("hop2_name", used=name[:120], in_text=key in haystack)
            return name

        fallback = self._uncovered_span(question, titles)
        if not fallback and pool:
            fallback = pool[0]
        if not fallback and titles:
            anchor = titles[0]
            focus = [w for w in question.split()
                     if w.lower().strip(",.?\"'") not in STOP and _norm(w) not in _norm(anchor)]
            fallback = (anchor + " " + " ".join(focus[:4])).strip()
        if not fallback:
            fallback = question
        ctx.note("hop2_fallback", proposed=name[:120], used=fallback[:120])
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

        query_2 = self._resolve(t.get("missing_page", ""), question, hop1, ctx)
        ctx.note("hop2_plan", route=t.get("route", "")[:40], anchor=t.get("anchor_title", "")[:80])
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

        out = self.answerer(
            ctx.llm,
            tag="answer",
            question=question,
            solved_examples=shown or ["none available"],
            style_note=note,
            facts=facts or "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        atype = out.get("answer_type", "").strip() or answer_type
        answer = polish(raw, atype, question)

        # "too long" is measured against the reference answers of this question form, not a constant
        budget = max(len(d["a"].split()) for d in demos) + 2 if demos else 6
        if _needs_repair(answer, atype, question, budget):
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

        return {"answer": answer, "raw_answer": raw, "query_2": query_2, "answer_type": atype}
