"""The second search verifies a hypothesis instead of covering a page.

The frontier spends hop 2 on *coverage*: the first call is asked to name the one reference page not yet
retrieved, and the answering call then reads the union. That policy has been pushed twice -- once by
re-aiming the slot at an entity parsed out of the question, once by rewriting the query into a title-like
descriptor -- and neither moved accuracy, so the remaining retrieval loss is probably not a targeting-
surface problem. Reading the errors from the other end suggests why. On 57 of the frontier's 92 errors the
gold string is already inside the evidence the answering call was shown, and the ones it gets wrong are
questions with several modifiers where the system settles on a plausible entity and never tests it: it
answers a "who ... born in 1928" question with the relative named in the same sentence, and a "who ...
published with X" question with X's collaborator of the wrong discipline. The coverage plan cannot catch
this, because it chooses what to search for *before* there is a claim on the table, and the answering call
then sees the evidence that its own guess selected.

So this harness makes hop 2 a falsification step. The first call must commit: read hop 1, write the
`hypothesis` it would answer with right now, and write the ONE atomic proposition `check` that has to be
true of that hypothesis for it to be right -- the birth year, the language, the role, the date, whichever
modifier of the question the hop-1 passages do not already confirm. `next_query` then names the page whose
abstract would settle `check`, which for a bridge is the far side of the bridge and for a comparison is
the rival, so coverage is mostly preserved while the query is chosen for what it could disprove. The
answering call is shown the hypothesis and the check and must first record `check_result` -- confirmed,
contradicted or unsupported -- and is instructed that a contradicted hypothesis must be replaced by a
different span rather than defended. A contradicted or unsupported verdict, not verbosity, is what now
re-aims the spare repair call.

The first hop, the query gate, the evidence pool, the train-learned demos, the style note, the answer-form
rules and the polisher are unchanged, and no call or retrieval is added.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

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


def _one_line(s: str) -> str:
    """First non-empty line of a model field, stripped of markup and label prefixes."""
    s = re.sub(r"\[\[.*?\]\]", " ", s or "").replace("**", "").replace("`", "")
    for line in s.splitlines():
        if line.strip():
            s = line.strip()
            break
    s = re.sub(r"(?i)^(hypothesis|check|claim|answer)\s*:\s*", "", s).strip(" \"'")
    return "" if NULL_QUERY.match(s.strip()) else s


def _verdict(s: str) -> str:
    m = re.search(r"(?i)\b(confirmed|contradicted|unsupported)\b", s or "")
    return m.group(1).lower() if m else "unsupported"


# --------------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------------
TRIAGE_INSTRUCTIONS = """You are running the first of two searches for a question that needs facts from two
different encyclopaedia pages. The second search is not a second guess at the topic -- it is the test that
tells you whether your current reading is right. Set that test up.

`retrieved_titles` lists the pages already retrieved, and `passages` are their abstracts.

Produce, in order:
- `facts`: the sentences from `passages` that bear on the question, copied word for word, each on its own
  line and prefixed with the page title. Write `none` if nothing in the passages is relevant.
- `answer_type`: one of yesno, person, organisation, location, date, number, title, other.
- `hypothesis`: the answer you would give if you had to answer from `passages` alone -- a bare span, not a
  sentence. Commit even when the passages are thin; a wrong hypothesis that gets tested is worth more than
  no hypothesis. Write `none` only if the passages say nothing at all about the question.
- `check`: the ONE statement that has to be true for `hypothesis` to be the answer, written as a single
  short proposition about a named thing, e.g. `<name> was born in 1928`, `<film> is in Tamil`, `<person>
  directed <film>`, `<work> was released before <other work>`. Choose the requirement of the question that
  `passages` do NOT already settle -- a year, a nationality or language, a role or profession, an ordinal,
  a relationship to something else named in the question. Do not choose something the abstracts in front of
  you already confirm, and do not restate the whole question.
- `next_query`: the search query for the SECOND search: it must name the ONE page whose abstract would
  settle `check`. Follow these rules exactly:
  * for a bridge question that is the far side of the bridge (the entity the first page points at); for a
    question that weighs two named things against each other it is whichever of them is not yet in
    `retrieved_titles`; for a check about a person's dates or profession it is that person's own page;
  * write it the way that page would be titled: a bare noun phrase of at most six words, optionally with
    a short disambiguating word (for example a year, a nationality or a category);
  * it must NOT be a sentence, must NOT contain a question mark, must NOT restate the question, must NOT
    be a copy of `check`, and must NOT name a page that is already in `retrieved_titles`;
  * never state the answer in it;
  * `next_query` is never empty and is never `none`, `n/a` or `unknown`. If every page you would test is
    already retrieved, name instead the most specific *other* page the relevant passage points at -- the
    album, film, event, organisation or person the question depends on -- written as that page's title. If
    the passages were all irrelevant, name the main subject of the question the way an encyclopaedia would
    title its page."""

ANSWER_INSTRUCTIONS = """Answer `question` from `passages`, which are encyclopaedia abstracts written as
`Title | text`, and from `facts`, which were copied out of the first search.

`hypothesis` is the answer the first search suggested and `check` is the statement that had to be true for
it to hold; the second search was run to settle exactly that statement. Before anything else produce
`check_result`, one of:
- `confirmed` -- some passage states `check`, or states something that entails it;
- `contradicted` -- some passage states something incompatible with `check`;
- `unsupported` -- no passage speaks to `check` either way.
Then answer. If `check_result` is `contradicted`, `hypothesis` is out: find the span that does satisfy the
requirement in `check` and answer with that instead -- do not argue the hypothesis back in. If it is
`unsupported`, treat `hypothesis` as one option among the spans the passages offer and prefer whichever
one the passages actually support. Only a `confirmed` verdict lets you keep `hypothesis` unexamined.

`solved_examples` are questions of the SAME FORM as this one that have already been marked correct, each
shown with its reference answer. They are your specification for what an answer looks like: how long it is,
whether it keeps a qualifier such as a state, a country or a year, whether it is a proper name written in
full or a lower-case common noun, and whether it is simply yes or no. `style_note` summarises them.
Use them for FORM ONLY -- their subject matter is unrelated to this question and never supplies its answer.

Then produce `supporting_titles`: the titles of the one or two passages that actually justify the answer,
then `answer_type`, and then `answer`.

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


RESELECT_INSTRUCTIONS = """`rejected` is an answer to `question` that the passages contradict: `check` is
the requirement it failed. Return only `answer`: a different span, copied verbatim from `passages`, that
answers `question` and does satisfy that requirement. It must not be `rejected` or a rewording of it.
A bare span, normally one to four words -- no sentence, no explanation, no final period. Answer a yes/no
question with exactly `yes` or exactly `no`. When the question offers two named alternatives, return
exactly one of them, spelled as in the question. Never reply that the information is missing."""


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.triage = Module(
            inputs=["question", "retrieved_titles", "passages"],
            outputs=["facts", "answer_type", "hypothesis", "check", "next_query"],
            instructions=TRIAGE_INSTRUCTIONS,
            descs={
                "retrieved_titles": "pages already covered by the first search",
                "facts": "verbatim sentences from the passages, one per line, each prefixed by its page title",
                "answer_type": "yesno, person, organisation, location, date, number, title or other",
                "next_query": "a title-like noun phrase of at most six words naming the page still missing",
            },
        )
        self.answerer = Module(
            inputs=["question", "solved_examples", "style_note", "hypothesis", "check", "facts", "passages"],
            outputs=["check_result", "supporting_titles", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "solved_examples": "already-solved questions of the same form with their reference answers; a "
                                   "specification of answer form only, never a source of content",
                "style_note": "what the reference answers of this question form look like",
                "hypothesis": "the answer the first search suggested; it is on trial, not given",
                "check": "the proposition the second search was run to settle",
                "check_result": "confirmed, contradicted or unsupported",
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
        self.reselect = Module(
            inputs=["question", "check", "rejected", "passages"],
            outputs=["answer"],
            instructions=RESELECT_INSTRUCTIONS,
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
        hypothesis = _one_line(t.get("hypothesis", ""))
        check = _one_line(t.get("check", ""))

        query_2 = self._gate(t.get("next_query", ""), question, titles1, ctx)
        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        ctx.note("hypothesis_check", hypothesis=hypothesis[:120], check=check[:160], query_2=query_2[:120])

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

        out = self.answerer(
            ctx.llm,
            tag="answer",
            question=question,
            solved_examples=shown or ["none available"],
            style_note=note,
            hypothesis=hypothesis or "none",
            check=check or "none",
            facts=facts or "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        atype = out.get("answer_type", "").strip() or answer_type
        verdict = _verdict(out.get("check_result", ""))
        answer = polish(raw, atype, question)

        # a hypothesis that survived only because nothing tested it, and an answer that kept a
        # contradicted hypothesis anyway, are the two states the spare call is worth spending on
        stale = verdict == "contradicted" and hypothesis and normalize_text(answer) == normalize_text(hypothesis)
        budget = max(len(d["a"].split()) for d in demos) + 2 if demos else 6
        ctx.note("verdict", check_result=verdict, stale=bool(stale), answer=answer[:80])
        if stale:
            fixed = self.reselect(
                ctx.llm,
                tag="reselect",
                question=question,
                check=check or "none",
                rejected=hypothesis[:200],
                passages=pool,
            )
            candidate = polish(fixed.get("answer", ""), atype, question)
            ctx.note("reselect", before=answer[:120], after=candidate[:120])
            if candidate and not HEDGES.search(candidate) and normalize_text(candidate) != normalize_text(hypothesis):
                answer = candidate
        elif _needs_repair(answer, atype, question, budget):
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
                "hypothesis": hypothesis, "check": check, "check_result": verdict}
