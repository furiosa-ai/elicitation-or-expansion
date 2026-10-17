"""Answers as verified verbatim spans of a quoted evidence sentence.

Trace analysis of the frontier program isolates where its remaining loss actually sits. On the search-set
items where BOTH required pages were retrieved and the answer was still wrong, the gold string is present
*word for word* in the very passages the answering call already had in front of it in 36 of 37 cases. The
evidence is not the bottleneck; the selection of the span is. Two specific habits cause it:

  * the answerer is told to produce "the shortest span", so when the supporting sentence answers with a
    qualified phrase it hands back a compressed version of that phrase instead of the phrase;
  * nothing ties the emitted string to the text. The model routinely returns the form of a name it knows
    rather than the form the passage uses, and no downstream step can tell the two apart.

Mechanism of this harness (still 2 retrievals, k=7, at most 4 LLM calls): make the copy explicit and then
check it.

  * the answering call must first name the passage it is using and quote, character for character, the ONE
    sentence from that passage that answers the question. Only then may it write the answer, and the answer
    must be a contiguous piece of that quoted sentence, worded as the sentence words it -- not shortened to
    a bare head noun, not swapped for a more familiar form of the same name;
  * a deterministic grounding check then verifies the claim the model just made: the answer has to occur as
    a contiguous token span of the retrieved passages. Yes/no answers are exempt;
  * an answer that fails the grounding check is exactly the case the frontier could not detect, so the one
    spare call is re-aimed: instead of firing on verbosity alone, the repair call fires on ungrounded or
    malformed answers and is shown the quote it must cut the span out of;
  * the polisher is applied in full only to ungrounded answers. A span that is verifiably in the evidence
    keeps its qualifiers, because the trimming heuristics exist to rescue prose, and a verified copy is not
    prose.

Retrieval, the coverage hop and the query gate are byte-identical to the frontier, so the delta isolates the
answer-grounding mechanism.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
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


def _needs_repair(answer: str, answer_type: str, question: str) -> bool:
    if not answer.strip():
        return True
    if HEDGES.search(answer):
        return True
    if len(answer.split()) > 6:
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


def _tokens(s: str) -> list[str]:
    return _norm(s).split()


def is_grounded(answer: str, pool: list[str]) -> bool:
    """True when `answer` occurs as a contiguous token span of some retrieved passage."""
    at = _tokens(answer)
    if not at:
        return False
    if len(at) == 1 and at[0] in ("yes", "no"):
        return True
    needle = " " + " ".join(at) + " "
    return any(needle in " " + " ".join(_tokens(p)) + " " for p in pool)


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

Work in this order and do not skip a step.

1. `evidence_title`: the title of the one passage that answers the question.
2. `evidence_quote`: the single sentence from that passage which contains the answer, copied character for
   character. Do not paraphrase it, do not shorten it, do not fix its punctuation. If no sentence contains
   the answer outright, quote the sentence that comes closest.
3. `answer_type`: one of yesno, person, organisation, location, date, number, title, other.
4. `answer`: a contiguous piece of `evidence_quote` -- a run of words you could highlight inside that
   sentence without leaving it out of order.

Rules for `answer`:
- Word it exactly as the quoted sentence words it. Keep the spelling, capitalisation, abbreviations, units
  and qualifiers the sentence uses. If the sentence names a person, a place or a work with a qualifier
  attached, keep the qualifier; if it names it plainly, leave it plain. Never substitute the version of a
  name you happen to know for the version in front of you.
- Take the whole phrase that answers the question, not its head word, and nothing beyond it. Do not add a
  clause explaining it and do not restate the question.
- Never write a sentence, never explain, never add a closing period.
- A question that can be answered with yes or no must be answered with exactly `yes` or exactly `no`, and
  only then may the answer fall outside the quote.
- A question that asks which of two named things has some property must be answered with just that one
  name, spelled as in the question; do not name both and do not say why.
- If the passages are incomplete, combine what they do say with the most likely reading and still commit to
  a single span; never answer that the information is missing."""

REPAIR_INSTRUCTIONS = """`draft` is an attempt at answering `question` that could not be traced back to the
evidence: it is prose, it is evasive, or its wording appears nowhere in `passages`.

Return only `answer`: a contiguous run of words copied out of `passages`, worded exactly as the passage words
it, normally the phrase that answers the question and nothing more. No sentence, no explanation, no final
period. Answer a yes/no question with exactly `yes` or exactly `no`. When the question offers two named
alternatives, return exactly one of them, spelled as in the question. Never reply that the information is
missing -- commit to the best span in the evidence."""


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
            inputs=["question", "facts", "passages"],
            outputs=["evidence_title", "evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence_title": "title of the passage the answer is copied from",
                "evidence_quote": "the sentence containing the answer, copied verbatim from that passage",
                "answer": "a contiguous run of words taken from evidence_quote, worded as the quote words it",
            },
        )
        self.repair = Module(
            inputs=["question", "passages", "quote", "draft"],
            outputs=["answer"],
            instructions=REPAIR_INSTRUCTIONS,
            descs={"quote": "the sentence the previous attempt claimed to be copying from"},
            chain_of_thought=False,
        )

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
    def _repair_hint(self, question: str) -> str:
        opts = _two_options(question)
        if opts:
            return "The answer must be exactly one of these, copied as written: " + " / ".join(opts)
        return ""

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
        pool, seen = [], set()
        for d in hop2 + hop1:
            key = d["title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(d["text"])

        out = self.answerer(ctx.llm, tag="answer", question=question, facts=facts or "none", passages=pool)
        raw = out.get("answer", "")
        quote = (out.get("evidence_quote", "") or "").strip()
        atype = out.get("answer_type", "").strip() or answer_type

        # a span the evidence actually contains keeps its wording; only prose gets trimmed
        # a yes/no question is never answered by a quoted span, so it never takes the grounded path
        yesno = atype.lower().startswith("yes") or YESNO_Q.match(question.strip()) is not None
        grounded = (not yesno) and is_grounded(raw.strip(), pool)
        answer = raw.strip().strip("\"\u201c\u201d\u2018\u2019 ").rstrip(".;:, ") if grounded else polish(raw, atype, question)
        if grounded:
            answer = polish(answer, atype, question) if len(answer.split()) > 12 else answer
        ctx.note("grounding", grounded=grounded, answer=answer[:120])

        if (not yesno and not is_grounded(answer, pool)) or _needs_repair(answer, atype, question):
            hint = self._repair_hint(question)
            fixed = self.repair(
                ctx.llm,
                tag="repair",
                question=(question + ("\n" + hint if hint else "")),
                passages=pool,
                quote=quote[:600] or "none",
                draft=raw[:600] or answer,
            )
            cand_raw = fixed.get("answer", "")
            cand = cand_raw.strip().strip("\"\u201c\u201d\u2018\u2019 ").rstrip(".;:, ")
            if not is_grounded(cand, pool) or len(cand.split()) > 12:
                cand = polish(cand_raw, atype, question)
            ctx.note("repair", before=answer[:120], after=cand[:120])
            if cand and not HEDGES.search(cand):
                answer = cand

        return {"answer": answer, "raw_answer": raw, "query_2": query_2, "answer_type": atype,
                "evidence_quote": quote[:300], "grounded": grounded}
