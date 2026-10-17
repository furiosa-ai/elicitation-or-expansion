"""Question-structure routing of the two retrieval slots.

The frontier program spends its retrieval budget the same way on every item: slot 1 always goes to the raw
question, and slot 2 goes to a page name that an LLM hop has to guess from slot 1's passages. Trace analysis
shows that policy is wrong for a whole family of questions. On the search set, every single item whose gold
pages were only half covered and whose question *weighs two named things against each other* names BOTH of
those pages literally in the question text. Nothing has to be inferred: the retrieval targets are sitting in
the question. The single raw-question query splits the BM25 mass across the two names and typically brings
back only the more common one, and the guessing hop then has to recover a page whose name was never in doubt.

Mechanism of this harness (still 2 retrievals, k=7, at most 4 LLM calls):

  * a deterministic parse of the question -- no LLM call -- extracts the page-like names it mentions and asks
    whether the question is *two-anchored*: an `A or B` choice, a `both A and B` question, or two named things
    joined by a comparative or a "have in common / share" relation;
  * two-anchored questions take the PARALLEL plan: slot 1 is pinned to anchor A and slot 2 to anchor B, both
    fired straight away. There is no planning hop at all, because there is nothing to plan -- which also
    returns that LLM call to the answer stage;
  * every other question keeps the frontier's SEQUENTIAL plan byte for byte (raw question, coverage hop,
    gated query), because there the second page genuinely is unnamed and has to be inferred;
  * both routes converge on the same deduplicated verbatim evidence pool, the same typed answer contract and
    the same deterministic polisher, so the two routes differ only in how the retrieval slots were spent.

The router is falsifiable on its own terms: it must raise avg gold title recall above 0.8083 and it must not
change the answer layer at all.
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


# --------------------------------------------------------------------------------------------------
# deterministic question-structure parse: does the question pin BOTH pages by name?
# --------------------------------------------------------------------------------------------------
# capitalized runs that do NOT swallow a conjunction, so "X and Y" yields two anchors rather than one
CAP_ATOM = re.compile(r"[A-Z][\w.&'\u2019-]*(?:\s+(?:of|the|de|la|von|van|for|in)?\s*[A-Z0-9][\w.&'\u2019-]*)*")
# markers that mean the question is really about two named things at once
TWO_ANCHOR_MARK = re.compile(
    r"(?i)\b(both|either|in common|share[sd]?|respectively|same|more|less|fewer|earlier|later|older|younger|"
    r"larger|smaller|longer|shorter|higher|lower|greater|newer|bigger|first|closer|which of (?:the )?(?:two|these))\b"
)


def _atoms(text: str, drop_leading_cap: bool = False) -> list[str]:
    """Page-like names in `text`, conjunctions treated as separators."""
    out = [m.group(1).strip() for m in QUOTED.finditer(text)]
    body = text[1:] if (drop_leading_cap and text[:1].isupper()) else text
    for m in CAP_ATOM.finditer(body):
        span = m.group(0).strip(" ,.;:")
        if len(span) > 2 and _norm(span) not in STOP:
            out.append(span)
    seen, res = set(), []
    for s in out:
        key = _norm(s)
        if key and key not in seen:
            seen.add(key)
            res.append(s)
    return res


def _usable(span: str) -> bool:
    toks = [t for t in _norm(span).split() if t not in STOP]
    return bool(toks) and len(_norm(span)) >= 3 and len(_norm(span).split()) <= 8


CAP_RUN = r"(?:[\"\u201c\u2018'][^\"\u201d\u2019']{2,60}[\"\u201d\u2019']|[A-Z][\w.&'\u2019-]*(?:\s+(?:of|the|de|la|von|van|for|in|and|&)?\s*[A-Z0-9][\w.&'\u2019-]*)*)"
LEFT_ANCHOR = re.compile(CAP_RUN + r"[\s,;:]*$")
RIGHT_ANCHOR = re.compile(r"^[\s,;:]*(" + CAP_RUN + r")")
# an `and` only coordinates two retrieval targets when the question says it is about both of them
BOTH_MARK = re.compile(r"(?i)\b(both|in common|shares?|shared|the same|respectively|each of)\b")


def _strip_quotes(s: str) -> str:
    return s.strip().strip("\"\u201c\u201d\u2018\u2019' ").strip(" ,.;:")


def anchor_pair(question: str) -> list[str]:
    """The two pages a two-anchored question names, or [] when the question is not two-anchored.

    A question is two-anchored only when a coordinating conjunction stands BETWEEN two named things --
    `A or B` (an explicit choice) or `A and B` under a both/in-common/same marker. Requiring the anchors to
    flank the conjunction is what keeps ordinary bridge questions, which mention plenty of capitalised words
    in passing, out of this route.
    """
    for conj in re.finditer(r"(?i)\b(or|and)\b", question):
        word = conj.group(1).lower()
        if word == "and" and not BOTH_MARK.search(question):
            continue
        left = LEFT_ANCHOR.search(question[: conj.start()])
        right = RIGHT_ANCHOR.match(question[conj.end():])
        if not left or not right:
            continue
        a, b = _strip_quotes(left.group(0)), _strip_quotes(right.group(1))
        if not (_usable(a) and _usable(b)):
            continue
        na, nb = _norm(a), _norm(b)
        if not na or not nb or na == nb or na in nb or nb in na:
            continue
        return [a, b]
    return []

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

Produce `supporting_titles`: the titles of the one or two passages that actually justify the answer.
Then produce `answer_type`, and then `answer`.

Rules for `answer`:
- It must be the shortest span that answers the question: normally one to four words.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- Copy the span verbatim from the passages, keeping the original spelling, capitalisation, abbreviations
  and units, e.g. write a distance the way the passage writes it rather than expanding or converting it.
- A question that can be answered with yes or no must be answered with exactly `yes` or exactly `no`.
- A question that asks which of two named things has some property must be answered with just that one
  name, spelled as in the question; do not name both and do not say why.
- Drop leading articles, descriptors like "the film" or "the band", and parenthetical glosses.
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
            outputs=["supporting_titles", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
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

    def _pool(self, *hops: list[dict[str, Any]]) -> list[str]:
        pool, seen = [], set()
        for hop in hops:
            for d in hop:
                key = d["title"].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(d["text"])
        return pool

    def _finish(self, question: str, facts: str, pool: list[str], prior_type: str, route: str,
                ctx: RunContext, extra: dict[str, Any]) -> dict[str, Any]:
        out = self.answerer(ctx.llm, tag="answer", question=question, facts=facts or "none", passages=pool)
        raw = out.get("answer", "")
        atype = out.get("answer_type", "").strip() or prior_type
        answer = polish(raw, atype, question)

        if _needs_repair(answer, atype, question):
            hint = self._repair_hint(question)
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

        result = {"answer": answer, "raw_answer": raw, "answer_type": atype, "route": route}
        result.update(extra)
        return result

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # ---- route A: the question itself pins both pages -> spend both slots on them, in parallel
        pair = anchor_pair(question)
        if pair:
            ctx.note("route_parallel", anchor_a=pair[0][:80], anchor_b=pair[1][:80])
            hop_a = ctx.tools.retrieve_docs(pair[0], k=self.K)
            hop_b = ctx.tools.retrieve_docs(pair[1], k=self.K)
            pool = self._pool(hop_a, hop_b)
            return self._finish(question, "none", pool, "", "parallel", ctx,
                                {"query_1": pair[0], "query_2": pair[1]})

        # ---- route B: the second page is unnamed -> the frontier's sequential coverage plan
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
        pool = self._pool(hop2, hop1)

        return self._finish(question, facts, pool, answer_type, "sequential", ctx, {"query_2": query_2})
