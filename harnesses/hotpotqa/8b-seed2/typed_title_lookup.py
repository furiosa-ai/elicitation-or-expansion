"""Resolve the second hop as a typed, disambiguated page title rather than as a topic string.

The frontier system already turned the second hop into a coverage step: the planner is shown the titles
already retrieved and must name the one page still missing, as a short noun phrase. Reading the 96
search-set items where a reference page was still never retrieved shows that the plan is frequently
*correct* and the query still fails:

  * on 21 of them the emitted query names the missing page exactly -- and the page still does not come back
    in the top 7. These are short, generic, heavily-shared names. A two-word query of common words matches
    millions of abstracts, and the one page actually titled that way is not what a bag-of-words scorer ranks
    first.
  * on 8 more the query is the right name in the wrong form: the corpus titles that name with a
    disambiguating parenthesis -- a year, a category, a profession, a nationality -- and the bare name loses
    to the unrelated page that owns the undisambiguated title.
  * and when a question weighs two named things against each other, the planner sometimes emits *both* names
    joined by "and", so the query names a page that does not exist.

The corpus unit is `Title | abstract`, so the fix is to make the query look like the title it is hunting.
This harness replaces the free-text `next_query` with a structured page descriptor -- the bare name, the
category word that would sit in that page's disambiguating parenthesis, and one rare cue term (a year, a
nationality, a domain word) that the page's own opening sentence would contain -- and then composes the
query deterministically in title-first order: name, then category, then cue. Every added term is a term the
target page contains, so it lifts the target rather than diluting it, while the unrelated page that shares
the bare name matches none of them. The composer also splits a descriptor that names two pages and keeps the
one not yet covered, and drops category or cue words the name already contains.

The answer layer, the evidence pool, the polisher, the repair trigger and the first hop are byte-identical to
the frontier, and the call and retrieval counts are unchanged, so the delta isolates query formulation.
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
META = {
    "date", "dates", "name", "names", "nationality", "release", "released", "premiered", "premiere",
    "born", "birth", "died", "death", "members", "member", "related", "relation", "type", "kind",
    "number", "year", "years", "age", "older", "younger", "common", "unknown", "none", "information",
    "answer", "question", "page", "article", "value", "amount", "occupation", "profession",
}
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
# prompts
# --------------------------------------------------------------------------------------------------
TRIAGE_INSTRUCTIONS = """You are running the first of two searches for a question that needs facts from two
different encyclopaedia pages.

`retrieved_titles` lists the pages already retrieved, and `passages` are their abstracts. The search index
stores each page as its title followed by its opening text, so a search finds a page when the words you use
are words that page itself uses.

Produce:
- `facts`: the sentences from `passages` that bear on the question, copied word for word, each on its own
  line and prefixed with the page title. Write `none` if nothing in the passages is relevant.
- `answer_type`: one of yesno, person, organisation, location, date, number, title, other.
- Then describe the ONE page that is still missing and is needed to finish the answer -- typically the entity
  the first page points to (a bridge), or, when the question weighs two named things against each other, the
  one of them that is not in `retrieved_titles` yet. Describe it with three separate fields:
  * `missing_page`: just its name, as that page would be titled. A bare noun phrase, at most six words, no
    sentence, no question mark, no restatement of the question, and never a page already in
    `retrieved_titles`. Name exactly ONE page -- never two names joined by "and" or a comma.
  * `missing_page_kind`: one or two words for what that page is about, the kind of word an encyclopaedia puts
    in brackets after a title to tell two same-named pages apart -- for example a form (film, album, song,
    novel, band, video game), a profession (playwright, footballer, politician, physicist), or a class of
    thing (river, county, plant, aircraft, company). Write the kind even when the name looks unambiguous, and
    write `none` only if you truly cannot tell.
  * `missing_page_cue`: one to three words that the missing page's own opening sentence almost certainly
    contains and that a page with the same name but a different subject would not -- a year, a nationality,
    a country, a genre or a field. Take them from `passages` or from the question. Write `none` if you have
    nothing specific.
  Never state the answer to the question in any of these fields. If the first search already seems to settle
  the question, use the second search to check it: describe the most specific *other* page the relevant
  passage points at -- the album, film, event, organisation or person the question depends on. If the
  passages were all irrelevant, describe the main subject of the question as an encyclopaedia would title
  its page. `missing_page` is never empty and never `none`."""

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
            outputs=["facts", "answer_type", "missing_page", "missing_page_kind", "missing_page_cue"],
            instructions=TRIAGE_INSTRUCTIONS,
            descs={
                "retrieved_titles": "pages already covered by the first search",
                "facts": "verbatim sentences from the passages, one per line, each prefixed by its page title",
                "answer_type": "yesno, person, organisation, location, date, number, title or other",
                "missing_page": "the name of the single page still missing, written as that page's title",
                "missing_page_kind": "one or two words for what kind of thing that page is, or none",
                "missing_page_cue": "one to three words that page's opening sentence would contain, or none",
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

    # -- compose the hop-2 query in title form -----------------------------------------------------
    def _pick_single_name(self, name: str, titles: list[str], question: str) -> str:
        """A descriptor that names two pages is split; keep the part not already retrieved."""
        parts = [p.strip() for p in re.split(r"(?i)\s+and\s+|\s*[,;/]\s*|\s+\bvs\.?\b\s+", name) if p.strip()]
        if len(parts) < 2:
            return name
        covered = " ".join(_norm(t) for t in titles)
        uncovered = [p for p in parts if _norm(p) and _norm(p) not in covered]
        pool = uncovered or parts
        # prefer the alternative the question actually names, longest first
        named = [p for p in pool if _norm(p) and _norm(p) in _norm(question)]
        return max(named or pool, key=lambda p: len(_norm(p)))

    def _compose(self, name: str, kind: str, cue: str, question: str, titles: list[str]) -> str:
        """name -> `name kind cue`, dropping every word the name already carries."""
        name = self._clean_query(name)
        name = self._pick_single_name(name, titles, question)
        # a parenthesis the model invented is a kind word in disguise: unwrap it
        name = re.sub(r"\s*[\(\[]([^)\]]{1,40})[\)\]]\s*$", r" \1", name).strip()
        words = name.split()[:6]
        while words and _norm(words[-1]) in META:
            words.pop()
        name = " ".join(words) or name.strip()
        have = set(_norm(name).split())

        def _extra(field: str, max_words: int) -> list[str]:
            f = self._clean_query(field)
            if not f or NULL_QUERY.match(f.strip()) or len(f.split()) > 6:
                return []
            words = []
            for w in _norm(f).split():
                if w in have or w in STOP or w in META or len(w) < 2:
                    continue
                have.add(w)
                words.append(w)
                if len(words) >= max_words:
                    break
            return words

        parts = [name] + _extra(kind, 2) + _extra(cue, 3)
        return " ".join(p for p in parts if p).strip()

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
            or len(q.split()) > 11
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

        composed = self._compose(
            t.get("missing_page", ""),
            t.get("missing_page_kind", ""),
            t.get("missing_page_cue", ""),
            question,
            titles1,
        )
        ctx.note(
            "hop2_descriptor",
            name=t.get("missing_page", "")[:80],
            kind=t.get("missing_page_kind", "")[:40],
            cue=t.get("missing_page_cue", "")[:40],
            composed=composed[:120],
        )
        query_2 = self._gate(composed, question, titles1, ctx)
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
        atype = out.get("answer_type", "").strip() or answer_type
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

        return {"answer": answer, "raw_answer": raw, "query_2": query_2, "answer_type": atype}
