"""Each lookup is made with an invented article opening, not with a name plus keywords.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%). Retrieval still decides the outcome:
0.817 accuracy when both reference articles reach the context, 0.51 with one, 0.125 with neither,
and in 40 of its 101 errors the reference span is nowhere in the pool. Two previous attempts moved
in the same direction and both lost ground -- one forced every entity in a query to be text copied
from the question or from a passage in hand (-3.7, recall fell 0.76 -> 0.708), the other compiled
queries from a bare name plus at most three qualifying words under a policy measured on the train
split (-5.7, recall 0.682). Both *narrowed* the query.

Reading the misses says narrowing is the wrong direction. The queries that fail are short relation
phrases -- "manufacturer of <work>", "replacement for <person> at <place>" -- three or four tokens
of which one is the only content word, aimed at an index that scores documents by term overlap with
an article *opening*. A relation phrase shares almost no vocabulary with the opening of the article
that answers it, and a bare ambiguous name shares its vocabulary with every namesake in the
encyclopedia. What the index rewards is text that reads like the target article's own first
sentence.

Mechanism under test (retrieval strategy; the hop structure, the call count, the evidence pool and
the whole answer contract are held byte-identical to the parent so the delta isolates it):

  PSEUDO-ABSTRACT QUERIES. No module writes a search query. Each module instead writes the opening
  sentence it expects the target article to have -- the name, then what kind of thing it is, then
  the type, field, place, date and affiliations an encyclopedia would print about it -- as a plain
  declarative of about twenty-five words. A deterministic compiler turns that sentence into the
  index query: it drops interrogatives and hedges, keeps the leading name so the name's term
  frequency stays highest, and caps the length. The generated sentence is a *lexical bridge*: its
  invented details cost nothing when wrong, because a term the true article does not contain simply
  scores zero, while every detail that is right adds a matching term the relation phrase never had.
  This widens the query surface where the two failed attempts narrowed it, which is precisely the
  hypothesis worth separating.

Budget: 3 LLM calls, 2 retrievals, k=7 -- inside the seed program's interaction depth.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PLAN_INSTRUCTIONS = """The question needs facts from two different encyclopedia articles. Plan both
lookups before anything is searched. You do NOT write search queries: you write, for each article,
the opening sentence you expect that article to have, and the search string is built from it.

Work out which two articles must be read. For a comparison question they are the two entities being
compared. For a bridge question the first is the entity the question names outright and the second
is the entity the question only describes, which cannot be identified until the first is read.

`name_a`: the title of the article for the lookup that is possible RIGHT NOW -- the subject the
question names outright. Spell it exactly as the question spells it. Never put a guessed identity
here: if the question does not name the subject, it is not the first lookup.
`name_b`: the title you expect the second article to have. If the question names that subject too
(a comparison), write it. If the question only describes it, write your best guess -- it will be
checked and replaced later if nothing supports it.

`opening_a` / `opening_b`: the first sentence of that article, written as an encyclopedia would
write it, about twenty-five words. Start with the name, then say what kind of thing it is, then
pack in the properties such an article always prints -- profession or genre or category, nationality
or place, the decade or year, the organisation, series, label or field it belongs to.
  - write a plain statement, never a question, never "I think", never "the article probably says".
  - do not describe the relation the question asks about ("the manufacturer of it", "the person who
    replaced him"); describe the *subject itself*, because that is the text the index compares
    against.
  - guessing is expected and is safe. A detail that turns out to be wrong contributes nothing; a
    detail that is right is a term the search would otherwise not have had. Write the sentence you
    would write if you had to fill the article in from memory.
  - for a name that many different subjects share, spend most of the sentence on what tells this one
    apart from the others.
  - the one thing that is NOT safe is inventing an identity. If you do not know which subject the
    question is describing, do not name a plausible one and write its article: describe only what
    the question itself says about it, in the question's own words plus the category it belongs to.
    A sentence about the wrong subject buries the right article under twenty-five wrong terms.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other."""

RESOLVE_INSTRUCTIONS = """You have the passages returned for the first lookup. Do three things.

`facts`: copy out, verbatim, only the sentences or clauses from the passages that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If no passage is relevant, write "none".

`name_b`: the title of the second and final article to read. If the question described this subject
indirectly and the passages have now named it, write that name. Otherwise write the other entity the
question names. It must be a different subject from the first lookup, whose expected article opening
is given as `previous_opening`.

`opening_b`: the first sentence you expect that second article to have, about twenty-five words,
written the same way -- start with the name, say what kind of thing it is, then its category, place,
period, field and affiliations. Use everything the passages just told you: a date, a nationality, an
organisation or a series read out of them belongs in this sentence, because those are terms the
target article is likely to share. Describe the subject, never the relation the question asks about,
and never write a question."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence: `facts` are the sentences already
extracted, `passages` are the raw article openings retrieved for this question.

First quote the single piece of evidence that settles the question, then declare the shape of the
answer, then give the answer itself.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer` is graded by exact string match against a short reference span, so it must be the bare
answer and nothing else:
- Never a sentence. Never restate the question. No explanation, no reason, no "because", no
  trailing clause, no second sentence, no markdown, no quotation marks, no final period.
- If `answer_type` is yes_no, `answer` must be exactly `yes` or `no` -- one word, nothing after it.
  Questions of the form "are both ...", "do both ...", "is X the same as Y", "can both ..." are
  yes_no questions, even when the interesting content is in the reasoning.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly as the question spells it, and nothing else.
- Otherwise copy the answer span verbatim from the evidence, in the shortest form that fully
  names it: just the name, the title, the place, the date or the number.
- Do not add appositives, descriptors, units of explanation or parentheses that the reference
  span would not contain.
- If the evidence is incomplete, still give the single most likely bare answer. Never answer
  "unknown", "not stated" or "cannot be determined"."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "name_a": "the title you expect the first article to have",
    "name_b": "the title you expect the second article to have",
    "opening_a": "the first sentence you expect the first article to have, about 25 words",
    "opening_b": "the first sentence you expect the second article to have, about 25 words",
    "previous_opening": "the expected opening already used for the first lookup",
    "planned_name_b": "the second subject as planned before searching, to revise if needed",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"

# tokens that carry no matching signal against an article opening
_INTERROGATIVE = re.compile(
    r"^(who|whom|whose|which|what|where|when|why|how|is|are|was|were|do|does|did|can|could|would|should|"
    r"probably|perhaps|maybe|likely|i|think|guess|article|wikipedia|page|the|a|an)$",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(i think|i believe|probably|presumably|it is likely that|the article (probably )?(says|states)|"
    r"this article is about|based on|according to my)\b",
    re.IGNORECASE,
)
_MAX_QUERY_WORDS = 42
_STOPWORD = re.compile(
    r"^(the|a|an|of|in|on|at|to|for|and|or|but|by|with|from|as|is|are|was|were|be|been|that|this|"
    r"these|those|it|its|his|her|their|there|which|who|whom|whose|what|where|when|why|how|did|do|"
    r"does|has|have|had|also|both|more|most|than|other|same|some|any|first|last)$",
    re.IGNORECASE,
)


def anchor_terms(question: str, limit: int = 12) -> str:
    """The question's own content words -- the only description of a subject nobody has named yet."""
    words = [w for w in re.split(r"[^\w'&+.-]+", str(question or "")) if w]
    kept = [w for w in words if not _STOPWORD.match(w)]
    return " ".join(kept[:limit])


def _mentioned(name: str, haystacks: list[str]) -> bool:
    """Is this identity supported by text we did not invent (the question, or a retrieved passage)?"""
    n = normalize_text(name)
    if not n:
        return False
    if len(n.split()) > 1:
        n_alt = " ".join(n.split()[:-1])
    else:
        n_alt = n
    for h in haystacks:
        nh = f" {normalize_text(h)} "
        if f" {n} " in nh or (len(n_alt.split()) >= 1 and len(n.split()) > 1 and f" {n_alt} " in nh):
            return True
    return False


def _clean(s: Any) -> str:
    text = str(s or "").strip()
    if not text:
        return ""
    text = text.split("\n")[0]
    text = _PREFIX.sub("", text)
    return text.strip(_STRIP_CHARS).strip()


def clean_name(text: Any) -> str:
    """A bare title: no leading article, no quotes, no trailing description after a dash."""
    name = str(text or "").strip().split("\n")[0]
    name = re.sub(r"^\s*(name|title)\s*:\s*", "", name, flags=re.IGNORECASE)
    name = re.split(r"\s+[-–—]\s+", name)[0]
    name = name.strip(_STRIP_CHARS).strip()
    return name if 0 < len(name.split()) <= 12 else name[:120].strip()


def compile_query(name: str, opening: Any, fallback: str) -> str:
    """Turn an invented article opening into a sparse-index query.

    The name leads so its term frequency stays highest; interrogatives, hedges and stopwords are
    dropped because they match every article equally; the whole thing is capped so one runaway
    sentence cannot dilute the entity terms.
    """
    text = str(opening or "").strip()
    text = _HEDGE.sub(" ", text)
    text = text.replace("\n", " ")
    # a sentence that came back as a question carries no descriptive terms -- drop it
    if "?" in text:
        text = text.split("?")[0]
    words = [w for w in re.split(r"[^\w'&+.-]+", text) if w]
    kept = [w for w in words if not _INTERROGATIVE.match(w)]
    body = " ".join(kept[:_MAX_QUERY_WORDS])
    head = clean_name(name)
    if head:
        query = f"{head} {body}".strip()
    else:
        query = body
    query = re.sub(r"\s+", " ", query).strip(_STRIP_CHARS).strip()
    return query if len(query) >= 3 else str(fallback or "").strip()


def _question_options(question: str) -> list[str]:
    if not _CLOSED_LIST.search(question or ""):
        return []
    q = str(question).strip().rstrip("?")
    sep = r",|\bor\b|\bversus\b|\bvs\.?\b"
    if re.match(r"\s*between\b", q, re.IGNORECASE):
        sep += r"|\bbetween\b|\band\b"
    opts: list[str] = []
    for chunk in re.split(sep, q, flags=re.IGNORECASE):
        cand = re.sub(r"^\s*(between|which|who|what|whom|whose)\b\s*", "", chunk, flags=re.IGNORECASE)
        cand = cand.strip(_STRIP_CHARS).strip()
        if 1 <= len(cand.split()) <= 8 and re.search(r"[A-Z0-9]", cand):
            opts.append(cand)
    return opts


def _echo_option(ans: str, question: str) -> str:
    norm_ans = normalize_text(ans)
    if not norm_ans:
        return ans
    prefix_hits, inside_hits = [], []
    for opt in _question_options(question):
        n = normalize_text(opt)
        if not n or n == norm_ans:
            return ans
        if len(n.split()) >= len(norm_ans.split()):
            continue
        if norm_ans.startswith(n + " "):
            prefix_hits.append(opt)
        elif f" {n} " in f" {norm_ans} ":
            inside_hits.append(opt)
    hits = prefix_hits or ([] if len(inside_hits) != 1 else inside_hits)
    return max(hits, key=lambda o: len(normalize_text(o))) if hits else ans


def project_answer(raw_answer: str, answer_type: str, question: str) -> str:
    ans = _clean(raw_answer)
    atype = str(answer_type or "").strip().lower()
    yn = _YESNO.match(ans)
    if yn and (atype.startswith("yes") or len(ans.split()) > 1):
        return yn.group(1).lower()
    if atype.startswith("yes") and normalize_text(ans) in ("yes", "no"):
        return normalize_text(ans)
    if len(ans.split()) > 4:
        parts = [p for p in _SENT.split(ans) if p.strip()]
        if len(parts) > 1 and len(parts[0].split()) >= 3:
            ans = _clean(parts[0])
    if len(ans.split()) > 1:
        ans = _echo_option(ans, question)
    return _clean(ans) or _clean(raw_answer)


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question"],
            outputs=["answer_type", "name_a", "opening_a", "name_b", "opening_b"],
            instructions=PLAN_INSTRUCTIONS,
            descs=DESCS,
        )
        self.resolve = Module(
            inputs=["question", "planned_name_b", "previous_opening", "passages"],
            outputs=["facts", "name_b", "opening_b"],
            instructions=RESOLVE_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )

    @staticmethod
    def _pool(*doc_lists: list[dict[str, Any]]) -> list[str]:
        """Deduplicated raw evidence pool, first occurrence wins, order preserved."""
        seen: set[str] = set()
        out: list[str] = []
        for docs in doc_lists:
            for d in docs:
                key = str(d.get("title", "")).strip().lower()
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(str(d.get("text", "")).strip())
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # 1. plan both lookups as expected article openings, before touching the index
        plan = self.plan(ctx.llm, tag="plan", question=question)
        name_a = clean_name(plan.get("name_a"))
        # a pseudo-abstract is only safe for a subject somebody has actually named: an invented
        # identity buries the right article under twenty-five wrong terms. Fall back to the
        # question's own words as the anchor when the first subject was not named in the question.
        if _mentioned(name_a, [question]):
            query_a = compile_query(name_a, plan.get("opening_a"), question)
        else:
            query_a = compile_query(anchor_terms(question), plan.get("opening_a"), question)
            ctx.note("ungrounded first subject, anchored on the question", name_a=name_a[:120])
        planned_name_b = clean_name(plan.get("name_b"))
        planned_query_b = compile_query(planned_name_b, plan.get("opening_b"), "")

        # 2. first hop
        docs_a = ctx.tools.retrieve_docs(query_a, k=self.K)

        # 3. extract verbatim facts and rewrite the expected opening of the second article
        res = self.resolve(
            ctx.llm,
            tag="resolve",
            question=question,
            planned_name_b=planned_name_b or "not named yet",
            previous_opening=query_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(res.get("facts", "")).strip()
        name_b = clean_name(res.get("name_b"))
        support = [question, facts] + [d["text"] for d in docs_a]
        if _mentioned(name_b, support):
            next_query = compile_query(name_b, res.get("opening_b"), planned_query_b or question)
        else:
            next_query = compile_query(anchor_terms(question), res.get("opening_b"), planned_query_b or question)
            ctx.note("ungrounded second subject, anchored on the question", name_b=name_b[:120])
        if normalize_text(next_query) == normalize_text(query_a):
            next_query = planned_query_b or question  # never spend the second hop on the same query

        # 4. second hop
        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        # 5. answer over the deduplicated raw pool
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("pseudo-abstract queries", query_a=query_a[:300], next_query=next_query[:300])
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": next_query,
            "facts": facts,
        }
