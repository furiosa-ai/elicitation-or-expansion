"""Decomposition-first retrieval with an entity-anchored second query and a raw evidence pool.

The seed program retrieves with the *raw question* first and then asks for a "query" given the
question and a summary; the model very often hands back the question again, so the two hops land
on the same passages and the second retrieval buys almost nothing.

Mechanism under test (retrieval strategy + context construction; the answer contract is held
fixed against `typed_answer_contract` so the comparison isolates this change):
  1. plan   -- one call *before* any retrieval decomposes the question into the two things that
               must be looked up, as keyword queries anchored on a single entity each (never the
               question verbatim), and declares the answer shape.
  2. hop A  -- retrieve(plan.query_a)
  3. resolve-- extract verbatim facts from hop A, resolve the bridge entity, and emit the second
               query anchored on that resolved entity, explicitly forbidden to repeat query_a.
  4. hop B  -- retrieve(resolve.next_query)
  5. answer -- the answerer sees the *deduplicated raw passages* from both hops (titles kept)
               plus the extracted facts, instead of two free-text summaries, so nothing that the
               summariser dropped is lost.

Budget: 3 LLM calls, 2 retrievals, k=7 -- inside the seed program's interaction depth.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PLAN_INSTRUCTIONS = """The question needs facts from two different encyclopedia articles. Plan the
two lookups before searching.

Work out which two things must be looked up. For a comparison question they are the two entities
being compared. For a bridge question the first is the entity named in the question and the second
is the entity that the question describes only indirectly and that must be identified first.

`query_a` and `query_b` are keyword queries for a sparse full-text index over encyclopedia article
openings. Therefore:
- each query names exactly ONE entity, plus two or three disambiguating content words (its type,
  field, place or date) -- e.g. a person plus their profession, a work plus its medium.
- never copy the question verbatim, never keep question words like "which", "who", "both", "same",
  "more", "first"; a query mixing both entities retrieves articles about neither.
- prefer the proper name exactly as the question spells it.
`query_a` must be the lookup that is possible right now; if the second entity is not yet named,
make `query_b` the best guess from the description in the question.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other."""

RESOLVE_INSTRUCTIONS = """You have the passages returned for the first lookup. Do two things.

`facts`: copy out, verbatim, only the sentences or clauses from the passages that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If no passage is relevant, write "none".

`next_query`: the keyword query for the second and final lookup. Use what you just learned: if the
question described the second entity indirectly and the passages have now named it, search for that
name. Otherwise search for the other entity in the question. Rules:
- name exactly ONE entity plus two or three disambiguating content words;
- it must NOT be the question and must NOT repeat `previous_query`; the second lookup is wasted if
  it returns the same articles;
- no question words, no punctuation, just the search terms."""

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
    "query_a": "keyword query for the first lookup, one entity only",
    "query_b": "keyword query for the second lookup, one entity only",
    "previous_query": "the query already used for the first lookup",
    "planned_query_b": "the second lookup planned before searching, to revise if needed",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "next_query": "keyword query for the second and final lookup",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
# a sentence boundary only counts after a word/number, never after an initial ("Albert R. Broccoli")
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"


def _clean(s: Any) -> str:
    text = str(s or "").strip()
    if not text:
        return ""
    text = text.split("\n")[0]
    text = _PREFIX.sub("", text)
    return text.strip(_STRIP_CHARS).strip()


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


def _as_query(text: Any, fallback: str) -> str:
    """Keep a generated query usable as a sparse-index query."""
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    return q if len(q) >= 3 else fallback


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question"],
            outputs=["answer_type", "query_a", "query_b"],
            instructions=PLAN_INSTRUCTIONS,
            descs=DESCS,
        )
        self.resolve = Module(
            inputs=["question", "planned_query_b", "previous_query", "passages"],
            outputs=["facts", "next_query"],
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

        # 1. plan both lookups before touching the index
        plan = self.plan(ctx.llm, tag="plan", question=question)
        query_a = _as_query(plan.get("query_a"), question)
        planned_b = _as_query(plan.get("query_b"), "")

        # 2. first hop
        docs_a = ctx.tools.retrieve_docs(query_a, k=self.K)

        # 3. extract verbatim facts and anchor the second query on what we just learned
        res = self.resolve(
            ctx.llm,
            tag="resolve",
            question=question,
            planned_query_b=planned_b,
            previous_query=query_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(res.get("facts", "")).strip()
        next_query = _as_query(res.get("next_query"), planned_b or question)
        if normalize_text(next_query) == normalize_text(query_a):
            next_query = planned_b or question  # never spend the second hop on the same query

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
        ctx.note("plan", query_a=query_a, next_query=next_query, planned_b=planned_b)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": next_query,
            "facts": facts,
        }
