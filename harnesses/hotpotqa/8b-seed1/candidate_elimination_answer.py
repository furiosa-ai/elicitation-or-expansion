"""The final call picks between evidence-backed candidates instead of composing one answer.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%). Splitting its 101 errors by the
relation between the prediction and the reference span:
  - 40  the reference span is nowhere in the retrieved passages          (retrieval / knowledge)
  - 34  the reference span IS in the passages and another string was returned
  -  4  the prediction is the reference span plus a trailing parenthetical disambiguator
  -  4  the prediction is empty because the final call spent its whole completion on reasoning
  - the rest are label-granularity noise (a middle name present or absent in the reference).

The 34-item bucket is the largest tractable one and reading those traces the failure is always the
same shape: the model returns a plausible entity of the right *type* rather than the one the
question's conditions pin down, and it often returns a name that no passage contains at all -- a
recalled association rather than a read fact. Two sub-shapes recur: answering with the bridge
entity when the question asks for a property of it, and satisfying one condition of a multi-clause
question while ignoring the others.

Mechanism under test (answer policy; retrieval, hop structure, queries, the evidence pool and the
call count are byte-identical to the parent so the delta isolates this change):

  ANSWER BY ELIMINATION OVER AN EVIDENCE-BACKED CANDIDATE SLATE.
  The final call no longer emits "a quote, then the answer". It must first state what the question
  literally demands -- the slot to fill and every condition on it, split into numbered atoms -- then
  put up two or three *competing* candidates, each one a string that occurs in the passages, each
  carrying the passage number and the verbatim clause that supports it, and each marked with the
  numbered condition it fails. Only then does it name the survivor. A candidate that no passage
  contains is inadmissible by construction, which is exactly the move that a recalled association
  cannot survive.

  Three deterministic guards ride along in the projection, each measured on the parent's own
  predictions before being written: strip a trailing parenthetical disambiguator (2 items repaired,
  0 broken); refuse `yes`/`no` for a question that opens with a wh-word and contains no
  both/same/either comparison (1 repaired, 0 broken, and none of the reference yes/no answers are
  of that shape); and when the completion is empty because reasoning consumed the budget, recover
  the answer from the tail of the reasoning instead of returning nothing.

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

ANSWER_INSTRUCTIONS = """Decide the answer by elimination, not by composition. `facts` are clauses
already copied out of the first lookup, `passages` are the raw article openings retrieved for this
question, numbered.

Work through the four output fields in order and keep every one of them short.

`demands`: what the question literally asks for, in two parts on two lines.
  SLOT: the wh-phrase, copied from the question, naming the kind of thing wanted -- "the
  manufacturer", "the year", "how many", "what the programme was at times". Watch the slot: when
  the wh-word asks for a property, a description, a count or a category ("was at times what", "what
  type of", "known for what", "how many"), the answer is the word or phrase that fills that slot in
  the evidence, NOT the name of the entity the question was routed through.
  CONDITIONS: every condition the answer must satisfy, numbered, each copied from the question. A
  question with three clauses has three conditions and the answer must meet all of them.

`candidates`: two or three competing answers, one per line, in the form
  <candidate> | p<passage number> | "<verbatim clause from that passage>" | fails: <condition
  numbers, or "none">
  Rules that decide what may appear here:
  - a candidate must be a string that occurs in `passages` or in `facts`. If your own memory
    supplies a name that no passage contains, it is not a candidate -- leave it out. Recalled
    associations are the single most common way this task is failed.
  - the candidates must be genuinely different answers, not spellings of one answer; put up the
    rival that a careless reader would pick, then show which condition kills it.
  - the quote must be copied character for character and must actually mention the candidate.
  - if the evidence supports only one candidate, write one line.

`answer_type`: exactly one of yes_no, person, organization, work_title, place, date, number, other.
  `yes_no` is admissible only when the question can be answered by the bare word yes or no --
  "are both ...", "do both ...", "is X the same as Y", "can both ...". A question that opens with
  who, which, what, where, when or how, or that offers a choice between named options, is never
  `yes_no`, however tempting a yes or no feels.

`answer`: the surviving candidate and nothing else. It is graded by exact string match against a
short reference span, so:
- never a sentence, never a restatement of the question, no explanation, no "because", no trailing
  clause, no second sentence, no markdown, no quotation marks, no final period.
- if `answer_type` is yes_no, write exactly `yes` or `no`, one word.
- if the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly as the question spells it.
- otherwise copy the span verbatim from the evidence, in the shortest form that fully names it, and
  without the parenthetical disambiguator an article title carries.
- if the evidence is incomplete, still name the single most likely surviving candidate. Never
  answer "unknown", "not stated" or "cannot be determined"."""

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
    "demands": "SLOT line then numbered CONDITIONS, all copied from the question",
    "candidates": "competing answers, one per line, each with passage number, verbatim quote and failed conditions",
    "answer": "the surviving candidate only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
# a sentence boundary only counts after a word/number, never after an initial ("Albert R. Broccoli")
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
# an open wh-question that carries no comparison marker cannot be answered by yes or no
_WH_OPEN = re.compile(r"^\s*(who|which|what|where|when|how|why|whose|whom|name)\b", re.IGNORECASE)
_COMPARISON = re.compile(r"\b(both|same|either|as well as)\b", re.IGNORECASE)
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


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


def yes_no_admissible(question: str) -> bool:
    """A wh-question with no comparison marker can never be answered by the bare word yes or no."""
    q = str(question or "")
    return not (_WH_OPEN.match(q) and not _COMPARISON.search(q))


def drop_title_disambiguator(ans: str) -> str:
    """`Alex Cross (film)` -> `Alex Cross`; reference spans never carry the article's disambiguator."""
    stripped = _TRAILING_PAREN.sub("", ans).strip(_STRIP_CHARS).strip()
    return stripped if stripped and len(stripped.split()) >= 1 else ans


def project_answer(raw_answer: str, answer_type: str, question: str) -> str:
    ans = _clean(raw_answer)
    atype = str(answer_type or "").strip().lower()
    allow_yn = yes_no_admissible(question)
    yn = _YESNO.match(ans)
    if allow_yn:
        if yn and (atype.startswith("yes") or len(ans.split()) > 1):
            return yn.group(1).lower()
        if atype.startswith("yes") and normalize_text(ans) in ("yes", "no"):
            return normalize_text(ans)
    elif yn and len(ans.split()) > 1:
        # "No, the school is Berea College" -- the yes/no was a prefix, the answer follows it
        ans = _clean(re.sub(r"^\W*(yes|no)\b[\s,;:.-]*", "", ans, flags=re.IGNORECASE)) or ans
    if len(ans.split()) > 4:
        parts = [p for p in _SENT.split(ans) if p.strip()]
        if len(parts) > 1 and len(parts[0].split()) >= 3:
            ans = _clean(parts[0])
    ans = drop_title_disambiguator(ans)
    if len(ans.split()) > 1:
        ans = _echo_option(ans, question)
    return _clean(ans) or _clean(raw_answer)


def _as_query(text: Any, fallback: str) -> str:
    """Keep a generated query usable as a sparse-index query."""
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    return q if len(q) >= 3 else fallback


def _mine_reasoning(reasoning: str) -> str:
    """Recover an answer from the tail of a completion that spent everything on reasoning."""
    text = str(reasoning or "").strip()
    if not text:
        return ""
    for line in reversed(text.splitlines()[-25:]):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(?:answer|it is|so it's|that's)\s*(?:is|:|=)?\s*[\"“']?([^\"”'\n.]{1,60})", line, re.IGNORECASE)
        if m and m.group(1).strip(_STRIP_CHARS).strip():
            return m.group(1).strip(_STRIP_CHARS).strip()
    return ""


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
            outputs=["demands", "candidates", "answer_type", "answer"],
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

        # 5. answer by elimination over an evidence-backed candidate slate
        msgs = self.generate_answer.messages(
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        resp = ctx.llm.chat(msgs, tag="answer")
        out = self.generate_answer.parse(resp.text)
        raw = out.get("answer", "")
        if not _clean(raw):
            # the completion was consumed by reasoning; the conclusion is usually still in there
            raw = _mine_reasoning(getattr(resp, "reasoning", "") or "")
            ctx.note("recovered from reasoning tail", recovered=str(raw)[:200])
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("plan", query_a=query_a, next_query=next_query, planned_b=planned_b)
        ctx.note("slate", candidates=str(out.get("candidates", ""))[:400], demands=str(out.get("demands", ""))[:300])
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "candidates": out.get("candidates", ""),
            "demands": out.get("demands", ""),
            "query_a": query_a,
            "next_query": next_query,
            "facts": facts,
        }
