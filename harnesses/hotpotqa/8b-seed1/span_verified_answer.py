"""Answer by selecting and cutting a span out of the evidence, then verifying it is really there.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%). Its 300 search-set items split by how
much of the gold evidence reached the context: with both gold articles present it scores 0.817
(147/180); with one, 0.51; with neither, 0.125. So 33 of its 101 errors happen with the evidence
fully in hand. Reading those 33, they are not reasoning failures -- they are failures of *what
counts as the answer string*:
  - the model answers with a name it composed or recalled rather than one the passages contain
    (a programme name where the passages say a one-word property; a film title where the passages
    say an ordinal; a genus name where the passages say a common noun);
  - or it hands back a truncation of the phrase the article actually writes, dropping the unit,
    the head noun, or the place that the sentence attaches directly to the name.
Both are the same defect: the answerer is *generating* a string, and nothing checks that the string
it generated exists in the evidence. The parent's prose even pushes it the wrong way -- "the
shortest form that fully names it" licenses cutting the phrase down below what the article writes.

Mechanism under test (answer policy only; the plan -> hop A -> resolve -> hop B retrieval pipeline
and the evidence pool are byte-identical to the parent, so the comparison isolates this change):

  SPAN SELECTION WITH DETERMINISTIC VERIFICATION AND REPAIR.
  1. The answering module no longer emits a free-form answer. It must first name the article it is
     using (`support_title`) and copy out, character for character, the one sentence from that
     article that settles the question (`support_sentence`). The answer must then be a *contiguous
     substring of that sentence*, cut at the boundaries of the phrase the question asks for --
     keeping whatever the phrase carries with it and inventing nothing.
  2. A deterministic verifier then checks the claim: for every non-yes/no answer, the normalised
     answer must occur inside the normalised evidence pool. This is a check the model cannot fake.
  3. If the check fails, the fourth call is spent on a repair: the model is shown its own rejected
     answer, told the exact rule it broke, and asked for a span that is in the passages. If the
     repair is also ungrounded, the better-grounded of the two is kept.
  A grounded answer is never re-asked, so the repair costs a call only on the items that need it.

Budget: 3 LLM calls on grounded items, 4 on repaired ones; 2 retrievals at k=7 -- inside the
interaction depth.
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

ANSWER_INSTRUCTIONS = """Answer the question by CUTTING THE ANSWER OUT OF THE EVIDENCE. You are not
writing an answer, you are selecting one. `facts` are sentences already extracted; `passages` are
the raw article openings, each as "Title | opening".

Fill the fields in this order and do not skip ahead:

`support_title`: the title of the one article, from `passages`, that settles the question. If two
articles are needed, name the one the answer phrase is written in.

`support_sentence`: that article's sentence containing the answer, copied character for character
from the passage. Do not paraphrase it, do not shorten it, do not merge two sentences. If the
question can only be settled by combining two articles, copy the sentence that contains the phrase
you are going to hand back.

`answer_type`: exactly one of yes_no, person, organization, work_title, place, date, number, other.

`answer`: graded by exact string match against a short reference phrase, so it must be the bare
answer and nothing else. No sentence, no restatement of the question, no explanation, no "because",
no trailing clause, no markdown, no quotation marks, no final period. And:
- If `answer_type` is yes_no, `answer` is exactly `yes` or `no`, one word. Questions of the form
  "are both ...", "do both ...", "is X the same as Y", "did both ..." are yes_no questions.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly as the question spells it.
- OTHERWISE `answer` must be a CONTIGUOUS SUBSTRING OF `support_sentence` -- every word of it, in
  the same order, with the same spelling, taken from that sentence. Never assemble it from two
  places, never substitute a word you consider equivalent, never supply a name from memory that the
  passages do not print.
- Cut the substring at the boundaries of the phrase the question asks for, and keep what that
  phrase carries in the sentence: the unit after a number, the head noun after a count, the
  region after a settlement, the qualifier the sentence attaches to the name. Do not trim it
  further than the sentence writes it, and do not extend it into a clause the question did not ask
  for ("which is ...", "and ...", "who ...", "located in" when the question asked only for a name).
- If the question asks what kind, what type, or what something is, the answer is the descriptive
  phrase the sentence uses, not the name of an entity.
- If the evidence is incomplete, still hand back the single most likely bare phrase. Never answer
  "unknown", "not stated" or "cannot be determined"."""

REPAIR_INSTRUCTIONS = """Your previous answer was REJECTED: the phrase you gave does not occur
anywhere in the retrieved passages, so it cannot be copied evidence -- you composed it, recalled it,
or altered its wording.

Give the answer again, this time as a phrase that is literally present in `passages`.

`located_in`: the title of the passage whose text contains your new answer, copied from `passages`.
`answer`: a contiguous substring of that passage. Every word in the same order and spelling as the
passage prints it. It must still be the bare answer to the question -- no sentence, no explanation,
no quotation marks, no final period.
- If the question is a yes/no question, answer exactly `yes` or `no` instead.
- If the question offers a closed list of options, copy the winning option as the question spells it.
- Choose the phrase that most directly answers the question; if several passages contain a
  candidate, prefer the one whose article the question is really about.
- Never answer "unknown" or "not stated": if nothing fits perfectly, give the closest phrase that
  the passages do print."""

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
    "support_title": "title of the article the answer is written in",
    "support_sentence": "that article's sentence containing the answer, copied verbatim",
    "rejected_answer": "the answer that was not found in the passages",
    "located_in": "title of the passage that literally contains the new answer",
    "answer": "the bare answer phrase only -- no sentence, no explanation",
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


def is_grounded(answer: str, haystack_norm: str, question: str) -> bool:
    """True when the answer is really present in the evidence (or is exempt from the rule)."""
    n = normalize_text(answer)
    if not n:
        return False
    if n in ("yes", "no"):
        return True
    if any(normalize_text(o) == n for o in _question_options(question)):
        return True  # an option echoed from the question is grounded by the question
    return f" {n} " in haystack_norm


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
            outputs=["support_title", "support_sentence", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )
        self.repair = Module(
            inputs=["question", "rejected_answer", "passages"],
            outputs=["located_in", "answer"],
            instructions=REPAIR_INSTRUCTIONS,
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
        pool = self._pool(docs_a, docs_b)

        # 5. select a span out of the pool rather than composing an answer
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        atype = out.get("answer_type", "")
        answer = project_answer(raw, atype, question)

        # 6. deterministic groundedness check, and one repair call if it fails
        haystack = " " + normalize_text(" ".join(pool)) + " "
        grounded = is_grounded(answer, haystack, question)
        repaired = ""
        if not grounded:
            fix = self.repair(
                ctx.llm,
                tag="repair",
                question=question,
                rejected_answer=answer,
                passages=pool,
            )
            repaired = project_answer(fix.get("answer", ""), atype, question)
            if repaired and is_grounded(repaired, haystack, question):
                answer, grounded = repaired, True
            elif repaired and not answer:
                answer = repaired

        ctx.note("plan", query_a=query_a, next_query=next_query, planned_b=planned_b)
        ctx.note(
            "span answer",
            answer_type=atype,
            support_title=str(out.get("support_title", ""))[:120],
            raw=str(raw)[:200],
            grounded=grounded,
            repaired=repaired[:120],
            final=answer,
        )
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": atype,
            "grounded": grounded,
            "support_title": out.get("support_title", ""),
            "query_a": query_a,
            "next_query": next_query,
            "facts": facts,
        }
