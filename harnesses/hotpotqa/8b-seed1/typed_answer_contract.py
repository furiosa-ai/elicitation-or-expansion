"""Typed answer contract: same retrieval and summarisation skeleton as the seed program, but the
final call emits a *typed, grounded, bare* answer instead of free prose, and a deterministic
projection maps whatever it emits onto the shortest scoring surface form.

Mechanism under test (answer policy only -- retrieval and the two summary hops are untouched):
    the last module's output schema becomes
        reasoning -> evidence_quote -> answer_type -> answer
    so the model must (a) ground itself in a verbatim quote, (b) commit to the *shape* of the
    answer, and only then (c) emit the bare span. A pure-Python projection then enforces the
    shape (yes/no collapse, first line, drop trailing explanatory clauses, echo an option that
    the question itself offered).

Budget: 4 LLM calls, 2 retrievals, k=7 -- identical to the seed program.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

ANSWER_INSTRUCTIONS = """Answer the question from the evidence in the summaries.

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
  span would not contain (write `Cincinnati`, not `Cincinnati, Ohio, in the United States`).
- If the evidence is incomplete, still give the single most likely bare answer. Never answer
  "unknown", "not stated" or "cannot be determined"."""

ANSWER_DESCS = {
    "question": "the question to answer",
    "summary_1": "evidence gathered in the first retrieval hop",
    "summary_2": "evidence gathered in the second retrieval hop",
    "evidence_quote": "the exact sentence from the summaries that settles the question",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
# a sentence boundary only counts after a word/number, never after an initial ("Albert R. Broccoli")
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"


def _clean(s: Any) -> str:
    """Surface cleanup shared by every projection step."""
    text = str(s or "").strip()
    if not text:
        return ""
    text = text.split("\n")[0]
    text = _PREFIX.sub("", text)
    return text.strip(_STRIP_CHARS).strip()


_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)


def _question_options(question: str) -> list[str]:
    """Closed-list options the question itself offers, e.g. 'Which of A or B ...'."""
    if not _CLOSED_LIST.search(question or ""):
        return []
    q = str(question).strip().rstrip("?")
    sep = r",|\bor\b|\bversus\b|\bvs\.?\b"
    if re.match(r"\s*between\b", q, re.IGNORECASE):
        sep += r"|\bbetween\b|\band\b"  # "Between A and B, which ...?"
    opts: list[str] = []
    for chunk in re.split(sep, q, flags=re.IGNORECASE):
        cand = re.sub(r"^\s*(between|which|who|what|whom|whose)\b\s*", "", chunk, flags=re.IGNORECASE)
        cand = cand.strip(_STRIP_CHARS).strip()
        # keep chunks that look like a named option, not clause fragments
        if 1 <= len(cand.split()) <= 8 and re.search(r"[A-Z0-9]", cand):
            opts.append(cand)
    return opts


def _echo_option(ans: str, question: str) -> str:
    """A declarative answer whose subject is one of the offered options scores as that option."""
    norm_ans = normalize_text(ans)
    if not norm_ans:
        return ans
    prefix_hits, inside_hits = [], []
    for opt in _question_options(question):
        n = normalize_text(opt)
        if not n or n == norm_ans:
            return ans  # already exactly an option
        if len(n.split()) >= len(norm_ans.split()):
            continue
        if norm_ans.startswith(n + " "):
            prefix_hits.append(opt)
        elif f" {n} " in f" {norm_ans} ":
            inside_hits.append(opt)
    hits = prefix_hits or ([] if len(inside_hits) != 1 else inside_hits)
    return max(hits, key=lambda o: len(normalize_text(o))) if hits else ans


def project_answer(raw_answer: str, answer_type: str, question: str) -> str:
    """Deterministically map the model's answer onto the shortest scoring surface form."""
    ans = _clean(raw_answer)
    atype = str(answer_type or "").strip().lower()

    yn = _YESNO.match(ans)
    if yn and (atype.startswith("yes") or len(ans.split()) > 1):
        return yn.group(1).lower()
    if atype.startswith("yes") and normalize_text(ans) in ("yes", "no"):
        return normalize_text(ans)

    # a trailing explanatory sentence is never part of the reference span
    if len(ans.split()) > 4:
        parts = [p for p in _SENT.split(ans) if p.strip()]
        if len(parts) > 1 and len(parts[0].split()) >= 3:
            ans = _clean(parts[0])

    # still a declarative clause: if the question offered a closed list, echo the option it picked
    if len(ans.split()) > 1:
        ans = _echo_option(ans, question)

    return _clean(ans) or _clean(raw_answer)


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.summarize1 = Module(inputs=["question", "passages"], outputs=["summary"])
        self.create_query_hop2 = Module(inputs=["question", "summary_1"], outputs=["query"])
        self.summarize2 = Module(inputs=["question", "context", "passages"], outputs=["summary"])
        self.generate_answer = Module(
            inputs=["question", "summary_1", "summary_2"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=ANSWER_DESCS,
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        # HOP 1
        hop1_docs = ctx.tools.retrieve(question, k=self.K)
        summary_1 = self.summarize1(ctx.llm, tag="summarize1", question=question, passages=hop1_docs)["summary"]
        # HOP 2
        query_2 = self.create_query_hop2(ctx.llm, tag="query_hop2", question=question, summary_1=summary_1)["query"]
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        summary_2 = self.summarize2(ctx.llm, tag="summarize2", question=question, context=summary_1, passages=hop2_docs)["summary"]
        # ANSWER under the typed contract
        out = self.generate_answer(
            ctx.llm, tag="answer", question=question, summary_1=summary_1, summary_2=summary_2
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "evidence_quote": out.get("evidence_quote", ""),
            "query_2": query_2,
            "summary_1": summary_1,
            "summary_2": summary_2,
        }
