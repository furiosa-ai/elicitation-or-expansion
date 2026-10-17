"""Typed answer contract: the last hop emits a *typed, span-shaped* answer and a deterministic
post-processor snaps it onto the surface form the grader expects.

Retrieval and both summarisation hops are byte-identical to the seed program, so this file isolates
one mechanism: the answer policy. The seed's final module inherits the generic
"Given the fields ..., produce the fields `answer`" instruction and therefore answers in prose
("Danny Kaye died on March 3, 1987."), which scores zero under exact match even when the fact is right.

Mechanism (all inside the same 4th call, no extra calls, no extra retrieval):
  1. the module first commits to an `answer_type` (yes/no, person, date, number, title, ...) and only
     then emits the `answer`, so the shape of the answer is decided before it is written;
  2. field descriptions + instructions define an explicit span contract (copy verbatim from the
     evidence, no sentence, no restatement, unit spelling preserved);
  3. a deterministic Python polisher enforces the contract afterwards: markdown/quote stripping,
     yes/no coercion, trailing-parenthetical and trailing-clause removal, and -- when the question
     offers a closed set of named options and exactly one of them occurs in a long answer -- the
     answer is replaced by that option copied verbatim out of the question.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

ANSWER_INSTRUCTIONS = """Give the final short answer to `question` using only the two summaries as evidence.

First decide `answer_type`, then write `answer` so that it matches that type exactly.

Rules for `answer`:
- It must be the shortest span that answers the question: normally one to four words.
- Never write a sentence, never restate the question, never explain, never add a closing period.
- Copy the span verbatim from the evidence (or from the question when the question lists the options):
  keep the original spelling, capitalisation, abbreviations and units, e.g. write a distance the way
  the evidence writes it rather than expanding or converting it.
- A question that can be answered with yes or no must be answered with exactly `yes` or exactly `no`.
- A question that asks which of two named things has some property must be answered with just that one
  name, spelled as in the question; do not name both and do not say why.
- Drop leading articles, titles like "the film" or "the band", parentheses and glosses.
- Always commit to exactly one span. When the evidence is thin, pick the most likely candidate the
  evidence does mention and write it plainly; an under-supported guess scores better than a refusal, and a
  reply about what the evidence lacks always scores zero."""

TYPE_DESC = (
    "one of: yesno, person, organisation, location, date, number, title, other -- the shape the answer must take"
)
ANSWER_DESC = "the final answer as a bare span, no sentence, no punctuation around it"

HEDGES = re.compile(
    r"(?i)\b(not (?:specified|mentioned|provided|available|determinable)|impossible to determine|"
    r"cannot be determined|no information|insufficient information|unknown)\b"
)
LEAD_IN = re.compile(
    r"(?i)^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*|^(?:answer|a)\s*:\s*|^(?:it|that)\s+(?:is|was)\s+"
)
YESNO_Q = re.compile(
    r"(?i)^(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should|both|the same)\b"
)
TRAILING_CLAUSE = re.compile(r"(?i)\s*[,;]\s*(?:while|whereas|although|though|but|because|since|so)\b.*$")
PAREN = re.compile(r"\s*\((?P<body>[a-z][^()]{0,80})\)\s*$")  # only a lower-case gloss, never a titled parenthetical
QUOTED = re.compile(r"[\"“‘']([^\"”’']{2,60})[\"”’']")
CAP_SPAN = re.compile(r"[A-Z][\w.&'’-]*(?:\s+(?:of|the|de|la|von|van|and|&)?\s*[A-Z0-9][\w.&'’-]*)*")
ARTICLES = re.compile(r"(?i)^(?:the|a|an)\s+")
CHOICE_Q = re.compile(  # the question itself lists the alternatives to choose between
    r"(?i)\bor\b|\bwhich (?:one|of)\b|\b(?:more|less|fewer|earlier|later|older|younger|larger|smaller|longer|shorter|higher|greater|first|closer|newer|bigger)\b"
)


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return " ".join(s.split())


def _question_options(question: str) -> list[str]:
    """Named alternatives the question itself offers (quoted titles and capitalised spans)."""
    opts = [m.group(1).strip() for m in QUOTED.finditer(question)]
    body = question[1:] if question[:1].isupper() else question  # the first word is capitalised by grammar
    for m in CAP_SPAN.finditer(body):
        span = m.group(0).strip(" ,.;:")
        if len(span) > 2:
            opts.append(span)
    seen, out = set(), []
    for o in opts:
        key = _norm(o)
        if key and key not in seen:
            seen.add(key)
            out.append(o)
    return out


def _first_sentence(a: str) -> str:
    """Cut a multi-sentence answer after its first sentence, without splitting on abbreviations."""
    toks = a.split()
    if len(toks) <= 8:
        return a
    for i, tok in enumerate(toks[:-1]):
        if tok.endswith(".") and len(tok.rstrip(".")) > 3 and toks[i + 1][:1].isupper():
            return " ".join(toks[: i + 1])
    return a


def polish(raw: str, answer_type: str, question: str) -> str:
    """Deterministically force the model's answer into the span shape exact match rewards."""
    a = (raw or "").strip()
    a = re.sub(r"\[\[.*?\]\]", " ", a)
    a = a.replace("**", "").replace("*", "").replace("`", "").strip()
    # keep the first non-empty line only: prose explanations live on later lines
    for line in a.splitlines():
        if line.strip():
            a = line.strip()
            break
    a = LEAD_IN.sub("", a).strip()
    a = a.strip("\"“”‘’ ")

    t = (answer_type or "").strip().lower()
    head = re.match(r"(?i)^(yes|no)\b", a)
    if head and (t.startswith("yes") or YESNO_Q.match(question.strip()) or len(a.split()) > 2):
        return head.group(1).lower()

    a = TRAILING_CLAUSE.sub("", a).strip()
    a = _first_sentence(a)
    a = a.rstrip(".;:, ")
    if len(a.split()) > 1:
        a = PAREN.sub("", a).strip()

    if len(a.split()) > 4 and CHOICE_Q.search(question):
        # the question offers a closed set of named alternatives and the long answer names exactly one
        # of them: that is a verbose way of choosing it, so answer with the alternative itself.
        na = _norm(a)
        hits = [o for o in _question_options(question) if _norm(o) and _norm(o) in na]
        hits = [o for o in hits if not any(_norm(o) != _norm(p) and _norm(o) in _norm(p) for p in hits)]
        if len(hits) == 1 and len(hits[0].split()) <= 6:
            a = hits[0]

    a = ARTICLES.sub("", a).strip()
    return a or (raw or "").strip()


CONNECTIVES = {
    "since", "while", "however", "although", "though", "both", "this", "that", "these", "those", "there",
    "based", "according", "when", "where", "which", "what", "who", "the", "and", "but", "for", "his", "her",
    "their", "its", "it", "they", "he", "she", "in", "on", "at", "no", "not", "given", "however the",
}


def _salvage(*texts_and_question: str) -> str:
    """When the model refuses to answer, name the entity the evidence talks about most.

    A refusal scores zero for certain, so the most repeated entity that the question did not already
    name is a strictly better bet than the refusal itself.
    """
    *texts, question = texts_and_question
    asked = _norm(question)
    blob = " ".join(t or "" for t in texts)
    nblob = _norm(blob)
    scored: dict[str, tuple[float, str]] = {}
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
            body = sentence.strip()
            if not body:
                continue
            body = body[1:] if body[:1].isupper() else body  # a sentence-initial word is capitalised by grammar
            for span in _question_options(body):
                ns = _norm(span)
                if not ns or ns in asked or "." in span:
                    continue
                if len(ns.split()) > 5 or len(ns) < 3 or ns in CONNECTIVES:
                    continue
                if all(w in CONNECTIVES for w in ns.split()):
                    continue
                count = nblob.count(ns)
                score = count + 0.5 * min(len(ns.split()), 3)
                if ns not in scored or score > scored[ns][0]:
                    scored[ns] = (score, span)
    if not scored:
        return ""
    return max(scored.values(), key=lambda x: x[0])[1]


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.summarize1 = Module(inputs=["question", "passages"], outputs=["summary"])
        self.create_query_hop2 = Module(inputs=["question", "summary_1"], outputs=["query"])
        self.summarize2 = Module(inputs=["question", "context", "passages"], outputs=["summary"])
        self.generate_answer = Module(
            inputs=["question", "summary_1", "summary_2"],
            outputs=["answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={"answer_type": TYPE_DESC, "answer": ANSWER_DESC},
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        hop1_docs = ctx.tools.retrieve(question, k=self.K)
        summary_1 = self.summarize1(ctx.llm, tag="summarize1", question=question, passages=hop1_docs)["summary"]

        query_2 = self.create_query_hop2(ctx.llm, tag="query_hop2", question=question, summary_1=summary_1)["query"]
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        summary_2 = self.summarize2(
            ctx.llm, tag="summarize2", question=question, context=summary_1, passages=hop2_docs
        )["summary"]

        out = self.generate_answer(
            ctx.llm, tag="answer", question=question, summary_1=summary_1, summary_2=summary_2
        )
        raw = out.get("answer", "")
        answer = polish(raw, out.get("answer_type", ""), question)
        if HEDGES.search(answer) or not answer:
            # a refusal scores zero for certain, so fall back to the most specific entity the evidence
            # names that the question did not already name
            salvage = _salvage(raw, out.get("reasoning", ""), summary_2, summary_1, question)
            if salvage:
                ctx.note("hedge_salvage", raw=answer[:120], used=salvage[:80])
                answer = salvage
        ctx.note("answer", raw=raw[:200], polished=answer, answer_type=out.get("answer_type", "")[:40])
        return {"answer": answer, "raw_answer": raw, "query_2": query_2}
