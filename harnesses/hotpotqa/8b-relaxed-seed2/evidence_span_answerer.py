"""Extractive, type-aware, verified final answer stage on top of the 2-hop retrieval pipeline.

The parent pipeline keeps the same two retrieval hops and summaries, but the free-form answer
module is replaced by an ANSWER POLICY LAYER:

  1. the final call is extractive: it sees the raw deduplicated passage text (not only the
     summaries) and is asked for the shortest span that answers the question;
  2. a deterministic post-processor strips sentence wrappers / markdown and detects the required
     answer form from the question itself (binary vs. pick-one-of-two-alternatives vs. span);
  3. binary questions are snapped to a single token, alternative questions are snapped to the
     matching alternative as it is written in the question;
  4. a verifier checks the span is short and actually occurs in the evidence; if it fails, ONE
     repair call re-extracts the minimal verbatim span and the shorter survivor wins.

Motivation: with the same evidence, a correct fact phrased as a sentence scores zero under exact
match. This candidate changes the answer policy, not the retrieval.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

BINARY_LEADS = {
    "are", "is", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "am",
}
YES_RE = re.compile(r"\b(yes|both|true|correct)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|not|neither|false|incorrect)\b", re.IGNORECASE)
LEAD_JUNK = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:short\s+)?answer\s*(?:is|:)\s*|^(?:it|that)\s+(?:is|was)\s+",
    re.IGNORECASE,
)
TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
MARKUP = re.compile(r"[*_`\"“”‘’]")
ABSTAIN = re.compile(
    r"(?i)\b(?:not (?:specified|mentioned|provided|stated|given|available|clear|possible|determined)|"
    r"no (?:information|evidence|mention|passage|details)|none of the|does not (?:specify|mention|state|provide)|"
    r"do not (?:specify|mention|state|provide)|cannot be|can(?:'|no)?t be|unknown|unclear|insufficient)\b"
)


def _clean(text: str) -> str:
    """Deterministic surface cleanup of a raw model answer."""
    s = (text or "").strip()
    s = s.split("\n")[0].strip()
    s = MARKUP.sub("", s).strip()
    s = LEAD_JUNK.sub("", s).strip()
    # a trailing parenthetical is usually a gloss, but it can be part of a title:
    # only drop it when what remains is still a multi-word phrase
    without_paren = TRAIL_PAREN.sub("", s).strip()
    if without_paren and len(without_paren.split()) >= 2:
        s = without_paren
    s = s.rstrip(".,;: ").strip()
    return s


def _words(text: str) -> list[str]:
    return normalize_text(text).split()


def _is_binary_question(question: str) -> bool:
    toks = re.findall(r"[a-z']+", (question or "").lower())
    return bool(toks) and toks[0] in BINARY_LEADS


def _alternatives(question: str) -> list[str]:
    """Surface alternatives of a 'X or Y' question, as written in the question."""
    q = (question or "").strip().rstrip("?")
    if " or " not in q.lower():
        return []
    # take the clause after the last comma / colon, which usually holds the two options
    clause = re.split(r"[,:;]", q)[-1]
    parts = re.split(r"\s+or\s+", clause, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []
    opts = []
    for p in parts:
        p = _clean(p)
        p = re.sub(
            r"^(?:which|who|what|whom|whose|the|of|in|was|were|is|are|did|does|do|has|have|had|"
            r"one|a|an|it|that|between|either|both)\s+",
            "",
            p,
            flags=re.IGNORECASE,
        ).strip()
        if not p or len(p.split()) > 8:
            return []
        opts.append(p)
    return opts


def _snap_alternative(answer: str, options: list[str]) -> str | None:
    """Return the option the answer refers to, written exactly as in the question."""
    a = set(_words(answer))
    if not a:
        return None
    scored = []
    for opt in options:
        o = set(_words(opt))
        if not o:
            return None
        scored.append((len(o & a) / len(o), opt))
    scored.sort(reverse=True)
    if scored[0][0] >= 0.6 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _binary_value(answer: str) -> str | None:
    """Read a yes/no verdict out of a (possibly verbose) answer."""
    s = _clean(answer)
    head = re.findall(r"[a-z']+", s.lower())[:1]
    if head:
        if head[0] == "yes":
            return "yes"
        if head[0] in ("no", "neither", "not"):
            return "no"
    first = s.split(",")[0]
    if NO_RE.search(first):
        return "no"
    if YES_RE.search(first):
        return "yes"
    return None


class Harness(BaseHarness):
    K = 7
    MAX_ANSWER_WORDS = 6

    EXTRACT_INSTR = (
        "Answer the question using only the given evidence.\n"
        "Write `answer` as the exact phrase the evidence uses, and nothing else: a person or work "
        "name, a place, a number or a date.\n"
        "Keep the phrase complete - a full name, the full title, the full date as written - but "
        "drop every word that is not part of the answer: no sentence, no verb, no explanation, "
        "no `because` clause and no restatement of the question.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )
    REPAIR_INSTR = (
        "The draft answer is phrased as a sentence, but the grader compares it literally with the "
        "expected phrase.\n"
        "Return in `answer` just the part of the draft that answers the question, copied verbatim "
        "from the evidence: the complete name, title, number or date, with no verb, no explanation "
        "and no trailing clause. Do not truncate it either - keep a full name whole and a full "
        "date whole."
    )

    GUESS_INSTR = (
        "The evidence is incomplete, but an answer must still be given: there is no credit for "
        "saying that the information is missing.\n"
        "Commit to the single most likely answer supported by the evidence and general knowledge, "
        "and write it in `answer` as a bare phrase - a name, title, place, number or date - with "
        "no hedging, no explanation and no sentence.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.summarize1 = Module(inputs=["question", "passages"], outputs=["summary"])
        self.create_query_hop2 = Module(inputs=["question", "summary_1"], outputs=["query"])
        self.summarize2 = Module(inputs=["question", "context", "passages"], outputs=["summary"])
        self.extract = Module(
            inputs=["question", "evidence", "notes"],
            outputs=["answer"],
            instructions=self.EXTRACT_INSTR,
        )
        self.repair = Module(
            inputs=["question", "evidence", "draft_answer"],
            outputs=["answer"],
            instructions=self.REPAIR_INSTR,
        )
        self.guess = Module(
            inputs=["question", "evidence", "notes"],
            outputs=["answer"],
            instructions=self.GUESS_INSTR,
        )

    # -- policy -------------------------------------------------------------------------------
    def _finalize(self, question: str, draft: str, evidence_blob: str) -> tuple[str, bool]:
        """Apply the deterministic answer policy. Returns (answer, needs_repair)."""
        ans = _clean(draft)
        if not ans:
            return "", True
        if _is_binary_question(question):
            verdict = _binary_value(ans)
            if verdict is not None:
                return verdict, False
        options = _alternatives(question)
        if options:
            snapped = _snap_alternative(ans, options)
            if snapped is not None:
                return snapped, False
        if ABSTAIN.search(ans):  # a hedge scores zero for sure; a committed guess does not
            return ans, True
        n_words = len(ans.split())
        in_evidence = " ".join(_words(ans)) in evidence_blob
        needs_repair = n_words > self.MAX_ANSWER_WORDS or (n_words > 2 and not in_evidence)
        return ans, needs_repair

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # HOP 1
        hop1_docs = ctx.tools.retrieve(question, k=self.K)
        summary_1 = self.summarize1(ctx.llm, tag="summarize1", question=question, passages=hop1_docs)["summary"]

        # HOP 2
        query_2 = self.create_query_hop2(ctx.llm, tag="query_hop2", question=question, summary_1=summary_1)["query"]
        hop2_docs = ctx.tools.retrieve(query_2 or question, k=self.K)
        summary_2 = self.summarize2(
            ctx.llm, tag="summarize2", question=question, context=summary_1, passages=hop2_docs
        )["summary"]

        # deduplicated raw evidence, in retrieval order
        seen: set[str] = set()
        evidence: list[str] = []
        for doc in list(hop1_docs) + list(hop2_docs):
            key = doc.split(" | ")[0].strip().lower()
            if key not in seen:
                seen.add(key)
                evidence.append(doc)
        evidence_blob = " ".join(_words(" ".join(evidence)))

        # EXTRACTIVE ANSWER
        notes = f"{summary_1}\n{summary_2}".strip()
        draft = self.extract(
            ctx.llm, tag="extract", question=question, evidence=evidence, notes=notes
        ).get("answer", "")
        answer, needs_repair = self._finalize(question, draft, evidence_blob)

        # VERIFY -> ONE REPAIR CALL
        repaired = ""
        if needs_repair:
            hedged = bool(answer) and bool(ABSTAIN.search(answer))
            if hedged:  # re-ask for a committed guess instead of trimming a refusal
                repaired = self.guess(
                    ctx.llm, tag="guess", question=question, evidence=evidence, notes=notes
                ).get("answer", "")
            else:
                repaired = self.repair(
                    ctx.llm, tag="repair", question=question, evidence=evidence,
                    draft_answer=answer or draft,
                ).get("answer", "")
            fixed, still_bad = self._finalize(question, repaired, evidence_blob)
            if fixed and not ABSTAIN.search(fixed) and (
                not still_bad or not answer or hedged or len(fixed.split()) < len(answer.split())
            ):
                answer = fixed
            ctx.note("answer repaired", hedged=hedged, final=answer[:200])

        if not answer:
            answer = _clean(draft) or _clean(summary_2)
        return {
            "answer": answer,
            "draft": draft,
            "repaired": repaired,
            "query_2": query_2,
            "summary_1": summary_1,
            "summary_2": summary_2,
        }
