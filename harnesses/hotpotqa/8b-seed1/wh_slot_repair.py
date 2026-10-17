"""Test the answer against the slot the QUESTION opened, and repair the ones that cannot fit it.

Diagnosis.  The frontier answers 161/189 items correctly when both gold articles are in context but only
46/111 when one is missing, and a countable share of that second bucket is not a knowledge failure at all
-- it is an answer that could not possibly be right, whatever the evidence said:

    "Who did the vocals on the song X?"                -> "Bush"          (the band the question named)
    "In what year was the man who founded Y born?"     -> "not stated"    (a refusal)
    "Stephen Oremus was presented a Tony Award ... in what theatre?" -> "unknown"
    "Which actor starred in both A and B?"             -> "no"            (a polar answer to a wh-question)

Each of those violates a constraint the question itself fixes before any passage is read: a "who" opens a
person slot, an "in what year" opens a four-digit slot, a wh-question forbids yes/no, and a reference span
is never a phrase the question already handed over (the exception being a comparison that offers a closed
list of options, where echoing one IS the answer).  The frontier has an `answer_type` field, but the model
picks it to fit the answer it just wrote, so it can never contradict it: the type has to be derived from
the question, deterministically, and used as a TEST.

The other half of the diagnosis is negative and comes from the last iteration.  A sibling that also
attacked the incomplete-evidence bucket gained +2 there and +1 at zero recall, yet lost 10 items in the
full-recall bucket (151/189 against 161/189) -- because it changed the answering call's output contract
for all 300 items to carry the signal its gate needed.  So the gate here reads nothing the model reports:
the answering call is byte-identical to the frontier's, the test is computed from the question and the
projected answer alone, and the repair call exists only for items that fail it.

Mechanism under test (answer policy):
  * a deterministic wh-parser reads the question once and returns the ANSWER SLOT it opens (the head noun
    phrase after the interrogative, or a year / number / date / person slot), whether the question is
    polar, and whether it offers a closed list of options;
  * an admissibility test rejects: an empty answer, a refusal, yes/no for a non-polar question, a
    digit-free answer in a numeric slot, and a span the question already contains when no closed list is
    offered;
  * a rejected answer -- and only a rejected answer -- costs one further call, which is told the slot, the
    rejected string and the rule it broke, keeps the same evidence, and is told that when the passages do
    not state the fact it must answer from its own knowledge of the subject rather than decline.

Budget: 2 LLM calls on admissible items, 3 on the rest; 2 retrievals, k=7.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

GAP_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the question
itself, and they are already in hand. Usually they cover only ONE of the two articles this question
needs. You get exactly one more lookup. Find the gap and aim it.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`missing_target`: name the article still needed and say why the question needs it -- the entity the
question mentions but no passage above covers, or the entity the question only describes
indirectly. If the passages have now NAMED that entity, give the name. If they have not, describe
it with the question's own words.
There is essentially always a second article: an entity that a passage merely NAMES is not covered
by that passage, so if the question asks for a property of it -- a date, a place, a number, a
person, a title -- that entity's own article is what is missing. Only write "none" if the exact
answer to the question is already quoted word for word in `facts`; a lookup you decline is a lookup
thrown away.

`next_query`: the query for the final lookup, scored by a sparse index over article openings, so
write the words that the missing article's OWN OPENING would contain: the subject's name, plus the
two or three words that pin down what kind of thing it is (its type, field, place, role or date).
- Aim it at `missing_target`, not at the question. Do not hand the question back: the titles listed
  in `titles_in_hand` are already retrieved, and a query that returns them again buys nothing.
- Drop question words ("which", "who", "both", "same", "more", "first") and punctuation.
- If the question named two entities and only one was retrieved, this query is for the other one.
- Search terms only: no prose, no explanation, no quotation marks, at most about ten words."""

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
    "titles_in_hand": "titles of the articles already retrieved -- do not fetch these again",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "missing_target": "the article still needed, named if the passages name it",
    "next_query": "keyword query for the second and final lookup, aimed at the missing article",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
_WORD = re.compile(r"[A-Za-z0-9]+")
_QWORDS = {
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had", "that", "this", "these", "those", "it", "its", "by",
    "with", "from", "as", "both", "same", "more", "most", "first", "than", "there", "their", "his",
    "her", "he", "she", "they", "you", "not", "but", "also", "s",
}


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


_DECLINED = re.compile(r"^(none|n/?a|nothing|no\b.*|not\s|already\b)", re.IGNORECASE)


def _as_query(text: Any, fallback: str) -> str:
    """Sanitise a generated query; a declined or prose-wrapped lookup falls back deterministically."""
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    q = re.split(r"\s+[—–-]{1,2}\s+|\s*[—–]\s*", q)[0].strip(_STRIP_CHARS).strip()
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    if len(q) < 3 or _DECLINED.match(q):
        return fallback  # never spend the remaining lookup on a refusal
    return q


def _residual_query(question: str, titles: list[str]) -> str:
    """The question minus the words of the title it already matched best -- a complementary query.

    Deterministic fallback for when the model hands the question back: whatever hop A locked onto,
    the words that identify it are removed, so the index scores the rest of the question instead.
    """
    q_tokens = [t for t in _WORD.findall(question)]
    q_content = {t.lower() for t in q_tokens if t.lower() not in _QWORDS}
    if not q_content:
        return question
    best, best_hit = None, 0
    for title in titles:
        hit = len({t.lower() for t in _WORD.findall(title)} & q_content)
        if hit > best_hit:
            best, best_hit = title, hit
    if best is None or best_hit >= len(q_content):
        return question
    drop = {t.lower() for t in _WORD.findall(best)}
    kept = [t for t in q_tokens if t.lower() not in drop and t.lower() not in _QWORDS]
    return " ".join(kept) if len(kept) >= 2 else question


REPAIR_INSTRUCTIONS = """A first attempt at this question produced an answer that cannot be right,
whatever the evidence says: `rejected_answer` breaks the rule quoted in `rejection`. Replace it.

`answer_slot` is what the question asks for, read off the question itself. Your answer must be a thing
of that kind and nothing else.

Work in this order. Say what the question is really asking for. Check whether `facts` or `passages`
state it; if they do, copy the span. If they do NOT state it -- which is why the first attempt failed --
then answer from your own knowledge of the subjects named in the evidence. The passages are a starting
point, not a limit, and an incomplete pool is not a reason to decline.

`answer` is graded by exact string match against a short reference span:
- The bare answer only: no sentence, no explanation, no "because", no trailing clause, no quotation
  marks, no final period.
- It must fit `answer_slot`. A question that asks for a person is not answered by an organisation or a
  work; a question that asks for a year or a count is answered with digits; a question that asks which
  thing is not answered with yes or no.
- It must NOT be a phrase the question already gave you, unless the question offers a closed list of
  options, in which case copy the winning option exactly as the question spells it.
- Never answer "unknown", "not stated", "not mentioned", "none" or "cannot be determined". If you are
  unsure, give the single most likely answer anyway -- a guess of the right kind can be right, a refusal
  never is."""

DESCS_REPAIR = dict(DESCS)
DESCS_REPAIR.update({
    "answer_slot": "what kind of thing the question asks for, read off the question",
    "rejected_answer": "the answer produced first, which is inadmissible",
    "rejection": "the rule the rejected answer breaks",
})

_POLAR = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|am)\b", re.IGNORECASE)
_REFUSAL = re.compile(
    r"^(unknown|not\s+(stated|mentioned|specified|provided|given|available|found|determined|known)|"
    r"none|no\s+(answer|information|mention)|n/?a|cannot\s+be\s+determined|insufficient|unclear|"
    r"undetermined|unspecified|not\s+in\s+the\s+(passages|evidence)|there\s+is\s+no)\b",
    re.IGNORECASE)
_YEAR_Q = re.compile(r"\b(in|by|during)?\s*(what|which)\s+year\b|\byear\s+(was|did|were|is)\b", re.IGNORECASE)
_COUNT_Q = re.compile(r"\bhow\s+(many|much|old|long|tall|far|high)\b", re.IGNORECASE)
_DATE_Q = re.compile(r"\b(what|which)\s+(date|day|month)\b|\breleased\s+on\s+(what|which)\b", re.IGNORECASE)
_PERSON_Q = re.compile(r"^\s*who\b|\bwho\s+(is|was|are|were|did|does|do|played|wrote|directed|founded|"
                       r"starred|sang|created|won|had)\b|\bwhose\b", re.IGNORECASE)
_WH_NP = re.compile(r"\b(?:what|which)\s+((?:[A-Za-z0-9'’-]+\s*){1,5})", re.IGNORECASE)
_WH_START = re.compile(r"^\s*(who|whom|whose|what|which|when|where|why|how)\b", re.IGNORECASE)
_NP_STOP = {
    "is", "was", "are", "were", "did", "does", "do", "has", "have", "had", "the", "a", "an", "of", "in",
    "on", "at", "to", "for", "and", "or", "that", "which", "who", "with", "from", "by", "as", "be",
    "he", "she", "it", "they", "this", "these", "those", "his", "her", "their", "its", "you", "we",
}


def _answer_slot(question: str) -> tuple[str, bool]:
    """What the question asks for, and whether that slot must contain digits."""
    q = str(question or "")
    if _YEAR_Q.search(q):
        return "a year, written as four digits", True
    if _COUNT_Q.search(q):
        return "a number or a measurement, written with digits", True
    if _DATE_Q.search(q):
        return "a date", True
    if _PERSON_Q.search(q):
        return "the name of a person", False
    m = _WH_NP.search(q)
    if m:
        toks: list[str] = []
        for t in m.group(1).split():
            if t.lower() in _NP_STOP:
                break
            toks.append(t)
        toks = [t for t in toks if not t.isdigit()] or toks
        if toks:
            return "a " + " ".join(toks[:4]), False
    if re.search(r"^\s*when\b", q, re.IGNORECASE):
        return "a date or a year", True
    if re.search(r"^\s*where\b", q, re.IGNORECASE):
        return "the name of a place", False
    return "the short span the question asks for", False


def _inadmissible(answer: str, question: str) -> str:
    """The rule the answer breaks, or "" if it is admissible. Computed, never reported by the model."""
    a = normalize_text(answer or "")
    if not a:
        return "the answer is empty"
    if _REFUSAL.match(str(answer).strip()):
        return ("the answer declines to answer; a refusal always scores zero, so an answer of the right "
                "kind must be given even when the passages do not state it")
    # a question that opens with an interrogative is never polar, whatever else it contains
    polar = not _WH_START.match(question or "") and (
        bool(_POLAR.match(question or ""))
        or bool(re.search(r"\bboth\b|\bsame\b", question or "", re.IGNORECASE)))
    if a in ("yes", "no") and not polar:
        return ("the question is not a yes/no question -- it asks which or what or who -- so yes/no "
                "cannot be a valid answer")
    slot, numeric = _answer_slot(question)
    if numeric and not re.search(r"\d", a):
        return f"the question asks for {slot} and the answer contains no digits"
    if len(a.split()) > 12:
        return "the answer is a sentence rather than a short span"
    if a not in ("yes", "no") and not _question_options(question):
        nq = normalize_text(question)
        if f" {a} " in f" {nq} ":
            return ("the answer repeats a phrase the question already supplies; the question asks for "
                    "something it does NOT already name")
    return ""


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.gap = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "missing_target", "next_query"],
            instructions=GAP_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )
        self.repair = Module(
            inputs=["question", "answer_slot", "rejected_answer", "rejection", "facts", "passages"],
            outputs=["evidence_quote", "answer"],
            instructions=REPAIR_INSTRUCTIONS,
            descs=DESCS_REPAIR,
        )

    @staticmethod
    def _pool(*doc_lists: list[dict[str, Any]]) -> list[str]:
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

        # 1. the question itself is the first query -- broadest available, and costs no call
        docs_a = ctx.tools.retrieve_docs(question, k=self.K)
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]

        # 2. one call: read the evidence, then plan the remaining lookup against the gap it leaves
        gap = self.gap(
            ctx.llm,
            tag="gap",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(gap.get("facts", "")).strip()
        residual = _residual_query(question, titles_a)
        # fallback chain: the written query, else the named target, else the residual question
        target = _as_query(gap.get("missing_target"), residual)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = residual  # a repeat of the question returns the seven articles in hand
        if normalize_text(next_query) in {normalize_text(t) for t in titles_a}:
            next_query = target  # aim it at the target, not at an article already in hand

        # 3. second hop, aimed at the missing article
        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        # 4. answer over the deduplicated raw pool (contract identical to the parent)
        pool = self._pool(docs_a, docs_b)
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)

        # 5. the answer must fit the slot the question opened; only a failure costs a further call
        slot, _numeric = _answer_slot(question)
        rejection = _inadmissible(answer, question)
        if rejection:
            fixed = self.repair(
                ctx.llm,
                tag="repair",
                question=question,
                answer_slot=slot,
                rejected_answer=answer or "(empty)",
                rejection=rejection,
                facts=facts if facts else "none",
                passages=pool,
            )
            cand = project_answer(fixed.get("answer", ""), "", question)
            if cand and not _REFUSAL.match(cand):
                answer = cand
            ctx.note("slot repair", slot=slot, rejection=rejection, repaired=answer)

        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "answer_slot": slot,
            "rejection": rejection,
            "facts": facts,
        }
