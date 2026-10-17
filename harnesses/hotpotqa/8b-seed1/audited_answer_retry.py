"""Audit the produced answer against two admissibility tests, and spend a retry only when it fails.

Measured motivation (replayed over the frontier's 300 search-set traces; the frontier's two calls
are reproduced byte for byte, so every number below is a post-hoc bookkeeping count):

    test 1 -- SUPPORT.  Is the answer string present verbatim in the pooled passages?
        present                                             267 items    0.734
        absent                                               33 items    0.455

    test 2 -- REFUSAL.  Is it a non-answer ("unknown", "not stated")?
        5 items, 0 correct -- a bucket where a retry cannot lose.

    test 3 -- NOVELTY.  Does the answer repeat a span the question already supplies?
        repeats it, and the question offers a closed list     30 items    0.833
        repeats it, and the question offers no list            6 items    0.167
        does not repeat it                                   264 items    0.701

Both tests are properties of the *output*, computable without labels, and both isolate a bucket
that scores far below the system average. The second is the sharper one and it has a reason: a
question of this kind asks for a fact the passages add, so a reference span almost never lies
inside the question -- across the 300 items the reference appears in its own question 31 times and
29 of those are "A or B" choices, where echoing IS the contract. Outside that construction, an
answer made of the question's own words is a restatement, not an answer.

Mechanism under test (answer policy: abstain-and-retry inside the call budget, not prompting and
not retrieval): the answer is audited after projection, and the audit either passes -- in which
case the item is byte-identical to the parent, calls and all -- or names the specific defect and
buys ONE more call. The retry is not a re-read: it is told which test failed, is shown the string
that failed it, and is required to return a span that the passages actually contain and that the
question does not already supply. The replacement is accepted only if it clears the test that the
first answer failed; otherwise the first answer stands. So the swap can only ever move an item
from a measured-worse state into a measured-better one, and the ~88% of items that pass the audit
are untouched.

Both retrievals, the pool, both existing prompts and every projection rule are unchanged from
`source_named_answer`.

Budget: 2 LLM calls (3 on the audited minority), 2 retrievals, k=7.
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

RETRY_INSTRUCTIONS = """A first attempt at this question produced `rejected_answer`, and an
automatic check on that string failed: `defect` says which one. The reasoning that produced it is
not available to you and is not needed -- re-derive the answer from the evidence below.

`rejected_answer` is wrong as written. Do not return it, and do not return a longer phrase that
contains it unless the extra words come from the passages.

The two checks are:
- support -- the answer must occur, word for word, inside `passages`. An answer assembled from
  memory or paraphrased out of the passages fails.
- novelty -- the answer must be something the passages ADD. A question names what is already known
  and asks for what is not, so a span lifted out of the question itself is a restatement. The one
  exception is a question that offers a closed list of alternatives ("A or B"), where copying the
  winning option is exactly right; `defect` will not be raised in that case.

First name the entity or quantity the question is actually asking about -- the one it describes but
does not name -- then quote the passage sentence that states it, then give the answer.

`answer` is graded by exact string match against a short reference span, so it must be the bare
answer and nothing else: never a sentence, no explanation, no quotation marks, no final period.
Copy it verbatim from `passages`, in the shortest form that fully names it. If the question is a
yes/no question, `answer` must be exactly `yes` or `no`. Never answer "unknown" or "unclear"; if the
evidence is thin, give the single most likely bare answer."""


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
    "rejected_answer": "an answer that failed an automatic admissibility check",
    "defect": "which check the rejected answer failed, and why that makes it inadmissible",
    "asked_for": "the entity or quantity the question describes but does not name",
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


# --- source-article voicing ------------------------------------------------------------------
# A Wikipedia-style abstract opens by naming its subject and then breaks into a parenthesis, a
# comma, a quoted nickname or a copula.  Everything before that break is the article's own naming
# clause for itself; it is the form a reference span uses when the question asks for that subject.
_LEAD_STOP = re.compile(
    r"\s*(?:\(|,|\"|\u201c|\u201d|\u2018|\u2019\s|\bor\b|\bis\b|\bwas\b|\bare\b|\bwere\b|\bserves\b|\brefers\b)"
)
# a pure name: capitalised tokens, initials and at most short lowercase nobiliary particles
_PURE_NAME = re.compile(r"^[A-Z][A-Za-z'\u2019.\-]*(?: (?:[A-Z][A-Za-z'\u2019.\-]*|[a-z]{2,4}))*$")


def _naming_clause(text: str) -> str:
    """The clause an article opens with, or "" when the opening is not a bare name."""
    stop = _LEAD_STOP.search(text or "")
    head = (text[: stop.start()] if stop else str(text or "")).strip()
    if not head or len(head.split()) > 6 or not _PURE_NAME.match(head):
        return ""
    return head


def _voice_through_source(answer: str, pool: list[str]) -> str:
    """Restate `answer` in the wording of the article that is ABOUT it, when the gate allows.

    The gate is deliberately narrow: same final word, compatible first word, strictly more words,
    and a naming clause that is nothing but a name.  Anything looser was measured to lose.
    """
    norm = normalize_text(answer)
    if not norm or norm in ("yes", "no") or len(norm.split()) < 2:
        return answer
    for passage in pool:
        title, sep, body = str(passage).partition("|")
        if not sep or normalize_text(title) != norm:
            continue  # not the article about the answer
        lead = _naming_clause(body.strip())
        if not lead:
            return answer
        lead_toks, ans_toks = lead.split(), str(answer).split()
        if len(lead_toks) <= len(ans_toks):
            return answer
        if normalize_text(lead_toks[-1]) != normalize_text(ans_toks[-1]):
            return answer  # a different entity, or a type word glued on ("Rosati's Pizza")
        first_lead, first_ans = normalize_text(lead_toks[0]), normalize_text(ans_toks[0])
        if not first_lead or not first_ans:
            return answer
        if first_lead[:3] != first_ans[:3]:
            return answer  # not the same given name written out (Ron -> Ronald is; Bob -> James is not)
        return lead
    return answer


# --- admissibility audit -----------------------------------------------------------------------
# Two label-free tests on the produced string.  Both were measured on 300 replayed items of the
# parent; each names a bucket that scores well below the system average, and each has a defect
# message precise enough for a retry to act on.
_CLOSED_LIST_Q = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|\beither\b|^\s*between\b", re.IGNORECASE)
# A refusal dressed as an answer.  Measured on the same 300 replayed items: 5 occurrences, 0 correct
# -- the one bucket where a retry cannot possibly lose, since the string it replaces scores zero.
_NON_ANSWER = re.compile(
    r"^(unknown|unclear|not stated|not specified|not mentioned|not provided|not given|not known"
    r"|cannot be determined|undetermined|none|n/?a|no answer|not related|nothing)$", re.IGNORECASE)


def _supported(answer: str, pool: list[str]) -> bool:
    """Is the answer written, word for word, somewhere in the passages the reader was shown?"""
    a = normalize_text(answer)
    if not a:
        return True
    haystack = " " + normalize_text(" ".join(pool)) + " "
    return (" " + a + " ") in haystack


def _echoes_question(answer: str, question: str) -> bool:
    """Is the answer a span the question already supplies?"""
    a = normalize_text(answer)
    if not a:
        return False
    return (" " + a + " ") in (" " + normalize_text(question) + " ")


def audit_answer(answer: str, question: str, pool: list[str]) -> str:
    """Return the defect message for an inadmissible answer, or "" when the answer passes.

    Order matters: novelty is the sharper test, so it is reported first when both fire.
    """
    a = normalize_text(answer)
    if not a or _NON_ANSWER.match(a.strip()):
        return ("refusal -- this is not an answer at all; the question has one, and an answer that "
                "declines to name it scores nothing, so any concrete candidate is strictly better")
    if a in ("yes", "no"):
        return ""  # a polar answer is neither copied from the passages nor new information
    if _echoes_question(answer, question) and not _CLOSED_LIST_Q.search(question or ""):
        return ("novelty -- this string is already written in the question itself, and the question "
                "does not offer a closed list of alternatives to choose between, so repeating its "
                "own words restates what was given instead of supplying what was asked")
    if not _supported(answer, pool):
        return ("support -- this string does not occur anywhere in the passages, so nothing that "
                "was retrieved actually states it")
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
        self.retry_answer = Module(
            inputs=["question", "rejected_answer", "defect", "facts", "passages"],
            outputs=["asked_for", "evidence_quote", "answer_type", "answer"],
            instructions=RETRY_INSTRUCTIONS,
            descs=DESCS,
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
        projected = project_answer(raw, out.get("answer_type", ""), question)
        answer = _voice_through_source(projected, pool)
        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300],
                 projected=projected, voiced=answer)

        # 5. audit the produced string; a clean answer ends the item here, exactly as the parent
        defect = audit_answer(answer, question, pool)
        if not defect:
            return {
                "answer": answer,
                "raw_answer": raw,
                "answer_type": out.get("answer_type", ""),
                "query_a": question,
                "next_query": next_query,
                "missing_target": str(gap.get("missing_target", "")),
                "facts": facts,
                "defect": "",
            }

        # 6. one retry, told which test failed.  The replacement is accepted only if it clears the
        #    test the first answer failed -- otherwise the first answer stands, so a failed retry
        #    costs a call and nothing else.
        again = self.retry_answer(
            ctx.llm,
            tag="retry",
            question=question,
            rejected_answer=answer,
            defect=defect,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw2 = again.get("answer", "")
        cand = _voice_through_source(project_answer(raw2, again.get("answer_type", ""), question), pool)
        new_defect = audit_answer(cand, question, pool)
        if defect.startswith("refusal"):
            # the rejected string is worth zero, so the only bar a replacement must clear is
            # being a different string that is not itself a refusal
            accepted = bool(cand) and normalize_text(cand) != normalize_text(answer) and not new_defect.startswith("refusal")
        else:
            accepted = bool(cand) and normalize_text(cand) != normalize_text(answer) and not new_defect
        ctx.note("audit", defect=defect.split(" --")[0], rejected=answer, retry=cand, accepted=accepted)
        return {
            "answer": cand if accepted else answer,
            "raw_answer": raw2 if accepted else raw,
            "answer_type": (again if accepted else out).get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
            "defect": defect,
            "first_answer": answer,
        }
