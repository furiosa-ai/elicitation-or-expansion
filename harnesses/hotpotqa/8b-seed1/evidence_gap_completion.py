"""Give the program a policy for the state it is actually in most often: evidence that does not settle it.

Measured motivation (replay over the frontier's own search-set traces; the index is deterministic so the
pools were rebuilt exactly):

    state of the evidence pool                                     items   accuracy
    both target articles retrieved                                   189      0.852
    one retrieved                                                     93      0.484
    neither retrieved                                                 18      0.056

    of the 93 errors:  the reference span is NOWHERE in the pool      42
                       the span is in the pool but not returned       23
                       both articles present, surface/reasoning loss  28

So the single largest bucket is items whose answer no retrieved passage contains, and the frontier has no
policy for them: one prompt is asked to answer whether or not the evidence supports an answer, and under a
pool of fourteen passages that mostly discuss the *other* hop it either refuses ("unknown", "not stated",
6 items) or hands back an entity the question already named (14 items -- the band instead of its singer,
the genus name instead of the genus).  Its own instruction to "still give the single most likely bare
answer" is issued in the same breath as "copy the answer span verbatim from the evidence", and the passages
win.

Mechanism under test (answer policy; retrieval is byte-identical to the parent -- same first query, same
gap module, same second query -- so the comparison isolates it):
  the answering call now REPORTS the state of its evidence (`evidence_status`) and, when the evidence stops
  short, writes the leftover question as a self-contained one naming the entity it resolved
  (`residual_question`).  A deterministic gate fires only on the failure signatures above -- a refusal, an
  empty answer, an answer that merely echoes an entity the question supplied, or a "partial" status whose
  answer occurs nowhere in the pool (i.e. the grounded call was already guessing from memory while looking
  at fourteen distracting passages).  When it fires, one more call answers the residual question CLOSED
  BOOK: no passages at all, only the resolved facts and the provisional answer to revise.  Recall from
  parametric memory is what that state calls for, and it is the one thing the frontier never asks for
  cleanly.

Budget: 2 LLM calls when the evidence settles the question, 3 when it does not; 2 retrievals, k=7.
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
  "unknown", "not stated" or "cannot be determined".

Two more fields report the state of the evidence, so that a second pass can be spent where it is needed.

`evidence_status` must be exactly one of:
- `complete`  -- a passage above states the answer itself, and `evidence_quote` is that statement.
- `partial`   -- the passages establish part of the chain (they identify the entity the question describes
                 indirectly, or one side of a comparison) but none of them states the answer.
- `absent`    -- the passages do not bear on the question at all.
Judge this by what the passages SAY, not by how confident you feel. If you had to supply the answer from
your own knowledge of the world rather than read it above, the status is `partial` or `absent`.

`residual_question`: the part of the question the passages leave unanswered, rewritten as one
self-contained question that names every entity explicitly instead of describing it -- substitute the names
the passages resolved for the question's descriptions ("the singer of <song>" becomes the singer's actual
name if a passage gave it). If `evidence_status` is `complete`, write "none"."""


MEMORY_INSTRUCTIONS = """A full-text lookup was already tried and the articles it returned do not settle
this question, so there is nothing left to read. Answer `residual_question` from your own knowledge of the
world.

`facts` are the sentences the lookup did establish -- treat them as fixed, and do not contradict them; they
usually identify the entity `residual_question` is about. `provisional_answer` is what a first pass said
while it was looking at unrelated passages: treat it as a hypothesis, not as evidence. Keep it only if your
own knowledge agrees with it; replace it when you know better; it is often the entity the question already
named rather than the thing the question asks for, and that is never the answer.

`known_fact`: state, in one sentence, the fact about the named entity that settles the question -- the date,
the place, the person, the number, the title. If several facts compete, pick the one you are most sure of.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer` is graded by exact string match against a short reference span, so it must be the bare answer and
nothing else: no sentence, no explanation, no hedging, no quotation marks, no final period. A name, a title,
a place, a date or a number.
- If the question offers a closed list of options, copy the winning option exactly as the question spells it.
- Give the most specific form you are sure of, and no descriptors around it.
- You must commit. "Unknown", "not stated", "unclear" and "cannot be determined" are all wrong answers, and
  so is naming an entity the question itself supplied. If you are unsure, name the single most likely
  candidate anyway."""

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
    "evidence_status": "one of: complete, partial, absent",
    "residual_question": "the unanswered part of the question, self-contained and with every entity named",
    "known_fact": "the fact from your own knowledge that settles the question, in one sentence",
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



_REFUSAL = re.compile(
    r"^(unknown|unclear|none|n/?a|nothing|no\s+(information|answer|data)|"
    r"not\s+(stated|specified|mentioned|given|provided|available|known|determined|clear|found|listed|related)|"
    r"cannot|can\s*not|can't|insufficient|unspecified|unanswerable)\b",
    re.IGNORECASE,
)


def _norm_words(text: str) -> str:
    return " " + normalize_text(str(text or "")) + " "


def needs_memory(answer: str, status: Any, question: str, pool_norm: str) -> str:
    """Deterministic gate: the failure signatures the parent's own traces show, and nothing else.

    Returns the reason it fired, or "" when the grounded answer stands. Every branch corresponds to a
    counted bucket, so the extra call is spent only where the evidence-first policy is known to break.
    """
    ans = str(answer or "").strip()
    if not ans:
        return "empty"
    if _REFUSAL.match(ans):
        return "refusal"
    norm = normalize_text(ans)
    if norm in ("yes", "no"):
        return ""  # yes/no answering is already reliable; leave it alone
    state = str(status or "").strip().lower()
    if not state.startswith("complete"):
        # The call has itself declared that no passage states the answer, so whatever it wrote is not a
        # grounded span: either memory with fourteen distractors in front of it, or -- the counted
        # failure -- an entity the passages merely NAME, which is the subject of the leftover question
        # rather than its answer.  `in_pool` records which, for the trace.
        return "unsupported" if norm and f" {norm} " in pool_norm else "absent-span"
    if norm and f" {norm} " in _norm_words(question) and not _CLOSED_LIST.search(question or ""):
        return "echoes-question"  # the answer is an entity the question supplied, not the thing asked for
    return ""

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
            outputs=["evidence_quote", "answer_type", "answer", "evidence_status", "residual_question"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )
        self.complete_from_memory = Module(
            inputs=["question", "residual_question", "facts", "provisional_answer"],
            outputs=["known_fact", "answer_type", "answer"],
            instructions=MEMORY_INSTRUCTIONS,
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
        residual_q = _residual_query(question, titles_a)
        # fallback chain: the written query, else the named target, else the residual question
        target = _as_query(gap.get("missing_target"), residual_q)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = residual_q  # a repeat of the question returns the seven articles in hand
        if normalize_text(next_query) in {normalize_text(t) for t in titles_a}:
            next_query = target  # aim it at the target, not at an article already in hand

        # 3. second hop, aimed at the missing article
        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        # 4. answer over the deduplicated raw pool (contract identical to the parent, plus a state report)
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
        status = out.get("evidence_status", "")
        residual = str(out.get("residual_question", "")).strip()

        # 5. only where the evidence-first policy is known to break: answer the leftover question
        #    closed book, with no passages in front of the model at all
        pool_norm = _norm_words(" ".join(pool))
        reason = needs_memory(answer, status, question, pool_norm)
        source = "evidence"
        if reason:
            sub = residual
            if not sub or _DECLINED.match(sub) or len(sub) < 8:
                sub = question  # no usable residual: ask the whole question from memory
            mem = self.complete_from_memory(
                ctx.llm,
                tag="memory",
                question=question,
                residual_question=sub,
                facts=facts if facts else "none",
                provisional_answer=answer or "none",
            )
            mem_raw = mem.get("answer", "")
            mem_ans = project_answer(mem_raw, mem.get("answer_type", ""), question)
            keep = bool(mem_ans) and not _REFUSAL.match(mem_ans) and len(mem_ans.split()) <= 12
            if keep and reason == "echoes-question":
                # an answer that still echoes the question has not moved off the failure signature
                keep = f" {normalize_text(mem_ans)} " not in _norm_words(question)
            if keep:
                answer, source = mem_ans, "memory"
            ctx.note("memory completion", reason=reason, residual=sub[:200],
                     known_fact=str(mem.get("known_fact", ""))[:200], mem_answer=mem_ans, kept=keep)

        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual_q)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300],
                 projected=answer, evidence_status=status, source=source)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "evidence_status": status,
            "answer_source": source,
            "gate_reason": reason,
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }
