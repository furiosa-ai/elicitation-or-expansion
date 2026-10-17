"""Spend the reasoning call AFTER the first lookup, not before it, and aim hop 2 at the gap.

Measured motivation (per-hop attribution of gold-article recall over the search set, read off the
traces of the two systems that already ran):

    first-query shape            hop-1 recall   both target articles in hop 1
    single-entity planned query        0.445                 18 / 300
    the question itself                0.625                102 / 300

A pre-retrieval planning call therefore *costs* recall: rewriting the question into a one-entity
keyword query throws away exactly the incidental words a sparse index scores on, and the planner is
guessing about a second entity it has not seen yet. The reason the planned pipeline still ends up
ahead on union recall is only that its two hops are complementary, while a system that queries with
the question twice gets the same seven articles back.

Mechanism under test (retrieval architecture; the answering stage is held byte-identical to
`decomposed_evidence_pool` so the comparison isolates it):
  1. hop A   -- retrieve(question) immediately, with no LLM call spent on it: the broadest, most
                faithful query available, and free.
  2. gap     -- ONE call reads those seven articles and does the work the planner used to do blind,
                now with evidence in hand: copy the facts that bear on the question, state which
                article is still MISSING, and write the one remaining query for it.  The seven
                titles already in hand are shown to it, so "wasted duplicate" is checkable rather
                than merely forbidden.
  3. hop B   -- retrieve(gap.next_query); a deterministic residual query (the question minus the
                words of the article hop A already matched) replaces it if the model hands the
                question back.
  4. answer  -- the frontier's answer contract over the deduplicated raw pool.

Budget: 2 LLM calls, 2 retrievals, k=7 -- one call *below* the seed program's interaction depth.
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
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
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
            "facts": facts,
        }
