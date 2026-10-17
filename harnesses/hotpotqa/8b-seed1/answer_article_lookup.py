"""The remaining lookup is always spent on a NAMED article, because the planner no longer gets to
judge whether the evidence is already sufficient.

Measured motivation (replayed over the frontier's own 300 search-set traces, so every number is a
post-hoc count of what that system actually did):

    planner writes "none" for the missing article                        154 / 300
    of those, hop A had in fact retrieved only ONE of the two gold
        articles (a wrong decline)                                        55
        ... plus 5 more where it had retrieved NEITHER                     5
    accuracy on items where hop A found exactly one gold article
        when the planner wrote a real query      (n=119)                 0.714
        when the planner declined                (n=52)                  0.635
    the missing gold article is recovered by hop B
        when the planner wrote a real query                              64-81%
        when it declined and the deterministic fallback fired              11%
    accuracy by pool coverage: both target articles 0.868 (n=189),
        one 0.495 (n=93), neither 0.056 (n=18)

Recovering the second target article is worth 0.453 -> 0.912 inside the very bucket the declines
land in, and a decline throws the lookup at a word-subtraction fallback that essentially never
lands it. The decline is not a query-wording problem: it is a sufficiency judgement, made by a call
that has seen one target article and mistakes it for two.

Mechanism under test (query formulation): the planning call is given a task that has no decline
branch. It is told the lookup will be spent no matter what, and asked not whether anything is
missing but WHERE the exact answer span is written -- name that article. Two exhaustive cases keep
the output total unchanged:
  * the article holding the asked fact is not in hand -> name it (from the passages if they name
    it, otherwise from what is known about the subject);
  * everything needed really is in hand -> name the article about the ANSWER ITSELF, because that
    article is what a checker would cite and it is also what supplies the answer's own naming
    clause downstream.
So "nothing is missing" stops being an answer and becomes a routing decision that still produces a
name. The retrieval count, the call count, the pool construction, the answering contract and every
answer projection (including source voicing) are byte-identical to the parent; only the planner's
task and the field that carries its target change.

Budget: 2 LLM calls, 2 retrievals, k=7.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

GAP_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the question
itself, and they are already in hand. Usually they cover only ONE of the two articles this question
needs. Exactly one more lookup will be made, and it will be made no matter what you write -- there
is no way to skip it and no credit for declining it.

So do not decide whether the passages are already enough. Decide only WHERE the exact answer span
to this question is written, and name that article.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`lookup_target`: the name of the ONE article to look up. Never "none", never a question, never a
sentence -- a name. Pick it by these two cases, which between them cover everything:
- The article whose SUBJECT owns the asked fact is NOT in `titles_in_hand`: name that article. An
  entity that a passage merely MENTIONS is not covered by that passage, so if the question asks for
  a property of it -- a date, a place, a number, a person, a title, a genre -- that entity's own
  article is the one to name. If the passages have NAMED it, use their spelling. If they only
  describe it, name it anyway from what you know about the subject.
- Everything the question needs really is in `titles_in_hand`: then name the article about the
  ANSWER ITSELF -- the entity, work, place or person that the answer is -- because that article is
  the one that would confirm it and the one that spells its name in full.

`next_query`: the query for that lookup, scored by a sparse index over article openings, so write
the words that `lookup_target`'s OWN OPENING would contain: its name, plus the two or three words
that pin down what kind of thing it is (its type, field, place, role or date).
- Aim it at `lookup_target`, not at the question. Do not hand the question back.
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
    "lookup_target": "the name of the one article to look up -- always a name, never a refusal",
    "next_query": "keyword query for the second and final lookup, aimed at `lookup_target`",
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


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.gap = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "lookup_target", "next_query"],
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
        # Fallback chain, now anchored on a name rather than on a sufficiency verdict: the written
        # query, else the named target article, else -- only if the model refused to name anything
        # at all, which the contract no longer permits -- the parent's residual question.
        target = _as_query(gap.get("lookup_target"), residual)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = target  # a repeat of the question returns the seven articles in hand
        # No re-route when the query names an article already in hand: under this contract that is
        # case two -- a deliberate confirmation lookup on the article about the answer itself.

        # 3. second hop, aimed at the named target article
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
        ctx.note("aimed lookup", lookup_target=str(gap.get("lookup_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300],
                 projected=projected, voiced=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "lookup_target": str(gap.get("lookup_target", "")),
            "facts": facts,
        }
