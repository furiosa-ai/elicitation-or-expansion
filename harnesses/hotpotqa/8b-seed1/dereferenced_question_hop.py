"""Rewrite the question into a self-contained one and use THAT SENTENCE as the second query.

Measured motivation (per-hop attribution over the frontier's own 300 search-set traces):

    query shape used for a hop                 gold articles it recovers
    the raw question (hop A, free)                        0.625
    a written keyword query (hop B)                       0.160 of the remainder

    conversion by where the gold articles came from
    hop A found one, hop B found the other        78 / 86   = 0.907
    hop A found both, hop B found nothing         82 / 102  = 0.804
    hop A found one, hop B found nothing          41 / 85   = 0.482

The one query shape that has ever worked at this depth is a whole natural-language question: every
narrowing of hop B into keywords -- bare name, name+kind card, missing_target verbatim, term
subtraction, answer-conditioned verification -- has scored BELOW the raw question on recall
(0.707, 0.763, 0.760, 0.750, 0.765 vs 0.785 union). A sparse index over article openings scores on
the incidental words a full sentence carries, and a keyword query throws them away.

So hop B should also be asked as a QUESTION. What makes hop B different from hop A is not that it
is shorter; it is that the entity the question only *describes* has by then been *named*. The
mechanism under test replaces the keyword query with de-referencing:

  1. hop A   -- retrieve(question), free, unchanged.
  2. bridge  -- ONE call reads those seven articles and (a) copies the facts, (b) names the entity
                the question refers to indirectly, (c) rewrites the question as a SELF-CONTAINED
                sentence in which that description is replaced by the name and nothing else is
                dropped -- same interrogative, same qualifiers, same incidental words.
  3. hop B   -- retrieve(the rewritten question). Full sentence, not keywords: complementary to hop
                A because the name is new, as rich as hop A because the sentence is kept.
                A deterministic check rejects a rewrite that is the question unchanged or that
                dropped the question's own content words, falling back to the residual query.
  4. answer  -- the frontier's answer contract, but asked the DE-REFERENCED question (the original
                is still shown), so the final call no longer has to resolve the reference itself.

Budget: 2 LLM calls, 2 retrievals, k=7 -- identical to the parent.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

BRIDGE_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the
question itself, and they are already in hand. Usually they cover only ONE of the two articles this
question needs, because a question of this kind mentions one entity by NAME and refers to the other
one only by DESCRIPTION ("the man who founded X", "the film that starred Y", "the band whose singer
...", "the country where Z started"). The passages in hand normally let you put a name to that
description. You get exactly one more lookup.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`resolved_entity`: the name of the entity the question describes but does not name, taken from
`passages` if they name it. Write the name alone, as an article title would spell it. If the
question names every entity it mentions, or the passages do not name it, write "none".

`direct_question`: rewrite the question so that it is self-contained and answerable by ONE article.
- Replace the description with the name from `resolved_entity`; leave everything else in place.
- Keep the same question word and the same asked property, and keep every qualifier and incidental
  word the original question carried (dates, places, adjectives, titles in quotes, nationalities).
  Do NOT compress it into keywords: it must read as a full question sentence.
- Drop only the clause that described the entity you just named.
- If `resolved_entity` is "none", instead rewrite the question so that it asks about the entity the
  passages in hand do NOT cover, keeping all of that entity's qualifying words.
- Never write the original question back unchanged, and never write a bare name."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence: `facts` are the sentences already
extracted, `passages` are the raw article openings retrieved for this question.

`question` has already had any indirect reference resolved to a name, so answer it as written.
`original_question` is how it was first asked -- consult it only for the property being asked and
for any closed list of options it offered; if the two disagree about what is being asked, follow
`original_question`.

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
    "original_question": "the question as it was originally asked",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "titles_in_hand": "titles of the articles already retrieved -- do not fetch these again",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "resolved_entity": "the name of the entity the question only described, or none",
    "direct_question": "the same question rewritten as a self-contained sentence naming that entity",
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


def _as_question_query(text, question, fallback):
    """Accept a rewritten QUESTION as the query; reject a non-rewrite or a keyword stub.

    Unlike the parent, length is not trimmed away: a full sentence is the point of the mechanism.
    A rewrite is rejected only when it cannot be complementary -- it repeats the question, it is
    too short to be a question at all, or it kept none of the question's own content words (which
    would mean the model wrote about something else entirely).
    """
    q = str(text or "").strip().replace("\n", " ")
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(rewritten|direct)?\s*(question|query)\s*:\s*", "", q, flags=re.IGNORECASE)
    words = q.split()
    if len(words) > 30:
        q = " ".join(words[:30])
    if len(q) < 8 or _DECLINED.match(q):
        return fallback, "declined"
    if normalize_text(q) == normalize_text(question):
        return fallback, "unchanged"
    kept = {w.lower() for w in _WORD.findall(q)} & {
        w.lower() for w in _WORD.findall(question) if w.lower() not in _QWORDS
    }
    if not kept:
        return fallback, "ungrounded"
    return q, "rewrite"


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
        self.bridge = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "resolved_entity", "direct_question"],
            instructions=BRIDGE_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "original_question", "facts", "passages"],
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

    def run(self, item, ctx: RunContext):
        question = item["question"]

        # 1. the question itself is the first query -- broadest available, and costs no call
        docs_a = ctx.tools.retrieve_docs(question, k=self.K)
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]

        # 2. one call: copy the evidence, name the entity the question only described, and rewrite
        #    the question around that name
        br = self.bridge(
            ctx.llm,
            tag="bridge",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(br.get("facts", "")).strip()
        entity = _clean(br.get("resolved_entity", ""))
        if _DECLINED.match(entity or "none"):
            entity = ""
        residual = _residual_query(question, titles_a)
        direct, status = _as_question_query(br.get("direct_question"), question, residual)

        # the rewrite is only complementary if the resolved name actually reached it; if the model
        # named an entity but left it out of the sentence, put it in front deterministically
        if status == "rewrite" and entity:
            have = {w.lower() for w in _WORD.findall(direct)}
            if not {w.lower() for w in _WORD.findall(entity)} <= have:
                direct = f"{entity} {direct}"
                status = "rewrite+name"

        # 3. second hop, asked as a full self-contained question
        docs_b = ctx.tools.retrieve_docs(direct, k=self.K)

        # 4. answer the DE-REFERENCED question over the deduplicated raw pool; the original is
        #    still shown so the asked property and any closed list of options stay visible
        asked = direct if status.startswith("rewrite") else question
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=asked,
            original_question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("dereference", resolved_entity=entity[:120], status=status, direct_question=direct[:300])
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": direct,
            "resolved_entity": entity,
            "rewrite_status": status,
            "facts": facts,
        }
