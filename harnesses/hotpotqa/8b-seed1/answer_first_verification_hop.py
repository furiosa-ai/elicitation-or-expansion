"""Aim the second lookup at the ANSWER, not at the missing article: hypothesise, then retrieve to verify.

Measured motivation (replay over the frontier's search-set traces; the index is deterministic so every
alternative query below was actually issued and scored):

    what the second lookup was aimed at                      gold_title_recall
    the missing article, as the model described it (parent)              0.785
    the model's `missing_target` field used verbatim                     0.760
    just the name inside `missing_target`                                0.707
    the parent query with already-held title tokens removed              0.750

Every reformulation that keeps aiming at the *missing article* loses, and the parent is already the best of
them, so the article-directed second hop is saturated at this depth. Meanwhile 42 of its 93 errors are items
whose reference span occurs nowhere in the pool, and its second lookup spends 1.8 of its 7 slots refetching
articles it already holds.

Mechanism under test: turn the second lookup into a verification of a guess. The single reading call
resolves the entity the question only describes and commits to a `provisional_answer` -- from the passages
if they state it, otherwise from its own knowledge of the world, which is the only source available for the
42. A deterministic compiler builds the second query out of that answer, the resolved entity's name and the
question's own leftover content words, so the query is the *conjunction a confirming article would satisfy*:
if the guess is right, BM25 lands the article that contains both the entity and the answer and the guess
comes back grounded; if it is wrong, the answer terms simply match nothing and the entity and question terms
still carry the lookup. The final call is told the hypothesis is a hypothesis and not evidence, and must
either keep it (if a passage bears it out) or replace it with what the passages support.

Budget: 2 LLM calls, 2 retrievals, k=7 -- identical to the parent, with the second query rebuilt.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PROBE_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the question
itself, and they are already in hand. Usually they cover only ONE of the two articles this question needs.
Exactly one more lookup is left, and it will be used to CHECK your best answer, so commit to one now.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the question, each
prefixed with the article title in brackets. Copy numbers, dates and full names exactly. If nothing is
relevant, write "none".

`resolved_entity`: the question describes some entity indirectly ("the singer of X", "the film that Y
starred in", "the town near Z"). Name it. Prefer a name a passage above supplies; if no passage does, and
you know the entity, name it from your own knowledge; if you truly cannot name it, copy the question's own
description of it. One name, no explanation.

`provisional_answer`: your best answer to the question as asked, right now, in the bare form a reference
answer would take -- a name, a title, a place, a date, a number, or `yes`/`no` for a yes/no question. Read
it out of `passages` if they state it. If they do not, answer it from your own knowledge of the world about
`resolved_entity`: a lookup that checks a guess is worth far more than a blank, and there is no third
lookup coming. Never write "unknown", "not stated" or "cannot be determined", and never answer with an
entity the question itself already named -- that is never what a question asks for.

`property_words`: the two or three words naming what the question asks FOR (for example `born year`,
`director`, `altitude metres`, `nickname`, `population`) -- the words the confirming article would use
around the answer. Words only."""


VERIFY_INSTRUCTIONS = """Answer the question from the evidence. `facts` are the sentences a first pass
extracted, `passages` are the raw article openings retrieved for this question, and `hypothesis` is the
answer that first pass committed to before the last lookup was made.

The `hypothesis` is a guess, not evidence. The lookup you now have was aimed at checking it.
- If a passage bears the hypothesis out, keep it, spelled exactly as that passage spells it.
- If the passages support a different answer, give theirs and ignore the hypothesis.
- If the passages neither confirm nor contradict it, keep the hypothesis rather than inventing a third
  answer or refusing: it is the best available estimate.
Say which of these three happened in `verdict`: exactly one of `confirmed`, `corrected`, `unchecked`.

First quote the single piece of evidence that settles the question (write "none" if there is none), then
declare the shape of the answer, then give the answer itself.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer` is graded by exact string match against a short reference span, so it must be the bare answer and
nothing else:
- Never a sentence. Never restate the question. No explanation, no reason, no "because", no trailing
  clause, no second sentence, no markdown, no quotation marks, no final period.
- If `answer_type` is yes_no, `answer` must be exactly `yes` or `no` -- one word, nothing after it.
  Questions of the form "are both ...", "do both ...", "is X the same as Y", "can both ..." are yes_no
  questions, even when the interesting content is in the reasoning.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning option
  exactly as the question spells it, and nothing else.
- Otherwise copy the answer span verbatim from the evidence, in the shortest form that fully names it:
  just the name, the title, the place, the date or the number.
- Do not add appositives, descriptors, units of explanation or parentheses that the reference span would
  not contain.
- Never answer "unknown", "not stated" or "cannot be determined", and never answer with an entity the
  question itself already named."""


DESCS = {
    "question": "the question to answer",
    "resolved_entity": "the entity the question describes indirectly, named",
    "provisional_answer": "best answer to the question right now, in bare reference form",
    "property_words": "two or three words naming what the question asks for",
    "hypothesis": "the answer committed to before the last lookup -- a guess, not evidence",
    "verdict": "one of: confirmed, corrected, unchecked",
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



_REFUSALS = re.compile(
    r"^(unknown|unclear|none|n/?a|nothing|not\s+\w+|cannot|can't|no\s+\w+)\b", re.IGNORECASE
)


def _query_tokens(text: str) -> list[str]:
    """Content tokens of a field, in order, punctuation dropped and question words removed."""
    out = []
    for w in _WORD.findall(str(text or "")):
        if w.lower() in _QWORDS or len(w) > 30:
            continue
        out.append(w)
    return out


def compile_verify_query(entity: str, answer: str, props: str, question: str, held: list[str]) -> str:
    """The conjunction a CONFIRMING article would satisfy: entity + hypothesised answer + what was asked.

    Order matters only for the length cap, not for BM25 scoring, so the two hypothesis-bearing parts go
    first and the question's leftover content words fill the rest. An answer that is a refusal, a yes/no or
    a phrase the question already contains carries no information about any article and is dropped, so the
    query degrades to an entity-and-question lookup rather than to noise.
    """
    parts: list[str] = []
    seen: set[str] = set()

    def add(tokens: list[str], limit: int) -> None:
        n = 0
        for w in tokens:
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(w)
            n += 1
            if n >= limit:
                return

    ent = str(entity or "").strip()
    if ent and not _REFUSALS.match(ent):
        add(_query_tokens(ent), 6)
    ans = str(answer or "").strip()
    q_norm = _norm(question)
    if (
        ans
        and not _REFUSALS.match(ans)
        and normalize_text(ans) not in ("yes", "no")
        and f" {normalize_text(ans)} " not in q_norm
    ):
        add(_query_tokens(ans), 5)
    add(_query_tokens(props), 3)
    # the question's own content words, minus anything the articles in hand already matched
    held_tokens = {w.lower() for t in held for w in _WORD.findall(str(t))}
    add([w for w in _query_tokens(question) if w.lower() not in held_tokens], 4)
    add(_query_tokens(question), 3)
    return " ".join(parts[:14])


def _norm(text: str) -> str:
    return " " + normalize_text(str(text or "")) + " "

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
        self.probe = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "resolved_entity", "provisional_answer", "property_words"],
            instructions=PROBE_INSTRUCTIONS,
            descs=DESCS,
        )
        self.verify_answer = Module(
            inputs=["question", "facts", "hypothesis", "passages"],
            outputs=["evidence_quote", "verdict", "answer_type", "answer"],
            instructions=VERIFY_INSTRUCTIONS,
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

        # 2. one call: read what is in hand, resolve the described entity and COMMIT to an answer
        probe = self.probe(
            ctx.llm,
            tag="probe",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(probe.get("facts", "")).strip()
        entity = _clean(probe.get("resolved_entity", ""))
        hypothesis = project_answer(probe.get("provisional_answer", ""), "", question)
        props = str(probe.get("property_words", ""))

        # 3. the second lookup checks the guess instead of chasing the missing article
        verify_query = compile_verify_query(entity, hypothesis, props, question, titles_a)
        if len(verify_query.split()) < 2:
            verify_query = _residual_query(question, titles_a)
        docs_b = ctx.tools.retrieve_docs(verify_query, k=self.K)
        titles_b = [str(d.get("title", "")).strip() for d in docs_b]

        # 4. confirm, correct or keep -- the hypothesis is shown, and shown to be a guess
        out = self.verify_answer(
            ctx.llm,
            tag="verify",
            question=question,
            facts=facts if facts else "none",
            hypothesis=hypothesis if hypothesis else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        if not answer or _REFUSALS.match(answer):
            answer = hypothesis or answer  # never return a blank when a committed guess exists
        ctx.note("verification hop", resolved_entity=entity[:120], hypothesis=hypothesis,
                 property_words=props[:80], verify_query=verify_query, new_titles=titles_b[:7])
        ctx.note("typed answer", verdict=out.get("verdict", ""), answer_type=out.get("answer_type", ""),
                 raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "verdict": out.get("verdict", ""),
            "hypothesis": hypothesis,
            "resolved_entity": entity,
            "query_a": question,
            "verify_query": verify_query,
            "facts": facts,
        }
