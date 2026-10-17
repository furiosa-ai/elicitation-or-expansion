"""Aim the second lookup with a phrase COPIED from the evidence, whenever the composed query names nobody.

Measured motivation (replayed over the frontier's 300 search-set traces; no new runs, pure post-hoc counts):

    items where the first lookup found exactly one of the two needed articles      171
    of those, the second lookup recovered the other one                             86  -> 0.895 accuracy
    of those, it did not                                                            85  -> 0.494 accuracy

So the whole remaining headroom sits in those 85 items, and they have a signature. Tokenising each
written second query and asking whether it names ANY entity that is not already an article in hand:

    query names a new entity        128 items, second lookup completes the pair  78  (60.9%)
    query names nobody               43 items, second lookup completes the pair   8  (18.6%)

The 43 are queries like "college located city district", "female artist", "son feudal prince
country", "genre band released cover album" -- the reader aimed at the asked PROPERTY, and a sparse
index over article openings cannot score a property phrase against an opening. Yet in 30 of the 85
failures the article that is missing is named verbatim inside the reader's OWN `facts` digest: it
read the bridge and then did not search for it.

Mechanism under test (query formulation only): in that measured state -- and only there -- the
query stops being composed and becomes a QUOTE. The digest is parsed back into (title, sentence)
pairs, the entity mentions it introduces are collected, those already covered by an article in hand
are dropped, the survivor that the reader's own `missing_target` points at is chosen, and the query
becomes a contiguous window of the sentence that mentions it -- the words another article's opening
would itself contain. A blanket version of this was tested first and refuted (replaying it on every
item loses the target name on 33 lookups that currently find it to gain 13), which is exactly why
the mechanism is gated on the content-free state instead of replacing the reader's aim.

Both prompts, both LLM calls, the pool, the answer contract and every projection rule are byte for
byte those of `source_named_answer`, so any delta is a second-lookup delta.

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


# --- evidence-copied second query ---------------------------------------------------------------
# The reader writes its digest back in the shape it received the passages: «Title | sentences».
# That is enough structure to recover which article each sentence came from, and therefore which
# entity mentions are NEW -- introduced by an article in hand but not covered by one.
_FACT_BLOCK = re.compile(r"\u00ab(.*?)\u00bb", re.DOTALL)
_BRACKET_BLOCK = re.compile(r"\s*\[([^\]]{1,90})\]\s*(.*)", re.DOTALL)
_CLAUSE = re.compile(r"(?<=[a-z0-9\)\]\"'])[.;]\s+(?=[A-Z0-9(])")
_MENTION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Z][\w'\u2019.\-]*|\d{4})"
    r"(?:\s+(?:of|the|and|de|van|von|di|du|da|for|in|at|on|&)\s+[A-Z][\w'\u2019.\-]*"
    r"|\s+[A-Z][\w'\u2019.\-]*|\s+\d{4})*"
)
_FILLER = _QWORDS | {
    "also", "during", "after", "before", "one", "two", "three", "who", "which", "what", "when",
    "where", "including", "such", "other", "another", "many", "some", "each", "only", "later",
}


def _tokset(text: Any) -> set:
    return {t.lower() for t in _WORD.findall(str(text or ""))}


def _digest_blocks(facts: str) -> list:
    """Recover (article title, sentence) pairs from the digest the reader wrote."""
    blocks = _FACT_BLOCK.findall(facts or "")
    if not blocks:
        blocks = [ln for ln in str(facts or "").splitlines() if ln.strip()]
    out = []
    for block in blocks:
        title, sep, body = block.partition("|")
        if sep:
            out.append((title.strip(), body.strip()))
        else:
            m = _BRACKET_BLOCK.match(block)
            out.append((m.group(1).strip(), m.group(2).strip()) if m else ("", block.strip()))
    return out


def _names_someone_new(query: str, titles: list) -> bool:
    """True when the query mentions an entity that is not already an article in hand."""
    held = set()
    for t in titles:
        held |= _tokset(t)
    for m in _MENTION.finditer(str(query or "")):
        toks = [w.lower() for w in _WORD.findall(m.group(0))]
        if any(w not in held and w not in _FILLER for w in toks):
            return True
    return False


def _copied_evidence_query(facts: str, titles: list, target: str) -> str:
    """A contiguous span of the digest, centred on the uncovered mention the reader aimed at."""
    held = set()
    for t in titles:
        held |= _tokset(t)
    aim = _tokset(target)
    if not aim:
        return ""
    best = None
    for title, body in _digest_blocks(facts):
        local = held | _tokset(title)
        for sentence in _CLAUSE.split(body):
            for m in _MENTION.finditer(sentence):
                cand = m.group(0).strip(" ,.;:'\"")
                toks = [w.lower() for w in _WORD.findall(cand)]
                if not toks or len(toks) > 6:
                    continue
                if all(w in local or w in _FILLER for w in toks):
                    continue  # this article, or a filler capitalisation
                if len(toks) == 1 and (toks[0] in _FILLER or m.start() == 0):
                    continue  # a sentence-initial word is not evidence of a name
                overlap = len(set(toks) & aim) / len(toks)
                if overlap <= 0:
                    continue  # the reader is not pointing at this entity
                novel = sum(1 for w in toks if w not in local)
                score = (overlap, novel, len(toks))
                if best is None or score > best[0]:
                    best = (score, cand, sentence, m.start(), local)
    if best is None:
        return ""
    _, cand, sentence, pos, local = best
    words = sentence.split()
    head = len(sentence[:pos].split())
    lo, hi = max(0, head - 3), min(len(words), head + len(cand.split()) + 3)
    cand_toks = _tokset(cand)
    window = []
    for w in words[lo:hi]:
        raw = _WORD.findall(w)
        if not raw:
            continue
        low = raw[0].lower()
        if low not in cand_toks and (low in local or low in _FILLER):
            continue  # keep the window free of the article we already hold
        window.append(w.strip(" ,.;:\"'()[]\u00ab\u00bb"))
    if not any(_tokset(w) & cand_toks for w in window):
        window = cand.split()
    return " ".join(window[:9]).strip()


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

        # THE MECHANISM: a query that names nobody new completes the pair 18.6% of the time against
        # 60.9% for one that does.  In that state only, stop composing and copy the evidence.
        copied = ""
        if not _names_someone_new(next_query, titles_a):
            copied = _copied_evidence_query(
                facts, titles_a, f"{gap.get('missing_target', '')} {gap.get('next_query', '')}"
            )
            if copied and _names_someone_new(copied, titles_a):
                next_query = copied

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
                 next_query=next_query, residual=residual, copied_query=copied)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300],
                 projected=projected, voiced=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }
