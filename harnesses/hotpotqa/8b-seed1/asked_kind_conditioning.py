"""Condition both calls on the KIND of thing the question asks for, parsed from the question itself.

Measured motivation (the frontier's 300 search-set traces, replayed):

    errors whose reference string is nowhere in the answering prompt        43 / 89
    of those, the prediction is the intermediate entity the question
        already names or that a passage merely mentions -- a band where a
        singer is asked for, a film where its two stars are asked for,
        an album title where a release date is asked for                     9
    of those, the prediction is a bare refusal ("unknown", "not stated",
        "not mentioned") that the contract already forbids                    5
    a wh-question answered with a polar "no"                                  1

Every one of those is a *kind* error, and the kind is fixed by the question before any passage is
read: "who did the vocals" opens a person slot, "in what theatre" opens a theatre slot, "which two
actors" opens a two-people slot, "on which date" opens a date slot. The frontier does carry an
`answer_type` field, but the model writes it after it has written the answer, so it can only ever
agree with whatever it already chose; and its eight-way enum cannot express "a school of fighting"
or "a type of media", which is what these questions actually ask for.

Mechanism under test (reasoning structure, not retrieval and not answer surface): a deterministic
one-pass parser reads the question's own interrogative phrase and returns the ASKED KIND as a
natural-language noun phrase lifted from the question ("a theatre", "two actors", "a school of
fighting", "a year", "a person"). That string is injected as an INPUT to BOTH existing calls:

  * the gap call is told what kind of thing the answer is, so `missing_target` must be the article
    whose SUBJECT is a thing of that kind -- for "who did the vocals on X" the missing article is
    the singer's, not the band's -- and `next_query` is aimed at that subject;
  * the answering call is told the asked kind before it reads anything, with the rule that an
    intermediate entity of a different kind is not an answer even when it is the only name the
    passages offer, and that a guess of the right kind beats a refusal.

No output field is added to either call, no call is added, retrieval and the pool are untouched, and
the answer projection is the parent's, so the delta is confined to what the two calls are steered
toward. This differs from a post-hoc admissibility test on the finished answer: the constraint is
present while the evidence is being read and while the second lookup is being aimed, which is the
only point at which a wrong-kind read can still be avoided rather than patched.

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
`asked_kind` says what kind of thing the answer is. The article you name here is normally the
article whose SUBJECT is a thing of that kind, not the article that merely mentions it: if the
answer must be a person, name the person's article, not the group, work or company they belong to;
if the answer must be a date or a number, name the article about the thing that HAS that date or
number. Naming the intermediate entity again is the commonest way this lookup is wasted.
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
- Write the words that an article about a thing of kind `asked_kind` would open with.
- Drop question words ("which", "who", "both", "same", "more", "first") and punctuation.
- If the question named two entities and only one was retrieved, this query is for the other one.
- Search terms only: no prose, no explanation, no quotation marks, at most about ten words."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence: `facts` are the sentences already
extracted, `passages` are the raw article openings retrieved for this question.

`asked_kind` is what kind of thing the answer has to be. It is read off the question's own wording,
so it is not a hint and not a preference -- it is what the question is asking for, and a string that
is not a thing of that kind is not an answer to this question. Before you quote anything, hold it in
mind and look for a thing of that kind.
- The passages usually put an intermediate entity in your way: the group a singer sings in, the film
  its actors star in, the record a song sits on, the company a founder founded. That entity is the
  route to the answer, never the answer, unless it is itself a thing of kind `asked_kind`.
- If the passages name the intermediate entity but not the thing of kind `asked_kind`, you must
  still return a thing of that kind: use what you know about the subject the passages have now
  identified. A wrong guess of the right kind and a refusal both score zero, so never spend the
  answer on the intermediate entity and never decline.

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
    "asked_kind": "the kind of thing the answer must be, read off the question's own wording",
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




# --- the asked kind, parsed from the question -------------------------------------------------
# A wh-question names the kind of thing it wants in its own interrogative phrase: the head noun
# after "which"/"what", or the fixed slot that "who", "when", "where" and "how many" open.  The
# phrase is lifted verbatim so that kinds the eight-way enum cannot express -- "a school of
# fighting", "a type of media", "two actors" -- survive intact.

_POLAR_START = re.compile(
    r"^\s*(are|is|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", re.IGNORECASE)
_WH_ANY = re.compile(r"\b(which|what|who|whom|whose|when|where|why|how)\b", re.IGNORECASE)
# words that close a noun phrase: finite verbs, relativisers and prepositions other than "of"
_NP_STOP = {
    "is", "are", "was", "were", "be", "been", "being", "did", "do", "does", "has", "have", "had",
    "that", "which", "who", "whom", "whose", "when", "where", "why", "how", "and", "or", "but",
    "in", "on", "at", "for", "from", "by", "with", "to", "as", "than", "into", "about", "between",
    "starred", "played", "wrote", "directed", "founded", "released", "born", "won", "made",
    "produced", "created", "published", "starring", "featured", "appeared", "came", "does",
}
_NP_KEEP_LINK = {"of"}
_COUNT_WORD = re.compile(r"^(two|three|four|both)$", re.IGNORECASE)
_YEAR_HEAD = re.compile(r"^(year|decade|century)s?$", re.IGNORECASE)


def _noun_phrase_after(tokens: list[str], start: int) -> str:
    """The noun phrase a `which`/`what` opens, at most six tokens, "of" allowed to continue it."""
    out: list[str] = []
    i = start
    while i < len(tokens) and len(out) < 6:
        low = re.sub(r"[^A-Za-z0-9'\-]", "", tokens[i]).lower()
        if not low:
            break
        if low in _NP_STOP and not (low in _NP_KEEP_LINK and out):
            break
        if low in _NP_KEEP_LINK and (not out or i + 1 >= len(tokens)):
            break
        out.append(tokens[i].strip(",;:?!\"'"))
        i += 1
    return " ".join(w for w in out if w).strip()


def asked_kind_of(question: str) -> str:
    """A natural-language description of the kind of thing the answer has to be."""
    q = str(question or "").strip()
    if not q:
        return "the thing the question asks for"
    low = q.lower()
    if _POLAR_START.match(q) and not _WH_ANY.search(q):
        return "yes or no -- this is a polar question, so the answer is the single word yes or no"
    if re.search(r"\bhow (many|much)\b", low):
        tail = _noun_phrase_after(q.split(), len(re.split(r"\bhow (?:many|much)\b", q, flags=re.IGNORECASE)[0].split()) + 2)
        return f"a number -- a count of {tail}" if tail else "a number"
    if re.search(r"\bhow (long|old|tall|far|big|wide|deep|fast|heavy)\b", low):
        return "a quantity, written with its unit exactly as the evidence writes it"
    if re.search(r"\b(what|which|in what|in which) (year|decade|century)\b", low):
        return "a year"
    if re.search(r"\bwhen\b", low):
        return "a date"
    if re.search(r"\bwhere\b", low):
        return "a place -- the name of a town, city, region, country or venue"
    if re.search(r"\bwhy\b", low):
        return "a reason"
    # the last which/what phrase governs the ask in these questions; else the leading who/whose
    tokens = q.split()
    last: str = ""
    for m in re.finditer(r"\b(which|what)\b", q, re.IGNORECASE):
        idx = len(q[: m.start()].split())
        np = _noun_phrase_after(tokens, idx + 1)
        if np:
            last = np
    if last:
        head = re.sub(r"[^A-Za-z0-9'\- ]", "", last).split()
        if head and _YEAR_HEAD.match(head[-1]):
            return "a year"
        article = "" if (head and _COUNT_WORD.match(head[0])) else "a "
        return f"{article}{last}"
    if re.search(r"\bwho\b|\bwhom\b|\bwhose\b", low):
        return "a person -- the name of a human being, not the name of a group, band, company or work"
    return "the thing the question asks for"


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.gap = Module(
            inputs=["question", "asked_kind", "titles_in_hand", "passages"],
            outputs=["facts", "missing_target", "next_query"],
            instructions=GAP_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "asked_kind", "facts", "passages"],
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
        asked_kind = asked_kind_of(question)

        gap = self.gap(
            ctx.llm,
            tag="gap",
            question=question,
            asked_kind=asked_kind,
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
            asked_kind=asked_kind,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        projected = project_answer(raw, out.get("answer_type", ""), question)
        answer = _voice_through_source(projected, pool)
        ctx.note("asked kind", asked_kind=asked_kind)
        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300],
                 projected=projected, voiced=answer)
        return {
            "answer": answer,
            "asked_kind": asked_kind,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }
