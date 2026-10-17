"""Hop B buys only what hop A did not: an identity card for the missing article, not a property query.

Diagnosis (read off the frontier's own search-set traces).  Its 93 errors split 17 / 48 / 28 across
gold-title recall 0.0 / 0.5 / 1.0, so two thirds of the loss is an article that never arrived.  Of the
82 gold articles the pipeline never retrieved, 24 have a title the question spells out verbatim, and in
13 of those the model DID put that name in its second query -- and the index still did not return the
article, because the name was carrying the asked property with it:

    "San Marino area square kilometers"        target article: San Marino
    "Maroon 5 band members"                    target article: Maroon 5
    "David Edgar publications country"         target article: David Edgar (playwright)
    "Snoop Dogg birth year"                    target article: Snoop Dogg
    "William Kennedy author books genre"       target article: William Kennedy (author)

A sparse index over article OPENINGS has no title prior: an entity's own article competes with every
article that merely mentions it, and property words ("area", "members", "birth year") are exactly the
words its opening does NOT contain, so they hand the seven slots to whatever article is about that
property instead.  Worse, those words came from the question -- and the question was already spent, in
full, as hop A's query.  Re-spending them buys the articles hop A has already shown us.

Mechanism under test (second-hop targeting; the answering stage is byte-identical to the parent, whose
full-recall bucket already converts at 0.852, so the comparison isolates retrieval):

  * the gap call no longer writes a query.  It writes an identity CARD for the article still missing:
    `target_name`, the bare name and nothing else, and `target_kind`, the words that article's own first
    sentence would use to say what kind of thing it is.
  * a deterministic compiler builds the query as name + kind and enforces COMPLEMENTARITY: a kind token
    that already occurs in the question is deleted, because hop A's query was the question, so that term
    has already been paid for.  What survives is exactly the vocabulary hop A could not have used -- the
    resolved name and its type.
  * a saturation guard: if three or more of hop A's seven titles already contain every content token of
    `target_name`, the index has shown everything it ranks for that name and the lookup is spent on the
    residual side of the question instead.

Budget: 2 LLM calls, 2 retrievals, k=7 -- identical to the parent.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

GAP_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the question
itself, and they are already in hand. Usually they cover only ONE of the two articles this question
needs. You get exactly one more lookup, and it is spent on ONE article. Say which article that is.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`target_name`: the NAME of the article still needed, and nothing else -- no properties, no relation,
no explanation, no question words. Write it the way an encyclopedia would title that article.
- The entity the question mentions but no passage above covers, or the entity the question only
  describes indirectly and the passages have now NAMED: if a passage names it, copy that name exactly.
- An entity that a passage merely NAMES is not covered by that passage. If the question asks for a
  property of it -- a date, a place, a number, a person, a title -- then that entity's own article is
  what is missing, and its name is what goes here.
- Never write the property you want to look up. "when was X born", "how many members X has",
  "X's area" are all the same lookup: `target_name` is X.
- If the question named two entities and only one of them has its own article above, this is the other one.
- Only write "none" if the exact answer to the question is already quoted word for word in `facts`;
  a lookup you decline is a lookup thrown away.

`target_kind`: what kind of thing `target_name` is, in the words that article's OWN FIRST SENTENCE
would use -- its type, field, nationality, place, role or era. Three to six words, no punctuation.
Examples of the register wanted: "English rock band", "microstate in southern Europe", "American
attorney and defense lawyer", "1975 Hindi thriller film", "Soviet mathematician".
- Describe what it IS, never what you want to know about it, and never why the question needs it.
- Say it the way the article introduces itself, not the way the question refers to it."""

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
    "target_name": "the bare name of the article still needed -- no properties, no relation",
    "target_kind": "what that article is, in its own first sentence's words: type, field, place, role",
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
# words that name a RELATION to an entity rather than the entity's own type: never identify an article
_RELATION_WORDS = {
    "birth", "birthday", "born", "death", "died", "age", "year", "years", "date", "dates", "day",
    "month", "number", "numbers", "count", "total", "amount", "area", "size", "length", "height",
    "width", "population", "member", "members", "membership", "release", "released", "publication",
    "published", "publish", "founded", "founding", "formed", "location", "located", "career",
    "achievement", "achievements", "legacy", "information", "details", "detail", "article", "page",
    "entry", "record", "records", "value", "figure", "figures", "square", "kilometre", "kilometres",
    "kilometer", "kilometers", "mile", "miles", "name", "names", "title", "titles", "about",
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


_DECLINED = re.compile(r"^(none|n/?a|nothing|no\b.*|not\s|already\b|unknown)", re.IGNORECASE)
_CARD_NOISE = re.compile(
    r"^\s*(the\s+)?(article|wikipedia\s+article|entry|page|passage)\s+(about|on|for|titled)?\s*|"
    r"^\s*(the\s+)?(missing|needed|target)\s+(article|entity|name)\s*(:|is|about|for)?\s*",
    re.IGNORECASE,
)


def _stem(tok: str) -> str:
    """Crude conflation so 'publications'/'published' and 'member'/'members' collide."""
    t = tok.lower()
    for suf in ("ations", "ation", "ings", "ies", "ing", "ers", "er", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t




def _name_of(card: Any) -> str:
    """The bare name from `target_name`: first clause, prose wrappers and trailing gloss removed."""
    n = str(card or "").strip().split("\n")[0]
    n = _CARD_NOISE.sub("", n).strip(_STRIP_CHARS).strip()
    # a name never carries a relative clause, a colon gloss or a parenthetical reason
    n = re.split(r"\s*[:;]\s*|\s+[—–]\s+|\s*\b(?:which|who|whose|that|because|since|as the|in order)\b\s+",
                 n, maxsplit=1, flags=re.IGNORECASE)[0]
    n = re.sub(r"\s*\((?![^()]*\b\d{4}\b)[^()]*\)\s*", " ", n)  # keep "(1981 film)", drop reasons
    n = n.strip(_STRIP_CHARS).strip()
    words = n.split()
    return " ".join(words[:8])


_RELATION_STEMS = {_stem(w) for w in _RELATION_WORDS}


def _kind_tokens(card: Any, question: str) -> list[str]:
    """Type words that the question did NOT already supply -- hop A already spent those."""
    q_stems = {_stem(t) for t in _WORD.findall(question or "")}
    out: list[str] = []
    for tok in _WORD.findall(str(card or "").split("\n")[0]):
        low = tok.lower()
        st = _stem(tok)
        if low in _QWORDS or low in _RELATION_WORDS or st in _RELATION_STEMS:
            continue
        if st in q_stems:
            continue  # complementarity: hop A's query was the question, this term is already paid for
        if st in {_stem(w) for w in out}:
            continue
        out.append(tok)
        if len(out) >= 4:
            break
    return out


def _saturated(name: str, titles: list[str]) -> bool:
    """Has hop A already shown everything the index ranks for this name?"""
    toks = {t.lower() for t in _WORD.findall(name) if t.lower() not in _QWORDS}
    if not toks:
        return False
    hits = sum(1 for t in titles if toks <= {w.lower() for w in _WORD.findall(t)})
    return hits >= 3


def _residual_query(question: str, titles: list[str]) -> str:
    """The question minus the words of the title it already matched best -- a complementary query."""
    q_tokens = list(_WORD.findall(question))
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
            outputs=["facts", "target_name", "target_kind"],
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

    def _compile_query(self, name: str, kind: Any, question: str, titles_a: list[str]) -> tuple[str, str]:
        residual = _residual_query(question, titles_a)
        if len(name) < 2 or _DECLINED.match(name):
            return residual, "declined"
        if normalize_text(name) == normalize_text(question):
            return residual, "handed-back-question"
        if _saturated(name, titles_a):
            return residual, "saturated"          # the index has nothing new under that name
        kind_toks = _kind_tokens(kind, question)
        query = " ".join([name] + kind_toks).strip()
        return (query if len(query) >= 2 else residual), ("name+kind" if kind_toks else "name-only")

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # 1. the question itself is the first query -- broadest available, and costs no call
        docs_a = ctx.tools.retrieve_docs(question, k=self.K)
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]

        # 2. one call: read the evidence, then name the article still missing -- as a card, not a query
        gap = self.gap(
            ctx.llm,
            tag="gap",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(gap.get("facts", "")).strip()
        name = _name_of(gap.get("target_name"))
        next_query, route = self._compile_query(name, gap.get("target_kind"), question, titles_a)

        # 3. second hop: identity terms only, and only the ones hop A could not already have used
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
        ctx.note("entity card", target_name=name, target_kind=str(gap.get("target_kind", ""))[:120],
                 next_query=next_query, route=route)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "route": route,
            "target_name": name,
            "target_kind": str(gap.get("target_kind", "")),
            "facts": facts,
        }
