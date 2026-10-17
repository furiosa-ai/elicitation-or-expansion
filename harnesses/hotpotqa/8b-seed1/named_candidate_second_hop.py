"""Aim the one remaining lookup with NAMES, not with a description of something unnamed.

Measured motivation (read off the frontier system's own 300 search-set traces):

    hop A (the question itself, k=7)   both target articles: 102/300   exactly one: 171/300
    of those 171 single-hit items, hop B recovered the second article for  86
                                    and failed for                        85
    of those 85 failures, the hop-B query shared
        every content word with the missing article's title:  17
        some words:                                           20
        NO word at all:                                       48

Reading the 48: they are all descriptions of an entity the model could not name -- "actress starred
American biographical comedy drama", "genre band released cover album", "type animal use voice TV
commercials", "year new stadium open". A sparse full-text index scores word overlap against article
openings; a paraphrase of an unnamed subject overlaps nothing, so those lookups return nothing
useful. Meanwhile the missing article is very often the answer entity's own article, i.e. exactly
the thing whose NAME is in the model's head even when it is not in the passages.

Mechanism under test (query TYPE, not query wording): the gap call may no longer hand back prose.
It must commit to two or three concrete proper NAMES for the missing article -- taken from the
passages if they name it, otherwise from its own knowledge of the world -- plus the handful of
words that article's own opening would contain. A deterministic compiler then builds the lookup as
a disjunction of those names: a sparse index scores each candidate's own opening on its own rare
name tokens, so a slate of three names costs nothing and buys three chances instead of zero. Only
if nothing name-shaped comes back does the old description/residual chain run.

Everything downstream -- the pool construction and the answering call -- is byte-identical to
`question_first_gap_hop`, so any delta is a retrieval delta.

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
needs. You get exactly one more lookup. Find the gap and aim it -- by NAME.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`missing_article`: one short line naming what the still-missing article is about. There is always a
second article: an entity that a passage merely MENTIONS is not covered by that passage, so if the
question asks for a property of it -- a date, a place, a number, a person, a title -- that entity's
own article is what is missing.

`name_candidates`: the heart of this task. Two or three CONCRETE PROPER NAMES for the missing
article, separated by semicolons -- the actual titles you believe that article carries.
There is no "none" here, and no empty answer. The lookup happens whether or not you aim it, so an
unaimed lookup is simply thrown away. Even when you think `facts` already settles the question,
name the article of the entity the answer refers to, so the lookup confirms the answer instead of
being wasted -- an abstract that merely mentions a work, a person or a place is not that work's,
person's or place's own article, and the detail actually asked for (an exact date, a full name, a
figure, a birthplace) almost always lives in the article you have not fetched.
- If a passage already names the entity, give that name first, spelled exactly as the passage
  spells it.
- If no passage names it, name it from your own knowledge of the world. Which real person, work,
  place, band, film, team, species or event fits the description the question gives? Commit to your
  two or three best guesses even if you are unsure -- three guesses cost the same as one.
- NEVER write a description here. The lookup is word matching against article openings, so
  "the actress who starred in that comedy" or "the genre of the band" matches nothing at all,
  while a wrong name at least matches nothing worse. A name is a name: capitalised words, or a
  year, or a species binomial.
- Do not list an entity whose article is already in `titles_in_hand`; that one is retrieved.

`type_words`: two to five ordinary words that the missing article's OWN OPENING would contain
around those names -- what kind of thing it is: its type, field, nationality, place, role or year
("American actress", "Hindi film 1975", "death metal band", "city Michigan"). No question words,
no punctuation, no prose."""

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
    "missing_article": "one short line: what the still-missing article is about",
    "name_candidates": "two or three concrete proper names for the missing article, semicolon-separated",
    "type_words": "two to five words the missing article's own opening would contain",
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


_NAME_SPLIT = re.compile(r"[;\n]|\s{2,}|\s*\|\s*")
_ENUM = re.compile(r"^\s*(?:[-*\u2022]|\(?\d+[.)])\s*")
_DESCRIPTIVE = re.compile(
    r"^(?:the|a|an|that|this|its|his|her|their|which|who|what|whose|some|any|unknown|unnamed)\b",
    re.IGNORECASE,
)


def _looks_like_name(chunk: str) -> bool:
    """A name is capitalised words, a year, or a binomial -- not a description of something."""
    words = chunk.split()
    if not (1 <= len(words) <= 7):
        return False
    if _DESCRIPTIVE.match(chunk):
        return False
    caps = sum(1 for w in words if re.match(r"[A-Z\u00C0-\u00DC]", w))
    if any(re.fullmatch(r"(1[0-9]|20)\d\d", w) for w in words):
        # a year is as discriminative as a capital: "2016 in video gaming" is a real article title
        return caps + 1 >= 1
    # a proper name is mostly capitalised; a description is mostly lower case
    return caps >= max(1, (len(words) + 1) // 2)


def _name_slate(raw: Any, titles_in_hand: list[str], limit: int = 3) -> list[str]:
    """Parse the model's semicolon slate into at most `limit` name-shaped candidates."""
    text = str(raw or "").strip()
    if not text or _DECLINED.match(text):
        return []
    held = {normalize_text(t) for t in titles_in_hand}
    out: list[str] = []
    for chunk in _NAME_SPLIT.split(text):
        cand = _ENUM.sub("", str(chunk)).strip(_STRIP_CHARS).strip()
        cand = re.split(r"\s+\(|\s+[\u2014\u2013]\s+", cand)[0].strip(_STRIP_CHARS).strip()
        if not cand or not _looks_like_name(cand):
            continue
        n = normalize_text(cand)
        if not n or n in held or any(n == normalize_text(o) for o in out):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _type_terms(raw: Any, budget: int) -> list[str]:
    """The few ordinary words the missing article's opening would carry, stripped of question words."""
    terms: list[str] = []
    for w in _WORD.findall(str(raw or ""))[:12]:
        if w.lower() in _QWORDS or len(w) < 3 and not w.isdigit():
            continue
        if any(w.lower() == t.lower() for t in terms):
            continue
        terms.append(w)
        if len(terms) >= budget:
            break
    return terms


def _slate_query(names: list[str], type_raw: Any, cap: int = 12) -> str:
    """A disjunction of names: the index scores each candidate's own opening on its own rare tokens.

    Names come first and whole (they carry the rare, discriminative tokens); whatever word budget
    is left goes to the type words, which only have to break ties between same-named articles.
    """
    words: list[str] = []
    seen: set[str] = set()
    for name in names:
        for w in name.split():
            key = w.lower()
            if key not in seen and len(words) < cap:
                seen.add(key)
                words.append(w)
    for w in _type_terms(type_raw, max(0, min(4, cap - len(words)))):
        if w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    return " ".join(words[:cap])


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
            outputs=["facts", "missing_article", "name_candidates", "type_words"],
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

        # the slate: names only.  A description-shaped answer is discarded here rather than spent.
        names = _name_slate(gap.get("name_candidates"), titles_a)
        if names:
            next_query = _slate_query(names, gap.get("type_words"))
        else:
            # nothing name-shaped came back: fall back to the parent's description/residual chain
            next_query = _as_query(gap.get("missing_article"), residual)
        if normalize_text(next_query) == normalize_text(question) or len(next_query) < 3:
            next_query = residual  # a repeat of the question returns the seven articles in hand

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
        ctx.note("name slate", names=names, type_words=str(gap.get("type_words", ""))[:120],
                 missing_article=str(gap.get("missing_article", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "name_candidates": names,
            "missing_article": str(gap.get("missing_article", "")),
            "facts": facts,
        }
