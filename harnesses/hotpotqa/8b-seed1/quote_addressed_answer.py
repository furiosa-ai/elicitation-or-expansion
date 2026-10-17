"""Treat the answer as an ADDRESS into the sentence the model quoted, not as a phrasing of it.

Measured motivation (replayed over the parent system's own 300 search-set traces; the parent's two
calls, its queries and its pool are reproduced byte for byte, so every number is a post-hoc count):

    the quoted sentence contains the reference span      214 items, accuracy 0.902
    it does not                                           86 items, accuracy 0.209
    the answer is a contiguous run of words of its
        own quoted sentence                              254 items, accuracy 0.752
    it is not                                             29 items, accuracy 0.310

So the quote is already almost always right, and the loss concentrates in the step AFTER quoting:
the model quotes the settling sentence and then re-words it. Of the 21 errors made with the
reference span sitting inside the model's own quote, the prediction converts the quote's ordinal
word to a digit, drops a measurement unit, swaps a head noun for a near synonym, glues a second
name on with "and", supplies a middle name the quote does not use, or -- five times -- refuses
outright ("not stated", "unknown", "not mentioned") while its own quote holds the span.

Mechanism under test (answer policy; retrieval, queries and evidence are frozen): the answer is
required to be a verbatim contiguous run of words of `evidence_quote`, and three deterministic
repairs re-anchor it there afterwards. Each repair was replay-validated against the parent's own
outputs and only rules that never lose were kept:

    snap-to-quote   (answer not a run of the quote, but a shorter run of the quote is a
                     subsequence of it with the same first and last word -> use the quote's
                     wording)                                            1 win  / 0 losses
    unit restore    (bare number answer, quote continues with a measuring
                     or counting noun -> append it)                      2 wins / 0 losses
    refusal rescue  (answer normalises to a refusal -> read a span of the
                     declared type out of the quote)          5 sure losses, no downside

Two extension rules that looked equally plausible were REFUTED by the same replay and are not
here: appending a following ", ProperNoun" qualifier scored 2 wins / 8 losses, and prepending a
leading year modifier 0 / 2. Extent is only widened where the added token is bound to the span
(a unit of measure), never where it is a separate appositive.

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

`evidence_quote`: copy out, word for word, the ONE sentence from `passages` that settles the
question, prefixed with its article title in brackets. If no sentence settles it outright, quote the
sentence that comes closest -- never leave this empty.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer` is an ADDRESS INTO `evidence_quote`, not a description of it. It must be a contiguous run
of words taken out of `evidence_quote` and copied character for character, and it is graded by
exact string match against a short reference span:
- Copy. Do not re-word. Do not translate an ordinal word into a digit or a digit into a word, do
  not swap a noun for a synonym, do not abbreviate, do not translate, do not correct the spelling
  or the capitalisation of the quoted sentence.
- Take ONE run of words. Never join two separate spans with "and" or a comma when the question
  asks for one thing.
- Take the SHORTEST run that fully names the answer. Drop the category descriptor the sentence
  uses to introduce it: from "the British rock band Bush" the answer is `Bush`, from "the 1965
  film Help!" it is `Help!`. Keep the name, drop the words that say what kind of thing it is.
- Use the name exactly as the sentence you quoted writes it. Do not supply a middle name, a
  fuller title, a nickname or a short form that the quoted sentence itself does not contain.
- Keep whatever is bound to the span inside the quote: a measurement keeps its unit
  ("about 12 km" -> `12 km`), a counted noun keeps its noun when the question asks "how many
  <noun>".
- Stop at the span. Never a sentence, never a restatement of the question, no explanation, no
  "because", no trailing clause, no second sentence, no markdown, no quotation marks, no final
  period.
- If `answer_type` is yes_no, `answer` must be exactly `yes` or `no` -- one word, nothing after
  it, and it need not appear in the quote. Questions of the form "are both ...", "do both ...",
  "is X the same as Y", "can both ..." are yes_no questions, even when the interesting content is
  in the reasoning.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly as the question spells it, and nothing else.
- If the evidence is incomplete, still give the single most likely bare answer, still as a run of
  words out of the quote where one fits. Never answer "unknown", "not stated", "not mentioned" or
  "cannot be determined": a refusal is scored exactly like a wrong guess and forfeits the item."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "titles_in_hand": "titles of the articles already retrieved -- do not fetch these again",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "missing_target": "the article still needed, named if the passages name it",
    "next_query": "keyword query for the second and final lookup, aimed at the missing article",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span, copied verbatim out of evidence_quote",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:«»"
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
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    q = re.split(r"\s+[—–-]{1,2}\s+|\s*[—–]\s*", q)[0].strip(_STRIP_CHARS).strip()
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    if len(q) < 3 or _DECLINED.match(q):
        return fallback
    return q


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


# --- source-article voicing (inherited unchanged) ---------------------------------------------
_LEAD_STOP = re.compile(
    r"\s*(?:\(|,|\"|“|”|‘|’\s|\bor\b|\bis\b|\bwas\b|\bare\b|\bwere\b|\bserves\b|\brefers\b)"
)
_PURE_NAME = re.compile(r"^[A-Z][A-Za-z'’.\-]*(?: (?:[A-Z][A-Za-z'’.\-]*|[a-z]{2,4}))*$")


def _naming_clause(text: str) -> str:
    stop = _LEAD_STOP.search(text or "")
    head = (text[: stop.start()] if stop else str(text or "")).strip()
    if not head or len(head.split()) > 6 or not _PURE_NAME.match(head):
        return ""
    return head


def _voice_through_source(answer: str, pool: list[str]) -> str:
    norm = normalize_text(answer)
    if not norm or norm in ("yes", "no") or len(norm.split()) < 2:
        return answer
    for passage in pool:
        title, sep, body = str(passage).partition("|")
        if not sep or normalize_text(title) != norm:
            continue
        lead = _naming_clause(body.strip())
        if not lead:
            return answer
        lead_toks, ans_toks = lead.split(), str(answer).split()
        if len(lead_toks) <= len(ans_toks):
            return answer
        if normalize_text(lead_toks[-1]) != normalize_text(ans_toks[-1]):
            return answer
        first_lead, first_ans = normalize_text(lead_toks[0]), normalize_text(ans_toks[0])
        if not first_lead or not first_ans:
            return answer
        if first_lead[:3] != first_ans[:3]:
            return answer
        return lead
    return answer


# --- quote addressing ------------------------------------------------------------------------
# A measuring or counting noun is BOUND to the number in front of it: "1065 m", "six children".
# An appositive qualifier is not, so ", California" style tails are deliberately absent here --
# replaying them over the parent's outputs scored 2 wins against 8 losses.
_MEASURE_UNIT = re.compile(
    r"^(?:m|mi|km|cm|mm|ft|kg|lb|lbs|ha|acres?|hectares?|miles?|metres?|meters?|feet|inches|"
    r"kilometres?|kilometers?|centimetres?|centimeters?|pounds?|kilograms?|tonnes?|tons?|"
    r"sq\s?mi|km2|m2)$"
)
_REFUSAL = re.compile(
    r"^(?:unknown|unclear|unspecified|none|n/?a|nothing|no answer|not\s+\w+|cannot\b.*|can not\b.*|"
    r"insufficient\b.*|no\s+(?:information|evidence|mention|data)\b.*)$",
    re.IGNORECASE,
)
_TOKENS = re.compile(r"\S+")


def _quote_words(quote: str) -> tuple[list[str], list[str]]:
    """Raw whitespace tokens of the quote plus their normalised forms."""
    raw = _TOKENS.findall(str(quote or ""))
    return raw, [normalize_text(t) for t in raw]


def _locate(raw: list[str], norm: list[str], target: str) -> tuple[int, int] | None:
    """Token window of the quote whose normalisation equals `target`'s, or None."""
    want = normalize_text(target).split()
    if not want:
        return None
    for i in range(len(raw)):
        acc: list[str] = []
        for j in range(i, min(i + len(want) + 4, len(raw))):
            if norm[j]:
                acc.append(norm[j])
            if len(acc) > len(want):
                break
            if acc == want:
                return (i, j + 1)
    return None


def _snap_to_quote(answer: str, quote: str) -> str:
    """Re-anchor an answer that is not a run of the quote onto the quote's own wording.

    Fires only when the quote holds a SHORTER run whose words are a subsequence of the answer's
    and which starts and ends on the same word -- i.e. the model padded the quoted name with
    material from elsewhere ("Adam Henry Carter" for a sentence that says "Adam Carter").
    """
    raw, norm = _quote_words(quote)
    if not raw:
        return answer
    if _locate(raw, norm, answer) is not None:
        return answer
    want = normalize_text(answer).split()
    if len(want) < 2:
        return answer
    best: tuple[int, int] | None = None
    best_len = 0
    for i in range(len(raw)):
        if norm[i] != want[0]:
            continue
        for j in range(i + 1, min(i + len(want), len(raw)) + 1):
            win = [t for t in norm[i:j] if t]
            if len(win) < 2 or win[-1] != want[-1] or len(win) >= len(want):
                continue
            cursor = iter(want)
            if all(any(w == x for x in cursor) for w in win) and len(win) > best_len:
                best, best_len = (i, j), len(win)
    if best is None:
        return answer
    return " ".join(raw[best[0]:best[1]]).strip(_STRIP_CHARS).strip() or answer


def _restore_unit(answer: str, quote: str) -> str:
    """A bare number answer takes back the noun the quote binds to it.

    One admissible tail only: a unit of measure, which is bound to the number rather than
    appositive to it.  Replaying looser variants over the parent's outputs refutes them -- "any
    following noun" produced "1943 in" and "1976 album", and extending across a plural noun the
    question itself counted produced "24 survivors" where the reference was the bare count.
    """
    if not re.fullmatch(r"[\d.,/\u2013-]+", str(answer).strip()):
        return answer
    raw, norm = _quote_words(quote)
    span = _locate(raw, norm, answer)
    if span is None or span[1] >= len(raw):
        return answer
    tail = raw[span[1]].strip(_STRIP_CHARS).strip()
    ntail = normalize_text(tail)
    if not ntail:
        return answer
    if not _MEASURE_UNIT.match(ntail):
        return answer
    return f"{answer} {tail}".strip()


def _rescue_refusal(answer: str, quote: str, answer_type: str) -> str:
    """A refusal forfeits the item, so read the best-typed span out of the quote instead."""
    if not _REFUSAL.match(normalize_text(answer) or "x"):
        return answer
    body = str(quote or "")
    body = re.sub(r"^\s*\[[^\]]{0,120}\]\s*", "", body).strip(_STRIP_CHARS)
    atype = str(answer_type or "").strip().lower()
    if atype.startswith(("date", "number")):
        hit = re.search(r"\b\d{1,4}(?:[–-]\d{2,4})?\b", body)
        if hit:
            return hit.group(0)
    names = re.findall(r"\b[A-Z][A-Za-z'’.\-]*(?:\s+[A-Z][A-Za-z'’.\-]*){0,3}", body)
    names = [n for n in names if len(n.split()) >= 2] or names
    if names:
        return names[-1].strip(_STRIP_CHARS).strip()
    return answer


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

        docs_a = ctx.tools.retrieve_docs(question, k=self.K)
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]

        gap = self.gap(
            ctx.llm,
            tag="gap",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(gap.get("facts", "")).strip()
        residual = _residual_query(question, titles_a)
        target = _as_query(gap.get("missing_target"), residual)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = residual
        if normalize_text(next_query) in {normalize_text(t) for t in titles_a}:
            next_query = target

        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        pool = self._pool(docs_a, docs_b)
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        atype = out.get("answer_type", "")
        quote = str(out.get("evidence_quote", "") or "")

        answer = project_answer(raw, atype, question)
        answer = _voice_through_source(answer, pool)
        if normalize_text(answer) not in ("yes", "no"):
            answer = _snap_to_quote(answer, quote)
            answer = _restore_unit(answer, quote)
            answer = _rescue_refusal(answer, quote, atype)
        answer = _clean(answer) or project_answer(raw, atype, question)

        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("quote-addressed answer", answer_type=atype, raw=str(raw)[:300],
                 quote=quote[:400], answer=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": atype,
            "evidence_quote": quote,
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }
