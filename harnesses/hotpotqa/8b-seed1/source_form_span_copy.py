"""Hand over a span of the source, in the source's own wording -- not an answer of your own.

Measured motivation (300 search-set traces of the frontier system, gold answer string searched for
inside the exact prompt its answering call received):

    union gold articles retrieved   correct   wrong, gold answer IS in the prompt   wrong, it is not
                    both               161                  27                             1
                exactly one             45                  21                            27
                       none              1                   2                            15

So 48 of the 93 errors -- more than half -- are made with the reference string sitting in the
prompt. Reading them, the dominant pattern is granularity, and it points one way every time: the
reference is the article's own wording and the prediction is the model's own wording.

    reference                                   prediction
    Ronald Eugene Rivera                        Ron Rivera
    Molly Helen Shannon                         Molly Shannon
    Adolfo Jose Valencia Mosquera               Adolfo Valencia
    Dirk Dries David Damiaan, Viscount Frimout  Dirk Frimout
    Newport Beach, California                   Newport Beach
    University of California, San Francisco     University of California
    141 mi                                      141
    six children                                six
    second                                      2

Every reference above is a contiguous run of words from a retrieved article opening, and every
prediction is a sensible paraphrase of it. The frontier's answering contract asks for "the shortest
form that fully names it" with "no units of explanation" -- which is precisely the instruction that
produces the right-hand column.

Mechanism under test (answer policy + evidence addressing; the retrieval half of
`question_first_gap_hop` is kept byte-identical so the delta is an answering delta): make the
source text addressable and make copying checkable. The evidence pool is rendered as individually
numbered SENTENCES under their titles, and the answering call must cite the sentence id that
settles the question and then copy the answering phrase out of that sentence character for
character, keeping the source's granularity -- full name at first mention, the qualifier attached
to a place, the unit or counted noun attached to a number, the source's spelling of a count. A
deterministic layer then verifies the cited phrase really occurs in the pool and, when the model's
own `answer` is a shortened paraphrase strictly inside that verified phrase, hands over the phrase
instead.

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

ANSWER_INSTRUCTIONS = """Answer the question from `evidence`: the article openings retrieved for
this question, broken into individually numbered sentences under their titles. `facts` are the
sentences an earlier pass already pulled out.

The reference you are graded against was produced by copying a short run of words OUT of one of
these articles, and grading is exact string match. So your job is not to phrase an answer; it is to
locate that run of words and hand it over unchanged. Where your own wording and the article's
wording differ, the article's wording is the one that scores.

`support_id`: the id -- for example S14 -- of the single sentence in `evidence` that settles the
question. Exactly one id, nothing else.

`source_phrase`: the shortest contiguous run of words inside that sentence which answers the
question, copied character for character as it appears there. A run of words, not the sentence:
one to six words is the normal length, and a `source_phrase` longer than eight words is thrown
away unread. Granularity is decided here, and it is decided by the source, not by you:
- Keep the full name the article gives the subject at first mention -- middle names, second
  surnames, titles of nobility and all -- rather than the short everyday name you would use.
- Keep the qualifier the source attaches: the state or country after a city, the campus or city
  after an institution, the year in front of a named event.
- Keep the word the source attaches to a bare number: the unit of measurement, or the noun being
  counted ("141 mi", "six children", "48 murders").
- Keep the source's own spelling of a count or an ordinal; do not turn a spelled-out word into a
  digit or a digit into a word.
- But do not reach past the answer: start at the first answering word and stop at the last. No verb
  of the sentence, no "born", no "which" or "because", no leading or trailing preposition, no
  trailing clause, no appositive that merely explains what the thing is, no parentheses.
- If the answering words are genuinely in no sentence of `evidence`, write "none" here.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer`: normally `source_phrase`, character for character. Depart from it only in the cases the
question forces:
- If `answer_type` is yes_no, `answer` must be exactly `yes` or `no` -- one word, nothing after it.
  Questions of the form "are both ...", "do both ...", "is X the same as Y", "can both ..." are
  yes_no questions, even when the interesting content is in the reasoning.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly and completely as the question spells it -- the whole option, not a shortened
  head of it.
- If `source_phrase` is "none", give the single most likely bare answer from your own knowledge, in
  the shortest form that fully names it. Never answer "unknown", "not stated" or "cannot be
  determined".
Never a sentence, never a restatement of the question, no explanation, no markdown, no quotation
marks, no final period."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "titles_in_hand": "titles of the articles already retrieved -- do not fetch these again",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "missing_target": "the article still needed, named if the passages name it",
    "next_query": "keyword query for the second and final lookup, aimed at the missing article",
    "evidence": "the retrieved article openings, split into numbered sentences under their titles",
    "support_id": "the id of the one sentence in `evidence` that settles the question, e.g. S14",
    "source_phrase": "the answering words, copied character for character out of that sentence",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[\"\'\(]?[A-Z0-9])")
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
_ABBREV = re.compile(
    r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|no|No|Inc|Ltd|Co|Mt|Ft|Op|ca|c|eds?|Vol)\.\s+$"
)


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
    """Hand back the option the question spells -- whole, whether the answer over- or under-shot."""
    norm_ans = normalize_text(ans)
    if not norm_ans:
        return ans
    prefix_hits, inside_hits, over_hits = [], [], []
    for opt in _question_options(question):
        n = normalize_text(opt)
        if not n or n == norm_ans:
            return ans
        if len(n.split()) < len(norm_ans.split()):
            if norm_ans.startswith(n + " "):
                prefix_hits.append(opt)
            elif f" {n} " in f" {norm_ans} ":
                inside_hits.append(opt)
        elif f" {norm_ans} " in f" {n} ":
            # the answer is a truncated head of an option ("University of California"): restore it
            over_hits.append(opt)
    hits = prefix_hits or ([] if len(inside_hits) != 1 else inside_hits)
    if not hits and len(over_hits) == 1:
        return over_hits[0]
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
    """The question minus the words of the title it already matched best -- a complementary query."""
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


# -- addressable evidence ---------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on the abbreviations article openings are full of."""
    out: list[str] = []
    buf = ""
    for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
        buf = f"{buf} {piece}".strip() if buf else piece
        if _ABBREV.search(buf + " ") or len(buf) < 25:
            continue
        out.append(buf)
        buf = ""
    if buf:
        if out and len(buf) < 25:
            out[-1] = f"{out[-1]} {buf}"
        else:
            out.append(buf)
    return out or ([str(text).strip()] if str(text).strip() else [])


def build_evidence(docs: list[dict[str, Any]]) -> tuple[str, dict[str, str], list[str]]:
    """Render the pool as numbered sentences under their titles; return (block, id->sentence, sents)."""
    lines: list[str] = []
    index: dict[str, str] = {}
    flat: list[str] = []
    n = 0
    for d in docs:
        title = str(d.get("title", "")).strip()
        body = str(d.get("text", "")).strip()
        if title and body.lower().startswith(title.lower()):
            body = body[len(title):].lstrip(" |").strip()
        lines.append(f"«{title}»")
        for sent in _split_sentences(body):
            n += 1
            sid = f"S{n}"
            index[sid] = sent
            flat.append(_norm_phrase(sent))
            lines.append(f"  {sid}: {sent}")
    return "\n".join(lines), index, flat


# A reference span never dangles a preposition or a copula: if handing over the source phrase would
# add one of these, the phrase is a clause rather than a span and the model's own answer stands.
_GLUE = {
    "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been",
    "being", "by", "with", "from", "as", "that", "which", "who", "whose", "whom", "when", "where",
    "it", "its", "his", "her", "their", "he", "she", "they", "has", "have", "had", "also", "but",
    "not", "than", "there", "this", "these", "those", "do", "does", "did", "known", "called",
    "born", "including", "such", "about", "after", "before", "during", "between", "both",
}


_NUMWORD = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "dozen", "hundred", "thousand", "million", "billion", "first", "second", "third",
    "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
}


def _extras_are_attached(answer: str, extras: list[str], phrase: str) -> bool:
    """Only a qualifier the source attaches may be added -- never a category noun.

    Two shapes qualify, and they are the two the reference conventions actually use:
      * name completion -- every added word is capitalised or a number where the source writes it
        ("Ronald Eugene Rivera", "Newport Beach, California", "January 7, 1962", "1999 Columbine ...");
      * a bare count or measurement gaining the single word attached to it ("141 mi", "six children").
    A lowercase category noun ("... motion picture", "a 1964 documentary film") is neither.
    """
    if not extras:
        return True
    surface = {w.strip(_STRIP_CHARS + "()").lower(): w.strip(_STRIP_CHARS + "()") for w in phrase.split()}
    if all(
        (surface.get(t, t)[:1].isupper() or t.isdigit() or any(c.isdigit() for c in t))
        for t in extras
    ):
        return True
    atoks = normalize_text(answer).split()
    return len(extras) == 1 and bool(atoks) and all(
        t.isdigit() or t in _NUMWORD for t in atoks
    )


def _align(answer_toks: list[str], phrase_toks: list[str]) -> list[str] | None:
    """Consume the answer's tokens through the phrase in order; return the tokens left over.

    A token matches exactly, or as an abbreviated given name ("Ron" inside "Ronald"), which is the
    relation between the everyday name a model reaches for and the full name an article opens with.
    """
    extras: list[str] = []
    i = 0
    for tok in phrase_toks:
        if i < len(answer_toks):
            want = answer_toks[i]
            if tok == want or (len(want) >= 3 and tok.startswith(want) and want.isalpha()):
                i += 1
                continue
        extras.append(tok)
    return extras if i == len(answer_toks) else None


def _norm_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def prefer_source_phrase(answer: str, source_phrase: Any, support_id: Any,
                         index: dict[str, str], sents: list[str]) -> tuple[str, str]:
    """If `answer` is a shortened paraphrase strictly inside a phrase verified in the pool, use it.

    The phrase has to earn it: it must occur verbatim inside a SINGLE retrieved sentence (checking
    against the flattened pool instead would let a phrase straddle a sentence boundary and glue on
    the tail of the previous sentence), it must be short enough to be a reference span rather than a
    clause, and the model's own answer must sit contiguously inside it.  Otherwise the model's
    answer stands untouched.
    """
    phrase = _clean(source_phrase)
    if not phrase or _DECLINED.match(phrase):
        return answer, "no-phrase"
    words = phrase.split()
    if not (1 <= len(words) <= 8):
        return answer, "phrase-too-long"
    np_raw = _norm_phrase(phrase)
    cited = index.get(str(support_id or "").strip().upper().lstrip("[").rstrip("]"))
    if not (cited and np_raw in _norm_phrase(cited)):
        if not any(np_raw in sent for sent in sents):
            return answer, "phrase-not-in-evidence"
    na, np_ = normalize_text(answer), normalize_text(phrase)
    if not na or not np_ or na == np_:
        return answer, "same"
    extras = _align(na.split(), np_.split())
    if extras is None:
        return answer, "answer-not-inside-phrase"
    if len(extras) > 4:
        return answer, "phrase-much-longer"
    if any(t in _GLUE for t in extras):
        return answer, "phrase-is-a-clause"
    if not _extras_are_attached(answer, extras, phrase):
        return answer, "extras-not-attached"
    return phrase, "expanded-to-source"


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
            inputs=["question", "facts", "evidence"],
            outputs=["support_id", "source_phrase", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )

    @staticmethod
    def _pool(*doc_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for docs in doc_lists:
            for d in docs:
                key = str(d.get("title", "")).strip().lower()
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(d)
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # 1-3. retrieval half: byte-identical to the frontier system
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

        # 4. answer over sentence-addressable evidence, then hand over the verified source phrase
        block, index, sents = build_evidence(self._pool(docs_a, docs_b))
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            evidence=block,
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        atype = str(out.get("answer_type", "")).strip().lower()
        verdict = "yes_no-kept"
        if not atype.startswith("yes") and normalize_text(answer) not in ("yes", "no"):
            answer, verdict = prefer_source_phrase(
                answer, out.get("source_phrase"), out.get("support_id"), index, sents
            )
            answer = _clean(_echo_option(answer, question)) or answer
        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("source span", n_sentences=len(index), support_id=str(out.get("support_id", ""))[:20],
                 source_phrase=str(out.get("source_phrase", ""))[:200], raw=str(raw)[:200],
                 verdict=verdict, projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "source_phrase": str(out.get("source_phrase", "")),
            "support_id": str(out.get("support_id", "")),
            "span_verdict": verdict,
            "answer_type": out.get("answer_type", ""),
            "query_a": question,
            "next_query": next_query,
            "facts": facts,
        }
