"""Stop asking one answer contract to cover three different jobs; route to a specialist.

Measured motivation (the parent system's 300 search-set traces, bucketed by what the reference span
actually is):

    bucket                                            n     parent accuracy
    bridge / single-entity lookup                   244            0.684
    yes-no comparison                                12            0.917
    pick-one-of-the-two-named alternatives           28            0.857
    shared-property ("what are both X and Y")        ~10           0.300

The last two buckets fail in ways the shared contract cannot see. For a pick-one question the
reference is a string the QUESTION already spells, yet the parent re-words it from the passages and
loses the qualifier the question carries ("University of California" for an option written
"University of California, San Francisco"). For a shared-property question the reference is one
common noun that covers both subjects, yet the parent answers with a conjunction of the two
subjects' individual descriptions ("B'z and Sigue Sigue Sputnik" where the reference is a role,
"not related" where the reference is the rank both share). Both are contract failures: the generic
instruction "copy the answer span verbatim from the evidence" is the right rule for a bridge lookup
and the wrong rule for a question whose answer is a choice or a category.

Mechanism under test (answering architecture): a deterministic router reads the question shape --
no LLM call, no extra retrieval -- and dispatches the single answering call to a specialist module
with its own fields, its own instruction and its own projection.
  * pick-one  -> the call compares the two alternatives and returns a LABEL; the returned answer is
                 the question's own literal alternative string, with a leading lowercase descriptor
                 stripped, so the reference wording is copied rather than paraphrased.
  * shared    -> the call must describe each subject separately and then name the ONE category word
                 both descriptions share; a conjunction of the subjects is rejected deterministically
                 and replaced by the longest phrase the two descriptions have in common.
  * otherwise -> the parent's contract, byte for byte, including yes-no.
Retrieval, the gap call and the pool are unchanged, so the delta is confined to the ~38 items the
router pulls out of the default branch.

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
    "option_a": "the first alternative the question offers, spelled as the question spells it",
    "option_b": "the second alternative the question offers, spelled as the question spells it",
    "a_fact": "verbatim clause settling the first alternative, or 'not found'",
    "b_fact": "verbatim clause settling the second alternative, or 'not found'",
    "winner": "a single character: A or B",
    "x_is": "what the first subject is, in as few words as the passages allow",
    "y_is": "what the second subject is, in as few words as the passages allow",
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


# --- question-shape router ---------------------------------------------------------------------
_ALT = re.compile(r"\s+\bor\b\s+", re.IGNORECASE)
_SHARED_CUE = re.compile(
    r"\b(both\b|in common\b|which type\b|what type\b|which kind\b|what kind\b|are which\b"
    r"|are of which\b|share\b|related\b|are both\b|known for\b)", re.IGNORECASE
)
_LOWER_HEAD = re.compile(r"^(?:[a-z][a-z'\u2019.\-]*\s+)+")
# a comma or colon followed by a lowercase word resumes the question, it does not continue the name
_TRAILING_CLAUSE = re.compile(r"\s*[,;:]\s*(?=[a-z])")
# "Was Bald Eagle ...", "Which magazine ..." -- the interrogative head is not part of the option
_LEADING_INTERROGATIVE = re.compile(
    r"^(?:which|what|who|whom|whose|was|were|is|are|did|does|do|has|have|had|between)\b\s*",
    re.IGNORECASE,
)
_STOPWORDS_HEAD = {"the", "a", "an"}


def _alternatives(question: str) -> tuple[str, str] | None:
    """The two literal alternatives a pick-one question offers, or None when it offers none."""
    q = str(question or "").strip().rstrip("?").strip()
    parts = _ALT.split(q)
    if len(parts) != 2:
        return None
    left, right = parts[0], parts[1].strip()
    # the left alternative starts after the last comma (or colon) that precedes the "or"
    for cut in (",", ":", ";"):
        if cut in left:
            left = left.rsplit(cut, 1)[1]
    left = _TRAILING_CLAUSE.split(left.strip())[0].strip()
    left = _LEADING_INTERROGATIVE.sub("", left).strip()
    # the right alternative ends where the question's trailing clause resumes (", is native to ...")
    right = _TRAILING_CLAUSE.split(right)[0].strip()
    for side in (left, right):
        if not side or len(side.split()) > 8 or not re.search(r"[A-Z0-9]", side):
            return None
    return left, right


def _bare_option(option: str) -> str:
    """Drop a leading lowercase descriptor ("astrophysicist Dirk Frimout" -> "Dirk Frimout")."""
    text = str(option or "").strip().strip(_STRIP_CHARS).strip()
    head = _LOWER_HEAD.match(text)
    if head and re.search(r"[A-Z]", text[head.end():]):
        lead_words = [w.lower() for w in text[: head.end()].split()]
        if not (len(lead_words) == 1 and lead_words[0] in _STOPWORDS_HEAD):
            text = text[head.end():].strip()
    return text.strip(_STRIP_CHARS).strip()


def _is_shared_property(question: str) -> bool:
    q = str(question or "")
    if not _SHARED_CUE.search(q):
        return False
    if not re.search(r"\band\b", q, re.IGNORECASE):
        return False
    return bool(re.match(r"\s*(what|which|how)\b", q.strip(), re.IGNORECASE))


def _longest_common_phrase(a: str, b: str) -> str:
    """Longest run of words shared by two descriptions -- the category they have in common."""
    wa = [w for w in _WORD.findall(str(a or "").lower())]
    wb = [w for w in _WORD.findall(str(b or "").lower())]
    best: list[str] = []
    for i in range(len(wa)):
        for j in range(len(wb)):
            n = 0
            while i + n < len(wa) and j + n < len(wb) and wa[i + n] == wb[j + n]:
                n += 1
            if n > len(best):
                best = wa[i:i + n]
    while best and best[0] in _QWORDS:
        best = best[1:]
    while best and best[-1] in _QWORDS:
        best = best[:-1]
    return " ".join(best[:4])


CHOICE_INSTRUCTIONS = """The question offers exactly two alternatives, `option_a` and `option_b`,
and the reference answer is one of them written the way the QUESTION writes it.

`a_fact`: the clause from the passages that settles whether `option_a` satisfies what the question
asks. Copy it verbatim, with its numbers and dates. Write "not found" if the passages do not cover it.
`b_fact`: the same for `option_b`.
`winner`: exactly one character, `A` or `B` -- whichever alternative the question is asking for.
Compare the two facts directly (earlier date = older, larger number = more, and so on). If only one
alternative is covered by the evidence, judge from that one and your knowledge of the other. Write
nothing except a single `A` or `B`.
`answer`: the winning alternative as a bare name. Spell it exactly as the question spells it, unless
a passage is the article ABOUT it and introduces its subject by a fuller name -- then use that fuller
opening name. Nothing else: no sentence, no reason, no final period."""

SHARED_INSTRUCTIONS = """The question names two different subjects and asks what they HAVE IN COMMON.
The reference answer is a single short category -- one common noun or noun phrase, in the wording the
articles use -- that is true of BOTH subjects. It is never a list of the two subjects, never their
names, and never two descriptions joined by "and".

`x_is`: what the first subject is, copied from the passages in as few words as possible.
`y_is`: the same for the second subject.
`answer`: the single category both descriptions share, in the source's own wording -- a bare noun
phrase of one to four words. Lowercase unless it is a proper name, no article, no explanation, no
final period. If the two subjects are two members of the same rank, class, genus, medium, country,
profession or format, that rank word IS the answer. If the question asks what they are "known for" or
"what type of X" they are, answer with the role or the type, not with the works or the names."""


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.gap = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "missing_target", "next_query"],
            instructions=GAP_INSTRUCTIONS,
            descs=DESCS,
        )
        self.choice_answer = Module(
            inputs=["question", "option_a", "option_b", "facts", "passages"],
            outputs=["a_fact", "b_fact", "winner", "answer"],
            instructions=CHOICE_INSTRUCTIONS,
            descs=DESCS,
        )
        self.shared_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["x_is", "y_is", "answer"],
            instructions=SHARED_INSTRUCTIONS,
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
        pool = self._pool(docs_a, docs_b)
        facts_in = facts if facts else "none"

        # 4. route the single answering call to the contract this question shape needs
        alts = _alternatives(question)
        if alts is not None and normalize_text(question)[:3] not in ("do ", "is ", "are"):
            out = self.choice_answer(
                ctx.llm, tag="answer", question=question,
                option_a=alts[0], option_b=alts[1], facts=facts_in, passages=pool,
            )
            verdict = str(out.get("winner", "")).strip().upper()
            pick = _bare_option(alts[1] if verdict.startswith("B") else alts[0])
            raw = out.get("answer", "")
            # the question's own spelling is the default; the model may only make it FULLER, and only
            # if it is plainly the same name (a shared word), never a different entity.
            spoken = project_answer(raw, "other", question)
            answer = pick or _clean(pick)
            pick_toks = {normalize_text(w) for w in _WORD.findall(pick)}
            spoken_toks = {normalize_text(w) for w in _WORD.findall(spoken)}
            if spoken and pick_toks and (pick_toks & spoken_toks) and len(spoken.split()) >= len(pick.split()):
                answer = spoken
            route = "choice"
        elif _is_shared_property(question):
            out = self.shared_answer(
                ctx.llm, tag="answer", question=question, facts=facts_in, passages=pool,
            )
            raw = out.get("answer", "")
            answer = project_answer(raw, "other", question)
            norm_ans = normalize_text(answer)
            subjects = normalize_text(question)
            # a conjunction of the two subjects is not a shared category -- fall back to the overlap
            if norm_ans and (" and " in f" {norm_ans} " or norm_ans in subjects) and len(norm_ans.split()) > 1:
                overlap = _longest_common_phrase(out.get("x_is", ""), out.get("y_is", ""))
                if overlap:
                    answer = overlap
            route = "shared"
        else:
            out = self.generate_answer(
                ctx.llm,
                tag="answer",
                question=question,
                facts=facts_in,
                passages=pool,
            )
            raw = out.get("answer", "")
            answer = project_answer(raw, out.get("answer_type", ""), question)
            route = "default"
        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200],
                 next_query=next_query, residual=residual)
        ctx.note("typed answer", route=route, answer_type=out.get("answer_type", ""),
                 raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "route": route,
            "query_a": question,
            "next_query": next_query,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }
