"""Teach the planner to write second-hop queries from examples of queries that PROVABLY worked.

Measured motivation (post-hoc counts over the parent system's own 300 search-set traces):

    items where the first lookup found exactly one of the two needed articles   143
    of those, the second lookup recovered the other one                          68
    accuracy when it did / when it did not                              0.912 / 0.453
    accuracy by pool coverage: both needed articles 0.868 (n=189),
        one 0.495 (n=93), neither 0.056 (n=18)

Whether the second query lands the other article is worth ~46 points on the items where it matters,
which is far more than any presentational change to an already-fixed pool. Query formulation is
therefore the thing to supervise -- and it is the one step of this pipeline that can be supervised
without a grader, because the train split carries the titles each question needs. Given the first
lookup's titles, the article the second lookup OUGHT to return is known exactly, so a query can be
labelled right or wrong by running it.

Mechanism under test (prepare-time learning): the planning call is fitted with few-shot
demonstrations that were bootstrapped and then verified by retrieval on the train split. In
`prepare`, for train questions whose first lookup covers exactly one of the two needed articles,
the planner is run as-is; the query it produces is executed; the demonstration is kept ONLY if that
query actually returned the still-missing article. Kept demonstrations are then chosen for
diversity along the axis that decides how a query must be written: whether the missing article's
name appears in the first lookup's text (copy the name) or does not (supply the name from knowledge
of the subject). Passages inside a demonstration are trimmed to their opening naming sentence so
the demonstrations cost a few hundred tokens rather than a few thousand.

Nothing else moves: same two calls, same two lookups, same k, same instructions, same fallback
chain, same answering contract, same answer projection and source voicing. If `prepare` yields no
verified demonstration the module is built with none and the system is byte-identical to its
parent, so the experiment is a clean test of whether verified query demonstrations transfer.

Budget: 2 LLM calls, 2 retrievals, k=7 per item; prepare uses its own separate budget.
"""

from __future__ import annotations

import json
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


# --- prepare-time demonstration mining -------------------------------------------------------
_FIRST_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?](?=\s+[A-Z0-9]|\s*$)")
MAX_TRAIN_PROBES = 40      # cap prepare wall-clock: one planning call per probe
MAX_DEMOS = 2              # two demonstrations keep the added prompt cost to a few hundred tokens


def _opening_sentence(passage: str) -> str:
    """A passage trimmed to the sentence that says what its subject is."""
    title, sep, body = str(passage).partition("|")
    body = body.strip()
    m = _FIRST_SENT.search(body)
    head = body[: m.end()] if m else body
    if len(head.split()) > 45:
        head = " ".join(head.split()[:45])
    return f"{title.strip()} | {head}".strip() if sep else head


def _trim(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0]


def _two_sentences(text: Any) -> str:
    s = " ".join(str(text or "").split())
    parts = re.split(r"(?<=[.!?])\s+", s)
    return _trim(" ".join(parts[:2]), 320)


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.demos: list[dict[str, Any]] = []
        self._build_gap()
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )

    def _build_gap(self) -> None:
        """The planning module, rebuilt whenever the verified demonstration set changes."""
        self.gap = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "missing_target", "next_query"],
            instructions=GAP_INSTRUCTIONS,
            descs=DESCS,
            demos=self.demos,
        )

    # --- prepare-time learning ----------------------------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: Any) -> None:
        """Bootstrap planning demonstrations and keep only the ones retrieval proves correct.

        For each probed train question: run the first lookup, check the labelled titles it needs,
        and keep going only when exactly one of them is still missing -- that is the situation the
        second query exists for. Run the planner unchanged, execute the query it produced, and
        accept the demonstration only if the still-missing article actually came back.
        """
        kept: list[dict[str, Any]] = []
        probed = 0
        for tr_item in train_items:
            if probed >= MAX_TRAIN_PROBES or len(kept) >= 6:
                break
            want = [str(t).strip() for t in (tr_item.get("gold_titles") or []) if str(t).strip()]
            question = str(tr_item.get("question") or "").strip()
            if len(want) != 2 or not question:
                continue
            try:
                with ctx.open_tools(tr_item) as tools:
                    docs_a = tools.retrieve_docs(question, k=self.K)
                    titles_a = [str(d.get("title", "")).strip() for d in docs_a]
                    have = {normalize_text(t) for t in titles_a}
                    missing = [t for t in want if normalize_text(t) not in have]
                    if len(missing) != 1:
                        continue  # nothing for a second query to do, or the first lookup found neither
                    probed += 1
                    next_query, _residual, gap = self._plan_lookup(ctx.llm, question, docs_a)
                    docs_b = tools.retrieve_docs(next_query, k=self.K)
                    returned = {normalize_text(str(d.get("title", ""))) for d in docs_b}
                    if normalize_text(missing[0]) not in returned:
                        continue  # the query did not land the article it was for -- not a demonstration
            except Exception:  # noqa: BLE001 - a probe that fails simply yields no demonstration
                continue
            blob = normalize_text(" ".join(str(d.get("text", "")) for d in docs_a))
            kept.append({
                "question": question,
                "titles_in_hand": titles_a,
                "passages": [_opening_sentence(str(d.get("text", ""))) for d in docs_a],
                "reasoning": _two_sentences(gap.get("reasoning")),
                "facts": _trim(gap.get("facts"), 420),
                "missing_target": _trim(gap.get("missing_target"), 220),
                "next_query": next_query,
                "_named_in_text": normalize_text(missing[0]) in blob,
            })
        self.demos = self._select_demos(kept)
        self._build_gap()
        ctx.note("verified query demonstrations", probed=probed, verified=len(kept),
                 chosen=[d.get("next_query", "") for d in self.demos])

    @staticmethod
    def _select_demos(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pick along the axis that decides how a query must be written, not by score.

        A second query is written in one of two ways: copy the name the first lookup's text already
        gives, or supply the name from knowledge of the subject when the text only describes it.
        One verified demonstration of each is worth more than two of the same kind.
        """
        if not kept:
            return []
        chosen: list[dict[str, Any]] = []
        for flag in (True, False):
            same = [d for d in kept if bool(d.get("_named_in_text")) is flag]
            if same:
                chosen.append(min(same, key=lambda d: len(str(d.get("facts", "")))))
        for d in kept:  # top up if only one kind was verified
            if len(chosen) >= MAX_DEMOS:
                break
            if d not in chosen:
                chosen.append(d)
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in chosen[:MAX_DEMOS]]

    def get_state(self) -> str:
        return json.dumps({"demos": self.demos})

    def set_state(self, state: str) -> None:
        try:
            loaded = json.loads(state or "{}").get("demos") or []
        except ValueError:
            loaded = []
        self.demos = [d for d in loaded if isinstance(d, dict)][:MAX_DEMOS]
        self._build_gap()

    def _plan_lookup(self, llm: Any, question: str, docs_a: list[dict[str, Any]]) -> tuple[str, str, dict[str, str]]:
        """The parent's planning step and fallback chain, factored out so `prepare` runs it verbatim."""
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]
        gap = self.gap(
            llm,
            tag="gap",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        residual = _residual_query(question, titles_a)
        target = _as_query(gap.get("missing_target"), residual)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = residual
        if normalize_text(next_query) in {normalize_text(t) for t in titles_a}:
            next_query = target
        return next_query, residual, gap


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
        next_query, residual, gap = self._plan_lookup(ctx.llm, question, docs_a)
        facts = str(gap.get("facts", "")).strip()

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
                 next_query=next_query, residual=residual)
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
