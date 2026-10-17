"""Learn WHICH surface repairs are real by measuring them on labelled train items, not by guessing.

Every post-hoc repair this line of systems has shipped was justified by replaying it over search-set
traces and counting wins. That is the right method applied to the wrong split, and it is also the
reason prepare-time work has failed three times running: each of those candidates imported *text*
into a prompt (demos, query templates, span conventions), which transfers badly. Nothing has ever
used the train labels for what they are uniquely good at -- deciding, with an actual win/loss count,
whether a mechanical rule is worth switching on.

Mechanism under test (prepare-time policy selection; both prompts, both retrievals and the pool are
byte-identical to `source_named_answer`):

  prepare -- run the parent pipeline over a bounded slice of the train split, recording for each
             item only (question, pooled passages, produced answer, gold answer). Then replay a
             library of candidate surface repairs over those records offline -- no extra calls --
             and score each one by (items it fixes) minus (items it breaks). A repair is switched
             ON only if it fixes strictly more than it breaks and fires at least twice; the
             enabled set, a handful of rule names, is the whole of `get_state`.

  run     -- the parent, then the enabled repairs in descending measured net gain. Each repair is
             a first-class hypothesis about what a reference span looks like, and the train split
             is the referee:
               source_voice     restate the answer as its OWN article names it (the parent rule --
                                offered as a candidate rather than assumed, so it has to re-earn
                                its place on train)
               unit_attach      a bare numeral whose every pooled occurrence carries the same
                                measurement word is a quantity, not a number
               head_noun        a bare count whose every pooled occurrence is followed by the same
                                noun is a count OF that noun
               title_extend     a unique pooled title that extends the answer with a qualifier is
                                the disambiguated form of the same thing
               comma_qualifier  a place or institution whose every pooled occurrence continues
                                ", Somewhere" is named with that qualifier attached
               sole_conjunct    an answer joining two names with "and", for a question that asked
                                for one thing, keeps the conjunct the evidence attaches to the
                                asked role

Nothing is hard-coded about which of these survives: on train, most will not. If prepare never
runs, the enabled set defaults to the parent's single rule and the system is the parent exactly.

Budget: 2 LLM calls, 2 retrievals, k=7 at eval time -- identical to the parent on every item.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
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


# --- repair library ----------------------------------------------------------------------------
# Each repair maps (answer, question, pooled passage bodies) -> a replacement string, or "" when it
# does not apply.  None of them is trusted: `prepare` counts their wins and losses on train items
# and only the ones that come out positive are switched on.

_NUM = re.compile(r"^[\d][\d,.]*$")
_NUMWORD = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "dozen", "hundred", "thousand", "million", "billion"}
_UNITWORD = re.compile(
    r"^(m|km|mi|ft|cm|mm|kg|g|lb|lbs|yd|sq|ha|acres?|hectares?|miles?|metres?|meters?|kilometres?"
    r"|kilometers?|kilometres|feet|inches|pounds?|kilograms?|tonnes?|tons?|percent|degrees?)$",
    re.IGNORECASE)
_CAPTOK = re.compile(r"^[A-Z][A-Za-z'’.\-]*$")


def _bodies(pool: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pool:
        title, _sep, body = str(p).partition("|")
        out.append((title.strip(), body.strip()))
    return out


def _occurrences(ans: str, text: str):
    """(text before, text after) around each whole-token occurrence of `ans` in `text`."""
    if not ans:
        return
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(ans) + r"(?![A-Za-z0-9])")
    for m in pat.finditer(text):
        yield text[max(0, m.start() - 40):m.start()], text[m.end():m.end() + 40]


def _consensus_next_word(ans: str, pool: list[str]) -> str:
    """The word that follows the answer, when EVERY pooled occurrence agrees on it."""
    seen = []
    for _t, body in _bodies(pool):
        for _before, after in _occurrences(ans, body):
            m = re.match(r"[  ]([A-Za-z]+)\b", after)
            seen.append(m.group(1) if m else "")
    seen = [s for s in seen if s]
    if not seen or len({s.lower() for s in seen}) != 1:
        return ""
    return seen[0]


def _is_count(ans: str) -> bool:
    toks = ans.split()
    return len(toks) == 1 and (bool(_NUM.match(toks[0])) or toks[0].lower() in _NUMWORD)


def rep_source_voice(ans: str, question: str, pool: list[str]) -> str:
    out = _voice_through_source(ans, pool)
    return out if out != ans else ""


def rep_unit_attach(ans: str, question: str, pool: list[str]) -> str:
    """A measured quantity is the number AND its unit; a number alone is half the span."""
    if not _is_count(ans):
        return ""
    nxt = _consensus_next_word(ans, pool)
    return ans + " " + nxt if nxt and _UNITWORD.match(nxt) else ""


def rep_head_noun(ans: str, question: str, pool: list[str]) -> str:
    """A count answers "how many WHAT"; the source names the what right after the number."""
    if not _is_count(ans) or not re.search(r"\bhow many\b", question or "", re.IGNORECASE):
        return ""
    nxt = _consensus_next_word(ans, pool)
    if not nxt or _UNITWORD.match(nxt) or _CAPTOK.match(nxt) or len(nxt) < 3:
        return ""
    return ans + " " + nxt


def rep_title_extend(ans: str, question: str, pool: list[str]) -> str:
    """One pooled article is titled with the answer plus a qualifier: that is the disambiguated name."""
    na = normalize_text(ans)
    if not na or len(na.split()) < 2:
        return ""
    hits = [t for t, _b in _bodies(pool) if normalize_text(t).startswith(na + " ")]
    return hits[0] if len(hits) == 1 else ""


def rep_comma_qualifier(ans: str, question: str, pool: list[str]) -> str:
    """Every pooled occurrence continues ", Somewhere" -- the qualifier is part of the name."""
    if not ans or not _CAPTOK.match(ans.split()[0]):
        return ""
    seen = []
    for _t, body in _bodies(pool):
        for _before, after in _occurrences(ans, body):
            m = re.match(r", ([A-Z][A-Za-z'’.\-]*(?: [A-Z][A-Za-z'’.\-]*)?)(?![A-Za-z])", after)
            seen.append(m.group(1) if m else "")
    if not seen or any(s == "" for s in seen) or len(set(seen)) != 1:
        return ""
    return ans + ", " + seen[0]


def rep_sole_conjunct(ans: str, question: str, pool: list[str]) -> str:
    """A single-target question answered with "X and Y": keep the conjunct the pool is about."""
    parts = re.split(r"\s+and\s+", ans)
    if len(parts) != 2 or re.search(r"\bboth\b|\bcommon\b", question or "", re.IGNORECASE):
        return ""
    parts = [p.strip(_STRIP_CHARS).strip() for p in parts]
    if not all(p and _CAPTOK.match(p.split()[0]) for p in parts):
        return ""
    titles = {normalize_text(t) for t, _b in _bodies(pool)}
    owned = [p for p in parts if normalize_text(p) in titles]
    return owned[0] if len(owned) == 1 else ""


REPAIRS = {
    "source_voice": rep_source_voice,
    "unit_attach": rep_unit_attach,
    "head_noun": rep_head_noun,
    "title_extend": rep_title_extend,
    "comma_qualifier": rep_comma_qualifier,
    "sole_conjunct": rep_sole_conjunct,
}


class _PrepRun:
    """Minimal RunContext stand-in so `prepare` can drive the very pipeline it is measuring."""

    def __init__(self, llm: Any, tools: Any) -> None:
        self.llm = llm
        self.tools = tools

    def note(self, message: str, **data: Any) -> None:
        return None


class Harness(BaseHarness):
    K = 7
    TRAIN_ITEMS = 70          # bounded slice of the train split replayed in prepare
    MIN_FIRES = 2             # a rule seen once is not evidence

    def __init__(self) -> None:
        self.enabled: list[str] = ["source_voice"]   # parent behaviour until train says otherwise
        self.audit: dict[str, Any] = {}
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

    # --- prepare: measure every repair on labelled train items, keep the ones that pay ---------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        records: list[dict[str, Any]] = []
        for item in train_items[: self.TRAIN_ITEMS]:
            gold = str(item.get("answer", "")).strip()
            if not gold:
                continue
            try:
                with ctx.open_tools(item) as tools:
                    st = self._solve(item["question"], _PrepRun(ctx.llm, tools))
            except Exception:  # a train item that fails to solve simply does not vote
                continue
            records.append({"q": item["question"], "pool": st["pool"],
                            "projected": st["projected"], "gold": gold})
        ctx.note("train records", n=len(records))

        scores: dict[str, dict[str, int]] = {}
        for name, fn in REPAIRS.items():
            fires = wins = losses = 0
            for rec in records:
                try:
                    out = fn(rec["projected"], rec["q"], rec["pool"])
                except Exception:
                    continue
                if not out or normalize_text(out) == normalize_text(rec["projected"]):
                    continue
                fires += 1
                was = normalize_text(rec["projected"]) == normalize_text(rec["gold"])
                now = normalize_text(out) == normalize_text(rec["gold"])
                wins += int(now and not was)
                losses += int(was and not now)
            scores[name] = {"fires": fires, "wins": wins, "losses": losses, "net": wins - losses}

        kept = [n for n, s in scores.items() if s["net"] > 0 and s["fires"] >= self.MIN_FIRES]
        self.enabled = sorted(kept, key=lambda n: (-scores[n]["net"], -scores[n]["wins"], n))
        if not self.enabled:
            # nothing verified: fall back to the one rule the parent already shipped, so the
            # system degrades to its parent rather than to no policy at all
            self.enabled = ["source_voice"]
        self.audit = {"n_records": len(records), "scores": scores}
        ctx.note("repair audit", enabled=self.enabled, scores=scores)

    def get_state(self) -> str:
        return json.dumps({"enabled": self.enabled, "audit": self.audit})[:180000]

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except Exception:
            return
        enabled = [n for n in (data.get("enabled") or []) if n in REPAIRS]
        if enabled:
            self.enabled = enabled
        self.audit = data.get("audit") or {}

    def _solve(self, question: str, ctx: RunContext) -> dict[str, Any]:
        """The parent pipeline, unchanged, up to and including the projected (unrepaired) answer."""

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
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        projected = project_answer(raw, out.get("answer_type", ""), question)
        return {
            "projected": projected,
            "pool": pool,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "next_query": next_query,
            "residual": residual,
            "missing_target": str(gap.get("missing_target", "")),
            "facts": facts,
        }

    def _repair(self, projected: str, question: str, pool: list[str]) -> tuple[str, list[str]]:
        """Apply the repairs prepare left switched on, best-measured first."""
        answer, applied = projected, []
        for name in self.enabled:
            fn = REPAIRS.get(name)
            if fn is None:
                continue
            out = fn(answer, question, pool)
            if out and normalize_text(out) != normalize_text(answer):
                answer, _ = out, applied.append(name)
        return answer, applied

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        st = self._solve(question, ctx)
        answer, applied = self._repair(st["projected"], question, st["pool"])
        ctx.note("gap hop", missing_target=st["missing_target"][:200],
                 next_query=st["next_query"], residual=st["residual"])
        ctx.note("repaired answer", answer_type=st["answer_type"], raw=str(st["raw_answer"])[:300],
                 projected=st["projected"], repaired=answer, applied=applied, enabled=self.enabled)
        return {
            "answer": answer,
            "raw_answer": st["raw_answer"],
            "answer_type": st["answer_type"],
            "query_a": question,
            "next_query": st["next_query"],
            "missing_target": st["missing_target"],
            "facts": st["facts"],
            "repairs_applied": applied,
        }
