"""Answer-form conventions learned from the labelled train pool instead of guessed at run time.

Mechanism change vs the reranking pipeline it builds on: `prepare` is no longer a no-op. The train
split is mined offline for the *surface convention* the gold answers follow, which is the one thing a
solver cannot infer from the evidence alone. Two signals are extracted and frozen into state:

  1. Shape-bucketed gold examples. Every train question is routed by a deterministic wording test
     (yes/no, either-or comparison, date, quantity, person, place, work-title, category, other) and
     stored as a bare `question -> gold answer` pair. These pairs show, per shape, how much of the
     phrase the convention keeps: whether a county name carries the word "county", whether a person is
     given with middle names, whether a year is bare or carries a month.

  2. Observed correction pairs. `prepare` drafts an answer for each train question from freshly
     retrieved passages and compares it to the gold string. Drafts that overlap the gold without
     matching it are the model's own systematic form errors, and the `draft -> gold` pair is a far
     sharper lesson than a positive example: it names the exact edit the convention requires.

At run time the shape router fires again and the matching lessons are handed to the answerer as a
separate `answer_style` field. Nothing about the evidence or the retrieval changes.

Rationale: with both required articles in context the pipeline is still wrong on a quarter of items,
and much of that is granularity, not knowledge. A previous attempt arbitrated granularity with an
extra LLM call and a longest-span bias; it broke far more answers than it fixed, because the correct
granularity is a *convention*, not something the passage reveals. Conventions have to be learned from
labels, and labels exist only on train.

Calls: 3 LLM at run time (unchanged), ~10 + one per train item during prepare.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import em_score, f1_score, normalize_text

_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+|short\s+)?answer\s*(?:is|:)\s*|^(?:it|that|this)\s+(?:is|was)\s+",
    re.IGNORECASE,
)
_YES = re.compile(r"^\s*(?:yes|yeah|true|correct|both)\b", re.IGNORECASE)
_NUM = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NONANSWER = re.compile(
    r"^(?:unknown|unclear|undetermined|not\s+(?:determined|specified|available|stated|found)|n/?a|none|"
    r"cannot\b|can't\b|insufficient)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of in on at to for and or is was are were which what who whom whose when where why how "
    "that this these those it its by with from as be been being do does did has have had you your "
    "both same first also known other another between than more most".split()
)

# -- deterministic question-shape router -----------------------------------------------------
_YESNO = re.compile(
    r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|will|would|should|<)\b"
    r"|\bboth\b.*\?|\bthe same\b",
    re.IGNORECASE,
)
_EITHER_OR = re.compile(r"\b\w[\w'.\-]*\s+or\s+\w[\w'.\-]*\b", re.IGNORECASE)
_DATE = re.compile(r"\b(?:what|which)\s+year\b|\bwhen\s+(?:was|were|did|is)\b|\bwhat\s+date\b|\bwhat\s+decade\b", re.IGNORECASE)
_QUANT = re.compile(r"\bhow\s+(?:many|much|old|long|tall|far)\b|\bwhat\s+(?:number|age|population)\b", re.IGNORECASE)
_PERSON = re.compile(r"^\s*who\b|\bwhat\s+(?:actor|actress|singer|author|writer|director|player|artist|musician|band|politician|person)\b", re.IGNORECASE)
_PLACE = re.compile(r"\bwhere\b|\bwhat\s+(?:city|country|state|county|province|town|region|place)\b|\bwhich\s+(?:city|country|state|county|province|town|region)\b", re.IGNORECASE)
_WORK = re.compile(r"\bwhat\s+(?:film|movie|album|song|book|novel|show|series|magazine|newspaper|game)\b|\bwhich\s+(?:film|movie|album|song|book|novel|show|series|magazine)\b", re.IGNORECASE)


def question_shape(question: str) -> str:
    q = (question or "").strip()
    if _YESNO.match(q) and not _EITHER_OR.search(q):
        return "yes_no"
    if _EITHER_OR.search(q):
        return "either_or"
    if _QUANT.search(q):
        return "quantity"
    if _DATE.search(q):
        return "date"
    if _PLACE.search(q):
        return "place"
    if _WORK.search(q):
        return "work_title"
    if _PERSON.search(q):
        return "person"
    return "other"


_ASKED = re.compile(r"\b(?:what|which)\b", re.IGNORECASE)


def answer_type(question: str) -> str:
    """The head noun a `what X ... ?` / `which X ... ?` question asks for, e.g. `county`, `film`, `animal`.

    This is a far sharper bucket key than the wording family, because the convention that governs how much
    of a phrase the answer keeps is a property of the thing being asked for: a question that asks for a
    county gets an answer ending in the word county, a question that asks for a station keeps the word
    station, a question that asks for a person may or may not keep middle names.
    """
    m = _ASKED.search(question or "")
    if not m:
        return ""
    tail = re.findall(r"[A-Za-z][A-Za-z'\-]*", (question or "")[m.end():])[:3]
    head = ""
    for tok in tail:
        low = tok.lower()
        if tok[:1].isupper() or low in _STOP or len(low) < 3:
            continue
        head = low
    if not head:
        return ""
    if len(head) > 4 and head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


class Harness(BaseHarness):
    K1 = 8
    K2 = 8
    MAX_FIRST_QUERIES = 3
    MAX_FOLLOWUPS = 2
    MAX_KEPT = 9
    MIN_KEPT = 5
    MAX_ANSWER_WORDS = 12

    # prepare-side limits
    DEMOS_PER_SHAPE = 14
    CORRECTIONS_PER_SHAPE = 8
    SHOWN_DEMOS = 6
    SHOWN_CORRECTIONS = 5
    MIN_HEAD_SUPPORT = 2
    SHOWN_HEAD_DEMOS = 4
    PREP_K = 10

    DECOMPOSE_INSTR = (
        "A question of this kind can only be settled by combining facts from two different articles, "
        "and one search phrased from the whole question tends to find only one of them.\n"
        "Write `queries`: two or three search queries, one per line, no numbering, no explanation.\n"
        "- Give each distinct named entity, work or event mentioned in the question its own query, "
        "worded the way that entity's own article would be titled.\n"
        "- If the question describes an entity instead of naming it, write one query out of that "
        "description using the words an article about it would use.\n"
        "- Do not repeat the question verbatim as a query, and do not write questions: write titles "
        "and noun phrases."
    )

    ASSESS_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant "
        "first, at most six numbers, separated by commas. Judge relevance by whether the passage is about "
        "an entity the question actually asks about, not by superficial word overlap.\n"
        "`missing_info`: in one line, the fact the question still needs and that no listed passage states. "
        "Write exactly `none` if the passages together already settle the question.\n"
        "`followup_queries`: if something is missing, one or two queries, one per line. Each must START with the "
        "name of the intermediate entity the passages just revealed (the person, work, place or organisation "
        "that links the question to the missing fact), because that entity's own article is what must be "
        "retrieved next, and may add one or two words naming the missing property. Write titles and noun "
        "phrases, never questions. Write exactly `none` when nothing is missing."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, or the single "
        "word yes or no when the question asks whether something holds. Never write a sentence, never "
        "restate the question, never explain. If the evidence is incomplete, give the most likely short "
        "answer anyway rather than saying it cannot be determined.\n"
        "`answer_style` shows how questions worded like this one were answered elsewhere, and where a "
        "first draft had to be corrected. It carries no facts about the present question: read it only for "
        "how much of the phrase to keep -- which qualifiers, head nouns, given names or units belong in the "
        "answer and which do not -- and match that convention exactly."
    )

    PREP_DRAFT_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, or the single "
        "word yes or no when the question asks whether something holds. Never write a sentence, never "
        "restate the question, never explain. If the evidence is incomplete, give the most likely short "
        "answer anyway rather than saying it cannot be determined."
    )

    NOTES_INSTR = (
        "`examples` lists solved questions of one wording family, each with the answer string that was "
        "accepted. `corrections` lists first drafts that were judged wrong next to the accepted string.\n"
        "Write `notes`: at most four short imperative lines stating the SURFACE convention these answers "
        "follow -- how long an answer runs, whether a head noun or qualifier is kept or dropped, whether a "
        "person is given in full or short form, how a date or quantity is written, what an either-or "
        "question returns. Describe form only. Never mention any specific entity, title, place, person or "
        "number from the examples, and never state a fact. If a rule is not supported by the examples, "
        "leave it out."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.assess = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "missing_info", "followup_queries"],
            instructions=self.ASSESS_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "answer_style"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )
        self.prep_draft = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.PREP_DRAFT_INSTR)
        self.notes_mod = Module(inputs=["examples", "corrections"], outputs=["notes"], instructions=self.NOTES_INSTR)
        self.demos: dict[str, list[list[str]]] = {}
        self.corrections: dict[str, list[list[str]]] = {}
        self.heads: dict[str, list[list[str]]] = {}
        self.notes: dict[str, str] = {}

    # -- state -----------------------------------------------------------------------------
    def get_state(self) -> str:
        return json.dumps(
            {"demos": self.demos, "corrections": self.corrections, "heads": self.heads, "notes": self.notes},
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        try:
            blob = json.loads(state or "{}")
        except Exception:
            blob = {}
        self.demos = blob.get("demos") or {}
        self.corrections = blob.get("corrections") or {}
        self.heads = blob.get("heads") or {}
        self.notes = blob.get("notes") or {}

    # -- prepare ---------------------------------------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        demos: dict[str, list[list[str]]] = {}
        corrections: dict[str, list[list[str]]] = {}
        heads: dict[str, list[list[str]]] = {}

        for it in train_items:
            q = str(it.get("question") or "")
            goldset = it.get("answer")
            gold = str(goldset[0] if isinstance(goldset, list) and goldset else goldset or "")
            if not q or not gold:
                continue
            shape = question_shape(q)
            demos.setdefault(shape, []).append([q, gold])
            head = answer_type(q)
            if head:
                heads.setdefault(head, []).append([q, gold])

            # Draft the answer the way the run-time answerer would, and keep the near-misses:
            # they are exactly the edits the convention demands.
            try:
                with ctx.open_tools(it) as tools:
                    passages = tools.retrieve(q, k=self.PREP_K)
                evidence = [f"{i + 1}. {p}" for i, p in enumerate(passages[: self.MIN_KEPT + 2])]
                out = self.prep_draft(ctx.llm, tag="prep_draft", question=q, evidence=evidence)
                draft = self._clean(out.get("answer", ""))
            except Exception:
                continue
            if not draft or em_score(draft, gold):
                continue
            # Overlapping but unequal: a surface-form miss rather than a knowledge miss.
            if f1_score(draft, gold) > 0.0:
                corrections.setdefault(shape, []).append([q, draft, gold])

        # Keep the pools small and varied: prefer short questions so the style block stays cheap.
        self.demos = {
            s: sorted(v, key=lambda p: len(p[0]))[: self.DEMOS_PER_SHAPE] for s, v in demos.items()
        }
        self.corrections = {
            s: sorted(v, key=lambda p: len(p[0]))[: self.CORRECTIONS_PER_SHAPE] for s, v in corrections.items()
        }
        # Only head nouns the train pool actually attests are worth a bucket of their own.
        self.heads = {
            h: sorted(v, key=lambda p: len(p[0]))[: self.DEMOS_PER_SHAPE]
            for h, v in heads.items()
            if len(v) >= self.MIN_HEAD_SUPPORT
        }

        # One call per shape turns the raw pairs into an explicit, entity-free style rule.
        for shape in sorted(self.demos):
            ex = [f"{q} -> {a}" for q, a in self.demos[shape]]
            co = [f"{q}\n   drafted: {d}   accepted: {g}" for q, d, g in self.corrections.get(shape, [])]
            if len(ex) < 3:
                continue
            try:
                out = self.notes_mod(ctx.llm, tag=f"notes_{shape}", examples=ex, corrections=co or ["none"])
                text = " ".join((out.get("notes") or "").split("\n")[:4]).strip()
            except Exception:
                text = ""
            if text and len(text) < 700:
                self.notes[shape] = text
        ctx.note(
            "learned answer-form conventions",
            shapes={s: [len(self.demos.get(s, [])), len(self.corrections.get(s, []))] for s in self.demos},
            head_buckets=sorted(self.heads),
        )

    # -- style block -----------------------------------------------------------------------
    def _style_block(self, question: str) -> list[str]:
        shape = question_shape(question)
        qw = _content_words(question)
        block: list[str] = []
        note = self.notes.get(shape)
        if note:
            block.append(f"convention for questions worded like this one: {note}")

        # The head noun the question asks for is the sharpest available key; show those first.
        head = answer_type(question)
        shown: set[str] = set()
        for q, a in (self.heads.get(head) or [])[: self.SHOWN_HEAD_DEMOS]:
            block.append(f"asked: {q[:190]}  ->  answered: {a}")
            shown.add(q)

        pool = list(self.demos.get(shape) or [])
        if len(pool) < 3:  # thin bucket: fall back to the generic pool
            for s, v in self.demos.items():
                if s != shape:
                    pool.extend(v)
        pool.sort(key=lambda p: -len(qw & _content_words(p[0])))
        for q, a in [p for p in pool if p[0] not in shown][: self.SHOWN_DEMOS - len(shown)]:
            block.append(f"asked: {q[:190]}  ->  answered: {a}")

        cpool = list(self.corrections.get(shape) or [])
        cpool.sort(key=lambda p: -len(qw & _content_words(p[0])))
        for q, draft, goldv in cpool[: self.SHOWN_CORRECTIONS]:
            if normalize_text(draft) == normalize_text(goldv):
                continue
            block.append(f"corrected: a draft of \"{draft}\" should have been \"{goldv}\"")
        return block

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”').strip()
            if not line or line.lower() in {"none", "n/a", "null"}:
                continue
            if len(line) > 300:
                line = line[:300]
            if line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    def _add(self, pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = self._title_of(p)
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip()
        ans = ans.strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = queries[: self.MAX_FIRST_QUERIES - 1] + [question] if queries else [question]

        pool: list[str] = []
        seen: set[str] = set()
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K1))

        numbered = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        judged = self.assess(ctx.llm, tag="assess", question=question, passages=numbered)
        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]

        followups = self._lines(judged.get("followup_queries", ""), self.MAX_FOLLOWUPS)
        missing = (judged.get("missing_info", "") or "").strip().lower()
        sufficient = (not followups) or missing.startswith("none")
        if not sufficient:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for q in followups:
                self._add(round2, seen2, ctx.tools.retrieve(q, k=self.K2))
            kept = kept + round2
        if len(kept) < self.MIN_KEPT:
            have = {self._title_of(p) for p in kept}
            for p in pool:
                if self._title_of(p) not in have:
                    kept.append(p)
                    have.add(self._title_of(p))
                if len(kept) >= self.MIN_KEPT:
                    break

        style = self._style_block(question)
        ctx.note(f"shape={question_shape(question)} asked_for={answer_type(question)!r} style_lines={len(style)}")
        evidence = [f"{i + 1}. {p}" for i, p in enumerate(kept)]
        out = self.answer(
            ctx.llm, tag="answer", question=question, evidence=evidence, answer_style=style or ["none"]
        )
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = kept[0].split("|", 1)[0].strip() if kept else answer
        return {
            "answer": answer,
            "shape": question_shape(question),
            "asked_for": answer_type(question),
            "queries": queries,
            "followups": followups,
            "n_style": len(style),
            "n_evidence": len(kept),
        }
