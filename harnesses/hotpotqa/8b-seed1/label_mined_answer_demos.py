"""Answer-surface demonstrations mined from the labelled train split.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%): 33 of its 101 errors happen with BOTH
gold articles already in the context window -- the evidence is there and the reasoning is right,
but the surface form of the answer is wrong. The traces show one systematic sub-class: the model
answers with the name the *question* used while the reference span is the fuller name the *article*
uses (short form vs. the lead-sentence form), and a mirror-image sub-class where it copies a
disambiguator or a trailing qualifier that the reference span does not carry, or drops a unit or
head noun that it does. Prose instructions cannot settle this: "shortest form that fully names it"
and "copy the span verbatim" pull in opposite directions, and the correct granularity is a
*convention of the labels*, not something derivable from the question.

Measured on the labelled train split (no eval data touched): 114 of 139 non-yes/no reference
answers are a contiguous substring of the gold article openings, and the median answer is 2 words
long. So the convention is learnable -- and the only honest place to learn it from is the labels.

Mechanism under test (prepare-time learning; retrieval, the evidence pool and the call/retrieval
budget are held identical to the parent so the comparison isolates this change):

  LABEL-MINED DEMONSTRATIONS. `prepare` walks the train split, pulls each item's gold article
  openings out of the index, locates the reference answer inside them, and programmatically
  classifies the *surface relation* between answer and evidence (yes/no; span copied from the
  evidence; option echoed from the question; span that extends the article title; span that is
  shorter than the title). It keeps one compact, real worked example per class -- question, the
  actual passages, the sentence the answer sits in, and the reference answer exactly as the labels
  write it -- and turns them into few-shot demonstrations for the answering module, together with
  the statistics it just measured. At eval time the answering module sees those demonstrations, so
  the granularity question is answered by example instead of by adjective.

No labels, questions or titles are written into this file: everything comes out of `prepare`.

Budget: 3 LLM calls, 2 retrievals, k=7 -- inside the seed program's interaction depth.
`prepare` uses retrieval only (no LLM calls), so it costs seconds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PLAN_INSTRUCTIONS = """The question needs facts from two different encyclopedia articles. Plan the
two lookups before searching.

Work out which two things must be looked up. For a comparison question they are the two entities
being compared. For a bridge question the first is the entity named in the question and the second
is the entity that the question describes only indirectly and that must be identified first.

`query_a` and `query_b` are keyword queries for a sparse full-text index over encyclopedia article
openings. Therefore:
- each query names exactly ONE entity, plus two or three disambiguating content words (its type,
  field, place or date) -- e.g. a person plus their profession, a work plus its medium.
- never copy the question verbatim, never keep question words like "which", "who", "both", "same",
  "more", "first"; a query mixing both entities retrieves articles about neither.
- prefer the proper name exactly as the question spells it.
`query_a` must be the lookup that is possible right now; if the second entity is not yet named,
make `query_b` the best guess from the description in the question.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other."""

RESOLVE_INSTRUCTIONS = """You have the passages returned for the first lookup. Do two things.

`facts`: copy out, verbatim, only the sentences or clauses from the passages that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If no passage is relevant, write "none".

`next_query`: the keyword query for the second and final lookup. Use what you just learned: if the
question described the second entity indirectly and the passages have now named it, search for that
name. Otherwise search for the other entity in the question. Rules:
- name exactly ONE entity plus two or three disambiguating content words;
- it must NOT be the question and must NOT repeat `previous_query`; the second lookup is wasted if
  it returns the same articles;
- no question words, no punctuation, just the search terms."""

# The prose below is deliberately thin on granularity: the worked examples appended by `prepare`
# are what fixes the surface form. {STATS} is filled in with what prepare measured.
ANSWER_INSTRUCTIONS = """Answer the question from the evidence: `facts` are the sentences already
extracted, `passages` are the raw article openings retrieved for this question.

Work in three steps. `evidence_quote`: copy the one sentence from the evidence that settles the
question, character for character. `answer_type`: exactly one of yes_no, person, organization,
work_title, place, date, number, other. `answer`: the reference span itself.

The answer is graded by exact string match against a short reference span written by the people who
read these same article openings, so it must be the bare span and nothing else:
- Never a sentence, never a restatement of the question, no explanation, no "because", no trailing
  clause, no second sentence, no markdown, no quotation marks, no final period.
- If `answer_type` is yes_no, `answer` is exactly `yes` or `no` -- one word, nothing after it.
  Questions of the form "are both ...", "do both ...", "is X the same as Y", "can both ..." are
  yes_no questions, even when the interesting content is in the reasoning.
- Otherwise `answer` must be a contiguous run of words lifted out of `evidence_quote`. Take the
  span exactly as that sentence writes it: keep the words the sentence attaches to it, and do not
  add or remove words to make it read better. If the question offered a closed list of options,
  the span is the winning option as the question spells it.
- Never answer "unknown", "not stated" or "cannot be determined"; give the most likely span.
{STATS}
The worked examples above come from previously answered questions of this kind: match the way they
choose how much of the sentence to keep."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "query_a": "keyword query for the first lookup, one entity only",
    "query_b": "keyword query for the second lookup, one entity only",
    "previous_query": "the query already used for the first lookup",
    "planned_query_b": "the second lookup planned before searching, to revise if needed",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "next_query": "keyword query for the second and final lookup",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9'’.\-&]+")
_SPLIT_SENT = re.compile(r"(?<=[a-z0-9\)\]\"'])[.!?](?:\s+|$)(?=[A-Z0-9\"']|$)")


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


def _as_query(text: Any, fallback: str) -> str:
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    return q if len(q) >= 3 else fallback


# -- label mining -----------------------------------------------------------------------------

def _shrink(text: str, limit: int = 330) -> str:
    """Keep a passage short enough that five demonstrations still fit in the context window."""
    t = " ".join(str(text or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit // 2 else cut.rstrip() + " ...")


def _sentences(text: str) -> list[str]:
    body = str(text or "")
    parts = [p.strip() for p in _SPLIT_SENT.split(body) if p and p.strip()]
    return parts or [body.strip()]


def find_answer_sentence(answer: str, passages: list[str]) -> str:
    """The sentence of the evidence that literally contains the reference span."""
    na = normalize_text(answer)
    if not na:
        return ""
    for p in passages:
        for s in _sentences(p):
            if na in normalize_text(s):
                return " ".join(s.split())[:400]
    return ""


def classify_surface(answer: str, question: str, titles: list[str], quote: str) -> str:
    """How does the reference span relate to what the model has in front of it?"""
    na = normalize_text(answer)
    if na in ("yes", "no"):
        return "yes_no"
    if any(na == normalize_text(o) for o in _question_options(question)):
        return "option_echo"
    atok, qtok = _WORD.findall(answer), None
    for t in titles:
        bare = re.sub(r"\s*\([^)]*\)$", "", str(t).replace("&amp;", "&")).strip()
        ttok = _WORD.findall(bare)
        if normalize_text(bare) == na:
            return "equals_title"
        if len(atok) >= 2 and len(ttok) >= 2 and atok[0].lower() == ttok[0].lower() \
                and atok[-1].lower() == ttok[-1].lower() and len(atok) != len(ttok):
            return "extends_title" if len(atok) > len(ttok) else "trims_title"
    del qtok
    if quote and len(atok) >= 2:
        return "multiword_span"
    return "short_span"


# one demonstration per surface class, in the order they are most worth teaching
DEMO_CLASSES = ["extends_title", "option_echo", "multiword_span", "yes_no", "equals_title", "short_span"]
MAX_DEMOS = 5


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.demos: list[dict[str, Any]] = []
        self.stats_note: str = ""
        self._build_modules()

    # -- modules ------------------------------------------------------------------------------
    def _build_modules(self) -> None:
        stats = ("\n" + self.stats_note + "\n") if self.stats_note else "\n"
        self.plan = Module(
            inputs=["question"],
            outputs=["answer_type", "query_a", "query_b"],
            instructions=PLAN_INSTRUCTIONS,
            descs=DESCS,
        )
        self.resolve = Module(
            inputs=["question", "planned_query_b", "previous_query", "passages"],
            outputs=["facts", "next_query"],
            instructions=RESOLVE_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS.replace("{STATS}", stats),
            descs=DESCS,
            demos=self.demos,
        )

    # -- prepare: mine the labels ---------------------------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        by_class: dict[str, dict[str, Any]] = {}
        n_span = n_total = n_yesno = 0
        lengths: list[int] = []

        for it in train_items:
            answer = str(it.get("answer", "")).strip()
            titles = [str(t) for t in (it.get("gold_titles") or [])]
            if not answer or not titles:
                continue
            passages: list[str] = []
            try:
                with ctx.open_tools(it) as tools:
                    for t in titles[:2]:
                        probe = re.sub(r"\s*\([^)]*\)$", "", t.replace("&amp;", "&")).strip() or t
                        for d in tools.retrieve_docs(probe, k=3):
                            if str(d.get("title", "")).strip().lower() == t.strip().lower():
                                passages.append(str(d.get("text", "")).strip())
                                break
            except Exception:  # a train item we cannot fetch simply contributes no demo
                continue
            if not passages:
                continue

            na = normalize_text(answer)
            n_total += 1
            lengths.append(len(answer.split()))
            if na in ("yes", "no"):
                n_yesno += 1
            elif na and na in normalize_text(" ".join(passages)):
                n_span += 1

            quote = find_answer_sentence(answer, passages) if na not in ("yes", "no") else ""
            if na not in ("yes", "no") and not quote:
                continue  # cannot show a faithful worked example without the sentence
            cls = classify_surface(answer, str(it.get("question", "")), titles, quote)
            if cls not in DEMO_CLASSES or cls in by_class:
                continue
            if len(answer.split()) > 6:
                continue
            shown = [_shrink(p) for p in passages]
            if not quote:
                quote = _shrink(_sentences(passages[0])[0], 240)
            atype = self._demo_answer_type(answer, cls)
            by_class[cls] = {
                "question": " ".join(str(it.get("question", "")).split()),
                "facts": "\n".join(f"[{t}] {q}" for t, q in zip(titles, [_shrink(_sentences(p)[0], 200) for p in passages])),
                "passages": shown,
                "reasoning": self._demo_reasoning(cls, quote),
                "evidence_quote": quote,
                "answer_type": atype,
                "answer": answer,
            }

        self.demos = [by_class[c] for c in DEMO_CLASSES if c in by_class][:MAX_DEMOS]
        if n_total:
            med = sorted(lengths)[len(lengths) // 2]
            self.stats_note = (
                f"Reference spans from {n_total} previously answered questions of this kind: "
                f"{n_span} of the {n_total - n_yesno} non-yes/no answers appear word for word inside the "
                f"article openings, {n_yesno} are yes/no, and the median answer is {med} words long."
            )
        ctx.note("label mining", n_train=len(train_items), n_scored=n_total,
                 classes=sorted(by_class), stats=self.stats_note)
        self._build_modules()

    @staticmethod
    def _demo_answer_type(answer: str, cls: str) -> str:
        if cls == "yes_no":
            return "yes_no"
        a = answer.strip()
        if re.fullmatch(r"[\d,.]+(\s*[a-zA-Z%°]+)?", a):
            return "number"
        if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", a) and len(a.split()) <= 4:
            return "date"
        return "other"

    @staticmethod
    def _demo_reasoning(cls: str, quote: str) -> str:
        why = {
            "yes_no": "Both sides of the comparison are settled by the passages, so the answer is the bare yes/no verdict.",
            "option_echo": "The question offered a closed list; the evidence picks one of them, so the answer is that option as the question spells it.",
            "extends_title": "The passage's own sentence writes the name out more fully than the article title does; the answer keeps the sentence's wording.",
            "trims_title": "The article title carries a qualifier the sentence does not; the answer keeps only what the sentence says.",
            "equals_title": "The evidence sentence names the thing the question asks for, and the answer is that name exactly as written.",
            "multiword_span": "The answer is the run of words the evidence sentence attaches to what the question asks for, kept whole.",
            "short_span": "The evidence sentence states the fact directly; the answer is the span it uses.",
        }.get(cls, "The evidence sentence states the fact directly; the answer is the span it uses.")
        return f"{why} Relevant sentence: «{quote[:220]}»"

    # -- state ---------------------------------------------------------------------------------
    def get_state(self) -> str:
        return json.dumps({"demos": self.demos, "stats_note": self.stats_note}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            blob = json.loads(state or "{}")
        except ValueError:
            blob = {}
        self.demos = list(blob.get("demos") or [])
        self.stats_note = str(blob.get("stats_note") or "")
        self._build_modules()

    # -- run -----------------------------------------------------------------------------------
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

        plan = self.plan(ctx.llm, tag="plan", question=question)
        query_a = _as_query(plan.get("query_a"), question)
        planned_b = _as_query(plan.get("query_b"), "")

        docs_a = ctx.tools.retrieve_docs(query_a, k=self.K)

        res = self.resolve(
            ctx.llm,
            tag="resolve",
            question=question,
            planned_query_b=planned_b,
            previous_query=query_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(res.get("facts", "")).strip()
        next_query = _as_query(res.get("next_query"), planned_b or question)
        if normalize_text(next_query) == normalize_text(query_a):
            next_query = planned_b or question

        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("plan", query_a=query_a, next_query=next_query, n_demos=len(self.demos))
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": next_query,
            "facts": facts,
        }
