"""Query formulation grounded in copied text, with a verified bridge-entity slate.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%): 68 of its 101 errors miss at least
one gold article, and the traces show *why*. Both of its query modules are free to write any
string, so the model routinely fills the unnamed side of a bridge question from parametric memory
and then spends a lookup on an entity that never appeared anywhere:
  - "The star of <film> founded which school of fighting?" -> hop A "<invented actor> <film>",
    hop B "<invented actor> martial arts school"; neither gold article is reachable.
  - "In what year was this Irish director of <film> born?" -> hop B "<invented director> birth
    year", even though hop A had already returned the film's article naming the real director.
Once an invented name enters a query, BM25 confidently returns 7 articles about the wrong person
and both the second lookup and the answer are lost.

Mechanism under test (query formulation only; retrieval depth, evidence pool and the answer
contract are held fixed against the parent so the comparison isolates this change):

  GROUNDED-COPY CONSTRAINT -- every entity that reaches the index must be a string that was
  literally copied from the question or from a passage already in hand.

  1. plan   -- names the `anchor`, which must be *copied from the question*, and describes the
               still-unknown second article as a `target_desc` instead of guessing its name.
               A deterministic filter deletes from the anchor any capitalised token that is not
               in the question, so an invented name cannot survive to the index.
  2. hop A  -- retrieve(anchor)
  3. bridge -- reads hop A and emits a *slate*: up to four candidate names copied verbatim out of
               the passages (or the question) that could be the second article, then picks one and
               writes `next_query`. A deterministic verifier checks the head name of the chosen
               query against the question + hop-A text; if it is ungrounded it walks the slate for
               the first grounded candidate, and only then falls back to the planned backup.
  4. hop B  -- retrieve(next_query)
  5. answer -- unchanged typed contract over the deduplicated raw pool.

Budget: 3 LLM calls, 2 retrievals, k=7 -- inside the seed program's interaction depth.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PLAN_INSTRUCTIONS = """The question needs facts from two different encyclopedia articles. Plan the
first lookup, and describe -- do not guess -- the second.

`anchor` is the keyword query for the first lookup, against a sparse full-text index over
encyclopedia article openings. Rules:
- every word of `anchor` must be COPIED FROM THE QUESTION. Do not introduce a name, title or date
  that the question does not contain: at this point you have read no sources, so any name you
  supply yourself is a guess, and a guessed name sends the search to the wrong article entirely.
- it names exactly ONE entity, spelled exactly as the question spells it, plus at most three
  further words from the question that say what kind of thing it is.
- drop question words ("which", "who", "what", "both", "same", "more", "first"), drop the second
  entity, drop punctuation. A query that mixes two entities retrieves articles about neither.
- choose the entity that the question names most explicitly; that is the one that can be looked up
  right now.

`target_desc` describes the OTHER article -- the one the answer depends on -- in the question's own
words: what kind of thing it is and how it relates to the anchor (for example "the actor who starred
in that film", "the river that the bridge crosses"). If the question already names both entities,
`target_desc` is simply that second name copied from the question. Never invent a name here.

`backup_query` is a keyword query for the second lookup built ONLY from words in the question, to
be used if the first lookup turns up nothing usable.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other."""

BRIDGE_INSTRUCTIONS = """The passages are the result of the first lookup. You have one lookup left.
Everything you send to the index must be text you can point at, not text you remember.

`facts`: copy out, verbatim, only the sentences or clauses from the passages that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If no passage is relevant, write "none".

`candidates`: the shortlist for the second lookup. List up to four names, one per line, that could
be the title of the other article the question depends on. Every name must be COPIED CHARACTER FOR
CHARACTER out of the passages above or out of the question -- a name you merely believe to be right
is not allowed on this list. Take them from where `target_desc` points: the person, work, place or
organisation that the passages mention while describing what the question asks about. Order them
best first. Write "none" only if the passages and the question between them contain no usable name.

`next_query`: the keyword query for the second and final lookup. Take the first name from
`candidates`, copy it exactly, and add two or three disambiguating words (its type, field, place or
date) drawn from the passages or the question. Rules:
- exactly one entity; no question words; no punctuation beyond what the name itself contains;
- it must not repeat `previous_query`: a lookup that returns the same articles is wasted;
- if `candidates` is "none", fall back to `backup_query`."""

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
    "anchor": "keyword query for the first lookup, built only from words in the question",
    "target_desc": "what the second article is, described in the question's words -- never a guessed name",
    "backup_query": "fallback keyword query for the second lookup, built only from words in the question",
    "previous_query": "the query already used for the first lookup",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "candidates": "up to four names copied verbatim from the passages or the question, best first",
    "next_query": "keyword query for the second and final lookup, built on a copied name",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9'’.\-&]+")
# a proper-noun-looking token: starts with a capital or is a bare number/year
_PROPER = re.compile(r"^(?:[A-ZÀ-Þ][\w'’.\-&]*|\d[\d\-]*)$")
_STOPQ = {
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "the", "a", "an",
    "of", "in", "on", "at", "for", "to", "and", "or", "is", "are", "was", "were", "did", "does",
    "do", "both", "same", "more", "most", "first", "that", "this", "his", "her", "their", "it",
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


def _as_query(text: Any, fallback: str) -> str:
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    return q if len(q) >= 3 else fallback


# -- the grounded-copy constraint ------------------------------------------------------------

def _vocab(*texts: str) -> set[str]:
    """Lowercased word vocabulary of everything we are allowed to copy from."""
    out: set[str] = set()
    for t in texts:
        out.update(w.lower().strip(".,;:") for w in _WORD.findall(str(t or "")))
    return out


def ground_anchor(anchor: str, question: str, fallback: str) -> str:
    """Delete from the first query any proper-noun token the question never used.

    A name the model supplied itself, before reading a single source, is a guess; BM25 answers a
    guessed name with seven confident articles about the wrong entity.
    """
    q = _as_query(anchor, fallback)
    allowed = _vocab(question)
    kept = [w for w in q.split() if not _PROPER.match(w.strip(_STRIP_CHARS)) or w.lower().strip(_STRIP_CHARS + "") in allowed]
    kept = [w for w in kept if w.strip(_STRIP_CHARS)]
    out = " ".join(kept).strip(_STRIP_CHARS).strip()
    # if scrubbing gutted the query, keep the question's own content words instead
    if len(out) < 3 or not any(_PROPER.match(w) for w in out.split()):
        content = [w for w in str(question).split() if w.lower().strip(_STRIP_CHARS) not in _STOPQ]
        out = " ".join(content[:10]).strip(_STRIP_CHARS).strip() or fallback
    return out


def _head_name(query: str) -> str:
    """Longest run of proper-noun-looking tokens in a query -- the entity it will actually match."""
    toks = str(query or "").split()
    best: list[str] = []
    cur: list[str] = []
    for t in toks:
        if _PROPER.match(t.strip(_STRIP_CHARS)):
            cur.append(t.strip(_STRIP_CHARS))
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return " ".join(best)


def is_grounded(query: str, source_norm: str) -> bool:
    """True when the entity the query is about was actually copied from something we have read.

    Judged on the name, not the whole query: a copied name with descriptive words appended is
    still grounded, while a name that appears nowhere in the question or the retrieved text is not.
    """
    head = normalize_text(_head_name(query))
    if not head:
        return True  # no proper name at all: nothing was invented
    if head in source_norm:
        return True
    parts = head.split()
    if len(parts) == 1:
        return f" {parts[0]} " in f" {source_norm} "
    return any(f"{parts[i]} {parts[i + 1]}" in source_norm for i in range(len(parts) - 1))


_QUERY_JUNK = re.compile(
    r"\b(which|who|whom|whose|what|when|where|why|how|did|does|do|is|are|was|were|the|a|an)\b",
    re.IGNORECASE,
)


def scrub_query(query: str, fallback: str) -> str:
    """Keep the final lookup a keyword query: no question words, no stray punctuation."""
    q = str(query or "").strip().strip(_STRIP_CHARS)
    q = q.replace("?", " ").replace('"', " ").replace("\u201c", " ").replace("\u201d", " ")
    if _QUERY_JUNK.search(q):
        stripped = _QUERY_JUNK.sub(" ", q)
        stripped = " ".join(stripped.split()).strip(_STRIP_CHARS)
        # only accept the scrub if a real entity survives it
        if len(stripped) >= 3 and any(_PROPER.match(w) for w in stripped.split()):
            q = stripped
    q = " ".join(q.split()).strip(_STRIP_CHARS).strip()
    return q if len(q) >= 3 else fallback


def parse_slate(raw: Any) -> list[str]:
    out: list[str] = []
    for line in str(raw or "").splitlines():
        cand = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip(_STRIP_CHARS).strip()
        cand = re.sub(r"\s*\((?:from|source|title)[^)]*\)\s*$", "", cand, flags=re.IGNORECASE).strip()
        if not cand or normalize_text(cand) in ("none", ""):
            continue
        if len(cand.split()) > 10:
            continue
        out.append(cand)
        if len(out) >= 4:
            break
    return out


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question"],
            outputs=["answer_type", "anchor", "target_desc", "backup_query"],
            instructions=PLAN_INSTRUCTIONS,
            descs=DESCS,
        )
        self.bridge = Module(
            inputs=["question", "target_desc", "backup_query", "previous_query", "passages"],
            outputs=["facts", "candidates", "next_query"],
            instructions=BRIDGE_INSTRUCTIONS,
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

        # 1. plan: name only what the question names, describe the rest
        plan = self.plan(ctx.llm, tag="plan", question=question)
        raw_anchor = _as_query(plan.get("anchor"), question)
        query_a = ground_anchor(raw_anchor, question, question)
        target_desc = str(plan.get("target_desc", "")).strip().split("\n")[0][:300] or "the other article the answer depends on"
        backup = ground_anchor(_as_query(plan.get("backup_query"), ""), question, "") if plan.get("backup_query") else ""

        # 2. first hop
        docs_a = ctx.tools.retrieve_docs(query_a, k=self.K)
        texts_a = [d["text"] for d in docs_a]

        # 3. bridge: a slate of copied names, then a query built on one of them
        br = self.bridge(
            ctx.llm,
            tag="bridge",
            question=question,
            target_desc=target_desc,
            backup_query=backup or "(none)",
            previous_query=query_a,
            passages=texts_a,
        )
        facts = str(br.get("facts", "")).strip()
        slate = parse_slate(br.get("candidates"))
        next_query = _as_query(br.get("next_query"), "")

        # 4. verify the chosen entity was actually copied from something we have read
        source_norm = normalize_text(question + " " + " ".join(texts_a))
        chosen, why = next_query, "model"
        if not chosen or normalize_text(chosen) == normalize_text(query_a) or not is_grounded(chosen, source_norm):
            chosen, why = "", "rejected"
            for cand in slate:
                if is_grounded(cand, source_norm) and normalize_text(cand) != normalize_text(query_a):
                    chosen, why = cand, "slate"
                    break
            if not chosen:
                chosen, why = (backup or target_desc or question), "backup"
        if normalize_text(chosen) == normalize_text(query_a):
            chosen, why = (backup or question), "backup"
        chosen = scrub_query(chosen, backup or query_a)

        docs_b = ctx.tools.retrieve_docs(chosen, k=self.K)

        # 5. answer over the deduplicated raw pool
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("grounded queries", raw_anchor=raw_anchor, query_a=query_a, slate=slate,
                 model_query=next_query, chosen=chosen, source=why)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": chosen,
            "query_source": why,
            "candidates": slate,
            "facts": facts,
        }
