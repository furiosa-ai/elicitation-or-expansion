"""Align the evidence pool to the question's entities and expose what the two sides share.

The frontier hands the answerer a flat, deduplicated list of up to fourteen article openings and
lets it work out on its own which article covers which side of the question. Two-entity questions
are where that costs most: on the search set the frontier answers bridge questions with both target
articles in context 87.4% of the time but comparison questions only 64.4%, and reading those errors
the model is not reasoning badly, it is *paraphrasing a shared descriptor* -- it writes a synonym or
a plural of the word both articles literally print ("pizzeria" where both openings say pizza,
"authors" where the reference span is writer, "township" where both say city, "unrelated" where both
openings say genus), or it answers with a property of one side only.

Mechanism under test (context construction; retrieval is byte-identical to `decomposed_evidence_pool`
-- same plan call, same two queries, same k -- so the comparison isolates how the evidence is laid
out for the final call):
  * the pool is no longer a flat list. Proper-name spans are extracted from the question
    deterministically, every pooled passage is assigned to the span its title matches, and the
    answerer receives an evidence TABLE: one block per question entity, then the unassigned
    passages, so "which article covers which side" is resolved before reasoning starts.
  * for questions naming two entities that both have a block, a `shared_descriptors` field is
    computed deterministically: the word and phrase types that occur in BOTH blocks (stopword-only
    grams dropped, entity names dropped). This makes the *intersection of the two articles* an
    explicit, copyable object instead of something the model has to reconstruct in prose.
  * the answer contract gains one clause tied to that field: a question asking what two entities
    have in common, or what type both are, is answered by the shared descriptor as the articles
    print it, not by a paraphrase of it.

Budget: 3 LLM calls, 2 retrievals, k=7 -- identical to the parent's interaction depth.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from core.harness import BaseHarness, RunContext
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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence.

`evidence_table` groups every retrieved article opening under the entity from the question whose
article it is; articles that match no entity are listed last. `facts` are sentences already copied
out. `shared_descriptors` lists the words and phrases that occur in the articles of BOTH entities --
computed mechanically, so some are coincidental, but if the question asks what the two entities have
in common this is where the answer is printed.

Read the block for each entity the question is about before you answer, and use the blocks to check
that the fact you use is attached to the right entity.

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
- If the question asks what two entities have in common, what type or kind both are, or how they are
  related, the answer is the descriptor their two articles SHARE, written exactly as those articles
  print it -- one word or a short phrase, lower case, singular if the articles print it in the
  singular. Never substitute a synonym, a plural, or a description of your own.
- Otherwise copy the answer span verbatim from the evidence, in the shortest form that fully
  names it: just the name, the title, the place, the date or the number.
- Do not add appositives, descriptors, units of explanation or parentheses that the reference
  span would not contain.
- If the evidence is incomplete, still give the single most likely bare answer. Never answer
  "unknown", "not stated" or "cannot be determined"."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "query_a": "keyword query for the first lookup, one entity only",
    "query_b": "keyword query for the second lookup, one entity only",
    "previous_query": "the query already used for the first lookup",
    "planned_query_b": "the second lookup planned before searching, to revise if needed",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "evidence_table": "retrieved article openings grouped under the question entity they cover",
    "shared_descriptors": "words and phrases printed by the articles of both entities",
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
_WORD = re.compile(r"[A-Za-z0-9]+")
_CAP_RUN = re.compile(r"[A-Z][\w'’\.\-]*(?:\s+(?:of|the|de|di|da|van|von|der|del|la|le|for|on|in)?\s*[A-Z][\w'’\.\-]*)*")
_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,70})[\"“”']")
_LEADCAP = re.compile(r"^(which|who|whom|whose|what|when|where|why|how|are|is|was|were|do|does|did|has|have|had|can|between|both|the|a|an)$", re.IGNORECASE)
_STOP = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "from", "as",
    "by", "is", "are", "was", "were", "be", "been", "being", "it", "its", "he", "she", "they", "his",
    "her", "their", "them", "this", "that", "these", "those", "which", "who", "whom", "whose",
    "what", "when", "where", "why", "how", "has", "have", "had", "also", "not", "no", "than",
    "then", "there", "s", "t", "known", "may", "one", "two", "both", "other", "into", "about",
    "after", "before", "between", "over", "under", "while", "during", "such", "same", "more",
    "most", "some", "any", "all", "can", "will", "would", "may", "used", "including", "well",
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


def question_entities(question: str) -> list[str]:
    """Proper-name spans of the question, in order, longest form kept."""
    q = str(question or "")
    spans: list[str] = []
    for m in _QUOTED.finditer(q):
        spans.append(m.group(1).strip(_STRIP_CHARS))
    for m in _CAP_RUN.finditer(q):
        span = m.group(0).strip(_STRIP_CHARS).strip()
        if not span or _LEADCAP.match(span):
            continue
        toks = [t for t in _WORD.findall(span) if t.lower() not in _STOP]
        if not toks:
            continue
        if m.start() == 0 and len(toks) == 1:
            continue  # the question's own first word
        spans.append(span)
    out: list[str] = []
    for s in spans:
        key = normalize_text(s)
        if not key or any(key in normalize_text(o) for o in out):
            continue
        out = [o for o in out if normalize_text(o) not in key]
        out.append(s)
    return out[:4]


def _toks(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(str(text or ""))]


def _grams(text: str) -> set[str]:
    toks = [t for t in _toks(text)]
    grams: set[str] = set()
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if all(t in _STOP or t.isdigit() for t in gram):
                continue
            grams.add(" ".join(gram))
    return grams


def align_pool(entities: list[str], docs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Group passages under the question entity their title covers; compute shared descriptors."""
    blocks: dict[int, list[str]] = {i: [] for i in range(len(entities))}
    ent_toks = [[t for t in _toks(e) if t not in _STOP] for e in entities]
    leftover: list[str] = []
    for d in docs:
        title = str(d.get("title", ""))
        text = str(d.get("text", "")).strip()
        ttoks = set(_toks(title))
        best, best_score = -1, 0.0
        for i, et in enumerate(ent_toks):
            if not et:
                continue
            score = len([t for t in et if t in ttoks]) / len(et)
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= 0.5:
            blocks[best].append(text)
        else:
            leftover.append(text)
    table: list[str] = []
    for i, ent in enumerate(entities):
        body = blocks[i] or ["(no retrieved article covers this entity)"]
        table.append(f"### {ent}\n" + "\n".join(body))
    if leftover:
        table.append("### other retrieved articles\n" + "\n".join(leftover))

    filled = [i for i in range(len(entities)) if blocks[i]]
    shared: list[str] = []
    if len(filled) >= 2:
        i, j = filled[0], filled[1]
        gi, gj = _grams(" ".join(blocks[i])), _grams(" ".join(blocks[j]))
        common = sorted(gi & gj)
        counts = Counter()
        joined = " ".join(_toks(" ".join(blocks[i] + blocks[j])))
        for g in common:
            counts[g] = joined.count(g)
        unis = sorted([g for g in common if " " not in g], key=lambda g: (-counts[g], g))[:12]
        multis = sorted([g for g in common if " " in g], key=lambda g: (-len(g.split()), -counts[g], g))[:6]
        shared = unis + multis
    return table, shared


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
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
            inputs=["question", "facts", "evidence_table", "shared_descriptors"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )

    @staticmethod
    def _dedup(*doc_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

        # 1-4. retrieval identical to the parent
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

        # 5. lay the pool out as an entity-aligned table plus the two sides' shared vocabulary
        pool = self._dedup(docs_a, docs_b)
        entities = question_entities(question)
        table, shared = align_pool(entities, pool)
        if not table:  # no proper name in the question: fall back to the flat pool
            table = [str(d.get("text", "")).strip() for d in pool]

        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            evidence_table=table,
            shared_descriptors=", ".join(shared) if shared else "none",
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("aligned evidence", entities=entities, shared=shared[:12], next_query=next_query)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": next_query,
            "entities": entities,
            "shared_descriptors": shared,
            "facts": facts,
        }
