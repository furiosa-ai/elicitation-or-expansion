"""Candidate: spend the call budget on CONSENSUS instead of on evidence compression.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval is the frontier's:
raw question first, then a title-style keyword query aimed at the named coverage gap.

Every system evaluated so far spends 2 of its 4 calls compressing passages into verbatim
`Title :: sentence` lines and 1 call producing a single answer. That single answer is one sample
from a stochastic decoder, and its errors are not all knowledge errors: on the frontier's traces a
third of the wrong items already had both source articles retrieved. This candidate reallocates the
two compression calls to two extra answering calls and does the evidence selection in code (which
cannot paraphrase, drop a surface form or invent a line). The three answering calls read the SAME
retrieved passages through three different presentations -- question-focused sentence lines, whole
abstracts, and the lines re-ordered with the roster of every article searched -- so their errors
decorrelate; code then takes the majority of the normalized answers, breaking a three-way split in
favour of the answer that occurs verbatim in the passages.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

PLAN_INSTRUCTIONS = """One search has already been run. Plan the second and last one.

`titles_found` lists the articles that search returned and `passages` holds their opening
paragraphs. The question needs facts from two different articles, so exactly one article is still
missing.

First set `missing_target` to the name of that missing article, as precisely as you can guess the
title an encyclopedia would give it: the person, work, place, organisation, species or event whose
facts you still lack. Two cases:
- the question names two things and the passages cover only one of them -> the missing target is the
  other one, named exactly as the question names it;
- the question describes the missing thing indirectly ("the director of X", "the album that contains
  Y") -> resolve it first from the passages and name the entity itself, not the description.
`missing_target` must be a proper name that could stand as the title of an article -- a person, work,
band, film, place, organisation or species. Never write a role or a relation ("the vocalist of X",
"the director of Y", "birthplace of Z"): if that is all you can say, then the article you actually
need is the one that would CONTAIN that fact, so name that entity instead (the band, the film, the
album, the person the question names).

Then set `query` to the search string. The index matches words literally against article titles and
their opening paragraph, so:
- write 2 to 8 words: the missing target's name, plus one or two words that disambiguate it (a type
  word like film, band, river, novel, or a year, only if the name alone is ambiguous);
- never write a question or a sentence, and use no question words (who, what, when, where, which,
  how);
- never include words describing facts you already have, and never include a name that is already in
  `titles_found` unless it is part of the missing target's own name;
- drop filler words entirely.
Output the query as bare keywords."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence, then state the answer in the exact
form a grader can match against a short reference answer.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `supporting_quote` to the one sentence of the evidence that contains the answer, copied
exactly.

Then set `answer` under these rules:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: write the shortest span that answers the question, copied from the supporting quote in
  the exact surface form used there. A name, a title, a date, a number, or a short noun phrase.
- Never write a sentence, never restate the question, never add "the answer is", never explain why.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no unit the quote does not attach to the number, no gloss in
  brackets, and keep any unit or word the quote does attach to it.
- When the question offers a choice between two named alternatives ("which was released first, A or
  B?", "who is older, A or B?", "which one has more members, A or B?"), the answer is one of those two
  names, written as the question writes it -- never the date, number or property that settled the
  comparison, and never both names.
- Otherwise answer the kind of thing the question asks for. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the evidence only names the group
  or company responsible; a question about where wants the place; about when, the date; about how many,
  the bare number. Never hand back an entity of the wrong kind just because the evidence names it --
  the group is not the singer, the album is not the song, the studio is not the director.
- The evidence is often incomplete. Commit to the single most likely answer anyway. If the evidence
  identifies the subject but not the asked-for property, answer from what you know about that subject,
  in the same short form. Never answer "not specified", "unknown", "cannot be determined", and never
  mention the evidence or its gaps in the `answer` field."""

_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}
_STOP = _QUESTION_WORDS | {"by", "with", "at", "from", "as", "it", "its", "his", "her", "their", "was",
                           "were", "been", "be", "also", "than", "into", "between", "both", "same",
                           "other", "one", "two", "about", "after", "before", "during", "over", "not",
                           "but", "they", "he", "she", "them", "who's", "known", "born", "first"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def _tidy(answer: str) -> str:
    text = (answer or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`").strip()
    if len(text) > 1 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    low = text.lower()
    for pref in _PREFIXES:
        if low.startswith(pref):
            text = text[len(pref):].lstrip(" :,").strip()
            low = text.lower()
    verdict = re.match(r"^(yes|no)\b\s*[,.:;—-]?\s*(.*)$", low, re.DOTALL)
    if verdict and verdict.group(2).strip():
        return verdict.group(1)
    return text.strip().strip(".").strip()


def _keyword_query(raw: str, fallback: str) -> str:
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`\"'").strip()
    text = re.sub(r"^(query|search|search for|keywords)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("?.").strip()
    words = text.split()
    if len(words) > 10:
        content = [w for w in words if w.strip(",.;:").lower() not in _QUESTION_WORDS]
        words = content[:8] if content else words[:8]
        text = " ".join(words)
    return text if text else fallback


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _split_sentences(text: str) -> list[str]:
    body = (text or "").strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(body) if p.strip()]
    return parts or ([body] if body else [])


def _focused_lines(docs: list[dict[str, Any]], key_tokens: set[str], per_doc: int = 2,
                   cap: int = 16) -> list[str]:
    """Pick, in code, the sentences of each abstract that share the most content words with the
    question and the planned target. No paraphrase, no fabrication: the sentence is copied whole."""
    scored: list[tuple[float, int, str]] = []
    for rank, doc in enumerate(docs):
        title = (doc.get("title") or "").strip()
        sents = _split_sentences(doc.get("text") or "")
        title_tok = _tokens(title)
        best: list[tuple[float, str]] = []
        for si, sent in enumerate(sents):
            if len(sent) < 12:
                continue
            tok = _tokens(sent)
            overlap = len(tok & key_tokens)
            score = overlap + 0.5 * len(title_tok & key_tokens) + (0.75 if si == 0 else 0.0)
            best.append((score, sent))
        best.sort(key=lambda t: -t[0])
        for score, sent in best[:per_doc]:
            if score <= 0.75:
                continue
            scored.append((score - 0.05 * rank, rank, f"{title} :: {sent}"[:600]))
    scored.sort(key=lambda t: (t[1], -t[0]))
    out: list[str] = []
    seen: set[str] = set()
    for _, _, line in scored:
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(line)
        if len(out) >= cap:
            break
    return out


def _dedup_docs(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for doc in group:
            title = (doc.get("title") or "").strip().lower()
            if title and title in seen:
                continue
            seen.add(title)
            out.append(doc)
    return out


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.plan_hop2 = Module(
            inputs=["question", "titles_found", "passages"],
            outputs=["missing_target", "query"],
            instructions=PLAN_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "passages": "opening paragraphs of those articles",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.answer = Module(
            inputs=["question", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences taken from the source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence sentence containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    # -- evidence presentations -------------------------------------------------------------
    def _views(self, question: str, target: str, hop1: list[dict[str, Any]],
               hop2: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
        key = _tokens(question) | _tokens(target)
        docs1, docs2 = _dedup_docs(hop1), _dedup_docs(hop2, hop1)[: len(hop2)]
        lines1 = _focused_lines(docs1, key, per_doc=2, cap=9)
        lines2 = _focused_lines(docs2, key, per_doc=2, cap=9)
        if not lines1:
            lines1 = [f"{d.get('title')} :: {_split_sentences(d.get('text') or '')[0]}"[:600] for d in docs1[:3]]
        if not lines2:
            lines2 = [f"{d.get('title')} :: {_split_sentences(d.get('text') or '')[0]}"[:600] for d in docs2[:3]]

        focused = lines1 + [ln for ln in lines2 if ln not in lines1]
        whole = [f"{d.get('title')} | {(d.get('text') or '').strip()}"[:1400]
                 for d in (docs1[:4] + docs2[:4])]
        roster = "; ".join(dict.fromkeys([(d.get("title") or "").strip() for d in docs1 + docs2]))
        reordered = [ln for ln in lines2 if ln not in lines1] + lines1
        reordered = reordered + [f"Articles searched :: {roster}"[:600]]
        return [("focused", focused), ("whole", whole), ("reordered", reordered)]

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]

        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1,
            passages=[d["text"] for d in hop1],
        )
        target = (plan.get("missing_target") or "").strip().splitlines()
        target = target[0][:120] if target else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)

        # three answering calls over three presentations of the same passages
        views = self._views(question, target, hop1, hop2)
        haystack = normalize_text(" ".join(
            [(d.get("text") or "") for d in hop1 + hop2] + [question]))
        samples: list[dict[str, str]] = []
        for name, evidence in views:
            out = self.answer(ctx.llm, tag=f"answer_{name}", question=question, evidence=evidence)
            ans = _tidy(out.get("answer", ""))
            samples.append({"view": name, "answer": ans, "norm": normalize_text(ans),
                            "answer_type": out.get("answer_type", "")})

        votes = Counter(s["norm"] for s in samples if s["norm"])
        answer = samples[0]["answer"]
        agreement = 0
        if votes:
            top_norm, agreement = votes.most_common(1)[0]
            if agreement >= 2:
                answer = next(s["answer"] for s in samples if s["norm"] == top_norm)
            else:
                # three-way split: prefer the answer the passages actually contain verbatim
                grounded = [s for s in samples if s["norm"] and s["norm"] in haystack]
                answer = (grounded[0]["answer"] if grounded else samples[0]["answer"])
        ctx.note("vote", agreement=agreement,
                 answers=[s["answer"] for s in samples], chosen=answer)
        return {
            "answer": answer,
            "agreement": agreement,
            "samples": [s["answer"] for s in samples],
            "missing_target": target,
            "query_2": query_2,
        }
