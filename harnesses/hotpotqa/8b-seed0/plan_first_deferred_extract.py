"""Candidate: compress the passages only *after* the system knows what it is looking for.

Interaction depth is unchanged from the parent (`gap_directed_hop2`): 4 LLM calls, 2 retrievals, k=7.
Both queries are formed exactly as the parent forms them (hop 1 = the raw question, hop 2 = the
title-style gap query). The single mechanism under test is the ORDER of the four calls.

Parent order:   extract(pool1) -> plan(compressed evidence) -> extract(pool2) -> answer
This candidate: plan(raw pool1) -> extract(pool1) -> extract(pool2) -> answer

Two things follow from moving the plan to the front, and they are the hypothesis:

1. The planner reads the seven abstracts as retrieved instead of the handful of lines a previous call
   chose to keep. The name of the still-missing article is very often already printed somewhere in
   those abstracts -- inside an aside, a parenthesis, a list of collaborators -- exactly the material
   an extractor drops, because extraction is asked for sentences that "state a fact the question
   depends on" and a passing mention of a person does not look like one.

2. Both extraction calls then run *knowing what the second search was for*. In the parent, the first
   extraction is the very first thing that happens: it compresses seven abstracts having no idea which
   of the question's two hops those abstracts serve, so it keeps whatever reads relevant to the
   question as a whole and throws away the sentence that will turn out to matter once the bridge is
   resolved. Here both extractions are given the resolved `target` and an explicit `wanted` slot, and
   pool 1 is re-read under that knowledge rather than compressed blind.

No call is deleted and no call is added: the same two extractions still see seven passages each. Only
the information available to each call changes.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

PLAN_INSTRUCTIONS = """One search has already been run and returned the passages below. Plan the second
and last search.

The question needs facts from two different articles. The passages usually cover one of them, so
exactly one article is still missing.

First set `covered` to the title of the article among the passages that the question is already served
by -- the one whose facts the question can use. One title, copied from the passages.

Then set `target` to the name of the article still missing: the person, work, place, organisation,
species or event whose facts you still lack, named as precisely as you can guess the title an
encyclopedia would give it. Two cases:
- the question names two things and the passages cover only one of them -> the target is the other
  one, named exactly as the question names it;
- the question describes the missing thing indirectly ("the director of X", "the album that contains
  Y") -> resolve it from the passages and name the entity itself, not the description. Read the
  passages closely for this: the name you need is frequently printed in one of them in passing, in an
  aside or a parenthesis or a list, rather than in a sentence about the question's subject.
`target` must be a proper name that could stand as the title of an article -- a person, work, band,
film, place, organisation or species. Never write a role or a relation ("the vocalist of X", "the
director of Y", "birthplace of Z"): if that is all you can say, then the article you actually need is
the one that would CONTAIN that fact, so name that entity instead (the band, the film, the album, the
person the question names).

Then set `wanted` to the one fact the missing article has to supply, in a few words (for example
"her year of birth", "the country it is in", "how many members it had").

Then set `query` to the search string. The index matches words literally against article titles and
their opening paragraph, so:
- write 2 to 8 words: the target's name, plus one or two words that disambiguate it (a type word like
  film, band, river, novel, or a year, only if the name alone is ambiguous);
- never write a question or a sentence, and use no question words (who, what, when, where, which, how);
- never include words describing facts the passages already give you, and never include a title that
  is already among the passages unless it is part of the target's own name;
- drop filler words entirely.
Output the query as bare keywords."""

EXTRACT1_INSTRUCTIONS = """Build an evidence file from passages that have already been searched, now
that it is known what the question's two halves are.

`target` names the other article the question needs and `wanted` the fact that other article has to
supply. So these passages are here for the REST of the question: the part that identifies the
question's subject and links it to `target`.

Go through the passages and copy out, one per line, in the form

Title :: the sentence copied exactly as it appears in the passage

- the sentence that identifies each entity the question refers to;
- the sentence that states any property the question asks about;
- the sentence that connects the subject to `target` -- who did what with whom, which work belongs to
  which person, which place is in which country;
- and any sentence that happens to state `wanted` outright, since these passages may already contain
  it even though another search is being run for it.

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Include a line even when it only partly matters -- a later step decides what to use.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines. Output only the lines, nothing else."""

EXTRACT2_INSTRUCTIONS = """Extend an evidence file with a second batch of passages, searched for a known
purpose.

`target` names the article this search was aimed at and `wanted` the fact it was supposed to supply.
`context` holds the evidence gathered from the earlier passages.

Go through the new passages and copy out, one per line, in the form

Title :: the sentence copied exactly as it appears in the passage

- above all, the sentence that states `wanted` -- that is what this search was for;
- the sentence that identifies `target` itself, so a later step can tell it is the right entity;
- any other sentence that states a fact the question depends on and `context` does not already have.

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a line that `context` already contains; add only new facts.
- If no passage is about `target`, do not force it: copy the sentences that are about the question
  instead, and add nothing invented.
- At most 8 lines. Output only the lines, nothing else."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
form a grader can match against a short reference answer.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `supporting_quote` to the one evidence line that contains the answer, copied exactly.

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

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}


def _ledger_lines(text: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*0123456789.) ").strip()
        if not line or _LEDGER_JUNK.match(line):
            continue
        if line.startswith("[[") or line.lower().startswith("rules:"):
            continue
        out.append(line[:600])
        if len(out) >= limit:
            break
    return out


def _keyword_query(raw: str, fallback: str) -> str:
    """Force the planned query into a short keyword string even if the module drifts into prose."""
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`\"'").strip()
    text = re.sub(r"^(query|search|search for|keywords)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("?.").strip()
    words = text.split()
    if len(words) > 10:  # drifted into a sentence: keep the content words only
        content = [w for w in words if w.strip(",.;:").lower() not in _QUESTION_WORDS]
        words = content[:8] if content else words[:8]
        text = " ".join(words)
    return text if text else fallback


def _one_line(raw: str, limit: int = 120) -> str:
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip().strip("*`\"'").strip()[:limit]
    return ""


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


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.plan = Module(
            inputs=["question", "titles_found", "passages"],
            outputs=["covered", "target", "wanted", "query"],
            instructions=PLAN_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search returned",
                "passages": "the passages the first search returned, in full",
                "covered": "title of the article the question is already served by",
                "target": "name of the one article still missing",
                "wanted": "the fact that missing article has to supply",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.extract1 = Module(
            inputs=["question", "target", "wanted", "passages"],
            outputs=["facts"],
            instructions=EXTRACT1_INSTRUCTIONS,
            descs={
                "target": "the other article the question needs",
                "wanted": "the fact that other article has to supply",
                "facts": "one `Title :: sentence` line per relevant fact, copied verbatim",
            },
        )
        self.extract2 = Module(
            inputs=["question", "target", "wanted", "context", "passages"],
            outputs=["facts"],
            instructions=EXTRACT2_INSTRUCTIONS,
            descs={
                "target": "the article this search was aimed at",
                "wanted": "the fact it was supposed to supply",
                "context": "evidence already gathered from the earlier passages",
                "facts": "one `Title :: sentence` line per new relevant fact, copied verbatim",
            },
        )
        self.generate_answer = Module(
            inputs=["question", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # HOP 1 -- the raw question, exactly as the parent issues it
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]

        # CALL 1 -- plan the gap from the RAW passages, before anything is compressed
        plan = self.plan(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1, passages=hop1_docs
        )
        target = _one_line(plan.get("target", ""))
        wanted = _one_line(plan.get("wanted", ""), limit=160)
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", covered=_one_line(plan.get("covered", "")), target=target,
                 wanted=wanted, query=query_2)

        # HOP 2 -- the gap query
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)

        # CALLS 2 and 3 -- compress both pools, both knowing the plan
        facts_1 = _ledger_lines(
            self.extract1(
                ctx.llm, tag="extract1", question=question,
                target=target or "(not identified)", wanted=wanted or "(not stated)", passages=hop1_docs,
            )["facts"]
        )
        facts_2 = _ledger_lines(
            self.extract2(
                ctx.llm, tag="extract2", question=question,
                target=target or "(not identified)", wanted=wanted or "(not stated)",
                context="\n".join(facts_1), passages=hop2_docs,
            )["facts"]
        )

        # CALL 4 -- ANSWER (the parent's contract, unchanged)
        evidence: list[str] = []
        seen: set[str] = set()
        for line in facts_1 + facts_2:
            key = re.sub(r"\W+", " ", line.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                evidence.append(line)
        if not evidence:
            evidence = list(hop1_docs[:3]) + list(hop2_docs[:3])

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        return {
            "answer": _tidy(out.get("answer", "")),
            "answer_type": out.get("answer_type", ""),
            "target": target,
            "wanted": wanted,
            "query_2": query_2,
            "evidence": evidence,
        }
