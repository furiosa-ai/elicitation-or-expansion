"""Candidate: the final call solves the question as a chain of sub-questions instead of one extraction.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the gap-directed second
query and the verbatim evidence file are inherited unchanged from `gap_directed_hop2`; the one
mechanism under test is the reasoning structure of the last call.

The parent asks the last call for a type, a supporting quote and a span. That is an extraction
contract: it works when one evidence line already contains the answer, and it is exactly the shape
of the parent's remaining errors when no single line does. A question like "in what year was the man
who founded X with A and B born" needs the man identified from the article about X before the year
can be sought at all; asked for a quote straight away, the model picks the line about X and answers
from it, or gives up. Errors of that shape in the parent's traces include an entity of the wrong kind
handed back because it was the only name in the evidence, and a comparison settled by returning the
compared value rather than the compared thing.

So the last call is restructured to name the intermediate step explicitly: which question must be
settled first, what the evidence says its answer is and out of which article, the original question
rewritten with that answer substituted in, and only then the short span. Comparison questions take
the same shape with the two compared values as the intermediate step. The answer-form contract that
the parent depends on is kept verbatim for the final field.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

EXTRACT_INSTRUCTIONS = """Build an evidence file for a question that needs facts from several articles.

Go through the passages and copy out every sentence that states a fact the question depends on -- the
sentence that identifies an entity the question refers to, and the sentence that states the property
being asked about. Write one line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Include a line even when it only partly matters (for example it names a person, work or place the
  question mentions in passing) -- a later step decides what to use.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines. Output only the lines, nothing else."""

EXTRACT2_INSTRUCTIONS = """Extend an evidence file for a question that needs facts from several articles.

`context` holds the lines gathered so far. Go through the new passages and copy out every sentence
that adds a fact the question depends on. Write one line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a line that `context` already contains; add only new facts.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines. Output only the lines, nothing else."""

GAP_QUERY_INSTRUCTIONS = """One search has already been run. Plan the second and last one.

`titles_found` lists the articles that search returned and `evidence` the facts taken from them. The
question needs facts from two different articles, so exactly one article is still missing.

First set `missing_target` to the name of that missing article, as precisely as you can guess the title
an encyclopedia would give it: the person, work, place, organisation, species or event whose facts you
still lack. Two cases:
- the question names two things and the evidence covers only one of them -> the missing target is the
  other one, named exactly as the question names it;
- the question describes the missing thing indirectly ("the director of X", "the album that contains
  Y") -> resolve it first from the evidence and name the entity itself, not the description.
`missing_target` must be a proper name that could stand as the title of an article -- a person, work,
band, film, place, organisation or species. Never write a role or a relation ("the vocalist of X", "the
director of Y", "birthplace of Z"): if that is all you can say, then the article you actually need is
the one that would CONTAIN that fact, so name that entity instead (the band, the film, the album, the
person the question names).

Then set `query` to the search string. The index matches words literally against article titles and
their opening paragraph, so:
- write 2 to 8 words: the missing target's name, plus one or two words that disambiguate it (a type
  word like film, band, river, novel, or a year, only if the name alone is ambiguous);
- never write a question or a sentence, and use no question words (who, what, when, where, which, how);
- never include words describing facts you already have, and never include a name that is already in
  `titles_found` unless it is part of the missing target's own name;
- drop filler words entirely.
Output the query as bare keywords."""

ANSWER_INSTRUCTIONS = """Work out the answer to the question in steps, then state it in the exact form a
grader can match against a short reference answer.

`evidence` holds sentences copied verbatim out of the source articles, each prefixed by its title. The
question needs facts from more than one article, so it almost never has a single line that answers it
outright: one line ties the question to an entity, another states the property being asked about.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `sub_question` to the one thing that has to be settled before the question can be answered:
- when the question refers to something indirectly ("the man who founded X", "the album that contains
  Y", "the miniseries he researched"), the sub-question asks which entity that description picks out;
- when the question names two things and compares them, the sub-question asks for the compared
  property of each of the two by name;
- when the evidence plainly answers the question in one line, write `none`.

Then set `sub_answer` to the answer of `sub_question` as the evidence gives it, followed by the title
it came from in brackets -- for example `Jane Doe (X Project)`. For a comparison, give both values with
their titles. If the evidence does not settle the sub-question, answer it from what you know and mark
it `(from knowledge)`. Never write a sub-answer the evidence contradicts.

Then set `resolved_question` to the original question with the description replaced by `sub_answer`, so
that it asks directly for the thing wanted. Keep the question's own wording everywhere else. For a
comparison, write which of the two named things wins.

Then set `answer` under these rules:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: write the shortest span that answers `resolved_question`. Take it from the evidence in
  the exact surface form used there when the evidence has it; otherwise write it in the form an
  encyclopedia would.
- Never write a sentence, never restate the question, never add "the answer is", never explain why,
  and never answer with the sub-question's answer unless the question asked for exactly that.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no unit the evidence does not attach to the number, no gloss in
  brackets, and keep any unit or word the evidence does attach to it.
- When the question offers a choice between two named alternatives ("which was released first, A or
  B?", "who is older, A or B?", "which one has more members, A or B?"), the answer is one of those two
  names, written as the question writes it -- never the date, number or property that settled the
  comparison, and never both names.
- Otherwise answer the kind of thing the question asks for. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the evidence only names the group
  or company responsible; a question about where wants the place; about when, the date; about how many,
  the bare number. Never hand back an entity of the wrong kind just because the evidence names it --
  the group is not the singer, the album is not the song, the studio is not the director.
- The evidence is often incomplete. Commit to the single most likely answer anyway. Never answer
  "not specified", "unknown" or "cannot be determined", and never mention the evidence or its gaps in
  the `answer` field."""

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
    # the chain writes sub-answers as `Name (Source Title)`; never let that shape reach the answer
    stripped = re.sub(r"\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*$", "", text).strip()
    if stripped and stripped != text and len(stripped.split()) >= 1:
        text = stripped
    return text.strip().strip(".").strip()


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.extract1 = Module(
            inputs=["question", "passages"],
            outputs=["facts"],
            instructions=EXTRACT_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per relevant fact, copied verbatim"},
        )
        self.plan_hop2 = Module(
            inputs=["question", "titles_found", "evidence"],
            outputs=["missing_target", "query"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.extract2 = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts"],
            instructions=EXTRACT2_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per new relevant fact, copied verbatim"},
        )
        self.generate_answer = Module(
            inputs=["question", "evidence"],
            outputs=["answer_type", "sub_question", "sub_answer", "resolved_question", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "sub_question": "the question that must be settled first, or `none`",
                "sub_answer": "its answer with the source title in brackets",
                "resolved_question": "the question with the description replaced by `sub_answer`",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # HOP 1
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]
        facts_1 = _ledger_lines(
            self.extract1(ctx.llm, tag="extract1", question=question, passages=hop1_docs)["facts"]
        )
        ledger_1 = "\n".join(facts_1)

        # HOP 2 -- named coverage gap, then a title-style keyword query for it
        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1, evidence=facts_1 or hop1_docs[:3]
        )
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)

        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        facts_2 = _ledger_lines(
            self.extract2(ctx.llm, tag="extract2", question=question, context=ledger_1, passages=hop2_docs)["facts"]
        )

        # ANSWER
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
        answer = _tidy(out.get("answer", ""))
        ctx.note("chain", sub_question=out.get("sub_question", "")[:200], sub_answer=out.get("sub_answer", "")[:200])
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "sub_question": out.get("sub_question", ""),
            "sub_answer": out.get("sub_answer", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
