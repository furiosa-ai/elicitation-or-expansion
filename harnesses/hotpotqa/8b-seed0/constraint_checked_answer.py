"""Candidate: the answer is committed only after every condition the question states is checked off.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the gap-directed second hop
and the evidence file are byte-identical to the parent system; the one mechanism under test is the
reasoning structure of the final call.

The parent's answer call is flat: read the evidence, name the kind of thing asked for, quote a line, emit
a span. Its surviving errors on items where the evidence was complete are not knowledge failures but
under-constrained ones. A question typically pins its subject down with several conditions at once (a
year, an ordinal, a role, a relation) and the evidence names several entities that satisfy some of them;
the flat call latches onto the first entity that fits any condition. The same flatness decides span
extent by a blanket "shortest span" preference, which silently drops the parts of a name or phrase the
source sentence attaches to it.

So the final call is restructured as an explicit satisfaction check. It must first break the question
into `conditions` -- one line per atomic requirement, including the kind of thing the answer must be --
then write `check`, one line per condition naming what in the evidence satisfies it and, crucially,
recording a condition as unmet when nothing does. Only then may it quote and answer, and the answer must
carry every word the source phrase attaches to whatever satisfied the conditions. A condition left unmet
is a signal to move to a different candidate, not to hedge: the system still always commits.
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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines. The question pins its subject down
with several conditions at once, and the evidence usually names other things that meet only some of them,
so check the conditions before committing rather than after.

First set `conditions` to the question broken into atomic requirements, one per line, numbered. Include
every constraint the question states -- a date or year, an ordinal or count, a role or occupation, a
nationality or place, a relation to another named thing, a work someone appeared in -- and make the LAST
line the kind of thing the answer itself must be (a person, an organisation, a work, a place, a date, a
number, a yes/no verdict, or a short descriptive phrase).

Then set `check` to one line per condition, in the same order, in the form `n: <what meets it>` or
`n: unmet`. Name the specific thing from the evidence that meets each condition. If the evidence offers
two things and only one meets all the conditions, this is where you see it: an entity that fails any
condition is the wrong entity, so move to the one that fails none. Write `unmet` honestly when the
evidence settles nothing -- an unmet condition is not a reason to refuse, only a note that the last line
of the answer rests on what you know rather than on the evidence.

Then set `supporting_quote` to the one evidence line that contains the answer, copied exactly.

Then set `answer` under these rules:
- yes/no question (asks whether something holds -- are/is/was/were/do/does/did/can/has/have/both ... ?):
  write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: copy the answer out of the supporting quote in the quote's own surface form, and copy the
  WHOLE phrase the quote uses for it -- every given name, initial, particle, honorific, qualifying year,
  administrative region or counted noun the quote attaches to it, and nothing beyond that phrase. Do not
  shorten a name to the form you know it by and do not expand one the quote does not expand. A name, a
  title, a date, a number, or a short noun phrase.
- Never write a sentence, never restate the question, never add "the answer is", never explain why.
- Never append a qualifier the quote does not attach, never add a gloss in brackets, never attach a unit
  the quote leaves off, and never give two alternatives.
- When the question offers a choice between two named alternatives ("which was released first, A or B?",
  "who is older, A or B?"), the answer is one of those two names -- never the date, number or property
  that settled the comparison, and never both.
- The last condition governs the kind of thing you return. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the evidence only names the group or
  company responsible; where wants the place; when, the date; how many, the bare number; what position or
  installment, the ordinal word. Never hand back an entity of the wrong kind just because the evidence
  names it -- the group is not the singer, the album is not the song, the studio is not the director.
- Commit to a single answer even when a condition is unmet. If the evidence identifies the subject but
  not the asked-for property, answer from what you know about that subject, in the same short form. Never
  answer "not specified", "unknown", "cannot be determined", and never mention the evidence or its gaps
  in the `answer` field."""

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
            outputs=["conditions", "check", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "conditions": "the question's atomic requirements, one numbered line each, kind of answer last",
                "check": "one `n: <what meets it>` or `n: unmet` line per condition, same order",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
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
        return {
            "answer": answer,
            "conditions": out.get("conditions", ""),
            "check": out.get("check", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
