"""Candidate: the second retrieval is *routed* -- gap search when evidence is incomplete,
probe of the provisional answer's own article when it is already complete.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7): extract -> route -> extend/verify -> answer.

The frontier system always spends its second search on the article it thinks is missing. On the
traces that is wasted work whenever the first search already returned both source articles (a third
of all items): the search adds nothing, and those items still fail -- almost always with the
reference answer sitting inside the evidence but stated in the wrong extent (a short form of a name
whose own article opens with the full one) or with a plausible-but-unverified entity chosen from
several candidates.

So the routing call first decides coverage. When the evidence still lacks a source article, the
branch is the frontier's gap query. When it does not, the call commits to a provisional answer and
the search is aimed at THAT entity's own article: the returned opening sentence both names the entity
the way its own article names it and states the properties the question compares, so the last call
can confirm, flip, or re-extent the provisional answer against a source that was never seen before.
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

ROUTE_INSTRUCTIONS = """One search has already been run and one is left. Decide what to spend it on.

`titles_found` lists the articles that search returned and `evidence` the facts taken from them.

First set `coverage` to `incomplete` or `complete`.
- `incomplete`: the evidence does not yet contain a sentence stating some fact the question needs --
  a thing the question names or describes has no article among `titles_found`, or the asked-for
  property is stated nowhere.
- `complete`: every fact the question needs is already stated in a line of `evidence`, so you could
  answer now. Be honest: guessing the property from your own knowledge is NOT complete.

If `coverage` is `incomplete`:
- set `provisional_answer` to an empty string;
- set `target` to the name of the one missing article, as precisely as you can guess the title an
  encyclopedia would give it. If the question names two things and only one is covered, the target is
  the other one. If the question describes the missing thing indirectly ("the director of X", "the
  album containing Y"), resolve it from the evidence and name the entity itself. `target` must be a
  proper name that could stand as an article title -- a person, work, band, film, place, organisation
  or species -- never a role or a relation.

If `coverage` is `complete`:
- set `provisional_answer` to the short answer the evidence supports right now (`yes` or `no` for a
  question asking whether something holds);
- set `target` to the name of the entity that answer refers to, so its own article can be read: for a
  person, work, place or organisation, its name; for a comparison whose answer is one of two named
  things, that thing's name. If the answer is a date, a number or a word rather than a named thing,
  set `target` to the name of the entity the question asks about instead.

Then set `query` to the search string for `target`: 2 to 8 bare keywords -- the name, plus one or two
words that disambiguate it (a type word like film, band, river, novel, or a year) only if the name
alone is ambiguous. Never write a question or a sentence, use no question words, never include words
describing facts you already have, and never append the property being asked about."""

EXTEND_INSTRUCTIONS = """Extend an evidence file for a question that needs facts from several articles.

`context` holds the lines gathered so far. Go through the new passages and copy out every sentence
that adds a fact the question depends on. Write one line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a line that `context` already contains; add only new facts.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines. Output only the lines, nothing else.
Leave `naming` empty."""

VERIFY_INSTRUCTIONS = """A provisional answer has been formed from the evidence in `context`. The new
passages were fetched to check it against the article about `provisional_answer` itself.

First set `facts` to the sentences from the new passages that bear on the provisional answer, one per
line, in the form:

Title :: the sentence copied exactly as it appears in the passage

Copy verbatim, never paraphrase or merge. Include the sentence stating the property the question asks
about, and any sentence that contradicts the provisional answer. If no new passage is about the
provisional answer, write only the lines that are relevant anyway, or nothing at all. At most 6 lines.

Then set `naming`. If one of the new passages IS the article about the provisional answer -- its title
is that entity -- copy that article's first sentence up to the end of the naming part, exactly as
written, in the form:

Title :: first sentence copied exactly

This is the form that article uses for the entity's own name, which may be longer than the name the
question uses (middle names, a formal or full name, a title). If no new passage is that article,
leave `naming` empty. Never invent a name."""

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
  B?", "who is older, A or B?", "which one has more members, A or B?"), the answer is one of those
  two things -- never the date, number or property that settled the comparison, and never both.
- `naming`, when it is not empty, is the opening sentence of the article about the entity that was
  provisionally the answer. If you are answering with that entity AND the question asks for a fact
  about the entity itself (which of two people is older, who won an award, who was born where), name
  it exactly as that opening sentence names it -- keep the middle names, the full given names, the
  formal or titled form. If instead the fact was stated by some other article's sentence about it
  (who directed a film, who wrote a song, which studio released it), keep the shorter form that
  sentence uses.
- Otherwise answer the kind of thing the question asks for. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the evidence only names the group
  or company responsible; a question about where wants the place; about when, the date; about how many,
  the bare number. Never hand back an entity of the wrong kind just because the evidence names it --
  the group is not the singer, the album is not the song, the studio is not the director.
- The new evidence may contradict the provisional answer; when it does, follow the evidence.
- The evidence is often incomplete. Commit to the single most likely answer anyway. If the evidence
  identifies the subject but not the asked-for property, answer from what you know about that subject,
  in the same short form. Never answer "not specified", "unknown", "cannot be determined", and never
  mention the evidence or its gaps in the `answer` field."""

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+|empty)$", re.IGNORECASE)
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


def _one_line(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:limit]
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
        self.extract1 = Module(
            inputs=["question", "passages"],
            outputs=["facts"],
            instructions=EXTRACT_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per relevant fact, copied verbatim"},
        )
        self.route = Module(
            inputs=["question", "titles_found", "evidence"],
            outputs=["coverage", "provisional_answer", "target", "query"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "coverage": "`incomplete` or `complete`",
                "provisional_answer": "the short answer the evidence supports, or empty",
                "target": "name of the article the last search should return",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.extend = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts", "naming"],
            instructions=EXTEND_INSTRUCTIONS,
            descs={
                "facts": "one `Title :: sentence` line per new relevant fact, copied verbatim",
                "naming": "leave empty",
            },
        )
        self.verify = Module(
            inputs=["question", "provisional_answer", "context", "passages"],
            outputs=["facts", "naming"],
            instructions=VERIFY_INSTRUCTIONS,
            descs={
                "provisional_answer": "the answer the earlier evidence supported",
                "facts": "one `Title :: sentence` line per new sentence bearing on that answer, verbatim",
                "naming": "`Title :: first sentence` of the article about the provisional answer, or empty",
            },
        )
        self.generate_answer = Module(
            inputs=["question", "evidence", "naming"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "naming": "opening sentence of the article about the provisionally chosen entity, or empty",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
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

        # ROUTE the one remaining search: fill a coverage gap, or probe the answer's own article
        plan = self.route(
            ctx.llm, tag="route", question=question, titles_found=titles_1,
            evidence=facts_1 or hop1_docs[:3],
        )
        coverage = _one_line(plan.get("coverage"), 40).lower()
        complete = coverage.startswith("complete")
        provisional = _tidy(_one_line(plan.get("provisional_answer"), 200))
        if complete and not provisional:
            complete = False
        target = _one_line(plan.get("target"), 120)
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("route", coverage=("complete" if complete else "incomplete"),
                 provisional_answer=provisional, target=target, query=query_2)

        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)

        # EXTEND (gap branch) or VERIFY the provisional answer against its own article
        if complete:
            out2 = self.verify(ctx.llm, tag="verify", question=question, provisional_answer=provisional,
                               context=ledger_1, passages=hop2_docs)
        else:
            out2 = self.extend(ctx.llm, tag="extend", question=question, context=ledger_1,
                               passages=hop2_docs)
        facts_2 = _ledger_lines(out2.get("facts", ""), limit=8)
        naming = ""
        if complete:
            naming_lines = _ledger_lines(out2.get("naming", ""), limit=1)
            naming = naming_lines[0] if naming_lines else ""
            if naming and len(naming.split()) < 3:
                naming = ""
        ctx.note("verify_naming", naming=naming)

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

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence,
                                   naming=naming)
        answer = _tidy(out.get("answer", ""))
        if not answer:
            answer = provisional
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "coverage": "complete" if complete else "incomplete",
            "provisional_answer": provisional,
            "target": target,
            "query_2": query_2,
            "naming": naming,
            "evidence": evidence,
        }
