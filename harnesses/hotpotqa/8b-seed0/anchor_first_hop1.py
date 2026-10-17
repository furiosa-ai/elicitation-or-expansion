"""Candidate: plan BOTH searches. The first one is aimed at the article the question names outright,
instead of firing the whole question at the index.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7): analyse -> extract+plan -> extend -> answer.

Every system so far spends its first search on the raw question string. A question that needs two
articles names one of them and *describes* the other ("the star of <film>", "the album containing
<song>"), so the raw string mixes the vocabulary of both, and the seven returned passages are split
between them -- or, when the description is wordier than the name, miss the named article altogether.
Rewriting that string mechanically (dropping stop words, keeping capitalised spans) changes nothing,
because the index already weights rare words: the decision that matters is *which* of the two things
to search for, and only a reader of the question can make it.

So the first call reads the question alone and splits it into its two sides: `named` is the thing the
question identifies by name, `described` is the thing it identifies only through a relation. The first
query is then written in the vocabulary of *titles and opening lines* rather than of the question --
the title form of `named`, plus the one or two words naming what KIND of thing the described one is
(news agency, film, actress, album), and nothing else: no question words, no relation, and never the
property being asked about, which matches every article that discusses it. Prototyping showed why the
kind words must stay: a question whose second article is reachable only through "London based news
agency" loses it when the description is dropped altogether.

`named` is checked in code against the question's own word pairs, because the words of a thing the
question names must occur in the question; a planned query that fails that check is a guess at the
described thing, and the raw question is used instead.

That call also commits to `asked_for` -- the kind of thing the answer must be -- before any passage has
been seen, so the commitment cannot be dragged around by whatever the retrieved articles talk about.

The freed call is paid for by merging the first extraction with the planning of the second search:
one pass writes the evidence lines and then names the still-missing article from them.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

ANALYSE_INSTRUCTIONS = """A question needs facts from two different encyclopedia articles. Plan the
first search for it, reading the question only.

First set `named` to the thing the question identifies by name -- its words must appear in the
question itself, copied, never a name you supply from your own knowledge. It is a person, work, film, album, song, band, place,
organisation, species or event, written as the title of its article would write it. Drop
the question's words around it ("the film X" -> X). If the question names two things and asks you to
compare or connect them, write both, separated by a semicolon.

Then set `described` to the thing the question does NOT name but describes through a relation ("the
director of it", "the actress who played it", "the album it appears on"). Write the relation, not a
guess at the name. If the question names both things it needs, write `none`.

Then set `asked_for` to the kind of thing the answer must be, in one or two words: `yes or no`,
`person`, `organisation`, `film`, `song`, `album`, `book`, `place`, `date`, `year`, `number`,
`nationality`, `one of the two named things`, and so on. Decide this from the question alone.

Then set `query` to the search string, as 2 to 8 bare keywords. The index matches words literally
against article titles and their opening paragraphs, so write the words that the *titles and opening
lines* of the two wanted articles would contain, and nothing else:
- start with `named`, written as its own title would write it, plus one disambiguating word (a type
  word like film, band, river, novel, or a year) only when the name alone is ambiguous;
- then add the one or two words that say what KIND of thing `described` is -- the noun an article
  about it would open with (news agency, film, actress, album, river, university) -- because those
  words are how that unnamed article can be reached at all. Keep the kind, drop the relation;
- when `named` holds two things, write both names plus one shared kind word;
- never write a question or a sentence, use no question words (who, what, when, where, which, how),
  never write the relation as words ("the one that inspired", "who played"), and never write the
  property being asked about ("release date", "birth year", "nationality") -- every article that
  discusses that property matches those words."""

EXTRACT_PLAN_INSTRUCTIONS = """Build an evidence file from the passages the first search returned, then
plan the second and last search.

First set `facts` to the sentences that state a fact the question depends on -- the sentence that
identifies an entity the question refers to, and the sentence that states the property being asked
about. One line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

Rules for `facts`:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Include a line even when it only partly matters (for example it names a person, work or place the
  question mentions in passing) -- a later step decides what to use.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines.

The question needs facts from two different articles and `titles_found` lists what the first search
returned, so exactly one article is still missing. Set `missing_target` to its name, as precisely as
you can guess the title an encyclopedia would give it. Two cases:
- the question names two things and the passages cover only one of them -> the missing target is the
  other one, named exactly as the question names it;
- the question describes the missing thing indirectly -> resolve it from the facts you just wrote and
  name the entity itself, not the description.
`missing_target` must be a proper name that could stand as the title of an article -- a person, work,
band, film, place, organisation or species. Never write a role or a relation ("the vocalist of X",
"the director of Y"): if that is all you can say, the article you actually need is the one that would
CONTAIN that fact, so name that entity instead.

Then set `query` to the search string: 2 to 8 bare keywords, the missing target's name plus one or two
disambiguating words only if needed. Never write a question or a sentence, use no question words,
never include words describing facts you already have, and never include a name already in
`titles_found` unless it is part of the missing target's own name."""

EXTEND_INSTRUCTIONS = """Extend an evidence file for a question that needs facts from several articles.

`context` holds the lines gathered so far. Go through the new passages and copy out every sentence
that adds a fact the question depends on. Write one line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

Rules:
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a line that `context` already contains; add only new facts.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines. Output only the lines, nothing else."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
form a grader can match against a short reference answer.

`asked_for` was decided from the question before any article was read: the answer must be a thing of
that kind. If the evidence offers a thing of some other kind, that thing is not the answer.

First set `supporting_quote` to the one evidence line that contains the answer, copied exactly.

Then set `answer` under these rules:
- when `asked_for` is yes or no: write exactly `yes` or `no`. Nothing else -- no explanation, no
  restatement.
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
- Otherwise answer the kind of thing `asked_for` names. A question about who performed, directed,
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
    """Force a planned query into a short keyword string even if the module drifts into prose."""
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`\"'").strip()
    text = re.sub(r"^(query|search|search for|keywords)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("?.").strip()
    text = text.replace(";", " ")
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


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[\w'\u2019\-]+", text or "")]


def _query_is_usable(query: str, question: str) -> bool:
    """Keep a planned first query only if the name in it was COPIED from the question, not invented.

    The thing a question identifies by name occurs in the question, so a planned query must contain at
    least one adjacent pair of words that is also adjacent in the question (or one long question word
    when the name is a single word). A query that fails this test is a guess at the *described* thing,
    which is precisely what the first search must not chase: the question itself is used instead.
    """
    words = _tokens(query)
    if not words or len(words) > 9:
        return False
    qwords = _tokens(question)
    qset = set(qwords)
    qpairs = {(qwords[i], qwords[i + 1]) for i in range(len(qwords) - 1)}
    for i in range(len(words) - 1):
        if (words[i], words[i + 1]) in qpairs:
            return True
    long_grounded = [w for w in words if len(w) >= 4 and w in qset]
    return len(words) <= 2 and len(long_grounded) >= 1


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.analyse = Module(
            inputs=["question"],
            outputs=["named", "described", "asked_for", "query"],
            instructions=ANALYSE_INSTRUCTIONS,
            descs={
                "named": "the thing the question identifies by name, in article-title form",
                "described": "the thing the question only describes through a relation, or `none`",
                "asked_for": "the kind of thing the answer must be, in one or two words",
                "query": "2-8 bare keywords for the first search",
            },
        )
        self.extract_plan = Module(
            inputs=["question", "titles_found", "passages"],
            outputs=["facts", "missing_target", "query"],
            instructions=EXTRACT_PLAN_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search returned",
                "facts": "one `Title :: sentence` line per relevant fact, copied verbatim",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.extend = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts"],
            instructions=EXTEND_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per new relevant fact, copied verbatim"},
        )
        self.generate_answer = Module(
            inputs=["question", "asked_for", "evidence"],
            outputs=["supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "asked_for": "the kind of thing the answer must be, fixed before any article was read",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # PLAN HOP 1 -- search the article the question names, not the question
        head = self.analyse(ctx.llm, tag="analyse", question=question)
        named = _one_line(head.get("named"), 160)
        described = _one_line(head.get("described"), 160)
        asked_for = _one_line(head.get("asked_for"), 60) or "short answer"
        query_1 = _keyword_query(head.get("query", ""), fallback=(named or question))
        if not _query_is_usable(query_1, question):
            query_1 = _keyword_query(named, fallback=question) if named else question
            if not _query_is_usable(query_1, question):
                query_1 = question
        ctx.note("hop1_plan", named=named, described=described, asked_for=asked_for, query=query_1)

        hop1 = ctx.tools.retrieve_docs(query_1, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]

        # EXTRACT the evidence and, in the same pass, name the coverage gap for the last search
        step = self.extract_plan(ctx.llm, tag="extract_plan", question=question,
                                 titles_found=titles_1, passages=hop1_docs)
        facts_1 = _ledger_lines(step.get("facts", ""))
        ledger_1 = "\n".join(facts_1)
        target = _one_line(step.get("missing_target"), 120)
        query_2 = _keyword_query(step.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)

        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        facts_2 = _ledger_lines(
            self.extend(ctx.llm, tag="extend", question=question, context=ledger_1,
                        passages=hop2_docs)["facts"]
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

        out = self.generate_answer(ctx.llm, tag="answer", question=question, asked_for=asked_for,
                                   evidence=evidence)
        return {
            "answer": _tidy(out.get("answer", "")),
            "named": named,
            "described": described,
            "asked_for": asked_for,
            "query_1": query_1,
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
