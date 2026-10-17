"""Candidate: the last call fills a blank in a statement instead of answering a question.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the gap planner and both
verbatim-extraction calls are byte-identical to the parent. The one mechanism under test is the answer
policy: the final call must first rewrite the question as the statement it is asking about, with the
unknown replaced by a blank and every entity it can already resolve substituted in from the evidence,
and then produce only the words that fill that blank.

Why: a short reference answer is not "the shortest span that answers the question" -- it is the piece
of a source sentence that the question's own sentence is missing. The parent decides how much of a
phrase to keep from a prose rule, and the rule cuts in both directions: it drops a word the question's
grammar demands and keeps a word it does not. A blank has a syntactic position instead of a rule:
what fills "... is ___ ." is a category phrase, what fills "... ordered ___ of them ." is a count with
whatever noun the count needs, what fills "... was the ___ one" is an ordinal rather than a name. The
extent and the kind of thing wanted both fall out of the rewrite, which happens before any candidate
answer exists.
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

ANSWER_INSTRUCTIONS = """Turn the question into the statement it is asking about, then fill in the one
piece of that statement the question is missing, using the evidence lines.

The evidence lines are sentences copied word for word out of the source articles, each prefixed by its
article title.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `claim`: the question rewritten as a single statement whose one unknown piece is written as
`___`. Rules for the rewrite:
- keep the question's own wording and word order wherever you can, and keep its whole subject: every
  date, ordinal, role and relation it uses to pin down what it is asking about stays in the claim;
- put `___` exactly where the question's words leave a hole, so that reading the claim aloud says what
  kind of thing belongs there and how much of it: `... ran from ___ .`, `Both A and B are ___ .`,
  `... employed ___ of them .`, `... was the ___ one in the series.`;
- if the question refers to something only by description ("the director of X", "the album that
  contains Y"), replace that description with the name the evidence gives it, so the claim names it;
- exactly one `___`, never two, and never put `___` where the question already tells you the answer;
- for a yes_no question write the statement whose truth is being asked about, with no `___` at all;
- for a question offering a choice between two named alternatives, write `___` in the subject
  position, so the blank takes one of the two names: `___ was released first.`

Then set `supporting_quote` to the one evidence line that lets you fill the blank, copied exactly.

Then set `answer` to the filler and nothing else:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: write the words that go in place of `___`, copied from the supporting quote in the exact
  surface form used there. Substituting them for `___` must leave a complete, grammatical, true
  sentence -- so keep the noun a count counts, the whole place name the quote writes including any
  larger place it attaches, the whole category phrase, and the given or middle names the quote spells
  out. Do not add a word the claim already contains, and do not
  add an apposition, a unit, a bracketed gloss or a country the quote does not attach.
- A filler that is a name is never introduced by a description of it. Write the bare name, even where
  the quote puts a description in front of it (`the <nationality> <kind-of-group> <name>`, `<job title>
  <name>`): the words saying what kind of thing the name is stay out of the answer.
- Never write a sentence, never restate the claim, never add "the answer is", never explain why.
- When the question offers a choice between two named alternatives ("which was released first, A or
  B?", "who is older, A or B?", "which one has more members, A or B?"), the filler is one of those two
  names -- never the date, number or property that settled the comparison, and never both names.
- Otherwise the filler must be the kind of thing the claim's blank wants. A claim about who performed,
  directed, wrote or founded something wants the name of the person, even when the evidence only names
  the group or company responsible; a claim about where wants the place; about when, the date; about
  how many, the count. Never hand back an entity of the wrong kind just because the evidence names it
  -- the group is not the singer, the album is not the song, the studio is not the director.
- The evidence is often incomplete. Fill the blank with the single most likely filler anyway. If the
  evidence identifies the subject but not the asked-for property, fill it from what you know about that
  subject, in the same short form. Never answer "not specified", "unknown", "cannot be determined", and
  never mention the evidence or its gaps in the `answer` field."""

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


_EDGE_WORDS = {"from", "in", "on", "at", "of", "to", "by", "for", "during", "between", "and", "as",
               "with", "was", "is", "were", "are", "near", "the", "a", "an", "into", "about", "over",
               "under", "until", "through", "his", "her", "their", "its"}


def _key(word: str) -> str:
    return word.strip(".,;:\"'?!()[]").lower()


def _peel_claim(text: str, pre: str, post: str) -> str:
    """If the model echoed the whole completed statement, recover just the words it put in the blank."""
    tw, pw, ow = text.split(), pre.split(), post.split()
    t, p, o = [_key(w) for w in tw], [_key(w) for w in pw], [_key(w) for w in ow]
    i = 0
    while i < len(p) and i < len(t) and t[i] == p[i]:
        i += 1
    if i < 3 or (p and i < 0.6 * len(p)):  # not an echo of this claim -- leave the answer alone
        return ""
    j = 0
    while j < len(o) and j < len(t) - i and t[len(t) - 1 - j] == o[len(o) - 1 - j]:
        j += 1
    end = len(tw) - j
    return " ".join(tw[i:end]).strip() if end > i else ""


def _trim_edges(text: str, pre: str, post: str) -> str:
    """Drop a function word the filler repeats from the claim's own words either side of the blank."""
    tw = text.split()
    before = [_key(w) for w in pre.split()][-1:]
    after = [_key(w) for w in post.split()][:1]
    while len(tw) > 1 and before and _key(tw[0]) == before[0] and _key(tw[0]) in _EDGE_WORDS:
        tw.pop(0)
    while len(tw) > 1 and after and _key(tw[-1]) == after[0] and _key(tw[-1]) in _EDGE_WORDS:
        tw.pop()
    return " ".join(tw).strip()


def _fill(answer: str, claim: str) -> str:
    text = _tidy(answer)
    statement = (claim or "").strip()
    if "___" in statement and text:
        pre, _, post = statement.partition("___")
        peeled = _peel_claim(text, pre, post)
        if peeled:
            text = _tidy(peeled)
        text = _trim_edges(text, pre, post)
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
            outputs=["answer_type", "claim", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "claim": "the question rewritten as a statement with its unknown piece written `___`",
                "supporting_quote": "the evidence line that fills the blank, copied exactly",
                "answer": "only the words that replace `___` -- never a sentence",
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
        answer = _fill(out.get("answer", ""), out.get("claim", ""))
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "claim": (out.get("claim") or "").strip()[:300],
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
