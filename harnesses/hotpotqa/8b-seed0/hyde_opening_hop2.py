"""Candidate: the second search is issued as a *generated article opening*, not as keywords.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7) and the evidence/answer layer is the
frontier's, untouched. The single mechanism under test is what string the second retrieval receives.

Trace evidence for the change: on the frontier's own search-set runs the second hop is a 2-8 word
keyword string built around a guessed title, and it fails in a characteristic way -- the guessed name is
a common one, so a bag-of-words index returns seven unrelated people with that name, or the guess is
close but not the exact title an encyclopedia uses. Adding the asked-for property to the keywords
("<name> birth year") makes it worse: those words match every biography in the index.

Query-by-generated-document instead: the planner writes the sentence the missing article would OPEN
with -- full name, what kind of thing it is, field/nationality/period, and the works, people or places
it is connected to. Against a term-frequency index that whole sentence scores the intended article far
above its namesakes, because the disambiguating vocabulary (occupation, era, associated titles) is
exactly the vocabulary of the real opening paragraph, while a bare name is shared by all of them. The
planner is told to describe the article, never to ask for the fact it wants.
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
`missing_target` must be a proper name that could stand as the title of an article. Never write a role
or a relation ("the vocalist of X", "birthplace of Z"): if that is all you can say, then the article you
need is the one that would CONTAIN that fact, so name that entity instead.

Then set `article_opening` to the sentence that missing article would BEGIN with -- write it as an
encyclopedia writes an opening line, as if you had the article in front of you:
- start with the full form of the name, then what kind of thing it is (person, film, band, river,
  novel, company, species) and the field, nationality, period or genre it belongs to;
- go on to name the things it is connected to: the works it made or appeared in, the people it worked
  with, the place it is in, the group it belongs to -- especially the entities the question and the
  evidence already mention, since a real article names them too;
- include the dates you are reasonably confident about;
- 20 to 45 words, one flat sentence of plain prose, no markup and no list;
- write only details you actually believe: if you do not know the date, the genre or the collaborators,
  leave them out entirely -- never write a slot like [date], [genre] or [unknown] and never write a
  placeholder word in place of a fact;
- it is a description of the article, NOT a request: never phrase it as a question, never write the
  fact you are trying to look up as if you knew it, and never write words like "unknown", "missing",
  "the article about" or "I need".
Write your best guess even when you are unsure of the details: several plausible descriptive words cost
nothing, while a bare name is shared by dozens of unrelated articles."""

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
_META_WORDS = {"unknown", "missing", "unspecified", "presumably", "likely", "probably", "article",
               "wikipedia", "encyclopedia", "hypothetical", "placeholder", "todo"}


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


def _pseudo_document(raw: str, name: str, fallback: str) -> str:
    """Flatten the generated opening line into one search string; keep it descriptive, drop meta-talk."""
    text = " ".join((raw or "").split())
    text = re.sub(r"^(article_opening|opening|sentence)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    # a planner that does not know a detail writes a template slot for it; those words belong to no
    # article and only dilute the match, so they are removed rather than searched for.
    text = re.sub(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}", " ", text)
    text = text.replace("*", " ").replace("_", " ")
    text = " ".join(text.split()).strip().strip("`\"'").strip()
    # a planner that drifts into asking for the fact produces a question: keep only the assertion part
    if "?" in text:
        head = text.split("?")[0]
        text = head if len(head.split()) >= 6 else text.replace("?", " ")
    words = [w for w in text.split() if w.strip(",.;:").lower() not in _META_WORDS]
    if len(words) > 48:
        words = words[:48]
    text = " ".join(words).strip()
    if len(text.split()) < 4:  # planner produced nothing usable
        return (name or fallback).strip() or fallback
    if name and name.split()[0].lower() not in text.lower():
        text = f"{name} {text}"  # the title itself must be in the query
    return text


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
            outputs=["missing_target", "article_opening"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "missing_target": "name of the one article still missing",
                "article_opening": "the opening sentence the missing article would begin with (20-45 words)",
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
        query_2 = _pseudo_document(plan.get("article_opening", ""), target, fallback=question)
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
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
