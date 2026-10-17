"""Candidate: nothing rewrites the source text. The model only *chooses* articles; the answerer reads them.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). The gap-directed hop-2 planner is
inherited from `gap_directed_hop2`; the mechanism under test is the removal of the sentence-extraction
bottleneck.

Measured defect in the parent: its two extraction calls compress 14 retrieved passages down to a mean of
**3.1 sentences**, and every sentence is re-typed by an 8B model that is told to copy verbatim. Offline
re-retrieval of its own traced queries shows the reference answer is present in the retrieved passages on
253/300 items while it scores 206/300, and on 12 errors the answer is in a passage but was never copied
out. Asking a small model to be a lossless compressor is the wrong division of labour.

So the evidence channel here is the source text itself. The two calls that used to copy sentences instead
emit *passage numbers*: which articles bear on the question. Code splices the chosen abstracts, untouched
and under their real titles, into the answering prompt. Paraphrase survives only where it cannot hurt --
the `known` field, used to plan the second search and never to answer from.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

SELECT1_INSTRUCTIONS = """A question needs facts from two different articles. Seven numbered passages
came back from the first search. Decide which of them are worth reading in full.

Set `relevant` to the passage numbers that bear on the question, best first, separated by commas -- at
most four, and never a number that is not in the list. Include a passage if it is about an entity the
question names or describes, or if it states the property the question asks about. Exclude passages that
merely share a word with the question. If nothing bears on the question, write the number of the closest
passage anyway.

Then set `known` to one or two sentences saying what these passages settle about the question: which
entity the question's description resolves to, and which of the two facts the question needs is now in
hand. This note is only used to plan the next search -- do not try to answer the question in it, and do
not worry about exact wording, because the passages themselves are kept and read later."""

SELECT2_INSTRUCTIONS = """The second and last search has returned seven numbered passages. `known` says
what the first search established. Decide which of the new passages are worth reading in full.

Set `relevant` to the passage numbers that bear on the question, best first, separated by commas -- at
most four, and never a number that is not in the list. Prefer the passage about the entity that was still
missing: the second of the two things the question compares, or the entity that the first search's facts
pointed to. Beware of near-namesakes -- a passage about a different person, work or place with a similar
name is not relevant. Exclude passages that merely share a word with the question. If none of them bear
on the question, write the number of the closest one anyway."""

GAP_QUERY_INSTRUCTIONS = """One search has already been run. Plan the second and last one.

`titles_found` lists the articles that search returned and `known` what those passages established. The
question needs facts from two different articles, so exactly one article is still missing.

First set `missing_target` to the name of that missing article, as precisely as you can guess the title
an encyclopedia would give it: the person, work, place, organisation, species or event whose facts you
still lack. Two cases:
- the question names two things and `known` covers only one of them -> the missing target is the other
  one, named exactly as the question names it;
- the question describes the missing thing indirectly ("the director of X", "the album that contains
  Y") -> resolve it first from what is known and name the entity itself, not the description.
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

ANSWER_INSTRUCTIONS = """Answer the question from the articles, then state the answer in the exact form a
grader can match against a short reference answer.

`articles` holds the opening text of the articles selected while searching, each under a heading giving
its real title, exactly as the encyclopedia writes it. `other_titles` lists articles that were returned
but not selected. The text has not been edited: every name, spelling, date and number is as the source
has it, so the answer can be copied straight out of it.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the articles are thin.

Then set `supporting_quote` to the one sentence that contains the answer, copied exactly from the
article text, prefixed by the heading it sits under.

Then set `answer` under these rules:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: write the shortest span that answers the question, in the exact surface form the article
  text uses. A name, a title, a date, a number, or a short noun phrase.
- Use the headings. A sentence tells you something about the article it sits under, not about whatever
  entity it happens to mention: "It was directed by X" under heading H means H was directed by X. When
  the question asks which work, album, film, group, place or person something belongs to and the text
  answers it by *which article the sentence came from*, the answer is that heading, exactly as written.
- Never write a sentence, never restate the question, never add "the answer is", never explain why.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no unit the text does not attach to the number, no gloss in brackets,
  and keep any unit or word the text does attach to it.
- When the question offers a choice between two named alternatives ("which was released first, A or
  B?", "who is older, A or B?", "which one has more members, A or B?"), the answer is one of those two
  names, written as the question writes it -- never the date, number or property that settled the
  comparison, and never both names.
- Otherwise answer the kind of thing the question asks for. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the text only names the group or
  company responsible; a question about where wants the place; about when, the date; about how many, the
  bare number. Never hand back an entity of the wrong kind just because the text names it -- the group
  is not the singer, the album is not the song, the studio is not the director.
- The articles are often incomplete. Commit to the single most likely answer anyway. If they identify
  the subject but not the asked-for property, answer from what you know about that subject, in the same
  short form. Never answer "not specified", "unknown", "cannot be determined", and never mention the
  articles or their gaps in the `answer` field."""

_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}


def _numbered(docs: list[dict[str, Any]]) -> list[str]:
    return [f"[{i + 1}] {d.get('text', '')}" for i, d in enumerate(docs)]


def _picked(raw: str, docs: list[dict[str, Any]], cap: int = 4) -> list[dict[str, Any]]:
    """Read passage numbers out of the selection field; keep the model's order, drop invalid numbers."""
    head = (raw or "").strip()
    for line in head.splitlines():  # a stray sentence after the list is ignored
        if re.search(r"\d", line):
            head = line
            break
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for tok in re.findall(r"\d{1,2}", head):
        i = int(tok)
        if 1 <= i <= len(docs) and i not in used:
            used.add(i)
            out.append(docs[i - 1])
        if len(out) >= cap:
            break
    return out or docs[:1]


def _articles(docs: list[dict[str, Any]]) -> str:
    """Splice the chosen abstracts in verbatim, each under its real title."""
    blocks = []
    for d in docs:
        title = d.get("title", "") or "(untitled)"
        text = d.get("text", "")
        body = text.split(" | ", 1)[1] if text.startswith(title + " | ") else text
        blocks.append(f"## {title}\n{body.strip()}")
    return "\n\n".join(blocks)


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
    if len(words) > 10:
        content = [w for w in words if w.strip(",.;:").lower() not in _QUESTION_WORDS]
        words = content[:8] if content else words[:8]
        text = " ".join(words)
    return text if text else fallback


def _salvage(quote: str, target: str, kept: list[str]) -> str:
    """Last-resort answer when the answering call returns nothing at all.

    Order of preference: the heading of the quote it was building on, the article the planner said was
    missing, then the best-ranked article read. Any committed guess beats an empty string.
    """
    head = (quote or "").strip().splitlines()[0] if (quote or "").strip() else ""
    head = head.split("::")[0].strip().lstrip("#").strip()
    for cand in (head, (target or "").strip(), kept[0] if kept else ""):
        cand = re.sub(r"\s+", " ", cand or "").strip()
        if 1 < len(cand) < 60:
            return cand
    return "yes"


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
        self.select1 = Module(
            inputs=["question", "passages"],
            outputs=["relevant", "known"],
            instructions=SELECT1_INSTRUCTIONS,
            descs={
                "passages": "seven numbered passages; each starts with `[n]`",
                "relevant": "up to four passage numbers, best first, comma separated",
                "known": "one or two sentences on what these passages settle; used only to plan the next search",
            },
        )
        self.plan_hop2 = Module(
            inputs=["question", "titles_found", "known"],
            outputs=["missing_target", "query"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "known": "what the first search established",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.select2 = Module(
            inputs=["question", "known", "passages"],
            outputs=["relevant"],
            instructions=SELECT2_INSTRUCTIONS,
            descs={
                "passages": "seven numbered new passages; each starts with `[n]`",
                "relevant": "up to four passage numbers, best first, comma separated",
            },
        )
        self.generate_answer = Module(
            inputs=["question", "articles", "other_titles"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "articles": "unedited opening text of the selected articles, each under its real title",
                "other_titles": "titles returned by the searches but not selected",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "heading plus the sentence containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # HOP 1 -- choose which articles to keep; never re-type their text
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        sel1_out = self.select1(ctx.llm, tag="select1", question=question, passages=_numbered(hop1))
        keep_1 = _picked(sel1_out.get("relevant", ""), hop1)
        known = " ".join((sel1_out.get("known") or "").split())[:700]

        # HOP 2 -- named coverage gap, then a title-style keyword query for it
        plan = self.plan_hop2(ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1,
                              known=known or "nothing established yet")
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        keep_2 = _picked(self.select2(ctx.llm, tag="select2", question=question, known=known,
                                      passages=_numbered(hop2)).get("relevant", ""), hop2)

        # ANSWER -- read the selected articles as written
        kept: list[dict[str, Any]] = []
        seen_pid: set[Any] = set()
        for d in keep_1 + keep_2:
            pid = d.get("pid", d.get("title"))
            if pid not in seen_pid:
                seen_pid.add(pid)
                kept.append(d)
        rest = [d["title"] for d in hop1 + hop2 if d.get("pid", d.get("title")) not in seen_pid]
        rest_unique: list[str] = []
        for t in rest:
            if t not in rest_unique:
                rest_unique.append(t)
        ctx.note("selection", kept=[d["title"] for d in kept])

        out = self.generate_answer(ctx.llm, tag="answer", question=question,
                                   articles=_articles(kept), other_titles=rest_unique)
        answer = _tidy(out.get("answer", ""))
        if not answer:  # the call ran to its token limit and returned nothing; commit anyway
            answer = _salvage(out.get("supporting_quote", ""), target, [d["title"] for d in kept])
            ctx.note("empty_answer_salvaged", answer=answer)
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "kept_titles": [d["title"] for d in kept],
        }
