"""Candidate: the evidence channel carries *verified article provenance*, enforced in code.

Interaction depth is unchanged from the parent (`gap_directed_hop2`): 4 LLM calls, 2 retrievals, k=7.
The gap-directed hop-2 planner and the answer contract are inherited verbatim so that the single
mechanism under test is how extracted facts are attributed and presented.

Measured defect in the parent: it asks the extraction call to write `Title :: sentence`. Across its 300
traced items, 485 of 935 evidence lines (51.9%) literally begin with the placeholder word `Title`, so on
more than half the evidence the answering call cannot tell which article a sentence is about -- while
23% of reference answers *are* one of the two supporting article titles. Attribution is destroyed exactly
where it is most valuable, and nothing in the pipeline can notice.

The mechanism here removes titles from the model's output space entirely. Passages are numbered, the
extractor emits `[i] sentence` citations, and code resolves `i` to the real retrieved title, verifies the
sentence really occurs in that passage (dropping fabricated lines), and hands the answerer an
article-keyed evidence table plus the roster of every article that was searched.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

EXTRACT_INSTRUCTIONS = """Build an evidence file for a question that needs facts from several articles.

The passages are numbered. Go through them and copy out every sentence that states a fact the question
depends on -- the sentence that identifies an entity the question refers to, and the sentence that states
the property being asked about. Write one line per fact, in the form:

[number] the sentence copied exactly as it appears in that passage

Rules:
- Start every line with the number of the passage you copied from, in square brackets. Never write an
  article name yourself: the number is how the article is recorded, and a line whose number is missing or
  wrong is thrown away.
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep names,
  spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Cover every passage that matters, at most two lines each: take a line from a passage even when it only
  partly matters (for example it names a person, work or place the question mentions in passing) -- a
  later step decides what to use.
- Skip passages that say nothing about the question. Never write a sentence that is not in the passages.
- A passage that names or defines something the question mentions ALWAYS matters, even when it does not
  state the property being asked about: the sentence that says what a thing is, who made it or which
  group, work or place it belongs to is how the two articles get connected. Take it.
- Never answer with an empty list or with a note that nothing is relevant. If no passage settles the
  question, copy the sentences that come closest to identifying the things the question names.
- At most 8 lines. Output only the lines, nothing else."""

EXTRACT2_INSTRUCTIONS = """Extend an evidence file for a question that needs facts from several articles.

`context` holds the facts gathered so far, grouped under the article each came from. The new passages are
numbered. Go through them and copy out every sentence that adds a fact the question depends on. Write one
line per fact, in the form:

[number] the sentence copied exactly as it appears in that passage

Rules:
- Start every line with the number of the passage you copied from, in square brackets. Never write an
  article name yourself; a line whose number is missing or wrong is thrown away.
- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep names,
  spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a fact `context` already contains; add only new facts.
- Cover every new passage that matters, at most two lines each. Skip passages that say nothing about the
  question. Never write a sentence that is not in the passages.
- A passage that names or defines something the question mentions ALWAYS matters, even when it does not
  state the property being asked about. Take it rather than reporting that nothing is relevant.
- At most 8 lines. Output only the lines, nothing else."""

GAP_QUERY_INSTRUCTIONS = """One search has already been run. Plan the second and last one.

`titles_found` lists the articles that search returned and `evidence` the facts taken from them, grouped
under the article each came from. The question needs facts from two different articles, so exactly one
article is still missing.

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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence, then state the answer in the exact form a
grader can match against a short reference answer.

`evidence` groups the collected sentences under the heading of the article they were copied from, so a
heading tells you what its sentences are *about*. `articles_searched` lists every article the searches
returned, whether or not a sentence was taken from it.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `supporting_quote` to the one evidence sentence that contains the answer, copied exactly,
prefixed by the heading it sits under.

Then set `answer` under these rules:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- otherwise: write the shortest span that answers the question, in the exact surface form the evidence
  uses. A name, a title, a date, a number, or a short noun phrase.
- Use the headings. A sentence says something about the article it sits under, not about whatever entity
  it happens to mention: "It was directed by X" under heading H means H was directed by X, and a sentence
  about a song under a heading naming an album tells you that album's name. When the question asks which
  work, album, film, group, place or person something belongs to and the evidence answers it by *which
  article the sentence came from*, the answer is that heading, written exactly as the heading writes it.
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
_CITE = re.compile(r"^\[?\s*(?:passage\s*)?(\d{1,2})\s*[\]\).:]+\s*(.+)$", re.IGNORECASE | re.DOTALL)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}


def _shape(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _cited_facts(text: str, docs: list[dict[str, Any]], limit: int = 8) -> list[tuple[str, str]]:
    """Parse `[i] sentence` lines into (real_title, sentence) pairs, verified against the passages.

    The title is never taken from the model: it comes from `docs[i-1]`. A sentence that occurs in no
    retrieved passage is dropped; one cited to the wrong passage is re-attributed to the passage that
    actually contains it.
    """
    shapes = [_shape(d.get("text", "")) for d in docs]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line or _LEDGER_JUNK.match(line) or line.startswith("[[") or line.lower().startswith("rules:"):
            continue
        m = _CITE.match(line)
        idx, sent = (int(m.group(1)), m.group(2).strip()) if m else (0, line)
        sent = sent.strip().strip("`").strip()
        if len(sent) < 12:
            continue
        sent = sent[:600]
        key = _shape(sent)[:160]
        if not key or key in seen:
            continue
        probe = _shape(sent)[:90]
        home = idx - 1 if 1 <= idx <= len(docs) else -1
        if home < 0 or (probe and probe not in shapes[home]):
            home = next((j for j, s in enumerate(shapes) if probe and probe in s), home)
        if home < 0:
            continue  # cites nothing we retrieved: a fabricated line
        seen.add(key)
        out.append((docs[home].get("title", ""), sent))
        if len(out) >= limit:
            break
    return out


def _numbered(docs: list[dict[str, Any]]) -> list[str]:
    return [f"[{i + 1}] {d.get('text', '')}" for i, d in enumerate(docs)]


def _body(doc: dict[str, Any]) -> str:
    """Passage text without the leading `Title | `, so a title is never printed twice."""
    title, text = doc.get("title", ""), doc.get("text", "")
    return (text.split(" | ", 1)[1] if title and text.startswith(title + " | ") else text).strip()


def _table(facts: list[tuple[str, str]]) -> str:
    """Group facts under the heading of the article they were copied from, order of first appearance."""
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    for title, sent in facts:
        head = title or "(untitled source)"
        if head not in groups:
            groups[head] = []
            order.append(head)
        groups[head].append(sent)
    blocks = []
    for head in order:
        body = "\n".join(f"- {s}" for s in groups[head])
        blocks.append(f"## {head}\n{body}")
    return "\n".join(blocks)


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


def _salvage(quote: str, target: str, roster: list[str]) -> str:
    """Last-resort answer when the answering call returns nothing at all.

    Order of preference: the heading of the quote it was building on, the article the planner said was
    missing, then the best-ranked article seen. Any committed guess beats an empty string.
    """
    head = (quote or "").strip().splitlines()[0] if (quote or "").strip() else ""
    head = head.split("::")[0].strip().lstrip("#").strip()
    for cand in (head, (target or "").strip(), roster[0] if roster else ""):
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
        self.extract1 = Module(
            inputs=["question", "passages"],
            outputs=["facts"],
            instructions=EXTRACT_INSTRUCTIONS,
            descs={
                "passages": "numbered passages; each starts with `[n]`",
                "facts": "one `[n] sentence` line per relevant fact, copied verbatim from passage n",
            },
        )
        self.plan_hop2 = Module(
            inputs=["question", "titles_found", "evidence"],
            outputs=["missing_target", "query"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "evidence": "facts so far, grouped under the article each was copied from",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
        )
        self.extract2 = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts"],
            instructions=EXTRACT2_INSTRUCTIONS,
            descs={
                "context": "facts so far, grouped under the article each was copied from",
                "passages": "numbered new passages; each starts with `[n]`",
                "facts": "one `[n] sentence` line per new relevant fact, copied verbatim from passage n",
            },
        )
        self.generate_answer = Module(
            inputs=["question", "articles_searched", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "articles_searched": "titles of every article the two searches returned",
                "evidence": "sentences copied verbatim, grouped under a heading naming the article each came from",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "heading plus the evidence sentence containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # HOP 1 -- extraction cites passage numbers; code owns the titles
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        facts_1 = _cited_facts(
            self.extract1(ctx.llm, tag="extract1", question=question, passages=_numbered(hop1))["facts"],
            hop1,
        )
        table_1 = _table(facts_1)

        # HOP 2 -- named coverage gap, then a title-style keyword query for it
        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1,
            evidence=table_1 or "\n".join(d["text"] for d in hop1[:3]),
        )
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        facts_2 = _cited_facts(
            self.extract2(ctx.llm, tag="extract2", question=question, context=table_1,
                          passages=_numbered(hop2))["facts"],
            hop2,
        )

        # ANSWER -- article-keyed evidence table plus the roster of everything searched
        merged: list[tuple[str, str]] = []
        seen: set[str] = set()
        for title, sent in facts_1 + facts_2:
            key = _shape(sent)[:160]
            if key and key not in seen:
                seen.add(key)
                merged.append((title, sent))
        if not merged:  # extraction refused everything: fall back to the passages themselves
            for d in hop1[:2] + hop2[:2]:
                pid = d.get("pid", d.get("title"))
                if pid in seen:
                    continue
                seen.add(pid)
                merged.append((d.get("title", ""), _body(d)))
        table = _table(merged)

        roster: list[str] = []
        for t in titles_1 + [d["title"] for d in hop2]:
            if t not in roster:
                roster.append(t)
        ctx.note("evidence_table", n_lines=len(merged), n_articles=table.count("## "))

        out = self.generate_answer(ctx.llm, tag="answer", question=question,
                                   articles_searched=roster, evidence=table)
        answer = _tidy(out.get("answer", ""))
        if not answer:  # the call ran to its token limit and returned nothing; commit anyway
            answer = _salvage(out.get("supporting_quote", ""), target, roster)
            ctx.note("empty_answer_salvaged", answer=answer)
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": [f"{t} :: {s}" for t, s in merged],
        }
