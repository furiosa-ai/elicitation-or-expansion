"""Candidate: the second hop's articles carry a provenance flag, and the answer call must confirm the link.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, extraction and the hop-2
planner are inherited from `gap_directed_hop2` verbatim; the one mechanism under test is that the
evidence taken from the second search is no longer presented as equally authoritative as the rest.

The planner *guesses* which article is missing and the search then returns something for that guess, right
or wrong. When the guess is wrong the index still returns seven plausible articles -- a different person
with a similar name, a different work by the same describing phrase -- and the answering call, which sees
one undifferentiated evidence file, reads a fact out of that article and states it confidently.

Here code computes, for every article the second search returned, whether its text actually mentions
anything the question names (a proper-noun span, a quoted string or a year taken from the question) or
anything the first hop's evidence already established. Articles that mention none of it are listed to the
answering call as link-unconfirmed, together with the guess they were fetched for, and the contract makes
the model state the tie between the question and the article it is about to answer from before it answers.
An article it cannot tie to the question is discarded rather than quoted.
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
- Keep the sentence that ties an article to something the question or `context` names -- the line that
  says who a person worked with, what a work belongs to, where a place is -- even when it does not
  state the property being asked about. A later step needs it to tell a relevant article from a
  same-named one.
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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
form a grader can match against a short reference answer.

The evidence comes from two searches. The second search was run for `guessed_target` -- a guess at which
article was still missing -- and it returned seven articles whether or not the guess was right.
`link_unconfirmed` lists the articles among them whose text never mentions anything the question names
and nothing the first search established. Such an article is usually a decoy: a different person with a
similar name, a different work matching the same description, an entity that merely shares a word with
the guess. It states its facts just as confidently as a relevant article does.

So first set `link_check`, in one or two sentences: name the entity the question is anchored on, say
which article you are about to take the answer from, and quote the tie between them -- the evidence line
in which one names the other, or that places them in the same work, group, event or place. Then:
- if that article is listed in `link_unconfirmed` and you cannot point to such a tie, it is a decoy.
  Do not take the answer from it. Say so, drop it, and answer from the remaining evidence plus what you
  reliably know about the entity the question names.
- an article that is not listed in `link_unconfirmed`, or one you can tie to the question, is trusted
  normally.

Then set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `supporting_quote` to the one evidence line that contains the answer, copied exactly. If the
answer came from what you know rather than from a line, write `from knowledge`.

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
- The evidence is often incomplete, and dropping a decoy makes it thinner still. Commit to the single
  most likely answer anyway. If the evidence identifies the subject but not the asked-for property,
  answer from what you know about that subject, in the same short form. Never answer "not specified",
  "unknown", "none", "cannot be determined", and never mention the evidence or its gaps in the
  `answer` field."""

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}

# spans that look like the name of something: quoted strings, or runs of capitalised words
_ANCHOR = re.compile(r"\"[^\"]{2,60}\"|“[^”]{2,60}”|"
                     r"\b[A-Z][\w'\-\.]*(?:\s+(?:[A-Z][\w'\-\.]*|of|the|de|van|von|and))*")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_STOPTOK = _QUESTION_WORDS | {"both", "also", "known", "same", "one", "two", "first", "last", "other",
                              "american", "british", "his", "her", "their", "its", "with", "from",
                              "was", "were", "than", "who", "whose", "while", "along"}


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


def _content_tokens(phrase: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (phrase or "").lower())
            if len(w) > 2 and w not in _STOPTOK]


def _anchors(question: str) -> list[str]:
    """Names the question itself puts on the table: quoted titles, capitalised runs, years."""
    found: list[str] = []
    for m in _ANCHOR.finditer(question):
        span = m.group(0).strip().strip("\"“”").strip()
        if len(span) < 3:
            continue
        if not _content_tokens(span):
            continue
        found.append(span)
    found.extend(_YEAR.findall(question))
    # drop a span that is fully contained in a longer one we already keep
    keep: list[str] = []
    for span in sorted(found, key=len, reverse=True):
        low = span.lower()
        if any(low in k.lower() for k in keep):
            continue
        keep.append(span)
    return keep[:8]


def _mentions(text: str, phrase: str) -> bool:
    low = (text or "").lower()
    toks = _content_tokens(phrase)
    return bool(toks) and all(t in low for t in toks)


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
            inputs=["question", "guessed_target", "link_unconfirmed", "evidence"],
            outputs=["link_check", "answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "guessed_target": "the article the second search was run for -- a guess, possibly wrong",
                "link_unconfirmed": "articles from the second search that mention nothing the question names",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "link_check": "the tie between the question's entity and the article the answer comes from",
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
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        titles_2 = [d["title"] for d in hop2]
        hop2_docs = [d["text"] for d in hop2]

        # PROVENANCE -- which of the guess-fetched articles touch anything the question is about?
        anchors = _anchors(question)
        established = "\n".join(facts_1) + "\n" + "\n".join(titles_1)
        unconfirmed: list[str] = []
        for doc in hop2:
            text = doc["text"]
            title = doc["title"]
            if any(_mentions(text, a) for a in anchors):
                continue
            # an article the first hop already talks about is tied to the question through it
            if _mentions(established, title):
                continue
            unconfirmed.append(title)
        ctx.note("hop2_plan", missing_target=target, query=query_2,
                 n_unconfirmed=len(unconfirmed), unconfirmed=unconfirmed[:7])

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

        # only warn about articles the evidence actually quotes -- an unused decoy is not a risk
        quoted = "\n".join(evidence).lower()
        flagged = [t for t in unconfirmed if t.lower()[:40] in quoted]
        if not flagged:
            flagged_field = "none -- every article quoted below mentions something the question names"
        else:
            flagged_field = flagged

        out = self.generate_answer(
            ctx.llm, tag="answer", question=question,
            guessed_target=target or "(none)", link_unconfirmed=flagged_field, evidence=evidence,
        )
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "link_unconfirmed": flagged,
            "evidence": evidence,
        }
