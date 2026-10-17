"""Candidate: the second search targets the article that would CONTAIN the answer sentence, never an
article about the guessed answer itself.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7) and the evidence/answer layer is the
frontier's, untouched. The one mechanism under test is what the second retrieval is aimed at.

Trace evidence for the change: on the frontier's search-set errors where the first hop covered one of
the two needed articles and the second hop added nothing, the planner repeatedly named the ANSWER it
had already guessed instead of the link it still had to resolve -- it searched a guessed co-star rather
than the film that lists the cast, a guessed programme rather than the performer whose article names
the programme, a guessed character rather than the actor. Searching a guess can only confirm it: the
index returns an article about the guess, the extraction copies facts about the guess, and the answer
call reads a coherent file that supports the wrong entity. That is worse than retrieving nothing.

So the planner is made to write the two sides of the chain apart. `answer_side` is the thing the
question ultimately asks for, written as a description and explicitly excluded from the search.
`link_entity` is the article that would state it -- the entity the question names but the first search
missed, or the connecting entity the first hop's evidence resolves. The query is built from the link
only: code deletes from it every word that belongs to the answer-side description, and, when the link
name is grounded in neither the question nor the evidence (i.e. the planner invented it after all),
code appends the question's own rarest content words so the search still has a lexical anchor that does
not come from a guess.
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
question needs facts from two different articles, so one article is still missing -- and it is the
article that would STATE the answer, not an article about the answer.

First set `answer_side` to a short description of what the question ultimately asks for: "the year he
was born", "the show she is best known for", "the crest of that club", "which of the two came first".
Describe it; do not name it. If a name for it has already occurred to you, that is a guess -- put the
guess nowhere in this task. Searching a guess only returns an article about the guess, which confirms
it whether it is right or wrong.

Then set `link_entity` to the article that would state the answer_side fact. Work out which one it is:
- the question names a person, work, place, organisation or event whose article is not in
  `titles_found` -> that is the link, named exactly as the question names it;
- every thing the question names is already in `titles_found` -> the link is the entity those articles
  point AT: the performer behind the role, the film that lists the cast, the band that recorded the
  song, the parent company, the person holding the office. Take its name from the evidence lines if
  they state it, otherwise name it from what you know.
A link is an entity whose encyclopedia article exists, never a role, a relation or a property.

Then set `query` to the search string that finds the link's article. The index matches words literally
against titles and opening paragraphs, so:
- start with the link's name in full, then add at most two words that separate it from others of the
  same name (a type word like film, band, river, novel, footballer, or a year or a place);
- put NO word of `answer_side` in it: not the property asked about, not a date or number you hope to
  find, not any name you guessed for the answer;
- no question words, no sentence, no punctuation, at most eight words in total."""

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


_BAN_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "is", "are",
             "was", "were", "by", "with", "as", "it", "he", "she", "they", "his", "her", "its", "their",
             "from", "at", "which", "what", "who", "whom", "whose", "when", "where", "how", "why",
             "did", "does", "do", "has", "have", "had", "been", "be", "than", "also", "same", "both"}
_PROPER = re.compile(r"\b[A-Z][\w'\u2019-]{2,}")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w'-]+", (text or "").lower()) if t]


def _router_query(raw: str, link: str, answer_side: str, question: str, covered: str) -> str:
    """Build the second query from the LINK only.

    Two code steps back the instruction up: every word that belongs to the answer-side description is
    deleted from the query (a word describing what we are looking FOR steers a term-frequency index
    towards articles about the guess rather than the article that states it), and a link name that the
    planner invented -- one that appears neither in the question nor in anything hop 1 found -- gets one
    proper name from the question appended, so the search keeps an anchor that is not itself a guess.
    """
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip("*`\"'").strip()
    text = re.sub(r"^(query|search|search for|keywords)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = text.rstrip("?.").strip()
    if not text:
        text = link

    def _stem(t: str) -> str:
        return t[:-2] if t.endswith("es") and len(t) > 4 else (t[:-1] if t.endswith("s") and len(t) > 3 else t)

    ban = {_stem(t) for t in _tokens(answer_side) if t not in _BAN_STOP}
    link_toks = set(_tokens(link))
    words: list[str] = []
    for w in text.split():
        tok = _tokens(w)
        if not tok:
            continue
        t = tok[0]
        if t in link_toks:
            words.append(w)
            continue
        if t in _QUESTION_WORDS or _stem(t) in ban:
            continue
        words.append(w)
    if not words:
        words = link.split() or text.split()
    if len(words) > 8:
        words = words[:8]

    # the query must actually name the link: a planner that described it instead ("lead singer of the
    # band") gives the index nothing to match on the article we are after.
    link_content = [t for t in _tokens(link) if t not in _BAN_STOP and len(t) > 2]
    if link_content and not any(t in set(_tokens(" ".join(words))) for t in link_content):
        words = link.split() + words
        if len(words) > 8:
            words = words[:8]

    source = set(_tokens(question)) | set(_tokens(covered))
    if link_content and not any(t in source for t in link_content):
        have = set(_tokens(" ".join(words)))
        covered_toks = set(_tokens(covered))
        for name in _PROPER.findall(question):
            t = name.lower()
            if t not in have and t not in covered_toks and t not in _BAN_STOP:
                words = words + [name]
                break

    out = " ".join(words).strip()
    return out if out else question


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
            outputs=["answer_side", "link_entity", "query"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "answer_side": "what the question ultimately asks for, described and never searched",
                "link_entity": "the article that would state that fact",
                "query": "2-8 bare keywords naming the link entity",
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
        target = (plan.get("link_entity") or "").strip().splitlines()[0][:120]
        answer_side = (plan.get("answer_side") or "").strip().splitlines()[0][:200]
        query_2 = _router_query(plan.get("query", ""), target, answer_side, question,
                                covered=" ".join(titles_1) + " " + ledger_1)
        ctx.note("hop2_plan", link_entity=target, answer_side=answer_side, query=query_2)

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
            "link_entity": target,
            "query_2": query_2,
            "evidence": evidence,
        }
