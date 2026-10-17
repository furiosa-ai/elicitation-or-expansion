"""Candidate: evidence is partitioned by whether its article is reachable from something the question names.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, extraction and the hop-2
planner are inherited from `shared_property_alignment` verbatim; the one mechanism under test is a
code-computed link graph over the retrieved articles that restructures what the answering call reads.

A bag-of-words index returns an article whenever it shares a token with the query, so most of the 14
articles a run sees are look-alikes: an article about a different person with the same surname, a
different band whose name contains the same number, a different award-winner with the same profession.
The parent's ledger is a flat list of sentences with nothing to tell them apart, and its contract
("set supporting_quote to the one evidence line that contains the answer") is satisfiable by a
look-alike line, so the model hands back a property of the wrong entity.

Here the code builds a link graph before the answering call. An article is an ANCHOR when every
distinctive word of its title occurs in the question -- the question names it outright. An article is
LINKED when the anchors' own text mentions it, i.e. it is one hop away along a real encyclopedia
reference. Everything else is UNLINKED: nothing the question names points at it. The ledger is then
presented as two labelled blocks plus the anchor->mentions map, and the answering call must say whether
it took the answer from linked evidence or from its own knowledge of the entity it resolved.
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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
form a grader can match against a short reference answer.

First set `answer_type` to one of: yes_no, person, organisation, work, place, date, number, other.
Use yes_no whenever the question asks whether something holds (are/is/was/were/do/does/did/can/has/
have/both ... ?), even when the evidence is thin.

Then set `supporting_quote` to the one evidence line that contains the answer, copied exactly.

`evidence` comes in two labelled blocks and `link_map` shows which retrieved articles the ones the
question names actually mention.
- `[LINKED]` holds lines from articles the question names outright, and from articles those articles
  themselves mention. The answer chain runs through these.
- `[UNLINKED]` holds lines from articles that nothing the question names points at. The search index
  returned them only because they share a word, a name or a number with the question: a different
  person with the same surname, a different band with the same digits, a different holder of the same
  job or prize. Treat them as being about somebody else unless the question names them.

Take the answer from the `[LINKED]` block. Never take it from an `[UNLINKED]` line just because that
line happens to state the kind of thing the question asks for -- a count, a date, a name -- when its
subject is not the entity the question is about. If no `[LINKED]` line states the answer, first work
out from the linked lines WHICH entity the question is ultimately about, name it to yourself, and then
answer from what you reliably know about that entity. Set `answer_source` to `evidence` when a linked
line states the answer and to `knowledge` when you supplied it yourself.

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

PAIR_ANSWER_INSTRUCTIONS = """The question puts two named things side by side. Line them up one at a
time, decide, and then state the answer in the exact form a grader can match against a short reference
answer.

`comparison_kind` says which job the question is asking for:
- `choice` -- it offers two named alternatives and wants one of them handed back;
- `shared` -- it wants the property, class or relation the two things have in common.

Set `entity_a` and `entity_b` to the two things, named as the question names them.

Set `fact_a` to what is true of `entity_a` on the single point the question turns on -- the date, the
number, the occupation, the class of thing it belongs to, the country -- copied in the wording of the
article about `entity_a`. Then set `fact_b` the same way for `entity_b`, from the article about
`entity_b`. For both rows:
- take the fact from that entity's own article. A sentence from the other entity's article, or from an
  article about something else with a similar name, is not evidence about this one.
- copy the article's wording rather than rewording it, and keep it to the phrase that matters.
- if no line covers that entity, write what you reliably know about it in the same short form and add
  `(from knowledge)`. Never leave a row empty and never write that it is missing.
- `evidence` comes in two labelled blocks. `[LINKED]` holds lines from articles the question names and
  from articles those mention; `[UNLINKED]` holds lines from articles nothing in the question points
  at, which the index returned only because they share a word, a name or a number with it. A row about
  `entity_a` must come from a `[LINKED]` line about `entity_a` itself; an `[UNLINKED]` line about a
  same-named other thing is not evidence about it, so use your own knowledge instead.
- Set `answer_source` to `evidence` when both rows come from linked lines and to `knowledge` otherwise.

Set `decision` in one sentence:
- `choice`: compare the two rows on the point asked about -- earlier, older, larger, the one that did
  the thing -- and name the winning entity.
- `shared`: give what the two rows have in common. If the question asks which person, work or place is
  common to both, name it. Otherwise give the wording that covers both rows, preferring wording that
  actually occurs in both rows, or in the question, over a broader category of your own; if the two rows
  word the same thing differently, take the wording the articles use for the class itself.

Then set `answer`:
- if the question asks whether something holds rather than which or what, write exactly `yes` or `no`.
- `choice`: write one of `entity_a` or `entity_b`, spelled as the question spells it. Never the date,
  number or property that settled the comparison, never both names, never a sentence.
- `shared`: write only what the two have in common -- one short noun phrase, singular unless the
  question asks about several things at once, in the source's wording. No `and`, no pairing of two
  wordings, no "both are", no explanation. When the question asks which *person*, *work* or *place*
  they have in common ("which actor was in both A and B"), what they share is that entity, so write its
  name; otherwise write the class or property itself and no entity name.
- Never restate the question, never add "the answer is", never explain why.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no gloss in brackets.
- The evidence is often incomplete. Commit to the single most likely answer anyway, from what you know
  about the entities if need be. Never answer "not specified", "unknown", "neither", "not related" or
  "cannot be determined", and never mention the evidence or its gaps in the `answer` field."""

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}

# a yes/no question opens with an auxiliary: those keep the general contract
_AUX_OPENERS = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had",
                "will", "would", "should", "am", "must"}
_NAMEISH = re.compile(r"\"[^\"]{2,60}\"|“[^”]{2,60}”|"
                      r"\b[A-Z][\w'\-\.]*(?:\s+(?:[A-Z][\w'\-\.]*|of|the|de|van|von))*")
_SHARED_PAT = re.compile(
    r"\bin common\b|\bboth\b|\bshare[sd]?\b|\bhow (?:is|are|do|does) .{0,60}\brelated\b|"
    r"\b(?:what|which) (?:type|kind|sort|genre|category|nationality|profession|occupation)\b|"
    r"\bare which\b|\bwere which\b|\bare of what\b",
    re.IGNORECASE)
_TRIVIAL = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "what", "which", "who", "when",
            "where", "how", "why", "is", "are", "was", "were", "do", "does", "did", "both", "american",
            "british", "english", "he", "she", "it", "they", "this", "that", "his", "her", "their"}


_TITLE_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "was", "were", "are",
               "be", "by", "with", "at", "from", "as", "that", "this", "it", "his", "her", "their",
               "its", "de", "la", "le", "van", "von", "del"}


def _distinctive(title: str) -> list[str]:
    """Content words of an article title, minus any parenthetical disambiguator."""
    bare = re.sub(r"\([^)]*\)", " ", title or "")
    return [w for w in re.findall(r"[A-Za-z0-9']+", bare)
            if len(w) > 2 and w.lower() not in _TITLE_STOP]


def _covered(words: list[str], haystack: str) -> bool:
    """All but at most one distinctive word of a title occurs in `haystack` (already lowercased)."""
    if not words:
        return False
    hits = sum(1 for w in words if w.lower() in haystack)
    return hits >= max(1, len(words) - 1)


def _link_graph(docs: list[dict], question: str) -> tuple[set[str], set[str], list[str]]:
    """Split retrieved titles into anchors / linked / unlinked and describe the anchors' mentions.

    anchor  -- the question names this article outright
    linked  -- an anchor's own text mentions this article
    """
    ql = " " + question.lower() + " "
    # an anchor must be named OUTRIGHT: every distinctive word of its title occurs in the question.
    # the one-word tolerance in `_covered` is for the mention step, where prose abbreviates a title.
    anchors = [d for d in docs
               if _distinctive(d["title"])
               and all(w.lower() in ql for w in _distinctive(d["title"]))]
    anchor_titles = {d["title"] for d in anchors}
    anchor_text = " ".join((d.get("text") or "").lower() for d in anchors)
    linked: set[str] = set(anchor_titles)
    mention_map: list[str] = []
    for d in docs:
        if d["title"] in anchor_titles:
            continue
        if anchor_text and _covered(_distinctive(d["title"]), anchor_text):
            linked.add(d["title"])
    for a in anchors:
        text = (a.get("text") or "").lower()
        named = [d["title"] for d in docs
                 if d["title"] != a["title"] and _covered(_distinctive(d["title"]), text)]
        mention_map.append(f"{a['title']} -> mentions: " + (", ".join(named) if named else "(none retrieved)"))
    return anchor_titles, linked, mention_map


def _attribute(line: str, docs: list[dict]) -> str:
    """Best guess at which retrieved article an evidence line was copied from."""
    head, _, rest = line.partition("::")
    head = head.strip().strip('"')
    if head and head.lower() not in ("title", ""):
        for d in docs:
            if d["title"].lower() == head.lower():
                return d["title"]
        for d in docs:
            if head.lower() in d["title"].lower() or d["title"].lower() in head.lower():
                return d["title"]
    probe = re.sub(r"\s+", " ", (rest or line)).strip()[:45].lower()
    if len(probe) >= 15:
        for d in docs:
            if probe in re.sub(r"\s+", " ", (d.get("text") or "")).lower():
                return d["title"]
    return ""


def _nameish_spans(question: str) -> list[str]:
    spans: list[str] = []
    for m in _NAMEISH.finditer(question):
        span = m.group(0).strip().strip("\"“”").strip(" ,;:")
        if len(span) < 3:
            continue
        if span.lower() in _TRIVIAL:
            continue
        spans.append(span)
    return spans


def _comparison_kind(question: str) -> str:
    """`choice`, `shared`, or `` when the question does not put two named things side by side."""
    q = question.strip()
    words = re.findall(r"[A-Za-z']+", q)
    if not words:
        return ""
    if words[0].lower() in _AUX_OPENERS:  # yes/no phrasing keeps the general contract
        return ""
    spans = _nameish_spans(q)
    if len(spans) < 2:
        return ""
    # `A or B?` -- a name on each side of the disjunction, close to it
    for m in re.finditer(r"\bor\b", q, re.IGNORECASE):
        before, after = q[: m.start()], q[m.end():]
        if before.rstrip().endswith((",", ";", ":")):
            continue  # an aside, not two parallel alternatives ("known as X, or Y, is ...")
        left = _nameish_spans(before[-70:])
        right = _nameish_spans(after[:70])
        if left and right and len(after.strip()) < 90:
            return "choice"
    if _SHARED_PAT.search(q):
        return "shared"
    return ""


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
            inputs=["question", "link_map", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer_source", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "link_map": "which retrieved articles the ones the question names actually mention",
                "evidence": "verbatim source sentences in two blocks: [LINKED] and [UNLINKED]",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "answer_source": "`evidence` if a linked line states it, `knowledge` if you supplied it",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "link_map", "evidence"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer_source", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "link_map": "which retrieved articles the ones the question names actually mention",
                "evidence": "verbatim source sentences in two blocks: [LINKED] and [UNLINKED]",
                "entity_a": "the first of the two things, as the question names it",
                "fact_a": "what its own article says on the point the question turns on",
                "entity_b": "the second of the two things, as the question names it",
                "fact_b": "what its own article says on the point the question turns on",
                "decision": "one sentence: the winner, or the wording that covers both rows",
                "answer_source": "`evidence` if both rows come from linked lines, else `knowledge`",
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
        hop2_docs = [f"{d['title']} | {d['text']}" for d in hop2]
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

        # LINK GRAPH -- who does the question actually reach?
        docs: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for d in list(hop1) + list(hop2):
            if d["title"] not in seen_titles:
                seen_titles.add(d["title"])
                docs.append(d)
        anchors, linked, mention_map = _link_graph(docs, question)
        linked_lines: list[str] = []
        unlinked_lines: list[str] = []
        for line in evidence:
            src = _attribute(line, docs)
            shown = line if line.split("::")[0].strip().lower() not in ("", "title") else (
                f"{src} :: {line.partition('::')[2].strip()}" if src else line)
            if src and src not in linked:
                unlinked_lines.append(shown)
            else:
                linked_lines.append(shown)
        blocks = ["[LINKED]"] + (linked_lines or ["(no line could be traced to a linked article)"])
        if unlinked_lines:
            blocks += ["", "[UNLINKED]"] + unlinked_lines
        link_map = mention_map or [f"(the question names none of the retrieved titles outright; "
                                   f"retrieved: {', '.join(d['title'] for d in docs[:8])})"]

        kind = _comparison_kind(question)
        ctx.note("hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general",
                 anchors=sorted(anchors), n_linked=len(linked_lines), n_unlinked=len(unlinked_lines))
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind,
                link_map=link_map, evidence=blocks
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "answer_source": out.get("answer_source", ""),
                "unlinked": unlinked_lines,
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question,
                                   link_map=link_map, evidence=blocks)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "answer_source": out.get("answer_source", ""),
            "unlinked": unlinked_lines,
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
