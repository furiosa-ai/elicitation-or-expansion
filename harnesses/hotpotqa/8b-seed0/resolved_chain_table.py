"""Candidate: the second compression call stops appending sentences and resolves the chain instead.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the hop-1 extractor and the
hop-2 planner are inherited from `shared_property_alignment` verbatim, as is its pair routing. The one
mechanism under test is what the SECOND compression call produces and therefore what the answering call
reads.

In the parent both compression calls do the same job -- copy out sentences that matter -- so the final
call receives a flat bag of three or four `Title :: sentence` lines with no indication of which line
merely identifies something and which line actually states the thing being asked for. Measured on the
parent's own traces that bag contains the reference answer on 27 of its 33 failures where every needed
article was retrieved: the answer is present and the call picks a different span, usually the named
entity from the identifying line rather than the property from the answering line.

The second call is the one moment in the run where all the information exists at once -- it holds the
lines gathered from the first search and is looking at the passages fetched for the gap the planner
named. Here it is asked to lay that out as a resolved two-row chain: which entity the question starts
from and the fact in its own article that points onwards, then which entity that fact names and the
sentence in ITS article that carries the asked-for property. Parallel two-entity questions get the same
two rows read side by side rather than end to end. Every other verbatim line it would have emitted is
kept as an additional block, so nothing is thrown away and the answering call can overrule the chain
when it is wrong.
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

RESOLVE_INSTRUCTIONS = """Finish gathering evidence for a question that needs facts from two different
articles, and lay out how the two articles connect.

`context` holds the lines already taken from the first search. `passages` are the articles just fetched
for the part that was still missing. `link_kind` says how the two articles relate:
- `chain` -- the question identifies one thing and asks for a property that is stated somewhere else, so
  one article points at the other;
- `parallel` -- the question puts two named things side by side, so the two articles are read against
  each other rather than one after the other.

First set `start_entity` to the thing the question starts from, named as an article title would name it,
and `start_fact` to the one sentence, copied exactly from that entity's own article, that matters:
- `chain`: the sentence that points onwards -- the one that names the other article's subject, or states
  the date, work, place or person that identifies it;
- `parallel`: the sentence that states this entity's side of the point the question turns on.

Then set `target_entity` to the other article's subject and `target_fact` to the one sentence, copied
exactly from THAT entity's own article, that carries what the question is asking for:
- `chain`: the sentence stating the asked-for property of `target_entity`;
- `parallel`: this entity's side of the same point, worded as its own article words it.

Rules for both rows:
- Copy sentences verbatim from `context` or `passages`. Never paraphrase, never merge two sentences,
  never complete or correct anything. Keep names, spellings, dates, numbers and units exactly.
- Take each row's sentence from the article about that row's entity. A sentence from the other row's
  article, or from an article about something else with a similar name, does not belong in this row.
- If nothing available covers a row, write `MISSING` as that row's fact and still name the entity as
  well as you can. Never invent a sentence.

Then set `other_facts` to every remaining sentence worth keeping, one per line, in the form
`Title :: the sentence copied exactly as it appears`. Include anything that mentions something the
question names, anything that could be the asked-for fact if the rows above are wrong, and any line in
`context` that neither row used. At most 6 lines, no repeats of the two rows."""

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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence, then state the answer in the exact
form a grader can match against a short reference answer.

`chain` lays out the two articles the question needs: a `from:` row naming the thing the question starts
from with the sentence that points onwards, and a `to:` row naming the thing that sentence leads to with
the sentence that carries the asked-for property. Read the `to:` row for the answer and the `from:` row
only to check you followed the right link. `other_evidence` holds every other sentence gathered. The
chain was assembled by an earlier step and can be wrong: if a row names something the question does not
lead to, or its sentence does not state what was asked, drop that row and answer from `other_evidence`.
If the `to:` row says `MISSING`, the chain reached the right article and no sentence from it was found --
so answer from what you know about the entity that row names, in one step, and do not search the other
lines for a substitute. Never treat a row's entity as the answer just because it is named there.

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

PAIR_ANSWER_INSTRUCTIONS = """The question puts two named things side by side. Line them up one at a
time, decide, and then state the answer in the exact form a grader can match against a short reference
answer.

`chain` already holds one row per side, each naming an entity and giving the sentence from that entity's
own article on the point the question turns on; `other_evidence` holds every other sentence gathered.
The rows were assembled by an earlier step and can be wrong: a row naming the wrong thing or quoting the
other side's article is to be rebuilt from `other_evidence`. A row saying `MISSING` is to be filled in
from what you reliably know about the entity it names, in one step, and never left empty.

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



def _key(line: str) -> str:
    return re.sub(r"\W+", " ", (line or "").lower()).strip()


_DELIMS = re.compile(r"^(?:\[\d+\]\s*)?[\u00ab\u201c\"]?\s*(?:Title\s*::\s*)?")


def _clean_fact(text: str) -> str:
    """Peel the passage/ledger delimiters the extractor sometimes copies in with the sentence."""
    line = _DELIMS.sub("", (text or "").strip()).strip()
    line = line.rstrip("\u00bb\u201d").strip()
    head = re.match(r"^([^|]{2,80})\s\|\s(.+)$", line)  # `Title | abstract` passage form
    if head:
        line = head.group(2).strip()
    return line


def _one_line(text: str, limit: int = 400) -> str:
    """First non-empty line of a module field, stripped of list bullets and quote marks."""
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*0123456789.) ").strip().strip("`").strip()
        if line:
            if len(line) > 1 and line[0] in "\"'\u201c\u2018" and line[-1] in "\"'\u201d\u2019":
                line = line[1:-1].strip()
            return line[:limit]
    return ""


_MISSING = re.compile(r"^\(?(missing|none|n/?a|unknown|not (?:found|available|stated|specified))\b", re.IGNORECASE)


def _fallback(out: dict, res: dict) -> str:
    """A truncated call leaves no `answer` field at all; a blank is always wrong, so commit to
    whatever the run did name -- the chain's target entity, else its subject."""
    for key in ("decision", "supporting_quote"):
        cand = _tidy(_one_line(out.get(key, ""), 120))
        if cand and len(cand.split()) <= 6:
            return cand
    for key in ("target_entity", "start_entity"):
        cand = _one_line(res.get(key, ""), 120)
        if cand:
            return _tidy(cand)
    return ""


def _chain_rows(res: dict, link_kind: str) -> tuple[list[str], list[str]]:
    """Render the resolved chain as two labelled rows; return the rows and the sentences they used."""
    labels = ("from", "to") if link_kind == "chain" else ("side 1", "side 2")
    rows: list[str] = []
    used: list[str] = []
    pairs = (("start_entity", "start_fact"), ("target_entity", "target_fact"))
    for label, (ent_key, fact_key) in zip(labels, pairs):
        entity = _one_line(res.get(ent_key, ""), 120)
        fact = _clean_fact(_one_line(res.get(fact_key, ""), 560))[:500]
        if not entity and not fact:
            continue
        if fact and _MISSING.match(fact):
            fact = "MISSING"
        else:
            used.append(fact)
        rows.append("%s: %s -- %s" % (label, entity or "(unnamed)", fact or "MISSING"))
    return rows, used

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
        self.resolve_chain = Module(
            inputs=["question", "link_kind", "context", "passages"],
            outputs=["start_entity", "start_fact", "target_entity", "target_fact", "other_facts"],
            instructions=RESOLVE_INSTRUCTIONS,
            descs={
                "link_kind": "`chain` (one article points at the other) or `parallel` (two things side by side)",
                "context": "lines already gathered from the first search",
                "start_entity": "the thing the question starts from, or the first of two parallel things",
                "start_fact": "the sentence from its own article that points onwards, copied verbatim",
                "target_entity": "the other article's subject",
                "target_fact": "the sentence from its own article carrying the asked-for fact, copied verbatim",
                "other_facts": "every other useful `Title :: sentence` line, copied verbatim, one per line",
            },
        )
        self.generate_answer = Module(
            inputs=["question", "chain", "other_evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "chain": "the two-article link: a `from:` row and a `to:` row, each entity plus its own sentence",
                "other_evidence": "every other sentence gathered, copied verbatim and prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "chain", "other_evidence"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "chain": "one row per side: the entity plus the sentence from its own article",
                "other_evidence": "every other sentence gathered, copied verbatim and prefixed by its title",
                "entity_a": "the first of the two things, as the question names it",
                "fact_a": "what its own article says on the point the question turns on",
                "entity_b": "the second of the two things, as the question names it",
                "fact_b": "what its own article says on the point the question turns on",
                "decision": "one sentence: the winner, or the wording that covers both rows",
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

        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)

        # RESOLVE -- the second compression call lays the two articles out as a chain
        kind = _comparison_kind(question)
        link_kind = "parallel" if kind else "chain"
        res = self.resolve_chain(
            ctx.llm, tag="extract2", question=question, link_kind=link_kind,
            context=ledger_1 or "(nothing gathered yet)", passages=hop2_docs,
        )
        rows, chain_facts = _chain_rows(res, link_kind)
        facts_2 = _ledger_lines(res.get("other_facts", ""), limit=6)

        # ANSWER -- chain rows first, everything else kept as a second block
        other: list[str] = []
        seen = {_key(f) for f in chain_facts}
        seen.discard("")
        for line in facts_1 + facts_2:
            key = _key(line)
            if key and key not in seen:
                seen.add(key)
                other.append(line)
        if not rows and not other:
            other = list(hop1_docs[:3]) + list(hop2_docs[:3])
        chain = "\n".join(rows) if rows else "(the two articles could not be laid out)"

        ctx.note("hop2_plan", missing_target=target, query=query_2, link_kind=link_kind,
                 comparison_kind=kind or "general", chain=chain, n_other=len(other))
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind,
                chain=chain, other_evidence=other,
            )
            answer = _tidy(out.get("answer", "")) or _fallback(out, res)
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "chain": chain,
                "evidence": other,
            }

        out = self.generate_answer(
            ctx.llm, tag="answer", question=question, chain=chain, other_evidence=other
        )
        answer = _tidy(out.get("answer", "")) or _fallback(out, res)
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "chain": chain,
            "evidence": other,
        }
