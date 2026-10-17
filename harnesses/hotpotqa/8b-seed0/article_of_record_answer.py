"""Candidate: the answer may only come from the article that owns the fact being asked for.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, both extraction calls, the
gap-directed planner and the two-entity alignment path are inherited from `shared_property_alignment`
verbatim. The one mechanism under test is a *provenance-eligibility gate* on the general answer path.

Diagnosis from the parent's traces. Two thirds of its errors are items where one of the two needed
articles was never retrieved, and measured over those items the missing article is never even named in
the passages the first search returned -- so no better query can be written from what is on the page and
the retrieval side is at its ceiling at this depth. What goes wrong is what the answerer does instead:
its contract tells it to "set supporting_quote to the one evidence line that contains the answer", which
presupposes that some retrieved line does. When none does, the model satisfies the contract anyway by
lifting the most topically adjacent name out of a passage about something else -- a different performer,
a similarly named footballer, a co-star, a neighbouring film -- and answers with it confidently. The
same failure appears with full recall whenever the answer's owner is one hop away from the passage that
mentions the question's words.

The gate needs provenance it can trust, and the parent's does not survive contact with the extractor:
its evidence lines are supposed to read `Title :: sentence`, but the model routinely writes the literal
placeholder `Title ::`, so the answering call often cannot tell which article a line came from at all.
Because the sentences are copied verbatim, code can repair that for free -- find the passage containing
the sentence and put its real title back on the line. That happens only on the answering call's input;
the extraction and planning calls receive byte-identical prompts to the parent's, so retrieval is
unchanged.

So this candidate hands the answer call something it has never had: the complete inventory of titles the
two searches returned, which says by omission which articles are *absent*. The contract then makes
eligibility explicit -- name the entity whose own article would state the answer, say which retrieved
title is that entity's article or that none is, and take the answer from that article's lines when it is
present, from knowledge of the named owner when it is not. Copying a name out of a non-owner article is
ruled out by name. Two-entity questions keep the parent's alignment contract untouched.
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


OWNER_ANSWER_INSTRUCTIONS = """Answer the question, then state the answer in the exact form a grader can
match against a short reference answer.

`titles_retrieved` is the complete inventory of the articles the searches returned. `evidence` holds
sentences copied from them, each prefixed by its title. The inventory matters as much as the evidence:
it tells you which articles you have, and by omission which you do not have.

First set `property_owner` to the ONE entity that owns the fact being asked for -- the thing whose own
encyclopedia article would state the answer. Follow the question's chain to get there: when the question
points at that entity indirectly ("the director of X", "the album that contains Y", "the man who founded
Z"), resolve the description against the evidence and name the entity itself, not the description. When
the question asks for a property of something it names outright, that thing is the owner.

When the question asks whether something holds of two things, both of them own a share of the fact:
write both, joined by ` and `.

Then set `owner_article` to the title from `titles_retrieved` that is the owner's OWN article, copied
exactly, or to `none`. Judge this strictly: an article about a different thing that shares part of the
name, a list, a work the owner appears in, or an article that merely mentions the owner in passing is
not the owner's own article -- that case is `none`. With two owners, give both titles joined by ` ; `,
writing `none` in the place of either one whose own article the inventory does not contain.

An article you hold only as a title -- it is in `titles_retrieved` but no `evidence` line comes from
it -- does not state the fact either. Treat that as `none` too.

Then set `answer`. Where it may come from follows from `owner_article`:
- a title: answer with a span of an `evidence` line from that article, or of a line that states the
  owner's property directly, copied in that line's surface form.
- `none`: no retrieved article states this fact, so answer from what you know about `property_owner` by
  name. Do not lift a name, date or number out of the passages you do have -- they are about other
  things, and a name sitting next to the question's words in an article that is not the owner's is a
  wrong answer, not a clue. If the owner is unfamiliar, still commit to the most likely value for the
  kind of thing asked.

Form rules for `answer`:
- if the question asks whether something holds (is/are/was/were/do/does/did/can/has/have/both ... ?),
  write exactly `yes` or `no`, nothing else.
- otherwise write the shortest span that answers the question, in the surface form of the line it comes
  from: a name, a title, a date, a number or a short noun phrase. Never a sentence, never a restatement
  of the question, never "the answer is", never an explanation.
- give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no unit the source does not attach to the number, no gloss in
  brackets -- and keep any unit or word the source does attach.
- when the question offers a choice between two named alternatives, the answer is one of those two
  names, written as the question writes it -- never the date, number or property that settled it.
- answer the kind of thing asked for: who performed, directed, wrote or founded something wants a
  person, even when the evidence only names the group or company; where wants a place; when a date; how
  many a bare number. The group is not the singer, the album is not the song, the studio is not the
  director.
- never answer "not specified", "unknown" or "cannot be determined", never leave `answer` empty, and
  never mention the evidence or its gaps in it.
- `none` is a value for `owner_article` only. It is never an answer: an absent article means you answer
  about `property_owner` from memory, not that there is nothing to say."""

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


def _title_of(line: str) -> str:
    """Title prefix of a `Title :: sentence` evidence line."""
    return line.split("::", 1)[0].strip().lower() if "::" in line else ""


def _flat(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


# extraction modules frequently echo the field placeholder instead of the article name
_PLACEHOLDER_TITLES = {"title", "article", "source", "passage", "titles", "unknown", ""}


def _reattribute(lines: list[str], docs: list[dict[str, Any]]) -> list[str]:
    """Rewrite each evidence line's prefix with the real title of the passage the sentence came from.

    Nothing is sent to the model to do this and no call is added: the sentences were copied verbatim,
    so the passage that contains one identifies its article. Lines whose sentence matches no passage
    keep whatever prefix they had. This runs only on the answering call's input; the extraction and
    planning calls see exactly what the parent's do.
    """
    flat_docs = [(d.get("title") or "", _flat(d.get("text") or "")) for d in docs]
    out: list[str] = []
    for line in lines:
        prefix, sentence = ("", line)
        if "::" in line:
            prefix, sentence = line.split("::", 1)
        sentence = sentence.strip()
        probe = _flat(sentence)[:70]
        title = ""
        if len(probe) >= 25:
            for cand, flat in flat_docs:
                if probe in flat:
                    title = cand
                    break
        if not title and prefix.strip().lower() not in _PLACEHOLDER_TITLES:
            title = prefix.strip()
        out.append(f"{title} :: {sentence}" if title else sentence)
    return out


_NO_ARTICLE = {"none", "no", "n/a", "nothing", "absent", "not present", "not retrieved",
               "not in the inventory", "not applicable", "-", ""}
# `no` is a legitimate answer, so the non-answer set for the answer field is narrower
_NON_ANSWER = {"none", "n/a", "na", "nothing", "unknown", "not specified", "not applicable",
               "not present", "not retrieved", "cannot be determined", "-", ""}
_ROLE_START = re.compile(r"^(the|a|an|his|her|its|their)\b", re.IGNORECASE)


def _fallback_answer(owner: str, question: str, evidence: list[str]) -> str:
    """Something answer-shaped when the final call returns a non-answer token."""
    owner = (owner or "").strip().strip("\"'")
    if owner and not _ROLE_START.match(owner) and not owner.lower().startswith("none"):
        return owner                       # the entity that was said to own the fact
    spans = _nameish_spans(question)
    if spans:
        return spans[-1]
    for line in evidence:
        for span in _nameish_spans(line.split("::", 1)[-1]):
            return span
    return owner


def _snap_one(text: str, titles: list[str]) -> str:
    """Snap one declared title onto a retrieved title, or `` when it declares none."""
    text = re.sub(r"^(title|article)\s*[:\-]\s*", "", (text or "").strip().strip("*`\"'")).strip()
    low = text.lower().strip(" .")
    if low in _NO_ARTICLE:
        return ""
    for t in titles:
        if t.lower() == low:
            return t
    for t in titles:                      # tolerate a dropped or added disambiguator
        if low and (low in t.lower() or t.lower() in low):
            return t
    return ""


def _owner_articles(raw: str, titles: list[str]) -> list[str]:
    """The retrieved articles declared to own the fact -- one per owner, `` entries dropped."""
    first = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    out: list[str] = []
    for part in re.split(r"\s*(?:;|\|)\s*", first):
        snapped = _snap_one(part, titles)
        if snapped and snapped not in out:
            out.append(snapped)
    return out


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
        self.answer_owned = Module(
            inputs=["question", "titles_retrieved", "evidence"],
            outputs=["property_owner", "owner_article", "answer"],
            instructions=OWNER_ANSWER_INSTRUCTIONS,
            descs={
                "titles_retrieved": "every article the searches returned, one per line",
                "evidence": "sentences copied verbatim from those articles, each prefixed by its title",
                "property_owner": "the entity whose own article would state the answer",
                "owner_article": "the retrieved title that is that entity's own article, or `none`",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "evidence"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
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

        # HOP 1 -- unchanged from the parent
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]
        facts_1 = _ledger_lines(
            self.extract1(ctx.llm, tag="extract1", question=question, passages=hop1_docs)["facts"]
        )
        ledger_1 = "\n".join(facts_1)

        # HOP 2 -- unchanged from the parent
        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1, evidence=facts_1 or hop1_docs[:3]
        )
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        titles_2 = [d["title"] for d in hop2]
        hop2_docs = [d["text"] for d in hop2]
        facts_2 = _ledger_lines(
            self.extract2(ctx.llm, tag="extract2", question=question, context=ledger_1, passages=hop2_docs)["facts"]
        )

        evidence: list[str] = []
        seen: set[str] = set()
        for line in _reattribute(facts_1 + facts_2, hop1 + hop2):
            key = re.sub(r"\W+", " ", line.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                evidence.append(line)
        if not evidence:
            evidence = list(hop1_docs[:3]) + list(hop2_docs[:3])

        # the inventory of what the searches actually returned, in retrieval order and deduplicated
        inventory: list[str] = []
        for t in titles_1 + titles_2:
            if t not in inventory:
                inventory.append(t)

        kind = _comparison_kind(question)
        ctx.note("hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general")

        if kind:  # two named things side by side: the parent's alignment contract, untouched
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence
            )
            return {
                "answer": _tidy(out.get("answer", "")),
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.answer_owned(
            ctx.llm, tag="answer", question=question, titles_retrieved=inventory, evidence=evidence
        )
        answer = _tidy(out.get("answer", ""))
        owner = (out.get("property_owner") or "").strip().splitlines()[0][:120]
        if re.sub(r"\W+", " ", answer.lower()).strip() in _NON_ANSWER:
            answer = _tidy(_fallback_answer(owner, question, evidence))
            ctx.note("non_answer_replaced", answer=answer)
        articles = _owner_articles(out.get("owner_article", ""), inventory)

        # provenance report: was the answer really taken from an article declared to own the fact?
        owned = [ln for ln in evidence if any(_title_of(ln) == a.lower() for a in articles)]
        key = re.sub(r"\W+", " ", answer.lower()).strip()
        grounded = bool(key) and any(key in re.sub(r"\W+", " ", ln.lower()) for ln in owned)
        ctx.note("provenance", owner=owner, owner_article=" ; ".join(articles) or "none",
                 from_owner_article=grounded, owner_lines=len(owned))

        return {
            "answer": answer,
            "comparison_kind": "general",
            "property_owner": owner,
            "owner_article": " ; ".join(articles) or "none",
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
