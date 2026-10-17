"""Candidate: read a set of already-graded answers and let the solver itself write down the wording
conventions they follow, then answer under those conventions.

Interaction depth at evaluation time is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, both
extractors, the hop-2 planner and both answering contracts are inherited; the one mechanism under test
is a `prepare`-time induction step that turns a labelled reference pool into a short rule sheet, which
is then handed to the answering call as a separate field.

Every contract in this lineage states the form rule as an absolute written by hand -- shortest span, no
apposition, no unit the quote does not attach, no state after a place name -- and the residual errors
say those absolutes are wrong in both directions at once: a counted noun dropped that the reference
keeps, a unit dropped that the reference keeps, a place left bare that the reference qualifies, a short
name given where the reference uses the fuller one, and equally often the reverse. Hand-written rules
cannot settle this, because the right choice is a convention of the collection being graded against,
not a consequence of the question. Two earlier attempts learned from the reference pool by carrying
examples over as demonstrations, and by measuring word counts per question shape in code; both spoke
about answers in the aggregate.

Here `prepare` instead shows the solver the graded pairs themselves, in batches, and asks it to induce
the conventions: what the reference answers do about extent, register, units, counted nouns, plural vs
singular, and name length -- as imperative rules, each with the number of pairs supporting it, and with
no facts or names in them. A second call merges the batches into one ranked sheet; code drops rules
that carry names or lack support. At run time the sheet is a field of the answering call, bound to the
form of the answer and forbidden from touching which fact is given.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module

INDUCE_INSTRUCTIONS = """You are shown questions together with the reference answer each one was graded
against. The grader compares an answer to the reference after lowercasing it and stripping punctuation
and articles, so a correct fact worded differently from the reference still scores zero.

Your job is to induce the wording conventions the reference answers follow, for a system that has
already found the right fact and only has to decide how to write it down.

Look, pair by pair, at the DISTANCE between the question and its reference answer, and at what the
reference chose to include and to leave out:
- extent: how much of the phrase around the fact is kept -- a counted noun after a number, a unit after
  a quantity, a region after a place name, a qualifier before a class word, a preposition before a date
  span;
- register: whether a class or type is given as a plain common noun or as a specific subtype, singular
  or plural;
- name length: whether a person, work or organisation is written the short way a question would name
  it, or a fuller form;
- verdicts: how a question that asks whether something holds is answered, and how a question offering
  two named alternatives is answered.

Write each convention as one imperative sentence a solver can follow, followed by ` [n=<count>]` where
the count is how many of the pairs shown support it. Rules must be about FORM only: never mention a
person, place, work, organisation or any other name, never state a fact about the world, and never
refer to a specific pair. Prefer a rule that discriminates (says what to include AND what to leave
out) over a vague one. Do not write a rule that only reports a typical length ("answers are two words",
"answers are short"): a solver cannot apply it without cutting a fact in half, and the pairs that look
short are short because of what they leave out, which is what the rule has to say. If the pairs disagree on a point, say which way the majority goes and give the
majority count. At most 10 rules, one per line, no numbering, no other text."""

MERGE_INSTRUCTIONS = """Several batches of the same reference pool were each summarised into wording
conventions for a grader that matches an answer against a short reference answer.

Merge them into one sheet. Combine rules that say the same thing and add up their counts; keep the
wording of the more discriminating one. Drop a rule that another rule contradicts unless one clearly
carries more support, in which case keep only that one, with the count. Drop anything that is not about
the form of an answer, anything naming a person, place, work or organisation, and anything that would
tell a solver which fact to give rather than how to write it.

Order the surviving rules by support, highest first. Keep at most 8. Write each as one imperative
sentence followed by ` [n=<count>]`, one per line, no numbering, no other text."""

CONVENTION_CLAUSE = """
`answer_conventions` is a sheet of wording conventions induced from answers that were already graded
against this kind of reference. Each line ends with the number of graded answers supporting it. Use it
to settle exactly the choices the rules above leave open -- how much of the phrase around the fact to
keep, whether to keep a counted noun or a unit, plain class word or specific subtype, singular or
plural, short name or fuller form. A convention with more support outranks one with less, and a
convention outranks your own instinct about what reads best. It never changes WHICH fact you give, and
it never licenses a sentence, an explanation or a hedge.

One thing no convention can license: dropping part of the fact itself. A date the source gives in full
stays in full, a quantity keeps the unit its source attaches to it, and a name keeps the parts its
source spells out. Conventions decide the wording around the fact -- what to keep of the phrase it sits
in, which register to use -- not how much of the fact survives. A rule about typical length is a
description of other answers, never a licence to cut this one short."""

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
  mention the evidence or its gaps in the `answer` field.
""" + CONVENTION_CLAUSE

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
  "cannot be determined", and never mention the evidence or its gaps in the `answer` field.
""" + CONVENTION_CLAUSE


_RULE_SUPPORT = re.compile(r"\[\s*n\s*=\s*(\d+)\s*\]\s*$", re.IGNORECASE)
_CAPWORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
# words that legitimately open a rule or name a category of answer, so their capital is not a name
_RULE_SAFE = {"Answer", "Give", "Keep", "Drop", "Write", "Use", "Prefer", "Never", "Always", "When",
              "For", "If", "Copy", "Report", "State", "Leave", "Include", "Omit", "Choose", "Match",
              "Return", "Take", "Yes", "No", "The", "A", "An", "Where", "Which", "What", "How"}


def _clean_rules(text: str, limit: int = 8) -> list[str]:
    """Keep supported, name-free, one-line rules. A rule carrying two mid-sentence capitalised words is
    describing an example rather than a convention, so it goes."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*0123456789.) ").strip()
        if len(line) < 15 or len(line) > 240:
            continue
        if line.lower().startswith(("rules", "conventions", "here are", "batch", "note")):
            continue
        support = 0
        m = _RULE_SUPPORT.search(line)
        if m:
            support = int(m.group(1))
            line = line[: m.start()].strip()
        names = [w for w in _CAPWORD.findall(line) if w not in _RULE_SAFE]
        if len(names) >= 2:
            continue
        if support and support < 3:
            continue
        entry = f"{line} [supported by {support} graded answers]" if support else line
        if entry not in out:
            out.append(entry)
        if len(out) >= limit:
            break
    return out


def _render_pairs(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for it in items:
        q = " ".join(str(it.get("question") or "").split())[:300]
        a = " ".join(str(it.get("answer") or "").split())[:120]
        if q and a:
            lines.append(f"question: {q}\nreference answer: {a}")
    return lines


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


class Harness(BaseHarness):
    K = 7
    BATCH = 25

    def __init__(self) -> None:
        self.rules: list[str] = []
        self.induce = Module(
            inputs=["graded_pairs"],
            outputs=["conventions"],
            instructions=INDUCE_INSTRUCTIONS,
            descs={
                "graded_pairs": "questions each followed by the reference answer it was graded against",
                "conventions": "up to 10 imperative form rules, one per line, each ending in ` [n=<count>]`",
            },
        )
        self.merge = Module(
            inputs=["convention_sets"],
            outputs=["conventions"],
            instructions=MERGE_INSTRUCTIONS,
            descs={
                "convention_sets": "one set of induced conventions per batch of the pool",
                "conventions": "up to 8 merged form rules, most supported first, each ending in ` [n=<count>]`",
            },
        )
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
            inputs=["question", "evidence", "answer_conventions"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer_conventions": "wording conventions induced from already-graded answers, with their support",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "evidence", "answer_conventions"],
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
                "answer_conventions": "wording conventions induced from already-graded answers, with their support",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        """Induce the wording conventions of the graded pool, in batches, then merge them."""
        pairs = _render_pairs(list(train_items or []))
        if not pairs:
            return
        sheets: list[str] = []
        for start in range(0, len(pairs), self.BATCH):
            batch = pairs[start: start + self.BATCH]
            if len(batch) < 5:
                break
            try:
                got = self.induce(ctx.llm, tag="induce", graded_pairs=batch)["conventions"]
            except Exception:  # one flaky batch must not sink prepare
                continue
            kept = _clean_rules(got, limit=10)
            if kept:
                sheets.append("\n".join(kept))
        if not sheets:
            return
        if len(sheets) == 1:
            self.rules = _clean_rules(sheets[0])
        else:
            try:
                merged = self.merge(ctx.llm, tag="merge", convention_sets=sheets)["conventions"]
                self.rules = _clean_rules(merged)
            except Exception:
                self.rules = _clean_rules("\n".join(sheets))
        ctx.note("conventions induced", n_batches=len(sheets), rules=self.rules)

    def get_state(self) -> str:
        return json.dumps({"rules": self.rules})

    def set_state(self, state: str) -> None:
        try:
            self.rules = list((json.loads(state) or {}).get("rules") or [])
        except (ValueError, TypeError):
            self.rules = []

    def _sheet(self) -> str:
        if not self.rules:
            return "(no conventions were induced -- fall back on the rules above)"
        return "\n".join(f"- {r}" for r in self.rules)

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

        kind = _comparison_kind(question)
        ctx.note("hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general")
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence,
                answer_conventions=self._sheet(),
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence,
                                   answer_conventions=self._sheet())
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
