"""Candidate: the answer is predicted from knowledge before any passage is read, then reconciled.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7): a passage-free prediction call, one
call that extracts the first hop's facts and plans the second hop at the same time, one call that
extracts the second hop's facts, and the answering call. Retrieval queries, the extraction contract,
the gap-directed planner wording and the two-entity alignment path are inherited from
`shared_property_alignment`; what is new is the *order in which the model commits*.

Diagnosis. On two thirds of the frontier's errors one of the two needed articles was never retrieved,
and over those items the missing article is never named anywhere in the passages the first search
returned -- there is no better query to write, so the answer has to come from the model's own knowledge
of the entity the question describes. The frontier does ask for that as a fallback, but by then the
model has read fourteen passages, and the failures show it does not fall back at all: it lifts the most
topically adjacent name out of a passage about someone else (a co-star, a similarly named athlete, a
neighbouring release) and commits to it. Reading first anchors the guess.

So the order is inverted. The first call sees the question and nothing else: it writes the hop chain it
would have to follow and registers the answer it believes from memory, plus how firm that belief is.
The passages arrive afterwards and cannot retro-fit that commitment. The final call is then no longer
free-form generation but a reconciliation with a stated verdict: retrieved text that actually states
the asked-for property of the right subject overrules the registered prediction, and when no retrieved
line states it, the registered prediction stands rather than being replaced by a name that happens to
sit near the question's words. Two-entity questions keep the parent's alignment contract untouched.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module


PREDICT_INSTRUCTIONS = """You are given a question that needs facts from more than one article, and no
articles yet. Commit to an answer now, from memory alone, before any source material is available.

Set `chain` to the path the question forces, as two or three short steps naming what has to be
identified at each one -- the thing the question describes indirectly first, then the property asked of
it. Write it as `step -> step`, no prose.

Set `prediction` to the answer you believe right now, already in the form a short reference answer
would take: a name, a title, a date, a number, a short noun phrase, or exactly `yes` or `no` if the
question asks whether something holds. Never a sentence, never a hedge, never "unknown" -- if the
entities are unfamiliar, name the most likely value for the kind of thing asked and let the next steps
correct it.

Set `basis` to one of:
- `known` -- you recognise the entities and are confident this is the fact;
- `inferred` -- you recognise the entities and are reasoning from what you know about them;
- `guess` -- you do not recognise the entities and this is a plausible filler.

Answer only these three fields. Do not ask for sources and do not describe what you would look up."""


EXTRACT_PLAN_INSTRUCTIONS = """Build an evidence file from the passages of the first search, and plan
the second and last search.

`prediction` is an answer registered from memory before any passage was read. Treat it as a lead to be
checked, never as a fact: it tells you which entity to look for, and it may be wrong.

First set `facts`: go through the passages and copy out every sentence that states a fact the question
depends on -- the sentence that identifies an entity the question refers to, and the sentence that
states the property being asked about. One line per fact, in the form:

Title :: the sentence copied exactly as it appears in the passage

- Copy verbatim. Never paraphrase, never merge two sentences, never fix or complete anything. Keep
  names, spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Include a line even when it only partly matters (for example it names a person, work or place the
  question mentions in passing) -- a later step decides what to use.
- Skip passages that say nothing about the question. Never invent a line that is not in the passages.
- At most 8 lines.

The question needs facts from two different articles and `titles_found` lists what the first search
returned, so exactly one article is still missing.

Set `missing_target` to the name of that missing article, as precisely as you can guess the title an
encyclopedia would give it: the person, work, place, organisation, species or event whose facts you
still lack. Two cases:
- the question names two things and the passages cover only one of them -> the missing target is the
  other one, named exactly as the question names it;
- the question describes the missing thing indirectly ("the director of X", "the album that contains
  Y") -> resolve it from the facts you just copied and name the entity itself, not the description.
`missing_target` must be a proper name that could stand as the title of an article. Never write a role
or a relation ("the vocalist of X", "birthplace of Z"): if that is all you can say, name the entity
whose article would CONTAIN that fact instead (the band, the film, the album, the person the question
names).

Then set `query` to the search string. The index matches words literally against article titles and
their opening paragraph, so:
- write 2 to 8 words: the missing target's name, plus one or two words that disambiguate it (a type
  word like film, band, river, novel, or a year, only if the name alone is ambiguous);
- never write a question or a sentence, and use no question words (who, what, when, where, which, how);
- never include words describing facts you already have, and never include a name that is already in
  `titles_found` unless it is part of the missing target's own name;
- drop filler words entirely.
Output the query as bare keywords."""

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


RECONCILE_INSTRUCTIONS = """An answer was registered from memory before any passage was read. The
retrieved evidence is now available. Decide what the answer is, and state it in the exact form a grader
can match against a short reference answer.

`prediction` is that registered answer and `basis` says how firm it was. `evidence` holds sentences
copied verbatim from the retrieved articles, each prefixed by its title.

First set `support` to the one `evidence` line that states the asked-for property of the subject the
question actually asks about -- copied exactly -- or to `none`. Be strict about the subject: a line
about a different person, work or place, however close it sits to the question's words, does not
support an answer. A line that identifies the subject but says nothing about the property asked for is
not support either.

Then set `verdict` to one of:
- `confirmed` -- a supporting line agrees with `prediction`;
- `corrected` -- a supporting line states something else, so the prediction was wrong;
- `unaddressed` -- `support` is `none`: nothing retrieved states this fact.

Then set `answer`:
- `confirmed` or `corrected`: answer with a span of `support`, copied in its surface form. Retrieved
  text always outranks memory, including when it contradicts a `known` prediction.
- `unaddressed`: no retrieved line states this fact, so the answer comes from knowledge, not from the
  passages. Do NOT lift a name, date or number out of the passages you have -- none of them is about
  the right subject, and a name that merely sits near the question's words is a wrong answer, not a
  clue. Which knowledge to use depends on what the evidence settled:
  * the evidence has identified the subject the property belongs to (it named the person, work or place
    the question was pointing at) -- answer from what you know about THAT subject by name, even if that
    means discarding the registration, which was made before the subject was known;
  * the evidence has not identified the subject either -- keep the registered `prediction`, tightened
    into short-answer form, however thin its `basis` was. It was made from the question alone, which is
    still more than the passages support.

Form rules for `answer`:
- if the question asks whether something holds (is/are/was/were/do/does/did/can/has/have/both ... ?),
  write exactly `yes` or `no`, nothing else.
- otherwise write the shortest span that answers the question: a name, a title, a date, a number or a
  short noun phrase. Never a sentence, never a restatement of the question, never "the answer is",
  never an explanation.
- give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no unit the source does not attach to the number, no gloss in
  brackets -- and keep any unit or word the source does attach.
- when the question offers a choice between two named alternatives, the answer is one of those two
  names, written as the question writes it -- never the property that settled it.
- answer the kind of thing asked for: who performed, directed, wrote or founded something wants a
  person, even when the evidence only names the group or company; where wants a place; when a date; how
  many a bare number.
- never answer "not specified", "unknown" or "cannot be determined", and never leave `answer` empty."""

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


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.predict = Module(
            inputs=["question"],
            outputs=["chain", "prediction", "basis"],
            instructions=PREDICT_INSTRUCTIONS,
            descs={
                "chain": "the hops the question forces, as `step -> step`",
                "prediction": "the answer believed from memory, already in short-answer form",
                "basis": "known, inferred or guess",
            },
        )
        self.extract_plan = Module(
            inputs=["question", "prediction", "titles_found", "passages"],
            outputs=["facts", "missing_target", "query"],
            instructions=EXTRACT_PLAN_INSTRUCTIONS,
            descs={
                "prediction": "an answer registered from memory before any passage was read: a lead, not a fact",
                "titles_found": "titles the first search returned",
                "facts": "one `Title :: sentence` line per relevant fact, copied verbatim",
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
        self.reconcile = Module(
            inputs=["question", "prediction", "basis", "evidence"],
            outputs=["support", "verdict", "answer"],
            instructions=RECONCILE_INSTRUCTIONS,
            descs={
                "prediction": "the answer registered from memory before any passage was read",
                "basis": "how firm that registration was: known, inferred or guess",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "support": "the evidence line stating the asked-for property of the right subject, or `none`",
                "verdict": "confirmed, corrected or unaddressed",
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

        # 1. COMMIT before reading anything
        pre = self.predict(ctx.llm, tag="predict", question=question)
        prediction = _tidy(pre.get("prediction", ""))
        basis = (pre.get("basis") or "").strip().splitlines()[0][:40].lower() or "guess"
        chain = (pre.get("chain") or "").strip().splitlines()[0][:300]

        # 2. HOP 1 -- the question itself, as in the parent
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]

        # 3. extract the first hop's facts and plan the second search in the same call
        step = self.extract_plan(
            ctx.llm, tag="extract_plan", question=question, prediction=prediction or "(none)",
            titles_found=titles_1, passages=hop1_docs,
        )
        facts_1 = _ledger_lines(step.get("facts", ""))
        ledger_1 = "\n".join(facts_1)
        target = (step.get("missing_target") or "").strip().splitlines()[0][:120] if step.get("missing_target") else ""
        query_2 = _keyword_query(step.get("query", ""), fallback=(target or prediction or question))

        # 4. HOP 2
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        facts_2 = _ledger_lines(
            self.extract2(ctx.llm, tag="extract2", question=question, context=ledger_1, passages=hop2_docs)["facts"]
        )

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
        ctx.note("registered", prediction=prediction, basis=basis, chain=chain,
                 missing_target=target, query=query_2, comparison_kind=kind or "general")

        if kind:  # two named things side by side: the parent's alignment contract, untouched
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence
            )
            return {
                "answer": _tidy(out.get("answer", "")),
                "comparison_kind": kind,
                "prediction": prediction,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.reconcile(
            ctx.llm, tag="answer", question=question, prediction=prediction or "(none)", basis=basis,
            evidence=evidence,
        )
        answer = _tidy(out.get("answer", ""))
        verdict = (out.get("verdict") or "").strip().splitlines()[0][:40].lower()
        if not answer:                      # never return nothing: the registration is the fallback
            answer = prediction
        ctx.note("reconciled", verdict=verdict, prediction=prediction, answer=answer,
                 kept_prediction=bool(answer) and answer.lower() == prediction.lower())
        return {
            "answer": answer,
            "comparison_kind": "general",
            "prediction": prediction,
            "basis": basis,
            "verdict": verdict,
            "support": (out.get("support") or "")[:400],
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
