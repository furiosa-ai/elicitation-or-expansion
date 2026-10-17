"""Candidate: the answer's extent is chosen between two enumerated spans of the cited sentence.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, extraction and the hop-2
planner are inherited from `shared_property_alignment` verbatim. The one mechanism under test is how the
final call fixes the *extent* of the answer span, plus a per-shape form specification measured on the
train split.

The parent commits to one extent policy -- "the shortest span", "no apposition, no extra qualifier, no
country or state appended, no unit the quote does not attach" -- and applies it to every question. On
items where all the needed articles were retrieved, that policy now costs more than it saves: of its
full-evidence misses whose prediction overlaps the reference, the great majority are *under*-specified
rather than over-specified. A short name is handed back where the cited sentence spells out the full
one, a bare number where the sentence attaches a unit, a place without the region the sentence attaches,
an event name without its year.

Here the extent is not decided by a standing rule but enumerated and then chosen. The call cites one
sentence, writes the minimal span of that sentence that answers the question, writes the longest span of
the same sentence that still says only the answer, and then picks between the two. The choice is
informed by `form_spec`: a description of how long reference answers of this question's shape actually
are, counted in `prepare` over the train split (and nothing else). Code then checks the chosen span is
really a span of the cited sentence and falls back to the minimal one when it is not.
"""

from __future__ import annotations

import json
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

Then fix the extent of the answer in two steps, both of them spans of `supporting_quote`.

Set `answer_core` to the shortest span of the quote that answers the question -- a name, a title, a
date, a number, a short noun phrase -- copied in the exact surface form the quote uses.

Set `answer_full` to the longest span of the *same* quote that still says only the answer and nothing
else. It extends `answer_core` only where the quote itself writes the answer more fully:
- a person's name as that sentence spells it, with the middle names, initials, quoted nickname or title
  the sentence includes;
- a number together with the unit or counted noun the sentence attaches to it;
- a place together with the larger place the sentence attaches to it;
- a work or event together with the year or qualifier the sentence attaches to its name.
It must not reach into a neighbouring fact: no clause, no "which", no "and", no relative pronoun, no
apposition describing the answer, no words the answer is not made of. If nothing in the quote extends
the answer, set `answer_full` to exactly `answer_core`.

Then set `answer` to whichever of the two a reference answer would be, and to nothing else:
- yes_no: write exactly `yes` or `no`. Nothing else -- no explanation, no restatement.
- `form_spec` says how long reference answers of this question's shape run. Follow it: take
  `answer_full` when the quote is the answer entity's own article stating its name or value in full and
  the shape tolerates that length, and `answer_core` when the extra words are a qualifier the question
  did not ask for.
- Either way the answer must be one contiguous span copied from the quote, never a sentence, never a
  restatement of the question, never "the answer is", never an explanation.
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

Set `decision` in one sentence:
- `choice`: compare the two rows on the point asked about -- earlier, older, larger, the one that did
  the thing -- and name the winning entity.
- `shared`: give what the two rows have in common. If the question asks which person, work or place is
  common to both, name it. Otherwise give the wording that covers both rows, preferring wording that
  actually occurs in both rows, or in the question, over a broader category of your own; if the two rows
  word the same thing differently, take the wording the articles use for the class itself.

Then fix the extent of the answer. Set `answer_core` to the shortest form of what you are handing
back, and `answer_full` to the fullest form the evidence writes it in -- for a `choice`, the winning
entity's name as the article *about that entity* spells it at its start, with the middle names,
initials or quoted nickname that article includes; for a `shared` property, the wording plus any word
the sources attach to it. If nothing extends the short form, set `answer_full` to exactly `answer_core`.

Then set `answer` to whichever of the two a reference answer would be, guided by `form_spec`, which
says how long reference answers of this question's shape run:
- if the question asks whether something holds rather than which or what, write exactly `yes` or `no`.
- `choice`: write the winning entity and only it -- one of the two, in one of the two forms above.
  Never the date, number or property that settled the comparison, never both names, never a sentence.
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


_SHAPE_PATTERNS = (
    ("count", re.compile(r"\bhow many\b|\bhow much\b|\bhow long\b|\bhow old\b", re.I)),
    ("date", re.compile(r"\bwhat year\b|\bwhich year\b|\bwhat date\b|\bwhen (?:was|were|did|is|are|do|does)\b", re.I)),
    ("person", re.compile(r"\bwho\b|\bwhose\b|\bwhich (?:actor|actress|singer|author|writer|director|player|artist|musician|politician|man|woman|person)\b", re.I)),
    ("place", re.compile(r"\bwhere\b|\b(?:what|which) (?:city|country|state|county|province|town|region|island|street)\b", re.I)),
    ("class", re.compile(r"\b(?:what|which) (?:type|kind|sort|genre|category|nationality|profession|occupation)\b|\bin common\b|\brelated\b", re.I)),
)


def _question_shape(question: str) -> str:
    """A coarse shape label shared by `prepare` (over train answers) and `run` (over the question)."""
    q = question or ""
    words = re.findall(r"[A-Za-z']+", q)
    if words and words[0].lower() in _AUX_OPENERS:
        return "verdict"
    for name, pattern in _SHAPE_PATTERNS:
        if pattern.search(q):
            return name
    return "other"


def _measure_shapes(train_items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Count how long reference answers of each shape actually run, on the train split only."""
    acc: dict[str, dict[str, float]] = {}
    for it in train_items:
        answer = str(it.get("answer") or "").strip()
        if not answer:
            continue
        shape = "verdict" if answer.lower() in ("yes", "no") else _question_shape(it.get("question", ""))
        row = acc.setdefault(shape, {"n": 0.0, "tokens": 0.0, "long": 0.0, "titled": 0.0})
        tokens = len(answer.split())
        row["n"] += 1
        row["tokens"] += tokens
        if tokens >= 3:
            row["long"] += 1
        titles = [str(t).lower() for t in (it.get("gold_titles") or [])]
        if answer.lower() in titles:
            row["titled"] += 1
    return acc


_DEFAULT_SPEC = ("Reference answers are one contiguous span of the sentence they were taken from, as "
                 "that sentence writes it -- no longer, and no shorter than the sentence writes it.")


def _spec_from_row(shape: str, row: dict[str, float] | None) -> str:
    """Turn the measured counts into a sentence the answering call can act on."""
    if not row or row.get("n", 0) < 4:
        return _DEFAULT_SPEC
    n = row["n"]
    mean = row["tokens"] / n
    long_share = row["long"] / n
    titled = row["titled"] / n
    parts = [f"Reference answers to questions of this shape run {mean:.1f} words on average"]
    if long_share >= 0.35:
        parts.append(f"and {long_share*100:.0f}% of them run three words or more, so the full form the "
                     f"source sentence uses is usually what is wanted rather than the shortest one")
    elif long_share <= 0.15:
        parts.append(f"and only {long_share*100:.0f}% run three words or more, so anything beyond the "
                     f"bare value is usually more than was asked for")
    else:
        parts.append("so extend the short form only where the source sentence writes the answer itself "
                     "more fully")
    if titled >= 0.3:
        parts.append(f"; {titled*100:.0f}% are exactly the name of an article, so prefer a naming form "
                     f"over a description")
    return " ".join(parts).replace(" ;", ";") + "."


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


def _norm_for_span(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _choose_extent(out: dict[str, Any], quote: str) -> str:
    """Keep the call's own choice of extent, but only when it is really a span of the cited sentence."""
    chosen = _tidy(out.get("answer", ""))
    core = _tidy(out.get("answer_core", ""))
    full = _tidy(out.get("answer_full", ""))
    if chosen.lower() in ("yes", "no"):
        return chosen.lower()
    haystack = _norm_for_span(quote)
    candidates = [c for c in (chosen, full, core) if c]
    if not candidates:
        return chosen or core or full
    for cand in candidates:
        needle = _norm_for_span(cand)
        if needle and haystack and needle in haystack:
            return cand
    # nothing was copied from the quote: prefer the shortest stated form over a drifting one
    return chosen or core


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
            inputs=["question", "form_spec", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer_core", "answer_full", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "form_spec": "how long reference answers of this question's shape run",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer_core": "the shortest span of the quote that answers the question",
                "answer_full": "the longest span of the same quote that still says only the answer",
                "answer": "whichever of the two spans a reference answer would be",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "form_spec", "evidence"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer_core",
                     "answer_full", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "entity_a": "the first of the two things, as the question names it",
                "fact_a": "what its own article says on the point the question turns on",
                "entity_b": "the second of the two things, as the question names it",
                "fact_b": "what its own article says on the point the question turns on",
                "form_spec": "how long reference answers of this question's shape run",
                "decision": "one sentence: the winner, or the wording that covers both rows",
                "answer_core": "the shortest form of what is handed back",
                "answer_full": "the fullest form the sources write it in",
                "answer": "whichever of the two a reference answer would be",
            },
        )

        self.shape_stats: dict[str, dict[str, float]] = {}
        self.specs: dict[str, str] = {}

    def prepare(self, train_items: list[dict[str, Any]], ctx) -> None:
        """Measure reference-answer length per question shape on the train split. No model calls."""
        self.shape_stats = _measure_shapes(train_items)
        self.specs = {s: _spec_from_row(s, r) for s, r in self.shape_stats.items()}
        ctx.note("form_spec_learned", shapes={s: round(r["n"]) for s, r in self.shape_stats.items()})

    def get_state(self) -> str:
        return json.dumps({"specs": self.specs})

    def set_state(self, state: str) -> None:
        try:
            self.specs = dict((json.loads(state) or {}).get("specs") or {})
        except ValueError:
            self.specs = {}

    def _form_spec(self, question: str) -> str:
        return self.specs.get(_question_shape(question)) or _DEFAULT_SPEC

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

        spec = self._form_spec(question)
        kind = _comparison_kind(question)
        ctx.note("hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general")
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, form_spec=spec,
                evidence=evidence
            )
            answer = _choose_extent(out, quote="\n".join(evidence))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question, form_spec=spec,
                                   evidence=evidence)
        answer = _choose_extent(out, quote=out.get("supporting_quote", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "answer_core": out.get("answer_core", ""),
            "answer_full": out.get("answer_full", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
