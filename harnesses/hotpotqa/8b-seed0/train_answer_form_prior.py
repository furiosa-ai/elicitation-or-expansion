"""Candidate: learn from a reference collection what SHAPE an answer to this kind of question takes.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the extractors, the hop-2
planner and the two answering contracts are inherited verbatim; the one mechanism under test is a
prepare-time prior over answer FORM, keyed by a signature the code computes from the question alone.

Every answer contract in this lineage states the form rule as an absolute: shortest span, no apposition,
bare number, no unit the source does not attach. Absolute rules cannot express what the graded answers
actually look like, and the residual errors are precisely disagreements about extent and register at
full evidence: a counted noun kept or dropped (`six` against `six children`), a preposition kept or
dropped (`from 1897 until 1963` against `1897 until 1963`), a place qualified or not, a class question
answered with a specific subtype rather than the plain common noun, an ordinal question answered with a
title. No instruction settles these, because the right choice is a property of the collection, not of
logic.

So `prepare` reads the reference questions it is given, groups them by a code-computed template
signature -- the interrogative pattern plus the head noun it asks for, e.g. a count, a year, a date, a
class-of-thing, a who, a which-<noun> -- and measures, from those questions' reference answers only,
the shape they take: how many words, whether bare numerals, whether capitalised, and a few examples kept
purely as examples of form. At run time the code computes the signature of the question in hand and hands
the matching profile to the answering call as a separate field, with an instruction that binds it to the
form of the answer and forbids it from touching the content. No question is ever matched to a stored
question; only the shape of the answer is carried over.
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

FORM_RULE = """

`answer_form` is measured from an independent reference collection of questions built the same way as
this one, grouped by question shape. It reports how long the reference answers to questions of THIS
shape run, whether they are bare numerals, whether they are capitalised, and a few of them as examples
of form. Use it to settle the form of your answer and nothing else:
- how much to include -- whether the counted noun, the unit, the leading preposition, the qualifying
  place or the parenthetical belongs in the answer or not;
- which register -- a proper name, or a plain common noun in lower case;
- how many words to aim for. If your answer is far longer than the profile, you are including material
  the reference answer would leave out; cut it to the shape shown.
It says nothing whatever about the subject of this question: never take content from the examples, never
answer with an example, and never let it override what the evidence states. Content from the evidence,
form from the profile."""

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


# ---------------------------------------------------------------- form prior

_ORDINAL = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "last", "latest", "next"}
_SKIP_AFTER_WH = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "the", "a", "an",
                  "of", "in", "on", "at", "to", "for", "this", "that", "these", "those", "his", "her",
                  "its", "their", "one", "both", "other", "same", "also", "American", "British", "he",
                  "she", "it", "they", "you", "we", "there", "many", "much", "kind", "type"}
_CLASS_WORDS = ("type", "kind", "sort", "genre", "category", "class", "species", "nationality",
                "profession", "occupation")


def _signature(question: str) -> str:
    """A template signature computed from the question alone: what shape of thing it asks for."""
    q = " " + re.sub(r"\s+", " ", (question or "").strip()) + " "
    low = q.lower()
    words = re.findall(r"[A-Za-z']+", low)
    if not words:
        return "other"
    if re.search(r"\bhow (?:many|much)\b", low):
        return "count"
    if re.search(r"\b(?:what|which) (?:year|decade)\b|\bin (?:what|which) year\b", low):
        return "year"
    if re.search(r"\b(?:what|which) date\b|\bon (?:what|which) date\b|\bwhen (?:was|were|is|are|did|does|do)\b", low):
        return "date"
    if re.search(r"\b(?:what|which) (?:%s)\b" % "|".join(_CLASS_WORDS), low):
        return "class"
    if re.search(r"\b(?:are|were|is|was) (?:both )?(?:which|of what)\b|\bin common\b|\bboth\b.*\b(?:are|were)\b", low):
        return "class"
    if words[0] in _AUX_OPENERS:
        return "yes_no"
    if re.search(r"\bwhere\b|\b(?:what|which) (?:city|country|state|county|province|town|place|region)\b", low):
        return "place"
    if re.search(r"\bwho\b|\bwhose\b|\bwhom\b", low):
        return "who"
    if re.search(r"\bwhy\b|\bhow (?:is|are|does|do|did|can)\b", low):
        return "reason"
    m = re.search(r"\b(?:what|which)\b", low)
    if m:
        rest = re.findall(r"[A-Za-z']+", low[m.end():])
        for w in rest[:4]:
            if w in _SKIP_AFTER_WH or len(w) < 3:
                continue
            if w in _ORDINAL:
                return "ordinal"
            return "which_" + w
        return "which_other"
    return "other"


def _family(sig: str) -> str:
    """Coarser bucket used when a signature has too few reference answers of its own."""
    if sig.startswith("which_"):
        return "a named thing"
    if sig in ("year", "date"):
        return "a point in time"
    return sig


def _profile(answers: list[str]) -> dict[str, Any]:
    lens = sorted(len(a.split()) for a in answers)
    n = len(answers)
    numeric = sum(1 for a in answers if re.fullmatch(r"[\d][\d,\.]*", a.strip()))
    wordy = [a for a in answers if re.search(r"[A-Za-z]", a)]
    lowered = sum(1 for a in wordy if a == a.lower())
    examples: list[str] = []
    seen: set[str] = set()
    for a in sorted(answers, key=lambda s: (len(s.split()), len(s))):
        key = a.lower()
        if key in seen or len(a.split()) > 7:
            continue
        seen.add(key)
        examples.append(a[:60])
        if len(examples) >= 5:
            break
    return {
        "n": n,
        "typ": lens[n // 2] if n else 0,
        "lo": lens[0] if n else 0,
        "hi": lens[-1] if n else 0,
        "num": numeric,
        "low": lowered,
        "wordy": len(wordy),
        "ex": examples,
    }


def _render_profile(sig: str, p: dict[str, Any] | None) -> str:
    if not p or not p.get("n"):
        return "(no reference profile for this question shape -- keep the answer as short as the question allows)"
    n = p["n"]
    bits = [
        "question shape `%s`, measured over %d reference answers" % (sig, n),
        "length: usually %d word%s (seen %d to %d)" % (p["typ"], "" if p["typ"] == 1 else "s", p["lo"], p["hi"]),
    ]
    if p["num"]:
        bits.append("%d of %d are a bare numeral with nothing attached" % (p["num"], n))
    wordy = p.get("wordy", n)
    if p["low"] and wordy:
        bits.append("%d of the %d that contain words are written entirely in lower case "
                    "(a plain common noun, not a name)" % (p["low"], wordy))
    if p["ex"]:
        bits.append("examples of the FORM only: " + " | ".join(p["ex"]))
    return "; ".join(bits)


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
    MIN_GROUP = 4

    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
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
            inputs=["question", "evidence", "answer_form"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS + FORM_RULE,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_form": "measured shape of reference answers to questions of this shape (form only, never content)",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "evidence", "answer_form"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS + FORM_RULE,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "answer_form": "measured shape of reference answers to questions of this shape (form only, never content)",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "entity_a": "the first of the two things, as the question names it",
                "fact_a": "what its own article says on the point the question turns on",
                "entity_b": "the second of the two things, as the question names it",
                "fact_b": "what its own article says on the point the question turns on",
                "decision": "one sentence: the winner, or the wording that covers both rows",
                "answer": "the short answer span only -- never a sentence",
            },
        )

    def prepare(self, train_items: list[dict[str, Any]], ctx: Any = None) -> None:
        """Group the reference questions by computed signature and measure their answers' shape."""
        groups: dict[str, list[str]] = {}
        for it in train_items or []:
            ans = (it.get("answer") or "").strip()
            if not ans:
                continue
            sig = _signature(it.get("question") or "")
            groups.setdefault(sig, []).append(ans)
            fam = _family(sig)
            if fam != sig:
                groups.setdefault(fam, []).append(ans)
            groups.setdefault("_all", []).append(ans)
        self.profiles = {sig: _profile(ans) for sig, ans in groups.items()
                         if sig == "_all" or len(ans) >= self.MIN_GROUP}
        if ctx is not None and hasattr(ctx, "note"):
            ctx.note("form_profiles", shapes=sorted(k for k in self.profiles if k != "_all"))

    def get_state(self) -> str:
        return json.dumps({"profiles": self.profiles})

    def set_state(self, state: str) -> None:
        try:
            self.profiles = (json.loads(state) or {}).get("profiles") or {}
        except (ValueError, TypeError):
            self.profiles = {}

    def _form_for(self, question: str) -> tuple[str, str]:
        sig = _signature(question)
        if sig == "yes_no":
            return sig, ("question shape `yes_no`: the reference answer to a question that asks whether "
                         "something holds is the single word yes or no")
        prof = self.profiles.get(sig)
        used = sig
        if not prof:
            used = _family(sig)
            prof = self.profiles.get(used)
        if not prof:
            prof, used = self.profiles.get("_all"), "any shape"
        return used, _render_profile(used, prof)

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        sig, answer_form = self._form_for(question)

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
        ctx.note("hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general",
                 signature=sig)
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence,
                answer_form=answer_form,
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "signature": sig,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence,
                                   answer_form=answer_form)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "signature": sig,
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
