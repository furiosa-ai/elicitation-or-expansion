"""Candidate: build the second search in code and hedge it over every name still unaccounted for.

Interaction depth stays inside the seed program's envelope and gets shallower: 2 retrievals at k=7 as
before, but 3 LLM calls instead of 4. Extraction and both answering contracts are inherited from
`shared_property_alignment` verbatim. The one mechanism under test is that the planning call is
DELETED and the second query is computed from the question and the first search's titles.

Every system so far has spent one of its four calls asking the model to name the single article it is
still missing, and then searched for that one name. That bet is all-or-nothing: on the parent's own
run, 61 of its 102 incomplete-coverage items have the planner pointing at the wrong entity, usually
because it aimed at the entity it would have to infer -- the answer -- instead of a name the question
states outright.

The index scores each document independently as a bag of words, so a query holding several names is
not a compromise between them: a document that matches one of the names well still outranks documents
that match none, and the other names cost it nothing. A second query can therefore carry every name
the question mentions whose own article did not come back, and cover all of those bets at once instead
of one. Offline on the train split, replacing a single targeted second query with this code-built
hedge is worth several points of gold-title coverage over searching the question alone.

Two consequences are deliberate. The query never contains a property word, since it is assembled from
names and class words only. And the freed call is not spent elsewhere: if the hedge is what carries
coverage, the planner's call was never earning its keep, and a three-call system that holds the
frontier's accuracy says so unambiguously.
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

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}

# a yes/no question opens with an auxiliary: those keep the general contract
_AUX_OPENERS = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had",
                "will", "would", "should", "am", "must"}
# a capitalized run may run through lowercase connectives (`Back in the U.S.S.R`) and may carry an
# elided token (`Rock 'N Roll Music`), so those continue a name rather than breaking it in two
_NAMEISH = re.compile(r"\"[^\"]{2,60}\"|“[^”]{2,60}”|"
                      r"\b[A-Z][\w'\-\.]*(?:\s+(?:'?[A-Z][\w'\-\.]*|of|the|in|on|at|for|a|an|de|van|von|du|di|del|la|le)\b)*")

# a name may end on a real word only: a connective that trailed into the match is not part of it
_TRAIL = re.compile(r"\s+(?:of|the|in|on|at|for|a|an|de|van|von|du|di|del|la|le)$", re.IGNORECASE)
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
        while True:  # drop connectives the capitalized run swallowed at its end
            trimmed = _TRAIL.sub("", span).strip()
            if trimmed == span:
                break
            span = trimmed
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


# properties an article states about its subject without being asked: never part of a title query
_PROPERTY_WORDS = {
    "born", "birth", "birthdate", "birthplace", "date", "dates", "year", "years", "age", "died",
    "death", "length", "height", "width", "depth", "distance", "population", "genre", "genres",
    "style", "nationality", "nationalities", "citizenship", "location", "located", "situated",
    "founder", "founded", "founding", "release", "released", "premiere", "premiered", "aired",
    "published", "number", "occupation", "profession", "elevation", "altitude", "weight", "cost",
    "price", "revenue", "starring", "starred", "directed", "written", "wrote", "produced",
    "performed", "sang", "voiced", "oldest", "youngest", "earlier", "earliest", "later",
    "latest", "bigger", "biggest", "larger", "largest", "smaller", "smallest", "older",
    "younger", "more", "most", "fewer", "less", "least", "before", "after", "between", "during",
    "about", "info", "information", "details", "facts", "fact", "summary", "overview",
}


def _norm_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w]


def _title_core(title: str) -> str:
    """A title without its parenthetical disambiguator: `X (1981 film)` -> `X`."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", (title or "")).strip()


def _covers(span: str, title: str) -> bool:
    """True when `title` is plausibly the article ABOUT `span` (not merely a passage mentioning it)."""
    sw = [w for w in _norm_words(span) if w not in _TRIVIAL]
    tw = _norm_words(_title_core(title))
    if not sw or not tw:
        return False
    tset = set(tw)
    # every content word of the name occurs in the title, and the title adds little beyond it
    if all(w in tset for w in sw):
        return True
    # the title is the name plus a comma-qualifier, or the name is the title plus one
    if len(sw) >= 2 and len(tw) >= 2 and sw[:2] == tw[:2]:
        return True
    return False


def _coverage_audit(question: str, titles: list[str]) -> tuple[list[str], list[str]]:
    """Mark every name the question mentions as having its own article among `titles`, or not."""
    spans = _nameish_spans(question)
    # drop a span that is contained in a longer one (`Van Trump` inside `Philemon Van Trump`)
    kept: list[str] = []
    for span in sorted(spans, key=len, reverse=True):
        low = span.lower()
        if any(low in k.lower() for k in kept):
            continue
        kept.append(span)
    kept = [s for s in spans if s in kept]  # restore question order
    lines: list[str] = []
    missing: list[str] = []
    for span in kept[:8]:
        if any(_covers(span, t) for t in titles):
            lines.append(f"{span} -- has own article")
        else:
            lines.append(f"{span} -- NOT RETRIEVED")
            missing.append(span)
    return lines, missing


# class words: they say what kind of thing a name denotes, so they disambiguate a title query
_CLASS_WORDS = {
    "film", "movie", "album", "song", "single", "band", "group", "novel", "book", "series",
    "miniseries", "game", "castle", "river", "lake", "mountain", "island", "city", "town", "village",
    "suburb", "county", "school", "university", "college", "company", "magazine", "newspaper", "party",
    "team", "league", "season", "church", "temple", "museum", "airport", "station", "bridge", "road",
    "highway", "species", "genus", "plant", "bird", "fish", "opera", "ballet", "play", "musical",
    "poem", "painting", "sculpture", "comic", "manga", "anime", "episode", "tournament", "championship",
    "festival", "award", "war", "battle", "treaty", "election", "hotel", "restaurant", "brewery",
}


def _hedged_query(question: str, uncovered: list[str], covered: list[str]) -> str:
    """The second query, assembled from names rather than planned: no property words can enter it."""
    classes = [w for w in _norm_words(question) if w in _CLASS_WORDS]
    seen: set[str] = set()
    words: list[str] = []

    def push(text: str) -> None:
        for tok in (text or "").split():
            key = tok.strip(",.;:'\"()").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            words.append(tok.strip(",;:()"))

    if uncovered:
        # hedge: every name whose own article is still missing, all in one bag of words.
        # a name goes in whole -- a repeated word inside it is part of the name, not a duplicate
        for name in uncovered[:4]:
            for tok in name.split():
                cleaned = tok.strip(",;:()")
                if cleaned:
                    words.append(cleaned)
                    seen.add(cleaned.strip(",.;:'\"()").lower())
    else:
        # every name is covered, so the missing article is one the question only describes:
        # keep the description and drop the words that belong to the names already in hand
        owned = {w for name in covered for w in _norm_words(name)}
        for tok in re.findall(r"[\w'\-\.]+", question):
            parts = _norm_words(tok)
            if not parts or all(w in owned for w in parts):
                continue
            if any(w in _QUESTION_WORDS or w in _TRIVIAL or w in _PROPERTY_WORDS for w in parts):
                continue
            push(tok)
    if len(uncovered) == 1:  # a single bare name is the case a class word disambiguates
        for cls in classes[:1]:
            push(cls)
    out = " ".join(words[:14]).strip()
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

        # HOP 1
        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]
        facts_1 = _ledger_lines(
            self.extract1(ctx.llm, tag="extract1", question=question, passages=hop1_docs)["facts"]
        )
        ledger_1 = "\n".join(facts_1)

        # HOP 2 -- no planning call: the query is a hedge over every name still unaccounted for
        audit_lines, uncovered = _coverage_audit(question, titles_1)
        covered = [ln.split(" -- ")[0] for ln in audit_lines if "NOT RETRIEVED" not in ln]
        query_2 = _hedged_query(question, uncovered, covered)

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
        ctx.note(
            "hop2_query",
            coverage=audit_lines,
            uncovered=uncovered,
            query=query_2,
            comparison_kind=kind or "general",
        )

        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "uncovered": uncovered,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "uncovered": uncovered,
            "query_2": query_2,
            "evidence": evidence,
        }
