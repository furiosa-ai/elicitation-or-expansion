"""Candidate: a computed candidate set may only constrain the answer when the code can see both sides.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval, the hop-2 planner and the
extractors are inherited verbatim; the one mechanism under test is a coverage precondition on the
computed shared-wording channel.

The parent computes, from the retrieved articles, the phrases that occur word for word in the own
articles of the subjects a question names, and requires the answer to a shared-property question to come
out of that computed set. Measured against the system it was built from, that constraint is the best
answer policy yet seen on items where both subjects' articles were actually retrieved (168 of 198) and
the worst on items where only one of them was (38 of 102). The reason is structural, not a matter of
wording: an intersection taken over one article is not an intersection at all -- it degenerates into the
phrases of whichever article came back, and the ban on substituting your own wording then forces the
answer to be lifted from a single article that cannot settle a two-sided question.

So the channel here is gated on a precondition the code can check without the model's help: each of the
two named subjects must resolve to its OWN retrieved article (a retrieved title that matches that
subject's name, and a different one for the other subject). When both resolve, the computed wording is
handed over and the alignment contract binds the answer to it. When only one resolves -- or the question
is not a two-sided one at all -- no computed set is shown and the contract falls back to open
generation, which is what a system with half the evidence needs in order to commit at all.
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
  common to both, name it. Otherwise give the wording that covers both rows, taken from
  `shared_wording` -- see the rule on it below.

Then set `answer`:
- if the question asks whether something holds rather than which or what, write exactly `yes` or `no`.
- `choice`: write one of `entity_a` or `entity_b`, spelled as the question spells it. Never the date,
  number or property that settled the comparison, never both names, never a sentence.
- `shared`: write only what the two have in common -- one short noun phrase, singular unless the
  question asks about several things at once. No `and`, no pairing of two wordings, no "both are", no
  explanation. When the question asks which *person*, *work* or *place* they have in common ("which
  actor was in both A and B"), what they share is that entity, so write its name; otherwise write the
  class or property itself and no entity name.

`shared_wording` is the list of phrases that occur, word for word, in BOTH entities' own articles. It
was computed from the articles, not written by you, so it is the vocabulary the two subjects genuinely
agree on. For a `shared` question whose answer is a class or property rather than a named entity:
- pick your answer from this list. Copy the listed phrase exactly, including its number (singular or
  plural) and its capitalisation.
- prefer the most specific listed phrase that is still true of both and answers what was asked; among
  phrases that say the same thing, take the shortest.
- a word of your own that means the same as a listed phrase is wrong -- if the list offers `writer`,
  answer `writer` and not a synonym of it; if it offers a country's formal name, do not shorten it.
- the list is built by literal overlap, so it also contains phrases that answer nothing. Ignore those.
  If no listed phrase answers the question, and only then, answer in your own words.
- when the list is empty this rule does not apply.
- Never restate the question, never add "the answer is", never explain why.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no gloss in brackets.
- The evidence is often incomplete. Commit to the single most likely answer anyway, from what you know
  about the entities if need be. Never answer "not specified", "unknown", "neither", "not related" or
  "cannot be determined", and never mention the evidence or its gaps in the `answer` field."""

GENERAL_VOCAB_RULE = """

`shared_wording` lists phrases that occur, word for word, in the own articles of two or more of the
subjects this question names. It was computed from the articles, not written by you. It matters in one
case only: when what the question asks for is a class, property, relation or category rather than a
named thing -- the country two works come from, the profession two people share, the kind of thing two
places both are. In that case answer with the listed phrase, copied exactly, rather than a word of your
own that means the same: a grader matches your answer against the source's word, so a formal name must
not be shortened to its common one and a class word must not be swapped for a synonym.
When the question asks for a name, a date or a number, or when no listed phrase answers what was asked,
ignore this field entirely and follow the rules above. The list is built by literal overlap and contains
phrases that answer nothing; it is never a reason to change an answer you can support from a quote."""


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


PAIR_OPEN_INSTRUCTIONS = """The question puts two named things side by side. Line them up one at a
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

GENERAL_ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
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


_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_VOCAB_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "as", "by", "from", "with",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "it", "its", "he", "she",
    "his", "her", "they", "them", "their", "who", "which", "that", "this", "these", "those", "also",
    "than", "then", "there", "when", "where", "while", "into", "over", "under", "between", "both",
    "all", "any", "some", "more", "most", "other", "another", "such", "only", "own", "so", "not",
    "known", "best", "one", "two", "first", "second", "new", "may", "can", "after", "before", "about",
    "s", "t", "d", "ll", "born", "died", "years", "year", "later", "early", "since", "until", "during",
}


def _content_ngrams(text: str, nmax: int = 3) -> dict[str, int]:
    """Every 1..3-word content phrase in an abstract, mapped to where it first occurs."""
    out: dict[str, int] = {}
    for sent_start, sentence in _sentences_with_offsets(text):
        toks = [(m.group(0), m.start()) for m in _WORD.finditer(sentence)]
        for i in range(len(toks)):
            for n in range(1, nmax + 1):
                if i + n > len(toks):
                    break
                chunk = toks[i:i + n]
                words = [w for w, _ in chunk]
                if words[0].lower() in _VOCAB_STOP or words[-1].lower() in _VOCAB_STOP:
                    continue
                if any(len(w) < 2 for w in words):
                    continue
                phrase = " ".join(words)
                key = phrase.lower()
                if key not in out:
                    out[key] = sent_start + chunk[0][1]
    return out


def _sentences_with_offsets(text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+", text or ""):
        spans.append((start, text[start:m.start()]))
        start = m.end()
    if start < len(text or ""):
        spans.append((start, text[start:]))
    return spans


def _title_key(title: str) -> str:
    base = re.sub(r"\s*\([^)]*\)\s*$", "", title or "")
    return re.sub(r"[^a-z0-9]+", " ", base.lower()).strip()


def _article_for(span: str, docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The retrieved article that span names: exact title match first, then best token overlap."""
    want = set(_title_key(span).split())
    if not want:
        return None
    exact = [d for d in docs if _title_key(d.get("title", "")) == _title_key(span)]
    if exact:
        return exact[0]
    best, best_score = None, 0.0
    for d in docs:
        have = set(_title_key(d.get("title", "")).split())
        if not have:
            continue
        inter = len(want & have)
        if not inter:
            continue
        score = inter / max(len(want | have), 1)
        if score > best_score:
            best, best_score = d, score
    return best if best_score >= 0.5 else None


def _both_sides_covered(question: str, docs: list[dict[str, Any]]) -> bool:
    """True when each of the two named subjects resolves to its OWN, distinct retrieved article."""
    spans = _nameish_spans(question)
    resolved: list[str] = []
    for span in spans:
        art = _article_for(span, docs)
        if art is None:
            continue
        key = _title_key(art["title"])
        if key not in resolved:
            resolved.append(key)
        if len(resolved) >= 2:
            return True
    return False


def _shared_wording(question: str, docs: list[dict[str, Any]], limit: int = 24) -> list[str]:
    """Phrases occurring verbatim in the own articles of two or more subjects named in the question."""
    spans = _nameish_spans(question)
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for span in spans:
        art = _article_for(span, docs)
        if art is None:
            continue
        key = _title_key(art.get("title", ""))
        if key in used:
            continue
        used.add(key)
        picked.append(art)
    if len(picked) < 2:
        return []
    maps = [_content_ngrams(a.get("text", "")) for a in picked]
    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)
    if not common:
        return []
    # a phrase that is part of a longer shared phrase is dropped in favour of the longer one
    kept = [c for c in common if not any(c != o and (c + " ") in (o + " ") and c in o for o in common)]
    title_words: set[str] = set()
    for a in picked:
        title_words |= set(_title_key(a.get("title", "")).split())

    def rank(phrase: str) -> tuple:
        words = phrase.split()
        earliest = min(m[phrase] for m in maps)
        in_title = 1 if all(w in title_words for w in words) else 0
        return (in_title, earliest, -len(words), phrase)

    kept.sort(key=rank)
    surface: list[str] = []
    seen: set[str] = set()
    for phrase in kept[:limit]:
        best = phrase
        for a in picked:
            m = re.search(r"\b" + re.escape(phrase) + r"\b", a.get("text", ""), re.IGNORECASE)
            if m:
                best = m.group(0)
                break
        if best.lower() not in seen:
            seen.add(best.lower())
            surface.append(best)
    return surface


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
            inputs=["question", "evidence"],
            outputs=["answer_type", "supporting_quote", "answer"],
            instructions=GENERAL_ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "evidence", "shared_wording"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "shared_wording": "phrases found word for word in both subjects' own articles (computed, not written by you)",
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "entity_a": "the first of the two things, as the question names it",
                "fact_a": "what its own article says on the point the question turns on",
                "entity_b": "the second of the two things, as the question names it",
                "fact_b": "what its own article says on the point the question turns on",
                "decision": "one sentence: the winner, or the wording that covers both rows",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair_open = Module(
            inputs=["question", "comparison_kind", "evidence"],
            outputs=["entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=PAIR_OPEN_INSTRUCTIONS,
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

        # HOP 2 -- named coverage gap, then a title-style keyword query for it
        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1, evidence=facts_1 or hop1_docs[:3]
        )
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))

        hop2 = ctx.tools.retrieve_docs(query_2, k=self.K)
        hop2_docs = [d["text"] for d in hop2]
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
        docs_all = list(hop1) + list(hop2)
        both_sides = _both_sides_covered(question, docs_all)
        vocab = _shared_wording(question, docs_all) if (kind and both_sides) else []
        ctx.note(
            "hop2_plan", missing_target=target, query=query_2, comparison_kind=kind or "general",
            both_sides_covered=both_sides, shared_wording=vocab,
        )
        if kind and both_sides and vocab:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence,
                shared_wording=vocab,
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "gate": "computed",
                "decision": out.get("decision", ""),
                "shared_wording": vocab,
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }
        if kind:
            out = self.answer_pair_open(
                ctx.llm, tag="answer", question=question, comparison_kind=kind, evidence=evidence,
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "gate": "open",
                "decision": out.get("decision", ""),
                "missing_target": target,
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "gate": "general",
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
