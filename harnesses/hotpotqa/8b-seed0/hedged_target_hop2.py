"""Candidate: the one remaining search hedges across two candidate targets instead of betting on one.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Extraction and the answering contract
are inherited from `shared_property_alignment` verbatim. The one mechanism under test is what the second
and last search asks for.

The parent resolves the missing article to a single name and spends its whole remaining retrieval budget
on it. Measured on the search set, the second search fails to bring back the missing article on a quarter
of items, and the failures are not random: the planner frequently *guesses the answer entity* from a
partial first hop, searches for the guess, gets seven passages about the guess, and the answering call
then confirms it -- a wrong name in the query becomes a wrong answer with confident-looking support.

Because the index scores a query as a bag of words, one query can carry two names at once and return
passages for both. So here the planner is asked for two names for the same missing slot -- the one it
believes and the best rival the first hop leaves open -- and code packs both into the single query. The
answering call is then told which two articles the search was hedged across and that exactly one of them
lies on the chain the question describes, so it has to decide between them against the question's stated
constraints rather than accept whatever the query fetched. When the first hop already names the missing
article outright, the planner leaves the rival empty and the search is the parent's.
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

Then set `rival_target`. The first hop is incomplete, so `missing_target` is often a guess -- and when
it is the name of the very thing the question asks for, a search for it returns passages that agree with
the guess whether or not it is right. So name the strongest *different* candidate for the same slot: the
other entity that everything you have read so far would also allow to fill it. Take it from the first
hop's own text where you can (another name the passages associate with the question's subject), or from
what you know about the subject otherwise. It must be a different proper name of the same kind as
`missing_target`, never a description or a rewording of the same name.
Write `none` for `rival_target` only when the evidence you already have *names* the missing article
outright, so that there is nothing left to guess.

Then set `keywords`: one or two words that would disambiguate a name of this kind in an index of article
titles -- a type word (film, band, river, novel, album, footballer) or a year -- or leave it empty when
the names are already unambiguous. Never write a question or a sentence, no question words (who, what,
when, where, which, how), no words describing facts you already have, and no name that is already in
`titles_found` unless it is part of a target's own name. Bare words only."""

HEDGE_CLAUSE = """`link_candidates` names the two articles the second search was hedged across, because
the first hop did not settle which one the question's chain passes through. At most one of them is on
that chain. Before answering, decide which: the one whose own article satisfies every constraint the
question states about that step -- the role, the year, the work, the place, the relation. Then answer
from that article's lines and treat the other candidate's lines as decoys, however well they match the
question's other words. If neither candidate satisfies the constraints, ignore both and answer from the
lines that do. Never hand back a candidate's name merely because the search returned passages about it.
"""

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


_NONE_PAT = re.compile(r"^\s*(none|n/?a|nothing|unknown|no rival.*|-+)\s*$", re.IGNORECASE)


def _bare_name(raw: str) -> str:
    """First line, unquoted, no leading label: a name the index can match against titles."""
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`\"'\u201c\u201d").strip()
    text = re.sub(r"^(missing_target|rival_target|target|article|name)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:[^()]{0,40})\)\s*$", "", text).strip()  # drop a trailing parenthetical gloss
    return text.strip(" ,;:.")[:60]


def _name_tokens(name: str) -> list[str]:
    return [w for w in re.findall(r"[\w'\-\.]+", name or "") if w.lower() not in _TRIVIAL]


def _same_name(a: str, b: str) -> bool:
    ta, tb = {w.lower() for w in _name_tokens(a)}, {w.lower() for w in _name_tokens(b)}
    if not ta or not tb:
        return False
    return ta == tb or ta <= tb or tb <= ta


def _hedged_query(target: str, rival: str, extra: str) -> str:
    """One bag of words carrying both candidate names, so the single search covers both."""
    words: list[str] = []
    seen: set[str] = set()
    for source in (target, rival, extra):
        for word in _name_tokens(source):
            key = word.lower().strip(".")
            if key and key not in seen:
                seen.add(key)
                words.append(word)
        if len(words) >= 10:
            break
    return " ".join(words[:10])


def _candidate_note(target: str, rival: str) -> str:
    if target and rival:
        return f"A: {target}\nB: {rival}"
    if target:
        return f"A: {target}\nB: (none -- the first hop named this article outright)"
    return "A: (unnamed)\nB: (none)"


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
            outputs=["missing_target", "rival_target", "keywords"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "missing_target": "name of the one article still missing",
                "rival_target": "the strongest different candidate for the same slot, or `none`",
                "keywords": "0-2 bare disambiguating words (a type word or a year), or empty",
            },
        )
        self.extract2 = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts"],
            instructions=EXTRACT2_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per new relevant fact, copied verbatim"},
        )
        self.generate_answer = Module(
            inputs=["question", "link_candidates", "evidence"],
            outputs=["chain_article", "answer_type", "supporting_quote", "answer"],
            instructions=HEDGE_CLAUSE + "\n" + ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "link_candidates": "the two articles the second search was hedged across",
                "chain_article": "which candidate the question's chain passes through, or `neither`",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_pair = Module(
            inputs=["question", "comparison_kind", "link_candidates", "evidence"],
            outputs=["chain_article", "entity_a", "fact_a", "entity_b", "fact_b", "decision", "answer"],
            instructions=HEDGE_CLAUSE + "\n" + PAIR_ANSWER_INSTRUCTIONS,
            descs={
                "comparison_kind": "`choice` (pick one of the two) or `shared` (what they have in common)",
                "link_candidates": "the two articles the second search was hedged across",
                "chain_article": "which candidate the evidence about the two things actually comes from",
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
        target = _bare_name(plan.get("missing_target", ""))
        rival = _bare_name(plan.get("rival_target", ""))
        extra = _keyword_query(plan.get("keywords", ""), fallback="")
        if _same_name(rival, target) or _NONE_PAT.match(rival or ""):
            rival = ""
        query_2 = _hedged_query(target, rival, extra) or _keyword_query(target, fallback=question)

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
        candidates = _candidate_note(target, rival)
        ctx.note("hop2_plan", missing_target=target, rival_target=rival, query=query_2,
                 comparison_kind=kind or "general")
        if kind:
            out = self.answer_pair(
                ctx.llm, tag="answer", question=question, comparison_kind=kind,
                link_candidates=candidates, evidence=evidence
            )
            answer = _tidy(out.get("answer", ""))
            return {
                "answer": answer,
                "comparison_kind": kind,
                "decision": out.get("decision", ""),
                "missing_target": target,
                "rival_target": rival,
                "chain_article": out.get("chain_article", ""),
                "query_2": query_2,
                "evidence": evidence,
            }

        out = self.generate_answer(ctx.llm, tag="answer", question=question,
                                   link_candidates=candidates, evidence=evidence)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "comparison_kind": "general",
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "rival_target": rival,
            "chain_article": out.get("chain_article", ""),
            "query_2": query_2,
            "evidence": evidence,
        }
