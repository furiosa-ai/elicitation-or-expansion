"""Candidate: the final call's answer contract is selected in code from the question's own asked-for slot.

Interaction depth is unchanged from the parent (`gap_directed_hop2`): 4 LLM calls, 2 retrievals, k=7.
Retrieval, both extraction calls and the gap planner are byte-identical to the parent. The single
mechanism under test is *which* answer contract the last call runs under.

The parent asks one generic answering module to first label the answer with a fixed enum
(`yes_no, person, organisation, work, place, date, number, other`). Six of those eight labels are
named-entity kinds, so the contract quietly presupposes that the answer is a name, and the module's
job is reduced to picking which name. That presupposition is wrong for a whole class of questions
whose answer is a common noun, a shared category, a relation word or a counted amount -- for those the
module reliably hands back the most salient proper noun in the evidence instead of the word the
question asked for, or hedges.

Here a deterministic parser reads the question's interrogative phrase and routes the last call to one
of four *separate* modules, each with its own output fields and its own notion of what a well-formed
answer is:

  VERDICT   polar question            -> a bare yes/no, decided by comparing the two subjects
  CATEGORY  "what type/kind of X", "what do A and B have in common", "how are A and B related"
                                      -> the common noun the source sentences themselves use, with
                                         proper names explicitly ruled out
  AMOUNT    "how many/much/long/old"  -> a bare quantity, with the unit kept only if the source
                                         attaches one, and the source's own numeral style preserved
  NAME      everything else           -> the parent's contract (named entity / date / choice)

The router is code, not a model judgement: the parent already showed that a model asked to classify
its own situation mislabels exactly the items that needed the other branch. Routing on the question's
surface form cannot fail that way, and it fires on a narrow, explicit pattern set, so the default
branch keeps the parent's behaviour on the large majority of items.
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

# ---------------------------------------------------------------------------
# The four answer contracts. Only the branch the router picks is ever called.
# ---------------------------------------------------------------------------

NAME_INSTRUCTIONS = """Answer the question from the evidence lines, then state the answer in the exact
form a grader can match against a short reference answer.

`asked_for` is the question's own interrogative phrase, copied out for you. The answer must be a thing
of exactly that kind -- nothing else will be accepted.

First set `supporting_quote` to the one evidence line that contains the answer, copied exactly.

Then set `answer` under these rules:
- Write the shortest span that answers the question, copied from the supporting quote in the exact
  surface form used there. A name, a title, a date, or a short noun phrase.
- Never write a sentence, never restate the question, never add "the answer is", never explain why.
- Give exactly what was asked for and nothing more: no apposition, no extra qualifier, no country or
  state appended to a place name, no gloss in brackets.
- When the question offers a choice between two named alternatives ("which was released first, A or
  B?", "who is older, A or B?", "which one has more members, A or B?"), the answer is one of those two
  names, written as the question writes it -- never the date, number or property that settled the
  comparison, and never both names.
- Otherwise answer the kind of thing `asked_for` names. A question about who performed, directed,
  wrote or founded something wants the name of the person, even when the evidence only names the group
  or company responsible; a question about where wants the place; about when, the date. Never hand back
  an entity of the wrong kind just because the evidence names it -- the group is not the singer, the
  album is not the song, the studio is not the director.
- The evidence is often incomplete. Commit to the single most likely answer anyway. If the evidence
  identifies the subject but not the asked-for property, answer from what you know about that subject,
  in the same short form. Never answer "not specified", "unknown", "cannot be determined", and never
  mention the evidence or its gaps in the `answer` field."""

VERDICT_INSTRUCTIONS = """This question asks whether something holds. It is answered with a bare verdict.

First set `subjects` to the two things being asked about, named as the question names them.
Then set `test` to the one property the question makes the verdict depend on (same nationality, both
composers, from the same country, ...).
Then set `finding` to what the evidence says about that property for each subject, one short clause
each, quoting the words the evidence uses.

Then set `answer` to exactly `yes` or `no` -- one word, lower case, nothing else. No explanation, no
restatement, no hedging, no punctuation.
- `yes` only if the evidence supports the property for both subjects; otherwise `no`.
- The evidence is often incomplete. Decide anyway, from what you know about the subjects if you must.
  Never write "unclear", "not specified" or a sentence in the `answer` field."""

CATEGORY_INSTRUCTIONS = """This question asks what KIND of thing something is, or what category or
relation two things share. The answer is a common noun -- a word for a class of things -- not the name
of any particular thing.

`asked_for` is the question's own interrogative phrase, copied out for you.

First set `defining_sentences` to the sentences that say what each subject IS -- the "X is a ..." /
"X was an ..." clause for each thing the question mentions -- copied from the evidence exactly. One
per line. These are usually the opening words of an article and read like background; they are exactly
what this question needs.

Then set `shared_class` to the most specific class word or phrase that the defining sentences apply to
every subject the question asks about. Take the word from those sentences; do not invent a fancier one.
If the sentences use different words for the same idea, choose the one that fits all subjects.

Then set `answer` to that class word, under these rules:
- The answer is a common noun or a short common-noun phrase, normally lower case: the word for the
  category, type, medium, profession, nationality, material or relation being asked about.
- It must NOT name a particular thing. Never answer with the name of one person, one work, one film,
  one album, one band, one company or one place, and never with the title of a retrieved article --
  those are instances of the class, not the class itself. Class words that are conventionally
  capitalised (a nationality, a language, a named genre or style) keep their capitalisation.
- Copy the word the defining sentences use, in the grammatical number the question implies, and give
  the bare class: no article, no "type of", no restatement of the subjects, no explanation, no
  parenthetical, no second alternative joined by "and" or "or".
- If the two defining sentences use the same word, that word is the answer.
- The evidence is often incomplete. Commit to the single most likely class anyway, from what you know
  about the subjects if you must. Never answer "not related", "none", "not specified", "unknown", and
  never say the subjects are unrelated -- the question presupposes that they share this property, so
  find the class they share."""

AMOUNT_INSTRUCTIONS = """This question asks for an amount -- a count, a size, a duration, an age or a
distance. The answer is the quantity itself.

`asked_for` is the question's own interrogative phrase, copied out for you.

First set `supporting_quote` to the one evidence line that states the amount, copied exactly.
Then set `arithmetic` to the one-line reasoning that gets from that quote to the amount asked for
(often none is needed; if the quote states the amount directly, say so and repeat it).

Then set `answer` to the amount, under these rules:
- Write the quantity and nothing else: no subject, no verb, no sentence, no restatement of the
  question, no "approximately", no "about", no explanation.
- Keep the numeral exactly as the supporting quote writes it -- if the quote spells the number out as
  a word, spell it out; if the quote uses digits, use digits, with the quote's own separators.
- Keep the unit or the counted noun only when the quote attaches it to the number, and then copy it in
  the quote's form and abbreviation; do not add a unit the quote does not use, and do not drop one it
  does.
- If the question asks how many of something there are and the evidence lists them rather than
  counting them, count them and give the number.
- The evidence is often incomplete. Commit to the single most likely amount anyway, from what you know
  about the subject if you must. Never answer "not specified", "unknown" or "cannot be determined"."""

# ---------------------------------------------------------------------------
# Router: deterministic, on the question's surface form only.
# ---------------------------------------------------------------------------

_POLAR = re.compile(
    r"^\s*(?:are|is|was|were|do|does|did|can|could|has|have|had|will|would)\b", re.IGNORECASE
)

_CATEGORY = re.compile(
    r"""
      \bwh(?:at|ich)\s+(?:type|types|kind|kinds|sort|sorts|genre|genres|category|categories
                        |class|classes|species|breed|breeds|form|forms|medium|media)\s+of\b
    | \bwh(?:at|ich)\s+(?:type|kind|sort|genre|category|class|species)\b(?!\s+of\b)
    | \bhave\s+in\s+common\b
    | \bin\s+common\b
    | \bhow\s+(?:is|are|was|were)\b[^?]{0,70}\brelated\b
    | \bwh(?:at|ich)\s+(?:are|is|was|were)\s+both\b
    | \bwh(?:at|ich)\s+(?:are|is|was|were)\b[^?]{0,70}\bknown\s+for\b
    | \bwh(?:at|ich)\s+(?:profession|professions|occupation|occupations|nationality|nationalities
                        |language|languages|colour|color|material|shape|religion|ethnicity)\b
    | \bboth\b[^?]{0,70}\bwh(?:at|ich)\s+(?:type|kind|profession|occupation|nationality)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_AMOUNT = re.compile(r"\bhow\s+(?:many|much|long|old|tall|far|high|deep|wide|large)\b", re.IGNORECASE)

_WH = re.compile(r"\b(wh(?:at|ich|ose|om|o|en|ere|y)|how)\b", re.IGNORECASE)

_ASKED_STOP = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can",
               "could", "will", "would", "the", "a", "an", "his", "her", "their", "its", "this",
               "that", "these", "those", "in", "on", "at", "to", "for", "and", "or", "both",
               "there", "he", "she", "it", "they", "you", "with", "by", "from", "between"}


def _asked_for(question: str) -> str:
    """The question's own interrogative phrase, so the answer call is told the slot in its own words.

    The head noun after the wh-word is what actually names the wanted kind ("what TYPE OF ANIMAL",
    "how MANY STAFF MEMBERS", "which DATE"), so walk forward from the wh-word and stop at the first
    function word, which is where the slot description ends and the question's body begins.
    """
    text = (question or "").strip()
    m = _WH.search(text)
    if not m:
        return text[:80]
    head = m.group(1).lower()
    words: list[str] = []
    for raw in text[m.end():].split():
        w = raw.strip(",;:.?!\"'()").strip()
        if not w or w.lower() in _ASKED_STOP:
            break
        words.append(w)
        if len(words) >= 4:
            break
    return " ".join([head] + words).strip()


def _route(question: str) -> str:
    q = (question or "").strip()
    if _POLAR.match(q) and not _CATEGORY.search(q) and not _AMOUNT.search(q):
        return "VERDICT"
    if _CATEGORY.search(q):
        return "CATEGORY"
    if _AMOUNT.search(q):
        return "AMOUNT"
    return "NAME"


_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}


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


def _bare_verdict(answer: str) -> str:
    """The verdict branch must emit a literal yes/no; recover one if the module wrote a clause."""
    text = _tidy(answer)
    low = text.lower().strip().strip(".!")
    if low in ("yes", "no"):
        return low
    m = re.search(r"\b(yes|no)\b", low)
    if m:
        return m.group(1)
    if re.search(r"\b(not|neither|differ|different|isn't|aren't|wasn't|weren't|don't|doesn't|didn't)\b", low):
        return "no"
    return text


def _bare_class(answer: str) -> str:
    """Strip the framing a class word tends to arrive wrapped in."""
    text = _tidy(answer)
    text = re.sub(r"^(?:they\s+are|both\s+are|it\s+is|they\s+were)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:type|kind|sort|category|class|genre)\s+of\s+", "", text, flags=re.IGNORECASE)
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
        _evidence_desc = "sentences copied verbatim from source articles, each prefixed by its title"
        self.answer_name = Module(
            inputs=["question", "asked_for", "evidence"],
            outputs=["supporting_quote", "answer"],
            instructions=NAME_INSTRUCTIONS,
            descs={
                "asked_for": "the question's own interrogative phrase -- the kind of thing wanted",
                "evidence": _evidence_desc,
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the short answer span only -- never a sentence",
            },
        )
        self.answer_verdict = Module(
            inputs=["question", "evidence"],
            outputs=["subjects", "test", "finding", "answer"],
            instructions=VERDICT_INSTRUCTIONS,
            descs={
                "evidence": _evidence_desc,
                "subjects": "the two things the question compares",
                "test": "the property the verdict depends on",
                "finding": "what the evidence says about that property for each subject",
                "answer": "exactly `yes` or `no`",
            },
        )
        self.answer_category = Module(
            inputs=["question", "asked_for", "evidence"],
            outputs=["defining_sentences", "shared_class", "answer"],
            instructions=CATEGORY_INSTRUCTIONS,
            descs={
                "asked_for": "the question's own interrogative phrase -- the kind of thing wanted",
                "evidence": _evidence_desc,
                "defining_sentences": "the `X is a ...` clause for each subject, copied exactly, one per line",
                "shared_class": "the most specific class word the defining sentences share",
                "answer": "the bare common noun for that class -- never a proper name",
            },
        )
        self.answer_amount = Module(
            inputs=["question", "asked_for", "evidence"],
            outputs=["supporting_quote", "arithmetic", "answer"],
            instructions=AMOUNT_INSTRUCTIONS,
            descs={
                "asked_for": "the question's own interrogative phrase -- the amount wanted",
                "evidence": _evidence_desc,
                "supporting_quote": "the evidence line stating the amount, copied exactly",
                "arithmetic": "one line from the quote to the amount asked for",
                "answer": "the quantity only -- never a sentence",
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

        # ANSWER -- the routed contract is the mechanism under test
        slot = _route(question)
        asked = _asked_for(question)
        ctx.note("answer_slot", slot=slot, asked_for=asked, missing_target=target, query=query_2)

        if slot == "VERDICT":
            out = self.answer_verdict(ctx.llm, tag="answer", question=question, evidence=evidence)
            answer = _bare_verdict(out.get("answer", ""))
        elif slot == "CATEGORY":
            out = self.answer_category(
                ctx.llm, tag="answer", question=question, asked_for=asked, evidence=evidence
            )
            answer = _bare_class(out.get("answer", ""))
        elif slot == "AMOUNT":
            out = self.answer_amount(
                ctx.llm, tag="answer", question=question, asked_for=asked, evidence=evidence
            )
            answer = _tidy(out.get("answer", ""))
        else:
            out = self.answer_name(
                ctx.llm, tag="answer", question=question, asked_for=asked, evidence=evidence
            )
            answer = _tidy(out.get("answer", ""))

        return {
            "answer": answer,
            "slot": slot,
            "asked_for": asked,
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
