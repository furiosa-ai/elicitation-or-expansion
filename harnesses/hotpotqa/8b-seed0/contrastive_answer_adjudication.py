"""Candidate: the last call adjudicates between explicitly enumerated answer options.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Retrieval is the proven
gap-directed second hop; the one mechanism under test is the ANSWER POLICY.

Diagnosis this is built on: on the parent system, 42 of its 94 search-set errors carry the gold
answer verbatim inside the evidence lines the answering call already had. The single free-form
`answer` field loses them in three recurring ways -- it hands back the shorthand form of a name
while the source article introduces it in a fuller form, it returns the property that settled a
comparison instead of the named alternative asked for, and it returns an entity when the question
asks for a word the sentence attaches to that entity.

So the answer is no longer written from scratch. The third call drafts an answer together with the
sentence it comes from; the harness then derives, in code, a small contrastive option set from that
draft (the longer verbatim name spans the evidence actually contains, the two alternatives an
"A or B" question offers, yes/no when the question is a polar one); and the fourth call is a
selection task: decide what kind of thing the question asks for, then pick the one option that
matches, or overrule the whole set with a span copied from the evidence. The chosen string is
snapped back onto the option's exact surface form.
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

DRAFT_INSTRUCTIONS = """Extend an evidence file with the new passages, then draft an answer.

`context` holds the lines gathered from the first search. First set `facts`: go through the new
passages and copy out every sentence that adds a fact the question depends on, one line per fact:

Title :: the sentence copied exactly as it appears in the passage

- Copy verbatim: never paraphrase, never merge two sentences, never complete anything. Keep names,
  spellings, dates, numbers, units and abbreviations exactly as the passage writes them.
- Do not repeat a line `context` already contains, skip passages that say nothing about the question,
  and never invent a line. At most 8 lines.

Then set `asked_for`: name the kind of thing the question asks for in three or four words -- for
example "the name of the person", "the year", "the number of episodes", "yes or no", "which of the
two named films". Be strict about the kind: a question about who performed, directed, wrote or
founded something asks for a person even when the evidence only names a group or company.

Then set `answer_sentence`: the ONE evidence line (from `context` or from `facts`) that contains the
answer, copied exactly. If no line contains it, copy the line that comes closest.

Then set `draft_answer`: the shortest span of `answer_sentence` that answers the question, copied in
that sentence's exact surface form -- a name, a title, a date, a number, or a short noun phrase.
Write exactly `yes` or `no` if the question is a polar one. Never write a sentence, never restate the
question, never explain. If the evidence is incomplete, commit to the most likely answer anyway; never
write "unknown" or "not specified"."""

ADJUDICATE_INSTRUCTIONS = """Settle the final answer. A draft answer has been proposed and `options`
lists the answer strings worth considering, all of them taken from the evidence or from the question.

Work in this order.
1. Set `asked_for_check`: state in a few words what kind of thing the question asks for, and whether
   the draft is that kind of thing. A group is not a person, an album is not a song, a studio is not a
   director, a date is not a film, and a property is not a name.
2. Set `answer` to the string that a grader would accept as the short reference answer:
   - If the question is a polar one (are/is/was/were/do/does/did/can/has/have/both ... ?), write
     exactly `yes` or `no` and nothing else.
   - If the question offers a choice between two named alternatives ("which came first, A or B?",
     "who is older, A or B?"), the answer is one of those two names -- never the date, number or
     property that settled the comparison, and never both.
   - Otherwise choose the option that answers the question. Between two options naming the SAME thing,
     prefer the form the entity's own article uses when it introduces it -- articles open with the
     full form of a name, and that is the form to give -- but drop a leading honorific (Sir, Dr, Mr)
     and drop a trailing role, rank or title that follows a comma. Keep a leading year or ordinal when
     it is part of the name of an event.
   - Keep numbers and dates exactly as the evidence writes them, with the unit the evidence attaches
     and no unit it does not attach.
   - If every option is the wrong kind of thing, ignore them all and copy the correct short span
     straight out of the evidence instead.
3. `answer` holds the answer span alone: no sentence, no restatement of the question, no apposition,
   no added country or state, no bracketed gloss, no explanation. Commit to one answer even when the
   evidence is thin -- never write "unknown", "not specified" or "cannot be determined"."""

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}
_POLAR = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|both)\b", re.IGNORECASE)
_CHOICE = re.compile(r"[,:]\s*([^,:?]{2,60}?)\s+or\s+([^,:?]{2,60}?)\s*\??\s*$")
_HONORIFIC = re.compile(r"^(sir|dame|dr|mr|mrs|ms|lord|lady|saint|st)\.?\s+", re.IGNORECASE)
_NAMEY = re.compile(r"^[\"'“‘(]?[A-Z0-9]")
MIDDLE_PARTS = r"(?:\s+[A-Z][\w'’.-]*,?){0,4}\s+"


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


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _expansions(draft: str, lines: list[str]) -> list[str]:
    """Longer verbatim spans of the evidence that contain `draft` as a whole sub-phrase.

    Names are the case that matters: the question (and the other article) tends to use a short form
    while the entity's own article opens with the full one, and the reference answer is a span of the
    source. Expansion walks outward over name-like tokens only, so it never swallows a clause.
    """
    out: list[str] = []
    d = draft.strip()
    if not d or len(d) > 60 or len(d.split()) > 6:
        return out
    pats = [re.compile(re.escape(d), re.IGNORECASE)]
    dtoks = d.split()
    if len(dtoks) >= 2 and _NAMEY.match(dtoks[0]) and _NAMEY.match(dtoks[-1]):
        # the source may write the same name with extra middle parts: first ... last
        pats.append(re.compile(re.escape(dtoks[0]) + MIDDLE_PARTS + re.escape(dtoks[-1])))
    for line in lines:
        body = line.split("::", 1)[1] if "::" in line[:80] else line
        for m in [mm for pat in pats for mm in pat.finditer(body)]:
            left, right = body[:m.start()], body[m.end():]
            span = body[m.start():m.end()]
            # walk left over adjacent capitalised / numeric tokens
            ltoks = left.split()
            take = 0
            while take < 4 and len(ltoks) > take:
                tok = ltoks[len(ltoks) - 1 - take]
                if tok.endswith((".", ";", ":")) or not _NAMEY.match(tok):
                    break
                take += 1
            if take:
                span = " ".join(ltoks[len(ltoks) - take:]) + " " + span
            # walk right over adjacent capitalised tokens
            rtoks = right.split()
            take_r = 0
            while take_r < 3 and len(rtoks) > take_r:
                tok = rtoks[take_r]
                if not _NAMEY.match(tok):
                    break
                take_r += 1
                if tok.endswith((".", ",", ";", ":", ")")):
                    break
            if take_r:
                span = span + " " + " ".join(rtoks[:take_r])
            for cand in (span, _HONORIFIC.sub("", span)):
                cand = cand.strip().strip(",;:—-").strip()
                cand = re.sub(r"\s*\([^)]*$", "", cand).strip()
                ck, dk = _key(cand), _key(d)
                same_ends = dk.split() and ck.split() and ck.split()[0] == dk.split()[0] and ck.split()[-1] == dk.split()[-1]
                if cand and ck != dk and len(cand) < 90 and (dk in ck or same_ends):
                    if all(_key(cand) != _key(o) for o in out):
                        out.append(cand)
    return out[:4]


def _option_set(question: str, draft: str, lines: list[str]) -> list[str]:
    opts: list[str] = []

    def add(value: str) -> None:
        value = (value or "").strip()
        if value and len(value) < 120 and all(_key(value) != _key(o) for o in opts):
            opts.append(value)

    add(draft)
    for exp in _expansions(draft, lines):
        add(exp)
    choice = _CHOICE.search(question.strip())
    if choice:
        for side in choice.groups():
            side = side.strip().strip(" \"'“”")
            while len(side.split()) > 1 and not _NAMEY.match(side.split()[0]):
                side = side.split(" ", 1)[1].strip()   # drop a descriptor before the name
            if side and len(side.split()) <= 8:
                add(side)
    if _POLAR.match(question.strip()):
        add("yes")
        add("no")
    return opts[:8]


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
        self.draft = Module(
            inputs=["question", "context", "passages"],
            outputs=["facts", "asked_for", "answer_sentence", "draft_answer"],
            instructions=DRAFT_INSTRUCTIONS,
            descs={
                "context": "evidence lines already gathered",
                "facts": "one `Title :: sentence` line per new relevant fact, copied verbatim",
                "asked_for": "the kind of thing the question asks for, in a few words",
                "answer_sentence": "the single evidence line containing the answer, copied exactly",
                "draft_answer": "the short answer span only -- never a sentence",
            },
        )
        self.adjudicate = Module(
            inputs=["question", "evidence", "asked_for", "draft_answer", "options"],
            outputs=["asked_for_check", "answer"],
            instructions=ADJUDICATE_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "asked_for": "what the drafting step believed the question asks for",
                "draft_answer": "the answer proposed by the drafting step",
                "options": "answer strings worth considering, taken from the evidence or the question",
                "asked_for_check": "the kind of thing asked for, and whether the draft is that kind",
                "answer": "the final short answer span only -- never a sentence",
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

        # HOP 2 -- named coverage gap, then a title-style keyword query for it
        plan = self.plan_hop2(
            ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1, evidence=facts_1 or hop1_docs[:3]
        )
        target = (plan.get("missing_target") or "").strip().splitlines()[0][:120] if plan.get("missing_target") else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2)
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)

        # EXTEND THE EVIDENCE AND DRAFT AN ANSWER IN ONE CALL
        drafted = self.draft(
            ctx.llm, tag="draft", question=question, context="\n".join(facts_1) or "(nothing yet)", passages=hop2_docs
        )
        facts_2 = _ledger_lines(drafted.get("facts", ""))
        evidence: list[str] = []
        seen: set[str] = set()
        for line in facts_1 + facts_2:
            k = _key(line)
            if k and k not in seen:
                seen.add(k)
                evidence.append(line)
        if not evidence:
            evidence = list(hop1_docs[:3]) + list(hop2_docs[:3])
        sentence = _ledger_lines(drafted.get("answer_sentence", ""), limit=1)
        if sentence and all(_key(sentence[0]) != _key(e) for e in evidence):
            evidence = evidence + sentence
        draft_answer = _tidy(drafted.get("draft_answer", ""))
        asked_for = (drafted.get("asked_for") or "").strip().splitlines()[0][:120] if drafted.get("asked_for") else ""

        # CONTRASTIVE ADJUDICATION over an option set derived from the evidence in code
        options = _option_set(question, draft_answer, (sentence or []) + evidence)
        ctx.note("options", draft=draft_answer, options=options)
        out = self.adjudicate(
            ctx.llm,
            tag="adjudicate",
            question=question,
            evidence=evidence,
            asked_for=asked_for or "(not stated)",
            draft_answer=draft_answer or "(none)",
            options=options or [draft_answer or "(none)"],
        )
        answer = _tidy(out.get("answer", ""))
        for opt in options:  # snap onto the option's exact surface form when it is the same string
            if _key(opt) == _key(answer):
                answer = opt
                break
        if not answer:
            answer = draft_answer
        return {
            "answer": answer,
            "draft_answer": draft_answer,
            "asked_for": asked_for,
            "options": options,
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
