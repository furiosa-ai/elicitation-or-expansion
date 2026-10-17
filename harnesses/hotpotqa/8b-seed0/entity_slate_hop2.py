"""Candidate: the second-hop target is CHOSEN from a slate of names that actually occur, not invented.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7). Evidence extraction and the answer
contract are byte-identical to the parent system; the one mechanism under test is how the second-hop
target is decided.

The parent asks the planner to *invent* the title of the still-missing article from memory ("name the
article you still lack"). Measured on its own traces, that free-form guess names something resembling the
truly missing article only about a fifth of the time, and when the guess is wrong no wording of the query can
rescue the hop. Yet the name it needed was usually sitting in the text it had just read.

So this system turns naming into selection. After the first search, code mines every proper-name phrase
that literally occurs in the retrieved passages (sentence-bounded, quoted work titles included) plus every
name in the question, drops the ones that are already retrieved article titles, ranks them by where and
how often they occur, and presents a numbered slate. The planner picks ONE slate number. Code then uses
that slate string verbatim as the second query -- no type words, no question words, no re-spelling -- so a
correct choice cannot be spoiled by paraphrase. A free-form escape hatch remains for the case where the
needed article is genuinely named nowhere in the text.
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

SLATE_INSTRUCTIONS = """One search has already been run. Choose the target of the second and last one.

The question needs facts from two different articles. `titles_found` lists the articles the first search
returned and `evidence` the facts taken from them; exactly one article is still missing.

`candidates` is a numbered slate of names that literally occur in the question or in the text just
retrieved. The article you still need is almost always one of them -- the bridge that the retrieved text
mentions but does not describe, or, when the question names two things to compare, the one whose facts are
still absent. Read the slate and pick it.

Work in this order:
- say what the still-missing article is about: the entity whose own article would state the fact the
  evidence does not state (the person, work, band, film, place, organisation, species or event itself,
  never a role or a relation such as "the director of X" -- if that is all you can say, the article you
  need is the one that would CONTAIN that fact, so pick that entity instead);
- a name already in `titles_found` is not it, but a name the QUESTION mentions may well be it: the question
  naming something does not mean the search covered it, and often the missing article is exactly that;
- then set `target_number` to the single slate number of that name. Write the bare number, nothing else.
- Set `target_name` to that slate entry copied character for character. Only if no slate entry can
  possibly be the missing article, set `target_number` to 0 and write the name yourself in `target_name`
  as precisely as you can -- a proper name that could stand as the title of an article."""

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

_NAME_TOKEN = r"[A-Z][A-Za-z0-9'\u2019.\-]*"
_LINKER = r"(?:of|the|and|for|de|von|van|der|du|da|la|le|el|di|in|on|at|a|an|to|with|&)"
_NAME_RE = re.compile(rf"{_NAME_TOKEN}(?:\s+(?:{_LINKER}\s+)?{_NAME_TOKEN})*")
_QUOTED_RE = re.compile(r"[\"\u201c\u2018']([^\"\u201d\u2019']{3,70})[\"\u201d\u2019']")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])|\n+|\s\|\s")
_TRAIL_PRONOUN = re.compile(r"\s+(He|She|It|They|The|A|An|His|Her|Its|This|That|In|On|At|By|From|After|When|While|Both)$")
_NAME_STOP = {
    "the", "a", "an", "he", "she", "it", "they", "this", "that", "his", "her", "its", "their", "there",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "was", "were", "is", "are",
    "did", "does", "do", "has", "have", "had", "and", "but", "for", "from", "with", "also", "both",
    "however", "after", "before", "during", "however", "these", "those", "such", "other", "another",
}


def _canon(text: str) -> str:
    """Loose key for deduplicating names: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _strip_article(name: str) -> str:
    words = name.split()
    return " ".join(words[1:]) if len(words) > 1 and words[0].lower() in ("the", "a", "an") else name


def _clean_name(raw: str) -> str:
    name = (raw or "").strip().strip("\"\u201c\u201d\u2018\u2019'")
    name = name.strip(" .,;:-\u2013\u2014")
    prev = None
    while prev != name:          # peel trailing sentence-continuation words like "... Award. He"
        prev = name
        name = _TRAIL_PRONOUN.sub("", name).strip(" .,;:-")
    return name


def _mine_names(text: str) -> list[str]:
    """Proper-name phrases occurring in `text`, kept inside sentence boundaries."""
    found: list[str] = []
    for sentence in _SENT_SPLIT.split(text or ""):
        for m in _QUOTED_RE.finditer(sentence):
            found.append(m.group(1))
        for m in _NAME_RE.finditer(sentence):
            name = _clean_name(m.group(0))
            if not name:
                continue
            words = name.split()
            if len(words) > 9:
                continue
            if len(words) == 1 and (name.lower() in _NAME_STOP or len(name) < 4 or name.isupper() is False and name.islower()):
                continue
            if len(words) == 1 and name.lower() in _NAME_STOP:
                continue
            found.append(name)
            stripped = _strip_article(name)
            if stripped != name and len(stripped) >= 3:
                found.append(stripped)
    return found


def _build_slate(question: str, docs: list[dict], known_titles: list[str], limit: int = 24) -> list[str]:
    """Rank the names that could be the still-missing article; drop the ones already retrieved."""
    scored: dict[str, list] = {}

    def add(name: str, weight: int) -> None:
        name = _clean_name(name)
        key = _canon(name)
        if not key or len(key) < 3:
            return
        slot = scored.setdefault(key, [name, 0])
        slot[1] += weight
        if len(name) > len(slot[0]):     # keep the fullest surface form seen
            slot[0] = name

    for name in _mine_names(question):
        add(name, 4)
    for rank, doc in enumerate(docs):
        weight = 3 if rank < 3 else 1
        for name in _mine_names(doc.get("text", "")):
            add(name, weight)

    blocked = {_canon(t) for t in known_titles}
    blocked |= {_canon(_strip_article(t)) for t in known_titles}
    slate = [(sc, nm) for key, (nm, sc) in scored.items() if key not in blocked]
    slate.sort(key=lambda pair: (-pair[0], -len(pair[1].split()), -len(pair[1])))
    return [name for _, name in slate[:limit]]


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



def _resolve_target(plan: dict, slate: list[str]) -> tuple[str, str]:
    """Turn the planner's choice into the exact search string, preferring the slate's own surface form."""
    raw_num = (plan.get("target_number") or "").strip()
    raw_name = _clean_name((plan.get("target_name") or "").strip().splitlines()[0] if plan.get("target_name") else "")
    digits = re.search(r"\d+", raw_num)
    if digits:
        idx = int(digits.group(0))
        if 1 <= idx <= len(slate):
            return slate[idx - 1], "slate"
    # no usable number: fall back to the written name, snapped onto a slate entry when it matches one
    if raw_name:
        key = _canon(raw_name)
        for entry in slate:
            if _canon(entry) == key:
                return entry, "slate"
        return raw_name[:120], "written"
    return "", "none"


class Harness(BaseHarness):
    K = 7
    SLATE = 24

    def __init__(self) -> None:
        self.extract1 = Module(
            inputs=["question", "passages"],
            outputs=["facts"],
            instructions=EXTRACT_INSTRUCTIONS,
            descs={"facts": "one `Title :: sentence` line per relevant fact, copied verbatim"},
        )
        self.pick_hop2 = Module(
            inputs=["question", "titles_found", "evidence", "candidates"],
            outputs=["target_number", "target_name"],
            instructions=SLATE_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "candidates": "numbered slate of names occurring in the question or the retrieved text",
                "target_number": "the slate number of the still-missing article, or 0 for none of them",
                "target_name": "that slate entry copied exactly (or your own name when the number is 0)",
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
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "answer_type": "yes_no, person, organisation, work, place, date, number or other",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
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

        # HOP 2 -- pick the missing article off a slate of names that really occur, then query it verbatim
        slate = _build_slate(question, hop1, titles_1, limit=self.SLATE)
        numbered = [f"{i + 1}. {name}" for i, name in enumerate(slate)]
        plan = self.pick_hop2(
            ctx.llm,
            tag="plan_hop2",
            question=question,
            titles_found=titles_1,
            evidence=facts_1 or hop1_docs[:3],
            candidates=numbered or ["1. (none found)"],
        )
        target, source = _resolve_target(plan, slate)
        query_2 = target or _keyword_query(question, fallback=question)
        ctx.note("hop2_plan", missing_target=target, source=source, query=query_2, slate_size=len(slate))

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

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "target_source": source,
            "query_2": query_2,
            "evidence": evidence,
        }
