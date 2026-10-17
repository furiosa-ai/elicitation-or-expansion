"""Candidate: the answer convention is LEARNED from the train split instead of being written down.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7) and retrieval is the proven
gap-directed second hop. The one mechanism under test is where the final call's notion of "a correct
short answer" comes from.

The parent system states that convention as prose rules ("shortest span", "name it as the question
names it", "no apposition"). Those rules are guesses, and the traces show them misfiring in both
directions: they shorten a name the source article writes in full, and they keep a leading preposition
or a unit the reference form drops.

Here `prepare` reads the labelled train pool instead. For each train question it retrieves the
passages the run-time system would see, locates the sentence that literally contains the reference
answer, and keeps the item as a worked example only when the reference answer is a verbatim span of
that sentence -- i.e. only when the example demonstrates the convention rather than guessing at it.
The kept examples are bucketed by question shape (polar, two-way choice, quantity/date, named entity)
and at run time the answering call is shown the examples of the shape it is facing, in the same field
format it must produce. The prose rules that used to encode surface form are removed; the examples
carry that information.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, RunContext, PrepareContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

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

ANSWER_INSTRUCTIONS = """Answer the question from the evidence lines.

Set `asked_for` to the kind of thing the question asks for, in a few words. Then set
`supporting_quote` to the one evidence line that contains the answer, copied exactly. Then set
`answer`.

The worked examples above show the form `answer` must take: they are questions of the same kind with
their reference answers, so follow how they cut the answer out of the supporting line -- how much of a
name they keep, whether they keep a leading word, and how they write dates, quantities and verdicts.
Match that form rather than any habit of your own.

Beyond that: `answer` is the answer alone. Never a sentence, never a restatement of the question,
never an explanation, never a mention of the evidence. The evidence is often incomplete; commit to the
single most likely answer anyway, and never write "unknown", "not specified" or "cannot be determined".
If the question asks whether something holds, `answer` is exactly `yes` or `no`."""

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}
_POLAR = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|both)\b", re.IGNORECASE)
_CHOICE = re.compile(r"[,:]\s*([^,:?]{2,60}?)\s+or\s+([^,:?]{2,60}?)\s*\??\s*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
_NUMERIC = re.compile(r"^[\d][\d,.\s]*($|[a-z%°])|^\d{4}$|^\d{1,2} \w+ \d{4}$", re.IGNORECASE)
_STOP = set(_QUESTION_WORDS) | {"by", "with", "from", "at", "as", "was", "were", "it", "its", "his", "her"}

# question shapes the answer convention is learned separately for
BUCKETS = ("polar", "choice", "quantity", "entity")


def _bucket(question: str) -> str:
    q = question.strip()
    if _CHOICE.search(q) or (re.search(r"\bwhich (one |of )?", q, re.IGNORECASE) and " or " in q):
        return "choice"
    # a polar question asks for a verdict, so it carries no word asking for a value
    if _POLAR.match(q) and not re.search(r"\b(what|which|who|whom|whose|where|when|how)\b", q, re.IGNORECASE):
        return "polar"
    if re.search(r"\b(how many|how much|how long|how old|what year|which year|what date)\b", q, re.IGNORECASE) \
            or q.lower().startswith("when"):
        return "quantity"
    return "entity"


def _asked_for(question: str, bucket: str) -> str:
    q = question.lower()
    if bucket == "polar":
        return "yes or no"
    if bucket == "choice":
        return "which of the two named options"
    if "how many" in q or "how much" in q:
        return "the quantity"
    if "how long" in q or "how old" in q:
        return "the length of time"
    if "what year" in q or "which year" in q or "what date" in q or q.startswith("when"):
        return "the date"
    if q.startswith("who"):   # only the leading question word tells us what is being asked for
        return "the name of the person"
    if q.startswith("where"):
        return "the place"
    return "the name of the thing asked about"


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


def _sentences(text: str) -> list[str]:
    body = text.split(" | ", 1)[1] if " | " in text[:120] else text
    return [s.strip() for s in _SENT_SPLIT.split(body) if len(s.strip()) > 25]


def _overlap(sentence: str, question: str) -> int:
    qw = {w for w in _key(question).split() if w not in _STOP}
    sw = {w for w in _key(sentence).split()}
    return len(qw & sw)


class Harness(BaseHarness):
    K = 7
    MAX_PER_BUCKET = 8      # kept in state
    DEMOS_PER_CALL = 3      # shown to the answering call

    def __init__(self) -> None:
        self.pools: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKETS}
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

    # -- answering module, rebuilt per item so that it carries the demos of that question's shape --
    def _answer_module(self, demos: list[dict[str, Any]]) -> Module:
        return Module(
            inputs=["question", "evidence"],
            outputs=["asked_for", "supporting_quote", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            demos=demos,
            descs={
                "evidence": "sentences copied verbatim from source articles, each prefixed by its title",
                "asked_for": "the kind of thing the question asks for, in a few words",
                "supporting_quote": "the evidence line containing the answer, copied exactly",
                "answer": "the answer in the same form as the worked examples -- never a sentence",
            },
        )

    # -- prepare: mine worked examples of the answer convention from the labelled train pool --------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        kept = 0
        for item in train_items:
            if all(len(self.pools[b]) >= self.MAX_PER_BUCKET for b in BUCKETS):
                break
            try:
                demo = self._mine_demo(item, ctx)
            except Exception:  # a flaky retrieval must not sink prepare
                continue
            if demo is None:
                continue
            bucket = demo.pop("_bucket")
            if len(self.pools[bucket]) < self.MAX_PER_BUCKET:
                self.pools[bucket].append(demo)
                kept += 1
        for bucket in BUCKETS:  # short, span-shaped examples first; multi-word names before one-word ones
            self.pools[bucket].sort(key=lambda d: (0 if len(d["answer"].split()) >= 2 else 1,
                                                   sum(len(line) for line in d["evidence"])))
        ctx.note("demo pools built", kept=kept, sizes={b: len(self.pools[b]) for b in BUCKETS})

    def _mine_demo(self, item: dict[str, Any], ctx: PrepareContext) -> dict[str, Any] | None:
        question, answer = item["question"], str(item.get("answer") or "")
        gold_titles = [t for t in (item.get("gold_titles") or [])]
        if not answer or len(answer) > 80:
            return None
        with ctx.open_tools(item) as tools:
            docs = list(tools.retrieve_docs(question, k=self.K))
            found = {d["title"] for d in docs}
            missing = [t for t in gold_titles if t not in found]
            if missing:  # the run-time system would spend its second search here
                docs += [d for d in tools.retrieve_docs(missing[0], k=self.K) if d["title"] not in found]
        by_title = {d["title"]: d for d in docs}
        gold_docs = [by_title[t] for t in gold_titles if t in by_title]
        if len(gold_docs) < 2:
            return None
        akey = normalize_text(answer)
        support, other_lines = None, []
        for doc in gold_docs:
            best_answer_sent, best_question_sent = None, None
            for sent in _sentences(doc["text"]):
                if akey and akey in normalize_text(sent) and best_answer_sent is None:
                    best_answer_sent = sent
                if best_question_sent is None or _overlap(sent, question) > _overlap(best_question_sent, question):
                    best_question_sent = sent
            if best_answer_sent is not None and support is None:
                support = f"{doc['title']} :: {best_answer_sent}"
            elif best_question_sent is not None:
                other_lines.append(f"{doc['title']} :: {best_question_sent}")
        bucket = _bucket(question)
        if akey in ("yes", "no"):
            bucket = "polar"          # the reference answer settles the shape of the example
        elif bucket == "polar":
            bucket = "entity"         # a verdict-shaped question with a value answer is not a polar example
        if support is None:
            if bucket != "polar" or akey not in ("yes", "no"):
                return None  # the example would not demonstrate a verbatim span
            support = other_lines.pop(0) if other_lines else None
            if support is None:
                return None
        distractors = [f"{d['title']} :: {_sentences(d['text'])[0]}"
                       for d in docs[:4] if d["title"] not in gold_titles and _sentences(d["text"])][:2]
        evidence = []
        for line in (other_lines[:1] + distractors[:1] + [support] + distractors[1:2] + other_lines[1:2]):
            if line and all(_key(line) != _key(e) for e in evidence):
                evidence.append(line[:400])
        if len(evidence) < 2:
            return None
        asked = _asked_for(question, bucket)
        idx = evidence.index(support[:400]) + 1 if support[:400] in evidence else 1
        if bucket == "polar":
            reasoning = (f"The question asks for {asked}. Line {idx} covers what the question asserts, "
                         f"so the verdict is `{answer}` written on its own.")
        else:
            reasoning = (f"The question asks for {asked}. Line {idx} carries it, and that line writes it as "
                         f"«{answer}», so the answer is cut out of the line in exactly that form.")
        return {
            "_bucket": bucket,
            "question": question,
            "evidence": evidence,
            "reasoning": reasoning,
            "asked_for": asked,
            "supporting_quote": support[:400],
            "answer": answer,
        }

    def get_state(self) -> str:
        return json.dumps({"pools": self.pools}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except ValueError:
            return
        pools = data.get("pools") or {}
        self.pools = {b: list(pools.get(b) or []) for b in BUCKETS}

    def _demos_for(self, question: str) -> list[dict[str, Any]]:
        bucket = _bucket(question)
        order = [bucket] + [b for b in ("entity", "quantity", "choice", "polar") if b != bucket]
        demos: list[dict[str, Any]] = []
        for pool_name in order:
            for demo in self.pools.get(pool_name, []):
                if len(demos) >= self.DEMOS_PER_CALL:
                    return demos
                demos.append(demo)
                if pool_name != bucket and len(demos) >= self.DEMOS_PER_CALL:
                    return demos
            if pool_name == bucket and len(demos) >= self.DEMOS_PER_CALL - 1:
                demos = demos[: self.DEMOS_PER_CALL - 1]  # leave room for one example of another shape
        return demos[: self.DEMOS_PER_CALL]

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
        ctx.note("hop2_plan", missing_target=target, query=query_2)
        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        facts_2 = _ledger_lines(
            self.extract2(ctx.llm, tag="extract2", question=question, context=ledger_1, passages=hop2_docs)["facts"]
        )

        # ANSWER, shown worked examples of this question's shape taken from the labelled train pool
        evidence: list[str] = []
        seen: set[str] = set()
        for line in facts_1 + facts_2:
            k = _key(line)
            if k and k not in seen:
                seen.add(k)
                evidence.append(line)
        if not evidence:
            evidence = list(hop1_docs[:3]) + list(hop2_docs[:3])
        demos = self._demos_for(question)
        ctx.note("answer demos", bucket=_bucket(question), n_demos=len(demos))
        out = self._answer_module(demos)(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = _tidy(out.get("answer", ""))
        return {
            "answer": answer,
            "asked_for": out.get("asked_for", ""),
            "n_demos": len(demos),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
