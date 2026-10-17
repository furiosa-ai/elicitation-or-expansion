"""A demonstration bank mined from the train split, replacing hand-written answer-style instructions.

Three iterations have now attacked the answer-shape bucket with instructions and deterministic rules
(+10.3, then -9.0, then +0.0). Reading the frontier's remaining full-coverage failures shows why the last
two returned nothing: the residual errors are not one rule but a set of unstated CONVENTIONS of the
collection, and they point in opposite directions --

    a count is written out as words in one item and as a digit in another;
    a person is wanted with their middle name in one item and with their everyday name in another;
    a field of work is wanted bare where the passage says it with a modifier;
    a country is wanted under one of two names that both appear in the evidence.

No instruction can state all of them and no length preference can satisfy both directions at once, which
is exactly what the previous two candidates demonstrated empirically. But conventions of this kind are
precisely what few-shot DEMONSTRATIONS transmit, and the train split is available for learning them.

`prepare` therefore mines a demonstration bank with no model calls at all: for every train question it
reads the supporting articles by exact title, assembles the passage block an answerer would actually see
(supporting articles plus real distractors returned for the question), and pairs it with the reference
answer. Demonstrations are bucketed by the SHAPE of the question -- yes/no, a count, a date, a choice
between two named alternatives, a person, a place, other -- because the conventions are shape-specific.

At run time the retrieval half of the pipeline is byte-identical to the frontier; the final answering call
is conditioned on the demonstrations for this question's shape, so the model is shown how answers of this
shape are written instead of being told. Falls back to the frontier's answerer when no bank is loaded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module

_BULLET = re.compile(r"^\s*(?:[-*•]|\[?\d+[.)\]]?)\s*")
_INDEX = re.compile(r"\d+")
_LEADING_FILLER = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|short\s+)?answer(?:\s+is)?\s*[:\-]?\s*", re.IGNORECASE
)
_YESNO_PREFIX = re.compile(r"^\s*(yes|no)\b\s*[,.:;!\-]*", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_MARKUP = re.compile(r"[*_`]+")
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_TITLE_JUNK = re.compile(r"[^a-z0-9 ]+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are",
    "what", "which", "who", "whom", "whose", "when", "where", "how", "did", "do", "does", "that",
    "this", "by", "with", "from", "as", "it", "its", "his", "her", "their", "be", "been", "has", "have",
}

DECOMPOSE_INSTRUCTIONS = """Plan a keyword search for a question that can only be answered by combining
two different source documents.

Write 3 search queries, one per line, no numbering and no explanation:
1. the most specific description in the question, phrased as the title/topic of the document that describes it;
2. a short keyword-only variant of that same target using distinctive proper nouns, dates or numbers only
   (no question words, no stop words) -- it should retrieve the same document by different wording;
3. the second document: the entity or property that the question ultimately asks about, phrased as a topic.

The retriever matches words, not meaning, so prefer the exact names and rare terms that would appear in the
document, and never write a query as a question."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. In `reasoning`, decide which chain of facts the question needs, then
check which links of that chain the passages already establish.

Produce:
- `useful`: the numbers of the passages that state a needed fact, most important first, comma separated.
  Include a passage only if it carries a fact the answer depends on; write `none` if no passage does.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property
  must still be looked up). Write `none` if every needed fact is present.
- `next_query`: a keyword query for that missing fact, anchored on the exact entity name as it appears in the
  passages, plus the property being sought. Leave it empty when nothing is missing. Never repeat a query whose
  results are already listed.
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`."""

DEMO_ANSWER_INSTRUCTIONS = """Answer the question from the passages.

The preceding exchanges are worked examples from the same collection, answered the way this collection
expects. Match them: they show how short an answer should be, which wording is kept and which is dropped,
whether a quantity is written as a word or as a digit, and how much of a name is given. Copy that style
rather than deciding it yourself, and never reply with a sentence or an explanation."""

ANSWER_INSTRUCTIONS = """Answer the question from the passages with the shortest possible answer.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


def _title_key(title: str) -> str:
    t = _PARENTHETICAL.sub("", str(title or "").strip())
    return " ".join(_TITLE_JUNK.sub(" ", t.lower()).split())


def question_shape(question: str) -> str:
    """Coarse, domain-general shape of a question. Conventions differ per shape, so demos are bucketed by it."""
    q = " ".join(str(question or "").lower().split())
    if _BOOL_QUESTION.match(q):
        return "yesno"
    if "how many" in q or "how much" in q:
        return "count"
    if q.startswith("when") or "what year" in q or "which year" in q or "what date" in q:
        return "date"
    if re.search(r"\b\w+,?\s+or\s+\w+", q) and re.search(r"\b(which|who|what)\b", q):
        return "choice"
    if q.startswith("who"):
        return "person"
    if q.startswith("where") or "what country" in q or "what city" in q or "what state" in q:
        return "place"
    return "other"


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP and len(w) > 2}


class Harness(BaseHarness):
    K = 8
    K_TITLE = 20            # wide read so the exact-title article is found while mining demos
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380
    DEMO_SNIPPET = 260      # chars per passage inside a demonstration
    DEMO_PASSAGES = 4
    N_DEMOS = 5             # demonstrations shown at answering time
    PER_SHAPE = 14          # demonstrations retained per shape bucket

    def __init__(self) -> None:
        self.bank: dict[str, list[dict[str, Any]]] = {}
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.route = Module(
            inputs=["question", "passages"],
            outputs=["useful", "missing", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={"answer": "the bare short answer"},
        )
        self.commit = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=COMMIT_INSTRUCTIONS,
            descs={"answer": "the bare short answer, never an abstention"},
        )

    # -- prepare: mine demonstrations from the train split (no model calls) ---------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        buckets: dict[str, list[dict[str, Any]]] = {}
        n_seen = n_kept = 0
        for it in train_items:
            question = str(it.get("question") or "")
            gold = str(it.get("answer") or "").strip()
            titles = list(it.get("gold_titles") or [])
            if not question or not gold or not titles:
                continue
            n_seen += 1
            try:
                with ctx.open_tools(it) as tools:
                    support: list[str] = []
                    for title in titles[:2]:
                        key = _title_key(title)
                        for doc in tools.retrieve_docs(title, k=self.K_TITLE):
                            if _title_key(doc.get("title", "")) == key:
                                support.append(str(doc.get("text") or ""))
                                break
                    if len(support) < len(titles[:2]):
                        continue        # an unreadable demonstration teaches nothing
                    # real distractors: what a search for this question actually returns
                    keys = {_title_key(p.split(" | ")[0]) for p in support}
                    extra = [p for p in tools.retrieve(question, k=self.K)
                             if _title_key(p.split(" | ")[0]) not in keys]
            except Exception:           # a train item that cannot be read is simply skipped
                continue
            passages = [p[: self.DEMO_SNIPPET] for p in (support + extra)][: self.DEMO_PASSAGES]
            shape = question_shape(question)
            buckets.setdefault(shape, []).append(
                {"question": question, "passages": passages, "answer": gold,
                 "w": sorted(_content_words(question))[:24]}
            )
            n_kept += 1
        # Keep a balanced slice per shape by an even stride over a length-neutral ordering. Sorting by
        # answer length would hand each bucket its most verbose examples and teach exactly the verbosity
        # the answerer has to unlearn, so the ordering key is the question text and the stride preserves
        # the bucket's natural distribution of answer forms.
        self.bank = {}
        for shape, demos in buckets.items():
            demos.sort(key=lambda d: d["question"])
            step = max(1, len(demos) // self.PER_SHAPE)
            self.bank[shape] = demos[::step][: self.PER_SHAPE]
        ctx.note("demo bank mined", n_seen=n_seen, n_kept=n_kept,
                 shapes={k: len(v) for k, v in self.bank.items()})

    def get_state(self) -> str:
        return json.dumps({"bank": self.bank}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            self.bank = dict((json.loads(state) or {}).get("bank") or {})
        except Exception:
            self.bank = {}

    def _select_demos(self, question: str) -> list[dict[str, str]]:
        """Demonstrations of the same question shape, most lexically related first (deterministic)."""
        shape = question_shape(question)
        pool = list(self.bank.get(shape) or [])
        if len(pool) < self.N_DEMOS:
            for other, demos in sorted(self.bank.items()):
                if other != shape:
                    pool.extend(demos)
        if not pool:
            return []
        qw = _content_words(question)
        scored = sorted(
            pool,
            key=lambda d: (-len(qw & set(d.get("w") or [])), str(d.get("question", ""))),
        )
        chosen = scored[: self.N_DEMOS]
        chosen.reverse()        # most related demonstration sits closest to the real request
        return [{"question": d["question"], "passages": list(d["passages"]), "answer": d["answer"]}
                for d in chosen]

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("«»").strip()
        return " ".join(q.split())[:300]

    def _parse_queries(self, raw: str, question: str) -> list[str]:
        queries: list[str] = []
        for line in str(raw or "").splitlines():
            q = self._clean_query(line)
            if len(q) < 3 or q.lower() in {"none", "queries"}:
                continue
            if q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
            if len(queries) == 3:
                break
        if question not in queries:
            queries.insert(0, question)
        return queries[:4]

    @staticmethod
    def _parse_indices(raw: str, n: int) -> list[int]:
        picks: list[int] = []
        for tok in _INDEX.findall(str(raw or "")):
            i = int(tok)
            if 1 <= i <= n and i not in picks:
                picks.append(i)
        return picks

    @staticmethod
    def _tidy(text: str, question: str) -> str:
        ans = _MARKUP.sub("", str(text or "")).strip().split("\n")[0].strip()
        ans = _LEADING_FILLER.sub("", ans).strip()
        ans = ans.strip('"').strip("'").strip("«»").strip()
        m = _YESNO_PREFIX.match(ans)
        if m and (_BOOL_QUESTION.match(question or "") or len(ans.split()) > 1):
            return m.group(1).lower()
        while ans.endswith((".", ",", ";", ":")):
            ans = ans[:-1].strip()
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str) -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=self.K):
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(passage)
                added += 1
            return added

        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q)

        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[:8]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(routed.get("next_query", "")):
                break

        evidence = selected or pool[:8]
        demos = self._select_demos(question)
        ctx.note("pool routed", n_pool=len(pool), n_selected=len(selected),
                 retrievals=retrievals, n_demos=len(demos), shape=question_shape(question))

        if demos:
            # a fresh Module per item: `run` is called concurrently, so nothing shared is mutated
            answerer = Module(
                inputs=["question", "passages"],
                outputs=["answer"],
                instructions=DEMO_ANSWER_INSTRUCTIONS,
                descs={"passages": "numbered source passages", "answer": "the bare short answer"},
                chain_of_thought=False,
                demos=demos,
            )
        else:
            answerer = self.answer
        drafted = answerer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool),
                "n_demos": len(demos), "missing": missing}
