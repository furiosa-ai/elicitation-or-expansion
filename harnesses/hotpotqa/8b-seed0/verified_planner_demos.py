"""Candidate: teach the second-hop planner from the train split with retrieval-VERIFIED examples.

Interaction depth is unchanged (4 LLM calls, 2 retrievals, k=7) and every call's instructions are
the parent's. The one mechanism under test is where the planner's notion of "the article still
missing" comes from: hand-written prose in the parent, worked examples with ground truth here.

`prepare` replays only the FIRST search of each labelled train question, so it sees exactly the
situation the planner faces at run time: the titles the question's own words retrieved, and the
sentences of those articles. The labels then say which source article is still missing -- no
guessing. A query string is derived from that article's title and only kept if issuing it actually
brings the article back from the index, so every example pairs a real gap with a query that
demonstrably closes it. Examples are bucketed by the shape of the question (a two-way comparison
names its second article; a bridge question only describes it), and at run time the planner is shown
three examples of the shape it is facing. No LLM call is made in `prepare`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
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
`missing_target` must be a proper name that could stand as the title of an article. Never write a role
or a relation ("the vocalist of X", "the director of Y", "birthplace of Z"): if that is all you can
say, then the article you actually need is the one that would CONTAIN that fact, so name that entity
instead (the band, the film, the album, the person the question names).

Then set `query` to the search string. The index matches words literally against article titles and
their opening paragraph, so:
- write 2 to 8 words: the missing target's name, plus one or two words that disambiguate it (a type
  word like film, band, river, novel, or a year, only if the name alone is ambiguous);
- never write a question or a sentence, and use no question words (who, what, when, where, which, how);
- never include words describing facts you already have, and never include a name that is already in
  `titles_found` unless it is part of the missing target's own name;
- drop filler words entirely.
Output the query as bare keywords.

The worked examples above were checked against the index: in each of them the query shown is one that
actually returns the missing article. Match how they name the target and how short they keep the query."""

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

_LEDGER_JUNK = re.compile(r"^(none|n/?a|no relevant.*|nothing.*|-+)$", re.IGNORECASE)
_PREFIXES = ("answer:", "final answer:", "the answer is", "answer is")
_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom", "is", "are",
                   "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "the", "a",
                   "an", "of", "in", "on", "for", "to", "and", "or", "that", "this", "there", "any"}
_STOP = _QUESTION_WORDS | {"by", "with", "at", "from", "as", "it", "its", "his", "her", "their", "been",
                           "be", "also", "than", "into", "between", "both", "same", "other", "about",
                           "after", "before", "during", "over", "not", "but", "they", "he", "she", "them"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_COMPARE_CUES = re.compile(
    r"\b(both|either|same|older|younger|earlier|later|longer|shorter|taller|larger|smaller|bigger|"
    r"more|fewer|higher|lower|first|closer|greater)\b", re.IGNORECASE)


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
    text = (raw or "").strip()
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break
    text = text.strip().strip("*`\"'").strip()
    text = re.sub(r"^(query|search|search for|keywords)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("?.").strip()
    words = text.split()
    if len(words) > 10:
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


def _norm_title(title: str) -> str:
    return re.sub(r"\W+", " ", (title or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _split_sentences(text: str) -> list[str]:
    body = (text or "").strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(body) if p.strip()]
    return parts or ([body] if body else [])


def _shape(question: str) -> str:
    """Two-way comparison (the second article is NAMED) vs bridge (it is only described)."""
    q = question or ""
    if re.search(r"\bor\b", q, re.IGNORECASE) and _COMPARE_CUES.search(q):
        return "comparison"
    if re.search(r"\bboth\b|\bsame\b", q, re.IGNORECASE):
        return "comparison"
    if _COMPARE_CUES.search(q) and re.search(r"\band\b", q, re.IGNORECASE):
        return "comparison"
    return "bridge"


def _demo_evidence(docs: list[dict[str, Any]], question: str, cap: int = 5) -> list[str]:
    """Code-side stand-in for the extraction step, used only to fill the demo's `evidence` field."""
    key = _tokens(question)
    scored: list[tuple[float, str]] = []
    for rank, doc in enumerate(docs):
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        if title and body.startswith(f"{title} |"):
            body = body[len(title) + 1:].lstrip("| ").strip()  # lines read `Title :: sentence`
        for si, sent in enumerate(_split_sentences(body)):
            if len(sent) < 12:
                continue
            score = len(_tokens(sent) & key) + (0.5 if si == 0 else 0.0) - 0.1 * rank
            if score <= 0.5:
                continue
            scored.append((score, f"{title} :: {sent}"[:320]))
    scored.sort(key=lambda t: -t[0])
    out: list[str] = []
    for _, line in scored:
        if line not in out:
            out.append(line)
        if len(out) >= cap:
            break
    return out


def _query_form(title: str) -> str:
    """The title-style keyword query to try for an article: the name, with any parenthesised
    qualifier turned into plain disambiguating words -- the shape the run-time planner is asked to
    produce. One form only: the probe budget for a train item is two searches."""
    base = (title or "").strip()
    inner = re.sub(r"\s*\([^)]*\)\s*", " ", base).strip()
    paren = " ".join(re.findall(r"\(([^)]*)\)", base)).strip()
    form = f"{inner} {paren}".strip() if paren else (inner or base)
    return re.sub(r"\s+", " ", form).strip()


class Harness(BaseHarness):
    K = 7
    N_DEMOS = 3
    POOL_PER_SHAPE = 12

    def __init__(self) -> None:
        self.demo_pool: dict[str, list[dict[str, Any]]] = {"bridge": [], "comparison": []}
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

    # -- prepare: mine verified gap-planning examples from the train split -----------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        mined = {"bridge": [], "comparison": []}
        n_seen = n_gap = n_verified = 0
        for item in train_items:
            if all(len(v) >= self.POOL_PER_SHAPE for v in mined.values()):
                break
            question = (item.get("question") or "").strip()
            gold = [t for t in (item.get("gold_titles") or []) if t]
            if not question or len(gold) != 2:
                continue
            n_seen += 1
            try:
                with ctx.open_tools(item) as tools:
                    docs = tools.retrieve_docs(question, k=self.K)
                    titles = [d["title"] for d in docs]
                    found = {_norm_title(t) for t in titles}
                    missing = [t for t in gold if _norm_title(t) not in found]
                    covered = [t for t in gold if _norm_title(t) in found]
                    if len(missing) != 1 or not covered:
                        continue
                    n_gap += 1
                    target = missing[0]
                    form = _query_form(target)
                    if not form:
                        continue
                    hits = tools.retrieve_docs(form, k=self.K)
                    if not any(_norm_title(h["title"]) == _norm_title(target) for h in hits):
                        continue  # keep only gaps whose query demonstrably closes them
                    query = form
            except Exception:  # noqa: BLE001 -- a failed train probe must not break prepare
                continue
            n_verified += 1
            shape = "comparison" if (item.get("type") or "") == "comparison" else "bridge"
            if len(mined[shape]) >= self.POOL_PER_SHAPE:
                continue
            evidence = _demo_evidence(docs, question)
            named = _norm_title(target) in _norm_title(question)
            reasoning = (
                f"The search returned articles about {covered[0]}, so that side of the question is "
                f"covered. The remaining fact belongs to a different article: "
                + (f"the question names it, {target}." if named
                   else f"the passages point to {target}, which is the article that would state it.")
                + f" A short title-style query for it is: {query}."
            )
            mined[shape].append({
                "question": question,
                "titles_found": titles,
                "evidence": evidence,
                "reasoning": reasoning,
                "missing_target": target,
                "query": query,
                "named_in_question": named,
            })
        # examples where the target had to be resolved from the evidence teach more than copies
        for shape, pool in mined.items():
            pool.sort(key=lambda d: (d["named_in_question"] if shape == "bridge" else not d["named_in_question"],
                                     len(d["question"])))
        self.demo_pool = mined
        ctx.note("mined_planner_demos", seen=n_seen, one_gap=n_gap, verified=n_verified,
                 bridge=len(mined["bridge"]), comparison=len(mined["comparison"]))

    def get_state(self) -> str:
        return json.dumps({"demo_pool": self.demo_pool}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except ValueError:
            data = {}
        pool = data.get("demo_pool") or {}
        self.demo_pool = {"bridge": list(pool.get("bridge") or []),
                          "comparison": list(pool.get("comparison") or [])}

    def _planner(self, question: str) -> Module:
        shape = _shape(question)
        pool = list(self.demo_pool.get(shape) or [])
        if len(pool) < self.N_DEMOS:
            other = "comparison" if shape == "bridge" else "bridge"
            pool = pool + list(self.demo_pool.get(other) or [])
        demos = [{k: d[k] for k in ("question", "titles_found", "evidence", "reasoning",
                                    "missing_target", "query")}
                 for d in pool[: self.N_DEMOS]]
        return Module(
            inputs=["question", "titles_found", "evidence"],
            outputs=["missing_target", "query"],
            instructions=GAP_QUERY_INSTRUCTIONS,
            descs={
                "titles_found": "titles the first search already returned",
                "missing_target": "name of the one article still missing",
                "query": "2-8 bare keywords for the second and last search",
            },
            demos=demos,
        )

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        hop1 = ctx.tools.retrieve_docs(question, k=self.K)
        titles_1 = [d["title"] for d in hop1]
        hop1_docs = [d["text"] for d in hop1]
        facts_1 = _ledger_lines(
            self.extract1(ctx.llm, tag="extract1", question=question, passages=hop1_docs)["facts"]
        )
        ledger_1 = "\n".join(facts_1)

        planner = self._planner(question)
        plan = planner(ctx.llm, tag="plan_hop2", question=question, titles_found=titles_1,
                       evidence=facts_1 or hop1_docs[:3])
        target = (plan.get("missing_target") or "").strip().splitlines()
        target = target[0][:120] if target else ""
        query_2 = _keyword_query(plan.get("query", ""), fallback=(target or question))
        ctx.note("hop2_plan", missing_target=target, query=query_2, shape=_shape(question),
                 n_demos=len(planner.demos))

        hop2_docs = ctx.tools.retrieve(query_2, k=self.K)
        facts_2 = _ledger_lines(
            self.extract2(ctx.llm, tag="extract2", question=question, context=ledger_1,
                          passages=hop2_docs)["facts"]
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

        out = self.generate_answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        return {
            "answer": _tidy(out.get("answer", "")),
            "answer_type": out.get("answer_type", ""),
            "missing_target": target,
            "query_2": query_2,
            "evidence": evidence,
        }
