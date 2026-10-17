"""Deep retrieval that is read at SENTENCE granularity instead of passage granularity.

Three separate attempts to buy recall (link expansion, a title-level wide scan, article-space
queries) all raised `gold_title_recall` and all LOST accuracy, because every one of them paid for
the extra recall in the same currency: whole additional passages poured into the one context the
answer is written from.  A fourth attempt moved the extra hop into an isolated sub-solver and lost
too.  The question those four leave open is whether breadth itself is harmful, or only breadth
delivered as passages.

This harness separates the two.  The parent pipeline is left exactly as it is -- the same planning
call, the same queries, the same first eight hits per query in the same order forming the same
triage pool, the same selected passages read in full by the same answerer.  What changes is that
every query is now issued at maximum depth and the part of each result list the parent never sees
(ranks nine and below) is kept aside as a DEEP TAIL, then distilled deterministically:

  * each tail lead is split into sentences, and every sentence is scored, with no LLM call, by the
    question's own content words weighted by how rare each word is inside the pool (a word shared by
    half the pool discriminates nothing; a word in two leads is the question's anchor), plus the
    words of the linking fact triage established, plus a bonus when the article's title carries a
    question word;
  * the best sentences are kept, at most two per article, and handed to the answerer as one line
    each, `title -- sentence`, under a field that says plainly they are unverified extra leads and
    that the evidence passages come first.

Sixteen such lines cost about the same as two passages, so the answering context stays the size the
parent's is while the material behind it grows from roughly thirty articles to over a hundred.  If
accuracy rises, the earlier breadth experiments failed on the packaging rather than on the breadth;
if it falls again with the passage context held fixed, breadth itself is what hurts.
"""

from __future__ import annotations

import math
import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are", "did", "does", "do",
    "has", "have", "had", "that", "this", "with", "by", "from", "as", "it", "its", "his", "her",
    "their", "he", "she", "they", "both", "also", "than", "then", "there", "these", "those", "name",
    "named", "known", "called", "same", "other", "another", "any", "between", "about", "into",
}


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip()
        if len(line) >= 3 and not line.startswith("[["):
            out.append(line[:220])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


def _words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in STOP]


def _sentences(passage: str) -> list[str]:
    _, _, body = passage.partition(" | ")
    out = []
    for sent in SENT_SPLIT.split(body.strip()):
        sent = sent.strip()
        if 20 <= len(sent) <= 320:
            out.append(sent)
        elif len(sent) > 320:
            out.append(sent[:320].rsplit(" ", 1)[0])
    return out[:6]


def _weights(question: str, linking_fact: str, corpus: list[str]) -> dict[str, float]:
    """Rarity weight, inside the retrieved pool, of every content word the question cares about."""
    df: dict[str, int] = {}
    for passage in corpus:
        for w in set(_words(passage)):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(corpus))
    weights: dict[str, float] = {}
    for w in _words(question):
        weights[w] = math.log(1.0 + n / (1.0 + df.get(w, 0)))
    for w in _words(linking_fact):
        weights[w] = max(weights.get(w, 0.0), 0.75 * math.log(1.0 + n / (1.0 + df.get(w, 0))))
    return weights


class Harness(BaseHarness):
    K_PLAN = 8          # hits per query the parent pipeline reads (unchanged)
    K_DEEP = 20         # depth every query is now issued at; ranks K_PLAN.. feed the tail only
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    DIGEST_LINES = 16   # sentences distilled out of the deep tail
    PER_TITLE = 2       # at most this many sentences from one article
    MIN_HITS = 2        # a sentence must match at least this many distinct question words

    PLAN_INSTR = (
        "Break the question into the separate facts that must be looked up in an encyclopaedia.\n"
        "Write `queries` as one short keyword search query per line (2 or 3 lines, no numbering). "
        "Each query must stand on its own: name the entity it is about explicitly instead of using "
        "a pronoun or a description that depends on another query.\n"
        "If one fact can only be looked up after another is known, still write the query for the "
        "part that is already named."
    )
    TRIAGE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia "
        "leads, shown as `id: title | beginning of the article`.\n"
        "`relevant_ids` : the ids worth reading in full, most useful first, at most 8, comma separated.\n"
        "`linking_fact` : the fact the pool establishes that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); say `unknown` if none.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`.\n"
        "`next_query`   : if not sufficient, one keyword search query for the missing fact, naming "
        "the entity from `linking_fact` explicitly; otherwise leave it empty."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "`extra_leads` holds single sentences pulled from further down the search results, one per "
        "line as `title -- sentence`. They are unverified: use one only when the evidence passages "
        "do not settle the question, and never prefer one over a passage that does.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact", "extra_leads"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = _title_of(p).lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    def _deep(self, ctx: RunContext, query: str, pool: list[str], seen: set[str],
              tail: list[str], tail_seen: set[str], k_read: int) -> None:
        """One query, issued at full depth: the head feeds the parent's pool, the rest the tail."""
        hits = ctx.tools.retrieve(query, k=self.K_DEEP)
        self._add(pool, seen, hits[:k_read])
        for p in hits[k_read:]:
            key = _title_of(p).lower()
            if key and key not in tail_seen:
                tail_seen.add(key)
                tail.append(p)

    def _digest(self, question: str, linking_fact: str, tail: list[str],
                read: set[str]) -> list[str]:
        """Rank every tail sentence by pool-rarity overlap with the question; keep the best few."""
        corpus = [p for p in tail if _title_of(p).lower() not in read]
        if not corpus:
            return []
        weights = _weights(question, linking_fact, corpus)
        if not weights:
            return []
        qwords = set(weights)
        scored: list[tuple[float, str, str]] = []
        for passage in corpus:
            title = _title_of(passage)
            title_hit = 0.5 * sum(weights.get(w, 0.0) for w in set(_words(title)) & qwords)
            for sent in _sentences(passage):
                hit = set(_words(sent)) & qwords
                if len(hit) < self.MIN_HITS:
                    continue
                score = sum(weights[w] for w in hit) + title_hit
                score /= 1.0 + 0.004 * len(sent)
                scored.append((score, title, sent))
        scored.sort(key=lambda s: -s[0])
        per: dict[str, int] = {}
        out: list[str] = []
        for score, title, sent in scored:
            if per.get(title, 0) >= self.PER_TITLE:
                continue
            per[title] = per.get(title, 0) + 1
            out.append(f"{title} -- {sent}")
            if len(out) >= self.DIGEST_LINES:
                break
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        tail: list[str] = []
        tail_seen: set[str] = set()

        # 1. decompose (unchanged) --------------------------------------------------------------
        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )

        # 2. every query at full depth: head -> parent pool, ranks below -> deep tail -----------
        self._deep(ctx, question, pool, seen, tail, tail_seen, self.K_PLAN)
        for q in queries:
            self._deep(ctx, q, pool, seen, tail, tail_seen, self.K_PLAN)

        # 3-4. triage with an adaptive follow-up hop (pool identical to the parent's) -----------
        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_TRIAGE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.triage(ctx.llm, tag=f"triage{step + 1}", question=question, pool=table)
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
                range(1, min(len(shown), self.SELECT_CAP) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in ("unknown", "none", "n/a"):
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_TRIAGE - 1:
                break
            nq = _split_lines(out.get("next_query", ""), 1)
            if not nq:
                break
            followups.append(nq[0])
            before = len(pool)
            self._deep(ctx, nq[0], pool, seen, tail, tail_seen, self.K_FOLLOW)
            if len(pool) == before:
                break
            pool = [pool[i - 1] for i in ids] + [p for i, p in enumerate(pool, 1) if i not in ids]
            ctx.note("follow-up hop", query=nq[0], pool_size=len(pool), tail=len(tail))

        # 5. answer from the raw selected passages plus the distilled deep tail -----------------
        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]
        read = {_title_of(p).lower() for p in evidence}
        digest = self._digest(question, linking_fact, tail, read)
        ctx.note("deep tail distilled", tail=len(tail), lines=len(digest))
        ans = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
            extra_leads=digest or ["(none)"],
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return {
            "answer": ans,
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "tail_size": len(tail),
            "n_digest": len(digest),
            "n_selected": len(evidence),
        }
