"""Link expansion re-tried at SENTENCE granularity.

Iteration 2 harvested the proper names written inside the passages the system had already read,
looked those names up, and poured the resulting articles into the answering context as passages.
Recall rose (.807 -> .818) and accuracy fell (-0.7).  Iteration 7 then showed that the same breadth
that hurt as passages helps as sentences: issuing every query at maximum depth and distilling the
unread tail into sixteen single lines took recall to .85 with no accuracy loss at all.

Those two results isolate a question the frontier leaves open.  Link expansion reaches articles that
NO reformulation of the question can reach -- an article whose only lexical bridge to the question is
a name printed inside another article -- and deep ranks reach articles the question's own words
already rank, just lower.  The frontier only has the second kind.  This harness feeds the first kind
into exactly the same channel:

  * proper names are harvested with no LLM call from the passages triage selected -- runs of
    capitalised tokens, quoted titles, names glued by particles -- and every name already carried by
    a pool title or already named in the question is dropped, because those have been searched;
  * the surviving mentions are ranked by the question-word rarity of the sentence they were printed
    in, and the best few are looked up by bare name (retrieval only, no LLM call);
  * the articles that come back never enter the passage pool.  They join the deep tail and are
    distilled by the same sentence scorer, competing for the same sixteen lines, with a reserved
    share so a linked article cannot be crowded out by the tail of a keyword query.

The answering context therefore has the size, the shape and the passage list of the frontier's, and
the only thing that changes is which articles the sixteen distilled sentences may come from.  If
accuracy rises, iteration 2 failed on packaging rather than on link following; if it does not, the
frontier's reach is limited by what the question's own words rank and link following adds nothing.
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
PARTICLE = {"of", "the", "and", "de", "del", "der", "den", "di", "da", "van", "von", "la", "le", "el"}
NAME_RUN = re.compile(r"[A-Z][\w&.'’\-]*(?:\s+(?:of|the|and|de|del|der|den|di|da|van|von|la|le|el)?\s*[A-Z][\w&.'’\-]*)*")
QUOTED = re.compile(r"[\"“]([^\"”]{3,60})[\"”]")
NOT_A_NAME = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "american", "british", "english", "french", "german",
    "italian", "canadian", "australian", "russian", "japanese", "chinese", "indian", "spanish",
    "he", "she", "it", "they", "his", "her", "the", "a", "an", "in", "on", "at", "however",
    "after", "before", "during", "although", "while", "this", "that", "there", "these", "those",
    "university", "national", "international", "city", "county", "state", "north", "south",
    "east", "west", "world", "war", "new", "united", "states",
}
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


def _clean_name(raw: str) -> str:
    name = raw.strip().strip(",;:\u2013-").strip()
    if name.lower().startswith("the "):
        name = name[4:]
    parts = name.split()
    while parts and parts[-1].lower() in PARTICLE:
        parts.pop()
    if parts and parts[-1].endswith(".") and len(parts[-1]) > 2:
        parts[-1] = parts[-1][:-1]          # sentence-final full stop, not an initial
    return " ".join(parts).strip()


def _mentions(passage: str) -> list[tuple[str, str]]:
    """Every proper name printed inside a passage, paired with the sentence it was printed in."""
    out: list[tuple[str, str]] = []
    for sent in _sentences(passage):
        raws: list[str] = list(QUOTED.findall(sent)) + list(NAME_RUN.findall(sent))
        for raw in list(raws):
            if " and " in raw:              # a run glued by `and` is usually two names
                raws.extend(raw.split(" and "))
        for raw in raws:
            name = _clean_name(raw)
            parts = name.split()
            if not 1 <= len(parts) <= 6 or len(name) < 4:
                continue
            low = name.lower()
            if low in NOT_A_NAME or parts[0].lower() in NOT_A_NAME:
                continue
            if len(parts) == 1 and (len(name) < 6 or name.isupper()):
                continue
            out.append((name, sent))
    return out


class Harness(BaseHarness):
    K_PLAN = 8          # hits per query the parent pipeline reads (unchanged)
    K_DEEP = 20         # depth every query is now issued at; ranks K_PLAN.. feed the tail only
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    K_LINK = 6          # depth a harvested name is looked up at
    MAX_LINKS = 4       # names looked up per item (retrieval only, no LLM call)
    LINK_LINES = 6      # lines of the digest reserved for linked articles
    MAX_RETRIEVALS = 10  # leave headroom under the per-item retrieval cap

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

    def _score_sentences(self, weights: dict, corpus: list[str]) -> list[tuple[float, str, str]]:
        """Rank every sentence of a set of leads by pool-rarity overlap with the question."""
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
        return scored

    def _take(self, scored: list[tuple[float, str, str]], out: list[str], per: dict[str, int],
              limit: int) -> None:
        for _, title, sent in scored:
            if len(out) >= limit:
                return
            if per.get(title, 0) >= self.PER_TITLE:
                continue
            line = f"{title} -- {sent}"
            if line in out:
                continue
            per[title] = per.get(title, 0) + 1
            out.append(line)

    def _digest(self, question: str, linking_fact: str, tail: list[str], linked: list[str],
                read: set[str]) -> list[str]:
        """Distil tail and linked leads into one line each, with a reserved share for linked ones."""
        tail_corpus = [p for p in tail if _title_of(p).lower() not in read]
        link_corpus = [p for p in linked if _title_of(p).lower() not in read]
        corpus = tail_corpus + link_corpus
        if not corpus:
            return []
        weights = _weights(question, linking_fact, corpus)
        if not weights:
            return []
        out: list[str] = []
        per: dict[str, int] = {}
        # the articles reached only through a printed name get their own share of the lines first
        self._take(self._score_sentences(weights, link_corpus), out, per,
                   min(self.LINK_LINES, self.DIGEST_LINES))
        self._take(self._score_sentences(weights, tail_corpus), out, per, self.DIGEST_LINES)
        # anything the reserve did not use goes back to the linked articles
        self._take(self._score_sentences(weights, link_corpus), out, per, self.DIGEST_LINES)
        return out

    def _follow_names(self, ctx: RunContext, question: str, evidence: list[str],
                      weights: dict, seen: set[str]) -> list[str]:
        """Harvest the proper names printed in the read passages and look the best few up."""
        qwords = set(_words(question))
        qnames = {n.lower() for n, _ in _mentions("q | " + question)}
        best: dict[str, float] = {}
        for passage in evidence:
            for name, sent in _mentions(passage):
                key = name.lower()
                if key in seen or key in qnames:
                    continue
                if any(key in q or q in key for q in qnames):
                    continue
                if all(w in qwords for w in _words(name)):
                    continue
                hit = set(_words(sent)) & qwords
                score = sum(weights.get(w, 0.0) for w in hit) + 0.3 * len(_words(name))
                best[name] = max(best.get(name, 0.0), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        linked: list[str] = []
        link_seen: set[str] = set(seen)
        for name, _score in ranked[: self.MAX_LINKS]:
            if ctx.tools.calls >= self.MAX_RETRIEVALS:
                break
            for p in ctx.tools.retrieve(name, k=self.K_LINK):
                key = _title_of(p).lower()
                if key and key not in link_seen:
                    link_seen.add(key)
                    linked.append(p)
        ctx.note("names followed", names=[n for n, _ in ranked[: self.MAX_LINKS]],
                 linked=len(linked))
        return linked

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
        weights = _weights(question, linking_fact, shown or evidence)
        linked = self._follow_names(ctx, question, evidence, weights, set(seen))
        digest = self._digest(question, linking_fact, tail, linked, read)
        ctx.note("deep tail distilled", tail=len(tail), linked=len(linked), lines=len(digest))
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
            "n_linked": len(linked),
            "n_selected": len(evidence),
        }
