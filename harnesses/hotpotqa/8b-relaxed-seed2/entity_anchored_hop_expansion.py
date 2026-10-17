"""Follow named entities to their own articles instead of reformulating a keyword query.

Retrieval architecture change over the parent pooled-decomposition harness. The parent grows its
evidence pool with ONE model-invented follow-up query; this harness replaces that step with an
explicit link-following mechanism that imitates how a reader navigates an encyclopaedia:

  1. planning + pooled retrieval build a title-keyed candidate pool (as in the parent);
  2. a triage call selects the passages worth reading and reports the linking fact;
  3. MENTION HARVEST (deterministic): every proper-name / quoted-title mention occurring in the
     SELECTED passages is extracted with a regex, mentions whose own article is already in the
     pool or that merely repeat the question are dropped, and the survivors are ranked by how
     often they are mentioned;
  4. LINK SELECTION: one call picks the handful of harvested mentions whose own article would
     supply the still-missing fact -- i.e. which link to click, not which words to search;
  5. each picked mention is issued to the index AS A BARE NAME, which is what makes the index
     return that entity's own article rather than more articles about the first entity;
  6. a second triage re-selects over the enlarged pool and the answer is read off raw passages.

Motivation: on the parent system the bridge article is missing from the pool for roughly a third
of the items, and for well over half of those the missing article's exact title is already
written inside a passage the parent had already retrieved and read. The fact needed is one
lookup away, but a reformulated keyword query keeps landing on the entity that was already
found. Anchoring the extra lookups on harvested mentions turns those into direct hits.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)

# --- mention harvesting ---------------------------------------------------------------------
_W = r"[A-Z][\w.'’&+-]*"
_CONN = r"(?:of|the|and|for|in|on|at|de|del|van|von|der|di|da|du|la|le|el|dos|bin|ibn)"
MENTION_RE = re.compile(rf"{_W}(?:\s+(?:{_CONN}\s+)?{_W}){{0,6}}")
QUOTED_RE = re.compile(r"[\"“]([^\"“”]{2,70})[\"”]")
TRAIL_CONN = re.compile(rf"\s+{_CONN}$", re.IGNORECASE)

# sentence-initial function words and calendar words that look like proper names
BLOCK = {
    "the", "a", "an", "he", "she", "it", "they", "this", "that", "these", "those", "his", "her",
    "its", "their", "there", "then", "when", "while", "after", "before", "however", "although",
    "and", "but", "for", "in", "on", "at", "of", "to", "by", "from", "with", "as", "is", "was",
    "were", "are", "be", "been", "has", "have", "had", "also", "both", "who", "which", "what",
    "she", "we", "you", "i", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "american", "british", "english", "french",
    "german", "one", "two", "three", "first", "second", "third", "new", "st", "mr", "mrs", "dr",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (text or "").lower())).strip()


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


def _body_of(passage: str) -> str:
    return passage.partition(" | ")[2]


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


SENT_BREAK = re.compile(r"(?<=[a-z0-9]{3}[.!?])\s+")
CONJ = re.compile(r"\s+(?:and|&|or)\s+")


def _clean_mention(raw: str) -> str:
    s = TRAIL_CONN.sub("", (raw or "").strip().strip(".,;:'’ "))
    return re.sub(r"\s+", " ", s).strip()


def _variants(raw: str) -> list[str]:
    """A raw regex match may run across a sentence break or join two names with `and`.

    Emit the pieces as well as the whole, so a conjoined phrase still yields each name.
    """
    out: list[str] = []
    for chunk in SENT_BREAK.split(raw):
        chunk = _clean_mention(chunk)
        if not chunk:
            continue
        out.append(chunk)
        parts = [_clean_mention(p) for p in CONJ.split(chunk)]
        if len(parts) > 1:
            out.extend(p for p in parts if p)
    return out


def harvest_mentions(passages: list[str], question: str, known_titles: set[str], cap: int) -> list[str]:
    """Proper-name mentions inside `passages` whose own article is not in the pool yet."""
    q = _norm(question)
    counts: Counter[str] = Counter()
    surface: dict[str, str] = {}
    order: dict[str, int] = {}
    for passage in passages:
        body = _body_of(passage)
        cands = [m.group(1) for m in QUOTED_RE.finditer(body)]
        cands += [m.group(0) for m in MENTION_RE.finditer(body)]
        for raw in [v for c in cands for v in _variants(c)]:
            name = raw
            key = _norm(name)
            if not key or len(key) < 3 or key in known_titles:
                continue
            words = key.split()
            if words[0] in BLOCK and len(words) == 1:
                continue
            if all(w in BLOCK for w in words):
                continue
            if key in q or key.isdigit():
                continue
            counts[key] += 1
            if key not in surface or len(name) > len(surface[key]):
                surface[key] = name
            order.setdefault(key, len(order))
    ranked = sorted(counts, key=lambda k: (-counts[k], order[k]))
    return [surface[k] for k in ranked[:cap]]


class Harness(BaseHarness):
    K_PLAN = 8          # hits per decomposed query
    K_LINK = 6          # hits per entity lookup
    MAX_QUERIES = 3     # decomposed queries issued alongside the raw question
    POOL_CAP = 30       # candidates shown to triage
    SELECT_CAP = 10     # passages handed to the answerer
    MENTION_CAP = 18    # harvested mentions offered to link selection
    MAX_LINKS = 4       # entity articles fetched per item

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
        "`missing_fact` : what still has to be looked up before the question can be answered; "
        "say `nothing` if the pool already contains everything.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`."
    )
    LINK_INSTR = (
        "The evidence read so far mentions the names below. Each name has its own article in the "
        "encyclopaedia, which has not been read yet.\n"
        "Decide which of those articles to open in order to obtain `missing_fact`.\n"
        "`link_ids` : the ids of the names to open, most promising first, at most 4, comma "
        "separated. Choose the name that the question refers to indirectly but never names, or "
        "whose article would state the property being asked for (a birth date, a nationality, a "
        "location, a creator, a number, a parent work). Prefer a specific full name over a "
        "generic phrase. Open an article whenever the question turns on something about that "
        "name that the evidence only mentions in passing. Write `none` only when every thing the "
        "question depends on has already been read in its own article."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "missing_fact", "sufficient"],
            instructions=self.TRIAGE_INSTR,
        )
        self.link = Module(
            inputs=["question", "missing_fact", "names"],
            outputs=["link_ids"],
            instructions=self.LINK_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> int:
        added = 0
        for p in passages:
            key = _norm(_title_of(p))
            if key and key not in seen:
                seen.add(key)
                pool.append(p)
                added += 1
        return added

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    def _triage_step(self, ctx: RunContext, question: str, pool: list[str], tag: str):
        shown = pool[: self.POOL_CAP]
        table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
        out = self.triage(ctx.llm, tag=tag, question=question, pool=table)
        ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
            range(1, min(len(shown), self.SELECT_CAP) + 1)
        )
        selected = [shown[i - 1] for i in ids[: self.SELECT_CAP]]
        fact = (out.get("linking_fact", "") or "").strip()
        if fact.lower() in ("unknown", "none", "n/a", ""):
            fact = ""
        missing = (out.get("missing_fact", "") or "").strip().split("\n")[0][:200]
        enough = bool(AFFIRM.match(out.get("sufficient", "")))
        return selected, fact, missing, enough

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()

        # 1. decompose -------------------------------------------------------------------------
        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )

        # 2. pooled retrieval ------------------------------------------------------------------
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K_PLAN))
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_PLAN))

        # 3. triage ------------------------------------------------------------------------------
        selected, linking_fact, missing, enough = self._triage_step(ctx, question, pool, "triage1")

        # 4-5. link following ---------------------------------------------------------------------
        # the sufficiency self-assessment is unreliable -- it claims the pool is complete on
        # roughly 40% of the items whose bridge article is in fact missing -- so the decision of
        # whether to open anything is left to the link selector itself, which may answer `none`.
        followed: list[str] = []
        if True:
            mentions = harvest_mentions(selected, question, seen, self.MENTION_CAP)
            if mentions:
                names = [f"{i + 1}: {m}" for i, m in enumerate(mentions)]
                out = self.link(
                    ctx.llm,
                    tag="link",
                    question=question,
                    missing_fact=(missing if not enough else "") or "confirm the answer",
                    names=names,
                )
                picks = self._parse_ids(out.get("link_ids", ""), len(mentions))[: self.MAX_LINKS]
                grew = 0
                for i in picks:
                    name = mentions[i - 1]
                    followed.append(name)
                    grew += self._add(pool, seen, ctx.tools.retrieve(name, k=self.K_LINK))
                if grew:
                    ctx.note("followed links", names=followed, new_docs=grew, pool_size=len(pool))
                    # keep already-selected evidence in front so it survives the pool cap
                    rest = [p for p in pool if p not in selected]
                    pool = selected + rest
                    selected, fact2, _, _ = self._triage_step(ctx, question, pool, "triage2")
                    linking_fact = fact2 or linking_fact

        # 6. answer from raw passages ------------------------------------------------------------
        evidence = selected[: self.SELECT_CAP] or pool[: self.SELECT_CAP]
        ans = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return {
            "answer": ans,
            "queries": queries,
            "followed_links": followed,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "n_selected": len(evidence),
        }
