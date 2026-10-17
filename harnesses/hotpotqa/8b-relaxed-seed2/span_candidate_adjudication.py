"""Answer by adjudicating between competing verbatim spans instead of writing one answer.

Same retrieval front-end as the parent pooled-decomposition harness (so the variable under test
is isolated), but the single free-form answer call is replaced by a three-stage answer
architecture built around the idea that the answer is a SPAN OF THE SOURCE, not a phrasing:

  1. ENUMERATE: one call proposes several competing answers, each given as
     `span :: the sentence it was copied from`, so alternatives (and the evidence for each) are
     laid side by side rather than collapsed into a single guess.
  2. GROUND + VARY (deterministic, no call): every proposed span is checked against the evidence
     text and discarded if it is not actually there. Each surviving span is then expanded into
     the SURFACE FORMS the evidence itself uses for the same thing -- the maximal proper-name run
     that the span is a subsequence of, a `<span>, <Place>` continuation, a preceding year, and
     any passage title the span sits inside. These variants are added as further candidates.
  3. ADJUDICATE: one call sees the numbered candidates with their source sentences and picks the
     id; the returned text is deterministically snapped back to the chosen span, so the final
     answer is always a string that occurs in the evidence.

Motivation: with the parent system half of the remaining errors already have the gold answer
sitting verbatim in the passages the answerer read, and most of the errors on items whose
evidence is complete are granularity slips -- the model writes the name it knows instead of the
name the article uses, or a plural for a singular. Making the answerer choose among grounded
surface forms turns those from generation errors into a selection problem.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
YESNO_RE = re.compile(r"^(yes|no)\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")
MARKUP = re.compile(r"[*_`\"“”«»]")
LIST_NUM = re.compile(r"^\s*\d+\s*[:.)-]\s*")
PLACEHOLDER = {"span", "sentence", "answer", "candidate", "candidates", "none", "reasoning"}

# a run of capitalised / numeric tokens, optionally joined by lower-case connectors
_W = r"[A-Z0-9][\w.'’&+-]*"
_CONN = r"(?:of|the|and|for|in|on|de|del|van|von|der|di|da|du|la|le|el)"
RUN_RE = re.compile(rf"{_W}(?:,?\s+(?:{_CONN}\s+)?{_W}){{0,7}}")
YEAR_LEFT = re.compile(r"(1[5-9]\d{2}|20[0-2]\d)\s+$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (text or "").lower())).strip()


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip()
        if len(line) >= 3 and not line.startswith("[["):
            out.append(line[:300])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


def _clean_span(text: str) -> str:
    s = MARKUP.sub("", (text or "").strip()).strip()
    return s.strip().rstrip(".,;: ").strip()


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(tok in it for tok in small)


LEAD_CHARS = 140
TAIL_PLACE = re.compile(r",\s+([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})")


def surface_variants(span: str, evidence: list[str], cap: int = 2) -> list[str]:
    """Longer forms of the span that the evidence itself uses for the same thing.

    Deliberately narrow: only the wording an article uses for its OWN subject (its title, or the
    fuller name in its opening definition) and the `<span>, <Place>` / `<year> <span>` forms
    written in running text. Anything looser offers the answerer longer strings for things the
    question was not asking about.
    """
    toks = _norm(span).split()
    if not toks or len(toks) > 6:
        return []
    out: list[str] = []
    seen = {_norm(span)}
    for passage in evidence:
        title = _title_of(passage)
        body = passage.partition(" | ")[2]
        ntitle = _norm(title)
        about_span = ntitle == _norm(span) or _is_subsequence(toks, ntitle.split())
        if about_span and ntitle != _norm(span) and 0 < len(ntitle.split()) - len(toks) <= 4:
            if ntitle not in seen:
                seen.add(ntitle)
                out.append(title)
        if about_span:
            for m in RUN_RE.finditer(body[:LEAD_CHARS]):
                run = _clean_span(m.group(0))
                nrun = _norm(run)
                if nrun in seen or not nrun:
                    continue
                if 0 < len(nrun.split()) - len(toks) <= 4 and _is_subsequence(toks, nrun.split()):
                    seen.add(nrun)
                    out.append(run)
        for m in re.finditer(re.escape(span), passage):
            pre = YEAR_LEFT.search(passage[max(0, m.start() - 6): m.start()])
            if pre:
                cand = f"{pre.group(1)} {span}"
                if _norm(cand) not in seen:
                    seen.add(_norm(cand))
                    out.append(cand)
            tail = TAIL_PLACE.match(passage[m.end(): m.end() + 40])
            if tail:
                cand = f"{span}, {tail.group(1)}"
                if _norm(cand) not in seen:
                    seen.add(_norm(cand))
                    out.append(cand)
        if len(out) >= cap:
            break
    return out[:cap]


def find_sentence(span: str, evidence: list[str]) -> str:
    """The sentence of the evidence the span was copied from ('' if it is not there)."""
    nspan = _norm(span)
    if not nspan:
        return ""
    for passage in evidence:
        if nspan not in _norm(passage):
            continue
        title, _, body = passage.partition(" | ")
        for sent in SENT_SPLIT.split(body):
            if nspan in _norm(sent):
                return f"{title.strip()}: {sent.strip()[:260]}"
        return _lead(passage, 240)
    return ""


class Harness(BaseHarness):
    K_PLAN = 8
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10
    MAX_PROPOSED = 4     # spans the enumerate call may propose
    CAND_CAP = 8         # candidates shown to the adjudicator

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
    ENUM_INSTR = (
        "List the possible answers to the question that the evidence supports. Do not settle on "
        "one yet.\n"
        "Write `candidates` as up to 4 lines, best first. Each line holds ONE possible answer and "
        "nothing else: no numbering, no label, no explanation, no quotation marks.\n"
        "Each answer must be copied character for character out of a passage, and must be a bare "
        "name, title, number, date or noun phrase -- never a sentence. Copy the wording the "
        "passage uses, including the fuller form of a name when that is how the passage writes "
        "it.\n"
        "Give a rival reading its own line whenever the evidence could be read another way, or "
        "whenever the passages spell the same thing two different ways.\n"
        "If the question asks whether something holds, the first line must be `yes` or `no`."
    )
    ADJ_INSTR = (
        "One of the numbered candidates answers the question. Pick it.\n"
        "`choice` : the id of the candidate, a single number.\n"
        "`answer` : that candidate's span, copied exactly, with nothing added or removed.\n"
        "Judge on two things: it must answer precisely what was asked, and it must be written the "
        "way the source writes it. When two candidates name the same thing, keep the one whose "
        "wording matches how the source introduces it rather than the one you would use yourself, "
        "and keep the number and tense the question asks for.\n"
        "If the question asks whether something holds, choose the `yes` or `no` candidate."
    )

    def __init__(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
        self.enumerate_ = Module(
            inputs=["question", "evidence", "linking_fact"],
            outputs=["candidates"],
            instructions=self.ENUM_INSTR,
        )
        self.adjudicate = Module(
            inputs=["question", "candidates"],
            outputs=["choice", "answer"],
            instructions=self.ADJ_INSTR,
        )

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = _norm(_title_of(p))
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

    # -- answer stage -------------------------------------------------------------------------
    def _propose(self, ctx: RunContext, question: str, evidence: list[str], fact: str) -> list[tuple[str, str]]:
        raw = self.enumerate_(
            ctx.llm, tag="enumerate", question=question, evidence=evidence, linking_fact=fact or "unknown"
        ).get("candidates", "")
        proposed: list[tuple[str, str]] = []
        for line in _split_lines(raw, self.MAX_PROPOSED + 2):
            span = _clean_span(LIST_NUM.sub("", line.partition("::")[0]))
            if not span or _norm(span) in PLACEHOLDER or len(span.split()) > 14:
                continue
            proposed.append((span, ""))
            if len(proposed) >= self.MAX_PROPOSED:
                break
        return proposed

    def _build_candidates(self, proposed: list[tuple[str, str]], evidence: list[str]) -> list[tuple[str, str]]:
        cands: list[tuple[str, str]] = []
        seen: set[str] = set()

        def push(span: str, sent: str) -> None:
            key = _norm(span)
            if key and key not in seen:
                seen.add(key)
                cands.append((span, sent))

        for span, sent in proposed:
            if YESNO_RE.match(span):
                push(YESNO_RE.match(span).group(1).lower(), sent)
                continue
            grounded = find_sentence(span, evidence)
            if not grounded:
                continue  # not actually in the evidence
            push(span, sent or grounded)
            for var in surface_variants(span, evidence):
                push(var, find_sentence(var, evidence) or grounded)
        if not cands:  # nothing survived grounding -- fall back to the raw proposals
            for span, sent in proposed:
                push(span, sent)
        return cands[: self.CAND_CAP]

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()

        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K_PLAN))
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_PLAN))

        linking_fact, ids = "", []
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
            before = len(pool)
            self._add(pool, seen, ctx.tools.retrieve(nq[0], k=self.K_FOLLOW))
            if len(pool) == before:
                break
            pool = [pool[i - 1] for i in ids] + [p for i, p in enumerate(pool, 1) if i not in ids]

        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)] or shown[: self.SELECT_CAP]

        # --- contrastive span answering -------------------------------------------------------
        proposed = self._propose(ctx, question, evidence, linking_fact)
        cands = self._build_candidates(proposed, evidence)
        if not cands:
            return {"answer": "", "queries": queries, "n_candidates": 0}
        if len(cands) == 1:
            ans = cands[0][0]
            choice = 1
        else:
            listing = [f"{i + 1}. {s}   [source: {sent}]" for i, (s, sent) in enumerate(cands)]
            out = self.adjudicate(ctx.llm, tag="adjudicate", question=question, candidates=listing)
            picked = self._parse_ids(out.get("choice", ""), len(cands))
            choice = picked[0] if picked else 1
            ans = _clean_span(LIST_NUM.sub("", _clean_span(out.get("answer", "").split("\n")[0])))
            spans = {_norm(s) for s, _ in cands}
            if _norm(ans) not in spans:  # snap back onto a grounded span
                ans = cands[choice - 1][0]
        ctx.note("span candidates", candidates=[s for s, _ in cands], choice=choice)
        return {
            "answer": ans,
            "queries": queries,
            "linking_fact": linking_fact,
            "n_candidates": len(cands),
            "candidates": [s for s, _ in cands],
        }
