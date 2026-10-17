"""Search the index in article space instead of question space.

The index is a keyword index over `title | lead paragraph` records, so a query scores best when it
looks like the *article* being sought, not like the question being asked. Every system so far
queries in question space: it splits the question into sub-questions and sends those words to the
engine. That works for the entity the question names outright and fails for the entity the question
only describes ("the 1960s adventure film that starred ...") -- the descriptor words are spread over
thousands of articles, so the one article that matters never surfaces at any depth.

This harness asks the model to write down, before any search, WHAT ARTICLES it expects to exist:

  1. one naming call returns
       `article_titles` : the exact headword each needed article would be filed under -- the
                          canonical name for entities the question states, and the model's best
                          guess at the name for an entity the question merely describes;
       `lead_sentence`  : a hypothetical opening sentence of the article that would contain the
                          answer, written in encyclopaedic register, used verbatim as a query so the
                          engine matches article prose against article prose;
  2. each hypothesised title and the hypothetical lead are searched alongside the raw question and
     merged into a title-keyed pool;
  3. a triage call selects the passages worth reading, reports the linking fact and, when evidence
     is missing, names the TITLE of the article that is still missing rather than inventing more
     question words -- the follow-up hop stays in article space too;
  4. the answer is read off the raw selected passages.

Hypothesis: a wrong title guess is harmless (its hits are simply not selected) while a right one
lands the article at rank 1, so moving the query distribution from question space to article space
raises recall at unchanged depth and unchanged answering context.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
LABEL = re.compile(r"^(?:title|article|entity|headword|name)\s*\d*\s*[:\-]\s*", re.IGNORECASE)
UNKNOWN = ("unknown", "none", "n/a", "na", "unclear", "not applicable", "")


def _clean(line: str) -> str:
    line = BULLET.sub("", line or "").strip()
    line = LABEL.sub("", line).strip()
    return line.strip().strip('"').strip("'").strip("[]").strip()


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _clean(raw)
        if len(line) >= 2 and line.lower() not in UNKNOWN:
            out.append(line[:220])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {' '.join(body.split())[:chars]}"


class Harness(BaseHarness):
    K = 8               # depth per query, same as the parent
    MAX_TITLES = 3      # hypothesised article titles searched
    MAX_TRIAGE = 2      # adaptive hops
    POOL_CAP = 30       # candidates shown to triage
    SELECT_CAP = 8      # passages handed to the answerer

    NAME_INSTR = (
        "An encyclopaedia is searched by matching words against `title | opening paragraph` records. "
        "Before searching, say what articles you expect to exist.\n"
        "`article_titles` : one article headword per line, 2 or 3 lines, no numbering and no "
        "explanation. Give the canonical headword for every entity the question names outright "
        "(the name the article itself would be filed under, disambiguating parenthesis included if "
        "the plain name is ambiguous). If the question only describes an entity instead of naming "
        "it, write your single best guess at that entity's name from your own knowledge; a wrong "
        "guess is acceptable, a description is not -- never write a phrase like `the film that ...`.\n"
        "`lead_sentence`  : one sentence, written the way the opening sentence of the article "
        "holding the answer would be written -- subject first, then what it is, then the "
        "defining details the question mentions. Use encyclopaedic wording, not question wording."
    )
    TRIAGE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia "
        "leads, shown as `id: title | beginning of the article`.\n"
        "`relevant_ids` : the ids worth reading in full, most useful first, at most 6, comma "
        "separated.\n"
        "`linking_fact` : the fact the pool establishes that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); `unknown` if none.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`.\n"
        "`missing_title`: if not sufficient, the headword of the article that is still missing -- "
        "a bare article name, taken from `linking_fact` when the pool has just revealed the entity, "
        "otherwise your best guess at its name. Never write a description or a question; leave it "
        "empty if the pool is sufficient."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    def __init__(self) -> None:
        self.name = Module(
            inputs=["question"],
            outputs=["article_titles", "lead_sentence"],
            instructions=self.NAME_INSTR,
        )
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "missing_title"],
            instructions=self.TRIAGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )

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

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()

        # 1. name the articles that ought to exist ---------------------------------------------
        named = self.name(ctx.llm, tag="name", question=question)
        titles = _split_lines(named.get("article_titles", ""), self.MAX_TITLES)
        lead = " ".join(_split_lines(named.get("lead_sentence", ""), 1))

        # 2. search in article space, plus the raw question as a safety net --------------------
        asked: set[str] = set()
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K))
        for t in titles:
            key = t.lower()
            if key in asked:
                continue
            asked.add(key)
            self._add(pool, seen, ctx.tools.retrieve(t, k=self.K))
        if lead and lead.lower() not in asked:
            asked.add(lead.lower())
            self._add(pool, seen, ctx.tools.retrieve(lead, k=self.K))

        # 3. triage; the follow-up hop stays in article space -----------------------------------
        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_TRIAGE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.triage(ctx.llm, tag=f"triage{step + 1}", question=question, pool=table)
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown))[: self.SELECT_CAP] or list(
                range(1, min(len(shown), 5) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in UNKNOWN:
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_TRIAGE - 1:
                break
            nxt = _split_lines(out.get("missing_title", ""), 1)
            if not nxt or nxt[0].lower() in asked:
                break
            asked.add(nxt[0].lower())
            followups.append(nxt[0])
            picked = [pool[i - 1] for i in ids]
            before = len(pool)
            self._add(pool, seen, ctx.tools.retrieve(nxt[0], k=self.K))
            if len(pool) == before:
                break
            keep = {_title_of(p).lower() for p in picked}
            pool = picked + [p for p in pool if _title_of(p).lower() not in keep]
            ctx.note("article-space follow-up", query=nxt[0], pool_size=len(pool))

        # 4. answer from raw passages -----------------------------------------------------------
        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[:5]
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
            "hypothesised_titles": titles,
            "lead_query": lead,
            "followups": followups,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "n_selected": len(evidence),
        }
