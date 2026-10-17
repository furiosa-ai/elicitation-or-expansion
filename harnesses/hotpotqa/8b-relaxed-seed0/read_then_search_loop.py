"""A closed loop: read what has actually been fetched, then write the next query from what was read.

Every system evaluated so far is a fixed pipeline -- decompose, retrieve, screen, hop once, answer --
that plans all of its searching before it has read anything, and then spends 3 of its 12 permitted
LLM calls and 6 of its 12 permitted searches. The plan is therefore written in the question's
vocabulary, which is exactly the vocabulary that the second article does not use: a question can
describe the article it needs ("the college team in its 67th season", "the film that sampled the
song") without ever printing a word of its title, so no amount of rephrasing the question reaches it.
The one system that hopped a second time did so from a single up-front judgement about what was
missing, made while looking at a pool it never read.

This system inverts the order. It keeps a small evidence board and, on every round, one call reads
the board together with the passages that just arrived and does four things at once: it says which of
the new passages to keep, states in one line what the board now establishes, decides whether that is
enough to answer, and -- if it is not -- writes the next query. Because that query is written after
reading, it can be worded out of a name the retrieved text just printed rather than guessed from the
question, and because the decision to stop is made by the same reader, the loop spends searches only
on the questions that still need them. Rounds are capped, one search round is mandatory before any
stop, and a query that would merely repeat an earlier one is rewritten deterministically, so the loop
cannot idle.

Calls: 3-6 LLM, 2-5 retrievals.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+|short\s+)?answer\s*(?:is|:)\s*|^(?:it|that|this)\s+(?:is|was)\s+",
    re.IGNORECASE,
)
_YES = re.compile(r"^\s*(?:yes|yeah|true|correct|both)\b", re.IGNORECASE)
_NUM = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NONANSWER = re.compile(
    r"^(?:unknown|unclear|undetermined|not\s+(?:determined|specified|available|stated|found)|n/?a|none|"
    r"cannot\b|can't\b|insufficient)\b",
    re.IGNORECASE,
)
_PROPER = re.compile(r"\b(?:[A-Z][\w'’.-]*)(?:\s+(?:of|the|de|van|von|and|&)\s+[A-Z][\w'’.-]*|\s+[A-Z][\w'’.-]*)*")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PLACEHOLDER = re.compile(r"[\[\]{}<>]|\b(?:unknown|someone|somebody|the\s+(?:person|entity|name)|tbd)\b", re.IGNORECASE)
_ENOUGH = re.compile(r"\benough|sufficient|complete|done|yes\b", re.IGNORECASE)

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "another",
    "known", "called", "born", "name", "named", "many", "much", "one", "two", "first", "last", "year", "years",
    "would", "will", "can", "could", "into", "about", "between", "during", "along", "such", "some", "any", "not",
    "think", "know", "tell", "say", "said", "featured", "starred", "made", "part", "well", "including",
}


class Harness(BaseHarness):
    K = 10
    MAX_ROUNDS = 3          # search rounds after the opening one
    MIN_ROUNDS = 2          # never stop before two queries written from something read
    QUERIES_PER_ROUND = 2   # a question that compares two things is missing two articles, not one
    BOARD_MAX = 11
    KEEP_PER_ROUND = 4
    MAX_ANSWER_WORDS = 12

    READ_INSTR = (
        "You are collecting the two or three passages needed to settle `question`, one search at a time.\n"
        "`board` is what you have kept so far; `new_passages` are the results of the last search, most of "
        "them irrelevant.\n"
        "`keep`: the numbers of the NEW passages worth keeping, at most four, comma separated, most useful "
        "first; write `none` if the search returned nothing worth keeping. Keep a passage only if it is "
        "ABOUT something the question asks about, not because it repeats a word from the question.\n"
        "`established`: one line stating the facts the kept passages actually prove, naming entities "
        "explicitly. If a passage handed you the name of something the question only described, print that "
        "name here.\n"
        "`still_missing`: one line naming the single fact that is still needed to answer, or the word "
        "`nothing` when the board already contains every fact the question needs.\n"
        "`status`: exactly `ENOUGH` when the board answers the question, otherwise exactly `MORE`.\n"
        "`next_query`: when `status` is MORE, the searches to run next, one per line, at most two. Each is "
        "a title or noun phrase of at most eight words, worded the way the wanted article would be titled, "
        "and built around a name that the board or the question actually prints. When the question sets two "
        "things against each other and the board is missing an article for each of them, write one search "
        "per thing. Never write a question, never write a placeholder or brackets, and never repeat a "
        "search that has already been run (`searched` lists them). Leave it empty when `status` is ENOUGH."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.read = Module(
            inputs=["question", "board", "new_passages", "searched"],
            outputs=["keep", "established", "still_missing", "status", "next_query"],
            instructions=self.READ_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _one_line(raw: str, limit: int = 200) -> str:
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”`* ').strip()
            if line and line.lower() not in {"none", "n/a", "null"}:
                return line[:limit]
        return ""

    @classmethod
    def _some_lines(cls, raw: str, limit: int, width: int = 120) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”`* ').strip()
            if not line or line.lower() in {"none", "n/a", "null"} or line in out:
                continue
            out.append(line[:width])
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _toks(text: str) -> list[str]:
        return [w.lower() for w in _WORD.findall(text or "")]

    def _key(self, text: str) -> frozenset:
        return frozenset(w for w in self._toks(text) if w not in _STOP)

    @staticmethod
    def _title(doc: dict[str, Any]) -> str:
        return (doc.get("title") or "").strip()

    def _fresh(self, query: str, searched: list[str]) -> bool:
        k = self._key(query)
        return bool(k) and not any(k <= self._key(s) for s in searched)

    def _repair(self, board: list[dict[str, Any]], question: str, missing: str, searched: list[str]) -> str:
        """A stalled loop is restarted in code: cross a name the board printed with the words the
        question uses for the property it wants. Never a bare repeat of something already searched."""
        qk = self._key(question)
        tail_words = [w for w in _WORD.findall(missing or question) if w.lower() not in _STOP and len(w) > 2]
        lower = [w for w in tail_words if not w[:1].isupper()]
        tail = " ".join((lower or tail_words)[-4:])
        counts: dict[str, int] = {}
        for doc in board:
            for m in _PROPER.finditer(doc.get("text", "")):
                name = m.group(0).strip(" .,;:'’-")
                if not (1 <= len(name.split()) <= 5) or len(name) < 4:
                    continue
                if self._key(name) <= qk:
                    continue
                counts[name] = counts.get(name, 0) + 1
        for name, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
            for cand in (f"{name} {tail}".strip(), name):
                if self._fresh(cand, searched):
                    return cand
        return ""

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip().strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        searched: list[str] = [question]
        seen_titles: set[str] = set()
        board: list[dict[str, Any]] = []
        spare: list[dict[str, Any]] = []
        established: list[str] = []
        missing = ""

        new_docs = ctx.tools.retrieve_docs(question, k=self.K)
        for rnd in range(self.MAX_ROUNDS + 1):
            fresh = []
            for doc in new_docs:
                key = self._title(doc).lower()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    fresh.append(doc)
            board_view = [f"{i + 1}. {d['text']}" for i, d in enumerate(board)] or ["(empty)"]
            new_view = [f"{i + 1}. {d['text']}" for i, d in enumerate(fresh)] or ["(nothing new)"]

            step = self.read(
                ctx.llm,
                tag=f"read{rnd}",
                question=question,
                board=board_view,
                new_passages=new_view,
                searched=[f"- {s}" for s in searched],
            )

            kept_now = []
            for tok in _NUM.findall(step.get("keep", "") or "")[: self.KEEP_PER_ROUND]:
                idx = int(tok) - 1
                if 0 <= idx < len(fresh) and fresh[idx] not in kept_now:
                    kept_now.append(fresh[idx])
            board.extend(kept_now[: max(0, self.BOARD_MAX - len(board))])
            spare.extend(d for d in fresh[:3] if d not in kept_now)

            fact = self._one_line(step.get("established", ""))
            if fact:
                established.append(fact)
            missing = self._one_line(step.get("still_missing", "")) or missing
            status = (step.get("status", "") or "").strip().upper()
            wanted = [
                q for q in self._some_lines(step.get("next_query", ""), self.QUERIES_PER_ROUND)
                if not _PLACEHOLDER.search(q) and not q.endswith("?")
            ]

            done = status.startswith("ENOUGH") or _ENOUGH.match(status or "")
            if done and rnd >= self.MIN_ROUNDS:
                ctx.note(f"stopped after round {rnd}", established=established, board=[self._title(d) for d in board])
                break
            if rnd >= self.MAX_ROUNDS:
                break
            queries = [q for q in wanted if self._fresh(q, searched)]
            if not queries:
                repaired = self._repair(board or fresh, question, missing, searched)
                queries = [repaired] if repaired else []
            if not queries:
                break
            new_docs = []
            for q in queries[: self.QUERIES_PER_ROUND]:
                searched.append(q)
                new_docs.extend(ctx.tools.retrieve_docs(q, k=self.K))
            ctx.note(f"round {rnd} -> {queries}", missing=missing, kept=[self._title(d) for d in kept_now])

        # The board is what the reader vouched for; top-ranked leftovers backfill a thin board.
        evidence_docs = list(board)
        have = {self._title(d).lower() for d in evidence_docs}
        for doc in spare:
            if len(evidence_docs) >= max(5, len(board)):
                break
            if self._title(doc).lower() not in have:
                have.add(self._title(doc).lower())
                evidence_docs.append(doc)

        evidence = [f"{i + 1}. {d['text']}" for i, d in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = self._title(evidence_docs[0]) if evidence_docs else answer
        return {
            "answer": answer,
            "searched": searched,
            "established": established,
            "still_missing": missing,
            "n_evidence": len(evidence_docs),
        }
