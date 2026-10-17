"""Retrieval as browsing: scan a wide index of article NAMES, then open a few of them in full.

Mechanism change vs the parent (composed bridge hop). The parent reads whole abstracts, so the number
of candidate articles it can consider is set by how much text fits in a prompt: three searches at k=8
give it about twenty-four candidates, and its own traces show the article it needs is often simply not
among them. A sibling system widened that pool by compressing every candidate to two sentences, which
roughly doubled it.

This system drops the idea that a candidate must be read at all in order to be judged. An encyclopedia
article's NAME is written to identify it uniquely -- it carries the entity's spelling, and, when the
name is ambiguous, an explicit parenthetical saying which sense it is. Six tokens of name therefore
distinguish candidates about as well as a hundred tokens of prose, so the same prompt that held two
dozen abstracts holds a hundred names. The system runs its searches at the largest k the index allows,
shows the model nothing but a numbered index of names, and asks it to OPEN the handful whose articles
would state a fact the question needs. Opening costs nothing extra: the text came back with the name.
The same call also says what to search for next, having just seen a hundred names' worth of evidence
about what the index does and does not contain; the new names are indexed and browsed a second time,
and only the opened articles -- untruncated -- reach the answerer.

Calls: 4 LLM, 5 retrievals.
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
    r"^(?:unknown|unclear|undetermined|not\s+(?:determined|specified|available|stated|found|present|listed|"
    r"mentioned|given|provided|in\s)|no\s+(?:information|answer|article)|n/?a|none|"
    r"cannot\b|can't\b|insufficient)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z0-9']+")

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "another",
}


class Harness(BaseHarness):
    K = 20                    # the widest scan the index allows: names are cheap to look at
    MAX_FIRST_QUERIES = 3
    MAX_FOLLOWUPS = 2
    INDEX_CAP = 110           # names shown in one browse
    OPEN_1 = 6
    OPEN_2 = 4
    MIN_OPEN = 4
    MAX_ANSWER_WORDS = 12

    DECOMPOSE_INSTR = (
        "A question of this kind can only be settled by combining facts from two different articles, "
        "and one search phrased from the whole question tends to find only one of them.\n"
        "Write `queries`: two or three search queries, one per line, no numbering, no explanation.\n"
        "- Give each distinct named entity, work or event mentioned in the question its own query, "
        "worded the way that entity's own article would be titled.\n"
        "- If the question describes an entity instead of naming it, write one query out of that "
        "description using the words an article about it would use.\n"
        "- Do not repeat the question verbatim as a query, and do not write questions: write titles "
        "and noun phrases."
    )

    BROWSE_INSTR = (
        "`index` is a numbered list of article NAMES from an encyclopedia -- names only, no text. Most are "
        "irrelevant. You are browsing the index to decide which articles are worth reading.\n"
        "`open_ids`: the numbers of the articles you want to read, best first, at most six numbers separated "
        "by commas. An article is worth opening when the question needs a fact that THIS article would state: "
        "the article about an entity the question names, and the article about the entity that holds the fact "
        "the question is really asking for. A name in brackets after the title says which sense of an ambiguous "
        "name it is -- use it, and do not open a same-named article of the wrong sense, wrong year or wrong "
        "kind. Do not open an article merely because its name repeats words of the question.\n"
        "`next_queries`: one or two further searches, one per line, for an article the index does not seem to "
        "list yet -- write it as the name and defining words that article would itself use. If everything "
        "needed is already listed, write none."
    )

    BROWSE2_INSTR = (
        "`opened` names the articles already chosen. `index` is a fresh numbered list of article NAMES the "
        "latest searches turned up -- names only, no text.\n"
        "`open_ids`: the numbers of any of these fresh articles that would state a fact the question still "
        "needs, best first, at most four numbers separated by commas. Prefer an article whose name IS the "
        "entity the question is asking about over one that merely mentions it. If none of them adds anything, "
        "write none."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `articles`.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the articles spell it, or the single word yes or no when the question asks whether something "
        "holds. Copy a date whole, in the form the article prints it, rather than reducing it to the year. "
        "Never write a sentence, never restate the question, never explain. If the articles are "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.browse = Module(
            inputs=["question", "index"], outputs=["open_ids", "next_queries"], instructions=self.BROWSE_INSTR
        )
        self.browse2 = Module(
            inputs=["question", "opened", "index"], outputs=["open_ids"], instructions=self.BROWSE2_INSTR
        )
        self.answer = Module(inputs=["question", "articles"], outputs=["answer"], instructions=self.ANSWER_INSTR)

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”').strip()
            if not line or line.lower().strip(".") in {"none", "n/a", "null", "no", "nothing"}:
                continue
            if len(line) > 300:
                line = line[:300]
            if line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _key(text: str) -> frozenset:
        return frozenset(w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP)

    def _scan(self, ctx: RunContext, queries: list[str], docs: dict[str, str], order: list[str]) -> list[str]:
        """Run searches at full width and record every NEW article name, keeping its text aside."""
        fresh: list[str] = []
        for q in queries:
            for h in ctx.tools.retrieve_docs(q, k=self.K):
                title = (h.get("title") or "").strip()
                key = title.lower()
                if not title or key in docs:
                    continue
                docs[key] = h.get("text") or f"{title} |"
                order.append(key)
                fresh.append(key)
        return fresh

    @staticmethod
    def _pick(raw: str, names: list[str], limit: int) -> list[str]:
        out: list[str] = []
        for tok in _NUM.findall(raw or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(names) and names[idx] not in out:
                out.append(names[idx])
            if len(out) >= limit:
                break
        return out

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip()
        ans = ans.strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if not queries:
            queries = [question]

        docs: dict[str, str] = {}
        order: list[str] = []
        self._scan(ctx, queries, docs, order)
        # Round-robin over the searches is already implicit in scan order; cap what one browse sees.
        shown = order[: self.INDEX_CAP]
        index = [f"{i + 1}. {docs[k].split('|', 1)[0].strip()}" for i, k in enumerate(shown)]
        ctx.note(f"index scan: {len(queries)} searches -> {len(order)} article names", queries=queries)

        seen = self.browse(ctx.llm, tag="browse", question=question, index=index)
        opened = self._pick(seen.get("open_ids", ""), shown, self.OPEN_1)
        follow = [
            q for q in self._lines(seen.get("next_queries", ""), self.MAX_FOLLOWUPS)
            if self._key(q) and not any(self._key(q) <= self._key(p) for p in queries)
        ]

        if follow:
            fresh = self._scan(ctx, follow, docs, order)
            shown2 = fresh[: self.INDEX_CAP]
            if shown2:
                index2 = [f"{i + 1}. {docs[k].split('|', 1)[0].strip()}" for i, k in enumerate(shown2)]
                names = [docs[k].split("|", 1)[0].strip() for k in opened] or ["(none yet)"]
                seen2 = self.browse2(
                    ctx.llm, tag="browse2", question=question, opened=names, index=index2
                )
                for k in self._pick(seen2.get("open_ids", ""), shown2, self.OPEN_2):
                    if k not in opened:
                        opened.append(k)
            ctx.note(f"followup: {follow} -> {len(fresh)} new names", opened=len(opened))

        # Never answer from an empty desk: fall back to what the index ranked highest.
        for k in order:
            if len(opened) >= self.MIN_OPEN:
                break
            if k not in opened:
                opened.append(k)

        articles = [f"{i + 1}. {docs[k]}" for i, k in enumerate(opened)]
        out = self.answer(ctx.llm, tag="answer", question=question, articles=articles)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = docs[opened[0]].split("|", 1)[0].strip() if opened else answer
        return {
            "answer": answer,
            "queries": queries,
            "next_queries": follow,
            "n_indexed": len(order),
            "n_opened": len(opened),
        }
