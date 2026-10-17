"""Screen a much wider candidate pool by compressing each article to its identifying sentences.

Mechanism change vs the parent (decomposed multi-query + LLM passage rerank): what limits the pool the
parent can consider is not the retrieval budget -- it uses three of twelve searches -- but the reading
budget of the screening call, which sees whole abstracts and therefore only ~24 candidates. Its own
traces show the second article is often simply never fetched, while the pooled output of the different
query styles tried across systems contains it far more often than any one style does.

So this system separates SCREENING from ANSWERING. It fires more, shorter-ranked queries and pools far
more articles, then compresses every candidate down to the two sentences that identify it and connect
it to the question -- a lead sentence, which is what tells the reader who or what the article is about,
plus the highest-scoring sentence under an inverse-document-frequency weighting computed over the pool
itself. Roughly twice as many candidates then fit in the same screening prompt. The passages the screen
selects are handed to the answerer in FULL, uncompressed text, so no answer span can be lost to the
compression; compression is a device for looking at more candidates, not for reading them.

Calls: 3 LLM, 4-6 retrievals.
"""

from __future__ import annotations

import math
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
_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(\d])')
_WORD = re.compile(r"[A-Za-z0-9']+")

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "one", "two",
    "would", "will", "can", "could", "into", "about", "between", "during", "such", "some", "any", "not", "s",
}


class Harness(BaseHarness):
    K1 = 12
    K2 = 10
    MAX_FIRST_QUERIES = 4
    MAX_FOLLOWUPS = 2
    POOL_CAP = 46
    MAX_KEPT = 9
    MIN_KEPT = 5
    SENTS_PER_CANDIDATE = 2
    MAX_SENT_WORDS = 42
    MAX_ANSWER_WORDS = 12

    DECOMPOSE_INSTR = (
        "A question of this kind can only be settled by combining facts from two different articles, and one "
        "search phrased from the whole question tends to find only one of them.\n"
        "Write `queries`: three or four search queries, one per line, no numbering, no explanation.\n"
        "- Give each distinct named entity, work or event mentioned in the question its own query, worded the "
        "way that entity's own article would be titled.\n"
        "- If the question describes an entity instead of naming it, write one query out of that description "
        "using the words an article about it would use.\n"
        "- Write one query for the fact the question is asking for, worded as the sentence that states it "
        "would be worded.\n"
        "- Do not repeat the question verbatim as a query, and do not write questions: write titles and noun "
        "phrases."
    )

    SCREEN_INSTR = (
        "`candidates` is a numbered list of search results, most of them irrelevant. Each is shown as a title "
        "followed by an extract of its article, so judge it by what the title is ABOUT, not by superficial word "
        "overlap with the question.\n"
        "`useful_ids`: the numbers of the candidates that carry a fact the question needs, most relevant first, "
        "at most six numbers separated by commas. The question needs facts from two different articles, so try "
        "to name both of them.\n"
        "`missing_info`: in one line, the fact the question still needs and that no listed candidate states. "
        "Write exactly `none` if the candidates together already settle the question.\n"
        "`followup_queries`: if something is missing, one or two queries, one per line. Each must START with the "
        "name of the intermediate entity the candidates just revealed (the person, work, place or organisation "
        "that links the question to the missing fact) and then add two or three words naming the missing "
        "property, because the article that states the fact is the one containing both. Write titles and noun "
        "phrases, never questions. Write exactly `none` when nothing is missing."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.screen = Module(
            inputs=["question", "candidates"],
            outputs=["useful_ids", "missing_info", "followup_queries"],
            instructions=self.SCREEN_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”').strip()
            if not line or line.lower() in {"none", "n/a", "null"}:
                continue
            if len(line) > 300:
                line = line[:300]
            if line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP and len(w) > 1]

    def _merge(self, ranked_lists: list[list[str]], cap: int) -> list[str]:
        """Round-robin over the per-query result lists so no single query monopolises the pool."""
        pool: list[str] = []
        seen: set[str] = set()
        depth = max((len(r) for r in ranked_lists), default=0)
        for i in range(depth):
            for lst in ranked_lists:
                if i >= len(lst):
                    continue
                key = self._title_of(lst[i])
                if key and key not in seen:
                    seen.add(key)
                    pool.append(lst[i])
                    if len(pool) >= cap:
                        return pool
        return pool

    def _compress(self, pool: list[str], question: str) -> list[str]:
        """Each candidate -> title + its lead sentence + its best question-bearing sentence.

        Term weights are inverse document frequencies computed over the pool itself, so a word that
        every candidate shares (it came back for the same query) counts for nothing and a word only
        one candidate has counts for a lot.
        """
        bodies = [p.split("|", 1)[1].strip() if "|" in p else p for p in pool]
        n = len(pool) or 1
        df: dict[str, int] = {}
        for b in bodies:
            for t in set(self._terms(b)):
                df[t] = df.get(t, 0) + 1
        qterms = set(self._terms(question))
        weight = {t: math.log((1.0 + n) / (1.0 + df.get(t, 0))) + 0.3 for t in qterms}

        out: list[str] = []
        for passage, body in zip(pool, bodies):
            title = passage.split("|", 1)[0].strip()
            sents = [s.strip() for s in _SENT.split(body) if s.strip()] or [body]
            lead = sents[0]
            best, best_score = None, 0.0
            for s in sents[1:]:
                present = qterms & set(self._terms(s))
                if not present:
                    continue
                score = sum(weight[t] for t in present) / math.sqrt(len(self._terms(s)) + 8)
                if score > best_score:
                    best, best_score = s, score
            picked = [lead] + ([best] if best and self.SENTS_PER_CANDIDATE > 1 else [])
            shown = []
            for s in picked:
                words = s.split()
                shown.append(" ".join(words[: self.MAX_SENT_WORDS]) + ("…" if len(words) > self.MAX_SENT_WORDS else ""))
            out.append(f"{title} | " + " ".join(shown))
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
        if question not in queries:
            queries = queries[: self.MAX_FIRST_QUERIES - 1] + [question] if queries else [question]

        results = [ctx.tools.retrieve(q, k=self.K1) for q in queries]
        pool = self._merge(results, self.POOL_CAP)
        ctx.note(f"wide pool: {len(queries)} queries -> {len(pool)} unique candidates", queries=queries)

        # SCREEN over the compressed view: twice the candidates for the same reading budget.
        compressed = self._compress(pool, question)
        judged = self.screen(
            ctx.llm,
            tag="screen",
            question=question,
            candidates=[f"{i + 1}. {c}" for i, c in enumerate(compressed)],
        )
        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]  # FULL text, not the compressed view

        followups = self._lines(judged.get("followup_queries", ""), self.MAX_FOLLOWUPS)
        missing = (judged.get("missing_info", "") or "").strip().lower()
        sufficient = (not followups) or missing.startswith("none")
        if not sufficient:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for q in followups:
                for p in ctx.tools.retrieve(q, k=self.K2):
                    key = self._title_of(p)
                    if key and key not in seen2:
                        seen2.add(key)
                        round2.append(p)
            ctx.note(f"round2: {len(followups)} followups -> {len(round2)} new passages", followups=followups)
            kept = kept + round2[: self.MAX_KEPT]

        if len(kept) < self.MIN_KEPT:
            have = {self._title_of(p) for p in kept}
            for p in pool:
                if self._title_of(p) not in have:
                    kept.append(p)
                    have.add(self._title_of(p))
                if len(kept) >= self.MIN_KEPT:
                    break

        evidence = [f"{i + 1}. {p}" for i, p in enumerate(kept)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = kept[0].split("|", 1)[0].strip() if kept else answer
        return {
            "answer": answer,
            "queries": queries,
            "pool_size": len(pool),
            "followups": followups,
            "sufficient": sufficient,
            "n_evidence": len(kept),
        }
