"""Second hop queries composed as `<name lifted from the text> + <the property still missing>`.

Mechanism change vs the parent (decomposed multi-query + rerank + sufficiency gate): the parent
decides *whether* to hop with a free-text sufficiency check and, when it does, lets the model write
the follow-up query in free text. Two things go wrong, both visible in its traces: the gate declares
the pool sufficient on questions whose second article was never retrieved, so no hop happens at all;
and the follow-ups that are written are guesses at the answer or a placeholder for a name the model
never learned.

This system removes the gate and changes what a follow-up query IS. The retrieval index scores whole
abstracts, so a query that is only an entity name is answered by the many short articles that repeat
that name rather than by the entity's own article -- a bare name is a bad query. A name paired with
the property the question is still missing is a good one, because the entity's own article is the
document that contains both. So the model is asked for two things it can actually read off the page
(a verbatim name span for the pivot entity, and the property the question asks about), the code
composes the cross-product of those two into queries deterministically, and the hop is unconditional.

Calls: 3 LLM, 5-7 retrievals.
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
_PLACEHOLDER = re.compile(r"[\[\]{}<>]|\b(?:name|entity|person|someone|unknown|the\s+star|his|her|their)\b", re.IGNORECASE)

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "another",
    "known", "called", "born", "name", "named", "many", "much", "one", "two", "first", "last", "year", "years",
    "would", "will", "can", "could", "into", "about", "between", "during", "along", "such", "some", "any", "not",
    "think", "know", "tell", "say", "said", "does", "featured", "starred", "made", "part", "well", "including",
}


class Harness(BaseHarness):
    K1 = 8
    K2 = 8
    MAX_FIRST_QUERIES = 3
    MAX_BRIDGES = 2
    MAX_HOP_QUERIES = 3
    MAX_KEPT = 9
    MIN_KEPT = 5
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

    BRIDGE_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant. The question needs a "
        "fact from an article that is probably NOT in the pool yet; your job is to say what to look up next, "
        "using only what the pool already prints.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant first, "
        "at most six numbers separated by commas. Judge relevance by whether the passage is about an entity the "
        "question actually asks about, not by superficial word overlap.\n"
        "`pivot_names`: one or two names, one per line. A pivot is the person, work, place or organisation that "
        "stands between the question and the fact it wants -- the one whose own article would state that fact. "
        "Copy each name EXACTLY as some passage spells it, or exactly as the question spells it; never invent a "
        "name, never write a description, never write brackets or a placeholder. If the question names two things "
        "and asks how they compare, write both of those names.\n"
        "`missing_property`: two to five words naming the property the question is asking for, in the vocabulary "
        "an encyclopedia article would use, not as a question. Never write `none`."
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
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "pivot_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
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

    def _add(self, pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = self._title_of(p)
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    @staticmethod
    def _key(text: str) -> frozenset:
        return frozenset(w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP)

    def _content_words(self, question: str, limit: int = 4) -> str:
        """The words the question uses for the property it asks about, minus the entities it names."""
        words = [w for w in _WORD.findall(question) if w.lower() not in _STOP and len(w) > 2]
        # Names are usually capitalised mid-sentence; the asked-for property usually is not.
        lower = [w for w in words if not w[:1].isupper()]
        picked = (lower or words)[-limit:]
        return " ".join(picked)

    def _harvest(self, passages: list[str], question: str, limit: int) -> list[str]:
        """Fallback pivots: proper-noun runs the retrieved text prints, that the question does not."""
        qk = self._key(question)
        counts: dict[str, int] = {}
        for p in passages:
            body = p.split("|", 1)[1] if "|" in p else p
            for m in _PROPER.finditer(body):
                name = m.group(0).strip(" .,;:'’-")
                toks = name.split()
                if not (2 <= len(toks) <= 5) or len(name) < 5:
                    continue
                if self._key(name) <= qk:
                    continue
                counts[name] = counts.get(name, 0) + 1
        return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]

    def _compose(self, pivots: list[str], prop: str, question: str, already: list[str]) -> list[str]:
        """Cross a name with the missing property. A bare name is never issued: the index scores whole
        abstracts, so a name on its own is won by articles that merely repeat it."""
        seen = [self._key(q) for q in already]
        out: list[str] = []
        base = [t for t in (prop.strip(), self._content_words(question)) if t]
        if not base:
            base = [""]

        def tails_for(name: str) -> list[str]:
            """When the question pivots on two entities, the property it asks about is often stated
            with the OTHER entity's name inside it. Gluing that onto this name yields a query about
            neither article, so each name is crossed only with the part of the property that is not
            another name."""
            foreign = set()
            for other in pivots:
                if other != name:
                    foreign |= {w.lower() for w in _WORD.findall(other)}
            if not foreign:
                return base
            cleaned = []
            for tail in base:
                words = [w for w in tail.split() if w.strip(".,'’-").lower() not in foreign]
                cleaned.append(" ".join(words))
            return [t for t in cleaned if t] or [""]

        for name in pivots:
            for tail in tails_for(name):
                q = f"{name} {tail}".strip()
                k = self._key(q)
                # Drop only a query that adds nothing an earlier one already asked for; a query that
                # merely EXTENDS an earlier one is the whole point here and must survive.
                if not k or any(k <= s for s in seen):
                    continue
                seen.append(k)
                out.append(q)
                break  # one composition per pivot first; extras appended below
        for name in pivots:
            for tail in tails_for(name)[1:]:
                q = f"{name} {tail}".strip()
                k = self._key(q)
                if not k or any(k <= s for s in seen):
                    continue
                seen.append(k)
                out.append(q)
        return out[: self.MAX_HOP_QUERIES]

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

        # ROUND 1: decomposed, entity-anchored searches issued side by side.
        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = queries[: self.MAX_FIRST_QUERIES - 1] + [question] if queries else [question]

        pool: list[str] = []
        seen: set[str] = set()
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K1))
        ctx.note(f"round1: {len(queries)} queries -> {len(pool)} unique passages", queries=queries)

        # PIVOT + PROPERTY, both read off the retrieved text rather than invented.
        numbered = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        judged = self.bridge(ctx.llm, tag="bridge", question=question, passages=numbered)
        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]

        pivots = [
            n for n in self._lines(judged.get("pivot_names", ""), self.MAX_BRIDGES + 2)
            if len(n) > 2 and not _PLACEHOLDER.search(n)
        ][: self.MAX_BRIDGES]
        if not pivots:
            pivots = self._harvest(kept or pool, question, self.MAX_BRIDGES)
        prop = " ".join((judged.get("missing_property", "") or "").split())[:60]
        if prop.lower().startswith("none"):
            prop = ""

        # HOP: unconditional, and every query is a name crossed with the missing property.
        hop_queries = self._compose(pivots, prop, question, queries)
        if hop_queries:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for q in hop_queries:
                self._add(round2, seen2, ctx.tools.retrieve(q, k=self.K2))
            ctx.note(f"hop: {hop_queries} -> {len(round2)} new passages", pivots=pivots, prop=prop)
            kept = kept + round2

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
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop_queries,
            "n_evidence": len(kept),
        }
