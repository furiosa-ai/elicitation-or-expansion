"""The second hop is a link followed out of the first hop's text, never a query invented from scratch.

Mechanism change vs the parent (decomposed_multiquery_rerank): the parent lets the model *write* its
follow-up searches in free text and only spends them when it declares itself unsatisfied. Here the
follow-up vocabulary is closed. After the first round, a deterministic harvester reads the retrieved
passages and extracts every entity name they actually print -- their own titles plus the proper-noun runs
inside their text -- minus the entities the question already names. That numbered list is the *only*
thing the model may choose from: it points at the entity whose own article has to be read next, and that
name, spelled as the passage spells it, becomes the second-round query. The hop is unconditional.

Rationale: a two-article question is a link. The article that has to be read second is, by construction,
one that the first article names, so it is already sitting in the retrieved text and does not have to be
guessed. Left to write its own follow-ups, the model instead guesses at the answer (searching for the
film it suspects, the actress it half-remembers, or a bracketed placeholder for a name it never learned),
and its sufficiency gate lets it skip the hop entirely on the very items where the linking entity was
never retrieved. Both failures disappear when the hop is compulsory and its wording is copied out of
evidence rather than imagined.

Calls: 3 LLM, 3-5 retrievals.
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

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
# A run of capitalised words, allowing the lower-case particles that sit inside real names and titles.
_ENTITY = re.compile(
    r"\b(?:[A-Z][^\W\d_][\w’'\-]*|\d{4})"
    r"(?:(?:\s+|\s*[-–]\s*)(?:of|the|and|de|del|della|di|da|van|von|der|la|le|los|dos|bin|ibn|al|"
    r"[A-Z][^\W\d_][\w’'\-]*|\d{4}))*",
    re.UNICODE,
)
_PARTICLE = {
    "of", "the", "and", "de", "del", "della", "di", "da", "van", "von", "der", "la", "le", "los",
    "dos", "bin", "ibn", "al",
}
# Words that start a capitalised run without naming anything (sentence openers, calendar words).
_NOTNAME = {
    "the", "a", "an", "he", "she", "it", "they", "his", "her", "its", "their", "this", "that",
    "these", "those", "in", "on", "at", "as", "by", "for", "from", "after", "before", "when",
    "while", "during", "since", "however", "although", "both", "there", "then", "also", "later",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "american", "british", "english", "french", "german", "italian",
    "spanish", "russian", "japanese", "chinese", "indian", "canadian", "australian",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "was", "were",
    "is", "are", "did", "do", "does", "can", "will", "would", "name", "between", "besides",
    "united states", "united kingdom", "u.s.", "us", "uk", "america", "europe", "asia",
    "africa", "north america", "south america", "world war i", "world war ii", "internet",
}


class Harness(BaseHarness):
    K1 = 8
    K2 = 8
    MAX_FIRST_QUERIES = 3
    MAX_LINKS = 2
    MAX_CANDIDATES = 22
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

    LINK_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant. `candidates` is a "
        "numbered list of the names those passages mention; nothing outside that list can be searched for.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant "
        "first, at most six numbers separated by commas. Judge relevance by whether the passage is about an "
        "entity the question actually asks about, not by superficial word overlap.\n"
        "`link_ids`: one or two numbers from `candidates`, separated by commas. Settling this question "
        "means reading a second article, and the entity it is about is named somewhere in the passages "
        "above. Pick the name of that entity -- the person, work, place or organisation that stands between "
        "what the question names and what it asks for. If the passages already state the answer outright, "
        "pick instead the name of the entity the answer is about, so it can be confirmed. Choose by number "
        "only; never write a name that is not in the list.\n"
        "`why`: one short line naming the fact you expect that second article to supply."
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
        self.link = Module(
            inputs=["question", "passages", "candidates"],
            outputs=["useful_ids", "link_ids", "why"],
            instructions=self.LINK_INSTR,
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

    # -- the closed link vocabulary ---------------------------------------------------------
    @staticmethod
    def _trim(name: str) -> str:
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip(" ,-–")
        parts = name.split()
        while parts and parts[0].lower() in _PARTICLE:
            parts.pop(0)
        while parts and parts[-1].lower() in _PARTICLE:
            parts.pop()
        return " ".join(parts)

    def _harvest(self, pool: list[str], question: str) -> list[str]:
        """Every name the retrieved passages print, minus the ones the question already names.

        Passage titles come first (an article the pool already links to by name is the likeliest next
        hop), then in-text names ordered by how many distinct passages mention them: an entity that two
        retrieved articles both name is exactly the joint the question turns on.
        """
        q_words = {w.lower() for w in _WORD.findall(question) if len(w) > 2}
        counts: dict[str, int] = {}
        first: dict[str, tuple[int, int]] = {}
        display: dict[str, str] = {}
        asked: dict[str, str] = {}
        have_titles = {self._title_of(p) for p in pool}

        # An entity the question names outright, whose own article the first round never brought back,
        # is the one search that certainly still has to be made.
        for m in _ENTITY.finditer(question):
            name = self._trim(m.group(0))
            words = name.split()
            if not (1 <= len(words) <= 6) or len(name) < 3:
                continue
            low = name.lower()
            if low in _NOTNAME or words[0].lower() in _NOTNAME or low in have_titles:
                continue
            if any(low in t or t in low for t in have_titles):
                continue
            asked.setdefault(low, name)

        for rank, passage in enumerate(pool):
            title, _, body = passage.partition("|")
            seen_here: set[str] = set()
            spans = [(self._trim(title.strip()), -1)] + [
                (self._trim(m.group(0)), m.start()) for m in _ENTITY.finditer(body)
            ]
            for name, at in spans:
                words = name.split()
                if not (1 <= len(words) <= 6) or len(name) < 3:
                    continue
                low = name.lower()
                if low in _NOTNAME or words[0].lower() in _NOTNAME:
                    continue
                if re.fullmatch(r"[\d\s,\-–]+", name):
                    continue
                # already asked about, or already retrieved as an article
                content = [w.lower() for w in words if w.lower() not in _PARTICLE]
                if content and all(w in q_words for w in content):
                    continue
                if low in have_titles:
                    continue
                if low in seen_here:
                    continue
                seen_here.add(low)
                counts[low] = counts.get(low, 0) + 1
                if low not in first:
                    first[low] = (rank, at)
                    display[low] = name

        ranked = sorted(counts, key=lambda k: (-counts[k], first[k][0], first[k][1]))
        out = [asked[k] for k in asked]
        for k in ranked:
            if k not in asked:
                out.append(display[k])
        return out[: self.MAX_CANDIDATES]

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

        # CLOSED LINK VOCABULARY harvested from the retrieved text itself.
        cands = self._harvest(pool, question)
        numbered = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        judged = self.link(
            ctx.llm,
            tag="link",
            question=question,
            passages=numbered,
            candidates=[f"{i + 1}. {c}" for i, c in enumerate(cands)],
        )

        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]

        # ROUND 2: unconditional, and worded with names lifted out of round 1.
        links: list[str] = []
        for tok in _NUM.findall(judged.get("link_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(cands) and cands[idx] not in links:
                links.append(cands[idx])
            if len(links) >= self.MAX_LINKS:
                break
        if not links:
            links = cands[: self.MAX_LINKS]
        if links:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for name in links:
                self._add(round2, seen2, ctx.tools.retrieve(name, k=self.K2))
            ctx.note(f"hop2: followed {links} -> {len(round2)} new passages", links=links)
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
            "links": links,
            "n_candidates": len(cands),
            "n_evidence": len(kept),
        }
