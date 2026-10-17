"""Evidence chosen as a SET by marginal gain, instead of as the top of a pointwise ranking.

Mechanism change vs the parent (`deep_pool_title_rerank`). The parent scores every pooled article
independently -- fusion rank, title alignment, question-word coverage -- and then simply reads the
highest twelve. Independent scoring has a structural blind spot: a question that mentions two things
pools one large family of articles that all match the first thing and one thin family that matches
the second, and the family with the higher scores fills the whole reading window. Measured on the
parent's own traces, the pool holds both required articles for 89.8% of items but the twelve that
were read hold both for only 86.2%: on 21 items an article the parent had already retrieved was
scored out of its own reading window, and a name that several articles share (a footballer, a
swimmer, a playwright) routinely takes six of the twelve slots and crowds out the second hop.

This system keeps the parent's scorer and throws away the `sort` behind it. The reading window is
filled one slot at a time by marginal gain: each candidate is judged on its own score PLUS how much
question vocabulary it adds that nothing already selected covers, MINUS how much it duplicates what
is already selected -- title overlap with an already-chosen article, and a penalty that grows with
every further article drawn from the same anchor family. Every anchor the question names therefore
buys its own slots, near-duplicates of an already-read article pay for their redundancy, and a
low-scoring article that is the only thing covering the question's remaining rare words is read
instead of being dropped. Selection becomes a covering problem over the question, not a ranking.

Calls: 3 LLM, 5-6 retrievals.
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
_PROPER = re.compile(r"\b(?:[A-Z][\w'’.-]*)(?:\s+(?:of|the|de|van|von|and|&)\s+[A-Z][\w'’.-]*|\s+[A-Z][\w'’.-]*)*")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
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
    K = 20                  # the index's full width; the rerank, not the cut-off, decides what is read
    MAX_FIRST_QUERIES = 3
    MAX_BRIDGES = 2
    MAX_HOP_QUERIES = 3
    SHORTLIST = 14          # how many reranked abstracts the screen reads
    MAX_EVIDENCE = 12       # how many the answerer reads
    RRF_C = 10.0
    W_TITLE = 3.0
    W_IDF = 1.6
    W_PICKED = 0.9
    W_COVER = 1.5           # marginal gain: question vocabulary no selected article covers yet
    W_DUP = 1.1             # redundancy: title overlap with an article already selected
    W_FAMILY = 0.55         # per extra article taken from the same anchor family
    FAMILY_FLOOR = 0.34     # title-anchor strength above which an article joins an anchor's family
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

    # -- text helpers ----------------------------------------------------------------------
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
    def _toks(text: str) -> list[str]:
        return [w.lower() for w in _WORD.findall(text or "")]

    def _content(self, text: str) -> set[str]:
        return {w for w in self._toks(text) if w not in _STOP and len(w) > 1}

    @staticmethod
    def _norm_phrase(text: str) -> str:
        return " ".join(_WORD.findall((text or "").lower()))

    # -- the rerank ------------------------------------------------------------------------
    def _title_anchor(self, title: str, anchors: list[str]) -> float:
        """How strongly this article's NAME matches a phrase the question or a pivot uses.

        The index cannot express this: it scores an abstract as a bag of words, so the article that
        IS the entity competes on equal terms with every article that merely mentions it.
        """
        bare = _PAREN.sub("", title).strip()
        tp, bp = self._norm_phrase(title), self._norm_phrase(bare)
        if not bp:
            return 0.0
        btoks = set(bp.split())
        best = 0.0
        for anchor in anchors:
            ap = self._norm_phrase(anchor)
            if not ap:
                continue
            atoks = set(ap.split())
            if bp == ap or tp == ap:
                score = 1.0                              # the anchor names this article outright
            elif f" {bp} " in f" {ap} ":
                score = 0.85                             # the anchor quotes the whole title
            elif f" {ap} " in f" {bp} " and len(atoks) > 1:
                score = 0.6                              # the title extends the anchor
            else:
                shared = btoks & atoks - _STOP
                if not shared:
                    continue
                score = 0.55 * len(shared) / max(len(btoks - _STOP), 1)
            best = max(best, score)
        return best

    def _rerank(
        self,
        ranked_lists: list[list[dict[str, Any]]],
        anchors: list[str],
        question: str,
        picked: set[str],
    ) -> tuple[list[tuple[float, dict[str, Any]]], dict[str, float]]:
        docs: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        for hits in ranked_lists:
            for rank, h in enumerate(hits):
                key = (h.get("title") or "").strip().lower()
                if not key:
                    continue
                docs.setdefault(key, h)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.RRF_C + rank)

        # Inverse document frequency over the pool itself: a question word that every candidate
        # repeats separates nothing, a rare one separates everything.
        n = max(len(docs), 1)
        df: dict[str, int] = {}
        for h in docs.values():
            for w in set(self._toks(h.get("text", ""))):
                df[w] = df.get(w, 0) + 1
        qwords = self._content(question)
        idf = {w: math.log(1.0 + n / (1.0 + df.get(w, 0))) for w in qwords}
        total_idf = sum(idf.values()) or 1.0

        scored: list[tuple[float, dict[str, Any]]] = []
        for key, h in docs.items():
            body = set(self._toks(h.get("text", "")))
            cover = sum(v for w, v in idf.items() if w in body) / total_idf
            score = (
                fused[key] * (self.RRF_C + 1.0)
                + self.W_TITLE * self._title_anchor(h.get("title", ""), anchors)
                + self.W_IDF * cover
                + (self.W_PICKED if key in picked else 0.0)
            )
            scored.append((score, h))
        scored.sort(key=lambda sh: -sh[0])
        return scored, idf

    def _families(self, docs: list[dict[str, Any]], anchors: list[str]) -> list[int]:
        """Which anchor family each article belongs to (-1 = none).

        A family is the set of articles whose TITLE answers to the same anchor phrase: the several
        articles that share a person's name, the several films that share a title. Grouping them is
        what lets the selector stop after reading two or three of them.
        """
        out: list[int] = []
        for h in docs:
            title = h.get("title", "")
            best, best_i = 0.0, -1
            for i, anchor in enumerate(anchors):
                strength = self._title_anchor(title, [anchor])
                if strength > best:
                    best, best_i = strength, i
            out.append(best_i if best >= self.FAMILY_FLOOR else -1)
        return out

    def _select(
        self,
        scored: list[tuple[float, dict[str, Any]]],
        idf: dict[str, float],
        anchors: list[str],
        budget: int,
    ) -> list[dict[str, Any]]:
        """Fill the reading window one slot at a time by marginal gain over the whole set.

        The parent read `scored[:budget]`. Here a candidate competes on its own score plus the
        question vocabulary it is the first to cover, minus what it duplicates: an article whose
        title all but repeats an already-selected title, or the fourth article from a family that
        already has three slots, has to be much stronger than a fresh one to be worth a slot.
        """
        pool = scored[: max(budget * 4, budget)]
        if len(pool) <= budget:
            return [h for _, h in pool]
        docs = [h for _, h in pool]
        raw = [sc for sc, _ in pool]
        span = (max(raw) - min(raw)) or 1.0
        norm = [(sc - min(raw)) / span for sc in raw]
        families = self._families(docs, anchors)
        bodies = [set(self._toks(h.get("text", ""))) for h in docs]
        titles = [set(self._norm_phrase(_PAREN.sub("", h.get("title", ""))).split()) - _STOP for h in docs]
        total_idf = sum(idf.values()) or 1.0

        chosen: list[int] = []
        covered: set[str] = set()
        fam_count: dict[int, int] = {}
        remaining = set(range(len(docs)))
        while remaining and len(chosen) < budget:
            best_gain, best_i = None, -1
            for i in sorted(remaining):
                fresh = sum(v for w, v in idf.items() if w not in covered and w in bodies[i]) / total_idf
                dup = 0.0
                for j in chosen:
                    inter = titles[i] & titles[j]
                    if inter:
                        dup = max(dup, len(inter) / max(len(titles[i] | titles[j]), 1))
                fam = fam_count.get(families[i], 0) if families[i] >= 0 else 0
                gain = norm[i] + self.W_COVER * fresh - self.W_DUP * dup - self.W_FAMILY * fam
                if best_gain is None or gain > best_gain:
                    best_gain, best_i = gain, i
            chosen.append(best_i)
            remaining.discard(best_i)
            covered |= {w for w in idf if w in bodies[best_i]}
            if families[best_i] >= 0:
                fam_count[families[best_i]] = fam_count.get(families[best_i], 0) + 1
        return [docs[i] for i in chosen]

    def _harvest(self, passages: list[str], question: str, limit: int) -> list[str]:
        """Fallback pivots: proper-noun runs the retrieved text prints, that the question does not."""
        qk = self._content(question)
        counts: dict[str, int] = {}
        for p in passages:
            body = p.split("|", 1)[1] if "|" in p else p
            for m in _PROPER.finditer(body):
                name = m.group(0).strip(" .,;:'’-")
                if not (2 <= len(name.split()) <= 5) or len(name) < 5:
                    continue
                if self._content(name) <= qk:
                    continue
                counts[name] = counts.get(name, 0) + 1
        return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]

    def _content_words(self, question: str, limit: int = 4) -> str:
        words = [w for w in _WORD.findall(question) if w.lower() not in _STOP and len(w) > 2]
        lower = [w for w in words if not w[:1].isupper()]
        return " ".join((lower or words)[-limit:])

    def _hop_queries(self, pivots: list[str], prop: str, question: str, already: list[str]) -> list[str]:
        """One query per pivot for the pivot's OWN article, plus the pivot crossed with the property.

        The parent never asked for an entity's own article directly because the index answers a bare
        name with the articles that repeat it. With a title-anchored rerank in front of the answerer
        that failure mode is handled downstream, so the cheapest and most precise query is available
        again.
        """
        seen = [frozenset(self._content(q)) for q in already]
        out: list[str] = []
        foreign_of = {
            name: {w for other in pivots if other != name for w in self._toks(other)} for name in pivots
        }
        tails: list[tuple[str, str]] = []
        for name in pivots:
            tails.append((name, ""))
        for name in pivots:
            for tail in (prop.strip(), self._content_words(question)):
                if not tail:
                    continue
                # A property phrase that quotes the OTHER pivot describes neither article; drop it.
                words = [w for w in tail.split() if w.strip(".,'’-").lower() not in foreign_of[name]]
                if words:
                    tails.append((name, " ".join(words)))
        for name, tail in tails:
            q = f"{name} {tail}".strip()
            k = frozenset(self._content(q))
            if not k or any(k <= s for s in seen):
                continue
            seen.append(k)
            out.append(q)
            if len(out) >= self.MAX_HOP_QUERIES:
                break
        return out

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

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = (queries[: self.MAX_FIRST_QUERIES - 1] + [question]) if queries else [question]

        lists = [ctx.tools.retrieve_docs(q, k=self.K) for q in queries]
        anchors = [question] + queries
        ranked, idf = self._rerank(lists, anchors, question, picked=set())
        shortlist = self._select(ranked, idf, anchors, self.SHORTLIST)
        ctx.note(
            f"round1: {len(queries)} queries at k={self.K} -> {len(ranked)} unique articles",
            queries=queries,
            top=[h["title"] for h in shortlist[:8]],
        )

        numbered = [f"{i + 1}. {h['text']}" for i, h in enumerate(shortlist)]
        judged = self.bridge(ctx.llm, tag="bridge", question=question, passages=numbered)

        picked: set[str] = set()
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(shortlist):
                picked.add((shortlist[idx].get("title") or "").strip().lower())

        pivots = [
            n for n in self._lines(judged.get("pivot_names", ""), self.MAX_BRIDGES + 2)
            if len(n) > 2 and not _PLACEHOLDER.search(n)
        ][: self.MAX_BRIDGES]
        if not pivots:
            pivots = self._harvest([h["text"] for h in shortlist], question, self.MAX_BRIDGES)
        prop = " ".join((judged.get("missing_property", "") or "").split())[:60]
        if prop.lower().startswith("none"):
            prop = ""

        hop = self._hop_queries(pivots, prop, question, queries)
        for q in hop:
            lists.append(ctx.tools.retrieve_docs(q, k=self.K))

        # One rerank over everything both rounds saw, now anchored on the pivots as well, and one
        # covering pass over that ranking: the second hop's article competes against the first
        # anchor's family for a slot on what it ADDS, not on its absolute score.
        ranked, idf = self._rerank(lists, anchors + pivots, question, picked)
        evidence_docs = self._select(ranked, idf, anchors + pivots, self.MAX_EVIDENCE)
        ctx.note(
            f"hop: {hop} -> {len(ranked)} articles pooled, {len(evidence_docs)} read",
            pivots=pivots,
            prop=prop,
            evidence=[h["title"] for h in evidence_docs],
        )

        evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "n_evidence": len(evidence_docs),
        }
