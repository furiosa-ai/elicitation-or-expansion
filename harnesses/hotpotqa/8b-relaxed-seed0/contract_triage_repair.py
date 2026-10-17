"""Answering is triaged against a contract derived from the question, and rejected answers are repaired.

Mechanism change vs the parent (`deep_pool_title_rerank`). The parent spends three of its twelve
permitted calls and six of its twelve permitted searches, then whatever its single answering call
emits is the answer -- with one exception: when the emitted text looks like a refusal the parent
silently substitutes the title of its top-ranked article. Its traces show what that costs. On items
where every required article is in the answerer's own context it still fails 47 times, and a large
share of those failures announce themselves in the text the answerer produced: it wrote that the
fact was not specified, that it could not be determined, that nobody matched; it echoed back a name
the question had already given it; it named a work when the question asked who; it returned two
people joined by `and` where one was asked for; or it emitted nothing parseable at all. Each of
those is a violation of a contract the question itself states, and each is currently accepted
(or replaced by a title, which is worse).

This system reads the question as a contract -- what type of thing is being asked for, whether one
item or several, whether a yes/no was requested, and which names are already given and therefore
cannot be the answer -- and checks the draft against it in code. A draft that satisfies the contract
is returned untouched, so items the parent already answers are not disturbed. A draft that violates
it is not patched by rule and not replaced by a title: the spare budget is spent on a repair round
that names the violation, asks what single thing must be looked up to satisfy it, searches for that,
and re-answers over the enlarged pool under an explicit type requirement. The repaired answer is
accepted only if it satisfies the contract the draft broke; otherwise the draft stands.

The lever is triage, not persuasion: the same answerer, told exactly which requirement it failed and
given the article it said it was missing.

Calls: 3 LLM and 5-6 retrievals when the draft passes, 5 LLM and 7-8 retrievals when it does not.
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

_HEDGE = re.compile(
    r"^(?:not\s|no\s+(?:one|body|overlap|information|match|mention)|none\b|nobody\b|unknown\b|unclear\b|"
    r"unrelated\b|cannot\b|can\'t\b|undetermined\b|indeterminate\b|insufficient\b|n/?a$|same$|both$|"
    r"neither\b|does\s+not\b|do\s+not\b|is\s+not\b|unspecified\b)",
    re.IGNORECASE,
)
_COUNT_Q = re.compile(r"\bhow\s+(?:many|much)\b", re.IGNORECASE)
_YESNO_Q = re.compile(r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE)
_WHO_Q = re.compile(r"\b(?:who|whom|whose)\b", re.IGNORECASE)
_WHEN_Q = re.compile(r"\b(?:when\b|what\s+year|which\s+year|what\s+date|which\s+date)\b", re.IGNORECASE)
_NUMWORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion)\b",
    re.IGNORECASE,
)
_PAREN_TAIL = re.compile(r"\s*\([^)]*\)\s*$")
_LIST_Q = re.compile(r"\bboth\b|\bwho\s+are\b|\bwhat\s+are\b|\bname\s+(?:the|two|all)\b", re.IGNORECASE)
_CHOICE_Q = re.compile(r"\bor\b", re.IGNORECASE)
_TWO_ENTITIES = re.compile(r"^([A-Z][^,]*?)\s+and\s+([A-Z].*)$")

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

    PLAN_INSTR = (
        "A draft answer to `question` was REJECTED because it broke a requirement the question states; "
        "`rejection` says which one. Usually the draft failed because the one article that prints the "
        "wanted fact was never fetched, so the draft hedged, echoed a name the question already gave, or "
        "named the wrong kind of thing.\n"
        "`lookup`: one or two search queries, one per line, for the article that would PRINT the wanted "
        "fact. Each query must name a concrete entity -- copy the spelling from the question or the "
        "evidence -- followed by two to five words for the property being asked for, in the vocabulary an "
        "encyclopedia article uses. No questions, no numbering, no brackets, no placeholders.\n"
        "`target`: two to six words naming what kind of thing the answer must be (for example a person's "
        "name, a year, a quantity, a place, the title of a work). Nothing else."
    )

    REPAIR_INSTR = (
        "Answer the question from the numbered `evidence` passages. A previous attempt answered "
        "`rejected`, which was thrown out because it broke the requirement in `requirement`; the evidence "
        "below has been re-searched since.\n"
        "`answer` must satisfy `requirement` exactly: the bare answer and nothing else, spelled exactly as "
        "the evidence spells it, never a sentence, never a restatement of the question. Do not repeat the "
        "rejected answer and do not hedge -- saying the evidence is insufficient is not an answer. If the "
        "evidence still does not state the fact outright, commit to the reading the evidence best supports."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.repair_plan = Module(
            inputs=["question", "rejection", "evidence"],
            outputs=["lookup", "target"],
            instructions=self.PLAN_INSTR,
        )
        self.repair_answer = Module(
            inputs=["question", "requirement", "rejected", "evidence"],
            outputs=["answer"],
            instructions=self.REPAIR_INSTR,
        )
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
    ) -> list[tuple[float, dict[str, Any]]]:
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
        return scored

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

    # -- the contract ----------------------------------------------------------------------
    def _contract(self, question: str) -> dict[str, Any]:
        """What the question itself says the answer has to be, read off its surface."""
        q = question.strip()
        choice = bool(_CHOICE_Q.search(q))
        kind = "open"
        if _COUNT_Q.search(q):
            kind = "count"
        elif _YESNO_Q.match(q) and not choice:
            kind = "yesno"
        elif _WHEN_Q.search(q):
            kind = "date"
        elif _WHO_Q.search(q):
            kind = "person"
        # Names the question already hands over cannot themselves be the answer, unless the
        # question is offering a choice between them.
        given: set[str] = set()
        if not choice:
            for m in _PROPER.finditer(q):
                name = self._norm_phrase(m.group(0).strip(" .,;:'\u2019-"))
                if name and name not in {"who", "what", "which", "when", "where", "how"}:
                    given.add(name)
        return {
            "kind": kind,
            "choice": choice,
            "given": given,
            "single": not _LIST_Q.search(q),
        }

    REQUIREMENT = {
        "count": "a quantity: the number itself, written the way the evidence writes it",
        "yesno": "the single word yes or no",
        "date": "a date or a year, nothing else",
        "person": "the name of one person, group or organisation as the evidence spells it, "
                  "never the title of a work and never a description",
        "open": "one bare noun phrase copied from the evidence, as short as the question allows",
    }

    def _violations(self, question: str, answer: str, evidence_docs: list[dict[str, Any]], con: dict[str, Any]) -> list[str]:
        """Which requirements this candidate answer breaks. Empty list means the draft stands."""
        ans = (answer or "").strip()
        na = self._norm_phrase(ans)
        out: list[str] = []
        if not na:
            return ["the draft was empty: no answer was produced"]
        if _HEDGE.match(ans):
            out.append("the draft refused to answer instead of naming something")
        if con["kind"] == "yesno" and na not in {"yes", "no"}:
            out.append("the question asks whether something holds, so the answer must be yes or no")
        if con["kind"] == "count" and not (_NUM.search(ans) or _NUMWORD.search(ans)):
            out.append("the question asks for a quantity, so the answer must be a number")
        if na in con["given"]:
            out.append("the draft repeated a name the question already supplies, which cannot be the answer")
        if _PAREN_TAIL.search(ans) and con["kind"] in {"person", "count", "date"}:
            out.append("the draft copied an article title with a parenthesised qualifier instead of the fact itself")
        if con["single"] and na not in {"yes", "no"}:
            m = _TWO_ENTITIES.match(ans)
            if m:
                out.append("the question asks for one thing and the draft named two joined by `and`")
        if len(ans.split()) > 8:
            out.append("the draft is a sentence rather than a bare answer")
        # An answer no fetched article prints is a recollection, not a reading of the evidence.
        if na not in {"yes", "no"} and not _NUM.search(ans):
            body = self._norm_phrase(" ".join(h.get("text", "") for h in evidence_docs))
            if na and f" {na} " not in f" {body} ":
                out.append("no fetched article prints the draft answer, so it is unsupported")
        return out

    def _repair_queries(self, lookups: list[str], pivots: list[str], target: str, question: str, already: list[str]) -> list[str]:
        seen = [frozenset(self._content(q)) for q in already]
        out: list[str] = []
        cands = list(lookups)
        for name in pivots:
            tail = " ".join(w for w in target.split() if w.lower() not in _STOP)[:40]
            if tail:
                cands.append(f"{name} {tail}")
        for q in cands:
            q = " ".join(q.split())[:120]
            if not q or _PLACEHOLDER.search(q):
                continue
            k = frozenset(self._content(q))
            if not k or any(k <= s for s in seen):
                continue
            seen.append(k)
            out.append(q)
            if len(out) >= 2:
                break
        return out

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = (queries[: self.MAX_FIRST_QUERIES - 1] + [question]) if queries else [question]

        lists = [ctx.tools.retrieve_docs(q, k=self.K) for q in queries]
        anchors = [question] + queries
        ranked = self._rerank(lists, anchors, question, picked=set())
        ctx.note(
            f"round1: {len(queries)} queries at k={self.K} -> {len(ranked)} unique articles",
            queries=queries,
            top=[h["title"] for _, h in ranked[:8]],
        )

        shortlist = [h for _, h in ranked[: self.SHORTLIST]]
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

        # One rerank over everything both rounds saw, now anchored on the pivots as well.
        ranked = self._rerank(lists, anchors + pivots, question, picked)
        evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
        ctx.note(
            f"hop: {hop} -> {len(ranked)} articles pooled, {len(evidence_docs)} read",
            pivots=pivots,
            prop=prop,
            evidence=[h["title"] for h in evidence_docs],
        )

        evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        draft = self._clean(out.get("answer", ""))

        # -- triage: does the draft satisfy the contract the question states? ---------------
        con = self._contract(question)
        viol = self._violations(question, draft, evidence_docs, con)
        answer, repaired, rq = draft, "", []
        if viol:
            reason = "; ".join(viol[:3])
            ctx.note(f"rejected draft {draft!r}: {reason}", contract=con["kind"], violations=viol)
            plan = self.repair_plan(
                ctx.llm, tag="repair_plan", question=question, rejection=reason, evidence=evidence[:6]
            )
            lookups = [q for q in self._lines(plan.get("lookup", ""), 3) if not q.endswith("?")]
            target = " ".join((plan.get("target", "") or "").split())[:60]
            rq = self._repair_queries(lookups, pivots, target, question, queries + hop)
            for q in rq:
                lists.append(ctx.tools.retrieve_docs(q, k=self.K))
            if rq:
                ranked = self._rerank(lists, anchors + pivots + rq, question, picked)
            repair_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
            requirement = self.REQUIREMENT[con["kind"]]
            if con["single"] and con["kind"] != "yesno":
                requirement += "; exactly one item, never two joined by `and`"
            if target and con["kind"] == "open":
                requirement += f"; specifically {target}"
            out2 = self.repair_answer(
                ctx.llm,
                tag="repair_answer",
                question=question,
                requirement=requirement,
                rejected=draft or "(nothing)",
                evidence=[f"{i + 1}. {h['text']}" for i, h in enumerate(repair_docs)],
            )
            repaired = self._clean(out2.get("answer", ""))
            # Accept the repair only if it clears the bar the draft failed to clear.
            if repaired and not self._violations(question, repaired, repair_docs, con):
                answer = repaired
            elif not draft or _HEDGE.match(draft):
                answer = repaired or draft
            ctx.note(f"repair {rq} -> {repaired!r}; kept {answer!r}", target=target)
            evidence_docs = repair_docs or evidence_docs

        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer
        answer = _PAREN_TAIL.sub("", answer).strip() or answer
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "violations": viol,
            "repair_queries": rq,
            "draft": draft,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
