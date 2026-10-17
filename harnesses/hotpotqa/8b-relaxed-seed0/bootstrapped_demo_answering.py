"""Answering taught by demonstrations mined from the train split, retrieved per question.

Mechanism change vs the parent (`deep_pool_title_rerank`). The retrieval half is left byte-identical
so the comparison isolates one thing: every system evaluated so far answers ZERO-SHOT, from an
instruction that *describes* the wanted answer in prose ("a name, title, date or quantity ... never a
sentence"). Prose cannot express a labelling convention, and the convention is what a large share of
the remaining losses are about -- with both required articles already in context the parent still
returns a familiar short name where the accepted string carries the middle names its lead sentence
prints, a place stripped of the larger place the source attaches, a bare quantity without its head
noun, and, in the other direction, an officially-long name where the accepted string is the everyday
short one. A run-time longest-span rule got this backwards half the time; a distilled per-bucket style
rule was too coarse. Conventions of this kind are learned from examples, not from rules.

So `prepare` -- a no-op in the parent -- builds a demonstration bank out of the 150 labelled train
questions. Each train question is put through the parent's own retrieval, a teacher call answers it
under the parent's own answer signature, and the demonstration is kept two ways: verified, when the
teacher's answer matches the label (its reasoning is kept as written), or REPAIRED, when it does not
but some retrieved passage prints the label verbatim -- then the demonstration is rebuilt around the
label with a quote of the sentence that prints it. The repaired half is the point: those are exactly
the questions whose convention the model does not already know, and a purely self-consistent bootstrap
would have thrown every one of them away.

At run time a deterministic router reads the question's wording into a shape and picks the four
nearest demonstrations by shape and content-word overlap; they are prepended to the answer call as
worked examples in the same field format as the request. Nothing else changes.

Calls: 3 LLM, 5-6 retrievals (unchanged). Prepare: <= 1 LLM call per train question.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import em_score, normalize_text

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


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# Deterministic reading of the question's wording into a shape. Entity-free: only function words,
# interrogatives and the common nouns an encyclopedia question uses to name what it wants.
_SHAPES: list[tuple[str, Any]] = [
    ("yesno", re.compile(r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.I)),
    ("count", re.compile(r"^\s*how\s+(?:many|much|old|long|tall|far)\b", re.I)),
    ("date", re.compile(r"\b(?:what|which)\s+(?:year|date|day|month|decade)\b|^\s*when\b", re.I)),
    ("kind", re.compile(r"\b(?:what|which)\s+(?:type|kind|sort|genre|species|nationality|language|colou?r|category)\b", re.I)),
    ("place", re.compile(r"^\s*where\b|\b(?:what|which)\s+(?:country|city|state|county|province|region|town|village|island)\b", re.I)),
    ("compare", re.compile(r"\b\w+\s+or\s+\w+\s*[?,]|\b(?:older|younger|larger|smaller|more|fewer|earlier|later|first|both|same)\b", re.I)),
    ("person", re.compile(r"^\s*who\b|\b(?:what|which)\s+(?:person|actor|actress|director|author|writer|singer|player|politician|artist|band|group)\b", re.I)),
    ("work", re.compile(r"\b(?:what|which)\s+(?:film|movie|album|song|book|novel|series|show|game|magazine)\b", re.I)),
]


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

    # demonstration bank
    N_DEMOS = 4                 # worked examples prepended to the answer call
    DEMO_EVIDENCE = 2           # passages shown per worked example
    MIN_REPAIRED = 2            # worked examples that must carry a correction, when the bank has them
    DEMO_PASSAGE_CHARS = 340
    DEMO_REASONING_CHARS = 420
    PREPARE_K = 20
    PREPARE_SHORTLIST = 8       # what the teacher reads per train question
    PREPARE_SECONDS = 2700      # wall-clock guard; the bank is useful even if truncated

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
        self.demo_bank: list[dict[str, Any]] = []

    # -- the demonstration bank ------------------------------------------------------------
    @classmethod
    def _shape(cls, question: str) -> str:
        q = " " + " ".join((question or "").split())
        for name, pat in _SHAPES:
            if pat.search(q):
                return name
        return "other"

    def _select_demos(self, question: str) -> list[dict[str, Any]]:
        """Nearest worked examples: same question shape only, half of them corrections.

        A convention is shape-local -- what a question about a person accepts as a name is not what a
        question about a place accepts -- so an example of the wrong shape is worse than no example,
        and the bank is filtered by shape before anything else. Within the shape the selection is
        deliberately split: the repaired examples are the ones whose convention the teacher did NOT
        already follow, so they are the only ones carrying new information, while the verified ones
        keep the extraction behaviour that already works from drifting.
        """
        if not self.demo_bank:
            return []
        shape = self._shape(question)
        qk = self._content(question)
        pool = [d for d in self.demo_bank if d.get("shape") == shape] or list(self.demo_bank)

        def rank(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
            scored = []
            for i, d in enumerate(cands):
                dk = set(d.get("kw") or [])
                scored.append((len(qk & dk) / max(len(qk | dk), 1), -i, d))
            scored.sort(key=lambda s: (-s[0], -s[1]))
            return [d for _, _, d in scored]

        repaired = rank([d for d in pool if not d.get("verified")])[: self.MIN_REPAIRED]
        rest = rank([d for d in pool if d not in repaired])
        chosen = (repaired + rest)[: self.N_DEMOS]
        chosen = rank(chosen)          # most similar example sits closest to the request
        chosen.reverse()
        return [{"question": d["q"], "evidence": d["ev"], "reasoning": d["r"], "answer": d["a"]} for d in chosen]

    def get_state(self) -> str:
        return json.dumps({"demos": self.demo_bank}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            self.demo_bank = list((json.loads(state or "{}") or {}).get("demos") or [])
        except Exception:
            self.demo_bank = []

    @staticmethod
    def _sentences(text: str) -> list[str]:
        body = text.split("|", 1)[1] if "|" in text else text
        return [s.strip() for s in _SENT.split(body.strip()) if s.strip()]

    def _quote_for(self, gold: str, docs: list[dict[str, Any]]) -> tuple[int, str] | None:
        """The passage and sentence that print the label verbatim, if any passage does."""
        target = normalize_text(gold)
        if not target:
            return None
        for i, h in enumerate(docs):
            if target not in normalize_text(h.get("text", "")):
                continue
            sents = self._sentences(h.get("text", ""))
            for sent in sents:
                if target in normalize_text(sent):
                    return i, sent[: self.DEMO_PASSAGE_CHARS]
            return i, (sents[0][: self.DEMO_PASSAGE_CHARS] if sents else "")
        return None

    def _demo_evidence(self, docs: list[dict[str, Any]], keep_first: int | None) -> tuple[list[str], int]:
        """A short, run-time-shaped evidence block: the answer-bearing passage plus the top others."""
        order: list[int] = []
        if keep_first is not None and 0 <= keep_first < len(docs):
            order.append(keep_first)
        for i in range(len(docs)):
            if i not in order:
                order.append(i)
            if len(order) >= self.DEMO_EVIDENCE:
                break
        block = []
        pos_of_first = 1
        for n, i in enumerate(order[: self.DEMO_EVIDENCE]):
            text = " ".join(docs[i].get("text", "").split())[: self.DEMO_PASSAGE_CHARS]
            block.append(f"{n + 1}. {text}")
            if i == keep_first:
                pos_of_first = n + 1
        return block, pos_of_first

    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        """Mine worked examples from the labelled train questions.

        One teacher call per question under the very signature `run` will use. Verified examples keep
        the teacher's own reasoning; examples the teacher got wrong are REPAIRED around the label
        whenever a retrieved passage prints it, with a quote of the sentence that does. The repaired
        ones carry the conventions the model does not already follow, which is what the bank is for.
        """
        started = time.time()
        bank: list[dict[str, Any]] = []
        n_ver = n_rep = n_drop = 0
        for item in train_items:
            if time.time() - started > self.PREPARE_SECONDS:
                break
            gold = str(item.get("answer") or "").strip()
            question = str(item.get("question") or "").strip()
            if not gold or not question or len(gold.split()) > self.MAX_ANSWER_WORDS:
                continue
            try:
                with ctx.open_tools(item) as tools:
                    hits = tools.retrieve_docs(question, k=self.PREPARE_K)
                ranked = self._rerank([hits], [question], question, picked=set())
                docs = [h for _, h in ranked[: self.PREPARE_SHORTLIST]]
                if not docs:
                    continue
                numbered = [f"{i + 1}. {h['text']}" for i, h in enumerate(docs)]
                out = self.answer(ctx.llm, tag="teacher", question=question, evidence=numbered)
                draft = self._clean(out.get("answer", ""))
                reasoning = " ".join((out.get("reasoning", "") or "").split())[: self.DEMO_REASONING_CHARS]
            except Exception:
                continue

            found = self._quote_for(gold, docs)
            verified = em_score(draft, gold)
            if verified:
                keep = found[0] if found else None
                block, _ = self._demo_evidence(docs, keep)
                if not reasoning:
                    reasoning = "The evidence names it directly, so the bare answer is copied from the passage."
            elif found is not None:
                # repaired: rebuild the example around the label and the sentence that prints it
                keep, sent = found
                block, pos = self._demo_evidence(docs, keep)
                reasoning = (
                    f"Passage {pos} states: \"{sent}\" That sentence carries what the question asks for, "
                    f"so the answer is that phrase exactly as the passage prints it, and nothing more."
                )
            else:
                n_drop += 1
                continue
            bank.append(
                {
                    "q": question,
                    "ev": block,
                    "r": reasoning,
                    "a": gold,
                    "shape": self._shape(question),
                    "kw": sorted(self._content(question))[:24],
                    "verified": 1 if verified else 0,
                }
            )
            n_ver += 1 if verified else 0
            n_rep += 0 if verified else 1
        self.demo_bank = bank
        ctx.note(
            f"demonstration bank: {len(bank)} examples ({n_ver} verified, {n_rep} repaired, {n_drop} unusable)",
            shapes=sorted({d["shape"] for d in bank}),
            seconds=round(time.time() - started, 1),
        )

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
        demos = self._select_demos(question)
        answerer = Module(
            inputs=["question", "evidence"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
            demos=demos,
        )
        ctx.note(f"answering with {len(demos)} worked examples", shape=self._shape(question))
        out = answerer(ctx.llm, tag="answer", question=question, evidence=evidence)
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
            "n_demos": len(demos),
        }
