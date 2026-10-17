"""Evidence assembled as competing two-article chains rather than one flat ranked list.

Mechanism change vs the parent (`deep_pool_title_rerank`). The parent's retrieval is kept verbatim,
including the deep pool and the title-anchored fusion rerank; what changes is the unit of evidence.
The parent ranks articles INDEPENDENTLY and hands the answerer the best twelve, so the pool it reads
is twelve unrelated topics and the answerer has to discover, unaided, which two of them chain -- and
when it fails it answers the wrong slot of the right topic: the ride instead of its manufacturer, the
film instead of the person who directed it, the sequel's writers instead of its director, the spouse
instead of the office-holder. Independent ranking also mis-selects: two articles that both echo the
question's wording outrank a pair where one article is only reachable through the other, which is the
shape every such question actually has.

So this system scores PAIRS. Over the reranked pool it computes, in code, whether one article's text
prints the other's title (a link), how much of the question's inverse-document-frequency mass the
union of the two covers, and how COMPLEMENTARY they are -- each member must carry question content the
other does not, which is what separates a genuine two-article chain from two paraphrases of the same
topic. The best few chains, each introducing at least one article the earlier ones did not, are handed
over as labelled blocks with their link named, the remaining top-ranked articles follow as loose
passages so nothing highly ranked is lost, and the answerer must first commit to the chain that
settles the question and only then extract from it.

Calls: 3 LLM, 5-6 retrievals (unchanged).
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
    PAIR_POOL = 12          # articles the pair scorer considers
    N_CHAINS = 3            # competing chains shown
    MAX_LOOSE = 4           # top-ranked articles shown outside any chain
    W_LINK = 1.4
    W_COVER = 1.2
    W_COMPL = 1.0
    W_PRIOR = 0.6
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
        "A question of this kind is settled by two articles read together: one is reachable from the words of "
        "the question, the other only through the first. `chains` holds the candidate pairings, each labelled "
        "with the link that joins them; `other_passages` holds further articles that were ranked highly but "
        "paired with nothing.\n"
        "`chain_id`: the label of the single chain that settles the question (for example `C1`), or `none` if "
        "the answer comes from `other_passages` instead. Choose the pair where one article leads to the other "
        "and the second one states the fact that was asked for -- not the pair that merely repeats the "
        "question's wording.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the chosen passage spells it, or the single word yes or no when the question asks whether something "
        "holds. Take care to answer the property the question asks about rather than naming the article you "
        "found it in. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "pivot_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "chains", "other_passages"],
            outputs=["chain_id", "answer"],
            instructions=self.ANSWER_INSTR,
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

    # -- pairing ---------------------------------------------------------------------------
    def _bare_title(self, title: str) -> str:
        return self._norm_phrase(_PAREN.sub("", title or "").strip())

    def _links(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """Does one article's prose name the other? That is the shape of a two-article chain.

        A hyperlink is what makes the second article reachable at all, and in an abstract the link
        survives as the other article's title printed in the text.
        """
        best = 0.0
        for src_doc, dst_doc in ((a, b), (b, a)):
            dst = self._bare_title(dst_doc.get("title", ""))
            if not dst:
                continue
            if len(dst.split()) < 2 and len(dst) < 6:
                continue
            body = " " + self._norm_phrase(src_doc.get("text", "")) + " "
            if f" {dst} " in body:
                best = max(best, 1.0)
            else:
                dt = [w for w in dst.split() if w not in _STOP]
                if len(dt) >= 2 and all(f" {w} " in body for w in dt):
                    best = max(best, 0.55)
        return best

    def _chains(
        self,
        docs: list[dict[str, Any]],
        question: str,
    ) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any], float]], list[dict[str, Any]]]:
        """Score every pair of the top-ranked articles, then keep the best few competing chains."""
        n = max(len(docs), 1)
        df: dict[str, int] = {}
        for h in docs:
            for w in set(self._toks(h.get("text", ""))):
                df[w] = df.get(w, 0) + 1
        qwords = self._content(question)
        idf = {w: math.log(1.0 + n / (1.0 + df.get(w, 0))) for w in qwords}
        total = sum(idf.values()) or 1.0
        covered = [
            {w for w in idf if w in set(self._toks(h.get("title", "") + " " + h.get("text", "")))} for h in docs
        ]
        prior = [1.0 / (1.0 + i) for i in range(len(docs))]

        scored: list[tuple[float, int, int, float]] = []
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                union = covered[i] | covered[j]
                cover = sum(idf[w] for w in union) / total
                only_i = sum(idf[w] for w in covered[i] - covered[j]) / total
                only_j = sum(idf[w] for w in covered[j] - covered[i]) / total
                compl = min(only_i, only_j)          # both members must earn their place
                link = self._links(docs[i], docs[j])
                score = (
                    self.W_LINK * link
                    + self.W_COVER * cover
                    + self.W_COMPL * compl
                    + self.W_PRIOR * (prior[i] + prior[j])
                )
                scored.append((score, i, j, link))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))

        chains: list[tuple[str, dict[str, Any], dict[str, Any], float]] = []
        used: set[int] = set()
        for score, i, j, link in scored:
            if len(chains) >= self.N_CHAINS:
                break
            if chains and i in used and j in used:
                continue                              # every chain must bring something new
            chains.append((f"C{len(chains) + 1}", docs[i], docs[j], link))
            used.update((i, j))
        loose = [h for k, h in enumerate(docs) if k not in used][: self.MAX_LOOSE]
        return chains, loose

    def _render_chains(self, chains: list[tuple[str, dict[str, Any], dict[str, Any], float]]) -> list[str]:
        out = []
        for label, a, b, link in chains:
            if link >= 1.0:
                joint = "one of these articles names the other in its text"
            elif link > 0.0:
                joint = "the two articles share the words of a name"
            else:
                joint = "no explicit link; they cover different parts of the question"
            out.append(f"{label} ({joint}):\n  {label}a. {a['text']}\n  {label}b. {b['text']}")
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
        pool = [h for _, h in ranked[: self.PAIR_POOL]]
        chains, loose = self._chains(pool, question)
        ctx.note(
            f"hop: {hop} -> {len(ranked)} pooled, {len(chains)} chains over {self.PAIR_POOL} candidates",
            pivots=pivots,
            prop=prop,
            chains=[f"{lab}: {a['title']} + {b['title']} (link={lk})" for lab, a, b, lk in chains],
            loose=[h["title"] for h in loose],
        )

        out = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            chains=self._render_chains(chains),
            other_passages=[h["text"] for h in loose] or ["(none)"],
        )
        answer = self._clean(out.get("answer", ""))
        chosen = (out.get("chain_id", "") or "").strip()[:8]
        if not answer or _NONANSWER.match(answer):
            answer = chains[0][1]["title"] if chains else answer
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "chain_id": chosen,
            "n_chains": len(chains),
            "n_loose": len(loose),
        }
