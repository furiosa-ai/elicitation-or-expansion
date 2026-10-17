"""Many narrow hop probes, each accepted or rejected as a whole before anything is read.

Mechanism change vs the parent (`grounded_answer_repair`). Measured on the parent's own traces, the
binding constraint is coverage, not ranking: of its 79 errors, 34 are items where a needed article was
never retrieved at all, while only 8 are items where it was retrieved and then not read. The parent
spends 5-6 of its 12 permitted searches, because its second hop is three wide queries built from at most
two pivot names that one LLM call had to commit to. When that call names the wrong pivot -- and on a
description-side hop it often does -- there is no second chance and the article is simply absent.

The obvious fix, hop on more candidates and merge everything, has been tried and loses: a wider merged
pool pushes the right abstract out of the twelve that get read. So this system separates POOL WIDTH from
READ WIDTH by making each hop a self-contained probe that must earn its way into the reading window:

  1. Candidate hop targets are enumerated cheaply and generously from the anchor abstracts -- the names
     the LLM proposed, plus every quoted string and every multi-word proper-noun run the anchors print
     that the question does not already contain. Six targets, not two.
  2. Each target gets its own narrow search. A probe is CONFIRMED only if the index round-trips it, i.e.
     some result's own title is that target rather than merely mentioning it; an unconfirmed probe is a
     name that is not an article and is thrown away whole.
  3. A confirmed probe is then scored on RESIDUAL FIT: how much of the question's inverse-document-
     frequency mass that the anchors do NOT already cover this one abstract supplies. A probe that only
     repeats what the anchors say adds nothing and is rejected too.
  4. Only the two best-fitting probe winners are admitted, and every other document any probe returned is
     discarded before the rerank runs. The reading window therefore never grows, no matter how wide the
     search went.

So the search budget is spent almost in full while the answerer's context stays exactly as small and as
clean as the parent's.

Calls: 3 LLM and 3-9 retrievals normally, 4 LLM and up to 12 retrievals when the grounding test fires.
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
_QUOTED = re.compile(r"[\"\u201c\u2018]([^\"\u201c\u201d\u2018\u2019]{3,60})[\"\u201d\u2019]")
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
    MAX_ANSWER_WORDS = 12
    MAX_REPAIR_QUERIES = 2

    # probe stage
    MAX_SEARCHES = 12       # hard cap imposed by the environment; never exceed it
    PROBE_MAX = 6           # candidate hop targets probed
    PROBE_K = 8             # narrow: a probe only has to surface the target's own article
    PROBE_ACCEPT = 2        # winners admitted to the reading window
    CONFIRM_MIN = 0.85      # title-anchor score at which the index has round-tripped the target
    ANCHOR_DOCS = 6         # abstracts treated as "what we already know" when scoring residual fit

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

    REPAIR_INSTR = (
        "Answer the question from the numbered `evidence` passages. An earlier attempt produced "
        "`rejected_draft`, a string that no passage prints; treat it as a hint about what was being "
        "looked for, not as the answer, and do not repeat it unless a passage now prints it.\n"
        "`answer` must be copied from the evidence: a name, title, date or quantity written exactly as "
        "some passage spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. Choose the passage "
        "that is about the entity the question asks about, not one that merely shares a word with it, "
        "and give the most likely short answer rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "pivot_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)
        self.repair = Module(
            inputs=["question", "rejected_draft", "evidence"],
            outputs=["answer"],
            instructions=self.REPAIR_INSTR,
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

    # -- the probe stage -------------------------------------------------------------------
    def _probe_targets(self, anchors: list[dict[str, Any]], pivots: list[str], question: str) -> list[str]:
        """Names worth looking up next, read straight off the anchor abstracts.

        The parent asks one LLM call to commit to two pivots. Here the call's answer is merely the head
        of a longer list: whatever the anchors quote or capitalise is a hop candidate, because a HotpotQA
        second article is nearly always a title the first article prints.
        """
        qk = self._content(question)
        out: list[str] = []
        seen: set[str] = set()

        def offer(name: str) -> None:
            name = " ".join((name or "").split()).strip(' .,;:\'’“”"-')
            if not (4 < len(name) < 70) or _PLACEHOLDER.search(name):
                return
            key = self._norm_phrase(name)
            if not key or key in seen:
                return
            seen.add(key)
            out.append(name)

        for p in pivots:
            offer(p)
        for h in anchors:
            body = h.get("text", "")
            body = body.split("|", 1)[1] if "|" in body else body
            for m in _QUOTED.finditer(body):
                offer(m.group(1))
        for h in anchors:
            body = h.get("text", "")
            body = body.split("|", 1)[1] if "|" in body else body
            for m in _PROPER.finditer(body):
                name = m.group(0).strip(" .,;:'’-")
                if len(name.split()) < 2 or self._content(name) <= qk:
                    continue
                offer(name)
        return out[: self.PROBE_MAX]

    def _confirm(self, target: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The index round-trip: does an article actually GO BY this name?

        A hop target harvested from prose is often a phrase, a role or a partial name. If nothing the
        index returns is titled that way, the target is not an entity and the whole probe is dropped --
        which is what keeps a generous enumeration from flooding the pool.
        """
        best, best_score = None, 0.0
        for h in hits:
            score = self._title_anchor(h.get("title", ""), [target])
            if score > best_score:
                best, best_score = h, score
        return best if best_score >= self.CONFIRM_MIN else None

    def _residual_fit(
        self, doc: dict[str, Any], question: str, prop: str, covered: set[str], idf: dict[str, float]
    ) -> float:
        """How much of what the question still wants this one abstract supplies.

        Scored against the question terms the anchors do NOT already carry, so an abstract that merely
        re-states the anchor -- the usual result of hopping on a name the anchor is about -- scores zero
        and is not admitted.
        """
        body = set(self._toks(f"{doc.get('title', '')} {doc.get('text', '')}"))
        gain = sum(w for t, w in idf.items() if t not in covered and t in body)
        if prop:
            pk = {t for t in self._toks(prop) if t not in _STOP and len(t) > 2}
            if pk:
                gain *= 1.0 + 0.4 * len(pk & body) / len(pk)
        return gain

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

    # -- the grounding test and its repair -------------------------------------------------
    def _norm_answer(self, text: str) -> str:
        """Comparison form: lowercase, articles and punctuation dropped, whitespace collapsed."""
        words = [w for w in _WORD.findall((text or "").lower()) if w not in {"a", "an", "the"}]
        return " ".join(words)

    def _grounded(self, answer: str, docs: list[dict[str, Any]], question: str) -> bool:
        """True when some read passage prints the answer verbatim AND is about what was asked.

        The second half matters: a common word or a stray name occurs in half the pool, so a passage
        only counts as support when it also carries at least two of the question's content words.
        """
        norm = self._norm_answer(answer)
        if not norm:
            return False
        if norm in {"yes", "no"}:
            return True
        qk = self._content(question)
        for h in docs:
            blob = f"{h.get('title', '')} {h.get('text', '')}"
            if norm in self._norm_answer(blob) and len(self._content(blob) & qk) >= 2:
                return True
        return False

    def _uncovered(self, question: str, docs: list[dict[str, Any]], limit: int = 5) -> str:
        """The question's content words that NO read passage mentions -- the unexplored facet."""
        seen: set[str] = set()
        for h in docs:
            seen |= set(self._toks(f"{h.get('title', '')} {h.get('text', '')}"))
        missing = [w for w in self._toks(question) if w not in _STOP and len(w) > 2 and w not in seen]
        out: list[str] = []
        for w in missing:
            if w not in out:
                out.append(w)
        return " ".join(out[:limit])

    def _repair_queries(self, draft: str, prop: str, question: str, already: list[str]) -> list[str]:
        """Searches anchored on the draft itself, plus the facet the pool never touched."""
        seen = [frozenset(self._content(q)) for q in already]
        draft = " ".join(draft.split())[:80]
        cands = [draft]
        if prop:
            cands.append(f"{draft} {prop}")
        out: list[str] = []
        for q in cands:
            k = frozenset(self._content(q))
            if not k or any(k <= s for s in seen):
                continue
            seen.append(k)
            out.append(q)
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

        # ---- the probe stage --------------------------------------------------------------
        anchor_docs = [h for _, h in ranked[: self.ANCHOR_DOCS]]
        covered: set[str] = set()
        for h in anchor_docs:
            covered |= set(self._toks(f"{h.get('title', '')} {h.get('text', '')}"))
        qwords = self._content(question)
        n_pool = max(len(ranked), 1)
        pool_df: dict[str, int] = {}
        for _, h in ranked:
            for w in set(self._toks(h.get("text", ""))):
                pool_df[w] = pool_df.get(w, 0) + 1
        idf = {w: math.log(1.0 + n_pool / (1.0 + pool_df.get(w, 0))) for w in qwords}

        targets = self._probe_targets(anchor_docs, pivots, question)
        budget = self.MAX_SEARCHES - len(queries) - self.MAX_REPAIR_QUERIES
        winners: list[tuple[float, str, dict[str, Any]]] = []
        probed: list[str] = []
        rejected: list[str] = []
        for target in targets:
            if budget <= 0:
                break
            hits = ctx.tools.retrieve_docs(target, k=self.PROBE_K)
            budget -= 1
            probed.append(target)
            doc = self._confirm(target, hits)
            if doc is None:
                rejected.append(f"{target} (no such article)")
                continue
            fit = self._residual_fit(doc, question, prop, covered, idf)
            if fit <= 0.0:
                rejected.append(f"{target} (adds nothing)")
                continue
            winners.append((fit, target, doc))
        winners.sort(key=lambda w: -w[0])
        admitted = winners[: self.PROBE_ACCEPT]

        # Everything else any probe returned is discarded here: only these documents cross over.
        admitted_docs = [d for _, _, d in admitted]
        admitted_keys = {(d.get("title") or "").strip().lower() for d in admitted_docs}
        hop = [t for _, t, _ in admitted]
        ranked = self._rerank(lists, anchors + pivots + hop, question, picked)
        evidence_docs = list(admitted_docs)
        for _, h in ranked:
            if len(evidence_docs) >= self.MAX_EVIDENCE:
                break
            if (h.get("title") or "").strip().lower() not in admitted_keys:
                evidence_docs.append(h)
        ctx.note(
            f"probes: {len(probed)} issued, {len(winners)} confirmed+useful, {len(admitted)} admitted",
            probed=probed,
            rejected=rejected,
            admitted=[f"{t} -> {d['title']} (fit {f:.2f})" for f, t, d in admitted],
            evidence=[h["title"] for h in evidence_docs],
        )

        evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer

        # The grounding test. A draft the read passages do not print is not an answer this pool can
        # justify, so the pool -- not the wording -- is what gets another chance.
        repaired = False
        repair_qs: list[str] = []
        if not self._grounded(answer, evidence_docs, question):
            asked = queries + hop
            repair_qs = self._repair_queries(answer, prop, question, asked)
            gap = self._uncovered(question, evidence_docs)
            if gap and len(repair_qs) < self.MAX_REPAIR_QUERIES:
                repair_qs.append(gap)
            repair_qs = repair_qs[: max(0, min(self.MAX_REPAIR_QUERIES, self.MAX_SEARCHES - len(queries) - len(probed)))]
            for q in repair_qs:
                lists.append(ctx.tools.retrieve_docs(q, k=self.K))
            if repair_qs:
                ranked = self._rerank(lists, anchors + pivots + [answer], question, picked)
                evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
                evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
                fixed = self.repair(
                    ctx.llm,
                    tag="repair",
                    question=question,
                    rejected_draft=answer,
                    evidence=evidence,
                )
                cand = self._clean(fixed.get("answer", ""))
                # Only accept the repair when it is better grounded than what it replaces; an
                # ungrounded replacement is no improvement over an ungrounded draft.
                if cand and not _NONANSWER.match(cand) and self._grounded(cand, evidence_docs, question):
                    answer = cand
                    repaired = True
            ctx.note(
                f"grounding test failed -> {len(repair_qs)} repair searches, repaired={repaired}",
                repair_queries=repair_qs,
                final=answer,
            )
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "probes": probed,
            "admitted": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
