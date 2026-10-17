"""The bridge entity must be corroborated by the pool AND by the index before the hop commits to it.

Mechanism change vs the parent (`grounded_answer_repair`). The parent asks one call to name the pivot
that stands between the question and the fact it wants, and then spends its whole second retrieval
round on that name. Its own traces show what happens when the first round did not surface the right
article: the call answers anyway, out of the model's memory or out of whatever passage shares a word
with the question, and names a person or work that has nothing to do with the question. The second
round then fills the pool with articles about the wrong entity, the answerer reads them, and the item
is lost with a confident, well-formed, wrong answer. The name is never checked before it is used.

This system makes the pivot decision EVIDENCE-BOUND by auditing candidates through the retriever
itself, which no earlier system in this line has used as a verifier rather than as a source:

  * a slate is assembled from the call's pivots AND from link candidates harvested deterministically
    out of the passages that are about entities the question names, so the model is not the only
    source of hypotheses;
  * each candidate is scored on whether the pool prints it at all, whether it is printed by a passage
    that is about something the question names, whether the index has an article actually titled with
    it, and whether that article points back at the question (mentions a name the question uses, or
    repeats its rare words);
  * candidates that the pool never prints and the index cannot title are treated as inventions and
    dropped; only corroborated candidates get the crossed property query.

When nothing is corroborated the system stops asserting a name at all and hops on the question's own
description of the entity instead, which is what the index can still match.

Calls: 3 LLM (4 with repair) and up to 8 retrievals (11 with repair).
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
_QUOTED = re.compile(r"[\"“']([^\"”']{3,60})[\"”']")
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
    K = 20
    PROBE_K = 8             # the audit only needs the head of the list: does this name title an article?
    MAX_FIRST_QUERIES = 3
    MAX_SLATE = 5           # candidates considered
    MAX_PROBES = 3          # candidates actually probed against the index
    MAX_PIVOTS = 2          # candidates that survive the audit and get a crossed query
    MAX_CROSSED = 2
    SHORTLIST = 14
    MAX_EVIDENCE = 12
    RRF_C = 10.0
    W_TITLE = 3.0
    W_IDF = 1.6
    W_PICKED = 0.9
    MAX_ANSWER_WORDS = 12
    MAX_REPAIR_QUERIES = 3
    CORROBORATION_FLOOR = 1.5

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
        "Copy each name EXACTLY as some passage spells it, or exactly as the question spells it. If no passage "
        "and not the question prints such a name, write `none` rather than a name you remember or guess: a "
        "guessed name sends the next search to the wrong article.\n"
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

    @staticmethod
    def _body(text: str) -> str:
        return text.split("|", 1)[1].strip() if "|" in text else text

    # -- the rerank (unchanged from the parent) --------------------------------------------
    def _title_anchor(self, title: str, anchors: list[str]) -> float:
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
                score = 1.0
            elif f" {bp} " in f" {ap} ":
                score = 0.85
            elif f" {ap} " in f" {bp} " and len(atoks) > 1:
                score = 0.6
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

    # -- the pivot slate and its audit -----------------------------------------------------
    def _question_anchors(self, question: str) -> list[str]:
        """The names the question itself asserts: quoted works and proper-noun runs."""
        out: list[str] = []
        for m in _QUOTED.finditer(question):
            name = m.group(1).strip(" .,;:'’-")
            if len(name) > 2 and name not in out:
                out.append(name)
        for m in _PROPER.finditer(question):
            name = m.group(0).strip(" .,;:'’-")
            if len(name) < 3 or self._norm_phrase(name) in {"what", "which", "who", "when", "where", "the"}:
                continue
            if len(name.split()) == 1 and question.strip().startswith(name):
                continue                      # the sentence-initial capital is not a name
            if name not in out:
                out.append(name)
        return out[:6]

    def _anchor_docs(self, docs: list[dict[str, Any]], anchors: list[str]) -> list[dict[str, Any]]:
        """Pool articles that ARE one of the entities the question names."""
        return [h for h in docs if self._title_anchor(h.get("title", ""), anchors) >= 0.6]

    def _harvest(self, docs: list[dict[str, Any]], question: str, limit: int) -> list[str]:
        """Link candidates: proper-noun runs printed by anchor articles that the question does not use."""
        qk = self._content(question)
        counts: dict[str, float] = {}
        for h in docs:
            # Sentence by sentence: a run read across a full stop is two names glued together.
            for sent in re.split(r"(?<=[.!?])\s+", self._body(h.get("text", ""))):
                for m in _PROPER.finditer(sent):
                    name = m.group(0).strip(" .,;:'’-")
                    if not (2 <= len(name.split()) <= 5) or len(name) < 5:
                        continue
                    nk = self._content(name)
                    if not nk or nk <= qk:
                        continue
                    counts[name] = counts.get(name, 0.0) + 1.0
            # Works are printed in quotes and are often a single word, which the run filter drops.
            for m in _QUOTED.finditer(self._body(h.get("text", ""))):
                name = m.group(1).strip(" .,;:'’-")
                nk = self._content(name)
                if len(name) < 4 or len(name.split()) > 6 or not nk or nk <= qk:
                    continue
                counts[name] = counts.get(name, 0.0) + 1.5
        return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]

    def _printed_in(self, name: str, docs: list[dict[str, Any]]) -> bool:
        n = self._norm_phrase(name)
        if not n:
            return False
        for h in docs:
            if f" {n} " in f" {self._norm_phrase(h.get('text', ''))} ":
                return True
        return False

    def _probe_scores(
        self,
        hits: list[dict[str, Any]],
        cand: str,
        anchors: list[str],
        question: str,
        prop: str,
    ) -> tuple[float, float]:
        """(does the index title an article with this name, does that article point back at the question)."""
        exists = back = 0.0
        ncand = set(self._norm_phrase(cand).split())
        akeys = [self._norm_phrase(a) for a in anchors]
        rare = self._content(question) | self._content(prop)
        for h in hits[:4]:
            title = _PAREN.sub("", h.get("title") or "").strip()
            ntitle = set(self._norm_phrase(title).split())
            if not ntitle or not ncand:
                continue
            overlap = len(ntitle & ncand) / max(len(ntitle | ncand), 1)
            if overlap < 0.6:
                continue
            exists = 1.0
            blob = f" {self._norm_phrase(h.get('text', ''))} "
            if any(a and f" {a} " in blob for a in akeys):
                back = max(back, 1.0)
            elif len(self._content(h.get("text", "")) & rare) >= 3:
                back = max(back, 0.5)
        return exists, back

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

    def _norm_answer(self, text: str) -> str:
        words = [w for w in _WORD.findall((text or "").lower()) if w not in {"a", "an", "the"}]
        return " ".join(words)

    def _grounded(self, answer: str, docs: list[dict[str, Any]], question: str) -> bool:
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
        seen: set[str] = set()
        for h in docs:
            seen |= set(self._toks(f"{h.get('title', '')} {h.get('text', '')}"))
        missing = [w for w in self._toks(question) if w not in _STOP and len(w) > 2 and w not in seen]
        out: list[str] = []
        for w in missing:
            if w not in out:
                out.append(w)
        return " ".join(out[:limit])

    def _description_query(self, question: str, prop: str, anchors: list[str]) -> str:
        """What to search when no name can be corroborated: the question minus the names it already has."""
        used = {w for a in anchors for w in self._toks(a)}
        words = [
            w for w in _WORD.findall(question)
            if w.lower() not in _STOP and len(w) > 2 and w.lower() not in used
        ]
        return " ".join(dict.fromkeys([*words[:6], *prop.split()[:3]]))

    def _repair_queries(self, draft: str, prop: str, question: str, already: list[str]) -> list[str]:
        seen = [frozenset(self._content(q)) for q in already]
        draft = " ".join(draft.split())[:80]
        out: list[str] = []
        for q in ([draft, f"{draft} {prop}"] if prop else [draft]):
            k = frozenset(self._content(q))
            if not k or any(k <= s for s in seen):
                continue
            seen.append(k)
            out.append(q)
        return out

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        qanchors = self._question_anchors(question)

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = (queries[: self.MAX_FIRST_QUERIES - 1] + [question]) if queries else [question]

        lists = [ctx.tools.retrieve_docs(q, k=self.K) for q in queries]
        anchors = [question] + queries
        ranked = self._rerank(lists, anchors, question, picked=set())

        shortlist = [h for _, h in ranked[: self.SHORTLIST]]
        numbered = [f"{i + 1}. {h['text']}" for i, h in enumerate(shortlist)]
        judged = self.bridge(ctx.llm, tag="bridge", question=question, passages=numbered)

        picked: set[str] = set()
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(shortlist):
                picked.add((shortlist[idx].get("title") or "").strip().lower())
        prop = " ".join((judged.get("missing_property", "") or "").split())[:60]
        if prop.lower().startswith("none"):
            prop = ""

        proposed = [
            n for n in self._lines(judged.get("pivot_names", ""), 3)
            if len(n) > 2 and not _PLACEHOLDER.search(n)
        ]
        # Hypotheses the model did not have to invent: names printed by the articles that ARE the
        # entities the question names, plus the passages the model itself marked useful.
        source_docs = self._anchor_docs(shortlist, qanchors) or [
            h for h in shortlist if (h.get("title") or "").strip().lower() in picked
        ][:3]
        harvested = self._harvest(source_docs, question, self.MAX_SLATE)

        slate: list[str] = []
        for name in [*proposed, *harvested]:
            if not any(self._norm_phrase(name) == self._norm_phrase(s) for s in slate):
                slate.append(name)
        slate = slate[: self.MAX_SLATE]

        # Audit: what the pool already says about each candidate, before any search is spent.
        prior: dict[str, float] = {}
        for name in slate:
            in_pool = 1.0 if (self._printed_in(name, shortlist) or self._norm_phrase(name) in self._norm_phrase(question)) else 0.0
            anchored = 1.2 if any(self._printed_in(name, [h]) for h in source_docs) else 0.0
            prior[name] = in_pool + anchored + (0.4 if name in proposed else 0.0)
        slate.sort(key=lambda n: -prior[n])

        # Round trip through the index for the leading candidates: is this a real article, and does
        # it point back at the question?
        audit: list[tuple[float, str]] = []
        probed: list[str] = []
        for name in slate[: self.MAX_PROBES]:
            hits = ctx.tools.retrieve_docs(name, k=self.PROBE_K)
            lists.append(hits)
            probed.append(name)
            exists, back = self._probe_scores(hits, name, qanchors, question, prop)
            audit.append((prior[name] + 1.0 * exists + 1.3 * back, name))
        for name in slate[self.MAX_PROBES:]:
            audit.append((prior[name], name))
        audit.sort(key=lambda sn: -sn[0])

        pivots = [n for s, n in audit if s >= self.CORROBORATION_FLOOR][: self.MAX_PIVOTS]
        crossed: list[str] = []
        if pivots and prop:
            foreign = {w for n in pivots for w in self._toks(n)}
            for name in pivots[: self.MAX_CROSSED]:
                tail = " ".join(w for w in prop.split() if w.lower() not in foreign)
                q = f"{name} {tail}".strip()
                if q and q not in crossed:
                    crossed.append(q)
        if not pivots:
            # Nothing survived the audit: search the description instead of asserting a name.
            desc = self._description_query(question, prop, qanchors)
            if desc:
                crossed.append(desc)
        for q in crossed[: self.MAX_CROSSED]:
            lists.append(ctx.tools.retrieve_docs(q, k=self.K))

        ranked = self._rerank(lists, anchors + pivots, question, picked)
        evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
        ctx.note(
            f"slate={slate} audited -> pivots={pivots}",
            proposed=proposed,
            harvested=harvested,
            probed=probed,
            audit=[[round(s, 2), n] for s, n in audit],
            crossed=crossed,
            evidence=[h["title"] for h in evidence_docs],
        )

        evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer

        repaired = False
        repair_qs: list[str] = []
        if not self._grounded(answer, evidence_docs, question):
            repair_qs = self._repair_queries(answer, prop, question, queries + probed + crossed)
            gap = self._uncovered(question, evidence_docs)
            if gap and len(repair_qs) < self.MAX_REPAIR_QUERIES:
                repair_qs.append(gap)
            repair_qs = repair_qs[: self.MAX_REPAIR_QUERIES]
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
                if cand and not _NONANSWER.match(cand) and self._grounded(cand, evidence_docs, question):
                    answer = cand
                    repaired = True

        return {
            "answer": answer,
            "queries": queries,
            "slate": slate,
            "probed": probed,
            "pivots": pivots,
            "missing_property": prop,
            "crossed": crossed,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
