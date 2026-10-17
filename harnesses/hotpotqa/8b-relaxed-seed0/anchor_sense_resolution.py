"""Every retrieved article that carries an anchor's name is a rival SENSE of it, and only one is meant.

Mechanism change vs the parent (`grounded_answer_repair`). The parent's rerank rewards an article for
having a title the question quotes (`W_TITLE`), and it applies that reward to every article that quotes
it. When the anchor phrase is ambiguous -- a short film title, a song name, a place name, a common
person's name -- the index returns several unrelated articles that all wear that name, they all collect
the full title bonus, and the most popular sense wins the reading window because nothing in the pipeline
can tell the senses apart. Everything downstream then inherits the wrong article: the bridge call reads
a name out of it, the hop queries are built from that name, and the repair round re-anchors on it again.
On the parent's traces this is the single largest error family -- roughly half of the items whose pool
never contained a required article had committed, in round one, to a same-named article about something
else.

This system makes sense selection an explicit step. Articles are grouped into TITLE FAMILIES: all pooled
articles whose bare title (parenthetical qualifier stripped) is the same phrase, or whose bare title
equals the same anchor the question quotes. Within a family, the members are rivals for one slot, and the
winner is decided by RESIDUAL FIT -- how much of the question that is *not* the anchor itself the article
accounts for. The residual is the question's content words minus the anchor's own words, weighted by
inverse document frequency, and a title's parenthetical qualifier (`(2011 film)`, `(album)`, `(singer)`)
is scored against it too, since a qualifier is exactly the word a disambiguation page would use. The
family winner keeps the title bonus; its rivals lose it and are pushed down, so a homonym flood can no
longer crowd the window.

When the best member of a family still explains almost none of the residual, no retrieved sense of the
anchor is the one the question means, and the anchor is re-searched as `anchor + residual`, a
disambiguating query the parent never issues because its queries are bare titles by design. Those
searches come out of the retrieval budget the parent leaves unspent, not out of its LLM calls.

Calls: 3 LLM and 5-8 retrievals normally, 4 LLM and up to 11 retrievals when the grounding test fires.
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
_QUAL = re.compile(r"\(([^)]*)\)\s*$")
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
    MAX_REPAIR_QUERIES = 3
    W_SENSE = 1.0           # how hard a beaten sense of a contested name loses its title bonus
    SENSE_EPS = 0.02        # a family the residual cannot separate is left alone
    SENSE_TIE = 0.15        # relative fit gap below which the rivals count as unresolved
    MAX_DISAMB_QUERIES = 2
    DISAMB_WORDS = 4

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


    @staticmethod
    def _stem(word: str) -> str:
        """Crude morphology so `director` and `directors` count as the same constraint."""
        for suf in ("ies", "ing", "ed", "es", "s"):
            if len(word) > len(suf) + 3 and word.endswith(suf):
                return word[: -len(suf)] + ("y" if suf == "ies" else "")
        return word

    def _stems(self, text: str) -> set[str]:
        return {self._stem(w) for w in self._toks(text)}

    @staticmethod
    def _bare_title(title: str) -> str:
        return _PAREN.sub("", title or "").strip()

    @staticmethod
    def _qualifier(title: str) -> str:
        m = _QUAL.search(title or "")
        return m.group(1) if m else ""

    # -- sense resolution ------------------------------------------------------------------
    def _resolve_senses(
        self,
        docs: dict[str, dict[str, Any]],
        idf: dict[str, float],
        anchors: list[str],
        question: str,
    ) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
        """Decide which article a contested name refers to, and say when none of them does.

        A TITLE FAMILY is every pooled article sharing one bare title -- `Lucky`, `Lucky (2011 film)`
        and `Lucky (2017 film)` are one family. A family is contested when the question or one of its
        queries actually quotes that name, which is exactly when the parent hands every member the
        same title bonus. The members are then rivals for one slot and are separated by RESIDUAL FIT:
        the share of the question's idf mass, counting only words the name itself does not contain,
        that the article's text or its parenthetical qualifier accounts for. The winner keeps its
        title bonus; each rival loses its own and is pushed down in proportion to the gap.

        Returns the additive score adjustments and the anchors no retrieved sense explains.
        """
        fams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key, h in docs.items():
            bare = self._norm_phrase(self._bare_title(h.get("title", "")))
            if bare:
                fams.setdefault(bare, []).append((key, h))

        adjust: dict[str, float] = {}
        weak: list[tuple[str, str, float]] = []
        qtoks = self._toks(question)
        for bare, members in fams.items():
            ta_name = self._title_anchor(bare, anchors)
            if ta_name < 0.8:
                continue                                   # a name nothing in the question quotes
            fam_stems = {self._stem(w) for w in bare.split()}
            resid = {self._stem(w): v for w, v in idf.items()}
            resid = {w: v for w, v in resid.items() if w not in fam_stems}
            total = sum(resid.values())
            if total <= 0:
                continue
            fits: list[tuple[float, str, dict[str, Any]]] = []
            for key, h in members:
                body = self._stems(h.get("text", "")) | self._stems(self._qualifier(h.get("title", "")))
                fits.append((sum(v for w, v in resid.items() if w in body) / total, key, h))
            fits.sort(key=lambda f: -f[0])
            best = fits[0][0]

            # Demotion is proportional to how clearly the residual separates the rivals, so a family
            # the question cannot resolve is left exactly as the parent ranked it.
            if len(fits) > 1:
                for fit, key, h in fits[1:]:
                    share = (fit + self.SENSE_EPS) / (best + self.SENSE_EPS)
                    ta = self._title_anchor(h.get("title", ""), anchors)
                    if share < 1.0:
                        adjust[key] = -self.W_TITLE * ta * (1.0 - share) * self.W_SENSE

            # Two situations mean the pool cannot say which article the name refers to: several
            # rivals the residual ranks equally, or a single exactly-named article whose abstract
            # shares nothing at all with the rest of the question.
            tie = len(fits) > 1 and best > 0.0 and (best - fits[1][0]) / best < self.SENSE_TIE
            # A short name with nothing behind it is the homonym-prone case; a long, specific
            # name that simply has a terse abstract is not worth another search.
            # Only an EXACT name match counts here: a short title that merely sits inside a
            # longer phrase the question uses is a fragment, and demoting it is enough.
            orphan = best <= 0.0 and ta_name >= 0.99 and len(bare.split()) <= 3
            if not (tie or orphan):
                continue
            extra = sorted(
                {w for w in qtoks if w not in _STOP and len(w) > 2 and self._stem(w) not in fam_stems},
                key=lambda w: -resid.get(self._stem(w), 0.0),
            )[: self.DISAMB_WORDS]
            keep = set(extra)
            seen: list[str] = []
            for w in qtoks:
                if w in keep and w not in seen:
                    seen.append(w)
            if len(seen) < 2:
                continue
            display = self._bare_title(fits[0][2].get("title", "")) or bare
            weak.append((display, " ".join(seen), best))
        weak.sort(key=lambda w: w[2])
        return adjust, weak

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
        weak_out: list[tuple[str, str, float]] | None = None,
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

        # Which article a contested name refers to is decided here, before anything is ranked.
        adjust, weak = self._resolve_senses(docs, idf, anchors, question)
        if weak_out is not None:
            weak_out.extend(weak)

        scored: list[tuple[float, dict[str, Any]]] = []
        for key, h in docs.items():
            body = set(self._toks(h.get("text", "")))
            cover = sum(v for w, v in idf.items() if w in body) / total_idf
            score = (
                fused[key] * (self.RRF_C + 1.0)
                + self.W_TITLE * self._title_anchor(h.get("title", ""), anchors)
                + self.W_IDF * cover
                + (self.W_PICKED if key in picked else 0.0)
                + adjust.get(key, 0.0)
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
        weak: list[tuple[str, str, float]] = []
        ranked = self._rerank(lists, anchors, question, picked=set(), weak_out=weak)
        ctx.note(
            f"round1: {len(queries)} queries at k={self.K} -> {len(ranked)} unique articles",
            queries=queries,
            top=[h["title"] for _, h in ranked[:8]],
            weak_anchors=[(w[0], w[1], round(w[2], 3)) for w in weak],
        )

        # An anchor no retrieved sense explains is re-searched as name + the rest of the question.
        disamb: list[str] = []
        asked = [frozenset(self._content(q)) for q in queries if q != question]
        for name, residual, _fit in weak:
            if not residual:
                continue
            q = f"{name} {residual}".strip()
            k = frozenset(self._content(q))
            if not k or any(k <= a for a in asked):
                continue
            asked.append(k)
            disamb.append(q)
            if len(disamb) >= self.MAX_DISAMB_QUERIES:
                break
        if disamb:
            for q in disamb:
                lists.append(ctx.tools.retrieve_docs(q, k=self.K))
            ranked = self._rerank(lists, anchors, question, picked=set())
            ctx.note(
                f"sense disambiguation: {len(disamb)} searches",
                disamb_queries=disamb,
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

        hop = self._hop_queries(pivots, prop, question, queries + disamb)
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
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer

        # The grounding test. A draft the read passages do not print is not an answer this pool can
        # justify, so the pool -- not the wording -- is what gets another chance.
        repaired = False
        repair_qs: list[str] = []
        if not self._grounded(answer, evidence_docs, question):
            asked_qs = queries + disamb + hop
            repair_qs = self._repair_queries(answer, prop, question, asked_qs)
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
            "disamb_queries": disamb,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
