"""Evidence handed over as a two-sided ledger with an explicit intersection, not as one flat list.

Mechanism change vs the parent (`grounded_answer_repair`). Every question here is settled by two
articles that play different roles: one the question NAMES, and one it only reaches through the first.
The parent flattens both roles, plus ten distractors, into a single numbered list and leaves the
answerer to rediscover the structure inside its own reasoning. The measurable cost shows up in the
items where both required articles were read and the answer was still wrong: the answerer returns the
entity the question already named, or it returns both halves joined by `and`, or -- on questions that
ask what two things share -- it paraphrases the shared property into a synonym the gold answer does
not use, or reports each side separately instead of the one thing they agree on.

This system replaces the flat list with a LEDGER built deterministically before the answering call:

  named      articles whose own title the question quotes -- the entities it starts from;
  reached    articles arrived at through the hop: titles matching a pivot, or titles that a `named`
             article's text prints, i.e. links out of the starting article;
  shared     the literal intersection of the two sides -- words and short phrases that some `named`
             article and some `reached` article both print, question vocabulary removed, ranked by
             how rare they are in the pool.

and it routes the answer contract on the question's own shape, computed from its wording rather than
from a model judgement: a whether-question must answer yes or no; a question offering two named
alternatives must return exactly one of them; a question asking what two things have in common must
return the shared property itself, and `shared` is where that property is literally written down; any
other question is a bridge, whose answer is a fact printed on the `reached` side -- the article the
question names is what it asks *through*, never what it asks *for*.

The intersection is the part the answerer cannot compute for itself: agreeing on a common property
means comparing two abstracts word by word, which is precisely what a language model paraphrases
instead of doing.

Calls: 3 LLM and 5-6 retrievals normally, 4 LLM and up to 9 retrievals when the grounding test fires.
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
_AUX = re.compile(r"^\s*(?:are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE)
_SHARED_Q = re.compile(r"\bin\s+common\b|\bboth\b|\bsame\b|\bshare[ds]?\b", re.IGNORECASE)
_ALT_Q = re.compile(r",?\s+\bor\b\s+[^,?]{2,70}\?*\s*$", re.IGNORECASE)
_CAPRUN = re.compile(r"\b[A-Z][\w'\u2019.-]*(?:\s+[A-Z][\w'\u2019.-]*){0,2}")
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
        "The evidence is split by role. `named_articles` are the articles the question itself names; "
        "`reached_articles` are the articles those lead to; `shared_phrases` are the words and phrases "
        "that an article on each side both print, listed rarest first. Some passages on both sides are "
        "irrelevant.\n"
        "`answer_form` states what shape the answer has to take; obey it exactly.\n"
        "`answer` must be the bare answer and nothing else -- a name, title, date, quantity or single "
        "word, spelled exactly as the evidence spells it. Never write a sentence, never restate the "
        "question, never explain, never join two answers with `and` unless the evidence prints them as "
        "one name. If the evidence is incomplete, give the most likely short answer anyway rather than "
        "saying it cannot be determined."
    )

    REPAIR_INSTR = (
        "The evidence is split by role: `named_articles` are the articles the question names, "
        "`reached_articles` are the articles those lead to. An earlier attempt produced "
        "`rejected_draft`, a string that no passage prints; treat it as a hint about what was being "
        "looked for, not as the answer, and do not repeat it unless a passage now prints it.\n"
        "`answer_form` states what shape the answer has to take; obey it exactly.\n"
        "`answer` must be copied from the evidence: a name, title, date, quantity or single word "
        "written exactly as some passage spells it. Never write a sentence, never restate the "
        "question, never explain. Choose the passage that is about the entity the question asks "
        "about, not one that merely shares a word with it, and give the most likely short answer "
        "rather than saying it cannot be determined."
    )

    FORM_YESNO = (
        "The question asks whether something holds of the things it names. `answer` is the single "
        "word yes or no, nothing else."
    )
    FORM_ALTERNATIVE = (
        "The question offers named alternatives and asks which one. `answer` is exactly one of those "
        "alternatives, copied with the spelling the question uses. Never answer with both, never "
        "answer `same` or `neither`, never add a reason."
    )
    FORM_SHARED = (
        "The question asks what the named things have in common. `answer` is that shared property "
        "itself -- the one word or short phrase that describes both of them, preferring a phrase the "
        "evidence actually prints (see `shared_phrases`) over a synonym of your own. It is never one "
        "of the named things, never a property of only one of them, and never two properties joined "
        "by `and`: if each side has its own, give the more general term that covers both."
    )
    FORM_BRIDGE = (
        "The question names one thing in order to reach another; what it asks for is a fact stated on "
        "the `reached_articles` side. `answer` is that fact -- a name, title, date, quantity or single "
        "word -- and it is almost never the entity the question already named. Give the bare value: "
        "if the question asks how many, answer with the number alone; if it asks which one of an "
        "ordered set, answer with the position, not the thing's title."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "pivot_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "answer_form", "named_articles", "reached_articles", "shared_phrases"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )
        self.repair = Module(
            inputs=["question", "answer_form", "rejected_draft", "named_articles", "reached_articles"],
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


    # -- the ledger ------------------------------------------------------------------------
    def _shape(self, question: str) -> tuple[str, str]:
        """Route on the question's own wording: whether / which-of-these / what-in-common / bridge."""
        q = question.strip()
        alt = bool(_ALT_Q.search(q))
        if _AUX.match(q):
            return ("alternative", self.FORM_ALTERNATIVE) if alt else ("yesno", self.FORM_YESNO)
        if alt:
            return "alternative", self.FORM_ALTERNATIVE
        if _SHARED_Q.search(q):
            return "shared", self.FORM_SHARED
        return "bridge", self.FORM_BRIDGE

    def _sides(
        self,
        docs: list[dict[str, Any]],
        question: str,
        pivots: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split the read evidence into what the question names and what that leads to.

        An article belongs to the named side when the question quotes its title. It belongs to the
        reached side when a pivot names it, or when some named article's own text prints its title --
        a link out of the starting article. Everything else falls to the reached side too, since the
        question did not start there.
        """
        named: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for h in docs:
            if self._title_anchor(h.get("title", ""), [question]) >= 0.8:
                named.append(h)
            else:
                rest.append(h)
        if not named:                       # nothing the question quotes was retrieved
            return [], docs
        named_blob = " ".join(self._norm_phrase(h.get("text", "")) for h in named)
        def link_rank(h: dict[str, Any]) -> tuple[int, int]:
            bare = self._norm_phrase(_PAREN.sub("", h.get("title", "")))
            by_pivot = any(self._title_anchor(h.get("title", ""), [p]) >= 0.8 for p in pivots)
            linked = bool(bare) and f" {bare} " in f" {named_blob} "
            return (0 if by_pivot else 1, 0 if linked else 1)
        rest.sort(key=link_rank)
        return named, rest

    def _shared_phrases(
        self,
        named: list[dict[str, Any]],
        reached: list[dict[str, Any]],
        question: str,
        limit: int = 10,
    ) -> list[str]:
        """Words and short phrases that an article on each side both print, rarest first.

        This is the one thing the answerer cannot do in its head: deciding what two abstracts agree on
        requires comparing them literally, and a model asked to do it returns a paraphrase instead.
        """
        if not named or not reached:
            return []
        qk = self._content(question)

        def units(group: list[dict[str, Any]]) -> dict[str, str]:
            found: dict[str, str] = {}
            for h in group:
                text = f"{h.get('title', '')} {h.get('text', '')}"
                for w in _WORD.findall(text):
                    lw = w.lower()
                    if lw in _STOP or len(lw) < 4 or lw in qk or lw.isdigit():
                        continue
                    found.setdefault(lw, w)
                for m in _CAPRUN.finditer(h.get("text", "")):
                    phrase = m.group(0).strip(" .,;:'\u2019-")
                    key = self._norm_phrase(phrase)
                    if len(key.split()) < 2 or not key or set(key.split()) <= qk:
                        continue
                    found.setdefault(key, phrase)
            return found

        left, right = units(named), units(reached)
        pool = named + reached
        df: dict[str, int] = {}
        for h in pool:
            blob = self._norm_phrase(f"{h.get('title', '')} {h.get('text', '')}")
            for key in set(left) & set(right):
                if f" {key} " in f" {blob} ":
                    df[key] = df.get(key, 0) + 1
        shared = sorted(set(left) & set(right), key=lambda k: (df.get(k, 99), -len(k.split())))
        out: list[str] = []
        for key in shared:
            surface = left[key] if len(left[key]) >= len(right[key]) else right[key]
            if surface not in out:
                out.append(surface)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _block(docs: list[dict[str, Any]]) -> list[str]:
        return [h["text"] for h in docs]

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

        shape, form = self._shape(question)
        named, reached = self._sides(evidence_docs, question, pivots)
        shared = self._shared_phrases(named, reached, question) if shape == "shared" else []
        ctx.note(
            f"ledger: shape={shape}, {len(named)} named / {len(reached)} reached",
            named=[h["title"] for h in named],
            reached=[h["title"] for h in reached],
            shared=shared,
        )

        out = self.answer(
            ctx.llm,
            tag="answer",
            question=question,
            answer_form=form,
            named_articles=self._block(named),
            reached_articles=self._block(reached),
            shared_phrases=shared,
        )
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
            repair_qs = repair_qs[: self.MAX_REPAIR_QUERIES]
            for q in repair_qs:
                lists.append(ctx.tools.retrieve_docs(q, k=self.K))
            if repair_qs:
                ranked = self._rerank(lists, anchors + pivots + [answer], question, picked)
                evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
                named, reached = self._sides(evidence_docs, question, pivots)
                fixed = self.repair(
                    ctx.llm,
                    tag="repair",
                    question=question,
                    answer_form=form,
                    rejected_draft=answer,
                    named_articles=self._block(named),
                    reached_articles=self._block(reached),
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
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
            "shape": shape,
            "n_named": len(named),
        }
