"""Audit the drafted answer against the kind of thing the question asks for.

Mechanism change vs the parent (`grounded_answer_repair`). The parent's grounding test asks whether the
draft is PRINTED in the evidence. That catches hallucinated strings, but it is blind to the opposite
failure, which the parent's own traces show is the one left over: a draft that is printed everywhere and is
still the wrong KIND of thing. Asked who manufactured a ride, the answerer hands back the ride; asked which
director made a film, it hands back the film; asked for the installment, it hands back a title. The string
is quoted from the evidence, so the grounding test approves it, and there is no other stage that can object.

This system adds a KIND AUDIT after the answer is settled. Wikipedia abstracts have a rigid opening --
`X is a 1995 American film`, `X (born 1962) is an American football coach` -- so an entity's own article
states its category in its first sentence. The audit therefore:

  1. reads the kind the question asks for off the interrogative (`who` wants an agent; `which county`,
     `what album` want the named category) -- and stays silent when the question asks for something that
     has no category, such as a quantity or an abstract noun;
  2. finds the draft's HOME ARTICLE, the read article the draft names, and reads the categories its first
     sentence and its title state, including the disambiguator a title carries in brackets;
  3. fires only on an outright CONFLICT: the home article states a category and none of the categories it
     states is the one the question wants.

A conflict means the answerer returned a neighbouring entity instead of the one asked for, and the entity
asked for is usually already described in that same article. The repair therefore spends one answering call
that is told the required kind and is forbidden the conflicting draft, backed by at most two searches that
cross the home article with the required kind; the replacement is accepted only if it passes the same audit
and is itself printed in the evidence. Otherwise the draft stands.

Calls: 3 LLM and 5-6 retrievals normally, +1 LLM when the grounding test fires, +1 LLM and up to 2 searches
when the kind audit fires.
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
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"“(A-Z0-9])")
_WHKIND = re.compile(r"\b(?:what|which)\b", re.IGNORECASE)
_AGENT_Q = re.compile(r"\b(?:who|whom|whose)\b", re.IGNORECASE)
_TITLE_TAG = re.compile(r"\(([^)]*)\)\s*$")

# Closed categories. Wikipedia's first sentence states an entity's category almost without exception,
# so a category word found there is a reliable type signal; words outside these sets are ignored, which
# keeps the audit silent rather than wrong whenever the question asks for something uncategorisable.
_KINDS: dict[str, set[str]] = {
    "agent": {
        "actor", "actress", "singer", "songwriter", "director", "writer", "author", "player",
        "musician", "politician", "artist", "coach", "president", "footballer", "journalist", "poet",
        "composer", "rapper", "producer", "comedian", "scientist", "mathematician", "physicist",
        "novelist", "engineer", "architect", "painter", "photographer", "chef", "lawyer", "judge",
        "general", "soldier", "king", "queen", "emperor", "minister", "governor", "senator", "founder",
        "businessman", "entrepreneur", "wrestler", "boxer", "driver", "pitcher", "quarterback",
        "company", "corporation", "band", "team", "club", "university", "college", "school", "label",
        "network", "organization", "agency", "publisher", "studio", "airline", "manufacturer", "firm",
        "duo", "group", "orchestra", "institute", "society", "association", "foundation",
    },
    "work": {
        "film", "movie", "album", "song", "single", "novel", "book", "series", "show", "game", "play",
        "magazine", "newspaper", "comic", "opera", "painting", "sitcom", "documentary", "musical",
        "poem", "story", "episode", "soundtrack", "franchise", "anthology", "manga", "anime",
    },
    "place": {
        "city", "town", "village", "county", "state", "province", "country", "river", "lake",
        "mountain", "island", "park", "street", "road", "highway", "region", "district", "borough",
        "neighbourhood", "neighborhood", "municipality", "capital", "township", "castle", "stadium",
        "airport", "station", "bridge", "museum", "hotel", "prefecture", "suburb", "peak", "valley",
    },
    "thing": {
        "coaster", "attraction", "ride", "aircraft", "airliner", "ship", "locomotive", "species",
        "genus", "award", "festival", "battle", "war", "massacre", "election", "tournament", "race",
        "championship", "hurricane", "earthquake", "mission", "spacecraft", "weapon", "rifle", "tank",
        "engine", "processor", "console", "breed", "plant", "disease", "protein", "mineral",
    },
}
_KIND_OF = {word: kind for kind, words in _KINDS.items() for word in words}

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

    KIND_INSTR = (
        "Answer the question from the numbered `evidence` passages. An earlier attempt returned "
        "`wrong_kind_draft`, which names the wrong sort of thing: the question asks for a "
        "`required_kind`, and that draft is not one. The thing being asked for is usually described "
        "inside the very passage the draft names, so read that passage again and take the right thing "
        "out of it.\n"
        "`answer` must be a `required_kind`, copied from the evidence exactly as some passage spells it, "
        "and it must not be `wrong_kind_draft`. Never write a sentence, never restate the question, never "
        "explain, and give the most likely short answer rather than saying it cannot be determined."
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
        self.rekind = Module(
            inputs=["question", "required_kind", "wrong_kind_draft", "evidence"],
            outputs=["answer"],
            instructions=self.KIND_INSTR,
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


    # -- the kind audit --------------------------------------------------------------------
    def _wanted_kind(self, question: str) -> str:
        """The category the question asks for, or "" when it asks for something uncategorisable."""
        # A named category beats a bare `who`: `which actress, who won ...` asks for an actress.
        for m in _WHKIND.finditer(question):
            for tok in question[m.end():].split()[:4]:
                bare = tok.strip(".,;:?!\"'()").lower()
                if not bare or bare in _STOP:
                    break
                kind = _KIND_OF.get(bare) or _KIND_OF.get(bare.rstrip("s"))
                if kind:
                    return kind
        return "agent" if _AGENT_Q.search(question) else ""

    def _stated_kinds(self, doc: dict[str, Any]) -> set[str]:
        """The categories an article states about itself: its opening sentence and its title tag."""
        text = doc.get("text", "") or ""
        title = (doc.get("title") or "").strip()
        body = text.split("|", 1)[1] if "|" in text else text
        lead = _SENT.split(body.strip())[0] if body.strip() else ""
        tag = _TITLE_TAG.search(title)
        words = self._toks(lead)[:45] + (self._toks(tag.group(1)) if tag else [])
        found = set()
        for w in words:
            kind = _KIND_OF.get(w) or _KIND_OF.get(w.rstrip("s"))
            if kind:
                found.add(kind)
        return found

    def _home_article(self, answer: str, docs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The read article the draft NAMES -- the one whose own first sentence types the draft."""
        key = self._norm_answer(answer)
        if not key or len(key) < 3:
            return None
        for h in docs:
            if self._norm_answer(_PAREN.sub("", h.get("title") or "")) == key:
                return h
        for h in docs:
            if self._norm_answer(h.get("title") or "") == key:
                return h
        return None

    def _kind_conflict(self, answer: str, docs: list[dict[str, Any]], question: str):
        """(wanted, home) when the draft's own article states categories and none is the wanted one."""
        wanted = self._wanted_kind(question)
        if not wanted or self._norm_answer(answer) in {"", "yes", "no"}:
            return None
        home = self._home_article(answer, docs)
        if home is None:
            return None
        stated = self._stated_kinds(home)
        if not stated or wanted in stated:
            return None
        return wanted, home

    def _kind_queries(self, home_title: str, wanted: str, question: str, already: list[str]) -> list[str]:
        """Cross the article the draft names with the relation the question asks about."""
        seen = [frozenset(self._content(q)) for q in already]
        base = _PAREN.sub("", home_title).strip()
        rel = [w for w in _WORD.findall(question) if w.lower() not in _STOP and len(w) > 2]
        rel = [w for w in rel if w.lower() not in self._content(base)][-4:]
        out: list[str] = []
        for tail in (" ".join(rel), wanted):
            q = f"{base} {tail}".strip()
            k = frozenset(self._content(q))
            if not k or any(k <= t for t in seen):
                continue
            seen.append(k)
            out.append(q)
        return out[:2]

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

        # The kind audit. A draft that is printed but is the wrong sort of thing is a different failure
        # from a draft that is not printed, and the evidence itself states the sort of thing it is.
        conflict = self._kind_conflict(answer, evidence_docs, question)
        rekinded = False
        kind_qs: list[str] = []
        if conflict is not None:
            wanted, home = conflict
            spent = len(queries) + len(hop) + len(repair_qs)
            if spent < 10:
                kind_qs = self._kind_queries(home.get("title", ""), wanted, question, queries + hop + repair_qs)
                kind_qs = kind_qs[: 10 - spent]
                for q in kind_qs:
                    lists.append(ctx.tools.retrieve_docs(q, k=self.K))
                if kind_qs:
                    ranked = self._rerank(lists, anchors + pivots, question, picked)
                    evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
                    if home not in evidence_docs:
                        evidence_docs = [home] + evidence_docs[: self.MAX_EVIDENCE - 1]
                    evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
            fixed = self.rekind(
                ctx.llm,
                tag="rekind",
                question=question,
                required_kind=wanted,
                wrong_kind_draft=answer,
                evidence=evidence,
            )
            cand = self._clean(fixed.get("answer", ""))
            # Accept only a replacement that the evidence prints and that the audit no longer rejects.
            if (
                cand
                and not _NONANSWER.match(cand)
                and self._norm_answer(cand) != self._norm_answer(answer)
                and self._grounded(cand, evidence_docs, question)
                and self._kind_conflict(cand, evidence_docs, question) is None
            ):
                answer = cand
                rekinded = True
            ctx.note(
                f"kind audit: wanted={conflict[0]} home={home.get('title')} -> rekinded={rekinded}",
                kind_queries=kind_qs,
                final=answer,
            )
        return {
            "answer": answer,
            "kind_queries": kind_qs,
            "rekinded": rekinded,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
