"""A failed draft gets NEW EVIDENCE, not a second opinion on the evidence that already failed it.

Mechanism change vs the parent (`shared_attribute_intersection`). The parent already knows when its draft
is bad: a deterministic grounding test asks whether any passage it read prints the draft verbatim. That
test fires on roughly a tenth of the items and the accuracy inside that slice is barely one in two. But the
parent's remedy is to re-rank the SAME documents (plus a couple of searches anchored on the draft it just
rejected) and ask the model to think again. Replaying the parent's traces shows the remedy converts
essentially nothing: the model reads the same abstracts and re-derives the same answer, because the reason
the draft was ungrounded is that the article carrying the fact was never in the window, not that the model
mis-read the window.

This system replaces the remedy. When a draft is rejected it opens a SECOND, DISJOINT evidence window:

  * the link names are mined from the passages that were actually read -- proper-noun runs the evidence
    prints and the question does not -- because in a two-article chain the second article's title is
    almost always spelled out inside the first article's abstract;
  * each link name is searched CROSSED with the property the question is missing, never bare, so the
    index returns the entity's own article rather than the crowd of articles that merely repeat its name;
  * every document the first window already showed is EXCLUDED, so the second window is guaranteed to
    contain articles the model has not seen. A rerank that merely re-scores the same pool cannot do this.

The trigger is widened by one deterministic condition beyond ungroundedness: a draft whose content words
are all words the question itself already uses is a draft that names nothing new -- the model stopped at
the entity it was told about instead of traversing to the one it was asked for. Questions that offer their
own answer as one of two named alternatives are exempt, since there the question's own words are the
answer.

The second-pass answerer is constrained, not free: it is told the rejected draft, told the answer must be
printed in the new passages, and its output is accepted only when the new window really prints it. A draft
that survives is kept unchanged, so items the first pass got right are untouched.

Everything else -- the two-sided overlap route, the decomposition, the title-anchored rerank, the hop --
is the parent's, unchanged.

Calls: 3 LLM and 5-6 retrievals normally; 4 LLM and up to 10 retrievals on a rejected draft.
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

# The two-sided shape: the question names two things and asks for what they SHARE.
_SHARE_STRONG = re.compile(
    r"\bin\s+common\b|\bboth\b|\bshare[sd]?\b|\bhow\s+(?:is|are|was|were)\b[^?]*\brelated\b|"
    r"\b(?:are|were|is|was)\s+(?:both\s+)?which\s+(?:type|kind|sort|genre)\b",
    re.IGNORECASE,
)
_SHARE_KIND = re.compile(
    r"\bwhat\s+(?:kind|type|sort|genre|nationality|profession|occupation|category)\b|"
    r"\bwhich\s+(?:type|kind|sort|genre|nationality|profession|occupation|category|country|language)\b",
    re.IGNORECASE,
)
# "X and Y" where both sides open with a capital: the coordination of two named things.
_COORD = re.compile(r"[A-Z][\w'\u2019.&-]*(?:\s+[\w'\u2019.&-]+){0,5}\s+(?:and|or)\s+[A-Z][\w'\u2019.&-]*")
_QUOTED = re.compile(r"[\"\u201c\u2018']([^\"\u201c\u201d\u2018\u2019']{2,60})[\"\u201d\u2019']")
# A question that prints its own answer as one of two named alternatives.
_OPTION = re.compile(r"\bwhich\s+of\b|,\s*(?:or|and)\s+[A-Z]|\b[a-z]{3,}\s+or\s+[A-Z]", re.IGNORECASE)
_DISAMB = re.compile(r"\s*\([^)]*\)")

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

    # the fresh-evidence second pass
    MAX_LINK_NAMES = 3      # link titles mined out of the passages that were read
    MAX_SECOND_QUERIES = 4  # searches spent on the second window
    SECOND_DOCS = 8         # how many UNSEEN abstracts the second window shows

    # two-sided route
    SIDE_DOCS = 3           # abstracts kept per side; the two blocks are never merged
    SIDE_MIN = 0.8          # a side's abstract must really be TITLED that side, not merely share a name
    NGRAM_MAX = 5           # longest overlap phrase mined
    SLATE = 10              # overlap phrases shown to the answerer
    ECHO_PENALTY = 0.8      # a phrase the question already uses is still eligible, just weaker
    LEN_BONUS = 0.0         # rank by distinctiveness alone: the shared category is usually one word

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

    SECOND_INSTR = (
        "`evidence` is a fresh set of passages: none of them was available when an earlier attempt "
        "produced `rejected_draft`. The draft was rejected because nothing supported it, so it is a hint "
        "about the chain, not the answer.\n"
        "`answer` must be the bare answer, copied from these new passages exactly as one of them spells "
        "it: a name, title, date or quantity, or the single word yes or no when the question asks whether "
        "something holds. Follow the chain one step further than the draft did -- if the draft names the "
        "thing the question already talks about, the answer is the thing that thing leads to. Never write "
        "a sentence, never restate the question, never explain, and give the most likely short answer "
        "rather than saying it cannot be determined."
    )

    SPLIT_INSTR = (
        "The question names two different things and asks for something they have in common.\n"
        "`side_a` and `side_b`: the two things, one per field, each written the way its own "
        "encyclopedia article would be titled. Copy the wording from the question; never merge them, "
        "never write a description of the relationship, never write more than one name in a field.\n"
        "`shared_kind`: two to five words naming the kind of thing they are asked to share -- the "
        "vocabulary an article would use, not a question."
    )

    OVERLAP_INSTR = (
        "`evidence_a` and `evidence_b` are abstracts about two different things. The question asks for "
        "something the two SHARE, so the answer is a phrase that BOTH sides state -- not a fact about "
        "one of them, and never a sentence covering them separately.\n"
        "`overlap` lists phrases that literally occur on both sides, most distinctive first; they are "
        "candidates, not a menu -- ignore them if the evidence supports a better shared phrase.\n"
        "`answer` must be the bare shared phrase and nothing else, spelled as the evidence spells it: "
        "a common noun or adjective when the question asks what kind of thing they both are, a name "
        "when it asks who or what appears in both. Prefer the SHORTEST shared phrase that answers the "
        "question -- one or two words -- over a longer phrase that contains it, and use the singular, "
        "evidence-printed wording rather than a plural or a paraphrase of your own. Never write a "
        "location or a founding fact when the question asks what kind of thing something is. "
        "Never restate the question, never explain, and give "
        "the most likely short answer rather than saying there is none."
    )

    def __init__(self) -> None:
        self.split = Module(
            inputs=["question"],
            outputs=["side_a", "side_b", "shared_kind"],
            instructions=self.SPLIT_INSTR,
        )
        self.overlap_answer = Module(
            inputs=["question", "evidence_a", "evidence_b", "overlap"],
            outputs=["answer"],
            instructions=self.OVERLAP_INSTR,
        )
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
        self.second = Module(
            inputs=["question", "rejected_draft", "evidence"],
            outputs=["answer"],
            instructions=self.SECOND_INSTR,
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

    # -- the two-sided route ---------------------------------------------------------------
    @staticmethod
    def _is_yes_no(question: str) -> bool:
        return bool(re.match(r"\s*(?:are|is|was|were|do|does|did|can|could|has|have|had)\b", question, re.I))

    def _two_sided(self, question: str) -> bool:
        """Fires on questions that name two things and ask for their overlap (not a yes/no check)."""
        if self._is_yes_no(question):
            return False
        if _SHARE_STRONG.search(question):
            return True
        return bool(_SHARE_KIND.search(question) and _COORD.search(question))

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+|\s*\|\s*", text or "") if s.strip()]

    def _phrase_map(self, texts: list[str]) -> dict[str, str]:
        """Every 1..NGRAM_MAX word phrase inside a sentence -> the surface form it was printed as."""
        out: dict[str, str] = {}
        for text in texts:
            for sent in self._sentences(text):
                surf = re.findall(r"[A-Za-z0-9'\u2019&.-]+", sent)
                low = [w.lower().strip(".,;:'\u2019-&") for w in surf]
                clean = [w.strip(".,;:'\u2019") for w in surf]
                for n in range(1, self.NGRAM_MAX + 1):
                    for i in range(len(surf) - n + 1):
                        keyw = low[i : i + n]
                        if not all(keyw):
                            continue
                        key = " ".join(keyw)
                        if key not in out:
                            out[key] = " ".join(clean[i : i + n]).strip()
        return out

    def _mine_overlap(self, a_texts: list[str], b_texts: list[str], question: str) -> list[str]:
        """Phrases printed on BOTH sides, ranked by how much they discriminate.

        This is the whole point of keeping the sides apart: a merged pool cannot tell a phrase that
        two articles independently use from a phrase one article happens to repeat.
        """
        a, b = self._phrase_map(a_texts), self._phrase_map(b_texts)
        shared = set(a) & set(b)
        if not shared:
            return []
        # idf over the sentences of both sides, so a word every abstract uses scores near zero
        units = [s for t in a_texts + b_texts for s in self._sentences(t)]
        n_units = max(len(units), 1)
        df: dict[str, int] = {}
        for u in units:
            for w in set(self._toks(u)):
                df[w] = df.get(w, 0) + 1
        qwords = self._content(question)

        scored: list[tuple[float, str]] = []
        for key in shared:
            toks = key.split()
            content = [w for w in toks if w not in _STOP and len(w) > 1]
            if not content or _STOP.issuperset({toks[0], toks[-1]}) or toks[0] in _STOP or toks[-1] in _STOP:
                continue
            mass = sum(math.log(1.0 + n_units / (1.0 + df.get(w, 0))) for w in content)
            score = mass * (1.0 + self.LEN_BONUS * (len(toks) - 1))
            if set(content) <= qwords:
                score *= self.ECHO_PENALTY      # the question's own words can still BE the overlap
            scored.append((score, a[key]))
        scored.sort(key=lambda sp: -sp[0])

        out: list[str] = []
        seen: set[str] = set()
        for _, surf in scored:
            k = self._norm_phrase(surf)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(surf)
            if len(out) >= self.SLATE:
                break
        return out

    def _side_docs(self, lists: list[list[dict[str, Any]]], side: str, taken: set[str]) -> list[dict[str, Any]]:
        """The abstracts that are ABOUT this side, chosen by how well their title names it."""
        docs: dict[str, dict[str, Any]] = {}
        for hits in lists:
            for h in hits:
                key = (h.get("title") or "").strip().lower()
                if key and key not in taken:
                    docs.setdefault(key, h)
        scored = sorted(
            ((self._title_anchor(h.get("title", ""), [side]), i, h) for i, h in enumerate(docs.values())),
            key=lambda t: (-t[0], t[1]),
        )
        # Anything that merely shares a name with the side ("Sharon Miller" for "Sharon Kay Penman")
        # is not about it, and a phrase it happens to share with the other side is not an overlap.
        return [h for sc, _, h in scored[: self.SIDE_DOCS] if sc >= self.SIDE_MIN]

    def _printed_on(self, answer: str, texts: list[str]) -> bool:
        norm = self._norm_answer(answer)
        return bool(norm) and any(norm in self._norm_answer(t) for t in texts)

    def _overlap_route(self, question: str, ctx: RunContext) -> dict[str, Any]:
        parts = self.split(ctx.llm, tag="split", question=question)
        side_a = " ".join((parts.get("side_a", "") or "").split())[:70].strip('"\u201c\u201d ')
        side_b = " ".join((parts.get("side_b", "") or "").split())[:70].strip('"\u201c\u201d ')
        kind = " ".join((parts.get("shared_kind", "") or "").split())[:50]
        if not side_a or not side_b or self._norm_phrase(side_a) == self._norm_phrase(side_b):
            return {}

        a_lists = [ctx.tools.retrieve_docs(side_a, k=self.K)]
        b_lists = [ctx.tools.retrieve_docs(side_b, k=self.K)]
        q_list = ctx.tools.retrieve_docs(question, k=self.K)

        a_docs = self._side_docs(a_lists + [q_list], side_a, taken=set())
        taken = {(h.get("title") or "").strip().lower() for h in a_docs}
        b_docs = self._side_docs(b_lists + [q_list], side_b, taken=taken)

        # A side whose own article never surfaced is re-asked once, crossed with the shared kind.
        for label, side, docs, bucket in (("a", side_a, a_docs, a_lists), ("b", side_b, b_docs, b_lists)):
            if docs or not kind:
                continue
            bucket.append(ctx.tools.retrieve_docs(f"{side} {kind}", k=self.K))
        if not a_docs:
            a_docs = self._side_docs(a_lists + [q_list], side_a, taken=set())
        if not b_docs:
            taken = {(h.get("title") or "").strip().lower() for h in a_docs}
            b_docs = self._side_docs(b_lists + [q_list], side_b, taken=taken)
        if not a_docs or not b_docs:
            return {}

        a_texts = [h["text"] for h in a_docs]
        b_texts = [h["text"] for h in b_docs]
        slate = self._mine_overlap(a_texts, b_texts, question)
        ctx.note(
            f"two-sided route: {side_a!r} vs {side_b!r} ({kind!r}) -> {len(slate)} overlap phrases",
            side_a_titles=[h["title"] for h in a_docs],
            side_b_titles=[h["title"] for h in b_docs],
            overlap=slate,
        )

        out = self.overlap_answer(
            ctx.llm,
            tag="overlap_answer",
            question=question,
            evidence_a=[f"{i + 1}. {t}" for i, t in enumerate(a_texts)],
            evidence_b=[f"{i + 1}. {t}" for i, t in enumerate(b_texts)],
            overlap=slate or ["(no phrase occurs on both sides)"],
        )
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = slate[0] if slate else ""
        if not answer:
            return {}

        # Snap only when the draft is NOT shared: an answer both sides print is already the overlap.
        snapped = ""
        if self._norm_answer(answer) not in {"yes", "no"} and not (
            self._printed_on(answer, a_texts) and self._printed_on(answer, b_texts)
        ):
            draft = self._norm_answer(answer)
            for phrase in slate:
                pk = self._norm_answer(phrase)
                if pk and pk != draft and pk in draft:
                    snapped, answer = phrase, phrase
                    break
        ctx.note(f"overlap answer={answer!r} snapped={snapped!r}")
        return {
            "answer": answer,
            "route": "overlap",
            "side_a": side_a,
            "side_b": side_b,
            "shared_kind": kind,
            "overlap": slate,
            "snapped": snapped,
        }

    # -- the fresh-evidence second pass ----------------------------------------------------
    def _echoes_question(self, answer: str, question: str) -> bool:
        """True when the draft introduces no word the question did not already use.

        Such a draft is the entity the question handed the model, not the one it asked for: the chain
        was never traversed. Questions that list their own answer as one of two alternatives are exempt,
        and so are yes/no and numeric answers, where reusing the question's vocabulary is correct.
        """
        norm = self._norm_answer(answer)
        if not norm or norm in {"yes", "no"} or _NUM.search(norm):
            return False
        if _OPTION.search(question):
            return False
        bare = _DISAMB.sub("", answer)
        words = self._content(bare)
        return bool(words) and words <= self._content(question)

    def _second_queries(self, names: list[str], prop: str, question: str, already: list[str]) -> list[str]:
        """One search per mined link name, crossed with the missing property.

        A bare name is a bad query against an abstract index -- it returns everything that repeats the
        name -- so every query carries the property or the question's own distinguishing words.
        """
        seen = [frozenset(self._content(q)) for q in already]
        tails = [t for t in (prop.strip(), self._content_words(question)) if t]
        out: list[str] = []
        for name in names:
            own = self._content(name)
            for tail in tails or [""]:
                words = [w for w in tail.split() if w.lower().strip(".,'\u2019-") not in own]
                q = f"{name} {' '.join(words)}".strip()
                k = frozenset(self._content(q))
                if not k or any(k <= s for s in seen):
                    continue
                seen.append(k)
                out.append(q)
                break
            if len(out) >= self.MAX_SECOND_QUERIES:
                break
        return out

    def _unseen(
        self,
        ranked: list[tuple[float, dict[str, Any]]],
        seen_titles: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """The best-ranked articles the first window did NOT show."""
        out: list[dict[str, Any]] = []
        for _, h in ranked:
            key = (h.get("title") or "").strip().lower()
            if key in seen_titles:
                continue
            out.append(h)
            if len(out) >= limit:
                break
        return out

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        if self._two_sided(item["question"]):
            out = self._overlap_route(item["question"], ctx)
            if out:
                return out
            ctx.note("two-sided route abstained -> bridge pipeline")
        return self._bridge_run(item, ctx)

    def _bridge_run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
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

        # Two deterministic rejections, one remedy. A draft no read passage prints is a draft this
        # window cannot justify; a draft made only of the question's own words is a draft that never
        # left the entity the question started from. Either way the window, not the wording, is wrong.
        ungrounded = not self._grounded(answer, evidence_docs, question)
        echo = self._echoes_question(answer, question)
        repaired = False
        second_qs: list[str] = []
        link_names: list[str] = []
        if ungrounded or echo:
            read_titles = {(h.get("title") or "").strip().lower() for h in evidence_docs}
            # The next article's title is normally spelled out inside the abstracts already read.
            link_names = self._harvest([h["text"] for h in evidence_docs], question, self.MAX_LINK_NAMES)
            second_qs = self._second_queries(link_names, prop, question, queries + hop)
            if len(second_qs) < self.MAX_SECOND_QUERIES:
                gap = self._uncovered(question, evidence_docs)
                extra = f"{answer} {prop}".strip() if prop else answer
                for q in (gap, extra):
                    k = frozenset(self._content(q))
                    if k and all(not (k <= frozenset(self._content(s))) for s in queries + hop + second_qs):
                        second_qs.append(q)
                    if len(second_qs) >= self.MAX_SECOND_QUERIES:
                        break
            fresh_lists = [ctx.tools.retrieve_docs(q, k=self.K) for q in second_qs[: self.MAX_SECOND_QUERIES]]
            fresh_docs: list[dict[str, Any]] = []
            if fresh_lists:
                fresh_ranked = self._rerank(
                    fresh_lists + lists, anchors + pivots + link_names, question, picked
                )
                # The whole point: the second window is disjoint from the first.
                fresh_docs = self._unseen(fresh_ranked, read_titles, self.SECOND_DOCS)
            if fresh_docs:
                fresh_ev = [f"{i + 1}. {h['text']}" for i, h in enumerate(fresh_docs)]
                fixed = self.second(
                    ctx.llm,
                    tag="second_pass",
                    question=question,
                    rejected_draft=answer,
                    evidence=fresh_ev,
                )
                cand = self._clean(fixed.get("answer", ""))
                # Accept only what the NEW passages really print, and only if it moved.
                if (
                    cand
                    and not _NONANSWER.match(cand)
                    and self._norm_answer(cand) != self._norm_answer(answer)
                    and self._grounded(cand, fresh_docs, question)
                    and not self._echoes_question(cand, question)
                ):
                    answer = cand
                    repaired = True
            ctx.note(
                f"rejected (ungrounded={ungrounded}, echo={echo}) -> {len(second_qs)} fresh searches, "
                f"{len(fresh_docs)} unseen articles, repaired={repaired}",
                link_names=link_names,
                second_queries=second_qs,
                fresh=[h["title"] for h in fresh_docs],
                final=answer,
            )
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "second_queries": second_qs,
            "link_names": link_names,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
