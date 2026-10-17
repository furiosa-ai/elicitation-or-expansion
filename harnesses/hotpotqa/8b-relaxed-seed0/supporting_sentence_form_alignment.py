"""Spell the answer the way the sentence that supports it spells it.

Mechanism change vs the parent (`grounded_answer_repair`). The parent tests whether the drafted answer is
PRINTED anywhere in the evidence and, if it is not, spends retrieval to find it. That test is blind to a
different and surprisingly large failure: the draft is printed, so the test passes, but it is printed in a
DIFFERENT SURFACE FORM from the one the question is really asking for. A short answer is graded by string
equality after light normalisation, so `Molly Shannon` and `Molly Helen Shannon`, `Lewis and Clark` and
`Lewis and Clark County`, `six` and `six children` are different answers even though the model understood
the question perfectly. The parent has no stage that can even represent this choice: the answering call
emits one string and nothing downstream may vary its extent.

This system adds a FORM ALIGNMENT stage after the answer is settled. It is built on the observation that a
short answer is a span of some sentence, so its correct extent is a property of that sentence, not of the
model's habits of naming things:

  1. locate the supporting sentences -- the sentences of the read passages that print the draft and that
     share the most (idf-weighted) content with the question once the draft's own words are discounted;
  2. enumerate, deterministically, every rival extent of the draft that those sentences actually print:
     the draft itself, the maximal name run the draft sits inside (so a formal name given in full is
     recoverable), the completion of the draft with the noun the question asks for (`... county`,
     `... children`), the comma qualifier a sentence attaches to a place, and the single conjuncts of a
     drafted `A and B`;
  3. if the sentences print only one extent, nothing happens -- no call, no change;
  4. otherwise ONE arbitration call sees only the supporting sentences and the numbered rival extents and
     returns an index; anything that is not one of the enumerated extents is discarded and the draft kept.

The stage therefore cannot invent a string: it can only choose among extents the evidence prints, and it is
silent on every item where the evidence offers no alternative.

Calls: 3 LLM and 5-6 retrievals normally, +1 LLM when the grounding test fires, +1 LLM when the evidence
offers rival extents of the draft.
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
_CAPTOK = re.compile(r"^(?:[A-Z0-9][\w'’.&-]*|[\"“'][A-Z][\w'’ .-]*[\"”'])$")
_WHHEAD = re.compile(
    r"\b(?:what|which|how\s+many|how\s+much)\s+((?:[a-z][a-z-]*\s+){0,3}[a-z][a-z-]*)\b",
    re.IGNORECASE,
)
_TAILHEAD = re.compile(r"\b(?:what|which)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s*\??$", re.IGNORECASE)

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

    ALIGN_INSTR = (
        "`sentences` are the sentences of the read articles that mention the answer to the question. "
        "`candidates` are the different extents of that answer which those sentences literally print.\n"
        "Exactly one candidate is the span the question asks for, written the way the sentence that "
        "establishes the fact writes it. Decide first which sentence establishes the fact, then read off "
        "how that sentence spells the thing asked about.\n"
        "The sentence that establishes the fact is the one carrying the question's own descriptive wording; "
        "a sentence that merely mentions the answer in passing is not it.\n"
        "- Copy that sentence's spelling exactly: neither shorten a name it writes out nor lengthen one it "
        "keeps short. Length is not evidence -- the sentence is.\n"
        "- When the question asks for a kind of thing (a county, an industry, a number of something), keep "
        "the word for that kind if the sentence attaches it to the span, and drop trailing words the "
        "question did not ask about.\n"
        "Write `choice`: the number of that candidate, a bare digit, nothing else."
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
        self.align = Module(
            inputs=["question", "sentences", "candidates"],
            outputs=["choice"],
            instructions=self.ALIGN_INSTR,
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


    # -- form alignment: the answer's extent belongs to its supporting sentence -------------
    @staticmethod
    def _sentences(doc: dict) -> list[tuple[str, str]]:
        """(title, sentence) pairs. Passages arrive as "Title | abstract"."""
        text = doc.get("text", "") or ""
        title = (doc.get("title") or "").strip()
        body = text.split("|", 1)[1] if "|" in text else text
        out = []
        for sent in _SENT.split(body.strip()):
            sent = " ".join(sent.split())
            if 10 <= len(sent) <= 600:
                out.append((title, sent))
        return out

    @staticmethod
    def _loose_pattern(phrase: str) -> "re.Pattern[str] | None":
        """Match a phrase in raw text ignoring punctuation and spacing differences."""
        toks = _WORD.findall(phrase or "")
        if not toks:
            return None
        body = r"[^A-Za-z0-9]{0,3}".join(re.escape(t) for t in toks)
        try:
            return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)
        except re.error:
            return None

    def _wh_head(self, question: str) -> str:
        """The noun the question asks for: `which Montana county` -> county, `what 1975 thriller` -> thriller."""
        m = re.search(r"\b(?:what|which|how\s+many|how\s+much)\b", question, re.IGNORECASE)
        if not m:
            return ""
        head = ""
        for tok in question[m.end():].split()[:4]:
            bare = tok.strip(".,;:?!\"'()").lower()
            if not bare or bare in _STOP or bare.endswith("ing") or bare.endswith("ed"):
                break
            if tok[:1].islower() and bare.isalpha():
                head = bare
            if tok.endswith((".", "?", ",", ";")):
                break
        return head

    @staticmethod
    def _lead_run(sent: str) -> str:
        """The proper-noun run an article opens with -- how the subject names itself in full."""
        toks = sent.split()
        run: list[str] = []
        for tok in toks[:7]:
            bare = tok.strip(".,;:()!?\"'")
            if not bare:
                break
            if _CAPTOK.match(bare) or (run and bare.lower() in {"of", "de", "van", "von", "der", "da", "bin"}):
                run.append(bare)
                if tok.endswith((",", "(", ".", ";")):
                    break
            else:
                break
        return " ".join(run)

    def _support_sentences(self, answer: str, docs: list[dict], question: str, limit: int = 3):
        """Sentences that print the draft, ranked by how much of the QUESTION they explain.

        The draft's own words are discounted: a sentence earns its place by carrying the constraints the
        question imposes, not by repeating the answer. The lead sentence of an article the draft NAMES is
        always eligible, since that is where an entity spells its own name in full.
        """
        pat = self._loose_pattern(answer)
        if pat is None:
            return []
        akey = self._norm_answer(answer)
        atoks = self._content(answer)
        qk = self._content(question) - atoks
        scored = []
        for h in docs:
            title = _PAREN.sub("", (h.get("title") or "")).strip()
            tkey = self._norm_answer(title)
            self_named = bool(tkey) and (tkey == akey or (atoks and self._content(title) & atoks))
            for order, (_t, sent) in enumerate(self._sentences(h)):
                if not pat.search(sent) and not (self_named and order == 0):
                    continue
                overlap = len(self._content(sent) & qk)
                bonus = 1.0 if (self_named and order == 0) else 0.0
                scored.append((overlap + bonus - 0.15 * order, title, sent))
        scored.sort(key=lambda x: -x[0])
        return [(t, s) for _, t, s in scored[:limit]]

    def _extents(self, answer: str, sentences: list[tuple[str, str]], question: str) -> list[str]:
        """Every extent of the draft that the supporting sentences literally print."""
        pat = self._loose_pattern(answer)
        if pat is None:
            return []
        head = self._wh_head(question)
        atoks = self._content(answer)
        out: list[str] = []

        def offer(text: str) -> None:
            text = " ".join((text or "").split()).strip(" ,;:.")
            if not text or len(text.split()) > self.MAX_ANSWER_WORDS:
                return
            key = self._norm_answer(text)
            if not key or any(self._norm_answer(o) == key for o in out):
                return
            out.append(text)

        offer(answer)
        qk = self._content(question) - atoks
        for _title, sent in sentences:
            # (a) The name the sentence's subject gives itself, when that subject IS the draft -- but
            #     only when that opening sentence is the one the question is quoting. An entity's own
            #     article always spells its name in full, so offering that form wherever it exists would
            #     rewrite every answer; it is a live option only when the words that qualify the entity
            #     as the answer are the words that opening sentence uses.
            lead = self._lead_run(sent)
            if (
                lead
                and atoks
                and self._content(lead) & atoks
                and len(lead.split()) <= 6
                and len(self._content(sent) & qk) >= 2
            ):
                offer(lead)

            m = pat.search(sent)
            if not m:
                continue
            offer(m.group(0))

            # (b) the maximal name run the draft sits inside, and the comma qualifier a sentence
            #     attaches to a place or a work.
            pre, rest = sent[: m.start()], sent[m.end():]
            left: list[str] = []
            while len(left) < 4:
                lm = re.search(r"([A-Z0-9\"“][\w'’.&-]*)\s+$", pre)
                if not lm:
                    break
                bare = lm.group(1).strip(".,;:")
                if not bare or bare.lower() in _STOP or not _CAPTOK.match(bare):
                    break
                left.insert(0, bare)
                pre = pre[: lm.start()]
            right: list[tuple[str, str]] = []
            commas = 0
            while len(right) < 4:
                rm = re.match(r"^\s*(,?)\s*([A-Za-z0-9\"“][\w'’.&-]*)", rest)
                if not rm:
                    break
                bare = rm.group(2).strip(".,;:")
                commas += 1 if rm.group(1) else 0
                if not bare or bare.lower() in _STOP or not _CAPTOK.match(bare) or commas > 1:
                    break
                right.append((rm.group(1), bare))
                ended = rm.group(2).endswith((".", ";", ":"))
                rest = rest[rm.end():]
                if ended:
                    break

            def join_right(n: int) -> str:
                acc = m.group(0)
                for sep, tok in right[:n]:
                    acc += (", " if sep else " ") + tok
                return acc

            if left:
                offer(" ".join(left) + " " + m.group(0))
            if right:
                offer(join_right(len(right)))
                if len(right) > 1:
                    offer(join_right(1))
            if left and right:
                offer(" ".join(left) + " " + join_right(len(right)))

            # (c) completion with the noun the question asks for, when the sentence attaches it.
            #     The noun need not be capitalised -- `six children`, `alternate history novels`.
            if head:
                tail = sent[m.end():].split()[:3]
                for j, tok in enumerate(tail):
                    if tok.strip(".,;:()").lower().rstrip("s") == head.rstrip("s"):
                        offer(m.group(0) + " " + " ".join(t.strip(".,;:") for t in tail[: j + 1]))
                        break

        # (d) a drafted conjunction is two answers; offer each conjunct the sentences print.
        parts = re.split(r"\s+and\s+|\s*&\s*", answer)
        if len(parts) == 2 and all(len(_WORD.findall(p)) >= 2 for p in parts):
            for part in parts:
                sub = self._loose_pattern(part)
                if sub and any(sub.search(sent) for _t, sent in sentences):
                    offer(part)

        return out[:5]

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

        # Form alignment. The answer is a span of some sentence; which extent of that span the question
        # wants is decided by the sentence, so the rival extents are read off the evidence and arbitrated.
        aligned = False
        extents: list[str] = []
        if self._norm_answer(answer) not in {"", "yes", "no"}:
            support = self._support_sentences(answer, evidence_docs, question)
            extents = self._extents(answer, support, question)
            if len(extents) > 1:
                shown = [f"{t} -- {sent}" for t, sent in support]
                numbered = [f"{i + 1}. {e}" for i, e in enumerate(extents)]
                pick = self.align(
                    ctx.llm, tag="align", question=question, sentences=shown, candidates=numbered
                )
                digits = _NUM.findall(pick.get("choice", "") or "")
                if digits:
                    idx = int(digits[0]) - 1
                    if 0 <= idx < len(extents) and self._norm_answer(extents[idx]) != self._norm_answer(answer):
                        answer = extents[idx]
                        aligned = True
                ctx.note(
                    f"form alignment: {len(extents)} extents, changed={aligned}",
                    extents=extents,
                    support=[t for t, _ in support],
                    final=answer,
                )
        return {
            "answer": answer,
            "extents": extents,
            "aligned": aligned,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
