"""Answer surface form chosen from a lattice of literal alternatives, arbitrated by the supporting sentence.

Mechanism change vs the parent (`grounded_answer_repair`). The parent decides WHICH entity answers the
question and then commits to whatever spelling its answering call happened to emit. Reading its failures
shows that a large share of them are not knowledge failures at all: the entity is right and the string is
wrong, because the passages print the same referent in several different forms and the model picks the
one it finds most familiar rather than the one the question asks for. It shortens a person's full name to
the form used as an article title, or drops the region, county or type word that the sentence attaches to
a place, or keeps a title's disambiguating tail that no sentence uses.

Checking the train split shows there is no fixed direction to correct in -- for some questions the wanted
form is the article title and for others it is the longer name the article's own opening prints, at close
to even odds -- so no deterministic rewrite can win. What decides it is WHICH SENTENCE establishes the
fact that makes this entity the answer: the form to give is the form that sentence prints. An entity
qualified by its own opening description should be named the way that opening names it; an entity merely
mentioned inside another article's sentence should be named the way that mention spells it.

So this system adds a FORM LATTICE stage after the answer is fixed. The lattice is built deterministically
and contains only strings the pool literally prints: every occurrence of the draft in an evidence sentence,
that occurrence extended left and right over the proper-noun phrase that contains it, the occurrence
extended over a following ", <Region>" apposition, and the opening name of any pooled article that the
draft names. Each node carries the sentence it was read out of. Only when the lattice holds more than one
distinct form -- i.e. only when there is a real choice to make -- is one extra call spent asking which
sentence establishes the asked-for fact, and the answer becomes that sentence's spelling. The choice is
restricted to enumerated nodes, so this stage can never invent a string and never changes which entity
is answered.

Calls: 3 LLM and 5-6 retrievals normally, +1 LLM when the lattice branches, +1 LLM and up to 3 retrievals
when the parent's grounding test fires.
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
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_TOKEN = re.compile(r"[A-Za-z0-9][\w'\u2019.&-]*|[,]")
_CAPTOK = re.compile(r"^(?:[A-Z][\w'\u2019.&-]*|\d{1,4}[\w-]*|[\"'\u201c\u2018][A-Z][\w'\u2019.-]*[\"'\u201d\u2019]?)$")
# Deliberately empty of prepositions: extending a name across `of`/`in` attaches a prepositional
# phrase ("<singer> of <band>", "<bank> in <month>") that renames or relocates the referent instead
# of respelling it, which is the one thing this stage must never do.
_LOWER_LINK: set[str] = set()
_LEAD_CUT = re.compile(
    r"\s+(?:is|was|are|were|has|have|had|became|serves|served|refers|remains)\b|\s*\(|\s*[;:]\s*|,\s+(?:a|an|the|born|who|which|is|was)\b",
    re.IGNORECASE,
)

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

    FORM_INSTR = (
        "`options` lists ways the passages spell one and the same answer, each followed by the sentence "
        "it was read out of. The entity is already settled; only its wording is open.\n"
        "Write `choice`: the number of the option whose SENTENCE is the one that establishes the fact the "
        "question is asking for, spelled exactly as that sentence spells it.\n"
        "- If what qualifies this entity as the answer is stated by its own opening description, take the "
        "name that opening prints, in full, even when a shorter name is more familiar.\n"
        "- If the entity is instead named inside a sentence about something else, take the wording of that "
        "mention.\n"
        "- Keep the region, county, jurisdiction or type word that the chosen sentence attaches to the "
        "answer, and drop a tail that is only there to tell two articles apart.\n"
        "Answer with one number and nothing else."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "pivot_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)
        self.form = Module(
            inputs=["question", "options"],
            outputs=["choice"],
            instructions=self.FORM_INSTR,
        )
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

    # -- the form lattice ------------------------------------------------------------------
    MAX_FORMS = 5
    MAX_FORM_WORDS = 9
    EXTEND = 4

    @staticmethod
    def _sentences(doc: dict[str, Any]) -> list[str]:
        text = doc.get("text", "") or ""
        body = text.split("|", 1)[1] if "|" in text else text
        return [p.strip() for p in _SENT.split(body.strip()) if len(p.strip()) > 12]

    def _lead_name(self, doc: dict[str, Any]) -> str:
        """The name an article's own opening prints before it starts describing the subject."""
        sents = self._sentences(doc)
        if not sents:
            return ""
        head = _LEAD_CUT.split(sents[0])[0].strip().strip(",;: ")
        if not head or len(head.split()) > self.MAX_FORM_WORDS:
            return ""
        return head

    def _occurrences(self, sentence: str, target: str) -> list[tuple[list[str], int, int]]:
        """Token spans of `sentence` whose normalized text equals the normalized target."""
        toks = _TOKEN.findall(sentence)
        goal = self._norm_answer(target).split()
        if not goal or not toks:
            return []
        out: list[tuple[list[str], int, int]] = []
        for start in range(len(toks)):
            for end in range(start + 1, min(len(toks), start + len(goal) + 3) + 1):
                if self._norm_answer(" ".join(toks[start:end])).split() == goal:
                    out.append((toks, start, end))
                    break
        return out

    @staticmethod
    def _join(toks: list[str], start: int, end: int) -> str:
        out = ""
        for t in toks[start:end]:
            out = out + t if t == "," or not out else out + " " + t
        return out.strip(" ,;:.")

    def _extended_forms(self, toks: list[str], start: int, end: int) -> list[str]:
        """The proper-noun phrase around an occurrence, and the occurrence plus its apposition."""
        forms: list[str] = []

        left = start
        while left > 0 and start - left < self.EXTEND:
            prev = toks[left - 1]
            if _CAPTOK.match(prev) or (prev.lower() in _LOWER_LINK and left - 1 > 0 and _CAPTOK.match(toks[left - 2])):
                left -= 1
            else:
                break

        right = end
        while right < len(toks) and right - end < self.EXTEND:
            nxt = toks[right]
            if _CAPTOK.match(nxt) or (nxt.lower() in _LOWER_LINK and right + 1 < len(toks) and _CAPTOK.match(toks[right + 1])):
                right += 1
            else:
                break

        # ", <Region>" -- the container a sentence attaches to a place or an institution
        appos = end
        if end < len(toks) and toks[end] == "," and end + 1 < len(toks) and _CAPTOK.match(toks[end + 1]):
            appos = end + 1
            while appos < len(toks) and appos - end < self.EXTEND and _CAPTOK.match(toks[appos]):
                appos += 1

        for a, b in ((left, end), (start, right), (left, right), (start, appos)):
            if (a, b) == (start, end) or b <= a:
                continue
            if b - a > self.MAX_FORM_WORDS:
                continue
            forms.append(self._join(toks, a, b))
        return forms

    def _form_lattice(self, draft: str, docs: list[dict[str, Any]], question: str) -> list[tuple[str, str]]:
        """Literal spellings of the drafted answer that the pool prints, each with its sentence.

        Never invents text: every node is copied out of a passage, so arbitrating between nodes can
        only change how the answer is written, never what it refers to.
        """
        norm = self._norm_answer(draft)
        if not norm or norm in {"yes", "no"} or len(norm.split()) > self.MAX_FORM_WORDS:
            return []
        qnorm = self._norm_answer(question)
        out: list[tuple[str, str]] = [(draft, "")]
        seen = {norm}

        def add(form: str, sentence: str) -> None:
            form = " ".join((form or "").split()).strip(" ,;:.")
            key = self._norm_answer(form)
            if not key or key in seen or len(key.split()) > self.MAX_FORM_WORDS:
                return
            # A form that the question already spells out is a restatement, not an answer.
            if f" {key} " in f" {qnorm} ":
                return
            seen.add(key)
            out.append((form, sentence))

        for h in docs:
            for sent in self._sentences(h):
                for toks, start, end in self._occurrences(sent, draft):
                    literal = self._join(toks, start, end)
                    add(literal, sent)
                    for form in self._extended_forms(toks, start, end):
                        add(form, sent)
                if len(out) > 3 * self.MAX_FORMS:
                    break

        # The opening name of an article the draft names -- the fuller form of a person or work
        # whose title is the short one, or the short form when the title carries a disambiguator.
        for h in docs:
            bare = self._norm_answer(_PAREN.sub("", h.get("title", "")))
            if not bare:
                continue
            btoks, ntoks = set(bare.split()), set(norm.split())
            if not (btoks <= ntoks or ntoks <= btoks):
                continue
            lead = self._lead_name(h)
            # This branch exists to recover a FULLER name; a shorter lead is a fragment of the draft.
            lead_toks = self._norm_answer(lead).split()
            if lead and lead_toks[-1:] == norm.split()[-1:] and len(lead_toks) >= len(norm.split()):
                sents = self._sentences(h)
                add(lead, sents[0] if sents else "")
            plain = _PAREN.sub("", h.get("title", "")).strip()
            if len(self._norm_answer(plain).split()) > 1 or bare == norm:
                add(plain, "")

        return out[: self.MAX_FORMS]

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
        # The form lattice. The entity is settled; the remaining question is which of the spellings
        # the pool prints is the one the asked-for fact is stated in.
        forms = self._form_lattice(answer, evidence_docs, question)
        chosen = ""
        if len(forms) >= 2:
            options = []
            for n, (form, sent) in enumerate(forms):
                line = f"{n + 1}. {form}"
                if sent:
                    line += f"\n   printed in: {sent[:300]}"
                options.append(line)
            verdict = self.form(ctx.llm, tag="form", question=question, options=options)
            picked_ids = _NUM.findall(verdict.get("choice", "") or "")
            if picked_ids:
                idx = int(picked_ids[0]) - 1
                if 0 <= idx < len(forms) and forms[idx][0].strip():
                    chosen = self._clean(forms[idx][0])
                    if chosen:
                        answer = chosen
            ctx.note(
                f"form lattice: {len(forms)} spellings -> {answer!r}",
                forms=[f for f, _ in forms],
                chosen=chosen,
            )

        return {
            "answer": answer,
            "forms": [f for f, _ in forms],
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
