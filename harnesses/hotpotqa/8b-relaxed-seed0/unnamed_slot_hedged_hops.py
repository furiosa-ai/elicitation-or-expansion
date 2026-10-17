"""Search for the thing the question describes but never names, and hedge the hop across rivals.

Mechanism change vs the parent (`deep_pool_title_rerank`). Every query the parent issues is built
out of a NAME: its first round gives each named entity its own query, and its second round crosses
one or two committed pivot names with the property it is missing. Two failure modes follow from
that, and together they account for most of the parent's missing articles (its pool holds every
required article for only 82% of items).

First, the article the question needs is often one it never names -- "the album that song is on",
"the French professional league that club plays in", "the legislation the ruling relied on". No
name for it exists in the question or in the pool, so no name-shaped query can reach it. But such an
article introduces itself with exactly the words of the description: its opening line says that it
IS a French professional league. So this system asks for those descriptions as queries in their own
right, worded as an encyclopedia's opening line, with the asked-for property stripped out -- the
property words are what drag a name query away from the entity's own article and towards the
boilerplate of everything that mentions it.

Second, when the pool does suggest a name, the parent commits: it picks one or two pivots and spends
every remaining query on them, so a single wrong guess -- and the guesses are frequently recalled
from the model's own memory rather than read off a passage -- costs the whole second hop. This
system asks instead for up to three RIVAL names, each required to be printed by some passage, and
retrieves each rival's own article. Being wrong about which rival is right is no longer fatal,
because the rerank and the answerer see all of them; the hop hedges instead of betting.

Calls: 3 LLM, 6-8 retrievals.
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
    MAX_NAMED = 2
    MAX_DESCRIBED = 2
    MAX_FIRST_QUERIES = 5
    MAX_BRIDGES = 3         # rival resolutions carried side by side, never one committed pivot
    MAX_HOP_QUERIES = 3
    SHORTLIST = 14          # how many reranked abstracts the screen reads
    MAX_EVIDENCE = 12       # how many the answerer reads
    RRF_C = 10.0
    W_TITLE = 3.0
    W_IDF = 1.6
    W_PICKED = 0.9
    MAX_ANSWER_WORDS = 12

    DECOMPOSE_INSTR = (
        "A question of this kind can only be settled by combining facts from two different articles, "
        "and one search phrased from the whole question tends to find only one of them. Split the "
        "question into the things it is about, and write searches for those things -- never for the "
        "fact you are being asked for.\n"
        "`named_things`: one line per thing the question NAMES (a person, work, place, organisation, "
        "event). Write the name the way that thing's own article would be titled, and nothing else: "
        "no verbs, no property, no question words.\n"
        "`described_things`: one line per thing the question refers to WITHOUT naming it -- the album "
        "a song is on, the league a club plays in, the law a ruling relied on, the actress who played "
        "a lead role. Write each one as a DEFINITION, the way that thing's own article opens, never as "
        "a reference back to the question: begin with the general noun that classifies it (album, "
        "league, statute, film, actress, university, river) and add only the classifying words the "
        "question supplies -- year, country, language, genre, medium, sport. Do NOT write the words "
        "`the`, `that`, `which` or `who`, do NOT write a verb, and do NOT repeat any name you listed "
        "under `named_things`: that name is not in the article you are looking for. Write `none` if "
        "the question names everything it is about, or if you cannot classify the thing in at least "
        "two words of your own."
    )

    BRIDGE_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant. The question needs "
        "a fact from an article that is probably NOT in the pool yet; your job is to say whose article "
        "to fetch next, using only what the pool already prints.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant "
        "first, at most six numbers separated by commas. Judge relevance by whether the passage is "
        "about a thing the question actually asks about, not by superficial word overlap.\n"
        "`rival_names`: up to three names, one per line, best guess first. These are RIVAL answers to "
        "the same question -- 'whose own article would state the fact the question is missing?' -- and "
        "you are not being asked to choose between them. If two or three passages suggest different "
        "candidates, or one passage names several, list them all; all of them will be looked up, so a "
        "wrong guess in this list costs nothing while a missing one costs the answer. Copy each name "
        "EXACTLY as some passage spells it; never invent a name from memory, never write a description, "
        "never write brackets or a placeholder. If the question names two things and asks how they "
        "compare, write both of those names.\n"
        "`missing_property`: two to five words naming the property the question is asking for, in the "
        "vocabulary an encyclopedia article would use, not as a question. Never write `none`."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["named_things", "described_things"],
            instructions=self.DECOMPOSE_INSTR,
        )
        self.bridge = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "rival_names", "missing_property"],
            instructions=self.BRIDGE_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)

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

    _CLAUSE = {"that", "which", "who", "whom", "whose", "where", "when", "is", "was", "were", "are", "be",
               "been", "has", "have", "had", "on", "in", "of", "by", "with", "from", "for", "to", "at",
               "the", "a", "an", "and", "or", "its", "his", "her", "their", "it"}

    def _describe(self, line: str, named: list[str]) -> str:
        """Reduce a described slot to the classifying words its own article would print.

        A description that quotes the entity the question already names is a paraphrase of the
        question, not a search: the words of the name pull the query straight back to the article
        that is already in the pool. What is left after removing them -- `1976 studio album`,
        `Australian bank`, `French professional football league` -- is what the missing article says
        about itself in its opening line.
        """
        banned = {w for n in named for w in self._toks(n)}
        words = [w for w in _WORD.findall(line or "") if w.lower() not in banned]
        words = [w for w in words if w.lower() not in self._CLAUSE]
        if len(words) < 2:
            return ""
        return " ".join(words[:10])

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

    def _hop_queries(self, rivals: list[str], question: str, already: list[str]) -> list[str]:
        """One query per rival, for the rival's OWN article, with no property words attached.

        The parent crossed its committed pivot with the property it was missing. A property phrase
        is boilerplate: every article that merely mentions the entity repeats it, so the crossed
        query scores those above the entity's own article -- the one place the property is actually
        stated. Here each rival is looked up bare, and the budget buys breadth across rivals
        instead of depth on one guess.
        """
        seen = [frozenset(self._content(q)) for q in already]
        out: list[str] = []
        for name in rivals:
            q = name.strip()
            k = frozenset(self._content(q))
            if not k or any(k <= t for t in seen):
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
        named = self._lines(plan.get("named_things", ""), self.MAX_NAMED)
        described: list[str] = []
        for line in self._lines(plan.get("described_things", ""), self.MAX_DESCRIBED + 1):
            d = self._describe(line, named)
            if d and d not in described:
                described.append(d)
            if len(described) >= self.MAX_DESCRIBED:
                break
        queries = named + described
        if question not in queries:
            queries = (queries[: self.MAX_FIRST_QUERIES - 1] + [question]) if queries else [question]

        lists = [ctx.tools.retrieve_docs(q, k=self.K) for q in queries]
        anchors = [question] + queries
        ranked = self._rerank(lists, anchors, question, picked=set())
        ctx.note(
            f"round1: {len(queries)} queries at k={self.K} -> {len(ranked)} unique articles",
            queries=queries,
            described=described,
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

        # Every rival is carried forward; nothing here chooses between them.
        rivals: list[str] = []
        for line in self._lines(judged.get("rival_names", ""), self.MAX_BRIDGES + 2):
            for part in re.split(r",| and | or |;|/", line):
                name = part.strip().strip("'\"")
                if len(name) > 2 and not _PLACEHOLDER.search(name) and name not in rivals:
                    rivals.append(name)
        rivals = rivals[: self.MAX_BRIDGES]
        if not rivals:
            rivals = self._harvest([h["text"] for h in shortlist], question, self.MAX_BRIDGES)
        pool_text = " ".join(h.get("text", "") for h in shortlist).lower()
        grounded = [n for n in rivals if n.lower() in pool_text or n.lower() in question.lower()]
        if grounded:
            rivals = grounded          # a name no passage prints is a memory, not a lead
        prop = " ".join((judged.get("missing_property", "") or "").split())[:60]
        if prop.lower().startswith("none"):
            prop = ""

        hop = self._hop_queries(rivals, question, queries)
        if len(hop) < self.MAX_HOP_QUERIES and described:
            # Still nothing named for a described slot: ask the description again, now sharpened
            # with the property words, since no article title is available to ask for.
            tail = " ".join(w for w in prop.split() if w.lower() not in _STOP)[:40]
            extra = f"{described[0]} {tail}".strip()
            if frozenset(self._content(extra)) not in [frozenset(self._content(q)) for q in queries + hop]:
                hop.append(extra)
        for q in hop:
            lists.append(ctx.tools.retrieve_docs(q, k=self.K))

        # One rerank over everything both rounds saw, anchored on every rival as well as the question.
        ranked = self._rerank(lists, anchors + rivals, question, picked)
        evidence_docs = [h for _, h in ranked[: self.MAX_EVIDENCE]]
        ctx.note(
            f"hop: {hop} -> {len(ranked)} articles pooled, {len(evidence_docs)} read",
            rivals=rivals,
            prop=prop,
            evidence=[h["title"] for h in evidence_docs],
        )

        evidence = [f"{i + 1}. {h['text']}" for i, h in enumerate(evidence_docs)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = evidence_docs[0]["title"] if evidence_docs else answer
        return {
            "answer": answer,
            "queries": queries,
            "rivals": rivals,
            "missing_property": prop,
            "hop_queries": hop,
            "n_evidence": len(evidence_docs),
        }
