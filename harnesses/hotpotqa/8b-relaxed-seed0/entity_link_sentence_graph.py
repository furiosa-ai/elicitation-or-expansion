"""Evidence chosen by link structure: sentences that connect the question's entities to the answer's.

Mechanism change vs the parent (composed bridge hop). The parent selects evidence with an LLM screening
call that reads whole abstracts, so its pool is capped at what fits in that call -- about two dozen
articles -- and its own traces show the needed article is frequently outside it. Widening the pool has
been tried by feeding the answerer more prose, and it lost points: the extra articles arrive as noise.

The premise here is that in a pool of abstracts the useful text is not a document but a chain of
SENTENCES, and that the chain can be found in code rather than by a screening call, because a two-article
question is a link: some sentence names an entity the question names, that same sentence (or its
article's own name) introduces a second entity the question never names, and a sentence somewhere else
states the wanted property of that second entity. So this system searches much wider, splits every
abstract into sentences, and scores sentences by their position in that link structure -- anchored
(shares a proper name with the question), bridging (anchored and introducing a new proper name), or
target (carries a new proper name from a bridging sentence together with the words the question uses
for the property it wants). The answerer sees a compact chain of the highest-scoring sentences grouped
under the names of their articles, so pool width and reading cost are decoupled and no screening call
is spent at all.

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
_PLACEHOLDER = re.compile(r"[\[\]{}<>]|\b(?:name|entity|person|someone|unknown|his|her|their)\b", re.IGNORECASE)
# Abstracts print sentences run together; split on terminal punctuation that is not an initial or abbrev.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "another",
    "known", "called", "born", "name", "named", "many", "much", "one", "two", "first", "last", "year", "years",
    "would", "will", "can", "could", "into", "about", "between", "during", "along", "such", "some", "any", "not",
    "think", "know", "tell", "say", "said", "featured", "starred", "made", "part", "well", "including", "were",
}


def _covers(tname: str, names) -> bool:
    """True when an article name and a name printed in text denote the same thing: one is a whole-word
    run of the other. `Bush (British band)` and `Bush` match; `Bush` and `Rock Against Bush` do not,
    because the article name must lead."""
    if not tname:
        return False
    tw = tname.split()
    for nm in names:
        nw = nm.split()
        if not nw:
            continue
        if nw == tw:
            return True
        if len(nw) < len(tw) and tw[: len(nw)] == nw:
            return True
        if len(tw) < len(nw) and nw[: len(tw)] == tw:
            return True
    return False


class Harness(BaseHarness):
    K1 = 12
    K2 = 12
    MAX_FIRST_QUERIES = 3
    MAX_BRIDGES = 2
    MAX_HOP_QUERIES = 3
    MAX_SENTENCES = 22        # what the answerer reads
    MAX_ARTICLES = 12         # distinct articles the chain may span
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

    PIVOT_INSTR = (
        "`sentences` are lines lifted from search results; most are irrelevant. Do not answer the question. "
        "Say only what to look up next, using words these lines already print.\n"
        "`pivot_names`: one or two names, one per line. A pivot is the person, work, place or organisation "
        "that stands between the question and the fact it wants -- the one whose own article would state "
        "that fact. Copy each name EXACTLY as some line spells it, or exactly as the question spells it; "
        "never invent a name, never write a description or a placeholder. If the question names two things "
        "and asks how they compare, write both of those names.\n"
        "`missing_property`: two to five words naming the property the question asks for, in the vocabulary "
        "an encyclopedia article would use, not as a question. Never write `none`."
    )

    ANSWER_INSTR = (
        "`evidence` gives, under each article's name, the sentences from it that matter.\n"
        "Answer the question from those sentences.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.pivot = Module(
            inputs=["question", "sentences"],
            outputs=["pivot_names", "missing_property"],
            instructions=self.PIVOT_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)

    # -- text helpers ----------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”').strip()
            if not line or line.lower().strip(".") in {"none", "n/a", "null"}:
                continue
            if len(line) > 300:
                line = line[:300]
            if line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _toks(text: str) -> set[str]:
        return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP}

    @staticmethod
    def _names(text: str) -> set[str]:
        """Proper-noun runs, normalised, that are long enough to identify something."""
        out = set()
        for m in _PROPER.finditer(text or ""):
            run = m.group(0).strip(" .,;:'’-")
            words = [w for w in _WORD.findall(run) if w.lower() not in _STOP]
            if not words:
                continue
            norm = " ".join(w.lower() for w in words)
            if len(norm) >= 4:
                out.add(norm)
        return out

    def _split(self, text: str) -> list[str]:
        parts = [s.strip() for s in _SENT.split(text or "") if s and s.strip()]
        merged: list[str] = []
        for s in parts:
            # An abstract sometimes ends a "sentence" on an initial; glue the crumbs back on.
            if merged and len(_WORD.findall(s)) < 4:
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s[:400])
        return merged[:8]

    def _content_words(self, question: str, limit: int = 4) -> str:
        words = [w for w in _WORD.findall(question) if w.lower() not in _STOP and len(w) > 2]
        lower = [w for w in words if not w[:1].isupper()]
        picked = (lower or words)[-limit:]
        return " ".join(picked)

    def _compose(self, pivots: list[str], prop: str, question: str, already: list[str]) -> list[str]:
        """A bare name is never issued: the index scores whole abstracts, so a name alone is won by the
        articles that merely repeat it. Every hop query is a name crossed with the missing property."""
        seen = [self._toks(q) for q in already]
        base = [t for t in (prop.strip(), self._content_words(question)) if t] or [""]
        out: list[str] = []
        for name in pivots:
            foreign = {w.lower() for other in pivots if other != name for w in _WORD.findall(other)}
            for tail in base:
                words = [w for w in tail.split() if w.strip(".,'’-").lower() not in foreign]
                q = f"{name} {' '.join(words)}".strip()
                k = self._toks(q)
                if not k or any(k <= s for s in seen):
                    continue
                seen.append(k)
                out.append(q)
                break
        return out[: self.MAX_HOP_QUERIES]

    # -- the link graph --------------------------------------------------------------------
    def _chain(self, pool: list[tuple[str, str]], question: str, pivots: list[str]) -> list[tuple[str, str]]:
        """Score every sentence by where it sits on the question -> bridge -> answer link path."""
        qnames = self._names(question) | {" ".join(w.lower() for w in _WORD.findall(p)) for p in pivots}
        qnames = {n for n in qnames if n}
        prop_words = self._toks(question) - {w for n in qnames for w in n.split()}

        df: dict[str, int] = {}
        units: list[dict[str, Any]] = []
        for title, text in pool:
            body = text.split("|", 1)[1] if "|" in text else text
            tname = " ".join(w.lower() for w in _WORD.findall(title) if w.lower() not in _STOP)
            for pos, sent in enumerate(self._split(body)):
                u = {
                    "title": title,
                    "pos": pos,
                    "sent": sent,
                    "tname": tname,
                    "names": self._names(sent) | ({tname} if pos == 0 and tname else set()),
                    "toks": self._toks(sent),
                }
                units.append(u)
                for w in u["toks"]:
                    df[w] = df.get(w, 0) + 1
        if not units:
            return []
        n_units = len(units)

        def anchored(u: dict[str, Any]) -> bool:
            return any(qn in u["names"] or any(qn in nm or nm in qn for nm in u["names"]) for qn in qnames)

        # Hop 1: sentences the question's own names reach. The entities they introduce are the bridges.
        bridge_names: dict[str, float] = {}
        for u in units:
            u["anchor"] = anchored(u)
            if u["anchor"]:
                for nm in u["names"] - qnames:
                    if not any(qn in nm for qn in qnames):
                        bridge_names[nm] = bridge_names.get(nm, 0.0) + 1.0

        # Following a link lands you on a whole article, not on one sentence: once an article is on the
        # path -- because it mentions what the question names, or because its own name IS a bridge the
        # question's sentences handed over -- every sentence in it is a candidate, decayed by position.
        # Without this the pivot's article contributes only its lead sentence and the asked-for fact,
        # which is by construction stated further down, is pruned before the answerer sees it.
        on_path: dict[str, float] = {}
        for u in units:
            t = u["title"]
            if u["anchor"]:
                on_path[t] = max(on_path.get(t, 0.0), 1.5)
            # An article's name carries a parenthetical sense marker, so it is rarely equal to the name a
            # sentence prints for the same entity; match on whole words instead of on the whole string.
            if u["tname"] and (_covers(u["tname"], bridge_names) or _covers(u["tname"], qnames)):
                on_path[t] = max(on_path.get(t, 0.0), 2.4)

        def idf(w: str) -> float:
            return math.log(1.0 + n_units / (1.0 + df.get(w, 0)))

        scored: list[tuple[float, dict[str, Any]]] = []
        for u in units:
            hits = u["toks"] & prop_words
            prop_score = sum(idf(w) for w in hits)
            reach = sum(1.0 for nm in u["names"] if nm in bridge_names)
            s = 0.0
            if u["anchor"]:
                s += 3.0                      # names something the question names
            if reach:
                s += 1.6 * min(reach, 2.0)    # names something an anchored sentence introduced
            if u["anchor"] and (u["names"] - qnames):
                s += 1.2                      # a bridging sentence: it hands over a new entity
            s += 0.9 * prop_score             # states the property the question wants
            s += on_path.get(u["title"], 0.0) / (1.0 + 0.5 * u["pos"])  # inside an article on the path
            if u["pos"] == 0:
                s += 1.0                      # the lead sentence identifies its article
            if not u["anchor"] and not reach and prop_score <= 0 and u["title"] not in on_path:
                s -= 2.5                      # unreachable from the question: noise
            scored.append((s, u))

        scored.sort(key=lambda su: (-su[0], su[1]["title"], su[1]["pos"]))
        picked: list[dict[str, Any]] = []
        titles: list[str] = []
        for s, u in scored:
            if len(picked) >= self.MAX_SENTENCES:
                break
            if s <= 0:
                continue
            if u["title"] not in titles:
                if len(titles) >= self.MAX_ARTICLES:
                    continue
                titles.append(u["title"])
            picked.append(u)
        if not picked:
            picked = [u for _, u in scored[: self.MAX_SENTENCES]]
            titles = list(dict.fromkeys(u["title"] for u in picked))

        # Restore reading order inside each article, keep articles in the order they first scored.
        rank = {t: i for i, t in enumerate(titles)}
        picked.sort(key=lambda u: (rank.get(u["title"], 99), u["pos"]))
        return [(u["title"], u["sent"]) for u in picked]

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip()
        ans = ans.strip("*` ").strip()
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    @staticmethod
    def _render(chain: list[tuple[str, str]]) -> list[str]:
        """One entry per article: its name, then the sentences of it that scored onto the path."""
        out: list[str] = []
        order: list[str] = []
        grouped: dict[str, list[str]] = {}
        for title, sent in chain:
            if title not in grouped:
                grouped[title] = []
                order.append(title)
            grouped[title].append(sent)
        for title in order:
            out.append(f"{title} | " + " ".join(grouped[title]))
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if not queries:
            queries = [question]

        pool: list[tuple[str, str]] = []
        seen: set[str] = set()

        def absorb(qs: list[str], k: int) -> None:
            for q in qs:
                for h in ctx.tools.retrieve_docs(q, k=k):
                    title = (h.get("title") or "").strip()
                    if not title or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    pool.append((title, h.get("text") or f"{title} |"))

        absorb(queries, self.K1)
        ctx.note(f"round1: {len(queries)} queries -> {len(pool)} articles", queries=queries)

        # The pivot call reads the link-scored chain of round 1, not the raw pool.
        first = self._chain(pool, question, [])
        judged = self.pivot(ctx.llm, tag="pivot", question=question, sentences=self._render(first))
        pivots = [
            n for n in self._lines(judged.get("pivot_names", ""), self.MAX_BRIDGES + 2)
            if len(n) > 2 and not _PLACEHOLDER.search(n)
        ][: self.MAX_BRIDGES]
        prop = " ".join((judged.get("missing_property", "") or "").split())[:60]
        if prop.lower().startswith("none"):
            prop = ""

        hop = self._compose(pivots, prop, question, queries)
        if hop:
            absorb(hop, self.K2)
            ctx.note(f"hop: {hop} -> pool {len(pool)} articles", pivots=pivots, prop=prop)

        chain = self._chain(pool, question, pivots)
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=self._render(chain))
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = chain[0][0] if chain else answer
        return {
            "answer": answer,
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "n_pool": len(pool),
            "n_sentences": len(chain),
        }
