"""Answers are chosen from spans that the evidence itself contains, not written freehand.

Mechanism change vs the parent (decomposed_multiquery_rerank): the answerer's string is treated as a
*pointer into the evidence* rather than as the final answer. A deterministic aligner locates that string
in the retrieved text and enumerates the competing spans the text actually offers around it -- the fuller
proper-noun run the source uses at that point, the same run carrying its trailing common noun, and the
individual conjuncts when the draft welded two things together. When the evidence offers more than one
such span, a single arbitration call picks which one a person marking up the passage would have
highlighted. Retrieval is untouched.

Rationale: with both gold articles in hand the parent still misses a fifth of the time, and the misses
are overwhelmingly granularity, not knowledge -- a familiar short name substituted for the fuller one the
lead sentence uses, a place stripped of the qualifier the source attaches to it, or two names conjoined
where one was asked for. None of these can be fixed by instructing the answerer harder, because the
answerer never sees the alternatives side by side; the aligner manufactures that comparison from the text.

Calls: 3-4 LLM, 3-5 retrievals.
"""

from __future__ import annotations

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

# Word tokens with their character offsets, so a chosen span can be sliced back out of the source text
# with its original spelling, accents and casing intact.
_TOKEN = re.compile(r"[^\W_]+(?:['’‘\-][^\W_]+)*", re.UNICODE)
_PARTICLE = {
    "of", "the", "de", "del", "della", "di", "da", "dos", "du", "van", "von", "der", "den",
    "la", "le", "los", "las", "bin", "ibn", "al", "and", "y", "e",
}
_STOPTAIL = {
    "is", "was", "were", "are", "be", "been", "in", "on", "at", "to", "for", "from", "by", "with",
    "as", "of", "that", "which", "who", "and", "or", "but", "it", "he", "she", "they", "a", "an", "the",
    "his", "her", "its", "their", "this", "these", "has", "had", "have", "also", "then", "when",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(]?[A-Z0-9])")


def _toks(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _tok_eq(a: str, b: str) -> bool:
    """Token equality that tolerates a shortened forename ('Ron' vs 'Ronald')."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return len(short) >= 3 and long.startswith(short)


def _is_subseq(needle: list[str], hay: list[str]) -> bool:
    i = 0
    for h in hay:
        if i < len(needle) and _tok_eq(needle[i], h):
            i += 1
    return i == len(needle)


class Harness(BaseHarness):
    K1 = 8
    K2 = 8
    MAX_FIRST_QUERIES = 3
    MAX_FOLLOWUPS = 2
    MAX_KEPT = 9
    MIN_KEPT = 5
    MAX_ANSWER_WORDS = 12
    MAX_OPTIONS = 5
    MAX_GROW = 4  # a variant may be at most this many tokens longer than the draft

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

    ASSESS_INSTR = (
        "`passages` is a numbered pool of search results, most of them irrelevant.\n"
        "`useful_ids`: the numbers of the passages that carry a fact the question needs, most relevant "
        "first, at most six numbers, separated by commas. Judge relevance by whether the passage is about "
        "an entity the question actually asks about, not by superficial word overlap.\n"
        "`missing_info`: in one line, the fact the question still needs and that no listed passage states. "
        "Write exactly `none` if the passages together already settle the question.\n"
        "`followup_queries`: if something is missing, one or two queries, one per line. Each must START with the "
        "name of the intermediate entity the passages just revealed (the person, work, place or organisation "
        "that links the question to the missing fact), because that entity's own article is what must be "
        "retrieved next, and may add one or two words naming the missing property. Write titles and noun "
        "phrases, never questions. Write exactly `none` when nothing is missing."
    )

    ANSWER_INSTR = (
        "Answer the question from the numbered `evidence` passages.\n"
        "`answer` must be the bare answer and nothing else: a name, title, date or quantity, spelled exactly "
        "as the evidence spells it, or the single word yes or no when the question asks whether something "
        "holds. Never write a sentence, never restate the question, never explain. If the evidence is "
        "incomplete, give the most likely short answer anyway rather than saying it cannot be determined."
    )

    PICK_INSTR = (
        "`options` are the competing wordings the source text offers for the same answer. Exactly one of "
        "them is the span that someone marking up `source` as the answer to `question` would have "
        "highlighted. Write its number in `choice` and nothing else.\n"
        "Judge by these rules, in order:\n"
        "- The span must name one single thing. If the question asks for one person, work or place, an "
        "option that welds two of them together with `and` is wrong, even when both were involved.\n"
        "- Otherwise prefer the wording the source itself uses where it introduces the thing, not the "
        "shorter form you happen to be familiar with: keep the middle names, the title, the year or the "
        "administrative or regional qualifier when the source prints them next to the name.\n"
        "- The span must still be a direct answer on its own. Reject an option that has swallowed a word "
        "belonging to the surrounding sentence rather than to the thing itself, and reject an option that "
        "has been narrowed so far it no longer answers what was asked."
    )

    def __init__(self) -> None:
        self.decompose = Module(inputs=["question"], outputs=["queries"], instructions=self.DECOMPOSE_INSTR)
        self.assess = Module(
            inputs=["question", "passages"],
            outputs=["useful_ids", "missing_info", "followup_queries"],
            instructions=self.ASSESS_INSTR,
        )
        self.answer = Module(inputs=["question", "evidence"], outputs=["answer"], instructions=self.ANSWER_INSTR)
        self.pick = Module(
            inputs=["question", "source", "options"], outputs=["choice"], instructions=self.PICK_INSTR
        )

    # -- retrieval helpers (unchanged from the parent) -------------------------------------
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
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    def _add(self, pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = self._title_of(p)
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

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

    # -- span alignment --------------------------------------------------------------------
    @staticmethod
    def _grow_right(text: str, toks: list[tuple[str, int, int]], j: int, start: int, n: int) -> list[str]:
        """Spans made by absorbing up to `n` further word tokens, stopping at any punctuation."""
        out: list[str] = []
        end = toks[j][2]
        for step in range(n):
            k = j + 1 + step
            if k >= len(toks):
                break
            gap = text[end : toks[k][1]]
            if gap.strip() not in {"", "-", "–"}:
                break
            word = toks[k][0]
            if word.lower() in _STOPTAIL:
                break
            end = toks[k][2]
            out.append(text[start:end])
        return out

    def _variants(self, draft: str, sources: list[str]) -> list[str]:
        """Competing spans for `draft` that literally occur in one of the `sources`.

        Two families: *fuller* spans, found by locating a proper-noun run of the source whose tokens
        contain the draft's tokens in order (this is what recovers a middle name or a place qualifier
        that the draft dropped, even though the draft itself never occurs verbatim); and *narrower*
        spans, obtained by splitting a conjunction or dropping a leading modifier.
        """
        d_toks = [t[0] for t in _toks(draft)]
        if not d_toks:
            return []
        # A draft that is already a complete proper name is only ever missing more *name*; a common noun
        # sitting after it belongs to the sentence, not to the thing. Quantities and dated events are the
        # opposite case, and those drafts are not wholly capitalised.
        proper = all(t[:1].isupper() or t.isdigit() for t in d_toks)
        out: list[str] = []

        def offer(cand: str) -> None:
            cand = cand.strip().strip(",;:").strip()
            c_toks = [t[0] for t in _toks(cand)]
            if not c_toks or len(cand) > 90:
                return
            if c_toks[0].lower() in _STOPTAIL or c_toks[-1].lower() in _STOPTAIL:
                return
            if proper and len(c_toks) > len(d_toks):
                tail = [t for t in c_toks if t.lower() not in _PARTICLE]
                if any(not (t[:1].isupper() or t.isdigit()) for t in tail):
                    return
            if " ".join(c_toks).lower() == " ".join(d_toks).lower():
                return
            for o in out:
                if o.lower() == cand.lower():
                    return
            out.append(cand)

        for src in sources:
            toks = _toks(src)
            low = [t[0].lower() for t in toks]
            # fuller: proper-noun runs (possibly carrying one or two trailing common nouns) that
            # contain the draft as an ordered token subsequence.
            i = 0
            while i < len(toks):
                word = toks[i][0]
                if not (word[:1].isupper() or re.fullmatch(r"\d{4}", word)):
                    i += 1
                    continue
                j = i
                while j + 1 < len(toks):
                    nxt = toks[j + 1][0]
                    gap = src[toks[j][2] : toks[j + 1][1]]
                    if gap not in {" ", ", ", "-", "–", " - "}:
                        break
                    if not (nxt[:1].isupper() or re.fullmatch(r"\d{4}", nxt) or nxt.lower() in _PARTICLE):
                        break
                    j += 1
                while j > i and toks[j][0].lower() in _PARTICLE:
                    j -= 1
                run_start = toks[i][1]
                spans = [src[run_start : toks[j][2]]] + self._grow_right(src, toks, j, run_start, 2)
                for sp in spans:
                    s_toks = [t[0] for t in _toks(sp)]
                    if len(s_toks) > len(d_toks) and len(s_toks) <= len(d_toks) + self.MAX_GROW:
                        if _is_subseq(d_toks, s_toks):
                            offer(sp)
                i = j + 1
            # fuller: the draft occurs verbatim and the source attaches a common noun to it
            # ("six" printed as "six children").
            n = len(d_toks)
            for s in range(len(toks) - n + 1):
                if all(_tok_eq(d_toks[t], low[s + t]) for t in range(n)):
                    for sp in self._grow_right(src, toks, s + n - 1, toks[s][1], 2):
                        offer(sp)

        # narrower: the draft welded several things together, or piled a modifier on the head noun.
        parts = re.split(r"\s+(?:and|&)\s+|\s*;\s*", draft)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            for p in parts:
                offer(p)
        if 2 <= len(d_toks) <= 4 and draft == draft.lower():
            offer(d_toks[-1])
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", draft).strip()
        if stripped and stripped != draft:
            offer(stripped)
        return out[: self.MAX_OPTIONS - 1]

    @staticmethod
    def _source_window(probes: list[str], evidence_text: str) -> str:
        """The sentence of the retrieved text that the competing wordings all come out of.

        Probed with the longest variant first: a fuller wording occurs verbatim by construction, while
        the draft itself often does not (that is precisely what the arbitration is for).
        """
        flat = " ".join(evidence_text.split())
        low = flat.lower()
        for probe in probes:
            needle = " ".join(probe.split()).lower()
            if not needle:
                continue
            pos = low.find(needle)
            if pos < 0:
                continue
            left = flat.rfind(". ", 0, pos)
            start = left + 2 if left >= 0 else max(0, pos - 240)
            tail = flat[pos + len(needle) :]
            m = _SENT_SPLIT.search(tail)
            end = pos + len(needle) + (m.start() if m else min(240, len(tail)))
            return flat[start:end].strip()
        return flat[:400]

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        plan = self.decompose(ctx.llm, tag="decompose", question=question)
        queries = self._lines(plan.get("queries", ""), self.MAX_FIRST_QUERIES)
        if question not in queries:
            queries = queries[: self.MAX_FIRST_QUERIES - 1] + [question] if queries else [question]

        pool: list[str] = []
        seen: set[str] = set()
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K1))
        ctx.note(f"round1: {len(queries)} queries -> {len(pool)} unique passages", queries=queries)

        numbered = [f"{i + 1}. {p}" for i, p in enumerate(pool)]
        judged = self.assess(ctx.llm, tag="assess", question=question, passages=numbered)
        ids: list[int] = []
        for tok in _NUM.findall(judged.get("useful_ids", "") or ""):
            idx = int(tok) - 1
            if 0 <= idx < len(pool) and idx not in ids:
                ids.append(idx)
        kept = [pool[i] for i in ids[: self.MAX_KEPT]]

        followups = self._lines(judged.get("followup_queries", ""), self.MAX_FOLLOWUPS)
        missing = (judged.get("missing_info", "") or "").strip().lower()
        sufficient = (not followups) or missing.startswith("none")

        if not sufficient:
            round2: list[str] = []
            seen2 = {self._title_of(p) for p in kept}
            for q in followups:
                self._add(round2, seen2, ctx.tools.retrieve(q, k=self.K2))
            ctx.note(f"round2: {len(followups)} followups -> {len(round2)} new passages", followups=followups)
            kept = kept + round2
        if len(kept) < self.MIN_KEPT:
            have = {self._title_of(p) for p in kept}
            for p in pool:
                if self._title_of(p) not in have:
                    kept.append(p)
                    have.add(self._title_of(p))
                if len(kept) >= self.MIN_KEPT:
                    break

        evidence = [f"{i + 1}. {p}" for i, p in enumerate(kept)]
        out = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence)
        answer = self._clean(out.get("answer", ""))
        if not answer or _NONANSWER.match(answer):
            answer = kept[0].split("|", 1)[0].strip() if kept else answer

        # SPAN ARBITRATION: only when the retrieved text really offers a competing wording.
        options: list[str] = []
        if answer.lower() not in {"yes", "no"} and len(answer) <= 90:
            options = self._variants(answer, [s for s in kept[:12] if s.strip()])
        if options:
            choices = [answer] + options
            probes = sorted(options, key=len, reverse=True) + [answer]
            source = self._source_window(probes, "\n".join(kept))
            ctx.note(f"span options: {len(choices)}", options=choices)
            verdict = self.pick(
                ctx.llm,
                tag="pick",
                question=question,
                source=source,
                options=[f"{i + 1}. {c}" for i, c in enumerate(choices)],
            )
            hit = _NUM.search(verdict.get("choice", "") or "")
            if hit:
                idx = int(hit.group(0)) - 1
                if 0 <= idx < len(choices):
                    answer = self._clean(choices[idx])

        return {
            "answer": answer,
            "queries": queries,
            "followups": followups,
            "sufficient": sufficient,
            "n_evidence": len(kept),
            "n_options": len(options),
        }
