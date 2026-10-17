"""Answer-extent rules whose activation is MEASURED on the train split before they may fire.

Mechanism change vs the parent (`grounded_answer_repair`). The parent accepts whatever short string
its answerer writes as long as some passage prints it. But the accepted string is a span of a
supporting sentence, and a span has an EXTENT: the same entity is printed as a bare title in one
sentence and with a middle name, a trailing region, or a head noun in another. Reading the traces of
the parent, a large share of its near misses differ from the accepted string only in extent -- a
person given by the form used as an article title where the article's own opening prints a fuller
name, a place without its containing region, a count without the noun it counts.

Every earlier attempt in this direction let a model re-choose the wording, and re-choosing wording
breaks as many answers as it fixes. This system instead defines a handful of DETERMINISTIC extent
rules, and refuses to let any of them fire until it has measured, on the train split, whether that
rule's expansion agrees with accepted answers more often than the unexpanded form does:

    prepare(): for every train question, pull a pool with the real index, enumerate every
    (short form, long form) pair each rule could produce from that pool, and count the items where
    the long form matches the label and the short one does not (a win) against the items where the
    short form matches and the long one does not (a loss). A rule is switched on only with a
    positive measured margin; the counts are carried in the harness state.

At run time the enabled rules are tried in a fixed order on the final draft, at most one fires, and
the replacement must itself be printed verbatim by a passage that was read. A rule the train split
does not endorse never runs, so an unsupported rule costs nothing instead of costing accuracy.

Calls: unchanged from the parent (3 LLM, 5-6 retrievals; 4 and up to 9 when the grounding test fires).
The extent stage is pure string work over evidence already in hand.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

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

# The opening of an abstract, up to the parenthesis or copula that follows the article's self-name.
_SELFNAME = re.compile(r"^\s*([A-Z][^.!?]{0,120}?)\s*(?:\(|\bis\b|\bwas\b|\bare\b|\bwere\b|\bwho\b|\bwhich\b)")
_CAPRUN = re.compile(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}")
_HEADNOUN = re.compile(
    r"\b(?:what|which|whose|how\s+many|how\s+much)\s+(?:the\s+)?((?:[a-z]+\s+){0,2}[a-z]+)\b",
    re.IGNORECASE,
)

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "is", "was", "were", "are",
    "be", "been", "and", "or", "but", "that", "this", "these", "those", "it", "its", "his", "her", "their", "he",
    "she", "they", "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "did", "do", "does",
    "had", "has", "have", "you", "we", "i", "there", "than", "then", "also", "both", "same", "other", "another",
    "known", "called", "born", "name", "named", "many", "much", "one", "two", "first", "last", "year", "years",
    "would", "will", "can", "could", "into", "about", "between", "during", "along", "such", "some", "any", "not",
    "think", "know", "tell", "say", "said", "does", "featured", "starred", "made", "part", "well", "including",
}

# Nouns that are part of an entity's printed name rather than a description of it.
_UNIT_NOUNS = {
    "county", "city", "town", "village", "province", "state", "district", "borough", "parish", "township",
    "river", "lake", "mountain", "island", "university", "college", "school", "hospital", "airport", "station",
    "massacre", "battle", "war", "album", "film", "series", "team", "club", "company", "children", "people",
    "members", "episodes", "seasons", "songs", "films", "albums", "books", "novels", "languages", "countries",
    "students", "years", "months", "days", "miles", "kilometres", "kilometers",
}


class Harness(BaseHarness):
    K = 20
    MAX_FIRST_QUERIES = 3
    MAX_BRIDGES = 2
    MAX_HOP_QUERIES = 3
    SHORTLIST = 14
    MAX_EVIDENCE = 12
    RRF_C = 10.0
    W_TITLE = 3.0
    W_IDF = 1.6
    W_PICKED = 0.9
    MAX_ANSWER_WORDS = 12
    MAX_REPAIR_QUERIES = 3

    # calibration
    PREPARE_K = 20
    PREPARE_POOL = 16
    MIN_WINS = 2           # a rule needs this many demonstrated fixes on train ...
    MIN_MARGIN = 1         # ... and this much more than the answers it would have broken
    BASE_RULES = ("self_name", "self_name_full", "head_noun", "region_qualifier")
    # Every rule is offered in BOTH directions and the train split chooses: `expand_` rewrites the
    # short form of a pair as the long one, `contract_` rewrites the long one as the short one. Only
    # one direction of a base rule can clear the margin, so the two never compete at run time.
    RULE_ORDER = tuple(
        f"{d}_{b}" for b in ("self_name", "self_name_full", "head_noun", "region_qualifier")
        for d in ("expand", "contract")
    )

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
        self.enabled: dict[str, bool] = {r: False for r in self.RULE_ORDER}
        self.tally: dict[str, list[int]] = {}

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

    # -- the rerank ------------------------------------------------------------------------
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

    def _harvest(self, passages: list[str], question: str, limit: int) -> list[str]:
        qk = self._content(question)
        counts: dict[str, int] = {}
        for p in passages:
            for m in _PROPER.finditer(self._body(p)):
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

    def _repair_queries(self, draft: str, prop: str, question: str, already: list[str]) -> list[str]:
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

    # -- extent rules ----------------------------------------------------------------------
    @staticmethod
    def _subseq(short_toks: list[str], long_toks: list[str]) -> bool:
        """`short` is an ordered subsequence of `long`, sharing its first and last token."""
        if len(short_toks) < 2 or len(long_toks) <= len(short_toks):
            return False
        if short_toks[0] != long_toks[0] or short_toks[-1] != long_toks[-1]:
            return False
        i = 0
        for t in long_toks:
            if i < len(short_toks) and t == short_toks[i]:
                i += 1
        return i == len(short_toks)

    def _self_names(self, doc: dict[str, Any]) -> tuple[str, str]:
        """(name truncated at the first comma, full opening name) as the abstract prints them."""
        m = _SELFNAME.match(self._body(doc.get("text", "")))
        if not m:
            return "", ""
        full = m.group(1).strip().strip(",;:")
        if not full or len(full.split()) > 8:
            return "", ""
        return full.split(",")[0].strip(), full

    def _directed(self, rule: str, docs: list[dict[str, Any]], question: str) -> list[tuple[str, str]]:
        """(from, to) rewrites, i.e. the base pairs read in the direction this rule stands for."""
        direction, base = rule.split("_", 1)
        pairs = self._pairs(base, docs, question)
        if direction == "contract":
            pairs = [(long, short) for short, long in pairs]
        return [
            (a, b) for a, b in pairs
            if len(b) >= 3 and b.strip().lower() not in _STOP and self._norm_answer(a) != self._norm_answer(b)
        ]

    def _pairs(self, rule: str, docs: list[dict[str, Any]], question: str) -> list[tuple[str, str]]:
        """Every (short form, long form) this rule can read off the pool."""
        out: list[tuple[str, str]] = []
        if rule in ("self_name", "self_name_full"):
            for h in docs:
                title = _PAREN.sub("", (h.get("title") or "")).strip()
                short_c, full = self._self_names(h)
                cand = short_c if rule == "self_name" else full
                if not title or not cand or self._norm_phrase(title) == self._norm_phrase(cand):
                    continue
                if self._subseq(self._norm_phrase(title).split(), self._norm_phrase(cand).split()):
                    out.append((title, cand))
            return out
        if rule == "region_qualifier":
            for h in docs:
                body = self._body(h.get("text", ""))
                for m in re.finditer(r"([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3}),\s+([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,1})(?=[\s,.;)]|$)", body):
                    short, qual = m.group(1).strip(), m.group(2).strip()
                    if len(short) < 4 or len(qual) < 3:
                        continue
                    out.append((short, f"{short}, {qual}"))
            return out
        if rule == "head_noun":
            heads = [
                w for m in _HEADNOUN.finditer(question) for w in m.group(1).lower().split()
                if w in _UNIT_NOUNS
            ]
            if not heads:
                return out
            for h in docs:
                body = self._body(h.get("text", ""))
                for head in set(heads):
                    for m in re.finditer(r"([\w'’.-]+(?:\s+[\w'’.-]+){0,3})\s+(" + re.escape(head) + r")\b", body, re.IGNORECASE):
                        short = m.group(1).strip()
                        if len(short) < 2 or short.lower() in _STOP:
                            continue
                        out.append((short, f"{short} {m.group(2)}"))
            return out
        return out

    def _apply_extent(self, answer: str, docs: list[dict[str, Any]], question: str) -> tuple[str, str]:
        """The first enabled rule whose short form IS the draft rewrites it. At most one fires."""
        norm = self._norm_answer(answer)
        if not norm or norm in {"yes", "no"} or len(answer.split()) > 6:
            return answer, ""
        for rule in self.RULE_ORDER:
            if not self.enabled.get(rule):
                continue
            for src, dst in self._directed(rule, docs, question):
                if self._norm_answer(src) != norm or self._norm_answer(dst) == norm:
                    continue
                if self._grounded(dst, docs, question):
                    return dst, rule
        return answer, ""

    # -- prepare: measure each rule against the labels -------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        tally = {r: [0, 0] for r in self.RULE_ORDER}
        for item in train_items:
            gold = str(item.get("answer") or "").strip()
            question = str(item.get("question") or "").strip()
            if not gold or not question or normalize_text(gold) in {"yes", "no"}:
                continue
            try:
                with ctx.open_tools(item) as tools:
                    hits = tools.retrieve_docs(question, k=self.PREPARE_K)
            except Exception:
                continue
            ranked = self._rerank([hits], [question], question, picked=set())
            docs = [h for _, h in ranked[: self.PREPARE_POOL]]
            if not docs:
                continue
            g = self._norm_answer(gold)
            for rule in self.RULE_ORDER:
                win = loss = False
                for src, dst in self._directed(rule, docs, question):
                    ns, nd = self._norm_answer(src), self._norm_answer(dst)
                    if nd == g and ns != g:
                        win = True          # the rewrite would have produced the accepted answer
                    elif ns == g and nd != g:
                        loss = True         # the rewrite would have destroyed an accepted answer
                tally[rule][0] += int(win)
                tally[rule][1] += int(loss)
        self.tally = tally
        self.enabled = {
            r: (w >= self.MIN_WINS and w - l >= self.MIN_MARGIN) for r, (w, l) in tally.items()
        }
        ctx.note(f"extent rules measured on {len(train_items)} train items", tally=tally, enabled=self.enabled)

    def get_state(self) -> str:
        return json.dumps({"enabled": self.enabled, "tally": self.tally}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        try:
            blob = json.loads(state or "{}") or {}
        except Exception:
            blob = {}
        loaded = blob.get("enabled") or {}
        self.enabled = {r: bool(loaded.get(r)) for r in self.RULE_ORDER}
        self.tally = blob.get("tally") or {}

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
                if cand and not _NONANSWER.match(cand) and self._grounded(cand, evidence_docs, question):
                    answer = cand
                    repaired = True

        # The extent stage: only rules the train split endorsed may rewrite the draft.
        final, fired = self._apply_extent(answer, evidence_docs, question)
        if fired:
            ctx.note(f"extent rule {fired}: {answer!r} -> {final!r}", rule=fired)
        return {
            "answer": final,
            "draft": answer,
            "extent_rule": fired,
            "enabled_rules": [r for r in self.RULE_ORDER if self.enabled.get(r)],
            "queries": queries,
            "pivots": pivots,
            "missing_property": prop,
            "hop_queries": hop,
            "repair_queries": repair_qs,
            "repaired": repaired,
            "n_evidence": len(evidence_docs),
        }
