"""Retrieval driven by hypothesised answers, and a contrastive verdict over them.

Mechanism change: every previous system in this lineage searches *for* the answer -- queries are built
from the question, or from an entity the first round happened to surface, and whatever comes back is
summarised or reranked into one undifferentiated evidence pile. This one inverts the direction. After a
cheap seed round it asks the solver to name several competing answers outright, together with the one
condition each would have to satisfy, and then spends its remaining retrieval budget fetching *each
candidate's own article*. The answer string is the query.

Why that should retrieve better: a two-article question hides the second article behind an entity the
question never names, so no query written from the question can be worded like that article's title.
But a candidate answer usually IS that title, or close enough for a term-matching index to land on it --
and this works even when the guess is wrong, because a wrong guess still pulls in the near neighbours
that let the adjudicator see it is wrong.

Why that should answer better: the evidence is no longer a flat list. It arrives partitioned by the
candidate it was fetched for, so the last call is a comparison between a small number of fully-evidenced
hypotheses -- each condition in the question checked against the article of the thing being proposed --
rather than an open-ended extraction from a pile of passages.

Calls: 3 LLM, up to 9 retrievals.
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
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*|^\s*(?:candidate|answer|option)\s*[a-z0-9]*\s*[:.)-]\s*", re.IGNORECASE)
_NONANSWER = re.compile(
    r"^(?:unknown|unclear|undetermined|not\s+(?:determined|specified|available|stated|found)|n/?a|none|"
    r"cannot\b|can't\b|insufficient)\b",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"\s*(?:::|\|\||\t)\s*")
_LABEL = re.compile(r"^\s*(?:candidate\s+)?([A-H])\s*[:.)]\s*", re.IGNORECASE)


class Harness(BaseHarness):
    K_SEED = 10
    K_GUESS = 6
    K_VERIFY = 6
    MAX_ANCHORS = 2
    MAX_GUESS_QUERIES = 2
    MAX_CANDIDATES = 4
    MAX_SHARED = 6
    MAX_PER_CANDIDATE = 3
    MAX_ANSWER_WORDS = 12
    LABELS = "ABCDEFGH"

    SEED_INSTR = (
        "You are about to search a term-matching index of short reference articles, one per subject.\n"
        "`anchor_query`: one line per subject the question names outright, at most two, each worded the way "
        "that subject's own article would be titled -- a bare noun phrase, no question words, no verbs. When "
        "the question names two things, give each its own line: one query phrased from the whole question "
        "finds only whichever of them dominates the wording.\n"
        "`guesses`: up to two lines, each a complete answer to the question that you already consider "
        "plausible from what you know, written as the bare answer alone. These are hypotheses to be "
        "checked, not claims, so guess even when unsure. Write exactly `none` only if the question names "
        "nothing you recognise."
    )

    PROPOSE_INSTR = (
        "`passages` are search results for the question; several are irrelevant.\n"
        "Name the answers worth checking. `candidates`: up to four lines, most promising first, each in "
        "the form\n"
        "    <bare answer> :: <search query that would retrieve that answer's own article>\n"
        "- The bare answer is what you would submit: a name, title, date, quantity, or yes / no. Copy its "
        "spelling from a passage whenever a passage prints it.\n"
        "- Include a candidate the passages support AND, if the question is under-determined by them, one "
        "or two the passages merely hint at. Competing candidates are the point; do not list one line "
        "unless the passages leave no doubt at all.\n"
        "- The search query is normally just the candidate's own name or title, plus at most two words "
        "naming the property the question asks about. It is a noun phrase, never a question.\n"
        "- For a yes / no question, the two lines are `yes` and `no`, and each query must retrieve the "
        "article that would settle it.\n"
        "`unresolved`: one line naming the fact still needed to choose between the candidates, or `none`."
    )

    VERDICT_INSTR = (
        "`candidates` are competing answers. `candidate_evidence` groups, under each candidate's label, "
        "the articles retrieved for that candidate specifically; `shared_evidence` are the passages found "
        "from the question itself.\n"
        "Test each candidate against EVERY condition the question states -- the described property, the "
        "date, the role, the place -- using the articles filed under it. A candidate that matches the "
        "wording of the question but whose own article contradicts one condition is wrong.\n"
        "`answer` must be the bare answer and nothing else, spelled as the evidence spells it: a name, "
        "title, date or quantity, or the single word yes or no when the question asks whether something "
        "holds. It must also be of the KIND the question asks for -- a person for who, a place for where, a "
        "date for when, a work for which film or album -- so a candidate of the wrong kind is wrong even "
        "where the evidence mentions it prominently. Never write a sentence, never restate the question, "
        "never explain, never mention a label. "
        "If every candidate fails, answer from the evidence anyway with the most likely short answer "
        "rather than saying it cannot be determined."
    )

    def __init__(self) -> None:
        self.seed = Module(inputs=["question"], outputs=["anchor_query", "guesses"], instructions=self.SEED_INSTR)
        self.propose = Module(
            inputs=["question", "passages"], outputs=["candidates", "unresolved"], instructions=self.PROPOSE_INSTR
        )
        self.verdict = Module(
            inputs=["question", "candidates", "candidate_evidence", "shared_evidence"],
            outputs=["answer"],
            instructions=self.VERDICT_INSTR,
        )

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = _BULLET.sub("", line).strip().strip('"“”*` ').strip()
            if not line or line.lower().strip(".") in {"none", "n/a", "null", "unknown"}:
                continue
            out.append(line[:300])
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _title_of(passage: str) -> str:
        return passage.split("|", 1)[0].strip().lower()

    def _add(self, pool: list[str], seen: set[str], passages: list[str]) -> list[str]:
        fresh = []
        for p in passages:
            key = self._title_of(p)
            if key and key not in seen:
                seen.add(key)
                pool.append(p)
                fresh.append(p)
        return fresh

    def _clean(self, raw: str) -> str:
        ans = (raw or "").strip().split("\n")[0].strip()
        ans = ans.strip("*` ").strip()
        ans = _LABEL.sub("", ans)
        ans = _LEAD.sub("", ans).strip().strip('"“”‘’ ')
        if re.match(r"^(?:yes|no)\b\s*[,.:]?\s*$", ans, re.IGNORECASE) or re.match(
            r"^(?:yes|no)\b\s*[,.:]", ans, re.IGNORECASE
        ):
            return "yes" if _YES.match(ans) else "no"
        ans = re.sub(r"\s*[.;,]\s*$", "", ans).strip()
        if len(ans.split()) > self.MAX_ANSWER_WORDS:
            ans = " ".join(ans.split()[: self.MAX_ANSWER_WORDS])
        return ans

    def _parse_candidates(self, raw: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for line in self._lines(raw, self.MAX_CANDIDATES * 2):
            parts = _SPLIT.split(line, maxsplit=1)
            text = _LABEL.sub("", parts[0]).strip().strip('"“”‘’ ')
            if not text or len(text.split()) > self.MAX_ANSWER_WORDS:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            query = parts[1].strip() if len(parts) > 1 and parts[1].strip() else text
            out.append((text, query[:200]))
            if len(out) >= self.MAX_CANDIDATES:
                break
        return out

    # -- run -------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # SEED: the named subject's article, the question itself, and whatever the solver already suspects.
        plan = self.seed(ctx.llm, tag="seed", question=question)
        anchors = self._lines(plan.get("anchor_query", ""), self.MAX_ANCHORS)
        guesses = self._lines(plan.get("guesses", ""), self.MAX_GUESS_QUERIES)

        shared: list[str] = []
        seen: set[str] = set()
        self._add(shared, seen, ctx.tools.retrieve(question, k=self.K_SEED))
        for a in anchors:
            if a.lower() != question.lower():
                self._add(shared, seen, ctx.tools.retrieve(a[:200], k=self.K_SEED))
        for g in guesses:
            self._add(shared, seen, ctx.tools.retrieve(g, k=self.K_GUESS))
        ctx.note(f"seed: anchors={anchors} guesses={guesses} -> {len(shared)} passages")

        # PROPOSE: competing answers, each with the query that fetches its own article.
        numbered = [f"{i + 1}. {p}" for i, p in enumerate(shared)]
        prop = self.propose(ctx.llm, tag="propose", question=question, passages=numbered)
        cands = self._parse_candidates(prop.get("candidates", ""))
        unresolved = (prop.get("unresolved", "") or "").strip().splitlines()[0][:200] if prop.get("unresolved") else ""

        # VERIFY: one retrieval per candidate, filed under that candidate.
        grouped: list[tuple[str, str, list[str]]] = []
        for label, (text, query) in zip(self.LABELS, cands):
            try:
                hits = ctx.tools.retrieve(query, k=self.K_VERIFY)
            except Exception:
                break
            local: list[str] = []
            local_seen: set[str] = set()
            for p in hits:
                key = self._title_of(p)
                if key in local_seen:
                    continue
                local_seen.add(key)
                local.append(p)
                if len(local) >= self.MAX_PER_CANDIDATE:
                    break
            seen.update(local_seen)
            grouped.append((label, text, local))
        ctx.note(
            f"verified {len(grouped)} candidates; unresolved={unresolved!r}",
            candidates=[c[0] for c in cands],
            queries=[c[1] for c in cands],
        )

        if not grouped:
            # No usable hypothesis: fall back to a flat read of the seed pool.
            out = self.verdict(
                ctx.llm,
                tag="verdict",
                question=question,
                candidates=["none proposed"],
                candidate_evidence=["none"],
                shared_evidence=numbered[: self.MAX_SHARED + 4],
            )
            answer = self._clean(out.get("answer", ""))
            if not answer or _NONANSWER.match(answer):
                answer = shared[0].split("|", 1)[0].strip() if shared else answer
            return {"answer": answer, "n_candidates": 0, "fallback": True}

        cand_lines = [f"{label}. {text}" for label, text, _ in grouped]
        ev_lines: list[str] = []
        for label, text, local in grouped:
            ev_lines.append(f"--- filed under {label} ({text}) ---")
            ev_lines.extend(f"{label}{i + 1}. {p}" for i, p in enumerate(local))
            if not local:
                ev_lines.append(f"{label}1. (no article found)")
        shared_lines = numbered[: self.MAX_SHARED]

        out = self.verdict(
            ctx.llm,
            tag="verdict",
            question=question,
            candidates=cand_lines,
            candidate_evidence=ev_lines,
            shared_evidence=shared_lines,
        )
        answer = self._clean(out.get("answer", ""))
        # Abstaining scores the same as a wrong guess, so fall back to the leading hypothesis.
        if not answer or _NONANSWER.match(answer):
            answer = grouped[0][1]
        return {
            "answer": answer,
            "n_candidates": len(grouped),
            "candidates": [c[1] for c in grouped],
            "unresolved": unresolved,
            "fallback": False,
        }
