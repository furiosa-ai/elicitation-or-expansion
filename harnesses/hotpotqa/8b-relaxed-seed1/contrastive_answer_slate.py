"""Contrastive slate: enumerate the competing answers with their evidence, then discriminate between them.

Measured on the parent's own 300 traces, its dominant loss is not retrieval and not wording. On 53 of its 95
wrong items the gold answer string is present in the very prompt the final call read, and the call returned a
DIFFERENT span of that prompt: the successor instead of the predecessor, the aircraft instead of the project it
served, the live album instead of the studio album a song came from, the co-star instead of the director. Only
31 wrong items lack the answer anywhere in the retrieved text.

A single generative call cannot be audited on that axis, because the competing span is never written down --
by the time the answer is emitted the alternative has been silently discarded inside the reasoning. So the
answering stage stops being generation and becomes enumeration followed by discrimination:

    1-3. decomposed sweep, evidence pool, router      (unchanged from the parent)
    4.   draft exactly as the parent does             (unchanged: the draft is the default, never overruled lightly)
    5.   slate: enumerate the mutually exclusive answers the passages support -- including the draft and the
         near-miss a careless reader would take -- each as `answer :: passage number :: verbatim sentence`
    6.   validate the slate DETERMINISTICALLY: a candidate survives only if its string really occurs in the
         passage it cites, and the title is re-attached from the pool so it cannot be garbled
    7.   discriminate: one call picks the surviving line whose evidence fills the slot the question asks for,
         with an explicit tie-break back to the draft

Three earlier attempts changed the answer's WORDING (a length preference, a support-sentence rule, mined
demonstrations); this changes what the decision is over. Nothing is rewritten, the choice is confined to strings
the passages literally contain, and when the slate offers no genuine competitor the draft is returned with no
further call, so the two thirds of items the parent already solves are only touched where a rival answer exists.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_BULLET = re.compile(r"^\s*(?:[-*•]|\[?\d+[.)\]]?)\s*")
_INDEX = re.compile(r"\d+")
_LEADING_FILLER = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|short\s+)?answer(?:\s+is)?\s*[:\-]?\s*", re.IGNORECASE
)
_YESNO_PREFIX = re.compile(r"^\s*(yes|no)\b\s*[,.:;!\-]*", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_MARKUP = re.compile(r"[*_`]+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)

DECOMPOSE_INSTRUCTIONS = """Plan a keyword search for a question that can only be answered by combining
two different source documents.

Write 3 search queries, one per line, no numbering and no explanation:
1. the most specific description in the question, phrased as the title/topic of the document that describes it;
2. a short keyword-only variant of that same target using distinctive proper nouns, dates or numbers only
   (no question words, no stop words) -- it should retrieve the same document by different wording;
3. the second document: the entity or property that the question ultimately asks about, phrased as a topic.

The retriever matches words, not meaning, so prefer the exact names and rare terms that would appear in the
document, and never write a query as a question."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. In `reasoning`, decide which chain of facts the question needs, then
check which links of that chain the passages already establish.

Produce:
- `useful`: the numbers of the passages that state a needed fact, most important first, comma separated.
  Include a passage only if it carries a fact the answer depends on; write `none` if no passage does.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property
  must still be looked up). Write `none` if every needed fact is present.
- `next_query`: a keyword query for that missing fact, anchored on the exact entity name as it appears in the
  passages, plus the property being sought. Leave it empty when nothing is missing. Never repeat a query whose
  results are already listed.
- `sufficient`: `yes` if the passages already contain every fact needed to answer, otherwise `no`."""

ANSWER_INSTRUCTIONS = """Answer the question from the passages with the shortest possible answer.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

SLATE_INSTRUCTIONS = """A short answer has already been drafted for this question from these passages.
Multi-step questions are usually lost to a NEAR MISS: a passage states some closely related thing -- the
person on the other side of the same relation, the neighbouring work by the same author, the containing
organisation, the year of the wrong event -- and that thing gets returned instead of the one asked for.

Your job is not to answer. It is to lay out, side by side, every answer these passages could plausibly be read
as supporting, so that the choice between them can be made deliberately.

Write one candidate per line and nothing else, in this exact form:

    <answer> :: <passage number> :: <sentence copied verbatim from that passage>

Rules:
- the first line must be the drafted answer, with the passage and sentence that support it;
- then add the rival readings: the things a careless reader of these same passages would return instead.
  Whenever the passages name any other thing of the SAME KIND as the drafted answer -- another person where
  the draft is a person, another date, another work title, another number, another place -- that thing is a
  rival and must get a line, even when the draft looks right to you. In particular list the thing on the other
  side of the relation the question is about (the one who came after where the draft came before, the container
  where the draft is the contained), and the intermediate thing the question passes through on the way to its
  answer. Write a single line only when no passage names any other thing of the drafted answer's kind.
- `<answer>` must be a bare short answer copied from the passage wording: an entity, name, title, date,
  number or verdict. Never a sentence, never an explanation.
- the sentence must be copied from the passage you cite, not paraphrased.
- at most 4 lines. If the passages support no rival reading at all, write only the first line.

Also produce `demanded`: the single thing the question finally asks for, restated as a short noun phrase,
including the direction of the relation when the question involves one thing acting on another (for example
`the person who held the post BEFORE him, not the one who came after`). Never answer it."""

DISCRIMINATE_INSTRUCTIONS = """Choose between candidate answers that were each read off the same passages.

`demanded` restates what the question asks for. `candidates` are numbered; each shows an answer, the article
it came from, and the sentence in that article which states it.

In `reasoning`, work out for each candidate what role its sentence actually gives it, and check that role
against `demanded` -- in particular the DIRECTION of any relation (who replaced whom, which came first, which
work contains which) and whether the sentence is about the thing the question asks about or merely about
something adjacent to it.

Then output `choice`: the number of the single candidate whose sentence states the thing demanded.

Candidate 1 is the drafted answer. Choose a different candidate only when its sentence clearly states the
demanded thing and candidate 1's sentence clearly does not; if the evidence is balanced, or if two candidates
are two wordings of the same thing, choose 1."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3          # router passes (each may add one retrieval)
    MAX_RETRIEVALS = 6
    MAX_CANDIDATES = 4      # size of the contrastive slate
    SLATE_EXTRA = 6         # pooled-but-unselected passages the slate may also read
    QUOTE_CHARS = 300       # evidence sentence shown per candidate
    POOL_SNIPPET = 380      # chars of each passage shown to the router

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.route = Module(
            inputs=["question", "passages"],
            outputs=["useful", "missing", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={"answer": "the bare short answer"},
        )
        self.slate = Module(
            inputs=["question", "draft", "passages"],
            outputs=["demanded", "candidates"],
            instructions=SLATE_INSTRUCTIONS,
            descs={
                "draft": "the answer drafted so far",
                "demanded": "the thing the question asks for, restated in a few words",
                "passages": "numbered search results, each `Title | abstract`",
                "candidates": "one `answer :: passage number :: verbatim sentence` per line",
            },
        )
        self.discriminate = Module(
            inputs=["question", "demanded", "candidates"],
            outputs=["choice"],
            instructions=DISCRIMINATE_INSTRUCTIONS,
            descs={
                "demanded": "what the question asks for",
                "candidates": "numbered candidate answers with their source article and evidence sentence",
                "choice": "the number of the chosen candidate",
            },
        )
        self.commit = Module(
            inputs=["question", "passages"],
            outputs=["answer"],
            instructions=COMMIT_INSTRUCTIONS,
            descs={"answer": "the bare short answer, never an abstention"},
        )

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("«»").strip()
        return " ".join(q.split())[:300]

    def _parse_queries(self, raw: str, question: str) -> list[str]:
        queries: list[str] = []
        for line in str(raw or "").splitlines():
            q = self._clean_query(line)
            if len(q) < 3 or q.lower() in {"none", "queries"}:
                continue
            if q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
            if len(queries) == 3:
                break
        if question not in queries:
            queries.insert(0, question)      # the plain question is a reliable fallback query
        return queries[:4]

    @staticmethod
    def _parse_indices(raw: str, n: int) -> list[int]:
        picks: list[int] = []
        for tok in _INDEX.findall(str(raw or "")):
            i = int(tok)
            if 1 <= i <= n and i not in picks:
                picks.append(i)
        return picks

    @staticmethod
    def _tidy(text: str, question: str) -> str:
        ans = _MARKUP.sub("", str(text or "")).strip().split("\n")[0].strip()
        ans = _LEADING_FILLER.sub("", ans).strip()
        ans = ans.strip('"').strip("'").strip("«»").strip()
        m = _YESNO_PREFIX.match(ans)
        if m and (_BOOL_QUESTION.match(question or "") or len(ans.split()) > 1):
            return m.group(1).lower()
        while ans.endswith((".", ",", ";", ":")):
            ans = ans[:-1].strip()
        return ans

    # -- slate handling ------------------------------------------------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(_NONWORD.sub(" ", str(text or "").lower()).split())

    @staticmethod
    def _title_of(passage: str) -> str:
        return str(passage).split(" | ")[0].strip()

    @staticmethod
    def _body_of(passage: str) -> str:
        parts = str(passage).split(" | ", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    def _grounded_in(self, answer: str, evidence: list[str]) -> int:
        """Index of the first passage whose text contains the answer string, else -1."""
        want = self._norm(answer)
        if not want:
            return -1
        for i, passage in enumerate(evidence):
            if want in self._norm(passage):
                return i
        return -1

    def _repair_sentence(self, sentence: str, passage: str, answer: str) -> str:
        """Keep the quote only if the passage really contains it; otherwise quote the passage itself."""
        body = self._body_of(passage)
        norm_body = self._norm(body)
        head = " ".join(self._norm(sentence).split()[:6])
        if head and head in norm_body:
            return " ".join(str(sentence).split())[: self.QUOTE_CHARS]
        want = self._norm(answer)
        for sent in _SENT_SPLIT.split(body):
            if want and want in self._norm(sent):
                return " ".join(sent.split())[: self.QUOTE_CHARS]
        return " ".join(body.split())[: self.QUOTE_CHARS]

    def _parse_slate(self, raw: str, draft: str, question: str, evidence: list[str]) -> list[dict[str, str]]:
        """Lines -> validated candidates: answer must occur in the evidence, title comes from the pool."""
        out: list[dict[str, str]] = []
        seen_answers: set[str] = set()
        for line in str(raw or "").splitlines():
            line = _MARKUP.sub("", line).strip()
            line = _BULLET.sub("", line).strip()
            if "::" not in line:
                continue
            parts = [x.strip() for x in line.split("::")]
            answer = self._tidy(parts[0], question)
            if not answer or _NON_ANSWER.match(answer) or len(answer.split()) > 12:
                continue
            key = self._norm(answer)
            if not key or key in seen_answers:
                continue
            idx = -1
            for tok in _INDEX.findall(parts[1] if len(parts) > 1 else ""):
                i = int(tok)
                if 1 <= i <= len(evidence):
                    idx = i - 1
                    break
            if idx < 0 or key not in self._norm(evidence[idx]):
                idx = self._grounded_in(answer, evidence)      # cited the wrong passage: relocate it
            if idx < 0:
                continue                                        # ungrounded string: drop the candidate
            sentence = self._repair_sentence(parts[2] if len(parts) > 2 else "", evidence[idx], answer)
            seen_answers.add(key)
            out.append({"answer": answer, "title": self._title_of(evidence[idx]), "quote": sentence})
            if len(out) >= self.MAX_CANDIDATES:
                break
        # the draft is always candidate 1, whether or not the slate remembered to list it
        dkey = self._norm(draft)
        if dkey and dkey not in {self._norm(c["answer"]) for c in out}:
            idx = self._grounded_in(draft, evidence)
            head = {
                "answer": draft,
                "title": self._title_of(evidence[idx]) if idx >= 0 else "(drafted)",
                "quote": self._repair_sentence("", evidence[idx], draft) if idx >= 0 else "",
            }
            out.insert(0, head)
        elif dkey:
            out.sort(key=lambda c: self._norm(c["answer"]) != dkey)
        return out[: self.MAX_CANDIDATES]

    @staticmethod
    def _render_candidates(cands: list[dict[str, str]]) -> list[str]:
        return [f"{c['answer']}  --  from «{c['title']}»: {c['quote']}" for c in cands]

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str) -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=self.K):
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(passage)
                added += 1
            return added

        # 1. decomposed opening sweep
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q)

        # 2-4. route over the pool, following up on what is missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            shown = [f"{p[: self.POOL_SNIPPET]}" for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[:8]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            next_query = routed.get("next_query", "")
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(next_query):
                break     # follow-up query brought nothing new: stop paying for rounds

        evidence = selected or pool[:8]
        ctx.note("pool routed", n_pool=len(pool), n_selected=len(selected), retrievals=retrievals, missing=missing[:200])

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        # 5-7. contrastive stage: enumerate the rival readings of the SAME passages, then discriminate.
        # Skipped entirely when the draft is unusable (the forced commit below owns that case).
        if final and not _NON_ANSWER.match(final):
            # The slate ranges over more than the router's pick: on 11 of the parent's wrong items the gold
            # answer sits in a POOLED passage the router dropped, so a rival read only off the selection could
            # never name it. The draft call above still sees exactly what the parent's does.
            wider = evidence + [pp for pp in pool if pp not in evidence][: self.SLATE_EXTRA]
            slated = self.slate(
                ctx.llm, tag="slate", question=question, draft=final, passages=wider
            )
            cands = self._parse_slate(slated.get("candidates", ""), final, question, wider)
            demanded = " ".join(_MARKUP.sub("", str(slated.get("demanded", ""))).split())[:300]
            ctx.note("slate", n_candidates=len(cands), answers=[c["answer"][:60] for c in cands])
            if len(cands) >= 2:
                picked = self.discriminate(
                    ctx.llm, tag="discriminate", question=question,
                    demanded=demanded or "the thing the question asks for",
                    candidates=self._render_candidates(cands),
                ).get("choice", "")
                nums = self._parse_indices(picked, len(cands))
                if nums:
                    chosen = self._tidy(cands[nums[0] - 1]["answer"], question)
                    if chosen and not _NON_ANSWER.match(chosen):
                        if chosen != final:
                            ctx.note("discriminated", before=final[:120], after=chosen[:120])
                        final = chosen

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- and show the WHOLE pool, since selection may have dropped the right passage.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "queries": queries, "missing": missing}
