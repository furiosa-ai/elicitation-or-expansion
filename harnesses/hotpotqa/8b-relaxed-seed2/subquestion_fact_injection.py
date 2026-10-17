"""Missing facts are delegated to an isolated sub-solver and injected as facts, not as passages.

Three iterations in a row bought retrieval recall and lost accuracy: link expansion (recall .807 ->
.818, accuracy -0.7), deep scanning (recall -> .825, accuracy -9.7) and article-space querying
(recall -> .812, accuracy -4.3).  The common factor is not the retrieval -- it is that everything each
of them found was poured back into the ONE context the final answer is written from.  The parent's own
follow-up hop does the same thing: it re-retrieves and re-sorts the pool, so a successful second hop
and a failed one differ only in how much noise the answerer must now read past.

This inverts that.  Retrieval for the missing fact still happens, but its passages never reach the
answering call:

  * triage states what is missing as a standalone QUESTION (`missing_question`) instead of as a
    keyword query -- it must name its entity, because it will be asked on its own;
  * that question is resolved by a separate call with its OWN small context: passages retrieved for it
    and nothing else, no original question, no original pool.  It answers in one short clause or says
    `unknown`, so it cannot bluff its way into the main context;
  * only the resolved clause is injected, as one line of `established_facts`.  The evidence list handed
    to the answerer is exactly the passages triage selected out of the first-hop pool -- the same list,
    in the same order, at the same size, as if no second hop had happened.

So the answering context grows by one sentence rather than by four passages, which is the falsifiable
part: if the last three iterations lost accuracy because of context damage rather than because the
extra retrieval was useless, fact-level injection keeps the gain and drops the damage.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

NUM_RE = re.compile(r"\d+")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
AFFIRM = re.compile(r"^\s*(?:yes|true|sufficient|enough)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "is", "was", "were", "are", "did", "does", "do",
    "has", "have", "had", "that", "this", "with", "by", "from", "as", "it", "its", "his", "her",
    "their", "he", "she", "they", "both", "also", "than", "then", "there", "these", "those", "name",
    "named", "known", "first", "one", "two", "were", "born", "film", "movie", "song", "album", "band",
}


def _split_lines(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = BULLET.sub("", raw).strip().strip('"').strip()
        if len(line) >= 3 and not line.startswith("[["):
            out.append(line[:220])
        if len(out) >= limit:
            break
    return out


def _title_of(passage: str) -> str:
    return passage.split(" | ")[0].strip()


def _lead(passage: str, chars: int) -> str:
    title, _, body = passage.partition(" | ")
    return f"{title.strip()} | {body.strip()[:chars]}"


def _family(question: str) -> str:
    """Expected answer shape, derived from the question alone (identical in prepare and run)."""
    s = (question or "").strip().lower()
    if re.match(r"^(are|is|was|were|do|does|did|has|have|had|can|could|will|would|would)\b", s):
        return "yesno"
    if re.search(r"\bhow (many|much|long|old|tall)\b", s):
        return "count"
    if re.search(r"^when\b|\b(what|which) (year|date)\b|\bin what year\b|\bon what date\b", s):
        return "date"
    if re.search(r"\b(which|who|what)\b[^?]*\b or \b", s):
        return "alternative"
    if re.search(r"^where\b|\bwhat (country|city|state|nationality|county)\b|\bwhich (country|city|state)\b", s):
        return "place"
    if re.search(r"^(who|whom|whose)\b|\bwhich (actor|actress|director|singer|player|author|writer|artist|person)\b", s):
        return "person"
    return "entity"


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in STOP}


def _answer_sentence(passage: str, answer: str) -> str:
    """The sentence of `passage` that carries `answer`, deterministically."""
    _, _, body = passage.partition(" | ")
    body = body.strip()
    norm_ans = normalize_text(answer)
    for sent in SENT_SPLIT.split(body):
        if norm_ans and norm_ans in normalize_text(sent):
            return sent.strip()[:300]
    return body.split(". ")[0].strip()[:300]



def _clean_fact(text: str) -> str:
    """One short clause, or nothing at all if the sub-solver did not find the fact."""
    line = (text or "").strip().split("\n")[0].strip().strip('"\u00ab\u00bb*_`').strip()
    if not line or len(line.split()) < 2 or len(line) > 220:
        return ""
    if re.match(r"^\s*(unknown|none|n/?a|not (stated|specified|mentioned|found))\b", line, re.IGNORECASE):
        return ""
    return line


class Harness(BaseHarness):
    K_PLAN = 8
    K_FOLLOW = 8
    K_SUB = 8             # depth of the sub-question's own retrieval
    SUB_PASSAGES = 5      # passages the sub-solver reads (its context, not the answerer's)
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    N_DEMOS = 3           # demonstrations attached to the answering call
    DEMO_PASSAGES = 4     # passages shown inside one demonstration
    DEMO_CHARS = 300      # per-passage truncation inside a demonstration
    BANK_PER_FAMILY = 6   # demonstrations retained per family

    PLAN_INSTR = (
        "Break the question into the separate facts that must be looked up in an encyclopaedia.\n"
        "Write `queries` as one short keyword search query per line (2 or 3 lines, no numbering). "
        "Each query must stand on its own: name the entity it is about explicitly instead of using "
        "a pronoun or a description that depends on another query.\n"
        "If one fact can only be looked up after another is known, still write the query for the "
        "part that is already named."
    )
    TRIAGE_INSTR = (
        "You are assembling the evidence needed to answer the question from a pool of encyclopaedia "
        "leads, shown as `id: title | beginning of the article`.\n"
        "`relevant_ids` : the ids worth reading in full, most useful first, at most 8, comma separated.\n"
        "`linking_fact` : the fact the pool establishes that connects the parts of the question "
        "(for example the intermediate entity the question refers to indirectly); say `unknown` if none.\n"
        "`sufficient`   : `yes` only if the pool already contains every fact needed to answer; "
        "otherwise `no`.\n"
        "`missing_question` : if not sufficient, the ONE fact still missing, written as a complete "
        "standalone question that names its own entity explicitly (someone who has never seen the "
        "original question must be able to answer it); otherwise leave it empty."
    )
    SUB_INSTR = (
        "Answer this single factual question from the passages below.\n"
        "Write `fact` as one short clause stating the fact, naming the entity it is about, for example "
        "`X was directed by Y` or `X is located in Y`.\n"
        "If the passages do not state it, write `fact` as exactly `unknown`. Never guess."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages, the way the worked examples do.\n"
        "Write `reasoning` as the sentence of the evidence that settles the question, quoted, and "
        "nothing more.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "Copy that span whole, exactly as the evidence writes it: do not cut a full date down to its "
        "year, a full name down to a surname, or a title down to part of itself.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim.\n"
        "`established_facts` holds facts looked up separately because the evidence passages do not "
        "state them. Treat them as true, but take the wording of the answer from the evidence "
        "whenever the evidence contains it."
    )

    def __init__(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "missing_question"],
            instructions=self.TRIAGE_INSTR,
        )
        self.sub = Module(inputs=["sub_question", "passages"], outputs=["fact"], instructions=self.SUB_INSTR)
        self.bank: dict[str, list[dict[str, Any]]] = {}

    # -- prepare: build the demonstration bank from labelled train items ----------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        bank: dict[str, list[dict[str, Any]]] = {}
        for it in train_items:
            question = str(it.get("question") or "")
            answer = str(it.get("answer") or "").strip()
            gold_titles = [str(t) for t in (it.get("gold_titles") or [])]
            if not question or not answer or not gold_titles:
                continue
            fam = _family(question)
            if len(bank.get(fam, [])) >= self.BANK_PER_FAMILY:
                continue
            try:
                with ctx.open_tools(it) as tools:
                    near = tools.retrieve(question, k=6)
                    found: dict[str, str] = {}
                    for t in gold_titles[:2]:
                        for p in list(near) + tools.retrieve(t, k=4):
                            if _title_of(p).strip().lower() == t.strip().lower():
                                found[t] = p
                                break
            except Exception:  # a retrieval hiccup must not lose the whole bank
                continue
            if len(found) < len(gold_titles[:2]):
                continue
            golds = [found[t] for t in gold_titles[:2] if t in found]
            is_binary = normalize_text(answer) in ("yes", "no")
            carrier = None
            for p in golds:
                if normalize_text(answer) and normalize_text(answer) in normalize_text(p):
                    carrier = p
                    break
            if carrier is None and not is_binary:
                continue  # not readable off its own evidence -> a misleading demonstration
            distractors = [p for p in near if _title_of(p).lower() not in {_title_of(g).lower() for g in golds}]

            # interleave so the answer is never in the same slot twice
            evidence: list[str] = []
            order = [distractors[:1], golds[:1], distractors[1:2], golds[1:2]]
            if len(bank.get(fam, [])) % 2:
                order = [golds[:1], distractors[:1], golds[1:2], distractors[1:2]]
            for chunk in order:
                evidence.extend(chunk)
            evidence = [_lead(p, self.DEMO_CHARS) for p in evidence[: self.DEMO_PASSAGES]]

            if is_binary:
                quotes = " ".join(f'«{_title_of(p)}» says: "{_answer_sentence(p, answer)}"' for p in golds)
                reasoning = f"{quotes} Comparing the two, the answer is {answer}."
            else:
                reasoning = (
                    f'«{_title_of(carrier)}» says: "{_answer_sentence(carrier, answer)}" '
                    f"That names the {fam if fam != 'entity' else 'thing'} the question asks for."
                )
            link = next(
                (
                    _title_of(g)
                    for g in golds
                    if not (_content_words(_title_of(g)) & _content_words(question))
                ),
                "unknown",
            )
            bank.setdefault(fam, []).append(
                {
                    "question": question,
                    "evidence": evidence,
                    "linking_fact": link,
                    "established_facts": "none",
                    "reasoning": reasoning[:600],
                    "answer": answer,
                    "keys": sorted(_content_words(question))[:24],
                }
            )
        self.bank = bank
        ctx.note("demo bank built", sizes={k: len(v) for k, v in bank.items()})

    def get_state(self) -> str:
        return json.dumps({"bank": self.bank})

    def set_state(self, state: str) -> None:
        try:
            self.bank = (json.loads(state) or {}).get("bank") or {}
        except Exception:
            self.bank = {}

    # -- demo selection: family first, then nearest neighbour + answer-shape coverage ----------
    @staticmethod
    def _shape(answer: str) -> str:
        """Coarse shape of a labelled answer, used to keep a demo set from teaching one granularity."""
        norm = normalize_text(answer)
        if norm in ("yes", "no"):
            return norm  # balanced on the label itself, so demos carry no prior
        return str(min(len(norm.split()), 4))

    def _demos(self, question: str) -> list[dict[str, Any]]:
        if not self.bank:
            return []
        fam = _family(question)
        keys = _content_words(question)
        def by_sim(ds: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(ds, key=lambda d: (-len(keys & set(d.get("keys") or [])), d.get("question", "")))

        def by_fullest(ds: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # a demonstration can only ever teach the model to answer with LESS than the evidence
            # holds, so after the nearest neighbour the fullest-form answers go in first.
            return sorted(ds, key=lambda d: (-len(normalize_text(d.get("answer", "")).split()), d.get("question", "")))

        # the family pool is drained first: a demonstration of the wrong answer shape is worse than
        # a less similar one of the right shape.
        picked: list[dict[str, Any]] = []
        shapes: set[str] = set()
        fam_pool = self.bank.get(fam, [])
        rest = [d for f, ds in sorted(self.bank.items()) if f != fam for d in ds]
        for pool, orders in ((fam_pool, (by_sim, by_fullest)), (rest, (by_sim, by_fullest))):
            for order, want_new in ((orders[0], True), (orders[1], True), (orders[1], False)):
                for d in order(pool):
                    if len(picked) >= self.N_DEMOS:
                        break
                    if any(d is q for q in picked):
                        continue
                    sh = self._shape(d.get("answer", ""))
                    # nearest neighbour always gets in; after that prefer an answer shape the set
                    # does not have yet, so three demos never all answer with a bare year.
                    if not picked or not want_new or sh not in shapes:
                        picked.append(d)
                        shapes.add(sh)
                        if order is by_sim:
                            break  # nearest neighbour only; coverage is filled fullest-form first
        return [
            {k: d[k] for k in ("question", "evidence", "linking_fact", "established_facts",
                              "reasoning", "answer") if k in d}
            for d in picked
        ]

    # -- pool helpers -------------------------------------------------------------------------
    @staticmethod
    def _add(pool: list[str], seen: set[str], passages: list[str]) -> None:
        for p in passages:
            key = _title_of(p).lower()
            if key and key not in seen:
                seen.add(key)
                pool.append(p)

    @staticmethod
    def _parse_ids(text: str, n: int) -> list[int]:
        ids: list[int] = []
        for tok in NUM_RE.findall(text or ""):
            i = int(tok)
            if 1 <= i <= n and i not in ids:
                ids.append(i)
        return ids

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()

        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )
        self._add(pool, seen, ctx.tools.retrieve(question, k=self.K_PLAN))
        for q in queries:
            self._add(pool, seen, ctx.tools.retrieve(q, k=self.K_PLAN))

        shown = pool[: self.POOL_CAP]
        table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
        out = self.triage(ctx.llm, tag="triage", question=question, pool=table)
        ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
            range(1, min(len(shown), self.SELECT_CAP) + 1)
        )
        linking_fact = (out.get("linking_fact", "") or "").strip()
        if linking_fact.lower() in ("unknown", "none", "n/a"):
            linking_fact = ""

        # -- delegation: the missing fact is resolved away from the main context ---------------
        established: list[str] = []
        followups: list[str] = []
        if not AFFIRM.match(out.get("sufficient", "")):
            asked = _split_lines(out.get("missing_question", ""), 1)
            sub_q = asked[0] if asked else ""
            if len(sub_q.split()) >= 3:
                followups.append(sub_q)
                side: list[str] = []
                side_seen: set[str] = set()
                self._add(side, side_seen, ctx.tools.retrieve(sub_q, k=self.K_SUB))
                if linking_fact:
                    self._add(side, side_seen, ctx.tools.retrieve(linking_fact[:200], k=self.K_SUB))
                if side:
                    fact = self.sub(
                        ctx.llm,
                        tag="subanswer",
                        sub_question=sub_q,
                        passages=[_lead(p, 300) for p in side[: self.SUB_PASSAGES]],
                    ).get("fact", "")
                    fact = _clean_fact(fact)
                    if fact:
                        established.append(fact)
                        ctx.note("sub-question resolved", sub_question=sub_q, fact=fact)
                    else:
                        ctx.note("sub-question unresolved", sub_question=sub_q)

        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]

        demos = self._demos(question)
        answerer = Module(  # per-item module: demos differ per question and items run concurrently
            inputs=["question", "evidence", "linking_fact", "established_facts"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
            demos=demos,
        )
        ans = answerer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
            established_facts="\n".join(established) if established else "none",
        ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        return {
            "answer": ans,
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "family": _family(question),
            "n_demos": len(demos),
            "n_selected": len(evidence),
            "established_facts": established,
        }
