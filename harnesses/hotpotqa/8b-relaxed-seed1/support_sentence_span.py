"""Support-sentence grounding: the sentence that licenses the answer also fixes its wording.

On the parent router's 206 items where every needed document WAS retrieved, 38 are still wrong and 18 of
those overlap the target string without matching it -- the substance is right and the surface form is not.
A previous attempt read this as "the model answers too briefly" and had it choose among enumerated longer
surface forms; that fixed 11 items and broke 38, because expanding a name is right only sometimes: the
same question type wants `Ronald Eugene Rivera` here and a bare `Elon Musk` there, so no length preference
can be correct.

What separates the two cases is not the name, it is WHICH sentence states the fact being asked about.  A
person's own article opens by writing their name out in full, whereas a sentence that mentions them while
describing something else uses the everyday short form -- and the target is a span of the sentence that
answers the question.  So this drops the free-form answer stage and replaces it with a stage that has to
commit to its evidence at SENTENCE granularity: the selected passages are split into numbered
`Title | sentence` lines, the model names the one line that states the answer and copies the answer out of
that line, and the copy is then verified deterministically as a literal substring of the line it cited
(re-sliced from the source so casing and accents come from the document, not from the model).  Wording is
therefore never chosen by preference; it is inherited from the cited sentence.

Retrieval is byte-identical to the parent -- same planner queries, same pool, same router, same selection --
so any delta is attributable to the answer stage.
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
_YESNO_ONLY = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_MARKUP = re.compile(r"[*_`]+")
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)

# -- sentence splitting -------------------------------------------------------------------------
_BREAK = re.compile(r"(?<=[.!?])[\"'’)\]]*\s+")
_ABBR = re.compile(
    r"(?:^|[\s(\"'])(?:mr|mrs|ms|dr|prof|st|mt|sr|jr|rev|gen|col|lt|sgt|capt|hon|fr|rep|sen|gov|pres"
    r"|ave|rd|no|vs|etc|al|inc|ltd|co|corp|dept|est|approx|ca|cf|op|ed|vol|pp|fig"
    r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.$",
    re.IGNORECASE,
)
_INITIAL = re.compile(r"(?:^|[\s(\"'])[A-Za-z]\.$")
_WS = re.compile(r"\s+")
# a copied span that runs on into a predicate is a sentence fragment, not an answer
_CLAUSE = re.compile(
    r"\s*(?:,\s*)?\b(?:is|was|are|were|be|been|being|has|have|had|does|did|do|will|would|can|could"
    r"|which|who|whom|whose|that|where|when|while|because|although|though|however"
    r"|operating|located|known|born|founded|released|published|starring|featuring|including)\b\s.*$",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    """Split an abstract into sentences, keeping initials and common abbreviations intact."""
    body = _WS.sub(" ", str(text or "")).strip()
    if not body:
        return []
    out: list[str] = []
    start = 0
    for m in _BREAK.finditer(body):
        head = body[start:m.start()].rstrip()
        if not head:
            continue
        if _ABBR.search(head) or _INITIAL.search(head):
            continue                      # not a real sentence boundary
        out.append(head)
        start = m.end()
    tail = body[start:].strip()
    if tail:
        out.append(tail)
    return [x for x in out if len(x) >= 3]


def _trim_clause(span: str) -> str:
    """Cut a verbatim span back to the answer itself when a predicate was copied along with it."""
    head = _CLAUSE.sub("", str(span or "")).strip().strip(" ,;:-")
    return head if len(head) >= 2 else str(span or "").strip()


_NAME_TOK = re.compile(r"[^\s]+")
_HONORIFIC = {"sir", "dame", "lord", "lady", "dr", "mr", "mrs", "ms", "prof", "professor", "saint",
              "st", "rev", "gen", "col", "capt", "sgt", "lt", "president", "senator", "governor",
              "baron", "baroness", "earl", "duke", "count", "king", "queen", "prince", "princess"}
_NAME_PARTICLE = {"of", "the", "de", "del", "della", "der", "den", "des", "di", "do", "du", "da",
                  "la", "le", "van", "von", "y", "el", "bin", "ibn", "al"}


def _name_runs(sentence: str) -> list[list[str]]:
    """Maximal runs of capitalised tokens (particles allowed inside) occurring in a sentence."""
    runs: list[list[str]] = []
    run: list[str] = []
    toks = _NAME_TOK.findall(str(sentence or ""))
    for i, tok in enumerate(toks):
        bare = tok.strip("(),;:.\"'’")
        if not bare:
            if run:
                runs.append(run)
                run = []
            continue
        low = bare.lower()
        if bare[:1].isupper() or bare[:1].isdigit():
            run.append(bare)
            if tok.rstrip("\"'’").endswith((",", ";", ":", ")", ".")):
                runs.append(run)          # punctuation ends the name; never span a comma
                run = []
            continue
        nxt = toks[i + 1].strip("(),;:.\"'’") if i + 1 < len(toks) else ""
        if run and low in _NAME_PARTICLE and nxt[:1].isupper():
            run.append(bare)
            continue
        if run:
            runs.append(run)
            run = []
        if tok.endswith((",", ".", ";", ":", ")")):
            continue
    if run:
        runs.append(run)
    return runs


def _complete_name(draft: str, sentence: str) -> str:
    """Restore words the cited sentence writes inside the drafted name (e.g. a middle name).

    Deterministic and strictly gated: the sentence must contain one capitalised run that BEGINS with the
    draft's first word, ENDS with the draft's last word, and contains every drafted word in order. Any
    other shape -- a longer entity the draft merely sits inside, an appended region, a coordination --
    leaves the draft untouched, so this can only ever fill a gap the sentence itself spells out.
    """
    words = [w for w in str(draft or "").split() if w]
    if not 2 <= len(words) <= 4 or any(not w[:1].isupper() for w in (words[0], words[-1])):
        return ""
    for run in _name_runs(sentence):
        while len(run) > 1 and run[0].lower().strip(".") in _HONORIFIC:
            run = run[1:]          # an honorific is not part of the name
        if len(run) <= len(words) or len(run) > len(words) + 3:
            continue
        if run[0].lower() != words[0].lower() or run[-1].lower() != words[-1].lower():
            continue
        pos = 0
        for w in words:
            while pos < len(run) and run[pos].lower() != w.lower():
                pos += 1
            if pos == len(run):
                break
            pos += 1
        else:
            return " ".join(run)
    return ""


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

CITE_INSTRUCTIONS = """A short answer to this question has already been drafted from these documents. Your
only job is to point at the sentence that states it.

The evidence lines are `Title | sentence` pairs. In `reasoning`, check that the draft really is what the
question asks for, then find the ONE line whose sentence actually states that fact -- the sentence a reader
would cite as the source of the draft, not merely a line that mentions the same entity.

- `support`: the number of that single line, and nothing else.
- `verdict`: `ok` if that sentence supports the draft, or `wrong` if no line does.

Do not rewrite, shorten or lengthen the draft; you are only citing its source."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380
    MAX_UNITS = 44          # numbered sentence lines shown to the span stage
    SENT_CHARS = 320

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
        self.cite = Module(
            inputs=["question", "draft", "evidence"],
            outputs=["support", "verdict"],
            instructions=CITE_INSTRUCTIONS,
            descs={
                "draft": "the short answer already drafted",
                "evidence": "numbered evidence lines, each `Title | sentence`",
                "support": "the number of the one line that states the drafted fact",
                "verdict": "ok or wrong",
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
            queries.insert(0, question)
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

    def _units(self, passages: list[str]) -> list[str]:
        """`Title | sentence` lines; the title stays copyable, the sentences carry the wording."""
        units: list[str] = []
        for passage in passages:
            raw = str(passage)
            title, _, body = raw.partition(" | ")
            title = _WS.sub(" ", title).strip() or raw[:60]
            for sent in _sentences(body) or [_WS.sub(" ", raw)[: self.SENT_CHARS]]:
                units.append(f"{title} | {sent[: self.SENT_CHARS]}")
                if len(units) >= self.MAX_UNITS:
                    return units
        return units

    @staticmethod
    def _locate(span: str, units: list[str], preferred: int) -> str:
        """Re-slice `span` out of the cited line so the source's own characters win; '' if absent."""
        needle = _WS.sub(" ", str(span or "")).strip().strip('"').strip("'").strip("«»")
        needle = needle.strip(" ,;:")
        if not needle:
            return ""
        order = ([preferred] if 1 <= preferred <= len(units) else []) + [
            i for i in range(1, len(units) + 1) if i != preferred
        ]
        low = needle.lower()
        for i in order:
            hay = units[i - 1]
            pos = hay.lower().find(low)
            if pos >= 0:
                return hay[pos : pos + len(needle)]
        return ""

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

        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q)

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
                break

        evidence = selected or pool[:8]

        # 1. draft exactly as the parent does -- same prompt, same passages, same tidying
        drafted = self._tidy(
            self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", ""),
            question,
        )
        drafted = drafted if _YESNO_ONLY.match(drafted or "") else _trim_clause(drafted)
        final = drafted
        grounded = "draft"

        # 2. cite the sentence that states it, then let THAT sentence -- not a preference for longer or
        #    shorter names -- decide the wording.
        units = self._units(evidence)
        if units and drafted and not _YESNO_ONLY.match(drafted):
            cited = self.cite(ctx.llm, tag="cite", question=question, draft=drafted, evidence=units)
            picks = self._parse_indices(cited.get("support", ""), len(units))
            if picks:
                sentence = units[picks[0] - 1].partition(" | ")[2]
                filled = _complete_name(drafted, sentence)
                if filled and filled.lower() != drafted.lower():
                    final = filled
                    grounded = "completed"
                else:
                    grounded = "confirmed"
            ctx.note(
                "citation",
                support=picks[:1],
                verdict=str(cited.get("verdict", ""))[:20],
                draft=drafted[:120],
                final=str(final)[:120],
                grounded=grounded,
            )

        ctx.note("pool routed", n_pool=len(pool), n_selected=len(selected), n_units=len(units), retrievals=retrievals, missing=missing[:200])

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=drafted[:120], after=str(final)[:120])
            grounded = "commit"

        return {"answer": final or drafted, "n_pool": len(pool), "grounded": grounded, "missing": missing}
