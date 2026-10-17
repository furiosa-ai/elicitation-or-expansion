"""Claim-table context: broad retrieval, then a FIXED-SIZE answering context of verbatim quotes.

Three retrieval candidates in a row (iters 2-4) raised gold-document recall and still lost accuracy, and the
item-level diagnosis is identical for all three: the extra recall converts (on the 12-14 items whose coverage
improved, accuracy went 5-6 -> 8), but on the ~195 items where the parent ALREADY had both documents, accuracy
fell 3-4 points -- 6-7 questions that were solved before break when the pool grows or is reordered, because the
final call reads a pool-shaped context. Retrieval breadth and answering context are coupled, and the coupling
costs more than the breadth is worth.

This decouples them. Retrieval goes as wide as the budget allows (the parent's decomposed sweep plus one bare
lookup for every name the question itself prints, mined deterministically), but the pool is never shown to the
answerer. Instead one distillation call converts the whole pool into a CLAIM TABLE: at most 9 lines, each a
VERBATIM quoted clause copied out of one numbered passage, with the passage's title re-attached
deterministically from the pool so the title can never be garbled. The answerer sees the table plus the 3 raw
passages the table cites most -- a context whose size and ordering are fixed no matter how large the pool grew.

Quotes, not summaries: the parent's advantage over the seed was that source wording survived to the answer, so
distillation copies clauses instead of paraphrasing them, and the top cited passages are pinned raw as a
backstop against the distiller dropping the decisive detail.

Falsifiable: if the both-covered regression really is context-shaped, this keeps the breadth gain without the
collateral and lands above 68.3%. If instead the regression was caused by the retrieved passages themselves
being distractors, a fixed-size context will not help and accuracy will be flat or lower.
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
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)
# a distilled line: `<passage number> :: <verbatim quote>`
_CLAIM_LINE = re.compile(r"^\s*\[?(\d{1,3})\]?\s*(?:::|\||-|\u2013)\s*(.+?)\s*$")


_QUOTED = re.compile(r"[\"“”«»]([^\"“”«»]{3,80})[\"“”«»]")
_TOKEN = re.compile(r"[A-Za-z0-9][\w.'&/-]*|[^\sA-Za-z0-9]")
# a token that can sit INSIDE a name without ending the capitalised run
_PARTICLE = {
    "of", "the", "in", "on", "at", "for", "to", "a", "an", "de", "del", "della", "der",
    "den", "des", "di", "do", "dos", "du", "da", "la", "le", "les", "van", "von", "y", "el",
    "bin", "ibn", "al", "ter", "op", "mac",
}
# words that are never an entity on their own even when capitalised
_STOP_ALONE = {
    "which", "what", "who", "whom", "whose", "when", "where", "why", "how", "the", "a", "an",
    "this", "that", "these", "those", "both", "either", "neither", "and", "or", "but", "if",
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could",
    "will", "would", "should", "in", "on", "at", "of", "for", "to", "by", "from", "with",
    "between", "besides", "along", "after", "before", "during", "there", "it", "he", "she",
    "they", "his", "her", "their", "its", "one", "two", "name", "many", "much", "other",
    "another", "same", "different", "american", "english", "british", "born", "also", "known",
}
_ALPHA = re.compile(r"[A-Za-z]")


def _join(run: list[str]) -> str:
    return " ".join(run).replace(" ' ", " '").replace(" ’ ", " ’")


def _mentions(question: str) -> list[str]:
    """Every span of the question that looks like the title of an article, longest first."""
    text = " ".join(str(question or "").split())
    found: list[str] = []

    for m in _QUOTED.finditer(text):
        cand = m.group(1).strip(" ,;:.-")
        if len(cand) >= 3 and _ALPHA.search(cand):
            found.append(cand)

    tokens = _TOKEN.findall(text)
    run: list[str] = []
    skip = 0
    for idx, tok in enumerate(tokens):
        if skip:
            skip -= 1
            continue
        low = tok.lower().strip(".")
        is_cap = bool(tok[:1].isupper()) or (tok[:1].isdigit() and len(tok) >= 3)
        if is_cap:
            # a capitalised word right after sentence start is usually the question word
            if not run and low in _STOP_ALONE:
                continue
            run.append(tok)
            continue
        if run and tok in ("'", "’") and idx + 1 < len(tokens) and tokens[idx + 1][:1].isupper():
            run.append(tok)
            continue
        if run and low in _PARTICLE and not skip:
            # absorb a chain of particles only when a capitalised word closes it ("Back in the U.S.S.R")
            chain = [tok]
            j = idx + 1
            while j < len(tokens) and tokens[j].lower().strip(".") in _PARTICLE:
                chain.append(tokens[j])
                j += 1
            if j < len(tokens) and tokens[j][:1].isupper():
                run.extend(chain)
                skip = len(chain) - 1
                continue
        if run:
            found.append(_join(run))
            run = []
    if run:
        found.append(_join(run))

    out: list[str] = []
    for cand in found:
        cand = cand.strip(" ,;:-").strip()
        while cand.lower().split(" ")[0] in _STOP_ALONE and " " in cand:
            cand = cand.split(" ", 1)[1].strip()
        if len(cand) < 3 or not _ALPHA.search(cand):
            continue
        words = [w for w in cand.split(" ") if w]
        if len(words) == 1 and (words[0].lower().strip(".") in _STOP_ALONE or len(words[0]) < 3):
            continue
        if cand.lower() not in {o.lower() for o in out}:
            out.append(cand)

    # drop a mention fully contained in a longer one (the longer span is the better query)
    out.sort(key=lambda s: -len(s))
    kept: list[str] = []
    for cand in out:
        if not any(cand.lower() in k.lower() for k in kept):
            kept.append(cand)
    return kept


DECOMPOSE_INSTRUCTIONS = """Plan a keyword search for a question that can only be answered by combining
two different source documents.

Write 3 search queries, one per line, no numbering and no explanation:
1. the most specific description in the question, phrased as the title/topic of the document that describes it;
2. a short keyword-only variant of that same target using distinctive proper nouns, dates or numbers only
   (no question words, no stop words) -- it should retrieve the same document by different wording;
3. the second document: the entity or property that the question ultimately asks about, phrased as a topic.

The retriever matches words, not meaning, so prefer the exact names and rare terms that would appear in the
document, and never write a query as a question."""

DISTIL_INSTRUCTIONS = """You are building a compact evidence table for a question that needs facts from several
different documents.

The passages are numbered search results; most of them are irrelevant. In `reasoning`, work out the chain of
facts the question needs, then find where the passages state the links of that chain.

Produce:
- `claims`: at most 9 lines, most important first, each written exactly as

      <passage number> :: <quote copied word for word from that passage>

  A quote is a single clause or sentence fragment -- long enough to stand on its own, never a whole paragraph.
  Copy it character for character out of the passage: never paraphrase it, never shorten a name inside it,
  never merge two passages into one line. Only include a line if the quote states a fact the answer depends on,
  and do not write a title on the line -- the title is added for you. Write one line for EVERY link of the
  chain, from different passages: a question that needs facts from two documents needs a quote from each, so a
  table with a single line is almost always incomplete. Write `none` if no passage states anything the question
  needs.
- `missing`: the single fact still needed, named concretely (for example the specific entity whose property must
  still be looked up). Write `none` if the quotes already establish every link.
- `next_query`: a keyword query for that missing fact, anchored on the exact entity name as it appears in the
  passages, plus the property being sought. Leave it empty when nothing is missing. Never repeat a query whose
  results are already listed.
- `sufficient`: `yes` if the quotes already contain every fact needed to answer, otherwise `no`."""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence with the shortest possible answer.

`evidence` is a table of quotes copied out of source documents, each labelled with the document it came from,
followed by the few source documents the table draws on most, in full.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
evidence uses. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from this evidence returned nothing usable.

The evidence may cover the question only partially. Commit anyway to the single most likely short answer it
supports: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8                   # results per decomposed query
    K_PROBE = 4             # results per bare-name lookup: probes are for coverage, not for bulk
    MAX_PROBES = 4
    MAX_ROUNDS = 3          # distillation passes (each may add one retrieval)
    MAX_RETRIEVALS = 10
    POOL_SNIPPET = 320      # chars of each passage shown to the distiller
    MAX_CLAIMS = 9          # the answering context is capped here, whatever the pool size
    N_PINNED = 3            # raw passages pinned behind the table

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"],
            outputs=["queries"],
            instructions=DECOMPOSE_INSTRUCTIONS,
            descs={"queries": "3 search queries, one per line"},
        )
        self.distil = Module(
            inputs=["question", "passages"],
            outputs=["claims", "missing", "next_query", "sufficient"],
            instructions=DISTIL_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "claims": "up to 9 lines of `<passage number> :: <verbatim quote>`, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "evidence"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "evidence": "quotes from the sources, then the most-cited sources in full",
                "answer": "the bare short answer",
            },
        )
        self.commit = Module(
            inputs=["question", "evidence"],
            outputs=["answer"],
            instructions=COMMIT_INSTRUCTIONS,
            descs={"answer": "the bare short answer, never an abstention"},
        )

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("\u00ab\u00bb").strip()
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
    def _tidy(text: str, question: str) -> str:
        ans = _MARKUP.sub("", str(text or "")).strip().split("\n")[0].strip()
        ans = _LEADING_FILLER.sub("", ans).strip()
        ans = ans.strip('"').strip("'").strip("\u00ab\u00bb").strip()
        m = _YESNO_PREFIX.match(ans)
        if m and (_BOOL_QUESTION.match(question or "") or len(ans.split()) > 1):
            return m.group(1).lower()
        while ans.endswith((".", ",", ";", ":")):
            ans = ans[:-1].strip()
        return ans

    # -- the claim table ----------------------------------------------------------------------
    @staticmethod
    def _split_passage(passage: str) -> tuple[str, str]:
        text = str(passage)
        title, _, body = text.partition(" | ")
        return title.strip(), body.strip()

    def _parse_claims(self, raw: str, pool: list[str]) -> tuple[list[str], list[int]]:
        """Turn the distiller's lines into `Title :: quote` rows, re-attaching titles from the pool itself."""
        rows: list[str] = []
        cited: list[int] = []
        seen_rows: set[str] = set()
        for line in str(raw or "").splitlines():
            line = _MARKUP.sub("", line).strip()
            if not line or line.lower() in {"none", "claims"}:
                continue
            m = _CLAIM_LINE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            if not 1 <= idx <= len(pool):
                continue
            quote = m.group(2).strip().strip('"').strip("\u00ab\u00bb").strip()
            quote = re.sub(r"^[a-z]+\s*::\s*", "", quote).strip()
            if len(quote) < 8:
                continue
            title, body = self._split_passage(pool[idx - 1])
            # a quote the passage does not contain is a paraphrase; fall back to the passage's own opening
            if quote[:40].lower() not in body.lower():
                quote = body[:300].strip()
            row = f"{title} :: {quote}"[:600]
            key = row.lower()
            if key in seen_rows:
                continue
            seen_rows.add(key)
            rows.append(row)
            if idx not in cited:
                cited.append(idx)
            if len(rows) >= self.MAX_CLAIMS:
                break
        return rows, cited

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str, k: int) -> int:
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=k):
                key = str(passage).split(" | ")[0].strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(passage)
                added += 1
            return added

        # 1. go wide: decomposed sweep + one bare lookup per name the question prints
        plan = self.decompose(ctx.llm, tag="decompose", question=question).get("queries", "")
        queries = self._parse_queries(plan, question)
        for q in queries:
            gather(q, self.K)
        probes = _mentions(question)[: self.MAX_PROBES]
        for name in probes:
            gather(name, self.K_PROBE)

        # 2-4. distil the pool into a fixed-size claim table, following up on what is missing
        claims: list[str] = []
        cited: list[int] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            out = self.distil(ctx.llm, tag=f"distil{round_i + 1}", question=question, passages=shown)
            rows, idxs = self._parse_claims(out.get("claims", ""), pool)
            if rows:
                claims, cited = rows, idxs
            missing = str(out.get("missing", "")).strip()
            sufficient = "yes" in str(out.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(out.get("next_query", ""), self.K):
                break     # follow-up query brought nothing new: stop paying for rounds

        # 5. the answering context: the table, then the raw passages it cites most -- fixed size by construction
        # the table's own sources first, then the best unread passages -- always exactly N_PINNED, so the
        # answering context has the same shape whether the distiller cited one passage or nine
        pinned = [pool[i - 1] for i in cited[: self.N_PINNED]]
        for passage in pool:
            if len(pinned) >= self.N_PINNED:
                break
            if passage not in pinned:
                pinned.append(passage)
        evidence = claims + [f"SOURCE: {p}" for p in pinned]
        ctx.note(
            "claim table",
            n_pool=len(pool), n_claims=len(claims), n_probes=len(probes),
            retrievals=retrievals, missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, evidence=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        # Abstain-and-retry: an empty or non-committal answer scores zero anyway, so spend a spare call
        # forcing a best guess -- over the raw pool, since distillation may have dropped the decisive line.
        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, evidence=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "n_claims": len(claims)}
