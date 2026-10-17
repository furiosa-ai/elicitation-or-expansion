"""Learn the answer's SURFACE-FORM convention from the labelled train split instead of guessing it.

Two thirds of the frontier system's misses (64 of 92) already have the gold string somewhere in their own
context, and the largest coherent cluster inside them is not a reasoning error at all: the model returns the
name it would use in conversation while the reference uses the form the article writes.  It answers with the
familiar short name where the reference carries the middle names, and with the bare place name where the
reference carries the administrative qualifier the passage attached to it.  The reverse also happens -- the
model adds a modifier the reference does not have -- so "always expand" is exactly as wrong as "always
shorten"; earlier attempts to adjudicate surface forms by prompting lost ground precisely because the
direction was guessed.

The direction does not have to be guessed. The train split is labelled, and its labels are drawn from the
same convention as the evaluation labels, so the convention is measurable.  `prepare` looks the gold articles
up in the index, reads how each gold answer is actually written inside them, and scores two candidate
rewrite rules -- fill in a name's inner words from the article, and keep a comma qualifier the article
attaches -- by counting, on real labelled items, how many answers each rule would REPAIR against how many it
would BREAK.  A rule is switched on only when the counts favour it, and the measured counts are carried into
the answer prompt so the model sees the convention as well.

At run time the rewrite is model-free and evidence-bound: a form is only ever adopted if it occurs verbatim
in the evidence the router selected, so the mechanism can change how an answer is written but never what it
claims.  Retrieval, routing and evidence selection are untouched, so the whole delta is attributable to
learning the answer convention from data.
"""

from __future__ import annotations

import re
from typing import Any

import json

from core.harness import BaseHarness, PrepareContext, RunContext
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
_HEDGE_PREFIX = re.compile(
    r"^(?:none\s+of\b|no\s+(?:information|passage|mention|document|evidence|such)\b|not\s+(?:enough|clear)\b"
    r"|the\s+(?:passages?|documents?|text)\b|insufficient\b|unable\s+to\b|i\s+(?:cannot|can't|do\s+not)\b)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w']+")
_PAREN = re.compile(r"\([^)]*\)")
_TITLE_STOP = frozenset(
    "a an the of in on at to for and or de la le du von van der los las el il".split()
)
_QSTOP = frozenset(
    """a an the of in on at to for from by with and or is was were are am be been being do does did done
    who what which when where how why whose whom that this these those it its their his her he she they them
    there here as than then also both same not no nor but if while during between about into over under
    after before above below out up down off again further once any some such only own so too very can could
    will would shall should may might must have has had known does you your i we our""".split()
)
_CONVENTION_GUARD = (
    " None of this makes the answer longer than it has to be: it is still the bare shortest span, never a"
    " phrase describing the answer, and a question asking whether something is or is not the case is still"
    " answered with exactly `yes` or `no`."
)
_LABEL = re.compile(
    r"^\s*(?:article|title|document|doc|entity|bridge(?:\s+entity)?|first|second|third)\s*\d*\s*[:\-]\s*",
    re.IGNORECASE,
)

HYPOTHESIS_INSTRUCTIONS = """This question can only be answered by combining two different encyclopedia
articles. Name those articles.

Use your own knowledge of the encyclopedia, not just the words of the question: work out which entity the
question describes, then which entity or work its answer depends on, and recall what the article about each
one is actually called.

In `reasoning`, spell out the chain: the entity described by the question, the link that connects it to the
answer, and the article that would state the final fact. Name the connecting entity explicitly even if the
question does not, and say when you are unsure of a name.

Write 3 or 4 article titles in `titles`, one per line, best guess first, no numbering, no explanation and no
extra words -- just the titles as they would appear at the top of the article:
1. the article about the entity the question describes most specifically;
2. the article about the connecting entity or work you recalled (the bridge), if you can name it;
3. the article that would state the fact the question finally asks for;
4. optionally one alternative name for whichever of the above you are least sure about.

A title you are unsure of is still worth writing -- it is checked against the encyclopedia before it is
trusted -- but never write a question, a sentence, or a description instead of a title."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. Passages found by looking up a title that the encyclopedia
confirmed are listed first; the rest come from a plain search of the question and may be off topic. A title
that was guessed but NOT confirmed is listed under `unconfirmed`: treat those names as unreliable, and never
use one as a fact.

In `reasoning`, decide which chain of facts the question needs, then check which links of that chain the
passages already establish.

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

`convention` reports how reference answers were actually written, measured on labelled examples. Follow it.

Output the bare answer only: an entity, name, title, date, number or verdict, copied with the wording the
passages use. No sentence, no restatement of the question, no explanation, no trailing punctuation. If the
question asks whether something is or is not the case, output exactly `yes` or `no`."""

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8
    K_TITLE = 5             # a title probe is precise; it does not need width
    MAX_HYPOTHESES = 3
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 10
    POOL_SNIPPET = 380
    N_EVIDENCE = 8
    CALIB_ITEMS = 150           # the whole labelled pool: the measurement is cheap and model-free
    K_GOLD = 3

    def __init__(self) -> None:
        self.hypothesize = Module(
            inputs=["question"],
            outputs=["titles"],
            instructions=HYPOTHESIS_INSTRUCTIONS,
            descs={"titles": "3-4 encyclopedia article titles, one per line, best guess first"},
        )
        self.route = Module(
            inputs=["question", "passages", "unconfirmed"],
            outputs=["useful", "missing", "next_query", "sufficient"],
            instructions=ROUTE_INSTRUCTIONS,
            descs={
                "passages": "numbered search results, each `Title | abstract`",
                "unconfirmed": "guessed titles the encyclopedia did not confirm",
                "useful": "comma separated passage numbers, or `none`",
                "missing": "the fact still needed, or `none`",
                "next_query": "keyword query for the missing fact",
                "sufficient": "yes or no",
            },
        )
        self.answer = Module(
            inputs=["question", "passages", "convention"],
            outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs={
                "convention": "how reference answers are written, measured on labelled examples",
                "answer": "the bare short answer",
            },
        )
        self.rules = {"fill_inner_words": False, "keep_qualifier": False}
        self.convention = "No measurement available; copy the wording the passages use."
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

    def _parse_titles(self, raw: str) -> list[str]:
        """One hypothesised article title per line; drop sentences, questions and boilerplate."""
        titles: list[str] = []
        for line in str(raw or "").splitlines():
            t = _LABEL.sub("", self._clean_query(line)).strip().strip('"').strip("'")
            if len(t) < 2 or "?" in t or t.lower() in {"none", "titles", "unknown", "n/a"}:
                continue
            if len(_WORD.findall(t)) > 10:          # a description, not a title
                continue
            if t.lower() not in {x.lower() for x in titles}:
                titles.append(t)
            if len(titles) == self.MAX_HYPOTHESES:
                break
        return titles

    @staticmethod
    def _keyterms(question: str) -> str:
        """Model-free keyword-only variant: the recall floor a title hypothesis must never fall below."""
        toks = [w for w in _WORD.findall(str(question or "")) if w.lower() not in _QSTOP]
        return " ".join(toks)[:300]

    @staticmethod
    def _key_tokens(title: str) -> set[str]:
        base = _PAREN.sub(" ", str(title or "")).lower()
        return {w for w in _WORD.findall(base) if w not in _TITLE_STOP}

    def _confirms(self, hypothesis: str, found_title: str) -> bool:
        """The index confirms a hypothesised name when the retrieved article is essentially that article."""
        h = self._key_tokens(hypothesis)
        f = self._key_tokens(found_title)
        if not h or not f:
            return False
        if h <= f:                                   # retrieved title contains the whole hypothesis
            return True
        overlap = len(h & f)                         # or differs only in a qualifier / spelling variant
        return overlap >= 2 and overlap / len(h | f) >= 0.75

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

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        confirmed: list[str] = []       # passages whose title the index confirmed: the pinned front of the pool
        rest: list[str] = []            # everything else, in arrival order
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str, k: int, pin_for: str | None = None) -> tuple[int, bool]:
            """Retrieve `query`; when it is a title hypothesis, pin the article only if the index confirms it."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return 0, False
            used_queries.add(q.lower())
            retrievals += 1
            added = 0
            verified = False
            for rank, passage in enumerate(ctx.tools.retrieve(q, k=k)):
                title = str(passage).split(" | ")[0].strip()
                key = title.lower()
                pinned = pin_for is not None and rank < 2 and self._confirms(pin_for, title)
                if pinned:
                    verified = True
                if key in seen:
                    if pinned and passage in rest:   # promote an already seen article to the pinned front
                        rest.remove(passage)
                        confirmed.append(passage)
                    continue
                seen.add(key)
                (confirmed if pinned else rest).append(passage)
                added += 1
            return added, verified

        # 1. name the articles the chain needs, from the model's own knowledge of the encyclopedia
        raw = self.hypothesize(ctx.llm, tag="hypothesize", question=question).get("titles", "")
        hypotheses = self._parse_titles(raw)

        # 2. the paraphrase sweep still runs, so a wrong guess can never cost recall: the plain question
        #    plus its model-free keyword-only variant are the floor the hypotheses are added on top of.
        gather(question, self.K)
        gather(self._keyterms(question), self.K)

        # 3. probe every hypothesis as a title lookup and check whether the index confirms the name
        unconfirmed: list[str] = []
        for h in hypotheses:
            _, ok = gather(h, self.K_TITLE, pin_for=h)
            if not ok:
                unconfirmed.append(h)

        # 4. route over the pool (pinned articles first), following up on whatever link is still missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            pool = confirmed + rest
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(
                ctx.llm,
                tag=f"route{round_i + 1}",
                question=question,
                passages=shown,
                unconfirmed=", ".join(unconfirmed) if unconfirmed else "none",
            )
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[: self.N_EVIDENCE]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(routed.get("next_query", ""), self.K)[0]:
                break

        pool = confirmed + rest
        evidence = selected or pool[: self.N_EVIDENCE]
        ctx.note(
            "verified title hypotheses",
            hypotheses=hypotheses,
            n_confirmed=len(confirmed),
            unconfirmed=unconfirmed,
            n_pool=len(pool),
            retrievals=retrievals,
            missing=missing[:200],
        )

        drafted = self.answer(
            ctx.llm, tag="answer", question=question, passages=evidence, convention=self.convention
        ).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        # 5. rewrite the answer into the measured convention, but only into a form the evidence writes
        rewritten = self._apply_rules(final, "\n".join(evidence))
        if rewritten != final:
            ctx.note("answer form rewritten", before=final[:120], after=rewritten[:120], rules=self.rules)
            final = rewritten

        return {
            "answer": final or str(drafted).strip(),
            "hypotheses": hypotheses,
            "n_confirmed": len(confirmed),
            "n_pool": len(pool),
            "missing": missing,
        }

    # -- learned answer-form rules -------------------------------------------------------------
    @staticmethod
    def _cap_words(text: str) -> list[str]:
        return _WORD.findall(str(text or ""))

    @staticmethod
    def _is_name(text: str) -> bool:
        """A multi-word proper name: every word starts uppercase and none is a digit."""
        words = str(text or "").split()
        return (
            len(words) >= 2
            and all(w[:1].isupper() and not w[:1].isdigit() for w in words)
            and not any(ch.isdigit() for ch in text)
        )

    def _fill_inner_words(self, draft: str, evidence: str) -> str:
        """`First Last` -> `First Middle Last` when the evidence itself writes the longer form."""
        words = draft.split()
        if len(words) < 2 or not self._is_name(draft):
            return draft
        pattern = (
            re.escape(words[0])
            + r"\s+(?:[A-Z][\w.'’-]*\.?\s+){1,3}"
            + re.escape(words[-1])
            + ("" if len(words) == 2 else "")
        )
        if len(words) > 2:                       # only the pure `First Last` shape is ambiguous
            return draft
        for m in re.finditer(pattern, evidence):
            found = " ".join(m.group(0).split())
            if len(found) > len(draft):
                return found
        return draft

    def _keep_qualifier(self, draft: str, evidence: str) -> str:
        """`Place` -> `Place, Qualifier` when the evidence attaches one to that exact mention."""
        if not draft or _BOOL_QUESTION.match(draft) or any(ch.isdigit() for ch in draft):
            return draft
        if not draft[:1].isupper():
            return draft
        pattern = re.escape(draft) + r",\s+([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,2})\b"
        m = re.search(pattern, evidence)
        if m:
            return f"{draft}, {m.group(1)}"
        return draft

    def _trim_inner_words(self, draft: str, evidence: str) -> str:
        """`First Middle Last` -> `First Last` when the evidence itself also writes the shorter form."""
        words = draft.split()
        if len(words) < 3 or not self._is_name(draft):
            return draft
        short = f"{words[0]} {words[-1]}"
        return short if short.lower() in evidence.lower() else draft

    @staticmethod
    def _trim_qualifier(draft: str, evidence: str) -> str:
        """`Place, Qualifier` -> `Place` when the evidence also writes the bare head."""
        head = draft.split(",")[0].strip()
        if "," not in draft or not head:
            return draft
        return head if head.lower() in evidence.lower() else draft

    def _apply_rules(self, draft: str, evidence: str) -> str:
        if not draft or not evidence or draft.strip().lower() in {"yes", "no"}:
            return draft
        if draft.lower() not in evidence.lower():        # never invent a form the evidence does not write
            return draft
        out = draft
        if self.rules.get("fill_inner_words"):
            out = self._fill_inner_words(out, evidence)
        elif self.rules.get("trim_inner_words"):
            out = self._trim_inner_words(out, evidence)
        if self.rules.get("keep_qualifier"):
            out = self._keep_qualifier(out, evidence)
        elif self.rules.get("trim_qualifier"):
            out = self._trim_qualifier(out, evidence)
        return out

    # -- calibration on the labelled train split -----------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        tally = {
            "fill_inner_words": [0, 0],      # [repairs, breakages]
            "trim_inner_words": [0, 0],
            "keep_qualifier": [0, 0],
            "trim_qualifier": [0, 0],
            "n_items": 0,
            "n_verbatim": 0,
        }
        for item in train_items[: self.CALIB_ITEMS]:
            gold = str(item.get("answer") or "").strip()
            titles = list(item.get("gold_titles") or [])
            if not gold or not titles or gold.lower() in {"yes", "no"}:
                continue
            evidence = self._gold_text(item, titles, ctx)
            if not evidence:
                continue
            tally["n_items"] += 1
            if gold.lower() in evidence.lower():
                tally["n_verbatim"] += 1

            words = gold.split()
            # family 1 -- a name whose inner words the evidence supplies: fill them in, or cut back to
            # the short form? Score both directions on this label, one repairs where the other breaks.
            if self._is_name(gold) and gold.lower() in evidence.lower():
                if len(words) >= 3:
                    short = f"{words[0]} {words[-1]}"
                    if self._fill_inner_words(short, evidence) == gold:
                        tally["fill_inner_words"][0] += 1       # a short draft would be repaired
                    if self._trim_inner_words(gold, evidence) != gold:
                        tally["trim_inner_words"][1] += 1       # the label itself would be broken
                elif len(words) == 2:
                    longer = self._fill_inner_words(gold, evidence)
                    if longer != gold:
                        tally["fill_inner_words"][1] += 1       # the label itself would be broken
                        if self._trim_inner_words(longer, evidence) == gold:
                            tally["trim_inner_words"][0] += 1   # a padded draft would be repaired
            # family 2 -- a comma qualifier the evidence attaches: keep it, or drop it?
            if gold.lower() in evidence.lower() and gold[:1].isupper():
                head = gold.split(",")[0].strip()
                if "," in gold and head:
                    if self._keep_qualifier(head, evidence) == gold:
                        tally["keep_qualifier"][0] += 1
                    if self._trim_qualifier(gold, evidence) != gold:
                        tally["trim_qualifier"][1] += 1
                elif "," not in gold:
                    qualified = self._keep_qualifier(gold, evidence)
                    if qualified != gold:
                        tally["keep_qualifier"][1] += 1
                        if self._trim_qualifier(qualified, evidence) == gold:
                            tally["trim_qualifier"][0] += 1

        # within each family the direction with the better repair-minus-breakage margin wins; if neither
        # direction is a net gain the family stays off, so a rule is never applied against the evidence.
        self.rules = {}
        for expand, trim in (("fill_inner_words", "trim_inner_words"), ("keep_qualifier", "trim_qualifier")):
            scores = {r: tally[r][0] - tally[r][1] for r in (expand, trim)}
            winner = max(scores, key=lambda r: scores[r])
            self.rules[expand] = winner == expand and scores[expand] > 0
            self.rules[trim] = winner == trim and scores[trim] > 0
        self.convention = self._describe(tally, self.rules)
        ctx.note("answer-form calibration", tally=tally, rules=self.rules, convention=self.convention)

    def _gold_text(self, item: dict[str, Any], titles: list[str], ctx: PrepareContext) -> str:
        """The abstracts the label was written from: look the gold articles up by title."""
        texts: list[str] = []
        try:
            with ctx.open_tools(item) as tools:
                for title in titles[:2]:
                    for passage in tools.retrieve(str(title), k=self.K_GOLD):
                        found = str(passage).split(" | ")[0].strip()
                        if self._confirms(str(title), found):
                            texts.append(str(passage))
                            break
        except Exception:                                # a missing article simply drops out of the sample
            return ""
        return "\n".join(texts)

    @staticmethod
    def _describe(tally: dict[str, Any], rules: dict[str, bool]) -> str:
        n = max(tally["n_items"], 1)
        lines = [
            f"Measured on {tally['n_items']} labelled examples: {round(100 * tally['n_verbatim'] / n)}% of "
            "reference answers appear word for word inside the passage, so copy the passage's wording rather "
            "than rephrasing it.",
        ]
        fw, fb = tally["fill_inner_words"]
        tw, tb = tally["trim_inner_words"]
        qw, qb = tally["keep_qualifier"]
        rw, rb = tally["trim_qualifier"]
        if rules.get("fill_inner_words"):
            lines.append(
                f"When the passage writes a person's name in full, the reference uses that full form, "
                f"middle names included ({fw} examples for, {fb} against): give the name as the passage writes it."
            )
        elif rules.get("trim_inner_words"):
            lines.append(
                f"Do not pad a name with the extra words an article carries: the reference keeps the short "
                f"everyday form ({tw} examples for, {tb} against)."
            )
        if rules.get("keep_qualifier"):
            lines.append(
                f"When the passage qualifies a place or work with a following comma, the reference keeps the "
                f"qualifier ({qw} examples for, {qb} against)."
            )
        elif rules.get("trim_qualifier"):
            lines.append(
                f"Drop a trailing comma qualifier the question did not ask for, even when the passage "
                f"attaches one ({rw} examples for, {rb} against)."
            )
        return " ".join(lines)

    def get_state(self) -> str:
        return json.dumps({"rules": self.rules, "convention": self.convention})

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except ValueError:
            return
        if isinstance(data.get("rules"), dict):
            self.rules = {k: bool(v) for k, v in data["rules"].items()}
        if data.get("convention"):
            self.convention = str(data["convention"]) + _CONVENTION_GUARD
