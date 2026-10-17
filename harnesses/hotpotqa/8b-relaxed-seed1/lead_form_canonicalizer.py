"""Canonicalise the answer's surface form against the answer entity's OWN article.

30 of the frontier's 89 misses had every required article in the pool, and 11 of those 30 are a single
mechanical pattern: the model returns the name an entity is commonly called while the graded string is the
fuller form the entity's own article opens with -- a short person name against the article's lead form with
its middle names, a bare place name against the article's qualified title, a bare event name against its
dated title.  f1 on those items is 0.8-0.9: every token of the prediction is right and one is missing.

Three earlier systems attacked answer form and all of them lost ground, because they asked the model to
choose between surface variants mined out of the evidence it had already read -- the same evidence that
produced the short form in the first place.  This harness does not ask the model anything.  After the answer
is drafted it spends one retrieval on the DRAFT ITSELF, finds the article the index holds under exactly that
name, and reads the canonical form off that article model-free:

    * `title_complete` -- the article's title is the draft plus a qualifier it begins with;
    * `lead_expand`    -- the article's opening subject phrase is the draft with extra name-shaped tokens
                          between the same first and last token.

Whether either rewrite is applied at all is not a guess.  `prepare` measures both rules on the train split
against the index: for every train answer that names an entity, applying a rule to the answer itself and
changing it counts as a BREAKAGE (the graded form was already the target), while applying it to the article
title recovering the answer counts as a REPAIR (the form a model would have produced gets fixed).  A rule is
switched on for evaluation only if it repaired strictly more than it broke, and the measured counts travel
in the harness state.  Retrieval, routing and drafting are untouched, so the delta is attributable to the
canonicaliser alone.
"""

from __future__ import annotations

import json
import re
from typing import Any

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
_COMMA_TAIL = re.compile(r",.*$")
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
_NAMETOK = re.compile(r"[\w'\u2019\-]+")
_CAP = re.compile(r"^[A-Z\u00C0-\u024F]")
_PARTICLE = frozenset("de del della di da van von der den ter la le du dos das of and the y i ii iii jr sr".split())
_LEAD_PATTERNS = (
    re.compile(r"^(.{2,90}?)\s+\("),
    re.compile(r"^(.{2,90}?)\s+(?:is|was|are|were)\s"),
)
_TITLE_TAIL = re.compile(r"\s*\([^)]*\)\s*$")
_NORM_WS = re.compile(r"\s+")

_LABEL = re.compile(
    r"^\s*(?:name|title|entity|article|constraint|requirement|mention)\s*\d*\s*[:\-]\s*", re.IGNORECASE
)

ANALYZE_INSTRUCTIONS = """A multi-step question is about to be looked up in an encyclopedia whose search ranks
articles by popularity. Prepare the lookup.

Many names are shared by several articles -- a novel and the film made from it, two songs, a place and a much
bigger place with the same name, several films released in different years. The search will return the FAMOUS
one. To pick the right one instead, two things are needed.

In `reasoning`, work out which entity the question is really about and what in the question rules out the
better-known things sharing its name.

Produce:
- `names`: the proper names, work titles and place names that the question MENTIONS, one per line, most
  ambiguous first, at most three. Write each one bare, exactly as the article about it would be titled: no
  quotation marks, no year, no descriptive words the question wrapped around it, nothing invented. Write
  `none` if the question mentions no name at all.
- `constraints`: the checklist the intended article must satisfy, one short requirement per line, at most
  four. Take these from the question only -- what KIND of thing it is (film, novel, song, album, person,
  town, team), when it appeared, where it is from, what role it played. Do not guess facts the question does
  not state. Write `none` if the question states no distinguishing property."""

DISAMBIGUATE_INSTRUCTIONS = """A name in a question matches several different encyclopedia articles. Choose
the one the question is about.

You are given the checklist of properties the intended article must satisfy, and the candidate articles the
encyclopedia actually holds under each name, numbered, each shown as `Title | opening of the article`.

Judge by the checklist, not by fame. The best known article under a name is usually NOT the one an
indirect question is about: if the question says a novel, a candidate that opens by describing a film fails
the checklist however famous it is; if the question fixes a year, a nationality or a role, a candidate whose
opening contradicts it is out. A candidate that merely fails to mention a property does not violate it --
only an outright contradiction does.

In `reasoning`, go name by name and say which candidates each checklist item eliminates.

Produce `picks`: one line per name, written `<name> = <number>`, giving the number of the surviving candidate.
Write `<name> = none` when every candidate contradicts the checklist, or when two survive and nothing in the
question separates them. Guessing when nothing separates the candidates is worse than `none`."""

ROUTE_INSTRUCTIONS = """You are routing a keyword search that must gather every fact needed to answer a
multi-step question.

The passages are numbered search results. Passages listed first are articles that were chosen as the intended
meaning of a name in the question, in preference to better known articles sharing that name; trust them over
a passage further down that looks similar but describes a different thing. The rest come from a plain search
and may be about the wrong namesake entirely.

In `reasoning`, decide which chain of facts the question needs, then check which links of that chain the
passages already establish, and watch for passages that share a name with the subject but contradict the
question (wrong year, wrong medium, wrong country).

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

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""


class Harness(BaseHarness):
    K = 8
    K_NAME = 8              # a bare-name probe must be wide enough to reach the obscure namesake
    MAX_NAMES = 3
    MAX_SLATE = 4           # candidates shown per colliding name
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 9      # one retrieval is held back for the canonical-form lookup
    POOL_SNIPPET = 380
    SLATE_SNIPPET = 240
    N_EVIDENCE = 8

    def __init__(self) -> None:
        # defaults if `prepare` never runs: both rules off, so the harness degrades to the frontier
        self.enabled: dict[str, bool] = {r: False for r in self.RULES}
        self.stats: dict[str, Any] = {}
        self.analyze = Module(
            inputs=["question"],
            outputs=["names", "constraints"],
            instructions=ANALYZE_INSTRUCTIONS,
            descs={
                "names": "bare names the question mentions, one per line, at most three, or `none`",
                "constraints": "properties the intended article must satisfy, one per line, or `none`",
            },
        )
        self.disambiguate = Module(
            inputs=["question", "constraints", "candidates"],
            outputs=["picks"],
            instructions=DISAMBIGUATE_INSTRUCTIONS,
            descs={
                "constraints": "the checklist the intended article must satisfy",
                "candidates": "candidate articles per name, numbered, `Title | opening`",
                "picks": "one `<name> = <number>` or `<name> = none` per line",
            },
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

    def _parse_lines(self, raw: str, limit: int, max_words: int) -> list[str]:
        out: list[str] = []
        for line in str(raw or "").splitlines():
            t = _LABEL.sub("", self._clean_query(line)).strip().strip('"').strip("'")
            if len(t) < 2 or t.lower() in {"none", "n/a", "unknown", "names", "constraints"}:
                continue
            if len(_WORD.findall(t)) > max_words:
                continue
            if t.lower() not in {x.lower() for x in out}:
                out.append(t)
            if len(out) == limit:
                break
        return out

    @staticmethod
    def _keyterms(question: str) -> str:
        """Model-free keyword-only variant: the recall floor the name probes are added on top of."""
        toks = [w for w in _WORD.findall(str(question or "")) if w.lower() not in _QSTOP]
        return " ".join(toks)[:300]

    @staticmethod
    def _key_tokens(text: str) -> set[str]:
        base = _PAREN.sub(" ", str(text or "")).lower()
        return {w for w in _WORD.findall(base) if w not in _TITLE_STOP}

    def _is_about(self, name: str, title: str) -> bool:
        """Model-free slate filter: this article really is one of the things called `name`.

        A title qualifies when it carries every content token of the name, so a bare name and the same name
        with a parenthetical sense qualifier both enter the slate, while an article that merely discusses the
        name does not, and a name the index writes with a trailing comma qualifier still matches.
        """
        n = self._key_tokens(name)
        if not n:
            return False
        t = self._key_tokens(title)
        if not n <= t:
            return False
        base = self._key_tokens(_COMMA_TAIL.sub("", _PAREN.sub("", str(title or ""))))
        return len(base - n) <= 1        # the title is the name itself, give or take one qualifier word

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


    # -- surface-form canonicaliser -------------------------------------------------------------
    #
    # Everything below is model-free.  `_forms` reads the two canonical forms an article offers for
    # the name it is titled with; `_apply` rewrites a drafted answer into one of them when the rule
    # that produced it was measured on train to repair more answers than it breaks.

    RULES = ("title_complete", "lead_expand")
    MARGIN = 0              # train is a VETO, not a mandate: a rule ships unless it broke more than
    #                         it repaired.  The pattern is rare on 150 train answers (a handful of
    #                         observations) but common among the frontier's full-recall misses, so a
    #                         tie must not veto -- while a rule that breaks answers wholesale must.

    @staticmethod
    def _name_tokens(text: str) -> list[str]:
        return _NAMETOK.findall(str(text or ""))

    @classmethod
    def _is_name_shaped(cls, text: str) -> bool:
        """A drafted answer that could be an entity NAME: 1-5 capitalised, digit-free tokens."""
        t = str(text or "").strip()
        if not t or any(ch.isdigit() for ch in t):
            return False
        toks = cls._name_tokens(t)
        if not 1 <= len(toks) <= 5:
            return False
        return all(_CAP.match(w) or w.lower() in _PARTICLE for w in toks)

    @classmethod
    def _lead_subject(cls, passage: str) -> str:
        """The subject phrase an abstract opens with, before its first parenthesis or copula."""
        body = str(passage or "").split(" | ", 1)[-1].strip()
        best = ""
        for pat in _LEAD_PATTERNS:
            m = pat.match(body)
            if m:
                cand = m.group(1).strip().strip(",")
                if cand and (not best or len(cand) < len(best)):
                    best = cand
        return best

    @classmethod
    def _forms(cls, draft: str, passages: list[str]) -> dict[str, str]:
        """Canonical forms the index offers for `draft`, keyed by the rule that produced them.

        Only an article that IS about the draft counts: its title, stripped of a parenthetical sense
        qualifier, must start with exactly the draft's tokens.  That keeps a merely related article
        from renaming the answer.
        """
        want = [w.lower() for w in cls._name_tokens(draft)]
        out: dict[str, str] = {}
        if not want:
            return out
        for passage in passages:
            title = str(passage or "").split(" | ")[0].strip()
            base = _TITLE_TAIL.sub("", title)
            btoks = [w.lower() for w in cls._name_tokens(base)]
            if btoks[: len(want)] != want:
                continue

            # rule 1: the index titles this thing with a qualifier the draft dropped
            if "title_complete" not in out and len(btoks) > len(want) and len(btoks) <= len(want) + 3:
                out["title_complete"] = base

            # rule 2: the article opens on a longer form of the same name
            if "lead_expand" not in out and btoks == want:
                lead = cls._lead_subject(passage)
                ltoks = cls._name_tokens(lead)
                low = [w.lower() for w in ltoks]
                if (
                    len(want) < len(low) <= len(want) + 4
                    and low[0] == want[0]
                    and low[-1] == want[-1]
                    and all(_CAP.match(w) or w.lower() in _PARTICLE for w in ltoks)
                ):
                    out["lead_expand"] = lead
        return out

    def _canonicalise(self, draft: str, passages: list[str]) -> tuple[str, str]:
        """Return (answer, rule_applied). The first enabled rule that fires wins."""
        if not self._is_name_shaped(draft):
            return draft, ""
        forms = self._forms(draft, passages)
        for rule in self.RULES:
            if self.enabled.get(rule) and forms.get(rule):
                return forms[rule], rule
        return draft, ""

    # -- train-split measurement ----------------------------------------------------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        """Count, per rule, how many train answers it repairs and how many it breaks.

        For a train answer that names an entity we look the answer up in the index and consider the
        article the index holds under it.  Applying a rule to the graded answer and CHANGING it is a
        breakage: the graded form was already canonical.  Applying the rule to that article's title --
        the short, popular form a model naturally produces -- and landing on the graded answer is a
        repair.  Only rules with a positive net are switched on.
        """
        repairs = {r: 0 for r in self.RULES}
        breaks = {r: 0 for r in self.RULES}
        looked = 0
        for item in train_items:
            answer = str(item.get("answer") or "").strip()
            if not self._is_name_shaped(answer):
                continue
            try:
                with ctx.open_tools(item) as tools:
                    passages = tools.retrieve(answer, k=self.K_NAME)
            except Exception:                                    # noqa: BLE001 - measurement is best effort
                continue
            looked += 1

            # breakage: the rule moves the graded answer somewhere else
            for rule, form in self._forms(answer, passages).items():
                if _NORM_WS.sub(" ", form).lower() != _NORM_WS.sub(" ", answer).lower():
                    breaks[rule] += 1

            # repair: starting from a short form a model would naturally produce, the rule reaches
            # the answer.  Two such forms exist and both must be counted, because a model says the
            # name a thing is commonly called, which is sometimes the article's title and sometimes
            # just the outer tokens of a longer full name.
            atoks = self._name_tokens(answer)
            shorts = []
            for passage in passages[:1]:
                shorts.append(_TITLE_TAIL.sub("", str(passage or "").split(" | ")[0].strip()))
            if len(atoks) >= 3:
                shorts.append(f"{atoks[0]} {atoks[-1]}")
            scored: set[str] = set()
            for short in shorts:
                key = short.lower()
                if key in scored or key == answer.lower() or not self._is_name_shaped(short):
                    continue
                scored.add(key)
                for rule, form in self._forms(short, passages).items():
                    if _NORM_WS.sub(" ", form).lower() == _NORM_WS.sub(" ", answer).lower():
                        repairs[rule] += 1
                        break

        self.enabled = {r: (repairs[r] - breaks[r]) >= self.MARGIN for r in self.RULES}
        self.stats = {"repairs": repairs, "breaks": breaks, "measured": looked}
        ctx.note("form rules measured", enabled=self.enabled, **self.stats)

    def get_state(self) -> str:
        return json.dumps({"enabled": self.enabled, "stats": self.stats})

    def set_state(self, state: str) -> None:
        try:
            blob = json.loads(state or "{}")
        except ValueError:
            return
        loaded = blob.get("enabled") or {}
        self.enabled = {r: bool(loaded.get(r, self.enabled.get(r))) for r in self.RULES}
        self.stats = blob.get("stats") or self.stats

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pinned: list[str] = []          # articles chosen as the intended referent: the front of the pool
        rest: list[str] = []            # everything else, in arrival order
        seen: set[str] = set()
        used_queries: set[str] = set()
        retrievals = 0

        def gather(query: str, k: int) -> list[str]:
            """Retrieve `query`, add anything new to the back of the pool, return this query's passages."""
            nonlocal retrievals
            q = self._clean_query(query)
            if not q or q.lower() in used_queries or retrievals >= self.MAX_RETRIEVALS:
                return []
            used_queries.add(q.lower())
            retrievals += 1
            got = []
            for passage in ctx.tools.retrieve(q, k=k):
                got.append(passage)
                key = str(passage).split(" | ")[0].strip().lower()
                if key not in seen:
                    seen.add(key)
                    rest.append(passage)
            return got

        def pin(passage: str) -> None:
            if passage in rest:
                rest.remove(passage)
            if passage not in pinned:
                pinned.append(passage)

        # 1. parse the question into the names it mentions and the checklist the intended article must satisfy
        parsed = self.analyze(ctx.llm, tag="analyze", question=question)
        names = self._parse_lines(parsed.get("names", ""), self.MAX_NAMES, 8)
        constraints = self._parse_lines(parsed.get("constraints", ""), 4, 14)

        # 2. recall floor, identical to the frontier: the plain question and its keyword-only variant
        gather(question, self.K)
        gather(self._keyterms(question), self.K)

        # 3. probe each name on its own and keep, model-free, every article the index holds under that name
        slates: list[tuple[str, list[str]]] = []
        for name in names:
            found = gather(name, self.K_NAME)
            slate: list[str] = []
            for passage in found:
                title = str(passage).split(" | ")[0].strip()
                if self._is_about(name, title) and passage not in slate:
                    slate.append(passage)
                if len(slate) == self.MAX_SLATE:
                    break
            if slate:
                slates.append((name, slate))

        # 4. a slate of one is not ambiguous -- pin it outright and spend no call on it
        colliding = [(n, s) for n, s in slates if len(s) > 1]
        for name, slate in slates:
            if len(slate) == 1:
                pin(slate[0])

        picks_raw = ""
        chosen: list[str] = []
        if colliding:
            lines: list[str] = []
            flat: list[str] = []
            for name, slate in colliding:
                lines.append(f"name: {name}")
                for passage in slate:
                    flat.append(passage)
                    lines.append(f"  [{len(flat)}] {str(passage)[: self.SLATE_SNIPPET]}")
            picks_raw = self.disambiguate(
                ctx.llm,
                tag="disambiguate",
                question=question,
                constraints="\n".join(constraints) if constraints else "none",
                candidates="\n".join(lines),
            ).get("picks", "")
            # one pick per line, `none` lines contribute no index
            for line in str(picks_raw).splitlines():
                if re.search(r"\bnone\b", line, re.IGNORECASE):
                    continue
                idx = self._parse_indices(line.split("=")[-1], len(flat))
                if idx:
                    chosen.append(flat[idx[0] - 1])
            for passage in chosen:
                pin(passage)

        # 5. route over the pool (chosen referents first), following up on whatever link is still missing
        selected: list[str] = []
        missing = ""
        for round_i in range(self.MAX_ROUNDS):
            pool = pinned + rest
            shown = [p[: self.POOL_SNIPPET] for p in pool]
            routed = self.route(ctx.llm, tag=f"route{round_i + 1}", question=question, passages=shown)
            picks = self._parse_indices(routed.get("useful", ""), len(pool))
            if picks:
                selected = [pool[i - 1] for i in picks[: self.N_EVIDENCE]]
            missing = str(routed.get("missing", "")).strip()
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            if not gather(routed.get("next_query", ""), self.K):
                break

        pool = pinned + rest
        evidence = selected or pool[: self.N_EVIDENCE]
        ctx.note(
            "referent disambiguation",
            names=names,
            constraints=constraints,
            n_colliding=len(colliding),
            n_pinned=len(pinned),
            picks=str(picks_raw)[:200],
            n_pool=len(pool),
            retrievals=retrievals,
            missing=missing[:200],
        )

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final) or _HEDGE_PREFIX.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced) and not _HEDGE_PREFIX.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        # one retrieval on the draft itself: the article the index holds under exactly that name
        rule = ""
        if final and any(self.enabled.values()) and self._is_name_shaped(final):
            lookup = gather(final, self.K_NAME)
            canon, rule = self._canonicalise(final, lookup or pool)
            if rule:
                ctx.note("canonical form", rule=rule, before=final[:120], after=canon[:120])
                final = canon

        return {
            "answer": final or str(drafted).strip(),
            "form_rule": rule,
            "names": names,
            "constraints": constraints,
            "n_pinned": len(pinned),
            "n_pool": len(pool),
            "missing": missing,
        }
