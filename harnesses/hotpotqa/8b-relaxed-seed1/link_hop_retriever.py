"""Link-graph retrieval: bare-title probes + deterministic mention mining + parallel link following.

Diagnosis of the current best system (94/300 search-set items never retrieve both required documents,
and those items score 0.46 / 0.00 instead of 0.82):

  * every query mixes the ENTITY with the PROPERTY being asked about ("Ligue 1 sponsor company",
    "Monaco Ligue 1 sponsor").  A word-matching retriever scores the property words too, so the short
    article that is literally titled after the entity gets diluted away by long articles that happen to
    talk about sponsorship.  The document is never seen even though its title is known.
  * the second hop is ONE reformulated query.  If the reformulation guesses the wrong bridge, the item is
    lost; there is no way to hedge across several plausible bridges.

This harness replaces query reformulation with GRAPH TRAVERSAL over the corpus:

    1. anchor   the question is turned into bare article-TITLE probes (names only, never a property,
                never a question word) -- one per document the question presupposes.
    2. mine     candidate links are extracted DETERMINISTICALLY from the abstracts already retrieved:
                every proper-name span an abstract mentions is a title that probably exists in the corpus.
                No LLM guessing about wording is involved, so the second hop cannot be lost to phrasing.
    3. traverse the model picks WHICH mentioned names to open (several at once, not one), and each is
                retrieved by its bare name.  Hedging across links is affordable because the retrieval
                budget is barely used today (4.5 of 12 calls).
    4. repeat   mining from the newly opened abstracts gives a third hop when the pool is still short.
    5. answer   directly from the raw pooled abstracts.

Passages are never summarised and never discarded: the pool is title-keyed and only grows.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module

_BULLET = re.compile(r"^\s*(?:[-*•]|\[?\d+[.)\]]?)\s*")
_INDEX = re.compile(r"\d+")
_MARKUP = re.compile(r"[*_`]+")
_LEADING_FILLER = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|short\s+)?answer(?:\s+is)?\s*[:\-]?\s*", re.IGNORECASE
)
_YESNO_PREFIX = re.compile(r"^\s*(yes|no)\b\s*[,.:;!\-]*", re.IGNORECASE)
_BOOL_QUESTION = re.compile(
    r"^\s*(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE
)
_NON_ANSWER = re.compile(
    r"^(?:none|n/?a|unknown|unclear|no answer|not\s+(?:specified|mentioned|found|available|provided|stated)"
    r"|(?:the\s+)?(?:answer|information)\b.*(?:not|cannot|unavailable).*|cannot\b.*|there is no\b.*)[.!]?$",
    re.IGNORECASE,
)

# A proper-name span: capitalised words, optionally joined by lowercase particles.
_WORD = r"[A-Z][A-Za-z0-9'’&\-]*\.?"
_CONT = r"[A-Z0-9][A-Za-z0-9'’&\-]*\.?"
# A proper-name span: capitalised words, optionally joined by lowercase particles.
_NAME_SPAN = re.compile(
    _CONT + r"(?:\s+(?:of|the|and|for|at|in|de|di|da|del|della|van|von|der|la|le|el|dos|bin|al)(?:\s+the)?\s+"
    + _CONT + r"|\s+" + _CONT + r")*"
)
# The same without particles, so that a list like `A, B and C` also yields its members separately.
_NAME_RUN = re.compile(_CONT + r"(?:\s+" + _CONT + r")*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")
# a trailing date or dangling particle is part of a sentence, not part of a title
_TRAIL = re.compile(
    r"(?:\s+(?:of|the|and|for|at|in|on|de|di|da|del|della|van|von|der|la|le|el)(?:\s+the)?)?"
    r"\s+\d{4}$|(?:\s+(?:of|the|and|for|at|in|on|de|van|von))+$",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"“]([^\"”]{2,60})[\"”]")
# Sentence-initial / grammatical words that start a span without naming anything.
_STOP_HEAD = {
    "the", "a", "an", "he", "she", "it", "they", "this", "that", "these", "those", "his", "her",
    "its", "their", "in", "on", "at", "for", "and", "but", "as", "by", "from", "with", "after",
    "before", "during", "when", "while", "there", "here", "who", "which", "what", "however",
    "also", "both", "some", "other", "one", "two", "later", "then", "after", "since", "until",
    "much", "many", "most", "several", "all", "each", "more", "less", "first", "second", "third",
    "another", "such", "only", "part", "new", "old", "its", "began", "born", "although",
    "based", "written", "released", "founded", "located", "following", "according", "produced",
    "directed", "known", "set", "score", "starring", "originally", "currently", "together",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
}
_LEAD_PARTICLE = {"of", "and", "for", "at", "in", "on", "to", "by", "with", "from", "de", "van", "von"}
_STOP_WHOLE = {
    "he", "she", "it", "they", "the", "a", "an", "his", "her", "its", "their", "this", "that",
    "american", "british", "english", "french", "german", "italian", "spanish", "canadian",
    "australian", "indian", "japanese", "chinese", "russian", "united states", "u.s.", "us",
    "world war ii", "world war i", "however", "although",
}

ANCHOR_INSTRUCTIONS = """A question can only be answered by combining several source documents. Name the
documents.

Write one line per document you are confident exists, at most 4 lines, no numbering and no explanation.
Each line must be a bare TITLE: the name of the person, work, place, organisation, team, event or concept
itself, exactly as an encyclopedia would title the article about it.

Rules:
- Names only. Never include the property being asked about (no `sponsor`, `nationality`, `release date`,
  `founder`, `population`), never a question word, never a verb, never a preposition phrase.
- If the question describes an entity without naming it, write the closest naming phrase it does give,
  plus a disambiguating word the title itself would carry (a year, a medium, a country).
- Write the entity that the question names first; add a separate line for any second entity it names.
- Do not invent facts you are unsure of: an unfamiliar name copied from the question is better than a guess."""

TRAVERSE_INSTRUCTIONS = """You are opening encyclopedia articles by following the names they mention.

You get the question, the abstracts already collected, and a numbered list of NAMES that those abstracts
mention but whose own article has not been opened yet.

In `reasoning`, state the chain of facts the question needs and which link of the chain is still missing.

Produce:
- `useful`: numbers of the collected passages that state a needed fact, most important first, comma
  separated; `none` if no passage does.
- `open`: the numbers of the mentioned names whose own article would supply the missing fact, best first,
  comma separated, at most 4. Hedge: if several names could be the right link, list them all rather than
  betting on one. Write `none` only when nothing is missing.
- `probe`: a bare name to look up that is NOT in the list, used only when the missing document is named
  nowhere above; leave empty otherwise. Names only, never a property or a question.
- `sufficient`: `yes` if the collected passages already state every fact needed to answer, otherwise `no`."""

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
    K_ANCHOR = 8          # results per opening title probe
    K_LINK = 6            # results per followed link
    MAX_RETRIEVALS = 10   # of the 12 allowed, leaving headroom
    MAX_ROUNDS = 2        # traversal rounds
    MAX_OPEN = 4          # links followed per round
    SNIPPET = 340
    MAX_LINKS_SHOWN = 28

    def __init__(self) -> None:
        self.anchor = Module(
            inputs=["question"],
            outputs=["titles"],
            instructions=ANCHOR_INSTRUCTIONS,
            descs={"titles": "bare article titles, one per line, at most 4"},
        )
        self.traverse = Module(
            inputs=["question", "passages", "mentioned_names"],
            outputs=["useful", "open", "probe", "sufficient"],
            instructions=TRAVERSE_INSTRUCTIONS,
            descs={
                "passages": "numbered abstracts already collected, each `Title | abstract`",
                "mentioned_names": "numbered names mentioned by those abstracts, not yet opened",
                "useful": "comma separated passage numbers, or `none`",
                "open": "comma separated name numbers to open, at most 4, or `none`",
                "probe": "a bare name not in the list, or empty",
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

    # -- text helpers -------------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        q = _MARKUP.sub("", str(text or "")).strip()
        q = _BULLET.sub("", q).strip().strip('"').strip("«»").strip()
        return " ".join(q.split())[:200]

    def _parse_titles(self, raw: str, question: str) -> list[str]:
        out: list[str] = []
        for line in str(raw or "").splitlines():
            t = self._clean(line)
            if len(t) < 2 or t.lower() in {"none", "titles"} or t.endswith("?"):
                continue
            if t.lower() not in {x.lower() for x in out}:
                out.append(t)
            if len(out) == 4:
                break
        return out or [self._clean(question)]

    @staticmethod
    def _parse_indices(raw: str, n: int, limit: int) -> list[int]:
        picks: list[int] = []
        for tok in _INDEX.findall(str(raw or "")):
            i = int(tok)
            if 1 <= i <= n and i not in picks:
                picks.append(i)
            if len(picks) >= limit:
                break
        return picks

    @staticmethod
    def _split(passage: str) -> tuple[str, str]:
        head, _, body = str(passage).partition(" | ")
        return head.strip(), body.strip()

    def _mine_links(self, passages: list[str], known: set[str]) -> list[str]:
        """Every proper name mentioned by an abstract is a candidate article title."""
        score: dict[str, int] = {}
        form: dict[str, str] = {}
        for p in passages:
            _, body = self._split(p)
            seen_here: set[str] = set()
            cands: list[str] = [m.group(1) for m in _QUOTED.finditer(body)]
            for sent in _SENTENCE.split(body):
                cands += [m.group(0) for m in _NAME_SPAN.finditer(sent)]
                cands += [m.group(0) for m in _NAME_RUN.finditer(sent)]
            for raw in cands:
                name = _TRAIL.sub("", raw.strip().strip(".,;:'’").strip()).strip()
                words = name.split()
                if not words or len(name) < 3 or len(name) > 60:
                    continue
                while words and words[0].lower().strip(".,") in _STOP_HEAD | _LEAD_PARTICLE:
                    words = words[1:]
                name = " ".join(words)
                if not words or len(name) < 3:
                    continue
                key = name.lower()
                if key in _STOP_WHOLE or key in known or key in seen_here:
                    continue
                if not any(c.isalpha() for c in name):
                    continue
                seen_here.add(key)
                score[key] = score.get(key, 0) + 1
                form.setdefault(key, name)
        ranked = sorted(score, key=lambda k: (-score[k], -len(k.split()), k))
        # a fragment that only ever occurs inside a longer candidate carries no extra information
        out: list[str] = []
        for k in ranked:
            if any(k != o and score[k] <= score[o] and (" " + k + " ") in (" " + o + " ") for o in ranked):
                continue
            out.append(form[k])
            if len(out) >= self.MAX_LINKS_SHOWN:
                break
        return out

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

    # -- main ---------------------------------------------------------------------------------
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        titles: set[str] = set()          # lowercase titles already in the pool
        opened: set[str] = set()          # lowercase queries already issued
        retrievals = 0

        def gather(query: str, k: int) -> int:
            nonlocal retrievals
            q = self._clean(query)
            if not q or q.lower() in opened or retrievals >= self.MAX_RETRIEVALS:
                return 0
            opened.add(q.lower())
            retrievals += 1
            added = 0
            for passage in ctx.tools.retrieve(q, k=k):
                head, _ = self._split(passage)
                key = head.lower()
                if key in titles:
                    continue
                titles.add(key)
                pool.append(passage)
                added += 1
            return added

        # 1. bare-title anchor probes for the documents the question presupposes
        raw_titles = self.anchor(ctx.llm, tag="anchor", question=question).get("titles", "")
        anchors = self._parse_titles(raw_titles, question)
        for t in anchors:
            gather(t, self.K_ANCHOR)
        if not pool:                       # anchors were unusable: fall back to the literal question
            gather(question, self.K_ANCHOR)

        # 2-4. follow the names the abstracts mention
        selected: list[str] = []
        followed: list[str] = []
        for round_i in range(self.MAX_ROUNDS):
            links = self._mine_links(pool, titles | opened)
            shown = [p[: self.SNIPPET] for p in pool]
            routed = self.traverse(
                ctx.llm, tag=f"traverse{round_i + 1}",
                question=question, passages=shown, mentioned_names=links or ["none"],
            )
            picks = self._parse_indices(routed.get("useful", ""), len(pool), 8)
            if picks:
                selected = [pool[i - 1] for i in picks]
            sufficient = "yes" in str(routed.get("sufficient", "")).strip().lower()[:4]
            if sufficient or round_i == self.MAX_ROUNDS - 1 or retrievals >= self.MAX_RETRIEVALS:
                break
            chosen = [links[i - 1] for i in self._parse_indices(routed.get("open", ""), len(links), self.MAX_OPEN)]
            probe = self._clean(routed.get("probe", ""))
            if probe and len(probe) > 2 and not probe.lower().startswith("none"):
                chosen.append(probe)
            grew = 0
            for name in chosen:
                grew += gather(name, self.K_LINK)
                followed.append(name)
            if not grew:
                break

        evidence = selected or pool[:10]
        ctx.note("links followed", n_pool=len(pool), n_selected=len(selected),
                 retrievals=retrievals, anchors=anchors[:4], followed=followed[:8])

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:14]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        return {"answer": final or str(drafted).strip(), "n_pool": len(pool), "anchors": anchors, "followed": followed}
