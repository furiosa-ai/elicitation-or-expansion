"""A second hop that is a CHAIN, not a single extra query, carried as a compact fact ledger.

Fifty-two of the frontier's ninety-five errors are missing at least one gold article, and its
accuracy where both are present is 80% against 40% where one is missing, so the gap is worth more
than anything else on the board.  Five previous attempts bought recall and every one of them lost
accuracy, and they failed the same way: whatever they found -- deeper ranks, linked mentions, wide
title scans -- was poured back into the SAME flat pool of leads that the answering call already
struggles to read, so recall went up and precision of attention went down.

The trace says the mechanism that is actually missing is different.  The frontier does exactly one
follow-up: after triage reports the pool insufficient it invents a single query, drops the hits into
the same pool, triages once more and stops -- there is no third look, and the one query it does get
is written against the leads rather than against anything that has been RESOLVED.  A two-hop bridge
whose first hop itself needs resolving therefore cannot be walked: the query for hop two has to name
an entity that only hop one can supply.

So this harness replaces the single follow-up with a sequential chain.  When, and only when, triage
reports the pool insufficient, it runs up to three rounds; each round issues one query, shows the
model ONLY the handful of leads that query returned, and asks it for one thing: a single resolved
fact, written as one line naming concrete entities, plus the id of the lead that fact came from and
the next query -- which can now name what the previous round resolved.  The rounds carry a ledger of
those one-line facts, never a growing pool, so the state the chain accumulates costs a line each
instead of a passage each.  Only the cited leads are promoted into the final answering context, and
the ledger is handed to the answering call as its own field beside the parent's evidence.

Recall therefore grows along a path the question's own wording cannot reach, while the answering
context grows by at most three passages and three lines.  Items whose pool triage already calls
sufficient never enter the chain and are byte-identical to the parent.

Falsifiable: if the bridge misses survive because the second hop is one shot rather than a walk,
chaining resolved facts should convert them; if the real obstacle is that the answering call cannot
use extra material however compactly it arrives, recall will rise again and accuracy will not.
"""

from __future__ import annotations

import json
import math
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
    "named", "known", "called", "same", "other", "another", "any", "between", "about", "into",
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


def _words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in STOP]


def _sentences(passage: str) -> list[str]:
    _, _, body = passage.partition(" | ")
    out = []
    for sent in SENT_SPLIT.split(body.strip()):
        sent = sent.strip()
        if 20 <= len(sent) <= 320:
            out.append(sent)
        elif len(sent) > 320:
            out.append(sent[:320].rsplit(" ", 1)[0])
    return out[:6]


def _weights(question: str, linking_fact: str, corpus: list[str]) -> dict[str, float]:
    """Rarity weight, inside the retrieved pool, of every content word the question cares about."""
    df: dict[str, int] = {}
    for passage in corpus:
        for w in set(_words(passage)):
            df[w] = df.get(w, 0) + 1
    n = max(1, len(corpus))
    weights: dict[str, float] = {}
    for w in _words(question):
        weights[w] = math.log(1.0 + n / (1.0 + df.get(w, 0)))
    for w in _words(linking_fact):
        weights[w] = max(weights.get(w, 0.0), 0.75 * math.log(1.0 + n / (1.0 + df.get(w, 0))))
    return weights


# -- answer-form machinery ---------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9\u00c0-\u017f'\u2019\-]+")
PARTICLE = {"of", "the", "de", "del", "der", "den", "di", "da", "van", "von", "la", "le", "el"}
DETERMINER = {"the", "a", "an"}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}
NUMBER_WORD = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty",
}
QUAL_RE = re.compile(r"^\s*,\s+([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,2})")
HEAD_NOUN_RE = re.compile(r"^\s+([a-z]{3,})")
PLACE_WORDS = {"where", "city", "town", "country", "state", "county", "located", "based", "born",
               "province", "village", "headquartered"}
COUNT_WORDS = {"many", "number"}


def _ntoks(text: str) -> list[str]:
    return [normalize_text(t) for t in TOKEN_RE.findall(text or "") if normalize_text(t)]


def _name_like(text: str) -> bool:
    raw = TOKEN_RE.findall(text or "")
    if not 2 <= len(raw) <= 5 or any(ch.isdigit() for ch in text):
        return False
    return all(t[:1].isupper() or t.lower() in PARTICLE for t in raw)


def _opening_run(body: str) -> str:
    """The name an article's own lead opens with, e.g. `Thomas Lance Rentzel`."""
    first = SENT_SPLIT.split((body or "").strip())[0]
    head = re.split(r"[(\"\u201c]", first)[0]
    run: list[str] = []
    for tok in TOKEN_RE.findall(head):
        if tok[:1].isupper() or (run and tok.lower() in PARTICLE):
            run.append(tok)
        else:
            break
    return " ".join(run).strip(" ,")


def _is_expansion(short: str, long_: str) -> bool:
    """`long_` names the same thing as `short` with up to three extra name tokens threaded in."""
    a, b = _ntoks(short), _ntoks(long_)
    if not a or len(b) <= len(a) or len(b) - len(a) > 3:
        return False
    extra = [t for t in b if t not in a]
    if not extra or any(t in MONTHS or t in DETERMINER for t in extra):
        return False
    for tok in TOKEN_RE.findall(long_):
        if tok.isupper() and len(tok) > 1 and normalize_text(tok) in extra:
            return False          # an acronym gloss, not a fuller name
    i = 0
    for t in b:
        if i < len(a) and t == a[i]:
            i += 1
    return i == len(a)


def _find(text: str, phrase: str) -> int:
    """Case-insensitive whole-token occurrence of `phrase` in `text`; -1 if absent."""
    if not phrase:
        return -1
    pat = r"\b" + r"\W+".join(re.escape(t) for t in TOKEN_RE.findall(phrase)) + r"\b"
    m = re.search(pat, text, re.IGNORECASE)
    return m.start() if m else -1


def _span_end(text: str, phrase: str, start: int) -> int:
    pat = r"\b" + r"\W+".join(re.escape(t) for t in TOKEN_RE.findall(phrase)) + r"\b"
    m = re.search(pat, text[start:], re.IGNORECASE)
    return start + m.end() if m else -1


def _is_count(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\d[\d,]*", t)) or t.lower() in NUMBER_WORD


def _singular(word: str) -> str:
    w = word
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


class Harness(BaseHarness):
    K_PLAN = 8          # hits per query the parent pipeline reads (unchanged)
    K_DEEP = 20         # depth every query is now issued at; ranks K_PLAN.. feed the tail only
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    DIGEST_LINES = 16   # sentences distilled out of the deep tail
    PER_TITLE = 2       # at most this many sentences from one article
    MIN_HITS = 2        # a sentence must match at least this many distinct question words

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
        "`next_query`   : if not sufficient, one keyword search query for the missing fact, naming "
        "the entity from `linking_fact` explicitly; otherwise leave it empty."
    )
    ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "`extra_leads` holds single sentences pulled from further down the search results, one per "
        "line as `title -- sentence`. They are unverified: use one only when the evidence passages "
        "do not settle the question, and never prefer one over a passage that does.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    # -- the chain -----------------------------------------------------------------------------
    MAX_CHAIN = 3        # rounds of resolve-then-query, entered only when triage says insufficient
    K_CHAIN = 6          # leads shown per round -- the round sees these and nothing else
    MAX_PROMOTED = 3     # cited leads allowed into the final answering context
    STEP_INSTR = (
        "You are walking a chain of look-ups towards the facts a question needs.\n"
        "`ledger` holds what earlier rounds already resolved, one line each; `new_leads` holds the "
        "encyclopaedia leads the latest search returned, as `id: title | beginning of the article`. "
        "Read only these.\n"
        "`fact`       : one line stating the single most useful thing `new_leads` establishes for "
        "the question, naming its entities explicitly so a later round can search for them; write "
        "`none` if the leads establish nothing useful.\n"
        "`cite`       : the id of the lead `fact` came from, or empty.\n"
        "`status`     : `answered` if the ledger plus these leads now settle the question, "
        "otherwise `continue`.\n"
        "`next_query` : if not answered, one short keyword search query for the next missing fact. "
        "Name the entity the ledger has just resolved instead of describing it; do not repeat a "
        "query whose results you have already seen."
    )
    CHAIN_ANSWER_INSTR = (
        "Answer the question from the evidence passages.\n"
        "`facts` holds the lines a chain of look-ups resolved, earliest first; they are about the "
        "parts of the question the evidence passages do not cover. Trust a passage over a line "
        "wherever the two disagree.\n"
        "`extra_leads` holds single sentences pulled from further down the search results, one per "
        "line as `title -- sentence`. They are unverified: use one only when the evidence passages "
        "and `facts` do not settle the question.\n"
        "Write `answer` as the shortest span that answers it: a bare name, title, number, date or "
        "noun phrase, copied from the evidence. Never write a sentence or an explanation.\n"
        "If the question asks whether something holds, `answer` must be exactly `yes` or `no`.\n"
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )


    # -- policy fitted on labelled training answers ---------------------------------------------
    MIN_OBS = 4          # a family needs this many training observations before it may fire
    MIN_SHARE = 0.6      # and this share of them must back the rewrite

    def __init__(self) -> None:
        self.policy: dict[str, bool] = {}
        self.votes: dict[str, int] = {}
        Harness._init_modules(self)

    def _vote(self, answer: str, articles: list[tuple[str, str]]) -> None:
        """Count, on ONE labelled item, which form the label took where both were available."""
        body_all = " ".join(b for _, b in articles)
        # (1) short title vs long lead: which one does a label use?
        if _name_like(answer):
            for title, body in articles:
                run = _opening_run(body)
                if normalize_text(title) == normalize_text(answer) and _is_expansion(answer, run):
                    self.votes["name_keep"] = self.votes.get("name_keep", 0) + 1
                elif normalize_text(run) == normalize_text(answer) and _is_expansion(title, answer):
                    self.votes["name_expand"] = self.votes.get("name_expand", 0) + 1
        # (2) trailing `, Qualifier`
        head = re.match(r"^(.+?), ([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*){0,2})$", answer or "")
        pos = _find(body_all, answer)
        if head and pos >= 0:
            self.votes["qual_attach"] = self.votes.get("qual_attach", 0) + 1
        elif not head and pos >= 0 and answer[:1].isupper() and len(answer.split()) <= 4:
            end = _span_end(body_all, answer, pos)
            if end > 0 and QUAL_RE.match(body_all[end:]):
                self.votes["qual_keep"] = self.votes.get("qual_keep", 0) + 1
        # (3) bare count vs count plus its head noun
        both = re.match(r"^(\d[\d,]*|[A-Za-z]+)\s+([a-z]{3,})$", (answer or "").strip())
        if both and _is_count(both.group(1)) and _find(body_all, answer) >= 0:
            self.votes["noun_attach"] = self.votes.get("noun_attach", 0) + 1
        elif _is_count(answer):
            pos = _find(body_all, answer)
            end = _span_end(body_all, answer, pos) if pos >= 0 else -1
            if end > 0 and HEAD_NOUN_RE.match(body_all[end:]):
                self.votes["noun_keep"] = self.votes.get("noun_keep", 0) + 1
        # (4) a common noun that the source offers in both numbers
        word = (answer or "").strip()
        if word.islower() and len(word.split()) == 1 and len(word) > 3:
            if word.endswith("s") and _singular(word) != word and _find(body_all, _singular(word)) >= 0:
                self.votes["plural_keep"] = self.votes.get("plural_keep", 0) + 1
            elif not word.endswith("s") and _find(body_all, word + "s") >= 0:
                self.votes["singularise"] = self.votes.get("singularise", 0) + 1

    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        for item in train_items:
            answer = (item.get("answer") or "").strip()
            if not answer or normalize_text(answer) in ("yes", "no"):
                continue
            articles: list[tuple[str, str]] = []
            try:
                with ctx.open_tools(item) as tools:
                    for title in (item.get("gold_titles") or [])[:2]:
                        for passage in tools.retrieve(title, k=3):
                            got, _, body = passage.partition(" | ")
                            if normalize_text(got) == normalize_text(title):
                                articles.append((got.strip(), body.strip()))
                                break
            except Exception:            # a training item that cannot be fetched simply abstains
                continue
            if articles:
                self._vote(answer, articles)
        for rewrite, keep in (("name_expand", "name_keep"), ("qual_attach", "qual_keep"),
                              ("noun_attach", "noun_keep"), ("singularise", "plural_keep")):
            won = self.votes.get(rewrite, 0)
            lost = self.votes.get(keep, 0)
            total = won + lost
            self.policy[rewrite] = total >= self.MIN_OBS and won >= self.MIN_SHARE * total
        ctx.note("answer-form policy fitted", votes=self.votes, policy=self.policy)

    def get_state(self) -> str:
        return json.dumps({"policy": self.policy, "votes": self.votes})

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state or "{}")
        except ValueError:
            data = {}
        self.policy = {k: bool(v) for k, v in (data.get("policy") or {}).items()}
        self.votes = data.get("votes") or {}

    # -- run-time projection onto the source sentence -------------------------------------------
    def _host_sentence(self, question: str, draft: str, evidence: list[str],
                       digest: list[str]) -> str:
        """The sentence the draft most plausibly came from: it contains it and matches the question."""
        corpus = list(evidence) + [f"{line} " for line in digest]
        weights = _weights(question, "", evidence or corpus)
        best, best_score = "", -1.0
        for passage in corpus:
            for sent in _sentences(passage):
                if _find(sent, draft) < 0:
                    continue
                score = sum(weights.get(w, 0.0) for w in set(_words(sent)) & set(weights))
                if score > best_score:
                    best, best_score = sent, score
        return best

    def _project(self, question: str, draft: str, evidence: list[str],
                 digest: list[str]) -> tuple[str, str]:
        """Rewrite the draft into the form the training labels back. Deterministic, no LLM call."""
        if not draft or normalize_text(draft) in ("yes", "no"):
            return draft, ""
        qwords = set(_words(question)) | {w for w in re.findall(r"[a-z]+", question.lower())}
        # (1) the answer's own article opens with a fuller name
        if _name_like(draft):
            for passage in evidence:
                title, _, body = passage.partition(" | ")
                if normalize_text(title.strip()) != normalize_text(draft):
                    continue
                run = _opening_run(body)
                if run and _is_expansion(draft, run):
                    if self.policy.get("name_expand"):
                        return run, "name_expand"
                    # votes inconclusive: expand only where this article's own lead is the draft's
                    # only home in the evidence. If another article refers to the thing by the short
                    # name, the short name is how the sources themselves write it, so keep it.
                    elsewhere = any(
                        _find(other.partition(" | ")[2], draft) >= 0
                        for other in evidence
                        if normalize_text(_title_of(other)) != normalize_text(draft)
                    )
                    if not elsewhere:
                        return run, "name_expand_unique"
                break
        host = self._host_sentence(question, draft, evidence, digest)
        if not host:
            return draft, ""
        pos = _find(host, draft)
        end = _span_end(host, draft, pos) if pos >= 0 else -1
        if end > 0:
            tail = host[end:]
            # (2) the source prints a qualifier right behind it and the question asks for a place
            if self.policy.get("qual_attach") and qwords & PLACE_WORDS:
                m = QUAL_RE.match(tail)
                if m and not re.match(r"^(?:.+?), (?:[A-Z])", draft):
                    return f"{draft}, {m.group(1)}", "qual_attach"
            # (3) the source prints a head noun right behind a bare count
            if self.policy.get("noun_attach") and _is_count(draft) and qwords & COUNT_WORDS:
                m = HEAD_NOUN_RE.match(tail)
                if m:
                    return f"{draft} {m.group(1)}", "noun_attach"
        # (4) the source names the kind of thing in the singular
        if self.policy.get("singularise") and len(draft.split()) == 1 and draft.islower():
            single = _singular(draft)
            if single != draft and _find(host, single) >= 0:
                return single, "singularise"
        return draft, ""

    def _init_modules(self) -> None:
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
        self.answer = Module(
            inputs=["question", "evidence", "linking_fact", "extra_leads"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
        )
        self.step = Module(
            inputs=["question", "ledger", "new_leads"],
            outputs=["fact", "cite", "status", "next_query"],
            instructions=self.STEP_INSTR,
        )
        self.chain_answer = Module(
            inputs=["question", "evidence", "facts", "linking_fact", "extra_leads"],
            outputs=["answer"],
            instructions=self.CHAIN_ANSWER_INSTR,
        )

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

    def _deep(self, ctx: RunContext, query: str, pool: list[str], seen: set[str],
              tail: list[str], tail_seen: set[str], k_read: int) -> None:
        """One query, issued at full depth: the head feeds the parent's pool, the rest the tail."""
        hits = ctx.tools.retrieve(query, k=self.K_DEEP)
        self._add(pool, seen, hits[:k_read])
        for p in hits[k_read:]:
            key = _title_of(p).lower()
            if key and key not in tail_seen:
                tail_seen.add(key)
                tail.append(p)

    def _digest(self, question: str, linking_fact: str, tail: list[str],
                read: set[str]) -> list[str]:
        """Rank every tail sentence by pool-rarity overlap with the question; keep the best few."""
        corpus = [p for p in tail if _title_of(p).lower() not in read]
        if not corpus:
            return []
        weights = _weights(question, linking_fact, corpus)
        if not weights:
            return []
        qwords = set(weights)
        scored: list[tuple[float, str, str]] = []
        for passage in corpus:
            title = _title_of(passage)
            title_hit = 0.5 * sum(weights.get(w, 0.0) for w in set(_words(title)) & qwords)
            for sent in _sentences(passage):
                hit = set(_words(sent)) & qwords
                if len(hit) < self.MIN_HITS:
                    continue
                score = sum(weights[w] for w in hit) + title_hit
                score /= 1.0 + 0.004 * len(sent)
                scored.append((score, title, sent))
        scored.sort(key=lambda s: -s[0])
        per: dict[str, int] = {}
        out: list[str] = []
        for score, title, sent in scored:
            if per.get(title, 0) >= self.PER_TITLE:
                continue
            per[title] = per.get(title, 0) + 1
            out.append(f"{title} -- {sent}")
            if len(out) >= self.DIGEST_LINES:
                break
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]
        pool: list[str] = []
        seen: set[str] = set()
        tail: list[str] = []
        tail_seen: set[str] = set()

        # 1. decompose (unchanged) --------------------------------------------------------------
        queries = _split_lines(
            self.plan(ctx.llm, tag="plan", question=question).get("queries", ""), self.MAX_QUERIES
        )

        # 2. every query at full depth: head -> parent pool, ranks below -> deep tail -----------
        self._deep(ctx, question, pool, seen, tail, tail_seen, self.K_PLAN)
        for q in queries:
            self._deep(ctx, q, pool, seen, tail, tail_seen, self.K_PLAN)

        # 3. one triage over the parent's pool -------------------------------------------------
        shown = pool[: self.POOL_CAP]
        table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
        out = self.triage(ctx.llm, tag="triage1", question=question, pool=table)
        ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
            range(1, min(len(shown), self.SELECT_CAP) + 1)
        )
        fact = (out.get("linking_fact", "") or "").strip()
        linking_fact = fact if fact and fact.lower() not in ("unknown", "none", "n/a") else ""
        sufficient = bool(AFFIRM.match(out.get("sufficient", "")))
        next_query = "" if sufficient else (_split_lines(out.get("next_query", ""), 1) or [""])[0]

        # 4. the pool is not enough: walk a chain, carrying one resolved line per round ---------
        ledger: list[str] = []
        promoted: list[str] = []
        asked: set[str] = {normalize_text(question)}
        for round_ in range(0 if sufficient else self.MAX_CHAIN):
            query = (next_query or "").strip()
            if not query or normalize_text(query) in asked:
                break
            asked.add(normalize_text(query))
            hits = ctx.tools.retrieve(query, k=self.K_CHAIN)
            if not hits:
                break
            leads = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(hits)]
            step = self.step(
                ctx.llm,
                tag=f"chain{round_ + 1}",
                question=question,
                ledger=ledger or ["(nothing resolved yet)"],
                new_leads=leads,
            )
            line = (step.get("fact", "") or "").strip().split("\n")[0].strip()
            if len(line) > 8 and normalize_text(line) not in ("none", "unknown", ""):
                ledger.append(line[:260])
                for i in self._parse_ids(step.get("cite", ""), len(hits))[:1]:
                    passage = hits[i - 1]
                    key = _title_of(passage).lower()
                    if key not in seen and len(promoted) < self.MAX_PROMOTED:
                        seen.add(key)
                        promoted.append(passage)
            status = (step.get("status", "") or "").strip().lower()
            ctx.note("chain round", query=query, fact=line[:140], status=status[:20])
            if status.startswith("answer") or AFFIRM.match(status):
                break
            next_query = (_split_lines(step.get("next_query", ""), 1) or [""])[0]

        # 5. answer from the selected passages, the promoted sources and the ledger ------------
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]
        evidence = evidence + promoted
        read = {_title_of(p).lower() for p in evidence}
        digest = self._digest(question, linking_fact, tail, read)
        ctx.note("chain done", rounds=len(ledger), promoted=len(promoted), lines=len(digest))
        if ledger:
            ans = self.chain_answer(
                ctx.llm,
                tag="answer",
                question=question,
                evidence=evidence,
                facts=ledger,
                linking_fact=linking_fact or "unknown",
                extra_leads=digest or ["(none)"],
            ).get("answer", "")
        else:
            ans = self.answer(
                ctx.llm,
                tag="answer",
                question=question,
                evidence=evidence,
                linking_fact=linking_fact or "unknown",
                extra_leads=digest or ["(none)"],
            ).get("answer", "")
        ans = (ans or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()
        projected, family = self._project(question, ans, evidence, digest)
        if family:
            ctx.note("answer form projected", family=family, draft=ans, answer=projected)
        return {
            "answer": projected or ans,
            "draft": ans,
            "form_family": family,
            "queries": queries,
            "ledger": ledger,
            "linking_fact": linking_fact,
            "pool_size": len(pool),
            "tail_size": len(tail),
            "n_digest": len(digest),
            "n_selected": len(evidence),
        }
