"""An answer-anchored retrieval hop: the drafted answer is checked against its OWN article.

Every retrieval in every system so far is anchored on the QUESTION.  That leaves one source
systematically unread: the article about the thing the answer names.  Measured on the parent's 300
traces, 46 of its 97 errors have the gold string sitting verbatim in the very context the answerer
read, and the single largest coherent pattern inside that bucket is that the answer is *named from
the article that mentions it* rather than from the article that is about it (a mentioning sentence
writes a short name, a passing description, or glues two names together with `and`), with a second
pattern where the draft names a neighbouring concept whose own lead does not in fact satisfy the
question.

The mechanism added here is one extra hop whose query is the drafted answer itself:

  * the parent pipeline runs untouched and produces its draft (same prompts, same demonstrations,
    same span alignment), so items the parent gets right stay byte-identical up to this point;
  * the hop is skipped entirely for yes/no answers, for bare numbers and dates, and for drafts that
    no longer look like a name, because those have no article of their own;
  * otherwise the index is queried with nothing but the draft, and the returned leads whose TITLE
    shares a content word with the draft become `pages` -- the answer's own article(s);
  * one adjudication call sees the question, the draft, the sentence the draft was taken from and
    those pages, and either confirms the draft or rewrites it;
  * every rewrite must survive a deterministic filter: it has to occur verbatim inside the pages or
    the source sentence, be at most six words, and not be the draft again.  A rewrite that fails the
    filter is discarded and the draft stands.

So the hop can only ever move the answer onto wording that the answer's own encyclopaedia article
uses.  Cost: one retrieval and one LLM call on the items that reach it.
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



TOK_RE = re.compile(r"[A-Za-z0-9'\u2019]+")
QUOTED_RE = re.compile(r'"([^"]{12,400})"')
# words the answerer may glue onto a span without changing what is being named
FILLER = {
    "about", "approximately", "around", "roughly", "nearly", "some", "over", "under", "the", "a",
    "an", "of", "in", "on", "at", "from", "circa", "almost", "just",
}
# a span that joins the draft's tokens across one of these is naming something else
SPLICE = {"and", "or", "&", "with", "plus", "versus", "vs"}


def _clean(text: str) -> str:
    return (text or "").strip().split("\n")[0].strip().strip("*_`\"").rstrip(".,;: ").strip()


def _tok_match(word: str, want: str) -> bool:
    """`want` matches `word` exactly, or is a shortening of it (Ron -> Ronald)."""
    w, d = word.lower(), want.lower()
    return w == d or (len(d) >= 3 and w.startswith(d)) or (len(w) >= 3 and d.startswith(w))


def _cover_span(sentence: str, draft: str) -> str | None:
    """Shortest contiguous run of `sentence` covering every content token of `draft`, in order."""
    toks = [(m.group(0), m.start(), m.end()) for m in TOK_RE.finditer(sentence)]
    want = TOK_RE.findall(draft)
    core = [t for t in want if t.lower() not in FILLER] or want
    if not toks or not core or len(core) > 8:
        return None
    best = None
    for start in range(len(toks)):
        i, last = 0, None
        for j in range(start, len(toks)):
            if _tok_match(toks[j][0], core[i]):
                i += 1
                last = j
                if i == len(core):
                    break
            elif last is not None and j - last > 2:
                break  # the tokens are too far apart to be one name
        if i == len(core) and last is not None:
            if any(toks[j][0].lower() in SPLICE for j in range(start, last + 1)):
                continue  # the span splices two things together
            span = sentence[toks[start][1] : toks[last][2]]
            if best is None or len(span) < len(best):
                best = span
    return best


def _acceptable(draft: str, span: str) -> bool:
    """The span may only expand the draft or trim filler off it -- never re-inflect or replace it."""
    d, s = TOK_RE.findall(draft), TOK_RE.findall(span)
    if not s or len(s) > 9:
        return False
    if len(s) == len(d):
        return False  # same length means a re-inflection (State -> states), no information gained
    if len(s) > len(d):
        return all(any(_tok_match(w, t) for w in s) for t in d)  # every draft token survives
    dropped = [t for t in d if not any(_tok_match(w, t) for w in s)]
    return bool(dropped) and all(t.lower() in FILLER for t in dropped)


def _align(draft: str, reasoning: str, evidence: list[str]) -> tuple[str, str]:
    """Re-write `draft` as the wording of the sentence that actually states it."""
    if not draft or normalize_text(draft) in ("yes", "no") or len(draft.split()) > 5:
        return draft, ""
    # the parent already writes `reasoning` as the quoted evidence sentence; that quote is the source
    sources: list[str] = list(QUOTED_RE.findall(reasoning or ""))
    for passage in evidence:
        sources.extend(SENT_SPLIT.split(passage))
    for sentence in sources:
        sentence = sentence.strip()
        if len(sentence) < 8:
            continue
        span = _cover_span(sentence, draft)
        if not span:
            continue
        span = span.strip().strip('"\u201c\u201d\u00ab\u00bb').strip()
        if span.lower().endswith("'s") or span.lower().endswith("\u2019s"):
            span = span[:-2].strip()
        if not span or normalize_text(span) == normalize_text(draft):
            return draft, sentence  # already written the way its source writes it
        if _acceptable(draft, span):
            return span, sentence
        return draft, sentence  # the first real occurrence decides; do not shop for a better one
    return draft, ""


# -- answer-anchored hop helpers -----------------------------------------------------------------
NUMERIC_ANS = re.compile(r"^[\d\s.,:/%$\u2013-]+$")
DATE_ANS = re.compile(
    r"^(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b", re.IGNORECASE)


def _has_page(draft: str) -> bool:
    """Only strings that can plausibly have an encyclopaedia article of their own are checked."""
    if not draft or normalize_text(draft) in ("yes", "no"):
        return False
    words = TOK_RE.findall(draft)
    if not words or len(words) > 6:
        return False
    if NUMERIC_ANS.match(draft) or DATE_ANS.match(draft.strip()):
        return False
    return any(w[:1].isupper() or w[:1].isdigit() for w in words)


def _about_draft(passage: str, draft: str) -> bool:
    """True when the passage's TITLE is about the drafted answer rather than merely mentioning it."""
    title = set(WORD_RE.findall(_title_of(passage).lower()))
    want = {w.lower() for w in TOK_RE.findall(draft) if len(w) > 2 and w.lower() not in FILLER}
    if not title or not want:
        return False
    return len(title & want) >= max(1, min(len(want), 2) - 1) and bool(title & want)


def _subsequence(short: list[str], long: list[str]) -> bool:
    i = 0
    for w in long:
        if i < len(short) and _tok_match(w, short[i]):
            i += 1
    return i == len(short)


def _revision_ok(draft: str, revised: str, haystack: str) -> bool:
    """A rewrite is kept only if the answer's own article really writes it that way.

    It must also change WHICH thing is named, never merely how fully the same name is written:
    a revision that just threads extra words around every token of the draft (a middle name, a
    qualifier) is rejected, because the parent's alignment already owns that decision and the
    answering convention is as often the short form as the long one.
    """
    if not revised or len(TOK_RE.findall(revised)) > 6:
        return False
    n_rev, n_draft = normalize_text(revised), normalize_text(draft)
    if not n_rev or n_rev == n_draft:
        return False
    if _subsequence(TOK_RE.findall(draft), TOK_RE.findall(revised)):
        return False  # pure expansion of the same name
    return n_rev in normalize_text(haystack)


class Harness(BaseHarness):
    K_PLAN = 8
    K_FOLLOW = 8
    MAX_QUERIES = 3
    MAX_TRIAGE = 2
    POOL_CAP = 30
    SELECT_CAP = 10

    K_PAGE = 5            # hits for the answer-anchored query
    MAX_PAGES = 3         # answer-own-article leads shown to the adjudicator

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
        "`next_query`   : if not sufficient, one keyword search query for the missing fact, naming "
        "the entity from `linking_fact` explicitly; otherwise leave it empty."
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
        "If the question offers two alternatives, `answer` must be one of them, copied verbatim."
    )

    CONFIRM_INSTR = (
        "A reader has proposed an answer to the question. You are shown `pages`: the encyclopaedia "
        "articles about the proposed answer itself, and `source`: the sentence the reader took it from.\n"
        "Decide two things, in this order.\n"
        "1. Is the proposed answer the thing the question asks for? If `pages` show it is something "
        "else (a different kind of thing, a related concept, a whole rather than the part asked for), "
        "set `verdict` to `revise`.\n"
        "Work through it concretely: name the kind of thing the question asks for (a person, an "
        "aircraft, a company, a river, a war), then read the opening words of each page, which define "
        "what that page is about. If the page about the proposed answer defines a different kind of "
        "thing, and another page here defines the right one, `verdict` is `revise` and `answer` is "
        "that other page's title.\n"
        "2. Does it name the whole of what was asked, and only that? If the reader glued two names "
        "together with `and` where the question asks for one of them, or wrote a description where "
        "`pages` give the thing a name of its own, `verdict` is `revise`.\n"
        "Otherwise set `verdict` to `confirm`. Do not revise merely to write a longer or shorter "
        "version of the same name -- that is not your decision to make.\n"
        "`answer` : the proposed answer unchanged when you confirm, otherwise the corrected span "
        "copied verbatim out of `pages` or `source`. Always a bare name, title or noun phrase, never "
        "a sentence, at most six words. Never invent wording that is not written in `pages` or `source`."
    )

    def __init__(self) -> None:
        self.confirm = Module(
            inputs=["question", "proposed_answer", "source", "pages"],
            outputs=["verdict", "answer"],
            instructions=self.CONFIRM_INSTR,
        )
        self.plan = Module(inputs=["question"], outputs=["queries"], instructions=self.PLAN_INSTR)
        self.triage = Module(
            inputs=["question", "pool"],
            outputs=["relevant_ids", "linking_fact", "sufficient", "next_query"],
            instructions=self.TRIAGE_INSTR,
        )
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
            {k: d[k] for k in ("question", "evidence", "linking_fact", "reasoning", "answer")}
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

        linking_fact, ids, followups = "", [], []
        for step in range(self.MAX_TRIAGE):
            shown = pool[: self.POOL_CAP]
            table = [f"{i + 1}: {_lead(p, 260)}" for i, p in enumerate(shown)]
            out = self.triage(ctx.llm, tag=f"triage{step + 1}", question=question, pool=table)
            ids = self._parse_ids(out.get("relevant_ids", ""), len(shown)) or list(
                range(1, min(len(shown), self.SELECT_CAP) + 1)
            )
            fact = (out.get("linking_fact", "") or "").strip()
            if fact and fact.lower() not in ("unknown", "none", "n/a"):
                linking_fact = fact
            if AFFIRM.match(out.get("sufficient", "")) or step == self.MAX_TRIAGE - 1:
                break
            nq = _split_lines(out.get("next_query", ""), 1)
            if not nq:
                break
            followups.append(nq[0])
            before = len(pool)
            self._add(pool, seen, ctx.tools.retrieve(nq[0], k=self.K_FOLLOW))
            if len(pool) == before:
                break
            pool = [pool[i - 1] for i in ids] + [p for i, p in enumerate(pool, 1) if i not in ids]
            ctx.note("follow-up hop", query=nq[0], pool_size=len(pool))

        shown = pool[: self.POOL_CAP]
        evidence = [shown[i - 1] for i in ids[: self.SELECT_CAP] if i <= len(shown)]
        if not evidence:
            evidence = shown[: self.SELECT_CAP]

        demos = self._demos(question)
        answerer = Module(  # per-item module: demos differ per question and items run concurrently
            inputs=["question", "evidence", "linking_fact"],
            outputs=["answer"],
            instructions=self.ANSWER_INSTR,
            demos=demos,
        )
        out = answerer(
            ctx.llm,
            tag="answer",
            question=question,
            evidence=evidence,
            linking_fact=linking_fact or "unknown",
        )
        draft = _clean(out.get("answer", ""))
        ans, source = _align(draft, out.get("reasoning", ""), evidence)
        if ans != draft:
            ctx.note("answer aligned to source wording", draft=draft, aligned=ans, source=source[:200])

        # -- answer-anchored hop: read the article about the answer itself ---------------------
        aligned, pages = ans, []
        if _has_page(ans):
            hits = ctx.tools.retrieve(ans, k=self.K_PAGE)
            pages = [_lead(p, 320) for p in hits if _about_draft(p, ans)][: self.MAX_PAGES]
        if pages:
            if not source:
                source = _answer_sentence(evidence[0], ans) if evidence else ""
            ver = self.confirm(
                ctx.llm,
                tag="confirm",
                question=question,
                proposed_answer=ans,
                source=source[:300] or "(not recorded)",
                pages=pages,
            )
            revised = _clean(ver.get("answer", ""))
            if "revise" in (ver.get("verdict", "") or "").lower():
                if _revision_ok(ans, revised, " \n ".join(pages) + " \n " + source):
                    ctx.note("answer revised against its own article", before=ans, after=revised)
                    ans = revised
                else:
                    ctx.note("revision rejected", before=ans, proposed=revised[:120])
        return {
            "answer": ans,
            "aligned": aligned,
            "n_pages": len(pages),
            "draft": draft,
            "queries": queries,
            "followups": followups,
            "linking_fact": linking_fact,
            "family": _family(question),
            "n_demos": len(demos),
            "n_selected": len(evidence),
        }
