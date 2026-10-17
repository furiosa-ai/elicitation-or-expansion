"""Queries are compiled from structured fields under a policy measured on the labelled train split.

Diagnosis of the parent (`decomposed_evidence_pool`, 66.3%). Accuracy is almost entirely decided by
how much gold evidence reaches the context: 0.817 when both gold articles are retrieved (180 items),
0.51 when one is (96), 0.125 when neither is (24). Attributing every retrieved title to its hop
shows the loss is lopsided: when exactly one gold article is found, it is the FIRST hop's target
74 times and the SECOND hop's target only 22 times. The lookup for the entity the question names
lands almost always; the lookup for the entity that has to be identified first is what misses.

Two things go wrong there, and both come from letting a module emit a free-text query string:
  - the string is a sentence-like blend of the resolved name and leftover question words, so the
    sparse index scores articles that merely share vocabulary with the question;
  - nobody knows how much qualifying text an entity-name query should carry. The parent's prose
    asserts "two or three disambiguating content words". That is a guess, and it is the one thing
    in this pipeline that can be *measured* rather than asserted -- the train split is labelled
    with the very article titles the index is supposed to return.

Mechanism under test (query construction; the number of calls, the number of retrievals, k, the
evidence pool and the answer contract are all held identical to the parent):

  STRUCTURED QUERY FIELDS + A COMPILER WITH A TRAIN-MEASURED POLICY.
  1. `prepare` runs no LLM at all. For each labelled train item it takes the target article that
     the question does not name -- the one a second hop would have to find -- and probes the index
     twice with the two competing query shapes for the same entity: the bare name, and the name
     carrying qualifying words. It records which shape returns the target article at k=7. The
     paired win rates become the compilation policy, and the measured counts are written into the
     instructions so the modules are told what was actually observed, not what someone assumed.
  2. At run time no module writes a query. The planner emits `entity_a` / `qualifier_a` and a
     description of the second target; the resolver emits `entity_b` / `qualifier_b`, where
     `entity_b` is the name it has just read out of the first hop's passages. A deterministic
     compiler assembles the index query from those fields under the measured policy, strips
     question words and punctuation that survived into the name field, and falls back to the
     planned description when the resolver failed to name anything.

Nothing about any evaluation item is used; the policy is a single boolean plus counts learned from
train labels.

Budget: 3 LLM calls, 2 retrievals, k=7 -- inside the interaction depth. `prepare` uses two
retrievals per train item and no LLM calls, so it costs seconds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "did", "do", "does",
    "that", "this", "these", "those", "with", "by", "from", "as", "it", "its", "his", "her", "their",
    "both", "same", "more", "most", "first", "than", "then", "also", "has", "have", "had", "been",
    "be", "not", "he", "she", "they", "you", "we", "i", "one", "two", "other", "another", "there",
}
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
_QUESTION_WORDS = re.compile(
    r"\b(which|who|whom|whose|what|when|where|why|how|did|does|do|is|are|was|were|both|same|"
    r"more|most|first|the|a|an|of|in|on|at|to|for|and|or|that|this|by|from|as)\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(text or "")) if w.lower() not in _STOP]


def _strip_paren(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(title or "").replace("&amp;", "&")).strip()


def _paren(title: str) -> str:
    m = re.search(r"\(([^)]*)\)\s*$", str(title or ""))
    return m.group(1).strip() if m else ""


def clean_entity(text: Any) -> str:
    """A name field is a name: one line, no question words, no leading article, no punctuation noise."""
    s = str(text or "").strip().split("\n")[0]
    s = s.strip(_STRIP_CHARS).strip()
    s = re.sub(r"^\s*(entity|name|search|query|look up)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip(_STRIP_CHARS).strip()
    if normalize_text(s) in ("none", "unknown", "n a", "na", "null", ""):
        return ""
    if len(s.split()) > 12:
        s = " ".join(s.split()[:12])
    return s


def clean_qualifier(text: Any) -> str:
    """A qualifier is at most three content words, with question words removed."""
    s = str(text or "").strip().split("\n")[0].strip(_STRIP_CHARS)
    if normalize_text(s) in ("none", "unknown", "n a", "na", "null", ""):
        return ""
    s = _QUESTION_WORDS.sub(" ", s)
    words = _content_words(s)
    return " ".join(words[:3])


def compile_query(entity: str, qualifier: str, use_qualifier: bool, fallback: str) -> str:
    """Assemble the index query from the structured fields under the measured policy."""
    ent = clean_entity(entity)
    if not ent:
        ent = clean_entity(fallback)
    if not ent:
        ent = " ".join(_content_words(fallback)[:6])
    qual = clean_qualifier(qualifier)
    # Only a name with too little signal of its own is ever allowed to carry qualifying words.
    if qual and (use_qualifier or len(ent.split()) < 2):
        low = ent.lower()
        extra = [w for w in qual.split() if w.lower() not in low]
        if extra:
            ent = ent + " " + " ".join(extra)
    return ent.strip() or str(fallback or "").strip()


PLAN_INSTRUCTIONS = """The question needs facts from two different encyclopedia articles. Plan both
lookups before anything is searched. You do NOT write search queries: you fill in name fields, and
the search string is assembled from them mechanically.

Work out which two articles must be read. For a comparison question they are the two entities being
compared. For a bridge question the first is the entity the question names outright and the second
is the entity the question only describes, which cannot be looked up until the first is read.

`entity_a`: the name of the article to read FIRST -- the lookup that is possible right now. Write
the proper name and nothing else: no question words, no "the", no verbs, no punctuation, no
description. Spell it exactly as the question spells it.
`qualifier_a`: at most three content words that tell two same-named articles apart -- a profession,
a medium, a place, a year. Write `none` if the name is already distinctive; a well-known full name
needs nothing.
`target_b`: what the SECOND article is about. If the question names it, write that name. If the
question only describes it, write the description exactly as the question words it, because the
name is not known yet.
`answer_type`: exactly one of yes_no, person, organization, work_title, place, date, number, other.

MEASURED ON LABELLED EXAMPLES: {POLICY}"""

RESOLVE_INSTRUCTIONS = """You have the passages from the first lookup. Do three things. Again you do
NOT write a search query -- you fill in a name field and the search string is assembled from it.

`facts`: copy out, verbatim, only the sentences or clauses from the passages that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. Write "none" if no passage is relevant.

`entity_b`: the name of the article the SECOND and final lookup must return. This is the decisive
field, so get it from the text in front of you:
- if the question described the second thing indirectly and a passage has now named it, write that
  name exactly as the passage prints it -- this is the whole point of having read the passages;
- otherwise write the other name the question itself gives;
- write a NAME only: no question words, no "the", no verbs, no clause, no punctuation, no
  explanation. It must not be `previous_entity` -- repeating the first lookup wastes the second.
- Prefer a name you can point at in the passages or in the question over one you remember. If the
  passages have genuinely not named it, write the description from `planned_target_b` instead.

`qualifier_b`: at most three content words that would tell two same-named articles apart, or `none`.

MEASURED ON LABELLED EXAMPLES: {POLICY}"""

ANSWER_INSTRUCTIONS = """Answer the question from the evidence: `facts` are the sentences already
extracted, `passages` are the raw article openings retrieved for this question.

First quote the single piece of evidence that settles the question, then declare the shape of the
answer, then give the answer itself.

`answer_type` must be exactly one of: yes_no, person, organization, work_title, place, date, number, other.

`answer` is graded by exact string match against a short reference span, so it must be the bare
answer and nothing else:
- Never a sentence. Never restate the question. No explanation, no reason, no "because", no
  trailing clause, no second sentence, no markdown, no quotation marks, no final period.
- If `answer_type` is yes_no, `answer` must be exactly `yes` or `no` -- one word, nothing after it.
  Questions of the form "are both ...", "do both ...", "is X the same as Y", "can both ..." are
  yes_no questions, even when the interesting content is in the reasoning.
- If the question offers a closed list of options ("A or B", "which of ..."), copy the winning
  option exactly as the question spells it, and nothing else.
- Otherwise copy the answer span verbatim from the evidence, in the shortest form that fully
  names it: just the name, the title, the place, the date or the number.
- Do not add appositives, descriptors, units of explanation or parentheses that the reference
  span would not contain.
- If the evidence is incomplete, still give the single most likely bare answer. Never answer
  "unknown", "not stated" or "cannot be determined"."""

DESCS = {
    "question": "the question to answer",
    "answer_type": "one of: yes_no, person, organization, work_title, place, date, number, other",
    "entity_a": "name of the article to read first -- a name only, no other words",
    "qualifier_a": "up to three disambiguating content words, or none",
    "target_b": "the second article: its name if the question gives one, otherwise the question's description of it",
    "planned_target_b": "the second target as described before searching",
    "previous_entity": "the name already used for the first lookup",
    "entity_b": "name of the article the second lookup must return -- a name only",
    "qualifier_b": "up to three disambiguating content words, or none",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)

DEFAULT_POLICY = (
    "an entity-name lookup works best as the bare name; qualifying words are added only when the "
    "name is a single word."
)


def _clean(s: Any) -> str:
    text = str(s or "").strip()
    if not text:
        return ""
    text = text.split("\n")[0]
    text = _PREFIX.sub("", text)
    return text.strip(_STRIP_CHARS).strip()


def _question_options(question: str) -> list[str]:
    if not _CLOSED_LIST.search(question or ""):
        return []
    q = str(question).strip().rstrip("?")
    sep = r",|\bor\b|\bversus\b|\bvs\.?\b"
    if re.match(r"\s*between\b", q, re.IGNORECASE):
        sep += r"|\bbetween\b|\band\b"
    opts: list[str] = []
    for chunk in re.split(sep, q, flags=re.IGNORECASE):
        cand = re.sub(r"^\s*(between|which|who|what|whom|whose)\b\s*", "", chunk, flags=re.IGNORECASE)
        cand = cand.strip(_STRIP_CHARS).strip()
        if 1 <= len(cand.split()) <= 8 and re.search(r"[A-Z0-9]", cand):
            opts.append(cand)
    return opts


def _echo_option(ans: str, question: str) -> str:
    norm_ans = normalize_text(ans)
    if not norm_ans:
        return ans
    prefix_hits, inside_hits = [], []
    for opt in _question_options(question):
        n = normalize_text(opt)
        if not n or n == norm_ans:
            return ans
        if len(n.split()) >= len(norm_ans.split()):
            continue
        if norm_ans.startswith(n + " "):
            prefix_hits.append(opt)
        elif f" {n} " in f" {norm_ans} ":
            inside_hits.append(opt)
    hits = prefix_hits or ([] if len(inside_hits) != 1 else inside_hits)
    return max(hits, key=lambda o: len(normalize_text(o))) if hits else ans


def project_answer(raw_answer: str, answer_type: str, question: str) -> str:
    ans = _clean(raw_answer)
    atype = str(answer_type or "").strip().lower()
    yn = _YESNO.match(ans)
    if yn and (atype.startswith("yes") or len(ans.split()) > 1):
        return yn.group(1).lower()
    if atype.startswith("yes") and normalize_text(ans) in ("yes", "no"):
        return normalize_text(ans)
    if len(ans.split()) > 4:
        parts = [p for p in _SENT.split(ans) if p.strip()]
        if len(parts) > 1 and len(parts[0].split()) >= 3:
            ans = _clean(parts[0])
    if len(ans.split()) > 1:
        ans = _echo_option(ans, question)
    return _clean(ans) or _clean(raw_answer)


class Harness(BaseHarness):
    K = 7

    def __init__(self) -> None:
        self.use_qualifier = False
        self.policy_note = DEFAULT_POLICY
        self._build()

    # -- modules ---------------------------------------------------------------------------
    def _build(self) -> None:
        self.plan = Module(
            inputs=["question"],
            outputs=["answer_type", "entity_a", "qualifier_a", "target_b"],
            instructions=PLAN_INSTRUCTIONS.replace("{POLICY}", self.policy_note),
            descs=DESCS,
        )
        self.resolve = Module(
            inputs=["question", "planned_target_b", "previous_entity", "passages"],
            outputs=["facts", "entity_b", "qualifier_b"],
            instructions=RESOLVE_INSTRUCTIONS.replace("{POLICY}", self.policy_note),
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )

    # -- prepare: measure the compilation policy against the train labels -------------------
    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:
        bare_hits = qual_hits = trials = 0
        for it in train_items:
            titles = [str(t) for t in (it.get("gold_titles") or []) if str(t).strip()]
            question = str(it.get("question", ""))
            if len(titles) < 2 or not question:
                continue
            qnorm = " " + normalize_text(question) + " "
            # the target a second hop would have to find: the one the question does not name
            unnamed = [t for t in titles if f" {normalize_text(_strip_paren(t))} " not in qnorm]
            target = unnamed[0] if unnamed else titles[-1]
            bare = _strip_paren(target)
            if not bare:
                continue
            # qualifying words: the article's own disambiguator if it has one, else the question's
            # most distinctive content words that the name does not already contain
            low = bare.lower()
            qual_words = [w for w in _content_words(_paren(target)) if w.lower() not in low]
            if not qual_words:
                qual_words = [w for w in _content_words(question) if w.lower() not in low][:3]
            qualified = (bare + " " + " ".join(qual_words[:3])).strip()
            if normalize_text(qualified) == normalize_text(bare):
                continue
            try:
                with ctx.open_tools(it) as tools:
                    got_bare = [str(d.get("title", "")).strip().lower() for d in tools.retrieve_docs(bare, k=7)]
                    got_qual = [str(d.get("title", "")).strip().lower() for d in tools.retrieve_docs(qualified, k=7)]
            except Exception:  # a train item we cannot probe simply contributes no trial
                continue
            key = target.strip().lower()
            trials += 1
            bare_hits += 1 if key in got_bare else 0
            qual_hits += 1 if key in got_qual else 0

        if trials >= 20:
            bare_rate = bare_hits / trials
            qual_rate = qual_hits / trials
            self.use_qualifier = qual_rate > bare_rate + 0.02
            better, worse = ("adding them", "the bare name") if self.use_qualifier else ("the bare name", "adding them")
            self.policy_note = (
                f"on {trials} labelled lookups where the wanted article was known, searching the bare "
                f"name returned it {bare_rate:.0%} of the time and adding qualifying words returned it "
                f"{qual_rate:.0%} of the time, so {better} wins over {worse}. Keep the name field a "
                f"name; put anything else in the qualifier field, where it is used only when it helps."
            )
            ctx.note("query policy measured", trials=trials, bare_rate=round(bare_rate, 3),
                     qual_rate=round(qual_rate, 3), use_qualifier=self.use_qualifier)
        self._build()

    def get_state(self) -> str:
        return json.dumps({"use_qualifier": self.use_qualifier, "policy_note": self.policy_note})

    def set_state(self, state: str) -> None:
        try:
            data = json.loads(state) if state else {}
        except ValueError:
            data = {}
        self.use_qualifier = bool(data.get("use_qualifier", False))
        self.policy_note = str(data.get("policy_note") or DEFAULT_POLICY)
        self._build()

    # -- run --------------------------------------------------------------------------------
    @staticmethod
    def _pool(*doc_lists: list[dict[str, Any]]) -> list[str]:
        """Deduplicated raw evidence pool, first occurrence wins, order preserved."""
        seen: set[str] = set()
        out: list[str] = []
        for docs in doc_lists:
            for d in docs:
                key = str(d.get("title", "")).strip().lower()
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(str(d.get("text", "")).strip())
        return out

    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        question = item["question"]

        # 1. plan both lookups as structured fields, before touching the index
        plan = self.plan(ctx.llm, tag="plan", question=question)
        entity_a = clean_entity(plan.get("entity_a"))
        target_b = str(plan.get("target_b", "")).strip()
        query_a = compile_query(entity_a, plan.get("qualifier_a"), self.use_qualifier, question)

        # 2. first hop
        docs_a = ctx.tools.retrieve_docs(query_a, k=self.K)

        # 3. read the passages, extract facts, and NAME the second target
        res = self.resolve(
            ctx.llm,
            tag="resolve",
            question=question,
            planned_target_b=target_b or "not named yet",
            previous_entity=entity_a or query_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(res.get("facts", "")).strip()
        entity_b = clean_entity(res.get("entity_b"))
        if entity_b and normalize_text(entity_b) == normalize_text(entity_a):
            entity_b = ""  # never spend the second hop on the first lookup again
        query_b = compile_query(entity_b, res.get("qualifier_b"), self.use_qualifier, target_b or question)
        if normalize_text(query_b) == normalize_text(query_a):
            query_b = compile_query("", "", self.use_qualifier, target_b or question)

        # 4. second hop
        docs_b = ctx.tools.retrieve_docs(query_b, k=self.K)

        # 5. answer over the deduplicated raw pool
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=self._pool(docs_a, docs_b),
        )
        raw = out.get("answer", "")
        answer = project_answer(raw, out.get("answer_type", ""), question)
        ctx.note("compiled queries", query_a=query_a, query_b=query_b,
                 entity_a=entity_a, entity_b=entity_b, use_qualifier=self.use_qualifier)
        ctx.note("typed answer", answer_type=out.get("answer_type", ""), raw=str(raw)[:300], projected=answer)
        return {
            "answer": answer,
            "raw_answer": raw,
            "answer_type": out.get("answer_type", ""),
            "query_a": query_a,
            "next_query": query_b,
            "facts": facts,
        }
