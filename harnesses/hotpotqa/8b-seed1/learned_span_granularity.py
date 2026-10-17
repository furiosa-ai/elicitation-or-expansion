"""Learn the label convention for how far an answer span extends, then complete the span.

Measured motivation (classifying all 93 search-set errors of the frontier by the surface relation
between its prediction and the reference):

    disjoint strings                                     60
    partial overlap (f1 > 0, no containment)             20
    one string strictly CONTAINS the other               13   -- 10 of them the prediction is SHORTER

and the ten short ones are all the same move, a truncation the frontier's own answer contract asks
for ("the shortest form that fully names it", "do not add ... units of explanation"):

    number without its unit; a count without its head noun; a settlement without its region;
    an institution without its campus; an event without its year; a year for a full date;
    a surname without its given name.

Every one of those references keeps text the evidence sentence writes immediately beside the span.
But the opposite instruction is not safe either: an earlier attempt to expand every answer to the
fullest form the article offers scored 5 wins against 15 losses on these same predictions, because
the convention differs by KIND of extension -- and which kinds are kept is a property of the labels,
not something prose can settle.

Mechanism under test: measure it. `prepare` reads the train split only, fetches each train item's own
labelled articles from the index (no LLM calls), and for every reference answer that occurs in them
asks a symmetric question per extension class: the sentence offered both the short and the extended
form -- which did the label take? A class where the labels keep the extension gets a KEEP vote, a
class where they drop it gets a STRIP vote, and only classes the train labels keep are enabled at
run time.

At run time the answering call is byte-identical to the frontier's, so the 189 items that already
convert at 0.852 see the same prompt and the same completion. Afterwards a deterministic operator
looks the projected answer up in the evidence pool and extends it -- only in an enabled class, only
when the extension is written verbatim beside the span, and never past one extension.

Budget: 2 LLM calls, 2 retrievals, k=7 -- identical to the parent.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.harness import BaseHarness, PrepareContext, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

GAP_INSTRUCTIONS = """`passages` are the articles a sparse full-text index returned for the question
itself, and they are already in hand. Usually they cover only ONE of the two articles this question
needs. You get exactly one more lookup. Find the gap and aim it.

`facts`: copy out, verbatim, only the sentences or clauses from `passages` that bear on the
question, each prefixed with the article title in brackets. Copy numbers, dates and full names
exactly. If nothing is relevant, write "none".

`missing_target`: name the article still needed and say why the question needs it -- the entity the
question mentions but no passage above covers, or the entity the question only describes
indirectly. If the passages have now NAMED that entity, give the name. If they have not, describe
it with the question's own words.
There is essentially always a second article: an entity that a passage merely NAMES is not covered
by that passage, so if the question asks for a property of it -- a date, a place, a number, a
person, a title -- that entity's own article is what is missing. Only write "none" if the exact
answer to the question is already quoted word for word in `facts`; a lookup you decline is a lookup
thrown away.

`next_query`: the query for the final lookup, scored by a sparse index over article openings, so
write the words that the missing article's OWN OPENING would contain: the subject's name, plus the
two or three words that pin down what kind of thing it is (its type, field, place, role or date).
- Aim it at `missing_target`, not at the question. Do not hand the question back: the titles listed
  in `titles_in_hand` are already retrieved, and a query that returns them again buys nothing.
- Drop question words ("which", "who", "both", "same", "more", "first") and punctuation.
- If the question named two entities and only one was retrieved, this query is for the other one.
- Search terms only: no prose, no explanation, no quotation marks, at most about ten words."""

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
    "titles_in_hand": "titles of the articles already retrieved -- do not fetch these again",
    "passages": "article openings returned by the index, as 'Title | opening'",
    "facts": "verbatim evidence sentences, each prefixed with its article title in brackets",
    "missing_target": "the article still needed, named if the passages name it",
    "next_query": "keyword query for the second and final lookup, aimed at the missing article",
    "evidence_quote": "the exact sentence from the evidence that settles the question",
    "answer": "the bare answer span only -- no sentence, no explanation",
}

_YESNO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_SENT = re.compile(r"(?<=[a-z0-9\)\]\'\"])[.!?]\s+(?=[A-Z0-9])")
_PREFIX = re.compile(r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)\s*", re.IGNORECASE)
_CLOSED_LIST = re.compile(r"\bor\b|\bversus\b|\bvs\.?\b|^\s*between\b", re.IGNORECASE)
_STRIP_CHARS = " \t*_`~\"'“”‘’.,;:"
_WORD = re.compile(r"[A-Za-z0-9]+")
_QWORDS = {
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had", "that", "this", "these", "those", "it", "its", "by",
    "with", "from", "as", "both", "same", "more", "most", "first", "than", "there", "their", "his",
    "her", "he", "she", "they", "you", "not", "but", "also", "s",
}


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


_DECLINED = re.compile(r"^(none|n/?a|nothing|no\b.*|not\s|already\b)", re.IGNORECASE)


def _as_query(text: Any, fallback: str) -> str:
    """Sanitise a generated query; a declined or prose-wrapped lookup falls back deterministically."""
    q = str(text or "").strip().split("\n")[0]
    q = q.strip(_STRIP_CHARS).strip()
    q = re.sub(r"^\s*(search|query|search for|look up)\s*:\s*", "", q, flags=re.IGNORECASE)
    q = re.split(r"\s+[—–-]{1,2}\s+|\s*[—–]\s*", q)[0].strip(_STRIP_CHARS).strip()
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    if len(q) < 3 or _DECLINED.match(q):
        return fallback  # never spend the remaining lookup on a refusal
    return q


def _residual_query(question: str, titles: list[str]) -> str:
    """The question minus the words of the title it already matched best -- a complementary query.

    Deterministic fallback for when the model hands the question back: whatever hop A locked onto,
    the words that identify it are removed, so the index scores the rest of the question instead.
    """
    q_tokens = [t for t in _WORD.findall(question)]
    q_content = {t.lower() for t in q_tokens if t.lower() not in _QWORDS}
    if not q_content:
        return question
    best, best_hit = None, 0
    for title in titles:
        hit = len({t.lower() for t in _WORD.findall(title)} & q_content)
        if hit > best_hit:
            best, best_hit = title, hit
    if best is None or best_hit >= len(q_content):
        return question
    drop = {t.lower() for t in _WORD.findall(best)}
    kept = [t for t in q_tokens if t.lower() not in drop and t.lower() not in _QWORDS]
    return " ".join(kept) if len(kept) >= 2 else question


# --------------------------------------------------------------------------------------------
# Span-completion classes.  Each class names ONE kind of text that sits immediately beside an
# answer span in the sentence that states it.  `_completions` reads a span forwards out of the
# evidence; `_shrinks` reads the same relation backwards out of a label.  The two together turn a
# reference answer into a vote: the sentence offered both forms, and the label picked one.
# --------------------------------------------------------------------------------------------

CLASSES = ("unit", "count_noun", "comma_qualifier", "year_prefix", "given_name", "middle_name",
           "full_date")

_UNITS = {
    "mi", "mile", "miles", "km", "kilometre", "kilometres", "kilometer", "kilometers", "m",
    "metre", "metres", "meter", "meters", "ft", "feet", "foot", "yd", "yards", "in", "inches",
    "cm", "mm", "acre", "acres", "ha", "hectares", "sqkm", "sqmi", "kg", "lb", "lbs", "pounds",
    "tons", "tonnes", "percent", "%", "million", "billion", "thousand", "years", "days", "hours",
    "minutes", "seconds", "points", "votes", "seats", "episodes", "members",
}
_NUM = re.compile(r"^[0-9][0-9,.]*$")
_NUMWORD = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred",
}
_CAP = re.compile(r"^[A-Z][A-Za-z'’.\-]*$")
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE_BEFORE = re.compile(r"(?:([0-9]{1,2})\s+)?(" + _MONTH + r")[,]?\s+$")
_DATE_SPAN = re.compile(r"^(?:[0-9]{1,2}\s+" + _MONTH + r"|" + _MONTH + r"\s+[0-9]{1,2},?|" + _MONTH + r")\s+(1[0-9]{3}|20[0-9]{2})$")
_STOPCONT = {
    "which", "who", "whose", "that", "and", "or", "a", "an", "the", "is", "was", "were", "in",
    "of", "for", "with", "he", "she", "it", "they", "his", "her", "their", "but", "as", "at",
    "on", "to", "by", "from", "also", "including", "based", "known", "born",
}


def _is_num(tok):
    return bool(_NUM.match(tok)) or tok.lower() in _NUMWORD


def _tok(s):
    return [t for t in re.split(r"\s+", str(s or "").strip()) if t]


def _occurrences(span, text):
    """Case-insensitive, boundary-respecting occurrences of `span` in `text`, as (start, end)."""
    if not span or len(span) < 2:
        return []
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(span) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _completions(span, text):
    """Every (class, extended_span) the evidence text offers immediately beside `span`."""
    out = []
    words = _tok(span)
    if not words or len(words) > 6:
        return out
    last, first = words[-1], words[0]
    for start, end in _occurrences(span, text):
        after = text[end:end + 60]
        before = text[max(0, start - 40):start]
        # unit / head noun: a number in the evidence carries what it counts
        m = re.match(r"^[  ]*(?:sq[ .]?)?([A-Za-z%]{1,12})\b", after)
        if m and _is_num(last):
            nxt = m.group(0).strip()
            key = re.sub(r"[^A-Za-z%]", "", nxt).lower()
            if key in _UNITS:
                out.append(("unit", text[start:end] + " " + nxt))
            elif key and key not in _STOPCONT and nxt[:1].islower() and len(words) == 1:
                out.append(("count_noun", text[start:end] + " " + nxt))
        # comma qualifier: a place or institution followed by the region that disambiguates it
        m = re.match(r"^[  ]*,[  ]*((?:[A-Z][A-Za-z.\-]*)(?:[  ]+[A-Z][A-Za-z.\-]*)?)", after)
        if m and _CAP.match(first):
            tail = m.group(1).strip()
            if tail.split()[0].lower() not in _STOPCONT:
                out.append(("comma_qualifier", text[start:end] + ", " + tail))
        # full date: a year the sentence writes out with its day and month
        if len(words) == 1 and _YEAR.match(last):
            md = _DATE_BEFORE.search(before)
            if md:
                out.append(("full_date", md.group(0).strip().rstrip(",") + " " + text[start:end]))
        # year prefix: an event named with the year in front of it
        pre = _tok(before)
        if pre and _YEAR.match(pre[-1]) and _CAP.match(first) and not _is_num(last):
            out.append(("year_prefix", pre[-1] + " " + text[start:end]))
        # given name: a surname that the sentence prints with its given name
        if len(words) == 1 and _CAP.match(first) and pre and _CAP.match(pre[-1]) \
                and pre[-1].lower() not in _STOPCONT:
            out.append(("given_name", pre[-1] + " " + text[start:end]))
    # middle name: the fuller self-naming form, same first and last token.  Looked up over the
    # whole text rather than per occurrence, because a name an article only ever writes in full
    # gives the short form no occurrence to anchor on.
    if len(words) >= 2 and _CAP.match(first) and _CAP.match(last):
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(first)
                         + r"(?:\s+[A-Z][A-Za-z'\u2019.\-]*){1,3}\s+"
                         + re.escape(last) + r"(?![A-Za-z0-9])")
        for mm in pat.finditer(text):
            if len(_tok(mm.group(0))) > len(words):
                out.append(("middle_name", mm.group(0)))

    # keep one candidate per class, the shortest
    best = {}
    for cls, cand in out:
        cand = cand.strip(_STRIP_CHARS).strip()
        if cand and (cls not in best or len(cand) < len(best[cls])):
            best[cls] = cand
    return [(c, best[c]) for c in CLASSES if c in best]


def _shrinks(span):
    """Every (class, shorter_span) obtained by undoing one extension of `span`."""
    out = []
    words = _tok(span)
    if len(words) < 2 or len(words) > 6:
        return out
    if _is_num(words[0]) and len(words) == 2:
        key = re.sub(r"[^A-Za-z%]", "", words[1]).lower()
        out.append(("unit" if key in _UNITS else "count_noun", words[0]))
    m = re.match(r"^(.+?),[  ]*([A-Z].*)$", span.strip())
    if m and _CAP.match(words[0]):
        out.append(("comma_qualifier", m.group(1).strip()))
    if _YEAR.match(words[0]) and len(words) >= 2:
        out.append(("year_prefix", " ".join(words[1:])))
    md = _DATE_SPAN.match(" ".join(words))
    if md:
        out.append(("full_date", md.group(1)))
    if len(words) == 2 and all(_CAP.match(w) for w in words):
        out.append(("given_name", words[1]))
    if len(words) >= 3 and _CAP.match(words[0]) and _CAP.match(words[-1]):
        out.append(("middle_name", words[0] + " " + words[-1]))
    return out


def _apply_completion(answer, pool_text, enabled):
    """Extend the projected answer by at most ONE enabled, evidence-written extension."""
    words = _tok(answer)
    if not answer or not words or len(words) > 6 or not enabled:
        return answer, ""
    if normalize_text(answer) in ("yes", "no", ""):
        return answer, ""
    cands = [(cls, cand) for cls, cand in _completions(answer, pool_text) if cls in enabled]
    if not cands:
        return answer, ""
    cls, cand = max(cands, key=lambda cc: enabled[cc[0]])
    cand = _clean(cand)
    if not cand or normalize_text(cand) == normalize_text(answer) or len(_tok(cand)) > len(words) + 3:
        return answer, ""
    return cand, cls


class Harness(BaseHarness):
    K = 7
    MIN_VOTES = 2
    MAX_PROBES = 150   # one index fetch per labelled article; bounds prepare-time wall clock

    def __init__(self):
        self.gap = Module(
            inputs=["question", "titles_in_hand", "passages"],
            outputs=["facts", "missing_target", "next_query"],
            instructions=GAP_INSTRUCTIONS,
            descs=DESCS,
        )
        self.generate_answer = Module(
            inputs=["question", "facts", "passages"],
            outputs=["evidence_quote", "answer_type", "answer"],
            instructions=ANSWER_INSTRUCTIONS,
            descs=DESCS,
        )
        # class -> measured keep-rate; empty until `prepare` measures it, so an unprepared run is
        # byte-identical to the parent rather than guessing at the convention
        self.enabled = {}
        self.votes = {}

    # -- prepare: measure the convention on the train labels ------------------------------------
    def prepare(self, train_items, ctx: PrepareContext):
        try:
            self._measure(train_items, ctx)
        except Exception as exc:  # no measurement -> no completion -> the parent, exactly
            self.enabled, self.votes = {}, {}
            ctx.note("span convention unmeasured", error=str(exc)[:300])

    def _measure(self, train_items, ctx: PrepareContext):
        keeps = {c: 0 for c in CLASSES}
        strips = {c: 0 for c in CLASSES}
        scanned = 0
        probes = 0
        for it in train_items:
            if probes >= self.MAX_PROBES:
                break
            answer = str(it.get("answer", "")).strip()
            titles = [str(t).strip() for t in (it.get("gold_titles") or []) if str(t).strip()]
            if not answer or normalize_text(answer) in ("yes", "no") or not titles:
                continue
            texts = []
            for title in titles[:2]:
                try:
                    probes += 1
                    with ctx.open_tools(it) as tools:
                        docs = tools.retrieve_docs(title, k=5)
                except Exception:  # a train item we cannot fetch simply contributes no vote
                    continue
                for d in docs:
                    if str(d.get("title", "")).strip().lower() == title.lower():
                        texts.append(str(d.get("text", "")))
                        break
            if not texts:
                continue
            body = "\n".join(texts)
            if not _occurrences(answer, body):
                continue  # the label is not a span of its own articles: it votes on nothing
            scanned += 1
            try:
                # the sentence offers these extensions and the label declined every one of them
                for cls, _cand in _completions(answer, body):
                    strips[cls] += 1
                # ... and the label itself already carries these extensions, whose short form the
                # sentence also offers, so they were a real choice
                for cls, short in _shrinks(answer):
                    if _occurrences(short, body):
                        keeps[cls] += 1
            except Exception:
                continue
        self.votes = {c: [keeps[c], strips[c]] for c in CLASSES}
        self.enabled = {
            c: round(keeps[c] / (keeps[c] + strips[c]), 3)
            for c in CLASSES
            if keeps[c] >= self.MIN_VOTES and keeps[c] > strips[c]
        }
        ctx.note("span convention measured", scanned=scanned, votes=self.votes, enabled=self.enabled)

    def get_state(self):
        return json.dumps({"enabled": self.enabled, "votes": self.votes})

    def set_state(self, state):
        try:
            data = json.loads(state) if state else {}
        except ValueError:
            data = {}
        self.enabled = {str(k): float(v) for k, v in (data.get("enabled") or {}).items() if k in CLASSES}
        self.votes = data.get("votes") or {}

    @staticmethod
    def _pool(*doc_lists):
        seen = set()
        out = []
        for docs in doc_lists:
            for d in docs:
                key = str(d.get("title", "")).strip().lower()
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(str(d.get("text", "")).strip())
        return out

    def run(self, item, ctx: RunContext):
        question = item["question"]

        # 1-3. retrieval and gap planning: byte-identical to the parent
        docs_a = ctx.tools.retrieve_docs(question, k=self.K)
        titles_a = [str(d.get("title", "")).strip() for d in docs_a]
        gap = self.gap(
            ctx.llm,
            tag="gap",
            question=question,
            titles_in_hand=titles_a,
            passages=[d["text"] for d in docs_a],
        )
        facts = str(gap.get("facts", "")).strip()
        residual = _residual_query(question, titles_a)
        target = _as_query(gap.get("missing_target"), residual)
        next_query = _as_query(gap.get("next_query"), target)
        if normalize_text(next_query) == normalize_text(question):
            next_query = residual
        if normalize_text(next_query) in {normalize_text(t) for t in titles_a}:
            next_query = target
        docs_b = ctx.tools.retrieve_docs(next_query, k=self.K)

        pool = self._pool(docs_a, docs_b)
        out = self.generate_answer(
            ctx.llm,
            tag="answer",
            question=question,
            facts=facts if facts else "none",
            passages=pool,
        )
        raw = out.get("answer", "")
        projected = project_answer(raw, out.get("answer_type", ""), question)

        # 4. the only change: complete the span the answering call truncated, in the classes the
        #    train labels were measured to keep, using text the evidence writes beside it
        evidence = "\n".join([facts] + pool)
        answer, applied = _apply_completion(projected, evidence, self.enabled)
        ctx.note("gap hop", missing_target=str(gap.get("missing_target", ""))[:200], next_query=next_query)
        ctx.note("span completion", projected=projected, applied=applied, final=answer,
                 enabled=self.enabled)
        return {
            "answer": answer,
            "raw_answer": raw,
            "projected": projected,
            "completion_class": applied,
            "answer_type": out.get("answer_type", ""),
            "next_query": next_query,
            "facts": facts,
        }
