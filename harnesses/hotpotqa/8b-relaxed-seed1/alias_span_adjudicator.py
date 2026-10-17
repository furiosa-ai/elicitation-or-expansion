"""Surface-form adjudication: mine the evidence for every wording of the drafted answer, then choose.

Retrieval and routing are kept identical to the current best system, so the whole delta is attributable
to the ANSWER SURFACE FORM stage.

Diagnosis: on the 206 search-set items where both required documents WERE retrieved, 38 are still wrong,
and 16 of those are right in substance and wrong only in wording. The bias is strongly one-directional --
14 of the 16 want a LONGER form than the model produced (the full given names an article opens with, the
ordinal or year attached to an event, the administrative or regional suffix of a place, the head noun of a
genre) and 2 want a shorter one (a trailing descriptor, a second person the question did not ask about).
Telling a model to "copy the wording of the passage" does not fix this: the drafting call is optimising for
a short answer and re-reads the passage through that intent.

So wording is not decided in the drafting call at all.  Instead:

    1. draft   the model answers as before (short, from the raw passages).
    2. mine    DETERMINISTICALLY locate the draft inside the evidence and enumerate the surface forms that
               actually occur there: the matched span, its left extension (leading ordinal/year/name), its
               right extensions (following capitalised words, a `, Region` tail, one trailing head noun),
               the titles of passages whose title contains or is contained in the draft, and those titles
               with their parenthetical disambiguator stripped.  Nothing is invented -- every candidate is
               a literal substring of the corpus.
    3. adjudicate  ONE call shows the question, the draft and the candidate forms, each with the sentence
               it was taken from, and asks which form is exactly what the question asks for.

The choice is between concrete strings shown side by side with their sources, which is a different and much
easier decision than recalling a style rule while composing.  When fewer than two forms exist, or the answer
is a verdict, the stage is skipped and no call is spent.
"""

from __future__ import annotations

import re
from typing import Any

from core.harness import BaseHarness, RunContext
from domains.hotpotqa.dspy_format import Module
from domains.hotpotqa.evaluator import normalize_text

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
_TOKEN = re.compile(r"[A-Za-z0-9À-ÿ'’.&\-]+")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_COORD = re.compile(r"\s+(?:and|&|or)\s+|\s*,\s+", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")
_YEAR = re.compile(r"^\d{3,4}(?:st|nd|rd|th)?$")
_ORDINAL = re.compile(r"^\d{1,4}(?:st|nd|rd|th)$")
# lowercase words that legitimately end a title-like phrase ("Hindustani classical music")
_GLUE = {"of", "the", "and", "for", "de", "di", "da", "del", "della", "van", "von", "der", "la",
         "le", "el", "dos", "bin", "al", "at", "in", "on"}
_TAIL_STOP = {"is", "was", "were", "are", "be", "been", "has", "have", "had", "who", "which", "that",
              "and", "or", "but", "with", "from", "by", "for", "to", "as", "than", "when", "while",
              "he", "she", "it", "they", "his", "her", "its", "their", "a", "an", "the", "also"}

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

COMMIT_INSTRUCTIONS = """A first attempt to answer this question from these passages returned nothing usable.

The passages may cover the question only partially. Commit anyway to the single most likely short answer they
support: weigh the partial evidence and pick the best-supported entity, name, title, date, number or verdict.
Never reply that the information is missing, unclear or not provided, and never abstain -- an unanswered
question is simply wrong, whereas a well-reasoned guess can be right. Output the bare answer and nothing
else. If the question asks whether something is or is not the case, output exactly `yes` or `no`."""

ADJUDICATE_INSTRUCTIONS = """A draft answer has already been judged correct in substance. Your only job is
to pick the WORDING for it.

You get the question and numbered candidate forms. Every candidate is a literal string taken from a source
document, shown with the sentence it came from. They differ only in how much of the name or phrase they keep.

Pick the LONGEST candidate that is still purely a name or label for the one thing asked about.

Words that BELONG to a name, and must therefore be kept when a candidate offers them:
- middle names and full given names, even where another sentence uses a short form;
- an ordinal or year that is part of what an event is called;
- the county, state, region or country a place is written with;
- the head noun that completes a label -- the genre, network, award, category or type word.
Never pick a shorter candidate on the grounds that the extra naming words are unnecessary or that the
question did not ask for them. Completeness of the name is exactly what is wanted here.

Words that are NOT part of a name, so reject a candidate that carries them:
- an honorific or rank in front of it (Sir, Dame, Dr, Lord, President, Saint);
- a role, occupation, birth date, or any descriptive clause appended after it;
- a second, different thing joined by `and` or a comma, when the question asks for one thing;
- a trailing verb, preposition or dangling fragment;
- a parenthetical the source only adds to tell two articles apart.

When the question asks for a count, a date or a verdict, pick the plain value.

Output `choice`: the number of the single best form, and nothing else."""


class Harness(BaseHarness):
    K = 8
    MAX_ROUNDS = 3
    MAX_RETRIEVALS = 6
    POOL_SNIPPET = 380
    MAX_FORMS = 7          # candidate surface forms shown to the adjudicator
    MAX_GAP = 3            # tokens the draft may skip while matching inside a passage

    def __init__(self) -> None:
        self.decompose = Module(
            inputs=["question"], outputs=["queries"],
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
            inputs=["question", "passages"], outputs=["answer"],
            instructions=ANSWER_INSTRUCTIONS, descs={"answer": "the bare short answer"},
        )
        self.commit = Module(
            inputs=["question", "passages"], outputs=["answer"],
            instructions=COMMIT_INSTRUCTIONS, descs={"answer": "the bare short answer, never an abstention"},
        )
        self.adjudicate = Module(
            inputs=["question", "candidate_forms"], outputs=["choice"],
            instructions=ADJUDICATE_INSTRUCTIONS,
            descs={
                "candidate_forms": "numbered wordings of the same answer, each with its source sentence",
                "choice": "the number of the chosen form",
            },
        )

    # -- shared helpers -----------------------------------------------------------------------
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

    # -- surface-form mining ------------------------------------------------------------------
    @staticmethod
    def _split_passage(passage: str) -> tuple[str, str]:
        head, _, body = str(passage).partition(" | ")
        return head.strip(), body.strip()

    @staticmethod
    def _sentence_of(text: str, start: int, end: int) -> str:
        pos = 0
        for sent in _SENT_SPLIT.split(text):
            nxt = text.find(sent, pos)
            if nxt < 0:
                nxt = pos
            if nxt <= start < nxt + len(sent) or nxt < end <= nxt + len(sent):
                return sent.strip()[:300]
            pos = nxt + len(sent)
        return text[max(0, start - 80): end + 80].strip()[:300]

    def _match_spans(self, text: str, want: list[str]) -> list[tuple[int, int, list[tuple[int, int]]]]:
        """Locate the draft's tokens, in order, inside `text`. Returns (start, end, all_token_offsets)."""
        toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        norm = [normalize_text(t[0]) for t in toks]
        out: list[tuple[int, int, list[tuple[int, int]]]] = []
        offsets = [(t[1], t[2]) for t in toks]
        for i in range(len(toks)):
            if norm[i] != want[0]:
                continue
            j, w, gap = i, 1, 0
            while w < len(want) and j + 1 < len(toks) and gap <= self.MAX_GAP:
                j += 1
                if norm[j] == want[w]:
                    w += 1
                elif norm[j] and (toks[j][0][0].isupper() or norm[j] in _GLUE or norm[j].isdigit()):
                    gap += 1
                else:
                    break
            if w == len(want):
                out.append((i, j, offsets))
            if len(out) >= 4:
                break
        return out

    def _forms(self, draft: str, passages: list[str]) -> list[tuple[str, str]]:
        """Enumerate literal wordings of `draft` occurring in the evidence, each with its source sentence."""
        want = normalize_text(draft).split()
        if not want or len(want) > 8:
            return []
        found: list[tuple[str, str]] = []

        def add(form: str, sent: str) -> None:
            form = form.strip().strip(",;:").strip()
            form = form[:-1] if form.endswith(".") and not form.endswith("..") else form
            form = form.strip()
            if not form or len(form) > 120 or not normalize_text(form):
                return
            if ". " in form or form.count("(") != form.count(")") or form.count('"') % 2:
                return
            if normalize_text(form.split()[-1]) in _GLUE or normalize_text(form.split()[0]) in _GLUE:
                return
            found.append((form, sent))

        for passage in passages[:10]:
            title, body = self._split_passage(passage)
            ntitle = normalize_text(title)
            plain = normalize_text(_PAREN.sub("", title))
            wnorm = " ".join(want)
            # a passage whose title is a longer or shorter wording of the draft
            if ntitle and (wnorm in ntitle or ntitle in wnorm) and ntitle != wnorm:
                add(title, body[:300])
            if plain and plain != ntitle and (wnorm in plain or plain in wnorm):
                add(_PAREN.sub("", title), body[:300])
            for text in (title, body):
              for i, j, off in self._match_spans(text, want):
                  toks = [(text[a:b], a, b) for a, b in off]
                  sent = self._sentence_of(text, off[i][0], off[j][1])
                  base_l, base_r = i, j
                  add(text[off[i][0]: off[j][1]], sent)
                  # left extension: a leading ordinal, year or capitalised name the draft dropped
                  li = i
                  for _ in range(2):
                      if li - 1 < 0:
                          break
                      w = toks[li - 1][0]
                      nw = normalize_text(w)
                      if _YEAR.match(w) or _ORDINAL.match(w) or (w[:1].isupper() and nw.isalpha()):
                          li -= 1
                          add(text[off[li][0]: off[base_r][1]], sent)
                      else:
                          break
                  # right extension: following capitalised words, a `, Region` tail, one trailing head noun
                  rj, lowered = j, 0
                  for _ in range(3):
                      if rj + 1 >= len(toks):
                          break
                      w = toks[rj + 1][0]
                      nw = normalize_text(w)
                      if not nw or nw in _TAIL_STOP:
                          break
                      if w[:1].isupper() or nw in _GLUE or _YEAR.match(w):
                          rj += 1
                      elif nw.isalpha() and lowered == 0:
                          rj += 1
                          lowered += 1
                      else:
                          break
                      add(text[off[base_l][0]: off[rj][1]], sent)
                      add(text[off[li][0]: off[rj][1]], sent)
            if len(found) > 40:
                break

        # the draft may name several things where the question asked for one
        for part in _COORD.split(draft):
            part = part.strip(" ,;:.")
            if part and part != draft.strip() and len(normalize_text(part).split()) >= 2:
                for passage in passages[:10]:
                    if normalize_text(part) and normalize_text(part) in normalize_text(passage):
                        add(part, "")
                        break

        # deduplicate on the grading normalisation, keeping the first (shortest-path) wording
        seen: set[str] = set()
        uniq: list[tuple[str, str]] = []
        for form, sent in sorted(found, key=lambda x: len(x[0])):
            key = normalize_text(form)
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append((form, sent))
        return uniq

    # -- main ---------------------------------------------------------------------------------
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
            shown = [p[: self.POOL_SNIPPET] for p in pool]
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
        ctx.note("pool routed", n_pool=len(pool), n_selected=len(selected), retrievals=retrievals)

        drafted = self.answer(ctx.llm, tag="answer", question=question, passages=evidence).get("answer", "")
        final = self._tidy(drafted, question)

        if not final or _NON_ANSWER.match(final):
            forced = self.commit(ctx.llm, tag="commit", question=question, passages=pool[:12]).get("answer", "")
            forced = self._tidy(forced, question)
            if forced and not _NON_ANSWER.match(forced):
                final = forced
            ctx.note("forced commit", before=str(drafted)[:120], after=final[:120])

        # -- surface-form adjudication ---------------------------------------------------------
        chosen = final
        forms: list[tuple[str, str]] = []
        if final and final.lower() not in {"yes", "no"} and not _NON_ANSWER.match(final):
            forms = self._forms(final, (selected or []) + pool)
            # keep the draft itself as an option even if the evidence spells it differently
            if not any(normalize_text(f) == normalize_text(final) for f, _ in forms):
                forms.insert(0, (final, ""))
            if len(forms) > 1:
                forms = forms[: self.MAX_FORMS]
                # the field renderer numbers list entries itself, so entry i is candidate i
                shown_forms = [f + (f"   [source: {s}]" if s else "") for f, s in forms]
                picked = self.adjudicate(
                    ctx.llm, tag="adjudicate", question=question, candidate_forms=shown_forms,
                ).get("choice", "")
                idx = self._parse_indices(picked, len(forms))
                if idx:
                    chosen = self._tidy(forms[idx[0] - 1][0], question)
                ctx.note("form adjudicated", draft=final[:100],
                         forms=[f for f, _ in forms], chosen=chosen[:100])

        return {"answer": chosen or final or str(drafted).strip(), "draft": final,
                "n_pool": len(pool), "n_forms": len(forms)}
