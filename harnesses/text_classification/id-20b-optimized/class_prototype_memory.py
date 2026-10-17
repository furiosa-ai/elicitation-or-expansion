"""Class-prototype memory: store compressed, discriminative class descriptions
instead of raw solved cases.

Every memory system in this line so far stores whole examples and shows the
model a handful of them. That representation is redundant - ten cases with the
same answer repeat the same evidence ten times - and it is expensive, filling
tens of thousands of characters while carrying one bit of supervision each.

Here the unit of memory is the *answer class*, not the case. As cases arrive,
each one is reduced to a set of document features (words and word bigrams for
alphabetic script, 2-3 character grams for unsegmented script) and folded into
per-class document-frequency counters. A class is then described by the
features that most distinguish it, ranked by a log-odds z-score against the
rest of the corpus with an informative Dirichlet prior, subject to a recall
floor (the feature must appear in a real share of the class's cases) and a
precision floor (it must occur mostly inside the class). Redundant substrings
are suppressed so the description covers distinct evidence.

At prediction time the classes are ranked by how many of their distinguishing
features the query exhibits, and the top ones are injected as a compact
"candidate answers and what distinguishes them" table. Raw cases survive only
as a two-example format anchor.

Offline replay of the iteration-1 traces: ten to twelve mined features per
class recover the correct answer at rank 1 for 54.3% of steps and inside the
top 5 for 78.6% on a small label space, in about 1.8k characters - roughly
half the context the exemplar-based system spends and a fifth of what the
few-shot baseline spends.

Classes seen only once cannot support a profile, so tasks whose answers never
repeat fall back to plain similarity retrieval over raw cases.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

# Profile mining
MIN_CLASS_DOCS = 2
CUES_PER_CLASS = 12
MIN_CUE_DOCS = 2
MIN_CUE_RECALL = 0.3
MIN_CUE_PRECISION = 0.2
MAX_GLOBAL_DF_FRACTION = 0.5
CANDIDATE_CLASSES = 8
MAX_PROFILE_CHARS = 4000

# Format anchors
ANCHOR_EXAMPLES = 2
MAX_ANCHOR_CHARS = 1500
# Used only while no class has enough cases to be profiled.
FALLBACK_TOP_K = 12
FALLBACK_EXEMPLAR_CHARS = 9000
PER_ANSWER_CAP = 2

# Answer-space gating
MIN_EXAMPLES_FOR_GATE = 15
MIN_REUSE_RATE = 0.25
MAX_INVENTORY_CHARS = 4000

# Canonicalisation
SNAP_THRESHOLD = 0.65
SNAP_MARGIN = 0.04
MIN_AFFIX_OBS = 4

NGRAM_SIZES = (2, 3)
SPLIT_PRECEDENCE = [";", "；", "\n", "|", "，", ",", "、"]

PROMPT = """Solve the problem below using the reference material learned from previously solved cases.

{profile_section}{inventory_section}{anchor_section}
**Problem:**
{input}

**Instructions:**
- Follow the patterns, granularity and wording conventions shown above.
{profile_rule}{inventory_rule}- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

PROFILE_RULE = (
    "- Each candidate answer is listed with the evidence that distinguishes it from "
    "the others. Check the problem against those distinguishing features and pick "
    "the answer whose features it actually matches.\n"
)

INVENTORY_RULE = (
    "- If the correct answer is one of the listed answer strings, reproduce that "
    "string EXACTLY, character for character. Otherwise answer in the same style "
    "and level of detail as those strings.\n"
)

TERSE_RETRY = """{input}

Give your answer immediately. Do not deliberate and do not explain.
Reply with only this JSON object:

{{"final_answer": "[your answer]"}}"""

_PUNCT = re.compile(r"[\s、。,;；，.\[\]/_\-()（）:：]+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]+")
_STOP = set(
    "the a an and or of to in is are was were am be been being have has had do does "
    "did not no as by from but so very they them he she his her you your we our us "
    "it its that this these those for on with at also there then when what which who "
    "i my me".split()
)


def _features(text: str) -> set[str]:
    """Script-aware document features: words + word bigrams, or short char grams."""
    t = text.lower()
    feats: set[str] = set()
    words = [w for w in _WORD.findall(t) if w not in _STOP and len(w) > 2 and not w.isdigit()]
    feats.update(words)
    feats.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    for run in _CJK.findall(t):
        for n in NGRAM_SIZES:
            for i in range(max(0, len(run) - n + 1)):
                feats.add(run[i : i + n])
    return feats


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def _answer_ngrams(text: str, n: int = 2) -> Counter:
    text = _normalize(text)
    grams = Counter(text[i : i + n] for i in range(max(0, len(text) - n + 1)))
    return grams or Counter([text])


def _dice(a: Counter, b: Counter) -> float:
    total = sum(a.values()) + sum(b.values())
    return 2 * sum((a & b).values()) / total if total else 0.0


def _coverage(a: Counter, b: Counter) -> float:
    total = sum(a.values())
    return sum((a & b).values()) / total if total else 0.0


class _Affix:
    """Modal output markup, learned from (prediction, gold) pairs."""

    def __init__(self) -> None:
        self.prefixes: Counter = Counter()
        self.suffixes: Counter = Counter()
        self.n = 0

    def observe(self, prediction: str, gold: str) -> None:
        idx = prediction.find(gold)
        if idx < 0 or len(prediction) == len(gold):
            return
        self.n += 1
        self.prefixes[prediction[:idx]] += 1
        self.suffixes[prediction[idx + len(gold) :]] += 1

    def _modal(self, counts: Counter) -> str:
        if self.n < MIN_AFFIX_OBS or not counts:
            return ""
        text, hits = counts.most_common(1)[0]
        return text if text and hits >= 0.5 * self.n else ""

    def strip(self, text: str) -> str:
        text = text.strip()
        pre, suf = self._modal(self.prefixes), self._modal(self.suffixes)
        if pre and text.startswith(pre):
            text = text[len(pre) :]
        if suf and text.endswith(suf):
            text = text[: -len(suf)]
        return text.strip()


class _Template:
    """The boilerplate shared by every input for a task.

    Inputs arrive as a fixed instruction template wrapped around the part that
    actually varies. That template can be a large share of the characters and,
    when it contains worked format examples, it injects answer strings that have
    nothing to do with the case at hand. It is exactly the longest prefix and
    suffix common to all inputs seen so far, so it can be learned and removed.
    """

    MIN_INPUTS = 5

    def __init__(self) -> None:
        self.prefix: str | None = None
        self.suffix: str | None = None
        self.n = 0

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        i = 0
        limit = min(len(a), len(b))
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    def observe(self, text: str) -> None:
        self.n += 1
        if self.prefix is None:
            self.prefix, self.suffix = text, text
            return
        self.prefix = self._common_prefix(self.prefix, text)
        self.suffix = self._common_prefix(self.suffix[::-1], text[::-1])[::-1]

    def strip(self, text: str) -> str:
        if self.n < self.MIN_INPUTS or self.prefix is None:
            return text
        body = text
        if self.prefix and body.startswith(self.prefix):
            body = body[len(self.prefix) :]
        if self.suffix and body.endswith(self.suffix):
            body = body[: -len(self.suffix)]
        return body.strip() or text


class _Inventory:
    """Distinct atomic answer units plus the delimiter that composes them."""

    def __init__(self) -> None:
        self.units: Counter = Counter()
        self.delim_hits: Counter = Counter()
        self.n_answers = 0
        self.golds: list[str] = []
        self._grams: dict[str, Counter] = {}
        self._unigrams: dict[str, Counter] = {}
        self._built_with: str | None = None

    def delimiter(self) -> str | None:
        for d in SPLIT_PRECEDENCE:
            if self.delim_hits[d] >= max(2, 0.03 * self.n_answers):
                return d
        return None

    def split(self, text: str) -> list[str]:
        d = self.delimiter()
        if d and d in text:
            return [p.strip() for p in text.split(d) if p.strip()]
        return [text.strip()] if text.strip() else []

    def _register(self, unit: str) -> None:
        self.units[unit] += 1
        if unit not in self._grams:
            self._grams[unit] = _answer_ngrams(unit)
            self._unigrams[unit] = _answer_ngrams(unit, 1)

    def observe(self, gold: str) -> bool:
        """Store an answer; returns True if the unit inventory was rebuilt."""
        self.n_answers += 1
        self.golds.append(gold)
        for d in SPLIT_PRECEDENCE:
            if d in gold:
                self.delim_hits[d] += 1
        # The best delimiter is only identifiable once enough answers have been
        # seen; when it changes, every unit split under the old one is wrong.
        current = self.delimiter()
        if current != self._built_with:
            self._built_with = current
            self.units = Counter()
            self._grams = {}
            self._unigrams = {}
            for g in self.golds:
                for unit in self.split(g):
                    self._register(unit)
            return True
        for unit in self.split(gold):
            self._register(unit)
        return False

    def reuse_rate(self) -> float:
        total = sum(self.units.values())
        if not total:
            return 0.0
        singletons = sum(1 for c in self.units.values() if c == 1)
        return 1.0 - singletons / total


class _Profiles:
    """Per-class document-frequency counters and the cues mined from them."""

    def __init__(self) -> None:
        self.class_df: dict[str, Counter] = defaultdict(Counter)
        self.class_docs: Counter = Counter()
        self.global_df: Counter = Counter()
        self.n_docs = 0
        self._cue_cache: dict[str, list[str]] = {}

    def add(self, text: str, labels: list[str]) -> None:
        feats = _features(text)
        for label in dict.fromkeys(labels):
            counts = self.class_df[label]
            for f in feats:
                counts[f] += 1
            self.class_docs[label] += 1
            self._cue_cache.pop(label, None)
        for f in feats:
            self.global_df[f] += 1
        self.n_docs += 1
        # Global statistics shifted, so every cached description is stale.
        self._cue_cache.clear()

    def _z_score(self, class_count: int, class_docs: int, feature: str) -> float:
        """Log-odds ratio with an informative Dirichlet prior (Monroe et al.)."""
        prior = 0.5
        rest = max(0, self.global_df[feature] - class_count)
        others = max(1, self.n_docs - class_docs)
        p_in = min((class_count + prior) / (class_docs + 2 * prior), 0.999)
        p_out = min((rest + prior) / (others + 2 * prior), 0.999)
        variance = 1.0 / (class_count + prior) + 1.0 / (rest + prior)
        return (
            math.log(p_in / (1 - p_in)) - math.log(p_out / (1 - p_out))
        ) / math.sqrt(variance)

    def cues(self, label: str) -> list[str]:
        cached = self._cue_cache.get(label)
        if cached is not None:
            return cached
        docs = self.class_docs[label]
        if docs < MIN_CLASS_DOCS:
            self._cue_cache[label] = []
            return []
        scored = []
        for feature, count in self.class_df[label].items():
            if count < MIN_CUE_DOCS or count < MIN_CUE_RECALL * docs:
                continue
            if self.global_df[feature] > MAX_GLOBAL_DF_FRACTION * self.n_docs:
                continue
            if count / (self.global_df[feature] or 1) < MIN_CUE_PRECISION:
                continue
            scored.append((self._z_score(count, docs, feature), feature))
        scored.sort(reverse=True)
        picked: list[str] = []
        for _z, feature in scored:
            # Skip cues already covered by a stronger one.
            if any(feature in p or p in feature for p in picked):
                continue
            picked.append(feature)
            if len(picked) >= CUES_PER_CLASS:
                break
        self._cue_cache[label] = picked
        return picked

    def profiled(self) -> list[str]:
        return [c for c in self.class_docs if self.cues(c)]

    def rank(self, text: str) -> list[tuple[float, str]]:
        """Rank classes by the share of their distinguishing cues the text exhibits."""
        feats = _features(text)
        scored = []
        for label in self.class_docs:
            cues = self.cues(label)
            if not cues:
                continue
            hits = sum(1 for c in cues if c in feats)
            scored.append((hits / len(cues), label))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored


class _Index:
    """Incremental TF-IDF inverted index over char n-grams (script agnostic)."""

    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.df: Counter = Counter()
        self.n_docs = 0

    def add(self, text: str) -> None:
        grams: Counter = Counter()
        t = text.lower()
        for n in NGRAM_SIZES:
            for i in range(max(0, len(t) - n + 1)):
                grams[t[i : i + n]] += 1
        weights = {g: 1.0 + math.log(c) for g, c in grams.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        doc_id = self.n_docs
        for g, w in weights.items():
            self.postings[g].append((doc_id, w / norm))
            self.df[g] += 1
        self.n_docs += 1

    def search(self, text: str) -> list[tuple[float, int]]:
        if not self.n_docs:
            return []
        grams: Counter = Counter()
        t = text.lower()
        for n in NGRAM_SIZES:
            for i in range(max(0, len(t) - n + 1)):
                grams[t[i : i + n]] += 1
        scores: dict[int, float] = defaultdict(float)
        for g, c in grams.items():
            postings = self.postings.get(g)
            if not postings:
                continue
            qw = (1.0 + math.log(c)) * math.log(1 + self.n_docs / (1 + self.df[g]))
            for doc_id, dw in postings:
                scores[doc_id] += qw * dw
        return sorted(((s, i) for i, s in scores.items()), key=lambda x: (-x[0], x[1]))


class ClassPrototypeMemory(MemorySystem):
    """Memory of compressed class descriptions rather than of raw cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._profiles = _Profiles()
        self._affix = _Affix()
        self._template = _Template()

    # ---------------- gating ----------------

    def _inventory_enabled(self) -> bool:
        return (
            len(self.examples) >= MIN_EXAMPLES_FOR_GATE
            and self._inventory.reuse_rate() >= MIN_REUSE_RATE
        )

    def _profiles_enabled(self) -> bool:
        return self._inventory_enabled() and bool(self._profiles.profiled())

    # ---------------- prompt sections ----------------

    def _format_profiles(self, ranked: list[tuple[float, str]]) -> str:
        if not ranked:
            return ""
        lines, total = [], 0
        for score, label in ranked[:CANDIDATE_CLASSES]:
            # A class none of whose distinguishing evidence is present is not a
            # candidate; listing it as one only invites a guess.
            if score <= 0:
                break
            cues = self._profiles.cues(label)
            if not cues:
                continue
            line = f"- {label} -> " + "; ".join(cues)
            if total + len(line) > MAX_PROFILE_CHARS:
                break
            lines.append(line)
            total += len(line) + 1
        if not lines:
            return ""
        return (
            "**Candidate answers, with the evidence that distinguishes each one:**\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def _format_inventory(self, exclude: list[str]) -> str:
        if not self._inventory_enabled():
            return ""
        skip = set(exclude)
        parts, total = [], 0
        for answer, _c in self._inventory.units.most_common():
            if answer in skip:
                continue
            line = f"- {answer}"
            if total + len(line) > MAX_INVENTORY_CHARS:
                break
            parts.append(line)
            total += len(line) + 1
        if not parts:
            return ""
        header = (
            "**Other answer strings used in previously solved cases:**\n"
            if skip
            else "**Answer strings used in previously solved cases:**\n"
        )
        return header + "\n".join(parts) + "\n\n"

    def _retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        used: Counter = Counter()
        picked: list[dict[str, str]] = []
        for _score, doc_id in self._index.search(query):
            ex = self.examples[doc_id]
            if used[ex["target"]] >= PER_ANSWER_CAP:
                continue
            picked.append(ex)
            used[ex["target"]] += 1
            if len(picked) >= top_k:
                break
        return picked

    def _format_cases(self, query: str) -> str:
        """Format anchors when profiles carry the signal, full retrieval otherwise."""
        profiled = self._profiles_enabled()
        top_k = ANCHOR_EXAMPLES if profiled else FALLBACK_TOP_K
        max_chars = MAX_ANCHOR_CHARS if profiled else FALLBACK_EXEMPLAR_CHARS
        picked = self._retrieve(query, top_k)
        if not picked:
            return ""
        parts, total = [], 0
        for ex in reversed(picked):
            question = ex.get("raw_question", ex["input"])
            part = f"Q: {question}\nA: {ex['target']}"
            if total + len(part) > max_chars:
                continue
            parts.append(part)
            total += len(part) + 2
        if not parts:
            return ""
        header = (
            "**Solved cases, for answer format:**\n\n"
            if profiled
            else "**Similar solved cases:**\n\n"
        )
        return header + "\n\n".join(parts) + "\n\n"

    # ---------------- canonicalisation ----------------

    def _snap_unit(self, unit: str, targets: list[str]) -> str:
        if unit in self._inventory.units:
            return unit
        bigrams = _answer_ngrams(unit)
        unigrams = _answer_ngrams(unit, 1)
        scored = []
        for t in targets:
            t_uni = self._inventory._unigrams[t]
            # Symmetric agreement, plus asymmetric evidence that the answer is a
            # sloppy rendering of this known one.
            sym = 0.5 * _dice(unigrams, t_uni) + 0.5 * _dice(
                bigrams, self._inventory._grams[t]
            )
            scored.append((0.5 * sym + 0.5 * _coverage(unigrams, t_uni), t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored or scored[0][0] < SNAP_THRESHOLD:
            return unit
        if len(scored) > 1 and scored[0][0] - scored[1][0] < SNAP_MARGIN:
            return unit
        return scored[0][1]

    def _canonicalize(self, answer: str) -> str:
        if not answer.strip() or not self._inventory_enabled():
            return answer
        body = self._affix.strip(answer)
        if not body:
            return answer
        targets = list(self._inventory.units)
        snapped = [self._snap_unit(u, targets) for u in self._inventory.split(body)]
        if not snapped:
            return answer
        joined = (self._inventory.delimiter() or ";").join(dict.fromkeys(snapped))
        if joined == body:
            return answer
        return answer.replace(body, joined) if body in answer else joined

    # ---------------- interface ----------------

    def _sections(self, input: str) -> tuple[str, str, str, list[str]]:
        query = self._template.strip(input)
        ranked = self._profiles.rank(query) if self._profiles_enabled() else []
        shown = [label for s, label in ranked[:CANDIDATE_CLASSES] if s > 0]
        return (
            self._format_profiles(ranked),
            self._format_inventory(shown),
            self._format_cases(query),
            shown,
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profile_section, inventory_section, case_section, shown = self._sections(input)
        prompt = PROMPT.format(
            profile_section=profile_section,
            inventory_section=inventory_section,
            anchor_section=case_section,
            input=input,
            profile_rule=PROFILE_RULE if profile_section else "",
            inventory_rule=INVENTORY_RULE if self._inventory_enabled() else "",
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")

        retried = False
        if not answer.strip():
            retried = True
            response = self.call_llm(TERSE_RETRY.format(input=input))
            answer = extract_json_field(response, "final_answer")
            if not answer.strip():
                answer = response.strip()[:200]

        canonical = self._canonicalize(answer)
        return canonical, {
            "full_response": response,
            "candidates": shown,
            "profiles_on": bool(profile_section),
            "snapped": canonical != answer,
            "retried": retried,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            target = r["ground_truth"]
            question = r.get("raw_question", r["input"])
            self._affix.observe(r.get("prediction", ""), target)
            ex = {"input": r["input"], "target": target}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            ex["prediction"] = r.get("prediction", "")
            self.examples.append(ex)
            self._template.observe(r["input"])
            self._index.add(question)
            rebuilt = self._inventory.observe(target)
            if rebuilt:
                # Units were re-split, so the class keys the profiles are
                # accumulated under are no longer the right ones.
                self._profiles = _Profiles()
                for stored in self.examples:
                    q = stored.get("raw_question", stored["input"])
                    self._profiles.add(q, self._inventory.split(stored["target"]))
            else:
                self._profiles.add(question, self._inventory.split(target))

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        profile_section, inventory_section, case_section, _ = self._sections(
            self.examples[-1]["input"]
        )
        return len(profile_section) + len(inventory_section) + len(case_section)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = []
        self._index = _Index()
        self._inventory = _Inventory()
        self._profiles = _Profiles()
        self._affix = _Affix()
        self._template = _Template()
        for ex in data.get("examples", []):
            record = {
                "input": ex["input"],
                "ground_truth": ex["target"],
                "prediction": ex.get("prediction", ""),
                "was_correct": False,
            }
            if "raw_question" in ex:
                record["raw_question"] = ex["raw_question"]
            self.learn_from_batch([record])
