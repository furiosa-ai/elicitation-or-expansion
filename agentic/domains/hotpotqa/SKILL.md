---
name: meta-harness
description: Run one iteration of harness evolution for HotpotQA multi-hop QA. Called by core/meta_harness.py.
---

# Meta-Harness (HotpotQA harness evolution)

Run ONE iteration of harness evolution. Do all work in the main session; do NOT delegate to subagents
(constraints get lost when you delegate, leading to parameter-only changes and skipped prototyping).

**You do NOT run the search-set benchmark.** You analyze results and traces, prototype, and implement new
harnesses. The outer loop evaluates every candidate on the full search set (300 questions) afterwards.

## CRITICAL CONSTRAINTS

- Produce exactly the number of candidates the task prompt asks for (normally 2), every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", and do not abort early.
- One mechanism per candidate; each candidate tests one falsifiable hypothesis. Mix exploitation and exploration.
- Never edit existing files in `agents/`; copy one to a NEW file and modify the copy. `core/`, `domains/`, `logs/`
  and `data/` are read-only.
- Candidate code must not do file or network I/O and must not construct LLM clients, retrievers or `AppWorld` objects
  itself. Use only `ctx.llm` and `ctx.tools`. The static audit REJECTS: imports of os, sys, subprocess, socket,
  requests, urllib, httpx, shutil, pathlib, glob, io, importlib, ctypes, pickle, multiprocessing, appworld, openai,
  litellm, sqlite3, tempfile, asyncio; and calls to the builtins `open(`, `exec(`, `eval(`, `compile(`, `__import__(`
  (use `ast.parse` for syntax checks, never `compile`). Module attribute calls such as `re.compile` are fine.
- Experiment controls are fixed and enforced (solver model, temperature, reasoning effort, max_tokens, item
  budgets). Do not try to work around them (e.g. internal best-of-N sampling): exceeding the budget fails the item.

### Time budget (hard)

The session has a hard wall-clock limit (given in the task prompt). When it expires, the session is killed and
nothing counts: unregistered files are discarded and the iteration is recorded as "no proposal". Therefore:
- write `proposals/pending_eval.json` as soon as each candidate import-checks and passes ONE small prototype run;
- refine afterwards if time remains and re-write the manifest, but never leave it for the end;
- cap prototyping: a few `--limit 5` runs per candidate, not repeated 20-item sweeps of many variants.

### Anti-parameter-tuning rules

The most common failure mode is a candidate that is just a parameter variant of an existing one (k=7 -> k=10,
2 hops -> 3 hops, a reworded instruction). Check `logs/evolution_summary.jsonl` for what has been tried.

Good candidates change a fundamental mechanism:
- a new retrieval strategy (query decomposition, entity-anchored second-hop queries, reranking with the LLM,
  deduplicated evidence pools, adaptive hop count driven by an explicit sufficiency check)
- a new context-construction architecture (structured evidence tables, quote extraction instead of summaries,
  passing raw passages to the answerer, contrastive "candidate answers with evidence")
- a new answer policy (answer normalization to entity/number/yes-no, verification against evidence, abstain-and-retry
  within the call budget)
- a new use of the train split in `prepare` (bootstrapped few-shot demos selected by question type, learned
  query templates, failure-pattern notes) -- the train split is the ONLY data you may learn from

Bad candidates just tune numbers or reword one instruction. If `run()` is identical to the parent except for
constants, rewrite with a genuinely novel mechanism. Combining mechanisms from two systems is valid.

### Anti-overfitting / anti-leakage rules

- Never hard-code questions, answers, entity names or Wikipedia titles from the data. The audit rejects any
  candidate containing held-out literals, and memorized search-set answers do not transfer to the test split.
- General multi-hop QA strategies are fine ("resolve the bridge entity first", "answer comparison questions with
  yes/no", "return the entity, not a sentence"). Dataset-specific hints are not.
- Do not mention dataset or benchmark names in code, prompts or comments.

## WORKFLOW

### Step 0: Post-eval reports (write if missing)

For each past iteration in `logs/evolution_summary.jsonl` that has no report in `proposals/reports/`, write
`proposals/reports/iter<N>.md` (<= 30 lines): what changed, why it helped/hurt (cite item ids from traces),
takeaway for future iterations.

### Step 1: Analyze

1. Read `logs/evolution_summary.jsonl`, `logs/frontier_val.json`, `logs/val_table.txt`.
2. Deep-read traces of the current best system and of the baseline: `logs/val/<agent>/items.jsonl` lists each
   question with `correct`, `answer`, `metrics.f1`, `metrics.gold_title_recall`, `metrics.retrieval_calls`;
   `logs/val/<agent>/items/<id>.json` has every prompt, response, retrieval query and the titles retrieved.
   Categorize failures: retrieval misses (gold_title_recall < 1), wrong bridge entity, verbose answers that fail
   exact match (f1 > 0 but not correct), reasoning errors with correct evidence, budget/format failures.
3. Formulate one falsifiable hypothesis per candidate, each targeting a different mechanism.

### Step 2: Prototype (MANDATORY)

Run the candidate on a handful of search-set items with the real solver before finalizing:

    cd /workspace && python -m core.evaluate --config config.container.yaml --agent agents/<name>.py \
        --split val --out proposals/proto/<name> --limit 5

Read `proposals/proto/<name>/items.jsonl` and the traces. Fix crashes and format bugs. Try 2-3 variants of the
mechanism if cheap. Prototyping is capped at 20 items per call.

### Step 3: Implement

For each candidate:
1. Copy the most relevant existing agent to `agents/<snake_case_name>.py` and implement the mechanism.
2. Keep `class Harness(BaseHarness)` and the interface below.
3. Self-critique: is this a NEW mechanism or a parameter variant? If the latter, rewrite.
4. Validate the import: `cd /workspace && python -c "from core.harness import load_harness_class; load_harness_class('agents/<name>.py')"`

### Step 4: Write proposals/pending_eval.json

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<snake_case_name>",
      "file": "agents/<name>.py",
      "hypothesis": "<falsifiable claim about what improves exact match and why>",
      "axis": "exploitation|exploration",
      "base_system": "<agent it builds on>",
      "components": ["tag1", "tag2"]
    }
  ]
}
```

Finish with the line: `CANDIDATES: <name1>, <name2>`

## Harness interface

```python
from core.harness import BaseHarness, RunContext, PrepareContext

class Harness(BaseHarness):
    def prepare(self, train_items, ctx: PrepareContext) -> None: ...   # optional; train items include "answer" and "gold_titles"
    def get_state(self) -> str: ...                                     # JSON string saved after prepare, restored before eval
    def set_state(self, state: str) -> None: ...
    def run(self, item, ctx: RunContext) -> dict:                       # item = {"id", "question"}; return {"answer": "<short answer>", ...}
```

- `ctx.llm.chat(messages, stop=None, max_tokens=None, tag=None)` -> response with `.text` (and `.reasoning`);
  `ctx.llm(prompt, system=None)` -> str. Each call counts against the per-item budget (see task prompt).
- `ctx.tools.retrieve(query, k=7)` -> list of passages formatted `"Title | abstract"` (deterministic BM25 over 5.2M
  Wikipedia abstracts); `ctx.tools.retrieve_docs(query, k)` -> dicts with `text`, `title`, `pid`, `score`.
  Retrieval calls are capped per item and k is capped (see task prompt).
- `ctx.note(message)` adds a free-form annotation to the trace.
- In `prepare`, `ctx.llm` is available with a separate (larger) budget. Keep `get_state()` small (< ~200 KB).
- `run()` must be stateless across items and thread-safe (items run concurrently in one process).
- Scoring: exact match after normalization (lowercase, strip punctuation and articles) against the gold answer.
  A correct fact phrased as a sentence scores 0. Yes/no questions expect literally `yes` or `no`.
- Helpers you may import: `domains.hotpotqa.dspy_format.Module` (DSPy ChatAdapter-style prompts with reasoning),
  `domains.hotpotqa.evaluator.normalize_text / em_score` (e.g. to self-check on TRAIN items in prepare).

## Directory structure (inside the sandbox)

- `agents/<name>.py`                      candidates and baselines
- `logs/val/<agent>/summary.json`         accuracy, n_errors, avg_calls, avg_prompt_tokens, avg_f1, avg_gold_title_recall
- `logs/val/<agent>/items.jsonl`          per-item results
- `logs/val/<agent>/items/<id>.json`      full traces
- `logs/evolution_summary.jsonl`          one row per candidate: iteration, agent, status, accuracy, delta, hypothesis, components
- `logs/frontier_val.json`                `_best` and `_pareto` (accuracy vs avg_prompt_tokens)
- `proposals/`                            the only writable place besides new files in `agents/`
