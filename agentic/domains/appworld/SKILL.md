---
name: meta-harness
description: Run one iteration of harness evolution for AppWorld interactive-coding agents. Called by core/meta_harness.py.
---

# Meta-Harness (AppWorld agent harness evolution)

Run ONE iteration of harness evolution. Do all work in the main session; do NOT delegate to subagents.

**You do NOT run the search-set benchmark.** You analyze results and trajectories, prototype, and implement new
agent harnesses. The outer loop evaluates every candidate on the full search set (57 dev tasks) afterwards.

## CRITICAL CONSTRAINTS

- Produce exactly the number of candidates the task prompt asks for (normally 2), every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", and do not abort early.
- One mechanism per candidate; each candidate tests one falsifiable hypothesis. Mix exploitation and exploration.
- Never edit existing files in `agents/`; copy one to a NEW file and modify the copy. `core/`, `domains/`, `logs/`,
  `data/` and `appworld_dev/data/` are read-only.
- Candidate code must not do file or network I/O and must not construct LLM clients, retrievers or `AppWorld` objects
  itself. Use only `ctx.llm` and `ctx.tools`. The static audit REJECTS: imports of os, sys, subprocess, socket,
  requests, urllib, httpx, shutil, pathlib, glob, io, importlib, ctypes, pickle, multiprocessing, appworld, openai,
  litellm, sqlite3, tempfile, asyncio; and calls to the builtins `open(`, `exec(`, `eval(`, `compile(`, `__import__(`
  (use `ast.parse` for syntax checks, never `compile`). Module attribute calls such as `re.compile` are fine.
- Experiment controls are fixed and enforced (solver model, temperature, reasoning effort, max_tokens, max steps,
  per-task output-token budget). Exceeding a budget fails the task.

### AppWorld benchmark rules (from the benchmark authors)

- Only the `train` (90 tasks) and `dev` (57 tasks, the search set) splits exist in this sandbox. Held-out tasks are
  not available and must not be guessed at.
- Do NOT hard-code API calls into the agent's logic (e.g. logging into every app in a fixed first step and stashing
  tokens). It IS fine to *instruct* the agent in the prompt to do such things itself.
- Generic hints inferred from train/dev failures are allowed ("read the API doc before calling", "paginate to the
  end", "call complete_task exactly once"). Task-specific hints or scenario-specific logic are not.
- Do not mention benchmark or split names in code, prompts or comments.

### Time budget (hard)

The session has a hard wall-clock limit (given in the task prompt). When it expires, the session is killed and
nothing counts: unregistered files are discarded and the iteration is recorded as "no proposal". Therefore:
- write `proposals/pending_eval.json` as soon as each candidate import-checks and passes ONE small prototype run;
- refine afterwards if time remains and re-write the manifest, but never leave it for the end;
- cap prototyping: a few `--limit 5` runs per candidate, not repeated 20-item sweeps of many variants.

### Anti-parameter-tuning rules

Check `logs/evolution_summary.jsonl` for what has been tried. Good candidates change a mechanism:
- how the API documentation is surfaced (pre-fetched app/API docs relevant to the task, structured doc summaries,
  doc-on-demand policies)
- how execution feedback is handled (error-triage step, explicit state/plan tracking between steps, output trimming
  and summarization for long histories, retry policies for exceptions)
- how the episode is structured (plan-then-act, checklist verification before `complete_task`, self-check of the
  answer format, early termination on repeated failures)
- how the train split is used in `prepare` (playbooks of reusable code idioms and pitfalls mined from train
  trajectories, bootstrapped demonstrations chosen by app) -- train is the only data you may learn from

Bad candidates just reword the instructions or change a number (max steps, truncation lengths). If `run()` is
identical to the parent except for constants or a paragraph of prompt text, rewrite with a real mechanism.

## WORKFLOW

### Step 0: Post-eval reports (write if missing)

For each past iteration in `logs/evolution_summary.jsonl` without a report in `proposals/reports/`, write
`proposals/reports/iter<N>.md` (<= 30 lines): what changed, which tasks/scenarios improved or regressed and why
(cite task ids), takeaway for future iterations.

### Step 1: Analyze

1. Read `logs/evolution_summary.jsonl`, `logs/frontier_val.json`, `logs/val_table.txt`.
2. Deep-read trajectories of the current best system and the baseline: `logs/val/<agent>/items.jsonl` lists each
   task with `correct` (Task Goal Completion), `metrics.steps`, `metrics.completed_call`, `error`;
   `logs/val/<agent>/items/<id>.json` has every LLM call (delta-encoded messages), every executed code block and
   its output (`env` events). Categorize failures: wrong API usage, missing pagination/time handling, wrong answer
   format, premature or missing `complete_task`, step-budget exhaustion, context overflow, exceptions.
3. Formulate one falsifiable hypothesis per candidate, each targeting a different mechanism.

### Step 2: Prototype (MANDATORY)

    cd /workspace && python -m core.evaluate --config config.container.yaml --agent agents/<name>.py \
        --split val --out proposals/proto/<name> --limit 3

Each dev task takes 30-120 s. Read `proposals/proto/<name>/items.jsonl` and the traces; fix crashes and prompt
bugs. Prototyping is capped at 20 tasks per call.

### Step 3: Implement

1. Copy the most relevant existing agent to `agents/<snake_case_name>.py` and implement the mechanism.
2. Keep `class Harness(BaseHarness)` and the interface below.
3. Self-critique: new mechanism or parameter variant? Rewrite if the latter.
4. Validate the import: `cd /workspace && python -c "from core.harness import load_harness_class; load_harness_class('agents/<name>.py')"`

### Step 4: Write proposals/pending_eval.json

```json
{
  "iteration": <N>,
  "candidates": [
    {"name": "<snake_case_name>", "file": "agents/<name>.py", "hypothesis": "<falsifiable claim>",
     "axis": "exploitation|exploration", "base_system": "<agent it builds on>", "components": ["tag1", "tag2"]}
  ]
}
```

Finish with the line: `CANDIDATES: <name1>, <name2>`

## Harness interface

```python
from core.harness import BaseHarness, RunContext, PrepareContext

class Harness(BaseHarness):
    def prepare(self, train_items, ctx: PrepareContext) -> None: ...   # optional; train_items = [{"id","task_id",...}]
    def get_state(self) -> str: ...                                     # JSON string saved after prepare, restored before eval
    def set_state(self, state: str) -> None: ...
    def run(self, item, ctx: RunContext) -> dict:                       # item = {"id","task_id"}; return {"completed": bool, "steps": int, ...}
```

- `env = ctx.tools` is the task environment: `env.instruction`, `env.supervisor` (dict: first_name, last_name, email,
  phone_number), `env.app_descriptions` (dict), `env.execute(code) -> str` (runs Python in the task's stateful REPL
  where the `apis` object is available; counts one step; `max_steps` enforced), `env.task_completed() -> bool`.
- `ctx.llm.chat(messages, stop=None, max_tokens=None, tag=None)` -> response with `.text`; each call counts
  against the per-task call and output-token budget.
- In `prepare`, `with ctx.open_tools(train_item) as env:` opens a TRAIN task; there `env.evaluate() -> bool` gives
  the ground-truth outcome so you can mine successful trajectories. Keep `get_state()` small (< ~500 KB).
- Scoring is done by the benchmark's state checks after `run()` returns (Task Goal Completion). The returned dict
  is informational. A task is only checked if `apis.supervisor.complete_task(...)` was called by the executed code.
- Helpers you may import: `domains.appworld.prompting` (`render_react_prompt`, `text_to_messages`,
  `extract_code_and_fix_content`, `format_output_message`, `REACT_INSTRUCTIONS`).
- Context window is 32k tokens for the solver; long episodes must manage history length.

## Directory structure (inside the sandbox)

- `agents/<name>.py`                      candidates and baselines
- `logs/val/<agent>/summary.json`         accuracy (TGC), sgc, n_errors, avg_calls, avg_prompt_tokens, avg_steps
- `logs/val/<agent>/items.jsonl`          per-task results
- `logs/val/<agent>/items/<id>.json`      full trajectories
- `logs/evolution_summary.jsonl`          one row per candidate: iteration, agent, status, accuracy, delta, hypothesis, components
- `logs/frontier_val.json`                `_best` and `_pareto` (accuracy vs avg_prompt_tokens)
- `proposals/`                            the only writable place besides new files in `agents/`
