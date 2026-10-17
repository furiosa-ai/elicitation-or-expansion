"""Autonomous evolution loop for memory systems.

Validation-only during evolution. Test evaluation requires explicit finalization.
Uses claude_wrapper + meta-harness skill to propose new memory systems.

    uv run python meta_harness.py --iterations 20 --fresh
    uv run python meta_harness.py --iterations 10 --run-name my-run
    uv run python meta_harness.py --run-name my-run --test
"""

import argparse
import json
import os
import signal
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import claude_wrapper
from benchmark import get_model_short_name, load_results
from process_control import run_command

EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = EVOLVE_DIR / "config.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}

# These are updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
RESULTS_DIR = LOGS_DIR / "results"
FINALIZED = LOGS_DIR / "finalized.json"

# ---------------------------------------------------------------------------
# Changes relative to upstream (the evolution loop, evaluation, prompts and SKILL are unchanged):
#   - Per-run isolation: every run uses its own copy of this checkout.
#   - Test isolation: test results and test data live outside the proposer-visible checkout.
#   - Proposer sandbox: only this checkout is mounted into an unprivileged container.
#   - The proposer model is pinned instead of using an alias.
#   - Claude Code auto-memory is disabled and the proposer HOME's project state is cleared before every
#     iteration, so the filesystem history is the only memory across iterations.
#   - An iteration that produces no valid candidate is recorded as failed instead of ending the run.
#   - The run name is part of the container name so that two runs can execute at the same time.
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = EVOLVE_DIR.parent.parent          # checkout root of this run
CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_EVOLVE_DIR = (
    CONTAINER_WORKSPACE / "reference_examples" / "text_classification"
)
PROPOSER_IMAGE = os.environ.get("TEXT_PROPOSER_IMAGE")
PROPOSER_HOME = os.environ.get("TEXT_PROPOSER_HOME")
PROPOSER_CPUS = os.environ.get("TEXT_PROPOSER_CPUS")
PROPOSER_MEMORY = os.environ.get("TEXT_PROPOSER_MEMORY")
PROPOSER_MODEL = os.environ.get("TEXT_PROPOSER_MODEL", "claude-opus-5")


def to_container_path(host_path):
    """Map a host path inside the checkout to the path the proposer sees."""
    host_path = Path(host_path).resolve()
    return CONTAINER_WORKSPACE / host_path.relative_to(WORKSPACE_ROOT.resolve())

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

_interrupted = False

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=7200, cwd=None):
    return run_command(cmd, timeout=timeout, cwd=cwd, cancelled=lambda: _interrupted)


def run_benchmark(args):
    return run_cmd(
        [
            "uv",
            "run",
            "python",
            "benchmark.py",
            "--logs-dir",
            str(LOGS_DIR),
            "--results-dir",
            str(RESULTS_DIR),
        ]
        + args,
        cwd=str(EVOLVE_DIR),
        timeout=None,  # Dataset attempts own the deadline and retry budget.
    )


def complete_scores(systems, datasets, model_short):
    """A score is defined only when all requested datasets completed."""
    results = load_results(LOGS_DIR, "val.json")
    missing = [(name, ds) for name in systems for ds in datasets
               if (model_short, ds, name) not in results
               or not isinstance(results[(model_short, ds, name)].get("accuracy"), (int, float))]
    if missing:
        raise ValueError(f"Incomplete evaluation: {missing}")
    return {name: sum(results[(model_short, ds, name)]["accuracy"] * 100
                      for ds in datasets) / len(datasets) for name in systems}


def evaluation_failed(phase, detail):
    """Stop before consuming an evaluation iteration or exposing partial scores."""
    (LOGS_DIR / "evaluation_failure.json").write_text(json.dumps(
        {"phase": phase, "detail": str(detail), "time": datetime.now().isoformat()}, indent=2) + "\n")
    raise SystemExit(f"Evaluation stopped ({phase}): {detail}. Fix and resume this run; no evaluation iteration was consumed.")


def render_task_prompt(iteration, num_datasets):
    """Build the prompt for the proposer session; paths are given as seen inside the container."""
    c_logs = to_container_path(LOGS_DIR)
    return (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        + f"## Run directories\n"
        f"All logs and results for this run are under `{c_logs}/`.\n"
        f"- `{to_container_path(EVOLUTION_SUMMARY)}` — past results\n"
        f"- `{to_container_path(FRONTIER_VAL)}` — frontier\n"
        f"- `{c_logs / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{to_container_path(PENDING_EVAL)}`"
    )


def count_iterations_from_summary():
    """Highest iteration number in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def _run_tag():
    """Run identifier used in container names."""
    return LOGS_DIR.name or "run"


def container_spec(iteration):
    """Sandbox contract for the proposer container."""
    return {
        "image": PROPOSER_IMAGE,
        # PYTHONPATH lets the SKILL import check (`from text_classification.agents.<name> import *`) run from the
        # package directory; the proposer's uv environment lives in its HOME so the host .venv is never touched.
        "extra_env": {
            "PYTHONPATH": str(CONTAINER_WORKSPACE / "reference_examples"),
            "UV_PROJECT_ENVIRONMENT": "/proposer-home/uv-venv",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        },
        "workspace_root": str(WORKSPACE_ROOT),
        "home_dir": PROPOSER_HOME,
        "cwd": str(CONTAINER_EVOLVE_DIR),
        # The run name keeps container names unique when runs execute concurrently.
        "name": f"text-proposer-{_run_tag()}-iter{iteration}",
        "cpus": PROPOSER_CPUS,
        "memory": PROPOSER_MEMORY,
        # A tmpfs hides the checkout's .venv from the proposer.
        "shadow_venv": str(CONTAINER_EVOLVE_DIR / ".venv"),
    }


def propose_claude(task_prompt, iteration, timeout=2400):
    """Returns True if candidates were produced (pending_eval.json exists)."""
    os.environ.pop("CLAUDECODE", None)
    # Strip API key so claude CLI uses subscription auth (avoids rate limits)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    # the proposer never runs directly on the host.
    if not (PROPOSER_IMAGE and PROPOSER_HOME and os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        raise RuntimeError(
            "TEXT_PROPOSER_IMAGE, TEXT_PROPOSER_HOME, CLAUDE_CODE_OAUTH_TOKEN "
            "are all required (see the repository README)."
        )
    # Credentials must not be inside the mounted checkout, where the proposer could read them.
    exposed = [p for p in (WORKSPACE_ROOT / ".env", EVOLVE_DIR.parent / ".env", EVOLVE_DIR / ".env") if p.exists()]
    if exposed:
        raise RuntimeError(f"{exposed[0]} is inside the proposer-visible checkout; remove it (see the repository README).")
    _scrub_proposer_home()
    container = container_spec(iteration)
    container["oauth_token"] = True   # the value is passed by environment-variable name only
    result = claude_wrapper.run(
        prompt=task_prompt,
        model=PROPOSER_MODEL,   # pinned model instead of the alias "opus"
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(EVOLVE_DIR / ".claude/skills/meta-harness")],
        cwd=str(CONTAINER_EVOLVE_DIR),
        container=container,
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="max",
    )
    # Restore API key
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def _scrub_proposer_home():
    """Delete Claude Code project state (including auto-memory) left in the proposer HOME."""
    import shutil

    d = Path(PROPOSER_HOME) / ".claude" / "projects"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def propose(task_prompt, iteration, timeout):
    return propose_claude(task_prompt, iteration, timeout=timeout)


def load_pending_candidates(path):
    """Read pending_eval.json as the candidate list.

    Accepts {"iteration": N, "candidates": [...]} (the SKILL format) or a bare list of candidates.
    Anything else raises ValueError, and the caller records only that iteration as failed.
    """
    manifest = json.loads(Path(path).read_text())
    if isinstance(manifest, list):
        manifest = {"candidates": manifest}
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object, got {type(manifest).__name__}")
    cands = manifest.get("candidates", [])
    if not isinstance(cands, list) or not all(isinstance(c, dict) for c in cands):
        raise ValueError("manifest 'candidates' must be a list of objects")
    return cands


def record_failed_iteration(iteration, status, reason="", propose_time=None):
    """Record an iteration that produced no candidate, so that it counts against the budget on resume."""
    row = {
        "iteration": iteration,
        "system": None,
        "avg_val": 0,
        "axis": "?",
        "hypothesis": "",
        "delta": None,
        "outcome": status,
    }
    if reason:
        row["reason"] = reason[:500]
    if propose_time is not None:
        row["timing_s"] = {"propose": round(propose_time, 1)}
    with open(EVOLUTION_SUMMARY, "a") as f:
        f.write(json.dumps(row) + "\n")


def validate_candidates(candidates):
    """Import-check each candidate. Returns list of valid candidates."""
    valid = []
    for c in candidates:
        name = c["name"]
        result = run_cmd(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"from text_classification.agents.{name} import *; print('OK')",
            ],
            cwd=str(EVOLVE_DIR.parent),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    propose_time=None,
    bench_time=None,
    wall_time=None,
):
    """Append one JSONL row per candidate to evolution_summary.jsonl."""
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])
    best_val = pareto[0].get("val_accuracy", 0) if pareto else 0

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1),
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": round(avg_val - best_val, 1) if best_val else None,
                "outcome": f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                if avg_val > 0
                else "failed",
            }
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row) + "\n")


def fresh_start():
    """Clear proposed memory systems and reset logs for a fresh run."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    if AGENTS_DIR.exists():
        files = [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]
        for f in files:
            f.unlink()
        if files:
            print(f"  Cleared {len(files)} candidate file(s) from agents/")

    for f in [
        EVOLUTION_SUMMARY,
        FRONTIER_VAL,
        LOGS_DIR / "frontier.json",
        PENDING_EVAL,
    ]:
        if f.exists():
            f.unlink()

    if LOGS_DIR.exists():
        val_files = list(LOGS_DIR.rglob("val.json"))
        for f in val_files:
            f.unlink()
        if val_files:
            print(f"  Cleared {len(val_files)} val result files")

    print(f"  {_green('Fresh start')}: cleared generated agents and log files")


def finalize_run(baselines, datasets, model_short):
    """Evaluate test once, after freezing this run against further evolution."""
    if not FRONTIER_VAL.exists():
        print(f"ERROR: no validation frontier for run at {LOGS_DIR}")
        raise SystemExit(1)

    if FINALIZED.exists():
        state = json.loads(FINALIZED.read_text())
        if state.get("status") == "complete":
            print(f"Run already finalized: {LOGS_DIR.name}")
            result = run_benchmark(["--results", "--test"])
            if result.stdout:
                print(result.stdout)
            return

    frontier = json.loads(FRONTIER_VAL.read_text())
    pareto = frontier.get("_pareto", [])
    test_systems = set(baselines)
    test_systems.update(entry["system"] for entry in pareto)
    for key, value in frontier.items():
        if not key.startswith("_") and isinstance(value, dict):
            if "best_system" in value:
                test_systems.add(value["best_system"])

    FINALIZED.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "systems": sorted(test_systems),
            },
            indent=2,
        )
    )

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")
    failed = False
    # Test evaluation runs several systems concurrently.
    _fw = max(1, int(os.environ.get("TEXT_FINALIZE_WORKERS", "3")))
    _names = sorted(test_systems)

    def _test_one(name):
        t0 = time.time()
        return name, run_benchmark(["--memory", name, "--test"]), time.time() - t0

    print(f"  {_ts()} {len(_names)} system, {_fw} concurrent", flush=True)
    for name in _names:
        print(f"    {_ts()} start {_bold(name)}", flush=True)
    with ThreadPoolExecutor(max_workers=_fw) as _exe:
        for name, result, secs in sorted(_exe.map(_test_one, _names)):
            if result.returncode != 0:
                failed = True
                print(f"    {_red('FAIL')} {name} test eval failed ({_elapsed(secs)})")
            else:
                print(f"    {_green('OK')} {name} ({_elapsed(secs)})")

    frontier_result = run_benchmark(["--frontier", "--test", "--model", model_short])
    failed = failed or frontier_result.returncode != 0

    result = run_benchmark(["--results", "--test"])
    failed = failed or result.returncode != 0
    if result.stdout:
        print(result.stdout)

    test_results = load_results(RESULTS_DIR, "test.json")
    missing = [
        (model_short, dataset, system)
        for system in test_systems
        for dataset in datasets
        if (model_short, dataset, system) not in test_results
    ]
    if missing:
        failed = True
        print(f"Missing {len(missing)} complete test result group(s).")

    if failed:
        print("Test finalization incomplete. Fix failures, then rerun --test.")
        raise SystemExit(1)

    state = json.loads(FINALIZED.read_text())
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    FINALIZED.write_text(json.dumps(state, indent=2))
    print(f"\n{_ts()} {_bold('Test finalization complete.')}")


def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global RESULTS_DIR, FINALIZED

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["datasets"]

    config_model_ids = [m["model"] for m in cfg.get("models", [])]
    if args.model not in config_model_ids:
        print(f"ERROR: --model {args.model} not in config.yaml: {config_model_ids}")
        sys.exit(1)

    model_short = get_model_short_name(args.model)

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    # Test results are kept outside the checkout, where the proposer cannot read them.
    _results_root = os.environ.get("TEXT_RESULTS_ROOT")
    RESULTS_DIR = (
        Path(_results_root) / run_name if _results_root else LOGS_DIR / "results"
    )
    FINALIZED = LOGS_DIR / "finalized.json"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = cfg["memory_systems"]["baselines"]
    if args.test:
        finalize_run(baselines, datasets, model_short)
        return

    if (
        FINALIZED.exists()
        and json.loads(FINALIZED.read_text()).get("status") == "complete"
    ):
        print(
            f"ERROR: run '{run_name}' is finalized; use a new --run-name to continue evolution"
        )
        raise SystemExit(1)

    if args.fresh:
        fresh_start()

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  model={_cyan(args.model)}  "
        f"iters={args.iterations}  datasets={datasets}"
    )
    print(f"{_ts()} proposer={_cyan('claude')} model={_cyan(PROPOSER_MODEL)} effort=max")

    # ── Phase 0: Baselines ─────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl])
            elapsed = time.time() - t0
            if result.returncode != 0:
                if _interrupted:
                    return
                evaluation_failed("baseline", f"{bl}: {result.stderr[-500:]}")
            try:
                complete_scores([bl], datasets, model_short)
            except ValueError as exc:
                evaluation_failed("baseline", exc)
            print(f"    {_green('OK')} ({_elapsed(elapsed)})")

        run_benchmark(["--frontier", "--model", model_short])

        if _interrupted:
            return
        try:
            baseline_scores = complete_scores(baselines, datasets, model_short)
        except ValueError as exc:
            evaluation_failed("baseline", exc)
        for bl, avg in baseline_scores.items():
            print(f"    {_bold(bl)}: avg_val={_pct(avg)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        # Show frontier status
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        retry_file = LOGS_DIR / "incomplete_evaluation.json"
        if retry_file.exists():
            retry_state = json.loads(retry_file.read_text())
            if retry_state["iteration"] != iteration:
                evaluation_failed("resume", "Pending evaluation iteration does not match summary")
            candidates = retry_state["candidates"]
            valid_candidates = validate_candidates(candidates)
            if len(valid_candidates) != len(candidates):
                evaluation_failed("resume", "Pending candidate import failed")
            propose_time = retry_state["propose_time"]
            print(f"  Resuming evaluation of {len(candidates)} existing candidates", flush=True)
        else:
            task_prompt = render_task_prompt(iteration, len(datasets))

            if PENDING_EVAL.exists():
                PENDING_EVAL.unlink()

            # Propose
            propose_start = time.time()
            print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
            ok = propose(task_prompt, iteration, timeout=args.propose_timeout)
            propose_time = time.time() - propose_start

            if not ok:
                print(
                    f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
                )
                record_failed_iteration(iteration, "no_proposal", propose_time=propose_time)
                continue

            # a broken manifest fails that iteration, not the whole run.
            try:
                candidates = load_pending_candidates(PENDING_EVAL)
            except (json.JSONDecodeError, ValueError, OSError) as e:
                print(f"  {_red('FAIL')} cannot read pending_eval.json: {str(e)[:200]}")
                try:   # keep the original so the breakage can be inspected later
                    bad = LOGS_DIR / f"pending_eval_iter{iteration:03d}.bad.json"
                    bad.write_text(PENDING_EVAL.read_text())
                    print(f"  {_dim(f'original kept: {bad}')}")
                except OSError:
                    pass
                record_failed_iteration(
                    iteration, "bad_pending_eval", str(e), propose_time=propose_time
                )
                continue
            print(
                f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
            )
            for ci, c in enumerate(candidates):
                hyp = c.get("hypothesis", "")
                print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

            # Validate
            print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
            valid_candidates = validate_candidates(candidates)

            if not valid_candidates:
                print(
                    f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
                )
                update_evolution_summary(
                    iteration, candidates, {}, propose_time=propose_time
                )
                continue
            print(
                f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
            )

        # Persist before launching any evaluator so a restart cannot lose this proposal.
        retry_file.write_text(json.dumps({"iteration": iteration, "candidates": valid_candidates,
                                          "propose_time": propose_time}, indent=2) + "\n")
        # Benchmark
        bench_start = time.time()
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(datasets)} datasets"
        )
        # Candidates of an iteration are evaluated concurrently; prompts, temperature, seeds and batch size are
        # unchanged. run_cmd wraps subprocess.run, so calling it from threads is safe.
        def _bench_one(item):
            ci, c = item
            name = c["name"]
            t0 = time.time()
            result = run_benchmark(["--memory", name])
            return ci, name, result, time.time() - t0

        pending = [(ci, c) for ci, c in enumerate(valid_candidates) if not _interrupted]
        for ci, c in pending:
            print(
                f"    [{ci + 1}/{len(valid_candidates)}] {_bold(c['name'])} start", flush=True
            )
        failed_benchmarks = []
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as _exe:
                for ci, name, result, elapsed in sorted(_exe.map(_bench_one, pending)):
                    tag = (
                        f"{_red('FAIL')} benchmark crashed"
                        if result.returncode != 0
                        else _green("OK")
                    )
                    print(f"      [{ci + 1}] {_bold(name)} {tag} ({_elapsed(elapsed)})")
                    if result.returncode != 0:
                        failed_benchmarks.append({"system": name, "code": result.returncode,
                                                  "stderr": result.stderr[-500:]})
        bench_time = time.time() - bench_start

        # Compute scores only after every dataset and every candidate succeeded.
        if _interrupted:
            return
        if failed_benchmarks:
            evaluation_failed(f"iteration {iteration}", failed_benchmarks)
        try:
            val_scores = complete_scores([c["name"] for c in valid_candidates], datasets, model_short)
        except ValueError as exc:
            evaluation_failed(f"iteration {iteration}", exc)
        for c in valid_candidates:
            name = c["name"]
            delta = val_scores[name] - (best_val * 100 if best_val <= 1 else best_val)
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}"
            )

        run_benchmark(["--frontier", "--model", model_short])
        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
        )

        retry_file.unlink()
        (LOGS_DIR / "evaluation_failure.json").unlink(missing_ok=True)
        # Show iteration summary
        improved = any(
            v > (best_val * 100 if best_val <= 1 else best_val)
            for v in val_scores.values()
        )
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Validation evolution complete.')}")
    print(
        f"Finalize once with: uv run python meta_harness.py --run-name {run_name} --test"
    )


def main():
    parser = argparse.ArgumentParser(description="Evolution loop for memory systems")
    parser.add_argument("--iterations", type=int, default=20)
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _default_model = _cfg["models"][0]["model"] if _cfg.get("models") else None
    parser.add_argument(
        "--model",
        default=_default_model,
        help=f"Solver model (default: {_default_model})",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout per propose step (default: 2400s)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs. Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed systems and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Finalize an existing named run with one held-out test evaluation",
    )
    args = parser.parse_args()

    if args.test and not args.run_name:
        parser.error("--test requires --run-name")
    if args.test and args.fresh:
        parser.error("--test cannot be combined with --fresh")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
