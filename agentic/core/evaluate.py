"""Evaluate one harness on one split; the inner loop of the Meta-Harness search.

CLI (invoked by meta_harness.py in a subprocess so candidate crashes/hangs are contained):

    python -m core.evaluate --config <resolved_config.yaml> --agent agents/<name>.py \
        --split val --out <dir> [--state <state.json>] [--prepare] [--limit N] [--items id1,id2]

Outputs in --out:
    items.jsonl     one line per item: id, correct, metrics, answer, error, usage, elapsed
    items/<id>.json full trace (prompts, responses, tool calls, outcome) for the proposer
    summary.json    accuracy and cost aggregates; `complete` is False if any item is missing

Worker modes (domain.worker_mode):
    thread   items run in a ThreadPool inside this process (HotpotQA)
    process  each item runs in its own child interpreter with a hard timeout (AppWorld)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .credentials import recordable_config
from .domain import Domain, get_domain
from .harness import BaseHarness, PrepareContext, RunContext, load_harness_class
from .clock import wall
from .llm import Budget, BudgetExceeded, ItemLLM, LLMClient, LLMConfig, Trace


# --------------------------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = recordable_config(yaml.safe_load(f))
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def make_client(cfg: dict[str, Any], cache_salt: str = "") -> LLMClient:
    llm_cfg = LLMConfig.from_dict(cfg["solver"])
    cache_dir = (cfg.get("paths") or {}).get("llm_cache")
    if cfg.get("solver", {}).get("cache", True) is False:
        cache_dir = None
    salt = cache_salt or str(cfg.get("solver", {}).get("cache_salt") or "")  # optional salt that separates cache entries of repeated runs
    return LLMClient(llm_cfg, cache_dir=cache_dir, cache_salt=salt)


def _budget(cfg: dict[str, Any], key: str) -> Budget:
    return Budget.from_dict((cfg.get("budgets") or {}).get(key))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------------------------------
# one item
# --------------------------------------------------------------------------------------------
def run_one_item(
    domain: Domain,
    harness: BaseHarness,
    item: dict[str, Any],
    client: LLMClient,
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    item_id = str(item["id"])
    trace = Trace(item_id)
    llm = ItemLLM(client, _budget(cfg, "per_item"), trace)
    t0 = wall()
    outcome: dict[str, Any] | None = None
    error: str | None = None
    correct = False
    metrics: dict[str, Any] = {}
    try:
        with domain.open_tools({**item, "_trace": trace}, cfg) as tools:
            ctx = RunContext(item_id, llm, tools, trace, cfg.get("domain_config") or {})
            try:
                outcome = harness.run(domain.public_item(item), ctx)
                if outcome is not None and not isinstance(outcome, dict):
                    outcome = {domain.answer_key: str(outcome)}
            except BudgetExceeded as e:
                error = f"BudgetExceeded: {e}"
            except Exception:  # noqa: BLE001 - candidate code may raise anything
                error = "HarnessError: " + traceback.format_exc(limit=8)[-4000:]
            try:
                correct, metrics = domain.score(item, outcome, tools)
            except Exception:  # noqa: BLE001
                correct, metrics = False, {}
                error = (error or "") + "\nScoreError: " + traceback.format_exc(limit=5)[-2000:]
    except Exception:  # noqa: BLE001 - tools failed to open (env crash etc.)
        error = (error or "") + "\nToolsError: " + traceback.format_exc(limit=5)[-2000:]
    elapsed = wall() - t0
    result = {
        "id": item_id,
        "correct": bool(correct),
        "metrics": metrics,
        "answer": (outcome or {}).get(domain.answer_key) if isinstance(outcome, dict) else None,
        "error": error,
        "usage": llm.usage(),
        "elapsed": round(elapsed, 2),
    }
    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        **trace.to_dict(),
        "item": domain.public_item(item),
        "outcome": outcome,
        "result": result,
    }
    _atomic_write(items_dir / f"{_safe(item_id)}.json", json.dumps(trace_payload, ensure_ascii=False, default=str))
    return result


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# --------------------------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------------------------
def run_prepare(domain: Domain, harness: BaseHarness, cfg: dict[str, Any], client: LLMClient, out_dir: Path) -> str:
    train_items = domain.load_split("train", cfg)
    trace = Trace("__prepare__")
    llm = ItemLLM(client, _budget(cfg, "prepare"), trace)

    def open_tools(item: dict[str, Any]):
        return domain.open_tools({**item, "_trace": trace}, cfg)

    ctx = PrepareContext(llm, trace, cfg.get("domain_config") or {}, open_tools)
    t0 = time.time()
    harness.prepare(train_items, ctx)
    state = harness.get_state()
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / "prepare_trace.json", json.dumps({**trace.to_dict(), "usage": llm.usage(), "elapsed": round(time.time() - t0, 1)}, ensure_ascii=False, default=str))
    return state


# --------------------------------------------------------------------------------------------
# split driver
# --------------------------------------------------------------------------------------------
def _load_done(out_dir: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    p = out_dir / "items.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[str(r["id"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def summarize(domain: Domain, agent_name: str, split: str, results: list[dict[str, Any]], n_items: int, cfg: dict[str, Any], elapsed: float) -> dict[str, Any]:
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_err = sum(1 for r in results if r.get("error"))
    n_budget = sum(1 for r in results if (r.get("error") or "").startswith("BudgetExceeded"))
    def avg(key: str) -> float:
        vals = [r["usage"].get(key, 0) for r in results if r.get("usage")]
        return round(sum(vals) / len(vals), 1) if vals else 0.0
    summary = {
        "agent": agent_name,
        "split": split,
        "domain": domain.name,
        "n_items": n_items,
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_items, 4) if n_items else 0.0,  # missing items count as wrong
        "complete": n == n_items,
        "n_errors": n_err,
        "n_budget_exceeded": n_budget,
        "avg_calls": avg("calls"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_completion_tokens": avg("completion_tokens"),
        "avg_elapsed": round(sum(r.get("elapsed", 0) for r in results) / n, 1) if n else 0.0,
        "model": cfg["solver"].get("model"),
        "temperature": cfg["solver"].get("temperature"),
        "wall_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary.update(domain.summarize(results))
    return summary


def evaluate_split(
    cfg: dict[str, Any],
    agent_path: Path,
    split: str,
    out_dir: Path,
    state_path: Path | None = None,
    limit: int | None = None,
    item_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    domain = get_domain(cfg["domain"])
    items = domain.load_split(split, cfg)
    if item_ids:
        wanted = set(item_ids)
        items = [it for it in items if str(it["id"]) in wanted]
    max_items = (cfg.get("eval") or {}).get("max_items")  # sandbox cap for proposer prototyping runs
    if max_items and not item_ids:
        limit = min(limit or int(max_items), int(max_items))
    if limit:
        items = items[:limit]
    cfg["_agent"] = agent_path.stem
    cfg["_split"] = split
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {} if force else _load_done(out_dir)
    todo = [it for it in items if str(it["id"]) not in done]
    agent_name = agent_path.stem
    t0 = time.time()
    results: list[dict[str, Any]] = [done[str(it["id"])] for it in items if str(it["id"]) in done]
    lock = threading.Lock()
    items_file = out_dir / "items.jsonl"
    if force and items_file.exists():
        items_file.unlink()

    def record(r: dict[str, Any]) -> None:
        with lock:
            with open(items_file, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            results.append(r)
            n_done = len(results)
            if n_done % 10 == 0 or n_done == len(items):
                acc = sum(1 for x in results if x["correct"]) / n_done
                print(f"  [{agent_name}/{split}] {n_done}/{len(items)} acc={acc:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

    concurrency = int((cfg.get("eval") or {}).get("concurrency", 8))
    if todo:
        if domain.worker_mode == "process":
            _run_process_mode(cfg, agent_path, split, out_dir, state_path, todo, concurrency, record)
        else:
            _run_thread_mode(cfg, domain, agent_path, state_path, todo, concurrency, out_dir, record)

    summary = summarize(domain, agent_name, split, results, len(items), cfg, time.time() - t0)
    _atomic_write(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _instantiate(agent_path: Path, state_path: Path | None) -> BaseHarness:
    cls = load_harness_class(agent_path)
    harness = cls()
    if state_path and Path(state_path).exists():
        harness.set_state(Path(state_path).read_text())
    return harness


def _run_thread_mode(cfg, domain, agent_path, state_path, todo, concurrency, out_dir, record) -> None:
    client = make_client(cfg)
    harness = _instantiate(agent_path, state_path)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_one_item, domain, harness, it, client, cfg, out_dir) for it in todo]
        for fut in as_completed(futures):
            record(fut.result())


def _child_limits(cfg):
    """preexec_fn for process-mode children: nice +10 (CPU-heavy AppWorld children must not freeze the host) and a heap
    cap (eval.child_mem_gb, default 8) so a runaway harness dies with MemoryError instead of the kernel OOM-killing it."""
    mem_gb = float((cfg.get("eval") or {}).get("child_mem_gb", 8))

    def _apply():
        os.nice(10)
        if mem_gb > 0:
            import resource
            lim = int(mem_gb * (1 << 30))
            resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))
    return _apply


def _run_process_mode(cfg, agent_path, split, out_dir, state_path, todo, concurrency, record) -> None:
    """One child interpreter per item, hard-killed at item_timeout_s.

    Children re-read the config from disk, so the EFFECTIVE config (including in-memory overrides such as
    sampling temperature, cache salt or per-rep tags) is written next to the outputs and passed to them.
    """
    timeout = float((cfg.get("eval") or {}).get("item_timeout_s", 1800))
    py = sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(Path(cfg["paths"]["checkout"]).resolve()), env.get("PYTHONPATH", "")] if p)
    child_dir = out_dir / "_children"
    child_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg_path = out_dir / "effective_config.yaml"
    effective_cfg = recordable_config({k: v for k, v in cfg.items() if k != "_config_path"})
    _atomic_write(effective_cfg_path, yaml.safe_dump(effective_cfg, sort_keys=False))

    def one(item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item["id"])
        result_file = child_dir / f"{_safe(item_id)}.result.json"
        log_file = child_dir / f"{_safe(item_id)}.log"
        if result_file.exists():
            result_file.unlink()
        cmd = [py, "-m", "core.evaluate", "--config", str(effective_cfg_path), "--agent", str(agent_path), "--split", split,
               "--out", str(out_dir), "--items", item_id, "--child-result", str(result_file)]
        if state_path:
            cmd += ["--state", str(state_path)]
        t0 = time.time()
        with open(log_file, "w") as lf:
            try:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=cfg["paths"]["checkout"],
                                      preexec_fn=_child_limits(cfg))
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        if result_file.exists():
            try:
                return json.loads(result_file.read_text())
            except json.JSONDecodeError:
                pass
        tail = ""
        try:
            tail = log_file.read_text()[-2000:]
        except OSError:
            pass
        return {"id": item_id, "correct": False, "metrics": {}, "answer": None,
                "error": f"ChildFailed(exit={code}{', timeout' if code == 124 else ''}): {tail}",
                "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}, "elapsed": round(time.time() - t0, 2)}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, it) for it in todo]
        for fut in as_completed(futures):
            record(fut.result())


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a harness on a split")
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True, help="path to agents/<name>.py")
    ap.add_argument("--split", default="val", choices=["train", "val", "selection", "test"])
    ap.add_argument("--out", required=True, help="output directory for this (agent, split)")
    ap.add_argument("--state", default=None, help="harness state json (written by --prepare, read otherwise)")
    ap.add_argument("--prepare", action="store_true", help="run harness.prepare on the train split and write --state")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--items", default=None, help="comma-separated item ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--child-result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    agent_path = Path(args.agent).resolve()
    out_dir = Path(args.out).resolve()
    state_path = Path(args.state).resolve() if args.state else None

    if args.prepare:
        domain = get_domain(cfg["domain"])
        cfg["_agent"] = agent_path.stem
        cfg["_split"] = "train"
        harness = _instantiate(agent_path, None)
        state = run_prepare(domain, harness, cfg, make_client(cfg), out_dir)
        if state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(state_path, state)
            print(f"state written to {state_path} ({len(state)} chars)")
        return 0

    if args.child_result:  # single-item child process
        domain = get_domain(cfg["domain"])
        items = [it for it in domain.load_split(args.split, cfg) if str(it["id"]) == args.items]
        if not items:
            print(f"item {args.items} not found in split {args.split}", file=sys.stderr)
            return 2
        cfg["_agent"] = agent_path.stem
        cfg["_split"] = args.split
        harness = _instantiate(agent_path, state_path)
        result = run_one_item(domain, harness, items[0], make_client(cfg), cfg, out_dir)
        _atomic_write(Path(args.child_result), json.dumps(result, ensure_ascii=False, default=str))
        return 0

    item_ids = [s for s in args.items.split(",") if s] if args.items else None
    summary = evaluate_split(cfg, agent_path, args.split, out_dir, state_path, args.limit, item_ids, args.force)
    print(json.dumps({k: summary[k] for k in ("agent", "split", "n", "n_items", "accuracy", "complete", "n_errors", "avg_calls", "avg_prompt_tokens")}))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
