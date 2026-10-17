"""Meta-Harness outer loop, domain-agnostic.

    python -m core.meta_harness --run-dir runs/<run_id> [--iterations N] [--fresh] [--skip-baseline] [--skip-smoke]
    python -m core.meta_harness --run-dir runs/<run_id> --test        # one-time held-out evaluation

Per iteration:
    1. proposer (Claude Code, inside an unprivileged container that sees only train/val data) writes
       new `agents/<name>.py` files and `proposals/pending_eval.json`;
    2. each candidate is audited (static anti-leakage rules), import-checked and smoke-tested;
    3. survivors are prepared on train and evaluated on the search set (val) in a subprocess;
    4. frontier + evolution_summary.jsonl are updated from COMPLETE results only.
Test evaluation happens only via --test, on the host, with the test split that never enters the sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import claude_wrapper
from .credentials import recordable_config, solver_key_env
from .audit import audit_file
from .domain import get_domain
from .results import compute_frontier, format_table, load_summaries

PROPOSER_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,48}$")
_interrupted = False


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _handle_signal(signum, frame):  # noqa: ARG001
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


class Run:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.cfg_path = self.run_dir / "resolved_config.yaml"
        self.cfg: dict[str, Any] = recordable_config(yaml.safe_load(self.cfg_path.read_text()))
        p = self.cfg["paths"]
        self.checkout = Path(p["checkout"])
        self.logs = Path(p["logs"])
        self.results = Path(p["results"])
        self.agents = self.checkout / "agents"
        self.proposals = self.checkout / "proposals"
        self.snapshot = self.checkout / ".baseline_snapshot"
        self.summary_path = self.logs / "evolution_summary.jsonl"
        self.frontier_path = self.logs / "frontier_val.json"
        self.pending_path = self.proposals / "pending_eval.json"
        self.finalized_path = self.results / "finalized.json"
        self.domain = get_domain(self.cfg["domain"])
        for d in (self.logs, self.results, self.proposals / "reports", self.proposals / "proto", self.logs / "claude_sessions"):
            d.mkdir(parents=True, exist_ok=True)
        self.baselines: list[str] = list(self.cfg.get("baselines") or [])
        self.py = sys.executable

    # ------------------------------------------------------------------ subprocess helpers
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.checkout)
        env.pop("AGENTIC_APPWORLD_ROOT", None)
        return env

    def _eval(self, agent: str, split: str, out_dir: Path, extra: list[str] | None = None, state: Path | None = None, timeout: float | None = None) -> dict[str, Any] | None:
        cmd = [self.py, "-m", "core.evaluate", "--config", str(self.cfg_path), "--agent", str(self.agents / f"{agent}.py"), "--split", split, "--out", str(out_dir)]
        if state is not None:
            cmd += ["--state", str(state)]
        cmd += extra or []
        timeout = timeout or float((self.cfg.get("eval") or {}).get("split_timeout_s", 4 * 3600))
        log = out_dir / "eval.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as lf:
            lf.write(f"\n### {datetime.now().isoformat()} {' '.join(cmd)}\n")
            lf.flush()
            try:
                subprocess.run(cmd, cwd=str(self.checkout), env=self._env(), stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                lf.write(f"\nTIMEOUT after {timeout}s\n")
        summary_file = out_dir / "summary.json"
        if summary_file.exists():
            try:
                return json.loads(summary_file.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def state_path(self, agent: str) -> Path:
        return self.logs / "state" / f"{agent}.json"

    def prepare(self, agent: str) -> bool:
        sp = self.state_path(agent)
        if sp.exists():
            return True
        out = self.logs / "prepare" / agent
        cmd = [self.py, "-m", "core.evaluate", "--config", str(self.cfg_path), "--agent", str(self.agents / f"{agent}.py"), "--prepare", "--state", str(sp), "--out", str(out)]
        out.mkdir(parents=True, exist_ok=True)
        timeout = float((self.cfg.get("budgets", {}).get("prepare") or {}).get("max_seconds", 7200)) + 600
        with open(out / "prepare.log", "a") as lf:
            try:
                r = subprocess.run(cmd, cwd=str(self.checkout), env=self._env(), stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                ok = r.returncode == 0 and sp.exists()
            except subprocess.TimeoutExpired:
                lf.write("\nTIMEOUT\n")
                ok = False
        if not ok:
            print(f"  {_ts()} prepare FAILED for {agent} (see {out / 'prepare.log'})")
        return ok

    def evaluate_val(self, agent: str) -> dict[str, Any] | None:
        if not self.prepare(agent):
            return None
        return self._eval(agent, "val", self.logs / "val" / agent, state=self.state_path(agent))

    # ------------------------------------------------------------------ frontier / summary
    def refresh_frontier(self) -> dict[str, Any]:
        summaries = load_summaries(self.logs / "val")
        frontier = compute_frontier(summaries)
        frontier["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.frontier_path.write_text(json.dumps(frontier, indent=2))
        extra = ("avg_f1", "avg_gold_title_recall") if self.cfg["domain"] == "hotpotqa" else ("sgc", "avg_steps")
        (self.logs / "val_table.txt").write_text(format_table(summaries, extra_keys=extra) + "\n")
        return frontier

    def best_val(self) -> tuple[str | None, float]:
        if not self.frontier_path.exists():
            return None, 0.0
        b = json.loads(self.frontier_path.read_text()).get("_best")
        return (b["agent"], float(b["accuracy"])) if b else (None, 0.0)

    def count_iterations(self) -> int:
        n = 0
        if self.summary_path.exists():
            for line in self.summary_path.read_text().splitlines():
                try:
                    n = max(n, int(json.loads(line).get("iteration", 0)))
                except (json.JSONDecodeError, ValueError):
                    pass
        return n

    def append_summary(self, rows: list[dict[str, Any]]) -> None:
        with open(self.summary_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ baseline integrity
    def snapshot_baselines(self) -> None:
        self.snapshot.mkdir(exist_ok=True)
        for name in self.baselines:
            src = self.agents / f"{name}.py"
            if src.exists() and not (self.snapshot / f"{name}.py").exists():
                shutil.copy2(src, self.snapshot / f"{name}.py")

    def restore_protected_files(self, protected: dict[str, str]) -> list[str]:
        """Restore any pre-existing agent file the proposer modified. Returns names restored."""
        restored = []
        for name, digest in protected.items():
            f = self.agents / f"{name}.py"
            if not f.exists() or _sha(f) != digest:
                snap = self.snapshot / f"{name}.py"
                if snap.exists():
                    shutil.copy2(snap, f)
                restored.append(name)
        return restored

    def sweep_orphans(self, iteration: int, registered: set[str], protected: dict[str, str]) -> list[str]:
        """Move agent files the proposer created but never registered in pending_eval.json out of agents/."""
        moved = []
        dest = self.logs / "orphans" / f"iter{iteration:03d}"
        for f in sorted(self.agents.glob("*.py")):
            if f.name.startswith("_") or f.stem in protected or f.stem in registered:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / f.name))
            moved.append(f.stem)
        if moved:
            print(f"  {_ts()} moved {len(moved)} unregistered candidate file(s) to {dest}: {moved}")
        return moved

    def park_rejected(self, iteration: int, name: str, protected: dict[str, str]) -> None:
        """Move a rejected candidate file out of agents/ (kept under logs/rejected/ for inspection)."""
        f = self.agents / f"{name}.py"
        if name and f.exists() and name not in protected:
            dest = self.logs / "rejected" / f"iter{iteration:03d}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / f.name))

    def protect_all_agents(self) -> dict[str, str]:
        """Snapshot every existing agent file (baselines + past candidates) before the proposer runs."""
        self.snapshot.mkdir(exist_ok=True)
        digests = {}
        for f in sorted(self.agents.glob("*.py")):
            if f.name.startswith("_"):
                continue
            shutil.copy2(f, self.snapshot / f.name)
            digests[f.stem] = _sha(f)
        return digests

    # ------------------------------------------------------------------ proposer
    def task_prompt(self, iteration: int) -> str:
        pc = self.cfg["proposer"]
        sc = self.cfg["solver"]
        b = self.cfg.get("budgets", {}).get("per_item", {})
        best_agent, best_acc = self.best_val()
        n = int(pc.get("candidates_per_iteration", 2))
        lines = [
            f"Run iteration {iteration} of the harness evolution loop.",
            f"Domain: {self.cfg['domain']} — {self.cfg.get('description', '')}",
            f"Current frontier on the search set: {best_agent} @ {best_acc * 100:.1f}%.",
            f"Propose exactly {n} new candidate harness file(s) in /workspace/agents/ and register them in /workspace/proposals/pending_eval.json.",
            "",
            "## Where things are (paths inside this sandbox)",
            "- /workspace/agents/                        baseline(s) and every past candidate. Write ONLY new files here; never edit existing ones.",
            "- /workspace/logs/evolution_summary.jsonl   one JSON row per evaluated candidate (scores, deltas, hypotheses)",
            "- /workspace/logs/frontier_val.json         current best / Pareto set on the search set",
            "- /workspace/logs/val_table.txt             all search-set results as a table",
            "- /workspace/logs/val/<agent>/summary.json  aggregate metrics;  items.jsonl = per-item results;  items/<id>.json = FULL traces (prompts, responses, tool calls)",
            "- /workspace/logs/smoke/<agent>/            smoke-test traces of candidates that failed validation",
            "- /workspace/proposals/reports/             write post-eval reports here",
            "- /workspace/proposals/pending_eval.json    write the candidate manifest here",
            "- /workspace/core/, /workspace/domains/     framework + domain code (read-only)",
            "- /workspace/data/                          train and search-set data (read-only). The held-out test split does not exist in this sandbox.",
            "",
            "## Prototyping (real solver, capped)",
            "  cd /workspace && python -m core.evaluate --config config.container.yaml --agent agents/<name>.py --split val --out proposals/proto/<name> --limit 5",
            f"  (at most {int((self.cfg.get('proposer') or {}).get('prototype_max_items', 20))} items per call; full search-set evaluation is done by the outer loop)",
            "",
            "## Time budget",
            f"- Hard limit for this session: {int(pc.get('timeout_s', 2400)) // 60} minutes. When it expires the session is killed and NOTHING you did counts:",
            "  unregistered files are discarded. Write proposals/pending_eval.json as soon as your candidates import-check and pass one",
            "  small prototype run; you may keep refining afterwards and re-write the manifest, but it must exist well before the limit.",
            "",
            "## Fixed experiment controls (never change or work around these)",
            f"- solver model {sc['model']}, temperature {sc['temperature']}, reasoning effort {sc.get('reasoning_effort')}, max_tokens/call {sc['max_tokens']}",
            f"- per-item budget: {json.dumps(b)}",
            f"- domain config: {json.dumps(self.cfg.get('domain_config', {}), default=str)}",
        ]
        return "\n".join(lines)

    def propose(self, iteration: int) -> bool:
        pc = self.cfg["proposer"]
        if self.pending_path.exists():
            self.pending_path.unlink()
        os.environ.pop("CLAUDECODE", None)
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)  # force subscription auth
        token_env = pc.get("oauth_env", "CLAUDE_CODE_OAUTH_TOKEN")
        if not os.environ.get(token_env):
            raise SystemExit(f"{token_env} is not set (see .env.example); the proposer never runs without it")
        if token_env != "CLAUDE_CODE_OAUTH_TOKEN":
            os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ[token_env]
        p = self.cfg["paths"]
        readonly = {str(self.checkout / "core"): "/workspace/core", str(self.checkout / "domains"): "/workspace/domains",
                    str(self.logs): "/workspace/logs", str(self.checkout / "data"): "/workspace/data"}
        extra_mounts = {}
        if self.cfg["domain"] == "appworld":
            dev_root = self.checkout / "appworld_dev"
            readonly[str(dev_root / "data")] = "/workspace/appworld_dev/data"
        extra_env = {"PYTHONPATH": "/workspace", "AGENTIC_CONFIG": "/workspace/config.container.yaml", "UV_PROJECT_ENVIRONMENT": "/home/proposer/uv-venv",
                     "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}  # no cross-iteration notes outside the filesystem history
        # also clear any project state the CLI persisted in the per-run HOME
        for sub in ("projects",):
            d = Path(p["proposer_home"]) / ".claude" / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        if self.cfg["domain"] == "appworld":
            extra_env["AGENTIC_APPWORLD_ROOT"] = "/workspace/appworld_dev"
            extra_env["APPWORLD_ROOT"] = "/workspace/appworld_dev"
        container = {
            "image": pc["image"],
            "workspace_root": str(self.checkout),
            "home_dir": p["proposer_home"],
            "readonly_mounts": readonly,
            "extra_mounts": extra_mounts,
            "oauth_token": True,
            "cwd": "/workspace",
            "name": f"mh-proposer-{self.cfg['run_id']}-iter{iteration}",
            "cpus": pc.get("cpus", 8),
            "memory": pc.get("memory", "16g"),
            "extra_env": extra_env,
            "pass_env": [solver_key_env(self.cfg["solver"])] if pc.get("allow_solver_access", True) else [],
            "host_gateway": bool(pc.get("allow_solver_access", True)),
            "network": pc.get("network"),
        }
        result = claude_wrapper.run(
            prompt=self.task_prompt(iteration),
            model=pc.get("model", "claude-opus-5"),
            allowed_tools=PROPOSER_ALLOWED_TOOLS,
            skills=[str(self.checkout / ".claude" / "skills" / "meta-harness")],
            cwd="/workspace",
            container=container,
            log_dir=str(self.logs / "claude_sessions"),
            name=f"iter{iteration}",
            timeout_seconds=int(pc.get("timeout_s", 2400)),
            effort=pc.get("effort", "max"),
        )
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
        if result.exit_code != 0:
            print(f"  {_ts()} proposer exit={result.exit_code} {(result.stderr or '')[:400]}")
        try:
            result.show()
        except Exception:  # noqa: BLE001
            pass
        return self.pending_path.exists()

    # ------------------------------------------------------------------ validation
    def validate(self, cand: dict[str, Any], protected: dict[str, str], skip_smoke: bool) -> tuple[bool, dict[str, Any]]:
        info: dict[str, Any] = {"audit_warnings": []}
        name = str(cand.get("name", ""))
        if not NAME_RE.match(name):
            return False, {**info, "status": "rejected", "reason": f"bad name {name!r}"}
        if name in protected:
            return False, {**info, "status": "rejected", "reason": "name collides with an existing agent"}
        f = self.agents / f"{name}.py"
        if not f.exists():
            return False, {**info, "status": "rejected", "reason": "file missing"}
        audit = audit_file(f, self.domain.audit_literals(self.cfg), (self.cfg.get("audit") or {}).get("warn_patterns"))
        info["audit_warnings"] = audit["warnings"]
        if not audit["ok"]:
            return False, {**info, "status": "rejected_audit", "reason": "; ".join(audit["errors"])[:1500]}
        r = subprocess.run([self.py, "-c", f"from core.harness import load_harness_class; load_harness_class('agents/{name}.py'); print('OK')"], cwd=str(self.checkout), env=self._env(), capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or "OK" not in r.stdout:
            return False, {**info, "status": "import_fail", "reason": (r.stderr or r.stdout)[-1500:]}
        if not self.prepare(name):  # runs harness.prepare on the train split once; state reused by smoke + val + test
            return False, {**info, "status": "prepare_fail", "reason": f"prepare() failed, see logs/prepare/{name}/prepare.log"}
        if skip_smoke:
            return True, {**info, "status": "validated"}
        n_smoke = int((self.cfg.get("search") or {}).get("smoke_items", 2))
        ids = [str(it["id"]) for it in self.domain.smoke_items(self.cfg, n_smoke)]
        out = self.logs / "smoke" / name
        if out.exists():
            shutil.rmtree(out)
        summary = self._eval(name, "val", out, extra=["--items", ",".join(ids), "--force"], state=self.state_path(name), timeout=float((self.cfg.get("eval") or {}).get("item_timeout_s", 1800)) * 2 + 120)
        if not summary or not summary.get("complete"):
            return False, {**info, "status": "smoke_fail", "reason": "smoke evaluation did not complete"}
        errors = []
        for line in (out / "items.jsonl").read_text().splitlines():
            r_ = json.loads(line)
            if r_.get("error") and not r_["error"].startswith("BudgetExceeded"):
                errors.append(r_["error"][:300])
        if errors:
            return False, {**info, "status": "smoke_fail", "reason": errors[0]}
        return True, {**info, "status": "validated"}

    # ------------------------------------------------------------------ phases
    def phase_baselines(self) -> None:
        print(f"\n{_ts()} Phase 0: baselines {self.baselines}")
        for name in self.baselines:
            if _interrupted:
                return
            existing = (self.logs / "val" / name / "summary.json")
            if existing.exists() and json.loads(existing.read_text()).get("complete"):
                print(f"  {name}: cached {json.loads(existing.read_text())['accuracy']:.3f}")
                continue
            t0 = time.time()
            s = self.evaluate_val(name)
            print(f"  {name}: acc={s['accuracy']:.3f} complete={s['complete']} ({time.time() - t0:.0f}s)" if s else f"  {name}: FAILED")
        self.refresh_frontier()

    def iteration(self, it: int, skip_smoke: bool) -> None:
        best_agent, best_acc = self.best_val()
        print(f"\n{_ts()} Iteration {it}  frontier={best_agent} @ {best_acc * 100:.1f}%\n" + "─" * 70)
        protected = self.protect_all_agents()
        t0 = time.time()
        ok = self.propose(it)
        propose_s = time.time() - t0
        restored = self.restore_protected_files(protected)
        if restored:
            print(f"  {_ts()} WARNING proposer modified protected agent files, restored: {restored}")
        if not ok:
            print(f"  {_ts()} proposer produced no pending_eval.json ({propose_s:.0f}s)")
            orphans = self.sweep_orphans(it, set(), protected)
            self.append_summary([{"iteration": it, "agent": None, "status": "no_proposal", "orphans": orphans, "timing_s": {"propose": round(propose_s)}}])
            return
        try:
            cands = load_manifest(self.pending_path)
        except (json.JSONDecodeError, ValueError, OSError) as e:  # malformed manifest = a failed iteration, never a crash of the run
            print(f"  {_ts()} bad pending_eval.json: {str(e)[:200]}")
            orphans = self.sweep_orphans(it, set(), protected)
            self.append_summary([{"iteration": it, "agent": None, "status": "bad_pending_eval", "reason": str(e)[:500], "orphans": orphans}])
            return
        orphans = self.sweep_orphans(it, {str(c.get("name", "")) for c in cands}, protected)
        # archive the manifest next to the logs
        shutil.copy2(self.pending_path, self.logs / f"pending_eval_iter{it:03d}.json")
        rows = []
        valid = []
        for c in cands:
            ok_c, info = self.validate(c, protected, skip_smoke)
            print(f"  {_ts()} {c.get('name')}: {info['status']}" + (f" — {info.get('reason', '')[:200]}" if not ok_c else ""))
            if ok_c:
                valid.append(c)
            else:
                rows.append({"iteration": it, "agent": c.get("name"), "hypothesis": c.get("hypothesis", ""), "axis": c.get("axis"), "components": c.get("components"), **info, "accuracy": None})
                self.park_rejected(it, str(c.get("name", "")), protected)
        for c in valid:  # always finish the iteration (SIGTERM only stops at the iteration boundary)
            name = c["name"]
            t1 = time.time()
            s = self.evaluate_val(name)
            bench_s = time.time() - t1
            if s and s.get("complete"):
                delta = s["accuracy"] - best_acc
                print(f"  {_ts()} {name}: acc={s['accuracy'] * 100:.1f}% delta={delta * 100:+.1f} calls={s['avg_calls']} in_tok={s['avg_prompt_tokens']:.0f} ({bench_s:.0f}s)")
                rows.append({"iteration": it, "agent": name, "status": "evaluated", "accuracy": s["accuracy"], "delta": round(delta, 4), "n_errors": s["n_errors"], "n_budget_exceeded": s["n_budget_exceeded"], "avg_calls": s["avg_calls"], "avg_prompt_tokens": s["avg_prompt_tokens"], "avg_completion_tokens": s["avg_completion_tokens"], "domain_metrics": {k: s[k] for k in s if k in ("avg_f1", "avg_gold_title_recall", "sgc", "avg_steps")}, "hypothesis": c.get("hypothesis", ""), "axis": c.get("axis"), "components": c.get("components"), "timing_s": {"propose": round(propose_s), "bench": round(bench_s)}})
            else:
                print(f"  {_ts()} {name}: evaluation INCOMPLETE/crashed ({bench_s:.0f}s)")
                rows.append({"iteration": it, "agent": name, "status": "eval_incomplete", "accuracy": None, "hypothesis": c.get("hypothesis", ""), "axis": c.get("axis"), "timing_s": {"propose": round(propose_s), "bench": round(bench_s)}})
        if orphans:
            rows.append({"iteration": it, "agent": None, "status": "orphans_swept", "orphans": orphans})
        self.append_summary(rows)
        fr = self.refresh_frontier()
        nb = fr.get("_best") or {}
        print(f"  {_ts()} frontier now {nb.get('agent')} @ {float(nb.get('accuracy', 0)) * 100:.1f}%")

    def finalize(self) -> None:
        """Held-out evaluation. With a `selection` split: frontier candidates are first scored on selection (never
        seen by the proposer) and the best one becomes the Optimized harness; baselines, the selected harness and
        (for analysis) the rest of the Pareto set are then evaluated on test."""
        if self.finalized_path.exists() and json.loads(self.finalized_path.read_text()).get("status") == "complete":
            print("already finalized")
            return
        frontier = json.loads(self.frontier_path.read_text()) if self.frontier_path.exists() else {}
        # selection pool = Pareto set (accuracy vs cost) + top-N by search accuracy, so that strong-but-costlier
        # candidates dominated on the Pareto front still get a chance to be picked on the hidden selection split
        top_n = int((self.cfg.get("search") or {}).get("selection_top_n", 5))
        candidates = [e["agent"] for e in frontier.get("_pareto", []) if e["agent"] not in self.baselines]
        ranked_search = sorted(((n, v) for n, v in (frontier.get("agents") or {}).items() if n not in self.baselines and v.get("accuracy") is not None),
                               key=lambda kv: (-kv[1]["accuracy"], kv[1].get("avg_prompt_tokens") or 0))
        for n, _ in ranked_search[:top_n]:
            if n not in candidates:
                candidates.append(n)
        if frontier.get("_best") and frontier["_best"]["agent"] not in candidates and frontier["_best"]["agent"] not in self.baselines:
            candidates.append(frontier["_best"]["agent"])
        extra = ("avg_f1", "avg_gold_title_recall") if self.cfg["domain"] == "hotpotqa" else ("sgc", "avg_steps")
        has_selection = self._has_split("selection")
        selected = None
        sel_summaries: dict[str, Any] = {}
        if has_selection and candidates:
            print(f"{_ts()} Selection evaluation ({len(candidates)} frontier candidates + baselines)")
            for name in list(self.baselines) + candidates:
                if not self.prepare(name):
                    continue
                s = self._eval(name, "selection", self.results / "selection" / name, state=self.state_path(name))
                if s and s.get("complete"):
                    sel_summaries[name] = s
                    print(f"  {_ts()} {name}: selection acc={s['accuracy'] * 100:.1f}%")
            ranked = sorted((n for n in candidates if n in sel_summaries), key=lambda n: (-sel_summaries[n]["accuracy"], sel_summaries[n]["avg_prompt_tokens"]))
            selected = ranked[0] if ranked else None
            (self.results / "selection_table.txt").write_text(format_table(sel_summaries, extra_keys=extra) + "\n")
            (self.results / "selection.json").write_text(json.dumps({"selected": selected, "ranking": ranked, "search_best": (frontier.get("_best") or {}).get("agent"),
                                                                    "selection_accuracy": {n: sel_summaries[n]["accuracy"] for n in sel_summaries}}, indent=2))
            print(f"  selected Optimized harness: {selected} (search-set best was {(frontier.get('_best') or {}).get('agent')})")
        # test order: baselines, then the selected harness, then the remaining pool by selection rank
        rest = [n for n in candidates if n != selected]
        if sel_summaries:
            rest.sort(key=lambda n: (-(sel_summaries.get(n) or {}).get("accuracy", -1.0), (sel_summaries.get(n) or {}).get("avg_prompt_tokens") or 0))
        targets = list(self.baselines) + ([selected] if selected else []) + rest
        self.finalized_path.write_text(json.dumps({"status": "in_progress", "agents": targets, "selected": selected, "started_at": datetime.now().isoformat()}, indent=2))
        print(f"{_ts()} Test evaluation for {targets}")
        summaries = {}

        def _write_tables() -> None:
            complete = {k: v for k, v in summaries.items() if v.get("complete")}
            (self.results / "test_table.txt").write_text(format_table(complete, extra_keys=extra) + "\n")
            (self.results / "frontier_test.json").write_text(json.dumps({**compute_frontier(complete), "selected": selected}, indent=2))

        for name in targets:
            if not self.prepare(name):
                continue
            s = self._eval(name, "test", self.results / "test" / name, state=self.state_path(name))
            if s:
                summaries[name] = s
                tag = "  <- Optimized (selected)" if name == selected else ("  (baseline)" if name in self.baselines else "")
                print(f"  {_ts()} {name}: test acc={s['accuracy'] * 100:.1f}% complete={s['complete']}{tag}")
                _write_tables()
        _write_tables()
        status = "complete" if all(s.get("complete") for s in summaries.values()) and len(summaries) == len(targets) else "incomplete"
        self.finalized_path.write_text(json.dumps({"status": status, "agents": targets, "selected": selected, "completed_at": datetime.now().isoformat()}, indent=2))
        print((self.results / "test_table.txt").read_text())
        print(f"finalize status: {status}")

    def _has_split(self, split: str) -> bool:
        try:
            return len(self.domain.load_split(split, self.cfg)) > 0
        except Exception:  # noqa: BLE001
            return False


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Parse proposals/pending_eval.json into the candidate list.

    Accepted: {"iteration": N, "candidates": [{...}, ...]} (the documented form) or a bare list of candidate objects.
    Anything else raises ValueError, which the loop records as `bad_pending_eval`."""
    manifest = json.loads(path.read_text())
    if isinstance(manifest, list):
        manifest = {"candidates": manifest}
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object, got {type(manifest).__name__}")
    cands = manifest.get("candidates", [])
    if not isinstance(cands, list) or not all(isinstance(c, dict) for c in cands):
        raise ValueError("manifest 'candidates' must be a list of objects")
    return cands


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run_cls: type[Run] = Run) -> int:
    """`run_cls` lets another proposer backend reuse this loop unchanged."""
    ap = argparse.ArgumentParser(description="Meta-Harness outer loop")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iterations", type=int, default=None, help="how many MORE iterations to run (default: up to search.iterations total)")
    ap.add_argument("--fresh", action="store_true", help="delete candidates/logs of this run and start over")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--propose-timeout", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="finalize: evaluate baselines + frontier on the held-out test split")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    load_dotenv(run_dir / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    run = run_cls(run_dir)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.propose_timeout:
        run.cfg["proposer"]["timeout_s"] = args.propose_timeout
    if args.test:
        run.finalize()
        return 0
    if run.finalized_path.exists() and json.loads(run.finalized_path.read_text()).get("status") == "complete":
        print("run is finalized; create a new run to continue evolving")
        return 1
    if args.fresh:
        run.snapshot_baselines()
        for f in run.agents.glob("*.py"):
            if f.stem not in run.baselines and not f.name.startswith("_"):
                f.unlink()
        for d in (run.logs, run.proposals):
            if d.exists():
                shutil.rmtree(d)
        run = run_cls(run_dir)
    run.snapshot_baselines()

    total = int((run.cfg.get("search") or {}).get("iterations", 20))
    done = run.count_iterations()
    n_more = args.iterations if args.iterations is not None else max(0, total - done)
    print(f"{_ts()} run={run.cfg['run_id']} domain={run.cfg['domain']} solver={run.cfg['solver']['model']} proposer={run.cfg['proposer'].get('model')} iterations done={done} planned={n_more}")

    if not args.skip_baseline:
        run.phase_baselines()
    else:
        run.refresh_frontier()
    for i in range(n_more):
        if _interrupted:
            break
        run.iteration(done + i + 1, args.skip_smoke)
    print(f"\n{_ts()} evolution stopped after iteration {run.count_iterations()}. Finalize with: python -m core.meta_harness --run-dir {run_dir} --test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
