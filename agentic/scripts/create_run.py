#!/usr/bin/env python
"""Create an isolated run workspace under runs/<run_id>/ from configs/common.yaml + configs/<run>.yaml.

    python scripts/create_run.py configs/hotpotqa-8b.yaml [--run-id NAME] [--force]

Layout:
    runs/<run_id>/
      resolved_config.yaml   host config (credential environment names only)
      checkout/              mounted at /workspace in the proposer sandbox
        core/ domains/<d>/   framework + domain code (copied; read-only in the sandbox)
        agents/              baseline copies + candidates (rw)
        data/                train + val views ONLY (ro in sandbox)
        appworld_dev/        (appworld) filtered APPWORLD_ROOT with train+dev tasks only
        logs/                val results, traces, frontier, evolution_summary (ro in sandbox)
        proposals/           pending_eval.json, reports, prototypes (rw)
        .claude/skills/meta-harness/SKILL.md
        config.container.yaml
      proposer-home/         per-run $HOME for the Claude CLI
      test-data/             held-out split (host only)
      results/               test results (host only)
      llm-cache/             per-run solver response cache
      appworld_root/         (appworld) host APPWORLD_ROOT: data -> shared, private experiment outputs
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.credentials import recordable_config


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def copy_tree(src: Path, dst: Path, exclude: tuple[str, ...] = ("__pycache__", ".pytest_cache", "agents")) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*exclude), dirs_exist_ok=True)


def to_container(cfg: dict) -> dict:
    c = recordable_config(cfg)
    c["paths"] = {
        "checkout": "/workspace", "data": "/workspace/data", "test_data": "/nonexistent/test-data",
        "llm_cache": "/tmp/llm-cache", "logs": "/workspace/logs", "results": "/tmp/results", "proposer_home": "/home/proposer",
    }
    def host_to_gateway(url: str | None) -> str | None:
        if not url:
            return url
        return url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    c["solver"]["api_base"] = host_to_gateway(c["solver"].get("api_base"))
    dc = c.get("domain_config") or {}
    if "retriever_url" in dc:
        dc["retriever_url"] = host_to_gateway(dc["retriever_url"])
    if "appworld_root" in dc:
        dc["appworld_root"] = "/workspace/appworld_dev"
    c["domain_config"] = dc
    c["eval"] = {**(c.get("eval") or {}), "concurrency": min(4, int((c.get("eval") or {}).get("concurrency", 4))), "max_items": int((c.get("proposer") or {}).get("prototype_max_items", 20))}
    c.pop("proposer", None)
    c["_sandbox"] = True
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_config")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--force", action="store_true", help="delete an existing run dir with the same id")
    a = ap.parse_args()

    common = yaml.safe_load((REPO / "configs" / "common.yaml").read_text()) or {}
    run_cfg = yaml.safe_load(Path(a.run_config).read_text()) or {}
    cfg = recordable_config(deep_merge(common, run_cfg))
    if os.environ.get("AGENTIC_SOLVER_API_BASE"):
        cfg["solver"]["api_base"] = os.environ["AGENTIC_SOLVER_API_BASE"]
        print("  solver.api_base <- $AGENTIC_SOLVER_API_BASE")
    run_id = a.run_id or cfg.get("run_id") or Path(a.run_config).stem
    cfg["run_id"] = run_id
    domain = cfg["domain"]
    run_dir = REPO / "runs" / run_id
    hotpot_splits = {split: REPO / "data" / ("splits" if split in ("train", "val") else "splits_test") / "hotpotqa" / f"{split}.jsonl"
                     for split in ("train", "val", "selection", "test")}
    if domain == "hotpotqa":
        missing = [str(path) for path in hotpot_splits.values() if not path.exists()]
        if missing:
            raise SystemExit(f"HotpotQA splits missing ({', '.join(missing)}) - build them as in data/hotpotqa/DATA.md")
    if domain == "appworld" and not (REPO / "data" / "appworld_root" / "data" / "tasks").is_dir():
        raise SystemExit(f"AppWorld data missing under {REPO / 'data' / 'appworld_root'}/data - run scripts/appworld_data_setup.sh first")
    if run_dir.exists():
        if not a.force:
            raise SystemExit(f"{run_dir} exists (use --force to recreate)")
        shutil.rmtree(run_dir)
    checkout = run_dir / "checkout"
    for d in ("proposer-home", "test-data", "results", "llm-cache"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    (checkout / "agents").mkdir(parents=True)
    (checkout / "logs").mkdir()
    (checkout / "proposals" / "reports").mkdir(parents=True)
    (checkout / "proposals" / "proto").mkdir(parents=True)
    (checkout / "data").mkdir()

    # code
    copy_tree(REPO / "core", checkout / "core")
    (checkout / "domains").mkdir()
    (checkout / "domains" / "__init__.py").write_text("")
    if domain == "hotpotqa":
        # Prior candidates and their evaluation results must not seed a fresh
        # optimization. Only the configured baselines are copied below.
        copy_tree(REPO / "domains" / domain, checkout / "domains" / domain,
                  exclude=("__pycache__", ".pytest_cache", "agents", "evolved"))
    else:
        copy_tree(REPO / "domains" / domain, checkout / "domains" / domain)
    for fn in ("pyproject.toml", "uv.lock"):
        if (REPO / fn).exists():
            shutil.copy2(REPO / fn, checkout / fn)
    (checkout / "agents" / "__init__.py").write_text("")
    for name in cfg.get("baselines") or []:
        src = REPO / "domains" / domain / "agents" / f"{name}.py"
        if not src.exists():
            raise SystemExit(f"baseline missing: {src}")
        shutil.copy2(src, checkout / "agents" / f"{name}.py")
    skill_src = REPO / "domains" / domain / "SKILL.md"
    if not skill_src.exists():
        raise SystemExit(f"SKILL.md missing for domain {domain}: {skill_src}")
    skill_dst = checkout / ".claude" / "skills" / "meta-harness" / "SKILL.md"
    skill_dst.parent.mkdir(parents=True)
    shutil.copy2(skill_src, skill_dst)

    # data views
    if domain == "hotpotqa":
        (checkout / "data" / "hotpotqa").mkdir()
        (run_dir / "test-data" / "hotpotqa").mkdir(parents=True)
        for split, src in hotpot_splits.items():
            visible = split in ("train", "val")  # selection and test stay host-only
            shutil.copy2(src, (checkout / "data" if visible else run_dir / "test-data") / "hotpotqa" / f"{split}.jsonl")
    elif domain == "appworld":
        from domains.appworld.data import make_filtered_root, make_run_root

        shared = REPO / "data" / "appworld_root"
        for split_file in sorted((REPO / "domains" / "appworld" / "splits").glob("*.txt")):  # custom splits
            dst = shared / "data" / "datasets" / split_file.name
            if not dst.exists():
                shutil.copy2(split_file, dst)
                print(f"  installed custom split {split_file.name} into {dst.parent}")
        make_run_root(shared, run_dir / "appworld_root")
        roles = (cfg.get("domain_config") or {}).get("splits") or {}
        visible = tuple(sorted({roles.get("train", "train"), roles.get("val", "dev")}))
        make_filtered_root(shared, checkout / "appworld_dev", datasets=visible)
        print(f"  proposer-visible AppWorld datasets: {visible}")
        cfg.setdefault("domain_config", {})["appworld_root"] = str(run_dir / "appworld_root")
        (checkout / "data" / "README.md").write_text("AppWorld data lives in /workspace/appworld_dev (train + dev tasks only).\n")
    else:
        raise SystemExit(f"unknown domain {domain}")

    cfg["paths"] = {
        "checkout": str(checkout), "data": str(checkout / "data"), "test_data": str(run_dir / "test-data"),
        "llm_cache": str(run_dir / "llm-cache"), "logs": str(checkout / "logs"), "results": str(run_dir / "results"),
        "proposer_home": str(run_dir / "proposer-home"),
    }
    cfg["created_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (checkout / "config.container.yaml").write_text(yaml.safe_dump(to_container(cfg), sort_keys=False))

    # provenance
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    (run_dir / "git_state.json").write_text(json.dumps({"head": git("rev-parse", "HEAD"), "status": git("status", "--short"), "created_at": cfg["created_at"]}, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True).stdout
    (run_dir / "resolved-versions.txt").write_text(f"python {sys.version.split()[0]}\n" + "\n".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in {"openai", "appworld", "bm25s", "numba", "httpx", "jinja2", "pyyaml", "numpy"}) + "\n")
    print(f"created {run_dir}")
    print(f"  domain={domain} baselines={cfg.get('baselines')} solver={cfg['solver']['model']} proposer={cfg['proposer']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
