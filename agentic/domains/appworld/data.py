"""AppWorld data helpers: split task ids, per-run roots, and the proposer-visible filtered root.

Official splits (Trivedi et al. 2024, used by ACE): train 90, dev 57, test_normal 168, test_challenge 417.
Mapping here:  train -> "train" (harness prepare pool)   val -> "dev" (search set)   test -> "test_normal" (held out).

AppWorld resolves everything relative to APPWORLD_ROOT (data/ and experiments/outputs/), so:
  * on the host each run gets `runs/<id>/appworld_root/` with `data -> <shared full data>` symlink and its own outputs;
  * the proposer container gets `runs/<id>/appworld_dev_root/` whose data/ contains ONLY train+dev tasks
    (hard-linked copies), so test tasks do not exist inside the sandbox at all.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

SPLIT_TO_DATASET = {"train": "train", "val": "dev", "test": "test_normal", "test_challenge": "test_challenge"}


def read_task_ids(root: str | Path, dataset: str) -> list[str]:
    p = Path(root) / "data" / "datasets" / f"{dataset}.txt"
    if not p.exists():
        raise FileNotFoundError(f"AppWorld dataset file missing: {p}")
    ids = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def task_specs(root: str | Path, task_id: str) -> dict[str, Any]:
    p = Path(root) / "data" / "tasks" / task_id / "specs.json"
    return json.loads(p.read_text())


def scenario_id(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def make_run_root(shared_root: str | Path, run_root: str | Path) -> Path:
    """Host-side per-run APPWORLD_ROOT: shared data via symlink, private experiments/outputs."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    data_link = run_root / "data"
    if not data_link.exists():
        data_link.symlink_to(Path(shared_root).resolve() / "data", target_is_directory=True)
    (run_root / "experiments" / "outputs").mkdir(parents=True, exist_ok=True)
    return run_root


def make_filtered_root(shared_root: str | Path, out_root: str | Path, datasets: tuple[str, ...] = ("train", "dev")) -> Path:
    """Proposer-visible APPWORLD_ROOT containing only the given datasets' tasks (no test tasks anywhere)."""
    shared = Path(shared_root).resolve() / "data"
    out = Path(out_root)
    data = out / "data"
    if data.exists():
        shutil.rmtree(data)
    (data / "datasets").mkdir(parents=True)
    (out / "experiments" / "outputs").mkdir(parents=True, exist_ok=True)
    for sub in ("api_docs", "base_dbs"):
        _copytree_hardlink(shared / sub, data / sub)
    keep: list[str] = []
    for ds in datasets:
        ids = read_task_ids(shared.parent, ds)
        keep.extend(ids)
        (data / "datasets" / f"{ds}.txt").write_text("\n".join(ids) + "\n")
    for tid in keep:
        _copytree_hardlink(shared / "tasks" / tid, data / "tasks" / tid)
    # sanity: no other task dirs
    present = {p.name for p in (data / "tasks").iterdir()}
    assert present == set(keep), "filtered root contains unexpected tasks"
    return out


def _copytree_hardlink(src: Path, dst: Path) -> None:
    def link(s: str, d: str) -> None:
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=link, dirs_exist_ok=True)
