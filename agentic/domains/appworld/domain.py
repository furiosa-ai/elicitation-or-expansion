"""AppWorld domain plugin (official splits; TGC per task, SGC per scenario)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.domain import Domain
from core.llm import Trace

from .data import SPLIT_TO_DATASET, read_task_ids, scenario_id, task_specs
from .tools import AppWorldEnv


def _root(cfg: dict[str, Any]) -> str:
    return os.environ.get("AGENTIC_APPWORLD_ROOT") or (cfg.get("domain_config") or {})["appworld_root"]


class AppWorldDomain(Domain):
    name = "appworld"
    worker_mode = "process"
    answer_key = "completed"

    def load_split(self, split: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Split roles -> official AppWorld datasets, configurable via domain_config.splits, e.g.
        {train: train, val: train, selection: dev, test: test_normal}. Unlisted roles use SPLIT_TO_DATASET."""
        root = _root(cfg)
        mapping = {**SPLIT_TO_DATASET, **((cfg.get("domain_config") or {}).get("splits") or {})}
        if split not in mapping:
            return []
        ids = read_task_ids(root, mapping[split])
        return [{"id": tid, "task_id": tid, "scenario": scenario_id(tid), "split": split} for tid in ids]

    def public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"id": item["id"], "task_id": item["task_id"]}

    @contextmanager
    def open_tools(self, item: dict[str, Any], cfg: dict[str, Any]) -> Iterator[AppWorldEnv]:
        root = _root(cfg)
        os.environ["APPWORLD_ROOT"] = root  # must be set before appworld is imported (path_store reads it once)
        from appworld import AppWorld  # noqa: PLC0415 - deferred on purpose

        dc = cfg.get("domain_config") or {}
        experiment = "/".join(x for x in ("mh", str(cfg.get("run_id", "run")), str(cfg.get("_agent", "agent")), str(cfg.get("_split", item.get("split", "split"))) + str(cfg.get("_exp_tag", ""))) if x)
        trace = item.get("_trace") or Trace(str(item["id"]))
        with AppWorld(
            task_id=item["task_id"],
            experiment_name=experiment,
            random_seed=int(dc.get("random_seed", 100)),
            raise_on_extra_parameters=bool(dc.get("raise_on_extra_parameters", True)),
        ) as world:
            yield AppWorldEnv(world, trace, max_steps=int(dc.get("max_steps", 50)), allow_evaluate=(item.get("split") == "train"))

    def score(self, item: dict[str, Any], outcome: dict[str, Any] | None, tools: AppWorldEnv | None) -> tuple[bool, dict[str, Any]]:
        if tools is None:
            return False, {}
        tracker = tools._evaluate()
        stats = tracker.to_dict(stats_only=True) if hasattr(tracker, "to_dict") else {}
        success = bool(getattr(tracker, "success", stats.get("success", False)))
        metrics = {
            "difficulty": stats.get("difficulty"),
            "num_tests": stats.get("num_tests"),
            "steps": tools.steps,
            "completed_call": tools.task_completed(),
        }
        return success, metrics

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        by_scenario: dict[str, list[bool]] = {}
        for r in results:
            by_scenario.setdefault(scenario_id(str(r["id"])), []).append(bool(r["correct"]))
        sgc = sum(1 for v in by_scenario.values() if all(v)) / len(by_scenario) if by_scenario else 0.0
        steps = [r["metrics"].get("steps", 0) for r in results if r.get("metrics")]
        return {"sgc": round(sgc, 4), "n_scenarios": len(by_scenario), "avg_steps": round(sum(steps) / len(steps), 1) if steps else 0.0}

    def audit_literals(self, cfg: dict[str, Any]) -> dict[str, list[str]]:
        root = (cfg.get("domain_config") or {}).get("appworld_root")  # host full root, never the filtered one
        if not root or not Path(root, "data", "datasets", "test_normal.txt").exists():
            return {}
        out: dict[str, list[str]] = {"test_task_id": [], "test_instruction": []}
        for ds in ("test_normal", "test_challenge"):
            try:
                ids = read_task_ids(root, ds)
            except FileNotFoundError:
                continue
            out["test_task_id"].extend(ids)
            for tid in ids:
                try:
                    out["test_instruction"].append(task_specs(root, tid)["instruction"])
                except (OSError, KeyError, ValueError):
                    pass
        return out

    def smoke_items(self, cfg: dict[str, Any], n: int = 1) -> list[dict[str, Any]]:
        return self.load_split("val", cfg)[:n]


DOMAIN = AppWorldDomain()
