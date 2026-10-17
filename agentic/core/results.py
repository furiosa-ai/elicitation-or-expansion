"""Aggregate summaries across agents: frontier, pareto, tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summaries(split_dir: Path) -> dict[str, dict[str, Any]]:
    """split_dir/<agent>/summary.json -> {agent: summary}. Only COMPLETE results are returned."""
    out: dict[str, dict[str, Any]] = {}
    if not split_dir.exists():
        return out
    for p in sorted(split_dir.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if s.get("complete"):
            out[p.parent.name] = s
    return out


def pareto(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """(name, accuracy, cost) -> non-dominated set sorted by accuracy desc, cost asc."""
    keep = []
    for name, acc, cost in points:
        dominated = any(
            (a2 >= acc and c2 <= cost and (a2 > acc or c2 < cost)) for _, a2, c2 in points
        )
        if not dominated:
            keep.append((name, acc, cost))
    return sorted(keep, key=lambda x: (-x[1], x[2]))


def compute_frontier(summaries: dict[str, dict[str, Any]], cost_key: str = "avg_prompt_tokens") -> dict[str, Any]:
    if not summaries:
        return {"_best": None, "_pareto": [], "agents": {}}
    pts = [(name, float(s["accuracy"]), float(s.get(cost_key, 0.0))) for name, s in summaries.items()]
    best_name, best_acc, best_cost = max(pts, key=lambda x: (x[1], -x[2]))
    return {
        "_best": {"agent": best_name, "accuracy": best_acc, cost_key: best_cost, "n_items": summaries[best_name]["n_items"]},
        "_pareto": [{"agent": n, "accuracy": a, cost_key: c} for n, a, c in pareto(pts)],
        "agents": {
            n: {k: s.get(k) for k in ("accuracy", "n_items", "n_errors", "n_budget_exceeded", "avg_calls", "avg_prompt_tokens", "avg_completion_tokens", "avg_elapsed")}
            for n, s in summaries.items()
        },
    }


def format_table(summaries: dict[str, dict[str, Any]], extra_keys: tuple[str, ...] = ()) -> str:
    rows = sorted(summaries.items(), key=lambda kv: -kv[1]["accuracy"])
    head = f"{'agent':<36}{'acc':>7}{'n':>6}{'err':>5}{'calls':>7}{'in_tok':>9}{'out_tok':>9}{'sec':>7}" + "".join(f"{k:>10}" for k in extra_keys)
    lines = [head, "-" * len(head)]
    for name, s in rows:
        lines.append(
            f"{name:<36}{s['accuracy']*100:>6.1f}%{s['n_items']:>6}{s['n_errors']:>5}{s['avg_calls']:>7.1f}"
            f"{s['avg_prompt_tokens']:>9.0f}{s['avg_completion_tokens']:>9.0f}{s['avg_elapsed']:>7.0f}"
            + "".join(f"{str(s.get(k, '-')):>10}" for k in extra_keys)
        )
    return "\n".join(lines)
