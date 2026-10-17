"""Keep solver credentials in the environment, never in saved run configs."""

from __future__ import annotations

import copy
import os
import re
from typing import Any

DEFAULT_API_KEY_ENV = "AGENTIC_SOLVER_API_KEY"


def solver_key_env(solver: dict[str, Any]) -> str:
    name = solver.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("solver.api_key_env must be an environment variable name")
    return name


def normalize_solver_config(solver: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(solver)
    # Accept old unauthenticated-local configs, but never silently discard a real key.
    if out.pop("api_key", None) not in (None, "", "local"):
        raise ValueError("Inline solver.api_key is not supported; use solver.api_key_env and export the key")
    out["api_key_env"] = solver_key_env(out)
    return out


def recordable_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["solver"] = normalize_solver_config(out.get("solver") or {})
    return out


def resolve_api_key(name: str) -> str:
    name = solver_key_env({"api_key_env": name})
    value = os.environ.get(name)
    if value:
        return value
    if name == DEFAULT_API_KEY_ENV:
        return "local"  # Preserve support for an unauthenticated local solver.
    raise ValueError(f"Solver credential environment variable {name} is not set")
