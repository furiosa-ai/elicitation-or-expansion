"""Static audit of candidate harness files (anti-cheating / anti-leakage).

Hard failures (candidate is rejected before evaluation):
  - imports of I/O, network, process or dynamic-import modules
  - calls to open()/exec()/eval()/__import__
  - direct LLM clients (openai, litellm, requests to the solver) that bypass ctx.llm budgets
  - direct environment construction (e.g. `AppWorld(`) that bypasses ctx.tools
  - literal strings from the held-out split (test ids, questions, answers) supplied by the domain
Warnings (reported, not rejected):
  - dataset/task name mentions, hard-coded API call patterns (domain-specific regexes)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
    "shutil", "pathlib", "glob", "io", "importlib", "ctypes", "pickle", "multiprocessing", "signal",
    "openai", "litellm", "anthropic", "appworld", "sqlite3", "tempfile", "builtins", "pty", "fcntl",
    "webbrowser", "ftplib", "smtplib", "xmlrpc", "asyncio",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "__import__", "compile", "breakpoint", "input"}
FORBIDDEN_NAMES = {"LLMClient", "OpenAI", "AppWorld", "make_client", "StubClient"}


def _iter_strings(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def audit_file(path: str | Path, literals: dict[str, list[str]] | None = None, warn_patterns: dict[str, str] | None = None) -> dict[str, Any]:
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as e:
        return {"ok": False, "errors": [f"parse error: {e}"], "warnings": []}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    errors.append(f"line {node.lineno}: forbidden import `{a.name}`")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                errors.append(f"line {node.lineno}: forbidden import `from {node.module}`")
            if node.module and node.module.startswith("core.llm"):
                for a in node.names:
                    if a.name in FORBIDDEN_NAMES:
                        errors.append(f"line {node.lineno}: `{a.name}` bypasses the item budget")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:  # builtins only; re.compile / str.join etc. are fine
                errors.append(f"line {node.lineno}: forbidden call `{fn.id}(`")
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if name in FORBIDDEN_NAMES:
                errors.append(f"line {node.lineno}: forbidden construction `{name}(` (use ctx.llm / ctx.tools)")
        elif isinstance(node, ast.Attribute) and node.attr in {"__subclasses__", "__globals__", "__builtins__", "f_back", "gi_frame"}:
            errors.append(f"line {node.lineno}: introspection escape `{node.attr}`")

    # held-out literals: exact substring match on any string constant or the raw source
    literals = literals or {}
    lowered_src = src.lower()
    for kind, values in literals.items():
        hits = 0
        for v in values:
            v = (v or "").strip()
            if len(v) < 4:  # too short to be meaningful (e.g. "yes", "no", years)
                continue
            if v.lower() in lowered_src:
                hits += 1
                if hits <= 5:
                    errors.append(f"held-out {kind} literal found in source: {v[:60]!r}")
        if hits > 5:
            errors.append(f"... {hits} held-out {kind} literals total")

    for label, pattern in (warn_patterns or {}).items():
        n = len(re.findall(pattern, src))
        if n:
            warnings.append(f"{label}: {n} match(es) for /{pattern}/")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def audit_dir(agents_dir: str | Path, literals: dict[str, list[str]] | None = None, warn_patterns: dict[str, str] | None = None, skip: set[str] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(agents_dir).glob("*.py")):
        if p.name.startswith("_") or (skip and p.stem in skip):
            continue
        out[p.stem] = audit_file(p, literals, warn_patterns)
    return out


if __name__ == "__main__":
    import json
    import sys

    for f in sys.argv[1:]:
        print(f, json.dumps(audit_file(f), indent=2))
