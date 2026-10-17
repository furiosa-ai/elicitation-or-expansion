"""Harness interface every candidate must implement.

A candidate is a single Python file `agents/<name>.py` defining `class Harness(BaseHarness)`.

Lifecycle per evaluation:
    1. `prepare(train_items, ctx)` runs ONCE with the train split (labels included). Use it to
       build few-shot pools, playbooks, or any learned state. The state returned by `get_state()`
       is saved and re-loaded via `set_state()` before val/test evaluation, so `run()` must be
       stateless with respect to the evaluation stream (no learning from val/test items).
    2. `run(item, ctx)` is called once per evaluation item and returns an outcome dict
       (domain-specific, e.g. {"answer": "..."} for HotpotQA). It receives the public item
       (no labels) and a `RunContext` with the budgeted LLM handle and domain tools.

Rules enforced by the framework/audit:
    - No file or network I/O in candidate code (use ctx.llm / ctx.tools only).
    - Model, temperature, top_p, reasoning effort and per-item budgets are fixed by the run config.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from .llm import ItemLLM, Trace


class RunContext:
    """What a harness sees while solving one item."""

    def __init__(self, item_id: str, llm: ItemLLM, tools: Any, trace: Trace, config: dict[str, Any]):
        self.item_id = item_id
        self.llm = llm
        self.tools = tools
        self.trace = trace
        self.config = config  # read-only view of domain config (e.g. retriever k cap, max steps)

    def note(self, message: str, **data: Any) -> None:
        """Free-form trace annotation for debugging (visible to the proposer)."""
        self.trace.add("note", message=str(message)[:4000], **data)


class PrepareContext:
    """What a harness sees during `prepare` (train split only)."""

    def __init__(self, llm: ItemLLM, trace: Trace, config: dict[str, Any], open_tools: Callable[[dict[str, Any]], Any]):
        self.llm = llm
        self.trace = trace
        self.config = config
        self._open_tools = open_tools

    def open_tools(self, item: dict[str, Any]):
        """Context manager yielding domain tools for a TRAIN item (e.g. an AppWorld world)."""
        return self._open_tools(item)

    def note(self, message: str, **data: Any) -> None:
        self.trace.add("note", message=str(message)[:4000], **data)


class BaseHarness(ABC):
    """Base class for all candidates and baselines."""

    def prepare(self, train_items: list[dict[str, Any]], ctx: PrepareContext) -> None:  # noqa: B027
        """Optional one-time learning from the train split. Default: nothing."""
        return None

    def get_state(self) -> str:
        """JSON-serializable state produced by `prepare`. Default: empty."""
        return "{}"

    def set_state(self, state: str) -> None:  # noqa: B027
        """Restore state produced by `get_state`. Default: ignore."""
        return None

    @abstractmethod
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        """Solve one item. Return a domain-specific outcome dict."""


def load_harness_class(path: str | Path, class_name: str = "Harness") -> type[BaseHarness]:
    """Import `agents/<name>.py` from a file path and return its `Harness` class."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"_mh_agent_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, BaseHarness):
        raise TypeError(f"{path} must define `class {class_name}(BaseHarness)`")
    return cls


def state_to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
