"""Domain plugin interface. Each domain lives in `domains/<name>/domain.py` and exposes `DOMAIN`."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator


class Domain(ABC):
    name: str = "base"
    #: "thread" for I/O-bound single-process work (HotpotQA), "process" when each item needs an
    #: isolated interpreter (AppWorld freezes time and monkeypatches builtins per task).
    worker_mode: str = "thread"
    #: outcome key holding the final answer, used for compact logging
    answer_key: str = "answer"

    @abstractmethod
    def load_split(self, split: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return items for `split` in {"train","val","test"}. Each item has a unique "id"."""

    @abstractmethod
    def public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Strip labels/ground truth; this is what harness.run receives."""

    @contextmanager
    def open_tools(self, item: dict[str, Any], cfg: dict[str, Any]) -> Iterator[Any]:
        """Domain tools for one item (retriever client, AppWorld world, ...)."""
        yield None

    @abstractmethod
    def score(self, item: dict[str, Any], outcome: dict[str, Any] | None, tools: Any) -> tuple[bool, dict[str, Any]]:
        """Return (correct, metrics) for one item. Called on the host with the labeled item."""

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Extra aggregate metrics beyond accuracy (e.g. SGC). Default: none."""
        return {}

    def audit_literals(self, cfg: dict[str, Any]) -> dict[str, list[str]]:
        """Strings that must never appear in candidate code (test ids, test answers, ...)."""
        return {}

    def smoke_items(self, cfg: dict[str, Any], n: int = 2) -> list[dict[str, Any]]:
        """Cheap items for the post-proposal smoke test (default: first n of val)."""
        return self.load_split("val", cfg)[:n]


def get_domain(name: str) -> Domain:
    module = importlib.import_module(f"domains.{name}.domain")
    domain = getattr(module, "DOMAIN")
    if not isinstance(domain, Domain):
        raise TypeError(f"domains.{name}.domain.DOMAIN must be a Domain instance")
    return domain
