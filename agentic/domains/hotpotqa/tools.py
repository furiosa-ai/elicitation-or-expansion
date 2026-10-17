"""Tools handed to HotpotQA harnesses: a budgeted, traced retriever client."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from core.llm import BudgetExceeded, Trace


class Retriever:
    """HTTP client for the wiki17 BM25 service. Deterministic: same query -> same passages."""

    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        for attempt in range(4):
            try:
                r = self._client.get(f"{self.base_url}/wiki17_abstracts", params={"query": query[:2000], "k": k})
                r.raise_for_status()
                return r.json()["topk"]
            except (httpx.HTTPError, KeyError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return []


class HotpotTools:
    """`ctx.tools` for HotpotQA. Records every retrieval in the item trace and enforces caps.

    retrieve(query, k=7) -> list[str] of passages formatted "Title | abstract" (DSPy ColBERTv2 style)
    retrieve_docs(query, k=7) -> list[dict] with text/title/pid/score
    """

    def __init__(self, retriever: Retriever, trace: Trace, k_max: int = 20, max_calls: int = 12):
        self._r = retriever
        self._trace = trace
        self.k_max = k_max
        self.max_calls = max_calls
        self.calls = 0
        self.retrieved_titles: list[str] = []
        self._lock = threading.Lock()

    def retrieve_docs(self, query: str, k: int = 7) -> list[dict[str, Any]]:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"max retrieval calls={self.max_calls} reached")
            self.calls += 1
        k = max(1, min(int(k), self.k_max))
        t0 = time.time()
        hits = self._r.search(str(query), k)
        titles = [h["title"] for h in hits]
        with self._lock:
            self.retrieved_titles.extend(titles)
        self._trace.add("retrieve", query=str(query)[:1000], k=k, titles=titles, pids=[h["pid"] for h in hits], elapsed=round(time.time() - t0, 3))
        return hits

    def retrieve(self, query: str, k: int = 7) -> list[str]:
        return [h["text"] for h in self.retrieve_docs(query, k)]
