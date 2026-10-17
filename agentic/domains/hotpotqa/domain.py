"""HotpotQA domain plugin (fullwiki, multi-hop QA over a fixed BM25 retriever)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.domain import Domain
from core.llm import Trace

from .data import load_split_file
from .evaluator import answer_exact_match, f1_score, gold_title_recall
from .tools import HotpotTools, Retriever

_RETRIEVERS: dict[str, Retriever] = {}


def _retriever(url: str) -> Retriever:
    if url not in _RETRIEVERS:
        _RETRIEVERS[url] = Retriever(url)
    return _RETRIEVERS[url]


class HotpotQADomain(Domain):
    name = "hotpotqa"
    worker_mode = "thread"
    answer_key = "answer"

    def load_split(self, split: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        paths = cfg["paths"]
        root = Path(paths["test_data"] if split in ("test", "selection") else paths["data"])
        return load_split_file(root / "hotpotqa" / f"{split}.jsonl")

    def public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"id": item["id"], "question": item["question"]}

    @contextmanager
    def open_tools(self, item: dict[str, Any], cfg: dict[str, Any]) -> Iterator[HotpotTools]:
        dc = cfg.get("domain_config") or {}
        trace = item.get("_trace")  # set by evaluate via wrapper below
        yield HotpotTools(_retriever(dc["retriever_url"]), trace or Trace(str(item["id"])), k_max=int(dc.get("retriever_k_max", 20)), max_calls=int(dc.get("max_retrieval_calls", 12)))

    def score(self, item: dict[str, Any], outcome: dict[str, Any] | None, tools: HotpotTools | None) -> tuple[bool, dict[str, Any]]:
        pred = ""
        if isinstance(outcome, dict):
            pred = str(outcome.get("answer") or "")
        correct = answer_exact_match(pred, item["answer"]) if pred else False
        metrics = {"f1": round(f1_score(pred, item["answer"]), 4) if pred else 0.0}
        if tools is not None:
            metrics["gold_title_recall"] = round(gold_title_recall(tools.retrieved_titles, item.get("gold_titles", [])), 3)
            metrics["retrieval_calls"] = tools.calls
        return correct, metrics

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        f1s = [r["metrics"].get("f1", 0.0) for r in results if r.get("metrics")]
        rec = [r["metrics"].get("gold_title_recall") for r in results if r.get("metrics") and r["metrics"].get("gold_title_recall") is not None]
        return {
            "avg_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "avg_gold_title_recall": round(sum(rec) / len(rec), 4) if rec else None,
        }

    def audit_literals(self, cfg: dict[str, Any]) -> dict[str, list[str]]:
        try:
            test = self.load_split("test", cfg)
        except FileNotFoundError:
            return {}
        # Answer strings that also occur in the proposer-visible splits (train/val) are not evidence of
        # memorizing the held-out split (e.g. "World War II"), so only test-unique answers are flagged.
        visible = " ".join(f"{t['question']} {t['answer']}" for split in ("train", "val") for t in self.load_split(split, cfg)).lower()
        answers = [t["answer"] for t in test if len(t["answer"]) >= 12 and t["answer"].lower() not in visible]
        return {"test_question": [t["question"] for t in test], "test_answer": answers, "test_id": [t["id"] for t in test]}


DOMAIN = HotpotQADomain()
