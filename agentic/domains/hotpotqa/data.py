"""HotpotQA splits, replicating the LangProBe/GEPA procedure exactly.

LangProBe (Tan et al. 2025), used by GEPA:
    dataset  = HF `hotpot_qa/fullwiki` train split, original order (all difficulty levels)
    dev_set  = dataset[: 40%]            -> trimmed to 300 with random.Random(1).sample   -> our TEST (held out)
    val_set  = dataset[40% : 80%]        -> trimmed to 300 with random.Random(1).sample   -> our VAL (search set)
    train_set= dataset[80% :]            -> trimmed to 150 with random.Random(1).sample   -> our TRAIN (few-shot / prepare pool)
GEPA reports these as 150 train / 300 val / 300 test.

Files: <data_root>/hotpotqa/{train,val}.jsonl and <test_data_root>/hotpotqa/test.jsonl.
The test file lives outside the proposer-visible tree (see scripts/create_run.py).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SPLIT_SIZES = {"test": 300, "val": 300, "train": 150, "selection": 300}
# selection: a second, disjoint sample from the same LangProBe val region (rows[40%:80%]), seed 2.
# It is NEVER shown to the proposer; finalize uses it to pick the Optimized harness among the frontier.


def _to_item(raw: dict[str, Any]) -> dict[str, Any]:
    sf = raw.get("supporting_facts") or {}
    titles = sf.get("title") if isinstance(sf, dict) else None
    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw.get("type"),
        "level": raw.get("level"),
        "gold_titles": sorted(set(titles or [])),
    }


def _trim(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(rows):
        return rows
    rng = random.Random()
    rng.seed(1)
    return rng.sample(rows, size)


def build_splits(raw_train_jsonl: str | Path, out_dir: str | Path, test_out_dir: str | Path, sizes: dict[str, int] | None = None) -> dict[str, int]:
    sizes = {**SPLIT_SIZES, **(sizes or {})}
    rows = [json.loads(l) for l in Path(raw_train_jsonl).read_text().splitlines() if l.strip()]
    total = len(rows)
    dev = rows[: int(0.4 * total)]
    val = rows[int(0.4 * total): int(0.8 * total)]
    train = rows[int(0.8 * total):]
    val_sample = _trim(val, sizes["val"])
    val_ids = {r["id"] for r in val_sample}
    rng2 = random.Random(2)
    selection_sample = rng2.sample([r for r in val if r["id"] not in val_ids], sizes["selection"])
    splits = {
        "test": [_to_item(r) for r in _trim(dev, sizes["test"])],
        "val": [_to_item(r) for r in val_sample],
        "selection": [_to_item(r) for r in selection_sample],
        "train": [_to_item(r) for r in _trim(train, sizes["train"])],
    }
    out_dir = Path(out_dir) / "hotpotqa"
    test_dir = Path(test_out_dir) / "hotpotqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        target = (test_dir if name in ("test", "selection") else out_dir) / f"{name}.jsonl"
        with open(target, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    ids = [set(it["id"] for it in v) for v in splits.values()]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not (ids[i] & ids[j]), "splits overlap"
    return {k: len(v) for k, v in splits.items()}


def load_split_file(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"split file missing: {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/hotpot_qa/fullwiki_train.jsonl")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--test-out", default="data/splits_test")
    a = ap.parse_args()
    print(build_splits(a.raw, a.out, a.test_out))
