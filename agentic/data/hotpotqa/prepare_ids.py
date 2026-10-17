"""Check the HotpotQA splits written by domains/hotpotqa/data.py against the public task-ID list (no model calls).

    python data/hotpotqa/prepare_ids.py verify --tasks data/hotpotqa/split-tasks.csv \
        --train data/splits/hotpotqa/train.jsonl --val data/splits/hotpotqa/val.jsonl \
        --selection data/splits_test/hotpotqa/selection.jsonl --test data/splits_test/hotpotqa/test.jsonl

Ordered-ID digests are sha256("\\n".join(ids) + "\\n").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROLES = {"train": "train", "validation": "val", "selection": "selection", "test": "test"}


def ordered_digest(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_verify(a) -> int:
    expected: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(a.tasks, newline="")):
        expected.setdefault(row["split"], []).append(row)
    ok = True
    for split, arg in ROLES.items():
        path = getattr(a, arg)
        if path is None:
            continue
        want = sorted(expected[split], key=lambda r: int(r["split_index"]))
        got = read_jsonl(Path(path))
        same_ids = [r["task_id"] for r in want] == [str(r["id"]) for r in got]
        same_labels = same_ids and all(w["difficulty"] == g.get("level") and w["question_type"] == g.get("type")
                                       for w, g in zip(want, got))
        ok &= same_ids and same_labels
        print(f"{split:10} n={len(got):4} ordered_ids={'match' if same_ids else 'MISMATCH'} "
              f"labels={'match' if same_labels else 'MISMATCH'} sha256={ordered_digest([str(r['id']) for r in got])}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--tasks", required=True)
    for arg in ROLES.values():
        v.add_argument(f"--{arg}", default=None)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
