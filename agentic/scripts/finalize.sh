#!/usr/bin/env bash
# One-time held-out test evaluation for a finished run: scripts/finalize.sh <run_id>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/runs/${1:?usage: $0 <run_id>}"
if [[ -f "$RUN_DIR/run.pid" ]] && kill -0 "$(cat "$RUN_DIR/run.pid")" 2>/dev/null; then echo "run still alive"; exit 1; fi
cd "$ROOT" && "$ROOT/.venv/bin/python" -m core.meta_harness --run-dir "$RUN_DIR" --test 2>&1 | tee -a "$RUN_DIR/finalize.log"
