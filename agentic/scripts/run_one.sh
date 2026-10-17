#!/usr/bin/env bash
# Create (if needed) and launch one evolution run in the background.
#   scripts/run_one.sh configs/hotpotqa-8b.yaml [--run-id hotpotqa-8b-seed1] [extra meta_harness args...]
# Without --run-id the run_id in the config is used.
# Resume: run the same command again (iterations are "up to search.iterations total").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?usage: $0 configs/<config>.yaml [--run-id NAME] [args]}"; shift || true
if [[ "${1:-}" == "--run-id" ]]; then RUN_ID="${2:?--run-id needs a value}"; shift 2
else RUN_ID="$("$ROOT/.venv/bin/python" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")"; fi
RUN_DIR="$ROOT/runs/$RUN_ID"
if [[ -f "$ROOT/.env" ]]; then set -a; . "$ROOT/.env"; set +a; fi
: "${CLAUDE_CODE_OAUTH_TOKEN:?CLAUDE_CODE_OAUTH_TOKEN missing (put it in $ROOT/.env)}"
if [[ ! -f "$RUN_DIR/resolved_config.yaml" ]]; then "$ROOT/.venv/bin/python" "$ROOT/scripts/create_run.py" "$CFG" --run-id "$RUN_ID"; fi
if [[ -f "$RUN_DIR/run.pid" ]] && kill -0 "$(cat "$RUN_DIR/run.pid")" 2>/dev/null; then
  echo "run $RUN_ID already running (pid $(cat "$RUN_DIR/run.pid"))"; exit 1; fi
bash "$ROOT/scripts/preflight.sh" "$RUN_DIR"
cd "$ROOT"
setsid nohup "$ROOT/.venv/bin/python" -m core.meta_harness --run-dir "$RUN_DIR" "$@" >> "$RUN_DIR/run.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/run.pid"
echo "launched $RUN_ID pid=$(cat "$RUN_DIR/run.pid")  log: $RUN_DIR/run.log"
