#!/usr/bin/env bash
# Gate + isolation acceptance tests before launching a run: scripts/preflight.sh <run_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(cd "${1:?usage: $0 <run_dir>}" && pwd)"
PY="$ROOT/.venv/bin/python"
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
CFG="$RUN_DIR/resolved_config.yaml"; [[ -f "$CFG" ]] || fail "missing $CFG"
IMAGE="$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['proposer']['image'])" "$CFG")"
DOMAIN="$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['domain'])" "$CFG")"

echo "[1] host tools"
command -v docker >/dev/null || fail "docker missing"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image $IMAGE missing (docker build -f docker/proposer/Dockerfile .)"
HOST_CLI="$(claude --version 2>/dev/null | awk '{print $1}')"
IMG_CLI="$(docker run --rm --entrypoint claude "$IMAGE" --version 2>/dev/null | awk '{print $1}')"
EXPECTED_CLI="${PREFLIGHT_CLAUDE_VERSION:-$HOST_CLI}"
[[ -n "$EXPECTED_CLI" && "$EXPECTED_CLI" == "$IMG_CLI" ]] || fail "claude CLI mismatch expected=$EXPECTED_CLI host=$HOST_CLI image=$IMG_CLI"
[[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] || fail "CLAUDE_CODE_OAUTH_TOKEN not set"
echo "    claude image=$IMG_CLI expected=$EXPECTED_CLI host=$HOST_CLI, token present"

echo "[2] services"
"$PY" - "$ROOT" "$CFG" <<'PY' || fail "solver health check failed (check endpoint and credential environment)"
import json
import sys
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml

sys.path.insert(0, sys.argv[1])
from core.credentials import recordable_config, resolve_api_key, solver_key_env

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the solver credential to a redirect target.

try:
    solver = recordable_config(yaml.safe_load(Path(sys.argv[2]).read_text()))["solver"]
    request = Request(solver["api_base"].rstrip("/") + "/models", headers={
        "Authorization": "Bearer " + resolve_api_key(solver_key_env(solver)),
    })
    with build_opener(NoRedirect()).open(request, timeout=5) as response:
        models = json.load(response)
    if not any(isinstance(model, dict) and "id" in model for model in models.get("data", [])):
        raise ValueError("No model IDs returned")
except Exception:
    # Do not print HTTP bodies or exception text that might echo a credential.
    raise SystemExit(1)
PY
if [[ "$DOMAIN" == "hotpotqa" ]]; then
  RURL="$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['domain_config']['retriever_url'])" "$CFG")"
  curl -s -m 5 "$RURL/health" | grep -q '"ok":true' || fail "retriever not reachable at $RURL"
fi
echo "    solver + domain services OK"

echo "[3] sandbox isolation probes"
CHECKOUT="$RUN_DIR/checkout"; HOME_DIR="$RUN_DIR/proposer-home"; mkdir -p "$HOME_DIR"
MARK="TESTLEAK$RANDOM$RANDOM"; mkdir -p "$RUN_DIR/test-data/_probe"; echo "$MARK" > "$RUN_DIR/test-data/_probe/marker.txt"
RO_MOUNTS=(-v "$CHECKOUT/core:/workspace/core:ro" -v "$CHECKOUT/logs:/workspace/logs:ro" -v "$CHECKOUT/data:/workspace/data:ro")
[[ -d "$CHECKOUT/appworld_dev/data" ]] && RO_MOUNTS+=(-v "$CHECKOUT/appworld_dev/data:/workspace/appworld_dev/data:ro")
probe() { docker run --rm --user "$(id -u):$(id -g)" --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:exec,mode=1777 \
  -v "$CHECKOUT:/workspace:rw" -v "$HOME_DIR:/home/proposer:rw" "${RO_MOUNTS[@]}" -e HOME=/home/proposer -w /workspace "$IMAGE" bash -c "$1" 2>&1; }
probe "test -f /workspace/config.container.yaml" || fail "checkout not visible"
probe "test ! -e /var/run/docker.sock" || fail "docker.sock visible"
probe "test ! -e /workspace/../test-data && test ! -e '$ROOT' && test ! -e '$RUN_DIR'" || fail "host paths visible"
probe "[ \"\$(id -u)\" = \"$(id -u)\" ]" || fail "not running as host uid"
probe "touch /workspace/core/_probe 2>/dev/null && exit 1 || exit 0" || fail "core/ writable"
probe "touch /workspace/logs/_probe 2>/dev/null && exit 1 || exit 0" || fail "logs/ writable"
probe "touch /workspace/agents/_probe && rm /workspace/agents/_probe" || fail "agents/ not writable"
HITS="$(probe "grep -rls $MARK /workspace /home /tmp /opt /etc /usr /var 2>/dev/null | wc -l")"; [[ "$HITS" == "0" ]] || fail "test marker visible inside sandbox ($HITS hits)"
probe "find /workspace -name 'test.jsonl' -o -name 'selection.jsonl' -o -name 'test_normal.txt' -o -name 'test_challenge.txt' -o -name 'dev.txt' | grep -q . && exit 1 || exit 0" || fail "held-out split files inside sandbox"
probe "python -c 'import core.harness, domains'" || fail "python stack broken inside sandbox"
rm -rf "$RUN_DIR/test-data/_probe"
echo "    checkout rw / core+logs ro / no docker.sock / no host paths / no held-out data / host uid  -- all OK"
echo "PREFLIGHT OK"
