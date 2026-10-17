#!/usr/bin/env bash
# One-time AppWorld setup, after the Python environment is installed (see ../README.md):
#   scripts/appworld_data_setup.sh
# Unpacks the appworld package, downloads the official data bundle into data/appworld_root (tasks, api_docs, base_dbs,
# datasets train/dev/test_normal/test_challenge), installs the custom split files, and fetches the official
# simplified ReAct instructions from the public AppWorld repository (read by domains/appworld/prompting.py).
# Stops unless the data version is 0.1.0 (the version used) and `appworld verify tasks` succeeds
# (set APPWORLD_SKIP_VERIFY=1 to skip only the latter).
# AppWorld data is NOT redistributed here (see data/appworld_root/data/README_BEFORE_SHARING.md after download).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
AW="${APPWORLD_CLI:-$ROOT/.venv/bin/appworld}"
command -v "$AW" >/dev/null || { echo "appworld CLI not found ($AW): install the environment first or set APPWORLD_CLI"; exit 1; }
"$AW" install
mkdir -p data/appworld_root
if [ ! -d data/appworld_root/data/tasks ]; then
  "$AW" download data --root data/appworld_root
else
  echo "data/appworld_root/data already present - skipping download"
fi
DATA_VERSION="$(cat data/appworld_root/data/version.txt 2>/dev/null || true)"
[ "$DATA_VERSION" = "0.1.0" ] || { echo "expected AppWorld data version 0.1.0, found '${DATA_VERSION:-none}'"; exit 1; }
for f in domains/appworld/splits/*.txt; do
  dst="data/appworld_root/data/datasets/$(basename "$f")"
  [ -e "$dst" ] || { cp "$f" "$dst"; echo "installed split $(basename "$f")"; }
done
PROMPT=domains/appworld/prompts/react_instructions.txt
PROMPT_URL=https://raw.githubusercontent.com/StonyBrookNLP/appworld/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/react_code_agent/instructions.txt
PROMPT_SHA256=c41f7852217d46586047a38f68e88b827cb9dcbe624e9651f9db8301547a534b
[ -f "$PROMPT" ] || { mkdir -p "$(dirname "$PROMPT")"; curl -fsSL "$PROMPT_URL" -o "$PROMPT"; }
echo "$PROMPT_SHA256  $PROMPT" | sha256sum -c -
[ "${APPWORLD_SKIP_VERIFY:-0}" = 1 ] || "$AW" verify tasks --root data/appworld_root
echo "datasets: $(ls data/appworld_root/data/datasets | tr '\n' ' ')"
echo "tasks: $(ls data/appworld_root/data/tasks | wc -l)"
