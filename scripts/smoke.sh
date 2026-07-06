#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 .github/scripts/ci_check.py

bash -n scripts/e2e.sh
bash -n scripts/smoke.sh

python3 -m compileall scripts .github/scripts >/dev/null

echo "smoke: ok"

