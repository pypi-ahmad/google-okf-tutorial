#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
SKIP_SYNC=0

usage() {
  cat <<'USAGE'
Usage: scripts/e2e.sh [--skip-sync]

Runs both notebooks headlessly and writes executed outputs to ./artifacts/.

Requirements:
- uv installed
- Kaggle credentials via ~/.kaggle/kaggle.json OR KAGGLE_USERNAME/KAGGLE_KEY
- Ollama installed + running at http://127.0.0.1:11434
- Models pulled: qwen3.5:4b and nomic-embed-text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sync) SKIP_SYNC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

check_kaggle_creds() {
  if [[ -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]]; then
    return 0
  fi
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    return 0
  fi
  echo "Missing Kaggle credentials. Provide ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY." >&2
  exit 1
}

check_ollama_running() {
  # `ollama list` is the fastest end-to-end check (daemon reachable + client works).
  if ! ollama list >/dev/null 2>&1; then
    echo "Ollama is not reachable. Start it (e.g. 'ollama serve') and retry." >&2
    exit 1
  fi
}

model_available() {
  local name="$1"
  # Match with or without tag (e.g. nomic-embed-text:latest).
  ollama list | awk 'NR>1 {print $1}' | awk -F: '{print $1}' | grep -Fxq "$name"
}

echo "[e2e] repo: $ROOT_DIR"
echo "[e2e] artifacts: $ARTIFACTS_DIR"

need_cmd uv
need_cmd ollama
need_cmd python3

check_kaggle_creds
check_ollama_running

if ! model_available "qwen3.5"; then
  echo "Missing model: qwen3.5 (expected qwen3.5:4b). Run: ollama pull qwen3.5:4b" >&2
  exit 1
fi
if ! model_available "nomic-embed-text"; then
  echo "Missing model: nomic-embed-text. Run: ollama pull nomic-embed-text" >&2
  exit 1
fi

mkdir -p "$ARTIFACTS_DIR"

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  echo "[e2e] uv sync --locked"
  uv sync --locked
fi

run_nb() {
  local nb="$1"
  local out="$2"
  echo "[e2e] execute: $nb -> $ARTIFACTS_DIR/$out"
  uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$ARTIFACTS_DIR" \
    --output "$out" \
    "$nb"
}

run_nb "google_okf_zero_to_mastery.ipynb" "google_okf_zero_to_mastery.executed.ipynb"
run_nb "agentic_rag_chromadb.ipynb" "agentic_rag_chromadb.executed.ipynb"

if [[ ! -f bundle/index.md ]]; then
  echo "[e2e] expected bundle/index.md not found; notebook 1 may have failed" >&2
  exit 1
fi
if [[ ! -f bundle/viz.html ]]; then
  echo "[e2e] expected bundle/viz.html not found; notebook 1 may have failed" >&2
  exit 1
fi

echo "[e2e] ok"

