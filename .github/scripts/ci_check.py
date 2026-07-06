#!/usr/bin/env python3
"""
CI checks that avoid network/secrets and avoid installing third-party deps.

Runs on GitHub Actions and locally.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def check_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing required path: {path.relative_to(REPO_ROOT)}")


def check_gitignore_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    if needle not in lines:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} must contain line: {needle!r}")


def check_notebook_json(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Invalid JSON notebook: {path.relative_to(REPO_ROOT)}: {e}") from e


def main() -> int:
    required_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / ".gitignore",
        REPO_ROOT / "google_okf_zero_to_mastery.ipynb",
        REPO_ROOT / "agentic_rag_chromadb.ipynb",
        REPO_ROOT / "scripts" / "e2e.sh",
        REPO_ROOT / "scripts" / "paired_eval.py",
        REPO_ROOT / "eval" / "eval_set_v1.yaml",
        REPO_ROOT / "reports" / "paired_eval_report.md",
        REPO_ROOT / "DATA_LICENSE.md",
    ]
    for p in required_files:
        check_exists(p)

    check_gitignore_contains(REPO_ROOT / ".gitignore", "bundle/")
    check_gitignore_contains(REPO_ROOT / ".gitignore", "artifacts/")

    check_notebook_json(REPO_ROOT / "google_okf_zero_to_mastery.ipynb")
    check_notebook_json(REPO_ROOT / "agentic_rag_chromadb.ipynb")

    print("ci_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
