from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def _run_git(args: list[str], repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"NA: git command failed: {exc}"


def collect_git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "git_branch": _run_git(["branch", "--show-current"], repo_root),
        "git_commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "git_status_short": _run_git(["status", "--short"], repo_root),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "virtual_env": os.environ.get("VIRTUAL_ENV", "NA"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "NA"),
    }


def write_git_snapshot(path: Path, repo_root: Path) -> dict[str, Any]:
    snapshot = collect_git_snapshot(repo_root)
    lines = []
    for key, value in snapshot.items():
        lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot
