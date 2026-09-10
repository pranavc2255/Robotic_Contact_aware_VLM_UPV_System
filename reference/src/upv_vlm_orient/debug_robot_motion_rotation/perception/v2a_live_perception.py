"""Run V2a perception on a live capture for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r1b_realsense_v2a_perception_once.py
and terminal_scripts/run_v2a_axis_proportional_anchor_count.py integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from upv_vlm_orient.debug_robot_motion_rotation import REPO_ROOT


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerceptionResult:
    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    color_path: Path
    requested_class: str
    axis_mode: str


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _parse_v2a_run_dir(stdout_text: str, perception_root: Path) -> Path:
    match = re.search(r"^run_dir:\s*(.+)$", stdout_text, flags=re.MULTILINE)
    if match:
        run_dir = Path(match.group(1).strip()).resolve()
        if run_dir.exists():
            return run_dir
    child_dirs = [path for path in perception_root.iterdir() if path.is_dir()]
    if not child_dirs:
        raise PerceptionError(f"V2a completed but no output folder was found under {perception_root}")
    return max(child_dirs, key=lambda path: path.stat().st_mtime).resolve()


def run_v2a_on_capture(
    color_path: Path,
    requested_class: str,
    axis_mode: str,
    output_root: Path,
    config: dict[str, Any],
) -> PerceptionResult:
    v2a_config = config.get("v2a", {})
    script_path = _resolve_repo_path(v2a_config.get("script_path", "terminal_scripts/run_v2a_axis_proportional_anchor_count.py"))
    if not script_path.exists():
        raise PerceptionError(f"V2a script does not exist: {script_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--image",
        str(color_path),
        "--requested-class",
        requested_class,
        "--axis-mode",
        axis_mode,
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    stdout_path = output_root / "v2a_stdout.txt"
    stderr_path = output_root / "v2a_stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise PerceptionError(
            "V2a perception failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {completed.returncode}\n"
            f"stdout tail:\n{completed.stdout[-3000:]}\n"
            f"stderr tail:\n{completed.stderr[-3000:]}"
        )
    return PerceptionResult(
        run_dir=_parse_v2a_run_dir(completed.stdout, output_root),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        color_path=color_path,
        requested_class=requested_class,
        axis_mode=axis_mode,
    )

