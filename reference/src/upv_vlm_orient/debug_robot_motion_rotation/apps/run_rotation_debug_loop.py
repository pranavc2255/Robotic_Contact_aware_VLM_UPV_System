"""Interactive rotation debug loop.

Provenance: copied/adapted from terminal_scripts/run_r22_interactive_repeated_upv_gap_tests.py
and terminal_scripts/run_r27_interactive_simple_mask_motion_debug.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from upv_vlm_orient.debug_robot_motion_rotation import REPO_ROOT
from upv_vlm_orient.debug_robot_motion_rotation.apps.run_rotation_debug_once import VALID_MODES, run_once


COMMANDS = {"status", "capture_only", "no_motion", "motion_no_xy", "motion_xy", "quit", "exit"}


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def run_loop(config_path: Path) -> int:
    print("Rotation debug loop")
    print("Commands: status, capture_only, no_motion, motion_no_xy, motion_xy, quit")
    test_index = 1
    while True:
        try:
            command = input("rotation-debug> ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting.")
            return 0
        if command not in COMMANDS:
            print("Commands: status, capture_only, no_motion, motion_no_xy, motion_xy, quit")
            continue
        if command in {"quit", "exit"}:
            print("Exiting.")
            return 0
        if command == "status":
            print(f"config: {config_path}")
            print(f"next_test_index: {test_index}")
            continue
        operator_confirmed = False
        if command in {"motion_no_xy", "motion_xy"}:
            confirmation = input("Type RUN_ROTATION_DEBUG_MOTION to execute physical motion: ").strip()
            operator_confirmed = confirmation == "RUN_ROTATION_DEBUG_MOTION"
            if not operator_confirmed:
                print("Confirmation mismatch; motion command skipped.")
                continue
        run_dir = run_once(config_path, command, operator_confirmed=operator_confirmed)
        print(f"test_{test_index:03d}_output: {run_dir}")
        test_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive robot rotation debug loop.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_loop(_resolve_repo_path(args.config))
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

