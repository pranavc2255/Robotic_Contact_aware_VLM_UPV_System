"""Safe observation pose helpers for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r21_move_to_safe_observation_pose_operator_confirmed.py.
This module intentionally does not import RTDEControlInterface; physical moves are executed by
execution/staged_gap_executor.py only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


DEFAULT_SAFE_OBSERVATION_TCP_POSE = [
    -0.04451708887209529,
    -0.30996360185460287,
    0.44493861019776015,
    1.6641551719613543e-05,
    -3.1411789271540913,
    -8.944466236178852e-05,
]


def load_safe_observation_pose(config: dict[str, Any] | None = None, path: Path | None = None) -> list[float]:
    if config and config.get("safe_observation_tcp_pose"):
        return [float(value) for value in config["safe_observation_tcp_pose"]]
    if path is not None:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return [float(value) for value in data["tcp_pose_ur_format"]]
    return list(DEFAULT_SAFE_OBSERVATION_TCP_POSE)


def is_near_safe_observation(
    actual_tcp_pose: list[float],
    safe_observation_tcp_pose: list[float],
    position_tolerance_m: float = 0.010,
    orientation_tolerance_rad: float = 0.050,
) -> bool:
    position_distance = math.sqrt(sum((float(actual_tcp_pose[idx]) - float(safe_observation_tcp_pose[idx])) ** 2 for idx in range(3)))
    orientation_distance = math.sqrt(sum((float(actual_tcp_pose[idx]) - float(safe_observation_tcp_pose[idx])) ** 2 for idx in range(3, 6)))
    return bool(position_distance <= position_tolerance_m and orientation_distance <= orientation_tolerance_rad)


def move_to_safe_observation_confirmed(*_args, **_kwargs) -> None:
    raise RuntimeError("Use execution.staged_gap_executor.execute_staged_gap_motion for physical safe-observation moves.")

