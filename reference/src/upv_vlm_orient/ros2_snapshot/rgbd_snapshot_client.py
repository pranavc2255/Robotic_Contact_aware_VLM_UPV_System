"""Client helpers for requesting a ROS2 RGB-D snapshot from Python scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rgbd_snapshot_node import capture_ros2_rgbd_snapshot


def load_snapshot_config(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def capture_snapshot_from_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    config = load_snapshot_config(config_path)
    if output_dir is None:
        output_dir = Path(config.get("output_root", "outputs/r35_ros2_rgbd_snapshot_once"))
    return capture_ros2_rgbd_snapshot(output_dir, config, timeout_s=timeout_s)
