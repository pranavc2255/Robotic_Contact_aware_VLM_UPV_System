"""Transform math helpers for camera/base scaffolding.

Provenance: copied/adapted from terminal_scripts/run_r5_camera_to_robot_base_transform_scaffold.py
and terminal_scripts/run_r12_plan_axis_aware_hover_orientation.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TransformMathError(RuntimeError):
    pass


def _rotation_class():
    try:
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415
    except ImportError as exc:
        raise TransformMathError("scipy is required for rotation-vector transform math.") from exc
    return R


def pose_vec_to_T_base_tool(pose: list[float]) -> np.ndarray:
    if not isinstance(pose, list) or len(pose) != 6:
        raise TransformMathError("TCP pose must be a six-value list.")
    values = np.array([float(item) for item in pose], dtype=float)
    if not np.isfinite(values).all():
        raise TransformMathError("TCP pose contains non-finite values.")
    rotation = _rotation_class().from_rotvec(values[3:]).as_matrix()
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = values[:3]
    return transform


def make_T_tool_camera_from_config(config: dict[str, Any]) -> np.ndarray:
    transform_cfg = config.get("transform", {})
    matrix = transform_cfg.get("T_tool_camera")
    if matrix is None:
        return np.eye(4, dtype=float)
    array = np.array(matrix, dtype=float)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise TransformMathError("transform.T_tool_camera must be a finite 4x4 matrix.")
    return array


def transform_camera_point_to_base(camera_xyz_m: list[float], T_base_tool: np.ndarray, T_tool_camera: np.ndarray) -> list[float]:
    point = np.array([float(camera_xyz_m[0]), float(camera_xyz_m[1]), float(camera_xyz_m[2]), 1.0], dtype=float)
    base = T_base_tool @ T_tool_camera @ point
    return [float(base[0]), float(base[1]), float(base[2])]

