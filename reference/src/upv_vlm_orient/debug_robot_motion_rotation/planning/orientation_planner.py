"""Axis-aware orientation planning for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r12_plan_axis_aware_hover_orientation.py,
terminal_scripts/run_r18_generate_orientation_mapping_candidates.py, and
terminal_scripts/run_r20_save_selected_orientation_mapping.py.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class OrientationPlanError(RuntimeError):
    pass


def _rotation_class():
    try:
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415
    except ImportError as exc:
        raise OrientationPlanError("scipy is required for orientation planning.") from exc
    return R


def _normalize_xy(vector_xy: list[float] | np.ndarray, label: str) -> np.ndarray:
    vector = np.array(vector_xy[:2], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise OrientationPlanError(f"{label} has zero-length XY projection.")
    return vector / norm


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def apply_selected_orientation_mapping(rotation, mapping: dict[str, Any] | None):
    yaw_offset_deg = 0.0
    candidate = None
    if mapping:
        candidate = mapping.get("selected_candidate_name")
        yaw_offset_deg = float(mapping.get("yaw_offset_deg", 0.0))
    # tcp_x_axis with yaw_offset_deg=0.0 intentionally adds no +90 offset.
    if abs(yaw_offset_deg) <= 1e-12:
        return rotation, {"selected_candidate_name": candidate, "yaw_offset_deg": yaw_offset_deg}
    R = _rotation_class()
    return R.from_euler("z", yaw_offset_deg, degrees=True) * rotation, {
        "selected_candidate_name": candidate,
        "yaw_offset_deg": yaw_offset_deg,
    }


def plan_axis_aware_orientation(
    *,
    current_tcp_pose: list[float],
    selected_axis_base_xy: list[float],
    selected_orientation_mapping: dict[str, Any] | None,
) -> dict[str, Any]:
    R = _rotation_class()
    current_pose = np.array([float(item) for item in current_tcp_pose], dtype=float)
    if current_pose.shape != (6,) or not np.isfinite(current_pose).all():
        raise OrientationPlanError("current_tcp_pose must be a finite six-value list.")
    current_rotation = R.from_rotvec(current_pose[3:])
    current_matrix = current_rotation.as_matrix()
    current_tcp_x_axis_base_xy = _normalize_xy(current_matrix[:, 0], "current TCP x-axis")
    selected_axis = _normalize_xy(selected_axis_base_xy, "selected axis")
    if float(np.dot(current_tcp_x_axis_base_xy, selected_axis)) < 0.0:
        selected_axis_for_yaw = -selected_axis
        axis_direction_adjusted = True
    else:
        selected_axis_for_yaw = selected_axis
        axis_direction_adjusted = False
    current_yaw = math.atan2(float(current_tcp_x_axis_base_xy[1]), float(current_tcp_x_axis_base_xy[0]))
    desired_yaw = math.atan2(float(selected_axis_for_yaw[1]), float(selected_axis_for_yaw[0]))
    yaw_error_rad = _wrap_pi(desired_yaw - current_yaw)
    yaw_rotation = R.from_euler("z", yaw_error_rad)
    desired_rotation = yaw_rotation * current_rotation
    desired_rotation, mapping_used = apply_selected_orientation_mapping(desired_rotation, selected_orientation_mapping)
    axis_aligned_pose = [
        float(current_pose[0]),
        float(current_pose[1]),
        float(current_pose[2]),
        *[float(value) for value in desired_rotation.as_rotvec()],
    ]
    return {
        "orientation_source": "axis_aware_tcp_x_axis",
        "tool_alignment_axis": "tcp_x_axis",
        "selected_axis_base_xy": selected_axis.tolist(),
        "selected_axis_for_yaw_base_xy": selected_axis_for_yaw.tolist(),
        "axis_direction_adjusted_for_minimal_yaw": axis_direction_adjusted,
        "current_tcp_x_axis_base_xy": current_tcp_x_axis_base_xy.tolist(),
        "yaw_error_deg": math.degrees(yaw_error_rad),
        "selected_orientation_mapping": mapping_used,
        "yaw_offset_deg": float(mapping_used["yaw_offset_deg"]),
        "axis_aligned_hover_tcp_pose": axis_aligned_pose,
        "robot_motion_allowed": False,
    }


def verify_descent_orientation_constant(prehover_oriented_pose: list[float], gap_pose: list[float], retract_pose: list[float]) -> dict[str, bool]:
    same_prehover_gap = all(abs(float(prehover_oriented_pose[idx]) - float(gap_pose[idx])) <= 1e-9 for idx in (3, 4, 5))
    same_gap_retract = all(abs(float(gap_pose[idx]) - float(retract_pose[idx])) <= 1e-9 for idx in (3, 4, 5))
    return {
        "prehover_oriented_and_gap_same_orientation": same_prehover_gap,
        "gap_and_retract_same_orientation": same_gap_retract,
        "ok": bool(same_prehover_gap and same_gap_retract),
    }

