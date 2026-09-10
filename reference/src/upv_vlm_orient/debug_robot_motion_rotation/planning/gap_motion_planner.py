"""Gap motion planning for staged rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r7_plan_hover_pose_from_base_candidate.py,
terminal_scripts/run_r11_update_hover_offset_from_measured_gap.py, and
terminal_scripts/run_r13_make_axis_aware_gap_motion_plan.py.
"""

from __future__ import annotations

import math
from typing import Any

from upv_vlm_orient.debug_robot_motion_rotation.planning.orientation_planner import verify_descent_orientation_constant


class GapPlanError(RuntimeError):
    pass


def build_uncorrected_prehover(base_contact_xyz_m: list[float], orientation: list[float], gap_z: float, clearance_m: float) -> list[float]:
    return [float(base_contact_xyz_m[0]), float(base_contact_xyz_m[1]), float(gap_z + clearance_m), *[float(v) for v in orientation]]


def build_prehover_oriented(uncorrected_prehover: list[float], orientation: list[float]) -> list[float]:
    return [float(uncorrected_prehover[0]), float(uncorrected_prehover[1]), float(uncorrected_prehover[2]), *[float(v) for v in orientation]]


def build_corrected_prehover(uncorrected_prehover: list[float], delta_x_base_m: float, delta_y_base_m: float) -> list[float]:
    return [
        float(uncorrected_prehover[0]) + float(delta_x_base_m),
        float(uncorrected_prehover[1]) + float(delta_y_base_m),
        float(uncorrected_prehover[2]),
        *[float(v) for v in uncorrected_prehover[3:]],
    ]


def build_gap_pose(corrected_prehover: list[float], gap_z: float) -> list[float]:
    return [float(corrected_prehover[0]), float(corrected_prehover[1]), float(gap_z), *[float(v) for v in corrected_prehover[3:]]]


def build_retract_pose(gap_pose: list[float], retract_extra_clearance_m: float) -> list[float]:
    return [float(gap_pose[0]), float(gap_pose[1]), float(gap_pose[2]) + float(retract_extra_clearance_m), *[float(v) for v in gap_pose[3:]]]


def _workspace_check(pose: list[float], limits: dict[str, float]) -> bool:
    x, y, z = pose[:3]
    return bool(
        float(limits["x_min"]) <= x <= float(limits["x_max"])
        and float(limits["y_min"]) <= y <= float(limits["y_max"])
        and float(limits["z_min"]) <= z <= float(limits["z_max"])
    )


def plan_gap_motion(
    *,
    base_contact_xyz_m: list[float],
    axis_aligned_hover_tcp_pose: list[float],
    visual_correction: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    motion = config.get("motion", {})
    desired_gap_m = float(config.get("desired_gap_m", 0.050))
    effective_offset_m = float(config.get("effective_lowest_tool_offset_from_active_tcp_m", 0.105))
    gap_z = float(base_contact_xyz_m[2]) + effective_offset_m + desired_gap_m
    orientation = [float(v) for v in axis_aligned_hover_tcp_pose[3:]]
    uncorrected = build_uncorrected_prehover(
        base_contact_xyz_m,
        orientation,
        gap_z,
        float(motion.get("prehover_extra_clearance_m", 0.080)),
    )
    prehover_oriented = build_prehover_oriented(uncorrected, orientation)
    corrected = build_corrected_prehover(
        prehover_oriented,
        float(visual_correction.get("delta_x_base_m", 0.0)),
        float(visual_correction.get("delta_y_base_m", 0.0)),
    )
    gap = build_gap_pose(corrected, gap_z)
    retract = build_retract_pose(gap, float(motion.get("retract_extra_clearance_m", 0.080)))
    orientation_check = verify_descent_orientation_constant(prehover_oriented, gap, retract)
    limits = motion.get("workspace_limits_base_m", {})
    workspace_ok = all(_workspace_check(pose, limits) for pose in [uncorrected, corrected, gap, retract]) if limits else True
    if not orientation_check["ok"]:
        raise GapPlanError("Prehover/gap/retract orientations are not constant.")
    if gap[2] > corrected[2]:
        raise GapPlanError("Gap pose is above corrected prehover pose.")
    return {
        "desired_gap_m": desired_gap_m,
        "effective_lowest_tool_offset_from_active_tcp_m": effective_offset_m,
        "uncorrected_prehover_tcp_pose": uncorrected,
        "prehover_oriented_pose": prehover_oriented,
        "corrected_prehover_tcp_pose": corrected,
        "prehover_tcp_pose": corrected,
        "gap_tcp_pose": gap,
        "retract_tcp_pose": retract,
        "delta_x_base_m": float(visual_correction.get("delta_x_base_m", 0.0)),
        "delta_y_base_m": float(visual_correction.get("delta_y_base_m", 0.0)),
        "delta_xy_norm_m": math.hypot(float(visual_correction.get("delta_x_base_m", 0.0)), float(visual_correction.get("delta_y_base_m", 0.0))),
        "visual_xy_correction_enabled": bool(visual_correction.get("enabled", False)),
        "orientation_constant_check": orientation_check,
        "workspace_all_poses_within_limits": workspace_ok,
        "motion_sequence": [
            "safe_observation",
            "uncorrected_high_prehover",
            "rotate_at_high_prehover",
            "corrected_high_prehover_lateral",
            "vertical_gap_descent",
            "hold",
            "vertical_retract",
            "safe_observation",
        ],
        "robot_motion_allowed": False,
    }

