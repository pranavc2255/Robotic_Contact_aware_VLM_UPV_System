"""Base-frame motion planning for v2 real execution.

This ports the T2/T4 planning idea into v2: use the current RTDE TCP pose as
T_base_tool0, combine it with the calibrated T_tool0_camera transform, project
the selected anchor pixel/depth into camera XYZ, and build T4-style staged
poses in the robot base frame. No hardware imports live in this module.
"""

from __future__ import annotations

from typing import Any
import math

from upv_vlm_v2.planning.tool_geometry import load_tool_geometry, tool_point
from upv_vlm_v2.planning.transforms import (
    apply_transform,
    load_transform,
    matmul4,
    matmul3,
    matrix3_to_rotvec,
    matrix_from_transform,
    pose_vector_to_matrix_ur,
    rotate_vector,
    z_rotation_matrix,
)


def _path(config: dict[str, Any], section: str, key: str) -> str | None:
    value = (config.get(section) or {}).get(key)
    return str(value) if value else None


def _workspace_limits(config: dict[str, Any]) -> dict[str, Any] | None:
    return load_transform(_path(config, "planning", "workspace_limits_file"))


def _within(pose: list[float], limits: dict[str, Any] | None, label: str) -> tuple[bool, list[str]]:
    if not limits:
        return False, [f"{label}: workspace limits missing"]
    workspace = limits.get("workspace_limits_base_m") or {}
    reasons: list[str] = []
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    for key, ok in [
        ("x_min", x >= float(workspace.get("x_min", -999.0))),
        ("x_max", x <= float(workspace.get("x_max", 999.0))),
        ("y_min", y >= float(workspace.get("y_min", -999.0))),
        ("y_max", y <= float(workspace.get("y_max", 999.0))),
        ("z_min", z >= float(workspace.get("z_min", -999.0))),
        ("z_max", z <= float(workspace.get("z_max", 999.0))),
    ]:
        if not ok:
            reasons.append(f"{label}.{key} violation")
    safety = limits.get("safety") or {}
    min_final_z = safety.get("min_final_tcp_z_m")
    if label == "final_pose" and min_final_z is not None and z < float(min_final_z):
        reasons.append("final_pose.min_final_tcp_z_m violation")
    return not reasons, reasons


def _dist_xy(a: list[float], b: list[float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _normalize_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _norm2(v: list[float]) -> list[float] | None:
    n = math.hypot(float(v[0]), float(v[1]))
    if n <= 1e-12:
        return None
    return [float(v[0]) / n, float(v[1]) / n]


def plan_t4_style_orientation(current_tcp_pose: list[float], selected_axis_image_px: list[float] | None, config: dict[str, Any]) -> dict[str, Any]:
    current_matrix = pose_vector_to_matrix_ur(current_tcp_pose)
    current_r = [row[:3] for row in current_matrix[:3]]
    current_x = _norm2([current_r[0][0], current_r[1][0]])
    if current_x is None:
        return {"success": False, "failure_reason": "current_tcp_x_axis_projection_degenerate", "orientation_after": current_tcp_pose[3:]}
    if not selected_axis_image_px:
        return {"success": True, "orientation_skipped": True, "orientation_after": current_tcp_pose[3:], "reason": "selected_axis_image_px_missing"}
    image_axis = _norm2([float(selected_axis_image_px[0]), float(selected_axis_image_px[1])])
    if image_axis is None:
        return {"success": False, "failure_reason": "selected_axis_image_degenerate", "orientation_after": current_tcp_pose[3:]}
    selected_axis_base = _norm2([-image_axis[0], image_axis[1]])
    assert selected_axis_base is not None
    orient = config.get("orientation") or {}
    current_yaw = math.degrees(math.atan2(current_x[1], current_x[0]))
    base_desired = math.degrees(math.atan2(selected_axis_base[1], selected_axis_base[0]))
    desired_after_extra = _normalize_deg(base_desired + float(orient.get("extra_yaw_deg", 90.0)))
    offsets = orient.get("candidate_yaw_offsets_deg", [0.0, 180.0, -180.0])
    candidates = [_normalize_deg(desired_after_extra + float(offset)) for offset in offsets]
    deltas = [_normalize_deg(candidate - current_yaw) for candidate in candidates]
    selected_idx = min(range(len(deltas)), key=lambda idx: abs(deltas[idx]))
    selected_delta = float(deltas[selected_idx])
    abs_delta = abs(selected_delta)
    deadband = float(orient.get("yaw_deadband_deg", 5.0))
    max_rotation = float(orient.get("max_single_rotation_deg", 120.0))
    if abs_delta > max_rotation:
        return {
            "success": False,
            "failure_reason": "T4_ORIENTATION_ROTATION_EXCEEDS_LIMIT",
            "orientation_after": current_tcp_pose[3:],
            "selected_yaw_delta_deg": selected_delta,
        }
    if abs_delta < deadband or not bool(orient.get("enabled", True)):
        target_r = current_r
        skipped = True
    else:
        target_r = matmul3(z_rotation_matrix(math.radians(selected_delta)), current_r)
        skipped = False
    return {
        "success": True,
        "orientation_source": "selected_axis_image_px",
        "selected_axis_image_px": image_axis,
        "selected_axis_base_xy": selected_axis_base,
        "current_tcp_x_axis_base_xy": current_x,
        "current_yaw_deg": current_yaw,
        "base_desired_yaw_deg": base_desired,
        "desired_yaw_after_extra_deg": desired_after_extra,
        "candidate_yaws_deg": candidates,
        "candidate_yaw_deltas_deg": deltas,
        "selected_yaw_delta_deg": selected_delta,
        "orientation_skipped_by_deadband": skipped,
        "orientation_after": matrix3_to_rotvec(target_r),
    }


def build_real_robot_motion_plan_from_current_tcp(
    *,
    current_tcp_pose: list[float],
    robot_plan_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(current_tcp_pose) != 6:
        return {"success": False, "failure_reason": "current_tcp_pose_invalid"}
    nested_diagnostics = robot_plan_payload.get("diagnostics") if isinstance(robot_plan_payload.get("diagnostics"), dict) else {}
    anchor_camera_xyz = (
        robot_plan_payload.get("anchor_camera_xyz_m")
        or robot_plan_payload.get("selected_anchor_camera_xyz_m")
        or nested_diagnostics.get("anchor_camera_xyz_m")
        or nested_diagnostics.get("selected_anchor_camera_xyz_m")
    )
    if not anchor_camera_xyz:
        return {
            "success": False,
            "failure_reason": "anchor_camera_xyz_missing_for_real_base_plan",
            "available_robot_plan_keys": sorted(robot_plan_payload.keys()),
            "available_nested_diagnostics_keys": sorted(nested_diagnostics.keys()),
        }
    diagnostics = {**nested_diagnostics, **robot_plan_payload}
    camera_to_tcp_path = _path(config, "tool", "camera_to_tcp_transform_file")
    tool_geometry_path = _path(config, "tool", "upv_tool_geometry_file")
    camera_to_tcp = load_transform(camera_to_tcp_path)
    tool_geometry = load_tool_geometry(tool_geometry_path)
    workspace = _workspace_limits(config)
    t_tool0_camera = matrix_from_transform(camera_to_tcp)
    if t_tool0_camera is None:
        return {"success": False, "failure_reason": "camera_to_tcp_transform_missing_or_invalid"}
    if tool_geometry is None:
        return {"success": False, "failure_reason": "upv_tool_geometry_missing"}
    if workspace is None:
        return {"success": False, "failure_reason": "workspace_limits_missing"}

    t_base_tool0_current = pose_vector_to_matrix_ur(current_tcp_pose)
    t_base_camera = matmul4(t_base_tool0_current, t_tool0_camera)
    anchor_base_xyz = apply_transform(t_base_camera, anchor_camera_xyz)
    if anchor_base_xyz is None:
        return {"success": False, "failure_reason": "anchor_base_transform_failed"}

    orientation = plan_t4_style_orientation(current_tcp_pose, robot_plan_payload.get("selected_axis_image_px"), config)
    if not orientation.get("success"):
        return {"success": False, "failure_reason": orientation.get("failure_reason", "orientation_plan_failed"), "orientation_plan": orientation}
    desired_rotvec = [float(v) for v in orientation.get("orientation_after", current_tcp_pose[3:])]
    desired_pose_at_origin = [0.0, 0.0, 0.0, *desired_rotvec]
    r_des = pose_vector_to_matrix_ur(desired_pose_at_origin)
    surface_target = tool_point(tool_geometry, "p_tool0_object_surface_target_m") or [0.0, -0.023, 0.085]
    rotated_surface = rotate_vector(r_des, surface_target)
    final_xyz = [
        anchor_base_xyz[0] - rotated_surface[0],
        anchor_base_xyz[1] - rotated_surface[1],
        anchor_base_xyz[2] - rotated_surface[2],
    ]
    robot_cfg = config.get("robot", {})
    planning_cfg = config.get("planning", {})
    midhover_z = float(robot_cfg.get("midhover_z_m", planning_cfg.get("midhover_z_m", 0.3)))
    preview_above = float(tool_geometry.get("preview_above_final_z_m", tool_geometry.get("approach_offset_m", 0.03)))
    final_pose = [*final_xyz, *desired_rotvec]
    move_midhover_pose = [float(current_tcp_pose[0]), float(current_tcp_pose[1]), midhover_z, *[float(v) for v in current_tcp_pose[3:6]]]
    orient_pose = [float(current_tcp_pose[0]), float(current_tcp_pose[1]), midhover_z, *desired_rotvec]
    xy_pose = [final_xyz[0], final_xyz[1], midhover_z, *desired_rotvec]
    approach_pose = [final_xyz[0], final_xyz[1], final_xyz[2] + preview_above, *desired_rotvec]

    poses = {
        "move_midhover_pose": move_midhover_pose,
        "orient_pose": orient_pose,
        "xy_pose": xy_pose,
        "approach_preview_pose": approach_pose,
        "approach_final_pose": final_pose,
    }
    workspace_check: dict[str, Any] = {"valid": True, "checks": {}, "reasons": []}
    for label, pose in poses.items():
        valid, reasons = _within(pose, workspace, label if label != "approach_final_pose" else "final_pose")
        workspace_check["checks"][label] = {"valid": valid, "reasons": reasons}
        if not valid:
            workspace_check["valid"] = False
            workspace_check["reasons"].extend(reasons)
    safety = workspace.get("safety") or {}
    xy_step = _dist_xy(current_tcp_pose, xy_pose)
    z_drop = float(move_midhover_pose[2]) - float(final_pose[2])
    workspace_check["xy_distance_from_current_m"] = xy_step
    workspace_check["final_z_drop_from_midhover_m"] = z_drop
    if xy_step > float(safety.get("max_xy_step_m", safety.get("max_single_move_distance_m", 999.0))):
        workspace_check["valid"] = False
        workspace_check["reasons"].append("T4_XY_STEP_EXCEEDS_LIMIT")
    if z_drop < 0.0 or z_drop > float(safety.get("max_final_z_drop_m", 999.0)):
        workspace_check["valid"] = False
        workspace_check["reasons"].append("T4_FINAL_Z_DROP_OUT_OF_RANGE")

    return {
        "success": bool(workspace_check["valid"]),
        "failure_reason": None if workspace_check["valid"] else "workspace_check_failed",
        "pose_frame": "robot_base",
        "current_tcp_pose": [float(v) for v in current_tcp_pose],
        "T_base_tool0_current": t_base_tool0_current,
        "T_base_camera": t_base_camera,
        "anchor_camera_xyz_m": [float(v) for v in anchor_camera_xyz],
        "anchor_base_xyz_m": anchor_base_xyz,
        "selected_anchor_px": robot_plan_payload.get("selected_anchor_px"),
        "motion_target_source": "selected_anchor_px",
        "orientation_plan": orientation,
        "surface_target_point_tool0_m": surface_target,
        "move_midhover_pose": move_midhover_pose,
        "orient_pose": orient_pose,
        "xy_pose": xy_pose,
        "approach_preview_pose": approach_pose,
        "approach_final_pose": final_pose,
        "target_pose_base": final_pose,
        "hover_pose_base": xy_pose,
        "approach_pose_base": approach_pose,
        "final_pose_base": final_pose,
        "workspace_safety_check": workspace_check,
    }
