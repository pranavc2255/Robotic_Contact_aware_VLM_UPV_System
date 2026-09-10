"""v2 robot/clamp plan generation.

This module creates plan-only outputs. It does not import RTDE or serial.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from upv_vlm_v2.geometry.mask_geometry import load_camera_intrinsics, load_depth_array
from upv_vlm_v2.planning.clamp_plan import plan_clamp_opening
from upv_vlm_v2.planning.tool_geometry import load_tool_geometry, selected_path_length_mm, tool_point
from upv_vlm_v2.planning.transforms import apply_transform, camera_pixel_depth_to_xyz_m, load_transform, matrix_from_transform


def _anchor_depth_m(anchor_px: list[float] | None, depth_path: str | None) -> float | None:
    if not anchor_px or not depth_path:
        return None
    depth = load_depth_array(depth_path)
    if depth is None:
        return None
    x, y = int(round(anchor_px[0])), int(round(anchor_px[1]))
    if y < 0 or y >= depth.shape[0] or x < 0 or x >= depth.shape[1]:
        return None
    z = float(depth[y, x])
    if z <= 0:
        return None
    return z / 1000.0 if z > 20.0 else z


def _path_from_config(config: dict[str, Any], section: str, key: str) -> str | None:
    value = (config.get(section) or {}).get(key)
    return str(value) if value else None


def _load_workspace_limits(config: dict[str, Any]) -> dict[str, Any] | None:
    return load_transform(_path_from_config(config, "planning", "workspace_limits_file"))


def _pose_within_workspace(pose: list[float], limits: dict[str, Any] | None, label: str) -> tuple[bool, list[str]]:
    if not limits:
        return True, [f"{label}: workspace limits unavailable"]
    workspace = limits.get("workspace_limits_base_m") or {}
    reasons: list[str] = []
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    checks = [
        ("x_min", x >= float(workspace.get("x_min", -float("inf"))), x),
        ("x_max", x <= float(workspace.get("x_max", float("inf"))), x),
        ("y_min", y >= float(workspace.get("y_min", -float("inf"))), y),
        ("y_max", y <= float(workspace.get("y_max", float("inf"))), y),
        ("z_min", z >= float(workspace.get("z_min", -float("inf"))), z),
        ("z_max", z <= float(workspace.get("z_max", float("inf"))), z),
    ]
    for key, ok, value in checks:
        if not ok:
            reasons.append(f"{label}.{key} violation at {value:.6f}")
    safety = limits.get("safety") or {}
    min_final_z = safety.get("min_final_tcp_z_m")
    if label == "final_pose" and min_final_z is not None and z < float(min_final_z):
        reasons.append(f"{label}.min_final_tcp_z_m violation at {z:.6f}")
    return not reasons, reasons


def _workspace_check(poses: dict[str, list[float]], limits: dict[str, Any] | None, *, pose_frame: str) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "pose_frame": pose_frame,
        "limits_loaded": limits is not None,
        "checks": {},
        "valid": True,
        "reasons": [],
    }
    if pose_frame != "robot_base":
        checks["valid"] = None
        checks["reasons"].append("workspace check recorded but not enforced because pose frame is not robot_base")
        return checks
    for label, pose in poses.items():
        valid, reasons = _pose_within_workspace(pose, limits, label)
        checks["checks"][label] = {"valid": valid, "reasons": reasons}
        if not valid:
            checks["valid"] = False
            checks["reasons"].extend(reasons)
    return checks


def plan_robot_poses(
    *,
    axis_mode: str,
    anchor: Any,
    path_length: Any,
    config: dict[str, Any],
    capture_info: dict[str, Any],
    geometry: Any | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    planning = config.get("planning", {})
    simulation = bool(planning.get("simulation_enabled", False))
    backend = str((config.get("execution") or {}).get("backend", "none")).lower()
    allow_missing_sim_calibration = bool(planning.get("allow_missing_robot_calibration_in_simulation", False))
    warnings: list[str] = []
    local_path_mm, path_warnings = selected_path_length_mm(path_length)
    warnings.extend(path_warnings)
    depth_value = getattr(path_length, "depth_path_length_mm", None)
    try:
        depth_value_float = float(depth_value)
    except (TypeError, ValueError):
        depth_value_float = float("nan")
    path_length_source = "depth_edge_bin_3d" if math.isfinite(depth_value_float) and depth_value_float > 0.0 else "mask_fallback"
    clamp_plan = plan_clamp_opening(local_path_mm, config)
    if not clamp_plan.get("success"):
        warnings.extend(clamp_plan.get("warnings", []))
        return {"success": False, "failure_reason": clamp_plan.get("failure_reason"), "warnings": warnings, "clamp_plan": clamp_plan}
    camera_to_tcp_path = _path_from_config(config, "tool", "camera_to_tcp_transform_file")
    tool_geometry_path = _path_from_config(config, "tool", "upv_tool_geometry_file")
    workspace_path = _path_from_config(config, "planning", "workspace_limits_file")
    camera_to_tcp = load_transform(camera_to_tcp_path)
    tool_geometry = load_tool_geometry(tool_geometry_path)
    workspace_limits = _load_workspace_limits(config)
    calibration_missing = []
    if camera_to_tcp is None:
        calibration_missing.append(camera_to_tcp_path or "tool.camera_to_tcp_transform_file")
    if tool_geometry is None:
        calibration_missing.append(tool_geometry_path or "tool.upv_tool_geometry_file")
    if workspace_limits is None:
        calibration_missing.append(workspace_path or "planning.workspace_limits_file")
    real_calibration_loaded = not calibration_missing
    if not real_calibration_loaded:
        warnings.append("missing robot/tool calibration: " + ", ".join(calibration_missing))
        if backend == "real" or not (simulation and allow_missing_sim_calibration):
            return {
                "success": False,
                "failure_reason": "missing required calibration for real robot planning",
                "warnings": warnings,
                "diagnostics": {
                    "real_calibration_loaded_for_plan": False,
                    "missing_calibration": calibration_missing,
                    "execution_backend": backend,
                    "simulation_enabled": simulation,
                },
            }
    anchor_diagnostics = getattr(anchor, "diagnostics", {}) or {}
    robot_anchor_geometry = anchor_diagnostics.get("robot_anchor_geometry") if isinstance(anchor_diagnostics, dict) else None
    if isinstance(robot_anchor_geometry, dict) and robot_anchor_geometry.get("robot_candidate_anchor_center_px"):
        anchor_px = robot_anchor_geometry.get("robot_candidate_anchor_center_px")
        motion_target_source = "robot_anchor_geometry.robot_candidate_anchor_center_px"
    else:
        anchor_px = getattr(anchor, "selected_anchor_px", None)
        motion_target_source = "selected_anchor_px"
    selected_axis_image_px = getattr(geometry, "major_axis_vector", None) if axis_mode == "major" else getattr(geometry, "minor_axis_vector", None)
    intrinsics = load_camera_intrinsics(capture_info.get("camera_info_path"))
    z_m = _anchor_depth_m(anchor_px, capture_info.get("depth_path"))
    anchor_camera_xyz = None
    if anchor_px and z_m and intrinsics:
        anchor_camera_xyz = camera_pixel_depth_to_xyz_m(x_px=anchor_px[0], y_px=anchor_px[1], depth_m=z_m, intrinsics=intrinsics)
    if anchor_camera_xyz is None:
        warnings.append("anchor camera XYZ unavailable; using simulated nominal robot poses")
    if not simulation and anchor_camera_xyz is None:
        return {"success": False, "failure_reason": "missing required calibration/depth for real robot planning", "warnings": warnings}
    transform_matrix = matrix_from_transform(camera_to_tcp)
    anchor_tool0_xyz = apply_transform(transform_matrix, anchor_camera_xyz)
    if real_calibration_loaded and anchor_camera_xyz and anchor_tool0_xyz is None:
        return {"success": False, "failure_reason": "camera_to_tcp_transform_invalid", "warnings": warnings}
    if not real_calibration_loaded and simulation:
        warnings.append("simulation_plan_without_real_calibration")
    surface_target = tool_point(tool_geometry, "p_tool0_object_surface_target_m") or [0.0, -0.023, 0.085]
    hover = float(tool_geometry.get("hover_offset_m", planning.get("hover_offset_m", 0.15))) if tool_geometry else float(planning.get("hover_offset_m", 0.15))
    approach = float(tool_geometry.get("approach_offset_m", planning.get("approach_offset_m", 0.03))) if tool_geometry else float(planning.get("approach_offset_m", 0.03))
    final_contact = float(tool_geometry.get("final_contact_offset_m", planning.get("final_contact_offset_m", 0.0))) if tool_geometry else float(planning.get("final_contact_offset_m", 0.0))
    if anchor_tool0_xyz is not None:
        plan_xyz = [
            anchor_tool0_xyz[0] - surface_target[0],
            anchor_tool0_xyz[1] - surface_target[1],
            anchor_tool0_xyz[2] - surface_target[2] + final_contact,
        ]
        pose_frame = "tool0_relative_plan_frame"
        warnings.append("base-frame executable poses are deferred until real execute reads current RTDE TCP")
    else:
        plan_xyz = [0.0, 0.0, 0.20]
        pose_frame = "simulated_nominal_frame"
    final_pose = [plan_xyz[0], plan_xyz[1], plan_xyz[2], 0.0, 3.14159, 0.0]
    approach_pose = [plan_xyz[0], plan_xyz[1], plan_xyz[2] + approach, 0.0, 3.14159, 0.0]
    hover_pose = [plan_xyz[0], plan_xyz[1], plan_xyz[2] + hover, 0.0, 3.14159, 0.0]
    workspace = _workspace_check(
        {"hover_pose": hover_pose, "approach_pose": approach_pose, "final_pose": final_pose},
        workspace_limits,
        pose_frame=pose_frame,
    )
    requires_real_rebuild = backend == "real" and pose_frame != "robot_base"
    plan = {
        "success": True,
        "axis_mode": axis_mode,
        "selected_anchor_px": anchor_px,
        "selected_anchor_id": getattr(anchor, "final_anchor_id", None),
        "motion_target_source": motion_target_source,
        "qwen_required": bool(anchor_diagnostics.get("qwen_required", False)) if isinstance(anchor_diagnostics, dict) else False,
        "qwen_selected_anchor_id": anchor_diagnostics.get("qwen_selected_anchor_id") if isinstance(anchor_diagnostics, dict) else None,
        "final_anchor_source": anchor_diagnostics.get("final_anchor_source") if isinstance(anchor_diagnostics, dict) else None,
        "selected_axis_image_px": selected_axis_image_px,
        "anchor_camera_xyz_m": anchor_camera_xyz,
        "anchor_tool0_xyz_m": anchor_tool0_xyz,
        "T_tool0_camera": transform_matrix,
        "pose_frame": pose_frame,
        "requires_real_executor_rebuild": requires_real_rebuild,
        "workspace_check_deferred_until_real_executor": requires_real_rebuild,
        "target_pose_base": final_pose,
        "hover_pose_base": hover_pose,
        "approach_pose_base": approach_pose,
        "final_pose_base": final_pose,
        "clamp_opening_mm": clamp_plan["recommended_clamp_opening_mm"],
        "upv_path_length_mm": local_path_mm,
        "path_length_source": path_length_source,
        "clamp_width_source": path_length_source,
        "clamp_plan": clamp_plan,
        "warnings": warnings,
        "workspace_safety_check": workspace,
        "diagnostics": {
            "simulation_enabled": simulation,
            "execution_backend": backend,
            "real_calibration_loaded_for_plan": real_calibration_loaded,
            "requires_real_executor_rebuild": requires_real_rebuild,
            "workspace_check_deferred_until_real_executor": requires_real_rebuild,
            "motion_target_source": motion_target_source,
            "selected_anchor_px": anchor_px,
            "selected_anchor_id": getattr(anchor, "final_anchor_id", None),
            "qwen_required": bool(anchor_diagnostics.get("qwen_required", False)) if isinstance(anchor_diagnostics, dict) else False,
            "qwen_selected_anchor_id": anchor_diagnostics.get("qwen_selected_anchor_id") if isinstance(anchor_diagnostics, dict) else None,
            "final_anchor_source": anchor_diagnostics.get("final_anchor_source") if isinstance(anchor_diagnostics, dict) else None,
            "robot_anchor_geometry": robot_anchor_geometry,
            "selected_axis_image_px": selected_axis_image_px,
            "missing_calibration": calibration_missing,
            "camera_to_tcp_transform_file": camera_to_tcp_path,
            "T_tool0_camera": transform_matrix,
            "upv_tool_geometry_file": tool_geometry_path,
            "workspace_limits_file": workspace_path,
            "surface_target_point_tool0_m": surface_target,
            "local_path_length_source": path_length_source,
            "path_length_source": path_length_source,
            "clamp_width_source": path_length_source,
            "clamp_opening_is_not_upv_path_length": True,
            "workspace_safety_check": workspace,
        },
    }
    return plan
