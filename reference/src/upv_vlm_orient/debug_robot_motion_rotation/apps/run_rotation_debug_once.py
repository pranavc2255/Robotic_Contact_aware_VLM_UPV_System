"""One-shot rotation debug app.

Provenance: copied/adapted from terminal_scripts/run_r22_interactive_repeated_upv_gap_tests.py,
R1B/R3/R4/R5/R12/R13/R14/R16/R21 flow, organized as a package app.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from upv_vlm_orient.debug_robot_motion_rotation import REPO_ROOT
from upv_vlm_orient.debug_robot_motion_rotation.capture.realsense_capture import capture_realsense_rgbd
from upv_vlm_orient.debug_robot_motion_rotation.execution.staged_gap_executor import execute_staged_gap_motion
from upv_vlm_orient.debug_robot_motion_rotation.geometry.depth_projection import create_robot_point_candidate, project_pixel_to_camera_xyz
from upv_vlm_orient.debug_robot_motion_rotation.geometry.transform_math import (
    make_T_tool_camera_from_config,
    pose_vec_to_T_base_tool,
    transform_camera_point_to_base,
)
from upv_vlm_orient.debug_robot_motion_rotation.perception.perception_outputs import (
    get_selected_anchor,
    get_selected_anchor_pixel,
    get_selected_axis_unit_px,
    load_robot_anchor_geometry,
)
from upv_vlm_orient.debug_robot_motion_rotation.perception.v2a_live_perception import run_v2a_on_capture
from upv_vlm_orient.debug_robot_motion_rotation.planning.gap_motion_planner import plan_gap_motion
from upv_vlm_orient.debug_robot_motion_rotation.planning.orientation_planner import plan_axis_aware_orientation
from upv_vlm_orient.debug_robot_motion_rotation.planning.visual_xy_correction import (
    compute_legacy_pixel_scale_correction,
    compute_no_xy_correction,
)
from upv_vlm_orient.debug_robot_motion_rotation.robot.rtde_read_only import read_robot_state
from upv_vlm_orient.debug_robot_motion_rotation.visualization.overlays import save_rotation_debug_overlay


VALID_MODES = {"capture_only", "no_motion", "motion_no_xy", "motion_xy"}


def _timestamp_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _timestamp_folder() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_dir(config: dict[str, Any], mode: str) -> Path:
    root = _resolve_repo_path(config.get("output_root", "outputs/debug_robot_motion_rotation")) / mode
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / _timestamp_folder()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _axis_endpoint_pixels(anchor_px: list[float], axis_unit_px: list[float], distance_px: float = 80.0) -> tuple[list[float], list[float]]:
    return (
        [float(anchor_px[0]) - float(axis_unit_px[0]) * distance_px, float(anchor_px[1]) - float(axis_unit_px[1]) * distance_px],
        [float(anchor_px[0]) + float(axis_unit_px[0]) * distance_px, float(anchor_px[1]) + float(axis_unit_px[1]) * distance_px],
    )


def _selected_axis_base_xy(anchor_px: list[float], axis_unit_px: list[float], depth: np.ndarray, intrinsics: dict[str, Any], T_base_tool, T_tool_camera) -> list[float]:
    p0_px, p1_px = _axis_endpoint_pixels(anchor_px, axis_unit_px)
    depth_scale = float(intrinsics["depth_scale"])
    z0 = float(depth[int(round(anchor_px[1])), int(round(anchor_px[0]))]) * depth_scale
    if z0 <= 0:
        z0 = 0.35
    p0_camera = project_pixel_to_camera_xyz(p0_px, z0, intrinsics)
    p1_camera = project_pixel_to_camera_xyz(p1_px, z0, intrinsics)
    p0_base = transform_camera_point_to_base(p0_camera, T_base_tool, T_tool_camera)
    p1_base = transform_camera_point_to_base(p1_camera, T_base_tool, T_tool_camera)
    return [float(p1_base[0] - p0_base[0]), float(p1_base[1] - p0_base[1])]


def run_once(config_path: Path, mode: str, operator_confirmed: bool = False) -> Path:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
    config = _load_json(config_path)
    run_dir = _run_dir(config, mode)
    summary: dict[str, Any] = {
        "timestamp_start": _timestamp_now(),
        "mode": mode,
        "robot_motion_command_sent": False,
        "success": False,
        "error_reason": None,
    }
    start_s = time.perf_counter()
    try:
        capture_config = {**config, "output_root": str(run_dir)}
        capture = capture_realsense_rgbd(capture_config)
        summary["capture"] = {
            "run_dir": str(capture.run_dir),
            "color_path": str(capture.color_path),
            "depth_npy_path": str(capture.depth_npy_path),
            "intrinsics_path": str(capture.intrinsics_path),
            "metadata_path": str(capture.metadata_path),
        }
        if mode == "capture_only":
            summary["success"] = True
            return _finish(run_dir, summary, start_s)

        perception = run_v2a_on_capture(
            capture.color_path,
            str(config.get("requested_class", "brick")),
            str(config.get("axis_mode", "major")),
            run_dir / "perception",
            config,
        )
        geometry = load_robot_anchor_geometry(perception.run_dir)
        selected_anchor = get_selected_anchor(perception.run_dir)
        anchor_px = get_selected_anchor_pixel(perception.run_dir)
        axis_px = get_selected_axis_unit_px(perception.run_dir)
        depth = np.load(capture.depth_npy_path)
        intrinsics = _load_json(capture.intrinsics_path)
        point_candidate = create_robot_point_candidate(
            anchor_id=str(selected_anchor["anchor_id"]),
            anchor_center_px=anchor_px,
            selected_axis_unit_px=axis_px,
            depth_image=depth,
            intrinsics=intrinsics,
        )
        robot_state = read_robot_state(str(config["robot_ip"]))
        if robot_state.actual_tcp_pose is None:
            raise RuntimeError("RTDE read-only state did not provide actual_tcp_pose.")
        T_base_tool = pose_vec_to_T_base_tool(robot_state.actual_tcp_pose)
        T_tool_camera = make_T_tool_camera_from_config(config)
        base_contact = transform_camera_point_to_base(point_candidate["camera_xyz_m"], T_base_tool, T_tool_camera)
        selected_axis_base_xy = _selected_axis_base_xy(anchor_px, axis_px, depth, intrinsics, T_base_tool, T_tool_camera)
        orientation = plan_axis_aware_orientation(
            current_tcp_pose=robot_state.actual_tcp_pose,
            selected_axis_base_xy=selected_axis_base_xy,
            selected_orientation_mapping=config.get("selected_orientation_mapping"),
        )
        visual_enabled = mode == "motion_xy" or (mode == "no_motion" and bool(config.get("visual_correction", {}).get("enabled", False)))
        correction = (
            compute_legacy_pixel_scale_correction(
                desired_pixel=anchor_px,
                image_width=int(intrinsics["width"]),
                image_height=int(intrinsics["height"]),
                config=config,
            )
            if visual_enabled
            else compute_no_xy_correction()
        )
        plan = plan_gap_motion(
            base_contact_xyz_m=base_contact,
            axis_aligned_hover_tcp_pose=orientation["axis_aligned_hover_tcp_pose"],
            visual_correction=correction,
            config=config,
        )
        plan["output_dir"] = str(run_dir)
        overlay_path = save_rotation_debug_overlay(
            color_path=capture.color_path,
            output_path=run_dir / "rotation_debug_overlay.png",
            anchor_px=anchor_px,
            tool_center_px=correction.get("tool_center_pixel_px", [float(intrinsics["width"]) / 2.0, float(intrinsics["height"]) / 2.0]),
            axis_unit_px=axis_px,
            correction=correction,
        )
        summary.update(
            {
                "perception_run_dir": str(perception.run_dir),
                "selected_anchor": selected_anchor,
                "robot_anchor_geometry_path": str(perception.run_dir / "robot_anchor_geometry.json"),
                "point_candidate": point_candidate,
                "base_contact_xyz_m": base_contact,
                "orientation_plan": orientation,
                "visual_correction": correction,
                "gap_motion_plan": plan,
                "overlay_path": str(overlay_path),
                "target_geometry": geometry.get("target_geometry"),
            }
        )
        _write_json(run_dir / "selected_anchor_debug.json", summary)
        if mode in {"motion_no_xy", "motion_xy"}:
            motion_config = {**config, "motion": {**config.get("motion", {}), "move_robot": True}}
            execution = execute_staged_gap_motion(plan, motion_config, operator_confirmed=operator_confirmed)
            summary["execution"] = execution
            summary["robot_motion_command_sent"] = bool(execution.get("robot_motion_command_sent"))
        summary["success"] = True
    except Exception as exc:  # noqa: BLE001
        summary["error_reason"] = str(exc)
    return _finish(run_dir, summary, start_s)


def _finish(run_dir: Path, summary: dict[str, Any], start_s: float) -> Path:
    summary["timestamp_end"] = _timestamp_now()
    summary["total_time_s"] = time.perf_counter() - start_s
    summary_path = run_dir / "run_summary.json"
    _write_json(summary_path, summary)
    print(f"output_folder: {run_dir}")
    print(f"success: {summary['success']}")
    print(f"robot_motion_command_sent: {summary.get('robot_motion_command_sent')}")
    print(f"run_summary: {summary_path}")
    if summary.get("error_reason"):
        print(f"error_reason: {summary['error_reason']}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one robot rotation debug pass.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="no_motion")
    parser.add_argument("--operator-confirmed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_once(_resolve_repo_path(args.config), args.mode, operator_confirmed=args.operator_confirmed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
