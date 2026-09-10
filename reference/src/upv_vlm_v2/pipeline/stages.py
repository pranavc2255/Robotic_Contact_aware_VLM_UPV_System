from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from upv_vlm_v2.data.models import (
    AnchorSelectionResult,
    CaptureResult,
    ExecutionResult,
    GeometryResult,
    LocalPathLengthResult,
    RobotPlanResult,
    TargetSelectionResult,
)
from upv_vlm_v2.logging.timing import elapsed_ms, now_perf
from upv_vlm_v2.pipeline.context import PipelineContext
from upv_vlm_v2.acquisition.image_file_source import copy_saved_rgbd_inputs
from upv_vlm_v2.acquisition.ros2_realsense import capture_ros2_realsense_snapshot
from upv_vlm_v2.geometry.mask_geometry import add_dominant_rectangle_geometry, compute_mask_geometry, save_dominant_rectangle_overlay
from upv_vlm_v2.geometry.overlays import save_axis_overlay
from upv_vlm_v2.anchor_selection.anchor_selection_stage import run_anchor_selection
from upv_vlm_v2.geometry.local_chord import compute_local_chord
from upv_vlm_v2.perception.target_selection_stage import run_target_selection
from upv_vlm_v2.planning.robot_pose_planner import plan_robot_poses
from upv_vlm_v2.hardware.arduino_clamp_client import ArduinoClampClient, connect_clamp
from upv_vlm_v2.hardware.safety import validate_execution_request
from upv_vlm_v2.hardware.ur_rtde_client import connect_rtde
from upv_vlm_v2.execution.t4_style_executor import run_t4_style_execution


def _not_ported(stage: str, timing_ms: float, extra_warning: str | None = None) -> tuple[bool, str, list[str]]:
    warnings = [f"{stage} real implementation is not ported in v2 Phase 1."]
    if extra_warning:
        warnings.append(extra_warning)
    return False, "not_implemented_phase1", warnings


def run_capture_stage(context: PipelineContext) -> CaptureResult:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("capture")
    if context.dry_run:
        rgb = stage_dir / "synthetic_rgb.png"
        depth = stage_dir / "synthetic_depth.npy"
        info = stage_dir / "synthetic_camera_info.json"
        depth_png = stage_dir / "synthetic_depth_visualization.png"
        rgb.write_text("dry-run synthetic RGB placeholder\n", encoding="utf-8")
        depth.write_text("dry-run synthetic depth placeholder\n", encoding="utf-8")
        depth_png.write_text("dry-run synthetic depth visualization placeholder\n", encoding="utf-8")
        context.artifact_writer.write_stage_json("capture", "synthetic_camera_info.json", {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0})
        result = CaptureResult(
            success=True,
            rgb_path=str(rgb),
            depth_path=str(depth),
            camera_info_path=str(info),
            depth_visualization_path=str(depth_png),
            frame_id="dry_run_frame",
            timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
            diagnostics={"source": "synthetic", "hardware_used": False},
            timing_ms=elapsed_ms(start),
        )
    else:
        if context.live_ros2:
            snapshot = capture_ros2_realsense_snapshot(config=context.config, output_dir=stage_dir)
            if snapshot.get("success"):
                result = CaptureResult(
                    success=True,
                    rgb_path=snapshot.get("rgb_path"),
                    depth_path=snapshot.get("depth_path"),
                    camera_info_path=snapshot.get("camera_info_path"),
                    depth_visualization_path=snapshot.get("depth_visualization_path"),
                    frame_id="ros2_realsense_live",
                    timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                    diagnostics={"source": "ros2_realsense", "hardware_used": False, **snapshot.get("diagnostics", {})},
                    timing_ms=elapsed_ms(start),
                )
            else:
                result = CaptureResult(
                    success=False,
                    failure_reason=str(snapshot.get("failure_reason", "ros2_realsense_capture_failed")),
                    diagnostics=snapshot,
                    timing_ms=elapsed_ms(start),
                )
        elif context.input_rgb:
            input_cfg = context.config.get("input", {})
            camera_info = context.input_camera_info or input_cfg.get("camera_info_file")
            copied = copy_saved_rgbd_inputs(
                rgb_path=context.input_rgb,
                depth_path=context.input_depth,
                camera_info_path=camera_info,
                output_dir=stage_dir,
            )
            result = CaptureResult(
                success=True,
                rgb_path=copied["rgb_path"],
                depth_path=copied["depth_path"],
                camera_info_path=copied["camera_info_path"],
                depth_visualization_path=copied["depth_visualization_path"],
                frame_id="saved_input",
                timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                diagnostics={"source": "saved_image", "hardware_used": False, **copied},
                timing_ms=elapsed_ms(start),
            )
        else:
            result = CaptureResult(
                success=False,
                failure_reason="No input source provided. Use --live-ros2 or --input-rgb.",
                timing_ms=elapsed_ms(start),
            )
    context.config["_runtime_capture"] = result.to_dict()
    context.artifact_writer.write_stage_json("capture", "capture_result.json", result)
    return result


def run_target_selection_stage(context: PipelineContext, capture: CaptureResult) -> TargetSelectionResult:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("target_selection")
    if context.dry_run:
        mask = stage_dir / "selected_mask.png"
        overlay = stage_dir / "selected_mask_overlay.png"
        scores = stage_dir / "clip_scores.json"
        mask.write_text("dry-run selected mask placeholder\n", encoding="utf-8")
        overlay.write_text("dry-run selected mask overlay placeholder\n", encoding="utf-8")
        context.artifact_writer.write_stage_json("target_selection", "clip_scores.json", {"candidate_001": {context.requested_material: 1.0}})
        result = TargetSelectionResult(
            success=True,
            requested_material=context.requested_material,
            selected_candidate_id="candidate_001",
            selected_mask_path=str(mask),
            selected_rgb_path=capture.rgb_path,
            selected_mask_overlay_path=str(overlay),
            candidate_count=1,
            candidate_artifacts={"candidate_001": {"source": "dry_run"}},
            clip_scores_path=str(scores),
            diagnostics={"model_used": "synthetic", "primary_requested_class_only": True},
            timing_ms=elapsed_ms(start),
        )
    else:
        if not capture.rgb_path:
            result = TargetSelectionResult(
                success=False,
                requested_material=context.requested_material,
                failure_reason="capture_rgb_path_missing",
                timing_ms=elapsed_ms(start),
            )
        else:
            payload = run_target_selection(
                rgb_path=capture.rgb_path,
                requested_material=context.requested_material,
                config=context.config,
                output_dir=stage_dir,
            )
            result = TargetSelectionResult(
                success=bool(payload.get("success")),
                requested_material=context.requested_material,
                selected_candidate_id=payload.get("selected_candidate_id"),
                selected_mask_path=payload.get("selected_mask_path"),
                selected_rgb_path=payload.get("selected_rgb_path"),
                selected_mask_overlay_path=payload.get("selected_mask_overlay_path"),
                candidate_count=int(payload.get("candidate_count", 0) or 0),
                candidate_artifacts=payload.get("candidate_artifacts", {}),
                clip_scores_path=payload.get("clip_scores_path"),
                diagnostics={key: value for key, value in payload.items() if key not in {"candidate_artifacts"}},
                failure_reason=payload.get("failure_reason"),
                timing_ms=elapsed_ms(start),
            )
    context.artifact_writer.write_stage_json("target_selection", "target_selection_result.json", result)
    return result


def run_geometry_stage(context: PipelineContext, target: TargetSelectionResult) -> GeometryResult:
    start = now_perf()
    context.artifact_writer.stage_dir("geometry")
    if context.dry_run:
        result = GeometryResult(
            success=True,
            centroid_px=[320.0, 240.0],
            major_axis_vector=[1.0, 0.0],
            minor_axis_vector=[0.0, 1.0],
            major_axis_angle_deg=0.0,
            minor_axis_angle_deg=90.0,
            global_major_dimension_mm=200.0,
            global_minor_dimension_mm=95.0,
            overlay_paths={"geometry_overlay": str(context.artifact_writer.stage_dir("geometry") / "geometry_overlay.png")},
            diagnostics={"source": "synthetic"},
            timing_ms=elapsed_ms(start),
        )
        Path(result.overlay_paths["geometry_overlay"]).write_text("dry-run geometry overlay placeholder\n", encoding="utf-8")
    else:
        capture_info = context.config.get("_runtime_capture", {})
        if not target.selected_mask_path or not target.selected_rgb_path:
            result = GeometryResult(success=False, failure_reason="selected_mask_or_rgb_missing", timing_ms=elapsed_ms(start))
        else:
            try:
                geom = compute_mask_geometry(
                    target.selected_mask_path,
                    depth_path=capture_info.get("depth_path"),
                    camera_info_path=capture_info.get("camera_info_path"),
                )
                geom = add_dominant_rectangle_geometry(geom, target.selected_mask_path, context.config)
                overlay = save_axis_overlay(
                    rgb_path=target.selected_rgb_path,
                    mask_path=target.selected_mask_path,
                    geometry=geom,
                    output_path=context.artifact_writer.stage_dir("geometry") / "major_minor_axis_overlay.png",
                )
                overlay_paths = {"major_minor_axis_overlay": overlay, "global_dimension_overlay": overlay}
                if (geom.get("dominant_rectangle") or {}).get("success"):
                    dominant_overlay = save_dominant_rectangle_overlay(
                        rgb_path=target.selected_rgb_path,
                        geometry=geom,
                        output_path=context.artifact_writer.stage_dir("geometry") / "dominant_rectangle_geometry_overlay.png",
                    )
                    overlay_paths["dominant_rectangle_geometry_overlay"] = dominant_overlay
                    context.artifact_writer.write_stage_json("geometry", "dominant_rectangle_geometry.json", geom.get("dominant_rectangle", {}))
                context.artifact_writer.write_stage_json("geometry", "mask_geometry.json", geom)
                result = GeometryResult(
                    success=True,
                    centroid_px=geom["centroid_px"],
                    major_axis_vector=geom["major_axis_vector"],
                    minor_axis_vector=geom["minor_axis_vector"],
                    major_axis_angle_deg=geom["major_axis_angle_deg"],
                    minor_axis_angle_deg=geom["minor_axis_angle_deg"],
                    global_major_dimension_mm=geom.get("global_major_dimension_mm"),
                    global_minor_dimension_mm=geom.get("global_minor_dimension_mm"),
                    overlay_paths=overlay_paths,
                    diagnostics=geom,
                    timing_ms=elapsed_ms(start),
                )
            except Exception as exc:
                result = GeometryResult(success=False, failure_reason=str(exc), timing_ms=elapsed_ms(start))
    context.artifact_writer.write_stage_json("geometry", "geometry_result.json", result)
    return result


def run_anchor_selection_stage(context: PipelineContext, target: TargetSelectionResult, geometry: GeometryResult) -> dict[str, AnchorSelectionResult]:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("anchor_selection")
    anchor_cfg = context.config.get("anchor_selection", {})
    axes = anchor_cfg.get("axis_modes", ["major", "minor"]) if context.dry_run or bool(anchor_cfg.get("run_both_axis_modes", False)) else [context.axis_mode]
    results: dict[str, AnchorSelectionResult] = {}
    for axis in axes:
        if context.dry_run:
            overlay = stage_dir / f"{axis}_anchor_overlay.png"
            overlay.write_text("dry-run anchor overlay placeholder\n", encoding="utf-8")
            results[axis] = AnchorSelectionResult(
                success=True,
                axis_mode=axis,
                final_anchor_id=f"{axis}_anchor_001",
                selected_anchor_px=[320.0, 240.0],
                contact_point_a_px=[275.0, 240.0],
                contact_point_b_px=[365.0, 240.0],
                local_cross_axis_vector=[0.0, 1.0] if axis == "major" else [1.0, 0.0],
                score=1.0,
                candidate_count=1,
                overlay_paths={"anchor_overlay": str(overlay)},
                diagnostics={"source": "synthetic"},
                timing_ms=elapsed_ms(start),
            )
        else:
            if not target.selected_mask_path or not target.selected_rgb_path or not geometry.success:
                results[axis] = AnchorSelectionResult(success=False, axis_mode=axis, failure_reason="missing_target_or_geometry", timing_ms=elapsed_ms(start))
                continue
            geom_payload = geometry.diagnostics or {
                "centroid_px": geometry.centroid_px,
                "major_axis_vector": geometry.major_axis_vector,
                "minor_axis_vector": geometry.minor_axis_vector,
                "major_axis_length_px": None,
                "minor_axis_length_px": None,
            }
            payload = run_anchor_selection(
                mask_path=target.selected_mask_path,
                rgb_path=target.selected_rgb_path,
                geometry=geom_payload,
                axis_mode=axis,
                config=context.config,
                output_dir=stage_dir,
                depth_path=context.config.get("_runtime_capture", {}).get("depth_path"),
                camera_info_path=context.config.get("_runtime_capture", {}).get("camera_info_path"),
            )
            results[axis] = AnchorSelectionResult(
                success=bool(payload.get("success")),
                axis_mode=axis,
                final_anchor_id=payload.get("final_anchor_id"),
                selected_anchor_px=payload.get("selected_anchor_px"),
                contact_point_a_px=payload.get("contact_point_a_px"),
                contact_point_b_px=payload.get("contact_point_b_px"),
                local_cross_axis_vector=payload.get("local_cross_axis_vector"),
                score=payload.get("score"),
                candidate_count=int(payload.get("candidate_count", 0) or 0),
                candidate_artifacts=payload.get("candidate_artifacts", {}),
                overlay_paths=payload.get("overlay_paths", {}),
                diagnostics=payload.get("diagnostics", {}),
                failure_reason=payload.get("failure_reason"),
                timing_ms=elapsed_ms(start),
            )
    context.artifact_writer.write_stage_json("anchor_selection", "anchor_selection_result.json", results)
    return results


def run_local_path_length_stage(context: PipelineContext, target: TargetSelectionResult, geometry: GeometryResult, anchors: dict[str, AnchorSelectionResult]) -> dict[str, LocalPathLengthResult]:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("path_length")
    results: dict[str, LocalPathLengthResult] = {}
    for axis, anchor in anchors.items():
        if context.dry_run:
            overlay = stage_dir / f"{axis}_local_path_length_overlay.png"
            overlay.write_text("dry-run local path length overlay placeholder\n", encoding="utf-8")
            results[axis] = LocalPathLengthResult(
                success=True,
                axis_mode=axis,
                mask_path_length_mm=95.0 if axis == "major" else 200.0,
                depth_path_length_mm=94.2 if axis == "major" else 198.7,
                depth_valid=True,
                depth_mask_disagreement_ratio=0.01,
                endpoint_a_px=anchor.contact_point_a_px,
                endpoint_b_px=anchor.contact_point_b_px,
                overlay_paths={"local_path_length_overlay": str(overlay)},
                diagnostics={"source": "synthetic"},
                timing_ms=elapsed_ms(start),
            )
        else:
            capture_info = context.config.get("_runtime_capture", {})
            if not target.selected_mask_path or not target.selected_rgb_path or not anchor.success or not anchor.selected_anchor_px or not anchor.local_cross_axis_vector:
                results[axis] = LocalPathLengthResult(success=False, axis_mode=axis, failure_reason="missing_target_or_anchor", timing_ms=elapsed_ms(start))
                continue
            payload = compute_local_chord(
                mask_path=target.selected_mask_path,
                rgb_path=target.selected_rgb_path,
                anchor_px=anchor.selected_anchor_px,
                direction_xy=anchor.local_cross_axis_vector,
                output_dir=stage_dir,
                axis_mode=axis,
                depth_path=capture_info.get("depth_path"),
                camera_info_path=capture_info.get("camera_info_path"),
                config=context.config,
            )
            results[axis] = LocalPathLengthResult(
                success=bool(payload.get("valid")),
                axis_mode=axis,
                mask_path_length_mm=payload.get("mask_path_length_mm"),
                depth_path_length_mm=payload.get("depth_path_length_mm"),
                depth_valid=bool(payload.get("depth_valid", False)),
                depth_failure_reason=payload.get("depth_failure_reason"),
                depth_mask_disagreement_ratio=payload.get("depth_mask_disagreement_ratio"),
                endpoint_a_px=payload.get("endpoint_a_px"),
                endpoint_b_px=payload.get("endpoint_b_px"),
                overlay_paths={
                    "local_path_length_overlay": payload.get("overlay_path"),
                    "depth_used_points_overlay": payload.get("depth_used_points_overlay_path"),
                    "depth_edge_bins_overlay": payload.get("depth_edge_bins_overlay_path"),
                    "depth_projection_histogram": payload.get("depth_projection_histogram_path"),
                    "depth_local_pointcloud_npz": payload.get("depth_local_pointcloud_npz_path"),
                    "depth_local_pointcloud_ply": payload.get("depth_local_pointcloud_ply_path"),
                    "depth_width_diagnostics": payload.get("depth_width_diagnostics_path"),
                },
                diagnostics=payload,
                failure_reason=payload.get("failure_reason"),
                timing_ms=elapsed_ms(start),
            )
    context.artifact_writer.write_stage_json("path_length", "local_path_length_result.json", results)
    return results


def run_robot_planning_stage(context: PipelineContext, target: TargetSelectionResult, geometry: GeometryResult, anchors: dict[str, AnchorSelectionResult], path_lengths: dict[str, LocalPathLengthResult]) -> RobotPlanResult:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("robot_plan")
    if context.dry_run:
        result = RobotPlanResult(
            success=True,
            axis_mode=context.axis_mode,
            target_pose_base=[0.0, 0.0, 0.20, 0.0, 3.14, 0.0],
            hover_pose_base=[0.0, 0.0, 0.30, 0.0, 3.14, 0.0],
            approach_pose_base=[0.0, 0.0, 0.23, 0.0, 3.14, 0.0],
            final_pose_base=[0.0, 0.0, 0.20, 0.0, 3.14, 0.0],
            clamp_opening_mm=115.0,
            upv_path_length_mm=path_lengths.get(context.axis_mode, LocalPathLengthResult()).depth_path_length_mm,
            safety_warnings=["dry_run_plan_not_for_execution"],
            diagnostics={"source": "synthetic"},
            timing_ms=elapsed_ms(start),
        )
    else:
        anchor = anchors.get(context.axis_mode)
        path_length = path_lengths.get(context.axis_mode)
        if anchor is None or path_length is None or not anchor.success or not path_length.success:
            result = RobotPlanResult(
                success=False,
                axis_mode=context.axis_mode,
                failure_reason="missing_anchor_or_path_length_for_planning",
                timing_ms=elapsed_ms(start),
            )
        else:
            payload = plan_robot_poses(
                axis_mode=context.axis_mode,
                anchor=anchor,
                path_length=path_length,
                config=context.config,
                capture_info=context.config.get("_runtime_capture", {}),
                geometry=geometry,
                output_dir=stage_dir,
            )
            context.artifact_writer.write_stage_json("robot_plan", "robot_plan.json", payload)
            context.artifact_writer.write_stage_json("robot_plan", "clamp_plan.json", payload.get("clamp_plan", {}))
            context.artifact_writer.write_stage_json(
                "robot_plan",
                "anchor_target_adapter.json",
                {
                    "motion_target_source": payload.get("motion_target_source"),
                    "selected_anchor_id": payload.get("selected_anchor_id"),
                    "selected_anchor_px": payload.get("selected_anchor_px"),
                    "anchor_camera_xyz_m": payload.get("anchor_camera_xyz_m"),
                    "selected_axis_image_px": payload.get("selected_axis_image_px"),
                    "centroid_not_used_as_motion_target": True,
                    "qwen_required": payload.get("qwen_required"),
                    "qwen_selected_anchor_id": payload.get("qwen_selected_anchor_id"),
                    "final_anchor_source": payload.get("final_anchor_source"),
                },
            )
            context.artifact_writer.write_stage_json(
                "robot_plan",
                "t4_style_transform_chain.json",
                {
                    "selected_anchor_px": payload.get("selected_anchor_px"),
                    "selected_anchor_camera_xyz_m": payload.get("anchor_camera_xyz_m"),
                    "anchor_tool0_xyz_m": payload.get("anchor_tool0_xyz_m"),
                    "T_tool0_camera": payload.get("T_tool0_camera") or payload.get("diagnostics", {}).get("T_tool0_camera"),
                    "camera_to_tcp_transform_file": payload.get("diagnostics", {}).get("camera_to_tcp_transform_file"),
                    "T_base_tool0": "deferred_until_real_execute_current_tcp",
                    "T_base_camera": "deferred_until_real_execute_current_tcp",
                    "selected_anchor_base_xyz_m": "deferred_until_real_execute_current_tcp",
                    "tool_surface_target_point": payload.get("diagnostics", {}).get("surface_target_point_tool0_m"),
                    "planned_preview_pose": payload.get("approach_pose_base"),
                    "planned_final_pose": payload.get("final_pose_base"),
                    "planned_home_pose": (context.config.get("robot") or {}).get("home_pose"),
                    "workspace_check": payload.get("workspace_safety_check"),
                    "motion_target_source": payload.get("motion_target_source"),
                },
            )
            context.artifact_writer.write_stage_json(
                "robot_plan",
                "t4_style_planned_poses.json",
                {
                    "planned_midhover_pose": payload.get("hover_pose_base"),
                    "planned_preview_pose": payload.get("approach_pose_base"),
                    "planned_final_pose": payload.get("final_pose_base"),
                    "planned_home_pose": (context.config.get("robot") or {}).get("home_pose"),
                    "pose_frame": payload.get("pose_frame"),
                    "requires_real_executor_rebuild": payload.get("requires_real_executor_rebuild"),
                    "final_z_rule_used": "object_surface_equals_tool0_surface_target_point",
                },
            )
            context.artifact_writer.write_stage_json(
                "robot_plan",
                "safety_check.json",
                payload.get("workspace_safety_check") or {
                    "simulation_enabled": bool(context.config.get("planning", {}).get("simulation_enabled", False)),
                    "execution_backend": context.execution_backend,
                    "allow_real_hardware": context.allow_real_hardware,
                },
            )
            context.artifact_writer.write_stage_json(
                "robot_plan",
                "workspace_check.json",
                payload.get("workspace_safety_check") or {
                    "simulation_enabled": bool(context.config.get("planning", {}).get("simulation_enabled", False)),
                    "execution_backend": context.execution_backend,
                    "allow_real_hardware": context.allow_real_hardware,
                },
            )
            (stage_dir / "robot_plan_summary.txt").write_text(
                "\n".join([
                    f"axis_mode: {context.axis_mode}",
                    f"upv_path_length_mm: {payload.get('upv_path_length_mm')}",
                    f"clamp_opening_mm: {payload.get('clamp_opening_mm')}",
                    f"warnings: {payload.get('warnings', [])}",
                ]),
                encoding="utf-8",
            )
            result = RobotPlanResult(
                success=bool(payload.get("success")),
                axis_mode=context.axis_mode,
                target_pose_base=payload.get("target_pose_base"),
                hover_pose_base=payload.get("hover_pose_base"),
                approach_pose_base=payload.get("approach_pose_base"),
                final_pose_base=payload.get("final_pose_base"),
                clamp_opening_mm=payload.get("clamp_opening_mm"),
                upv_path_length_mm=payload.get("upv_path_length_mm"),
                safety_warnings=payload.get("warnings", []),
                diagnostics=payload,
                warnings=payload.get("warnings", []),
                failure_reason=payload.get("failure_reason"),
                timing_ms=elapsed_ms(start),
            )
    context.artifact_writer.write_stage_json("robot_plan", "robot_plan_result.json", result)
    return result


def run_hardware_execution_stage(context: PipelineContext, robot_plan: RobotPlanResult) -> ExecutionResult:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("execution")
    if context.dry_run:
        result = ExecutionResult(
            success=True,
            robot_moved=False,
            clamp_moved=False,
            upv_triggered=False,
            home_returned=False,
            diagnostics={"source": "synthetic", "hardware_used": False},
            warnings=["dry_run_does_not_execute_hardware"],
            timing_ms=elapsed_ms(start),
        )
    else:
        backend = context.execution_backend or str(context.config.get("execution", {}).get("backend", "simulated"))
        expected = str(context.config.get("safety", {}).get("execute_confirm_text", "RUN_V2_EXECUTE"))
        safety = validate_execution_request(
            backend=backend,
            confirm=context.confirm,
            expected_confirm=expected,
            allow_real_hardware=context.allow_real_hardware,
        )
        if not safety.get("ok"):
            result = ExecutionResult(
                success=False,
                robot_moved=False,
                clamp_moved=False,
                upv_triggered=False,
                failure_reason=safety.get("failure_reason"),
                warnings=["execution blocked before hardware import"],
                timing_ms=elapsed_ms(start),
            )
        else:
            robot_cfg = context.config.get("robot", {})
            clamp_cfg = context.config.get("clamp", {})
            robot = connect_rtde(
                backend=str(backend),
                robot_ip=robot_cfg.get("robot_ip"),
                allow_real_hardware=context.allow_real_hardware,
            )
            if str(backend) == "real":
                clamp = ArduinoClampClient(
                    backend=str(backend),
                    port=clamp_cfg.get("serial_port"),
                    baud=int(clamp_cfg.get("baud", 115200)),
                    allow_real_hardware=context.allow_real_hardware,
                )
            else:
                clamp = connect_clamp(
                    backend=str(backend),
                    port=clamp_cfg.get("serial_port"),
                    baud=int(clamp_cfg.get("baud", 115200)),
                    allow_real_hardware=context.allow_real_hardware,
                )
            plan_payload = robot_plan.diagnostics or robot_plan.to_dict()
            execution_payload = run_t4_style_execution(
                backend=str(backend),
                robot=robot,
                clamp=clamp,
                robot_plan_payload=plan_payload,
                config=context.config,
                output_dir=stage_dir,
            )
            robot_log = str(stage_dir / ("simulated_robot_execution_log.json" if backend == "simulated" else "real_robot_execution_log.json"))
            clamp_log = str(stage_dir / ("simulated_clamp_execution_log.json" if backend == "simulated" else "real_clamp_execution_log.json"))
            execution_success = bool(execution_payload.get("overall_success", execution_payload.get("success")))
            clamp_success = execution_payload.get("clamp_success")
            clamp_moved = bool(execution_payload.get("clamp_attempted")) and clamp_success is True
            home_returned = bool(execution_payload.get("home_success_after_execution"))
            result = ExecutionResult(
                success=execution_success,
                robot_moved=False if backend == "simulated" else bool(execution_payload.get("motion_success")),
                clamp_moved=False if backend == "simulated" else clamp_moved,
                upv_triggered=False,
                home_returned=False if backend == "simulated" else home_returned,
                execution_log_path=robot_log,
                diagnostics={
                    "backend": backend,
                    "sequence": "t4_full_final_home",
                    "motion_success": execution_payload.get("motion_success"),
                    "clamp_success": execution_payload.get("clamp_success"),
                    "home_success": execution_payload.get("home_success"),
                    "overall_success": execution_payload.get("overall_success"),
                    "arduino_available": execution_payload.get("arduino_available"),
                    "clamp_required": execution_payload.get("clamp_required"),
                    "clamp_attempted": execution_payload.get("clamp_attempted"),
                    "clamp_failure_reason": execution_payload.get("clamp_failure_reason"),
                    "release_attempted": execution_payload.get("release_attempted"),
                    "release_success": execution_payload.get("release_success"),
                    "home_attempted_after_execution": execution_payload.get("home_attempted_after_execution"),
                    "home_success_after_execution": execution_payload.get("home_success_after_execution"),
                    "final_robot_pose_if_available": execution_payload.get("final_robot_pose_if_available"),
                    "execution_payload": execution_payload,
                    "simulated_robot_moved": backend == "simulated",
                    "simulated_clamp_moved": backend == "simulated",
                    "simulated_home_returned": backend == "simulated",
                    "robot_log_path": robot_log,
                    "clamp_log_path": clamp_log,
                },
                warnings=["simulated_execution_no_hardware"] if backend == "simulated" else [],
                failure_reason=execution_payload.get("failure_reason"),
                timing_ms=elapsed_ms(start),
            )
    context.artifact_writer.write_stage_json("execution", "execution_result.json", result)
    return result
