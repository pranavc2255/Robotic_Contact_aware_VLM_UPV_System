from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from upv_vlm_v2.config import load_config, validate_config_schema
from upv_vlm_v2.data.models import (
    AnchorSelectionResult,
    CaptureResult,
    ExecutionResult,
    GeometryResult,
    LocalPathLengthResult,
    PipelineResult,
    RobotPlanResult,
    StageStatus,
    TargetSelectionResult,
)
from upv_vlm_v2.data.validation import validate_axis_mode, validate_requested_material
from upv_vlm_v2.logging.artifact_writer import ArtifactWriter
from upv_vlm_v2.logging.csv_logger import write_flat_csv
from upv_vlm_v2.logging.run_manifest import write_run_manifest
from upv_vlm_v2.logging.timing import elapsed_ms, now_perf
from upv_vlm_v2.pipeline.context import PipelineContext
from upv_vlm_v2.pipeline.managed_qwen import managed_qwen, validate_managed_qwen
from upv_vlm_v2.pipeline.weight_prefetch import prefetch_during_perception
from upv_vlm_v2.pipeline.modes import MODE_ORDER, PipelineMode, validate_mode
from upv_vlm_v2.pipeline.stages import (
    run_anchor_selection_stage,
    run_capture_stage,
    run_geometry_stage,
    run_hardware_execution_stage,
    run_local_path_length_stage,
    run_robot_planning_stage,
    run_target_selection_stage,
)
from upv_vlm_v2.hardware.arduino_clamp_client import ArduinoClampClient
from upv_vlm_v2.hardware.ur_rtde_client import connect_rtde, read_only_rtde_status
from upv_vlm_v2.planning.calibration import requires_robot_calibration, write_calibration_manifest


def _pose_distance(current: list[float], target: list[float]) -> tuple[float, float]:
    pos = math.sqrt(sum((float(current[i]) - float(target[i])) ** 2 for i in range(3)))
    ori = math.sqrt(sum((float(current[i + 3]) - float(target[i + 3])) ** 2 for i in range(3)))
    return pos, ori


def _stage_status(name: str, result: Any, artifact_dir: str | None) -> StageStatus:
    return StageStatus(
        name=name,
        success=bool(getattr(result, "success", False)),
        skipped=False,
        failure_reason=getattr(result, "failure_reason", None),
        warnings=list(getattr(result, "warnings", []) or []),
        diagnostics=dict(getattr(result, "diagnostics", {}) or {}),
        timing_ms=getattr(result, "timing_ms", None),
        artifact_dir=artifact_dir,
    )


def _skipped_status(name: str, artifact_dir: str | None) -> StageStatus:
    return StageStatus(name=name, success=True, skipped=True, artifact_dir=artifact_dir)


def _gpu_memory_snapshot(label: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"label": label, "available": False}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_used_mib": int(float(parts[2])),
                        "memory_free_mib": int(float(parts[3])),
                        "memory_total_mib": int(float(parts[4])),
                    }
                )
        payload.update({"available": bool(gpus), "source": "nvidia-smi", "gpus": gpus})
        return payload
    except Exception as exc:  # noqa: BLE001
        payload.update({"warning": str(exc), "exception_type": type(exc).__name__})
        return payload


def _flatten_result(result: PipelineResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "success": result.success,
        "mode": result.mode,
        "requested_material": result.requested_material,
        "axis_mode": result.axis_mode,
        "session_dir": result.session_dir,
        "failure_reason": result.failure_reason,
        "timing_ms": result.timing_ms,
    }
    if result.capture:
        row["rgb_path"] = result.capture.rgb_path
        row["depth_path"] = result.capture.depth_path
    if result.target:
        row["selected_candidate_id"] = result.target.selected_candidate_id
        row["selected_mask_path"] = result.target.selected_mask_path
    if result.geometry:
        row["centroid_px"] = result.geometry.centroid_px
        row["major_axis_angle_deg"] = result.geometry.major_axis_angle_deg
        row["minor_axis_angle_deg"] = result.geometry.minor_axis_angle_deg
        row["global_major_dimension_mm"] = result.geometry.global_major_dimension_mm
        row["global_minor_dimension_mm"] = result.geometry.global_minor_dimension_mm
    anchor = result.anchors.get(result.axis_mode) if result.anchors else None
    if anchor:
        row["final_anchor_id"] = anchor.final_anchor_id
        row["selected_anchor_px"] = anchor.selected_anchor_px
        row["contact_point_a_px"] = anchor.contact_point_a_px
        row["contact_point_b_px"] = anchor.contact_point_b_px
    path_length = result.path_lengths.get(result.axis_mode) if result.path_lengths else None
    if path_length:
        row["local_mask_path_length_mm"] = path_length.mask_path_length_mm
        row["local_depth_path_length_mm"] = path_length.depth_path_length_mm
        row["depth_valid"] = path_length.depth_valid
        row["depth_mask_disagreement_ratio"] = path_length.depth_mask_disagreement_ratio
    if result.robot_plan:
        row["clamp_opening_mm"] = result.robot_plan.clamp_opening_mm
        row["upv_path_length_mm"] = result.robot_plan.upv_path_length_mm
    if result.execution:
        row["robot_moved"] = result.execution.robot_moved
        row["clamp_moved"] = result.execution.clamp_moved
        row["upv_triggered"] = result.execution.upv_triggered
    return row


def _anchor_selection_runtime_provenance(config: dict[str, Any]) -> dict[str, Any]:
    anchor_cfg = config.get("anchor_selection") or {}
    backend = str(anchor_cfg.get("backend") or "")
    if backend == "llama32_single_anchor_l6_taxonomy_scoring" or str(anchor_cfg.get("vlm_mode") or "") == "single_image_loop":
        return {
            "anchor_selection_backend": "llama32_single_anchor_l6_taxonomy_scoring",
            "vlm_backend": "llama_single_image_loop",
            "anchor_layout_variant": str(anchor_cfg.get("anchor_layout_variant") or "L6_magenta_line_only_fixed"),
            "prompt_variant": str(anchor_cfg.get("prompt_variant") or "llama_single_anchor_v1_taxonomy"),
            "vlm_endpoint": str(anchor_cfg.get("vlm_server_endpoint") or "/infer"),
            "vlm_server_endpoint": str(anchor_cfg.get("vlm_server_endpoint") or "/infer"),
            "vlm_mode": "single_image_loop",
            "vlm_input_dir": "artifacts/04_anchor_selection/clean_single_anchor_inputs",
            "ranking_method": str(anchor_cfg.get("ranking_method") or "python_select_highest_usable_score_with_threshold"),
            "llama_min_selectable_score": float(anchor_cfg.get("llama_min_selectable_score", 85.0)),
            "anchor_selection_stage": "src.upv_vlm_v2.anchor_selection.anchor_selection_stage.run_anchor_selection",
        }
    if backend == "qwen32_multi_anchor_l6_batch_ranking" or str(anchor_cfg.get("qwen_mode") or "") == "multi_image":
        return {
            "anchor_selection_backend": "qwen32_multi_anchor_l6_batch_ranking",
            "anchor_layout_variant": str(anchor_cfg.get("anchor_layout_variant") or "L6_magenta_line_only_fixed"),
            "qwen_prompt_variant": str(anchor_cfg.get("qwen_prompt_variant") or "multi_image_v1_original"),
            "qwen_endpoint": str(anchor_cfg.get("qwen_server_endpoint") or "/infer_multi"),
            "qwen_mode": "multi_image",
            "qwen_input_dir": "artifacts/04_anchor_selection/clean_single_anchor_inputs",
            "anchor_selection_stage": "src.upv_vlm_v2.anchor_selection.anchor_selection_stage.run_anchor_selection",
        }
    return {}


def _controlled_failure_result(
    *,
    context: PipelineContext,
    mode: str,
    requested_material: str,
    axis_mode: str,
    stage_status: list[StageStatus],
    failure_reason: str,
    start: float,
) -> PipelineResult:
    return PipelineResult(
        success=False,
        mode=mode,
        requested_material=requested_material,
        axis_mode=axis_mode,
        session_dir=str(context.session_dir),
        stage_status=stage_status,
        failure_reason=failure_reason,
        timing_ms=elapsed_ms(start),
    )


def _run_pre_capture_home_guard(context: PipelineContext) -> tuple[StageStatus, dict[str, Any]]:
    start = now_perf()
    stage_dir = context.artifact_writer.stage_dir("pre_capture_home")
    robot_cfg = context.config.get("robot") or {}
    clamp_cfg = context.config.get("clamp") or {}
    payload: dict[str, Any] = {
        "pre_capture_hardware_check_performed": False,
        "ur_rtde_check_performed": False,
        "ur_rtde_available": None,
        "ur_rtde_failure_reason": None,
        "arduino_check_performed": False,
        "arduino_available": None,
        "arduino_failure_reason": None,
        "pre_capture_home_check_performed": False,
        "pre_capture_home_within_threshold": None,
        "pre_capture_home_move_attempted": False,
        "pre_capture_home_success": None,
        "pre_capture_ready_for_capture": None,
        "capture_started": False,
        "final_robot_pose_if_available": None,
    }
    enabled = bool(robot_cfg.get("home_before_capture_enabled", False))
    should_run = (
        context.mode == PipelineMode.EXECUTE.value
        and bool(context.live_ros2)
        and str(context.execution_backend).lower() == "real"
        and bool(context.allow_real_hardware)
        and enabled
    )
    if not should_run:
        payload["skipped_reason"] = "conditions_not_met_or_disabled"
        payload["pre_capture_ready_for_capture"] = True
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=True,
                skipped=True,
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )

    home_pose = robot_cfg.get("home_pose")
    robot_ip = robot_cfg.get("robot_ip")
    clamp_enabled = bool(clamp_cfg.get("enabled", False))
    payload["pre_capture_home_check_performed"] = True
    payload["pre_capture_hardware_check_performed"] = True
    payload["home_pose"] = home_pose
    payload["home_pose_tolerance_position_m"] = float(robot_cfg.get("home_pose_tolerance_position_m", 0.015))
    payload["home_pose_tolerance_orientation_rad"] = float(robot_cfg.get("home_pose_tolerance_orientation_rad", 0.10))
    payload["clamp_enabled"] = clamp_enabled
    if not robot_ip:
        payload["ur_rtde_check_performed"] = True
        payload["ur_rtde_available"] = False
        payload["ur_rtde_failure_reason"] = "robot_ip_missing"
        payload["pre_capture_ready_for_capture"] = False
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=False,
                skipped=False,
                failure_reason="pre_capture_ur_rtde_unavailable",
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )
    if not isinstance(home_pose, list) or len(home_pose) != 6:
        payload["failure_reason"] = "home_pose_missing_or_invalid"
        payload["pre_capture_ready_for_capture"] = False
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=False,
                skipped=False,
                failure_reason="home_pose_missing_or_invalid",
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )

    try:
        payload["ur_rtde_check_performed"] = True
        before = read_only_rtde_status(
            robot_ip=str(robot_ip),
            allow_real_hardware=True,
        )
        payload["ur_rtde_available"] = True
    except Exception as exc:  # noqa: BLE001
        payload["ur_rtde_available"] = False
        payload["ur_rtde_failure_reason"] = str(exc)
        payload["exception_type"] = type(exc).__name__
        payload["pre_capture_ready_for_capture"] = False
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=False,
                skipped=False,
                failure_reason="pre_capture_ur_rtde_unavailable",
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )

    if clamp_enabled:
        payload["arduino_check_performed"] = True
        try:
            port = clamp_cfg.get("serial_port")
            baud = int(clamp_cfg.get("baud", 115200))
            if not port or not Path(str(port)).exists():
                raise RuntimeError(f"serial_port_missing: {port}")
            clamp = ArduinoClampClient(
                backend="real",
                port=str(port),
                baud=baud,
                allow_real_hardware=True,
            )
            clamp.connect()
            clamp.configure_from_config(context.config)
            payload["arduino_ping_response"] = clamp.ping()
            payload["arduino_status_response"] = clamp.status()
            payload["arduino_available"] = True
            payload["arduino_command_log"] = clamp.commands
        except Exception as exc:  # noqa: BLE001
            payload["arduino_available"] = False
            payload["arduino_failure_reason"] = str(exc)
            payload["pre_capture_ready_for_capture"] = False
            context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
            return (
                StageStatus(
                    name="pre_capture_home",
                    success=False,
                    skipped=False,
                    failure_reason="pre_capture_arduino_unavailable",
                    timing_ms=elapsed_ms(start),
                    artifact_dir=str(stage_dir),
                ),
                payload,
            )
    else:
        payload["arduino_available"] = "not_checked_clamp_disabled"

    try:
        current_pose = before.get("actual_tcp_pose") or []
        payload["initial_robot_state"] = before
        payload["initial_robot_pose"] = current_pose
        if not isinstance(current_pose, list) or len(current_pose) != 6:
            raise RuntimeError("actual_tcp_pose_unavailable")
        pos_err, ori_err = _pose_distance(current_pose, home_pose)
        payload["initial_home_position_error_m"] = pos_err
        payload["initial_home_orientation_error_rad"] = ori_err
        within = (
            pos_err <= float(payload["home_pose_tolerance_position_m"])
            and ori_err <= float(payload["home_pose_tolerance_orientation_rad"])
        )
        payload["pre_capture_home_within_threshold"] = within
        if within:
            payload["pre_capture_home_success"] = True
            context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_robot_log.json", {"read_only_check": before})
        else:
            payload["pre_capture_home_move_attempted"] = True
            speed = float((robot_cfg.get("motion") or {}).get("speed_home_m_s", robot_cfg.get("speed_m_s", 0.008)))
            accel = float((robot_cfg.get("motion") or {}).get("accel_home_m_s2", robot_cfg.get("accel_m_s2", 0.015)))
            robot = connect_rtde(
                backend="real",
                robot_ip=str(robot_ip),
                allow_real_hardware=True,
            )
            robot.move_l(home_pose, speed, accel, "pre_capture_home")
            after = robot.snapshot()
            final_pose = after.get("actual_tcp_pose") or []
            payload["after_home_move_robot_state"] = after
            payload["final_robot_pose_if_available"] = final_pose
            if isinstance(final_pose, list) and len(final_pose) == 6:
                final_pos_err, final_ori_err = _pose_distance(final_pose, home_pose)
                payload["final_home_position_error_m"] = final_pos_err
                payload["final_home_orientation_error_rad"] = final_ori_err
                payload["pre_capture_home_success"] = (
                    final_pos_err <= float(payload["home_pose_tolerance_position_m"])
                    and final_ori_err <= float(payload["home_pose_tolerance_orientation_rad"])
                )
            else:
                payload["pre_capture_home_success"] = False
            robot.write_log(stage_dir / "pre_capture_home_robot_log.json")
        payload["pre_capture_ready_for_capture"] = bool(payload.get("pre_capture_home_success"))
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=bool(payload.get("pre_capture_home_success")),
                skipped=False,
                failure_reason=None if payload.get("pre_capture_home_success") else "pre_capture_home_failed",
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )
    except Exception as exc:  # noqa: BLE001
        payload["pre_capture_home_success"] = False
        payload["failure_reason"] = str(exc)
        payload["exception_type"] = type(exc).__name__
        payload["pre_capture_ready_for_capture"] = False
        context.artifact_writer.write_stage_json("pre_capture_home", "pre_capture_home_check.json", payload)
        return (
            StageStatus(
                name="pre_capture_home",
                success=False,
                skipped=False,
                failure_reason="pre_capture_home_failed",
                timing_ms=elapsed_ms(start),
                artifact_dir=str(stage_dir),
            ),
            payload,
        )


def run_full_main_upv_vlm_v2_pipeline(
    config_path: str | Path,
    requested_material: str,
    axis_mode: str = "major",
    mode: str = "dry_run",
    input_rgb: str | Path | None = None,
    input_depth: str | Path | None = None,
    input_camera_info: str | Path | None = None,
    live_ros2: bool = False,
    execution_backend: str | None = None,
    allow_real_hardware: bool = False,
    mock_qwen_response: str | Path | None = None,
    output_root: str | Path | None = None,
    confirm: str | None = None,
    anchor_selection_overrides: dict[str, Any] | None = None,
) -> PipelineResult:
    start = now_perf()
    selected_mode = validate_mode(mode)
    material = validate_requested_material(requested_material)
    axis = validate_axis_mode(axis_mode)
    config = validate_config_schema(load_config(config_path))
    if anchor_selection_overrides:
        anchor_cfg = config.setdefault("anchor_selection", {})
        for key, value in anchor_selection_overrides.items():
            if value is not None:
                anchor_cfg[key] = value
    configured_backend = str((config.get("execution") or {}).get("backend", "simulated" if selected_mode == PipelineMode.EXECUTE.value else "none"))
    backend = str(execution_backend or configured_backend).strip().lower()
    config.setdefault("execution", {})["backend"] = backend
    config["execution"]["allow_real_hardware"] = bool(allow_real_hardware or config.get("execution", {}).get("allow_real_hardware", False))
    config["_runtime_live_ros2"] = bool(live_ros2)
    config["_runtime_execution_backend"] = backend
    config["_runtime_requested_material"] = material
    anchor_selection_provenance = _anchor_selection_runtime_provenance(config)
    if anchor_selection_provenance:
        config["_runtime_anchor_selection_provenance"] = anchor_selection_provenance
    runtime_diagnostics: dict[str, Any] = {
        "gpu_memory_before_pipeline": _gpu_memory_snapshot("before_pipeline"),
        "target_selection_detector": str((config.get("perception") or {}).get("detector_model_id") or ""),
        "sam2_checkpoint": str((config.get("perception") or {}).get("sam2_checkpoint") or ""),
        "anchor_vlm_backend": anchor_selection_provenance.get("anchor_selection_backend"),
        "anchor_vlm_model": "Llama-3.2-11B-Vision-Instruct" if anchor_selection_provenance.get("anchor_selection_backend") == "llama32_single_anchor_l6_taxonomy_scoring" else None,
        "anchor_selection_provenance": anchor_selection_provenance,
    }
    config["_runtime_diagnostics"] = runtime_diagnostics
    if config.get("managed_qwen", {}).get("enabled"):
        runtime_diagnostics["anchor_vlm_model"] = config["managed_qwen"]["model_path"]
        runtime_diagnostics["model_residency_mode"] = "perception_then_managed_qwen_then_execution"
        runtime_diagnostics["managed_qwen_loading_in_total_pipeline_timing"] = True
    if mock_qwen_response:
        config["_runtime_mock_qwen_response"] = str(mock_qwen_response)
    dry_run = selected_mode == PipelineMode.DRY_RUN.value or bool(config.get("safety", {}).get("dry_run_default", False) and selected_mode == PipelineMode.DRY_RUN.value)
    root = Path(output_root or config.get("logging", {}).get("output_root", "outputs/v2_pipeline_runs"))
    session_dir = ArtifactWriter.make_session_dir(root)
    writer = ArtifactWriter(session_dir)
    writer.ensure_all_stage_dirs()
    writer.write_config_snapshot(config)
    simulation_enabled = bool((config.get("planning") or {}).get("simulation_enabled", False)) or backend == "simulated"
    calibration_required = requires_robot_calibration(
        mode=selected_mode,
        execution_backend=backend,
        simulation_enabled=simulation_enabled,
    )
    calibration_manifest, calibration_warnings = write_calibration_manifest(writer, config, required=calibration_required)
    write_run_manifest(writer, mode=selected_mode, requested_material=material, axis_mode=axis, config_path=str(config_path))

    context = PipelineContext(
        config=config,
        mode=selected_mode,
        requested_material=material,
        axis_mode=axis,
        input_rgb=Path(input_rgb) if input_rgb else None,
        input_depth=Path(input_depth) if input_depth else None,
        input_camera_info=Path(input_camera_info) if input_camera_info else None,
        live_ros2=bool(live_ros2),
        execution_backend=backend,
        allow_real_hardware=bool(allow_real_hardware),
        output_root=root,
        session_dir=session_dir,
        dry_run=dry_run,
        confirm=confirm,
        artifact_writer=writer,
    )

    if selected_mode == PipelineMode.EXECUTE.value and backend == "real":
        required = str(config.get("safety", {}).get("execute_confirm_text", "RUN_V2_EXECUTE"))
        if confirm != required or not allow_real_hardware:
            result = _controlled_failure_result(
                context=context,
                mode=selected_mode,
                requested_material=material,
                axis_mode=axis,
                stage_status=[],
                failure_reason=f"real execute mode requires --allow-real-hardware and confirm={required}",
                start=start,
            )
            result.runtime_diagnostics = runtime_diagnostics
            writer.write_json("pipeline_result.json", result)
            return result

    wanted = MODE_ORDER[selected_mode]
    statuses: list[StageStatus] = []
    capture: CaptureResult | None = None
    target: TargetSelectionResult | None = None
    geometry: GeometryResult | None = None
    anchors: dict[str, AnchorSelectionResult] = {}
    path_lengths: dict[str, LocalPathLengthResult] = {}
    robot_plan: RobotPlanResult | None = None
    execution: ExecutionResult | None = None
    pre_capture_home_payload: dict[str, Any] | None = None

    if not dry_run:
        validate_managed_qwen(config)
    pre_capture_home_status, pre_capture_home_payload = _run_pre_capture_home_guard(context)
    statuses.append(pre_capture_home_status)
    if not pre_capture_home_status.skipped and not pre_capture_home_status.success:
        wanted = []

    if "capture" in wanted:
        if pre_capture_home_payload is not None:
            pre_capture_home_payload["capture_started"] = True
        capture = run_capture_stage(context)
        if capture.diagnostics is not None and pre_capture_home_payload is not None:
            capture.diagnostics["pre_capture_home_check"] = pre_capture_home_payload
        statuses.append(_stage_status("capture", capture, str(writer.stage_dir("capture"))))
        if not capture.success:
            wanted = []
    else:
        statuses.append(_skipped_status("capture", str(writer.stage_dir("capture"))))

    if "target_selection" in wanted and capture:
        runtime_diagnostics["gpu_memory_before_target_selection"] = _gpu_memory_snapshot("before_target_selection")
        with prefetch_during_perception(config, writer.stage_dir("anchor_selection"), dry_run=dry_run):
            target = run_target_selection_stage(context, capture)
        runtime_diagnostics["gpu_memory_after_target_selection"] = _gpu_memory_snapshot("after_target_selection")
        statuses.append(_stage_status("target_selection", target, str(writer.stage_dir("target_selection"))))
        if not target.success:
            wanted = []
    elif "target_selection" not in wanted:
        statuses.append(_skipped_status("target_selection", str(writer.stage_dir("target_selection"))))

    if "geometry" in wanted and target:
        geometry = run_geometry_stage(context, target)
        statuses.append(_stage_status("geometry", geometry, str(writer.stage_dir("geometry"))))
        if not geometry.success:
            wanted = []
    elif "geometry" not in wanted:
        statuses.append(_skipped_status("geometry", str(writer.stage_dir("geometry"))))

    if "anchor_selection" in wanted and target and geometry:
        runtime_diagnostics["gpu_memory_before_anchor_selection"] = _gpu_memory_snapshot("before_anchor_selection")
        with managed_qwen(config, writer.stage_dir("anchor_selection"), dry_run=dry_run):
            anchors = run_anchor_selection_stage(context, target, geometry)
        runtime_diagnostics["gpu_memory_after_anchor_selection"] = _gpu_memory_snapshot("after_anchor_selection")
        anchor_diag = next((item.diagnostics for item in anchors.values() if item.diagnostics), {})
        if isinstance(anchor_diag, dict):
            llama_decision = anchor_diag.get("llama_decision")
            if isinstance(llama_decision, dict) and isinstance(llama_decision.get("vlm_server_health"), dict):
                runtime_diagnostics["vlm_server_health"] = llama_decision["vlm_server_health"]
        statuses.append(StageStatus(
            "anchor_selection",
            success=all(item.success for item in anchors.values()),
            timing_ms=round(sum(float(item.timing_ms or 0.0) for item in anchors.values()), 3),
            artifact_dir=str(writer.stage_dir("anchor_selection")),
            diagnostics={
                "gpu_memory_before_anchor_selection": runtime_diagnostics.get("gpu_memory_before_anchor_selection"),
                "gpu_memory_after_anchor_selection": runtime_diagnostics.get("gpu_memory_after_anchor_selection"),
                "anchor_selection_provenance": anchor_selection_provenance,
                "vlm_server_health": runtime_diagnostics.get("vlm_server_health"),
            },
        ))
        if not all(item.success for item in anchors.values()):
            wanted = []
    elif "anchor_selection" not in wanted:
        statuses.append(_skipped_status("anchor_selection", str(writer.stage_dir("anchor_selection"))))

    if "path_length" in wanted and target and geometry:
        path_lengths = run_local_path_length_stage(context, target, geometry, anchors)
        statuses.append(StageStatus(
            "path_length",
            success=all(item.success for item in path_lengths.values()),
            timing_ms=round(sum(float(item.timing_ms or 0.0) for item in path_lengths.values()), 3),
            artifact_dir=str(writer.stage_dir("path_length")),
        ))
        if not all(item.success for item in path_lengths.values()):
            wanted = []
    elif "path_length" not in wanted:
        statuses.append(_skipped_status("path_length", str(writer.stage_dir("path_length"))))

    if "robot_plan" in wanted and target and geometry:
        robot_plan = run_robot_planning_stage(context, target, geometry, anchors, path_lengths)
        statuses.append(_stage_status("robot_plan", robot_plan, str(writer.stage_dir("robot_plan"))))
        if not robot_plan.success:
            wanted = []
    elif "robot_plan" not in wanted:
        statuses.append(_skipped_status("robot_plan", str(writer.stage_dir("robot_plan"))))

    if "execution" in wanted and robot_plan:
        execution = run_hardware_execution_stage(context, robot_plan)
        statuses.append(_stage_status("execution", execution, str(writer.stage_dir("execution"))))
    elif "execution" not in wanted:
        statuses.append(_skipped_status("execution", str(writer.stage_dir("execution"))))

    failed = [status for status in statuses if not status.skipped and not status.success]
    result = PipelineResult(
        success=not failed,
        mode=selected_mode,
        requested_material=material,
        axis_mode=axis,
        session_dir=str(session_dir),
        capture=capture,
        target=target,
        geometry=geometry,
        anchors=anchors,
        path_lengths=path_lengths,
        robot_plan=robot_plan,
        execution=execution,
        stage_status=statuses,
        warnings=calibration_warnings,
        failure_reason=failed[0].failure_reason if failed else None,
        timing_ms=elapsed_ms(start),
        runtime_diagnostics=runtime_diagnostics,
    )
    writer.write_json("stage_status.json", [status.to_dict() for status in statuses])
    writer.write_json("timing_summary.json", {"total_timing_ms": result.timing_ms, "stages": [status.to_dict() for status in statuses], "runtime_diagnostics": runtime_diagnostics})
    writer.write_json("pipeline_result.json", result)
    write_flat_csv(session_dir / "tables" / "flat_result.csv", _flatten_result(result))
    writer.write_json("run_manifest.json", {
        "mode": selected_mode,
        "requested_material": material,
        "axis_mode": axis,
        "session_dir": str(session_dir),
        "calibration_manifest": str(calibration_manifest),
        "pre_capture_home_check": pre_capture_home_payload,
        "anchor_selection_provenance": anchor_selection_provenance,
        "runtime_diagnostics": runtime_diagnostics,
        "stage_dirs": writer.ensure_all_stage_dirs(),
    })
    return result
