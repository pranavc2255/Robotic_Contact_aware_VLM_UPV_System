from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NA = "NA"


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(flatten_dict(value, flat_key))
        else:
            out[flat_key] = value
    return out


def load_json(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        path = Path(path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _latest_existing(paths: list[str] | list[Path]) -> Path | None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda p: p.stat().st_mtime)[-1]


def discover_t4_files(t4_session_dir: str | Path | None) -> dict[str, Any]:
    if not t4_session_dir:
        return {}
    root = Path(t4_session_dir)
    if not root.exists():
        return {"source_t4_session_dir": str(root), "missing_expected_files": ["t4_session_dir"]}
    full_runs = sorted(root.glob("full_runs/full_*.json"))
    run_records = sorted(root.glob("runs/run_*/run_*_record.json"))
    clamp_plans = sorted([*root.glob("runs/run_*/clamp_plan/T4_clamp_plan.json"), *root.glob("runs/run_*/clamp_plan/T3_width_plan.json")])
    overlays = sorted(root.glob("**/*.png"))
    return {
        "source_t4_session_dir": str(root),
        "latest_full_run_json": str(full_runs[-1]) if full_runs else NA,
        "latest_run_record_json": str(run_records[-1]) if run_records else NA,
        "latest_clamp_plan_json": str(clamp_plans[-1]) if clamp_plans else NA,
        "latest_summary_csv": str(root / "full_runs" / "full_run_summary.csv") if (root / "full_runs" / "full_run_summary.csv").exists() else NA,
        "latest_overlay_paths": [str(path) for path in overlays],
        "missing_expected_files": [
            name
            for name, value in {
                "latest_full_run_json": full_runs,
                "latest_run_record_json": run_records,
                "latest_clamp_plan_json": clamp_plans,
            }.items()
            if not value
        ],
    }


def flatten_t4_summary(t4_summary: dict[str, Any], t4_files: dict[str, Any]) -> dict[str, Any]:
    full_run = load_json(t4_files.get("latest_full_run_json"))
    return {
        "t4_action": t4_summary.get("action", NA),
        "t4_success": t4_summary.get("success", NA),
        "t4_failure_reason": t4_summary.get("failure_reason", NA),
        "t4_session_dir": t4_summary.get("t4_session_dir", t4_files.get("source_t4_session_dir", NA)),
        "full_autonomous_success": (not bool(full_run.get("aborted"))) if full_run else t4_summary.get("success", NA),
        "abort_stage": full_run.get("abort_stage", NA),
        "abort_reason": full_run.get("abort_reason", t4_summary.get("failure_reason", NA)),
        "upv_measurement_triggered": full_run.get("measurement_triggered", False),
    }


def flatten_perception_geometry_results(t4_files: dict[str, Any]) -> dict[str, Any]:
    record = load_json(t4_files.get("latest_run_record_json"))
    clamp = load_json(t4_files.get("latest_clamp_plan_json"))
    return {
        "selected_upv_axis": clamp.get("selected_upv_axis", record.get("selected_upv_axis", NA)),
        "clamp_spacing_axis": clamp.get("clamp_spacing_axis", record.get("clamp_spacing_axis", NA)),
        "major_axis_span_mm": clamp.get("major_axis_span_mm", record.get("major_axis_span_mm", NA)),
        "minor_axis_span_mm": clamp.get("minor_axis_span_mm", record.get("minor_axis_span_mm", NA)),
        "selected_probe_spacing_mm": clamp.get("selected_probe_spacing_mm", record.get("selected_probe_spacing_mm", NA)),
        "recommended_clamp_opening_mm": clamp.get("recommended_clamp_opening_mm", record.get("recommended_clamp_opening_mm", NA)),
        "mask_projected_width_mm": clamp.get("mask_projected_width_mm", record.get("mask_projected_width_mm", NA)),
        "depth_refined_probe_spacing_mm": clamp.get("depth_refined_probe_spacing_mm", record.get("depth_refined_probe_spacing_mm", NA)),
        "depth_refined_probe_spacing_valid": clamp.get("depth_refined_probe_spacing_valid", record.get("depth_refined_probe_spacing_valid", NA)),
        "depth_refined_failure_reason": clamp.get("depth_refined_failure_reason", record.get("depth_refined_failure_reason", NA)),
        "depth_width_disagreement_ratio": clamp.get("depth_width_disagreement_ratio", record.get("depth_width_disagreement_ratio", NA)),
        "recommended_upv_path_length_mm": clamp.get("recommended_upv_path_length_mm", record.get("recommended_upv_path_length_mm", NA)),
        "upv_path_length_source": clamp.get("upv_path_length_source", record.get("upv_path_length_source", NA)),
        "total_clamp_closing_mm": clamp.get("total_clamp_closing_mm", record.get("total_clamp_closing_mm", NA)),
        "one_side_motion_mm": clamp.get("one_side_motion_mm", record.get("one_side_motion_mm", NA)),
        "estimated_clamp_steps": clamp.get("estimated_clamp_steps", record.get("estimated_clamp_steps", NA)),
        "object_depth_m": record.get("selected_depth_m", NA),
        "object_base_xyz": record.get("P_base_object", NA),
        "planned_preview_pose": record.get("planned_preview_pose", NA),
        "planned_final_pose": record.get("planned_final_pose", NA),
        "final_z_rule_used": record.get("final_z_rule_used", NA),
        "center_xy_error_to_object_xy_m": record.get("center_xy_error_to_object_xy_m", NA),
        "expected_surface_target_error_to_object_surface_m": record.get("expected_surface_target_error_to_object_surface_m", NA),
    }


def flatten_robot_results(t4_summary: dict[str, Any], t4_files: dict[str, Any]) -> dict[str, Any]:
    full_run = load_json(t4_files.get("latest_full_run_json"))
    stages = {stage.get("stage"): stage for stage in full_run.get("stages", []) if isinstance(stage, dict)}
    return {
        "robot_motion_command_sent": t4_summary.get("robot_motion_command_sent", False),
        "move_midhover_success": (stages.get("move_midhover") or {}).get("success", NA),
        "orient_success": (stages.get("orient") or {}).get("success", NA),
        "xy_success": (stages.get("xy") or {}).get("success", NA),
        "approach_preview_success": (stages.get("approach_preview") or {}).get("success", NA),
        "approach_final_success": (stages.get("approach_final") or {}).get("success", NA),
        "home_attempted": t4_summary.get("home_attempted", full_run.get("home_attempted", False)),
        "home_success": t4_summary.get("home_success", full_run.get("home_success", NA)),
        "failure_stage": full_run.get("abort_stage", NA),
        "failure_reason_auto": full_run.get("abort_reason", t4_summary.get("failure_reason", NA)),
    }


def flatten_clamp_results(t4_summary: dict[str, Any], t4_files: dict[str, Any]) -> dict[str, Any]:
    full_run = load_json(t4_files.get("latest_full_run_json"))
    clamp_plan = load_json(t4_files.get("latest_clamp_plan_json"))
    return {
        "arduino_command_sent": t4_summary.get("arduino_command_sent", False),
        "clamp_command_sent": t4_summary.get("clamp_command_sent", False),
        "selected_upv_axis": clamp_plan.get("selected_upv_axis", full_run.get("selected_upv_axis", NA)),
        "clamp_spacing_axis": clamp_plan.get("clamp_spacing_axis", full_run.get("clamp_spacing_axis", NA)),
        "selected_probe_spacing_mm": clamp_plan.get("selected_probe_spacing_mm", full_run.get("selected_probe_spacing_mm", NA)),
        "recommended_clamp_opening_mm": clamp_plan.get("recommended_clamp_opening_mm", full_run.get("recommended_clamp_opening_mm", NA)),
        "mask_projected_width_mm": clamp_plan.get("mask_projected_width_mm", full_run.get("mask_projected_width_mm", NA)),
        "depth_refined_probe_spacing_mm": clamp_plan.get("depth_refined_probe_spacing_mm", full_run.get("depth_refined_probe_spacing_mm", NA)),
        "depth_refined_probe_spacing_valid": clamp_plan.get("depth_refined_probe_spacing_valid", full_run.get("depth_refined_probe_spacing_valid", NA)),
        "depth_refined_failure_reason": clamp_plan.get("depth_refined_failure_reason", full_run.get("depth_refined_failure_reason", NA)),
        "depth_width_disagreement_ratio": clamp_plan.get("depth_width_disagreement_ratio", full_run.get("depth_width_disagreement_ratio", NA)),
        "recommended_upv_path_length_mm": clamp_plan.get("recommended_upv_path_length_mm", full_run.get("recommended_upv_path_length_mm", NA)),
        "upv_path_length_source": clamp_plan.get("upv_path_length_source", full_run.get("upv_path_length_source", NA)),
        "total_clamp_closing_mm": clamp_plan.get("total_clamp_closing_mm", full_run.get("total_clamp_closing_mm", NA)),
        "one_side_motion_mm": clamp_plan.get("one_side_motion_mm", full_run.get("one_side_motion_mm", NA)),
        "estimated_clamp_steps": clamp_plan.get("estimated_clamp_steps", full_run.get("estimated_clamp_steps", NA)),
        "clamp_close_success": (not bool(full_run.get("aborted"))) if full_run.get("command") else NA,
        "clamp_release_attempted": full_run.get("clamp_release_attempted", NA),
        "clamp_release_success": full_run.get("clamp_release_success", NA),
        "emergency_release_attempted": t4_summary.get("emergency_release_attempted", full_run.get("emergency_release_attempted", False)),
        "emergency_release_success": t4_summary.get("emergency_release_success", full_run.get("emergency_release_success", NA)),
    }


def flatten_timing_summary(t4_files: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_wall_time_s": NA,
        "total_wall_time_s_NA_reason": "timing_not_available_from_T4_log",
        "total_core_pipeline_time_s": NA,
        "total_core_pipeline_time_s_NA_reason": "timing_not_available_from_T4_log",
        "total_robot_motion_time_s": NA,
        "total_robot_motion_time_s_NA_reason": "timing_not_available_from_T4_log",
        "total_clamp_time_s": NA,
        "total_clamp_time_s_NA_reason": "timing_not_available_from_T4_log",
    }
