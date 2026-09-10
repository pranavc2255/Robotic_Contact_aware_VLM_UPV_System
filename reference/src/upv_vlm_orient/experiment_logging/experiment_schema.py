from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from upv_vlm_v1.prompts.prompt_registry import load_prompt_bundle


NA = "NA"


RUN_MODES = [
    "perception_only",
    "plan_only",
    "clamp_only",
    "full_autonomous",
    "full_autonomous_home",
]


PIPELINE_MODES = [
    "proposed_full_system",
    "baseline_detector_only_no_material_vlm",
    "baseline_centroid_or_image_center",
    "baseline_fixed_clamp_opening",
    "ablation_no_material_crop_vlm",
    "ablation_no_edge_crop_contact_vlm",
    "ablation_no_adaptive_clamp_width",
    "ablation_no_anchor_contact_quality_check",
]


EXPERIMENT_CATEGORIES = [
    "normal_single_object",
    "orientation_variation",
    "different_class_scene",
    "imperfection_variation",
    "baseline_comparison",
    "ablation_study",
    "repeatability",
    "UPV_reading",
    "other",
]


FAILURE_CLASSES = [
    "none",
    "perception_failure",
    "wrong_target",
    "mask_failure",
    "prompt_parse_failure",
    "vlm_failure",
    "depth_failure",
    "planning_failure",
    "workspace_failure",
    "robot_motion_failure",
    "clamp_failure",
    "UPV_reading_failure",
    "manual_abort",
    "timeout",
    "unknown",
]


METRIC_GROUPS = {
    "Perception": [
        "target_selection_correct",
        "selected_class_manual",
        "mask_usability",
        "mask_touches_boundary",
        "axis_orientation_error_deg",
        "centroid_error_mm_manual",
        "material_crop_verification_correct",
        "edge_crop_contact_quality_correct",
        "open_vocab_candidate_rank",
        "inference_time_per_model",
    ],
    "Geometry": [
        "major_axis_span_mm",
        "minor_axis_span_mm",
        "selected_probe_spacing_mm",
        "manual_width_mm",
        "width_error_mm",
        "clamp_plan_valid",
        "object_depth_m",
        "object_base_xyz",
    ],
    "Robot": [
        "move_midhover_success",
        "orient_success",
        "xy_success",
        "approach_preview_success",
        "approach_final_success",
        "final_pose_reached",
        "final_pose_error_mm_manual",
        "yaw_error_deg_manual",
        "home_success",
        "robot_motion_time_s",
    ],
    "Clamp": [
        "recommended_opening_mm",
        "total_closing_mm",
        "one_side_motion_mm",
        "estimated_steps",
        "clamp_close_success",
        "clamp_hold_success",
        "clamp_release_success",
        "manual_actual_opening_mm",
        "clamp_opening_error_mm",
    ],
    "UPV": [
        "time_of_flight_us",
        "path_length_mm",
        "velocity_m_per_s",
        "reading_quality",
        "repeated_reading_cv_percent",
    ],
    "Prompt/VLM": [
        "prompt_set_name",
        "prompt_set_version",
        "prompt_hash",
        "edited_prompt_used",
        "open_vocab_prompt_text",
        "material_verification_prompt_text",
        "edge_contact_prompt_text",
        "structured_output_prompt_text",
        "material_vlm_raw_response",
        "material_vlm_parsed_response",
        "material_vlm_parse_success",
        "edge_vlm_raw_response",
        "edge_vlm_parsed_response",
        "edge_vlm_parse_success",
        "structured_output_valid",
        "vlm_decision_used_by_pipeline",
        "vlm_stage_disabled_by_ablation",
        "prompt_engineering_notes",
    ],
    "Full task": [
        "full_autonomous_success",
        "failure_stage",
        "failure_reason_auto",
        "failure_reason_manual",
        "human_intervention_required",
        "total_wall_time_s",
        "core_pipeline_time_s",
        "visualization_logging_time_s",
        "task_mode",
        "experiment_category_tags",
    ],
}


TIMING_KEYS = [
    "total_wall_time_s",
    "camera_capture_time_s",
    "open_vocab_detection_time_s",
    "segmentation_time_s",
    "crop_generation_time_s",
    "material_vlm_time_s",
    "edge_crop_vlm_time_s",
    "prompt_preparation_time_s",
    "vlm_response_parsing_time_s",
    "mask_geometry_time_s",
    "depth_backprojection_time_s",
    "robot_pose_planning_time_s",
    "clamp_width_planning_time_s",
    "visualization_output_time_s",
    "json_csv_logging_time_s",
    "move_midhover_time_s",
    "orient_time_s",
    "xy_time_s",
    "approach_preview_time_s",
    "approach_final_time_s",
    "clamp_close_time_s",
    "clamp_hold_time_s",
    "clamp_release_time_s",
    "home_time_s",
    "total_robot_motion_time_s",
    "total_clamp_time_s",
    "total_core_pipeline_time_s",
    "total_io_visualization_time_s",
]


PROMPT_STAGES = [
    "open_vocab",
    "material_crop_verification",
    "crop_comparison",
    "edge_contact_evaluation",
    "structured_output",
]


PROMPT_FILE_NAMES = {
    "open_vocab": "open_vocab_prompt.txt",
    "material_crop_verification": "material_verification_prompt.txt",
    "crop_comparison": "crop_comparison_prompt.txt",
    "edge_contact_evaluation": "edge_contact_prompt.txt",
    "structured_output": "structured_output_prompt.txt",
}


def _canonical_prompt_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "upv_vlm_v1" / "prompts" / "default_prompts.yaml"


def _load_default_prompts() -> dict[str, str]:
    bundle = load_prompt_bundle(_canonical_prompt_path())
    return {
        "open_vocab": bundle.stages["open_vocab"].user_prompt_template,
        "material_crop_verification": bundle.stages["material_verification"].user_prompt_template,
        "crop_comparison": bundle.stages["candidate_crop_comparison"].user_prompt_template,
        "edge_contact_evaluation": bundle.stages["edge_contact"].user_prompt_template,
        "structured_output": bundle.stages["structured_output"].user_prompt_template,
    }


DEFAULT_PROMPTS = _load_default_prompts()


PROMPT_DISABLED_BY_MODE = {
    "baseline_detector_only_no_material_vlm": ["material_crop_verification", "edge_contact_evaluation", "structured_output"],
    "ablation_no_material_crop_vlm": ["material_crop_verification"],
    "ablation_no_edge_crop_contact_vlm": ["edge_contact_evaluation"],
    "ablation_no_anchor_contact_quality_check": ["edge_contact_evaluation"],
}


MODEL_STAGE_DEFAULTS = {
    "open_vocab": {"model": "repo_current_open_vocab_detector", "model_path": NA},
    "material_crop_verification": {"model": "repo_available_material_vlm_or_CLIP", "model_path": NA},
    "crop_comparison": {"model": "repo_available_crop_comparison_vlm", "model_path": NA},
    "edge_contact_evaluation": {"model": "Qwen2.5-VL or configured VLM if enabled", "model_path": "local_models/Qwen2.5-VL-3B-Instruct"},
    "structured_output": {"model": "deterministic_json_parser", "model_path": NA},
}


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def all_metric_keys() -> list[str]:
    keys: list[str] = []
    for group_keys in METRIC_GROUPS.values():
        keys.extend(group_keys)
    return keys


def na_metrics(selected_metrics: list[str] | None = None) -> dict[str, Any]:
    keys = selected_metrics or all_metric_keys()
    return {key: NA for key in keys} | {f"{key}_NA_reason": "not_available_in_this_run" for key in keys}


def default_manual_labels() -> dict[str, Any]:
    return {
        "target_selection_correct": NA,
        "requested_material_class": NA,
        "system_selected_class": NA,
        "manual_selected_class": NA,
        "wrong_failure_type": NA,
        "mask_usable": NA,
        "axis_correct": NA,
        "selected_contact_anchor_acceptable": NA,
        "visible_imperfection_near_contact": NA,
        "manual_major_dimension_mm": NA,
        "manual_minor_dimension_mm": NA,
        "manual_actual_clamp_opening_mm": NA,
        "manual_final_position_error_mm": NA,
        "manual_yaw_error_deg": NA,
        "human_intervention_required": NA,
        "intervention_stage": NA,
        "free_text_notes": "",
    }


def default_upv_results() -> dict[str, Any]:
    return {
        "upv_time_of_flight_us": NA,
        "upv_path_length_mm": NA,
        "upv_velocity_m_per_s": NA,
        "upv_reading_quality": NA,
        "upv_notes": "",
    }


def compute_upv_velocity(time_of_flight_value: float | None, units: str, path_length_mm: float | None) -> float | str:
    if time_of_flight_value is None or path_length_mm is None or time_of_flight_value <= 0 or path_length_mm <= 0:
        return NA
    tof_s = time_of_flight_value * 1e-6 if units == "microseconds" else time_of_flight_value
    return float((path_length_mm / 1000.0) / tof_s)
