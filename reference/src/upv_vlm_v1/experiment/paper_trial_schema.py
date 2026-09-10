from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


EXPERIMENT_GROUPS = ["target_selection", "geometry_axis", "anchor_selection", "path_length", "contact_ablation", "end_to_end_upv"]


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def slug_text(text: Any) -> str:
    value = str(text or "NA").strip().lower()
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).split("_"))


@dataclass
class PaperTrialMetadata:
    trial_id: str
    experiment_group: str
    material_query: str
    specimen_id: str
    scene_id: str
    trial_index: int
    axis_mode: str
    edge_condition: str = "clean"
    orientation_case: str = "0"
    objects_present: str = "NA"
    notes: str = ""
    operator_name: str = "Pranav"
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperTrialResult:
    success: bool
    failure_stage: str = "NA"
    failure_reason: str = "NA"
    backend_run_dir: str = "NA"
    linked_e1_or_t4_dir: str = "NA"
    key_artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    manual_ground_truth: dict[str, Any] = field(default_factory=dict)
    upv_reading: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_manual_ground_truth(experiment_group: str) -> dict[str, Any]:
    common = {
        "saved_at": now_iso(),
        "manual_notes": "",
    }
    if experiment_group == "target_selection":
        common.update(
            {
                "target_correct": "NA",
                "manual_selected_class": "NA",
                "objects_present": "NA",
                "failure_type": "NA",
            }
        )
    elif experiment_group == "geometry_axis":
        common.update(
            {
                "manual_major_axis_angle_deg": "NA",
                "manual_centroid_u": "NA",
                "manual_centroid_v": "NA",
                "mask_usable": "NA",
                "axis_correct": "NA",
            }
        )
    elif experiment_group == "anchor_selection":
        common.update(
            {
                "selected_anchor_label": "NA",
                "manual_best_anchor_id": "NA",
                "imperfection_present": "NA",
                "bad_anchor_reason": "NA",
            }
        )
    elif experiment_group == "path_length":
        common.update(
            {
                "manual_clamp_width_mm": "NA",
                "manual_upv_path_length_mm": "NA",
                "measurement_tool": "NA",
                "measurement_notes": "NA",
            }
        )
    elif experiment_group == "contact_ablation":
        common.update(
            {
                "source_anchor_trial_id": "NA",
                "ablation_condition": "NA",
                "anchor_acceptable": "NA",
                "manual_best_anchor_id": "NA",
                "ablation_notes": "NA",
            }
        )
    elif experiment_group == "end_to_end_upv":
        common.update(
            {
                "target_correct": "NA",
                "anchor_acceptable": "NA",
                "robot_success": "NA",
                "clamp_success": "NA",
                "valid_upv_tof": "NA",
                "returned_home": "NA",
                "human_intervention_count": "NA",
                "failure_stage": "NA",
                "time_of_flight_value": "NA",
                "time_of_flight_units": "microseconds",
                "path_length_source": "recommended_upv_path_length",
                "manual_path_length_mm": "NA",
                "reading_quality": "NA",
                "computed_velocity_m_per_s": "NA",
            }
        )
    return common


def flatten_trial_row(metadata: dict[str, Any], result: dict[str, Any], manual: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    upv = result.get("upv_reading") if isinstance(result.get("upv_reading"), dict) else {}
    return {
        **metadata,
        "success": result.get("success", "NA"),
        "failure_stage": result.get("failure_stage", "NA"),
        "failure_reason": result.get("failure_reason", "NA"),
        "backend_run_dir": result.get("backend_run_dir", "NA"),
        "linked_e1_or_t4_dir": result.get("linked_e1_or_t4_dir", "NA"),
        **{f"metric_{k}": v for k, v in metrics.items()},
        **{f"timing_{k}": v for k, v in timing.items()},
        **{f"manual_{k}": v for k, v in (manual or {}).items()},
        **{f"upv_{k}": v for k, v in upv.items()},
    }
