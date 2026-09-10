from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def json_ready(value: Any) -> Any:
    """Convert common Python/scientific values into JSON-compatible values."""
    if is_dataclass(value):
        return {key: json_ready(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@dataclass
class SerializableResult:
    def to_dict(self) -> JsonDict:
        return json_ready(self)


@dataclass
class StageStatus(SerializableResult):
    name: str
    success: bool = False
    skipped: bool = False
    failure_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: JsonDict = field(default_factory=dict)
    timing_ms: float | None = None
    artifact_dir: str | None = None


@dataclass
class CaptureResult(SerializableResult):
    success: bool = False
    rgb_path: str | None = None
    depth_path: str | None = None
    camera_info_path: str | None = None
    depth_visualization_path: str | None = None
    frame_id: str | None = None
    timestamp: str | None = None
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class TargetSelectionResult(SerializableResult):
    success: bool = False
    requested_material: str | None = None
    selected_candidate_id: str | None = None
    selected_mask_path: str | None = None
    selected_rgb_path: str | None = None
    selected_mask_overlay_path: str | None = None
    candidate_count: int = 0
    candidate_artifacts: JsonDict = field(default_factory=dict)
    clip_scores_path: str | None = None
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class GeometryResult(SerializableResult):
    success: bool = False
    centroid_px: list[float] | None = None
    major_axis_vector: list[float] | None = None
    minor_axis_vector: list[float] | None = None
    major_axis_angle_deg: float | None = None
    minor_axis_angle_deg: float | None = None
    global_major_dimension_mm: float | None = None
    global_minor_dimension_mm: float | None = None
    overlay_paths: JsonDict = field(default_factory=dict)
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class AnchorSelectionResult(SerializableResult):
    success: bool = False
    axis_mode: str | None = None
    final_anchor_id: str | None = None
    selected_anchor_px: list[float] | None = None
    contact_point_a_px: list[float] | None = None
    contact_point_b_px: list[float] | None = None
    local_cross_axis_vector: list[float] | None = None
    score: float | None = None
    candidate_count: int = 0
    candidate_artifacts: JsonDict = field(default_factory=dict)
    overlay_paths: JsonDict = field(default_factory=dict)
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class LocalPathLengthResult(SerializableResult):
    success: bool = False
    axis_mode: str | None = None
    mask_path_length_mm: float | None = None
    depth_path_length_mm: float | None = None
    depth_valid: bool = False
    depth_failure_reason: str | None = None
    depth_mask_disagreement_ratio: float | None = None
    endpoint_a_px: list[float] | None = None
    endpoint_b_px: list[float] | None = None
    overlay_paths: JsonDict = field(default_factory=dict)
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class RobotPlanResult(SerializableResult):
    success: bool = False
    axis_mode: str | None = None
    target_pose_base: list[float] | None = None
    hover_pose_base: list[float] | None = None
    approach_pose_base: list[float] | None = None
    final_pose_base: list[float] | None = None
    clamp_opening_mm: float | None = None
    upv_path_length_mm: float | None = None
    safety_warnings: list[str] = field(default_factory=list)
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class ExecutionResult(SerializableResult):
    success: bool = False
    robot_moved: bool = False
    clamp_moved: bool = False
    upv_triggered: bool = False
    home_returned: bool = False
    execution_log_path: str | None = None
    diagnostics: JsonDict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None


@dataclass
class PipelineResult(SerializableResult):
    success: bool
    mode: str
    requested_material: str
    axis_mode: str
    session_dir: str
    capture: CaptureResult | None = None
    target: TargetSelectionResult | None = None
    geometry: GeometryResult | None = None
    anchors: dict[str, AnchorSelectionResult] = field(default_factory=dict)
    path_lengths: dict[str, LocalPathLengthResult] = field(default_factory=dict)
    robot_plan: RobotPlanResult | None = None
    execution: ExecutionResult | None = None
    stage_status: list[StageStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_ms: float | None = None
    runtime_diagnostics: JsonDict = field(default_factory=dict)
