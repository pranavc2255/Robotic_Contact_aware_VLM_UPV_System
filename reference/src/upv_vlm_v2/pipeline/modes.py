from __future__ import annotations

from enum import Enum


class PipelineMode(str, Enum):
    DRY_RUN = "dry_run"
    SNAPSHOT_ONLY = "snapshot_only"
    TARGET_ONLY = "target_only"
    GEOMETRY_ONLY = "geometry_only"
    ANCHOR_ONLY = "anchor_only"
    PATH_LENGTH_ONLY = "path_length_only"
    PLAN_ONLY = "plan_only"
    EXECUTE = "execute"


MODE_ORDER = {
    PipelineMode.SNAPSHOT_ONLY.value: ["capture"],
    PipelineMode.TARGET_ONLY.value: ["capture", "target_selection"],
    PipelineMode.GEOMETRY_ONLY.value: ["capture", "target_selection", "geometry"],
    PipelineMode.ANCHOR_ONLY.value: ["capture", "target_selection", "geometry", "anchor_selection"],
    PipelineMode.PATH_LENGTH_ONLY.value: ["capture", "target_selection", "geometry", "anchor_selection", "path_length"],
    PipelineMode.PLAN_ONLY.value: ["capture", "target_selection", "geometry", "anchor_selection", "path_length", "robot_plan"],
    PipelineMode.EXECUTE.value: ["capture", "target_selection", "geometry", "anchor_selection", "path_length", "robot_plan", "execution"],
    PipelineMode.DRY_RUN.value: ["capture", "target_selection", "geometry", "anchor_selection", "path_length", "robot_plan", "execution"],
}


def validate_mode(mode: str) -> str:
    value = mode.strip().lower()
    if value not in MODE_ORDER:
        raise ValueError(f"Unsupported v2 mode: {mode}")
    return value

