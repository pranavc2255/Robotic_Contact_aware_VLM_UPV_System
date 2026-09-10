from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def _now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


@dataclass
class ProgressStage:
    name: str
    label: str
    percent: int
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    success: bool | None = None
    failure_reason: str | None = None


@dataclass
class RunProgress:
    run_mode: str
    current_stage: str
    percent_complete: int
    stages: list[ProgressStage]
    status: str
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None


_TEMPLATES: dict[str, list[tuple[str, str, int]]] = {
    "perception_only": [
        ("create_run_folder", "Creating run folder", 5),
        ("capture_rgbd", "Capturing RGB-D", 25),
        ("run_perception", "Running perception", 60),
        ("save_outputs", "Saving outputs", 90),
        ("complete", "Complete", 100),
    ],
    "plan_only": [
        ("create_run_folder", "Creating run folder", 5),
        ("capture_rgbd", "Capturing RGB-D", 15),
        ("run_perception", "Running perception", 35),
        ("plan_robot_pose", "Planning robot pose", 60),
        ("plan_clamp_width", "Planning clamp width", 80),
        ("save_outputs", "Saving outputs", 95),
        ("complete", "Complete", 100),
    ],
    "clamp_only": [
        ("create_run_folder", "Creating run folder", 5),
        ("arduino_check", "Checking Arduino clamp controller", 15),
        ("clamp_close", "Closing clamp", 45),
        ("clamp_hold", "Holding clamp", 70),
        ("clamp_release", "Releasing clamp", 90),
        ("complete", "Complete", 100),
    ],
    "full_autonomous": [
        ("create_run_folder", "Creating run folder", 5),
        ("snap_capture_rgbd", "Capturing RGB-D", 10),
        ("perception_and_target_selection", "Perception and target selection", 20),
        ("robot_and_clamp_planning", "Planning robot pose and clamp", 30),
        ("move_midhover_initial_height", "Moving to mid-hover initial height", 40),
        ("orient_tool", "Orienting robot/tool", 50),
        ("xy_alignment", "XY alignment", 60),
        ("approach_preview", "Preview approach", 68),
        ("approach_final", "Final UPV-ready approach", 75),
        ("clamp_close_to_planned_width", "Closing clamp to planned width", 83),
        ("upv_hold_reading_window", "UPV hold / reading window", 90),
        ("clamp_release", "Clamp release", 98),
        ("complete", "Complete", 100),
    ],
    "full_autonomous_home": [
        ("create_run_folder", "Creating run folder", 5),
        ("snap_capture_rgbd", "Capturing RGB-D", 10),
        ("perception_and_target_selection", "Perception and target selection", 20),
        ("robot_and_clamp_planning", "Planning robot pose and clamp", 30),
        ("move_midhover_initial_height", "Moving to mid-hover initial height", 40),
        ("orient_tool", "Orienting robot/tool", 50),
        ("xy_alignment", "XY alignment", 60),
        ("approach_preview", "Preview approach", 68),
        ("approach_final", "Final UPV-ready approach", 75),
        ("clamp_close_to_planned_width", "Closing clamp to planned width", 82),
        ("upv_hold_reading_window", "UPV hold / reading window", 88),
        ("clamp_release", "Clamp release", 93),
        ("return_home", "Returning robot home", 98),
        ("complete", "Complete", 100),
    ],
    "target_selection": [
        ("create_trial", "Creating trial", 5),
        ("capture_snapshot", "Capture / snapshot", 20),
        ("target_selection_crop_verification", "Target selection / crop verification", 50),
        ("save_overlays", "Saving overlays", 80),
        ("complete", "Complete", 100),
    ],
    "geometry_axis": [
        ("create_trial", "Creating trial", 5),
        ("capture_snapshot", "Capture / snapshot", 25),
        ("mask_geometry", "Mask geometry", 50),
        ("axis_estimation", "Axis estimation", 75),
        ("complete", "Complete", 100),
    ],
    "anchor_selection": [
        ("create_trial", "Creating trial", 5),
        ("capture_snapshot", "Capture / snapshot", 20),
        ("candidate_anchors", "Candidate anchors", 40),
        ("vlm_contact_evaluation", "VLM/contact evaluation", 70),
        ("save_overlays", "Saving overlays", 90),
        ("complete", "Complete", 100),
    ],
    "path_length": [
        ("create_trial", "Creating trial", 5),
        ("capture_snapshot", "Capture / snapshot", 20),
        ("mask_width", "Mask width", 40),
        ("depth_refined_width", "Depth-refined width", 70),
        ("save_overlay", "Saving overlay", 90),
        ("complete", "Complete", 100),
    ],
    "contact_ablation": [
        ("create_trial", "Creating trial", 5),
        ("load_saved_anchor_data", "Loading saved anchor-selection data", 25),
        ("run_ablation_replay", "Running contact ablation replay", 60),
        ("save_outputs", "Saving ablation outputs", 90),
        ("complete", "Complete", 100),
    ],
    "end_to_end_upv": [
        ("create_run_folder", "Creating run folder", 5),
        ("snap_capture_rgbd", "Capturing RGB-D", 10),
        ("perception_and_target_selection", "Target selection", 20),
        ("robot_and_clamp_planning", "Robot/clamp planning", 30),
        ("move_midhover_initial_height", "Moving mid-hover", 40),
        ("orient_tool", "Orienting tool", 50),
        ("xy_alignment", "XY alignment", 60),
        ("approach_preview", "Preview approach", 68),
        ("approach_final", "Final approach", 75),
        ("clamp_close_to_planned_width", "Clamp close", 82),
        ("upv_hold_reading_window", "UPV hold / reading window", 88),
        ("clamp_release", "Clamp release", 93),
        ("return_home", "Return home", 98),
        ("complete", "Complete", 100),
    ],
}


def make_progress_template(run_mode: str) -> RunProgress:
    template = _TEMPLATES.get(run_mode) or [
        ("create_run_folder", "Creating run folder", 5),
        ("complete", "Complete", 100),
    ]
    now = _now_iso()
    return RunProgress(
        run_mode=run_mode,
        current_stage="waiting",
        percent_complete=0,
        stages=[ProgressStage(name=name, label=label, percent=percent) for name, label, percent in template],
        status="ready",
        started_at=now,
        updated_at=now,
        finished_at=None,
    )


def _coerce_progress(progress: RunProgress | dict[str, Any]) -> RunProgress:
    if isinstance(progress, RunProgress):
        return progress
    stages = [
        ProgressStage(
            name=str(stage.get("name", "")),
            label=str(stage.get("label", stage.get("name", ""))),
            percent=int(stage.get("percent", 0)),
            status=str(stage.get("status", "pending")),
            started_at=stage.get("started_at"),
            finished_at=stage.get("finished_at"),
            success=stage.get("success"),
            failure_reason=stage.get("failure_reason"),
        )
        for stage in progress.get("stages", [])
        if isinstance(stage, dict)
    ]
    return RunProgress(
        run_mode=str(progress.get("run_mode", "unknown")),
        current_stage=str(progress.get("current_stage", "waiting")),
        percent_complete=int(progress.get("percent_complete", 0)),
        stages=stages,
        status=str(progress.get("status", "ready")),
        started_at=progress.get("started_at"),
        updated_at=progress.get("updated_at"),
        finished_at=progress.get("finished_at"),
    )


def update_stage(
    progress: RunProgress | dict[str, Any],
    stage_name: str,
    status: str,
    success: bool | None = None,
    failure_reason: str | None = None,
) -> RunProgress:
    out = _coerce_progress(progress)
    now = _now_iso()
    match = next((stage for stage in out.stages if stage.name == stage_name), None)
    if match is None:
        match = ProgressStage(name=stage_name, label=stage_name.replace("_", " "), percent=out.percent_complete)
        out.stages.append(match)
    if status == "running" and match.started_at is None:
        match.started_at = now
    if status in {"succeeded", "failed", "skipped"}:
        match.finished_at = now
    match.status = status
    match.success = success
    match.failure_reason = failure_reason
    out.current_stage = stage_name
    out.percent_complete = int(match.percent)
    out.updated_at = now
    if status == "failed":
        out.status = "failed"
        out.finished_at = now
    elif stage_name == "complete" and status in {"succeeded", "complete"}:
        match.status = "succeeded"
        match.success = True
        match.finished_at = now
        out.status = "complete"
        out.percent_complete = 100
        out.finished_at = now
    elif status == "running":
        out.status = "running"
    elif out.status not in {"failed", "complete"}:
        out.status = "running" if stage_name != "complete" else out.status
    return out


def progress_to_dict(progress: RunProgress | dict[str, Any]) -> dict[str, Any]:
    return asdict(_coerce_progress(progress))


def write_progress(progress: RunProgress | dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress_to_dict(progress), indent=2), encoding="utf-8")


def read_progress(path: Path) -> RunProgress | dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _coerce_progress(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return {}


def failed_stage(progress: RunProgress | dict[str, Any]) -> str | None:
    out = _coerce_progress(progress)
    for stage in out.stages:
        if stage.status == "failed" or stage.success is False:
            return stage.name
    return None


def write_progress_summary_csv(progress: RunProgress | dict[str, Any], path: Path) -> None:
    out = _coerce_progress(progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_mode", "stage", "label", "percent", "status", "success", "failure_reason", "started_at", "finished_at"],
        )
        writer.writeheader()
        for stage in out.stages:
            writer.writerow(
                {
                    "run_mode": out.run_mode,
                    "stage": stage.name,
                    "label": stage.label,
                    "percent": stage.percent,
                    "status": stage.status,
                    "success": stage.success,
                    "failure_reason": stage.failure_reason,
                    "started_at": stage.started_at,
                    "finished_at": stage.finished_at,
                }
            )
