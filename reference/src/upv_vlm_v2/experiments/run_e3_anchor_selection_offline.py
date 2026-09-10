"""Offline E3 contact-anchor selection evaluation from captured RGB-D scenes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Callable
from zoneinfo import ZoneInfo

from upv_vlm_v2.config.loader import load_config, snapshot_yaml
from upv_vlm_v2.data.models import json_ready
from upv_vlm_v2.experiments.e3_anchor_crop_utils import (
    build_clean_anchor_review_grid,
    render_clean_anchor_inputs_for_artifact_dir,
)
from upv_vlm_v2.experiments.run_width_offline_from_dataset import (
    _archive_case_folder,
    _copy,
    _failed_stage,
    _health_check,
    _parse_case_ids,
    _read_rows,
    _read_rows_if_exists,
    _stage_ms,
    _write_json,
    _write_rows,
)
from upv_vlm_v2.pipeline.main_pipeline import run_full_main_upv_vlm_v2_pipeline


MASTER_ANCHOR_COLUMNS = [
    "success",
    "case_index",
    "case_id",
    "material",
    "condition_type",
    "condition_notes",
    "axis_mode",
    "candidate_anchor_count",
    "candidate_anchor_ids",
    "geometry_only_selected_anchor_id",
    "geometry_only_score",
    "geometry_only_selection_source",
    "proposed_selected_anchor_id",
    "qwen_selected_anchor_id",
    "qwen_repaired",
    "final_selected_anchor_source",
    "qwen_no_safe_anchor",
    "qwen_reasoning_short",
    "target_selected_candidate_id",
    "target_selection_success",
    "anchor_selection_success",
    "rgb_path",
    "depth_path",
    "camera_info_path",
    "selected_mask_overlay_path",
    "crop_verification_panel_path",
    "major_minor_axis_overlay_path",
    "candidate_anchor_overlay_path",
    "wide_contact_grid_path",
    "clean_single_anchor_inputs_dir",
    "clean_anchor_review_grid_path",
    "anchor_crop_geometry_debug_path",
    "selected_anchor_overlay_path",
    "geometry_only_selected_anchor_overlay_path",
    "qwen_anchor_prompt_path",
    "qwen_anchor_response_raw_path",
    "qwen_anchor_decision_path",
    "final_anchor_decision_path",
    "deterministic_anchor_features_path",
    "anchor_selection_summary_path",
    "pipeline_result_path",
    "stage_status_path",
    "failed_stage",
    "failure_reason",
]


@dataclass
class E3OfflineSession:
    config: dict[str, Any]
    config_path: Path
    dataset_path: Path
    session_dir: Path


def _now_id() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _session(config_path: str | Path, dataset: str | Path, resume_session: str | Path | None = None) -> E3OfflineSession:
    config = load_config(config_path)
    session_dir = Path(resume_session) if resume_session else Path(config.get("output_root", "outputs/v2_experiments/e3_anchor_selection")) / f"session_{_now_id()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_yaml(config, session_dir / "config_snapshot.yaml")
    shutil.copy2(dataset, session_dir / "input_dataset_manifest.csv")
    return E3OfflineSession(config=config, config_path=Path(config_path), dataset_path=Path(dataset), session_dir=session_dir)


def _case_folder_name(row: dict[str, Any]) -> str:
    return f"case_{int(row['case_index']):03d}_{row['case_id']}"


def _stage_dir(result: Any, stage_name: str) -> Path | None:
    session_dir = Path(getattr(result, "session_dir", ""))
    candidates = {
        "target_selection": "02_target_selection",
        "geometry": "03_geometry",
        "anchor_selection": "04_anchor_selection",
    }
    if session_dir:
        path = session_dir / "artifacts" / candidates.get(stage_name, stage_name)
        if path.exists():
            return path
    return None


def _backend_requires_qwen(backend: str | None) -> bool:
    if not backend:
        return False
    text = str(backend).strip().lower()
    if not text:
        return False
    if "manual_prep" in text or "deterministic" in text:
        return False
    return "qwen" in text or "vlm" in text


def _read_json_if_exists(path: str | Path | None) -> Any:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact_from_stage(result: Any, stage: str, filename: str) -> str | None:
    stage_dir = _stage_dir(result, stage)
    if not stage_dir:
        return None
    path = stage_dir / filename
    return str(path) if path.exists() else None


def _candidate_ids(anchor: Any) -> list[str]:
    diag = getattr(anchor, "diagnostics", {}) or {}
    for key in ("candidate_anchor_ids", "candidate_ids", "anchor_candidate_ids"):
        values = diag.get(key)
        if isinstance(values, list):
            return [str(v) for v in values]
    features = diag.get("deterministic_features") or diag.get("features") or []
    if isinstance(features, list):
        ids = [str(item.get("anchor_id") or item.get("id")) for item in features if isinstance(item, dict) and (item.get("anchor_id") or item.get("id"))]
        if ids:
            return ids
    count = int(getattr(anchor, "candidate_count", 0) or 0)
    return [f"A{i}" for i in range(1, count + 1)]


def _feature_score(item: dict[str, Any]) -> float | None:
    for key in ("deterministic_score", "score", "anchor_score", "overall_score"):
        try:
            if item.get(key) is not None:
                return float(item[key])
        except (TypeError, ValueError):
            continue
    return None


def _extract_feature_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("anchors", "anchor_features", "candidates", "features", "deterministic_anchor_features"):
        values = payload.get(key)
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
    return []


def _geometry_only_selection(anchor: Any, deterministic_features_path: str | None) -> tuple[str | None, float | None, str]:
    payload = _read_json_if_exists(deterministic_features_path)
    features = _extract_feature_list(payload)
    diag = getattr(anchor, "diagnostics", {}) or {}
    for key in ("deterministic_selected_anchor_id", "geometry_only_selected_anchor_id"):
        selected = diag.get(key)
        if selected:
            score = None
            for item in features:
                if str(item.get("anchor_id") or item.get("id")) == str(selected):
                    score = _feature_score(item)
                    break
            return str(selected), score, key
    best_id: str | None = None
    best_score: float | None = None
    for item in features:
        anchor_id = item.get("anchor_id") or item.get("id")
        if not anchor_id:
            continue
        if bool(item.get("veto", False)):
            continue
        score = _feature_score(item)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_id = str(anchor_id)
            best_score = score
    if best_id:
        return best_id, best_score, "highest_deterministic_score_non_vetoed"
    return getattr(anchor, "final_anchor_id", None), getattr(anchor, "score", None), "fallback_final_anchor"


def _copy_artifacts(result: Any, axis: str, case_dir: Path) -> dict[str, str | None]:
    target = getattr(result, "target", None)
    geometry = getattr(result, "geometry", None)
    anchor = (getattr(result, "anchors", {}) or {}).get(axis)
    target_diag = getattr(target, "diagnostics", {}) or {}
    geometry_overlays = getattr(geometry, "overlay_paths", {}) or {}
    anchor_artifacts = getattr(anchor, "candidate_artifacts", {}) or {}
    anchor_overlays = getattr(anchor, "overlay_paths", {}) or {}
    copied = {
        "selected_mask_overlay_path": _copy(getattr(target, "selected_mask_overlay_path", None), case_dir),
        "crop_verification_panel_path": _copy(target_diag.get("crop_verification_panel_path") or _artifact_from_stage(result, "target_selection", "material_verification/crop_verification_panel.png"), case_dir),
        "major_minor_axis_overlay_path": _copy(
            geometry_overlays.get("major_minor_axis_overlay")
            or geometry_overlays.get("geometry_overlay")
            or _artifact_from_stage(result, "geometry", "major_minor_axis_overlay.png"),
            case_dir,
        ),
        "candidate_anchor_overlay_path": _copy(
            anchor_overlays.get("candidate_anchor_overlay")
            or anchor_overlays.get("anchor_overlay")
            or _artifact_from_stage(result, "anchor_selection", f"{axis}_anchor_overlay.png"),
            case_dir,
        ),
        "wide_contact_grid_path": _copy(
            anchor_artifacts.get("combined_wide_contact_guided_grid_2col")
            or anchor_artifacts.get("combined_wide_contact_guided_grid")
            or _artifact_from_stage(result, "anchor_selection", "combined_wide_contact_guided_grid_2col.png"),
            case_dir,
        ),
        "selected_anchor_overlay_path": _copy(anchor_overlays.get("selected_anchor_overlay") or _artifact_from_stage(result, "anchor_selection", "selected_anchor_overlay.png"), case_dir),
        "qwen_anchor_prompt_path": _copy(anchor_artifacts.get("qwen_anchor_prompt") or _artifact_from_stage(result, "anchor_selection", "qwen_anchor_prompt.txt"), case_dir),
        "qwen_anchor_response_raw_path": _copy(anchor_artifacts.get("qwen_anchor_response_raw") or _artifact_from_stage(result, "anchor_selection", "qwen_anchor_response_raw.txt"), case_dir),
        "qwen_anchor_decision_path": _copy(anchor_artifacts.get("qwen_anchor_decision") or _artifact_from_stage(result, "anchor_selection", "qwen_anchor_decision.json"), case_dir),
        "final_anchor_decision_path": _copy(anchor_artifacts.get("final_anchor_decision_wide_context") or _artifact_from_stage(result, "anchor_selection", "final_anchor_decision_wide_context.json"), case_dir),
        "deterministic_anchor_features_path": _copy(anchor_artifacts.get("deterministic_anchor_features_wide_context") or _artifact_from_stage(result, "anchor_selection", "deterministic_anchor_features_wide_context.json"), case_dir),
        "pipeline_result_path": _copy(Path(getattr(result, "session_dir", "")) / "pipeline_result.json", case_dir),
        "stage_status_path": _copy(Path(getattr(result, "session_dir", "")) / "stage_status.json", case_dir),
    }
    if copied.get("selected_anchor_overlay_path") and not copied.get("geometry_only_selected_anchor_overlay_path"):
        copied["geometry_only_selected_anchor_overlay_path"] = _copy(copied["selected_anchor_overlay_path"], case_dir, "geometry_only_selected_anchor_overlay.png")
    else:
        copied["geometry_only_selected_anchor_overlay_path"] = None
    anchor_stage = _stage_dir(result, "anchor_selection")
    clean_src_dir = anchor_artifacts.get("clean_single_anchor_inputs_dir")
    clean_grid_src = anchor_artifacts.get("clean_anchor_review_grid")
    clean_debug_src = anchor_artifacts.get("anchor_crop_geometry_debug")
    if clean_src_dir and Path(clean_src_dir).exists():
        clean_dir = case_dir / "clean_single_anchor_inputs"
        shutil.copytree(clean_src_dir, clean_dir, dirs_exist_ok=True)
        copied["clean_single_anchor_inputs_dir"] = str(clean_dir)
        copied["clean_anchor_review_grid_path"] = _copy(clean_grid_src, case_dir, "clean_anchor_review_grid.png")
        copied["anchor_crop_geometry_debug_path"] = _copy(clean_debug_src, case_dir, "anchor_crop_geometry_debug.json")
    elif anchor_stage:
        clean_dir = case_dir / "clean_single_anchor_inputs"
        try:
            debug = render_clean_anchor_inputs_for_artifact_dir(
                artifact_dir=anchor_stage,
                output_dir=clean_dir,
                config={},
            )
            image_paths = {
                aid: meta.get("rendered_input_path")
                for aid, meta in (debug.get("anchors") or {}).items()
                if isinstance(meta, dict) and meta.get("rendered_input_path")
            }
            grid_path = build_clean_anchor_review_grid(
                image_paths=image_paths,
                output_path=case_dir / "clean_anchor_review_grid.png",
            )
            copied["clean_single_anchor_inputs_dir"] = str(clean_dir)
            copied["clean_anchor_review_grid_path"] = grid_path
            copied["anchor_crop_geometry_debug_path"] = str(case_dir / "anchor_crop_geometry_debug.json")
        except Exception as exc:
            copied["clean_single_anchor_inputs_dir"] = str(clean_dir)
            copied["clean_anchor_review_grid_path"] = None
            copied["anchor_crop_geometry_debug_path"] = None
            copied["clean_anchor_crop_generation_error"] = repr(exc)
    else:
        copied["clean_single_anchor_inputs_dir"] = None
        copied["clean_anchor_review_grid_path"] = None
        copied["anchor_crop_geometry_debug_path"] = None
    return copied


def _row_from_result(dataset_row: dict[str, Any], result: Any, case_dir: Path, axis: str) -> dict[str, Any]:
    target = getattr(result, "target", None)
    anchor = (getattr(result, "anchors", {}) or {}).get(axis)
    anchor_diag = getattr(anchor, "diagnostics", {}) or {}
    qwen = anchor_diag.get("qwen", {}) if isinstance(anchor_diag, dict) else {}
    artifacts = _copy_artifacts(result, axis, case_dir)
    geometry_id, geometry_score, geometry_source = _geometry_only_selection(anchor, artifacts.get("deterministic_anchor_features_path"))
    candidate_ids = _candidate_ids(anchor)
    qwen_selected = anchor_diag.get("qwen_selected_anchor_id") or qwen.get("selected_anchor_id") or qwen.get("best_anchor")
    qwen_reason = qwen.get("reasoning_short") or qwen.get("best_anchor_reason") or anchor_diag.get("qwen_reasoning_short")
    summary = {
        "case_id": dataset_row.get("case_id"),
        "axis_mode": axis,
        "candidate_anchor_ids": candidate_ids,
        "geometry_only_selected_anchor_id": geometry_id,
        "geometry_only_score": geometry_score,
        "proposed_selected_anchor_id": getattr(anchor, "final_anchor_id", None),
        "qwen_selected_anchor_id": qwen_selected,
        "final_selected_anchor_source": anchor_diag.get("final_anchor_source"),
        "qwen_reasoning_short": qwen_reason,
        "artifacts": artifacts,
    }
    summary_path = case_dir / "anchor_selection_summary.json"
    _write_json(summary_path, summary)
    return {
        "success": bool(getattr(result, "success", False)),
        "case_index": dataset_row.get("case_index"),
        "case_id": dataset_row.get("case_id"),
        "material": dataset_row.get("material") or dataset_row.get("requested_material"),
        "condition_type": dataset_row.get("condition_type"),
        "condition_notes": dataset_row.get("condition_notes"),
        "axis_mode": axis,
        "candidate_anchor_count": getattr(anchor, "candidate_count", None),
        "candidate_anchor_ids": ",".join(candidate_ids),
        "geometry_only_selected_anchor_id": geometry_id,
        "geometry_only_score": geometry_score,
        "geometry_only_selection_source": geometry_source,
        "proposed_selected_anchor_id": getattr(anchor, "final_anchor_id", None),
        "qwen_selected_anchor_id": qwen_selected,
        "qwen_repaired": qwen.get("qwen_response_repaired") or anchor_diag.get("qwen_response_repaired"),
        "final_selected_anchor_source": anchor_diag.get("final_anchor_source"),
        "qwen_no_safe_anchor": str(getattr(anchor, "final_anchor_id", "")) == "NO_SAFE_ANCHOR" or qwen.get("safe_no_anchor") or qwen.get("no_safe_anchor"),
        "qwen_reasoning_short": qwen_reason,
        "target_selected_candidate_id": getattr(target, "selected_candidate_id", None),
        "target_selection_success": bool(getattr(target, "success", False)),
        "anchor_selection_success": bool(getattr(anchor, "success", False)),
        "rgb_path": dataset_row.get("rgb_path"),
        "depth_path": dataset_row.get("depth_path"),
        "camera_info_path": dataset_row.get("camera_info_path"),
        "failed_stage": _failed_stage(result),
        "failure_reason": getattr(result, "failure_reason", None),
        "anchor_selection_summary_path": str(summary_path),
        **artifacts,
    }


def _failure_row(dataset_row: dict[str, Any], exc: Exception, case_dir: Path, axis: str) -> dict[str, Any]:
    summary_path = case_dir / "anchor_selection_summary.json"
    _write_json(summary_path, {"case_id": dataset_row.get("case_id"), "axis_mode": axis, "failure_reason": str(exc)})
    return {
        "success": False,
        "case_index": dataset_row.get("case_index"),
        "case_id": dataset_row.get("case_id"),
        "material": dataset_row.get("material") or dataset_row.get("requested_material"),
        "condition_type": dataset_row.get("condition_type"),
        "condition_notes": dataset_row.get("condition_notes"),
        "axis_mode": axis,
        "rgb_path": dataset_row.get("rgb_path"),
        "depth_path": dataset_row.get("depth_path"),
        "camera_info_path": dataset_row.get("camera_info_path"),
        "failed_stage": "offline_runner",
        "failure_reason": str(exc),
        "anchor_selection_summary_path": str(summary_path),
    }


def run_offline(
    *,
    dataset: str | Path,
    config_path: str | Path,
    case_id: str | None = None,
    case_ids: str | None = None,
    resume_session: str | Path | None = None,
    replace_existing_case: bool = False,
    fail_fast: bool | None = None,
    skip_qwen_health_check: bool = False,
    pipeline_runner: Callable[..., Any] = run_full_main_upv_vlm_v2_pipeline,
    health_check: bool = True,
) -> Path:
    session = _session(config_path, dataset, resume_session)
    qwen_cfg = session.config.get("qwen") or {}
    anchor_cfg = session.config.get("anchor_selection") or {}
    backend = anchor_cfg.get("backend")
    needs_qwen = bool(qwen_cfg.get("require_server", False)) and _backend_requires_qwen(backend)
    if health_check and not skip_qwen_health_check and needs_qwen:
        _health_check(session.config)
    requested_case_ids = _parse_case_ids(case_id=case_id, case_ids=case_ids)
    rows_in = [row for row in _read_rows(dataset) if str(row.get("status", "saved")).lower() == "saved"]
    if requested_case_ids:
        wanted = set(requested_case_ids)
        rows_in = [row for row in rows_in if row.get("case_id") in wanted]
    if replace_existing_case and (not resume_session or not requested_case_ids):
        raise ValueError("--replace-existing-case requires --resume-session and --case-id/--case-ids")
    existing_rows = _read_rows_if_exists(session.session_dir / "master_anchor_results.csv") if resume_session else []
    rows = existing_rows if not replace_existing_case else [row for row in existing_rows if row.get("case_id") not in set(requested_case_ids)]
    archived: list[str] = []
    if replace_existing_case:
        print(f"Replacing existing case rows: {','.join(requested_case_ids)}")
        for item in rows_in:
            archived_path = _archive_case_folder(session.session_dir, item)
            if archived_path:
                archived.append(archived_path)
    added_count = 0
    removed_count = len(existing_rows) - len(rows)
    axis = str(session.config.get("axis_mode", "major"))
    mode = str(session.config.get("mode", "anchor_only"))
    for item in rows_in:
        case_dir = session.session_dir / "cases" / _case_folder_name(item)
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = pipeline_runner(
                config_path=session.config["main_pipeline_config"],
                requested_material=item.get("material") or item.get("requested_material"),
                axis_mode=axis,
                mode=mode,
                input_rgb=item["rgb_path"],
                input_depth=item["depth_path"],
                input_camera_info=item["camera_info_path"],
                live_ros2=False,
                execution_backend=session.config.get("execution_backend", "none"),
                output_root=str(case_dir / "pipeline_session"),
            )
            row = _row_from_result(item, result, case_dir, axis)
            _write_json(case_dir / "pipeline_result.json", result)
            _write_json(case_dir / "stage_status.json", getattr(result, "stage_status", []))
        except Exception as exc:
            if fail_fast if fail_fast is not None else bool((session.config.get("failure_policy") or {}).get("fail_fast", False)):
                raise
            row = _failure_row(item, exc, case_dir, axis)
        rows.append(row)
        added_count += 1
        _write_rows(session.session_dir / "master_anchor_results.csv", rows, MASTER_ANCHOR_COLUMNS)
    manifest = {
        "dataset": str(dataset),
        "config": str(config_path),
        "session_dir": str(session.session_dir),
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "row_count": len(rows),
        "axis_mode": axis,
        "mode": mode,
        "replace_existing_case": bool(replace_existing_case),
        "removed_old_rows": removed_count,
        "added_new_rows": added_count,
        "archived_case_dirs": archived,
        "target_selection_runtime_ms": [_stage_ms(row, "target_selection") for row in []],
    }
    _write_json(session.session_dir / "offline_run_manifest.json", manifest)
    if replace_existing_case:
        print(f"Removed old rows: {removed_count}")
        print(f"Added new rows: {added_count}")
        print(f"Final row count: {len(rows)}")
    return session.session_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 offline contact-anchor selection from a captured dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--case-ids")
    parser.add_argument("--resume-session")
    parser.add_argument("--replace-existing-case", action="store_true")
    parser.add_argument("--skip-qwen-health-check", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = run_offline(
        dataset=args.dataset,
        config_path=args.config,
        case_id=args.case_id,
        case_ids=args.case_ids,
        resume_session=args.resume_session,
        replace_existing_case=args.replace_existing_case,
        skip_qwen_health_check=args.skip_qwen_health_check,
        fail_fast=args.fail_fast,
    )
    print(f"offline_session: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
