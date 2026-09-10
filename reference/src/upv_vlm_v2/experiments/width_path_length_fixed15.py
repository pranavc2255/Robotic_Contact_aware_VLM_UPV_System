"""E2 width / UPV path-length validation wrapper for UPV_VLM_v2.

This wrapper calls the v2 main pipeline in `path_length_only` mode and records
manual measurements, errors, timings, and paper-ready tables/figures. It does
not duplicate target selection, Qwen anchor selection, or robot planning logic.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw

from upv_vlm_v2.config.loader import load_config, snapshot_yaml
from upv_vlm_v2.geometry.local_chord import compute_chord_endpoints, compute_local_chord
from upv_vlm_v2.geometry.mask_geometry import (
    compute_pixel_geometry,
    load_camera_intrinsics,
    load_depth_array,
    load_mask,
    normalize_vector,
    pixel_span_to_mm,
    representative_depth_m,
)
from upv_vlm_v2.pipeline.main_pipeline import run_full_main_upv_vlm_v2_pipeline


MASTER_COLUMNS = [
    "session_id", "case_index", "case_id", "material", "object_instance_id", "axis_mode",
    "axis_order_for_case", "measurement_row_index", "attempt_index", "timestamp",
    "requested_material", "selected_material", "selected_candidate_id", "target_selected",
    "centroid_x_px", "centroid_y_px", "major_axis_deg", "minor_axis_deg",
    "anchor_backend", "qwen_required", "qwen_selected_anchor_id", "final_selected_anchor_id",
    "final_anchor_source", "robot_candidate_anchor_center_x_px", "robot_candidate_anchor_center_y_px",
    "contact_a_x_px", "contact_a_y_px", "contact_b_x_px", "contact_b_y_px", "anchor_score",
    "local_mask_path_length_mm", "local_depth_path_length_mm", "local_depth_valid",
    "local_depth_valid_point_count", "local_depth_mask_disagreement_ratio",
    "depth_validity_policy", "depth_warnings", "depth_width_was_computed",
    "depth_width_used_for_clamp", "path_length_source",
    "local_mask_compute_ms", "local_depth_compute_ms", "qwen_inference_ms",
    "anchor_selection_ms", "path_length_ms", "full_pipeline_total_ms",
    "pointcloud_downsample_mode", "pointcloud_stride", "max_points", "min_valid_depth_points",
    "local_depth_band_px",
    "depth_width_method", "depth_scale_m_per_unit", "depth_scale_source",
    "raw_band_pixel_count", "valid_depth_point_count_before_z_filter",
    "valid_depth_point_count_after_z_filter", "used_point_count_after_downsampling",
    "rejected_point_count", "depth_median_m", "depth_mad_m",
    "span_mm_pixel_percentile_2_98", "span_mm_3d_minmax",
    "span_mm_3d_percentile_2_98", "span_mm_3d_edge_bin",
    "edge_bin_a_count", "edge_bin_b_count",
    "edge_bin_width_px_used_a", "edge_bin_width_px_used_b",
    "manual_local_path_length_mm",
    "manual_measurement_tool", "manual_notes", "mask_usable", "depth_usable",
    "local_mask_abs_error_mm", "local_mask_pct_error", "local_depth_abs_error_mm", "local_depth_pct_error",
    "pipeline_session_dir", "shared_capture_dir", "selected_mask_overlay_path", "rotated_object_overlay_path",
    "cropped_rotated_object_overlay_path", "wide_contact_grid_path", "qwen_response_path",
    "robot_anchor_geometry_path", "selected_anchor_overlay_path", "local_chord_overlay_path",
    "depth_used_points_overlay_path", "depth_edge_bins_overlay_path",
    "depth_projection_histogram_path", "depth_local_pointcloud_npz_path",
    "depth_local_pointcloud_ply_path", "depth_width_diagnostics_path",
    "trial_result_json_path",
]

ATTEMPT_COLUMNS = MASTER_COLUMNS + ["attempt_status", "failure_reason"]


@dataclass
class ExperimentSession:
    config: dict[str, Any]
    config_path: Path
    session_dir: Path
    session_id: str
    cases: list[dict[str, str]]


def _now_id() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _load_cases(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def _append_or_write(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in columns})


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _abs_err(value: float | None, manual: float | None) -> float | None:
    if value is None or manual is None:
        return None
    return abs(float(value) - float(manual))


def _pct_err(value: float | None, manual: float | None) -> float | None:
    if value is None or manual in (None, 0):
        return None
    return 100.0 * abs(float(value) - float(manual)) / abs(float(manual))


def _stage_ms(result: Any, *names: str) -> float | None:
    total = 0.0
    found = False
    for status in getattr(result, "stage_status", []) or []:
        if status.name in names and status.timing_ms is not None:
            total += float(status.timing_ms)
            found = True
    return total if found else None


def _qwen_ms(anchor_diag: dict[str, Any] | None) -> float | None:
    if not isinstance(anchor_diag, dict):
        return None
    qwen = anchor_diag.get("qwen_decision") or {}
    for key in ("timing_ms", "elapsed_ms", "inference_ms", "qwen_inference_ms"):
        if isinstance(qwen, dict) and qwen.get(key) is not None:
            return float(qwen[key])
    return None


def _material_for_pipeline(material: str) -> str:
    return "concrete block" if material == "concrete_block" else material


def _prepare_session(config_path: str | Path) -> ExperimentSession:
    config_path = Path(config_path)
    config = load_config(config_path)
    exp_cfg = config["experiment"]
    session_id = f"session_{_now_id()}"
    session_dir = Path(exp_cfg["output_root"]) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_yaml(config, session_dir / "config_snapshot.yaml")
    cases_path = Path(exp_cfg["cases_csv"])
    shutil.copy2(cases_path, session_dir / "cases_snapshot.csv")
    cases = _load_cases(cases_path)
    manifest = {
        "experiment": exp_cfg.get("name", "width_path_length_fixed15"),
        "session_id": session_id,
        "created_at": _timestamp(),
        "no_robot": True,
        "no_clamp": True,
        "no_rtde": True,
        "no_arduino": True,
        "case_count": len(cases),
    }
    (session_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ExperimentSession(config=config, config_path=config_path, session_dir=session_dir, session_id=session_id, cases=cases)


def _save_simple_overlay(path: Path, width: int = 640, height: int = 480, text: str = "synthetic overlay") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (235, 238, 232))
    draw = ImageDraw.Draw(img)
    draw.text((24, 24), text, fill=(20, 25, 28))
    draw.line((120, height // 2, width - 120, height // 2), fill=(255, 50, 50), width=4)
    draw.line((width // 2, 120, width // 2, height - 120), fill=(50, 90, 255), width=4)
    img.save(path)
    return str(path)


def _measurement_config(config: dict[str, Any]) -> dict[str, Any]:
    depth = config.get("depth_processing", {})
    return {
        "path_length": {
            **depth,
            "min_top_face_points": int(depth.get("min_valid_depth_points", depth.get("min_top_face_points", 1000))),
            "use_true_3d_local_pointcloud_width": True,
            "final_depth_width_method": depth.get("final_depth_width_method", "edge_bin_3d_endpoint_distance"),
        }
    }


def _create_case_pipeline_config(session: ExperimentSession, attempt_root: Path) -> Path:
    main_config = load_config(session.config["experiment"]["main_pipeline_config"])
    merged = dict(main_config.get("path_length", {}))
    depth_processing = dict(session.config.get("depth_processing", {}))
    merged.update(depth_processing)
    merged["min_top_face_points"] = int(depth_processing.get("min_valid_depth_points", merged.get("min_top_face_points", 1000)))
    merged["use_true_3d_local_pointcloud_width"] = True
    merged["final_depth_width_method"] = depth_processing.get("final_depth_width_method", "edge_bin_3d_endpoint_distance")
    main_config["path_length"] = merged
    anchor_overrides = dict(session.config.get("anchor_selection", {}))
    if anchor_overrides:
        anchor_cfg = dict(main_config.get("anchor_selection", {}))
        anchor_cfg.update(anchor_overrides)
        main_config["anchor_selection"] = anchor_cfg
    output = attempt_root / "e2_pipeline_config_with_depth_processing.yaml"
    snapshot_yaml(main_config, output)
    return output


def _print_pipeline_failure_details(result: Any, axis_mode: str) -> None:
    failed_stage = None
    for stage in getattr(result, "stage_status", []) or []:
        if not getattr(stage, "success", True) and not getattr(stage, "skipped", False):
            failed_stage = stage
            break
    print(f"Pipeline failed for axis_mode={axis_mode}:")
    print(f"  failed_stage: {getattr(failed_stage, 'name', 'unknown')}")
    print(f"  failure_reason: {getattr(result, 'failure_reason', None) or getattr(failed_stage, 'failure_reason', None)}")
    session_dir = Path(getattr(result, "session_dir", ""))
    anchor_dir = session_dir / "artifacts" / "04_anchor_selection"
    summary_path = anchor_dir / f"anchor_selection_{axis_mode}_summary.json"
    qwen_decision_path = anchor_dir / "qwen_anchor_decision.json"
    parsed_path = anchor_dir / "qwen_anchor_response_parsed.json"
    raw_path = anchor_dir / "qwen_anchor_response_raw.txt"
    prompt_path = anchor_dir / "qwen_anchor_prompt.txt"
    overlay_candidates = [anchor_dir / "selected_anchor_overlay.png", anchor_dir / f"{axis_mode}_selected_anchor_overlay.png", anchor_dir / "no_safe_anchor_overlay.png"]
    for path in (summary_path, qwen_decision_path):
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            diag = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
            qwen = diag.get("qwen", payload) if isinstance(diag, dict) else payload
            print(f"  anchor_summary: {path}")
            print(f"  qwen_failure_reason: {qwen.get('failure_reason') if isinstance(qwen, dict) else None}")
            print(f"  qwen_selected_anchor: {qwen.get('selected_anchor_id') if isinstance(qwen, dict) else None}")
            print(f"  missing_assessment_ids: {qwen.get('qwen_missing_assessment_ids') or qwen.get('missing_required_contact_flags') if isinstance(qwen, dict) else None}")
            print(f"  selected_anchor_before_repair: {qwen.get('selected_anchor_before_repair') if isinstance(qwen, dict) else None}")
            print(f"  selected_anchor_after_repair: {qwen.get('selected_anchor_after_repair') if isinstance(qwen, dict) else None}")
            break
    print(f"  qwen_prompt: {prompt_path if prompt_path.exists() else 'NA'}")
    print(f"  qwen_response_raw: {raw_path if raw_path.exists() else 'NA'}")
    print(f"  qwen_response_parsed: {parsed_path if parsed_path.exists() else 'NA'}")
    overlay = next((path for path in overlay_candidates if path.exists()), None)
    print(f"  anchor_overlay: {overlay or 'NA'}")


def compute_global_measurements(
    *,
    mask_path: str | Path,
    depth_path: str | Path | None,
    camera_info_path: str | Path | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    mask = load_mask(mask_path)
    depth = load_depth_array(depth_path)
    intrinsics = load_camera_intrinsics(camera_info_path)
    depth_cfg = config.get("depth_processing", {})
    stride = max(1, int(depth_cfg.get("pointcloud_stride", 1) or 1))
    max_points = depth_cfg.get("max_points")
    min_points = int(depth_cfg.get("min_valid_depth_points", 1000))

    t0 = time.perf_counter()
    geom = compute_pixel_geometry(mask)
    z_m = representative_depth_m(mask, depth)
    global_mask_major = pixel_span_to_mm(geom["major_axis_length_px"], geom["major_axis_vector"], z_m, intrinsics)
    global_mask_minor = pixel_span_to_mm(geom["minor_axis_length_px"], geom["minor_axis_vector"], z_m, intrinsics)
    mask_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    depth_major = None
    depth_minor = None
    valid_count = 0
    if depth is not None and intrinsics is not None:
        yy, xx = np.nonzero(mask > 0)
        xx = xx[::stride]
        yy = yy[::stride]
        if max_points is not None and len(xx) > int(max_points):
            xx = xx[: int(max_points)]
            yy = yy[: int(max_points)]
        d = depth.astype("float32")
        values: list[tuple[float, float, float]] = []
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        for x, y in zip(xx, yy):
            z = float(d[int(y), int(x)])
            if not np.isfinite(z) or z <= 0:
                continue
            z_mi = z / 1000.0 if z > 20.0 else z
            values.append(((float(x) - cx) * z_mi / fx, (float(y) - cy) * z_mi / fy, z_mi))
        valid_count = len(values)
        if valid_count >= min_points:
            pts = np.asarray(values, dtype=float)
            for key, axis in [("major", geom["major_axis_vector"]), ("minor", geom["minor_axis_vector"])]:
                ax, ay = normalize_vector(axis)
                proj = pts[:, 0] * ax + pts[:, 1] * ay
                lo, hi = np.percentile(proj, [2, 98])
                span_mm = float((hi - lo) * 1000.0)
                if key == "major":
                    depth_major = span_mm
                else:
                    depth_minor = span_mm
    depth_ms = (time.perf_counter() - t1) * 1000.0
    return {
        "geometry": geom,
        "global_mask_major_mm": global_mask_major,
        "global_mask_minor_mm": global_mask_minor,
        "global_mask_compute_ms": mask_ms,
        "global_depth_major_mm": depth_major,
        "global_depth_minor_mm": depth_minor,
        "global_depth_compute_ms": depth_ms,
        "global_depth_major_valid": depth_major is not None,
        "global_depth_minor_valid": depth_minor is not None,
        "global_depth_valid_point_count": valid_count,
        "pointcloud_stride": stride,
        "max_points": max_points,
        "min_valid_depth_points": min_points,
    }


def compute_local_measurements(
    *,
    mask_path: str | Path,
    rgb_path: str | Path,
    depth_path: str | Path | None,
    camera_info_path: str | Path | None,
    anchor_px: list[float],
    direction_xy: list[float],
    axis_mode: str,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    mask = load_mask(mask_path)
    depth = load_depth_array(depth_path)
    intrinsics = load_camera_intrinsics(camera_info_path)

    t0 = time.perf_counter()
    chord = compute_chord_endpoints(mask, anchor_px, direction_xy)
    z_m = representative_depth_m(mask, depth)
    local_mask = pixel_span_to_mm(chord.get("length_px", 0.0), chord.get("direction_xy", direction_xy), z_m, intrinsics) if chord.get("valid") else None
    mask_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    result = compute_local_chord(
        mask_path=mask_path,
        rgb_path=rgb_path,
        anchor_px=anchor_px,
        direction_xy=direction_xy,
        output_dir=output_dir,
        axis_mode=axis_mode,
        depth_path=depth_path,
        camera_info_path=camera_info_path,
        config=_measurement_config(config),
    )
    depth_ms = (time.perf_counter() - t1) * 1000.0
    return {
        "local_mask_path_length_mm": local_mask if local_mask is not None else result.get("mask_path_length_mm"),
        "local_depth_path_length_mm": result.get("depth_path_length_mm"),
        "local_mask_compute_ms": mask_ms,
        "local_depth_compute_ms": depth_ms,
        "local_depth_valid": bool(result.get("depth_valid", False)),
        "local_depth_valid_point_count": (result.get("depth_width_diagnostics") or result.get("depth_diagnostics") or {}).get("used_point_count_after_downsampling"),
        "local_depth_mask_disagreement_ratio": result.get("depth_mask_disagreement_ratio"),
        "depth_validity_policy": (result.get("depth_width_diagnostics") or result.get("depth_diagnostics") or {}).get("depth_validity_policy"),
        "depth_warnings": "; ".join((result.get("depth_width_diagnostics") or result.get("depth_diagnostics") or {}).get("depth_warnings", []) or []),
        "depth_width_was_computed": bool((result.get("depth_width_diagnostics") or result.get("depth_diagnostics") or {}).get("edge_bin_width_computed", False)),
        "depth_width_used_for_clamp": bool(result.get("depth_path_length_mm") is not None),
        "path_length_source": result.get("path_length_source"),
        "local_chord_overlay_path": result.get("overlay_path"),
        "depth_width_diagnostics": result.get("depth_width_diagnostics") or {},
        "depth_artifacts": result.get("depth_artifacts") or {},
    }


def _copy_artifact(src: str | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    p = Path(src)
    if not p.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / p.name
    if p.resolve() == dst.resolve():
        return str(dst)
    if dst.exists() and p.resolve() == dst.resolve():
        return str(dst)
    shutil.copy2(p, dst)
    return str(dst)


def _extract_row_from_pipeline(
    *,
    session: ExperimentSession,
    case: dict[str, str],
    axis_mode: str,
    axis_order_for_case: int,
    attempt_index: int,
    measurement_row_index: int | None,
    pipeline_result: Any,
    attempt_dir: Path,
    shared_capture_dir: Path | None,
    manual: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    target = pipeline_result.target
    geometry = pipeline_result.geometry
    anchor = (pipeline_result.anchors or {}).get(axis_mode)
    path_length = (pipeline_result.path_lengths or {}).get(axis_mode)
    capture = pipeline_result.capture
    anchor_diag = anchor.diagnostics if anchor else {}
    robot_geom = anchor_diag.get("robot_anchor_geometry") if isinstance(anchor_diag, dict) else {}
    artifacts = anchor.candidate_artifacts if anchor else {}
    depth_cfg = session.config.get("depth_processing", {})
    copied_dir = attempt_dir / "copied_pipeline_artifacts"
    overlays_dir = attempt_dir / "overlays"

    global_meas = compute_global_measurements(
        mask_path=target.selected_mask_path,
        depth_path=capture.depth_path if capture else None,
        camera_info_path=capture.camera_info_path if capture else None,
        config=session.config,
    )
    local_meas = compute_local_measurements(
        mask_path=target.selected_mask_path,
        rgb_path=target.selected_rgb_path,
        depth_path=capture.depth_path if capture else None,
        camera_info_path=capture.camera_info_path if capture else None,
        anchor_px=anchor.selected_anchor_px,
        direction_xy=anchor.local_cross_axis_vector,
        axis_mode=axis_mode,
        output_dir=overlays_dir,
        config=session.config,
    )
    depth_diag = local_meas.get("depth_width_diagnostics") if isinstance(local_meas.get("depth_width_diagnostics"), dict) else {}
    depth_artifacts = local_meas.get("depth_artifacts") if isinstance(local_meas.get("depth_artifacts"), dict) else {}
    edge_bin = depth_diag.get("edge_bin", {}) if isinstance(depth_diag, dict) else {}

    global_overlay = _save_simple_overlay(overlays_dir / "global_mask_dimension_overlay.png", text="global mask major/minor")
    depth_overlay = _save_simple_overlay(overlays_dir / "global_depth_dimension_overlay.png", text="global depth major/minor")
    local_overlay = _copy_artifact(local_meas.get("local_chord_overlay_path"), overlays_dir)
    depth_used_overlay = _copy_artifact(depth_artifacts.get("depth_used_points_overlay"), overlays_dir)
    depth_edge_overlay = _copy_artifact(depth_artifacts.get("depth_edge_bins_overlay"), overlays_dir)
    depth_hist = _copy_artifact(depth_artifacts.get("depth_projection_histogram"), overlays_dir)
    depth_npz = _copy_artifact(depth_artifacts.get("depth_local_pointcloud_npz"), copied_dir)
    depth_ply = _copy_artifact(depth_artifacts.get("depth_local_pointcloud_ply"), copied_dir)
    depth_diag_path = _copy_artifact(depth_artifacts.get("depth_width_diagnostics"), copied_dir)
    for key in ("depth_local_band_mask", "depth_valid_points_mask", "depth_rejected_points_mask"):
        _copy_artifact(depth_artifacts.get(key), overlays_dir)
    _copy_artifact(target.selected_mask_overlay_path, overlays_dir)
    _copy_artifact(anchor.overlay_paths.get("selected_anchor_overlay") if anchor else None, overlays_dir)
    _copy_artifact(artifacts.get("combined_wide_contact_guided_grid_2col"), overlays_dir)
    _copy_artifact(artifacts.get("rotated_object_overlay"), overlays_dir)
    _copy_artifact(artifacts.get("cropped_rotated_object_overlay"), overlays_dir)

    row: dict[str, Any] = {
        "session_id": session.session_id,
        "case_index": case["case_index"],
        "case_id": case["case_id"],
        "material": case["material"],
        "object_instance_id": case["object_instance_id"],
        "axis_mode": axis_mode,
        "axis_order_for_case": axis_order_for_case,
        "attempt_index": attempt_index,
        "measurement_row_index": measurement_row_index,
        "timestamp": _timestamp(),
        "requested_material": _material_for_pipeline(case["material"]),
        "selected_material": target.requested_material if target else None,
        "selected_candidate_id": target.selected_candidate_id if target else None,
        "target_selected": bool(target and target.success),
        "centroid_x_px": (geometry.centroid_px or [None, None])[0] if geometry else None,
        "centroid_y_px": (geometry.centroid_px or [None, None])[1] if geometry else None,
        "major_axis_deg": geometry.major_axis_angle_deg if geometry else None,
        "minor_axis_deg": geometry.minor_axis_angle_deg if geometry else None,
        "anchor_backend": anchor_diag.get("anchor_backend") if isinstance(anchor_diag, dict) else None,
        "qwen_required": anchor_diag.get("qwen_required") if isinstance(anchor_diag, dict) else None,
        "qwen_selected_anchor_id": anchor_diag.get("qwen_selected_anchor_id") if isinstance(anchor_diag, dict) else None,
        "final_selected_anchor_id": anchor.final_anchor_id if anchor else None,
        "final_anchor_source": anchor_diag.get("final_anchor_source") if isinstance(anchor_diag, dict) else None,
        "robot_candidate_anchor_center_x_px": (robot_geom.get("robot_candidate_anchor_center_px") or [None, None])[0] if isinstance(robot_geom, dict) else None,
        "robot_candidate_anchor_center_y_px": (robot_geom.get("robot_candidate_anchor_center_px") or [None, None])[1] if isinstance(robot_geom, dict) else None,
        "contact_a_x_px": (anchor.contact_point_a_px or [None, None])[0] if anchor else None,
        "contact_a_y_px": (anchor.contact_point_a_px or [None, None])[1] if anchor else None,
        "contact_b_x_px": (anchor.contact_point_b_px or [None, None])[0] if anchor else None,
        "contact_b_y_px": (anchor.contact_point_b_px or [None, None])[1] if anchor else None,
        "anchor_score": anchor.score if anchor else None,
        **{k: v for k, v in global_meas.items() if k != "geometry"},
        **local_meas,
        "pointcloud_downsample_mode": depth_cfg.get("pointcloud_downsample_mode"),
        "local_depth_band_px": depth_cfg.get("local_depth_band_px"),
        "depth_validity_policy": local_meas.get("depth_validity_policy") or depth_diag.get("depth_validity_policy"),
        "depth_warnings": local_meas.get("depth_warnings") or "; ".join(depth_diag.get("depth_warnings", []) or []),
        "depth_width_was_computed": local_meas.get("depth_width_was_computed") if local_meas.get("depth_width_was_computed") is not None else depth_diag.get("edge_bin_width_computed"),
        "depth_width_used_for_clamp": local_meas.get("depth_width_used_for_clamp"),
        "path_length_source": local_meas.get("path_length_source") or ("depth_edge_bin_3d" if local_meas.get("local_depth_path_length_mm") is not None else "mask_fallback"),
        "depth_width_method": depth_diag.get("final_depth_width_method"),
        "depth_scale_m_per_unit": depth_diag.get("depth_scale_m_per_unit"),
        "depth_scale_source": depth_diag.get("depth_scale_source"),
        "raw_band_pixel_count": depth_diag.get("raw_band_pixel_count"),
        "valid_depth_point_count_before_z_filter": depth_diag.get("valid_depth_point_count_before_z_filter"),
        "valid_depth_point_count_after_z_filter": depth_diag.get("valid_depth_point_count_after_z_filter"),
        "used_point_count_after_downsampling": depth_diag.get("used_point_count_after_downsampling"),
        "rejected_point_count": depth_diag.get("rejected_point_count"),
        "depth_median_m": depth_diag.get("depth_median_m"),
        "depth_mad_m": depth_diag.get("depth_mad_m"),
        "span_mm_pixel_percentile_2_98": depth_diag.get("span_mm_pixel_percentile_2_98"),
        "span_mm_3d_minmax": depth_diag.get("span_mm_3d_minmax"),
        "span_mm_3d_percentile_2_98": depth_diag.get("span_mm_3d_percentile_2_98"),
        "span_mm_3d_edge_bin": depth_diag.get("span_mm_3d_edge_bin"),
        "edge_bin_a_count": edge_bin.get("edge_bin_a_count"),
        "edge_bin_b_count": edge_bin.get("edge_bin_b_count"),
        "edge_bin_width_px_used_a": edge_bin.get("edge_bin_width_px_used_a"),
        "edge_bin_width_px_used_b": edge_bin.get("edge_bin_width_px_used_b"),
        "manual_local_path_length_mm": manual.get("manual_local_path_length_mm"),
        "manual_measurement_tool": manual.get("manual_measurement_tool"),
        "manual_notes": manual.get("manual_notes"),
        "mask_usable": manual.get("mask_usable"),
        "depth_usable": manual.get("depth_usable"),
        "capture_ms": _stage_ms(pipeline_result, "capture"),
        "perception_ms": _stage_ms(pipeline_result, "target_selection"),
        "geometry_ms": _stage_ms(pipeline_result, "geometry"),
        "anchor_selection_ms": _stage_ms(pipeline_result, "anchor_selection"),
        "qwen_inference_ms": _qwen_ms(anchor_diag),
        "path_length_ms": _stage_ms(pipeline_result, "path_length"),
        "full_pipeline_total_ms": pipeline_result.timing_ms,
        "pipeline_session_dir": pipeline_result.session_dir,
        "shared_capture_dir": str(shared_capture_dir) if shared_capture_dir else None,
        "selected_mask_overlay_path": target.selected_mask_overlay_path if target else None,
        "rotated_object_overlay_path": artifacts.get("rotated_object_overlay"),
        "cropped_rotated_object_overlay_path": artifacts.get("cropped_rotated_object_overlay"),
        "wide_contact_grid_path": artifacts.get("combined_wide_contact_guided_grid_2col"),
        "qwen_response_path": artifacts.get("qwen_anchor_decision"),
        "robot_anchor_geometry_path": artifacts.get("robot_anchor_geometry"),
        "selected_anchor_overlay_path": anchor.overlay_paths.get("selected_anchor_overlay") if anchor else None,
        "local_chord_overlay_path": local_overlay,
        "depth_used_points_overlay_path": depth_used_overlay,
        "depth_edge_bins_overlay_path": depth_edge_overlay,
        "depth_projection_histogram_path": depth_hist,
        "depth_local_pointcloud_npz_path": depth_npz,
        "depth_local_pointcloud_ply_path": depth_ply,
        "depth_width_diagnostics_path": depth_diag_path,
        "global_dimension_overlay_path": global_overlay,
        "depth_width_overlay_path": depth_overlay,
        "trial_result_json_path": str(attempt_dir / "trial_result.json"),
        "attempt_status": status,
        "failure_reason": pipeline_result.failure_reason,
    }
    row.update({
        "local_mask_abs_error_mm": _abs_err(row.get("local_mask_path_length_mm"), row.get("manual_local_path_length_mm")),
        "local_mask_pct_error": _pct_err(row.get("local_mask_path_length_mm"), row.get("manual_local_path_length_mm")),
        "local_depth_abs_error_mm": _abs_err(row.get("local_depth_path_length_mm"), row.get("manual_local_path_length_mm")),
        "local_depth_pct_error": _pct_err(row.get("local_depth_path_length_mm"), row.get("manual_local_path_length_mm")),
    })
    (attempt_dir / "pipeline_session").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "pipeline_session" / "source_pipeline_session_dir.txt").write_text(str(pipeline_result.session_dir), encoding="utf-8")
    (attempt_dir / "manual_entry.json").write_text(json.dumps(manual, indent=2), encoding="utf-8")
    (attempt_dir / "trial_result.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def _axis_order(axis_mode: str) -> int:
    return 1 if axis_mode == "major" else 2


def _dry_row(session: ExperimentSession, case: dict[str, str], axis_mode: str, attempt_dir: Path, measurement_idx: int) -> dict[str, Any]:
    base = 210.0 if case["material"] == "brick" else 245.0 if case["material"] == "timber" else 390.0
    minor = 95.0 if case["material"] == "brick" else 80.0 if case["material"] == "timber" else 190.0
    local = minor * (0.98 if axis_mode == "major" else 1.03)
    depth_cfg = session.config["depth_processing"]
    overlays = attempt_dir / "overlays"
    selected_mask = _save_simple_overlay(overlays / "selected_mask_overlay.png", text="synthetic selected mask")
    global_overlay = _save_simple_overlay(overlays / "global_mask_dimension_overlay.png", text="synthetic global dimensions")
    depth_overlay = _save_simple_overlay(overlays / "global_depth_dimension_overlay.png", text="synthetic depth dimensions")
    anchor_overlay = _save_simple_overlay(overlays / "selected_anchor_overlay.png", text="synthetic qwen anchor")
    local_overlay = _save_simple_overlay(overlays / "local_anchor_path_length_overlay.png", text="synthetic local path length")
    depth_used = _save_simple_overlay(overlays / "depth_used_points_overlay.png", text="synthetic depth used points")
    depth_edge = _save_simple_overlay(overlays / "depth_edge_bins_overlay.png", text="synthetic depth edge bins")
    depth_hist = _save_simple_overlay(overlays / "depth_projection_histogram.png", text="synthetic depth projection histogram")
    qwen_grid = _save_simple_overlay(overlays / "qwen_wide_contact_grid.png", text="synthetic qwen grid")
    rotated = _save_simple_overlay(overlays / "rotated_object_overlay.png", text="synthetic rotated object")
    cropped = _save_simple_overlay(overlays / "cropped_rotated_object_overlay.png", text="synthetic cropped rotated object")
    pipeline_session = attempt_dir / "pipeline_session"
    pipeline_session.mkdir(parents=True, exist_ok=True)
    robot_geom = attempt_dir / "copied_pipeline_artifacts" / "robot_anchor_geometry.json"
    robot_geom.parent.mkdir(parents=True, exist_ok=True)
    robot_geom.write_text(json.dumps({"robot_candidate_anchor_center_px": [320.0, 240.0], "qwen_selected_anchor_id": "A3"}, indent=2), encoding="utf-8")
    depth_npz = attempt_dir / "copied_pipeline_artifacts" / "depth_local_pointcloud_crop.npz"
    depth_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(depth_npz, xyz_m=np.zeros((1, 3)), uv_px=np.zeros((1, 2)))
    depth_ply = attempt_dir / "copied_pipeline_artifacts" / "depth_local_pointcloud_crop.ply"
    depth_ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")
    depth_diag = attempt_dir / "copied_pipeline_artifacts" / "depth_width_diagnostics.json"
    depth_diag.write_text(json.dumps({"method": "dry_run_true_3d_local_pointcloud_edge_bin"}, indent=2), encoding="utf-8")
    row = {
        "session_id": session.session_id, "case_index": case["case_index"], "case_id": case["case_id"],
        "material": case["material"], "object_instance_id": case["object_instance_id"], "axis_mode": axis_mode,
        "axis_order_for_case": _axis_order(axis_mode), "measurement_row_index": measurement_idx,
        "attempt_index": 1, "timestamp": _timestamp(),
        "requested_material": _material_for_pipeline(case["material"]), "selected_material": _material_for_pipeline(case["material"]),
        "selected_candidate_id": "candidate_001", "target_selected": True,
        "centroid_x_px": 320.0, "centroid_y_px": 240.0, "major_axis_deg": 0.0, "minor_axis_deg": 90.0,
        "anchor_backend": "v2a_wide_contact_qwen_required", "qwen_required": True, "qwen_selected_anchor_id": "A3",
        "final_selected_anchor_id": "A3", "final_anchor_source": "qwen_required",
        "robot_candidate_anchor_center_x_px": 320.0, "robot_candidate_anchor_center_y_px": 240.0,
        "contact_a_x_px": 320.0, "contact_a_y_px": 190.0, "contact_b_x_px": 320.0, "contact_b_y_px": 290.0,
        "anchor_score": 88.0,
        "local_mask_path_length_mm": local, "local_depth_path_length_mm": local * 0.99,
        "local_mask_compute_ms": 1.8, "local_depth_compute_ms": 6.8, "local_depth_valid": True,
        "local_depth_valid_point_count": 2200, "local_depth_mask_disagreement_ratio": 0.01,
        "depth_validity_policy": "computed_edge_bin_width_is_valid",
        "depth_warnings": "",
        "depth_width_was_computed": True,
        "depth_width_used_for_clamp": True,
        "path_length_source": "depth_edge_bin_3d",
        "pointcloud_downsample_mode": depth_cfg["pointcloud_downsample_mode"],
        "pointcloud_stride": depth_cfg["pointcloud_stride"], "max_points": depth_cfg["max_points"],
        "min_valid_depth_points": depth_cfg["min_valid_depth_points"], "local_depth_band_px": depth_cfg["local_depth_band_px"],
        "depth_width_method": "edge_bin_3d_endpoint_distance",
        "depth_scale_m_per_unit": 0.001,
        "depth_scale_source": "dry_run",
        "raw_band_pixel_count": 2400,
        "valid_depth_point_count_before_z_filter": 2300,
        "valid_depth_point_count_after_z_filter": 2250,
        "used_point_count_after_downsampling": 2250,
        "rejected_point_count": 150,
        "depth_median_m": 0.3,
        "depth_mad_m": 0.001,
        "span_mm_pixel_percentile_2_98": local * 0.95,
        "span_mm_3d_minmax": local * 1.01,
        "span_mm_3d_percentile_2_98": local * 0.97,
        "span_mm_3d_edge_bin": local * 0.99,
        "edge_bin_a_count": 50,
        "edge_bin_b_count": 52,
        "edge_bin_width_px_used_a": depth_cfg.get("edge_bin_width_px", 8.0),
        "edge_bin_width_px_used_b": depth_cfg.get("edge_bin_width_px", 8.0),
        "manual_local_path_length_mm": local,
        "manual_measurement_tool": "dry_run", "manual_notes": "synthetic", "mask_usable": True, "depth_usable": True,
        "anchor_selection_ms": 0.0, "qwen_inference_ms": 0.0, "path_length_ms": 0.0, "full_pipeline_total_ms": 0.0,
        "pipeline_session_dir": str(pipeline_session), "shared_capture_dir": str(attempt_dir.parent / "shared_capture"),
        "selected_mask_overlay_path": selected_mask,
        "rotated_object_overlay_path": rotated, "cropped_rotated_object_overlay_path": cropped,
        "wide_contact_grid_path": qwen_grid, "qwen_response_path": str(attempt_dir / "qwen_anchor_decision.json"),
        "robot_anchor_geometry_path": str(robot_geom), "selected_anchor_overlay_path": anchor_overlay,
        "local_chord_overlay_path": local_overlay,
        "depth_used_points_overlay_path": depth_used,
        "depth_edge_bins_overlay_path": depth_edge,
        "depth_projection_histogram_path": depth_hist,
        "depth_local_pointcloud_npz_path": str(depth_npz),
        "depth_local_pointcloud_ply_path": str(depth_ply),
        "depth_width_diagnostics_path": str(depth_diag),
        "global_dimension_overlay_path": global_overlay, "depth_width_overlay_path": depth_overlay,
        "trial_result_json_path": str(attempt_dir / "trial_result.json"),
        "attempt_status": "saved", "failure_reason": None,
        "global_debug": {
            "global_mask_major_mm": base,
            "global_mask_minor_mm": minor,
            "global_depth_major_mm": base * 0.99,
            "global_depth_minor_mm": minor * 1.01,
        },
    }
    row.update({
        "local_mask_abs_error_mm": _abs_err(row["local_mask_path_length_mm"], row["manual_local_path_length_mm"]),
        "local_mask_pct_error": _pct_err(row["local_mask_path_length_mm"], row["manual_local_path_length_mm"]),
        "local_depth_abs_error_mm": _abs_err(row["local_depth_path_length_mm"], row["manual_local_path_length_mm"]),
        "local_depth_pct_error": _pct_err(row["local_depth_path_length_mm"], row["manual_local_path_length_mm"]),
    })
    (attempt_dir / "manual_entry.json").write_text(json.dumps({"dry_run": True}, indent=2), encoding="utf-8")
    (attempt_dir / "trial_result.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def _summaries(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_material: list[dict[str, Any]] = []
    for material in sorted({row["material"] for row in rows}):
        subset = [row for row in rows if row["material"] == material]
        by_material.append({
            "material": material,
            "n": len(subset),
            "mean_local_mask_abs_error_mm": _mean([row.get("local_mask_abs_error_mm") for row in subset]),
            "mean_local_depth_abs_error_mm": _mean([row.get("local_depth_abs_error_mm") for row in subset]),
            "mean_local_mask_pct_error": _mean([row.get("local_mask_pct_error") for row in subset]),
            "mean_local_depth_pct_error": _mean([row.get("local_depth_pct_error") for row in subset]),
        })
    by_axis: list[dict[str, Any]] = []
    for axis_mode in sorted({row["axis_mode"] for row in rows}):
        subset = [row for row in rows if row["axis_mode"] == axis_mode]
        by_axis.append({
            "axis_mode": axis_mode,
            "n": len(subset),
            "mean_local_mask_abs_error_mm": _mean([row.get("local_mask_abs_error_mm") for row in subset]),
            "mean_local_depth_abs_error_mm": _mean([row.get("local_depth_abs_error_mm") for row in subset]),
            "mean_local_mask_pct_error": _mean([row.get("local_mask_pct_error") for row in subset]),
            "mean_local_depth_pct_error": _mean([row.get("local_depth_pct_error") for row in subset]),
            "mean_local_mask_compute_ms": _mean([row.get("local_mask_compute_ms") for row in subset]),
            "mean_local_depth_compute_ms": _mean([row.get("local_depth_compute_ms") for row in subset]),
        })
    by_method = [
        {"method": "local_mask", "mean_abs_error_mm": _mean([row.get("local_mask_abs_error_mm") for row in rows]), "mean_compute_ms": _mean([row.get("local_mask_compute_ms") for row in rows])},
        {"method": "local_depth", "mean_abs_error_mm": _mean([row.get("local_depth_abs_error_mm") for row in rows]), "mean_compute_ms": _mean([row.get("local_depth_compute_ms") for row in rows])},
    ]
    _write_rows(session_dir / "summary_by_material.csv", by_material, list(by_material[0].keys()) if by_material else ["material"])
    _write_rows(session_dir / "summary_by_axis_mode.csv", by_axis, list(by_axis[0].keys()) if by_axis else ["axis_mode"])
    _write_rows(session_dir / "summary_by_method.csv", by_method, list(by_method[0].keys()))
    paper = session_dir / "paper_tables"
    _write_rows(paper / "master_results.csv", rows, MASTER_COLUMNS)
    _write_rows(paper / "summary_by_material.csv", by_material, list(by_material[0].keys()) if by_material else ["material"])
    _write_rows(paper / "summary_by_axis_mode.csv", by_axis, list(by_axis[0].keys()) if by_axis else ["axis_mode"])
    _write_rows(paper / "summary_by_method.csv", by_method, list(by_method[0].keys()))
    (paper / "latex_table_local_path_error_summary.tex").write_text("% Regenerate from summary_by_method.csv for final paper formatting.\n", encoding="utf-8")


def _mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None and value != ""]
    return statistics.mean(nums) if nums else None


def generate_figures(session_dir: str | Path) -> None:
    session_dir = Path(session_dir)
    master = session_dir / "master_results.csv"
    if not master.exists():
        return
    with master.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = session_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    manual = [_safe_float(r.get("manual_local_path_length_mm")) for r in rows]
    mask = [_safe_float(r.get("local_mask_path_length_mm")) for r in rows]
    depth = [_safe_float(r.get("local_depth_path_length_mm")) for r in rows]

    plt.figure(figsize=(6, 4))
    plt.scatter(manual, mask, label="mask", marker="o")
    plt.scatter(manual, depth, label="depth", marker="x")
    vals = [v for v in manual + mask + depth if v is not None]
    if vals:
        plt.plot([min(vals), max(vals)], [min(vals), max(vals)], "k--", linewidth=1)
    plt.xlabel("Manual local path length (mm)")
    plt.ylabel("Estimated local path length (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "estimated_vs_manual_local_path.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["local mask", "local depth"], [_mean([r.get("local_mask_abs_error_mm") for r in rows]) or 0, _mean([r.get("local_depth_abs_error_mm") for r in rows]) or 0])
    plt.ylabel("Mean absolute error (mm)")
    plt.tight_layout()
    plt.savefig(fig_dir / "local_path_error_by_method.png", dpi=200)
    plt.close()

    labels = [f"{m}-{a}" for m in sorted({r["material"] for r in rows}) for a in ["major", "minor"]]
    plt.figure(figsize=(9, 4))
    data = [
        [_safe_float(r.get("local_depth_pct_error")) for r in rows if f"{r['material']}-{r['axis_mode']}" == label]
        for label in labels
    ]
    plt.boxplot([[v for v in vals if v is not None] or [0] for vals in data], labels=labels)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Local depth percent error")
    plt.tight_layout()
    plt.savefig(fig_dir / "percent_error_by_material_axis_method.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["mask compute", "depth compute"], [_mean([r.get("local_mask_compute_ms") for r in rows]) or 0, _mean([r.get("local_depth_compute_ms") for r in rows]) or 0])
    plt.ylabel("Compute time (ms)")
    plt.tight_layout()
    plt.savefig(fig_dir / "runtime_mask_vs_depth.png", dpi=200)
    plt.close()

    axis_labels = ["major", "minor"]
    mask_vals = [_mean([r.get("local_mask_abs_error_mm") for r in rows if r.get("axis_mode") == axis]) or 0 for axis in axis_labels]
    depth_vals = [_mean([r.get("local_depth_abs_error_mm") for r in rows if r.get("axis_mode") == axis]) or 0 for axis in axis_labels]
    x = np.arange(len(axis_labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.18, mask_vals, width=0.36, label="mask")
    plt.bar(x + 0.18, depth_vals, width=0.36, label="depth")
    plt.xticks(x, axis_labels)
    plt.ylabel("Mean absolute error (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "local_path_error_by_axis_mode.png", dpi=200)
    plt.close()


def summarize_session(session_dir: str | Path, *, figures: bool = True) -> None:
    session = Path(session_dir)
    master = session / "master_results.csv"
    if not master.exists():
        raise FileNotFoundError(f"master_results.csv not found: {master}")
    with master.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _summaries(session, rows)
    if figures:
        generate_figures(session)


def _prompt_manual() -> dict[str, Any]:
    print("\nNow enter physical manual measurement:")
    return {
        "manual_local_path_length_mm": _safe_float(input("Manual selected-anchor local path length in mm: ")),
        "manual_measurement_tool": input("Manual measurement tool [ruler/vernier_caliper/digital_caliper/tape/other]: ").strip() or "vernier_caliper",
        "manual_notes": input("Manual measurement notes: ").strip(),
        "mask_usable": (input("Mask usable? [y/n]: ").strip().lower() or "y").startswith("y"),
        "depth_usable": (input("Depth usable? [y/n]: ").strip().lower() or "y").startswith("y"),
    }


def _print_estimates(row: dict[str, Any]) -> None:
    print("-" * 60)
    print(f"Case {row.get('case_index')}/15: {row.get('case_id')}")
    print(f"Axis measurement {row.get('axis_order_for_case')}/2: {row.get('axis_mode')}")
    print(f"Qwen-selected anchor: {row.get('qwen_selected_anchor_id')}")
    print(f"Anchor pixel: [{row.get('robot_candidate_anchor_center_x_px')}, {row.get('robot_candidate_anchor_center_y_px')}]")
    print(f"Contact A pixel: [{row.get('contact_a_x_px')}, {row.get('contact_a_y_px')}]")
    print(f"Contact B pixel: [{row.get('contact_b_x_px')}, {row.get('contact_b_y_px')}]")
    print("\nComputed selected-anchor local path length:")
    print(f"Mask-based local path length: {row.get('local_mask_path_length_mm')} mm")
    print(f"Depth/point-cloud local path length: {row.get('local_depth_path_length_mm')} mm")
    print(f"Depth valid: {row.get('local_depth_valid')}")
    print(f"Depth validity policy: {row.get('depth_validity_policy')}")
    warnings = row.get("depth_warnings")
    if warnings:
        print(f"Depth warnings: {warnings}")
    print(f"Path length source: {row.get('path_length_source')}")
    ratio = row.get("local_depth_mask_disagreement_ratio")
    pct = None if ratio in (None, "") else float(ratio) * 100.0
    print(f"Depth/mask disagreement: {pct} %")
    print("\nTiming for measurement calculation only:")
    print(f"Mask local path calculation: {row.get('local_mask_compute_ms')} ms")
    print(f"Depth/point-cloud local path calculation: {row.get('local_depth_compute_ms')} ms")
    print("\nDepth point-cloud diagnostics:")
    print(f"method: {row.get('depth_width_method')}")
    print(f"depth scale: {row.get('depth_scale_m_per_unit')} ({row.get('depth_scale_source')})")
    print(f"raw band pixels: {row.get('raw_band_pixel_count')}")
    print(f"valid points before/after z filter: {row.get('valid_depth_point_count_before_z_filter')} / {row.get('valid_depth_point_count_after_z_filter')}")
    print(f"used points after downsampling: {row.get('used_point_count_after_downsampling')}")
    print(f"edge bin A/B point counts: {row.get('edge_bin_a_count')} / {row.get('edge_bin_b_count')}")
    print(f"pixel percentile width: {row.get('span_mm_pixel_percentile_2_98')} mm")
    print(f"3D minmax width: {row.get('span_mm_3d_minmax')} mm")
    print(f"3D percentile width: {row.get('span_mm_3d_percentile_2_98')} mm")
    print(f"3D edge-bin final width: {row.get('span_mm_3d_edge_bin')} mm")
    print(f"pointcloud NPZ: {row.get('depth_local_pointcloud_npz_path')}")
    print(f"pointcloud PLY: {row.get('depth_local_pointcloud_ply_path')}")
    print(f"depth layout overlay: {row.get('depth_used_points_overlay_path')}")
    print("\nImportant overlays:")
    print(f"selected_anchor_overlay: {row.get('selected_anchor_overlay_path')}")
    print(f"local_chord_overlay: {row.get('local_chord_overlay_path')}")
    print(f"wide_contact_grid: {row.get('wide_contact_grid_path')}")
    print(f"cropped_rotated_object_overlay: {row.get('cropped_rotated_object_overlay_path')}")


def _case_attempt_dir(session: ExperimentSession, case: dict[str, str], attempt_index: int) -> Path:
    return session.session_dir / "cases" / f"case_{int(case['case_index']):03d}_{case['case_id']}" / "attempts" / f"attempt_{attempt_index:03d}"


def _copy_shared_capture(pipeline_result: Any, shared_capture_dir: Path) -> dict[str, str | None]:
    capture = getattr(pipeline_result, "capture", None)
    shared_capture_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}
    for attr, name in [
        ("rgb_path", "raw_rgb.png"),
        ("depth_path", "depth_raw.npy"),
        ("camera_info_path", "camera_info.json"),
        ("depth_visualization_path", "depth_visualization.png"),
    ]:
        source = getattr(capture, attr, None) if capture else None
        if not source:
            paths[attr] = None
            continue
        src = Path(source)
        dst = shared_capture_dir / name
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            paths[attr] = str(dst)
        else:
            paths[attr] = None
    return paths


def _apply_manual_to_row(row: dict[str, Any], manual: dict[str, Any]) -> None:
    row.update({
        "manual_local_path_length_mm": manual.get("manual_local_path_length_mm"),
        "manual_measurement_tool": manual.get("manual_measurement_tool"),
        "manual_notes": manual.get("manual_notes"),
        "mask_usable": manual.get("mask_usable"),
        "depth_usable": manual.get("depth_usable"),
        "local_mask_abs_error_mm": _abs_err(row.get("local_mask_path_length_mm"), manual.get("manual_local_path_length_mm")),
        "local_mask_pct_error": _pct_err(row.get("local_mask_path_length_mm"), manual.get("manual_local_path_length_mm")),
        "local_depth_abs_error_mm": _abs_err(row.get("local_depth_path_length_mm"), manual.get("manual_local_path_length_mm")),
        "local_depth_pct_error": _pct_err(row.get("local_depth_path_length_mm"), manual.get("manual_local_path_length_mm")),
    })


def run_dry(config_path: str | Path) -> Path:
    session = _prepare_session(config_path)
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    axes = list(session.config["experiment"].get("run_modes") or ["major", "minor"])
    measurement_idx = 0
    for case in session.cases:
        for axis in axes:
            measurement_idx += 1
            attempt_dir = _case_attempt_dir(session, case, 1) / f"axis_{axis}"
            row = _dry_row(session, case, axis, attempt_dir, measurement_idx)
            rows.append(row)
            attempts.append(row)
    _write_rows(session.session_dir / "master_results.csv", rows, MASTER_COLUMNS)
    _write_rows(session.session_dir / "attempts_all.csv", attempts, ATTEMPT_COLUMNS)
    _write_rows(
        session.session_dir / "case_progress.csv",
        [{"case_id": case["case_id"], "status": "saved_both_axes"} for case in session.cases],
        ["case_id", "status"],
    )
    _summaries(session.session_dir, rows)
    generate_figures(session.session_dir)
    print(f"dry_run_session: {session.session_dir}")
    return session.session_dir


def run_live(args: argparse.Namespace) -> Path:
    session = _prepare_session(args.config)
    selected_cases = session.cases
    if args.case_index:
        selected_cases = [case for case in selected_cases if int(case["case_index"]) == int(args.case_index)]
    axis_modes = args.axis_mode or list(session.config["experiment"].get("run_modes") or ["major", "minor"])
    saved_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    measurement_idx = 0
    for case in selected_cases:
        attempt = 1
        while True:
            print("=" * 60)
            print(f"E2 Width / Path-Length Validation Case {case['case_index']}/15: {case['case_id']}")
            print(f"Requested material: {_material_for_pipeline(case['material'])}")
            print("Single-object scene only.")
            print("No robot, no clamp, no RTDE, no Arduino.")
            action = input("Place the object under the camera. Press ENTER to capture, type skip, retry, or quit: ").strip().lower()
            if action == "quit":
                _write_rows(session.session_dir / "master_results.csv", saved_rows, MASTER_COLUMNS)
                _write_rows(session.session_dir / "attempts_all.csv", attempt_rows, ATTEMPT_COLUMNS)
                summarize_session(session.session_dir)
                return session.session_dir
            if action == "skip":
                break
            attempt_root = _case_attempt_dir(session, case, attempt)
            pipeline_config_path = _create_case_pipeline_config(session, attempt_root)
            shared_capture_dir = attempt_root / "shared_capture"
            capture_paths: dict[str, str | None] | None = None
            first_axis_result: Any | None = None

            for axis_mode in axis_modes:
                axis_attempt = 1
                while True:
                    axis_dir = attempt_root / f"axis_{axis_mode}"
                    if capture_paths is None:
                        result = run_full_main_upv_vlm_v2_pipeline(
                            config_path=pipeline_config_path,
                            requested_material=_material_for_pipeline(case["material"]),
                            axis_mode=axis_mode,
                            mode="path_length_only",
                            live_ros2=bool(args.live_ros2),
                            execution_backend="none",
                        )
                        first_axis_result = result
                        if result.success:
                            capture_paths = _copy_shared_capture(result, shared_capture_dir)
                    elif first_axis_result is not None and first_axis_result.axis_mode == axis_mode and axis_attempt == 1:
                        result = first_axis_result
                    else:
                        result = run_full_main_upv_vlm_v2_pipeline(
                            config_path=pipeline_config_path,
                            requested_material=_material_for_pipeline(case["material"]),
                            axis_mode=axis_mode,
                            mode="path_length_only",
                            input_rgb=capture_paths.get("rgb_path"),
                            input_depth=capture_paths.get("depth_path"),
                            input_camera_info=capture_paths.get("camera_info_path"),
                            live_ros2=False,
                            execution_backend="none",
                        )
                    if not result.success:
                        fail = {
                            "case_id": case["case_id"],
                            "axis_mode": axis_mode,
                            "axis_order_for_case": _axis_order(axis_mode),
                            "attempt_index": attempt,
                            "attempt_status": "pipeline_failed",
                            "failure_reason": result.failure_reason,
                        }
                        attempt_rows.append(fail)
                        _append_or_write(session.session_dir / "attempts_all.csv", fail, ATTEMPT_COLUMNS)
                        _print_pipeline_failure_details(result, axis_mode)
                        decision = input("Retry this axis on the same capture? [y/n/recapture/quit]: ").strip().lower()
                        if decision == "quit":
                            summarize_session(session.session_dir)
                            return session.session_dir
                        if decision == "recapture":
                            attempt += 1
                            break
                        if decision.startswith("y"):
                            axis_attempt += 1
                            continue
                        break
                    row = _extract_row_from_pipeline(
                        session=session,
                        case=case,
                        axis_mode=axis_mode,
                        axis_order_for_case=_axis_order(axis_mode),
                        attempt_index=attempt,
                        measurement_row_index=None,
                        pipeline_result=result,
                        attempt_dir=axis_dir,
                        shared_capture_dir=shared_capture_dir,
                        manual={},
                        status="pending",
                    )
                    _print_estimates(row)
                    manual = _prompt_manual()
                    _apply_manual_to_row(row, manual)
                    (axis_dir / "manual_entry.json").write_text(json.dumps(manual, indent=2), encoding="utf-8")
                    decision = input("Save this axis measurement? [save/retry/skip/quit]: ").strip().lower()
                    if decision == "save":
                        measurement_idx += 1
                        row["measurement_row_index"] = measurement_idx
                        row["attempt_status"] = "saved"
                        (axis_dir / "trial_result.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
                        saved_rows.append(row)
                        attempt_rows.append(row)
                        _append_or_write(session.session_dir / "master_results.csv", row, MASTER_COLUMNS)
                        _append_or_write(session.session_dir / "attempts_all.csv", row, ATTEMPT_COLUMNS)
                        break
                    row["attempt_status"] = decision or "retry"
                    attempt_rows.append(row)
                    _append_or_write(session.session_dir / "attempts_all.csv", row, ATTEMPT_COLUMNS)
                    if decision == "quit":
                        summarize_session(session.session_dir)
                        return session.session_dir
                    if decision == "skip":
                        break
                    axis_attempt += 1
                if 'decision' in locals() and decision == "recapture":
                    break
            else:
                break
            if 'decision' in locals() and decision == "recapture":
                continue
            break
    _write_rows(session.session_dir / "master_results.csv", saved_rows, MASTER_COLUMNS)
    _write_rows(session.session_dir / "attempts_all.csv", attempt_rows, ATTEMPT_COLUMNS)
    summarize_session(session.session_dir, figures=not args.no_figures)
    return session.session_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 width/path-length fixed15 experiment wrapper.")
    parser.add_argument("--config", default="configs/v2/experiments/width_path_length_fixed15.yaml")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--axis-mode", action="append", choices=["major", "minor"])
    parser.add_argument("--live-ros2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-session")
    parser.add_argument("--summarize-session")
    parser.add_argument("--generate-figures")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generate_figures:
        generate_figures(args.generate_figures)
        print(f"generated_figures: {args.generate_figures}")
        return 0
    if args.summarize_session:
        summarize_session(args.summarize_session, figures=not args.no_figures)
        print(f"summarized_session: {args.summarize_session}")
        return 0
    if args.resume_session:
        summarize_session(args.resume_session, figures=not args.no_figures)
        print(f"resume currently summarizes existing session: {args.resume_session}")
        return 0
    if args.dry_run:
        run_dry(args.config)
        return 0
    run_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
