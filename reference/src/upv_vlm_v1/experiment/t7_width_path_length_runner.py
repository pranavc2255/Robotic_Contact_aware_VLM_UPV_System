from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from upv_vlm_orient.anchor_selection import build_axis_search_domain, compute_anchor_cross_section_hits
from upv_vlm_v1.experiment.t5_target_selection_runner import resolve_repo_path
from upv_vlm_v1.experiment.t5_v2_target_selection_fixed27_runner import (
    T5V2Trial,
    capture_ros2_snapshot,
    check_clip_subprocess_environment,
    resolve_clip_subprocess_python,
    run_candidate_pool_perception,
)
from upv_vlm_v1.geometry.depth_width_refinement import estimate_depth_refined_width


REPO_ROOT = Path(__file__).resolve().parents[3]
NA = "NA"

T7_COLUMNS = [
    "session_id",
    "saved_trial_number",
    "case_index",
    "case_id",
    "material_query",
    "expected_material",
    "object_id",
    "orientation_case",
    "attempt_index",
    "timestamp",
    "target_selected",
    "final_selected_candidate_id",
    "selected_candidate_mask_path",
    "raw_rgb_path",
    "depth_raw_path",
    "camera_info_path",
    "selected_mask_overlay_path",
    "major_minor_dimension_overlay_path",
    "mask_dimension_overlay_path",
    "depth_width_major_overlay_path",
    "depth_width_minor_overlay_path",
    "anchor_candidates_major_overlay_path",
    "anchor_candidates_minor_overlay_path",
    "anchor_selection_major_summary_path",
    "anchor_selection_minor_summary_path",
    "anchor_local_path_length_major_overlay_path",
    "anchor_local_path_length_minor_overlay_path",
    "centroid_x_px",
    "centroid_y_px",
    "major_axis_angle_deg",
    "minor_axis_angle_deg",
    "major_axis_vector_x",
    "major_axis_vector_y",
    "minor_axis_vector_x",
    "minor_axis_vector_y",
    "mask_major_dimension_mm",
    "mask_minor_dimension_mm",
    "mask_major_valid",
    "mask_minor_valid",
    "mask_major_failure_reason",
    "mask_minor_failure_reason",
    "depth_major_dimension_mm",
    "depth_minor_dimension_mm",
    "depth_major_valid",
    "depth_minor_valid",
    "depth_major_failure_reason",
    "depth_minor_failure_reason",
    "depth_major_top_face_point_count",
    "depth_minor_top_face_point_count",
    "depth_major_coverage",
    "depth_minor_coverage",
    "depth_major_mad_m",
    "depth_minor_mad_m",
    "major_depth_mask_disagreement_ratio",
    "minor_depth_mask_disagreement_ratio",
    "major_depth_mask_sanity_pass",
    "minor_depth_mask_sanity_pass",
    "manual_major_dimension_mm",
    "manual_minor_dimension_mm",
    "measurement_tool",
    "measurement_confidence",
    "mask_usable_manual",
    "manual_notes",
    "mask_major_error_mm",
    "mask_major_error_pct",
    "mask_minor_error_mm",
    "mask_minor_error_pct",
    "depth_major_error_mm",
    "depth_major_error_pct",
    "depth_minor_error_mm",
    "depth_minor_error_pct",
    "depth_closer_than_mask_major",
    "depth_closer_than_mask_minor",
    "major_mode_anchor_selected",
    "major_mode_final_anchor_id",
    "major_mode_anchor_x_px",
    "major_mode_anchor_y_px",
    "major_mode_anchor_score",
    "major_mode_anchor_overlay_path",
    "major_mode_anchor_summary_path",
    "minor_mode_anchor_selected",
    "minor_mode_final_anchor_id",
    "minor_mode_anchor_x_px",
    "minor_mode_anchor_y_px",
    "minor_mode_anchor_score",
    "minor_mode_anchor_overlay_path",
    "minor_mode_anchor_summary_path",
    "anchor_major_mode_local_mask_path_length_mm",
    "anchor_major_mode_local_depth_path_length_mm",
    "anchor_major_mode_local_mask_valid",
    "anchor_major_mode_local_depth_valid",
    "anchor_major_mode_local_depth_failure_reason",
    "anchor_major_mode_depth_mask_disagreement_ratio",
    "anchor_major_mode_depth_mask_sanity_pass",
    "anchor_major_mode_local_mask_error_mm",
    "anchor_major_mode_local_mask_error_pct",
    "anchor_major_mode_local_depth_error_mm",
    "anchor_major_mode_local_depth_error_pct",
    "anchor_minor_mode_local_mask_path_length_mm",
    "anchor_minor_mode_local_depth_path_length_mm",
    "anchor_minor_mode_local_mask_valid",
    "anchor_minor_mode_local_depth_valid",
    "anchor_minor_mode_local_depth_failure_reason",
    "anchor_minor_mode_depth_mask_disagreement_ratio",
    "anchor_minor_mode_depth_mask_sanity_pass",
    "anchor_minor_mode_local_mask_error_mm",
    "anchor_minor_mode_local_mask_error_pct",
    "anchor_minor_mode_local_depth_error_mm",
    "anchor_minor_mode_local_depth_error_pct",
    "manual_anchor_major_mode_local_path_length_mm",
    "manual_anchor_minor_mode_local_path_length_mm",
    "capture_time_ms",
    "perception_time_ms",
    "mask_geometry_time_ms",
    "mask_major_dimension_time_ms",
    "mask_minor_dimension_time_ms",
    "mask_total_dimension_time_ms",
    "depth_major_dimension_time_ms",
    "depth_minor_dimension_time_ms",
    "depth_total_dimension_time_ms",
    "anchor_selection_major_time_ms",
    "anchor_selection_minor_time_ms",
    "anchor_selection_total_time_ms",
    "anchor_local_mask_major_mode_time_ms",
    "anchor_local_mask_minor_mode_time_ms",
    "anchor_local_mask_total_time_ms",
    "anchor_local_depth_major_mode_time_ms",
    "anchor_local_depth_minor_mode_time_ms",
    "anchor_local_depth_total_time_ms",
    "total_attempt_time_ms",
]

ATTEMPT_COLUMNS = [
    "session_id",
    "case_index",
    "case_id",
    "material_query",
    "attempt_index",
    "attempt_status",
    "attempt_dir",
    "target_selected",
    "final_selected_candidate_id",
    "mask_major_dimension_mm",
    "mask_minor_dimension_mm",
    "depth_major_dimension_mm",
    "depth_minor_dimension_mm",
    "anchor_major_mode_local_mask_path_length_mm",
    "anchor_minor_mode_local_mask_path_length_mm",
    "discard_reason",
    "timestamp",
]


@dataclass
class T7Case:
    trial_index: int
    case_id: str
    material_query: str
    expected_material: str
    object_id: str
    orientation_case: str
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "T7Case":
        return cls(
            trial_index=int(row["trial_index"]),
            case_id=str(row["case_id"]),
            material_query=str(row["material_query"]),
            expected_material=str(row["expected_material"]),
            object_id=str(row.get("object_id") or row["case_id"]),
            orientation_case=str(row.get("orientation_case") or "NA"),
            enabled=str(row.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "n"},
            notes=str(row.get("notes") or ""),
        )

    def folder_name(self) -> str:
        return f"trial_{self.trial_index:03d}_{_slug(self.case_id)}"

    def to_t5_trial(self) -> T5V2Trial:
        return T5V2Trial(
            trial_index=self.trial_index,
            frame_id=self.case_id,
            classes_existing_in_frame=self.material_query,
            input_text=self.material_query,
            expected_selected_class=self.expected_material,
            enabled=self.enabled,
            notes=self.notes,
        )


def _slug(text: Any) -> str:
    return "_".join("".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).split("_"))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def read_cases(path: str | Path) -> list[T7Case]:
    with resolve_repo_path(path).open("r", newline="", encoding="utf-8") as handle:
        return [T7Case.from_row(row) for row in csv.DictReader(handle)]


def validate_cases(cases: list[T7Case]) -> None:
    enabled = [case for case in cases if case.enabled]
    if len(enabled) != 15:
        raise ValueError(f"T7 requires exactly 15 enabled cases, found {len(enabled)}")
    counts: dict[str, int] = {}
    for case in enabled:
        counts[case.material_query] = counts.get(case.material_query, 0) + 1
    expected = {"brick": 5, "timber": 5, "concrete block": 5}
    if counts != expected:
        raise ValueError(f"T7 expected material distribution {expected}, found {counts}")


def create_session_dir(output_root: str | Path, resume_session: str | Path | None = None) -> Path:
    session_dir = resolve_repo_path(resume_session) if resume_session else resolve_repo_path(output_root) / f"session_{now_stamp()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["cases", "paper_figures", "paper_tables", "logs"]:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    return session_dir


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_csv(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, NA) for key in columns})


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, NA) for key in columns})


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", NA):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    val = _safe_float(value)
    return NA if val is None else f"{val:.{digits}f}"


def _load_camera_info(path: str | Path) -> dict[str, Any]:
    info = read_json(path)
    if not info:
        raise ValueError(f"Missing camera_info JSON: {path}")
    if all(key in info for key in ("fx", "fy", "cx", "cy")):
        return {key: float(info[key]) for key in ("fx", "fy", "cx", "cy")} | {
            "width": info.get("width"),
            "height": info.get("height"),
        }
    matrix = info.get("K") or info.get("k") or info.get("camera_matrix")
    if isinstance(matrix, dict):
        matrix = matrix.get("data")
    if matrix is None or len(matrix) < 9:
        raise ValueError("camera_info must contain fx/fy/cx/cy or K matrix")
    return {
        "fx": float(matrix[0]),
        "fy": float(matrix[4]),
        "cx": float(matrix[2]),
        "cy": float(matrix[5]),
        "width": info.get("width"),
        "height": info.get("height"),
        "K": list(matrix),
    }


def _depth_to_m(depth: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    if "depth_unit_scale" in cfg:
        scale = float(cfg["depth_unit_scale"])
    elif np.issubdtype(depth.dtype, np.integer):
        scale = 0.001
    else:
        scale = 1.0
    return depth.astype(np.float32) * scale


def _mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    mask_bool = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask_bool)
    if len(xs) < 2:
        return {
            "valid": False,
            "failure_reason": "empty_or_too_small_mask",
            "centroid_x_px": NA,
            "centroid_y_px": NA,
            "major_axis_angle_deg": NA,
            "minor_axis_angle_deg": NA,
            "major_axis_vector_x": NA,
            "major_axis_vector_y": NA,
            "minor_axis_vector_x": NA,
            "minor_axis_vector_y": NA,
            "mask_area_px": int(len(xs)),
        }
    coords = np.column_stack([xs, ys]).astype(np.float32)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0].astype(float)
    major = major / max(1e-12, float(np.linalg.norm(major)))
    minor = np.array([-major[1], major[0]], dtype=float)
    major_angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    minor_angle = math.degrees(math.atan2(float(minor[1]), float(minor[0])))
    return {
        "valid": True,
        "failure_reason": NA,
        "centroid_x_px": round(float(centroid[0]), 3),
        "centroid_y_px": round(float(centroid[1]), 3),
        "major_axis_angle_deg": round(major_angle, 3),
        "minor_axis_angle_deg": round(minor_angle, 3),
        "major_axis_vector_x": round(float(major[0]), 8),
        "major_axis_vector_y": round(float(major[1]), 8),
        "minor_axis_vector_x": round(float(minor[0]), 8),
        "minor_axis_vector_y": round(float(minor[1]), 8),
        "mask_area_px": int(len(xs)),
    }


def _axis_mask_dimension_mm(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_info: dict[str, Any],
    axis: np.ndarray,
    depth_cfg: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    mask_bool = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask_bool)
    if len(xs) < 2:
        return {"valid": False, "dimension_mm": NA, "failure_reason": "empty_mask", "time_ms": _elapsed_ms(start)}
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    projections = coords @ axis.reshape(2)
    span_px = float(np.max(projections) - np.min(projections))
    depth_m = _depth_to_m(np.asarray(depth), depth_cfg)
    finite = np.isfinite(depth_m) & (depth_m > float(depth_cfg.get("min_depth_m", 0.05))) & (depth_m < float(depth_cfg.get("max_depth_m", 2.0)))
    valid = mask_bool & finite
    if not np.any(valid):
        return {"valid": False, "dimension_mm": NA, "failure_reason": "no_valid_depth_inside_mask", "time_ms": _elapsed_ms(start)}
    z_m = float(np.median(depth_m[valid]))
    fx, fy = float(camera_info["fx"]), float(camera_info["fy"])
    scale_m_per_px = z_m * math.sqrt((float(axis[0]) / fx) ** 2 + (float(axis[1]) / fy) ** 2)
    return {
        "valid": True,
        "dimension_mm": float(span_px * scale_m_per_px * 1000.0),
        "failure_reason": NA,
        "representative_depth_m": z_m,
        "span_px": span_px,
        "time_ms": _elapsed_ms(start),
    }


def _line_mask_hits(
    mask: np.ndarray,
    anchor_xy: tuple[float, float],
    direction_xy: np.ndarray,
    max_distance_px: float,
    step_px: float = 1.0,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    mask_bool = np.asarray(mask).astype(bool)
    direction = direction_xy.astype(float).reshape(2)
    direction = direction / max(1e-12, float(np.linalg.norm(direction)))

    def search(sign: float) -> tuple[float, float] | None:
        last = None
        dist = 0.0
        while dist <= max_distance_px:
            point = np.array(anchor_xy, dtype=float) + direction * sign * dist
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if y < 0 or y >= mask_bool.shape[0] or x < 0 or x >= mask_bool.shape[1] or not mask_bool[y, x]:
                break
            last = (float(point[0]), float(point[1]))
            dist += step_px
        return last

    return search(-1.0), search(1.0)


def _local_mask_path_length_mm(
    hit_a: tuple[float, float] | None,
    hit_b: tuple[float, float] | None,
    depth: np.ndarray,
    camera_info: dict[str, Any],
    depth_cfg: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    if hit_a is None or hit_b is None:
        return {"valid": False, "path_length_mm": NA, "failure_reason": "missing_mask_boundary_hits", "time_ms": _elapsed_ms(start)}
    depth_m = _depth_to_m(np.asarray(depth), depth_cfg)
    points = [hit_a, hit_b]
    zs = []
    for x, y in points:
        xi = max(0, min(depth_m.shape[1] - 1, int(round(x))))
        yi = max(0, min(depth_m.shape[0] - 1, int(round(y))))
        z = float(depth_m[yi, xi])
        if math.isfinite(z) and z > 0:
            zs.append(z)
    mask_valid = np.isfinite(depth_m) & (depth_m > float(depth_cfg.get("min_depth_m", 0.05))) & (depth_m < float(depth_cfg.get("max_depth_m", 2.0)))
    fallback_z = float(np.median(depth_m[mask_valid])) if np.any(mask_valid) else None
    z = float(np.median(zs)) if zs else fallback_z
    if z is None or not math.isfinite(z):
        return {"valid": False, "path_length_mm": NA, "failure_reason": "no_valid_depth_for_endpoint_conversion", "time_ms": _elapsed_ms(start)}
    fx, fy, cx, cy = float(camera_info["fx"]), float(camera_info["fy"]), float(camera_info["cx"]), float(camera_info["cy"])

    def backproject(point: tuple[float, float]) -> np.ndarray:
        x, y = point
        return np.array([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=float)

    length = float(np.linalg.norm(backproject(hit_b) - backproject(hit_a)) * 1000.0)
    return {
        "valid": True,
        "path_length_mm": length,
        "failure_reason": NA,
        "endpoint_a_px": list(hit_a),
        "endpoint_b_px": list(hit_b),
        "representative_depth_m": z,
        "time_ms": _elapsed_ms(start),
    }


def _local_depth_path_length_mm(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_info: dict[str, Any],
    anchor_xy: tuple[float, float],
    cross_axis: np.ndarray,
    mask_path_length_mm: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    depth_cfg = dict(config.get("depth_width", {}))
    depth_m = _depth_to_m(np.asarray(depth), depth_cfg)
    mask_bool = np.asarray(mask).astype(bool)
    finite = np.isfinite(depth_m) & (depth_m > float(depth_cfg.get("min_depth_m", 0.05))) & (depth_m < float(depth_cfg.get("max_depth_m", 2.0)))
    coords_y, coords_x = np.nonzero(mask_bool & finite)
    if len(coords_x) < 2:
        return {"valid": False, "path_length_mm": NA, "failure_reason": "too_few_valid_depth_points", "top_face_point_count": 0, "coverage": NA, "mad_m": NA, "time_ms": _elapsed_ms(start)}
    coords = np.column_stack([coords_x.astype(float), coords_y.astype(float)])
    axis = cross_axis.astype(float).reshape(2)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    perp = np.array([-axis[1], axis[0]], dtype=float)
    rel = coords - np.asarray(anchor_xy, dtype=float)
    line_distance = np.abs(rel @ perp)
    band_px = float(config.get("anchor_selection", {}).get("local_depth_line_band_px", 5.0))
    in_band = line_distance <= band_px
    if not np.any(in_band):
        return {"valid": False, "path_length_mm": NA, "failure_reason": "no_depth_points_near_anchor_line", "top_face_point_count": 0, "coverage": NA, "mad_m": NA, "time_ms": _elapsed_ms(start)}
    band_coords = coords[in_band]
    band_depths = depth_m[coords_y[in_band], coords_x[in_band]].astype(float)
    center_depth = float(np.median(band_depths))
    mad = float(np.median(np.abs(band_depths - center_depth))) if band_depths.size else float("nan")
    threshold = max(float(depth_cfg.get("depth_band_threshold_m", 0.012)), float(depth_cfg.get("mad_multiplier", 2.5)) * (mad if math.isfinite(mad) else 0.0))
    top = np.abs(band_depths - center_depth) <= threshold
    top_coords = band_coords[top]
    top_depths = band_depths[top]
    min_points = int(config.get("anchor_selection", {}).get("min_local_depth_points", 50))
    if len(top_coords) < min_points:
        return {"valid": False, "path_length_mm": NA, "failure_reason": "too_few_top_face_points_near_anchor_line", "top_face_point_count": int(len(top_coords)), "coverage": round(float(len(top_coords)) / max(1, int(mask_bool.sum())), 5), "mad_m": mad, "time_ms": _elapsed_ms(start)}
    projections = top_coords @ axis
    low, high = np.percentile(projections, [1.0, 99.0])
    z = float(np.median(top_depths))
    fx, fy = float(camera_info["fx"]), float(camera_info["fy"])
    scale_m_per_px = z * math.sqrt((float(axis[0]) / fx) ** 2 + (float(axis[1]) / fy) ** 2)
    length = float((high - low) * scale_m_per_px * 1000.0)
    mask_len = _safe_float(mask_path_length_mm)
    disagreement = abs(length - mask_len) / mask_len if mask_len and mask_len > 0 else NA
    return {
        "valid": True,
        "path_length_mm": length,
        "failure_reason": NA,
        "top_face_point_count": int(len(top_coords)),
        "coverage": round(float(len(top_coords)) / max(1, int(mask_bool.sum())), 5),
        "mad_m": mad,
        "disagreement_ratio": round(float(disagreement), 5) if disagreement != NA else NA,
        "time_ms": _elapsed_ms(start),
    }


def _run_anchor_mode(
    attempt_dir: Path,
    axis_mode: str,
    rgb_path: Path,
    mask: np.ndarray,
    depth: np.ndarray,
    camera_info: dict[str, Any],
    geom: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    anchor_cfg = config.get("anchor_selection", {})
    prefix = "major_mode" if axis_mode == "major" else "minor_mode"
    cross_prefix = "anchor_major_mode" if axis_mode == "major" else "anchor_minor_mode"
    mask_bool = np.asarray(mask).astype(bool)
    if not geom.get("valid") or not np.any(mask_bool):
        summary = {
            "axis_mode": axis_mode,
            "anchor_selected": False,
            "final_anchor_id": "NO_SAFE_ANCHOR",
            "failure_reason": "mask_geometry_invalid",
            "candidates": [],
            "selection_time_ms": _elapsed_ms(start),
        }
        summary_path = attempt_dir / f"anchor_selection_{axis_mode}_summary.json"
        write_json(summary_path, summary)
        return _anchor_mode_row(prefix, cross_prefix, summary, summary_path, NA, NA, NA, NA, NA, NA)

    major = np.array([float(geom["major_axis_vector_x"]), float(geom["major_axis_vector_y"])])
    minor = np.array([float(geom["minor_axis_vector_x"]), float(geom["minor_axis_vector_y"])])
    chosen_axis = major if axis_mode == "major" else minor
    cross_axis = minor if axis_mode == "major" else major
    axis_extent = _safe_float(geom.get("mask_area_px")) or 1.0
    ys, xs = np.nonzero(mask_bool)
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    centered = coords - np.array([float(geom["centroid_x_px"]), float(geom["centroid_y_px"])])
    axis_extent = float((centered @ chosen_axis).max() - (centered @ chosen_axis).min())
    domain = build_axis_search_domain(
        object_center_xy=(float(geom["centroid_x_px"]), float(geom["centroid_y_px"])),
        chosen_axis_unit_vector_xy=(float(chosen_axis[0]), float(chosen_axis[1])),
        axis_extent_px=axis_extent,
        usable_axis_margin_ratio=float(anchor_cfg.get("usable_axis_margin_ratio", 0.22)),
        num_anchor_samples=int(anchor_cfg.get("num_anchor_samples", 5)),
        axis_name=f"{axis_mode}_axis",
    )
    candidates = []
    for idx, s_value in enumerate(domain.candidate_s_values, start=1):
        anchor_xy, left_hit, right_hit = compute_anchor_cross_section_hits(
            axis_search_domain=domain,
            candidate_s=s_value,
            mask=mask_bool.astype(np.uint8),
            max_search_distance=max(20.0, axis_extent * 1.2),
            step_size_px=1.0,
        )
        mask_len = _local_mask_path_length_mm(left_hit.hit_xy, right_hit.hit_xy, depth, camera_info, config.get("depth_width", {}))
        center_bonus = 1.0 - min(1.0, abs(float(s_value)) / max(1.0, axis_extent / 2.0))
        score = (float(mask_len.get("path_length_mm") or 0.0) if mask_len.get("valid") else 0.0) + center_bonus * float(anchor_cfg.get("center_bonus_mm", 5.0))
        candidates.append(
            {
                "anchor_id": f"anchor_{idx:03d}",
                "anchor_xy": [float(anchor_xy[0]), float(anchor_xy[1])],
                "candidate_s": float(s_value),
                "left_hit_xy": list(left_hit.hit_xy) if left_hit.hit_xy else None,
                "right_hit_xy": list(right_hit.hit_xy) if right_hit.hit_xy else None,
                "valid": bool(left_hit.valid and right_hit.valid and mask_len.get("valid")),
                "local_mask_path_length_mm": mask_len.get("path_length_mm", NA),
                "score": round(score, 4),
            }
        )
    valid_candidates = [c for c in candidates if c["valid"]]
    selected = sorted(valid_candidates, key=lambda item: (-float(item["score"]), item["anchor_id"]))[0] if valid_candidates else None
    if selected:
        anchor_xy = (float(selected["anchor_xy"][0]), float(selected["anchor_xy"][1]))
        mask_len = selected["local_mask_path_length_mm"]
        depth_len = _local_depth_path_length_mm(mask_bool, depth, camera_info, anchor_xy, cross_axis, mask_len, config)
        threshold = float(config.get("depth_width", {}).get("max_depth_mask_disagreement_ratio_for_t7_sanity", 0.50))
        ratio = depth_len.get("disagreement_ratio", NA)
        sanity_pass = bool(_safe_float(ratio) is not None and float(ratio) <= threshold) if ratio != NA else NA
    else:
        anchor_xy = (float("nan"), float("nan"))
        mask_len = NA
        depth_len = {"valid": False, "path_length_mm": NA, "failure_reason": "no_valid_anchor", "time_ms": 0.0, "disagreement_ratio": NA}
        ratio = NA
        sanity_pass = NA

    overlay_path = attempt_dir / f"anchor_candidates_{axis_mode}_overlay.png"
    local_overlay_path = attempt_dir / f"anchor_local_path_length_{axis_mode}_overlay.png"
    _save_anchor_overlay(rgb_path, mask_bool, candidates, selected, axis_mode, overlay_path, local_overlay_path)
    summary = {
        "axis_mode": axis_mode,
        "anchor_stage": "existing_anchor_selection_axis_search_domain_cross_section_sampler",
        "anchor_selected": selected is not None,
        "final_anchor_id": selected["anchor_id"] if selected else "NO_SAFE_ANCHOR",
        "selected_anchor": selected,
        "candidates": candidates,
        "local_path_length_semantics": "major mode measures local minor-axis chord; minor mode measures local major-axis chord",
        "local_depth_result": depth_len,
        "depth_mask_disagreement_ratio": ratio,
        "depth_mask_sanity_pass": sanity_pass,
        "selection_time_ms": _elapsed_ms(start),
    }
    summary_path = attempt_dir / f"anchor_selection_{axis_mode}_summary.json"
    write_json(summary_path, summary)
    return _anchor_mode_row(prefix, cross_prefix, summary, summary_path, overlay_path, local_overlay_path, mask_len, depth_len, ratio, sanity_pass)


def _anchor_mode_row(
    prefix: str,
    cross_prefix: str,
    summary: dict[str, Any],
    summary_path: Path,
    overlay_path: Any,
    local_overlay_path: Any,
    mask_len: Any,
    depth_len: Any,
    ratio: Any,
    sanity_pass: Any,
) -> dict[str, Any]:
    selected = summary.get("selected_anchor") or {}
    depth_valid = bool(depth_len.get("valid")) if isinstance(depth_len, dict) else False
    return {
        f"{prefix}_anchor_selected": bool(summary.get("anchor_selected")),
        f"{prefix}_final_anchor_id": summary.get("final_anchor_id", "NO_SAFE_ANCHOR"),
        f"{prefix}_anchor_x_px": selected.get("anchor_xy", [NA, NA])[0] if selected else NA,
        f"{prefix}_anchor_y_px": selected.get("anchor_xy", [NA, NA])[1] if selected else NA,
        f"{prefix}_anchor_score": selected.get("score", NA) if selected else NA,
        f"{prefix}_anchor_overlay_path": str(overlay_path) if overlay_path != NA else NA,
        f"{prefix}_anchor_summary_path": str(summary_path),
        f"{cross_prefix}_local_mask_path_length_mm": round(float(mask_len), 3) if _safe_float(mask_len) is not None else NA,
        f"{cross_prefix}_local_depth_path_length_mm": round(float(depth_len.get("path_length_mm")), 3) if isinstance(depth_len, dict) and _safe_float(depth_len.get("path_length_mm")) is not None else NA,
        f"{cross_prefix}_local_mask_valid": _safe_float(mask_len) is not None,
        f"{cross_prefix}_local_depth_valid": depth_valid,
        f"{cross_prefix}_local_depth_failure_reason": depth_len.get("failure_reason", NA) if isinstance(depth_len, dict) else NA,
        f"{cross_prefix}_depth_mask_disagreement_ratio": ratio,
        f"{cross_prefix}_depth_mask_sanity_pass": sanity_pass,
        f"{cross_prefix}_local_depth_time_ms": depth_len.get("time_ms", 0.0) if isinstance(depth_len, dict) else 0.0,
        "anchor_candidates_major_overlay_path" if prefix == "major_mode" else "anchor_candidates_minor_overlay_path": str(overlay_path) if overlay_path != NA else NA,
        "anchor_selection_major_summary_path" if prefix == "major_mode" else "anchor_selection_minor_summary_path": str(summary_path),
        "anchor_local_path_length_major_overlay_path" if prefix == "major_mode" else "anchor_local_path_length_minor_overlay_path": str(local_overlay_path) if local_overlay_path != NA else NA,
        "anchor_selection_major_time_ms" if prefix == "major_mode" else "anchor_selection_minor_time_ms": summary.get("selection_time_ms", 0.0),
    }


def _save_anchor_overlay(
    rgb_path: Path,
    mask: np.ndarray,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    axis_mode: str,
    overlay_path: Path,
    local_overlay_path: Path,
) -> None:
    raw = Image.open(rgb_path).convert("RGB")
    overlay = raw.convert("RGBA")
    mask_img = Image.fromarray(np.asarray(mask).astype(np.uint8) * 255, mode="L")
    overlay.paste(Image.new("RGBA", raw.size, (80, 190, 255, 70)), (0, 0), mask_img)
    draw = ImageDraw.Draw(overlay)
    for candidate in candidates:
        x, y = candidate["anchor_xy"]
        color = (60, 160, 255, 255)
        if selected and candidate["anchor_id"] == selected["anchor_id"]:
            color = (255, 220, 0, 255)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(0, 0, 0, 255), width=2)
        draw.text((x + 7, y - 8), candidate["anchor_id"].replace("anchor_", "A"), fill=color, font=_font(12, True))
        if candidate.get("left_hit_xy") and candidate.get("right_hit_xy"):
            draw.line((*candidate["left_hit_xy"], *candidate["right_hit_xy"]), fill=(255, 255, 255, 140), width=2)
    header_h = 92
    canvas = Image.new("RGB", (raw.width, raw.height + header_h), (246, 246, 240))
    canvas.paste(overlay.convert("RGB"), (0, header_h))
    header = ImageDraw.Draw(canvas)
    header.text((14, 10), f"T7 anchor selection: {axis_mode}-mode", fill=(20, 20, 20), font=_font(20, True))
    selected_text = selected["anchor_id"] if selected else "NO_SAFE_ANCHOR"
    header.text((14, 42), f"selected anchor: {selected_text}", fill=(20, 20, 20), font=_font(16))
    header.text((14, 66), "major-mode local path=minor-axis chord; minor-mode local path=major-axis chord", fill=(20, 20, 20), font=_font(13))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(overlay_path)
    canvas.save(local_overlay_path)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _copy_if_exists(src: str | Path | None, dst: str | Path) -> str:
    if not src or str(src) == NA:
        return NA
    source = Path(src)
    target = Path(dst)
    if not source.exists():
        return NA
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return str(target)


def _load_mask(mask_path: str | Path | None, image_size: tuple[int, int]) -> np.ndarray | None:
    if not mask_path or str(mask_path) == NA:
        return None
    p = Path(mask_path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".npy":
        mask = np.load(p)
    else:
        mask = np.asarray(Image.open(p).convert("L"))
    if mask.shape[:2] != (image_size[1], image_size[0]):
        mask = cv2.resize(mask.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST)
    return mask > 0


def resolve_selected_mask_path(final_result: dict[str, Any], verification: dict[str, Any], pool_summary: dict[str, Any]) -> str:
    for key in ("selected_candidate_mask_path", "selected_target_mask_path", "mask_path"):
        value = final_result.get(key) or verification.get(key)
        if value and str(value) != NA and Path(value).exists():
            return str(value)
    selected_id = final_result.get("selected_candidate_id") or verification.get("selected_candidate_id")
    for candidate in pool_summary.get("candidates", []):
        if candidate.get("candidate_id") == selected_id:
            value = candidate.get("mask_path")
            if value and Path(value).exists():
                return str(value)
    return NA


def _fake_trial_data(attempt_dir: Path, case: T7Case) -> dict[str, Any]:
    width, height = 640, 480
    rgb = np.full((height, width, 3), 234, dtype=np.uint8)
    colors = {
        "brick": (178, 88, 62),
        "timber": (142, 98, 48),
        "concrete block": (150, 154, 150),
    }
    color = colors.get(case.material_query, (120, 120, 120))
    angle = {"0deg": 0, "30deg": 30, "60deg": 60, "90deg": 90}.get(case.orientation_case, 18)
    center = (width // 2, height // 2)
    size = (250, 105)
    rect = (center, size, float(angle))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(rgb, [box], color)
    cv2.polylines(rgb, [box], True, (30, 30, 30), 3)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    depth = np.full((height, width), 0.72, dtype=np.float32)
    depth[mask > 0] = 0.70
    camera_info = {"fx": 610.0, "fy": 610.0, "cx": width / 2.0, "cy": height / 2.0, "width": width, "height": height}

    snapshot = attempt_dir / "ros2_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    raw_rgb = attempt_dir / "raw_rgb.png"
    depth_vis = attempt_dir / "depth_visualization.png"
    depth_raw = snapshot / "depth_raw.npy"
    camera_info_path = snapshot / "camera_info.json"
    Image.fromarray(rgb).save(raw_rgb)
    Image.fromarray(rgb).save(snapshot / "raw_rgb.png")
    np.save(depth_raw, depth)
    np.save(attempt_dir / "depth_raw.npy", depth)
    Image.fromarray(np.clip((depth - float(depth.min())) / max(1e-6, float(depth.max() - depth.min())) * 255, 0, 255).astype(np.uint8)).save(depth_vis)
    shutil.copyfile(depth_vis, snapshot / "depth_visualization.png")
    write_json(camera_info_path, camera_info)
    write_json(snapshot / "rgbd_snapshot_metadata.json", {"dry_run": True, "direct_realsense_access": False})
    mask_path = attempt_dir / "selected_mask.png"
    Image.fromarray(mask).save(mask_path)
    return {
        "raw_rgb_path": str(raw_rgb),
        "depth_raw_path": str(depth_raw),
        "depth_visualization_path": str(depth_vis),
        "camera_info_path": str(camera_info_path),
        "mask_path": str(mask_path),
        "final_result": {
            "valid_target": True,
            "selected_candidate_id": f"candidate_001_{_slug(case.material_query)}",
            "selected_source_detection_query": case.material_query,
            "selected_candidate_mask_path": str(mask_path),
        },
        "pool_summary": {"candidate_count": 1, "detection_queries_used": [case.material_query], "candidates": [{"candidate_id": f"candidate_001_{_slug(case.material_query)}", "mask_path": str(mask_path)}]},
        "verification": {"selected_candidate_id": f"candidate_001_{_slug(case.material_query)}", "material_verification_method": "dry_run_synthetic_clip_crop_verifier"},
    }


def _save_mask_dimension_overlay(attempt_dir: Path, rgb_path: Path, mask: np.ndarray, geom: dict[str, Any], mask_dims: dict[str, Any]) -> dict[str, str]:
    raw = Image.open(rgb_path).convert("RGB")
    overlay = raw.convert("RGBA")
    mask_img = Image.fromarray((np.asarray(mask).astype(bool).astype(np.uint8) * 255), mode="L")
    tint = Image.new("RGBA", raw.size, (255, 205, 0, 90))
    overlay.paste(tint, (0, 0), mask_img)
    draw = ImageDraw.Draw(overlay)
    if geom.get("centroid_x_px") != NA:
        cx = float(geom["centroid_x_px"])
        cy = float(geom["centroid_y_px"])
        major = np.array([float(geom["major_axis_vector_x"]), float(geom["major_axis_vector_y"])])
        minor = np.array([float(geom["minor_axis_vector_x"]), float(geom["minor_axis_vector_y"])])
        for axis, color, label in [(major, (255, 40, 40, 255), "major"), (minor, (0, 160, 255, 255), "minor")]:
            p1 = (cx - axis[0] * 130, cy - axis[1] * 130)
            p2 = (cx + axis[0] * 130, cy + axis[1] * 130)
            draw.line((*p1, *p2), fill=color, width=4)
            draw.text((p2[0] + 6, p2[1] + 6), label, fill=color, font=_font(14, True))
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
    panel_h = 110
    canvas = Image.new("RGB", (raw.width, raw.height + panel_h), (246, 246, 240))
    canvas.paste(overlay.convert("RGB"), (0, panel_h))
    header = ImageDraw.Draw(canvas)
    header.text((14, 10), "T7 global major/minor object dimensions", fill=(20, 20, 20), font=_font(20, True))
    header.text((14, 42), f"mask major={_fmt(mask_dims.get('mask_major_dimension_mm'))} mm | mask minor={_fmt(mask_dims.get('mask_minor_dimension_mm'))} mm", fill=(20, 20, 20), font=_font(16))
    header.text((14, 70), f"major angle={_fmt(geom.get('major_axis_angle_deg'))} deg | minor angle={_fmt(geom.get('minor_axis_angle_deg'))} deg", fill=(20, 20, 20), font=_font(14))
    major_minor = attempt_dir / "major_minor_dimension_overlay.png"
    mask_overlay = attempt_dir / "mask_dimension_overlay.png"
    selected_overlay = attempt_dir / "selected_mask_overlay.png"
    canvas.save(major_minor)
    canvas.save(mask_overlay)
    canvas.save(selected_overlay)
    return {
        "major_minor_dimension_overlay_path": str(major_minor),
        "mask_dimension_overlay_path": str(mask_overlay),
        "selected_mask_overlay_path": str(selected_overlay),
    }


def _save_depth_axis_overlay(
    output_path: Path,
    rgb_path: Path,
    mask: np.ndarray,
    geom: dict[str, Any],
    result: dict[str, Any],
    axis_name: str,
) -> None:
    raw = Image.open(rgb_path).convert("RGB")
    overlay = raw.convert("RGBA")
    mask_img = Image.fromarray((np.asarray(mask).astype(bool).astype(np.uint8) * 255), mode="L")
    overlay.paste(Image.new("RGBA", raw.size, (0, 180, 255, 70)), (0, 0), mask_img)
    draw = ImageDraw.Draw(overlay)
    if geom.get("centroid_x_px") != NA:
        cx = float(geom["centroid_x_px"])
        cy = float(geom["centroid_y_px"])
        axis = np.array([float(geom[f"{axis_name}_axis_vector_x"]), float(geom[f"{axis_name}_axis_vector_y"])])
        p1 = (cx - axis[0] * 145, cy - axis[1] * 145)
        p2 = (cx + axis[0] * 145, cy + axis[1] * 145)
        draw.line((*p1, *p2), fill=(255, 255, 255, 255), width=5)
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 60, 60, 255))
    panel_h = 122
    canvas = Image.new("RGB", (raw.width, raw.height + panel_h), (246, 246, 240))
    canvas.paste(overlay.convert("RGB"), (0, panel_h))
    header = ImageDraw.Draw(canvas)
    header.text((14, 10), f"T7 depth-refined {axis_name} global dimension", fill=(20, 20, 20), font=_font(20, True))
    header.text((14, 42), f"depth {axis_name}={_fmt(result.get('dimension_mm'))} mm | valid={result.get('valid')}", fill=(20, 20, 20), font=_font(16))
    header.text((14, 68), f"top-face points={result.get('top_face_points_count', NA)} | coverage={_fmt(result.get('coverage'), 3)} | MAD={_fmt(result.get('mad_m'), 5)} m", fill=(20, 20, 20), font=_font(14))
    header.text((14, 92), f"failure={result.get('failure_reason', NA)}", fill=(20, 20, 20), font=_font(13))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _depth_refined_axis_dimension(
    axis_name: str,
    mask: np.ndarray,
    depth: np.ndarray,
    camera_info: dict[str, Any],
    geom: dict[str, Any],
    mask_dimension_mm: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    if not geom.get("valid"):
        return {
            "axis_name": axis_name,
            "valid": False,
            "dimension_mm": NA,
            "failure_reason": "mask_geometry_invalid",
            "top_face_points_count": 0,
            "coverage": NA,
            "mad_m": NA,
            "diagnostics": {},
            "raw_depth_width_result": {},
            "time_ms": _elapsed_ms(start),
            "mapping_note": "T7 depth dimension was not computed because mask geometry was invalid.",
        }
    major = [float(geom["major_axis_vector_x"]), float(geom["major_axis_vector_y"])]
    minor = [float(geom["minor_axis_vector_x"]), float(geom["minor_axis_vector_y"])]
    depth_cfg = dict(config.get("depth_width", {}))
    depth_cfg["max_width_disagreement_ratio"] = float(depth_cfg.get("max_depth_mask_disagreement_ratio_for_t7_sanity", 0.50))
    depth_cfg["centroid_px"] = [float(geom["centroid_x_px"]), float(geom["centroid_y_px"])]
    depth_cfg["width_semantics"] = f"T7 global {axis_name}-axis object dimension"
    selected_upv_axis = "minor" if axis_name == "major" else "major"
    mask_value = _safe_float(mask_dimension_mm) or 0.0
    result = estimate_depth_refined_width(
        mask=np.asarray(mask).astype(bool),
        depth_image=np.asarray(depth),
        camera_info=camera_info,
        major_axis_unit_px=major,
        minor_axis_unit_px=minor,
        selected_upv_axis=selected_upv_axis,
        mask_projected_width_mm=mask_value,
        config=depth_cfg,
    )
    payload = result.to_dict()
    return {
        "axis_name": axis_name,
        "valid": bool(payload.get("valid")),
        "dimension_mm": payload.get("depth_refined_width_mm", NA),
        "failure_reason": payload.get("failure_reason") or NA,
        "top_face_points_count": payload.get("top_face_points_count", 0),
        "coverage": payload.get("depth_coverage_ratio", NA),
        "mad_m": payload.get("top_face_depth_mad_m", NA),
        "diagnostics": payload.get("diagnostics", {}),
        "raw_depth_width_result": payload,
        "time_ms": _elapsed_ms(start),
        "mapping_note": f"T7 {axis_name} dimension uses depth_width_refinement with selected_upv_axis={selected_upv_axis} only to select the {axis_name} projection axis; no anchor/UPV path is selected.",
    }


def _compute_errors(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("mask", "depth"):
        for axis in ("major", "minor"):
            estimate = _safe_float(row.get(f"{method}_{axis}_dimension_mm"))
            manual = _safe_float(row.get(f"manual_{axis}_dimension_mm"))
            if estimate is None or manual is None or manual == 0:
                out[f"{method}_{axis}_error_mm"] = NA
                out[f"{method}_{axis}_error_pct"] = NA
            else:
                err = estimate - manual
                out[f"{method}_{axis}_error_mm"] = round(err, 3)
                out[f"{method}_{axis}_error_pct"] = round(abs(err) / manual * 100.0, 3)
    for axis in ("major", "minor"):
        m_err = abs(_safe_float(out.get(f"mask_{axis}_error_mm")) or float("nan"))
        d_err = abs(_safe_float(out.get(f"depth_{axis}_error_mm")) or float("nan"))
        out[f"depth_closer_than_mask_{axis}"] = bool(np.isfinite(d_err) and np.isfinite(m_err) and d_err < m_err)
    for mode in ("major", "minor"):
        manual = _safe_float(row.get(f"manual_anchor_{mode}_mode_local_path_length_mm"))
        for method in ("mask", "depth"):
            estimate = _safe_float(row.get(f"anchor_{mode}_mode_local_{method}_path_length_mm"))
            if estimate is None or manual is None or manual == 0:
                out[f"anchor_{mode}_mode_local_{method}_error_mm"] = NA
                out[f"anchor_{mode}_mode_local_{method}_error_pct"] = NA
            else:
                err = estimate - manual
                out[f"anchor_{mode}_mode_local_{method}_error_mm"] = round(err, 3)
                out[f"anchor_{mode}_mode_local_{method}_error_pct"] = round(abs(err) / manual * 100.0, 3)
    return out


def _prompt_manual_dimensions(no_manual: bool, defaults: dict[str, Any]) -> dict[str, Any]:
    if no_manual:
        return {
            "manual_major_dimension_mm": defaults.get("manual_major_dimension_mm", NA),
            "manual_minor_dimension_mm": defaults.get("manual_minor_dimension_mm", NA),
            "manual_anchor_major_mode_local_path_length_mm": defaults.get("manual_anchor_major_mode_local_path_length_mm", NA),
            "manual_anchor_minor_mode_local_path_length_mm": defaults.get("manual_anchor_minor_mode_local_path_length_mm", NA),
            "measurement_tool": "dry_run_caliper",
            "measurement_confidence": "high",
            "mask_usable_manual": "yes",
            "manual_notes": "dry-run synthetic manual dimensions",
        }
    major = input("Manual major-axis dimension in mm: ").strip()
    minor = input("Manual minor-axis dimension in mm: ").strip()
    anchor_major = input("Manual major-mode local path length in mm (blank=NA): ").strip()
    anchor_minor = input("Manual minor-mode local path length in mm (blank=NA): ").strip()
    tool = input("Measurement tool [caliper]: ").strip() or "caliper"
    usable = input("Mask usable? [y/n]: ").strip() or "y"
    confidence = input("Measurement confidence [high]: ").strip() or "high"
    notes = input("Notes: ").strip() or NA
    return {
        "manual_major_dimension_mm": major or NA,
        "manual_minor_dimension_mm": minor or NA,
        "manual_anchor_major_mode_local_path_length_mm": anchor_major or NA,
        "manual_anchor_minor_mode_local_path_length_mm": anchor_minor or NA,
        "measurement_tool": tool,
        "measurement_confidence": confidence,
        "mask_usable_manual": usable,
        "manual_notes": notes,
    }


def _next_attempt_dir(session_dir: Path, case: T7Case) -> tuple[Path, int]:
    case_dir = session_dir / "cases" / case.folder_name()
    attempts_root = case_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts_root.glob("attempt_*"))
    idx = len(existing) + 1
    attempt = attempts_root / f"attempt_{idx:03d}"
    attempt.mkdir(parents=True, exist_ok=True)
    return attempt, idx


def _existing_saved_case_ids(session_dir: Path) -> set[str]:
    return {row.get("case_id", "") for row in _read_rows(session_dir / "T7_master_results.csv")}


def _capture_or_fake(attempt_dir: Path, case: T7Case, config: dict[str, Any], *, dry_run: bool, live: bool) -> dict[str, Any]:
    capture_start = time.perf_counter()
    if dry_run:
        payload = _fake_trial_data(attempt_dir, case)
        payload["capture_time_ms"] = _elapsed_ms(capture_start)
        payload["perception_time_ms"] = 0.0
        return payload
    if not live:
        raise RuntimeError("T7 non-dry trial requires --live. Use --dry-run for no-camera validation.")
    snap = capture_ros2_snapshot(attempt_dir, config)
    payload = {
        "raw_rgb_path": snap["color_path"],
        "depth_raw_path": snap["depth_npy_path"],
        "depth_visualization_path": snap["depth_png_path"],
        "camera_info_path": snap["intrinsics_path"],
        "capture_time_ms": _elapsed_ms(capture_start),
    }
    _copy_if_exists(snap["depth_npy_path"], attempt_dir / "depth_raw.npy")
    perception_start = time.perf_counter()
    final_result, pool_summary, verification = run_candidate_pool_perception(
        attempt_dir,
        payload["raw_rgb_path"],
        Path(payload["depth_raw_path"]),
        case.to_t5_trial(),
        config,
    )
    mask_path = resolve_selected_mask_path(final_result, verification, pool_summary)
    payload.update(
        {
            "mask_path": mask_path,
            "final_result": final_result,
            "pool_summary": pool_summary,
            "verification": verification,
            "perception_time_ms": _elapsed_ms(perception_start),
        }
    )
    return payload


def run_attempt(session_dir: Path, case: T7Case, config: dict[str, Any], *, dry_run: bool, live: bool, no_manual: bool) -> dict[str, Any]:
    total_start = time.perf_counter()
    attempt_dir, attempt_index = _next_attempt_dir(session_dir, case)
    write_json(attempt_dir / "trial_metadata.json", case.__dict__)

    payload = _capture_or_fake(attempt_dir, case, config, dry_run=dry_run, live=live)
    raw_rgb_path = Path(payload["raw_rgb_path"])
    depth_path = Path(payload["depth_raw_path"])
    camera_info_path = Path(payload["camera_info_path"])
    final_result = payload.get("final_result", {})
    target_selected = bool(final_result.get("valid_target", False)) and payload.get("mask_path") not in (None, NA, "")
    mask = _load_mask(payload.get("mask_path"), Image.open(raw_rgb_path).size)
    if mask is None:
        mask = np.zeros((Image.open(raw_rgb_path).height, Image.open(raw_rgb_path).width), dtype=bool)
        target_selected = False
    selected_mask_root = attempt_dir / "selected_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(selected_mask_root)
    payload["mask_path"] = str(selected_mask_root)

    geom_start = time.perf_counter()
    geom = _mask_geometry(mask)
    mask_geometry_time_ms = _elapsed_ms(geom_start)

    depth = np.load(depth_path)
    camera_info = _load_camera_info(camera_info_path)
    depth_cfg = config.get("depth_width", {})
    major_axis = np.array([float(geom.get("major_axis_vector_x", 1.0) if geom.get("major_axis_vector_x") != NA else 1.0), float(geom.get("major_axis_vector_y", 0.0) if geom.get("major_axis_vector_y") != NA else 0.0)])
    minor_axis = np.array([float(geom.get("minor_axis_vector_x", 0.0) if geom.get("minor_axis_vector_x") != NA else 0.0), float(geom.get("minor_axis_vector_y", 1.0) if geom.get("minor_axis_vector_y") != NA else 1.0)])

    major_mask = _axis_mask_dimension_mm(mask, depth, camera_info, major_axis, depth_cfg)
    minor_mask = _axis_mask_dimension_mm(mask, depth, camera_info, minor_axis, depth_cfg)
    mask_total_dimension_time_ms = round(float(major_mask["time_ms"]) + float(minor_mask["time_ms"]), 3)
    mask_dims = {
        "mask_major_dimension_mm": round(float(major_mask["dimension_mm"]), 3) if major_mask["valid"] else NA,
        "mask_minor_dimension_mm": round(float(minor_mask["dimension_mm"]), 3) if minor_mask["valid"] else NA,
        "mask_major_valid": bool(major_mask["valid"]),
        "mask_minor_valid": bool(minor_mask["valid"]),
        "mask_major_failure_reason": major_mask["failure_reason"],
        "mask_minor_failure_reason": minor_mask["failure_reason"],
    }

    depth_major = _depth_refined_axis_dimension("major", mask, depth, camera_info, geom, mask_dims["mask_major_dimension_mm"], config)
    depth_minor = _depth_refined_axis_dimension("minor", mask, depth, camera_info, geom, mask_dims["mask_minor_dimension_mm"], config)
    depth_total_dimension_time_ms = round(float(depth_major["time_ms"]) + float(depth_minor["time_ms"]), 3)
    depth_dims = {
        "depth_major_dimension_mm": round(float(depth_major["dimension_mm"]), 3) if _safe_float(depth_major["dimension_mm"]) is not None else NA,
        "depth_minor_dimension_mm": round(float(depth_minor["dimension_mm"]), 3) if _safe_float(depth_minor["dimension_mm"]) is not None else NA,
        "depth_major_valid": bool(depth_major["valid"]),
        "depth_minor_valid": bool(depth_minor["valid"]),
        "depth_major_failure_reason": depth_major["failure_reason"],
        "depth_minor_failure_reason": depth_minor["failure_reason"],
        "depth_major_top_face_point_count": int(depth_major.get("top_face_points_count", 0) or 0),
        "depth_minor_top_face_point_count": int(depth_minor.get("top_face_points_count", 0) or 0),
        "depth_major_coverage": depth_major.get("coverage", NA),
        "depth_minor_coverage": depth_minor.get("coverage", NA),
        "depth_major_mad_m": depth_major.get("mad_m", NA),
        "depth_minor_mad_m": depth_minor.get("mad_m", NA),
    }

    sanity = _depth_mask_sanity(mask_dims, depth_dims, float(depth_cfg.get("max_depth_mask_disagreement_ratio_for_t7_sanity", 0.50)))
    if sanity["major_depth_mask_sanity_pass"] is False or sanity["minor_depth_mask_sanity_pass"] is False:
        print("WARNING: depth-vs-mask ±50% sanity check failed. Inspect overlays and consider retrying.")
        print(f"major ratio={sanity['major_depth_mask_disagreement_ratio']} minor ratio={sanity['minor_depth_mask_disagreement_ratio']}")

    overlays = _save_mask_dimension_overlay(attempt_dir, raw_rgb_path, mask, geom, mask_dims)
    _save_depth_axis_overlay(attempt_dir / "depth_width_major_overlay.png", raw_rgb_path, mask, geom, depth_major, "major")
    _save_depth_axis_overlay(attempt_dir / "depth_width_minor_overlay.png", raw_rgb_path, mask, geom, depth_minor, "minor")
    overlays["depth_width_major_overlay_path"] = str(attempt_dir / "depth_width_major_overlay.png")
    overlays["depth_width_minor_overlay_path"] = str(attempt_dir / "depth_width_minor_overlay.png")

    anchor_major = _run_anchor_mode(attempt_dir, "major", raw_rgb_path, mask, depth, camera_info, geom, config)
    anchor_minor = _run_anchor_mode(attempt_dir, "minor", raw_rgb_path, mask, depth, camera_info, geom, config)
    anchor_total_selection_ms = round(float(anchor_major.get("anchor_selection_major_time_ms", 0.0)) + float(anchor_minor.get("anchor_selection_minor_time_ms", 0.0)), 3)
    anchor_local_mask_total_ms = 0.0
    anchor_local_depth_total_ms = round(float(anchor_major.get("anchor_major_mode_local_depth_time_ms", 0.0)) + float(anchor_minor.get("anchor_minor_mode_local_depth_time_ms", 0.0)), 3)
    anchor_timings = {
        "anchor_selection_major_time_ms": anchor_major.get("anchor_selection_major_time_ms", 0.0),
        "anchor_selection_minor_time_ms": anchor_minor.get("anchor_selection_minor_time_ms", 0.0),
        "anchor_selection_total_time_ms": anchor_total_selection_ms,
        "anchor_local_mask_major_mode_time_ms": anchor_local_mask_total_ms,
        "anchor_local_mask_minor_mode_time_ms": anchor_local_mask_total_ms,
        "anchor_local_mask_total_time_ms": anchor_local_mask_total_ms,
        "anchor_local_depth_major_mode_time_ms": anchor_major.get("anchor_major_mode_local_depth_time_ms", 0.0),
        "anchor_local_depth_minor_mode_time_ms": anchor_minor.get("anchor_minor_mode_local_depth_time_ms", 0.0),
        "anchor_local_depth_total_time_ms": anchor_local_depth_total_ms,
    }
    anchor_major.pop("anchor_major_mode_local_depth_time_ms", None)
    anchor_minor.pop("anchor_minor_mode_local_depth_time_ms", None)
    overlays.update({key: value for key, value in {**anchor_major, **anchor_minor}.items() if key.endswith("_path")})

    defaults = {}
    if dry_run:
        defaults = {
            "manual_major_dimension_mm": round(float(mask_dims["mask_major_dimension_mm"]) + 2.0, 3) if _safe_float(mask_dims["mask_major_dimension_mm"]) is not None else NA,
            "manual_minor_dimension_mm": round(float(mask_dims["mask_minor_dimension_mm"]) - 1.0, 3) if _safe_float(mask_dims["mask_minor_dimension_mm"]) is not None else NA,
            "manual_anchor_major_mode_local_path_length_mm": round(float(anchor_major["anchor_major_mode_local_mask_path_length_mm"]) + 1.0, 3) if _safe_float(anchor_major.get("anchor_major_mode_local_mask_path_length_mm")) is not None else NA,
            "manual_anchor_minor_mode_local_path_length_mm": round(float(anchor_minor["anchor_minor_mode_local_mask_path_length_mm"]) - 1.0, 3) if _safe_float(anchor_minor.get("anchor_minor_mode_local_mask_path_length_mm")) is not None else NA,
        }
    total_so_far_ms = _elapsed_ms(total_start)
    print("\nT7 dimension estimates:")
    print(f"Global mask major/minor: {_fmt(mask_dims['mask_major_dimension_mm'])} mm / {_fmt(mask_dims['mask_minor_dimension_mm'])} mm")
    print(f"Global depth major/minor: {_fmt(depth_dims['depth_major_dimension_mm'])} mm / {_fmt(depth_dims['depth_minor_dimension_mm'])} mm")
    print("\nAnchor-selection:")
    print(f"Major-mode selected anchor: {anchor_major.get('major_mode_final_anchor_id')} at (x={_fmt(anchor_major.get('major_mode_anchor_x_px'))}, y={_fmt(anchor_major.get('major_mode_anchor_y_px'))}), score={_fmt(anchor_major.get('major_mode_anchor_score'))}")
    print(f"Major-mode local path length: mask={_fmt(anchor_major.get('anchor_major_mode_local_mask_path_length_mm'))} mm, depth={_fmt(anchor_major.get('anchor_major_mode_local_depth_path_length_mm'))} mm, valid={anchor_major.get('anchor_major_mode_local_depth_valid')}")
    print(f"Minor-mode selected anchor: {anchor_minor.get('minor_mode_final_anchor_id')} at (x={_fmt(anchor_minor.get('minor_mode_anchor_x_px'))}, y={_fmt(anchor_minor.get('minor_mode_anchor_y_px'))}), score={_fmt(anchor_minor.get('minor_mode_anchor_score'))}")
    print(f"Minor-mode local path length: mask={_fmt(anchor_minor.get('anchor_minor_mode_local_mask_path_length_mm'))} mm, depth={_fmt(anchor_minor.get('anchor_minor_mode_local_depth_path_length_mm'))} mm, valid={anchor_minor.get('anchor_minor_mode_local_depth_valid')}")
    print("\nDepth-vs-mask sanity:")
    print(f"Global major/minor: {sanity['major_depth_mask_sanity_pass']}/{sanity['minor_depth_mask_sanity_pass']}, disagreement={sanity['major_depth_mask_disagreement_ratio']}/{sanity['minor_depth_mask_disagreement_ratio']}")
    print(f"Major-mode local: {anchor_major.get('anchor_major_mode_depth_mask_sanity_pass')}, disagreement={anchor_major.get('anchor_major_mode_depth_mask_disagreement_ratio')}")
    print(f"Minor-mode local: {anchor_minor.get('anchor_minor_mode_depth_mask_sanity_pass')}, disagreement={anchor_minor.get('anchor_minor_mode_depth_mask_disagreement_ratio')}")
    print("\nTiming:")
    print(f"Capture: {payload.get('capture_time_ms', NA)} ms")
    print(f"Perception: {payload.get('perception_time_ms', NA)} ms")
    print(f"Mask geometry: {mask_geometry_time_ms} ms")
    print(f"Global mask dimensions: {mask_total_dimension_time_ms} ms")
    print(f"Global depth dimensions: {depth_total_dimension_time_ms} ms")
    print(f"Anchor selection major/minor/total: {anchor_timings['anchor_selection_major_time_ms']} / {anchor_timings['anchor_selection_minor_time_ms']} / {anchor_timings['anchor_selection_total_time_ms']} ms")
    print(f"Anchor-local depth major/minor/total: {anchor_timings['anchor_local_depth_major_mode_time_ms']} / {anchor_timings['anchor_local_depth_minor_mode_time_ms']} / {anchor_timings['anchor_local_depth_total_time_ms']} ms")
    print(f"Total attempt so far: {total_so_far_ms} ms")
    manual = _prompt_manual_dimensions(no_manual or dry_run, defaults)

    row = {
        "session_id": session_dir.name,
        "case_index": case.trial_index,
        "case_id": case.case_id,
        "material_query": case.material_query,
        "expected_material": case.expected_material,
        "object_id": case.object_id,
        "orientation_case": case.orientation_case,
        "attempt_index": attempt_index,
        "timestamp": now_iso(),
        "target_selected": target_selected,
        "final_selected_candidate_id": final_result.get("selected_candidate_id", NA),
        "selected_candidate_mask_path": str(payload.get("mask_path", NA)),
        "raw_rgb_path": str(raw_rgb_path),
        "depth_raw_path": str(depth_path),
        "camera_info_path": str(camera_info_path),
        **overlays,
        **geom,
        **mask_dims,
        **depth_dims,
        **sanity,
        **anchor_major,
        **anchor_minor,
        **manual,
        "capture_time_ms": payload.get("capture_time_ms", NA),
        "perception_time_ms": payload.get("perception_time_ms", NA),
        "mask_geometry_time_ms": mask_geometry_time_ms,
        "mask_major_dimension_time_ms": major_mask["time_ms"],
        "mask_minor_dimension_time_ms": minor_mask["time_ms"],
        "mask_total_dimension_time_ms": mask_total_dimension_time_ms,
        "depth_major_dimension_time_ms": depth_major["time_ms"],
        "depth_minor_dimension_time_ms": depth_minor["time_ms"],
        "depth_total_dimension_time_ms": depth_total_dimension_time_ms,
        **anchor_timings,
        "total_attempt_time_ms": _elapsed_ms(total_start),
    }
    row.update(_compute_errors(row))
    write_json(attempt_dir / "dimension_summary.json", row | {"depth_major_diagnostics": depth_major, "depth_minor_diagnostics": depth_minor})
    write_json(attempt_dir / "manual_measurements.json", manual)
    write_json(attempt_dir / "timing_summary.json", {key: row.get(key) for key in T7_COLUMNS if key.endswith("_time_ms") or key == "total_attempt_time_ms"})
    return row | {"attempt_dir": str(attempt_dir)}


def _depth_mask_sanity(mask_dims: dict[str, Any], depth_dims: dict[str, Any], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis in ("major", "minor"):
        mask_v = _safe_float(mask_dims.get(f"mask_{axis}_dimension_mm"))
        depth_v = _safe_float(depth_dims.get(f"depth_{axis}_dimension_mm"))
        if mask_v is None or mask_v <= 0 or depth_v is None:
            ratio = NA
            passed: bool | str = NA
        else:
            ratio = round(abs(depth_v - mask_v) / mask_v, 4)
            passed = bool(ratio <= threshold)
        out[f"{axis}_depth_mask_disagreement_ratio"] = ratio
        out[f"{axis}_depth_mask_sanity_pass"] = passed
    return out


def _prompt_save_action(dry_run: bool) -> str:
    if dry_run:
        return "s"
    action = input("Save this trial result? [s]ave / [r]etry / [q]uit: ").strip().lower()
    return action or "s"


def save_attempt_log(session_dir: Path, row: dict[str, Any], status: str, discard_reason: str = NA) -> None:
    attempt_row = {
        "session_id": session_dir.name,
        "case_index": row.get("case_index", NA),
        "case_id": row.get("case_id", NA),
        "material_query": row.get("material_query", NA),
        "attempt_index": row.get("attempt_index", NA),
        "attempt_status": status,
        "attempt_dir": row.get("attempt_dir", NA),
        "target_selected": row.get("target_selected", NA),
        "final_selected_candidate_id": row.get("final_selected_candidate_id", NA),
        "mask_major_dimension_mm": row.get("mask_major_dimension_mm", NA),
        "mask_minor_dimension_mm": row.get("mask_minor_dimension_mm", NA),
        "depth_major_dimension_mm": row.get("depth_major_dimension_mm", NA),
        "depth_minor_dimension_mm": row.get("depth_minor_dimension_mm", NA),
        "discard_reason": discard_reason,
        "timestamp": now_iso(),
    }
    _append_csv(session_dir / "T7_attempt_log.csv", ATTEMPT_COLUMNS, attempt_row)


def save_official_result(session_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    existing = _read_rows(session_dir / "T7_master_results.csv")
    row = dict(row)
    row["saved_trial_number"] = len(existing) + 1
    _append_csv(session_dir / "T7_master_results.csv", T7_COLUMNS, row)
    _copy_if_exists(row.get("major_minor_dimension_overlay_path"), session_dir / "paper_figures" / f"{row['case_id']}_major_minor_dimension_overlay.png")
    _copy_if_exists(row.get("depth_width_major_overlay_path"), session_dir / "paper_figures" / f"{row['case_id']}_depth_width_major_overlay.png")
    _copy_if_exists(row.get("depth_width_minor_overlay_path"), session_dir / "paper_figures" / f"{row['case_id']}_depth_width_minor_overlay.png")
    return row


def run_case_with_retry(session_dir: Path, case: T7Case, config: dict[str, Any], *, dry_run: bool, live: bool, no_manual: bool) -> str:
    while True:
        print("\n" + "=" * 60)
        print(f"T7 Width/Path-Length Validation Case {case.trial_index}/15: {case.case_id}")
        print(f"Requested material: {case.material_query}")
        print("Single-object scene only. No robot, clamp, RTDE, Arduino, or UPV measurement.")
        if not dry_run:
            arranged = input("Place the single object under the camera. Press ENTER to capture, or type skip: ").strip().lower()
            if arranged == "skip":
                save_attempt_log(session_dir, case.__dict__ | {"attempt_index": NA, "attempt_dir": NA, "target_selected": False}, "skipped", "operator_skip")
                return "skipped"
        row = run_attempt(session_dir, case, config, dry_run=dry_run, live=live, no_manual=no_manual)
        action = _prompt_save_action(dry_run)
        if action.startswith("s"):
            saved = save_official_result(session_dir, row)
            save_attempt_log(session_dir, saved, "saved")
            write_case_progress(session_dir, config)
            refresh_session_outputs(session_dir, config, generate_figures=bool(config.get("plotting", {}).get("enabled", True)))
            return "saved"
        if action.startswith("q"):
            save_attempt_log(session_dir, row, "quit_before_save", "operator_quit")
            return "quit"
        save_attempt_log(session_dir, row, "discarded_retry", "operator_retry")
        print("Attempt discarded from official results. Retrying same case.")


def write_case_progress(session_dir: Path, config: dict[str, Any]) -> None:
    cases = read_cases(config["cases_csv"])
    saved = _existing_saved_case_ids(session_dir)
    rows = []
    for case in cases:
        rows.append({
            "case_index": case.trial_index,
            "case_id": case.case_id,
            "material_query": case.material_query,
            "enabled": case.enabled,
            "saved": case.case_id in saved,
        })
    _write_csv(session_dir / "T7_case_progress.csv", ["case_index", "case_id", "material_query", "enabled", "saved"], rows)


def run_cases(session_dir: Path, cases: list[T7Case], config: dict[str, Any], *, dry_run: bool, live: bool, no_manual: bool) -> None:
    saved = _existing_saved_case_ids(session_dir)
    for case in [c for c in cases if c.enabled]:
        if case.case_id in saved:
            print(f"Skipping already saved case: {case.case_id}")
            continue
        result = run_case_with_retry(session_dir, case, config, dry_run=dry_run, live=live, no_manual=no_manual)
        if result == "quit":
            break
    write_case_progress(session_dir, config)
    refresh_session_outputs(session_dir, config, generate_figures=bool(config.get("plotting", {}).get("enabled", True)))


def _mean(values: list[float]) -> float | str:
    values = [v for v in values if math.isfinite(v)]
    return round(float(np.mean(values)), 4) if values else NA


def _abs_values(rows: list[dict[str, Any]], keys: list[str]) -> list[float]:
    vals = []
    for row in rows:
        for key in keys:
            value = _safe_float(row.get(key))
            if value is not None:
                vals.append(abs(value))
    return vals


def summarize_session(session_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    session = resolve_repo_path(session_dir)
    rows = _read_rows(session / "T7_master_results.csv")
    saved_by_material: dict[str, int] = {}
    for row in rows:
        mat = row.get("material_query", NA)
        saved_by_material[mat] = saved_by_material.get(mat, 0) + 1
    mask_major = _abs_values(rows, ["mask_major_error_mm"])
    mask_minor = _abs_values(rows, ["mask_minor_error_mm"])
    depth_major = _abs_values(rows, ["depth_major_error_mm"])
    depth_minor = _abs_values(rows, ["depth_minor_error_mm"])
    mask_all = mask_major + mask_minor
    depth_all = depth_major + depth_minor
    summary = {
        "session_dir": str(session),
        "total_saved_trials": len(rows),
        "expected_trials": 15,
        "saved_by_material": saved_by_material,
        "mean_abs_error_mask_major_mm": _mean(mask_major),
        "mean_abs_error_mask_minor_mm": _mean(mask_minor),
        "mean_abs_error_depth_major_mm": _mean(depth_major),
        "mean_abs_error_depth_minor_mm": _mean(depth_minor),
        "mean_abs_error_mask_all_mm": _mean(mask_all),
        "mean_abs_error_depth_all_mm": _mean(depth_all),
        "mean_percent_error_mask_all": _mean([abs(_safe_float(r.get(k)) or float("nan")) for r in rows for k in ("mask_major_error_pct", "mask_minor_error_pct")]),
        "mean_percent_error_depth_all": _mean([abs(_safe_float(r.get(k)) or float("nan")) for r in rows for k in ("depth_major_error_pct", "depth_minor_error_pct")]),
        "max_abs_error_mask_all_mm": round(max(mask_all), 4) if mask_all else NA,
        "max_abs_error_depth_all_mm": round(max(depth_all), 4) if depth_all else NA,
        "depth_valid_major_count": sum(str(r.get("depth_major_valid")).lower() == "true" for r in rows),
        "depth_valid_minor_count": sum(str(r.get("depth_minor_valid")).lower() == "true" for r in rows),
        "depth_valid_all_count": sum(str(r.get(k)).lower() == "true" for r in rows for k in ("depth_major_valid", "depth_minor_valid")),
        "depth_mask_sanity_pass_major_count": sum(str(r.get("major_depth_mask_sanity_pass")).lower() == "true" for r in rows),
        "depth_mask_sanity_pass_minor_count": sum(str(r.get("minor_depth_mask_sanity_pass")).lower() == "true" for r in rows),
        "depth_closer_than_mask_major_count": sum(str(r.get("depth_closer_than_mask_major")).lower() == "true" for r in rows),
        "depth_closer_than_mask_minor_count": sum(str(r.get("depth_closer_than_mask_minor")).lower() == "true" for r in rows),
        "mean_mask_total_dimension_time_ms": _mean([_safe_float(r.get("mask_total_dimension_time_ms")) or float("nan") for r in rows]),
        "mean_depth_total_dimension_time_ms": _mean([_safe_float(r.get("depth_total_dimension_time_ms")) or float("nan") for r in rows]),
        "robot_used": False,
        "clamp_used": False,
        "rtde_used": False,
        "arduino_used": False,
        "upv_measurement_used": False,
    }
    write_json(session / "T7_session_summary.json", summary)
    write_paper_tables(session, rows)
    if config is None or bool(config.get("plotting", {}).get("enabled", True)):
        generate_figures(session)
    return summary


def write_paper_tables(session: Path, rows: list[dict[str, Any]]) -> None:
    by_material: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_material.setdefault(row.get("material_query", NA), []).append(row)
    material_rows = []
    for material, group in sorted(by_material.items()):
        material_rows.append({
            "material_query": material,
            "n": len(group),
            "mean_abs_mask_error_mm": _mean(_abs_values(group, ["mask_major_error_mm", "mask_minor_error_mm"])),
            "mean_abs_depth_error_mm": _mean(_abs_values(group, ["depth_major_error_mm", "depth_minor_error_mm"])),
        })
    _write_csv(session / "paper_tables" / "T7_summary_by_material.csv", ["material_query", "n", "mean_abs_mask_error_mm", "mean_abs_depth_error_mm"], material_rows)

    axis_rows = []
    for axis in ("major", "minor"):
        axis_rows.append({
            "axis": axis,
            "mean_abs_mask_error_mm": _mean(_abs_values(rows, [f"mask_{axis}_error_mm"])),
            "mean_abs_depth_error_mm": _mean(_abs_values(rows, [f"depth_{axis}_error_mm"])),
            "depth_valid_count": sum(str(r.get(f"depth_{axis}_valid")).lower() == "true" for r in rows),
            "depth_mask_sanity_pass_count": sum(str(r.get(f"{axis}_depth_mask_sanity_pass")).lower() == "true" for r in rows),
        })
    _write_csv(session / "paper_tables" / "T7_axis_error_summary.csv", ["axis", "mean_abs_mask_error_mm", "mean_abs_depth_error_mm", "depth_valid_count", "depth_mask_sanity_pass_count"], axis_rows)

    method_rows = [
        {"method": "mask", "mean_abs_error_mm": _mean(_abs_values(rows, ["mask_major_error_mm", "mask_minor_error_mm"])), "mean_runtime_ms": _mean([_safe_float(r.get("mask_total_dimension_time_ms")) or float("nan") for r in rows])},
        {"method": "depth_refined", "mean_abs_error_mm": _mean(_abs_values(rows, ["depth_major_error_mm", "depth_minor_error_mm"])), "mean_runtime_ms": _mean([_safe_float(r.get("depth_total_dimension_time_ms")) or float("nan") for r in rows])},
    ]
    _write_csv(session / "paper_tables" / "T7_method_comparison_summary.csv", ["method", "mean_abs_error_mm", "mean_runtime_ms"], method_rows)

    anchor_rows = []
    for row in rows:
        for mode in ("major", "minor"):
            anchor_rows.append(
                {
                    "case_id": row.get("case_id", NA),
                    "material_query": row.get("material_query", NA),
                    "axis_mode": mode,
                    "final_anchor_id": row.get(f"{mode}_mode_final_anchor_id", NA),
                    "local_mask_path_length_mm": row.get(f"anchor_{mode}_mode_local_mask_path_length_mm", NA),
                    "local_depth_path_length_mm": row.get(f"anchor_{mode}_mode_local_depth_path_length_mm", NA),
                    "manual_local_path_length_mm": row.get(f"manual_anchor_{mode}_mode_local_path_length_mm", NA),
                    "mask_error_mm": row.get(f"anchor_{mode}_mode_local_mask_error_mm", NA),
                    "depth_error_mm": row.get(f"anchor_{mode}_mode_local_depth_error_mm", NA),
                    "depth_mask_sanity_pass": row.get(f"anchor_{mode}_mode_depth_mask_sanity_pass", NA),
                }
            )
    _write_csv(
        session / "paper_tables" / "T7_anchor_local_path_length_results.csv",
        ["case_id", "material_query", "axis_mode", "final_anchor_id", "local_mask_path_length_mm", "local_depth_path_length_mm", "manual_local_path_length_mm", "mask_error_mm", "depth_error_mm", "depth_mask_sanity_pass"],
        anchor_rows,
    )
    anchor_summary_rows = []
    for mode in ("major", "minor"):
        anchor_summary_rows.append(
            {
                "axis_mode": mode,
                "mean_abs_local_mask_error_mm": _mean(_abs_values(rows, [f"anchor_{mode}_mode_local_mask_error_mm"])),
                "mean_abs_local_depth_error_mm": _mean(_abs_values(rows, [f"anchor_{mode}_mode_local_depth_error_mm"])),
                "selected_anchor_count": sum(str(r.get(f"{mode}_mode_anchor_selected")).lower() == "true" for r in rows),
                "depth_valid_count": sum(str(r.get(f"anchor_{mode}_mode_local_depth_valid")).lower() == "true" for r in rows),
            }
        )
    _write_csv(
        session / "paper_tables" / "T7_anchor_local_error_summary.csv",
        ["axis_mode", "mean_abs_local_mask_error_mm", "mean_abs_local_depth_error_mm", "selected_anchor_count", "depth_valid_count"],
        anchor_summary_rows,
    )


def generate_figures(session_dir: str | Path) -> None:
    session = resolve_repo_path(session_dir)
    rows = _read_rows(session / "T7_master_results.csv")
    fig_dir = session / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        write_json(session / "logs" / "figure_generation_failed.json", {"error": str(exc)})
        return

    def pairs(method: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            for axis in ("major", "minor"):
                manual = _safe_float(row.get(f"manual_{axis}_dimension_mm"))
                estimate = _safe_float(row.get(f"{method}_{axis}_dimension_mm"))
                if manual is not None and estimate is not None:
                    xs.append(manual)
                    ys.append(estimate)
        return xs, ys

    for method, filename, title in [
        ("mask", "t7_mask_vs_manual_scatter.png", "T7 mask dimensions vs manual"),
        ("depth", "t7_depth_vs_manual_scatter.png", "T7 depth-refined dimensions vs manual"),
    ]:
        xs, ys = pairs(method)
        plt.figure(figsize=(5.2, 4.4))
        plt.scatter(xs, ys, color="#2f6f9f")
        if xs and ys:
            lo, hi = min(xs + ys), max(xs + ys)
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("Manual dimension (mm)")
        plt.ylabel(f"{method} estimate (mm)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    materials = sorted(set(row.get("material_query", NA) for row in rows))
    plt.figure(figsize=(6.0, 4.2))
    data = []
    labels = []
    for mat in materials:
        vals = _abs_values([r for r in rows if r.get("material_query") == mat], ["mask_major_error_mm", "mask_minor_error_mm", "depth_major_error_mm", "depth_minor_error_mm"])
        if vals:
            data.append(vals)
            labels.append(mat)
    if data:
        plt.boxplot(data, labels=labels)
    plt.ylabel("Absolute error (mm)")
    plt.title("T7 absolute error by material")
    plt.tight_layout()
    plt.savefig(fig_dir / "t7_error_by_material_boxplot.png", dpi=180)
    plt.close()

    bar_specs = [
        ("t7_abs_error_by_axis_method_bar.png", ["mask_major_error_mm", "mask_minor_error_mm", "depth_major_error_mm", "depth_minor_error_mm"], "Mean absolute error (mm)"),
        ("t7_runtime_by_method_bar.png", ["mask_total_dimension_time_ms", "depth_total_dimension_time_ms"], "Mean runtime (ms)"),
        ("t7_depth_mask_disagreement_by_axis.png", ["major_depth_mask_disagreement_ratio", "minor_depth_mask_disagreement_ratio"], "Mean depth-mask disagreement ratio"),
    ]
    for filename, keys, ylabel in bar_specs:
        vals = []
        for key in keys:
            vals.append(_mean([abs(_safe_float(row.get(key)) or float("nan")) for row in rows]))
        plt.figure(figsize=(7.0, 4.2))
        plt.bar([k.replace("_", "\n") for k in keys], [0 if v == NA else float(v) for v in vals], color="#5c8f64")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    for method, filename, title in [
        ("mask", "t7_anchor_local_mask_vs_manual_scatter.png", "T7 anchor-local mask path length vs manual"),
        ("depth", "t7_anchor_local_depth_vs_manual_scatter.png", "T7 anchor-local depth path length vs manual"),
    ]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            for mode in ("major", "minor"):
                manual = _safe_float(row.get(f"manual_anchor_{mode}_mode_local_path_length_mm"))
                estimate = _safe_float(row.get(f"anchor_{mode}_mode_local_{method}_path_length_mm"))
                if manual is not None and estimate is not None:
                    xs.append(manual)
                    ys.append(estimate)
        plt.figure(figsize=(5.2, 4.4))
        plt.scatter(xs, ys, color="#8a5d9f")
        if xs and ys:
            lo, hi = min(xs + ys), max(xs + ys)
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("Manual anchor-local path length (mm)")
        plt.ylabel(f"{method} estimate (mm)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(5.5, 4.2))
    labels = ["major mask", "major depth", "minor mask", "minor depth"]
    values = [
        _mean(_abs_values(rows, ["anchor_major_mode_local_mask_error_mm"])),
        _mean(_abs_values(rows, ["anchor_major_mode_local_depth_error_mm"])),
        _mean(_abs_values(rows, ["anchor_minor_mode_local_mask_error_mm"])),
        _mean(_abs_values(rows, ["anchor_minor_mode_local_depth_error_mm"])),
    ]
    plt.bar(labels, [0 if v == NA else float(v) for v in values], color="#8a5d9f")
    plt.ylabel("Mean absolute local path-length error (mm)")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "t7_anchor_local_error_by_axis_mode.png", dpi=180)
    plt.close()


def refresh_session_outputs(session_dir: str | Path, config: dict[str, Any], *, generate_figures: bool = True) -> dict[str, Any]:
    summary = summarize_session(session_dir, config if generate_figures else {**config, "plotting": {"enabled": False}})
    return summary


def initialize_session(session_dir: Path, config: dict[str, Any], cases: list[T7Case]) -> None:
    write_json(session_dir / "config_snapshot.json", config)
    write_json(session_dir / "fixed15_protocol.json", {"cases": [case.__dict__ for case in cases], "single_object_scenes": True, "global_major_minor_dimensions": True, "local_anchor_line_path_length_validated": False})
    write_case_progress(session_dir, config)


def clip_env_check(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_clip_subprocess_python(config)
    env = check_clip_subprocess_environment(str(resolved["clip_subprocess_python"]), output_dir)
    return {**resolved, "environment": env}
