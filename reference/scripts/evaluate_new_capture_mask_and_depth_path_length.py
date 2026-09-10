#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO_ROOT / "outputs/path_length_ros2_stream_capture_test/session_20260702_153357"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/new_capture_mask_depth_manual_eval"

COMPARISON_COLUMNS = [
    "case_id",
    "case_index",
    "material",
    "object_id",
    "path_label",
    "manual_mm",
    "manual_source_field",
    "mask_based_mm",
    "whole_roi_depth_mm",
    "strip_depth_mm",
    "mask_abs_err_mm",
    "whole_roi_depth_abs_err_mm",
    "strip_depth_abs_err_mm",
    "mask_pct_err",
    "whole_roi_depth_pct_err",
    "strip_depth_pct_err",
    "mask_valid",
    "whole_roi_depth_valid",
    "strip_depth_valid",
    "mask_failure_reason",
    "whole_roi_depth_failure_reason",
    "strip_depth_failure_reason",
    "mask_source",
    "direction_source",
    "anchor_source",
    "p0_x",
    "p0_y",
    "direction_x",
    "direction_y",
    "selected_mask_area_px",
    "valid_depth_fraction",
    "in_range_268_375_fraction",
    "evaluation_valid",
    "invalid_reason",
]

ALL_ANCHOR_COLUMNS = [
    "case_id",
    "case_index",
    "material",
    "object_id",
    "candidate_id",
    "candidate_family",
    "selected_for_major",
    "selected_for_minor",
    "candidate_p0_x",
    "candidate_p0_y",
    "candidate_direction_x",
    "candidate_direction_y",
    "candidate_endpoint1_x",
    "candidate_endpoint1_y",
    "candidate_endpoint2_x",
    "candidate_endpoint2_y",
    "candidate_path_length_px_from_csv",
    "mask_path_length_px_recomputed",
    "mask_path_length_mm",
    "strip_depth_path_length_mm",
    "strip_depth_valid",
    "strip_depth_failure_reason",
    "strip_depth_num_supported_points",
    "strip_depth_valid_depth_fraction",
    "strip_depth_depth_median_mm",
    "strip_depth_lower_endpoint_or_quantile_mm",
    "strip_depth_upper_endpoint_or_quantile_mm",
    "depth_endpoint_mode",
    "depth_lower_quantile",
    "depth_upper_quantile",
    "depth_support_projected_lower",
    "depth_support_projected_upper",
    "depth_support_projected_span_mm",
    "manual_major_selected",
    "manual_minor_selected",
]

SELECTED_COLUMNS = [
    "case_id",
    "case_index",
    "material",
    "object_id",
    "path_label",
    "manual_mm",
    "selected_anchor_id",
    "selection_source",
    "p0_x",
    "p0_y",
    "direction_x",
    "direction_y",
    "endpoint1_x",
    "endpoint1_y",
    "endpoint2_x",
    "endpoint2_y",
    "mask_path_length_px",
    "mask_path_length_mm",
    "mask_abs_error_mm",
    "mask_signed_error_mm",
    "mask_pct_error",
    "strip_depth_path_length_mm",
    "strip_depth_valid",
    "strip_depth_failure_reason",
    "strip_depth_num_supported_points",
    "strip_depth_valid_depth_fraction",
    "strip_depth_abs_error_mm",
    "strip_depth_signed_error_mm",
    "strip_depth_pct_error",
    "depth_endpoint_mode",
    "depth_lower_quantile",
    "depth_upper_quantile",
    "depth_support_projected_lower",
    "depth_support_projected_upper",
    "depth_support_projected_span_mm",
    "best_method_for_this_path_by_abs_error",
    "notes",
]


@dataclass
class CaseInput:
    case_id: str
    case_index: int
    object_id: str
    material: str
    case_dir: Path
    rgb_path: Path
    depth_path: Path
    camera_info_path: Path
    manual_path: Path
    capture_metadata_path: Path


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip() or None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _case_index(case_id: str, fallback: int) -> int:
    try:
        return int(case_id.split("_")[1])
    except Exception:
        return fallback


def _is_backup_case_name(name: str) -> bool:
    low = str(name).lower()
    return "backup" in low or "_backup_" in low or "_before_" in low


def _material_for_case(case_id: str, fallback: str | None = None) -> str:
    text = str(fallback or "").strip().lower().replace("_", " ")
    if text in {"timber", "wood", "t"}:
        return "timber"
    if text in {"brick", "b"}:
        return "brick"
    if text in {"concrete", "concrete block", "block", "c"}:
        return "concrete block"
    idx = _case_index(case_id, 0)
    if 1 <= idx <= 5:
        return "timber"
    if 6 <= idx <= 10:
        return "brick"
    if 11 <= idx <= 15:
        return "concrete block"
    return fallback or ""


def _read_cases(session_dir: Path, args: argparse.Namespace) -> list[CaseInput]:
    manifest = session_dir / "session_manifest.csv"
    rows: list[dict[str, Any]]
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = []
            for row in csv.DictReader(handle):
                if _is_backup_case_name(str(row.get("case_id") or "")):
                    continue
                saved_status = str(row.get("saved_status") or "saved")
                if saved_status.startswith("saved"):
                    rows.append(row)
    else:
        rows = []
        for idx, case_dir in enumerate(sorted((session_dir / "cases").glob("case_*")), start=1):
            if _is_backup_case_name(case_dir.name):
                continue
            rows.append(
                {
                    "case_id": case_dir.name,
                    "case_index": idx,
                    "object_id": case_dir.name,
                    "rgb_path": str(case_dir / "raw_rgb.png"),
                    "aligned_depth_z16_path": str(case_dir / "raw_depth_aligned_z16.png"),
                    "aligned_depth_camera_info_path": str(case_dir / "camera_info_aligned_depth.json"),
                    "manual_measurements_path": str(case_dir / "manual_measurements.json"),
                    "capture_metadata_path": str(case_dir / "capture_metadata.json"),
                }
            )
    cases: list[CaseInput] = []
    for fallback_idx, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        if args.case_id and case_id != args.case_id:
            continue
        idx = int(row.get("case_index") or _case_index(case_id, fallback_idx))
        material = _material_for_case(case_id, row.get("material"))
        case_dir = (session_dir / "cases" / case_id).resolve()
        cases.append(
            CaseInput(
                case_id=case_id,
                case_index=idx,
                object_id=str(row.get("object_id") or case_id),
                material=material,
                case_dir=case_dir,
                rgb_path=(case_dir / "raw_rgb.png").resolve(),
                depth_path=(case_dir / "raw_depth_aligned_z16.png").resolve(),
                camera_info_path=(case_dir / "camera_info_aligned_depth.json").resolve(),
                manual_path=(case_dir / "manual_measurements.json").resolve(),
                capture_metadata_path=(case_dir / "capture_metadata.json").resolve(),
            )
        )
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    return cases


def _case_from_folder(session_dir: Path, mask_run: Path, case_id: str) -> CaseInput:
    case_dir = session_dir / "cases" / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"selected_anchors.json exists for {case_id}, but case folder is missing: {case_dir}")
    manual_path = case_dir / "manual_measurements.json"
    material = ""
    object_id = case_id
    if manual_path.exists():
        try:
            manual = _read_json(manual_path)
            material = str(manual.get("material") or "")
            object_id = str(manual.get("object_id") or case_id)
        except Exception:
            pass
    return CaseInput(
        case_id=case_id,
        case_index=_case_index(case_id, 0),
        object_id=object_id,
        material=_material_for_case(case_id, material),
        case_dir=case_dir.resolve(),
        rgb_path=(case_dir / "raw_rgb.png").resolve(),
        depth_path=(case_dir / "raw_depth_aligned_z16.png").resolve(),
        camera_info_path=(case_dir / "camera_info_aligned_depth.json").resolve(),
        manual_path=manual_path.resolve(),
        capture_metadata_path=(case_dir / "capture_metadata.json").resolve(),
    )


def _selected_anchor_case_ids(manual_anchor_root: Path | None) -> list[str]:
    if manual_anchor_root is None:
        return []
    cases_dir = manual_anchor_root / "cases"
    if not cases_dir.exists():
        return []
    return sorted(path.parent.name for path in cases_dir.glob("*/selected_anchors.json") if not _is_backup_case_name(path.parent.name))


def _case_missing_reasons(case: CaseInput, mask_run: Path, manual_anchor_root: Path | None) -> list[str]:
    checks = {
        "raw_rgb.png": case.rgb_path,
        "raw_depth_aligned_z16.png": case.depth_path,
        "manual_measurements.json": case.manual_path,
        "camera_info_aligned_depth.json": case.camera_info_path,
        "capture_metadata.json": case.capture_metadata_path,
        "selected_mask.png": mask_run / "cases" / case.case_id / "selected_mask.png",
    }
    if manual_anchor_root is not None:
        checks["selected_anchors.json"] = manual_anchor_root / "cases" / case.case_id / "selected_anchors.json"
    return [f"{label}:{path}" for label, path in checks.items() if not path.exists()]


def _include_selected_anchor_cases(
    cases: list[CaseInput],
    *,
    session_dir: Path,
    mask_run: Path,
    manual_anchor_root: Path | None,
    args: argparse.Namespace,
) -> tuple[list[CaseInput], list[str], list[dict[str, Any]]]:
    selected_case_ids = _selected_anchor_case_ids(manual_anchor_root)
    by_id = {case.case_id: case for case in cases}
    skipped: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        if args.case_id and case_id != args.case_id:
            skipped.append({"case_id": case_id, "reason": f"filtered_by_case_id:{args.case_id}"})
            continue
        case = _case_from_folder(session_dir, mask_run, case_id)
        missing = _case_missing_reasons(case, mask_run, manual_anchor_root)
        if missing:
            skipped.append({"case_id": case_id, "reason": "missing_required_files", "missing": missing})
            continue
        by_id[case_id] = case
    included = sorted(by_id.values(), key=lambda case: (case.case_index, case.case_id))
    included_ids = [case.case_id for case in included]
    return included, included_ids, skipped


def _load_intrinsics(path: Path) -> dict[str, float]:
    data = _read_json(path)
    fx = data.get("fx")
    fy = data.get("fy")
    cx = data.get("cx")
    cy = data.get("cy")
    if None in (fx, fy, cx, cy):
        k = data.get("K") or data.get("k")
        if isinstance(k, list) and len(k) >= 6:
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    if None in (fx, fy, cx, cy):
        raise ValueError(f"Missing fx/fy/cx/cy in {path}")
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def _backproject(uv: np.ndarray, depth_mm: np.ndarray, intr: dict[str, float]) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    x = (uv[:, 0] - intr["cx"]) * z / intr["fx"]
    y = (uv[:, 1] - intr["cy"]) * z / intr["fy"]
    return np.column_stack([x, y, z])


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        raise ValueError("zero direction vector")
    return vec / n


def _depth_viridis(depth_mm: np.ndarray, *, invalid: str = "black") -> np.ndarray:
    valid = depth_mm > 0
    norm = colors.Normalize(vmin=268.0, vmax=375.0, clip=True)
    rgba = (matplotlib.colormaps["viridis"](norm(depth_mm.astype(float))) * 255).astype(np.uint8)
    rgba[~valid] = np.array([0, 0, 0, 255] if invalid == "black" else [255, 255, 255, 255], dtype=np.uint8)
    return rgba[:, :, :3]


def _read_mask(mask_run: Path, case_id: str) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    case_dir = mask_run / "cases" / case_id
    mask_path = case_dir / "selected_mask.png"
    meta_path = case_dir / "mask_metadata.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    if not mask_path.exists():
        return None, f"missing_gsam2_mask:{mask_path}", meta
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, f"unreadable_gsam2_mask:{mask_path}", meta
    return (mask > 0).astype(np.uint8), str(mask_path), meta


def _rect_axes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty mask")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.array(rect[0], dtype=float)
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_idx = int(np.argmax(lengths))
    long_dir = _normalize(edges[long_idx])
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=float)
    return center, long_dir, short_dir, box, rect


def _expand_box(box: np.ndarray, buffer_percent: float) -> np.ndarray:
    center = box.mean(axis=0)
    return center + (box - center) * (1.0 + float(buffer_percent) / 100.0)


def _pick_ref_depth(depth_mm: np.ndarray, p0: np.ndarray, radius: float) -> tuple[float | None, int]:
    h, w = depth_mm.shape
    x0 = max(0, int(math.floor(p0[0] - radius)))
    x1 = min(w, int(math.ceil(p0[0] + radius + 1)))
    y0 = max(0, int(math.floor(p0[1] - radius)))
    y1 = min(h, int(math.ceil(p0[1] + radius + 1)))
    vals = depth_mm[y0:y1, x0:x1]
    valid = vals[vals > 0]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def _direction3(p0: np.ndarray, direction: np.ndarray, ref_depth_mm: float, intr: dict[str, float]) -> tuple[np.ndarray, np.ndarray, float]:
    uv = np.vstack([p0, p0 + direction])
    xyz = _backproject(uv, np.array([ref_depth_mm, ref_depth_mm], dtype=float), intr)
    d3 = _normalize(xyz[1] - xyz[0])
    mm_per_px = 1000.0 * float(np.linalg.norm(xyz[1] - xyz[0]))
    return xyz[0], d3, mm_per_px


def _mask_chord(mask: np.ndarray, p0: np.ndarray, direction: np.ndarray, ref_depth_mm: float, intr: dict[str, float]) -> dict[str, Any]:
    h, w = mask.shape
    ts: list[float] = []
    max_extent = int(math.hypot(w, h))
    for t in np.linspace(-max_extent, max_extent, 2 * max_extent + 1):
        p = p0 + direction * t
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            ts.append(float(t))
    if not ts:
        return {"valid": False, "failure_reason": "no_mask_intersection"}
    t0, t1 = min(ts), max(ts)
    p1 = p0 + direction * t0
    p2 = p0 + direction * t1
    _, _, mm_per_px = _direction3(p0, direction, ref_depth_mm, intr)
    return {
        "valid": True,
        "length_mm": abs(t1 - t0) * mm_per_px,
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "t0_px": t0,
        "t1_px": t1,
        "failure_reason": None,
    }


def _endpoint_length(p1: np.ndarray, p2: np.ndarray, ref_depth_mm: float, intr: dict[str, float]) -> dict[str, Any]:
    delta = p2 - p1
    length_px = float(np.linalg.norm(delta))
    if length_px <= 1e-12:
        return {"valid": False, "failure_reason": "zero_endpoint_length", "length_px": None, "length_mm": None}
    p0 = 0.5 * (p1 + p2)
    direction = delta / length_px
    _, _, mm_per_px = _direction3(p0, direction, ref_depth_mm, intr)
    return {
        "valid": True,
        "failure_reason": None,
        "length_px": length_px,
        "length_mm": length_px * mm_per_px,
        "p0": p0,
        "direction": direction,
    }


def _strip_supported_count(strip: dict[str, Any]) -> int:
    results = strip.get("strip_results") or []
    return int(sum(int(r.get("supported_point_count") or 0) for r in results if r.get("valid")))


def _strip_valid_depth_count(strip: dict[str, Any]) -> int:
    results = strip.get("strip_results") or []
    return int(sum(int(r.get("valid_depth_count") or 0) for r in results))


def _strip_candidate_count(strip: dict[str, Any]) -> int:
    results = strip.get("strip_results") or []
    return int(sum(int(r.get("candidate_pixel_count") or 0) for r in results))


def _strip_quantile_mm(strip: dict[str, Any], key: str) -> float | None:
    vals = [float(r[key]) * 1000.0 for r in (strip.get("strip_results") or []) if r.get("valid") and r.get(key) is not None]
    if not vals:
        return None
    return float(np.median(vals))


def _strip_projected_span_mm(strip: dict[str, Any]) -> float | None:
    lows = [float(r["lower_projected_support_m"]) for r in (strip.get("strip_results") or []) if r.get("valid") and r.get("lower_projected_support_m") is not None]
    highs = [float(r["upper_projected_support_m"]) for r in (strip.get("strip_results") or []) if r.get("valid") and r.get("upper_projected_support_m") is not None]
    if not lows or not highs:
        return None
    return 1000.0 * abs(float(np.median(highs)) - float(np.median(lows)))


def _whole_roi_depth(depth_mm: np.ndarray, p0: np.ndarray, direction: np.ndarray, box: np.ndarray, intr: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    roi = np.zeros(depth_mm.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi, np.round(_expand_box(box, args.buffer_percent)).astype(np.int32), 1)
    ys, xs = np.nonzero(roi > 0)
    uv = np.column_stack([xs.astype(float), ys.astype(float)])
    ref, ref_count = _pick_ref_depth(depth_mm, p0, args.center_region_radius_px)
    if uv.size == 0:
        return {"valid": False, "failure_reason": "empty_roi", "support_uv": [], "length_mm": None}
    depths = depth_mm[ys, xs].astype(float)
    valid = depths > 0
    if ref is None:
        return {"valid": False, "failure_reason": "missing_reference_depth", "support_uv": [], "length_mm": None}
    supported = valid & (np.abs(depths - ref) <= args.depth_tolerance_mm)
    support_uv = uv[supported]
    support_depth = depths[supported]
    if support_uv.shape[0] < args.min_supported_points:
        return {"valid": False, "failure_reason": f"insufficient_supported_points:{support_uv.shape[0]}", "support_uv": support_uv.tolist(), "length_mm": None}
    p0_xyz, d3, _ = _direction3(p0, direction, ref, intr)
    xyz = _backproject(support_uv, support_depth, intr)
    proj = (xyz - p0_xyz) @ d3
    lo = float(np.quantile(proj, args.lower_quantile))
    hi = float(np.quantile(proj, args.upper_quantile))
    return {
        "valid": True,
        "failure_reason": None,
        "length_mm": 1000.0 * abs(hi - lo),
        "support_uv": support_uv[:: max(1, support_uv.shape[0] // 1000)].tolist(),
        "candidate_point_count": int(uv.shape[0]),
        "valid_depth_point_count": int(np.count_nonzero(valid)),
        "supported_point_count": int(support_uv.shape[0]),
        "center_depth_mm": ref,
        "center_region_valid_count": ref_count,
        "lower_projected_support_m": lo,
        "upper_projected_support_m": hi,
    }


def _strip_depth(depth_mm: np.ndarray, p0: np.ndarray, direction: np.ndarray, box: np.ndarray, intr: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    roi = np.zeros(depth_mm.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi, np.round(_expand_box(box, args.buffer_percent)).astype(np.int32), 1)
    ys, xs = np.nonzero(roi > 0)
    roi_uv = np.column_stack([xs.astype(float), ys.astype(float)])
    ref, ref_count = _pick_ref_depth(depth_mm, p0, args.center_region_radius_px)
    if ref is None:
        return {"valid": False, "failure_reason": "missing_reference_depth", "strip_results": [], "length_mm": None}
    p0_xyz, d3, _ = _direction3(p0, direction, ref, intr)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    strip_lengths: list[float] = []
    strip_results: list[dict[str, Any]] = []
    all_support: list[list[float]] = []
    for offset in args.strip_offsets_px:
        line_p0 = p0 + perp * float(offset)
        rel = roi_uv - line_p0[None, :]
        r = np.abs(rel @ perp)
        in_strip = r <= args.strip_half_width_px
        uv = roi_uv[in_strip]
        result = {"strip_offset_px": float(offset), "candidate_pixel_count": int(uv.shape[0]), "valid": False}
        if uv.shape[0] == 0:
            result["failure_reason"] = "empty_strip"
            strip_results.append(result)
            continue
        xi = uv[:, 0].astype(int)
        yi = uv[:, 1].astype(int)
        depths = depth_mm[yi, xi].astype(float)
        valid = depths > 0
        supported = valid & (np.abs(depths - ref) <= args.depth_tolerance_mm)
        support_uv = uv[supported]
        support_depth = depths[supported]
        result["valid_depth_count"] = int(np.count_nonzero(valid))
        result["supported_point_count"] = int(support_uv.shape[0])
        if support_uv.shape[0] < args.min_supported_points:
            result["failure_reason"] = f"insufficient_supported_points:{support_uv.shape[0]}"
            strip_results.append(result)
            continue
        xyz = _backproject(support_uv, support_depth, intr)
        proj = (xyz - p0_xyz) @ d3
        # V1 already used these robust projected quantile endpoints. The explicit
        # mode/quantile args make V2 percentile-endpoint runs self-documenting
        # without changing V1 defaults.
        lo = float(np.quantile(proj, args.lower_quantile))
        hi = float(np.quantile(proj, args.upper_quantile))
        length_mm = 1000.0 * abs(hi - lo)
        result.update(
            {
                "valid": True,
                "failure_reason": None,
                "length_mm": length_mm,
                "lower_projected_support_m": lo,
                "upper_projected_support_m": hi,
                "projected_span_mm": length_mm,
                "depth_endpoint_mode": args.depth_endpoint_mode,
                "depth_lower_quantile": args.lower_quantile,
                "depth_upper_quantile": args.upper_quantile,
            }
        )
        strip_lengths.append(length_mm)
        all_support.extend(support_uv[:: max(1, support_uv.shape[0] // 150)].tolist())
        strip_results.append(result)
    if not strip_lengths:
        return {"valid": False, "failure_reason": "no_valid_strips", "strip_results": strip_results, "length_mm": None, "support_uv": all_support}
    return {
        "valid": True,
        "failure_reason": None,
        "length_mm": float(np.median(strip_lengths)),
        "mean_length_mm": float(np.mean(strip_lengths)),
        "strip_results": strip_results,
        "n_valid_strips": len(strip_lengths),
        "support_uv": all_support,
        "center_depth_mm": ref,
        "center_region_valid_count": ref_count,
    }


def _err(pred: float | None, manual: float | None) -> tuple[float | None, float | None, float | None]:
    if pred is None or manual is None or manual <= 0:
        return None, None, None
    signed = float(pred - manual)
    abs_err = abs(signed)
    return abs_err, 100.0 * abs_err / manual, signed


INVALID_FALLBACK_REASON = (
    "manual major/minor labels were compared to fallback minAreaRect axes without selected anchor/path metadata"
)


def _load_manual_anchor_selection(root: Path | None, case_id: str, path_label: str) -> dict[str, Any] | None:
    if root is None:
        return None
    candidates = [
        root / "cases" / case_id / "selected_anchors.json",
        root / case_id / "selected_anchors.json",
        root / f"{case_id}_selected_anchors.json",
        root / "selected_anchors.json",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            f"manual anchor selection metadata was requested but no selected_anchors.json was found for {case_id} under {root}"
        )
    payload = _read_json(source_path)
    if isinstance(payload, dict) and case_id in payload:
        payload = payload[case_id]
    item: Any = None
    if isinstance(payload, dict):
        item = payload.get(path_label) or payload.get(f"{path_label}_path") or payload.get(f"{path_label}_axis")
        if item is None and "paths" in payload and isinstance(payload["paths"], dict):
            item = payload["paths"].get(path_label)
    elif isinstance(payload, list):
        for candidate in payload:
            if isinstance(candidate, dict) and candidate.get("path_label") == path_label:
                item = candidate
                break
    if not isinstance(item, dict):
        raise ValueError(f"No manual anchor/path metadata for {case_id} path_label={path_label} in {source_path}")

    def _point(*keys: str) -> np.ndarray | None:
        for key in keys:
            value = item.get(key)
            if isinstance(value, dict):
                x = value.get("x", value.get("u", value.get("col")))
                y = value.get("y", value.get("v", value.get("row")))
                if x is not None and y is not None:
                    return np.array([float(x), float(y)], dtype=float)
            if isinstance(value, list) and len(value) >= 2:
                return np.array([float(value[0]), float(value[1])], dtype=float)
        return None

    p0 = _point("p0", "p0_xy", "path_center_px", "anchor_center_px", "center_px")
    direction = None
    direction_value = item.get("direction_xy") or item.get("path_direction_xy") or item.get("direction")
    if isinstance(direction_value, dict):
        x = direction_value.get("x", direction_value.get("dx"))
        y = direction_value.get("y", direction_value.get("dy"))
        if x is not None and y is not None:
            direction = np.array([float(x), float(y)], dtype=float)
    elif isinstance(direction_value, list) and len(direction_value) >= 2:
        direction = np.array([float(direction_value[0]), float(direction_value[1])], dtype=float)
    p1 = _point("p1", "endpoint1_xy", "endpoint_1_px", "endpoint1_px", "line_p1_px")
    p2 = _point("p2", "endpoint2_xy", "endpoint_2_px", "endpoint2_px", "line_p2_px")
    if direction is None and p1 is not None and p2 is not None:
        direction = p2 - p1
    if p0 is None and p1 is not None and p2 is not None:
        p0 = 0.5 * (p1 + p2)
    if p0 is None or direction is None:
        raise ValueError(f"Manual anchor metadata for {case_id} {path_label} must provide p0/direction or endpoints.")
    return {
        "p0": p0,
        "direction": _normalize(direction),
        "source_path": str(source_path),
        "raw": item,
        "selected_anchor_id": item.get("selected_anchor_id"),
        "selection_source": item.get("selection_source"),
        "endpoint1": p1,
        "endpoint2": p2,
        "path_length_px": item.get("path_length_px"),
    }


def _draw_line(ax: Any, p0: np.ndarray, direction: np.ndarray, length_mm: float | None, ref_depth_mm: float, intr: dict[str, float], *, color: str, label: str, linestyle: str = "-") -> None:
    if length_mm is None:
        return
    _, _, mm_per_px = _direction3(p0, direction, ref_depth_mm, intr)
    half_px = 0.5 * length_mm / max(mm_per_px, 1e-9)
    a = p0 - direction * half_px
    b = p0 + direction * half_px
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2.0, linestyle=linestyle, label=label)


def _progress(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "debug_progress", False):
        print(message, flush=True)


def _step_grid(case: CaseInput, path_label: str, rgb: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray, box: np.ndarray, p0: np.ndarray, direction: np.ndarray, row: dict[str, Any], diag: dict[str, Any], out_png: Path, out_svg: Path | None, intr: dict[str, float]) -> None:
    depth_rgb = _depth_viridis(depth_mm)
    ref_depth = diag.get("reference_depth_mm") or np.nanmedian(depth_mm[depth_mm > 0])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axs = axes.ravel()
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(rgb); axs[0].set_title("RGB image")
    axs[1].imshow(depth_rgb); axs[1].set_title("Depth viridis 268-375 mm")
    overlay = rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    axs[2].imshow(overlay); axs[2].set_title("GSAM2 selected mask")
    for ax_idx, title in [(3, "Mask ROI geometry"), (4, "Whole-ROI depth method"), (5, "Strip-depth method")]:
        axs[ax_idx].imshow(rgb)
        axs[ax_idx].add_patch(Polygon(box, fill=False, edgecolor="orange", linewidth=2))
        axs[ax_idx].plot(p0[0], p0[1], "ko", markersize=4)
        d = direction * 60
        axs[ax_idx].plot([p0[0] - d[0], p0[0] + d[0]], [p0[1] - d[1], p0[1] + d[1]], "k-", linewidth=1)
        axs[ax_idx].set_title(title)
    if row.get("mask_based_mm"):
        _draw_line(axs[3], p0, direction, float(row["mask_based_mm"]), float(ref_depth), intr, color="blue", label="mask")
    support = np.asarray(diag.get("whole_roi", {}).get("support_uv", []), dtype=float)
    if support.size:
        axs[4].scatter(support[:, 0], support[:, 1], s=1, c="lime", alpha=0.35)
    _draw_line(axs[4], p0, direction, row.get("whole_roi_depth_mm"), float(ref_depth), intr, color="green", label="whole")
    strip_support = np.asarray(diag.get("strip_depth", {}).get("support_uv", []), dtype=float)
    if strip_support.size:
        axs[5].scatter(strip_support[:, 0], strip_support[:, 1], s=1, c="magenta", alpha=0.3)
    _draw_line(axs[5], p0, direction, row.get("strip_depth_mm"), float(ref_depth), intr, color="purple", label="strip", linestyle="--")
    strip_results = diag.get("strip_depth", {}).get("strip_results", [])
    xs = [r["strip_offset_px"] for r in strip_results if r.get("valid")]
    ys = [r["length_mm"] for r in strip_results if r.get("valid")]
    axs[6].axis("on")
    axs[6].plot(xs, ys, "o-", color="purple", label="strip")
    if row.get("whole_roi_depth_mm") is not None:
        axs[6].axhline(float(row["whole_roi_depth_mm"]), color="green", linestyle="-", label="whole")
    if row.get("manual_mm") is not None:
        axs[6].axhline(float(row["manual_mm"]), color="black", linestyle=":", label="manual")
    axs[6].set_title("Projected support / strip estimates")
    axs[6].set_xlabel("strip offset px")
    axs[6].set_ylabel("length mm")
    axs[6].legend(fontsize=7)
    text = (
        f"{case.case_id} {path_label}\n"
        f"manual: {row.get('manual_mm')}\n"
        f"mask: {row.get('mask_based_mm')} err {row.get('mask_abs_err_mm')}\n"
        f"whole: {row.get('whole_roi_depth_mm')} err {row.get('whole_roi_depth_abs_err_mm')}\n"
        f"strip: {row.get('strip_depth_mm')} err {row.get('strip_depth_abs_err_mm')}\n"
        f"fallback direction: {row.get('direction_source')}"
    )
    axs[7].text(0.02, 0.96, text, va="top", ha="left", fontsize=10, family="monospace")
    axs[7].set_title("Numerical comparison")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


def _summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for label in ["major", "minor", "all"]:
        subset = rows if label == "all" else [r for r in rows if r["path_label"] == label]
        srow: dict[str, Any] = {"path_label": label, "n_total": len(subset)}
        maes: dict[str, float | None] = {}
        rmses: dict[str, float | None] = {}
        mapes: dict[str, float | None] = {}
        for method, value_col, err_prefix in [
            ("mask", "mask_based", "mask"),
            ("whole_roi_depth", "whole_roi_depth", "whole_roi_depth"),
            ("strip_depth", "strip_depth", "strip_depth"),
        ]:
            vals = [float(r[f"{err_prefix}_abs_err_mm"]) for r in subset if r.get(f"{err_prefix}_abs_err_mm") not in (None, "")]
            pct = [float(r[f"{err_prefix}_pct_err"]) for r in subset if r.get(f"{err_prefix}_pct_err") not in (None, "")]
            signed = []
            for r in subset:
                pred = r.get(f"{value_col}_mm")
                manual = r.get("manual_mm")
                if pred not in (None, "") and manual not in (None, ""):
                    signed.append(float(pred) - float(manual))
            srow[f"{method}_n_valid"] = len(vals)
            srow[f"{method}_MAE_mm"] = float(np.mean(vals)) if vals else None
            srow[f"{method}_RMSE_mm"] = float(np.sqrt(np.mean(np.square(vals)))) if vals else None
            srow[f"{method}_MAPE_percent"] = float(np.mean(pct)) if pct else None
            srow[f"{method}_signed_mean_error_mm"] = float(np.mean(signed)) if signed else None
            maes[method] = srow[f"{method}_MAE_mm"]
            rmses[method] = srow[f"{method}_RMSE_mm"]
            mapes[method] = srow[f"{method}_MAPE_percent"]
        summary_rows.append(srow)
        ranking_rows.append(
            {
                "path_label": label,
                "best_method_by_MAE": min((k for k, v in maes.items() if v is not None), key=lambda k: maes[k], default=None),
                "best_method_by_RMSE": min((k for k, v in rmses.items() if v is not None), key=lambda k: rmses[k], default=None),
                "best_method_by_MAPE": min((k for k, v in mapes.items() if v is not None), key=lambda k: mapes[k], default=None),
                "mask_vs_manual_MAE_mm": maes.get("mask"),
                "whole_roi_vs_manual_MAE_mm": maes.get("whole_roi_depth"),
                "strip_vs_manual_MAE_mm": maes.get("strip_depth"),
            }
        )
    return summary_rows, ranking_rows


def _load_selection_payload(manual_anchor_root: Path, case_id: str) -> dict[str, Any]:
    path = manual_anchor_root / "cases" / case_id / "selected_anchors.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def _verify_selection_completeness(cases: list[CaseInput], manual_anchor_root: Path | None) -> dict[str, dict[str, Any]]:
    if manual_anchor_root is None:
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    bad: list[str] = []
    required = ["manual_mm", "selected_anchor_id", "p0_xy", "direction_xy", "endpoint1_xy", "endpoint2_xy", "path_length_px"]
    for case in cases:
        path = manual_anchor_root / "cases" / case.case_id / "selected_anchors.json"
        if not path.exists():
            missing.append(case.case_id)
            continue
        payload = _read_json(path)
        payloads[case.case_id] = payload
        for label in ["major", "minor"]:
            item = payload.get(label)
            if not isinstance(item, dict):
                bad.append(f"{case.case_id}:{label}:missing_object")
                continue
            missing_keys = [key for key in required if item.get(key) in (None, "")]
            if missing_keys:
                bad.append(f"{case.case_id}:{label}:missing_{','.join(missing_keys)}")
    if missing or bad:
        raise RuntimeError(f"Incomplete manual anchor selections. Missing files={missing}; bad entries={bad}")
    return payloads


def _all_anchor_measurements(
    *,
    case: CaseInput,
    manual_anchor_root: Path | None,
    selection_payload: dict[str, Any] | None,
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: dict[str, float],
    box: np.ndarray,
    ref_depth: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if manual_anchor_root is None:
        return []
    candidate_path = manual_anchor_root / "cases" / case.case_id / "anchor_candidates.csv"
    if not candidate_path.exists():
        return []
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    major_id = (selection_payload or {}).get("major", {}).get("selected_anchor_id")
    minor_id = (selection_payload or {}).get("minor", {}).get("selected_anchor_id")
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        if str(cand.get("valid_candidate")).lower() not in {"true", "1", "yes"}:
            continue
        try:
            p1 = np.array([float(cand["endpoint1_x"]), float(cand["endpoint1_y"])], dtype=float)
            p2 = np.array([float(cand["endpoint2_x"]), float(cand["endpoint2_y"])], dtype=float)
        except Exception:
            continue
        length = _endpoint_length(p1, p2, ref_depth, intr)
        if length.get("valid"):
            p0 = length["p0"]
            direction = length["direction"]
            strip = _strip_depth(depth_mm, p0, direction, box, intr, args)
        else:
            p0 = np.array([float(cand.get("p0_x") or 0.0), float(cand.get("p0_y") or 0.0)], dtype=float)
            direction = np.array([float(cand.get("direction_x") or 1.0), float(cand.get("direction_y") or 0.0)], dtype=float)
            strip = {"valid": False, "failure_reason": length.get("failure_reason")}
        candidate_count = _strip_candidate_count(strip)
        valid_depth_count = _strip_valid_depth_count(strip)
        rows.append(
            {
                "case_id": case.case_id,
                "case_index": case.case_index,
                "material": case.material,
                "object_id": case.object_id,
                "candidate_id": cand.get("candidate_id"),
                "candidate_family": cand.get("path_family"),
                "selected_for_major": cand.get("candidate_id") == major_id,
                "selected_for_minor": cand.get("candidate_id") == minor_id,
                "candidate_p0_x": float(p0[0]),
                "candidate_p0_y": float(p0[1]),
                "candidate_direction_x": float(direction[0]),
                "candidate_direction_y": float(direction[1]),
                "candidate_endpoint1_x": float(p1[0]),
                "candidate_endpoint1_y": float(p1[1]),
                "candidate_endpoint2_x": float(p2[0]),
                "candidate_endpoint2_y": float(p2[1]),
                "candidate_path_length_px_from_csv": cand.get("path_length_px"),
                "mask_path_length_px_recomputed": length.get("length_px"),
                "mask_path_length_mm": length.get("length_mm"),
                "strip_depth_path_length_mm": strip.get("length_mm"),
                "strip_depth_valid": bool(strip.get("valid")),
                "strip_depth_failure_reason": strip.get("failure_reason"),
                "strip_depth_num_supported_points": _strip_supported_count(strip),
                "strip_depth_valid_depth_fraction": (valid_depth_count / candidate_count) if candidate_count else None,
                "strip_depth_depth_median_mm": strip.get("center_depth_mm"),
                "strip_depth_lower_endpoint_or_quantile_mm": _strip_quantile_mm(strip, "lower_projected_support_m"),
                "strip_depth_upper_endpoint_or_quantile_mm": _strip_quantile_mm(strip, "upper_projected_support_m"),
                "depth_endpoint_mode": args.depth_endpoint_mode,
                "depth_lower_quantile": args.lower_quantile,
                "depth_upper_quantile": args.upper_quantile,
                "depth_support_projected_lower": _strip_quantile_mm(strip, "lower_projected_support_m"),
                "depth_support_projected_upper": _strip_quantile_mm(strip, "upper_projected_support_m"),
                "depth_support_projected_span_mm": _strip_projected_span_mm(strip),
                "manual_major_selected": (selection_payload or {}).get("major", {}).get("manual_mm") if cand.get("candidate_id") == major_id else None,
                "manual_minor_selected": (selection_payload or {}).get("minor", {}).get("manual_mm") if cand.get("candidate_id") == minor_id else None,
            }
        )
    return rows


def _metric_block(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    vals = [float(r[f"{prefix}_abs_error_mm"]) for r in rows if r.get(f"{prefix}_abs_error_mm") not in (None, "")]
    pct = [float(r[f"{prefix}_pct_error"]) for r in rows if r.get(f"{prefix}_pct_error") not in (None, "")]
    signed = [float(r[f"{prefix}_signed_error_mm"]) for r in rows if r.get(f"{prefix}_signed_error_mm") not in (None, "")]
    return {
        f"{prefix}_MAE_mm": float(np.mean(vals)) if vals else None,
        f"{prefix}_RMSE_mm": float(np.sqrt(np.mean(np.square(vals)))) if vals else None,
        f"{prefix}_MAPE_percent": float(np.mean(pct)) if pct else None,
        f"{prefix}_signed_mean_error_mm": float(np.mean(signed)) if signed else None,
        f"{prefix}_median_abs_error_mm": float(np.median(vals)) if vals else None,
        f"{prefix}_max_abs_error_mm": float(np.max(vals)) if vals else None,
    }


def _selected_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [
        ("major", [r for r in rows if r.get("path_label") == "major"]),
        ("minor", [r for r in rows if r.get("path_label") == "minor"]),
        ("all", rows),
        ("timber_all", [r for r in rows if r.get("material") == "timber"]),
        ("brick_all", [r for r in rows if r.get("material") == "brick"]),
        ("concrete_block_all", [r for r in rows if r.get("material") == "concrete block"]),
    ]
    out: list[dict[str, Any]] = []
    for name, subset in groups:
        row: dict[str, Any] = {
            "group": name,
            "n_total": len(subset),
            "mask_n_valid": sum(1 for r in subset if r.get("mask_path_length_mm") not in (None, "")),
            "strip_depth_n_valid": sum(1 for r in subset if str(r.get("strip_depth_valid")).lower() == "true"),
        }
        row["strip_depth_n_invalid"] = row["n_total"] - row["strip_depth_n_valid"]
        row.update(_metric_block(subset, "mask"))
        row.update(_metric_block(subset, "strip_depth"))
        row["strip_depth_valid_rate"] = row["strip_depth_n_valid"] / row["n_total"] if row["n_total"] else None
        mask_mae = row.get("mask_MAE_mm")
        strip_mae = row.get("strip_depth_MAE_mm")
        if mask_mae is not None and strip_mae is not None:
            row["improvement_mm"] = float(mask_mae) - float(strip_mae)
            row["improvement_percent"] = 100.0 * (float(mask_mae) - float(strip_mae)) / float(mask_mae) if float(mask_mae) else None
            row["best_method_by_MAE"] = "strip_depth" if float(strip_mae) < float(mask_mae) else "mask"
        else:
            row["improvement_mm"] = None
            row["improvement_percent"] = None
            row["best_method_by_MAE"] = "mask" if mask_mae is not None else ("strip_depth" if strip_mae is not None else None)
        out.append(row)
    return out


def _selected_summary_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["group"]
    return list(rows[0].keys())


def _failure_reason_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get(key) or "")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _make_selected_figures(out_dir: Path, selected_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['case_id']}\n{r['path_label']}" for r in selected_rows]
    x = np.arange(len(selected_rows))
    mask_err = [float(r["mask_abs_error_mm"]) if r.get("mask_abs_error_mm") not in (None, "") else np.nan for r in selected_rows]
    strip_err = [float(r["strip_depth_abs_error_mm"]) if r.get("strip_depth_abs_error_mm") not in (None, "") else np.nan for r in selected_rows]
    fig, ax = plt.subplots(figsize=(max(12, len(selected_rows) * 0.45), 5))
    ax.bar(x - 0.18, mask_err, width=0.36, label="mask", color="#1f77b4")
    ax.bar(x + 0.18, strip_err, width=0.36, label="strip-depth", color="#9467bd")
    ax.set_ylabel("absolute error (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "selected_anchor_error_barplot.png", dpi=180)
    plt.close(fig)

    def scatter(path: Path, pred_key: str, title: str, color: str) -> None:
        xs = []
        ys = []
        for row in selected_rows:
            if row.get("manual_mm") in (None, "") or row.get(pred_key) in (None, ""):
                continue
            xs.append(float(row["manual_mm"]))
            ys.append(float(row[pred_key]))
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(xs, ys, c=color)
        if xs and ys:
            lo = min(xs + ys)
            hi = max(xs + ys)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("manual mm")
        ax.set_ylabel("estimated mm")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    scatter(fig_dir / "selected_anchor_error_scatter_mask.png", "mask_path_length_mm", "Manual vs mask path length", "#1f77b4")
    scatter(fig_dir / "selected_anchor_error_scatter_strip_depth.png", "strip_depth_path_length_mm", "Manual vs strip-depth path length", "#9467bd")

    fig, ax = plt.subplots(figsize=(max(10, len(selected_rows) * 0.35), 2.8))
    validity = [1 if str(r.get("strip_depth_valid")).lower() == "true" else 0 for r in selected_rows]
    ax.imshow(np.array([validity]), aspect="auto", cmap=colors.ListedColormap(["#d62728", "#2ca02c"]))
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("Strip-depth validity: green=valid, red=invalid")
    fig.tight_layout()
    fig.savefig(fig_dir / "strip_depth_validity_grid.png", dpi=180)
    plt.close(fig)


def _write_selected_report(
    *,
    out_dir: Path,
    session_dir: Path,
    mask_run: Path,
    manual_anchor_root: Path | None,
    selection_payloads: dict[str, dict[str, Any]],
    all_anchor_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    selected_summary: list[dict[str, Any]],
) -> None:
    selected_table = []
    for case_id in sorted(selection_payloads):
        payload = selection_payloads[case_id]
        selected_table.append(
            {
                "case_id": case_id,
                "major_selected_id": payload.get("major", {}).get("selected_anchor_id"),
                "minor_selected_id": payload.get("minor", {}).get("selected_anchor_id"),
                "manual_major_mm": payload.get("major", {}).get("manual_mm"),
                "manual_minor_mm": payload.get("minor", {}).get("manual_mm"),
            }
        )
    invalid_selected = [r for r in selected_rows if str(r.get("strip_depth_valid")).lower() != "true"]
    all_invalid_reasons = _failure_reason_counts(all_anchor_rows, "strip_depth_failure_reason")
    selected_invalid_reasons = _failure_reason_counts(invalid_selected, "strip_depth_failure_reason")
    strip_valid_selected = len(selected_rows) - len(invalid_selected)
    all_anchor_valid = sum(1 for r in all_anchor_rows if str(r.get("strip_depth_valid")).lower() == "true")
    lines = [
        "# Selected-Anchor Mask vs Depth Point-Cloud Path-Length Report",
        "",
        f"Dataset: `{session_dir}`",
        f"Mask run: `{mask_run}`",
        f"Manual anchor selection root: `{manual_anchor_root}`",
        "",
        "## Source of Truth",
        f"- Selected anchor files found: {len(selection_payloads)}",
        f"- Selected manual paths evaluated: {len(selected_rows)}",
        "- Evaluation used `selected_anchors.json` A* metadata for p0, direction, endpoints, and manual target.",
        "- No major/minor swapping was performed.",
        "- No minAreaRect fallback axis comparison was used for these selected-path metrics.",
        "- Whole-ROI depth was not used in the main comparison.",
        f"- Depth endpoint mode: `{selected_rows[0].get('depth_endpoint_mode', 'current') if selected_rows else 'current'}`.",
        f"- Projected endpoint quantiles: `{selected_rows[0].get('depth_lower_quantile', '') if selected_rows else ''}` / `{selected_rows[0].get('depth_upper_quantile', '') if selected_rows else ''}`.",
        "- These metrics supersede the previous exploratory minAreaRect/cross-axis results.",
        "",
        "## Selected Anchor IDs",
        "",
        "| case_id | major selected ID | minor selected ID | manual major mm | manual minor mm |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in selected_table:
        lines.append(
            f"| {row['case_id']} | {row['major_selected_id']} | {row['minor_selected_id']} | {_fmt(row['manual_major_mm'])} | {_fmt(row['manual_minor_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## All A* Anchor Measurement Summary",
            f"- Total A* anchors measured: {len(all_anchor_rows)}",
            f"- Anchors with valid mask path length: {sum(1 for r in all_anchor_rows if r.get('mask_path_length_mm') not in (None, ''))}",
            f"- Anchors with valid depth point-cloud path length: {all_anchor_valid}",
            f"- Anchors with invalid depth point-cloud path length: {len(all_anchor_rows) - all_anchor_valid}",
            f"- Top all-anchor depth point-cloud failure reasons: `{all_invalid_reasons}`",
            "",
            "## Selected 30-Path Summary",
            "",
        ]
    )
    if selected_summary:
        headers = list(selected_summary[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in selected_summary:
            lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    lines.extend(
        [
            "",
            "## Was Depth Point-Cloud Path Measurement Possible?",
            f"Selected paths valid: {strip_valid_selected}/{len(selected_rows)}.",
            "Answer: "
            + ("yes, for all selected paths." if strip_valid_selected == len(selected_rows) else ("partially." if strip_valid_selected else "no.")),
            "",
            "## Which Method Is More Accurate?",
        ]
    )
    all_group = next((r for r in selected_summary if r.get("group") == "all"), {})
    mask_mae = all_group.get("mask_MAE_mm")
    strip_mae = all_group.get("strip_depth_MAE_mm")
    if mask_mae is not None and strip_mae is not None:
        better = "depth point-cloud-based" if float(strip_mae) < float(mask_mae) else "mask-based"
        lines.append(f"Across all selected paths with valid depth point-cloud estimates, the lower MAE method is: **{better}**.")
    else:
        lines.append("The comparison is incomplete because one or both methods have no valid MAE.")
    lines.extend(["", "## Invalid Selected Depth Point-Cloud Rows"])
    if invalid_selected:
        lines.append("| case_id | path_label | selected_anchor_id | reason |")
        lines.append("| --- | --- | --- | --- |")
        for row in invalid_selected:
            lines.append(
                f"| {row['case_id']} | {row['path_label']} | {row.get('selected_anchor_id')} | {row.get('strip_depth_failure_reason')} |"
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            f"Selected-path depth point-cloud failure reason counts: `{selected_invalid_reasons}`",
            "",
            "## Output Files",
            f"- `all_anchor_path_measurements.csv`",
            f"- `selected_anchor_mask_strip_depth_vs_manual.csv`",
            f"- `selected_anchor_mask_strip_depth_summary.csv`",
            f"- Diagnostic figures: `{out_dir / 'figures'}`",
        ]
    )
    (out_dir / "selected_anchor_mask_strip_depth_report.md").write_text("\n".join(lines), encoding="utf-8")


def _final_grid(rows: list[dict[str, Any]], cases_data: dict[str, dict[str, Any]], out_png: Path, out_svg: Path | None, label_filter: str | None = None) -> None:
    subset = rows if label_filter is None else [r for r in rows if r["path_label"] == label_filter]
    if not subset:
        return
    cols = 5
    rows_n = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, subset):
        data = cases_data[row["case_id"]]
        rgb = data["rgb"]
        mask = data["mask"]
        p0 = np.array([float(row["p0_x"]), float(row["p0_y"])])
        d = np.array([float(row["direction_x"]), float(row["direction_y"])])
        ref = data["ref_depth_mm"]
        intr = data["intrinsics"]
        img = rgb.copy()
        img[mask > 0] = (0.8 * img[mask > 0] + 0.2 * np.array([0, 80, 255])).astype(np.uint8)
        ax.imshow(img)
        ax.plot(p0[0], p0[1], "ko", markersize=3)
        _draw_line(ax, p0, d, row.get("mask_based_mm"), ref, intr, color="blue", label="mask")
        _draw_line(ax, p0, d, row.get("whole_roi_depth_mm"), ref, intr, color="green", label="whole")
        _draw_line(ax, p0, d, row.get("strip_depth_mm"), ref, intr, color="purple", label="strip", linestyle="--")
        ax.set_title(f"{row['case_id']} {row['path_label']}\nM {row.get('manual_mm')} / m {row.get('mask_based_mm')} / w {row.get('whole_roi_depth_mm')} / s {row.get('strip_depth_mm')}", fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    session_dir = _resolve(args.capture_session)
    mask_run = _resolve(args.mask_run)
    manual_anchor_root = _resolve(args.manual_anchor_selection_root) if args.manual_anchor_selection_root else None
    if manual_anchor_root is None and not args.allow_axis_fallback_error_metrics:
        print(
            "WARNING: No manual anchor/path metadata was provided. This script will use fallback minAreaRect axes "
            "only for diagnostic visualization. The resulting estimates are not valid for error metrics against "
            "manual_major_mm/manual_minor_mm unless those measurements were taken along the same minAreaRect axes. "
            "Pass --allow-axis-fallback-error-metrics to compute those invalid/diagnostic errors anyway.",
            flush=True,
        )
    out_dir = _resolve(args.output_root) / f"session_{_now_id()}"
    diagnostics_dir = out_dir / "diagnostics"
    grids_dir = out_dir / "visual_grids"
    out_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    all_anchor_rows: list[dict[str, Any]] = []
    cases_data: dict[str, dict[str, Any]] = {}
    base_cases = _read_cases(session_dir, args)
    cases, included_selected_cases, skipped_selected_cases = _include_selected_anchor_cases(
        base_cases,
        session_dir=session_dir,
        mask_run=mask_run,
        manual_anchor_root=manual_anchor_root,
        args=args,
    )
    print(f"Included cases: {', '.join(case.case_id for case in cases)}")
    if manual_anchor_root is not None:
        print(f"Selected-anchor cases included: {len(included_selected_cases)}")
        if skipped_selected_cases:
            print(f"Skipped selected-anchor cases: {json.dumps(skipped_selected_cases, indent=2)}")
    _progress(args, "Verifying selected-anchor metadata completeness")
    selection_payloads = _verify_selection_completeness(cases, manual_anchor_root)
    _progress(args, f"Selection payloads verified: {len(selection_payloads)}")
    _progress(args, "Starting per-case all-anchor and selected-path measurements")
    for case_i, case in enumerate(cases, start=1):
        _progress(args, f"[{case_i}/{len(cases)}] Loading case {case.case_id}")
        rgb_bgr = cv2.imread(str(case.rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(case.rgb_path)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_mm = cv2.imread(str(case.depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        intr = _load_intrinsics(case.camera_info_path)
        manual = _read_json(case.manual_path)
        capture_meta = _read_json(case.capture_metadata_path) if case.capture_metadata_path.exists() else {}
        mask, mask_source, mask_meta = _read_mask(mask_run, case.case_id)
        if mask is None:
            for path_label in ["major", "minor"]:
                rows.append({"case_id": case.case_id, "case_index": case.case_index, "material": case.material, "object_id": case.object_id, "path_label": path_label, "manual_mm": manual.get(f"{path_label}_axis_mm"), "mask_valid": False, "whole_roi_depth_valid": False, "strip_depth_valid": False, "mask_failure_reason": mask_source})
            continue
        p0, long_dir, short_dir, box, rect = _rect_axes(mask)
        ref_depth, _ = _pick_ref_depth(depth_mm, p0, args.center_region_radius_px)
        if ref_depth is None:
            ref_depth = float(np.nanmedian(depth_mm[depth_mm > 0]))
        cases_data[case.case_id] = {"rgb": rgb, "mask": mask, "intrinsics": intr, "ref_depth_mm": ref_depth}
        selection_payload = selection_payloads.get(case.case_id)
        _progress(args, f"[{case_i}/{len(cases)}] Measuring all A* anchors for {case.case_id}")
        try:
            case_all_anchor_rows = _all_anchor_measurements(
                    case=case,
                    manual_anchor_root=manual_anchor_root,
                    selection_payload=selection_payload,
                    mask=mask,
                    depth_mm=depth_mm,
                    intr=intr,
                    box=box,
                    ref_depth=ref_depth,
                    args=args,
            )
        except Exception as exc:
            _progress(args, f"[{case_i}/{len(cases)}] All-anchor measurement failed for {case.case_id}: {exc}")
            case_all_anchor_rows = []
        all_anchor_rows.extend(case_all_anchor_rows)
        _progress(args, f"[{case_i}/{len(cases)}] All-anchor rows for {case.case_id}: {len(case_all_anchor_rows)}")
        for path_label, fallback_direction in [("major", long_dir), ("minor", short_dir)]:
            _progress(args, f"[{case_i}/{len(cases)}] Measuring selected path {case.case_id} {path_label}")
            manual_field = f"{path_label}_axis_mm"
            manual_mm = manual.get(manual_field)
            anchor_meta = _load_manual_anchor_selection(manual_anchor_root, case.case_id, path_label)
            if anchor_meta is not None:
                p0_used = anchor_meta["p0"]
                direction = anchor_meta["direction"]
                endpoint1 = anchor_meta.get("endpoint1")
                endpoint2 = anchor_meta.get("endpoint2")
                direction_source = "manual_anchor_selection"
                anchor_source = "manual_anchor_selection"
                evaluation_valid = True
                invalid_reason = None
            else:
                p0_used = p0
                direction = fallback_direction
                endpoint1 = None
                endpoint2 = None
                # Fallback major/minor are geometric axes, not the user's recorded path labels.
                direction_source = f"minAreaRect_{'long' if path_label == 'major' else 'short'}_axis_fallback_no_anchor_metadata"
                anchor_source = "minAreaRect_center_fallback_no_anchor_metadata"
                evaluation_valid = bool(args.allow_axis_fallback_error_metrics)
                invalid_reason = None if evaluation_valid else INVALID_FALLBACK_REASON
            if endpoint1 is not None and endpoint2 is not None:
                endpoint_mask_result = _endpoint_length(endpoint1, endpoint2, ref_depth, intr)
                mask_result = {
                    "valid": bool(endpoint_mask_result.get("valid")),
                    "length_mm": endpoint_mask_result.get("length_mm"),
                    "length_px": endpoint_mask_result.get("length_px"),
                    "p1": endpoint1.tolist(),
                    "p2": endpoint2.tolist(),
                    "failure_reason": endpoint_mask_result.get("failure_reason"),
                    "source": "selected_anchor_endpoints",
                }
            else:
                mask_result = _mask_chord(mask, p0_used, direction, ref_depth, intr)
            whole = _whole_roi_depth(depth_mm, p0_used, direction, box, intr, args)
            strip = _strip_depth(depth_mm, p0_used, direction, box, intr, args)
            if evaluation_valid:
                mask_abs, mask_pct, mask_signed = _err(mask_result.get("length_mm"), manual_mm)
                whole_abs, whole_pct, _ = _err(whole.get("length_mm"), manual_mm)
                strip_abs, strip_pct, strip_signed = _err(strip.get("length_mm"), manual_mm)
            else:
                mask_abs = mask_pct = mask_signed = whole_abs = whole_pct = strip_abs = strip_pct = strip_signed = None
            row = {
                "case_id": case.case_id,
                "case_index": case.case_index,
                "material": case.material,
                "object_id": case.object_id,
                "path_label": path_label,
                "manual_mm": manual_mm,
                "manual_source_field": manual_field,
                "mask_based_mm": mask_result.get("length_mm"),
                "whole_roi_depth_mm": whole.get("length_mm"),
                "strip_depth_mm": strip.get("length_mm"),
                "mask_abs_err_mm": mask_abs,
                "whole_roi_depth_abs_err_mm": whole_abs,
                "strip_depth_abs_err_mm": strip_abs,
                "mask_pct_err": mask_pct,
                "whole_roi_depth_pct_err": whole_pct,
                "strip_depth_pct_err": strip_pct,
                "mask_valid": bool(mask_result.get("valid")),
                "whole_roi_depth_valid": bool(whole.get("valid")),
                "strip_depth_valid": bool(strip.get("valid")),
                "mask_failure_reason": mask_result.get("failure_reason"),
                "whole_roi_depth_failure_reason": whole.get("failure_reason"),
                "strip_depth_failure_reason": strip.get("failure_reason"),
                "mask_source": mask_source,
                "direction_source": direction_source,
                "anchor_source": anchor_source,
                "p0_x": float(p0_used[0]),
                "p0_y": float(p0_used[1]),
                "direction_x": float(direction[0]),
                "direction_y": float(direction[1]),
                "selected_mask_area_px": int(np.count_nonzero(mask)),
                "valid_depth_fraction": capture_meta.get("valid_depth_fraction"),
                "in_range_268_375_fraction": capture_meta.get("in_fixed_range_fraction"),
                "evaluation_valid": evaluation_valid,
                "invalid_reason": invalid_reason,
            }
            diag = {
                "case_id": case.case_id,
                "path_label": path_label,
                "manual_mm": manual_mm,
                "manual_source_field": manual_field,
                "note": (
                    "manual major/minor are path labels. Error metrics are valid only when manual anchor/path metadata "
                    "is used, or when --allow-axis-fallback-error-metrics is explicitly set for diagnostic-only fallback axes."
                ),
                "evaluation_valid": evaluation_valid,
                "invalid_reason": invalid_reason,
                "p0": p0_used.tolist(),
                "direction": direction.tolist(),
                "manual_anchor_selection": anchor_meta["raw"] if anchor_meta is not None else None,
                "manual_anchor_selection_source": anchor_meta["source_path"] if anchor_meta is not None else None,
                "reference_depth_mm": ref_depth,
                "minAreaRect_box": box.tolist(),
                "mask_result": mask_result,
                "whole_roi": whole,
                "strip_depth": strip,
                "mask_metadata": mask_meta,
                "parameters": vars(args),
            }
            _write_json(diagnostics_dir / f"{case.case_id}__{path_label}.json", diag)
            if not args.skip_figures:
                _progress(args, f"[{case_i}/{len(cases)}] Writing step grid {case.case_id} {path_label}")
                _step_grid(
                    case,
                    path_label,
                    rgb,
                    depth_mm,
                    mask,
                    box,
                    p0_used,
                    direction,
                    row,
                    diag,
                    grids_dir / f"{case.case_id}__{path_label}__step_grid.png",
                    None if args.no_svg else grids_dir / f"{case.case_id}__{path_label}__step_grid.svg",
                    intr,
                )
            rows.append(row)
            strip_candidate_count = _strip_candidate_count(strip)
            strip_valid_depth_count = _strip_valid_depth_count(strip)
            best_method = None
            if mask_abs is not None and strip_abs is not None:
                best_method = "strip_depth" if float(strip_abs) < float(mask_abs) else "mask"
            elif mask_abs is not None:
                best_method = "mask"
            elif strip_abs is not None:
                best_method = "strip_depth"
            selected_rows.append(
                {
                    "case_id": case.case_id,
                    "case_index": case.case_index,
                    "material": case.material,
                    "object_id": case.object_id,
                    "path_label": path_label,
                    "manual_mm": manual_mm,
                    "selected_anchor_id": anchor_meta.get("selected_anchor_id") if anchor_meta else None,
                    "selection_source": anchor_meta.get("selection_source") if anchor_meta else None,
                    "p0_x": float(p0_used[0]),
                    "p0_y": float(p0_used[1]),
                    "direction_x": float(direction[0]),
                    "direction_y": float(direction[1]),
                    "endpoint1_x": float(endpoint1[0]) if endpoint1 is not None else None,
                    "endpoint1_y": float(endpoint1[1]) if endpoint1 is not None else None,
                    "endpoint2_x": float(endpoint2[0]) if endpoint2 is not None else None,
                    "endpoint2_y": float(endpoint2[1]) if endpoint2 is not None else None,
                    "mask_path_length_px": mask_result.get("length_px") or abs(float(mask_result.get("t1_px", 0) or 0) - float(mask_result.get("t0_px", 0) or 0)),
                    "mask_path_length_mm": mask_result.get("length_mm"),
                    "mask_abs_error_mm": mask_abs,
                    "mask_signed_error_mm": mask_signed,
                    "mask_pct_error": mask_pct,
                    "strip_depth_path_length_mm": strip.get("length_mm"),
                    "strip_depth_valid": bool(strip.get("valid")),
                    "strip_depth_failure_reason": strip.get("failure_reason"),
                    "strip_depth_num_supported_points": _strip_supported_count(strip),
                    "strip_depth_valid_depth_fraction": (strip_valid_depth_count / strip_candidate_count) if strip_candidate_count else None,
                    "strip_depth_abs_error_mm": strip_abs,
                    "strip_depth_signed_error_mm": strip_signed,
                    "strip_depth_pct_error": strip_pct,
                    "depth_endpoint_mode": args.depth_endpoint_mode,
                    "depth_lower_quantile": args.lower_quantile,
                    "depth_upper_quantile": args.upper_quantile,
                    "depth_support_projected_lower": _strip_quantile_mm(strip, "lower_projected_support_m"),
                    "depth_support_projected_upper": _strip_quantile_mm(strip, "upper_projected_support_m"),
                    "depth_support_projected_span_mm": _strip_projected_span_mm(strip),
                    "best_method_for_this_path_by_abs_error": best_method,
                    "notes": "Whole-ROI depth was not used in this main selected-anchor comparison.",
                }
            )

    _progress(args, "Starting summary/report writing")
    summary_rows, ranking_rows = _summary(rows)
    selected_summary = _selected_summary_rows(selected_rows)
    _write_csv(out_dir / "new_capture_mask_depth_comparison_per_path.csv", COMPARISON_COLUMNS, rows)
    _write_csv(out_dir / "all_anchor_path_measurements.csv", ALL_ANCHOR_COLUMNS, all_anchor_rows)
    _write_csv(out_dir / "selected_anchor_mask_strip_depth_vs_manual.csv", SELECTED_COLUMNS, selected_rows)
    _write_csv(out_dir / "selected_anchor_mask_strip_depth_summary.csv", _selected_summary_columns(selected_summary), selected_summary)
    summary_cols = list(summary_rows[0].keys()) if summary_rows else ["path_label"]
    _write_csv(out_dir / "new_capture_mask_depth_summary.csv", summary_cols, summary_rows)
    _write_csv(out_dir / "method_ranking.csv", list(ranking_rows[0].keys()) if ranking_rows else ["path_label"], ranking_rows)
    if not args.skip_figures:
        _progress(args, "Writing final chord grids")
        _final_grid(rows, cases_data, grids_dir / "final_chord_grid_all_paths.png", None if args.no_svg else grids_dir / "final_chord_grid_all_paths.svg")
        _final_grid(rows, cases_data, grids_dir / "final_chord_grid_major.png", None if args.no_svg else grids_dir / "final_chord_grid_major.svg", "major")
        _final_grid(rows, cases_data, grids_dir / "final_chord_grid_minor.png", None if args.no_svg else grids_dir / "final_chord_grid_minor.svg", "minor")
    _write_json(
        out_dir / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_branch": _git_value(["git", "branch", "--show-current"]),
            "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
            "capture_session": str(session_dir),
            "mask_run": str(mask_run),
            "parameters": vars(args),
            "depth_endpoint_mode": args.depth_endpoint_mode,
            "depth_lower_quantile": args.lower_quantile,
            "depth_upper_quantile": args.upper_quantile,
            "included_cases": [case.case_id for case in cases],
            "included_selected_anchor_cases": included_selected_cases,
            "skipped_selected_anchor_cases": skipped_selected_cases,
            "robot_motion_used": False,
            "rtde_used": False,
        },
    )
    _write_report(out_dir, session_dir, mask_run, rows, summary_rows, ranking_rows)
    if not args.skip_figures:
        _progress(args, "Writing selected-path diagnostic figures")
        _make_selected_figures(out_dir, selected_rows)
    _write_selected_report(
        out_dir=out_dir,
        session_dir=session_dir,
        mask_run=mask_run,
        manual_anchor_root=manual_anchor_root,
        selection_payloads=selection_payloads,
        all_anchor_rows=all_anchor_rows,
        selected_rows=selected_rows,
        selected_summary=selected_summary,
    )
    _progress(args, f"Finished evaluation: {out_dir}")
    return out_dir


def _write_report(out_dir: Path, session_dir: Path, mask_run: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]) -> None:
    mask_success = len({r["case_id"] for r in rows if r.get("mask_valid")})
    mask_cases = len({r["case_id"] for r in rows})
    all_valid_for_metrics = all(bool(r.get("evaluation_valid")) for r in rows)
    invalid_count = sum(1 for r in rows if not r.get("evaluation_valid"))
    title = (
        "# New Capture Mask/Depth Manual Evaluation"
        if all_valid_for_metrics
        else "# DIAGNOSTIC ONLY - MANUAL ERROR METRICS DISABLED"
    )
    lines = [
        title,
        "",
        "## Purpose",
        "Evaluate GSAM2-mask, whole-ROI depth-image, and strip-depth path-length estimates against the manually recorded path-length labels.",
        "",
        f"Dataset: `{session_dir}`",
        f"Mask run: `{mask_run}`",
        "",
        "## Important Notes",
        "- Cases 001-005 are treated as timber, 006-010 as brick, and 011-015 as concrete block.",
        "- `major_axis_mm` and `minor_axis_mm` are manual path labels from the capture workflow, not proof of true object geometry.",
        (
            "- Manual anchor/path metadata was used, so manual error metrics are enabled."
            if all_valid_for_metrics
            else "- No explicit anchor metadata was provided for at least one row, so fallback minAreaRect axes are diagnostic only and manual error metrics are disabled."
        ),
        f"- Rows with disabled/invalid manual-error metrics: {invalid_count}",
        "",
        "## Depth Visualization Settings",
        "- Colormap: viridis",
        "- Fixed limits: 268-375 mm",
        "- Clipping: enabled",
        "- Invalid depth: black/white versions are generated by the separate depth recreation script; diagnostics use black invalid.",
        "",
        "## GSAM2 Mask Generation",
        f"- Masks usable in evaluation: {mask_success}/{mask_cases} cases",
        "",
        "## Quantitative Summary",
        "",
    ]
    if summary_rows:
        headers = list(summary_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in summary_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines += [
        "",
        "## Method Ranking",
        "",
    ]
    if ranking_rows:
        headers = list(ranking_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in ranking_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines += [
        "",
        "## Visual Outputs",
        f"- Per-case step grids: `{out_dir / 'visual_grids'}`",
        f"- Final chord grids: `{out_dir / 'visual_grids'}`",
        "",
        "## Interpretation",
        "Use the grids to check whether depth errors come from sparse/missing depth, GSAM2 foreground mistakes, or the minAreaRect fallback not matching the manual path setup.",
        "",
        "## Recommendation",
        "This evaluation is useful for a quick offline check. A stronger local-anchor depth comparison requires saving explicit anchor centers and path directions during capture.",
    ]
    (out_dir / "new_capture_mask_depth_manual_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSAM2 mask, whole-ROI depth, and strip-depth path lengths on a new ROS2 capture session.")
    parser.add_argument("--capture-session", default=str(DEFAULT_SESSION))
    parser.add_argument("--mask-run", default=str(DEFAULT_SESSION / "gsam2_masks"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--buffer-percent", type=float, default=4.0)
    parser.add_argument("--depth-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--lower-quantile", type=float, default=0.0025)
    parser.add_argument("--upper-quantile", type=float, default=0.9975)
    parser.add_argument(
        "--depth-endpoint-mode",
        choices=["current", "percentile"],
        default="current",
        help=(
            "Projected endpoint mode for the selected-anchor depth point-cloud estimate. "
            "The current implementation already uses robust quantile endpoints; percentile mode records that V2 intent explicitly."
        ),
    )
    parser.add_argument("--depth-lower-quantile", type=float, default=None, help="Alias for --lower-quantile.")
    parser.add_argument("--depth-upper-quantile", type=float, default=None, help="Alias for --upper-quantile.")
    parser.add_argument("--center-region-radius-px", type=float, default=15.0)
    parser.add_argument("--min-supported-points", type=int, default=100)
    parser.add_argument("--strip-half-width-px", type=float, default=6.0)
    parser.add_argument("--strip-offsets-px", default="-18,-12,-6,0,6,12,18")
    parser.add_argument(
        "--manual-anchor-selection-root",
        default=None,
        help=(
            "Optional folder containing selected_anchors.json files. When provided, selected p0/direction metadata "
            "is used and manual path-label error metrics are valid."
        ),
    )
    parser.add_argument(
        "--allow-axis-fallback-error-metrics",
        action="store_true",
        help=(
            "Explicitly compute diagnostic manual errors using minAreaRect long/short fallback axes. "
            "Do not use these metrics for method comparison unless manual measurements were taken along those axes."
        ),
    )
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--debug-progress", action="store_true", help="Print per-case/per-path progress while evaluating.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip step grids, final chord grids, and diagnostic figures.")
    parser.add_argument("--no-svg", action="store_true", help="When figures are enabled, write PNG only and skip SVG outputs.")
    parser.add_argument("--case-timeout-sec", type=float, default=120.0, help="Reserved diagnostic timeout value recorded in run_config; no subprocess timeout is used.")
    args = parser.parse_args()
    if args.depth_lower_quantile is not None:
        args.lower_quantile = args.depth_lower_quantile
    if args.depth_upper_quantile is not None:
        args.upper_quantile = args.depth_upper_quantile
    args.strip_offsets_px = [float(v.strip()) for v in str(args.strip_offsets_px).split(",") if v.strip()]
    return args


def main() -> int:
    out_dir = run(parse_args())
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
