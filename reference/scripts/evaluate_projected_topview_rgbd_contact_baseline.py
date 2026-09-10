#!/usr/bin/env python3
"""Evaluate a training-free projected top-view RGB-D contact baseline.

The script reads the saved E3/E45 82-anchor benchmark lineage. It does not run
segmentation, VLM inference, camera capture, robot control, or UPV hardware.

The fitted top plane is used only as a metric projection coordinate system.
All valid depth points in each contact ROI are retained for projected occupancy
analysis; no vertical side-plane model is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINEAGE = Path("outputs/genesis_82_pair_benchmark_per_case_lineage.csv")
DEFAULT_LABELS = Path("data/contact_anchor_selection_benchmark_E3_E45_82/labels_all.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs/projected_topview_rgbd_contact_baseline")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "good", "acceptable", "usable"}:
        return True
    if text in {"false", "0", "no", "bad", "unusable"}:
        return False
    return None


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def _intrinsics(camera_info: dict[str, Any]) -> tuple[float, float, float, float]:
    if all(key in camera_info for key in ("fx", "fy", "cx", "cy")):
        return tuple(float(camera_info[key]) for key in ("fx", "fy", "cx", "cy"))
    values = camera_info.get("k") or camera_info.get("K")
    if not values or len(values) < 6:
        raise ValueError("Camera info has neither fx/fy/cx/cy nor a valid K matrix.")
    return float(values[0]), float(values[4]), float(values[2]), float(values[5])


def _depth_to_mm(depth: np.ndarray) -> np.ndarray:
    result = depth.astype(np.float64)
    positive = result[result > 0]
    if positive.size and float(np.median(positive)) < 10.0:
        result *= 1000.0
    return result


def _backproject(depth_mm: np.ndarray, intr: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = intr
    yy, xx = np.indices(depth_mm.shape, dtype=np.float64)
    z = depth_mm
    return np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=-1)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = _normalize(vh[-1])
    if float(np.dot(normal, -center)) < 0:
        normal = -normal
    return normal, -float(np.dot(normal, center))


def _fit_top_plane(
    xyz: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    *,
    erode_px: int,
    ransac_threshold_mm: float,
    ransac_iterations: int,
    seed: int,
) -> dict[str, Any]:
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    interior = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    valid = interior & (depth_mm > 0)
    depths = depth_mm[valid]
    if depths.size < 300:
        raise ValueError(f"Insufficient top-plane points after erosion: {depths.size}")
    median_depth = float(np.median(depths))
    valid &= np.abs(depth_mm - median_depth) <= 20.0
    points = xyz[valid]
    if points.shape[0] < 300:
        raise ValueError(f"Insufficient top-plane points after depth filtering: {points.shape[0]}")

    rng = np.random.default_rng(seed)
    fit_points = points
    if fit_points.shape[0] > 7000:
        fit_points = fit_points[rng.choice(fit_points.shape[0], 7000, replace=False)]
    best_inliers: np.ndarray | None = None
    best_median = float("inf")
    for _ in range(ransac_iterations):
        sample = fit_points[rng.choice(fit_points.shape[0], 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        offset = -float(np.dot(normal, sample[0]))
        residual = np.abs(fit_points @ normal + offset)
        inliers = residual <= ransac_threshold_mm
        if not np.any(inliers):
            continue
        med = float(np.median(residual[inliers]))
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()) or (
            int(inliers.sum()) == int(best_inliers.sum()) and med < best_median
        ):
            best_inliers = inliers
            best_median = med
    if best_inliers is None or int(best_inliers.sum()) < 100:
        raise ValueError("Top-plane RANSAC did not find a supported plane.")

    normal, offset = _fit_plane_svd(fit_points[best_inliers])
    residual = fit_points @ normal + offset
    med = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - med)))
    refine = np.abs(residual - med) <= max(ransac_threshold_mm, 3.0 * sigma)
    normal, offset = _fit_plane_svd(fit_points[refine])
    residual = fit_points @ normal + offset
    med = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - med)))
    return {
        "normal": normal,
        "offset": offset,
        "noise_sigma_mm": sigma,
        "median_depth_mm": median_depth,
        "interior_point_count": int(points.shape[0]),
        "fit_point_count": int(fit_points.shape[0]),
        "inlier_count": int(refine.sum()),
        "inlier_fraction": float(refine.mean()),
        "eroded_mask": interior,
    }


def _pixel_rays(shape: tuple[int, int], intr: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = intr
    yy, xx = np.indices(shape, dtype=np.float64)
    return np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones(shape)), axis=-1)


def _intersect_rays_with_plane(rays: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    denominator = np.einsum("...i,i->...", rays, normal)
    scale = np.divide(-offset, denominator, out=np.full_like(denominator, np.nan), where=np.abs(denominator) > 1e-9)
    return rays * scale[..., None]


def _pixel_plane_point(
    point_xy: list[float] | tuple[float, float],
    intr: tuple[float, float, float, float],
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    fx, fy, cx, cy = intr
    x, y = float(point_xy[0]), float(point_xy[1])
    ray = np.asarray([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) <= 1e-9:
        raise ValueError("Pixel ray is parallel to the top plane.")
    return ray * (-offset / denominator)


def _fit_boundary_line_ransac(
    u: np.ndarray,
    v: np.ndarray,
    *,
    threshold_mm: float,
    iterations: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    if u.size < 12:
        return 0.0, 0.0, np.zeros(u.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_med = float("inf")
    for _ in range(iterations):
        idx = rng.choice(u.size, 2, replace=False)
        if abs(float(u[idx[1]] - u[idx[0]])) <= 1e-6:
            continue
        slope = float((v[idx[1]] - v[idx[0]]) / (u[idx[1]] - u[idx[0]]))
        intercept = float(v[idx[0]] - slope * u[idx[0]])
        residual = np.abs(v - (slope * u + intercept)) / math.sqrt(1.0 + slope * slope)
        inliers = residual <= threshold_mm
        med = float(np.median(residual[inliers])) if np.any(inliers) else float("inf")
        if best is None or int(inliers.sum()) > int(best.sum()) or (int(inliers.sum()) == int(best.sum()) and med < best_med):
            best, best_med = inliers, med
    if best is None or int(best.sum()) < 8:
        return 0.0, float(np.median(v)), np.ones(u.shape, dtype=bool)
    design = np.column_stack((u[best], np.ones(int(best.sum()))))
    slope, intercept = np.linalg.lstsq(design, v[best], rcond=None)[0]
    return float(slope), float(intercept), best


def _longest_true_run(values: np.ndarray, grid_mm: float) -> float:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * grid_mm)


def _component_metrics(
    grid: np.ndarray,
    distance_grid: np.ndarray,
    *,
    grid_mm: float,
    min_area_mm2: float,
    min_span_mm: float,
) -> dict[str, Any]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8), connectivity=8)
    components: list[dict[str, float]] = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA]) * grid_mm * grid_mm
        width = float(stats[label, cv2.CC_STAT_WIDTH]) * grid_mm
        height = float(stats[label, cv2.CC_STAT_HEIGHT]) * grid_mm
        component_values = distance_grid[labels == label]
        components.append(
            {
                "area_mm2": area,
                "u_span_mm": width,
                "v_span_mm": height,
                "max_distance_mm": float(np.max(component_values)) if component_values.size else 0.0,
                "mean_distance_mm": float(np.mean(component_values)) if component_values.size else 0.0,
                "coherent": bool(area >= min_area_mm2 and width >= min_span_mm),
            }
        )
    coherent = [item for item in components if item["coherent"]]
    return {
        "component_count": len(components),
        "coherent_component_count": len(coherent),
        "largest_area_mm2": max((item["area_mm2"] for item in coherent), default=0.0),
        "largest_u_span_mm": max((item["u_span_mm"] for item in coherent), default=0.0),
        "max_distance_mm": max((item["max_distance_mm"] for item in coherent), default=0.0),
        "components": components,
        "labels": labels,
    }


@dataclass
class SideResult:
    side_name: str
    contact_point_xy: list[float]
    lateral_half_width_mm: float
    effective_threshold_mm: float
    plane_noise_sigma_mm: float
    depth_valid_fraction: float
    boundary_support_fraction: float
    boundary_line_slope: float
    boundary_line_intercept_mm: float
    boundary_fit_inlier_fraction: float
    boundary_residual_p95_mm: float
    outward_max_mm: float
    outward_area_mm2: float
    outward_span_mm: float
    inward_max_mm: float
    inward_run_mm: float
    missing_boundary_run_mm: float
    raised_height_max_mm: float
    raised_height_area_mm2: float
    rgb_edge_support_fraction: float
    background_height_mm: float | None
    decision_good: bool
    veto_reasons: list[str]
    safety_score: float


def _analyse_side(
    *,
    contact_point: list[float],
    anchor_point: list[float],
    side_name: str,
    plane: dict[str, Any],
    xyz: np.ndarray,
    plane_xyz: np.ndarray,
    plane_intersections: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    canny: np.ndarray,
    intr: tuple[float, float, float, float],
    contour_plane_points: np.ndarray,
    lateral_half_width_mm: float,
    args: argparse.Namespace,
    seed: int,
) -> tuple[SideResult, dict[str, Any]]:
    normal = plane["normal"]
    offset = float(plane["offset"])
    contact_3d = _pixel_plane_point(contact_point, intr, normal, offset)
    anchor_3d = _pixel_plane_point(anchor_point, intr, normal, offset)
    outward = _normalize(contact_3d - anchor_3d)
    tangent = _normalize(np.cross(normal, outward))
    # Re-orthogonalize outward to avoid numeric drift.
    outward = _normalize(np.cross(tangent, normal))
    if float(np.dot(outward, contact_3d - anchor_3d)) < 0:
        outward, tangent = -outward, -tangent

    rel_plane = plane_intersections - contact_3d
    u_all = np.einsum("...i,i->...", rel_plane, tangent)
    v_all = np.einsum("...i,i->...", rel_plane, outward)
    local_mask_pixels = (
        (np.abs(u_all) <= lateral_half_width_mm)
        & (v_all >= -args.inside_margin_mm)
        & (v_all <= args.outside_margin_mm)
    )
    support_zone = local_mask_pixels & mask & (v_all <= 2.0) & (v_all >= -12.0)
    support_total = int(support_zone.sum())
    depth_valid_fraction = float(((depth_mm > 0) & support_zone).sum() / support_total) if support_total else 0.0

    rel_contour = contour_plane_points - contact_3d
    contour_u = rel_contour @ tangent
    contour_v = rel_contour @ outward
    contour_keep = (np.abs(contour_u) <= args.boundary_fit_half_width_mm) & (np.abs(contour_v) <= args.boundary_fit_search_mm)
    cu, cv = contour_u[contour_keep], contour_v[contour_keep]
    slope, intercept, line_inliers = _fit_boundary_line_ransac(
        cu,
        cv,
        threshold_mm=args.boundary_ransac_threshold_mm,
        iterations=args.boundary_ransac_iterations,
        seed=seed,
    )
    line_inlier_fraction = float(line_inliers.mean()) if line_inliers.size else 0.0

    bins = np.arange(-lateral_half_width_mm, lateral_half_width_mm + args.grid_mm, args.grid_mm)
    centers = (bins[:-1] + bins[1:]) * 0.5
    observed = np.full(centers.shape, np.nan, dtype=np.float64)
    for index in range(centers.size):
        selected = (contour_u >= bins[index]) & (contour_u < bins[index + 1]) & (np.abs(contour_v) <= args.boundary_fit_search_mm)
        if np.any(selected):
            observed[index] = float(np.max(contour_v[selected]))
    expected = slope * centers + intercept
    profile_residual = observed - expected
    finite = np.isfinite(profile_residual)
    boundary_support_fraction = float(finite.mean()) if finite.size else 0.0
    residual_abs = np.abs(profile_residual[finite])
    boundary_residual_p95 = float(np.quantile(residual_abs, 0.95)) if residual_abs.size else float("inf")

    effective_threshold = max(float(args.defect_threshold_mm), float(args.noise_multiplier) * float(plane["noise_sigma_mm"]))
    outward_profile = finite & (profile_residual > effective_threshold)
    inward_profile = finite & (profile_residual < -effective_threshold)
    missing_profile = ~finite
    outward_profile_max = float(np.max(profile_residual[outward_profile])) if np.any(outward_profile) else 0.0
    inward_profile_max = float(np.max(-profile_residual[inward_profile])) if np.any(inward_profile) else 0.0
    inward_run = _longest_true_run(inward_profile, args.grid_mm)
    missing_run = _longest_true_run(missing_profile, args.grid_mm)

    valid = (depth_mm > 0) & local_mask_pixels
    flat_xyz = xyz[valid]
    if flat_xyz.size:
        signed_height = flat_xyz @ normal + offset
        projected = flat_xyz - signed_height[:, None] * normal
        rel = projected - contact_3d
        du = rel @ tangent
        dv = rel @ outward
        source_mask = mask[valid]
        source_edges = canny[valid] > 0
    else:
        signed_height = np.empty(0)
        du = dv = np.empty(0)
        source_mask = source_edges = np.empty(0, dtype=bool)

    outside_background = (du >= -lateral_half_width_mm) & (du <= lateral_half_width_mm) & (dv >= 8.0)
    bg_values = signed_height[outside_background & ~source_mask]
    background_height = float(np.median(bg_values)) if bg_values.size >= 20 else None
    if background_height is None:
        object_like = source_mask | (signed_height >= -args.max_object_below_top_mm)
    else:
        # A sloped/noisy table can differ from its local median by several mm.
        # Require non-mask points to clear a meaningful fraction of the measured
        # top-to-background separation before treating them as foreground.
        background_clearance = max(
            effective_threshold,
            abs(background_height) * args.background_separation_fraction,
        )
        object_like = source_mask | (signed_height >= background_height + background_clearance)

    u_bins = bins
    v_bins = np.arange(-args.inside_margin_mm, args.outside_margin_mm + args.grid_mm, args.grid_mm)
    grid_shape = (len(v_bins) - 1, len(u_bins) - 1)
    occupancy = np.zeros(grid_shape, dtype=np.uint8)
    height_max = np.full(grid_shape, np.nan, dtype=np.float64)
    ui = np.floor((du - u_bins[0]) / args.grid_mm).astype(int) if du.size else np.empty(0, dtype=int)
    vi = np.floor((dv - v_bins[0]) / args.grid_mm).astype(int) if dv.size else np.empty(0, dtype=int)
    in_grid = (ui >= 0) & (ui < grid_shape[1]) & (vi >= 0) & (vi < grid_shape[0])
    for x_index, y_index, keep_obj, height in zip(ui[in_grid], vi[in_grid], object_like[in_grid], signed_height[in_grid]):
        if keep_obj:
            occupancy[y_index, x_index] = 1
        old = height_max[y_index, x_index]
        if not math.isfinite(old) or height > old:
            height_max[y_index, x_index] = height

    uu, vv = np.meshgrid(centers, (v_bins[:-1] + v_bins[1:]) * 0.5)
    line_grid = slope * uu + intercept
    distance_grid = vv - line_grid
    outward_grid = (occupancy > 0) & (distance_grid > effective_threshold)
    outward_components = _component_metrics(
        outward_grid,
        np.maximum(distance_grid, 0.0),
        grid_mm=args.grid_mm,
        min_area_mm2=args.min_component_area_mm2,
        min_span_mm=args.min_component_span_mm,
    )

    raised_grid = (
        np.isfinite(height_max)
        & (height_max > effective_threshold)
        & (np.abs(distance_grid) <= args.raised_edge_band_mm)
    )
    raised_components = _component_metrics(
        raised_grid,
        np.maximum(np.nan_to_num(height_max, nan=0.0), 0.0),
        grid_mm=args.grid_mm,
        min_area_mm2=args.min_component_area_mm2,
        min_span_mm=args.min_component_span_mm,
    )
    outward_max = max(outward_profile_max, float(outward_components["max_distance_mm"]))

    edge_near_line = source_edges & (np.abs(dv - (slope * du + intercept)) <= args.rgb_edge_band_mm)
    line_zone = np.abs(dv - (slope * du + intercept)) <= args.rgb_edge_band_mm
    rgb_edge_support = float(edge_near_line.sum() / line_zone.sum()) if np.any(line_zone) else 0.0

    veto: list[str] = []
    if depth_valid_fraction < args.min_depth_valid_fraction:
        veto.append("insufficient_depth_support")
    if boundary_support_fraction < args.min_boundary_support_fraction or missing_run > args.max_missing_boundary_run_mm:
        veto.append("boundary_support_gap")
    if outward_components["coherent_component_count"] > 0 or _longest_true_run(outward_profile, args.grid_mm) >= args.min_component_span_mm:
        veto.append("coherent_outward_obstruction")
    if inward_run >= args.min_component_span_mm:
        veto.append("coherent_inward_notch")
    if raised_components["coherent_component_count"] > 0:
        veto.append("coherent_raised_height")
    if line_inlier_fraction < args.min_boundary_line_inlier_fraction:
        veto.append("unstable_expected_edge_fit")

    risks = [
        outward_max / effective_threshold,
        inward_profile_max / effective_threshold,
        inward_run / max(args.min_component_span_mm, 1e-6),
        missing_run / max(args.max_missing_boundary_run_mm, 1e-6),
        float(raised_components["max_distance_mm"]) / effective_threshold,
        max(0.0, args.min_depth_valid_fraction - depth_valid_fraction) / max(args.min_depth_valid_fraction, 1e-6),
    ]
    safety = float(np.clip(100.0 * (1.0 - max(risks)), 0.0, 100.0))
    result = SideResult(
        side_name=side_name,
        contact_point_xy=[float(contact_point[0]), float(contact_point[1])],
        lateral_half_width_mm=float(lateral_half_width_mm),
        effective_threshold_mm=effective_threshold,
        plane_noise_sigma_mm=float(plane["noise_sigma_mm"]),
        depth_valid_fraction=depth_valid_fraction,
        boundary_support_fraction=boundary_support_fraction,
        boundary_line_slope=slope,
        boundary_line_intercept_mm=intercept,
        boundary_fit_inlier_fraction=line_inlier_fraction,
        boundary_residual_p95_mm=boundary_residual_p95,
        outward_max_mm=outward_max,
        outward_area_mm2=float(outward_components["largest_area_mm2"]),
        outward_span_mm=float(outward_components["largest_u_span_mm"]),
        inward_max_mm=inward_profile_max,
        inward_run_mm=inward_run,
        missing_boundary_run_mm=missing_run,
        raised_height_max_mm=float(raised_components["max_distance_mm"]),
        raised_height_area_mm2=float(raised_components["largest_area_mm2"]),
        rgb_edge_support_fraction=rgb_edge_support,
        background_height_mm=background_height,
        decision_good=not veto,
        veto_reasons=veto,
        safety_score=safety,
    )
    debug = {
        "u_centers": centers,
        "expected_boundary": expected,
        "observed_boundary": observed,
        "u_grid": uu,
        "v_grid": vv,
        "occupancy": occupancy,
        "height_max": height_max,
        "outward_grid": outward_grid,
        "outward_labels": outward_components["labels"],
        "raised_grid": raised_grid,
        "rgb_edge_support_fraction": rgb_edge_support,
        "contact_3d": contact_3d,
        "tangent_3d": tangent,
        "outward_3d": outward,
    }
    return result, debug


def _confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        manual = bool(row["manual_is_usable"])
        predicted = bool(row["predicted_usable"])
        if manual and predicted:
            tp += 1
        elif not manual and predicted:
            fp += 1
        elif not manual and not predicted:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else None,
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _save_side_visual(
    path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    anchor: dict[str, Any],
    side: SideResult,
    debug: dict[str, Any],
    group_id: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    endpoint = np.asarray(side.contact_point_xy)
    center = np.asarray(anchor["anchor_px"], dtype=float)
    direction = endpoint - center
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    tangent = np.asarray([-direction[1], direction[0]])
    half_px = 65
    polygon = np.asarray(
        [endpoint - tangent * half_px - direction * 65, endpoint + tangent * half_px - direction * 65,
         endpoint + tangent * half_px + direction * 55, endpoint - tangent * half_px + direction * 55]
    )
    axes[0, 0].imshow(rgb)
    layer = np.zeros((*mask.shape, 4), dtype=float)
    layer[mask] = (0.0, 0.7, 0.4, 0.22)
    axes[0, 0].imshow(layer)
    axes[0, 0].plot(*np.vstack((polygon, polygon[0])).T, color="yellow", linewidth=2)
    axes[0, 0].plot([center[0], endpoint[0]], [center[1], endpoint[1]], color="magenta", linewidth=3)
    axes[0, 0].scatter(*endpoint, color="white", edgecolor="black", s=65)
    axes[0, 0].set_title("RGB, mask, and contact ROI")
    axes[0, 0].axis("off")

    extent = [float(debug["u_grid"].min()), float(debug["u_grid"].max()), float(debug["v_grid"].min()), float(debug["v_grid"].max())]
    axes[0, 1].imshow(debug["occupancy"], origin="lower", extent=extent, aspect="auto", cmap="gray_r")
    axes[0, 1].contour(debug["u_grid"], debug["v_grid"], debug["outward_grid"].astype(float), levels=[0.5], colors=["red"], linewidths=2)
    axes[0, 1].plot(debug["u_centers"], debug["expected_boundary"], color="cyan", linewidth=2, label="expected edge")
    axes[0, 1].set_title("Projected top-view occupancy\nred = outward obstruction")
    axes[0, 1].set_xlabel("along edge u (mm)")
    axes[0, 1].set_ylabel("outward v (mm)")
    axes[0, 1].legend(loc="lower right")

    image = axes[1, 0].imshow(debug["height_max"], origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    axes[1, 0].plot(debug["u_centers"], debug["expected_boundary"], color="black", linewidth=1.5)
    axes[1, 0].set_title("Signed height retained after projection (mm)")
    axes[1, 0].set_xlabel("along edge u (mm)")
    axes[1, 0].set_ylabel("outward v (mm)")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].plot(debug["u_centers"], debug["observed_boundary"], color="#d55e00", linewidth=2, label="observed mask edge")
    axes[1, 1].plot(debug["u_centers"], debug["expected_boundary"], color="#0072b2", linewidth=2, label="robust expected edge")
    axes[1, 1].fill_between(
        debug["u_centers"],
        debug["expected_boundary"] - side.effective_threshold_mm,
        debug["expected_boundary"] + side.effective_threshold_mm,
        color="#009e73",
        alpha=0.18,
        label="accepted deviation band",
    )
    axes[1, 1].set_title("Boundary continuity and decision")
    axes[1, 1].set_xlabel("along edge u (mm)")
    axes[1, 1].set_ylabel("outward v (mm)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].text(
        0.02,
        0.03,
        f"decision={'GOOD' if side.decision_good else 'BAD'}\n"
        f"threshold={side.effective_threshold_mm:.2f} mm\n"
        f"outward={side.outward_max_mm:.2f} mm; inward={side.inward_max_mm:.2f} mm\n"
        f"depth valid={side.depth_valid_fraction:.2f}\n"
        f"veto={', '.join(side.veto_reasons) or 'none'}",
        transform=axes[1, 1].transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#444444"},
    )
    fig.suptitle(f"{group_id} | {anchor['anchor_id']} | {side.side_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_group_grid(path: Path, visuals: list[tuple[str, Path]]) -> None:
    if not visuals:
        return
    cols = 2
    rows = math.ceil(len(visuals) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 6.2))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (label, image_path) in zip(axes_array, visuals):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        axis.imshow(image)
        axis.set_title(label, fontsize=11)
        axis.axis("off")
    for axis in axes_array[len(visuals):]:
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_pipeline_overview(
    path: Path,
    *,
    group_id: str,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    plane: dict[str, Any],
    anchors: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    selected_anchor_id: str,
) -> None:
    """Render the deterministic stages without implying that guide-line color is a score."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Saved RGB input")

    valid_depth = depth_mm > 0
    depth_display = np.ma.masked_where(~valid_depth, depth_mm)
    if np.any(valid_depth):
        vmin, vmax = np.quantile(depth_mm[valid_depth], [0.01, 0.99])
    else:
        vmin, vmax = 0.0, 1.0
    image = axes[0, 1].imshow(depth_display, cmap="viridis", vmin=float(vmin), vmax=float(vmax))
    axes[0, 1].set_facecolor("black")
    axes[0, 1].set_title("2. Saved aligned depth (mm)")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(rgb)
    mask_layer = np.zeros((*mask.shape, 4), dtype=float)
    mask_layer[mask] = (0.0, 0.75, 0.35, 0.35)
    axes[0, 2].imshow(mask_layer)
    axes[0, 2].set_title(
        "3. Fixed mask + top-plane metric frame\n"
        f"RANSAC inliers={plane['inlier_fraction']:.2f}; MAD sigma={plane['noise_sigma_mm']:.2f} mm"
    )

    axes[1, 0].imshow(rgb)
    for anchor in anchors:
        center = np.asarray(anchor["anchor_px"], dtype=float)
        point_a = np.asarray(anchor["contact_point_a_px"], dtype=float)
        point_b = np.asarray(anchor["contact_point_b_px"], dtype=float)
        axes[1, 0].plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color="magenta", linewidth=2)
        axes[1, 0].scatter([point_a[0], point_b[0]], [point_a[1], point_b[1]], color="white", edgecolor="black", s=22)
        axes[1, 0].text(center[0] + 5, center[1] - 5, anchor["anchor_id"], color="yellow", fontsize=10, fontweight="bold")
    axes[1, 0].set_title("4. A* candidate geometry\nmagenta = candidate path, not quality")

    axes[1, 1].imshow(rgb)
    rows_by_id = {str(row["anchor_id"]): row for row in anchor_rows}
    for anchor in anchors:
        row = rows_by_id[str(anchor["anchor_id"])]
        point_a = np.asarray(anchor["contact_point_a_px"], dtype=float)
        point_b = np.asarray(anchor["contact_point_b_px"], dtype=float)
        color = "#00a65a" if row["predicted_usable"] else "#d62728"
        axes[1, 1].plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color=color, linewidth=4)
        midpoint = (point_a + point_b) * 0.5
        axes[1, 1].text(midpoint[0] + 5, midpoint[1], anchor["anchor_id"], color=color, fontsize=11, fontweight="bold")
    axes[1, 1].set_title("5. Pair decision\ngreen = both sides pass; red = vetoed")

    axes[1, 2].imshow(rgb)
    selected_anchor = next((anchor for anchor in anchors if anchor["anchor_id"] == selected_anchor_id), None)
    if selected_anchor is not None:
        point_a = np.asarray(selected_anchor["contact_point_a_px"], dtype=float)
        point_b = np.asarray(selected_anchor["contact_point_b_px"], dtype=float)
        axes[1, 2].plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color="#00a65a", linewidth=6)
        axes[1, 2].scatter([point_a[0], point_b[0]], [point_a[1], point_b[1]], color="yellow", edgecolor="black", s=70)
    decision_lines = [
        f"{row['anchor_id']}: {'GOOD' if row['predicted_usable'] else 'BAD'}"
        + (f" ({row['veto_reasons']})" if row["veto_reasons"] else "")
        for row in anchor_rows
    ]
    axes[1, 2].text(
        0.02,
        0.98,
        f"6. Selected: {selected_anchor_id}\n" + "\n".join(decision_lines),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#444444"},
    )
    axes[1, 2].set_title("6. Conservative final selection")

    for axis in axes.reshape(-1):
        axis.axis("off")
    fig.suptitle(f"Projected top-view RGB-D deterministic contact pipeline | {group_id}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_report(
    output_dir: Path,
    args: argparse.Namespace,
    anchor_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    overall = next(row for row in metrics_rows if row["group"] == "all")
    veto_counts = Counter(reason for row in anchor_rows for reason in str(row["veto_reasons"]).split(";") if reason)
    lines = [
        "# Projected Top-View RGB-D Deterministic Contact Baseline",
        "",
        "## Method",
        "",
        "This is an offline, training-free contact-selection baseline over saved RGB-D data. The upstream object masks were originally produced by the main perception pipeline; no model was run here. A robust top plane is fitted only to establish a metric projection coordinate system. All valid points in each candidate ROI are projected into that plane, while signed height is retained as an auxiliary channel. No vertical side plane is fitted.",
        "",
        f"- Base visually reviewed defect threshold: `{args.defect_threshold_mm:.2f} mm`",
        f"- Noise-adaptive threshold: `max(base threshold, {args.noise_multiplier:.1f} x top-plane MAD sigma)`",
        f"- Maximum contact-review width: `{2 * args.contact_half_width_mm:.1f} mm`; each candidate is clipped to its nearest-anchor lateral partition, matching the saved review crops.",
        f"- Projection grid: `{args.grid_mm:.2f} mm/cell`",
        f"- Minimum coherent component: `{args.min_component_area_mm2:.1f} mm^2` and `{args.min_component_span_mm:.1f} mm` along the edge",
        "- Binary safety policy: insufficient evidence is classified as bad.",
        "- Pair policy: both opposing sides must pass; selection maximizes the worst-side safety score.",
        "",
        "## Results",
        "",
        f"- Groups processed: `{len(group_rows)}`",
        f"- Anchor pairs processed: `{len(anchor_rows)}`",
        f"- Processing failures: `{len(failures)}`",
        f"- Top-ranked manually usable selections: `{sum(bool(row['selected_manual_is_usable']) for row in group_rows)}/{len(group_rows)}`",
        f"- NO_SAFE_ANCHOR groups: `{sum(row['selected_anchor_id'] == 'NO_SAFE_ANCHOR' for row in group_rows)}`",
        "",
        "| Split/group | n | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['TP']} | {row['FP']} | {row['TN']} | {row['FN']} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['specificity'])} | {_fmt(row['f1'])} | {_fmt(row['balanced_accuracy'])} |"
        )
    lines.extend(["", "## Most Common Veto Reasons", ""])
    for reason, count in veto_counts.most_common():
        lines.append(f"- `{reason}`: {count} anchors")
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "The method evaluates projected top-view evidence at the visible rim: outward occupancy, inward loss of support, boundary continuity, coherent raised geometry, and depth reliability. A defect completely occluded beneath a vertical side rim cannot be recovered from a single top view. E3 labels are formal manual labels; E45 labels are visual handoff labels and are therefore reported separately.",
            "",
            "The selected numerical threshold is a visually reviewed engineering setting, not a learned classifier parameter. If final labels are consulted to optimize it, the resulting benchmark must be described as development-set performance.",
            "",
            "## Artifacts",
            "",
            "- `per_anchor_results.csv`: per-candidate measurements and binary predictions.",
            "- `per_group_results.csv`: selected pair and top-1 usability.",
            "- `metrics.csv`: anchor-level metrics overall and by split.",
            "- `visuals/per_anchor/`: projected occupancy and boundary diagnostics.",
            "- `visuals/per_group/`: all side diagnostics grouped by case.",
            "- `visuals/pipeline_overview/`: six-stage input-to-selection overview for every case.",
            "- `threshold_visual_audit/`: representative threshold comparison outputs.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-csv", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append", help="Optional benchmark group ID; repeat to process several groups.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all required paths and candidate IDs only.")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--no-side-visuals", action="store_true", help="Create pipeline overviews but skip per-side diagnostic figures.")
    parser.add_argument("--defect-threshold-mm", type=float, default=3.0)
    parser.add_argument("--noise-multiplier", type=float, default=4.0)
    parser.add_argument("--contact-half-width-mm", type=float, default=25.0)
    parser.add_argument("--inside-margin-mm", type=float, default=24.0)
    parser.add_argument("--outside-margin-mm", type=float, default=20.0)
    parser.add_argument("--grid-mm", type=float, default=0.5)
    parser.add_argument("--min-component-area-mm2", type=float, default=4.0)
    parser.add_argument("--min-component-span-mm", type=float, default=2.0)
    parser.add_argument("--min-depth-valid-fraction", type=float, default=0.08)
    parser.add_argument("--min-boundary-support-fraction", type=float, default=0.70)
    parser.add_argument("--max-missing-boundary-run-mm", type=float, default=4.0)
    parser.add_argument("--min-boundary-line-inlier-fraction", type=float, default=0.45)
    parser.add_argument("--boundary-fit-half-width-mm", type=float, default=35.0)
    parser.add_argument("--boundary-fit-search-mm", type=float, default=16.0)
    parser.add_argument("--boundary-ransac-threshold-mm", type=float, default=1.5)
    parser.add_argument("--boundary-ransac-iterations", type=int, default=160)
    parser.add_argument("--rgb-edge-band-mm", type=float, default=1.5)
    parser.add_argument("--top-plane-erode-px", type=int, default=10)
    parser.add_argument("--top-plane-ransac-threshold-mm", type=float, default=1.5)
    parser.add_argument("--top-plane-ransac-iterations", type=int, default=240)
    parser.add_argument("--max-object-below-top-mm", type=float, default=45.0)
    parser.add_argument("--background-separation-fraction", type=float, default=0.25)
    parser.add_argument("--raised-edge-band-mm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    lineage_path = _resolve(args.lineage_csv)
    labels_path = _resolve(args.labels_csv)
    lineage_rows = _read_csv(lineage_path)
    label_rows = _read_csv(labels_path)
    if args.case_id:
        requested = set(args.case_id)
        lineage_rows = [row for row in lineage_rows if row["benchmark_group_id"] in requested]
        missing = requested - {row["benchmark_group_id"] for row in lineage_rows}
        if missing:
            raise SystemExit(f"Unknown --case-id values: {', '.join(sorted(missing))}")
    labels_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        group = f"{row['dataset_split']}_{row['case_or_session_id']}"
        labels_by_group[group].append(row)

    audit_rows: list[dict[str, Any]] = []
    for lineage in lineage_rows:
        group = lineage["benchmark_group_id"]
        required = {
            "rgb": _resolve(Path(lineage["original_raw_rgb_path"])),
            "depth": _resolve(Path(lineage["original_raw_depth_path"])),
            "camera_info": _resolve(Path(lineage["original_camera_info_path"])),
            "mask": _resolve(Path(lineage["original_selected_mask_path"])),
            "features": _resolve(Path(lineage["original_anchor_selection_folder"])) / "deterministic_anchor_features_wide_context.json",
        }
        feature_count = 0
        if required["features"].exists():
            feature_count = len(_load_json(required["features"]).get("anchor_features", []))
        audit_rows.append(
            {
                "benchmark_group_id": group,
                "dataset_split": lineage["dataset_split"],
                **{f"{key}_exists": path.exists() for key, path in required.items()},
                "candidate_count": feature_count,
                "label_count": len(labels_by_group.get(group, [])),
                "candidate_label_count_match": feature_count == len(labels_by_group.get(group, [])),
            }
        )
    failed_audit = [row for row in audit_rows if not all(row[key] for key in row if key.endswith("_exists")) or not row["candidate_label_count_match"]]
    print(f"Groups discovered: {len(lineage_rows)}", flush=True)
    print(f"Anchors expected: {sum(int(row['candidate_count']) for row in audit_rows)}", flush=True)
    if failed_audit:
        raise SystemExit(f"Input audit failed for {len(failed_audit)} group(s): {failed_audit}")
    if args.dry_run:
        for row in audit_rows:
            print(f"{row['benchmark_group_id']}: candidates={row['candidate_count']} labels={row['label_count']} inputs=OK")
        print("Dry run complete. No scoring or model/hardware operation was performed.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve(args.output_dir) if args.output_dir else _resolve(DEFAULT_OUTPUT_ROOT) / f"session_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "input_audit.csv", audit_rows)
    _write_json(
        output_dir / "run_config.json",
        {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "lineage_csv": str(lineage_path),
            "labels_csv": str(labels_path),
            "method": "projected_topview_rgbd_expected_edge_connected_components",
            "top_plane_role": "metric_projection_coordinate_frame_only",
            "side_plane_fitting": False,
            "model_inference_run": False,
            "robot_or_hardware_run": False,
        },
    )

    anchor_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for group_index, lineage in enumerate(lineage_rows):
        group_start = time.perf_counter()
        group_id = lineage["benchmark_group_id"]
        print(f"[{group_index + 1}/{len(lineage_rows)}] {group_id}", flush=True)
        try:
            rgb_bgr = cv2.imread(str(_resolve(Path(lineage["original_raw_rgb_path"]))), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(lineage["original_raw_rgb_path"])
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth_path = _resolve(Path(lineage["original_raw_depth_path"]))
            depth_raw = np.load(depth_path) if depth_path.suffix.lower() == ".npy" else cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_mm = _depth_to_mm(depth_raw)
            mask_image = cv2.imread(str(_resolve(Path(lineage["original_selected_mask_path"]))), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise FileNotFoundError(lineage["original_selected_mask_path"])
            mask = mask_image > 0
            camera = _load_json(_resolve(Path(lineage["original_camera_info_path"])))
            intr = _intrinsics(camera)
            xyz = _backproject(depth_mm, intr)
            plane = _fit_top_plane(
                xyz,
                depth_mm,
                mask,
                erode_px=args.top_plane_erode_px,
                ransac_threshold_mm=args.top_plane_ransac_threshold_mm,
                ransac_iterations=args.top_plane_ransac_iterations,
                seed=args.seed + group_index,
            )
            signed = np.einsum("...i,i->...", xyz, plane["normal"]) + float(plane["offset"])
            plane_xyz = xyz - signed[..., None] * plane["normal"]
            rays = _pixel_rays(depth_mm.shape, intr)
            plane_intersections = _intersect_rays_with_plane(rays, plane["normal"], float(plane["offset"]))
            canny = cv2.Canny(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), 60, 160)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_xy = max(contours, key=cv2.contourArea)[:, 0, :]
            contour_plane_points = np.asarray(
                [_pixel_plane_point(point, intr, plane["normal"], float(plane["offset"])) for point in contour_xy],
                dtype=np.float64,
            )
            feature_path = _resolve(Path(lineage["original_anchor_selection_folder"])) / "deterministic_anchor_features_wide_context.json"
            anchors = _load_json(feature_path)["anchor_features"]
            anchor_plane_points = [
                _pixel_plane_point(anchor["anchor_px"], intr, plane["normal"], float(plane["offset"]))
                for anchor in anchors
            ]
            lateral_half_widths: list[float] = []
            for anchor_index, anchor_plane_point in enumerate(anchor_plane_points):
                neighbor_distances = [
                    float(np.linalg.norm(anchor_plane_point - other_point))
                    for other_index, other_point in enumerate(anchor_plane_points)
                    if other_index != anchor_index
                ]
                partition_half_width = 0.5 * min(neighbor_distances) if neighbor_distances else args.contact_half_width_mm
                lateral_half_widths.append(min(float(args.contact_half_width_mm), partition_half_width))
            manual_by_anchor = {row["anchor_id"].upper(): row for row in labels_by_group[group_id]}
            group_visuals: list[tuple[str, Path]] = []
            group_anchor_rows: list[dict[str, Any]] = []
            for anchor_index, anchor in enumerate(anchors):
                sides: list[SideResult] = []
                for side_index, (side_name, key) in enumerate((("contact_a", "contact_point_a_px"), ("contact_b", "contact_point_b_px"))):
                    side, debug = _analyse_side(
                        contact_point=anchor[key],
                        anchor_point=anchor["anchor_px"],
                        side_name=side_name,
                        plane=plane,
                        xyz=xyz,
                        plane_xyz=plane_xyz,
                        plane_intersections=plane_intersections,
                        depth_mm=depth_mm,
                        mask=mask,
                        canny=canny,
                        intr=intr,
                        contour_plane_points=contour_plane_points,
                        lateral_half_width_mm=lateral_half_widths[anchor_index],
                        args=args,
                        seed=args.seed + group_index * 100 + anchor_index * 10 + side_index,
                    )
                    sides.append(side)
                    if not args.no_visuals and not args.no_side_visuals:
                        visual_path = output_dir / "visuals" / "per_anchor" / group_id / f"{anchor['anchor_id']}_{side_name}.png"
                        _save_side_visual(visual_path, rgb, mask, anchor, side, debug, group_id)
                        group_visuals.append((f"{anchor['anchor_id']} {side_name}", visual_path))
                pair_good = all(side.decision_good for side in sides)
                pair_score = min(side.safety_score for side in sides)
                manual = manual_by_anchor.get(str(anchor["anchor_id"]).upper())
                if manual is None:
                    raise ValueError(f"No manual label for {group_id} {anchor['anchor_id']}")
                row: dict[str, Any] = {
                    "benchmark_group_id": group_id,
                    "dataset_split": lineage["dataset_split"],
                    "material": manual.get("material") or lineage.get("material_or_object") or "",
                    "condition_type": manual.get("condition_type") or lineage.get("condition_type") or "",
                    "anchor_id": anchor["anchor_id"],
                    "manual_is_usable": bool(_safe_bool(manual["is_usable"])),
                    "manual_label": manual["manual_label"],
                    "predicted_usable": pair_good,
                    "pair_safety_score": pair_score,
                    "veto_reasons": ";".join(sorted(set(reason for side in sides for reason in side.veto_reasons))),
                    "top_plane_noise_sigma_mm": float(plane["noise_sigma_mm"]),
                    "top_plane_inlier_fraction": float(plane["inlier_fraction"]),
                }
                for side in sides:
                    for key, value in asdict(side).items():
                        if key in {"side_name", "contact_point_xy", "veto_reasons"}:
                            continue
                        row[f"{side.side_name}_{key}"] = value
                    row[f"{side.side_name}_veto_reasons"] = ";".join(side.veto_reasons)
                    row[f"{side.side_name}_contact_point_xy"] = json.dumps(side.contact_point_xy)
                anchor_rows.append(row)
                group_anchor_rows.append(row)
            eligible = [row for row in group_anchor_rows if row["predicted_usable"]]
            selected = max(eligible, key=lambda row: (float(row["pair_safety_score"]), -abs(int(row["anchor_id"][1:]) - (len(anchors) + 1) / 2))) if eligible else None
            group_row = {
                    "benchmark_group_id": group_id,
                    "dataset_split": lineage["dataset_split"],
                    "material": lineage.get("material_or_object") or "",
                    "candidate_count": len(anchors),
                    "good_candidate_count": len(eligible),
                    "selected_anchor_id": selected["anchor_id"] if selected else "NO_SAFE_ANCHOR",
                    "selected_pair_safety_score": selected["pair_safety_score"] if selected else "",
                    "selected_manual_is_usable": bool(selected["manual_is_usable"]) if selected else False,
                    "manual_good_anchor_ids": ",".join(row["anchor_id"] for row in group_anchor_rows if row["manual_is_usable"]),
                    "processing_time_ms": (time.perf_counter() - group_start) * 1000.0,
                }
            group_rows.append(group_row)
            if not args.no_visuals and group_visuals:
                _save_group_grid(output_dir / "visuals" / "per_group" / f"{group_id}.png", group_visuals)
            if not args.no_visuals:
                _save_pipeline_overview(
                    output_dir / "visuals" / "pipeline_overview" / f"{group_id}.png",
                    group_id=group_id,
                    rgb=rgb,
                    depth_mm=depth_mm,
                    mask=mask,
                    plane=plane,
                    anchors=anchors,
                    anchor_rows=group_anchor_rows,
                    selected_anchor_id=group_row["selected_anchor_id"],
                )
        except Exception as exc:
            failures.append({"benchmark_group_id": group_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    if anchor_rows:
        _write_csv(output_dir / "per_anchor_results.csv", anchor_rows)
        _write_json(output_dir / "per_anchor_results.json", anchor_rows)
    if group_rows:
        _write_csv(output_dir / "per_group_results.csv", group_rows)
        _write_json(output_dir / "per_group_results.json", group_rows)
    metrics_rows: list[dict[str, Any]] = []
    for group_name, selected_rows in [
        ("all", anchor_rows),
        ("E3", [row for row in anchor_rows if row["dataset_split"] == "E3"]),
        ("E45", [row for row in anchor_rows if row["dataset_split"] == "E45"]),
    ]:
        metrics_rows.append({"group": group_name, **_confusion(selected_rows)})
    _write_csv(output_dir / "metrics.csv", metrics_rows)
    _write_json(output_dir / "metrics.json", metrics_rows)
    _write_csv(output_dir / "failures.csv", failures, ["benchmark_group_id", "error"])
    _build_report(output_dir, args, anchor_rows, group_rows, metrics_rows, failures)
    print(f"wrote {output_dir}")
    print(f"groups={len(group_rows)} anchors={len(anchor_rows)} failures={len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
