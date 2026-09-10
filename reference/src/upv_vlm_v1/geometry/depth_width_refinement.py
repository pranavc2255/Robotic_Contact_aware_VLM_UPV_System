from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


NA = "NA"


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mask_erode_kernel_px": 5,
    "central_fraction": 0.50,
    "depth_band_threshold_m": 0.012,
    "mad_multiplier": 2.5,
    "min_depth_m": 0.05,
    "max_depth_m": 2.0,
    "min_top_face_points": 500,
    "min_depth_coverage_ratio": 0.20,
    "max_depth_mad_m": 0.015,
    "max_width_disagreement_ratio": 0.25,
    "erode_only_for_depth_reference": True,
    "use_original_mask_for_final_span": True,
    "percentile_low": 1.0,
    "percentile_high": 99.0,
    "diagnostic_percentiles": [[0, 100], [0.5, 99.5], [1, 99], [2, 98], [2.5, 97.5], [5, 95]],
    "edge_bin_count": 80,
    "edge_bin_min_points": 5,
    "overlay_max_points": 2000,
}


@dataclass
class DepthWidthRefinementResult:
    success: bool
    valid: bool
    failure_reason: str | None
    method: str
    mask_projected_width_mm: float | str
    depth_refined_width_mm: float | str
    recommended_upv_path_length_mm: float | str
    upv_path_length_source: str
    disagreement_ratio: float | str
    clamp_spacing_axis: str
    selected_upv_axis: str
    top_face_points_count: int
    valid_depth_points_count: int
    original_mask_points_count: int
    depth_coverage_ratio: float | str
    central_depth_median_m: float | str
    top_face_depth_median_m: float | str
    top_face_depth_mad_m: float | str
    depth_band_threshold_m: float
    percentile_low: float
    percentile_high: float
    endpoints_px: dict[str, Any]
    endpoints_camera_xyz_m: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


def _failure(reason: str, selected_upv_axis: str, mask_projected_width_mm: float | str, config: dict[str, Any]) -> DepthWidthRefinementResult:
    return DepthWidthRefinementResult(
        success=False,
        valid=False,
        failure_reason=reason,
        method="masked_top_face_depth_band_projected_percentile_span",
        mask_projected_width_mm=mask_projected_width_mm,
        depth_refined_width_mm=NA,
        recommended_upv_path_length_mm=NA,
        upv_path_length_source="NA_depth_refinement_invalid",
        disagreement_ratio=NA,
        clamp_spacing_axis=_clamp_axis_from_upv_axis_safe(selected_upv_axis),
        selected_upv_axis=selected_upv_axis,
        top_face_points_count=0,
        valid_depth_points_count=0,
        original_mask_points_count=0,
        depth_coverage_ratio=NA,
        central_depth_median_m=NA,
        top_face_depth_median_m=NA,
        top_face_depth_mad_m=NA,
        depth_band_threshold_m=float(config.get("depth_band_threshold_m", DEFAULT_CONFIG["depth_band_threshold_m"])),
        percentile_low=float(config.get("percentile_low", DEFAULT_CONFIG["percentile_low"])),
        percentile_high=float(config.get("percentile_high", DEFAULT_CONFIG["percentile_high"])),
        endpoints_px={},
        endpoints_camera_xyz_m={},
        diagnostics={},
    )


def _camera_intrinsics(camera_info: Any) -> tuple[float, float, float, float]:
    return _camera_model(camera_info, None)["scaled_intrinsics_tuple"]


def _camera_model(camera_info: Any, image_shape: tuple[int, int] | None) -> dict[str, Any]:
    width = None
    height = None
    if isinstance(camera_info, dict):
        if all(key in camera_info for key in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = float(camera_info["fx"]), float(camera_info["fy"]), float(camera_info["cx"]), float(camera_info["cy"])
        else:
            matrix = camera_info.get("K") or camera_info.get("k") or camera_info.get("camera_matrix")
            if isinstance(matrix, dict):
                matrix = matrix.get("data")
            if matrix is None:
                raise ValueError("camera_info must provide fx/fy/cx/cy or K[0,4,2,5]")
            values = list(matrix)
            if len(values) < 9:
                raise ValueError("camera_info K matrix must have 9 values")
            fx, fy, cx, cy = float(values[0]), float(values[4]), float(values[2]), float(values[5])
        width = camera_info.get("width")
        height = camera_info.get("height")
    else:
        matrix = getattr(camera_info, "k", None) or getattr(camera_info, "K", None)
        if matrix is None or len(matrix) < 9:
            raise ValueError("camera_info must provide fx/fy/cx/cy or K[0,4,2,5]")
        fx, fy, cx, cy = float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5])
        width = getattr(camera_info, "width", None)
        height = getattr(camera_info, "height", None)

    original = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
    scale_x = 1.0
    scale_y = 1.0
    intrinsics_scaled = False
    if image_shape is not None and width and height:
        image_h, image_w = image_shape
        scale_x = float(image_w) / float(width)
        scale_y = float(image_h) / float(height)
        if abs(scale_x - 1.0) > 1e-9 or abs(scale_y - 1.0) > 1e-9:
            fx *= scale_x
            cx *= scale_x
            fy *= scale_y
            cy *= scale_y
            intrinsics_scaled = True
    scaled = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": image_shape[1] if image_shape else width, "height": image_shape[0] if image_shape else height}
    return {
        "original_intrinsics": original,
        "scaled_intrinsics": scaled,
        "scaled_intrinsics_tuple": (fx, fy, cx, cy),
        "camera_info_width": width,
        "camera_info_height": height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "intrinsics_scaled": intrinsics_scaled,
    }


def _normalize_axis(axis: tuple[float, float] | list[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(axis, dtype=float).reshape(2)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} axis is degenerate")
    return arr / norm


def _clamp_axis_from_upv_axis(selected_upv_axis: str) -> str:
    if selected_upv_axis == "major":
        return "minor"
    if selected_upv_axis == "minor":
        return "major"
    raise ValueError(f"Unsupported selected_upv_axis: {selected_upv_axis}")


def _clamp_axis_from_upv_axis_safe(selected_upv_axis: str) -> str:
    try:
        return _clamp_axis_from_upv_axis(selected_upv_axis)
    except Exception:  # noqa: BLE001
        return NA


def _depth_to_meters(depth_image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if "depth_unit_scale" in config:
        scale = float(config["depth_unit_scale"])
    elif np.issubdtype(depth_image.dtype, np.integer):
        scale = 0.001
    else:
        scale = 1.0
    return depth_image.astype(np.float32) * scale


def _mad(values: np.ndarray, center: float | None = None) -> float:
    if values.size == 0:
        return float("nan")
    c = float(np.median(values)) if center is None else float(center)
    return float(np.median(np.abs(values - c)))


def _backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> list[float]:
    return [
        (float(u) - cx) * float(z) / fx,
        (float(v) - cy) * float(z) / fy,
        float(z),
    ]


def _axis_scale_m_per_px(axis: np.ndarray, depth_m: float, fx: float, fy: float) -> float:
    return float(depth_m) * math.sqrt((float(axis[0]) / fx) ** 2 + (float(axis[1]) / fy) ** 2)


def _width_mm_from_span_px(span_px: float, axis: np.ndarray, depth_m: float, fx: float, fy: float) -> float:
    return float(span_px) * _axis_scale_m_per_px(axis, depth_m, fx, fy) * 1000.0


def _centerline_endpoints_from_projection(centroid: np.ndarray, axis: np.ndarray, s_low: float, s_high: float) -> tuple[np.ndarray, np.ndarray]:
    c_proj = float(centroid @ axis)
    low = centroid + axis * (float(s_low) - c_proj)
    high = centroid + axis * (float(s_high) - c_proj)
    return low, high


def _endpoint_backproject_width_mm(
    centroid: np.ndarray,
    axis: np.ndarray,
    s_low: float,
    s_high: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    low_px, high_px = _centerline_endpoints_from_projection(centroid, axis, s_low, s_high)
    low_xyz = _backproject(low_px[0], low_px[1], depth_m, fx, fy, cx, cy)
    high_xyz = _backproject(high_px[0], high_px[1], depth_m, fx, fy, cx, cy)
    width_mm = float(np.linalg.norm(np.asarray(high_xyz) - np.asarray(low_xyz)) * 1000.0)
    endpoints_px = {"low": low_px.tolist(), "high": high_px.tolist(), "percentile_low_projection_px": float(s_low), "percentile_high_projection_px": float(s_high)}
    endpoints_xyz = {"low": low_xyz, "high": high_xyz}
    return width_mm, endpoints_px, endpoints_xyz


def _percentile_widths(
    projections: np.ndarray,
    axis: np.ndarray,
    depth_m: float,
    fx: float,
    fy: float,
    diagnostic_percentiles: list[list[float]],
) -> dict[str, Any]:
    widths: dict[str, Any] = {}
    for low, high in diagnostic_percentiles:
        s_low = float(np.percentile(projections, float(low)))
        s_high = float(np.percentile(projections, float(high)))
        key_low = str(low).replace(".", "p")
        key_high = str(high).replace(".", "p")
        widths[f"depth_width_percentile_{key_low}_{key_high}_mm"] = _width_mm_from_span_px(s_high - s_low, axis, depth_m, fx, fy)
        widths[f"projection_percentile_{key_low}_{key_high}_px"] = {"low": s_low, "high": s_high, "span": s_high - s_low}
    return widths


def _edge_bin_width(
    projections: np.ndarray,
    axis: np.ndarray,
    depth_m: float,
    fx: float,
    fy: float,
    bin_count: int,
    min_points: int,
) -> tuple[float | str, dict[str, Any]]:
    if projections.size < 2:
        return NA, {"failure_reason": "too_few_projection_points"}
    s_min = float(np.min(projections))
    s_max = float(np.max(projections))
    if s_max <= s_min:
        return NA, {"failure_reason": "degenerate_projection_range"}
    counts, edges = np.histogram(projections, bins=max(2, int(bin_count)), range=(s_min, s_max))
    occupied = np.where(counts >= int(min_points))[0]
    if occupied.size < 2:
        return NA, {"failure_reason": "not_enough_occupied_edge_bins", "max_bin_count": int(counts.max()) if counts.size else 0}
    low_idx = int(occupied[0])
    high_idx = int(occupied[-1])
    low_center = float((edges[low_idx] + edges[low_idx + 1]) / 2.0)
    high_center = float((edges[high_idx] + edges[high_idx + 1]) / 2.0)
    return _width_mm_from_span_px(high_center - low_center, axis, depth_m, fx, fy), {
        "low_bin_index": low_idx,
        "high_bin_index": high_idx,
        "low_bin_center_px": low_center,
        "high_bin_center_px": high_center,
        "edge_bin_count": int(bin_count),
        "edge_bin_min_points": int(min_points),
    }


def _point_near_projection(coords: np.ndarray, projections: np.ndarray, target: float) -> np.ndarray:
    return coords[int(np.argmin(np.abs(projections - target)))]


def _sample_points(coords: np.ndarray, max_points: int) -> list[list[int]]:
    if coords.size == 0:
        return []
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(int)
        coords = coords[idx]
    return [[int(round(float(x))), int(round(float(y)))] for x, y in coords]


def estimate_depth_refined_width(
    mask: np.ndarray,
    depth_image: np.ndarray,
    camera_info: dict[str, Any] | Any,
    major_axis_unit_px: tuple[float, float] | list[float] | np.ndarray,
    minor_axis_unit_px: tuple[float, float] | list[float] | np.ndarray,
    selected_upv_axis: str,
    mask_projected_width_mm: float,
    config: dict[str, Any] | None = None,
) -> DepthWidthRefinementResult:
    cfg = _merged_config(config)
    method = "masked_top_face_depth_band_projected_percentile_span"
    if not bool(cfg.get("enabled", True)):
        return _failure("depth_width_refinement_disabled", selected_upv_axis, mask_projected_width_mm, cfg)
    try:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.ndim != 2:
            return _failure("mask_must_be_2d", selected_upv_axis, mask_projected_width_mm, cfg)
        depth_m = _depth_to_meters(np.asarray(depth_image), cfg)
        if depth_m.shape[:2] != mask_bool.shape[:2]:
            return _failure("depth_shape_does_not_match_mask", selected_upv_axis, mask_projected_width_mm, cfg)
        image_shape = mask_bool.shape[:2]
        camera_model = _camera_model(camera_info, image_shape)
        fx, fy, cx, cy = camera_model["scaled_intrinsics_tuple"]
        major = _normalize_axis(major_axis_unit_px, "major")
        minor = _normalize_axis(minor_axis_unit_px, "minor")
        clamp_spacing_axis = _clamp_axis_from_upv_axis(selected_upv_axis)
        axis = minor if clamp_spacing_axis == "minor" else major
        centroid = np.asarray(cfg.get("centroid_px", []), dtype=float)
        if centroid.size != 2:
            ys_all, xs_all = np.nonzero(mask_bool)
            centroid = np.array([float(np.mean(xs_all)), float(np.mean(ys_all))]) if len(xs_all) else np.array([0.0, 0.0])
        original_mask_points_count = int(mask_bool.sum())
        if original_mask_points_count == 0:
            return _failure("empty_mask", selected_upv_axis, mask_projected_width_mm, cfg)

        kernel_px = int(cfg["mask_erode_kernel_px"])
        eroded = mask_bool
        if kernel_px > 1:
            kernel = np.ones((kernel_px, kernel_px), dtype=np.uint8)
            eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
            if not eroded.any():
                eroded = mask_bool

        min_depth = float(cfg["min_depth_m"])
        max_depth = float(cfg["max_depth_m"])
        finite_depth = np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
        ref_valid = eroded & finite_depth
        ref_ys, ref_xs = np.nonzero(ref_valid)
        valid_depth_points_count = int(len(ref_xs))
        depth_coverage_ratio = valid_depth_points_count / max(1, original_mask_points_count)
        if valid_depth_points_count == 0:
            result = _failure("no_valid_depth_points_inside_eroded_mask", selected_upv_axis, mask_projected_width_mm, cfg)
            result.original_mask_points_count = original_mask_points_count
            result.diagnostics = {
                "mask_shape": list(mask_bool.shape),
                "depth_shape": list(depth_m.shape),
                **camera_model,
            }
            return result

        ref_coords = np.column_stack([ref_xs.astype(float), ref_ys.astype(float)])
        ref_depths = depth_m[ref_ys, ref_xs].astype(float)
        ref_proj_major = ref_coords @ major
        ref_proj_minor = ref_coords @ minor
        central_fraction = float(cfg["central_fraction"])
        central_fraction = min(max(central_fraction, 0.05), 1.0)
        low_pct = 50.0 - central_fraction * 50.0
        high_pct = 50.0 + central_fraction * 50.0
        major_low, major_high = np.percentile(ref_proj_major, [low_pct, high_pct])
        minor_low, minor_high = np.percentile(ref_proj_minor, [low_pct, high_pct])
        central_mask = (ref_proj_major >= major_low) & (ref_proj_major <= major_high) & (ref_proj_minor >= minor_low) & (ref_proj_minor <= minor_high)
        if not np.any(central_mask):
            central_mask = np.ones_like(ref_depths, dtype=bool)
        central_depth_median = float(np.median(ref_depths[central_mask]))
        central_mad = _mad(ref_depths[central_mask], central_depth_median)
        threshold = max(float(cfg["depth_band_threshold_m"]), float(cfg["mad_multiplier"]) * central_mad)

        # Erosion is only for top-face depth reference. Final span uses original-mask
        # pixels that pass the top-face depth band, so the measured width is not an
        # eroded/interior width.
        final_domain = mask_bool if bool(cfg.get("use_original_mask_for_final_span", True)) else eroded
        final_valid = final_domain & finite_depth
        final_ys, final_xs = np.nonzero(final_valid)
        final_coords_all = np.column_stack([final_xs.astype(float), final_ys.astype(float)])
        final_depths_all = depth_m[final_ys, final_xs].astype(float)
        final_top_mask = np.abs(final_depths_all - central_depth_median) <= threshold if final_depths_all.size else np.array([], dtype=bool)
        top_coords = final_coords_all[final_top_mask]
        top_depths = final_depths_all[final_top_mask]
        rejected_coords = final_coords_all[~final_top_mask]
        top_count = int(len(top_depths))
        top_depth_median = float(np.median(top_depths)) if top_count else float("nan")
        top_depth_mad = _mad(top_depths, top_depth_median) if top_count else float("nan")
        accepted_point_coverage_ratio = top_count / max(1, original_mask_points_count)

        orig_ys, orig_xs = np.nonzero(mask_bool)
        orig_coords = np.column_stack([orig_xs.astype(float), orig_ys.astype(float)])
        er_ys, er_xs = np.nonzero(eroded)
        eroded_coords = np.column_stack([er_xs.astype(float), er_ys.astype(float)]) if len(er_xs) else orig_coords
        orig_s = orig_coords @ axis
        eroded_s = eroded_coords @ axis
        width_before_erosion_px = float(np.max(orig_s) - np.min(orig_s)) if orig_s.size else 0.0
        width_after_erosion_px = float(np.max(eroded_s) - np.min(eroded_s)) if eroded_s.size else 0.0
        width_after_depth_filter_px = 0.0
        if top_count:
            top_s = top_coords @ axis
            width_after_depth_filter_px = float(np.max(top_s) - np.min(top_s))
        else:
            top_s = np.array([], dtype=float)
        erosion_width_loss_px = max(0.0, width_before_erosion_px - width_after_erosion_px)

        failure_reasons: list[str] = []
        if top_count < int(cfg["min_top_face_points"]):
            failure_reasons.append("too_few_top_face_points")
        if depth_coverage_ratio < float(cfg["min_depth_coverage_ratio"]):
            failure_reasons.append("depth_coverage_below_minimum")
        if not np.isfinite(top_depth_mad) or top_depth_mad > float(cfg["max_depth_mad_m"]):
            failure_reasons.append("top_face_depth_mad_too_large")

        width_mm: float | str = NA
        disagreement: float | str = NA
        endpoints_px: dict[str, Any] = {}
        endpoints_xyz: dict[str, Any] = {}
        candidate_widths: dict[str, Any] = {}
        selected_method = "NA"
        diagnostic_3d_width_mm: float | str = NA
        edge_bin_width_mm: float | str = NA
        edge_bin_diag: dict[str, Any] = {}
        percentile_width_loss_px: float | str = NA
        corrected_formula_width_mm: float | str = NA
        old_formula_width_mm: float | str = NA

        if top_count >= 2 and np.isfinite(top_depth_median):
            diagnostic_percentiles = cfg.get("diagnostic_percentiles") or DEFAULT_CONFIG["diagnostic_percentiles"]
            candidate_widths = _percentile_widths(top_s, axis, top_depth_median, fx, fy, diagnostic_percentiles)
            p0 = float(np.percentile(top_s, 0.0))
            p100 = float(np.percentile(top_s, 100.0))
            depth_boundary_width_mm = _width_mm_from_span_px(p100 - p0, axis, top_depth_median, fx, fy)
            candidate_widths["depth_width_from_depth_boundary_mm"] = depth_boundary_width_mm
            edge_bin_width_mm, edge_bin_diag = _edge_bin_width(
                top_s,
                axis,
                top_depth_median,
                fx,
                fy,
                int(cfg.get("edge_bin_count", 80)),
                int(cfg.get("edge_bin_min_points", 5)),
            )
            candidate_widths["depth_width_from_edge_bins_mm"] = edge_bin_width_mm
            low_pct_final = float(cfg["percentile_low"])
            high_pct_final = float(cfg["percentile_high"])
            s_low = float(np.percentile(top_s, low_pct_final))
            s_high = float(np.percentile(top_s, high_pct_final))
            span_px = max(0.0, s_high - s_low)
            corrected_formula_width_mm = _width_mm_from_span_px(span_px, axis, top_depth_median, fx, fy)
            old_f_eff = math.sqrt((float(axis[0]) * fx) ** 2 + (float(axis[1]) * fy) ** 2)
            old_formula_width_mm = float(span_px * top_depth_median / old_f_eff * 1000.0)
            endpoint_width_mm, endpoints_px, endpoints_xyz = _endpoint_backproject_width_mm(centroid, axis, s_low, s_high, top_depth_median, fx, fy, cx, cy)
            candidate_widths["depth_width_old_effective_focal_formula_mm"] = old_formula_width_mm
            candidate_widths["depth_width_corrected_axis_scale_mm"] = corrected_formula_width_mm
            candidate_widths["depth_width_from_backprojected_endpoint_pixels_mm"] = endpoint_width_mm
            candidate_widths["mask_projected_width_mm_existing"] = float(mask_projected_width_mm)
            if edge_bin_width_mm != NA:
                width_mm = edge_bin_width_mm
                selected_method = "edge_bin_recovered_original_mask_top_depth_band"
                edge_low = edge_bin_diag.get("low_bin_center_px")
                edge_high = edge_bin_diag.get("high_bin_center_px")
                if edge_low is not None and edge_high is not None:
                    _edge_width, endpoints_px, endpoints_xyz = _endpoint_backproject_width_mm(centroid, axis, float(edge_low), float(edge_high), top_depth_median, fx, fy, cx, cy)
            else:
                width_mm = endpoint_width_mm
                selected_method = "endpoint_backprojected_1_99_original_mask_top_depth_band"
                if not np.isfinite(float(width_mm)) and corrected_formula_width_mm != NA:
                    width_mm = corrected_formula_width_mm
                    selected_method = "corrected_axis_scale_1_99_original_mask_top_depth_band"
            candidate_widths["depth_width_current_method_mm"] = width_mm
            candidate_widths["depth_width_selected_method_mm"] = width_mm
            percentile_width_loss_px = max(0.0, width_after_depth_filter_px - span_px)

            diagnostic_3d_width_mm = endpoint_width_mm
            candidate_widths["depth_width_from_3d_projection_mm"] = diagnostic_3d_width_mm
            if mask_projected_width_mm and float(mask_projected_width_mm) > 0:
                disagreement = abs(float(width_mm) - float(mask_projected_width_mm)) / float(mask_projected_width_mm)
                if disagreement > float(cfg["max_width_disagreement_ratio"]):
                    failure_reasons.append("depth_refined_width_disagrees_with_mask_width")
        else:
            failure_reasons.append("not_enough_top_face_points_for_width")

        valid_result = not failure_reasons
        erosion_width_loss_mm = _width_mm_from_span_px(erosion_width_loss_px, axis, top_depth_median, fx, fy) if np.isfinite(top_depth_median) else NA
        percentile_width_loss_mm = _width_mm_from_span_px(float(percentile_width_loss_px), axis, top_depth_median, fx, fy) if percentile_width_loss_px != NA and np.isfinite(top_depth_median) else NA
        min_accepted_depth = float(np.min(top_depths)) if top_count else NA
        max_accepted_depth = float(np.max(top_depths)) if top_count else NA
        axis_angle_deg = float(math.degrees(math.atan2(float(axis[1]), float(axis[0]))))
        diagnostics = {
            "rgb_shape": cfg.get("rgb_shape", NA),
            "manual_width_mm": cfg.get("manual_width_mm", NA),
            "mask_shape": list(mask_bool.shape),
            "depth_shape": list(depth_m.shape),
            "camera_info_width": camera_model["camera_info_width"],
            "camera_info_height": camera_model["camera_info_height"],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "detected_shape_mismatch": bool(camera_model["intrinsics_scaled"]),
            "scale_x": camera_model["scale_x"],
            "scale_y": camera_model["scale_y"],
            "intrinsics_scaled": bool(camera_model["intrinsics_scaled"]),
            "original_intrinsics": camera_model["original_intrinsics"],
            "scaled_intrinsics": camera_model["scaled_intrinsics"],
            "major_axis_unit_px": major.tolist(),
            "minor_axis_unit_px": minor.tolist(),
            "selected_upv_axis": selected_upv_axis,
            "clamp_spacing_axis": clamp_spacing_axis,
            "a_clamp": axis.tolist(),
            "a_clamp_norm": float(np.linalg.norm(axis)),
            "dot_major_minor": float(major @ minor),
            "axis_angle_deg": axis_angle_deg,
            **candidate_widths,
            "width_before_erosion_px": width_before_erosion_px,
            "width_after_erosion_px": width_after_erosion_px,
            "width_after_depth_filter_px": width_after_depth_filter_px,
            "erosion_width_loss_px": erosion_width_loss_px,
            "percentile_width_loss_px": percentile_width_loss_px,
            "erosion_width_loss_mm": erosion_width_loss_mm,
            "percentile_width_loss_mm": percentile_width_loss_mm,
            "central_depth_median_m": central_depth_median,
            "central_depth_mad_m": central_mad,
            "top_face_depth_median_m": top_depth_median if np.isfinite(top_depth_median) else NA,
            "top_face_depth_mad_m": top_depth_mad if np.isfinite(top_depth_mad) else NA,
            "depth_band_threshold_m": float(threshold),
            "min_accepted_depth_m": min_accepted_depth,
            "max_accepted_depth_m": max_accepted_depth,
            "accepted_points_count": top_count,
            "rejected_points_count": int(len(rejected_coords)),
            "accepted_point_coverage_ratio": accepted_point_coverage_ratio,
            "valid_depth_points_count_reference_eroded": valid_depth_points_count,
            "valid_depth_points_count_original_mask": int(len(final_xs)),
            "edge_bin_diagnostics": edge_bin_diag,
            "selected_depth_width_method": selected_method,
            "metric_conversion_note": "width_m = ds_px * z * sqrt((a_x/fx)^2 + (a_y/fy)^2); endpoint backprojection uses the same scaled intrinsics",
            "mask_erode_kernel_px": kernel_px,
            "erode_only_for_depth_reference": bool(cfg.get("erode_only_for_depth_reference", True)),
            "use_original_mask_for_final_span": bool(cfg.get("use_original_mask_for_final_span", True)),
            "central_fraction": central_fraction,
            "top_face_points_uv": _sample_points(top_coords, int(cfg["overlay_max_points"])),
            "rejected_points_uv": _sample_points(rejected_coords, int(cfg["overlay_max_points"])),
        }

        return DepthWidthRefinementResult(
            success=True,
            valid=valid_result,
            failure_reason=None if valid_result else ";".join(failure_reasons),
            method="masked_top_face_depth_band_original_mask_endpoint_backprojection",
            mask_projected_width_mm=float(mask_projected_width_mm),
            depth_refined_width_mm=width_mm if width_mm != NA else NA,
            recommended_upv_path_length_mm=width_mm if valid_result else NA,
            upv_path_length_source="depth_refined_top_face_width" if valid_result else "NA_depth_refinement_invalid",
            disagreement_ratio=disagreement,
            clamp_spacing_axis=clamp_spacing_axis,
            selected_upv_axis=selected_upv_axis,
            top_face_points_count=top_count,
            valid_depth_points_count=valid_depth_points_count,
            original_mask_points_count=original_mask_points_count,
            depth_coverage_ratio=float(depth_coverage_ratio),
            central_depth_median_m=central_depth_median,
            top_face_depth_median_m=top_depth_median if np.isfinite(top_depth_median) else NA,
            top_face_depth_mad_m=top_depth_mad if np.isfinite(top_depth_mad) else NA,
            depth_band_threshold_m=float(threshold),
            percentile_low=float(cfg["percentile_low"]),
            percentile_high=float(cfg["percentile_high"]),
            endpoints_px=endpoints_px,
            endpoints_camera_xyz_m=endpoints_xyz,
            diagnostics=diagnostics,
        )
    except Exception as exc:  # noqa: BLE001
        result = _failure(f"exception:{exc}", selected_upv_axis, mask_projected_width_mm, cfg)
        result.diagnostics = {"exception_type": type(exc).__name__}
        return result


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def save_depth_width_refinement_overlay(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    result: DepthWidthRefinementResult,
    output_path: str | Path,
    top_face_points_uv: list[list[int]] | None = None,
    rejected_points_uv: list[list[int]] | None = None,
    major_axis_unit_px: tuple[float, float] | list[float] | np.ndarray | None = None,
    minor_axis_unit_px: tuple[float, float] | list[float] | np.ndarray | None = None,
    centroid_px: tuple[float, float] | list[float] | np.ndarray | None = None,
) -> None:
    image = np.asarray(rgb_image).copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape[:2] != image.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 140
    overlay = np.where(mask_bool[:, :, None], cv2.addWeighted(overlay, 0.72, tint, 0.28, 0), overlay)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    top_points = top_face_points_uv if top_face_points_uv is not None else result.diagnostics.get("top_face_points_uv", [])
    rej_points = rejected_points_uv if rejected_points_uv is not None else result.diagnostics.get("rejected_points_uv", [])
    for u, v in rej_points:
        cv2.circle(overlay, (int(u), int(v)), 1, (80, 80, 255), -1)
    for u, v in top_points:
        cv2.circle(overlay, (int(u), int(v)), 1, (255, 180, 0), -1)

    if centroid_px is not None:
        c = np.asarray(centroid_px, dtype=float).reshape(2)
    else:
        ys, xs = np.nonzero(mask_bool)
        c = np.array([float(np.mean(xs)), float(np.mean(ys))]) if len(xs) else np.array([image.shape[1] / 2, image.shape[0] / 2])
    cv2.circle(overlay, (int(round(c[0])), int(round(c[1]))), 6, (0, 0, 255), -1)

    axes: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []
    if major_axis_unit_px is not None:
        axes.append(("major", _normalize_axis(major_axis_unit_px, "major"), (0, 210, 255)))
    if minor_axis_unit_px is not None:
        axes.append(("minor", _normalize_axis(minor_axis_unit_px, "minor"), (255, 150, 0)))
    for name, axis, color in axes:
        length = 95
        p1 = (int(round(c[0] - axis[0] * length)), int(round(c[1] - axis[1] * length)))
        p2 = (int(round(c[0] + axis[0] * length)), int(round(c[1] + axis[1] * length)))
        thickness = 4 if name == result.clamp_spacing_axis else 2
        cv2.arrowedLine(overlay, p1, p2, color, thickness, tipLength=0.08)

    endpoints = result.endpoints_px or {}
    if "low" in endpoints and "high" in endpoints:
        low = tuple(int(round(v)) for v in endpoints["low"])
        high = tuple(int(round(v)) for v in endpoints["high"])
        cv2.line(overlay, low, high, (255, 255, 255), 3)
        cv2.circle(overlay, low, 7, (255, 255, 255), -1)
        cv2.circle(overlay, high, 7, (255, 255, 255), -1)

    panel_w = 540
    panel = np.full((overlay.shape[0], panel_w, 3), 246, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel.shape[0] - 1), (60, 60, 60), 2)
    lines = [
        "Depth-Refined UPV Path Length",
        f"manual reference mm: {result.diagnostics.get('manual_width_mm', NA)}",
        f"selected_upv_axis: {result.selected_upv_axis}",
        f"clamp_spacing_axis: {result.clamp_spacing_axis}",
        f"mask width mm: {_fmt(result.mask_projected_width_mm)}",
        f"depth-refined mm: {_fmt(result.depth_refined_width_mm)}",
        f"0/100 width mm: {_fmt(result.diagnostics.get('depth_width_percentile_0_100_mm', NA))}",
        f"1/99 width mm: {_fmt(result.diagnostics.get('depth_width_percentile_1_99_mm', NA))}",
        f"2.5/97.5 width mm: {_fmt(result.diagnostics.get('depth_width_percentile_2p5_97p5_mm', NA))}",
        f"edge-bin width mm: {_fmt(result.diagnostics.get('depth_width_from_edge_bins_mm', NA))}",
        f"endpoint width mm: {_fmt(result.diagnostics.get('depth_width_from_backprojected_endpoint_pixels_mm', NA))}",
        f"old f_eff width mm: {_fmt(result.diagnostics.get('depth_width_old_effective_focal_formula_mm', NA))}",
        f"erosion loss mm: {_fmt(result.diagnostics.get('erosion_width_loss_mm', NA))}",
        f"percentile loss mm: {_fmt(result.diagnostics.get('percentile_width_loss_mm', NA))}",
        f"intrinsics_scaled: {result.diagnostics.get('intrinsics_scaled', NA)}",
        f"selected method: {result.diagnostics.get('selected_depth_width_method', NA)}",
        f"valid: {result.valid}",
        f"failure: {result.failure_reason}",
        f"disagreement: {_fmt(result.disagreement_ratio)}",
        f"top-face points: {result.top_face_points_count}",
        f"valid depth points: {result.valid_depth_points_count}",
        f"coverage: {_fmt(result.depth_coverage_ratio)}",
        f"central depth m: {_fmt(result.central_depth_median_m)}",
        f"top depth MAD m: {_fmt(result.top_face_depth_mad_m, 5)}",
        f"UPV path mm: {_fmt(result.recommended_upv_path_length_mm)}",
        f"source: {result.upv_path_length_source}",
    ]
    cv2.putText(panel, lines[0], (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    for i, line in enumerate(lines[1:]):
        cv2.putText(panel, str(line)[:68], (20, 72 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 1, cv2.LINE_AA)
    legend_y = min(panel.shape[0] - 76, 92 + 25 * len(lines))
    cv2.circle(panel, (32, legend_y), 5, (255, 180, 0), -1)
    cv2.putText(panel, "accepted top-face depth points", (52, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.circle(panel, (32, legend_y + 26), 5, (80, 80, 255), -1)
    cv2.putText(panel, "depth-rejected mask points", (52, legend_y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)

    composed = np.concatenate([overlay, panel], axis=1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), composed)
