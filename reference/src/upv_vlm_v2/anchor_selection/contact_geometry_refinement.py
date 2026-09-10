"""Contact-geometry refinement and contact-line gap checks for anchor preparation."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from upv_vlm_v2.anchor_selection.partition_contact_crop_builder import ideal_boundaries_from_mask, local_lateral_bounds
from upv_vlm_v2.anchor_selection.rotated_object_view_builder import rotate_image_and_mask_to_canonical
from upv_vlm_v2.geometry.mask_geometry import estimate_dominant_rectangle_geometry, load_camera_intrinsics, load_depth_array, normalize_vector


def _axis_vector_from_angle(angle_deg: float) -> list[float]:
    theta = math.radians(float(angle_deg) % 180.0)
    return normalize_vector([math.cos(theta), math.sin(theta)])


def _refinement_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    return {
        "dominant_rectangle_min_column_coverage_px": int(cfg.get("min_column_coverage_px", 6)),
        "dominant_rectangle_trim_fraction": float(cfg.get("trim_percentile", 0.10)),
        "dominant_rectangle_mad_multiplier": float(cfg.get("mad_multiplier", 2.5)),
        "dominant_rectangle_edge_method": str(cfg.get("method", "robust_core_rectangle")),
    }


def _normalize01(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.float32)
    vals = data[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo + 1e-9:
        return np.zeros_like(data, dtype=np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    out[finite] = (data[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)


def _dominant_rect_mask(mask: np.ndarray, geometry: dict[str, Any], cfg: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any]]:
    raw = {
        "centroid_px": geometry.get("centroid_px"),
        "major_axis_vector": geometry.get("major_axis_vector"),
    }
    rect = estimate_dominant_rectangle_geometry(mask, raw, _refinement_cfg(cfg))
    corners = np.asarray(rect["body_rect_corners_px"], dtype=np.float32)
    core_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(core_mask, [np.round(corners).astype(np.int32)], 1)
    return core_mask, rect


def _ensure_float_depth(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None
    arr = np.asarray(depth, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _crop_box_from_mask(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Mask has no foreground pixels for ROI crop.")
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def _sobel_edge_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    canny = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0
    return np.maximum(_normalize01(mag), canny)


def _depth_edge_map(depth: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray:
    if depth is None:
        return np.zeros(valid_mask.shape, dtype=np.float32)
    d = depth.copy()
    if float(np.nanmax(d)) > 20.0:
        d = d / 1000.0
    d[~np.isfinite(d)] = 0.0
    d[valid_mask <= 0] = 0.0
    fill = cv2.GaussianBlur(d, (5, 5), 0)
    gx = cv2.Sobel(fill, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(fill, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag[valid_mask <= 0] = 0.0
    return _normalize01(mag)


def _weighted_ransac_constant(
    values: np.ndarray,
    weights: np.ndarray,
    distance_threshold: float,
    min_inliers: int,
) -> tuple[float, np.ndarray]:
    vals = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    good = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[good]
    w = w[good]
    if vals.size == 0:
        raise ValueError("No candidate edge samples.")
    order = np.argsort(w)[::-1]
    vals = vals[order]
    w = w[order]
    best_center = float(np.median(vals))
    best_inliers = np.abs(vals - best_center) <= distance_threshold
    best_score = float(np.sum(w[best_inliers]))
    top_k = min(128, vals.size)
    for center in vals[:top_k]:
        inliers = np.abs(vals - center) <= distance_threshold
        count = int(np.sum(inliers))
        if count < max(1, min_inliers):
            continue
        score = float(np.sum(w[inliers]))
        if score > best_score:
            best_center = float(center)
            best_inliers = inliers
            best_score = score
    if int(np.sum(best_inliers)) < max(1, min_inliers):
        best_inliers = np.abs(vals - best_center) <= max(distance_threshold, 5.0)
    refined = float(np.average(vals[best_inliers], weights=w[best_inliers])) if bool(best_inliers.any()) else best_center
    return refined, best_inliers


def _sample_edge_positions(
    *,
    combined_edge: np.ndarray,
    roi_mask: np.ndarray,
    expected_top: float,
    expected_bottom: float,
    expected_left: float,
    expected_right: float,
    band_px: int,
    distance_bias: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    h, w = combined_edge.shape
    top_vals: list[float] = []
    top_weights: list[float] = []
    bottom_vals: list[float] = []
    bottom_weights: list[float] = []
    left_vals: list[float] = []
    left_weights: list[float] = []
    right_vals: list[float] = []
    right_weights: list[float] = []
    x_support = np.where(np.sum(roi_mask > 0, axis=0) > 0)[0]
    y_support = np.where(np.sum(roi_mask > 0, axis=1) > 0)[0]
    for x in x_support.tolist():
        y0 = max(0, int(math.floor(expected_top - band_px)))
        y1 = min(h, int(math.ceil(expected_top + band_px + 1)))
        col = combined_edge[y0:y1, x]
        if col.size:
            ys = np.arange(y0, y1, dtype=np.float32)
            penalized = col - distance_bias * (np.abs(ys - expected_top) / max(1.0, float(band_px)))
            idx = int(np.argmax(penalized))
            top_vals.append(float(y0 + idx))
            top_weights.append(float(col[idx]))
        y0 = max(0, int(math.floor(expected_bottom - band_px)))
        y1 = min(h, int(math.ceil(expected_bottom + band_px + 1)))
        col = combined_edge[y0:y1, x]
        if col.size:
            ys = np.arange(y0, y1, dtype=np.float32)
            penalized = col - distance_bias * (np.abs(ys - expected_bottom) / max(1.0, float(band_px)))
            idx = int(np.argmax(penalized))
            bottom_vals.append(float(y0 + idx))
            bottom_weights.append(float(col[idx]))
    for y in y_support.tolist():
        x0 = max(0, int(math.floor(expected_left - band_px)))
        x1 = min(w, int(math.ceil(expected_left + band_px + 1)))
        row = combined_edge[y, x0:x1]
        if row.size:
            xs = np.arange(x0, x1, dtype=np.float32)
            penalized = row - distance_bias * (np.abs(xs - expected_left) / max(1.0, float(band_px)))
            idx = int(np.argmax(penalized))
            left_vals.append(float(x0 + idx))
            left_weights.append(float(row[idx]))
        x0 = max(0, int(math.floor(expected_right - band_px)))
        x1 = min(w, int(math.ceil(expected_right + band_px + 1)))
        row = combined_edge[y, x0:x1]
        if row.size:
            xs = np.arange(x0, x1, dtype=np.float32)
            penalized = row - distance_bias * (np.abs(xs - expected_right) / max(1.0, float(band_px)))
            idx = int(np.argmax(penalized))
            right_vals.append(float(x0 + idx))
            right_weights.append(float(row[idx]))
    return {
        "top": (np.asarray(top_vals, dtype=np.float32), np.asarray(top_weights, dtype=np.float32)),
        "bottom": (np.asarray(bottom_vals, dtype=np.float32), np.asarray(bottom_weights, dtype=np.float32)),
        "left": (np.asarray(left_vals, dtype=np.float32), np.asarray(left_weights, dtype=np.float32)),
        "right": (np.asarray(right_vals, dtype=np.float32), np.asarray(right_weights, dtype=np.float32)),
    }


def _render_debug_overlay(
    image_rgb: np.ndarray,
    roi_box: tuple[int, int, int, int],
    top_y: float,
    bottom_y: float,
    left_x: float,
    right_x: float,
) -> np.ndarray:
    x0, y0, x1, y1 = roi_box
    overlay = image_rgb.copy()
    color = (255, 30, 180)
    cv2.line(overlay, (int(round(x0 + left_x)), int(round(y0 + top_y))), (int(round(x0 + right_x)), int(round(y0 + top_y))), color, 2)
    cv2.line(overlay, (int(round(x0 + left_x)), int(round(y0 + bottom_y))), (int(round(x0 + right_x)), int(round(y0 + bottom_y))), color, 2)
    cv2.line(overlay, (int(round(x0 + left_x)), int(round(y0 + top_y))), (int(round(x0 + left_x)), int(round(y0 + bottom_y))), color, 2)
    cv2.line(overlay, (int(round(x0 + right_x)), int(round(y0 + top_y))), (int(round(x0 + right_x)), int(round(y0 + bottom_y))), color, 2)
    return overlay


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if vals.size == 0:
        raise ValueError("No values for weighted median.")
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1]) if cdf.size else 0.0
    if total <= 0.0:
        return float(np.median(vals))
    idx = int(np.searchsorted(cdf, total * 0.5))
    idx = max(0, min(idx, vals.size - 1))
    return float(vals[idx])


def _trimmed_values(values: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    lo = float(np.quantile(vals, low_q))
    hi = float(np.quantile(vals, high_q))
    kept = vals[(vals >= lo) & (vals <= hi)]
    return kept if kept.size else vals


def _support_position(
    values: np.ndarray,
    *,
    low_q: float,
    high_q: float,
    support_band_px: float,
    min_support_fraction: float,
) -> tuple[float, dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No boundary samples for support profile.")
    trimmed = _trimmed_values(vals, low_q, high_q)
    rounded = np.round(trimmed).astype(np.int32)
    bins, counts = np.unique(rounded, return_counts=True)
    mode_bin = float(bins[int(np.argmax(counts))])
    inliers = vals[np.abs(vals - mode_bin) <= support_band_px]
    support_fraction = float(inliers.size / max(1, vals.size))
    if inliers.size == 0 or support_fraction < min_support_fraction:
        center = float(np.median(trimmed))
        inliers = vals[np.abs(vals - center) <= max(support_band_px, 3.0)]
        support_fraction = float(inliers.size / max(1, vals.size))
        mode_bin = center
    if inliers.size == 0:
        inliers = trimmed if trimmed.size else vals
    final = float(np.median(inliers))
    return final, {
        "mode_bin": mode_bin,
        "support_fraction": support_fraction,
        "sample_count": int(vals.size),
        "inlier_count": int(inliers.size),
        "support_band_px": float(support_band_px),
    }


def _boundary_samples_from_mask(roi_mask: np.ndarray, step_px: int) -> dict[str, np.ndarray]:
    top_vals: list[float] = []
    bottom_vals: list[float] = []
    top_xs: list[float] = []
    bottom_xs: list[float] = []
    for x in range(0, roi_mask.shape[1], max(1, step_px)):
        ys = np.where(roi_mask[:, x] > 0)[0]
        if ys.size:
            top_vals.append(float(ys.min()))
            bottom_vals.append(float(ys.max()))
            top_xs.append(float(x))
            bottom_xs.append(float(x))
    left_vals: list[float] = []
    right_vals: list[float] = []
    left_ys: list[float] = []
    right_ys: list[float] = []
    for y in range(0, roi_mask.shape[0], max(1, step_px)):
        xs = np.where(roi_mask[y, :] > 0)[0]
        if xs.size:
            left_vals.append(float(xs.min()))
            right_vals.append(float(xs.max()))
            left_ys.append(float(y))
            right_ys.append(float(y))
    return {
        "top_values": np.asarray(top_vals, dtype=np.float32),
        "top_axis": np.asarray(top_xs, dtype=np.float32),
        "bottom_values": np.asarray(bottom_vals, dtype=np.float32),
        "bottom_axis": np.asarray(bottom_xs, dtype=np.float32),
        "left_values": np.asarray(left_vals, dtype=np.float32),
        "left_axis": np.asarray(left_ys, dtype=np.float32),
        "right_values": np.asarray(right_vals, dtype=np.float32),
        "right_axis": np.asarray(right_ys, dtype=np.float32),
    }


def _render_side_support_profiles(
    samples: dict[str, np.ndarray],
    final_sides: dict[str, float],
    support_meta: dict[str, dict[str, Any]],
    size: tuple[int, int],
) -> np.ndarray:
    h, w = size
    canvas = np.full((h, w, 3), 248, dtype=np.uint8)
    panel_w = w // 2
    panel_h = h // 2
    specs = [
        ("top", (0, 0, panel_w, panel_h), (220, 70, 70)),
        ("bottom", (panel_w, 0, w, panel_h), (70, 180, 70)),
        ("left", (0, panel_h, panel_w, h), (80, 120, 230)),
        ("right", (panel_w, panel_h, w, h), (190, 80, 210)),
    ]
    for side, (x0, y0, x1, y1), color in specs:
        vals = samples[f"{side}_values"]
        if vals.size == 0:
            continue
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        span = max(1.0, hi - lo)
        bins: dict[int, int] = {}
        for v in vals.tolist():
            key = int(round(v))
            bins[key] = bins.get(key, 0) + 1
        max_count = max(bins.values()) if bins else 1
        for key, count in bins.items():
            t = (float(key) - lo) / span
            py = int(round((y1 - 12) - t * ((y1 - y0) - 24)))
            bar_w = int(round((count / max_count) * ((x1 - x0) - 40)))
            cv2.line(canvas, (x0 + 18, py), (x0 + 18 + bar_w, py), color, 2)
        final_v = final_sides[side]
        t = (final_v - lo) / span
        py = int(round((y1 - 12) - t * ((y1 - y0) - 24)))
        cv2.line(canvas, (x0 + 8, py), (x1 - 8, py), (20, 20, 20), 1)
        label = f"{side}: {final_v:.1f}px s={support_meta[side]['support_fraction']:.2f}"
        cv2.putText(canvas, label, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _render_boundary_samples_overlay(
    roi_rgb: np.ndarray,
    samples: dict[str, np.ndarray],
    final_sides: dict[str, float],
) -> np.ndarray:
    overlay = roi_rgb.copy()
    for x, v in zip(samples["top_axis"], samples["top_values"]):
        cv2.circle(overlay, (int(round(x)), int(round(v))), 1, (255, 0, 0), -1)
    for x, v in zip(samples["bottom_axis"], samples["bottom_values"]):
        cv2.circle(overlay, (int(round(x)), int(round(v))), 1, (0, 255, 0), -1)
    for y, v in zip(samples["left_axis"], samples["left_values"]):
        cv2.circle(overlay, (int(round(v)), int(round(y))), 1, (0, 180, 255), -1)
    for y, v in zip(samples["right_axis"], samples["right_values"]):
        cv2.circle(overlay, (int(round(v)), int(round(y))), 1, (255, 0, 255), -1)
    cv2.line(overlay, (0, int(round(final_sides["top"]))), (overlay.shape[1] - 1, int(round(final_sides["top"]))), (255, 30, 30), 1)
    cv2.line(overlay, (0, int(round(final_sides["bottom"]))), (overlay.shape[1] - 1, int(round(final_sides["bottom"]))), (30, 255, 30), 1)
    cv2.line(overlay, (int(round(final_sides["left"])), 0), (int(round(final_sides["left"])), overlay.shape[0] - 1), (30, 180, 255), 1)
    cv2.line(overlay, (int(round(final_sides["right"])), 0), (int(round(final_sides["right"])), overlay.shape[0] - 1), (255, 30, 255), 1)
    return overlay


def _refine_side_with_edge_band(
    *,
    combined_edge: np.ndarray,
    axis_positions: np.ndarray,
    side_name: str,
    initial_value: float,
    max_refine_px: float,
) -> tuple[float, dict[str, Any]]:
    if axis_positions.size == 0 or max_refine_px <= 0:
        return initial_value, {"used": False, "refined_value": initial_value}
    picked_vals: list[float] = []
    picked_w: list[float] = []
    if side_name in {"top", "bottom"}:
        for axis_pos in axis_positions.tolist():
            x = int(round(axis_pos))
            y0 = max(0, int(math.floor(initial_value - max_refine_px)))
            y1 = min(combined_edge.shape[0], int(math.ceil(initial_value + max_refine_px + 1)))
            col = combined_edge[y0:y1, x]
            if col.size == 0:
                continue
            ys = np.arange(y0, y1, dtype=np.float32)
            idx = int(np.argmax(col))
            picked_vals.append(float(ys[idx]))
            picked_w.append(float(col[idx]))
    else:
        for axis_pos in axis_positions.tolist():
            y = int(round(axis_pos))
            x0 = max(0, int(math.floor(initial_value - max_refine_px)))
            x1 = min(combined_edge.shape[1], int(math.ceil(initial_value + max_refine_px + 1)))
            row = combined_edge[y, x0:x1]
            if row.size == 0:
                continue
            xs = np.arange(x0, x1, dtype=np.float32)
            idx = int(np.argmax(row))
            picked_vals.append(float(xs[idx]))
            picked_w.append(float(row[idx]))
    if not picked_vals:
        return initial_value, {"used": False, "refined_value": initial_value}
    refined = _weighted_median(np.asarray(picked_vals, dtype=np.float32), np.asarray(picked_w, dtype=np.float32))
    refined = float(np.clip(refined, initial_value - max_refine_px, initial_value + max_refine_px))
    return refined, {
        "used": True,
        "refined_value": refined,
        "refine_delta_px": float(refined - initial_value),
        "sample_count": len(picked_vals),
    }


def _build_rgbd_edge_ransac_rectangle(
    *,
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    geometry: dict[str, Any],
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = cfg or {}
    rough_mask, rough_rect = _dominant_rect_mask(mask, geometry, cfg)
    center = rough_rect["body_rect_center_px"]
    axis = _axis_vector_from_angle(float(rough_rect["body_rect_major_axis_angle_deg"]))
    rotated_rgb, rotated_mask_u8, rotation_matrix, _ = rotate_image_and_mask_to_canonical(
        image_rgb=rgb,
        mask=mask,
        axis_center_xy=center,
        axis_dir_xy=axis,
    )
    rotated_mask = (rotated_mask_u8 > 0).astype(np.uint8)
    rotated_rough_rect_mask_u8 = cv2.warpAffine(
        _mask_to_u8(rough_mask),
        rotation_matrix,
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_rough_rect_mask = (rotated_rough_rect_mask_u8 > 0).astype(np.uint8)
    rotated_depth = None
    if depth is not None:
        rotated_depth = cv2.warpAffine(
            depth.astype(np.float32),
            rotation_matrix,
            dsize=(mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    roi_box = _crop_box_from_mask(rotated_mask, int(cfg.get("boundary_search_band_px", 35)) + 8)
    x0, y0, x1, y1 = roi_box
    roi_rgb = rotated_rgb[y0:y1, x0:x1].copy()
    roi_mask = rotated_mask[y0:y1, x0:x1].copy()
    roi_rect_mask = rotated_rough_rect_mask[y0:y1, x0:x1].copy()
    roi_depth = rotated_depth[y0:y1, x0:x1].copy() if rotated_depth is not None else None

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = _sobel_edge_map(gray)
    depth_edges = _depth_edge_map(roi_depth, roi_mask)
    rgb_w = float(cfg.get("rgb_edge_weight", 0.6))
    depth_w = float(cfg.get("depth_edge_weight", 0.4))
    combined = (rgb_w * rgb_edges) + (depth_w * depth_edges)
    combined *= (roi_mask > 0).astype(np.float32)
    combined = _normalize01(combined)

    ys, xs = np.nonzero(roi_rect_mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Refined ROI has no supported rectangle mask.")
    expected_top = float(np.min(ys))
    expected_bottom = float(np.max(ys))
    expected_left = float(np.min(xs))
    expected_right = float(np.max(xs))
    samples = _sample_edge_positions(
        combined_edge=combined,
        roi_mask=roi_mask,
        expected_top=expected_top,
        expected_bottom=expected_bottom,
        expected_left=expected_left,
        expected_right=expected_right,
        band_px=int(cfg.get("boundary_search_band_px", 35)),
        distance_bias=float(cfg.get("boundary_distance_bias", 0.75)),
    )
    dist = float(cfg.get("ransac_distance_threshold_px", 3.0))
    min_inliers = int(cfg.get("ransac_min_inliers", 30))
    top_y, top_inliers = _weighted_ransac_constant(samples["top"][0], samples["top"][1], dist, min_inliers)
    bottom_y, bottom_inliers = _weighted_ransac_constant(samples["bottom"][0], samples["bottom"][1], dist, min_inliers)
    left_x, left_inliers = _weighted_ransac_constant(samples["left"][0], samples["left"][1], dist, min_inliers)
    right_x, right_inliers = _weighted_ransac_constant(samples["right"][0], samples["right"][1], dist, min_inliers)
    if right_x <= left_x or bottom_y <= top_y:
        raise ValueError("RGB-D edge rectangle is degenerate.")

    corners_roi = np.asarray(
        [[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y]],
        dtype=np.float32,
    )
    corners_rot = corners_roi + np.asarray([x0, y0], dtype=np.float32)
    inv = cv2.invertAffineTransform(rotation_matrix)
    corners_orig = (inv @ np.hstack([corners_rot, np.ones((4, 1), dtype=np.float32)]).T).T
    center_rot = np.asarray([(left_x + right_x) / 2.0 + x0, (top_y + bottom_y) / 2.0 + y0, 1.0], dtype=np.float32)
    center_orig = inv @ center_rot

    core_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(core_mask, [np.round(corners_orig).astype(np.int32)], 1)

    overlay = _render_debug_overlay(rotated_rgb, roi_box, top_y, bottom_y, left_x, right_x)
    debug_images = {
        "contact_rgb_edge_map": (np.clip(rgb_edges, 0.0, 1.0) * 255).astype(np.uint8),
        "contact_depth_edge_map": (np.clip(depth_edges, 0.0, 1.0) * 255).astype(np.uint8),
        "contact_combined_edge_map": (np.clip(combined, 0.0, 1.0) * 255).astype(np.uint8),
        "contact_ransac_edge_lines_overlay": overlay,
    }
    diagnostics = {
        "roi_box_rotated": [int(x0), int(y0), int(x1), int(y1)],
        "expected_boundaries_roi": {
            "top": expected_top,
            "bottom": expected_bottom,
            "left": expected_left,
            "right": expected_right,
        },
        "fitted_boundaries_roi": {
            "top": float(top_y),
            "bottom": float(bottom_y),
            "left": float(left_x),
            "right": float(right_x),
        },
        "ransac_inlier_counts": {
            "top": int(np.sum(top_inliers)),
            "bottom": int(np.sum(bottom_inliers)),
            "left": int(np.sum(left_inliers)),
            "right": int(np.sum(right_inliers)),
        },
    }
    return {
        "success": True,
        "method": "rgbd_edge_ransac_rectangle",
        "contact_geometry_mask": core_mask,
        "dominant_rectangle": {
            "success": True,
            "body_geometry_mode": "contact_core_rectangle",
            "edge_method": "rgbd_edge_ransac_rectangle",
            "body_rect_center_px": [float(center_orig[0]), float(center_orig[1])],
            "body_rect_corners_px": [[float(x), float(y)] for x, y in corners_orig.tolist()],
            "body_rect_major_axis_angle_deg": float(rough_rect["body_rect_major_axis_angle_deg"]),
            "body_rect_minor_axis_angle_deg": float(rough_rect["body_rect_minor_axis_angle_deg"]),
            "body_rect_major_axis_length_px": float(right_x - left_x),
            "body_rect_minor_axis_length_px": float(bottom_y - top_y),
            "dominant_top_y_rotated": float(top_y + y0),
            "dominant_bottom_y_rotated": float(bottom_y + y0),
            "dominant_left_x_rotated": float(left_x + x0),
            "dominant_right_x_rotated": float(right_x + x0),
            "fallback_used": False,
            "rgbd_edge_ransac_diagnostics": diagnostics,
        },
        "centroid_px": [float(center_orig[0]), float(center_orig[1])],
        "major_axis_vector": _axis_vector_from_angle(float(rough_rect["body_rect_major_axis_angle_deg"])),
        "minor_axis_vector": _axis_vector_from_angle(float(rough_rect["body_rect_minor_axis_angle_deg"])),
        "major_axis_angle_deg": float(rough_rect["body_rect_major_axis_angle_deg"]),
        "minor_axis_angle_deg": float(rough_rect["body_rect_minor_axis_angle_deg"]),
        "major_axis_length_px": float(right_x - left_x),
        "minor_axis_length_px": float(bottom_y - top_y),
        "body_geometry_mode": "contact_core_rectangle",
        "debug_images": debug_images,
        "debug_diagnostics": diagnostics,
    }


def _build_orientation_fixed_core_support_rectangle(
    *,
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    geometry: dict[str, Any],
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = cfg or {}
    rough_mask, rough_rect = _dominant_rect_mask(mask, geometry, cfg)
    center = rough_rect["body_rect_center_px"]
    axis = _axis_vector_from_angle(float(rough_rect["body_rect_major_axis_angle_deg"]))
    rotated_rgb, rotated_mask_u8, rotation_matrix, _ = rotate_image_and_mask_to_canonical(
        image_rgb=rgb,
        mask=mask,
        axis_center_xy=center,
        axis_dir_xy=axis,
    )
    rotated_mask = (rotated_mask_u8 > 0).astype(np.uint8)
    rotated_depth = None
    if depth is not None:
        rotated_depth = cv2.warpAffine(
            depth.astype(np.float32),
            rotation_matrix,
            dsize=(mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    roi_box = _crop_box_from_mask(rotated_mask, int(cfg.get("boundary_search_band_px", 35)) + 8)
    x0, y0, x1, y1 = roi_box
    roi_rgb = rotated_rgb[y0:y1, x0:x1].copy()
    roi_mask = rotated_mask[y0:y1, x0:x1].copy()
    roi_depth = rotated_depth[y0:y1, x0:x1].copy() if rotated_depth is not None else None

    samples = _boundary_samples_from_mask(roi_mask, int(cfg.get("boundary_sample_step_px", 2)))
    low_q = float(cfg.get("trim_low_percentile", 0.10))
    high_q = float(cfg.get("trim_high_percentile", 0.90))
    min_support = max(
        float(cfg.get("min_side_support_fraction", 0.55)),
        float(cfg.get("side_support_threshold", 0.58)),
    )
    vertical_span = (
        float(np.median(samples["bottom_values"] - samples["top_values"]))
        if samples["top_values"].size and samples["bottom_values"].size
        else float(roi_mask.shape[0] * 0.25)
    )
    horizontal_span = (
        float(np.median(samples["right_values"] - samples["left_values"]))
        if samples["left_values"].size and samples["right_values"].size
        else float(roi_mask.shape[1] * 0.25)
    )
    vertical_band = max(2.0, vertical_span * 0.05)
    horizontal_band = max(2.0, horizontal_span * 0.05)

    final_sides: dict[str, float] = {}
    support_meta: dict[str, dict[str, Any]] = {}
    final_sides["top"], support_meta["top"] = _support_position(
        samples["top_values"], low_q=low_q, high_q=high_q, support_band_px=vertical_band, min_support_fraction=min_support
    )
    final_sides["bottom"], support_meta["bottom"] = _support_position(
        samples["bottom_values"], low_q=low_q, high_q=high_q, support_band_px=vertical_band, min_support_fraction=min_support
    )
    final_sides["left"], support_meta["left"] = _support_position(
        samples["left_values"], low_q=low_q, high_q=high_q, support_band_px=horizontal_band, min_support_fraction=min_support
    )
    final_sides["right"], support_meta["right"] = _support_position(
        samples["right_values"], low_q=low_q, high_q=high_q, support_band_px=horizontal_band, min_support_fraction=min_support
    )

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = _sobel_edge_map(gray)
    depth_edges = _depth_edge_map(roi_depth, roi_mask)
    combined = _normalize01(
        float(cfg.get("rgb_edge_weight", 0.5)) * rgb_edges
        + float(cfg.get("depth_edge_weight", 0.5)) * depth_edges
    )
    pre_tiebreak = dict(final_sides)
    tiebreak_meta: dict[str, Any] = {}
    if bool(cfg.get("use_rgbd_edge_tiebreak", True)):
        max_refine_px = float(cfg.get("max_edge_refine_px", 5))
        final_sides["top"], tiebreak_meta["top"] = _refine_side_with_edge_band(
            combined_edge=combined, axis_positions=samples["top_axis"], side_name="top", initial_value=final_sides["top"], max_refine_px=max_refine_px
        )
        final_sides["bottom"], tiebreak_meta["bottom"] = _refine_side_with_edge_band(
            combined_edge=combined, axis_positions=samples["bottom_axis"], side_name="bottom", initial_value=final_sides["bottom"], max_refine_px=max_refine_px
        )
        final_sides["left"], tiebreak_meta["left"] = _refine_side_with_edge_band(
            combined_edge=combined, axis_positions=samples["left_axis"], side_name="left", initial_value=final_sides["left"], max_refine_px=max_refine_px
        )
        final_sides["right"], tiebreak_meta["right"] = _refine_side_with_edge_band(
            combined_edge=combined, axis_positions=samples["right_axis"], side_name="right", initial_value=final_sides["right"], max_refine_px=max_refine_px
        )

    if final_sides["right"] <= final_sides["left"] or final_sides["bottom"] <= final_sides["top"]:
        raise ValueError("orientation_fixed_core_support_rectangle is degenerate.")

    corners_roi = np.asarray(
        [
            [final_sides["left"], final_sides["top"]],
            [final_sides["right"], final_sides["top"]],
            [final_sides["right"], final_sides["bottom"]],
            [final_sides["left"], final_sides["bottom"]],
        ],
        dtype=np.float32,
    )
    corners_rot = corners_roi + np.asarray([x0, y0], dtype=np.float32)
    inv = cv2.invertAffineTransform(rotation_matrix)
    corners_orig = (inv @ np.hstack([corners_rot, np.ones((4, 1), dtype=np.float32)]).T).T
    center_rot = np.asarray(
        [(final_sides["left"] + final_sides["right"]) / 2.0 + x0, (final_sides["top"] + final_sides["bottom"]) / 2.0 + y0, 1.0],
        dtype=np.float32,
    )
    center_orig = inv @ center_rot
    core_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(core_mask, [np.round(corners_orig).astype(np.int32)], 1)

    samples_overlay = _render_boundary_samples_overlay(roi_rgb, samples, final_sides)
    rect_overlay = _render_debug_overlay(rotated_rgb, roi_box, final_sides["top"], final_sides["bottom"], final_sides["left"], final_sides["right"])
    tiebreak_overlay = _render_debug_overlay(
        rotated_rgb, roi_box, pre_tiebreak["top"], pre_tiebreak["bottom"], pre_tiebreak["left"], pre_tiebreak["right"]
    )
    cv2.line(
        tiebreak_overlay,
        (int(round(x0 + final_sides["left"])), int(round(y0 + final_sides["top"]))),
        (int(round(x0 + final_sides["right"])), int(round(y0 + final_sides["top"]))),
        (60, 255, 255),
        1,
    )
    cv2.line(
        tiebreak_overlay,
        (int(round(x0 + final_sides["left"])), int(round(y0 + final_sides["bottom"]))),
        (int(round(x0 + final_sides["right"])), int(round(y0 + final_sides["bottom"]))),
        (60, 255, 255),
        1,
    )
    cv2.line(
        tiebreak_overlay,
        (int(round(x0 + final_sides["left"])), int(round(y0 + final_sides["top"]))),
        (int(round(x0 + final_sides["left"])), int(round(y0 + final_sides["bottom"]))),
        (60, 255, 255),
        1,
    )
    cv2.line(
        tiebreak_overlay,
        (int(round(x0 + final_sides["right"])), int(round(y0 + final_sides["top"]))),
        (int(round(x0 + final_sides["right"])), int(round(y0 + final_sides["bottom"]))),
        (60, 255, 255),
        1,
    )

    diagnostics = {
        "roi_box_rotated": [int(x0), int(y0), int(x1), int(y1)],
        "support_meta": support_meta,
        "pre_tiebreak_sides_roi": {k: float(v) for k, v in pre_tiebreak.items()},
        "final_sides_roi": {k: float(v) for k, v in final_sides.items()},
        "tiebreak_meta": tiebreak_meta,
    }
    debug_images = {
        "local_rotated_mask": _mask_to_u8(roi_mask),
        "side_support_profiles": _render_side_support_profiles(samples, final_sides, support_meta, (640, 900)),
        "side_boundary_samples_overlay": samples_overlay,
        "side_support_rectangle_overlay": rect_overlay,
        "rgbd_edge_tiebreak_overlay": tiebreak_overlay,
    }
    return {
        "success": True,
        "method": "orientation_fixed_core_support_rectangle",
        "contact_geometry_mask": core_mask,
        "dominant_rectangle": {
            "success": True,
            "body_geometry_mode": "contact_core_rectangle",
            "edge_method": "orientation_fixed_core_support_rectangle",
            "body_rect_center_px": [float(center_orig[0]), float(center_orig[1])],
            "body_rect_corners_px": [[float(x), float(y)] for x, y in corners_orig.tolist()],
            "body_rect_major_axis_angle_deg": float(rough_rect["body_rect_major_axis_angle_deg"]),
            "body_rect_minor_axis_angle_deg": float(rough_rect["body_rect_minor_axis_angle_deg"]),
            "body_rect_major_axis_length_px": float(final_sides["right"] - final_sides["left"]),
            "body_rect_minor_axis_length_px": float(final_sides["bottom"] - final_sides["top"]),
            "dominant_top_y_rotated": float(final_sides["top"] + y0),
            "dominant_bottom_y_rotated": float(final_sides["bottom"] + y0),
            "dominant_left_x_rotated": float(final_sides["left"] + x0),
            "dominant_right_x_rotated": float(final_sides["right"] + x0),
            "fallback_used": False,
            "orientation_fixed_core_support_diagnostics": diagnostics,
        },
        "centroid_px": [float(center_orig[0]), float(center_orig[1])],
        "major_axis_vector": _axis_vector_from_angle(float(rough_rect["body_rect_major_axis_angle_deg"])),
        "minor_axis_vector": _axis_vector_from_angle(float(rough_rect["body_rect_minor_axis_angle_deg"])),
        "major_axis_angle_deg": float(rough_rect["body_rect_major_axis_angle_deg"]),
        "minor_axis_angle_deg": float(rough_rect["body_rect_minor_axis_angle_deg"]),
        "major_axis_length_px": float(final_sides["right"] - final_sides["left"]),
        "minor_axis_length_px": float(final_sides["bottom"] - final_sides["top"]),
        "body_geometry_mode": "contact_core_rectangle",
        "debug_images": debug_images,
        "debug_diagnostics": diagnostics,
    }


def build_contact_geometry_refinement(
    *,
    mask: np.ndarray,
    geometry: dict[str, Any],
    cfg: dict[str, Any] | None,
    rgb: np.ndarray | None = None,
    depth_path: str | None = None,
    camera_info_path: str | None = None,
) -> dict[str, Any]:
    method = str((cfg or {}).get("method", "robust_core_rectangle"))
    if method == "orientation_fixed_core_support_rectangle":
        if rgb is None:
            raise ValueError("orientation_fixed_core_support_rectangle requires rgb image input.")
        depth = _ensure_float_depth(load_depth_array(depth_path))
        intrinsics = load_camera_intrinsics(camera_info_path)
        result = _build_orientation_fixed_core_support_rectangle(mask=mask, rgb=rgb, depth=depth, geometry=geometry, cfg=cfg)
        result["camera_intrinsics_available"] = bool(intrinsics)
        result["depth_available"] = depth is not None
        return result
    if method == "rgbd_edge_ransac_rectangle":
        if rgb is None:
            raise ValueError("rgbd_edge_ransac_rectangle requires rgb image input.")
        depth = _ensure_float_depth(load_depth_array(depth_path))
        intrinsics = load_camera_intrinsics(camera_info_path)
        result = _build_rgbd_edge_ransac_rectangle(mask=mask, rgb=rgb, depth=depth, geometry=geometry, cfg=cfg)
        result["camera_intrinsics_available"] = bool(intrinsics)
        result["depth_available"] = depth is not None
        return result

    core_mask, rect = _dominant_rect_mask(mask, geometry, cfg)
    return {
        "success": True,
        "method": method,
        "contact_geometry_mask": core_mask,
        "dominant_rectangle": rect,
        "centroid_px": rect["body_rect_center_px"],
        "major_axis_vector": _axis_vector_from_angle(rect["body_rect_major_axis_angle_deg"]),
        "minor_axis_vector": _axis_vector_from_angle(rect["body_rect_minor_axis_angle_deg"]),
        "major_axis_angle_deg": rect["body_rect_major_axis_angle_deg"],
        "minor_axis_angle_deg": rect["body_rect_minor_axis_angle_deg"],
        "major_axis_length_px": rect["body_rect_major_axis_length_px"],
        "minor_axis_length_px": rect["body_rect_minor_axis_length_px"],
        "body_geometry_mode": "contact_core_rectangle",
        "debug_images": {},
        "debug_diagnostics": {},
    }


def compute_contact_geometry_quality(
    *,
    rotated_raw_mask: np.ndarray,
    rotation_mappings: dict[str, Any],
    refined_top_y: float,
    refined_bottom_y: float,
    image_width: int,
    cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    cfg = cfg or {}
    max_gap_px = float(cfg.get("max_contact_line_gap_px", 18.0))
    x_positions: dict[str, float] = {}
    for aid, meta in rotation_mappings.items():
        if not isinstance(meta, dict):
            continue
        patch = meta.get("true_edge_pair_patch") or {}
        anchor_xy = patch.get("anchor_xy_rotated")
        if isinstance(anchor_xy, (list, tuple)) and len(anchor_xy) >= 2:
            try:
                x_positions[str(aid)] = float(anchor_xy[0])
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for aid in sorted(rotation_mappings):
        row: dict[str, Any] = {
            "anchor_id": aid,
            "max_contact_line_gap_px": max_gap_px,
            "within_gap_threshold": False,
            "gap_exceeds_threshold": True,
            "failure_reason": "missing_anchor_x_position",
        }
        if aid not in x_positions:
            rows.append(row)
            continue
        x0, x1, lateral_meta = local_lateral_bounds(anchor_id=aid, x_positions=x_positions, image_width=image_width, config=cfg)
        window = rotated_raw_mask[:, x0:x1]
        local = ideal_boundaries_from_mask(window, cfg)
        if not local.get("success"):
            row.update({**lateral_meta, "failure_reason": str(local.get("failure_reason") or "missing_local_boundary_window")})
            rows.append(row)
            continue
        raw_top = float(local["top_boundary_y"])
        raw_bottom = float(local["bottom_boundary_y"])
        top_gap = abs(float(refined_top_y) - raw_top)
        bottom_gap = abs(float(refined_bottom_y) - raw_bottom)
        max_gap = max(top_gap, bottom_gap)
        row.update(
            {
                **lateral_meta,
                "failure_reason": "",
                "raw_local_top_boundary_y": raw_top,
                "raw_local_bottom_boundary_y": raw_bottom,
                "refined_top_boundary_y": float(refined_top_y),
                "refined_bottom_boundary_y": float(refined_bottom_y),
                "top_contact_line_gap_px": top_gap,
                "bottom_contact_line_gap_px": bottom_gap,
                "largest_contact_line_gap_px": max_gap,
                "within_gap_threshold": bool(max_gap <= max_gap_px),
                "gap_exceeds_threshold": bool(max_gap > max_gap_px),
            }
        )
        rows.append(row)
    return rows
