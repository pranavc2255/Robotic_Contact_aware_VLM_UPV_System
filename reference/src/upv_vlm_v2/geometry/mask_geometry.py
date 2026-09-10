"""Mask geometry and global dimension estimates for v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_mask(mask_path: str | Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
    return str(path)


def _foreground_points(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask > 0)
    if len(rows) == 0:
        raise ValueError("Mask has no foreground pixels.")
    return np.column_stack((cols.astype(float), rows.astype(float)))


def normalize_vector(vector: list[float] | tuple[float, float] | np.ndarray) -> list[float]:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return [float(arr[0] / norm), float(arr[1] / norm)]


def canonicalize_angle_deg(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def _largest_external_contour(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask has no foreground contour.")
    return max(contours, key=cv2.contourArea)


def _axis_vector_from_angle(angle_deg: float) -> list[float]:
    theta = math.radians(canonicalize_angle_deg(angle_deg))
    return normalize_vector([math.cos(theta), math.sin(theta)])


def _long_and_short_rect_axes(box_points: np.ndarray) -> tuple[float, float, float, float]:
    edges: list[tuple[float, float]] = []
    for idx in range(4):
        p0 = box_points[idx]
        p1 = box_points[(idx + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length <= 1e-9:
            continue
        angle = canonicalize_angle_deg(float(math.degrees(math.atan2(edge[1], edge[0]))))
        edges.append((length, angle))
    if len(edges) < 2:
        raise ValueError("Minimum-area rectangle is degenerate.")

    unique_lengths = sorted((item[0] for item in edges), reverse=True)
    major_length = float(unique_lengths[0])
    minor_length = float(unique_lengths[-1])
    major_angle = max(edges, key=lambda item: item[0])[1]
    minor_angle = canonicalize_angle_deg(major_angle + 90.0)
    return major_angle, minor_angle, major_length, minor_length


def compute_pixel_geometry(mask: np.ndarray) -> dict[str, Any]:
    points = _foreground_points(mask)
    contour = _largest_external_contour(mask)
    rect = cv2.minAreaRect(contour)
    center_xy = rect[0]
    box = cv2.boxPoints(rect).astype(float)
    major_angle, minor_angle, major_span, minor_span = _long_and_short_rect_axes(box)
    major = _axis_vector_from_angle(major_angle)
    minor = _axis_vector_from_angle(minor_angle)
    return {
        "centroid_px": [float(center_xy[0]), float(center_xy[1])],
        "major_axis_vector": major,
        "minor_axis_vector": minor,
        "major_axis_angle_deg": major_angle,
        "minor_axis_angle_deg": minor_angle,
        "major_axis_length_px": major_span,
        "minor_axis_length_px": minor_span,
        "foreground_pixel_count": int(points.shape[0]),
    }


def _robust_edge_value(values: np.ndarray, *, trim_fraction: float, mad_multiplier: float, side: str) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite edge values.")
    if vals.size >= 6 and trim_fraction > 0:
        lo = float(np.quantile(vals, trim_fraction))
        hi = float(np.quantile(vals, 1.0 - trim_fraction))
        vals = vals[(vals >= lo) & (vals <= hi)]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if vals.size >= 6 and mad > 1e-9:
        keep = np.abs(vals - med) <= mad_multiplier * 1.4826 * mad
        if bool(keep.any()):
            vals = vals[keep]
    if side in {"top", "left"}:
        return float(np.median(vals))
    return float(np.median(vals))


def estimate_dominant_rectangle_geometry(
    mask: np.ndarray,
    raw_geometry: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    min_coverage = int(cfg.get("dominant_rectangle_min_column_coverage_px", 6))
    trim = float(cfg.get("dominant_rectangle_trim_fraction", 0.10))
    mad_mult = float(cfg.get("dominant_rectangle_mad_multiplier", 2.5))
    center = raw_geometry.get("centroid_px")
    axis = raw_geometry.get("major_axis_vector")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        raise ValueError("raw geometry centroid missing")
    if not isinstance(axis, (list, tuple)) or len(axis) < 2:
        raise ValueError("raw geometry major axis missing")
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotated_mask = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255,
        matrix,
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    top_edges: list[float] = []
    bottom_edges: list[float] = []
    for x in range(rotated_mask.shape[1]):
        ys = np.where(rotated_mask[:, x])[0]
        if ys.size >= min_coverage:
            top_edges.append(float(ys.min()))
            bottom_edges.append(float(ys.max()))
    left_edges: list[float] = []
    right_edges: list[float] = []
    for y in range(rotated_mask.shape[0]):
        xs = np.where(rotated_mask[y, :])[0]
        if xs.size >= min_coverage:
            left_edges.append(float(xs.min()))
            right_edges.append(float(xs.max()))
    if not top_edges or not bottom_edges or not left_edges or not right_edges:
        raise ValueError("dominant rectangle has insufficient supported rows or columns")

    top_y = _robust_edge_value(np.asarray(top_edges), trim_fraction=trim, mad_multiplier=mad_mult, side="top")
    bottom_y = _robust_edge_value(np.asarray(bottom_edges), trim_fraction=trim, mad_multiplier=mad_mult, side="bottom")
    left_x = _robust_edge_value(np.asarray(left_edges), trim_fraction=trim, mad_multiplier=mad_mult, side="left")
    right_x = _robust_edge_value(np.asarray(right_edges), trim_fraction=trim, mad_multiplier=mad_mult, side="right")
    if right_x <= left_x or bottom_y <= top_y:
        raise ValueError("dominant rectangle edges are degenerate")

    inv = cv2.invertAffineTransform(matrix)
    corners_rot = np.asarray(
        [[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y]],
        dtype=float,
    )
    ones = np.ones((4, 1), dtype=float)
    corners = (inv @ np.hstack([corners_rot, ones]).T).T
    center_rot = np.asarray([(left_x + right_x) / 2.0, (top_y + bottom_y) / 2.0, 1.0], dtype=float)
    center_px = inv @ center_rot
    raw_rect = cv2.minAreaRect(_largest_external_contour(mask))
    return {
        "success": True,
        "body_geometry_mode": "dominant_rectangle",
        "edge_method": str(cfg.get("dominant_rectangle_edge_method", "robust_median")),
        "body_rect_center_px": [float(center_px[0]), float(center_px[1])],
        "body_rect_corners_px": [[float(x), float(y)] for x, y in corners.tolist()],
        "body_rect_major_axis_angle_deg": canonicalize_angle_deg(angle),
        "body_rect_minor_axis_angle_deg": canonicalize_angle_deg(angle + 90.0),
        "body_rect_major_axis_length_px": float(right_x - left_x),
        "body_rect_minor_axis_length_px": float(bottom_y - top_y),
        "dominant_top_y_rotated": top_y,
        "dominant_bottom_y_rotated": bottom_y,
        "dominant_left_x_rotated": left_x,
        "dominant_right_x_rotated": right_x,
        "raw_min_area_rect_center_px": [float(raw_rect[0][0]), float(raw_rect[0][1])],
        "raw_foreground_centroid_px": raw_geometry.get("centroid_px"),
        "supported_column_count": len(top_edges),
        "supported_row_count": len(left_edges),
        "fallback_used": False,
    }


def load_depth_array(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    depth_path = Path(path)
    if not depth_path.exists():
        return None
    if depth_path.suffix.lower() == ".npy":
        return np.load(depth_path)
    return np.array(Image.open(depth_path))


def load_camera_intrinsics(path: str | Path | None) -> dict[str, float] | None:
    if path is None or not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "fx" in payload and "fy" in payload:
        return {key: float(payload[key]) for key in ("fx", "fy", "cx", "cy") if key in payload}
    if "k" in payload:
        k = payload["k"]
        return {"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])}
    if "K" in payload:
        k = payload["K"]
        return {"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])}
    intr = payload.get("intrinsics") if isinstance(payload, dict) else None
    if isinstance(intr, dict) and "fx" in intr and "fy" in intr:
        return {key: float(intr[key]) for key in ("fx", "fy", "cx", "cy") if key in intr}
    return None


def representative_depth_m(mask: np.ndarray, depth: np.ndarray | None) -> float | None:
    if depth is None:
        return None
    d = depth.astype("float32")
    values = d[(mask > 0) & np.isfinite(d) & (d > 0)]
    if values.size == 0:
        return None
    median = float(np.median(values))
    # RealSense uint16 depths are usually in millimetres.
    return median / 1000.0 if median > 20.0 else median


def pixel_span_to_mm(span_px: float, axis_vector: list[float], depth_m: float | None, intrinsics: dict[str, float] | None) -> float | None:
    if depth_m is None or intrinsics is None:
        return None
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    if fx <= 0 or fy <= 0:
        return None
    ax, ay = normalize_vector(axis_vector)
    meters_per_px = depth_m * math.sqrt((ax / fx) ** 2 + (ay / fy) ** 2)
    return float(span_px * meters_per_px * 1000.0)


def compute_mask_geometry(
    mask_path: str | Path,
    *,
    depth_path: str | Path | None = None,
    camera_info_path: str | Path | None = None,
) -> dict[str, Any]:
    mask = load_mask(mask_path)
    geom = compute_pixel_geometry(mask)
    depth = load_depth_array(depth_path)
    intrinsics = load_camera_intrinsics(camera_info_path)
    z_m = representative_depth_m(mask, depth)
    geom["representative_depth_m"] = z_m
    geom["camera_intrinsics"] = intrinsics
    geom["global_major_dimension_mm"] = pixel_span_to_mm(
        geom["major_axis_length_px"], geom["major_axis_vector"], z_m, intrinsics
    )
    geom["global_minor_dimension_mm"] = pixel_span_to_mm(
        geom["minor_axis_length_px"], geom["minor_axis_vector"], z_m, intrinsics
    )
    geom["body_geometry_mode"] = "raw_mask_existing"
    return geom


def add_dominant_rectangle_geometry(
    geometry: dict[str, Any],
    mask_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (config or {}).get("geometry", config or {})
    enabled = bool(cfg.get("dominant_rectangle_enabled", False)) or str(cfg.get("body_geometry_mode", "raw_mask_existing")) == "dominant_rectangle"
    geometry["body_geometry_mode"] = "dominant_rectangle" if enabled else "raw_mask_existing"
    if not enabled:
        geometry["dominant_rectangle"] = {"success": False, "fallback_used": False, "skipped_reason": "disabled"}
        return geometry
    try:
        rect = estimate_dominant_rectangle_geometry(load_mask(mask_path), geometry, cfg)
        geometry["dominant_rectangle"] = rect
    except Exception as exc:  # noqa: BLE001
        geometry["dominant_rectangle"] = {
            "success": False,
            "fallback_used": True,
            "failure_reason": str(exc),
            "body_geometry_mode": "raw_mask_existing",
        }
        geometry["body_geometry_mode"] = "raw_mask_existing"
        geometry["dominant_rectangle_failure_reason"] = str(exc)
    return geometry


def save_dominant_rectangle_overlay(
    *,
    rgb_path: str | Path,
    geometry: dict[str, Any],
    output_path: str | Path,
) -> str:
    image = np.array(Image.open(rgb_path).convert("RGB"))
    overlay = image.copy()
    rect = geometry.get("dominant_rectangle") or {}
    corners = rect.get("body_rect_corners_px") if isinstance(rect, dict) else None
    if isinstance(corners, list) and len(corners) >= 4:
        pts = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 220, 40), thickness=4)
        center = rect.get("body_rect_center_px")
        if isinstance(center, list) and len(center) >= 2:
            cv2.circle(overlay, (int(round(center[0])), int(round(center[1]))), 7, (255, 60, 60), -1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out)
    return str(out)
