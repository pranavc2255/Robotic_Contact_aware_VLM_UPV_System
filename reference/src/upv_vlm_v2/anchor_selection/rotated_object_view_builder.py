"""Canonical rotated object view utilities for v2 anchor crops."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _normalize_xy(vector_xy: list[float] | tuple[float, float]) -> tuple[float, float]:
    x, y = float(vector_xy[0]), float(vector_xy[1])
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError("Axis direction must have non-zero length.")
    return (x / norm, y / norm)


def _apply_affine_to_point(point_xy: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    point = np.array([point_xy[0], point_xy[1], 1.0], dtype=float)
    transformed = matrix @ point
    return (float(transformed[0]), float(transformed[1]))


def _mask_outline_points(mask: np.ndarray) -> list[tuple[int, int]]:
    foreground = mask > 0
    up = np.roll(foreground, -1, axis=0)
    down = np.roll(foreground, 1, axis=0)
    left = np.roll(foreground, -1, axis=1)
    right = np.roll(foreground, 1, axis=1)
    up[-1, :] = False
    down[0, :] = False
    left[:, -1] = False
    right[:, 0] = False
    boundary = foreground & ~(up & down & left & right)
    ys, xs = np.nonzero(boundary)
    return [(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())]


def rotate_image_and_mask_to_canonical(
    *,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    axis_center_xy: list[float] | tuple[float, float],
    axis_dir_xy: list[float] | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rotate image and mask so the selected axis is horizontal."""
    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be HxWxC")
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    axis_dir = _normalize_xy(axis_dir_xy)
    rotation_angle_deg = math.degrees(math.atan2(axis_dir[1], axis_dir[0]))
    matrix = cv2.getRotationMatrix2D(
        center=(float(axis_center_xy[0]), float(axis_center_xy[1])),
        angle=rotation_angle_deg,
        scale=1.0,
    )
    rotated_image = cv2.warpAffine(
        image_rgb,
        matrix,
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255,
        matrix,
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, ((rotated_mask > 0).astype(np.uint8) * 255), matrix, float(rotation_angle_deg)


def render_rotated_object_overlay(
    *,
    rotated_image_rgb: np.ndarray,
    rotated_mask: np.ndarray,
    rotation_matrix: np.ndarray,
    geometry: dict[str, Any],
    selected_axis_vector: list[float],
    candidate_points: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Render selected mask, selected axis, and candidate contact lines in rotated frame."""
    overlay = rotated_image_rgb.copy()
    alpha = ((rotated_mask > 0).astype(np.float32) * 0.28)[..., None]
    tint = np.zeros_like(rotated_image_rgb, dtype=np.uint8)
    tint[:, :] = np.array([61, 220, 151], dtype=np.uint8)
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)
    for x, y in _mask_outline_points(rotated_mask):
        cv2.circle(overlay, (x, y), 0, (40, 255, 140), 1)

    center = geometry.get("center_px") or geometry.get("centroid_px")
    if center:
        axis = _normalize_xy(selected_axis_vector)
        extent = float(geometry.get("major_axis_length_px") or geometry.get("minor_axis_length_px") or max(rotated_mask.shape))
        start = (float(center[0]) - axis[0] * extent / 2.0, float(center[1]) - axis[1] * extent / 2.0)
        end = (float(center[0]) + axis[0] * extent / 2.0, float(center[1]) + axis[1] * extent / 2.0)
        rs = _apply_affine_to_point(start, rotation_matrix)
        re = _apply_affine_to_point(end, rotation_matrix)
        rc = _apply_affine_to_point((float(center[0]), float(center[1])), rotation_matrix)
        cv2.line(overlay, (int(round(rs[0])), int(round(rs[1]))), (int(round(re[0])), int(round(re[1]))), (0, 110, 255), 5)
        cv2.circle(overlay, (int(round(rc[0])), int(round(rc[1]))), 6, (255, 60, 60), -1)

    for idx, candidate in enumerate(candidate_points or [], start=1):
        anchor_xy = _apply_affine_to_point(tuple(candidate["anchor_xy"]), rotation_matrix)
        cv2.circle(overlay, (int(round(anchor_xy[0])), int(round(anchor_xy[1]))), 6, (255, 110, 110), -1)
        cv2.putText(overlay, candidate.get("anchor_id", f"A{idx}"), (int(anchor_xy[0] + 8), int(anchor_xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        left = candidate.get("left_hit_xy")
        right = candidate.get("right_hit_xy")
        if left is not None and right is not None:
            rl = _apply_affine_to_point(tuple(left), rotation_matrix)
            rr = _apply_affine_to_point(tuple(right), rotation_matrix)
            cv2.line(overlay, (int(round(rl[0])), int(round(rl[1]))), (int(round(rr[0])), int(round(rr[1]))), (255, 220, 40), 3)
            cv2.circle(overlay, (int(round(rl[0])), int(round(rl[1]))), 5, (255, 70, 70), -1)
            cv2.circle(overlay, (int(round(rr[0])), int(round(rr[1]))), 5, (50, 200, 255), -1)
    return overlay


def _expanded_mask_bbox(mask: np.ndarray, margin_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Rotated mask is empty.")
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w, h = x_max - x_min + 1, y_max - y_min + 1
    mx, my = int(math.ceil(w * margin_ratio)), int(math.ceil(h * margin_ratio))
    return max(0, x_min - mx), max(0, y_min - my), min(mask.shape[1], x_max + mx + 1), min(mask.shape[0], y_max + my + 1)


def crop_rotated_object_view(
    *,
    rotated_image_rgb: np.ndarray,
    rotated_mask: np.ndarray,
    rotated_overlay_rgb: np.ndarray,
    margin_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = _expanded_mask_bbox(rotated_mask, margin_ratio)
    return (
        rotated_image_rgb[y0:y1, x0:x1].copy(),
        rotated_mask[y0:y1, x0:x1].copy(),
        rotated_overlay_rgb[y0:y1, x0:x1].copy(),
        (x0, y0, x1, y1),
    )
