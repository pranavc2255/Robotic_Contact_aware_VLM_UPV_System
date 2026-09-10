"""Mask cross-section sampling for v2 contact anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from upv_vlm_v2.anchor_selection.axis_parameterization import AxisSearchDomain, map_axis_scalar_to_point


@dataclass
class BoundaryHit:
    side: str
    hit_xy: tuple[float, float] | None
    distance_from_anchor: float | None
    valid: bool


def _normalize_xy(vector_xy: tuple[float, float] | list[float]) -> tuple[float, float]:
    x, y = float(vector_xy[0]), float(vector_xy[1])
    norm = math.hypot(x, y)
    if norm <= 0:
        raise ValueError("Direction must have non-zero length.")
    return (x / norm, y / norm)


def _is_in_mask(mask: np.ndarray, point_xy: tuple[float, float]) -> bool:
    x_idx, y_idx = int(round(point_xy[0])), int(round(point_xy[1]))
    if y_idx < 0 or y_idx >= mask.shape[0] or x_idx < 0 or x_idx >= mask.shape[1]:
        return False
    return bool(mask[y_idx, x_idx] > 0)


def _search_one_side(
    mask: np.ndarray,
    anchor_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    max_search_distance: float,
    step_size_px: float,
    side: str,
) -> BoundaryHit:
    last_xy: tuple[float, float] | None = None
    last_dist: float | None = None
    distance = 0.0
    while distance <= max_search_distance + 1e-9:
        point = (anchor_xy[0] + direction_xy[0] * distance, anchor_xy[1] + direction_xy[1] * distance)
        if _is_in_mask(mask, point):
            last_xy = point
            last_dist = distance
            distance += step_size_px
            continue
        break
    return BoundaryHit(side=side, hit_xy=last_xy, distance_from_anchor=last_dist, valid=last_xy is not None and (last_dist or 0) > 0)


def sample_cross_section_boundary_hits(
    mask: np.ndarray,
    anchor_xy: tuple[float, float],
    perpendicular_unit_direction_xy: tuple[float, float] | list[float],
    max_search_distance: float,
    step_size_px: float,
) -> tuple[BoundaryHit, BoundaryHit]:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array.")
    direction = _normalize_xy(perpendicular_unit_direction_xy)
    right = _search_one_side(mask, anchor_xy, direction, max_search_distance, step_size_px, "right")
    left = _search_one_side(mask, anchor_xy, (-direction[0], -direction[1]), max_search_distance, step_size_px, "left")
    return left, right


def compute_anchor_cross_section_hits(
    axis_search_domain: AxisSearchDomain,
    candidate_s: float,
    mask: np.ndarray,
    max_search_distance: float,
    step_size_px: float,
) -> tuple[tuple[float, float], BoundaryHit, BoundaryHit]:
    anchor_xy = map_axis_scalar_to_point(axis_search_domain.axis_center_xy, axis_search_domain.axis_dir_xy, candidate_s)
    left, right = sample_cross_section_boundary_hits(
        mask=mask,
        anchor_xy=anchor_xy,
        perpendicular_unit_direction_xy=axis_search_domain.axis_perp_xy,
        max_search_distance=max_search_distance,
        step_size_px=step_size_px,
    )
    return anchor_xy, left, right
