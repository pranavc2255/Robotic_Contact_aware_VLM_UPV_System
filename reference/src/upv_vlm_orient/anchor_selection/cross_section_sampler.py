import math

import numpy as np

from upv_vlm_orient.anchor_selection.axis_parameterization import map_axis_scalar_to_point
from upv_vlm_orient.anchor_selection.result_models import AxisSearchDomain, BoundaryHit


def _normalize_xy(vector_xy: tuple[float, float]) -> tuple[float, float]:
    x, y = vector_xy
    norm = math.hypot(x, y)

    if norm <= 0.0:
        raise ValueError("Perpendicular direction must have non-zero length.")

    return (x / norm, y / norm)


def _point_to_mask_indices(point_xy: tuple[float, float]) -> tuple[int, int]:
    x, y = point_xy
    return int(round(x)), int(round(y))


def _is_in_mask(mask: np.ndarray, point_xy: tuple[float, float]) -> bool:
    x_idx, y_idx = _point_to_mask_indices(point_xy)

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
    last_in_mask_xy: tuple[float, float] | None = None
    last_distance: float | None = None

    distance = 0.0
    while distance <= max_search_distance + 1e-9:
        point_xy = (
            anchor_xy[0] + direction_xy[0] * distance,
            anchor_xy[1] + direction_xy[1] * distance,
        )

        if _is_in_mask(mask, point_xy):
            last_in_mask_xy = point_xy
            last_distance = distance
            distance += step_size_px
            continue

        break

    return BoundaryHit(
        side=side,
        hit_xy=last_in_mask_xy,
        distance_from_anchor=last_distance,
        valid=last_in_mask_xy is not None and last_distance is not None and last_distance > 0.0,
    )


def sample_cross_section_boundary_hits(
    mask: np.ndarray,
    anchor_xy: tuple[float, float],
    perpendicular_unit_direction_xy: tuple[float, float],
    max_search_distance: float,
    step_size_px: float,
) -> tuple[BoundaryHit, BoundaryHit]:
    """
    Search outward from an anchor along +/- perpendicular directions and return
    the last in-mask point reached on each side before leaving the mask.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array.")

    if max_search_distance <= 0.0:
        raise ValueError("max_search_distance must be positive.")

    if step_size_px <= 0.0:
        raise ValueError("step_size_px must be positive.")

    direction_xy = _normalize_xy(perpendicular_unit_direction_xy)
    right_hit = _search_one_side(
        mask=mask,
        anchor_xy=anchor_xy,
        direction_xy=direction_xy,
        max_search_distance=max_search_distance,
        step_size_px=step_size_px,
        side="right",
    )
    left_hit = _search_one_side(
        mask=mask,
        anchor_xy=anchor_xy,
        direction_xy=(-direction_xy[0], -direction_xy[1]),
        max_search_distance=max_search_distance,
        step_size_px=step_size_px,
        side="left",
    )
    return left_hit, right_hit


def compute_anchor_cross_section_hits(
    axis_search_domain: AxisSearchDomain,
    candidate_s: float,
    mask: np.ndarray,
    max_search_distance: float,
    step_size_px: float,
) -> tuple[tuple[float, float], BoundaryHit, BoundaryHit]:
    """
    Map one scalar candidate on the axis to an anchor point and sample the
    cross-section boundary hits on both perpendicular sides.
    """
    anchor_xy = map_axis_scalar_to_point(
        axis_center_xy=axis_search_domain.axis_center_xy,
        axis_dir_xy=axis_search_domain.axis_dir_xy,
        s=candidate_s,
    )
    left_hit, right_hit = sample_cross_section_boundary_hits(
        mask=mask,
        anchor_xy=anchor_xy,
        perpendicular_unit_direction_xy=axis_search_domain.axis_perp_xy,
        max_search_distance=max_search_distance,
        step_size_px=step_size_px,
    )
    return anchor_xy, left_hit, right_hit
