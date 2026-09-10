"""Point mapping for v2 rotated anchor edge-pair crops."""

from __future__ import annotations

import numpy as np


def _apply_affine_to_point(point_xy: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    point = np.array([point_xy[0], point_xy[1], 1.0], dtype=float)
    transformed = matrix @ point
    return (float(transformed[0]), float(transformed[1]))


def map_candidate_points_to_rotated_crop_frame(
    *,
    anchor_xy: list[float] | tuple[float, float],
    left_hit_xy: list[float] | tuple[float, float] | None,
    right_hit_xy: list[float] | tuple[float, float] | None,
    rotation_matrix: np.ndarray,
    rotated_object_crop_box: tuple[int, int, int, int],
) -> dict:
    crop_x_min, crop_y_min, _, _ = rotated_object_crop_box
    anchor_full = _apply_affine_to_point((float(anchor_xy[0]), float(anchor_xy[1])), rotation_matrix)
    anchor_rotated = (anchor_full[0] - crop_x_min, anchor_full[1] - crop_y_min)
    transformed_hits: list[tuple[float, float]] = []
    for hit in (left_hit_xy, right_hit_xy):
        if hit is None:
            continue
        full = _apply_affine_to_point((float(hit[0]), float(hit[1])), rotation_matrix)
        transformed_hits.append((full[0] - crop_x_min, full[1] - crop_y_min))
    if len(transformed_hits) == 2:
        sorted_hits = sorted(transformed_hits, key=lambda item: item[1])
        top = sorted_hits[0]
        bottom = sorted_hits[1]
    else:
        top = None
        bottom = None
    return {
        "anchor_xy_rotated": anchor_rotated,
        "top_hit_xy_rotated": top,
        "bottom_hit_xy_rotated": bottom,
    }
