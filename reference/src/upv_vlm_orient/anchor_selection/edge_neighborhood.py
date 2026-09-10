import math

import cv2
import numpy as np

from upv_vlm_orient.anchor_selection.result_models import BoundaryHit, LocalEdgeSupport


def extract_ordered_mask_contour(mask: np.ndarray) -> list[tuple[float, float]]:
    """
    Extract the largest external contour from a cleaned binary mask as an ordered
    sequence of boundary points.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array.")

    contour_result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    if not contours:
        raise ValueError("No contour could be extracted from the mask.")

    contour = max(contours, key=cv2.contourArea)
    if contour.ndim != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
        raise ValueError("Unexpected contour shape returned by OpenCV.")

    return [(float(point[0][0]), float(point[0][1])) for point in contour]


def _nearest_contour_index(
    contour_points_xy: list[tuple[float, float]],
    hit_xy: tuple[float, float],
) -> int:
    if not contour_points_xy:
        raise ValueError("contour_points_xy must not be empty.")

    hit_x, hit_y = hit_xy
    best_index = 0
    best_distance_sq = math.inf

    for idx, (point_x, point_y) in enumerate(contour_points_xy):
        distance_sq = (point_x - hit_x) ** 2 + (point_y - hit_y) ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_index = idx

    return best_index


def _cyclic_window_indices(center_index: int, half_window: int, contour_size: int) -> list[int]:
    if contour_size <= 0:
        return []

    return [((center_index + offset) % contour_size) for offset in range(-half_window, half_window + 1)]


def extract_local_edge_support(
    contour_points_xy: list[tuple[float, float]],
    boundary_hit: BoundaryHit,
    support_half_window: int,
) -> LocalEdgeSupport:
    """
    Find the contour point nearest to the boundary hit and return a local ordered
    neighborhood around that location.
    """
    if boundary_hit.hit_xy is None:
        return LocalEdgeSupport(
            side=boundary_hit.side,
            hit_xy=(math.nan, math.nan),
            support_points_xy=[],
            support_size=0,
            valid=False,
            note="Boundary hit has no coordinates.",
        )

    if support_half_window <= 0:
        raise ValueError("support_half_window must be positive.")

    if not contour_points_xy:
        return LocalEdgeSupport(
            side=boundary_hit.side,
            hit_xy=boundary_hit.hit_xy,
            support_points_xy=[],
            support_size=0,
            valid=False,
            note="Contour is empty.",
        )

    nearest_index = _nearest_contour_index(contour_points_xy, boundary_hit.hit_xy)
    support_indices = _cyclic_window_indices(
        center_index=nearest_index,
        half_window=min(support_half_window, max(len(contour_points_xy) // 2, 1)),
        contour_size=len(contour_points_xy),
    )
    support_points_xy = [contour_points_xy[idx] for idx in support_indices]

    return LocalEdgeSupport(
        side=boundary_hit.side,
        hit_xy=boundary_hit.hit_xy,
        support_points_xy=support_points_xy,
        support_size=len(support_points_xy),
        valid=boundary_hit.valid and len(support_points_xy) > 0,
        note=None if boundary_hit.valid else "Boundary hit was invalid.",
    )


def compute_candidate_local_edge_supports(
    contour_points_xy: list[tuple[float, float]],
    left_hit: BoundaryHit,
    right_hit: BoundaryHit,
    support_half_window: int,
) -> tuple[LocalEdgeSupport, LocalEdgeSupport]:
    """
    Extract local ordered contour neighborhoods around the left and right
    boundary-hit points for one anchor candidate.
    """
    left_support = extract_local_edge_support(
        contour_points_xy=contour_points_xy,
        boundary_hit=left_hit,
        support_half_window=support_half_window,
    )
    right_support = extract_local_edge_support(
        contour_points_xy=contour_points_xy,
        boundary_hit=right_hit,
        support_half_window=support_half_window,
    )
    return left_support, right_support
