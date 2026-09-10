import math

import cv2
import numpy as np

from upv_vlm_orient.anchor_selection.result_models import BoundaryHit, PairObservation


def _normalize_xy(vector_xy: tuple[float, float]) -> tuple[float, float]:
    x, y = vector_xy
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError("Axis direction must have non-zero length.")
    return (x / norm, y / norm)


def _apply_affine_to_point(point_xy: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    point = np.array([point_xy[0], point_xy[1], 1.0], dtype=float)
    transformed = matrix @ point
    return (float(transformed[0]), float(transformed[1]))


def build_pair_observation(
    candidate_id: str,
    anchor_xy: tuple[float, float],
    left_hit: BoundaryHit,
    right_hit: BoundaryHit,
    axis_dir_xy: tuple[float, float],
    axis_context_half_width_px: float,
    inside_margin_px: float,
    outside_margin_px: float,
) -> PairObservation:
    """
    Define a canonical rotated crop around one opposite-edge hit pair.
    The chosen axis will be horizontal after rotation, while the contact band
    between the two hits will be vertical.
    """
    if left_hit.hit_xy is None or right_hit.hit_xy is None:
        return PairObservation(
            candidate_id=candidate_id,
            anchor_xy=anchor_xy,
            left_hit_xy=left_hit.hit_xy,
            right_hit_xy=right_hit.hit_xy,
            crop_center_xy=anchor_xy,
            rotation_angle_deg=0.0,
            crop_width_px=0,
            crop_height_px=0,
            image_path=None,
            valid=False,
            note="Missing left or right hit coordinates.",
        )

    axis_dir_xy = _normalize_xy(axis_dir_xy)
    pair_center_xy = (
        (left_hit.hit_xy[0] + right_hit.hit_xy[0]) / 2.0,
        (left_hit.hit_xy[1] + right_hit.hit_xy[1]) / 2.0,
    )
    pair_span_px = math.hypot(
        right_hit.hit_xy[0] - left_hit.hit_xy[0],
        right_hit.hit_xy[1] - left_hit.hit_xy[1],
    )
    rotation_angle_deg = math.degrees(math.atan2(axis_dir_xy[1], axis_dir_xy[0]))

    crop_width_px = int(round(max(96.0, axis_context_half_width_px * 2.0)))
    crop_height_px = int(round(max(96.0, pair_span_px + 2.0 * (inside_margin_px + outside_margin_px))))

    return PairObservation(
        candidate_id=candidate_id,
        anchor_xy=anchor_xy,
        left_hit_xy=left_hit.hit_xy,
        right_hit_xy=right_hit.hit_xy,
        crop_center_xy=pair_center_xy,
        rotation_angle_deg=float(rotation_angle_deg),
        crop_width_px=crop_width_px,
        crop_height_px=crop_height_px,
        image_path=None,
        valid=left_hit.valid and right_hit.valid,
        note=None if left_hit.valid and right_hit.valid else "One or both boundary hits are invalid.",
    )


def build_pair_observation_crop(
    image_rgb: np.ndarray,
    pair_observation: PairObservation,
) -> tuple[np.ndarray | None, dict]:
    """
    Rotate the full image around the crop center and extract a canonical local
    observation crop for the candidate pair.
    """
    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be an HxWxC array.")

    if not pair_observation.valid:
        return None, {"note": pair_observation.note}

    crop_center_x, crop_center_y = pair_observation.crop_center_xy
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(crop_center_x, crop_center_y),
        angle=pair_observation.rotation_angle_deg,
        scale=1.0,
    )
    rotated_image = cv2.warpAffine(
        image_rgb,
        rotation_matrix,
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    transformed_anchor_xy = _apply_affine_to_point(pair_observation.anchor_xy, rotation_matrix)
    transformed_left_hit_xy = _apply_affine_to_point(pair_observation.left_hit_xy, rotation_matrix)
    transformed_right_hit_xy = _apply_affine_to_point(pair_observation.right_hit_xy, rotation_matrix)

    crop_x0 = int(round(crop_center_x - pair_observation.crop_width_px / 2.0))
    crop_y0 = int(round(crop_center_y - pair_observation.crop_height_px / 2.0))
    crop_x1 = crop_x0 + pair_observation.crop_width_px
    crop_y1 = crop_y0 + pair_observation.crop_height_px

    pad_left = max(0, -crop_x0)
    pad_top = max(0, -crop_y0)
    pad_right = max(0, crop_x1 - rotated_image.shape[1])
    pad_bottom = max(0, crop_y1 - rotated_image.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        rotated_image = cv2.copyMakeBorder(
            rotated_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        crop_x0 += pad_left
        crop_x1 += pad_left
        crop_y0 += pad_top
        crop_y1 += pad_top

    crop_rgb = rotated_image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    relative_anchor_xy = (transformed_anchor_xy[0] - crop_x0, transformed_anchor_xy[1] - crop_y0)
    relative_left_hit_xy = (transformed_left_hit_xy[0] - crop_x0, transformed_left_hit_xy[1] - crop_y0)
    relative_right_hit_xy = (transformed_right_hit_xy[0] - crop_x0, transformed_right_hit_xy[1] - crop_y0)

    details = {
        "relative_anchor_xy": relative_anchor_xy,
        "relative_left_hit_xy": relative_left_hit_xy,
        "relative_right_hit_xy": relative_right_hit_xy,
    }
    return crop_rgb, details
