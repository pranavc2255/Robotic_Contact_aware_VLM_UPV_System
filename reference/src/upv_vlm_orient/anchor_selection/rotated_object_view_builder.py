import math

import cv2
import numpy as np

from upv_vlm_orient.anchor_selection.result_models import RotatedObjectView


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
    image_rgb: np.ndarray,
    mask: np.ndarray,
    axis_center_xy: tuple[float, float],
    axis_dir_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Rotate image and mask around the object center so the chosen axis becomes
    horizontal in image coordinates.
    """
    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be an HxWxC array.")

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary array.")

    axis_dir_xy = _normalize_xy(axis_dir_xy)
    rotation_angle_deg = math.degrees(math.atan2(axis_dir_xy[1], axis_dir_xy[0]))
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(axis_center_xy[0], axis_center_xy[1]),
        angle=rotation_angle_deg,
        scale=1.0,
    )

    rotated_image_rgb = cv2.warpAffine(
        image_rgb,
        rotation_matrix,
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = ((rotated_mask > 0).astype(np.uint8) * 255)
    return rotated_image_rgb, rotated_mask, rotation_matrix, float(rotation_angle_deg)


def _expanded_mask_bbox(mask: np.ndarray, margin_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Rotated mask is empty.")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    margin_x = int(math.ceil(width * margin_ratio))
    margin_y = int(math.ceil(height * margin_ratio))

    crop_x_min = max(0, x_min - margin_x)
    crop_y_min = max(0, y_min - margin_y)
    crop_x_max = min(mask.shape[1], x_max + margin_x + 1)
    crop_y_max = min(mask.shape[0], y_max + margin_y + 1)
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


def render_rotated_object_overlay(
    rotated_image_rgb: np.ndarray,
    rotated_mask: np.ndarray,
    rotation_matrix: np.ndarray,
    geometry: dict,
    candidate_points: list[dict] | None = None,
) -> np.ndarray:
    """
    Render overlay markers on the full rotated image while preserving the same
    transformed geometry as the rotated image and mask.
    """
    overlay = rotated_image_rgb.copy()

    mask_alpha = np.zeros_like(rotated_mask, dtype=np.uint8)
    mask_alpha[rotated_mask > 0] = 75
    tint = np.zeros_like(rotated_image_rgb, dtype=np.uint8)
    tint[:, :] = np.array([61, 220, 151], dtype=np.uint8)
    alpha = mask_alpha[..., None].astype(np.float32) / 255.0
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)

    for x, y in _mask_outline_points(rotated_mask):
        cv2.circle(overlay, (x, y), 0, (40, 255, 140), 1)

    center_xy = (float(geometry["center_px"][0]), float(geometry["center_px"][1]))
    major_vector = _normalize_xy((float(geometry["major_axis_vector"][0]), float(geometry["major_axis_vector"][1])))
    major_half = float(geometry["major_axis_length_px"]) / 2.0
    major_start = (center_xy[0] - major_vector[0] * major_half, center_xy[1] - major_vector[1] * major_half)
    major_end = (center_xy[0] + major_vector[0] * major_half, center_xy[1] + major_vector[1] * major_half)

    rotated_center_xy = _apply_affine_to_point(center_xy, rotation_matrix)
    rotated_major_start = _apply_affine_to_point(major_start, rotation_matrix)
    rotated_major_end = _apply_affine_to_point(major_end, rotation_matrix)

    cv2.line(
        overlay,
        (int(round(rotated_major_start[0])), int(round(rotated_major_start[1]))),
        (int(round(rotated_major_end[0])), int(round(rotated_major_end[1]))),
        (0, 110, 255),
        6,
    )
    cv2.circle(
        overlay,
        (int(round(rotated_center_xy[0])), int(round(rotated_center_xy[1]))),
        7,
        (255, 60, 60),
        -1,
    )

    if candidate_points:
        for idx, candidate in enumerate(candidate_points, start=1):
            anchor_xy = _apply_affine_to_point(candidate["anchor_xy"], rotation_matrix)
            cv2.circle(
                overlay,
                (int(round(anchor_xy[0])), int(round(anchor_xy[1]))),
                7,
                (255, 110, 110),
                -1,
            )
            cv2.putText(
                overlay,
                f"A{idx}",
                (int(round(anchor_xy[0] + 10)), int(round(anchor_xy[1] - 10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            left_hit_xy = candidate.get("left_hit_xy")
            right_hit_xy = candidate.get("right_hit_xy")
            if left_hit_xy is not None and right_hit_xy is not None:
                rotated_left_hit_xy = _apply_affine_to_point(left_hit_xy, rotation_matrix)
                rotated_right_hit_xy = _apply_affine_to_point(right_hit_xy, rotation_matrix)
                cv2.line(
                    overlay,
                    (int(round(rotated_left_hit_xy[0])), int(round(rotated_left_hit_xy[1]))),
                    (int(round(rotated_right_hit_xy[0])), int(round(rotated_right_hit_xy[1]))),
                    (255, 220, 40),
                    4,
                )
                cv2.circle(
                    overlay,
                    (int(round(rotated_left_hit_xy[0])), int(round(rotated_left_hit_xy[1]))),
                    6,
                    (255, 70, 70),
                    -1,
                )
                cv2.circle(
                    overlay,
                    (int(round(rotated_right_hit_xy[0])), int(round(rotated_right_hit_xy[1]))),
                    6,
                    (50, 200, 255),
                    -1,
                )

    return overlay


def crop_rotated_object_view(
    rotated_image_rgb: np.ndarray,
    rotated_mask: np.ndarray,
    rotated_overlay_rgb: np.ndarray,
    margin_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Crop the rotated view to the rotated mask bounding box enlarged by a fixed
    margin on each side.
    """
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = _expanded_mask_bbox(
        mask=rotated_mask,
        margin_ratio=margin_ratio,
    )
    cropped_image_rgb = rotated_image_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
    cropped_mask = rotated_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
    cropped_overlay_rgb = rotated_overlay_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
    return cropped_image_rgb, cropped_mask, cropped_overlay_rgb, (crop_x_min, crop_y_min, crop_x_max, crop_y_max)


def build_rotated_object_view(
    image_path: str,
    image_rgb: np.ndarray,
    cleaned_mask: np.ndarray,
    geometry: dict,
    candidate_points: list[dict] | None = None,
    margin_ratio: float = 0.10,
) -> tuple[RotatedObjectView, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the canonical rotated object view and its cropped artifacts.
    """
    rotated_image_rgb, rotated_mask, rotation_matrix, rotation_angle_deg = rotate_image_and_mask_to_canonical(
        image_rgb=image_rgb,
        mask=cleaned_mask,
        axis_center_xy=(float(geometry["center_px"][0]), float(geometry["center_px"][1])),
        axis_dir_xy=(float(geometry["major_axis_vector"][0]), float(geometry["major_axis_vector"][1])),
    )
    rotated_overlay_rgb = render_rotated_object_overlay(
        rotated_image_rgb=rotated_image_rgb,
        rotated_mask=rotated_mask,
        rotation_matrix=rotation_matrix,
        geometry=geometry,
        candidate_points=candidate_points,
    )
    cropped_image_rgb, cropped_mask, cropped_overlay_rgb, crop_box = crop_rotated_object_view(
        rotated_image_rgb=rotated_image_rgb,
        rotated_mask=rotated_mask,
        rotated_overlay_rgb=rotated_overlay_rgb,
        margin_ratio=margin_ratio,
    )
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box
    rotated_object_view = RotatedObjectView(
        image_path=image_path,
        rotated_image_path=None,
        rotated_overlay_path=None,
        rotation_angle_deg=rotation_angle_deg,
        crop_x_min=crop_x_min,
        crop_y_min=crop_y_min,
        crop_x_max=crop_x_max,
        crop_y_max=crop_y_max,
        crop_width_px=crop_x_max - crop_x_min,
        crop_height_px=crop_y_max - crop_y_min,
        valid=bool(cropped_mask.size > 0 and np.any(cropped_mask > 0)),
        note=None,
    )
    return (
        rotated_object_view,
        rotated_image_rgb,
        rotated_mask,
        rotated_overlay_rgb,
        cropped_image_rgb,
        cropped_overlay_rgb,
    )
