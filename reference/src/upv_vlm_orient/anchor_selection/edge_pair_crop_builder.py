import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from upv_vlm_orient.anchor_selection.result_models import EdgePairCrop


def _apply_affine_to_point(point_xy: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    point = np.array([point_xy[0], point_xy[1], 1.0], dtype=float)
    transformed = matrix @ point
    return (float(transformed[0]), float(transformed[1]))


def map_candidate_points_to_rotated_crop_frame(
    anchor_xy: tuple[float, float],
    left_hit_xy: tuple[float, float] | None,
    right_hit_xy: tuple[float, float] | None,
    rotation_matrix: np.ndarray,
    rotated_object_crop_box: tuple[int, int, int, int],
) -> dict:
    """
    Map anchor and opposite-edge hits into the cropped rotated-object frame.
    Top/bottom labels are derived from y-order in the canonical rotated frame.
    """
    crop_x_min, crop_y_min, _, _ = rotated_object_crop_box
    anchor_xy_rotated_full = _apply_affine_to_point(anchor_xy, rotation_matrix)
    anchor_xy_rotated = (
        anchor_xy_rotated_full[0] - crop_x_min,
        anchor_xy_rotated_full[1] - crop_y_min,
    )

    transformed_hits = []
    for hit_xy in [left_hit_xy, right_hit_xy]:
        if hit_xy is None:
            continue
        transformed_xy = _apply_affine_to_point(hit_xy, rotation_matrix)
        transformed_hits.append(
            (
                transformed_xy[0] - crop_x_min,
                transformed_xy[1] - crop_y_min,
            )
        )

    if len(transformed_hits) == 2:
        sorted_hits = sorted(transformed_hits, key=lambda point_xy: point_xy[1])
        top_hit_xy_rotated = sorted_hits[0]
        bottom_hit_xy_rotated = sorted_hits[1]
    else:
        top_hit_xy_rotated = None
        bottom_hit_xy_rotated = None

    return {
        "anchor_xy_rotated": anchor_xy_rotated,
        "top_hit_xy_rotated": top_hit_xy_rotated,
        "bottom_hit_xy_rotated": bottom_hit_xy_rotated,
    }


def build_edge_pair_crop(
    candidate_id: str,
    rotated_object_image_rgb: np.ndarray,
    rotated_object_overlay_rgb: np.ndarray,
    anchor_xy_rotated: tuple[float, float],
    top_hit_xy_rotated: tuple[float, float] | None,
    bottom_hit_xy_rotated: tuple[float, float] | None,
    pair_crop_half_width_px: int,
    outside_margin_px: int,
    inside_margin_px: int,
) -> tuple[EdgePairCrop, np.ndarray | None]:
    """
    Build one local edge-pair crop from the canonical rotated object frame.
    """
    image_height, image_width = rotated_object_image_rgb.shape[:2]

    if top_hit_xy_rotated is None or bottom_hit_xy_rotated is None:
        return (
            EdgePairCrop(
                candidate_id=candidate_id,
                anchor_xy_rotated=anchor_xy_rotated,
                top_hit_xy_rotated=top_hit_xy_rotated,
                bottom_hit_xy_rotated=bottom_hit_xy_rotated,
                crop_x_min=0,
                crop_y_min=0,
                crop_x_max=0,
                crop_y_max=0,
                crop_width_px=0,
                crop_height_px=0,
                image_path=None,
                valid=False,
                note="Missing top or bottom rotated hit point.",
            ),
            None,
        )

    anchor_x, anchor_y = anchor_xy_rotated
    top_y = top_hit_xy_rotated[1]
    bottom_y = bottom_hit_xy_rotated[1]

    crop_x_min = max(0, int(math.floor(anchor_x - pair_crop_half_width_px)))
    crop_x_max = min(image_width, int(math.ceil(anchor_x + pair_crop_half_width_px)))

    crop_y_min = max(
        0,
        int(math.floor(min(top_y - outside_margin_px, anchor_y - inside_margin_px))),
    )
    crop_y_max = min(
        image_height,
        int(math.ceil(max(bottom_y + outside_margin_px, anchor_y + inside_margin_px))),
    )

    crop_width_px = crop_x_max - crop_x_min
    crop_height_px = crop_y_max - crop_y_min

    valid = bool(crop_width_px > 0 and crop_height_px > 0)
    edge_pair_crop = EdgePairCrop(
        candidate_id=candidate_id,
        anchor_xy_rotated=anchor_xy_rotated,
        top_hit_xy_rotated=top_hit_xy_rotated,
        bottom_hit_xy_rotated=bottom_hit_xy_rotated,
        crop_x_min=crop_x_min,
        crop_y_min=crop_y_min,
        crop_x_max=crop_x_max,
        crop_y_max=crop_y_max,
        crop_width_px=crop_width_px,
        crop_height_px=crop_height_px,
        image_path=None,
        valid=valid,
        note=None if valid else "Degenerate crop bounds.",
    )

    if not valid:
        return edge_pair_crop, None

    crop_rgb = rotated_object_image_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
    overlay_rgb = rotated_object_overlay_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()

    anchor_xy_local = (anchor_x - crop_x_min, anchor_y - crop_y_min)
    top_hit_xy_local = (top_hit_xy_rotated[0] - crop_x_min, top_hit_xy_rotated[1] - crop_y_min)
    bottom_hit_xy_local = (
        bottom_hit_xy_rotated[0] - crop_x_min,
        bottom_hit_xy_rotated[1] - crop_y_min,
    )

    cv2.line(
        overlay_rgb,
        (int(round(top_hit_xy_local[0])), int(round(top_hit_xy_local[1]))),
        (int(round(bottom_hit_xy_local[0])), int(round(bottom_hit_xy_local[1]))),
        (255, 220, 40),
        4,
    )
    cv2.circle(
        overlay_rgb,
        (int(round(top_hit_xy_local[0])), int(round(top_hit_xy_local[1]))),
        6,
        (255, 70, 70),
        -1,
    )
    cv2.circle(
        overlay_rgb,
        (int(round(bottom_hit_xy_local[0])), int(round(bottom_hit_xy_local[1]))),
        6,
        (50, 200, 255),
        -1,
    )
    cv2.circle(
        overlay_rgb,
        (int(round(anchor_xy_local[0])), int(round(anchor_xy_local[1]))),
        7,
        (255, 110, 110),
        -1,
    )
    cv2.putText(
        overlay_rgb,
        candidate_id,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return edge_pair_crop, overlay_rgb


def build_edge_pair_crop_grid(
    candidate_items: list[dict],
    output_path: str,
    columns: int = 3,
    padding_px: int = 16,
) -> str:
    """
    Build one combined grid image from the saved individual edge-pair crops.
    """
    image_paths = [item["image_path"] for item in candidate_items if item.get("image_path")]
    if not image_paths:
        raise ValueError("No edge-pair crop image paths were provided.")

    tiles = [Image.open(path).convert("RGB") for path in image_paths]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    column_count = min(columns, len(tiles))
    row_count = (len(tiles) + column_count - 1) // column_count

    canvas = Image.new(
        "RGB",
        (
            column_count * tile_width + (column_count + 1) * padding_px,
            row_count * tile_height + (row_count + 1) * padding_px,
        ),
        (20, 20, 22),
    )
    draw = ImageDraw.Draw(canvas)

    for idx, (tile, item) in enumerate(zip(tiles, candidate_items)):
        if item.get("image_path") is None:
            continue
        row_idx = idx // column_count
        col_idx = idx % column_count
        x0 = padding_px + col_idx * (tile_width + padding_px)
        y0 = padding_px + row_idx * (tile_height + padding_px)
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 10, y0 + 8), item["candidate_id"], fill=(255, 255, 255))

    canvas.save(output_path)
    for tile in tiles:
        tile.close()
    return output_path
