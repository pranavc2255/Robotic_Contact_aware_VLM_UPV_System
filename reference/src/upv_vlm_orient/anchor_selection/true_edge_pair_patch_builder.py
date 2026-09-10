from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from upv_vlm_orient.anchor_selection.result_models import TrueEdgePairPatch


def _clip_bounds(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x_min),
        max(0, y_min),
        min(image_width, x_max),
        min(image_height, y_max),
    )


def _extract_patch_with_marker(
    base_overlay_rgb: np.ndarray,
    candidate_id: str,
    hit_xy_rotated: tuple[float, float] | None,
    anchor_xy_rotated: tuple[float, float],
    patch_half_width_px: int,
    inside_margin_px: int,
    outside_margin_px: int,
    patch_kind: str,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, str | None]:
    image_height, image_width = base_overlay_rgb.shape[:2]

    if hit_xy_rotated is None:
        return None, None, f"Missing {patch_kind} hit point."

    hit_x, hit_y = hit_xy_rotated
    anchor_x, _anchor_y = anchor_xy_rotated
    x_min = int(round(anchor_x - patch_half_width_px))
    x_max = int(round(anchor_x + patch_half_width_px))

    if patch_kind == "top":
        y_min = int(round(hit_y - outside_margin_px))
        y_max = int(round(hit_y + inside_margin_px))
        hit_color = (255, 70, 70)
    elif patch_kind == "bottom":
        y_min = int(round(hit_y - inside_margin_px))
        y_max = int(round(hit_y + outside_margin_px))
        hit_color = (50, 200, 255)
    else:
        raise ValueError(f"Unsupported patch kind: {patch_kind}")

    x_min, y_min, x_max, y_max = _clip_bounds(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        image_width=image_width,
        image_height=image_height,
    )

    if x_max <= x_min or y_max <= y_min:
        return None, None, f"Degenerate {patch_kind} patch bounds."

    patch_rgb = base_overlay_rgb[y_min:y_max, x_min:x_max].copy()
    local_hit_xy = (hit_x - x_min, hit_y - y_min)

    cv2.circle(
        patch_rgb,
        (int(round(local_hit_xy[0])), int(round(local_hit_xy[1]))),
        6,
        hit_color,
        -1,
    )
    cv2.line(
        patch_rgb,
        (int(round(local_hit_xy[0] - 12)), int(round(local_hit_xy[1]))),
        (int(round(local_hit_xy[0] + 12)), int(round(local_hit_xy[1]))),
        hit_color,
        2,
    )
    cv2.putText(
        patch_rgb,
        f"{candidate_id} {patch_kind}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return (x_min, y_min, x_max, y_max), patch_rgb, None


def build_true_edge_pair_patch(
    candidate_id: str,
    rotated_object_overlay_rgb: np.ndarray,
    anchor_xy_rotated: tuple[float, float],
    top_hit_xy_rotated: tuple[float, float] | None,
    bottom_hit_xy_rotated: tuple[float, float] | None,
    patch_half_width_px: int,
    edge_inside_margin_px: int,
    edge_outside_margin_px: int,
    patch_gap_px: int,
) -> tuple[TrueEdgePairPatch, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Build local top and bottom edge patches, then vertically club them into one
    combined candidate image.
    """
    top_bounds, top_patch_rgb, top_note = _extract_patch_with_marker(
        base_overlay_rgb=rotated_object_overlay_rgb,
        candidate_id=candidate_id,
        hit_xy_rotated=top_hit_xy_rotated,
        anchor_xy_rotated=anchor_xy_rotated,
        patch_half_width_px=patch_half_width_px,
        inside_margin_px=edge_inside_margin_px,
        outside_margin_px=edge_outside_margin_px,
        patch_kind="top",
    )
    bottom_bounds, bottom_patch_rgb, bottom_note = _extract_patch_with_marker(
        base_overlay_rgb=rotated_object_overlay_rgb,
        candidate_id=candidate_id,
        hit_xy_rotated=bottom_hit_xy_rotated,
        anchor_xy_rotated=anchor_xy_rotated,
        patch_half_width_px=patch_half_width_px,
        inside_margin_px=edge_inside_margin_px,
        outside_margin_px=edge_outside_margin_px,
        patch_kind="bottom",
    )

    note = top_note or bottom_note
    valid = top_patch_rgb is not None and bottom_patch_rgb is not None

    if not valid:
        return (
            TrueEdgePairPatch(
                candidate_id=candidate_id,
                anchor_xy_rotated=anchor_xy_rotated,
                top_hit_xy_rotated=top_hit_xy_rotated,
                bottom_hit_xy_rotated=bottom_hit_xy_rotated,
                top_patch_bounds=top_bounds,
                bottom_patch_bounds=bottom_bounds,
                combined_pair_image_path=None,
                valid=False,
                note=note or "Failed to build true edge pair patches.",
            ),
            top_patch_rgb,
            bottom_patch_rgb,
            None,
        )

    combined_width = max(top_patch_rgb.shape[1], bottom_patch_rgb.shape[1])
    combined_height = top_patch_rgb.shape[0] + patch_gap_px + bottom_patch_rgb.shape[0]
    combined_pair_rgb = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_pair_rgb[:, :] = np.array([20, 20, 22], dtype=np.uint8)

    top_x = (combined_width - top_patch_rgb.shape[1]) // 2
    bottom_x = (combined_width - bottom_patch_rgb.shape[1]) // 2
    combined_pair_rgb[0:top_patch_rgb.shape[0], top_x:top_x + top_patch_rgb.shape[1]] = top_patch_rgb
    bottom_y0 = top_patch_rgb.shape[0] + patch_gap_px
    combined_pair_rgb[
        bottom_y0:bottom_y0 + bottom_patch_rgb.shape[0],
        bottom_x:bottom_x + bottom_patch_rgb.shape[1],
    ] = bottom_patch_rgb

    cv2.putText(
        combined_pair_rgb,
        candidate_id,
        (8, combined_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    true_edge_pair_patch = TrueEdgePairPatch(
        candidate_id=candidate_id,
        anchor_xy_rotated=anchor_xy_rotated,
        top_hit_xy_rotated=top_hit_xy_rotated,
        bottom_hit_xy_rotated=bottom_hit_xy_rotated,
        top_patch_bounds=top_bounds,
        bottom_patch_bounds=bottom_bounds,
        combined_pair_image_path=None,
        valid=True,
        note=None,
    )
    return true_edge_pair_patch, top_patch_rgb, bottom_patch_rgb, combined_pair_rgb


def build_true_edge_pair_grid(
    candidate_items: list[dict],
    output_path: str,
    columns: int = 3,
    padding_px: int = 16,
) -> str:
    """
    Build one grid image containing all vertically clubbed candidate pair patch
    images.
    """
    valid_items = [item for item in candidate_items if item.get("combined_pair_image_path")]
    if not valid_items:
        raise ValueError("No combined true-edge pair images were provided.")

    tiles = [Image.open(item["combined_pair_image_path"]).convert("RGB") for item in valid_items]
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

    for idx, (tile, item) in enumerate(zip(tiles, valid_items)):
        row_idx = idx // column_count
        col_idx = idx % column_count
        x0 = padding_px + col_idx * (tile_width + padding_px)
        y0 = padding_px + row_idx * (tile_height + padding_px)
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 10, y0 + 8), item["candidate_id"], fill=(255, 255, 255))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    for tile in tiles:
        tile.close()
    return str(output_file)
