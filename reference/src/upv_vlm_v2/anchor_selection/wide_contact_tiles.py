"""V2a-style wide-contact tile and grid generation for v2 anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _crop_patch(rgb: np.ndarray, point: list[float] | tuple[float, float] | None, *, half_w: int, half_h: int) -> Image.Image | None:
    if point is None:
        return None
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    x0, x1 = max(0, x - half_w), min(rgb.shape[1], x + half_w + 1)
    y0, y1 = max(0, y - half_h), min(rgb.shape[0], y + half_h + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Image.fromarray(rgb[y0:y1, x0:x1].copy()).convert("RGB")


def _extract_patch_from_bounds(rgb: np.ndarray, bounds: tuple[int, int, int, int] | list[int] | None) -> Image.Image | None:
    if bounds is None:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in bounds]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Image.fromarray(rgb[y0:y1, x0:x1].copy()).convert("RGB")


def build_boxed_wide_contact_tile(
    *,
    candidate_id: str,
    rgb_array: np.ndarray,
    contact_point_a_px: list[float] | None,
    contact_point_b_px: list[float] | None,
    output_path: str | Path,
    half_width_px: int = 70,
    half_height_px: int = 38,
) -> dict[str, Any]:
    """Build a compact TOP/BOTTOM contact tile with labels outside texture."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    top = _crop_patch(rgb_array, contact_point_a_px, half_w=half_width_px, half_h=half_height_px)
    bottom = _crop_patch(rgb_array, contact_point_b_px, half_w=half_width_px, half_h=half_height_px)
    if top is None or bottom is None:
        return {
            "candidate_id": candidate_id,
            "tile_image_path": None,
            "valid": False,
            "note": "missing contact patch",
        }

    content_w = max(top.width, bottom.width)
    row_h = max(top.height, bottom.height)
    label_w = 92
    pad = 14
    header_h = 42
    gap = 14
    width = label_w + content_w + 2 * pad
    height = header_h + 2 * pad + row_h * 2 + gap
    canvas = Image.new("RGB", (width, height), (242, 243, 239))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, header_h - 1), fill=(31, 34, 38))
    draw.text((pad, 9), candidate_id, fill=(255, 255, 255), font=_font(22, bold=True))

    def paste_row(row: int, label: str, tile: Image.Image, guide_color: tuple[int, int, int]) -> tuple[int, int, int, int]:
        y0 = header_h + pad + row * (row_h + gap)
        draw.text((pad, y0 + row_h // 2 - 8), label, fill=(38, 40, 43), font=_font(14, bold=True))
        x = label_w + pad + (content_w - tile.width) // 2
        y = y0 + (row_h - tile.height) // 2
        draw.rectangle((x - 2, y - 2, x + tile.width + 1, y + tile.height + 1), fill=(118, 122, 126))
        canvas.paste(tile, (x, y))
        gy = y + tile.height // 2
        for gx in range(x, x + tile.width, 22):
            draw.line((gx, gy, min(x + tile.width - 1, gx + 14), gy), fill=guide_color, width=2)
        return (x, y, x + tile.width, y + tile.height)

    top_bounds = paste_row(0, "TOP", top, (255, 210, 66))
    bottom_bounds = paste_row(1, "BOTTOM", bottom, (66, 170, 255))
    sep_y = header_h + pad + row_h + gap // 2
    draw.line((pad, sep_y, width - pad, sep_y), fill=(188, 190, 184), width=1)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(74, 78, 84), width=2)
    canvas.save(out)
    return {
        "candidate_id": candidate_id,
        "tile_image_path": str(out),
        "valid": True,
        "tile_width_px": width,
        "tile_height_px": height,
        "top_content_bounds_in_tile": top_bounds,
        "bottom_content_bounds_in_tile": bottom_bounds,
        "note": None,
    }


def build_boxed_wide_contact_tile_from_bounds(
    *,
    candidate_id: str,
    rotated_object_image_rgb: np.ndarray,
    top_patch_bounds: tuple[int, int, int, int] | list[int] | None,
    bottom_patch_bounds: tuple[int, int, int, int] | list[int] | None,
    output_path: str | Path,
    top_contact_guide_y_px: float | int | None = None,
    bottom_contact_guide_y_px: float | int | None = None,
    header_height_px: int = 36,
    row_label_width_px: int = 96,
    content_padding_px: int = 10,
    row_gap_px: int = 10,
    border_px: int = 2,
    row_label_position: str = "above",
    row_label_height_px: int = 20,
    min_row_height_px: int | None = None,
) -> dict[str, Any]:
    """Build TOP/BOTTOM tile from canonical rotated-object patch bounds."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    top = _extract_patch_from_bounds(rotated_object_image_rgb, top_patch_bounds)
    bottom = _extract_patch_from_bounds(rotated_object_image_rgb, bottom_patch_bounds)
    if top is None or bottom is None:
        return {"candidate_id": candidate_id, "tile_image_path": None, "valid": False, "note": "missing rotated contact patch"}

    # Old working V2a/T4-style boxed tile layout:
    # labels live outside the image content, row labels are above patches,
    # and row height is kept large enough to preserve edge context.
    content_w = max(top.width, bottom.width)
    row_h = max(top.height, bottom.height)
    if min_row_height_px is not None:
        row_h = max(row_h, int(min_row_height_px))

    if row_label_position == "left":
        width = row_label_width_px + content_w + 2 * content_padding_px
        height = header_height_px + 2 * content_padding_px + 2 * row_h + row_gap_px
    elif row_label_position == "above":
        width = content_w + 2 * content_padding_px
        height = (
            header_height_px
            + 2 * content_padding_px
            + 2 * (row_label_height_px + row_h)
            + row_gap_px
        )
    else:
        raise ValueError(f"Unsupported row_label_position: {row_label_position}")

    canvas = Image.new("RGB", (width, height), (242, 243, 239))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, header_height_px - 1), fill=(31, 34, 38))
    draw.text((content_padding_px, 7), candidate_id, fill=(255, 255, 255), font=_font(22, bold=True))

    def draw_guide(x: int, y: int, tile: Image.Image, guide_y: float | int | None, color: tuple[int, int, int]) -> None:
        local_guide = tile.height // 2 if guide_y is None else int(round(float(guide_y)))
        local_guide = max(0, min(tile.height - 1, local_guide))
        gy = y + local_guide
        for gx in range(x, x + tile.width, 22):
            draw.line((gx, gy, min(x + tile.width - 1, gx + 14), gy), fill=color, width=2)

    def paste_left_label_row(row: int, label: str, tile: Image.Image, guide_y: float | int | None, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
        row_y0 = header_height_px + content_padding_px + row * (row_h + row_gap_px)
        draw.text((content_padding_px, row_y0 + row_h // 2 - 8), label, fill=(38, 40, 43), font=_font(14, bold=True))
        x = row_label_width_px + content_padding_px + (content_w - tile.width) // 2
        y = row_y0 + (row_h - tile.height) // 2
        draw.rectangle((x - border_px, y - border_px, x + tile.width + border_px - 1, y + tile.height + border_px - 1), fill=(118, 122, 126))
        canvas.paste(tile, (x, y))
        draw_guide(x, y, tile, guide_y, color)
        return (x, y, x + tile.width, y + tile.height)

    def paste_above_label_row(row: int, label: str, tile: Image.Image, guide_y: float | int | None, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
        row_y0 = header_height_px + content_padding_px + row * (row_label_height_px + row_h + row_gap_px)
        label_y0 = row_y0
        x = content_padding_px + (content_w - tile.width) // 2
        y = row_y0 + row_label_height_px + (row_h - tile.height) // 2
        draw.rectangle(
            (
                content_padding_px,
                label_y0,
                width - content_padding_px - 1,
                label_y0 + row_label_height_px - 1,
            ),
            fill=(229, 231, 225),
        )
        draw.text((content_padding_px + 8, label_y0 + 2), label, fill=(38, 40, 43), font=_font(14, bold=True))
        draw.rectangle((x - border_px, y - border_px, x + tile.width + border_px - 1, y + tile.height + border_px - 1), fill=(118, 122, 126))
        canvas.paste(tile, (x, y))
        draw_guide(x, y, tile, guide_y, color)
        return (x, y, x + tile.width, y + tile.height)

    if row_label_position == "left":
        top_bounds_in_tile = paste_left_label_row(0, "TOP", top, top_contact_guide_y_px, (255, 210, 66))
        bottom_bounds_in_tile = paste_left_label_row(1, "BOTTOM", bottom, bottom_contact_guide_y_px, (66, 170, 255))
        sep_y = header_height_px + content_padding_px + row_h + row_gap_px // 2
    else:
        top_bounds_in_tile = paste_above_label_row(0, "TOP", top, top_contact_guide_y_px, (255, 210, 66))
        bottom_bounds_in_tile = paste_above_label_row(1, "BOTTOM", bottom, bottom_contact_guide_y_px, (66, 170, 255))
        sep_y = header_height_px + content_padding_px + row_label_height_px + row_h + row_gap_px // 2

    draw.line((content_padding_px, sep_y, width - content_padding_px - 1, sep_y), fill=(188, 190, 184), width=1)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(74, 78, 84), width=border_px)
    canvas.save(out)
    return {
        "candidate_id": candidate_id,
        "tile_image_path": str(out),
        "valid": True,
        "tile_width_px": width,
        "tile_height_px": height,
        "top_content_bounds_in_tile": top_bounds_in_tile,
        "bottom_content_bounds_in_tile": bottom_bounds_in_tile,
        "top_patch_bounds": top_patch_bounds,
        "bottom_patch_bounds": bottom_patch_bounds,
        "tile_source": "cropped_rotated_object_view",
        "note": None,
    }


def build_boxed_candidate_tile_grid(
    candidate_items: list[dict[str, Any]],
    output_path: str | Path,
    *,
    columns: int,
    padding_px: int = 18,
) -> str:
    valid_items = [item for item in candidate_items if item.get("tile_image_path")]
    if not valid_items:
        raise ValueError("No boxed candidate tile images were provided.")
    tiles = [Image.open(item["tile_image_path"]).convert("RGB") for item in valid_items]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    cols = max(1, min(int(columns), len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * padding_px, rows * tile_h + (rows + 1) * padding_px),
        (226, 228, 222),
    )
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = padding_px + col * (tile_w + padding_px) + (tile_w - tile.width) // 2
        y = padding_px + row * (tile_h + padding_px) + (tile_h - tile.height) // 2
        canvas.paste(tile, (x, y))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    for tile in tiles:
        tile.close()
    return str(out)
