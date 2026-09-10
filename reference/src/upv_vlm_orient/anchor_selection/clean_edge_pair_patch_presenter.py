from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from upv_vlm_orient.anchor_selection.result_models import CleanEdgePairPatch


def _extract_patch_from_bounds(
    base_image_rgb: np.ndarray,
    bounds: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    if bounds is None:
        return None

    x_min, y_min, x_max, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        return None

    return base_image_rgb[y_min:y_max, x_min:x_max].copy()


def _add_header_band(
    patch_rgb: np.ndarray,
    header_text: str,
    header_height_px: int = 22,
) -> np.ndarray:
    patch_image = Image.fromarray(patch_rgb)
    canvas = Image.new(
        "RGB",
        (patch_image.width, patch_image.height + header_height_px),
        (20, 20, 22),
    )
    canvas.paste(patch_image, (0, header_height_px))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 3), header_text, fill=(255, 255, 255))
    return np.array(canvas)


def _load_font(size_px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size_px)
    except OSError:
        return ImageFont.load_default()


def _load_bold_font(size_px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size_px)
    except OSError:
        return _load_font(size_px)


def _center_text_y(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, y0: int, y1: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]
    return y0 + ((y1 - y0 - text_height) // 2) - bbox[1]


def _draw_contact_guide(
    patch_rgb: np.ndarray,
    guide_y_px: float | int | None,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    if guide_y_px is None:
        return patch_rgb

    guided = patch_rgb.copy()
    image = Image.fromarray(guided).convert("RGB")
    draw = ImageDraw.Draw(image)
    y = int(round(float(guide_y_px)))
    if y < 0 or y >= image.height:
        return guided

    # Short dashed segments keep the contact cue visible without covering texture.
    segment_px = 14
    gap_px = 8
    for x0 in range(0, image.width, segment_px + gap_px):
        draw.line((x0, y, min(image.width - 1, x0 + segment_px), y), fill=color_rgb, width=2)
    return np.array(image)


def build_clean_edge_pair_presentation(
    candidate_id: str,
    rotated_object_image_rgb: np.ndarray,
    top_patch_bounds: tuple[int, int, int, int] | None,
    bottom_patch_bounds: tuple[int, int, int, int] | None,
    patch_gap_px: int,
    header_height_px: int = 22,
) -> tuple[CleanEdgePairPatch, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Build clean VLM-oriented top and bottom patches with labels moved into
    separate header bands outside the edge texture content.
    """
    top_patch_rgb = _extract_patch_from_bounds(rotated_object_image_rgb, top_patch_bounds)
    bottom_patch_rgb = _extract_patch_from_bounds(rotated_object_image_rgb, bottom_patch_bounds)

    valid = top_patch_rgb is not None and bottom_patch_rgb is not None
    if not valid:
        clean_edge_pair_patch = CleanEdgePairPatch(
            candidate_id=candidate_id,
            top_patch_image_path=None,
            bottom_patch_image_path=None,
            combined_clean_pair_image_path=None,
            valid=False,
            note="Missing top or bottom patch content.",
        )
        return clean_edge_pair_patch, top_patch_rgb, bottom_patch_rgb, None

    top_clean_rgb = _add_header_band(
        patch_rgb=top_patch_rgb,
        header_text=f"{candidate_id} top",
        header_height_px=header_height_px,
    )
    bottom_clean_rgb = _add_header_band(
        patch_rgb=bottom_patch_rgb,
        header_text=f"{candidate_id} bottom",
        header_height_px=header_height_px,
    )

    combined_width = max(top_clean_rgb.shape[1], bottom_clean_rgb.shape[1])
    combined_height = top_clean_rgb.shape[0] + patch_gap_px + bottom_clean_rgb.shape[0]
    combined_clean_rgb = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_clean_rgb[:, :] = np.array([20, 20, 22], dtype=np.uint8)

    top_x = (combined_width - top_clean_rgb.shape[1]) // 2
    bottom_x = (combined_width - bottom_clean_rgb.shape[1]) // 2
    combined_clean_rgb[0:top_clean_rgb.shape[0], top_x:top_x + top_clean_rgb.shape[1]] = top_clean_rgb
    bottom_y0 = top_clean_rgb.shape[0] + patch_gap_px
    combined_clean_rgb[
        bottom_y0:bottom_y0 + bottom_clean_rgb.shape[0],
        bottom_x:bottom_x + bottom_clean_rgb.shape[1],
    ] = bottom_clean_rgb

    clean_edge_pair_patch = CleanEdgePairPatch(
        candidate_id=candidate_id,
        top_patch_image_path=None,
        bottom_patch_image_path=None,
        combined_clean_pair_image_path=None,
        valid=True,
        note=None,
    )
    return clean_edge_pair_patch, top_clean_rgb, bottom_clean_rgb, combined_clean_rgb


def build_boxed_candidate_tile(
    candidate_id: str,
    rotated_object_image_rgb: np.ndarray,
    top_patch_bounds: tuple[int, int, int, int] | None,
    bottom_patch_bounds: tuple[int, int, int, int] | None,
    output_path: str,
    header_height_px: int = 40,
    row_label_width_px: int = 96,
    content_padding_px: int = 14,
    row_gap_px: int = 14,
    border_px: int = 2,
    row_label_position: str = "left",
    row_label_height_px: int = 20,
    min_row_height_px: int | None = None,
    top_contact_guide_y_px: float | int | None = None,
    bottom_contact_guide_y_px: float | int | None = None,
) -> dict:
    """
    Build one VLM-readable tile with a candidate header and separate TOP/BOTTOM
    rows. All labels live outside the image content.
    """
    top_patch_rgb = _extract_patch_from_bounds(rotated_object_image_rgb, top_patch_bounds)
    bottom_patch_rgb = _extract_patch_from_bounds(rotated_object_image_rgb, bottom_patch_bounds)

    valid = top_patch_rgb is not None and bottom_patch_rgb is not None
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not valid:
        return {
            "candidate_id": candidate_id,
            "tile_image_path": None,
            "valid": False,
            "note": "Missing top or bottom patch content.",
            "top_patch_bounds": top_patch_bounds,
            "bottom_patch_bounds": bottom_patch_bounds,
        }

    top_patch_rgb = _draw_contact_guide(top_patch_rgb, top_contact_guide_y_px, color_rgb=(255, 210, 66))
    bottom_patch_rgb = _draw_contact_guide(bottom_patch_rgb, bottom_contact_guide_y_px, color_rgb=(66, 170, 255))
    top_image = Image.fromarray(top_patch_rgb).convert("RGB")
    bottom_image = Image.fromarray(bottom_patch_rgb).convert("RGB")

    content_width = max(top_image.width, bottom_image.width)
    row_height = max(top_image.height, bottom_image.height)
    if min_row_height_px is not None:
        row_height = max(row_height, min_row_height_px)
    if row_label_position == "left":
        tile_width = row_label_width_px + content_width + 2 * content_padding_px
        tile_height = header_height_px + 2 * content_padding_px + 2 * row_height + row_gap_px
    elif row_label_position == "above":
        tile_width = content_width + 2 * content_padding_px
        tile_height = (
            header_height_px
            + 2 * content_padding_px
            + 2 * (row_label_height_px + row_height)
            + row_gap_px
        )
    else:
        raise ValueError(f"Unsupported row_label_position: {row_label_position}")

    background_rgb = (242, 243, 239)
    tile_rgb = Image.new("RGB", (tile_width, tile_height), background_rgb)
    draw = ImageDraw.Draw(tile_rgb)

    header_fill = (31, 34, 38)
    border_rgb = (74, 78, 84)
    separator_rgb = (188, 190, 184)
    label_rgb = (38, 40, 43)
    patch_frame_rgb = (118, 122, 126)

    draw.rectangle((0, 0, tile_width - 1, header_height_px - 1), fill=header_fill)
    title_font = _load_bold_font(22)
    label_font = _load_bold_font(14)
    draw.text(
        (content_padding_px, _center_text_y(draw, candidate_id, title_font, 0, header_height_px)),
        candidate_id,
        fill=(255, 255, 255),
        font=title_font,
    )

    def paste_left_label_row(row_index: int, row_label: str, patch_image: Image.Image) -> tuple[int, int, int, int]:
        row_y0 = header_height_px + content_padding_px + row_index * (row_height + row_gap_px)

        draw.text(
            (content_padding_px, _center_text_y(draw, row_label, label_font, row_y0, row_y0 + row_height)),
            row_label,
            fill=label_rgb,
            font=label_font,
        )
        patch_x0 = row_label_width_px + content_padding_px
        patch_y0 = row_y0 + (row_height - patch_image.height) // 2
        frame_x0 = patch_x0 - border_px
        frame_y0 = patch_y0 - border_px
        frame_x1 = patch_x0 + patch_image.width + border_px - 1
        frame_y1 = patch_y0 + patch_image.height + border_px - 1
        draw.rectangle((frame_x0, frame_y0, frame_x1, frame_y1), fill=patch_frame_rgb)
        tile_rgb.paste(patch_image, (patch_x0, patch_y0))
        return (patch_x0, patch_y0, patch_x0 + patch_image.width, patch_y0 + patch_image.height)

    def paste_above_label_row(row_index: int, row_label: str, patch_image: Image.Image) -> tuple[int, int, int, int]:
        row_y0 = header_height_px + content_padding_px + row_index * (row_label_height_px + row_height + row_gap_px)
        label_y0 = row_y0
        patch_x0 = content_padding_px + (content_width - patch_image.width) // 2
        patch_y0 = row_y0 + row_label_height_px

        draw.rectangle(
            (
                content_padding_px,
                label_y0,
                tile_width - content_padding_px - 1,
                label_y0 + row_label_height_px - 1,
            ),
            fill=(229, 231, 225),
        )
        draw.text(
            (
                content_padding_px + 8,
                _center_text_y(draw, row_label, label_font, label_y0, label_y0 + row_label_height_px),
            ),
            row_label,
            fill=label_rgb,
            font=label_font,
        )
        frame_x0 = patch_x0 - border_px
        frame_y0 = patch_y0 - border_px
        frame_x1 = patch_x0 + patch_image.width + border_px - 1
        frame_y1 = patch_y0 + patch_image.height + border_px - 1
        draw.rectangle((frame_x0, frame_y0, frame_x1, frame_y1), fill=patch_frame_rgb)
        tile_rgb.paste(patch_image, (patch_x0, patch_y0))
        return (patch_x0, patch_y0, patch_x0 + patch_image.width, patch_y0 + patch_image.height)

    if row_label_position == "left":
        top_content_bounds = paste_left_label_row(0, "TOP", top_image)
        bottom_content_bounds = paste_left_label_row(1, "BOTTOM", bottom_image)
        separator_y = header_height_px + content_padding_px + row_height + row_gap_px // 2
    else:
        top_content_bounds = paste_above_label_row(0, "TOP", top_image)
        bottom_content_bounds = paste_above_label_row(1, "BOTTOM", bottom_image)
        separator_y = header_height_px + content_padding_px + row_label_height_px + row_height + row_gap_px // 2

    draw.line(
        (content_padding_px, separator_y, tile_width - content_padding_px - 1, separator_y),
        fill=separator_rgb,
        width=1,
    )
    draw.rectangle((0, 0, tile_width - 1, tile_height - 1), outline=border_rgb, width=border_px)

    tile_rgb.save(output_file)

    return {
        "candidate_id": candidate_id,
        "tile_image_path": str(output_file),
        "valid": True,
        "note": None,
        "tile_width_px": tile_width,
        "tile_height_px": tile_height,
        "top_content_bounds_in_tile": top_content_bounds,
        "bottom_content_bounds_in_tile": bottom_content_bounds,
        "top_patch_bounds": top_patch_bounds,
        "bottom_patch_bounds": bottom_patch_bounds,
    }


def build_clean_edge_pair_grid(
    candidate_items: list[dict],
    output_path: str,
    columns: int = 3,
    padding_px: int = 16,
    draw_tile_labels: bool = False,
) -> str:
    """
    Build one grid image from the combined clean pair images.
    """
    valid_items = [item for item in candidate_items if item.get("combined_clean_pair_image_path")]
    if not valid_items:
        raise ValueError("No clean combined pair images were provided.")

    tiles = [Image.open(item["combined_clean_pair_image_path"]).convert("RGB") for item in valid_items]
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
        if draw_tile_labels:
            draw.text((x0 + 8, y0 + 6), item["candidate_id"], fill=(255, 255, 255))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    for tile in tiles:
        tile.close()
    return str(output_file)


def build_boxed_candidate_tile_grid(
    candidate_items: list[dict],
    output_path: str,
    columns: int = 3,
    padding_px: int = 18,
) -> str:
    """
    Build one compact grid image from VLM-readable boxed candidate tiles.
    """
    valid_items = [item for item in candidate_items if item.get("tile_image_path")]
    if not valid_items:
        raise ValueError("No boxed candidate tile images were provided.")

    tiles = [Image.open(item["tile_image_path"]).convert("RGB") for item in valid_items]
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
        (226, 228, 222),
    )

    for idx, tile in enumerate(tiles):
        row_idx = idx // column_count
        col_idx = idx % column_count
        x0 = padding_px + col_idx * (tile_width + padding_px) + (tile_width - tile.width) // 2
        y0 = padding_px + row_idx * (tile_height + padding_px) + (tile_height - tile.height) // 2
        canvas.paste(tile, (x0, y0))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    for tile in tiles:
        tile.close()
    return str(output_file)
