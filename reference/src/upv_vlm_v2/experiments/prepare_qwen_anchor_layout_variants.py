"""Build alternate Qwen-input image layouts for adaptive prompt+layout search.

Reads the existing rendered ``clean_single_anchor_inputs/source_A*.png`` plus
the matching ``anchor_crop_geometry_debug.json`` metadata and creates layout
variants used by ``run_qwen_anchor_adaptive_prompt_search.py``.

This module never touches robot hardware, the live camera, RTDE, Arduino, the
clamp, or Pundit. It only reads saved PNG/JSON files and writes new PNGs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


LAYOUTS = (
    "L0_baseline",
    "L1_contact_band_zoom",
    "L2_full_plus_zoom_2x2",
    "L3_red_inspection_band",
    "L4_thick_line_no_text",
    "L5_zoom_no_text_thick_line",
    "L5_magenta_line_red_zone",
    "L6_magenta_line_only",
)


@dataclass(frozen=True)
class RenderedAnchorInputs:
    """Pointers to the existing rendered source_A*.png and its metadata."""

    rendered_path: Path
    geometry_debug_path: Path
    anchor_id: str

    def load_geometry(self) -> dict[str, Any]:
        data = json.loads(self.geometry_debug_path.read_text(encoding="utf-8"))
        return data.get("anchors", {}).get(self.anchor_id, {})


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contact_line_y_in_rendered(meta: dict[str, Any]) -> tuple[int, int]:
    """Return (top_panel_line_y, bottom_panel_line_y) absolute in the rendered image."""

    upscale = float(meta.get("single_anchor_crop_upscale") or 3.0)
    top_box = meta.get("top_crop_box_in_rendered_image") or [0, 0, 0, 0]
    bot_box = meta.get("bottom_crop_box_in_rendered_image") or [0, 0, 0, 0]
    top_bounds = meta.get("top_local_crop_bounds") or [0, 0, 0, 0]
    bot_bounds = meta.get("bottom_local_crop_bounds") or [0, 0, 0, 0]
    top_edge_y = float(meta.get("top_edge_y_rotated", 0))
    bot_edge_y = float(meta.get("bottom_edge_y_rotated", 0))

    top_local = (top_edge_y - float(top_bounds[1])) * upscale
    bot_local_pre_rotate = (bot_edge_y - float(bot_bounds[1])) * upscale
    panel_h = float(bot_box[3] - bot_box[1])
    bot_local = panel_h - bot_local_pre_rotate
    return (
        int(round(float(top_box[1]) + top_local)),
        int(round(float(bot_box[1]) + bot_local)),
    )


def _panel_boxes(meta: dict[str, Any]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    top = meta.get("top_crop_box_in_rendered_image") or [0, 0, 0, 0]
    bot = meta.get("bottom_crop_box_in_rendered_image") or [0, 0, 0, 0]
    return tuple(int(v) for v in top), tuple(int(v) for v in bot)  # type: ignore[return-value]


def _draw_dotted_horizontal(
    draw: ImageDraw.ImageDraw,
    y: int,
    x0: int,
    x1: int,
    color: tuple[int, int, int],
    width: int = 2,
    dash_px: int = 8,
    gap_px: int = 6,
) -> None:
    x = x0
    while x < x1:
        seg_end = min(x + dash_px, x1)
        draw.line((x, y, seg_end, y), fill=color, width=width)
        x += dash_px + gap_px


def _draw_solid_horizontal(
    draw: ImageDraw.ImageDraw,
    y: int,
    x0: int,
    x1: int,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line((x0, y, x1, y), fill=color, width=width)


def _band_strip(canvas: Image.Image, y_center: int, x0: int, x1: int, half_height: int) -> Image.Image:
    y0 = max(0, y_center - half_height)
    y1 = min(canvas.height, y_center + half_height)
    return canvas.crop((x0, y0, x1, y1)).convert("RGB")


def render_l0_baseline(*, source_path: Path, out_path: Path) -> dict[str, Any]:
    img = Image.open(source_path).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {"layout": "L0_baseline", "image_path": str(out_path), "image_size": list(img.size)}


def render_l1_contact_band_zoom(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    band_half_height_panel_px: int = 70,
    target_height_px: int = 720,
) -> dict[str, Any]:
    """Crop a tight band ±N px around the dashed contact line in each panel, stack vertically.

    No title text. No labels. Just the contact band, top above bottom, separated by a thin gap.
    """

    img = Image.open(source_path).convert("RGB")
    top_y, bot_y = _contact_line_y_in_rendered(geometry)
    top_box, bot_box = _panel_boxes(geometry)
    top_strip = _band_strip(img, top_y, top_box[0], top_box[2], band_half_height_panel_px)
    bot_strip = _band_strip(img, bot_y, bot_box[0], bot_box[2], band_half_height_panel_px)

    # Resize each strip to a generous height for the VLM.
    def _resize_keep_w(strip: Image.Image) -> Image.Image:
        if strip.height <= 0:
            return strip
        scale = float(target_height_px) / float(strip.height)
        new_w = max(1, int(round(strip.width * scale)))
        return strip.resize((new_w, target_height_px), Image.Resampling.LANCZOS)

    top_resized = _resize_keep_w(top_strip)
    bot_resized = _resize_keep_w(bot_strip)

    gap = 12
    pad = 14
    canvas_w = max(top_resized.width, bot_resized.width) + 2 * pad
    canvas_h = pad + top_resized.height + gap + bot_resized.height + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
    canvas.paste(top_resized, ((canvas_w - top_resized.width) // 2, pad))
    canvas.paste(bot_resized, ((canvas_w - bot_resized.width) // 2, pad + top_resized.height + gap))

    # Mark contact line in each strip with a bright dashed yellow line + small label-free arrow tick.
    draw = ImageDraw.Draw(canvas)
    yellow = (245, 215, 70)
    top_line_y = pad + top_resized.height // 2
    bot_line_y = pad + top_resized.height + gap + bot_resized.height // 2
    line_x0 = (canvas_w - top_resized.width) // 2
    line_x1 = line_x0 + top_resized.width
    _draw_dotted_horizontal(draw, top_line_y, line_x0, line_x1, yellow, width=3, dash_px=10, gap_px=7)
    line_x0 = (canvas_w - bot_resized.width) // 2
    line_x1 = line_x0 + bot_resized.width
    _draw_dotted_horizontal(draw, bot_line_y, line_x0, line_x1, yellow, width=3, dash_px=10, gap_px=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "layout": "L1_contact_band_zoom",
        "image_path": str(out_path),
        "band_half_height_panel_px": band_half_height_panel_px,
        "top_strip_size": list(top_resized.size),
        "bot_strip_size": list(bot_resized.size),
        "image_size": list(canvas.size),
    }


def render_l2_full_plus_zoom_2x2(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    band_half_height_panel_px: int = 70,
) -> dict[str, Any]:
    """2x2 composite: top full panel, top contact-band zoom, bottom full panel, bottom zoom."""

    img = Image.open(source_path).convert("RGB")
    top_box, bot_box = _panel_boxes(geometry)
    top_y, bot_y = _contact_line_y_in_rendered(geometry)
    top_full = img.crop(top_box).convert("RGB")
    bot_full = img.crop(bot_box).convert("RGB")
    top_zoom = _band_strip(img, top_y, top_box[0], top_box[2], band_half_height_panel_px)
    bot_zoom = _band_strip(img, bot_y, bot_box[0], bot_box[2], band_half_height_panel_px)

    target_h = 520
    def _resize_keep_aspect(im: Image.Image) -> Image.Image:
        if im.height <= 0:
            return im
        scale = float(target_h) / float(im.height)
        return im.resize((max(1, int(round(im.width * scale))), target_h), Image.Resampling.LANCZOS)

    cells = [_resize_keep_aspect(x) for x in (top_full, top_zoom, bot_full, bot_zoom)]
    pad = 14
    gap = 12
    col_w = max(cells[0].width, cells[2].width, cells[1].width, cells[3].width)
    row_h = target_h
    canvas_w = pad * 2 + col_w * 2 + gap
    canvas_h = pad * 2 + row_h * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
    positions = [
        (pad, pad),
        (pad + col_w + gap, pad),
        (pad, pad + row_h + gap),
        (pad + col_w + gap, pad + row_h + gap),
    ]
    for cell, pos in zip(cells, positions):
        canvas.paste(cell, (pos[0] + (col_w - cell.width) // 2, pos[1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "layout": "L2_full_plus_zoom_2x2",
        "image_path": str(out_path),
        "image_size": list(canvas.size),
    }


def render_l3_red_inspection_band(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    band_half_height_panel_px: int = 50,
) -> dict[str, Any]:
    """Take the L0 baseline and overlay a translucent red rectangle around the dashed line."""

    img = Image.open(source_path).convert("RGBA")
    top_y, bot_y = _contact_line_y_in_rendered(geometry)
    top_box, bot_box = _panel_boxes(geometry)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    red_fill = (220, 30, 30, 70)
    red_border = (220, 30, 30, 220)
    for x0, x1, ly in (
        (top_box[0], top_box[2], top_y),
        (bot_box[0], bot_box[2], bot_y),
    ):
        draw.rectangle(
            (x0 + 2, ly - band_half_height_panel_px, x1 - 2, ly + band_half_height_panel_px),
            fill=red_fill,
            outline=red_border,
            width=2,
        )
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return {
        "layout": "L3_red_inspection_band",
        "image_path": str(out_path),
        "band_half_height_panel_px": band_half_height_panel_px,
        "image_size": list(out.size),
    }


def render_l4_thick_line_no_text(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Re-render the panels (cropped from baseline) with a thicker dashed line and no text/title.

    Removes title bar and labels by cropping to the panel area only.
    """

    img = Image.open(source_path).convert("RGB")
    top_box, bot_box = _panel_boxes(geometry)
    top_panel = img.crop(top_box).convert("RGB")
    bot_panel = img.crop(bot_box).convert("RGB")

    top_y_abs, bot_y_abs = _contact_line_y_in_rendered(geometry)
    top_y_local = top_y_abs - top_box[1]
    bot_y_local = bot_y_abs - bot_box[1]

    def _scale(im: Image.Image) -> Image.Image:
        h = 640
        scale = float(h) / float(im.height) if im.height > 0 else 1.0
        return im.resize((max(1, int(round(im.width * scale))), h), Image.Resampling.LANCZOS), scale

    top_scaled, top_scale = _scale(top_panel)
    bot_scaled, bot_scale = _scale(bot_panel)
    top_y_scaled = int(round(top_y_local * top_scale))
    bot_y_scaled = int(round(bot_y_local * bot_scale))

    pad = 16
    gap = 14
    canvas_w = pad * 2 + top_scaled.width + gap + bot_scaled.width
    canvas_h = pad * 2 + max(top_scaled.height, bot_scaled.height)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
    canvas.paste(top_scaled, (pad, pad))
    canvas.paste(bot_scaled, (pad + top_scaled.width + gap, pad))

    draw = ImageDraw.Draw(canvas)
    yellow = (255, 220, 60)
    _draw_dotted_horizontal(
        draw, pad + top_y_scaled, pad, pad + top_scaled.width, yellow,
        width=5, dash_px=14, gap_px=8,
    )
    _draw_dotted_horizontal(
        draw,
        pad + bot_y_scaled,
        pad + top_scaled.width + gap,
        pad + top_scaled.width + gap + bot_scaled.width,
        yellow,
        width=5,
        dash_px=14,
        gap_px=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "layout": "L4_thick_line_no_text",
        "image_path": str(out_path),
        "image_size": list(canvas.size),
    }


def render_l5_zoom_no_text_thick_line(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    band_half_height_panel_px: int = 70,
    target_height_px: int = 720,
) -> dict[str, Any]:
    """Like L1 but with thicker dashed yellow line and slightly tighter band."""

    return render_l1_contact_band_zoom(
        source_path=source_path,
        out_path=out_path,
        geometry=geometry,
        band_half_height_panel_px=band_half_height_panel_px,
        target_height_px=target_height_px,
    ) | {"layout": "L5_zoom_no_text_thick_line", "image_path": str(out_path)}


def _detect_existing_contact_line_y_values(img: Image.Image) -> list[int]:
    """Detect likely horizontal yellow guide-line rows from an existing source crop.

    This fallback is used when anchor crop geometry metadata is unavailable, as
    in exported E3 scoring sessions. It deliberately looks for the already
    rendered yellow dashed contact guide and returns clustered row centers.
    """

    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    row_counts: list[tuple[int, int]] = []
    threshold = max(12, int(width * 0.08))
    for y in range(height):
        count = 0
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= 180 and g >= 145 and b <= 130 and (r - b) >= 80 and (g - b) >= 50:
                count += 1
        if count >= threshold:
            row_counts.append((y, count))
    if not row_counts:
        # Conservative fallback for two vertically stacked panels.
        return [height // 4, (3 * height) // 4] if height > width else [height // 2]

    clusters: list[list[tuple[int, int]]] = []
    for y, count in row_counts:
        if not clusters or y - clusters[-1][-1][0] > 8:
            clusters.append([(y, count)])
        else:
            clusters[-1].append((y, count))

    centers: list[int] = []
    for cluster in clusters:
        total = sum(count for _, count in cluster)
        if total <= 0:
            continue
        centers.append(int(round(sum(y * count for y, count in cluster) / total)))
    return centers[:4]


def _detect_yellow_guide_clusters(img: Image.Image) -> list[dict[str, Any]]:
    """Find row/x extents of existing yellow dashed guide lines.

    The detector uses row support, so sparse yellow title text should not be
    returned. It also ignores the upper header area unless a row has strong
    horizontal yellow support.
    """

    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    min_row_support = max(12, int(width * 0.08))
    yellow_rows: list[tuple[int, int, int, int]] = []
    for y in range(height):
        xs: list[int] = []
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= 180 and g >= 145 and b <= 135 and (r - b) >= 70 and (g - b) >= 45:
                xs.append(x)
        if len(xs) >= min_row_support:
            yellow_rows.append((y, len(xs), min(xs), max(xs)))
    clusters: list[list[tuple[int, int, int, int]]] = []
    for item in yellow_rows:
        y = item[0]
        if not clusters or y - clusters[-1][-1][0] > 8:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    out: list[dict[str, Any]] = []
    forbidden_top_y = max(72, int(height * 0.16))
    for cluster in clusters:
        total = sum(item[1] for item in cluster)
        if total <= 0:
            continue
        y_center = int(round(sum(item[0] * item[1] for item in cluster) / total))
        if y_center < forbidden_top_y:
            continue
        x0 = min(item[2] for item in cluster)
        x1 = max(item[3] for item in cluster)
        # Reject tiny/title-like spans.
        if x1 - x0 < int(width * 0.25):
            continue
        out.append({"y": y_center, "x0": x0, "x1": x1, "support": total})
    return out


def render_l5_magenta_line_red_zone(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    band_half_height_panel_px: int = 20,
    band_alpha: int = 64,
) -> dict[str, Any]:
    """Overlay a bright magenta contact line and translucent red tolerance zone.

    The original RGB crop content remains visible. If geometry metadata is
    available, it is used for the two panel contact-line locations; otherwise,
    the current yellow dashed guide line is detected directly from the image.
    """

    img = Image.open(source_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    magenta = (255, 0, 255, 255)
    red_fill = (255, 0, 0, int(band_alpha))
    red_border = (255, 0, 0, min(220, int(band_alpha) + 110))

    zones: list[tuple[int, int, int]] = []
    if geometry:
        try:
            top_y, bot_y = _contact_line_y_in_rendered(geometry)
            top_box, bot_box = _panel_boxes(geometry)
            zones = [
                (top_box[0], top_box[2], top_y),
                (bot_box[0], bot_box[2], bot_y),
            ]
        except Exception:
            zones = []
    if not zones:
        for y in _detect_existing_contact_line_y_values(img):
            zones.append((0, img.width, y))

    for x0, x1, y in zones:
        y0 = max(0, int(y) - int(band_half_height_panel_px))
        y1 = min(img.height - 1, int(y) + int(band_half_height_panel_px))
        draw.rectangle((x0, y0, x1, y1), fill=red_fill, outline=red_border, width=2)
        draw.line((x0, int(y), x1, int(y)), fill=magenta, width=6)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return {
        "layout": "L5_magenta_line_red_zone",
        "image_path": str(out_path),
        "band_half_height_panel_px": band_half_height_panel_px,
        "band_alpha": band_alpha,
        "contact_line_y_values": [z[2] for z in zones],
        "used_geometry_metadata": bool(geometry),
        "image_size": list(out.size),
    }


def _is_yellow_guide_pixel(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return bool(r >= 180 and g >= 145 and b <= 135 and (r - b) >= 70 and (g - b) >= 45)


def _recolor_yellow_guide_pixels_near_line(
    img: Image.Image,
    *,
    x0: int,
    x1: int,
    y: int,
    band_px: int,
    color: tuple[int, int, int],
) -> None:
    """Replace existing yellow guide pixels near the intended contact line.

    This preserves material content while preventing yellow dashed remnants from
    surviving under/around the solid L6 magenta line. The ROI is clipped to the
    panel contact-line segment, so header/title text is not touched.
    """

    rgb = img.convert("RGB")
    pixels = rgb.load()
    y0 = max(0, int(y) - int(band_px))
    y1 = min(rgb.height - 1, int(y) + int(band_px))
    xx0 = max(0, min(rgb.width - 1, int(x0)))
    xx1 = max(xx0, min(rgb.width - 1, int(x1)))
    for yy in range(y0, y1 + 1):
        for xx in range(xx0, xx1 + 1):
            if _is_yellow_guide_pixel(pixels[xx, yy]):
                pixels[xx, yy] = color
    if rgb is not img:
        img.paste(rgb)


def render_l6_magenta_line_only(
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
    line_width_px: int = 4,
) -> dict[str, Any]:
    """Overlay only a clean magenta contact line on the two crop panels.

    No red band, no red outlines, no labels, and no header-spanning guides are
    added. With geometry metadata, the line is clipped to panel boxes. Without
    metadata, the existing yellow dashed guide rows are detected and overdrawn
    only across their yellow support extents.
    """

    img = Image.open(source_path).convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    magenta = (255, 0, 255)
    lines: list[dict[str, Any]] = []
    used_geometry = False
    forbidden_top_y = max(72, int(out.height * 0.16))

    if geometry:
        try:
            top_y, bot_y = _contact_line_y_in_rendered(geometry)
            top_box, bot_box = _panel_boxes(geometry)
            for box, y in ((top_box, top_y), (bot_box, bot_y)):
                x0 = max(0, int(box[0]))
                x1 = min(out.width - 1, int(box[2]))
                yy = max(0, min(out.height - 1, int(y)))
                if yy < forbidden_top_y:
                    continue
                _recolor_yellow_guide_pixels_near_line(out, x0=x0, x1=x1, y=yy, band_px=max(8, int(line_width_px) * 3), color=magenta)
                draw.line((x0, yy, x1, yy), fill=magenta, width=int(line_width_px))
                lines.append({"x0": x0, "x1": x1, "y": yy, "source": "geometry"})
            used_geometry = True
        except Exception:
            lines = []
            used_geometry = False

    if not lines:
        clusters = _detect_yellow_guide_clusters(img)
        for cluster in clusters:
            x0 = max(0, int(cluster["x0"]))
            x1 = min(out.width - 1, int(cluster["x1"]))
            y = max(0, min(out.height - 1, int(cluster["y"])))
            _recolor_yellow_guide_pixels_near_line(out, x0=x0, x1=x1, y=y, band_px=max(8, int(line_width_px) * 3), color=magenta)
            draw.line((x0, y, x1, y), fill=magenta, width=int(line_width_px))
            lines.append({"x0": x0, "x1": x1, "y": y, "source": "yellow_line_fallback"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return {
        "layout": "L6_magenta_line_only",
        "image_path": str(out_path),
        "line_width_px": int(line_width_px),
        "used_geometry_metadata": used_geometry,
        "forbidden_top_y": forbidden_top_y,
        "line_segments": lines,
        "image_size": list(out.size),
    }


_RENDERERS = {
    "L0_baseline": render_l0_baseline,
    "L1_contact_band_zoom": render_l1_contact_band_zoom,
    "L2_full_plus_zoom_2x2": render_l2_full_plus_zoom_2x2,
    "L3_red_inspection_band": render_l3_red_inspection_band,
    "L4_thick_line_no_text": render_l4_thick_line_no_text,
    "L5_zoom_no_text_thick_line": render_l5_zoom_no_text_thick_line,
    "L5_magenta_line_red_zone": render_l5_magenta_line_red_zone,
    "L6_magenta_line_only": render_l6_magenta_line_only,
}


def render_layout(
    layout: str,
    *,
    source_path: Path,
    out_path: Path,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    if layout not in _RENDERERS:
        raise ValueError(f"unknown layout {layout!r}; choices: {sorted(_RENDERERS)}")
    fn = _RENDERERS[layout]
    if layout == "L0_baseline":
        return fn(source_path=source_path, out_path=out_path)
    return fn(source_path=source_path, out_path=out_path, geometry=geometry)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Path to a rendered source_A*.png.")
    ap.add_argument(
        "--geometry-debug",
        required=True,
        help="Path to anchor_crop_geometry_debug.json from the same artifact_dir.",
    )
    ap.add_argument("--anchor-id", required=True, help="Anchor id key inside the geometry-debug JSON.")
    ap.add_argument("--out-dir", required=True, help="Output directory for variant PNGs.")
    ap.add_argument(
        "--layouts",
        nargs="+",
        default=list(LAYOUTS),
        choices=list(LAYOUTS),
        help="Layouts to render (default: all).",
    )
    args = ap.parse_args()

    geom_all = json.loads(Path(args.geometry_debug).read_text(encoding="utf-8"))
    geom = geom_all.get("anchors", {}).get(args.anchor_id, {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for layout in args.layouts:
        out_path = out_dir / f"{layout}__{args.anchor_id}.png"
        info = render_layout(
            layout,
            source_path=Path(args.source),
            out_path=out_path,
            geometry=geom,
        )
        results.append(info)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
