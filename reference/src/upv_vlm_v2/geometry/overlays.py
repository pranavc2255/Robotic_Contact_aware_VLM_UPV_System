"""PIL-based overlay helpers for v2 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_binary_mask(path: str | Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 0


def _resize_mask_to_image(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.shape == (height, width):
        return mask.astype(bool)
    mask_img = Image.fromarray(mask.astype("uint8") * 255, mode="L")
    resized = mask_img.resize((width, height), resample=Image.NEAREST)
    return np.array(resized) > 0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    foreground = mask.astype(bool)
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    eroded = foreground.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + foreground.shape[0], 1 + dx : 1 + dx + foreground.shape[1]]
    return foreground & ~eroded


def draw_mask_overlay(
    base: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (40, 220, 120),
    alpha: int = 96,
    outline_color: tuple[int, int, int] = (0, 255, 80),
    outline_width: int = 2,
) -> Image.Image:
    rgb = base.convert("RGB")
    mask_bool = _resize_mask_to_image(mask, rgb.size)
    arr = np.asarray(rgb).astype(np.float32).copy()
    if np.any(mask_bool):
        tint = np.array(color, dtype=np.float32)
        alpha_f = float(alpha) / 255.0 if alpha > 1 else float(alpha)
        alpha_f = float(np.clip(alpha_f, 0.0, 1.0))
        arr[mask_bool] = arr[mask_bool] * (1.0 - alpha_f) + tint * alpha_f
        boundary = _mask_boundary(mask_bool)
        if outline_width > 1:
            dilated = boundary.copy()
            for _ in range(int(outline_width) - 1):
                padded = np.pad(dilated, 1, mode="constant", constant_values=False)
                neighbors = [
                    padded[1 + dy : 1 + dy + boundary.shape[0], 1 + dx : 1 + dx + boundary.shape[1]]
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                ]
                dilated = np.logical_or.reduce(neighbors)
            boundary = dilated
        arr[boundary] = np.array(outline_color, dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")


def save_selected_mask_overlay(
    *,
    rgb_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    header: list[str],
    points: dict[str, list[float] | None] | None = None,
    lines: list[tuple[list[float], list[float], tuple[int, int, int], int]] | None = None,
) -> str:
    rgb = load_rgb(rgb_path)
    mask = load_binary_mask(mask_path)
    body = draw_mask_overlay(rgb, mask)
    header_h = max(58, 24 + 18 * len(header))
    canvas = Image.new("RGB", (body.width, body.height + header_h), (245, 246, 248))
    canvas.paste(body, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    for idx, text in enumerate(header):
        draw.text((12, 10 + idx * 18), str(text), fill=(20, 22, 25), font=_font(13))
    offset = np.array([0.0, float(header_h)])
    if lines:
        for start, end, color, width in lines:
            s = np.asarray(start, dtype=float) + offset
            e = np.asarray(end, dtype=float) + offset
            draw.line([tuple(s), tuple(e)], fill=color, width=width)
    if points:
        for name, point in points.items():
            if point is None:
                continue
            p = np.asarray(point, dtype=float) + offset
            r = 4
            draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 40, 40), outline=(255, 255, 255))
            draw.text((p[0] + 6, p[1] - 8), name, fill=(255, 255, 255), font=_font(12))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out)


def save_axis_overlay(
    *,
    rgb_path: str | Path,
    mask_path: str | Path,
    geometry: dict[str, Any],
    output_path: str | Path,
) -> str:
    center = geometry["centroid_px"]
    major = np.asarray(geometry["major_axis_vector"], dtype=float)
    minor = np.asarray(geometry["minor_axis_vector"], dtype=float)
    major_half = float(geometry.get("major_axis_length_px", 0.0)) / 2.0
    minor_half = float(geometry.get("minor_axis_length_px", 0.0)) / 2.0
    lines = [
        ((np.asarray(center) - major * major_half).tolist(), (np.asarray(center) + major * major_half).tolist(), (0, 220, 60), 3),
        ((np.asarray(center) - minor * minor_half).tolist(), (np.asarray(center) + minor * minor_half).tolist(), (0, 180, 255), 3),
    ]
    header = [
        "v2 geometry: selected mask major/minor axes",
        f"major angle={geometry.get('major_axis_angle_deg')} deg, minor angle={geometry.get('minor_axis_angle_deg')} deg",
    ]
    return save_selected_mask_overlay(
        rgb_path=rgb_path,
        mask_path=mask_path,
        output_path=output_path,
        header=header,
        points={"centroid": center},
        lines=lines,
    )


def save_anchor_overlay(
    *,
    rgb_path: str | Path,
    mask_path: str | Path,
    anchors: list[dict[str, Any]],
    selected_anchor_id: str | None,
    output_path: str | Path,
    title: str,
) -> str:
    rgb = draw_mask_overlay(load_rgb(rgb_path), load_binary_mask(mask_path), color=(80, 200, 255), alpha=65)
    draw = ImageDraw.Draw(rgb)
    for item in anchors:
        point = item.get("anchor_px") or item.get("selected_anchor_px")
        if not point:
            continue
        x, y = float(point[0]), float(point[1])
        selected = item.get("anchor_id") == selected_anchor_id
        color = (0, 255, 70) if selected else (255, 180, 0)
        r = 6 if selected else 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))
        draw.text((x + 7, y - 8), str(item.get("anchor_id", "?")), fill=(0, 0, 0), font=_font(12))
        a = item.get("contact_point_a_px")
        b = item.get("contact_point_b_px")
        if a and b:
            draw.line([tuple(a), tuple(b)], fill=color, width=3 if selected else 1)
    header_h = 54
    canvas = Image.new("RGB", (rgb.width, rgb.height + header_h), (245, 246, 248))
    canvas.paste(rgb, (0, header_h))
    d2 = ImageDraw.Draw(canvas)
    d2.text((12, 10), title, fill=(20, 22, 25), font=_font(14))
    d2.text((12, 30), f"selected anchor: {selected_anchor_id or 'NONE'}", fill=(20, 22, 25), font=_font(13))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out)
