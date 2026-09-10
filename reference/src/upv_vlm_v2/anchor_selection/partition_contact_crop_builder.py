"""Partition-based UPV contact crop rendering for anchor-selection artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_E3_CROP_CONFIG: dict[str, Any] = {
    "single_anchor_background_rgb": [12, 12, 12],
    "single_anchor_text_rgb": [245, 215, 70],
    "single_anchor_contact_line_rgb": [245, 215, 70],
    "single_anchor_contact_line_style": "dotted",
    "single_anchor_contact_line_width_px": 2,
    "single_anchor_crop_upscale": 3.0,
    "single_anchor_min_canvas_width_px": 220,
    "single_anchor_section_gap_px": 10,
    "single_anchor_padding_px": 8,
    "single_anchor_lateral_span_fraction_of_spacing": 1.0,
    "single_anchor_min_lateral_span_px": 60,
    "single_anchor_max_lateral_span_px": 80,
    "single_anchor_edge_crop_depth_px": 84,
    "single_anchor_outer_context_fraction": 0.66,
    "single_anchor_min_outer_context_fraction": 0.52,
    "ideal_boundary_top_quantile": 0.12,
    "ideal_boundary_bottom_quantile": 0.88,
    "ideal_boundary_min_column_coverage_px": 6,
    "anchor_crop_coverage": {
        "mode": "legacy_spacing_window",
        "enforce_gap_free": False,
        "enforce_non_overlapping": False,
        "overlap_fraction": 0.0,
        "max_allowed_gap_px": 0.0,
        "max_allowed_overlap_px": 0.0,
        "continuity_axis": "edge_axis",
        "save_crop_pixel_metadata": False,
        "save_crop_overlays": False,
        "save_continuity_validation": False,
    },
}


def merged_crop_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_E3_CROP_CONFIG)
    if config:
        merged.update(config)
    return merged


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def guide_mask_rgb(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype("int16")
    g = arr[:, :, 1].astype("int16")
    b = arr[:, :, 2].astype("int16")
    yellow = (r > 170) & (g > 130) & (b < 130) & ((r - b) > 70) & ((g - b) > 50)
    blue = (b > 140) & (g > 80) & (r < 120) & ((b - r) > 60)
    cyan = (b > 120) & (g > 120) & (r < 120) & ((b - r) > 40) & ((g - r) > 40)
    return yellow | blue | cyan


def detect_contact_strip_centers(tile: Image.Image) -> tuple[int, int]:
    arr = np.array(tile.convert("RGB"))
    mask = guide_mask_rgb(arr)
    h = arr.shape[0]
    mid = h // 2

    def center(submask: np.ndarray, fallback: int) -> int:
        ys = np.where(submask)[0]
        if ys.size:
            return int(round(float(np.median(ys))))
        return fallback

    top_y = center(mask[:mid], max(0, int(round(h * 0.28))))
    bottom_y = center(mask[mid:], min(h - 1, int(round(h * 0.72))))
    if bottom_y < mid:
        bottom_y = min(h - 1, int(round(h * 0.72)))
    return top_y, bottom_y


def remove_guide_lines(tile: Image.Image) -> Image.Image:
    arr = np.array(tile.convert("RGB"))
    mask = guide_mask_rgb(arr)
    if not bool(mask.any()):
        return tile.convert("RGB")
    keep = ~mask
    if keep.any():
        med = np.median(arr[keep], axis=0).astype("uint8")
        arr[mask] = med
    return Image.fromarray(arr)


def crop_from_bounds(image: Image.Image, bounds: Any) -> Image.Image | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    x0, y0, x1, y1 = [int(round(float(v))) for v in bounds]
    x0 = max(0, min(image.width, x0))
    y0 = max(0, min(image.height, y0))
    x1 = max(0, min(image.width, x1))
    y1 = max(0, min(image.height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def median_spacing(values: list[float]) -> float | None:
    xs = sorted(values)
    diffs = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if not diffs:
        return None
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return float(diffs[mid])
    return float((diffs[mid - 1] + diffs[mid]) / 2.0)


def anchor_x_positions(candidate_map: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for aid, meta in candidate_map.items():
        if not isinstance(meta, dict):
            continue
        patch = meta.get("true_edge_pair_patch")
        if not isinstance(patch, dict):
            continue
        xy = patch.get("anchor_xy_rotated") or patch.get("top_hit_xy_rotated")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                out[str(aid)] = float(xy[0])
            except Exception:
                pass
    return out


def _resolve_artifact_path(artifact_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return artifact_dir / path.name


def _load_rotated_mask(
    *,
    artifact_dir: Path,
    rotated_artifacts: dict[str, Any],
    rotated_size: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    mask_path = _resolve_artifact_path(artifact_dir, rotated_artifacts.get("rotated_object_mask"))
    if mask_path and mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        if mask.shape[1] == rotated_size[0] and mask.shape[0] == rotated_size[1]:
            return mask, str(mask_path)
    cropped_mask_path = _resolve_artifact_path(artifact_dir, rotated_artifacts.get("cropped_rotated_object_mask"))
    if cropped_mask_path and cropped_mask_path.exists():
        mask = np.array(Image.open(cropped_mask_path).convert("L")) > 0
        return mask, str(cropped_mask_path)
    return None, str(mask_path) if mask_path else None


def ideal_boundaries_from_mask(mask: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = merged_crop_config(config)
    if mask.ndim != 2 or mask.size == 0:
        return {"success": False, "failure_reason": "missing_or_invalid_mask"}
    min_coverage = int(cfg.get("ideal_boundary_min_column_coverage_px", 6))
    top_edges: list[float] = []
    bottom_edges: list[float] = []
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x])[0]
        if ys.size >= min_coverage:
            top_edges.append(float(ys.min()))
            bottom_edges.append(float(ys.max()))
    if not top_edges or not bottom_edges:
        return {"success": False, "failure_reason": "no_supported_mask_columns"}
    top_q = float(cfg.get("ideal_boundary_top_quantile", 0.12))
    bottom_q = float(cfg.get("ideal_boundary_bottom_quantile", 0.88))
    top_q = max(0.0, min(1.0, top_q))
    bottom_q = max(0.0, min(1.0, bottom_q))
    top_y = float(np.quantile(np.array(top_edges, dtype=np.float32), top_q))
    bottom_y = float(np.quantile(np.array(bottom_edges, dtype=np.float32), bottom_q))
    return {
        "success": True,
        "method": "robust_column_edge_quantile_rectangular_envelope",
        "top_boundary_y": top_y,
        "bottom_boundary_y": bottom_y,
        "top_quantile": top_q,
        "bottom_quantile": bottom_q,
        "supported_column_count": len(top_edges),
        "mask_width_px": int(mask.shape[1]),
        "mask_height_px": int(mask.shape[0]),
    }


def local_lateral_bounds(
    *,
    anchor_id: str,
    x_positions: dict[str, float],
    image_width: int,
    config: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    cfg = merged_crop_config(config)
    center_x = float(x_positions[anchor_id])
    ordered = sorted(x_positions.items(), key=lambda kv: kv[1])
    ids = [aid for aid, _ in ordered]
    xs = [float(x) for _, x in ordered]
    idx = ids.index(anchor_id)
    spacing = median_spacing(xs) or 64.0

    zone_left = (xs[idx - 1] + xs[idx]) / 2.0 if idx > 0 else 0.0
    zone_right = (xs[idx] + xs[idx + 1]) / 2.0 if idx < len(xs) - 1 else float(image_width)

    fraction = float(cfg.get("single_anchor_lateral_span_fraction_of_spacing", 1.0))
    min_span = float(cfg.get("single_anchor_min_lateral_span_px", 60))
    max_span = float(cfg.get("single_anchor_max_lateral_span_px", 80))
    desired_span = max(min_span, min(max_span, spacing * fraction))
    half = desired_span / 2.0
    x0 = max(zone_left, center_x - half)
    x1 = min(zone_right, center_x + half)

    if (x1 - x0) < min_span:
        deficit = min_span - (x1 - x0)
        x0 = max(zone_left, x0 - deficit / 2.0)
        x1 = min(zone_right, x1 + deficit / 2.0)
    if (x1 - x0) < min_span:
        if x0 <= zone_left + 1e-6:
            x1 = min(zone_right, x0 + min_span)
        elif x1 >= zone_right - 1e-6:
            x0 = max(zone_left, x1 - min_span)

    ix0 = max(0, min(image_width - 1, int(math.floor(x0))))
    ix1 = max(ix0 + 1, min(image_width, int(math.ceil(x1))))
    return ix0, ix1, {
        "anchor_center_x_rotated": center_x,
        "anchor_lateral_zone_px": [zone_left, zone_right],
        "anchor_lateral_spacing_px": spacing,
        "requested_lateral_span_px": desired_span,
        "actual_lateral_span_px": ix1 - ix0,
    }


def build_distinct_gap_free_lateral_intervals(
    *,
    anchor_ids: list[str],
    interval_start_x: float,
    interval_end_x: float,
) -> dict[str, list[float]]:
    ordered = sorted(anchor_ids, key=lambda aid: int(aid[1:]) if aid.startswith("A") and aid[1:].isdigit() else 9999)
    n = len(ordered)
    if n <= 0:
        return {}
    start = float(interval_start_x)
    end = float(interval_end_x)
    if end <= start:
        raise ValueError("interval_end_x must be greater than interval_start_x for gap-free intervals.")
    boundaries = np.linspace(start, end, n + 1, dtype=np.float64)
    return {
        aid: [float(boundaries[idx]), float(boundaries[idx + 1])]
        for idx, aid in enumerate(ordered)
    }


def edge_normal_bounds(
    *,
    edge_y: float,
    image_height: int,
    side: str,
    config: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    cfg = merged_crop_config(config)
    depth = float(cfg.get("single_anchor_edge_crop_depth_px", 84))
    outer_fraction = float(cfg.get("single_anchor_outer_context_fraction", 0.66))
    min_outer_fraction = float(cfg.get("single_anchor_min_outer_context_fraction", 0.52))
    outer = max(1.0, depth * outer_fraction)
    inner = max(1.0, depth - outer)
    ey = float(edge_y)

    if side == "top":
        y0 = max(0.0, ey - outer)
        y1 = min(float(image_height), ey + inner)
        actual_outer = ey - y0
        actual_inner = y1 - ey
        if actual_outer > 1 and actual_outer / max(1.0, actual_outer + actual_inner) < min_outer_fraction:
            max_inner = actual_outer * (1.0 - min_outer_fraction) / min_outer_fraction
            y1 = min(y1, ey + max(1.0, max_inner))
    elif side == "bottom":
        y0 = max(0.0, ey - inner)
        y1 = min(float(image_height), ey + outer)
        actual_outer = y1 - ey
        actual_inner = ey - y0
        if actual_outer > 1 and actual_outer / max(1.0, actual_outer + actual_inner) < min_outer_fraction:
            max_inner = actual_outer * (1.0 - min_outer_fraction) / min_outer_fraction
            y0 = max(y0, ey - max(1.0, max_inner))
    else:
        raise ValueError(f"Unsupported side: {side}")

    iy0 = max(0, min(image_height - 1, int(math.floor(y0))))
    iy1 = max(iy0 + 1, min(image_height, int(math.ceil(y1))))

    if side == "top":
        actual_outer_px = max(0.0, ey - iy0)
        actual_inner_px = max(0.0, iy1 - ey)
    else:
        actual_outer_px = max(0.0, iy1 - ey)
        actual_inner_px = max(0.0, ey - iy0)
    total = max(1.0, actual_outer_px + actual_inner_px)
    return iy0, iy1, {
        f"{side}_edge_y_rotated": ey,
        f"{side}_outer_context_px": actual_outer_px,
        f"{side}_inner_material_px": actual_inner_px,
        f"{side}_outer_context_fraction": actual_outer_px / total,
    }


def load_metadata_contact_crops(
    artifact_dir: Path | None,
    anchor_id: str,
    config: dict[str, Any] | None = None,
) -> tuple[Image.Image | None, Image.Image | None, dict[str, Any]]:
    if artifact_dir is None:
        return None, None, {"crop_source": "missing_artifact_dir"}

    cfg = merged_crop_config(config)
    rotation_meta_path = Path(artifact_dir) / "rotation_metadata.json"
    rotation_meta = read_json(rotation_meta_path, default={})
    if not isinstance(rotation_meta, dict):
        return None, None, {"crop_source": "missing_rotation_metadata", "rotation_metadata_path": str(rotation_meta_path)}

    rotated_path = None
    rotated_artifacts = rotation_meta.get("rotated_artifacts")
    rotated_artifacts = rotated_artifacts if isinstance(rotated_artifacts, dict) else {}
    using_full_rotated = False
    rotated_path = rotated_artifacts.get("rotated_object_rgb") or rotated_artifacts.get("cropped_rotated_object_rgb")
    using_full_rotated = bool(rotated_artifacts.get("rotated_object_rgb"))
    if not rotated_path:
        rotated_path = str(Path(artifact_dir) / "cropped_rotated_object_rgb.png")

    rotated_image_path = Path(str(rotated_path))
    if not rotated_image_path.is_absolute() and not rotated_image_path.exists():
        rotated_image_path = Path(artifact_dir) / rotated_image_path.name
    if not rotated_image_path.exists():
        return None, None, {
            "crop_source": "missing_rotated_object_rgb",
            "rotation_metadata_path": str(rotation_meta_path),
            "rotated_image_path": str(rotated_image_path),
        }

    candidate_map = rotation_meta.get("candidate_point_mapping_original_to_rotated_crop") or {}
    candidate_meta = candidate_map.get(anchor_id) if isinstance(candidate_map, dict) else None
    patch_meta = candidate_meta.get("true_edge_pair_patch") if isinstance(candidate_meta, dict) else None
    if not isinstance(patch_meta, dict):
        return None, None, {
            "crop_source": "missing_patch_bounds",
            "rotation_metadata_path": str(rotation_meta_path),
            "rotated_image_path": str(rotated_image_path),
        }

    rotated = Image.open(rotated_image_path).convert("RGB")
    coord_offset_x = 0.0
    coord_offset_y = 0.0
    crop_box = rotation_meta.get("rotated_crop_box")
    if using_full_rotated and isinstance(crop_box, (list, tuple)) and len(crop_box) >= 2:
        coord_offset_x = float(crop_box[0])
        coord_offset_y = float(crop_box[1])

    candidate_map = candidate_map if isinstance(candidate_map, dict) else {}
    x_positions = anchor_x_positions(candidate_map)
    if coord_offset_x or coord_offset_y:
        x_positions = {aid: x + coord_offset_x for aid, x in x_positions.items()}
    top_hit = patch_meta.get("top_hit_xy_rotated")
    bottom_hit = patch_meta.get("bottom_hit_xy_rotated")
    dominant_boundaries = rotation_meta.get("dominant_rectangle_crop_boundaries_rotated")
    if isinstance(dominant_boundaries, dict) and dominant_boundaries.get("success"):
        ideal = {
            "success": True,
            "method": "dominant_rectangle_geometry",
            "body_geometry_mode": "dominant_rectangle",
            "top_boundary_y": dominant_boundaries.get("top_boundary_y"),
            "bottom_boundary_y": dominant_boundaries.get("bottom_boundary_y"),
            "left_boundary_x": dominant_boundaries.get("left_boundary_x"),
            "right_boundary_x": dominant_boundaries.get("right_boundary_x"),
            "source": "rotation_metadata.dominant_rectangle_crop_boundaries_rotated",
        }
        rotated_mask = None
        rotated_mask_path = None
    else:
        rotated_mask, rotated_mask_path = _load_rotated_mask(
            artifact_dir=Path(artifact_dir),
            rotated_artifacts=rotated_artifacts,
            rotated_size=rotated.size,
        )
        ideal = ideal_boundaries_from_mask(rotated_mask, cfg) if rotated_mask is not None else {"success": False, "failure_reason": "missing_rotated_mask"}
    top_ideal_y = ideal.get("top_boundary_y")
    bottom_ideal_y = ideal.get("bottom_boundary_y")
    if top_ideal_y is not None and using_full_rotated:
        top_ideal_y = float(top_ideal_y)
    if bottom_ideal_y is not None and using_full_rotated:
        bottom_ideal_y = float(bottom_ideal_y)

    if (
        anchor_id not in x_positions
        or not isinstance(top_hit, (list, tuple))
        or not isinstance(bottom_hit, (list, tuple))
        or len(top_hit) < 2
        or len(bottom_hit) < 2
    ):
        top = crop_from_bounds(rotated, patch_meta.get("top_patch_bounds"))
        bottom = crop_from_bounds(rotated, patch_meta.get("bottom_patch_bounds"))
        return top, bottom, {
            "crop_source": "rotation_metadata_legacy_patch_bounds",
            "rotation_metadata_path": str(rotation_meta_path),
            "rotated_image_path": str(rotated_image_path),
            "top_patch_bounds": patch_meta.get("top_patch_bounds"),
            "bottom_patch_bounds": patch_meta.get("bottom_patch_bounds"),
        }

    x0, x1, x_meta = local_lateral_bounds(anchor_id=anchor_id, x_positions=x_positions, image_width=rotated.width, config=cfg)
    top_edge_y_for_crop = float(top_ideal_y) if isinstance(top_ideal_y, (int, float)) else float(top_hit[1]) + coord_offset_y
    bottom_edge_y_for_crop = float(bottom_ideal_y) if isinstance(bottom_ideal_y, (int, float)) else float(bottom_hit[1]) + coord_offset_y
    top_y0, top_y1, top_y_meta = edge_normal_bounds(
        edge_y=top_edge_y_for_crop,
        image_height=rotated.height,
        side="top",
        config=cfg,
    )
    bottom_y0, bottom_y1, bottom_y_meta = edge_normal_bounds(
        edge_y=bottom_edge_y_for_crop,
        image_height=rotated.height,
        side="bottom",
        config=cfg,
    )
    top = rotated.crop((x0, top_y0, x1, top_y1)).convert("RGB")
    bottom = rotated.crop((x0, bottom_y0, x1, bottom_y1)).convert("RGB")
    return top, bottom, {
        "crop_source": "rotation_metadata_local_anchor_window",
        "rotation_metadata_path": str(rotation_meta_path),
        "rotated_image_path": str(rotated_image_path),
        "rotated_coordinate_offset_xy": [coord_offset_x, coord_offset_y],
        "legacy_top_patch_bounds": patch_meta.get("top_patch_bounds"),
        "legacy_bottom_patch_bounds": patch_meta.get("bottom_patch_bounds"),
        "top_local_crop_bounds": [x0, top_y0, x1, top_y1],
        "bottom_local_crop_bounds": [x0, bottom_y0, x1, bottom_y1],
        "observed_top_hit_y_rotated": float(top_hit[1]) + coord_offset_y,
        "observed_bottom_hit_y_rotated": float(bottom_hit[1]) + coord_offset_y,
        "ideal_top_boundary": ideal,
        "ideal_bottom_boundary": ideal,
        "body_geometry_mode": rotation_meta.get("body_geometry_mode", "raw_mask_existing"),
        "dominant_rectangle_crop_boundaries_rotated": dominant_boundaries if isinstance(dominant_boundaries, dict) else {},
        "ideal_top_boundary_y_rotated": top_edge_y_for_crop,
        "ideal_bottom_boundary_y_rotated": bottom_edge_y_for_crop,
        "rotated_mask_path": rotated_mask_path,
        **x_meta,
        **top_y_meta,
        **bottom_y_meta,
    }


def fallback_tile_contact_crops(source_tile_path: str | Path) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    tile = Image.open(source_tile_path).convert("RGB")
    top_y, bottom_y = detect_contact_strip_centers(tile)
    tile = remove_guide_lines(tile)
    split_y = max(1, min(tile.height - 1, (top_y + bottom_y) // 2))
    margin = max(35, int(round((bottom_y - top_y) * 0.48)))
    top = tile.crop((0, max(0, top_y - margin), tile.width, min(split_y, top_y + margin))).convert("RGB")
    bottom = tile.crop((0, max(split_y, bottom_y - margin), tile.width, min(tile.height, bottom_y + margin))).convert("RGB")
    return top, bottom, {
        "crop_source": "fallback_source_tile_halves",
        "top_contact_strip_center_y_px_source_tile": top_y,
        "bottom_contact_strip_center_y_px_source_tile": bottom_y,
    }


def resize_crop(im: Image.Image, upscale: float) -> Image.Image:
    if upscale <= 0:
        upscale = 1.0
    if abs(upscale - 1.0) < 1e-6:
        return im
    new_size = (max(1, int(round(im.width * upscale))), max(1, int(round(im.height * upscale))))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def _text_bbox(text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    probe = Image.new("RGB", (8, 8), (0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1]))


def _draw_dotted_horizontal_line(
    canvas: Image.Image,
    *,
    y: int,
    color: tuple[int, int, int],
    width: int = 2,
    dash_px: int = 7,
    gap_px: int = 5,
    x0: int = 0,
    x1: int | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x_end = canvas.width if x1 is None else max(0, min(canvas.width, x1))
    x = max(0, min(canvas.width, x0))
    while x < x_end:
        seg_end = min(x + dash_px, x_end)
        draw.line((x, y, seg_end, y), fill=color, width=width)
        x += dash_px + gap_px


def _annotate_crop_with_edge_line(
    crop: Image.Image,
    *,
    edge_line_y: float | None,
    cfg: dict[str, Any],
) -> Image.Image:
    annotated = crop.convert("RGB").copy()
    if edge_line_y is None:
        return annotated
    edge_y = int(round(float(edge_line_y) * float(cfg.get("single_anchor_crop_upscale", 3.0))))
    edge_y = max(0, min(annotated.height - 1, edge_y))
    line_color = tuple(int(x) for x in cfg.get("single_anchor_contact_line_rgb", cfg.get("single_anchor_text_rgb", [245, 215, 70])))
    line_width = int(cfg.get("single_anchor_contact_line_width_px", 2))
    line_style = str(cfg.get("single_anchor_contact_line_style", "dotted")).strip().lower()
    if line_style == "solid":
        draw = ImageDraw.Draw(annotated)
        draw.line((0, edge_y, annotated.width, edge_y), fill=line_color, width=line_width)
    else:
        _draw_dotted_horizontal_line(annotated, y=edge_y, color=line_color, width=line_width, dash_px=7, gap_px=5)
    return annotated


def render_clean_single_anchor_input_from_bounds(
    *,
    rotated_image_path: str | Path,
    anchor_id: str,
    output_path: str | Path,
    top_bounds: list[float] | tuple[float, float, float, float],
    bottom_bounds: list[float] | tuple[float, float, float, float],
    top_edge_y_rotated: float,
    bottom_edge_y_rotated: float,
    config: dict[str, Any] | None = None,
    metadata_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = merged_crop_config(config)
    rotated = Image.open(rotated_image_path).convert("RGB")

    def _crop(bounds):
        x0, y0, x1, y1 = [float(v) for v in bounds]
        ix0 = max(0, min(rotated.width - 1, int(math.floor(x0))))
        iy0 = max(0, min(rotated.height - 1, int(math.floor(y0))))
        ix1 = max(ix0 + 1, min(rotated.width, int(math.ceil(x1))))
        iy1 = max(iy0 + 1, min(rotated.height, int(math.ceil(y1))))
        return rotated.crop((ix0, iy0, ix1, iy1)).convert("RGB"), [ix0, iy0, ix1, iy1]

    top_crop, top_bounds_i = _crop(top_bounds)
    bottom_crop, bottom_bounds_i = _crop(bottom_bounds)

    upscale = float(cfg.get("single_anchor_crop_upscale", 3.0))
    top_crop = resize_crop(top_crop, upscale)
    bottom_crop = resize_crop(bottom_crop, upscale)
    top_edge_local = float(top_edge_y_rotated) - float(top_bounds_i[1])
    bottom_edge_local = float(bottom_edge_y_rotated) - float(bottom_bounds_i[1])
    top_crop = _annotate_crop_with_edge_line(top_crop, edge_line_y=top_edge_local, cfg=cfg)
    bottom_crop = _annotate_crop_with_edge_line(bottom_crop, edge_line_y=bottom_edge_local, cfg=cfg)
    bottom_crop = bottom_crop.transpose(Image.Transpose.ROTATE_180)

    bg = tuple(int(x) for x in cfg.get("single_anchor_background_rgb", [12, 12, 12]))
    text_rgb = tuple(int(x) for x in cfg.get("single_anchor_text_rgb", [245, 215, 70]))
    pad = int(cfg.get("single_anchor_padding_px", 8))
    panel_gap = int(cfg.get("single_anchor_panel_gap_px", 12))
    title_font = font(17)
    label_font = font(13)
    title_w, title_h = _text_bbox(f"Anchor {anchor_id}", title_font)
    top_label_w, label_h = _text_bbox("Top contact crop", label_font)
    bottom_label_w, _ = _text_bbox("Bottom contact crop", label_font)
    image_row_y = pad + title_h + 6 + label_h + 4
    left_panel_w = max(top_crop.width, top_label_w)
    right_panel_w = max(bottom_crop.width, bottom_label_w)
    width = max(
        int(cfg.get("single_anchor_min_canvas_width_px", 220)),
        pad * 2 + left_panel_w + panel_gap + right_panel_w,
        pad * 2 + title_w,
    )
    height = image_row_y + max(top_crop.height, bottom_crop.height) + pad

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), f"Anchor {anchor_id}", fill=text_rgb, font=title_font)
    top_label_y = pad + title_h + 6
    draw.text((pad, top_label_y), "Top contact crop", fill=text_rgb, font=label_font)
    bottom_panel_x = pad + left_panel_w + panel_gap
    draw.text((bottom_panel_x, top_label_y), "Bottom contact crop", fill=text_rgb, font=label_font)
    top_x = pad
    top_y = image_row_y
    bottom_x = bottom_panel_x
    bottom_y = image_row_y
    canvas.paste(top_crop, (top_x, top_y))
    canvas.paste(bottom_crop, (bottom_x, bottom_y))
    top_crop_box = [top_x, top_y, top_x + top_crop.width, top_y + top_crop.height]
    bottom_crop_box = [bottom_x, bottom_y, bottom_x + bottom_crop.width, bottom_y + bottom_crop.height]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return {
        "source_tile_path": "",
        "rendered_input_path": str(out),
        "crop_source": "distinct_gap_free_non_overlapping_bins",
        "top_local_crop_bounds": top_bounds_i,
        "bottom_local_crop_bounds": bottom_bounds_i,
        "top_edge_y_rotated": float(top_edge_y_rotated),
        "bottom_edge_y_rotated": float(bottom_edge_y_rotated),
        "top_crop_box_in_rendered_image": top_crop_box,
        "bottom_crop_box_in_rendered_image": bottom_crop_box,
        "bottom_crop_rotated_180": True,
        "single_anchor_crop_upscale": upscale,
        "image_size_px": [canvas.width, canvas.height],
        **(metadata_source or {}),
    }


def render_clean_single_anchor_input(
    *,
    source_tile_path: str | Path,
    anchor_id: str,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = merged_crop_config(config)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    top_crop, bottom_crop, crop_meta = load_metadata_contact_crops(
        Path(artifact_dir) if artifact_dir is not None else None,
        anchor_id,
        cfg,
    )
    if top_crop is None or bottom_crop is None:
        top_crop, bottom_crop, fallback_meta = fallback_tile_contact_crops(source_tile_path)
        crop_meta.update(fallback_meta)

    upscale = float(cfg.get("single_anchor_crop_upscale", 3.0))
    top_crop = resize_crop(top_crop, upscale)
    bottom_crop = resize_crop(bottom_crop, upscale)
    top_edge_line_y: float | None = None
    bottom_edge_line_y: float | None = None
    top_bounds = crop_meta.get("top_local_crop_bounds")
    bottom_bounds = crop_meta.get("bottom_local_crop_bounds")
    if isinstance(top_bounds, (list, tuple)) and len(top_bounds) == 4 and isinstance(crop_meta.get("top_edge_y_rotated"), (int, float)):
        top_edge_line_y = float(crop_meta["top_edge_y_rotated"]) - float(top_bounds[1])
    if isinstance(bottom_bounds, (list, tuple)) and len(bottom_bounds) == 4 and isinstance(crop_meta.get("bottom_edge_y_rotated"), (int, float)):
        bottom_edge_line_y = float(crop_meta["bottom_edge_y_rotated"]) - float(bottom_bounds[1])
    top_crop = _annotate_crop_with_edge_line(top_crop, edge_line_y=top_edge_line_y, cfg=cfg)
    bottom_crop = _annotate_crop_with_edge_line(bottom_crop, edge_line_y=bottom_edge_line_y, cfg=cfg)
    bottom_crop = bottom_crop.transpose(Image.Transpose.ROTATE_180)

    bg = tuple(int(x) for x in cfg.get("single_anchor_background_rgb", [12, 12, 12]))
    text_rgb = tuple(int(x) for x in cfg.get("single_anchor_text_rgb", [245, 215, 70]))
    pad = int(cfg.get("single_anchor_padding_px", 8))
    section_gap = int(cfg.get("single_anchor_section_gap_px", 10))
    panel_gap = int(cfg.get("single_anchor_panel_gap_px", 12))
    title_font = font(17)
    label_font = font(13)
    title_w, title_h = _text_bbox(f"Anchor {anchor_id}", title_font)
    top_label_w, label_h = _text_bbox("Top contact crop", label_font)
    bottom_label_w, _ = _text_bbox("Bottom contact crop", label_font)
    image_row_y = pad + title_h + 6 + label_h + 4
    panels_top_y = image_row_y
    left_panel_w = max(top_crop.width, top_label_w)
    right_panel_w = max(bottom_crop.width, bottom_label_w)
    width = max(
        int(cfg.get("single_anchor_min_canvas_width_px", 220)),
        pad * 2 + left_panel_w + panel_gap + right_panel_w,
        pad * 2 + title_w,
    )
    height = panels_top_y + max(top_crop.height, bottom_crop.height) + pad

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    y = pad
    draw.text((pad, y), f"Anchor {anchor_id}", fill=text_rgb, font=title_font)
    top_label_y = pad + title_h + 6
    draw.text((pad, top_label_y), "Top contact crop", fill=text_rgb, font=label_font)
    bottom_panel_x = pad + left_panel_w + panel_gap
    draw.text((bottom_panel_x, top_label_y), "Bottom contact crop", fill=text_rgb, font=label_font)
    top_x = pad
    top_y = panels_top_y
    bottom_x = bottom_panel_x
    bottom_y = panels_top_y
    canvas.paste(top_crop, (top_x, top_y))
    canvas.paste(bottom_crop, (bottom_x, bottom_y))
    top_crop_box = [top_x, top_y, top_x + top_crop.width, top_y + top_crop.height]
    bottom_crop_box = [bottom_x, bottom_y, bottom_x + bottom_crop.width, bottom_y + bottom_crop.height]
    canvas.save(out)

    return {
        "source_tile_path": str(source_tile_path),
        "rendered_input_path": str(out),
        **crop_meta,
        "top_crop_box_in_rendered_image": top_crop_box,
        "bottom_crop_box_in_rendered_image": bottom_crop_box,
        "bottom_crop_rotated_180": True,
        "single_anchor_crop_upscale": upscale,
        "image_size_px": [canvas.width, canvas.height],
    }


def canonical_tile_paths(tile_dir: str | Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for p in Path(tile_dir).glob("candidate_A*_wide_contact_guided_tile.png"):
        stem = p.name.removeprefix("candidate_").removesuffix("_wide_contact_guided_tile.png")
        if stem.startswith("A"):
            paths[stem] = p
    return dict(sorted(paths.items(), key=lambda item: int(item[0][1:]) if item[0][1:].isdigit() else 999))


def render_clean_anchor_inputs_for_artifact_dir(
    *,
    artifact_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_dir)
    tile_dir = artifact / "boxed_wide_contact_guided_tiles"
    tile_paths = canonical_tile_paths(tile_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchors: dict[str, Any] = {}
    for anchor_id, tile_path in tile_paths.items():
        image_path = out / f"source_{anchor_id}.png"
        meta = render_clean_single_anchor_input(
            source_tile_path=tile_path,
            anchor_id=anchor_id,
            output_path=image_path,
            config=config,
            artifact_dir=artifact,
        )
        anchors[anchor_id] = meta
    first_meta = next(iter(anchors.values()), {}) if anchors else {}
    lateral_partitions = {
        aid: {
            "anchor_lateral_zone_px": meta.get("anchor_lateral_zone_px"),
            "top_crop_x_bounds": meta.get("top_local_crop_bounds", [None, None, None, None])[0::2]
            if isinstance(meta.get("top_local_crop_bounds"), list)
            else None,
            "bottom_crop_x_bounds": meta.get("bottom_local_crop_bounds", [None, None, None, None])[0::2]
            if isinstance(meta.get("bottom_local_crop_bounds"), list)
            else None,
        }
        for aid, meta in anchors.items()
    }
    debug = {
        "artifact_dir": str(artifact),
        "output_dir": str(out),
        "anchor_count": len(anchors),
        "anchor_ids": list(anchors.keys()),
        "lateral_partition_bounds_by_anchor": lateral_partitions,
        "ideal_top_boundary": first_meta.get("ideal_top_boundary"),
        "ideal_bottom_boundary": first_meta.get("ideal_bottom_boundary"),
        "anchors": anchors,
        "config": merged_crop_config(config),
    }
    write_json(out.parent / "anchor_crop_geometry_debug.json", debug)
    return debug


def build_clean_anchor_review_grid(
    *,
    image_paths: dict[str, str | Path],
    output_path: str | Path,
    background_rgb: tuple[int, int, int] = (250, 250, 248),
    padding_px: int = 18,
    columns: int = 5,
) -> str:
    items = [(aid, Path(path)) for aid, path in sorted(image_paths.items(), key=lambda item: item[0]) if Path(path).exists()]
    if not items:
        raise ValueError("No clean anchor images were provided.")
    images = [(aid, Image.open(path).convert("RGB")) for aid, path in items]
    tile_w = max(im.width for _, im in images)
    tile_h = max(im.height for _, im in images)
    cols = max(1, min(int(columns), len(images)))
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * padding_px, rows * tile_h + (rows + 1) * padding_px),
        background_rgb,
    )
    for idx, (_, im) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding_px + col * (tile_w + padding_px) + (tile_w - im.width) // 2
        y = padding_px + row * (tile_h + padding_px) + (tile_h - im.height) // 2
        canvas.paste(im, (x, y))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    for _, im in images:
        im.close()
    return str(out)
