"""Candidate mask/crop pool construction for v2 target selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_MODE = "primary_requested_class_only"
CANDIDATE_POOL_MODE = "requested_class_instances_only"
CROP_PRIORITY = ["inner_texture_crop", "masked_texture_crop", "bbox_crop"]
PANEL_TILE_WIDTH = 340
PANEL_TILE_HEIGHT = 320
PANEL_HEADER_HEIGHT = 125
PANEL_PADDING = 12
PANEL_CROP_MARGIN_X = PANEL_PADDING
PANEL_CROP_MARGIN_BOTTOM = PANEL_PADDING


def _material_slug(material: str) -> str:
    return "_".join(str(material).strip().lower().split())


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def crop_panel_layout_metadata(*, columns: int, tile_width: int = PANEL_TILE_WIDTH, tile_height: int = PANEL_TILE_HEIGHT, header_height: int = PANEL_HEADER_HEIGHT) -> dict[str, Any]:
    crop_viewport_width = int(tile_width - 2 * PANEL_PADDING)
    crop_viewport_height = int(tile_height - header_height - 2 * PANEL_PADDING)
    return {
        "style": "t5_v2_candidate_crop_panel_fixed_viewport",
        "columns": int(columns),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "header_height": int(header_height),
        "padding": int(PANEL_PADDING),
        "crop_viewport_width": crop_viewport_width,
        "crop_viewport_height": crop_viewport_height,
        "fixed_crop_display_size": [crop_viewport_width, crop_viewport_height],
        "fixed_crop_display_viewport": True,
        "aspect_fit_centered": False,
        "aspect_fill_center_crop": True,
        "crop_viewport_fill_mode": "aspect_fill_center_crop",
        "crop_display_mode": "aspect_fill_center_crop",
        "text_area_reserved": True,
    }


def crop_panel_candidate_boxes(
    *,
    candidates: list[dict[str, Any]],
    columns: int,
    tile_width: int = PANEL_TILE_WIDTH,
    tile_height: int = PANEL_TILE_HEIGHT,
    header_height: int = PANEL_HEADER_HEIGHT,
) -> list[dict[str, Any]]:
    layout = crop_panel_layout_metadata(columns=columns, tile_width=tile_width, tile_height=tile_height, header_height=header_height)
    viewport_w = int(layout["crop_viewport_width"])
    viewport_h = int(layout["crop_viewport_height"])
    boxes: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        col = (idx - 1) % max(1, columns)
        row = (idx - 1) // max(1, columns)
        x0 = int(col * tile_width)
        y0 = int(row * tile_height)
        viewport_x0 = int(x0 + PANEL_PADDING)
        viewport_y0 = int(y0 + header_height + PANEL_PADDING)
        resized_box = None
        resized_size = None
        center_crop_box = None
        original_size = None
        crop_path = candidate.get("crop_used_for_verification") or candidate.get("crop_path_used_for_scoring")
        if crop_path and Path(str(crop_path)).exists():
            with Image.open(crop_path) as source:
                src_w, src_h = source.size
            if src_w > 0 and src_h > 0:
                original_size = [int(src_w), int(src_h)]
                scale = max(viewport_w / src_w, viewport_h / src_h)
                resized_w = max(viewport_w, int(np.ceil(src_w * scale)))
                resized_h = max(viewport_h, int(np.ceil(src_h * scale)))
                left = max(0, int((resized_w - viewport_w) // 2))
                top = max(0, int((resized_h - viewport_h) // 2))
                resized_size = [int(resized_w), int(resized_h)]
                center_crop_box = [left, top, left + viewport_w, top + viewport_h]
                resized_box = [0, 0, viewport_w, viewport_h]
        boxes.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "tile_box": [x0, y0, x0 + int(tile_width), y0 + int(tile_height)],
                "header_box": [x0 + PANEL_PADDING, y0 + PANEL_PADDING, x0 + int(tile_width) - PANEL_PADDING, y0 + int(header_height)],
                "crop_viewport_box": [viewport_x0, viewport_y0, viewport_x0 + viewport_w, viewport_y0 + viewport_h],
                "display_resize_mode": "aspect_fill_center_crop",
                "crop_original_size": original_size,
                "crop_resized_size_before_center_crop": resized_size,
                "crop_center_crop_box_in_resized": center_crop_box,
                "resized_crop_box_inside_viewport": resized_box,
            }
        )
    return boxes


def _resize_crop_to_fill_viewport(crop: Image.Image, viewport_w: int, viewport_h: int) -> Image.Image:
    crop_w, crop_h = crop.size
    if crop_w <= 0 or crop_h <= 0:
        return Image.new("RGB", (viewport_w, viewport_h), (255, 255, 255))
    scale = max(viewport_w / crop_w, viewport_h / crop_h)
    new_w = max(viewport_w, int(np.ceil(crop_w * scale)))
    new_h = max(viewport_h, int(np.ceil(crop_h * scale)))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    resized = crop.resize((new_w, new_h), resampling)
    left = max(0, int((new_w - viewport_w) // 2))
    top = max(0, int((new_h - viewport_h) // 2))
    return resized.crop((left, top, left + viewport_w, top + viewport_h))


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask has no foreground pixels.")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _save_mask(mask: np.ndarray, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype("uint8") * 255).save(output_path)
    return str(output_path)


def _crop_rgb(rgb: Image.Image, bbox: list[int], pad_px: int) -> Image.Image:
    left = max(0, bbox[0] - pad_px)
    top = max(0, bbox[1] - pad_px)
    right = min(rgb.width, bbox[2] + pad_px)
    bottom = min(rgb.height, bbox[3] + pad_px)
    return rgb.crop((left, top, right, bottom))


def _crop_box_from_mask(mask: np.ndarray, image_size: tuple[int, int], pad_px: int = 0) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    width, height = image_size
    return [
        max(0, int(xs.min()) - int(pad_px)),
        max(0, int(ys.min()) - int(pad_px)),
        min(width, int(xs.max()) + int(pad_px) + 1),
        min(height, int(ys.max()) + int(pad_px) + 1),
    ]


def _erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return (mask > 0).astype("uint8")
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.binary_erosion(mask > 0, iterations=erode_px).astype("uint8")
    except Exception:
        eroded = (mask > 0).astype("uint8")
        for _ in range(erode_px):
            padded = np.pad(eroded, 1, mode="constant")
            neighbors = [
                padded[1 + dy : 1 + dy + eroded.shape[0], 1 + dx : 1 + dx + eroded.shape[1]]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            eroded = np.logical_and.reduce(neighbors).astype("uint8")
        return eroded


def _save_masked_crop(rgb: Image.Image, mask: np.ndarray, bbox: list[int], output_path: Path, *, pad_px: int, erode_px: int = 0) -> str:
    crop_box = [
        max(0, bbox[0] - pad_px),
        max(0, bbox[1] - pad_px),
        min(rgb.width, bbox[2] + pad_px),
        min(rgb.height, bbox[3] + pad_px),
    ]
    crop = rgb.crop(tuple(crop_box)).convert("RGB")
    used_mask = _erode_mask(mask, erode_px)
    mask_crop = used_mask[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    arr = np.array(crop)
    arr[mask_crop <= 0] = 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path)
    return str(output_path)


def _save_tight_masked_crop(
    rgb: Image.Image,
    mask: np.ndarray,
    output_path: Path,
    *,
    crop_mask: np.ndarray,
    pad_px: int,
) -> dict[str, Any]:
    crop_box = _crop_box_from_mask(crop_mask, rgb.size, pad_px=pad_px)
    if crop_box is None:
        return {"success": False, "failure_reason": "empty_crop_mask", "crop_path": "NA"}
    crop = rgb.crop(tuple(crop_box)).convert("RGB")
    mask_crop = (crop_mask > 0).astype("uint8")[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    arr = np.array(crop)
    arr[mask_crop <= 0] = 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path)
    return {
        "success": True,
        "crop_path": str(output_path),
        "crop_box": crop_box,
        "foreground_pixels": int(mask_crop.sum()),
        "size": [int(arr.shape[1]), int(arr.shape[0])],
    }


def _save_inner_texture_crop(
    rgb: Image.Image,
    mask: np.ndarray,
    output_path: Path,
    *,
    erode_px: int,
    min_inner_crop_pixels: int,
) -> dict[str, Any]:
    eroded = _erode_mask(mask, erode_px)
    eroded_pixels = int(eroded.sum())
    if eroded_pixels < int(min_inner_crop_pixels):
        return {
            "success": False,
            "failure_reason": f"eroded_pixels_below_min_inner_crop_pixels:{eroded_pixels}<{int(min_inner_crop_pixels)}",
            "crop_path": "NA",
            "inner_crop_pixel_count": eroded_pixels,
            "inner_crop_erode_px": int(erode_px),
        }
    ys, xs = np.nonzero(eroded > 0)
    if len(xs) == 0:
        return {
            "success": False,
            "failure_reason": "empty_eroded_mask",
            "crop_path": "NA",
            "inner_crop_pixel_count": 0,
            "inner_crop_erode_px": int(erode_px),
        }
    # Match the old T5_v2 intent: use an interior percentile box, not the whole object.
    left = max(0, int(np.percentile(xs, 20)) - 2)
    right = min(rgb.width, int(np.percentile(xs, 80)) + 3)
    top = max(0, int(np.percentile(ys, 20)) - 2)
    bottom = min(rgb.height, int(np.percentile(ys, 80)) + 3)
    if right <= left or bottom <= top:
        return {
            "success": False,
            "failure_reason": "invalid_inner_percentile_box",
            "crop_path": "NA",
            "inner_crop_pixel_count": eroded_pixels,
            "inner_crop_erode_px": int(erode_px),
        }
    crop = rgb.crop((left, top, right, bottom)).convert("RGB")
    mask_crop = (eroded > 0).astype("uint8")[top:bottom, left:right]
    arr = np.array(crop)
    arr[mask_crop <= 0] = 255
    foreground_pixels = int(mask_crop.sum())
    if foreground_pixels < int(min_inner_crop_pixels):
        return {
            "success": False,
            "failure_reason": f"inner_crop_pixels_below_min_inner_crop_pixels:{foreground_pixels}<{int(min_inner_crop_pixels)}",
            "crop_path": "NA",
            "inner_crop_pixel_count": foreground_pixels,
            "inner_crop_erode_px": int(erode_px),
            "inner_crop_box": [left, top, right, bottom],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path)
    return {
        "success": True,
        "crop_path": str(output_path),
        "inner_crop_box": [left, top, right, bottom],
        "inner_crop_pixel_count": foreground_pixels,
        "inner_crop_erode_px": int(erode_px),
        "size": [int(arr.shape[1]), int(arr.shape[0])],
    }


def _save_masked_crop_alpha(rgb: Image.Image, mask: np.ndarray, bbox: list[int], output_path: Path, *, pad_px: int) -> str:
    crop_box = [
        max(0, bbox[0] - pad_px),
        max(0, bbox[1] - pad_px),
        min(rgb.width, bbox[2] + pad_px),
        min(rgb.height, bbox[3] + pad_px),
    ]
    crop = rgb.crop(tuple(crop_box)).convert("RGBA")
    mask_crop = (mask > 0).astype("uint8")[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    arr = np.array(crop)
    arr[:, :, 3] = mask_crop * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(output_path)
    return str(output_path)


def _choose_verification_crop(candidate: dict[str, Any]) -> tuple[str | None, str | None]:
    keys = (
        ("inner_texture_crop", "inner_texture_crop_path", "inner_texture_crop_invalid"),
        ("masked_texture_crop", "masked_texture_crop_path", "masked_texture_crop_missing"),
        ("bbox_crop", "bbox_crop_path", "bbox_crop_missing"),
    )
    fallback_reasons: list[str] = []
    for crop_type, key, reason in keys:
        value = candidate.get(key)
        if value and value != "NA" and Path(str(value)).exists():
            candidate["fallback_reason"] = "; ".join(fallback_reasons) if fallback_reasons else None
            return str(value), crop_type
        fallback_reasons.append(reason)
    candidate["fallback_reason"] = "; ".join(fallback_reasons) if fallback_reasons else "no_valid_crop"
    return None, None


def _save_detection_overlay(rgb: Image.Image, candidates: list[dict[str, Any]], output_path: Path) -> str:
    overlay = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for idx, candidate in enumerate(candidates, start=1):
        bbox = [int(v) for v in candidate.get("bbox", [0, 0, 0, 0])]
        color = (40, 200, 90) if idx == 1 else (255, 190, 40)
        draw.rectangle(tuple(bbox), outline=color, width=3)
        draw.text((bbox[0] + 4, max(0, bbox[1] - 16)), f"C{idx}", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return str(output_path)


def build_candidate_pool(
    *,
    rgb_path: str | Path,
    detections: list[dict[str, Any]],
    output_dir: str | Path,
    source_detection_query: str,
    pad_px: int = 8,
    erode_px: int = 3,
    min_inner_crop_pixels: int = 1000,
) -> list[dict[str, Any]]:
    out = Path(output_dir)
    pool_dir = out / "candidate_pool"
    rgb = Image.open(rgb_path).convert("RGB")
    candidates: list[dict[str, Any]] = []
    slug = _material_slug(source_detection_query)
    for idx, det in enumerate(detections, start=1):
        mask = np.asarray(det["mask"], dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype("uint8")
        try:
            bbox = [int(v) for v in det.get("bbox") or _bbox_from_mask(mask)]
        except ValueError:
            continue
        candidate_id = f"candidate_{idx:03d}_{slug}"
        mask_path = pool_dir / f"{candidate_id}_mask.png"
        bbox_crop_path = pool_dir / f"{candidate_id}_bbox_crop.png"
        masked_crop_path = pool_dir / f"{candidate_id}_masked_texture_crop.png"
        masked_alpha_path = pool_dir / f"{candidate_id}_masked_texture_crop_alpha.png"
        inner_crop_path = pool_dir / f"{candidate_id}_inner_texture_crop.png"
        _save_mask(mask, mask_path)
        bbox_crop = _crop_rgb(rgb, bbox, pad_px)
        bbox_crop_path.parent.mkdir(parents=True, exist_ok=True)
        bbox_crop.save(bbox_crop_path)
        masked_meta = _save_tight_masked_crop(
            rgb,
            mask,
            masked_crop_path,
            crop_mask=mask,
            pad_px=pad_px,
        )
        _save_masked_crop_alpha(rgb, mask, bbox, masked_alpha_path, pad_px=pad_px)
        inner_meta = _save_inner_texture_crop(
            rgb,
            mask,
            inner_crop_path,
            erode_px=erode_px,
            min_inner_crop_pixels=min_inner_crop_pixels,
        )
        inner_crop_valid = bool(inner_meta.get("success")) and Path(str(inner_meta.get("crop_path"))).exists()
        detection_score = det.get("detection_score", det.get("score", det.get("mask_score")))
        mask_score = det.get("mask_score", det.get("score", detection_score))
        candidate = {
            "candidate_id": candidate_id,
            "source_detection_query": source_detection_query,
            "candidate_generation_source": det.get("candidate_generation_source", "grounded_sam2_requested_class_detection"),
            "mask_path": str(mask_path),
            "bbox": bbox,
            "bbox_crop_path": str(bbox_crop_path),
            "masked_texture_crop_path": str(masked_crop_path) if masked_meta.get("success") else "NA",
            "masked_texture_crop_alpha_path": str(masked_alpha_path),
            "masked_texture_crop_metadata": masked_meta,
            "inner_texture_crop_path": str(inner_crop_path) if inner_crop_valid else "NA",
            "inner_crop_valid": inner_crop_valid,
            "inner_texture_crop_usable": inner_crop_valid,
            "inner_crop_erode_px": int(erode_px),
            "inner_crop_pixel_count": int(inner_meta.get("inner_crop_pixel_count", 0) or 0),
            "inner_texture_foreground_pixels": int(inner_meta.get("inner_crop_pixel_count", 0) or 0),
            "inner_texture_crop_metadata": inner_meta,
            "detection_score": detection_score,
            "detection_score_available": detection_score is not None,
            "score_mapping_failure_reason": None if detection_score is not None else "no_detection_score_or_mask_score_in_detection_payload",
            "mask_score": mask_score,
            "class_name": det.get("class_name", source_detection_query),
        }
        crop_path, crop_type = _choose_verification_crop(candidate)
        candidate["crop_used_for_verification"] = crop_path
        candidate["crop_used_for_verification_path"] = crop_path
        candidate["crop_used_for_verification_type"] = crop_type
        candidates.append(candidate)
    return candidates


def save_candidate_pool_manifest(
    *,
    output_dir: str | Path,
    requested_material: str,
    candidates: list[dict[str, Any]],
) -> str:
    pool_dir = Path(output_dir) / "candidate_pool"
    manifest = {
        "requested_material": requested_material,
        "mode": EXPERIMENT_MODE,
        "candidate_pool_mode": CANDIDATE_POOL_MODE,
        "allow_oracle_known_classes": False,
        "simple_detection_queries_only": True,
        "candidate_count": len(candidates),
        "crop_priority": CROP_PRIORITY,
        "candidates": candidates,
    }
    return _write_json(pool_dir / "candidate_pool_manifest.json", manifest)


def save_candidate_detection_overlay(
    *,
    rgb_path: str | Path,
    output_dir: str | Path,
    requested_material: str,
    candidates: list[dict[str, Any]],
) -> str:
    rgb = Image.open(rgb_path).convert("RGB")
    slug = _material_slug(requested_material)
    return _save_detection_overlay(rgb, candidates, Path(output_dir) / "candidate_pool" / f"detection_{slug}_overlay.png")


def save_empty_candidate_pool_failure(
    *,
    output_dir: str | Path,
    requested_material: str,
    failure_reason: str,
) -> str:
    pool_dir = Path(output_dir) / "candidate_pool"
    manifest = {
        "requested_material": requested_material,
        "mode": EXPERIMENT_MODE,
        "candidate_pool_mode": CANDIDATE_POOL_MODE,
        "allow_oracle_known_classes": False,
        "simple_detection_queries_only": True,
        "candidate_count": 0,
        "crop_priority": CROP_PRIORITY,
        "failure_reason": failure_reason,
        "candidates": [],
    }
    return _write_json(pool_dir / "candidate_pool_manifest.json", manifest)


def render_crop_verification_panel(
    *,
    candidates: list[dict[str, Any]],
    selected_candidate_id: str | None,
    requested_material: str,
    output_path: str | Path,
    columns: int = 2,
    tile_width: int = PANEL_TILE_WIDTH,
    tile_height: int = PANEL_TILE_HEIGHT,
    header_height: int = PANEL_HEADER_HEIGHT,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = max(len(candidates), 1)
    cols = max(1, min(columns, count))
    rows = int(np.ceil(count / cols))
    no_verified_match = str(selected_candidate_id or "") == "NO_VERIFIED_MATCH"
    footer_height = 26 if no_verified_match else 0
    panel = Image.new("RGB", (cols * tile_width, rows * tile_height + footer_height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    boxes_by_id = {
        str(item.get("candidate_id")): item
        for item in crop_panel_candidate_boxes(
            candidates=candidates,
            columns=cols,
            tile_width=tile_width,
            tile_height=tile_height,
            header_height=header_height,
        )
    }
    for idx, candidate in enumerate(candidates, start=1):
        col = (idx - 1) % cols
        row = (idx - 1) // cols
        x0 = col * tile_width
        y0 = row * tile_height
        crop_path = candidate.get("crop_used_for_verification") or candidate.get("crop_path_used_for_scoring")
        crop_type = candidate.get("crop_used_for_verification_type") or candidate.get("crop_type_used_for_scoring") or "unknown"
        box_meta = boxes_by_id.get(str(candidate.get("candidate_id")), {})
        crop_box = box_meta.get("crop_viewport_box") or [
            x0 + PANEL_PADDING,
            y0 + header_height + PANEL_PADDING,
            x0 + tile_width - PANEL_PADDING,
            y0 + tile_height - PANEL_PADDING,
        ]
        crop_area_x, crop_area_y, crop_area_x1, crop_area_y1 = [int(value) for value in crop_box]
        crop_area_w = max(32, crop_area_x1 - crop_area_x)
        crop_area_h = max(32, crop_area_y1 - crop_area_y)
        crop_canvas = Image.new("RGB", (crop_area_w, crop_area_h), (255, 255, 255))
        if crop_path and Path(str(crop_path)).exists():
            crop = Image.open(crop_path).convert("RGB")
            crop_canvas = _resize_crop_to_fill_viewport(crop, crop_area_w, crop_area_h)
        panel.paste(crop_canvas, (crop_area_x, crop_area_y))
        draw.rectangle(
            (crop_area_x, crop_area_y, crop_area_x + crop_area_w - 1, crop_area_y + crop_area_h - 1),
            outline=(185, 185, 185),
            width=1,
        )
        is_selected = (not no_verified_match) and (candidate.get("candidate_id") == selected_candidate_id or bool(candidate.get("final_selected")))
        border = (20, 170, 70) if is_selected else (70, 70, 70)
        draw.rectangle((x0 + 3, y0 + 3, x0 + tile_width - 4, y0 + tile_height - 4), outline=border, width=4 if is_selected else 1)
        req_score = candidate.get("requested_material_score", candidate.get("clip_score_for_requested", ""))
        det_score = candidate.get("detection_score", candidate.get("mask_score", ""))
        try:
            req_score_text = f"{float(req_score):.4f}"
        except Exception:
            req_score_text = str(req_score)
        try:
            det_score_text = f"{float(det_score):.4f}"
        except Exception:
            det_score_text = str(det_score)
        top_label = candidate.get("clip_top_label", candidate.get("top_label", "NA"))
        lines = [
            f"C{idx}: {candidate.get('candidate_id')}",
            f"source={candidate.get('source_detection_query', requested_material)}",
            f"det_score={det_score_text}",
            f"req_score={req_score_text}",
            f"top_label={top_label}",
            f"crop={crop_type}",
            f"mode={EXPERIMENT_MODE}",
        ]
        if is_selected:
            lines.append("SELECTED")
        max_chars = max(18, int((tile_width - 20) / 6))
        for line_idx, line in enumerate(lines):
            text = str(line)
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            y = y0 + 10 + line_idx * 13
            if y + 12 >= crop_area_y - 2:
                break
            draw.text((x0 + 10, y), text, fill=(0, 0, 0), font=font)
    if not candidates:
        draw.text((10, 10), f"No candidates for {requested_material}", fill=(160, 0, 0), font=font)
    if no_verified_match:
        y = rows * tile_height + 7
        draw.text(
            (10, y),
            f"NO_VERIFIED_MATCH - requested material not confidently present: {requested_material}",
            fill=(170, 0, 0),
            font=font,
        )
    panel.save(output)
    return str(output)
