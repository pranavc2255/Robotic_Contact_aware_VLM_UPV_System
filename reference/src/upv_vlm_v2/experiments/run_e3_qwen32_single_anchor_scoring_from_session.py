#!/usr/bin/env python3
"""
E3 Qwen32 single-anchor scoring experiment.

This script reuses an existing E3 anchor-selection session, renders one source
anchor at a time as clean TOP/BOTTOM contact crops, sends each image
independently to a Qwen server, ranks the source anchors from those independent
scores, and compares the selected anchor against manual labels.

No robot, RTDE, Arduino, clamp, live camera, or dataset capture is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from upv_vlm_v2.experiments.e3_anchor_crop_utils import (
    DEFAULT_E3_CROP_CONFIG,
    render_clean_single_anchor_input,
)
from upv_vlm_v2.experiments.run_e3_qwen32_shuffle_consensus_from_session import (
    anchor_sort_key,
    canonical_tile_paths,
    discover_cases,
    extract_response,
    load_deterministic_features,
    load_manual_labels,
    post_json,
    safe_float,
    write_csv,
    write_json,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": "outputs/v2_experiments/e3_qwen32_single_anchor_scoring",
    "server_url": "http://127.0.0.1:8899",
    "server_endpoint": "/infer",
    "model_name": "Qwen2.5-VL-32B-Instruct",
    "model_dir": "local_models/Qwen2.5-VL-32B-Instruct",
    "max_new_tokens": 700,
    "temperature": 0.0,
    "timeout_sec": 240,
    "copy_source_tiles": True,
    **DEFAULT_E3_CROP_CONFIG,
    "fail_fast": False,
    "case_glob": "case_*",
}


PROMPT_VERSION = "e3_qwen32_single_anchor_upv_contact_score_v1"


PROMPT_TEMPLATE = """You are scoring one physical UPV contact anchor.

Source anchor ID: {anchor_id}

This is a physical UPV contact-affordance scoring task. Evaluate this ONE source anchor only. Do not compare it to other anchors.

The image shows two clean inspection crops side by side:
- Left panel = Top contact crop
- Right panel = Bottom contact crop

The full TOP crop and full BOTTOM crop are the inspection regions. The entire TOP crop is the intended top contact region. The entire BOTTOM crop is the intended bottom contact region. These are the exact regions where flat UPV transducers will touch.

Judge the full visible crop area in each panel. Do not rely on overlays, colored markers, guide lines, or UI decoration.

Each crop intentionally includes outside-of-object context beyond the edge. Inspect the edge boundary and the outside/background side near the edge. Defects may appear exactly at the edge or just outside the edge and still affect contact usability.

A good anchor requires BOTH TOP and BOTTOM strips to be clean, continuous, open, approximately planar, and stable for flat probe contact.

Penalize these visible contact risks in either crop:
- raised or attached material entering from the edge
- mortar-like crust or blobs
- vertical seam, crack, line, or material boundary crossing the crop
- vertical seam or crack crossing the contact region
- abrupt color or texture change along the contact region suggesting mortar start, residue, attached material, chip, nonuniform surface, or unstable contact
- gray/white crust, mortar-like buildup, blobs, protrusions, debris, dust pile, nail, tape, splinter, or foreign object
- chipped, broken, jagged, missing, or non-planar contact edge
- visible non-planarity that would prevent stable flat probe contact

A candidate is usable only if BOTH top and bottom contact crops are usable. A candidate is poor if either TOP or BOTTOM crop is poor. If all anchors are imperfect, the downstream code will choose the least bad anchor from your independent scores.

Material awareness:
- Brick: normal red/brown/tan/beige/yellowish fired-clay texture, pores, speckles, dust, and roughness are not defects by themselves. Gray/white attached cement-like material, mortar, crust, protruding material, missing chip, jagged break, abrupt boundary, or debris inside the crop is a defect.
- Timber: normal grain, knots, color bands, and saw marks are not defects by themselves. Nails, screws, staples, tape, splinters, cracks, protruding chips, debris, or blocked strips are defects.
- Concrete/cinder block: normal gray cementitious texture, aggregate, pores, and roughness are not defects by themselves. Loose debris, protruding attached paste, broken/jagged edge, missing material, local obstruction, or material sticking out into the contact strip is a defect.

Scoring:
- 90-100: both strips are clean, continuous, planar, open, and clearly physically usable.
- 70-89: usable; only minor normal material texture or slight roughness; no obstruction.
- 40-69: questionable; minor contact uncertainty, limited visible support, nonuniformity, edge/corner/cutoff risk, or one side less reliable.
- 0-39: bad; blocked, chipped, debris-covered, mortar/crust/protrusion, broken/jagged, corner/cutoff-limited, or one side unusable.

Rules:
- If top_usable=false or bottom_usable=false, overall_usable should normally be false.
- If overall_usable=false, score should normally be below 40.
- If the image is uncertain, choose a lower score and explain the uncertainty.
- Do not give a generic reason; mention actual visible evidence.
- Return JSON only. Do not include markdown fences or text outside JSON.

Use exactly this compact JSON schema:
{{
  "anchor_id": "{anchor_id}",
  "top_defects": ["none"],
  "bottom_defects": ["none"],
  "top_usable": true,
  "bottom_usable": true,
  "overall_usable": true,
  "score": 0,
  "reason": "short physical-contact reason"
}}
"""


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as dict: {path}")

    cfg.update(data)
    return cfg


def build_prompt(anchor_id: str) -> str:
    return PROMPT_TEMPLATE.format(anchor_id=anchor_id)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _crop_from_bounds(image: Image.Image, bounds: Any) -> Image.Image | None:
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


def _median_spacing(values: list[float]) -> float | None:
    xs = sorted(values)
    diffs = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if not diffs:
        return None
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return float(diffs[mid])
    return float((diffs[mid - 1] + diffs[mid]) / 2.0)


def _anchor_x_positions(candidate_map: dict[str, Any]) -> dict[str, float]:
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


def _local_lateral_bounds(
    *,
    anchor_id: str,
    x_positions: dict[str, float],
    image_width: int,
    cfg: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    center_x = float(x_positions[anchor_id])
    ordered = sorted(x_positions.items(), key=lambda kv: kv[1])
    ids = [aid for aid, _ in ordered]
    xs = [float(x) for _, x in ordered]
    idx = ids.index(anchor_id)
    spacing = _median_spacing(xs) or 64.0

    if idx > 0:
        zone_left = (xs[idx - 1] + xs[idx]) / 2.0
    else:
        zone_left = 0.0
    if idx < len(xs) - 1:
        zone_right = (xs[idx] + xs[idx + 1]) / 2.0
    else:
        zone_right = float(image_width)

    fraction = float(cfg.get("single_anchor_lateral_span_fraction_of_spacing", 0.82))
    min_span = float(cfg.get("single_anchor_min_lateral_span_px", 46))
    max_span = float(cfg.get("single_anchor_max_lateral_span_px", 72))
    desired_span = max(min_span, min(max_span, spacing * fraction))
    half = desired_span / 2.0
    x0 = max(zone_left, center_x - half)
    x1 = min(zone_right, center_x + half)

    # If midpoint clipping made an endpoint crop too narrow, expand only inside
    # its local zone. This preserves locality and avoids spilling into neighbors.
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


def _edge_normal_bounds(
    *,
    edge_y: float,
    image_height: int,
    side: str,
    cfg: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
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


def _load_metadata_contact_crops(
    artifact_dir: Path | None,
    anchor_id: str,
    cfg: dict[str, Any],
) -> tuple[Image.Image | None, Image.Image | None, dict[str, Any]]:
    if artifact_dir is None:
        return None, None, {"crop_source": "missing_artifact_dir"}

    rotation_meta_path = artifact_dir / "rotation_metadata.json"
    rotation_meta = read_json(rotation_meta_path, default={})
    if not isinstance(rotation_meta, dict):
        return None, None, {"crop_source": "missing_rotation_metadata", "rotation_metadata_path": str(rotation_meta_path)}

    rotated_path = None
    rotated_artifacts = rotation_meta.get("rotated_artifacts")
    using_full_rotated = False
    if isinstance(rotated_artifacts, dict):
        rotated_path = rotated_artifacts.get("rotated_object_rgb") or rotated_artifacts.get("cropped_rotated_object_rgb")
        using_full_rotated = bool(rotated_artifacts.get("rotated_object_rgb"))
    if not rotated_path:
        rotated_path = str(artifact_dir / "cropped_rotated_object_rgb.png")

    rotated_image_path = Path(str(rotated_path))
    if not rotated_image_path.is_absolute() and not rotated_image_path.exists():
        rotated_image_path = artifact_dir / rotated_image_path.name
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
    x_positions = _anchor_x_positions(candidate_map)
    if coord_offset_x or coord_offset_y:
        x_positions = {aid: x + coord_offset_x for aid, x in x_positions.items()}
    top_hit = patch_meta.get("top_hit_xy_rotated")
    bottom_hit = patch_meta.get("bottom_hit_xy_rotated")

    if (
        anchor_id not in x_positions
        or not isinstance(top_hit, (list, tuple))
        or not isinstance(bottom_hit, (list, tuple))
        or len(top_hit) < 2
        or len(bottom_hit) < 2
    ):
        top = _crop_from_bounds(rotated, patch_meta.get("top_patch_bounds"))
        bottom = _crop_from_bounds(rotated, patch_meta.get("bottom_patch_bounds"))
        return top, bottom, {
            "crop_source": "rotation_metadata_legacy_patch_bounds",
            "rotation_metadata_path": str(rotation_meta_path),
            "rotated_image_path": str(rotated_image_path),
            "top_patch_bounds": patch_meta.get("top_patch_bounds"),
            "bottom_patch_bounds": patch_meta.get("bottom_patch_bounds"),
        }

    x0, x1, x_meta = _local_lateral_bounds(
        anchor_id=anchor_id,
        x_positions=x_positions,
        image_width=rotated.width,
        cfg=cfg,
    )
    top_y0, top_y1, top_y_meta = _edge_normal_bounds(
        edge_y=float(top_hit[1]) + coord_offset_y,
        image_height=rotated.height,
        side="top",
        cfg=cfg,
    )
    bottom_y0, bottom_y1, bottom_y_meta = _edge_normal_bounds(
        edge_y=float(bottom_hit[1]) + coord_offset_y,
        image_height=rotated.height,
        side="bottom",
        cfg=cfg,
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
        **x_meta,
        **top_y_meta,
        **bottom_y_meta,
    }


def _fallback_tile_contact_crops(source_tile_path: Path) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    tile = Image.open(source_tile_path).convert("RGB")
    top_y, bottom_y = _detect_contact_strip_centers(tile)
    tile = remove_artificial_guide_lines(tile, {"remove_yellow_guide_lines": True, "yellow_guide_inpaint_radius_px": 3})
    split_y = max(1, min(tile.height - 1, (top_y + bottom_y) // 2))
    top_margin = max(35, int(round((bottom_y - top_y) * 0.48)))
    bottom_margin = top_margin
    top = tile.crop((0, max(0, top_y - top_margin), tile.width, min(split_y, top_y + top_margin))).convert("RGB")
    bottom = tile.crop((0, max(split_y, bottom_y - bottom_margin), tile.width, min(tile.height, bottom_y + bottom_margin))).convert("RGB")
    return top, bottom, {
        "crop_source": "fallback_source_tile_halves",
        "top_contact_strip_center_y_px_source_tile": top_y,
        "bottom_contact_strip_center_y_px_source_tile": bottom_y,
    }


def _resize_crop(im: Image.Image, upscale: float) -> Image.Image:
    if upscale <= 0:
        upscale = 1.0
    if abs(upscale - 1.0) < 1e-6:
        return im
    new_size = (max(1, int(round(im.width * upscale))), max(1, int(round(im.height * upscale))))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def render_single_anchor_input(
    source_tile_path: Path,
    anchor_id: str,
    out_path: Path,
    cfg: dict[str, Any],
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for checks/importers."""
    return render_clean_single_anchor_input(
        source_tile_path=source_tile_path,
        anchor_id=anchor_id,
        output_path=out_path,
        config=cfg,
        artifact_dir=artifact_dir,
    )


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["none"]
    if value is None or value == "":
        return ["none"]
    return [str(value).strip()]


def normalize_single_anchor_response(parsed: dict[str, Any] | None, expected_anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    if parsed is None:
        return {"anchor_id": expected_anchor_id, "parse_ok": False}, False, "missing parsed object"

    anchor_id = str(parsed.get("anchor_id") or expected_anchor_id).strip()
    if anchor_id != expected_anchor_id:
        return {
            "anchor_id": anchor_id,
            "expected_anchor_id": expected_anchor_id,
            "parse_ok": False,
            "raw_parsed": parsed,
        }, False, f"anchor_id mismatch: expected {expected_anchor_id}, got {anchor_id}"

    score = safe_float(parsed.get("score"))
    top_usable = _as_bool(parsed.get("top_usable"))
    bottom_usable = _as_bool(parsed.get("bottom_usable"))
    overall_usable = _as_bool(parsed.get("overall_usable"))

    if score is None:
        return {
            "anchor_id": expected_anchor_id,
            "parse_ok": False,
            "raw_parsed": parsed,
        }, False, "missing numeric score"

    if top_usable is None:
        top_usable = False
    if bottom_usable is None:
        bottom_usable = False
    if overall_usable is None:
        overall_usable = bool(top_usable and bottom_usable and score >= 70)

    norm = {
        "anchor_id": expected_anchor_id,
        "top_defects": _string_list(parsed.get("top_defects")),
        "bottom_defects": _string_list(parsed.get("bottom_defects")),
        "top_usable": bool(top_usable),
        "bottom_usable": bool(bottom_usable),
        "overall_usable": bool(overall_usable),
        "score": float(score),
        "reason": str(parsed.get("reason") or "").strip(),
        "parse_ok": True,
        "raw_parsed": parsed,
    }
    return norm, True, ""


def rank_single_anchor_scores(
    source_ids: list[str],
    anchor_results: dict[str, dict[str, Any]],
    deterministic_features: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    deterministic_features = deterministic_features or {}
    parsed_ids = [aid for aid in source_ids if anchor_results.get(aid, {}).get("parse_ok")]

    def _key(aid: str) -> tuple:
        r = anchor_results[aid]
        det = deterministic_features.get(aid, {})
        det_score = safe_float(det.get("deterministic_score"))
        return (
            0 if bool(r.get("overall_usable")) else 1,
            -float(r.get("score") or -math.inf),
            -(det_score if det_score is not None else -math.inf),
            anchor_sort_key(aid),
        )

    return sorted(parsed_ids, key=_key)


def manual_eval(
    case_id: str,
    selected_anchor: str | None,
    labels: dict[tuple[str, str], str],
    manual_best: dict[str, str],
) -> dict[str, Any]:
    label = labels.get((case_id, selected_anchor or ""), "")
    if selected_anchor in {None, "", "NO_PARSE_RESULT"}:
        label = ""
    label_norm = label.strip().lower()
    best_anchor = manual_best.get(case_id, "")

    return {
        "manual_label": label,
        "manual_best_anchor": best_anchor,
        "is_top1_usable": label_norm in {"good", "acceptable", "accept", "ok"},
        "is_manual_best_match": bool(selected_anchor and best_anchor and selected_anchor == best_anchor),
        "bad_anchor_selected": label_norm == "bad",
    }


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def write_anchor_score_table(path: Path, source_ids: list[str], anchor_results: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for aid in source_ids:
        r = anchor_results.get(aid, {"anchor_id": aid, "parse_ok": False})
        rows.append(
            {
                "anchor_id": aid,
                "parse_ok": r.get("parse_ok", False),
                "score": r.get("score", ""),
                "overall_usable": r.get("overall_usable", ""),
                "top_usable": r.get("top_usable", ""),
                "bottom_usable": r.get("bottom_usable", ""),
                "top_defects": _join_list(r.get("top_defects")),
                "bottom_defects": _join_list(r.get("bottom_defects")),
                "reason": r.get("reason", ""),
                "parse_error": r.get("parse_error", ""),
                "rendered_input_path": r.get("rendered_input_path", ""),
                "raw_response_path": r.get("raw_response_path", ""),
                "parsed_response_path": r.get("parsed_response_path", ""),
            }
        )
    write_csv(
        path,
        rows,
        [
            "anchor_id",
            "parse_ok",
            "score",
            "overall_usable",
            "top_usable",
            "bottom_usable",
            "top_defects",
            "bottom_defects",
            "reason",
            "parse_error",
            "rendered_input_path",
            "raw_response_path",
            "parsed_response_path",
        ],
    )


def run_case(
    case: Any,
    output_case_dir: Path,
    cfg: dict[str, Any],
    labels: dict[tuple[str, str], str],
    manual_best: dict[str, str],
) -> dict[str, Any]:
    output_case_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = canonical_tile_paths(case.tile_dir)
    if not tile_paths:
        raise RuntimeError(f"No canonical tiles found: {case.tile_dir}")

    source_ids = list(tile_paths.keys())
    input_dir = output_case_dir / "single_anchor_inputs"
    response_dir = output_case_dir / "qwen32_responses"
    input_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("copy_source_tiles", True)):
        source_tile_dir = output_case_dir / "source_tiles"
        source_tile_dir.mkdir(parents=True, exist_ok=True)
        for _, p in tile_paths.items():
            shutil.copy2(p, source_tile_dir / p.name)

    write_json(
        output_case_dir / "case_input_manifest.json",
        {
            "case_name": case.case_name,
            "case_id": case.case_id,
            "case_index": case.case_index,
            "material": case.material,
            "condition": case.condition,
            "source_artifact_dir": str(case.artifact_dir),
            "source_tile_dir": str(case.tile_dir),
            "source_ids": source_ids,
            "model_name": cfg.get("model_name"),
            "model_dir": cfg.get("model_dir"),
            "prompt_version": PROMPT_VERSION,
        },
    )

    server_url = str(cfg.get("server_url", "http://127.0.0.1:8899")).rstrip("/")
    endpoint = str(cfg.get("server_endpoint", "/infer"))
    infer_url = server_url + endpoint
    timeout_sec = int(cfg.get("timeout_sec", 240))

    anchor_results: dict[str, dict[str, Any]] = {}

    for anchor_id, tile_path in tile_paths.items():
        prompt = build_prompt(anchor_id)
        prompt_path = response_dir / f"source_{anchor_id}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        image_path = input_dir / f"source_{anchor_id}.png"
        canonical_clean = case.artifact_dir / "clean_single_anchor_inputs" / f"source_{anchor_id}.png"
        if canonical_clean.exists():
            shutil.copy2(canonical_clean, image_path)
            render_meta = {
                "source_tile_path": str(tile_path),
                "rendered_input_path": str(image_path),
                "canonical_clean_input_path": str(canonical_clean),
                "crop_source": "main_anchor_selection_clean_artifact",
            }
        else:
            render_meta = render_clean_single_anchor_input(
                source_tile_path=tile_path,
                anchor_id=anchor_id,
                output_path=image_path,
                config=cfg,
                artifact_dir=case.artifact_dir,
            )

        server_response_path = response_dir / f"source_{anchor_id}_server_response.json"
        payload = {
            "image_path": str(image_path),
            "prompt_text": prompt,
            "prompt_version": PROMPT_VERSION,
            "max_new_tokens": int(cfg.get("max_new_tokens", 700)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "output_path": str(server_response_path),
        }

        t0 = time.perf_counter()
        response = post_json(infer_url, payload, timeout_sec)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        write_json(server_response_path, response)

        raw_text, parsed, parse_ok, parse_error = extract_response(response)
        raw_path = response_dir / f"source_{anchor_id}_raw.txt"
        parsed_path = response_dir / f"source_{anchor_id}_parsed.json"
        raw_path.write_text(raw_text or "", encoding="utf-8")

        normalized, norm_ok, norm_error = normalize_single_anchor_response(parsed, anchor_id)
        parse_ok = bool(parse_ok and norm_ok)
        final_error = "" if parse_ok else (norm_error or parse_error)

        normalized.update(
            {
                "parse_ok": parse_ok,
                "parse_error": final_error,
                "elapsed_ms": elapsed_ms,
                "rendered_input_path": str(image_path),
                "raw_response_path": str(raw_path),
                "parsed_response_path": str(parsed_path),
                "server_response_path": str(server_response_path),
                "prompt_path": str(prompt_path),
                **render_meta,
            }
        )

        write_json(parsed_path, normalized if parse_ok else {**normalized, "raw_parsed": parsed})
        anchor_results[anchor_id] = normalized

    deterministic_features = load_deterministic_features(case.artifact_dir)
    ranked = rank_single_anchor_scores(source_ids, anchor_results, deterministic_features)
    selected_anchor = ranked[0] if ranked else "NO_PARSE_RESULT"
    parse_count = sum(1 for r in anchor_results.values() if r.get("parse_ok"))

    write_anchor_score_table(output_case_dir / "anchor_score_table.csv", source_ids, anchor_results)

    selected_result = anchor_results.get(selected_anchor, {}) if selected_anchor != "NO_PARSE_RESULT" else {}
    selected_eval = manual_eval(case.case_id, selected_anchor, labels, manual_best)

    result = {
        "case_name": case.case_name,
        "case_id": case.case_id,
        "case_index": case.case_index,
        "material": case.material,
        "condition": case.condition,
        "num_source_anchors": len(source_ids),
        "anchors_attempted": len(source_ids),
        "anchors_parsed": parse_count,
        "all_anchors_parsed": parse_count == len(source_ids),
        "parse_failure": parse_count == 0,
        "selected_anchor": selected_anchor,
        "ranked_source_anchors": "|".join(ranked),
        "selected_score": selected_result.get("score", ""),
        "selected_overall_usable": selected_result.get("overall_usable", ""),
        "selected_reason": selected_result.get("reason", ""),
        "selected_top_defects": _join_list(selected_result.get("top_defects")),
        "selected_bottom_defects": _join_list(selected_result.get("bottom_defects")),
        "selected_manual_label": selected_eval["manual_label"],
        "manual_best_anchor": selected_eval["manual_best_anchor"],
        "is_top1_usable": selected_eval["is_top1_usable"],
        "is_manual_best_match": selected_eval["is_manual_best_match"],
        "bad_anchor_selected": selected_eval["bad_anchor_selected"],
        "case_output_dir": str(output_case_dir),
        "anchor_score_table_path": str(output_case_dir / "anchor_score_table.csv"),
    }

    write_json(
        output_case_dir / "case_result.json",
        {
            **result,
            "anchor_results": anchor_results,
            "deterministic_features": deterministic_features,
        },
    )
    return result


def _truthy(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key)).strip().lower() == "true"


def summarize_rows(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    anchors_attempted = sum(int(r.get("anchors_attempted") or 0) for r in rows)
    anchors_parsed = sum(int(r.get("anchors_parsed") or 0) for r in rows)
    cases_any = sum(int(r.get("anchors_parsed") or 0) > 0 for r in rows)
    cases_all = sum(str(r.get("all_anchors_parsed")).lower() == "true" for r in rows)
    usable = sum(_truthy(r, "is_top1_usable") for r in rows)
    best = sum(_truthy(r, "is_manual_best_match") for r in rows)
    bad = sum(_truthy(r, "bad_anchor_selected") for r in rows)

    denom = cases_any if cases_any else 0
    return {
        "n_cases": n,
        "n_errors": len(errors),
        "anchors_attempted": anchors_attempted,
        "anchors_parsed": anchors_parsed,
        "anchor_parse_success_rate": anchors_parsed / anchors_attempted if anchors_attempted else None,
        "cases_with_any_parse_success": cases_any,
        "cases_with_all_anchors_parsed": cases_all,
        "top1_usable_count": usable,
        "top1_usable_rate": usable / denom if denom else None,
        "manual_best_match_count": best,
        "manual_best_match_rate": best / denom if denom else None,
        "bad_anchor_selected_count": bad,
        "bad_anchor_selected_rate": bad / denom if denom else None,
    }


def group_summary(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_field) or "")].append(r)

    out: list[dict[str, Any]] = []
    for group_name, group_rows in sorted(groups.items()):
        s = summarize_rows(group_rows, [])
        s[group_field] = group_name
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="Existing E3 anchor-selection session dir")
    ap.add_argument("--manual-labels", default="", help="manual_anchor_labels.csv")
    ap.add_argument("--config", default="", help="YAML config")
    ap.add_argument("--server-url", default="", help="Override server URL")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--case-id", default="", help="Optional comma-separated case IDs")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    if args.server_url:
        cfg["server_url"] = args.server_url
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.fail_fast:
        cfg["fail_fast"] = True

    session_dir = Path(args.session)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    manual_labels_path = Path(args.manual_labels) if args.manual_labels else None
    labels, manual_best = load_manual_labels(manual_labels_path)

    output_session = Path(str(cfg["output_root"])) / f"session_{now_stamp()}"
    output_session.mkdir(parents=True, exist_ok=False)

    selected_case_ids = {x.strip() for x in args.case_id.split(",") if x.strip()}
    cases = discover_cases(session_dir, str(cfg.get("case_glob", "case_*")))
    if selected_case_ids:
        cases = [c for c in cases if c.case_id in selected_case_ids]

    write_json(
        output_session / "run_manifest.json",
        {
            "phase": "e3_qwen32_single_anchor_scoring_from_existing_session",
            "created": datetime.now().isoformat(),
            "input_session": str(session_dir),
            "manual_labels": str(manual_labels_path) if manual_labels_path else None,
            "output_session": str(output_session),
            "config": cfg,
            "num_cases": len(cases),
            "case_ids": [c.case_id for c in cases],
            "prompt_version": PROMPT_VERSION,
        },
    )
    write_json(output_session / "effective_config.json", cfg)

    print("output_session:", output_session)
    print("num_cases:", len(cases))
    print("server_url:", cfg.get("server_url"))
    print("model_dir:", cfg.get("model_dir"))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.case_id}")
        output_case_dir = output_session / "cases" / case.case_name

        try:
            row = run_case(case, output_case_dir, cfg, labels, manual_best)
            rows.append(row)
            print(
                "  selected=",
                row["selected_anchor"],
                row["selected_manual_label"],
                "manual_best=",
                row["manual_best_anchor"],
                "parsed=",
                f"{row['anchors_parsed']}/{row['anchors_attempted']}",
            )
        except Exception as exc:
            err = {
                "case_id": case.case_id,
                "case_name": case.case_name,
                "error": repr(exc),
                "case_output_dir": str(output_case_dir),
            }
            errors.append(err)
            print("[ERROR]", err, file=sys.stderr)
            if bool(cfg.get("fail_fast", False)):
                raise

    master_fields = [
        "case_name",
        "case_id",
        "case_index",
        "material",
        "condition",
        "num_source_anchors",
        "anchors_attempted",
        "anchors_parsed",
        "all_anchors_parsed",
        "parse_failure",
        "selected_anchor",
        "ranked_source_anchors",
        "selected_score",
        "selected_overall_usable",
        "selected_reason",
        "selected_top_defects",
        "selected_bottom_defects",
        "selected_manual_label",
        "manual_best_anchor",
        "is_top1_usable",
        "is_manual_best_match",
        "bad_anchor_selected",
        "case_output_dir",
        "anchor_score_table_path",
    ]

    write_csv(output_session / "master_single_anchor_results.csv", rows, master_fields)
    write_csv(output_session / "errors.csv", errors, ["case_id", "case_name", "error", "case_output_dir"])

    summary = {
        "phase": "e3_qwen32_single_anchor_scoring_from_existing_session",
        "input_session": str(session_dir),
        "manual_labels": str(manual_labels_path) if manual_labels_path else None,
        "output_session": str(output_session),
        **summarize_rows(rows, errors),
    }
    write_json(output_session / "summary_overall.json", summary)
    write_csv(output_session / "summary_by_material.csv", group_summary(rows, "material"))
    write_csv(output_session / "summary_by_condition.csv", group_summary(rows, "condition"))

    paper = output_session / "paper_ready_e3_single_anchor_results"
    paper.mkdir(exist_ok=True)
    for name in [
        "master_single_anchor_results.csv",
        "summary_overall.json",
        "summary_by_material.csv",
        "summary_by_condition.csv",
    ]:
        src = output_session / name
        if src.exists():
            shutil.copy2(src, paper / name)

    print("\nDONE")
    print("output_session:", output_session)
    print("summary:", output_session / "summary_overall.json")
    print("master:", output_session / "master_single_anchor_results.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
