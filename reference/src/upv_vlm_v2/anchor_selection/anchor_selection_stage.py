"""Anchor selection stage for v2."""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from upv_vlm_v2.anchor_selection.axis_parameterization import build_axis_search_domain
from upv_vlm_v2.anchor_selection.cross_section_sampler import compute_anchor_cross_section_hits
from upv_vlm_v2.anchor_selection.edge_quality import score_edge_quality
from upv_vlm_v2.anchor_selection.edge_pair_crop_builder import map_candidate_points_to_rotated_crop_frame
from upv_vlm_v2.anchor_selection.hybrid_selector import select_anchor_hybrid
from upv_vlm_v2.anchor_selection.partition_contact_crop_builder import (
    build_clean_anchor_review_grid,
    render_clean_anchor_inputs_for_artifact_dir,
)
from upv_vlm_v2.anchor_selection.contact_geometry_refinement import (
    build_contact_geometry_refinement,
    compute_contact_geometry_quality,
)
from upv_vlm_v2.anchor_selection.rotated_object_view_builder import (
    crop_rotated_object_view,
    render_rotated_object_overlay,
    rotate_image_and_mask_to_canonical,
)
from upv_vlm_v2.anchor_selection.true_edge_pair_patch_builder import build_true_edge_pair_patch
from upv_vlm_v2.anchor_selection.wide_contact_tiles import build_boxed_candidate_tile_grid, build_boxed_wide_contact_tile_from_bounds
from upv_vlm_v2.geometry.mask_geometry import load_mask
from upv_vlm_v2.geometry.overlays import draw_mask_overlay, load_rgb, save_anchor_overlay
from upv_vlm_v2.vlm.qwen_anchor_selector import run_required_qwen_anchor_selection
from upv_vlm_v2.anchor_selection.qwen32_single_anchor_contact_scorer import run_qwen32_single_anchor_contact_scoring
from upv_vlm_v2.anchor_selection.qwen32_multi_anchor_l6_batch_ranker import (
    BACKEND_NAME as QWEN32_MULTI_ANCHOR_L6_BACKEND,
    run_qwen32_multi_anchor_l6_batch_ranking,
)
from upv_vlm_v2.anchor_selection.llama32_single_anchor_l6_taxonomy_scorer import (
    BACKEND_NAME as LLAMA32_SINGLE_ANCHOR_L6_BACKEND,
    run_llama32_single_anchor_l6_taxonomy_scoring,
)


ANCHOR_COUNT_RULE_NAME = "physical_axis_length_floor_transducer_diameter"


def _axis_for_mode(axis_mode: str, geometry: dict[str, Any]) -> tuple[list[float], list[float], float]:
    if axis_mode == "minor":
        chosen = geometry["minor_axis_vector"]
        cross = geometry["major_axis_vector"]
        extent = float(geometry.get("minor_axis_length_px") or 0.0)
    else:
        chosen = geometry["major_axis_vector"]
        cross = geometry["minor_axis_vector"]
        extent = float(geometry.get("major_axis_length_px") or 0.0)
    return list(chosen), list(cross), extent


def _dominant_rectangle_for_anchor_mode(geometry: dict[str, Any], axis_mode: str) -> dict[str, Any]:
    rect = geometry.get("dominant_rectangle") or {}
    if not isinstance(rect, dict) or not rect.get("success"):
        return {}
    if str(geometry.get("body_geometry_mode")) not in {"dominant_rectangle", "contact_core_rectangle"}:
        return {}
    return rect


def _json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _get_axis_length_mm(geometry: dict[str, Any], axis_mode: str) -> tuple[float | None, str | None]:
    if axis_mode == "minor":
        keys = [
            "minor_axis_length_mm",
            "global_minor_dimension_mm",
            "minor_dimension_mm",
            "diagnostics.minor_axis_length_mm",
            "diagnostics.global_minor_dimension_mm",
            "diagnostics.minor_dimension_mm",
        ]
    else:
        keys = [
            "major_axis_length_mm",
            "global_major_dimension_mm",
            "major_dimension_mm",
            "diagnostics.major_axis_length_mm",
            "diagnostics.global_major_dimension_mm",
            "diagnostics.major_dimension_mm",
        ]
    for key in keys:
        value = _nested_get(geometry, key) if "." in key else geometry.get(key)
        try:
            if value is None:
                continue
            length = float(value)
            if math.isfinite(length):
                return length, key
        except (TypeError, ValueError):
            continue
    return None, None


def _compute_previous_v2a_anchor_count_metadata(geometry: dict[str, Any], requested_count: int, axis_mode: str, margin_ratio: float) -> dict[str, Any]:
    raw_major = float(geometry.get("major_axis_length_px") or 0.0)
    raw_minor = float(geometry.get("minor_axis_length_px") or 0.0)
    usable_major = raw_major * max(0.0, 1.0 - 2.0 * float(margin_ratio))
    usable_minor = raw_minor * max(0.0, 1.0 - 2.0 * float(margin_ratio))
    if axis_mode == "major":
        generated = max(1, int(requested_count))
        fallback = False
    elif usable_major <= 1e-6 or usable_minor <= 1e-6:
        generated = 1
        fallback = True
    else:
        floor_count = int(math.floor(float(requested_count) * (usable_minor / usable_major)))
        generated = max(1, floor_count)
        fallback = floor_count < 1
    return {
        "requested_num_anchors": int(requested_count),
        "axis_mode": axis_mode,
        "raw_major_length_px": raw_major,
        "raw_minor_length_px": raw_minor,
        "usable_major_length_px": usable_major,
        "usable_minor_length_px": usable_minor,
        "usable_axis_ratio_minor_over_major": (usable_minor / usable_major) if usable_major > 1e-6 else None,
        "anchor_count_generated": int(generated),
        "anchor_count_rule_name": "usable_axis_floor_ratio_min1",
        "center_anchor_fallback_used": bool(fallback),
    }


def _compute_v2a_anchor_count_metadata(
    geometry: dict[str, Any],
    requested_count: int,
    axis_mode: str,
    margin_ratio: float,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    mode = str(cfg.get("candidate_count_mode", "axis_proportional")).strip()
    if mode != "transducer_diameter_floor":
        metadata = _compute_previous_v2a_anchor_count_metadata(geometry, requested_count, axis_mode, margin_ratio)
        metadata["candidate_count_mode"] = mode
        return metadata

    axis_length_mm, source_key = _get_axis_length_mm(geometry, axis_mode)
    transducer_diameter_mm = float(cfg.get("transducer_diameter_mm", 50.0))
    min_samples = int(cfg.get("min_anchor_samples", 0))
    max_samples = int(cfg.get("max_anchor_samples", 20))
    if axis_length_mm is None or axis_length_mm < 0 or transducer_diameter_mm <= 0:
        fallback = _compute_previous_v2a_anchor_count_metadata(geometry, requested_count, axis_mode, margin_ratio)
        fallback.update(
            {
                "candidate_count_mode": mode,
                "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
                "fallback_used": True,
                "fallback_reason": "missing_physical_axis_length_mm" if axis_length_mm is None else "invalid_transducer_or_axis_length",
                "missing_physical_axis_length_mm": axis_length_mm is None,
                "physical_axis_length_mm": axis_length_mm,
                "physical_axis_length_source_key": source_key,
                "transducer_diameter_mm": transducer_diameter_mm,
            }
        )
        return fallback

    raw_floor_count = int(math.floor(axis_length_mm / transducer_diameter_mm))
    generated = raw_floor_count
    generated = max(min_samples, generated)
    generated = min(max_samples, generated)
    return {
        "requested_num_anchors": int(requested_count),
        "axis_mode": axis_mode,
        "candidate_count_mode": mode,
        "physical_axis_length_mm": axis_length_mm,
        "physical_axis_length_source_key": source_key,
        "transducer_diameter_mm": transducer_diameter_mm,
        "raw_floor_count": raw_floor_count,
        "min_anchor_samples": min_samples,
        "max_anchor_samples": max_samples,
        "anchor_count_generated": int(generated),
        "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
        "fallback_used": False,
        "center_anchor_fallback_used": False,
        "raw_major_length_px": float(geometry.get("major_axis_length_px") or 0.0),
        "raw_minor_length_px": float(geometry.get("minor_axis_length_px") or 0.0),
    }


def _feature_record(candidate: dict[str, Any]) -> dict[str, Any]:
    q = candidate.get("quality") or {}
    contact_quality = candidate.get("contact_geometry_quality") or {}
    side_quality = float(q.get("side_quality_score", 0.0) or 0.0)
    paired = float(q.get("paired_consistency_score", 0.0) or 0.0)
    roughness = round(max(0.0, 100.0 - side_quality), 3)
    return {
        "anchor_id": candidate["anchor_id"],
        "deterministic_score": candidate.get("score"),
        "edge_straightness": q.get("span_score"),
        "edge_roughness": roughness,
        "local_edge_consistency": paired,
        "contact_context_score": side_quality,
        "mortar_anomaly_score": roughness,
        "paired_consistency": paired,
        "side_quality": side_quality,
        "veto": bool(candidate.get("veto")),
        "veto_reasons": candidate.get("veto_reasons", []),
        "anchor_px": candidate.get("anchor_px"),
        "contact_point_a_px": candidate.get("contact_point_a_px"),
        "contact_point_b_px": candidate.get("contact_point_b_px"),
        "contact_geometry_gap_top_px": contact_quality.get("top_contact_line_gap_px"),
        "contact_geometry_gap_bottom_px": contact_quality.get("bottom_contact_line_gap_px"),
        "contact_geometry_gap_max_px": contact_quality.get("largest_contact_line_gap_px"),
        "contact_geometry_gap_exceeds_threshold": contact_quality.get("gap_exceeds_threshold"),
    }


def _failure_decision(*, requested_material: str, axis_mode: str, deterministic_selected: str | None, survivors: list[str], reason: str) -> dict[str, Any]:
    return {
        "phase": "upv_vlm_v2_v2a_wide_contact_qwen_required",
        "anchor_stage": "v2a_wide_contact_qwen_required",
        "requested_class": requested_material,
        "axis_mode": axis_mode,
        "qwen_required": True,
        "vlm_rerank_used": False,
        "deterministic_anchor_decision_is_final": False,
        "deterministic_selected_anchor": deterministic_selected,
        "qwen_selected_anchor": None,
        "selected_anchor": None,
        "final_decision": "NO_SAFE_ANCHOR",
        "final_selected_anchor_source": "qwen_required_failed",
        "survivor_list": survivors,
        "qwen_reasoning_short": None,
        "failure_reason": reason,
    }


def _save_rgb(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)
    return str(path)


def _save_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((mask > 0).astype(np.uint8) * 255)).save(path)
    return str(path)


def _save_contact_geometry_debug_overlays(
    *,
    rgb_path: str | Path,
    original_mask: np.ndarray,
    contact_mask: np.ndarray,
    rect_corners: list[list[float]] | None,
    output_dir: Path,
    method: str = "robust_core_rectangle",
) -> dict[str, str]:
    from PIL import ImageDraw

    base = load_rgb(rgb_path)
    mask_path = _save_mask(output_dir / "contact_geometry_mask.png", contact_mask)

    contact_overlay = draw_mask_overlay(base, contact_mask > 0, color=(255, 160, 80), alpha=85, outline_color=(255, 70, 30))
    if rect_corners:
        draw = ImageDraw.Draw(contact_overlay)
        pts = [tuple(map(float, pt)) for pt in rect_corners]
        if pts:
            draw.line(pts + [pts[0]], fill=(255, 30, 180), width=3)
    contact_overlay_path = output_dir / "contact_core_rectangle_overlay.png"
    contact_overlay.save(contact_overlay_path)

    original_vs = draw_mask_overlay(base, original_mask > 0, color=(60, 220, 120), alpha=75, outline_color=(0, 255, 80))
    original_vs = draw_mask_overlay(original_vs, contact_mask > 0, color=(255, 40, 190), alpha=40, outline_color=(255, 40, 190))
    original_vs_path = output_dir / "original_vs_contact_geometry_overlay.png"
    original_vs.save(original_vs_path)
    result = {
        "contact_geometry_mask": mask_path,
        "contact_core_rectangle_overlay": str(contact_overlay_path),
        "original_vs_contact_geometry_overlay": str(original_vs_path),
    }
    if method == "rgbd_edge_ransac_rectangle":
        method_specific_path = output_dir / "original_mask_vs_rgbd_edge_rectangle_overlay.png"
        original_vs.save(method_specific_path)
        result["original_mask_vs_rgbd_edge_rectangle_overlay"] = str(method_specific_path)
    if method == "orientation_fixed_core_support_rectangle":
        method_specific_path = output_dir / "original_mask_vs_core_support_rectangle_overlay.png"
        original_vs.save(method_specific_path)
        result["original_mask_vs_core_support_rectangle_overlay"] = str(method_specific_path)
    return result


def _save_contact_geometry_aux_images(output_dir: Path, debug_images: dict[str, np.ndarray]) -> dict[str, str]:
    saved: dict[str, str] = {}
    for key, arr in debug_images.items():
        if arr is None:
            continue
        path = output_dir / f"{key}.png"
        if arr.ndim == 2:
            Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
        else:
            Image.fromarray(arr.astype(np.uint8)).save(path)
        saved[key] = str(path)
    return saved


def _save_contact_geometry_anchor_lines_overlay(
    *,
    rgb_path: str | Path,
    original_mask: np.ndarray,
    contact_mask: np.ndarray,
    rect_corners: list[list[float]] | None,
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> str:
    base = draw_mask_overlay(load_rgb(rgb_path), original_mask > 0, color=(60, 220, 120), alpha=40, outline_color=(0, 255, 80))
    base = draw_mask_overlay(base, contact_mask > 0, color=(255, 40, 190), alpha=20, outline_color=(255, 40, 190))
    draw = ImageDraw.Draw(base)
    if rect_corners:
        pts = [tuple(map(float, pt)) for pt in rect_corners]
        if pts:
            draw.line(pts + [pts[0]], fill=(255, 30, 180), width=3)
    for item in candidates:
        point = item.get("anchor_px")
        if not point:
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x + 6, y - 8), str(item.get("anchor_id", "?")), fill=(15, 15, 15))
        a = item.get("contact_point_a_px")
        b = item.get("contact_point_b_px")
        if a and b:
            draw.line([tuple(a), tuple(b)], fill=(255, 30, 180), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path)
    return str(output_path)


def _contact_guide_y(hit_xy: tuple[float, float] | None, bounds: tuple[int, int, int, int] | list[int] | None) -> float | None:
    if hit_xy is None or bounds is None:
        return None
    return float(hit_xy[1]) - float(bounds[1])


def _build_clean_contact_artifacts(out: Path, cfg: dict[str, Any]) -> dict[str, str]:
    clean_dir = out / "clean_single_anchor_inputs"
    debug = render_clean_anchor_inputs_for_artifact_dir(artifact_dir=out, output_dir=clean_dir, config=cfg)
    image_paths = {
        aid: meta.get("rendered_input_path")
        for aid, meta in (debug.get("anchors") or {}).items()
        if isinstance(meta, dict) and meta.get("rendered_input_path")
    }
    review_grid = build_clean_anchor_review_grid(image_paths=image_paths, output_path=out / "clean_anchor_review_grid.png")
    return {
        "clean_single_anchor_inputs_dir": str(clean_dir),
        "clean_anchor_review_grid": review_grid,
        "anchor_crop_geometry_debug": str(out / "anchor_crop_geometry_debug.json"),
    }


def run_anchor_selection(
    *,
    mask_path: str | Path,
    rgb_path: str | Path,
    geometry: dict[str, Any],
    axis_mode: str,
    config: dict[str, Any],
    output_dir: str | Path,
    depth_path: str | None = None,
    camera_info_path: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask = load_mask(mask_path)
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    refinement_cfg = dict(config.get("contact_geometry_refinement") or {})
    working_geometry = dict(geometry)
    contact_mask = mask
    contact_geometry_debug_artifacts: dict[str, str] = {}
    contact_geometry_refinement: dict[str, Any] = {
        "success": False,
        "enabled": bool(refinement_cfg.get("enabled", False)),
        "method": str(refinement_cfg.get("method", "robust_core_rectangle")),
    }
    if bool(refinement_cfg.get("enabled", False)):
        try:
            contact_geometry_refinement = build_contact_geometry_refinement(
                mask=mask,
                geometry=geometry,
                cfg=refinement_cfg,
                rgb=rgb,
                depth_path=depth_path,
                camera_info_path=camera_info_path,
            )
            contact_mask = contact_geometry_refinement["contact_geometry_mask"]
            working_geometry.update(
                {
                    "centroid_px": contact_geometry_refinement["centroid_px"],
                    "major_axis_vector": contact_geometry_refinement["major_axis_vector"],
                    "minor_axis_vector": contact_geometry_refinement["minor_axis_vector"],
                    "major_axis_angle_deg": contact_geometry_refinement["major_axis_angle_deg"],
                    "minor_axis_angle_deg": contact_geometry_refinement["minor_axis_angle_deg"],
                    "major_axis_length_px": contact_geometry_refinement["major_axis_length_px"],
                    "minor_axis_length_px": contact_geometry_refinement["minor_axis_length_px"],
                    "body_geometry_mode": "contact_core_rectangle",
                    "dominant_rectangle": contact_geometry_refinement["dominant_rectangle"],
                }
            )
            if bool(refinement_cfg.get("save_debug_overlays", True)):
                contact_geometry_debug_artifacts = _save_contact_geometry_debug_overlays(
                    rgb_path=rgb_path,
                    original_mask=mask,
                    contact_mask=contact_mask,
                    rect_corners=contact_geometry_refinement["dominant_rectangle"].get("body_rect_corners_px"),
                    output_dir=out,
                    method=str(contact_geometry_refinement.get("method") or refinement_cfg.get("method") or "robust_core_rectangle"),
                )
                contact_geometry_debug_artifacts.update(
                    _save_contact_geometry_aux_images(out, contact_geometry_refinement.get("debug_images") or {})
                )
        except Exception as exc:
            contact_geometry_refinement = {
                "success": False,
                "enabled": True,
                "method": str(refinement_cfg.get("method", "robust_core_rectangle")),
                "failure_reason": str(exc),
            }

    chosen_axis, cross_axis, extent = _axis_for_mode(axis_mode, working_geometry)
    dominant_rect = _dominant_rectangle_for_anchor_mode(working_geometry, axis_mode)
    if dominant_rect:
        if axis_mode == "minor":
            extent = float(dominant_rect.get("body_rect_minor_axis_length_px") or extent)
        else:
            extent = float(dominant_rect.get("body_rect_major_axis_length_px") or extent)
    if extent <= 0:
        payload = {"success": False, "axis_mode": axis_mode, "failure_reason": "invalid_axis_extent"}
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload
    cfg = config.get("anchor_selection", {})
    requested_count = int(cfg.get("num_anchor_samples", cfg.get("candidate_count", 5)))
    margin = float(cfg.get("usable_axis_margin_ratio", 0.18))
    anchor_count_metadata = _compute_v2a_anchor_count_metadata(working_geometry, requested_count, axis_mode, margin, cfg)
    candidate_count = int(anchor_count_metadata["anchor_count_generated"])
    if candidate_count <= 0:
        payload = {
            "success": False,
            "axis_mode": axis_mode,
            "failure_reason": "axis_too_short_for_transducer_diameter",
            "anchor_count_generated": 0,
            "candidate_count": 0,
            "anchor_count_metadata": anchor_count_metadata,
            "selected_anchor": None,
            "final_decision": "NO_ANCHORS_GENERATED",
            "diagnostics": {
                "anchor_count_metadata": anchor_count_metadata,
                "source": "anchor_count_physical_transducer_diameter",
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload
    axis_center_xy = dominant_rect.get("body_rect_center_px") if dominant_rect else working_geometry["centroid_px"]
    domain = build_axis_search_domain(
        object_center_xy=axis_center_xy,
        chosen_axis_unit_vector_xy=chosen_axis,
        axis_extent_px=extent,
        usable_axis_margin_ratio=margin,
        num_anchor_samples=candidate_count,
        axis_name=axis_mode,
    )
    candidates: list[dict[str, Any]] = []
    for idx, s in enumerate(domain.candidate_s_values, start=1):
        anchor_xy, left, right = compute_anchor_cross_section_hits(
            axis_search_domain=domain,
            candidate_s=s,
            mask=contact_mask,
            max_search_distance=float(max(mask.shape) * 1.5),
            step_size_px=float(cfg.get("cross_section_step_px", 1.0)),
        )
        a = list(left.hit_xy) if left.hit_xy else None
        b = list(right.hit_xy) if right.hit_xy else None
        local_len = 0.0
        if a and b:
            local_len = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        center_ratio = float(s / max(abs(domain.s_min), abs(domain.s_max), 1.0))
        quality = score_edge_quality(
            rgb_array=rgb,
            contact_point_a_px=a,
            contact_point_b_px=b,
            local_path_length_px=local_len,
            object_extent_px=float(working_geometry.get("minor_axis_length_px" if axis_mode == "major" else "major_axis_length_px", local_len)),
            center_distance_ratio=center_ratio,
            patch_radius_px=int(cfg.get("contact_patch_radius_px", 18)),
        )
        anchor_id = f"A{idx}"
        candidates.append(
            {
                "anchor_id": anchor_id,
                "legacy_anchor_id": f"{axis_mode}_anchor_{idx:03d}",
                "axis_mode": axis_mode,
                "axis_scalar": float(s),
                "anchor_px": [float(anchor_xy[0]), float(anchor_xy[1])],
                "contact_point_a_px": a,
                "contact_point_b_px": b,
                "local_cross_axis_vector": [float(cross_axis[0]), float(cross_axis[1])],
                "local_path_length_px": local_len,
                "left_hit_valid": left.valid,
                "right_hit_valid": right.valid,
                "score": quality["deterministic_score"],
                "veto": quality["veto"],
                "veto_reasons": quality["veto_reasons"],
                "quality": quality,
            }
        )
    center_xy = axis_center_xy
    candidate_points = [
        {
            "anchor_id": item["anchor_id"],
            "anchor_xy": item["anchor_px"],
            "left_hit_xy": item.get("contact_point_a_px"),
            "right_hit_xy": item.get("contact_point_b_px"),
        }
        for item in candidates
    ]
    rotated_image_rgb, rotated_mask, rotation_matrix, rotation_angle_deg = rotate_image_and_mask_to_canonical(
        image_rgb=rgb,
        mask=contact_mask,
        axis_center_xy=center_xy,
        axis_dir_xy=chosen_axis,
    )
    _raw_rotated_image_rgb, rotated_raw_mask, _rotation_matrix_raw, _rotation_angle_deg_raw = rotate_image_and_mask_to_canonical(
        image_rgb=rgb,
        mask=mask,
        axis_center_xy=center_xy,
        axis_dir_xy=chosen_axis,
    )
    rotated_overlay_rgb = render_rotated_object_overlay(
        rotated_image_rgb=rotated_image_rgb,
        rotated_mask=rotated_mask,
        rotation_matrix=rotation_matrix,
        geometry=working_geometry,
        selected_axis_vector=chosen_axis,
        candidate_points=candidate_points,
    )
    cropped_image_rgb, cropped_mask, cropped_overlay_rgb, rotated_crop_box = crop_rotated_object_view(
        rotated_image_rgb=rotated_image_rgb,
        rotated_mask=rotated_mask,
        rotated_overlay_rgb=rotated_overlay_rgb,
        margin_ratio=float(cfg.get("rotated_object_crop_margin_ratio", 0.10)),
    )
    cropped_raw_mask = rotated_raw_mask[rotated_crop_box[1] : rotated_crop_box[3], rotated_crop_box[0] : rotated_crop_box[2]].copy()
    rotated_artifacts = {
        "rotated_object_rgb": _save_rgb(out / "rotated_object_rgb.png", rotated_image_rgb),
        "rotated_object_mask": _save_mask(out / "rotated_object_mask.png", rotated_mask),
        "rotated_object_overlay": _save_rgb(out / "rotated_object_overlay.png", rotated_overlay_rgb),
        "cropped_rotated_object_rgb": _save_rgb(out / "cropped_rotated_object_rgb.png", cropped_image_rgb),
        "cropped_rotated_object_overlay": _save_rgb(out / "cropped_rotated_object_overlay.png", cropped_overlay_rgb),
    }
    dominant_crop_boundaries: dict[str, Any] = {}
    if dominant_rect:
        corners = dominant_rect.get("body_rect_corners_px")
        if isinstance(corners, list) and len(corners) >= 4:
            pts = []
            for point in corners:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    p = np.array([float(point[0]), float(point[1]), 1.0], dtype=float)
                    rp = rotation_matrix @ p
                    pts.append([float(rp[0]), float(rp[1])])
            if pts:
                arr = np.asarray(pts, dtype=float)
                dominant_crop_boundaries = {
                    "success": True,
                    "body_geometry_mode": str(working_geometry.get("body_geometry_mode") or "dominant_rectangle"),
                    "top_boundary_y": float(np.min(arr[:, 1])),
                    "bottom_boundary_y": float(np.max(arr[:, 1])),
                    "left_boundary_x": float(np.min(arr[:, 0])),
                    "right_boundary_x": float(np.max(arr[:, 0])),
                    "rotated_points_full_frame": pts,
                    "rotated_crop_box": list(rotated_crop_box),
                }

    tile_dir = out / "boxed_wide_contact_guided_tiles"
    tile_items: list[dict[str, Any]] = []
    rotation_mappings: dict[str, Any] = {}
    for item in candidates:
        rotated_points = map_candidate_points_to_rotated_crop_frame(
            anchor_xy=item["anchor_px"],
            left_hit_xy=item.get("contact_point_a_px"),
            right_hit_xy=item.get("contact_point_b_px"),
            rotation_matrix=rotation_matrix,
            rotated_object_crop_box=rotated_crop_box,
        )
        true_patch, _top_patch, _bottom_patch, _combined_patch = build_true_edge_pair_patch(
            candidate_id=str(item["anchor_id"]),
            rotated_object_overlay_rgb=cropped_overlay_rgb,
            anchor_xy_rotated=rotated_points["anchor_xy_rotated"],
            top_hit_xy_rotated=rotated_points["top_hit_xy_rotated"],
            bottom_hit_xy_rotated=rotated_points["bottom_hit_xy_rotated"],
            patch_half_width_px=int(cfg.get("true_edge_patch_half_width_px", 136)),
            edge_inside_margin_px=int(cfg.get("edge_inside_margin_px", 66)),
            edge_outside_margin_px=int(cfg.get("edge_outside_margin_px", 56)),
            patch_gap_px=int(cfg.get("edge_pair_patch_gap_px", 10)),
        )
        tile = build_boxed_wide_contact_tile_from_bounds(
            candidate_id=str(item["anchor_id"]),
            rotated_object_image_rgb=cropped_image_rgb,
            top_patch_bounds=true_patch.get("top_patch_bounds"),
            bottom_patch_bounds=true_patch.get("bottom_patch_bounds"),
            output_path=tile_dir / f"candidate_{item['anchor_id']}_wide_contact_guided_tile.png",
            top_contact_guide_y_px=_contact_guide_y(rotated_points["top_hit_xy_rotated"], true_patch.get("top_patch_bounds")),
            bottom_contact_guide_y_px=_contact_guide_y(rotated_points["bottom_hit_xy_rotated"], true_patch.get("bottom_patch_bounds")),
            header_height_px=int(cfg.get("tile_header_height_px", 36)),
            content_padding_px=int(cfg.get("tile_content_padding_px", 10)),
            row_gap_px=int(cfg.get("tile_row_gap_px", 10)),
            row_label_position=str(cfg.get("tile_row_label_position", "above")),
            row_label_height_px=int(cfg.get("tile_row_label_height_px", 20)),
            min_row_height_px=int(cfg.get("tile_min_row_height_px", int(cfg.get("edge_inside_margin_px", 66)) + int(cfg.get("edge_outside_margin_px", 56)))),
        )
        rotation_mappings[item["anchor_id"]] = {
            "original_anchor_px": item["anchor_px"],
            "original_contact_point_a_px": item.get("contact_point_a_px"),
            "original_contact_point_b_px": item.get("contact_point_b_px"),
            "rotated_crop_points": rotated_points,
            "true_edge_pair_patch": true_patch,
        }
        tile_items.append({**tile, "source_anchor_id": item["anchor_id"], "qwen_grid_source": "cropped_rotated_object_view"})

    contact_geometry_quality: list[dict[str, Any]] = []
    if dominant_crop_boundaries.get("success"):
        contact_geometry_quality = compute_contact_geometry_quality(
            rotated_raw_mask=cropped_raw_mask > 0,
            rotation_mappings=rotation_mappings,
            refined_top_y=float(dominant_crop_boundaries["top_boundary_y"]) - float(rotated_crop_box[1]),
            refined_bottom_y=float(dominant_crop_boundaries["bottom_boundary_y"]) - float(rotated_crop_box[1]),
            image_width=cropped_image_rgb.shape[1],
            cfg=refinement_cfg,
        )
        quality_by_anchor = {row["anchor_id"]: row for row in contact_geometry_quality}
        invalidate_on_large_gap = bool(refinement_cfg.get("invalidate_on_large_gap", True))
        for item in candidates:
            q = quality_by_anchor.get(str(item["anchor_id"]))
            item["contact_geometry_quality"] = q
            if q and q.get("gap_exceeds_threshold") and invalidate_on_large_gap:
                veto_reasons = list(item.get("veto_reasons", []))
                if "contact_geometry_gap_exceeds_threshold" not in veto_reasons:
                    veto_reasons.append("contact_geometry_gap_exceeds_threshold")
                item["veto"] = True
                item["veto_reasons"] = veto_reasons
                item["score"] = min(float(item.get("score", 0.0)), 35.0)

    if contact_geometry_refinement.get("success"):
        contact_geometry_debug_artifacts["contact_geometry_anchor_lines_overlay"] = _save_contact_geometry_anchor_lines_overlay(
            rgb_path=rgb_path,
            original_mask=mask,
            contact_mask=contact_mask,
            rect_corners=(contact_geometry_refinement.get("dominant_rectangle") or {}).get("body_rect_corners_px"),
            candidates=candidates,
            output_path=out / "contact_geometry_anchor_lines_overlay.png",
        )

    decision = select_anchor_hybrid(candidates)
    deterministic_selected_id = decision.get("selected_anchor")
    survivors = [str(item["anchor_id"]) for item in candidates if not item.get("veto")]

    grid_path = build_boxed_candidate_tile_grid(
        tile_items,
        out / "combined_wide_contact_guided_grid.png",
        columns=max(1, min(5, len(tile_items))),
        padding_px=int(cfg.get("grid_padding_px", 14)),
    )
    grid_2col_path = build_boxed_candidate_tile_grid(
        tile_items,
        out / "combined_wide_contact_guided_grid_2col.png",
        columns=max(1, min(2, len(tile_items))),
        padding_px=int(cfg.get("grid_padding_px", 14)),
    )
    rotation_metadata = {
        "rotation_angle_deg": rotation_angle_deg,
        "rotation_matrix": rotation_matrix.tolist(),
        "rotated_crop_box": list(rotated_crop_box),
        "selected_axis_vector_original": [float(chosen_axis[0]), float(chosen_axis[1])],
        "selected_axis_mode": axis_mode,
        "qwen_grid_source": "cropped_rotated_object_view",
        "rotated_artifacts": rotated_artifacts,
        "candidate_point_mapping_original_to_rotated_crop": rotation_mappings,
        "body_geometry_mode": str(working_geometry.get("body_geometry_mode") or ("dominant_rectangle" if dominant_rect else "raw_mask_existing")),
        "dominant_rectangle": dominant_rect,
        "dominant_rectangle_crop_boundaries_rotated": dominant_crop_boundaries,
        "contact_geometry_refinement": contact_geometry_refinement,
        "contact_geometry_debug_artifacts": contact_geometry_debug_artifacts,
    }
    _json(out / "rotation_metadata.json", rotation_metadata)
    clean_contact_artifacts = _build_clean_contact_artifacts(out, cfg)
    rotation_metadata["clean_contact_artifacts"] = clean_contact_artifacts
    _json(out / "rotation_metadata.json", rotation_metadata)
    quality_json_path = _json(out / "anchor_contact_geometry_quality.json", {"rows": contact_geometry_quality, "config": refinement_cfg})
    quality_csv_path = _write_csv(
        out / "anchor_contact_geometry_quality.csv",
        contact_geometry_quality,
        [
            "anchor_id",
            "anchor_lateral_zone_px",
            "anchor_lateral_spacing_px",
            "requested_lateral_span_px",
            "actual_lateral_span_px",
            "raw_local_top_boundary_y",
            "raw_local_bottom_boundary_y",
            "refined_top_boundary_y",
            "refined_bottom_boundary_y",
            "top_contact_line_gap_px",
            "bottom_contact_line_gap_px",
            "largest_contact_line_gap_px",
            "max_contact_line_gap_px",
            "within_gap_threshold",
            "gap_exceeds_threshold",
            "failure_reason",
        ],
    )

    deterministic_features = {
        "phase": "upv_vlm_v2_v2a_wide_contact_qwen_required",
        "context_variant": "v2a_wide_contact_context",
        "qwen_grid_source": "cropped_rotated_object_view",
        "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
        "anchor_count_metadata": anchor_count_metadata,
        "body_geometry_mode": str(working_geometry.get("body_geometry_mode") or ("dominant_rectangle" if dominant_rect else "raw_mask_existing")),
        "dominant_rectangle_used_for_anchor_crops": bool(dominant_rect),
        "contact_geometry_refinement": contact_geometry_refinement,
        "anchor_features": [_feature_record(candidate) for candidate in candidates],
    }
    decision_trace = {
        "phase": "upv_vlm_v2_v2a_wide_contact_qwen_required",
        "anchor_stage": "v2a_wide_contact_qwen_required",
        "deterministic_selected_anchor": deterministic_selected_id,
        "deterministic_anchor_decision_is_final": False,
        "vlm_rerank_used": False,
        "survivor_list": survivors,
        "decision_reason": "deterministic stage generated candidates/features/vetoes only; Qwen is required for final selection",
        "deterministic_decision_trace": decision,
        "qwen_grid_source": "cropped_rotated_object_view",
    }

    requested_material = str(config.get("_runtime_requested_material") or config.get("requested_material") or "unknown")
    qwen_required = bool(cfg.get("qwen_required", True))
    if not qwen_required:
        selected_id = deterministic_selected_id
        selected = next((item for item in candidates if item["anchor_id"] == selected_id), None)
        final_decision = {
            "phase": "upv_vlm_v2_v2a_wide_contact_deterministic_manual_prep",
            "anchor_stage": "v2a_wide_contact_deterministic_manual_prep",
            "requested_class": requested_material,
            "axis_mode": axis_mode,
            "qwen_required": False,
            "vlm_rerank_used": False,
            "deterministic_anchor_decision_is_final": True,
            "deterministic_selected_anchor": selected_id,
            "qwen_selected_anchor": None,
            "selected_anchor": selected_id,
            "final_decision": selected_id,
            "final_selected_anchor_source": "deterministic_manual_prep",
            "survivor_list": survivors,
            "failure_reason": None if selected else "NO_DETERMINISTIC_ANCHOR",
            "anchor_count_generated": len(candidates),
            "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
        }
        robot_anchor_geometry = {
            "robot_candidate_anchor_id": selected_id,
            "robot_candidate_anchor_center_px": selected.get("anchor_px") if selected else None,
            "robot_candidate_selection_mode": "deterministic_manual_prep",
            "robot_candidate_is_safe": not bool(selected.get("veto")) if selected else False,
            "robot_candidate_for_motion_allowed": False,
            "qwen_required": False,
            "qwen_selected_anchor_id": None,
            "deterministic_candidate_source": selected,
            "axis_mode": axis_mode,
            "requested_class": requested_material,
        }
        overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / f"{axis_mode}_selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; deterministic manual prep",
        )
        final_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / "selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; deterministic manual prep",
        )
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
        _json(out / "final_anchor_decision_wide_context.json", final_decision)
        _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)
        payload = {
            "success": bool(selected),
            "axis_mode": axis_mode,
            "final_anchor_id": selected_id,
            "selected_anchor_px": robot_anchor_geometry["robot_candidate_anchor_center_px"],
            "contact_point_a_px": selected.get("contact_point_a_px") if selected else None,
            "contact_point_b_px": selected.get("contact_point_b_px") if selected else None,
            "local_cross_axis_vector": selected.get("local_cross_axis_vector") if selected else None,
            "score": selected.get("score") if selected else None,
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
                "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
                "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                "rotation_metadata": str(out / "rotation_metadata.json"),
                **rotated_artifacts,
                **clean_contact_artifacts,
                **contact_geometry_debug_artifacts,
                "anchor_contact_geometry_quality_json": quality_json_path,
                "anchor_contact_geometry_quality_csv": quality_csv_path,
                "deterministic_anchor_features_wide_context": str(out / "deterministic_anchor_features_wide_context.json"),
                "hybrid_decision_trace_wide_context": str(out / "hybrid_decision_trace_wide_context.json"),
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                "selected_anchor_overlay": final_overlay,
            },
            "overlay_paths": {"selected_anchor_overlay": final_overlay, "axis_selected_anchor_overlay": overlay, "anchor_candidates_overlay": overlay},
            "failure_reason": None if selected else "NO_DETERMINISTIC_ANCHOR",
            "diagnostics": {
                "anchor_backend": "v2a_wide_contact_deterministic_manual_prep",
                "qwen_required": False,
                "qwen_selected_anchor_id": None,
                "final_anchor_source": "deterministic_manual_prep",
                "robot_anchor_geometry": robot_anchor_geometry,
                "wide_contact_grid": grid_path,
                "wide_contact_grid_2col": grid_2col_path,
                "clean_contact_artifacts": clean_contact_artifacts,
                "contact_geometry_debug_artifacts": contact_geometry_debug_artifacts,
                "anchor_contact_geometry_quality_json": quality_json_path,
                "anchor_contact_geometry_quality_csv": quality_csv_path,
                "qwen_grid_source": "cropped_rotated_object_view",
                "rotation_metadata": rotation_metadata,
                "deterministic_survivors": survivors,
                "deterministic_decision": decision,
                "final_decision": final_decision,
                "anchor_count_metadata": anchor_count_metadata,
                "source": "v2a_wide_contact_deterministic_manual_prep",
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload

    llama_single_backend_enabled = (
        str(cfg.get("backend", "")).strip() == LLAMA32_SINGLE_ANCHOR_L6_BACKEND
        or str(cfg.get("vlm_mode", "")).strip() == "single_image_loop"
    )
    if llama_single_backend_enabled:
        llama = run_llama32_single_anchor_l6_taxonomy_scoring(
            config=config,
            artifact_dir=out,
            candidates=candidates,
            requested_material=requested_material,
            axis_mode=axis_mode,
        )
        provenance = llama.get("provenance") if isinstance(llama.get("provenance"), dict) else {}
        selected_id = llama.get("selected_anchor_id") if llama.get("success") else None
        selected = next((item for item in candidates if item["anchor_id"] == selected_id), None)
        final_source = LLAMA32_SINGLE_ANCHOR_L6_BACKEND
        final_decision = {
            "phase": "main_pipeline_llama32_single_anchor_l6_taxonomy_scoring",
            "anchor_stage": final_source,
            "requested_class": requested_material,
            "axis_mode": axis_mode,
            "qwen_required": True,
            "vlm_rerank_used": True,
            "deterministic_anchor_decision_is_final": False,
            "deterministic_selected_anchor": deterministic_selected_id,
            "qwen_selected_anchor": selected_id,
            "selected_anchor": selected_id,
            "final_decision": selected_id if selected else "NO_SAFE_ANCHOR",
            "final_selected_anchor_source": final_source if selected else "llama32_single_anchor_l6_taxonomy_scoring_failed",
            "survivor_list": survivors,
            "failure_reason": None if selected else llama.get("failure_reason", "NO_SAFE_ANCHOR"),
            "anchor_count_generated": len(candidates),
            "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
            "llama32_single_anchor_decision": str(out / "llama32_single_anchor_decision.json"),
            "llama32_ranked_usable_anchors": llama.get("ranked_usable_anchors", []),
            "llama_min_selectable_score": llama.get("llama_min_selectable_score"),
            **provenance,
        }
        robot_anchor_geometry = {
            "robot_candidate_anchor_id": selected_id,
            "robot_candidate_anchor_center_px": selected.get("anchor_px") if selected else None,
            "robot_candidate_selection_mode": final_decision["final_selected_anchor_source"],
            "robot_candidate_is_safe": not bool(selected.get("veto")) if selected else False,
            "robot_candidate_for_motion_allowed": not bool(selected.get("veto")) if selected else False,
            "qwen_required": True,
            "qwen_selected_anchor_id": selected_id,
            "deterministic_candidate_source": selected,
            "axis_mode": axis_mode,
            "requested_class": requested_material,
            **provenance,
        }
        overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / f"{axis_mode}_selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_decision['final_selected_anchor_source']}",
        )
        final_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / "selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_decision['final_selected_anchor_source']}",
        )
        _json(out / f"{axis_mode}_anchor_candidates.json", {"axis_search_domain": domain.__dict__, "candidates": candidates})
        _json(out / f"{axis_mode}_anchor_scores.json", decision)
        _json(out / f"{axis_mode}_final_anchor.json", selected)
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", {**decision_trace, **provenance, "llama_decision": llama})
        _json(out / "final_anchor_decision_wide_context.json", final_decision)
        _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)
        payload = {
            "success": bool(selected),
            "axis_mode": axis_mode,
            "final_anchor_id": selected_id,
            "selected_anchor_px": robot_anchor_geometry["robot_candidate_anchor_center_px"],
            "contact_point_a_px": selected.get("contact_point_a_px") if selected else None,
            "contact_point_b_px": selected.get("contact_point_b_px") if selected else None,
            "local_cross_axis_vector": selected.get("local_cross_axis_vector") if selected else None,
            "score": selected.get("score") if selected else None,
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
                "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
                "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                **clean_contact_artifacts,
                "original_clean_single_anchor_inputs_dir": str(out / "original_clean_single_anchor_inputs"),
                "l6_clean_input_conversion": str(out / "l6_clean_input_conversion.json"),
                "llama32_single_anchor_decision": str(out / "llama32_single_anchor_decision.json"),
                "llama32_single_anchor_score_table": str(out / "llama32_single_anchor_score_table.csv"),
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                "selected_anchor_overlay": final_overlay,
            },
            "overlay_paths": {
                "selected_anchor_overlay": final_overlay,
                "axis_selected_anchor_overlay": overlay,
                "anchor_candidates_overlay": overlay,
            },
            "failure_reason": None if selected else str(llama.get("failure_reason") or "NO_SAFE_ANCHOR"),
            "diagnostics": {
                "anchor_backend": LLAMA32_SINGLE_ANCHOR_L6_BACKEND,
                "qwen_required": True,
                "qwen_selected_anchor_id": selected_id,
                "final_anchor_source": final_decision["final_selected_anchor_source"],
                "robot_anchor_geometry": robot_anchor_geometry,
                "wide_contact_grid": grid_path,
                "wide_contact_grid_2col": grid_2col_path,
                "clean_contact_artifacts": clean_contact_artifacts,
                "qwen_grid_source": "clean_single_anchor_inputs",
                "rotation_metadata": rotation_metadata,
                "deterministic_survivors": survivors,
                "deterministic_decision": decision,
                "final_decision": final_decision,
                "llama_decision": llama,
                "anchor_count_metadata": anchor_count_metadata,
                "source": LLAMA32_SINGLE_ANCHOR_L6_BACKEND,
                **provenance,
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload

    multi_anchor_backend_enabled = (
        str(cfg.get("backend", "")).strip() == QWEN32_MULTI_ANCHOR_L6_BACKEND
        or str(cfg.get("qwen_mode", "")).strip() == "multi_image"
    )
    if multi_anchor_backend_enabled:
        qwen = run_qwen32_multi_anchor_l6_batch_ranking(
            config=config,
            artifact_dir=out,
            candidates=candidates,
            requested_material=requested_material,
            axis_mode=axis_mode,
        )
        provenance = qwen.get("provenance") if isinstance(qwen.get("provenance"), dict) else {}
        selected_id = qwen.get("selected_anchor_id") if qwen.get("success") else None
        selected = next((item for item in candidates if item["anchor_id"] == selected_id), None)

        if not selected:
            failure_reason = str(qwen.get("failure_reason") or "QWEN32_MULTI_ANCHOR_L6_BATCH_RANKING_FAILED")
            final_failure = _failure_decision(
                requested_material=requested_material,
                axis_mode=axis_mode,
                deterministic_selected=deterministic_selected_id,
                survivors=survivors,
                reason=failure_reason,
            )
            final_failure.update(
                {
                    "phase": "main_pipeline_qwen32_multi_anchor_l6_batch_ranking",
                    "anchor_stage": QWEN32_MULTI_ANCHOR_L6_BACKEND,
                    "qwen_required": True,
                    "vlm_rerank_used": True,
                    "qwen_selected_anchor": "NO_SAFE_ANCHOR" if qwen.get("safe_no_anchor") else None,
                    "selected_anchor": None,
                    "final_decision": "NO_SAFE_ANCHOR",
                    "final_selected_anchor_source": "qwen32_multi_anchor_l6_batch_ranking_failed",
                    "qwen32_multi_anchor_decision": str(out / "qwen32_multi_anchor_decision.json"),
                    **provenance,
                }
            )
            robot_anchor_geometry = {
                "robot_candidate_anchor_id": None,
                "robot_candidate_anchor_center_px": None,
                "robot_candidate_selection_mode": "qwen32_multi_anchor_l6_batch_ranking_failed",
                "robot_candidate_is_safe": False,
                "robot_candidate_for_motion_allowed": False,
                "qwen_required": True,
                "qwen_selected_anchor_id": None,
                "axis_mode": axis_mode,
                "requested_class": requested_material,
                "failure_reason": failure_reason,
                **provenance,
            }
            no_safe_overlay = save_anchor_overlay(
                rgb_path=rgb_path,
                mask_path=mask_path,
                anchors=candidates,
                selected_anchor_id=None,
                output_path=out / "no_safe_anchor_overlay.png",
                title=f"v2 anchor selection failed: {failure_reason}",
            )
            _json(out / f"{axis_mode}_anchor_candidates.json", {"axis_search_domain": domain.__dict__, "candidates": candidates})
            _json(out / f"{axis_mode}_anchor_scores.json", decision)
            _json(out / f"{axis_mode}_final_anchor.json", None)
            _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
            _json(out / "hybrid_decision_trace_wide_context.json", {**decision_trace, **provenance, "qwen_decision": qwen})
            _json(out / "final_anchor_decision_wide_context.json", final_failure)
            _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)
            payload = {
                "success": False,
                "axis_mode": axis_mode,
                "failure_reason": failure_reason,
                "candidate_count": len(candidates),
                "candidate_artifacts": {
                    "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
                    "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
                    "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
                    "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                    "combined_wide_contact_guided_grid": grid_path,
                    "combined_wide_contact_guided_grid_2col": grid_2col_path,
                    **clean_contact_artifacts,
                    "l6_clean_input_conversion": str(out / "l6_clean_input_conversion.json"),
                    "qwen32_multi_anchor_decision": str(out / "qwen32_multi_anchor_decision.json"),
                    "qwen32_multi_anchor_score_table": str(out / "qwen32_multi_anchor_score_table.csv"),
                    "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                    "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                    "no_safe_anchor_overlay": no_safe_overlay,
                },
                "overlay_paths": {"no_safe_anchor_overlay": no_safe_overlay},
                "diagnostics": {
                    "anchor_backend": QWEN32_MULTI_ANCHOR_L6_BACKEND,
                    "qwen_required": True,
                    "qwen_decision": qwen,
                    "decision": final_failure,
                    "robot_anchor_geometry": robot_anchor_geometry,
                    "clean_contact_artifacts": clean_contact_artifacts,
                    "source": QWEN32_MULTI_ANCHOR_L6_BACKEND,
                    **provenance,
                },
            }
            _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
            return payload

        final_source = QWEN32_MULTI_ANCHOR_L6_BACKEND
        final_decision = {
            "phase": "main_pipeline_qwen32_multi_anchor_l6_batch_ranking",
            "anchor_stage": QWEN32_MULTI_ANCHOR_L6_BACKEND,
            "requested_class": requested_material,
            "axis_mode": axis_mode,
            "qwen_required": True,
            "vlm_rerank_used": True,
            "deterministic_anchor_decision_is_final": False,
            "deterministic_selected_anchor": deterministic_selected_id,
            "qwen_selected_anchor": selected_id,
            "selected_anchor": selected_id,
            "final_decision": selected_id,
            "final_selected_anchor_source": final_source,
            "survivor_list": survivors,
            "failure_reason": None,
            "anchor_count_generated": len(candidates),
            "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
            "qwen32_multi_anchor_decision": str(out / "qwen32_multi_anchor_decision.json"),
            "qwen32_ranked_usable_anchors": qwen.get("ranked_usable_anchors", []),
            "qwen32_anchors_attempted": qwen.get("anchors_attempted"),
            "qwen32_anchors_parsed": qwen.get("anchors_parsed"),
            **provenance,
        }
        robot_anchor_geometry = {
            "robot_candidate_anchor_id": selected_id,
            "robot_candidate_anchor_center_px": selected.get("anchor_px"),
            "robot_candidate_selection_mode": final_source,
            "robot_candidate_is_safe": not bool(selected.get("veto")),
            "robot_candidate_for_motion_allowed": not bool(selected.get("veto")),
            "qwen_required": True,
            "qwen_selected_anchor_id": selected_id,
            "deterministic_candidate_source": selected,
            "axis_mode": axis_mode,
            "requested_class": requested_material,
            **provenance,
        }
        overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / f"{axis_mode}_selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_source}",
        )
        final_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / "selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_source}",
        )
        _json(out / f"{axis_mode}_anchor_candidates.json", {"axis_search_domain": domain.__dict__, "candidates": candidates})
        _json(out / f"{axis_mode}_anchor_scores.json", decision)
        _json(out / f"{axis_mode}_final_anchor.json", selected)
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", {**decision_trace, **provenance, "qwen_decision": qwen})
        _json(out / "final_anchor_decision_wide_context.json", final_decision)
        _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)

        payload = {
            "success": bool(selected),
            "axis_mode": axis_mode,
            "final_anchor_id": selected_id,
            "selected_anchor_px": robot_anchor_geometry["robot_candidate_anchor_center_px"],
            "contact_point_a_px": selected.get("contact_point_a_px"),
            "contact_point_b_px": selected.get("contact_point_b_px"),
            "local_cross_axis_vector": selected.get("local_cross_axis_vector"),
            "score": selected.get("score"),
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
                "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
                "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                **clean_contact_artifacts,
                "original_clean_single_anchor_inputs_dir": str(out / "original_clean_single_anchor_inputs"),
                "l6_clean_input_conversion": str(out / "l6_clean_input_conversion.json"),
                "rotation_metadata": str(out / "rotation_metadata.json"),
                **rotated_artifacts,
                "deterministic_anchor_features_wide_context": str(out / "deterministic_anchor_features_wide_context.json"),
                "hybrid_decision_trace_wide_context": str(out / "hybrid_decision_trace_wide_context.json"),
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                "qwen32_multi_anchor_decision": str(out / "qwen32_multi_anchor_decision.json"),
                "qwen32_multi_anchor_score_table": str(out / "qwen32_multi_anchor_score_table.csv"),
                "qwen32_multi_anchor_inputs_dir": str(out / "clean_single_anchor_inputs"),
                "qwen32_multi_anchor_responses_dir": str(out / "qwen32_multi_anchor_responses"),
                "selected_anchor_overlay": final_overlay,
            },
            "overlay_paths": {
                "selected_anchor_overlay": final_overlay,
                "axis_selected_anchor_overlay": overlay,
                "anchor_candidates_overlay": overlay,
            },
            "failure_reason": None,
            "diagnostics": {
                "anchor_backend": QWEN32_MULTI_ANCHOR_L6_BACKEND,
                "qwen_required": True,
                "qwen_selected_anchor_id": selected_id,
                "final_anchor_source": final_source,
                "final_selected_anchor_source": final_source,
                "robot_anchor_geometry": robot_anchor_geometry,
                "wide_contact_grid": grid_path,
                "wide_contact_grid_2col": grid_2col_path,
                "clean_contact_artifacts": clean_contact_artifacts,
                "qwen_grid_source": "clean_single_anchor_inputs",
                "rotation_metadata": rotation_metadata,
                "deterministic_survivors": survivors,
                "deterministic_decision": decision,
                "final_decision": final_decision,
                "qwen_decision": qwen,
                "anchor_count_metadata": anchor_count_metadata,
                "source": QWEN32_MULTI_ANCHOR_L6_BACKEND,
                **provenance,
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload

    single_anchor_backend_enabled = (
        str(cfg.get("backend", "")).strip() == "v2a_qwen32_single_anchor_contact_scoring"
        or bool(cfg.get("qwen32_single_anchor_contact_scoring_enabled", False))
    )
    if single_anchor_backend_enabled:
        qwen = run_qwen32_single_anchor_contact_scoring(
            config=config,
            artifact_dir=out,
            candidates=candidates,
            deterministic_features=deterministic_features,
            output_dir=out,
            requested_material=requested_material,
            axis_mode=axis_mode,
        )

        selected_id = qwen.get("selected_anchor_id") if qwen.get("success") else None
        selected = next((item for item in candidates if item["anchor_id"] == selected_id), None)

        if not selected:
            failure_reason = str(qwen.get("failure_reason") or "QWEN32_SINGLE_ANCHOR_SCORING_FAILED")
            final_failure = _failure_decision(
                requested_material=requested_material,
                axis_mode=axis_mode,
                deterministic_selected=deterministic_selected_id,
                survivors=survivors,
                reason=failure_reason,
            )
            final_failure.update(
                {
                    "phase": "main_pipeline_qwen32_single_anchor_contact_scoring",
                    "anchor_stage": "v2a_qwen32_single_anchor_contact_scoring",
                    "qwen_required": True,
                    "vlm_rerank_used": True,
                    "qwen_selected_anchor": None,
                    "selected_anchor": None,
                    "final_decision": "NO_SAFE_ANCHOR",
                    "final_selected_anchor_source": "qwen32_single_anchor_contact_scoring_failed",
                    "qwen32_single_anchor_decision": str(out / "qwen32_single_anchor_decision.json"),
                }
            )
            no_safe_overlay = save_anchor_overlay(
                rgb_path=rgb_path,
                mask_path=mask_path,
                anchors=candidates,
                selected_anchor_id=None,
                output_path=out / "no_safe_anchor_overlay.png",
                title=f"v2 anchor selection failed: {failure_reason}",
            )
            _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
            _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
            _json(out / "final_anchor_decision_wide_context.json", final_failure)
            payload = {
                "success": False,
                "axis_mode": axis_mode,
                "failure_reason": failure_reason,
                "candidate_count": len(candidates),
                "candidate_artifacts": {
                    "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                    "combined_wide_contact_guided_grid": grid_path,
                    "combined_wide_contact_guided_grid_2col": grid_2col_path,
                    **clean_contact_artifacts,
                    "qwen32_single_anchor_decision": str(out / "qwen32_single_anchor_decision.json"),
                    "qwen32_single_anchor_score_table": str(out / "qwen32_single_anchor_score_table.csv"),
                    "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                    "no_safe_anchor_overlay": no_safe_overlay,
                },
                "overlay_paths": {"no_safe_anchor_overlay": no_safe_overlay},
                "diagnostics": {
                    "qwen": qwen,
                    "decision": final_failure,
                    "source": "v2a_qwen32_single_anchor_contact_scoring",
                    "clean_contact_artifacts": clean_contact_artifacts,
                },
            }
            _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
            return payload

        final_source = "qwen32_single_anchor_contact_scoring"
        final_decision = {
            "phase": "main_pipeline_qwen32_single_anchor_contact_scoring",
            "anchor_stage": "v2a_qwen32_single_anchor_contact_scoring",
            "requested_class": requested_material,
            "axis_mode": axis_mode,
            "qwen_required": True,
            "vlm_rerank_used": True,
            "deterministic_anchor_decision_is_final": False,
            "deterministic_selected_anchor": deterministic_selected_id,
            "qwen_selected_anchor": selected_id,
            "selected_anchor": selected_id,
            "final_decision": selected_id,
            "final_selected_anchor_source": final_source,
            "survivor_list": survivors,
            "qwen_reasoning_short": qwen.get("reasoning_short"),
            "failure_reason": None,
            "anchor_count_generated": len(candidates),
            "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
            "qwen32_single_anchor_decision": str(out / "qwen32_single_anchor_decision.json"),
            "qwen32_ranked_source_anchors": qwen.get("ranked_source_anchors", []),
            "qwen32_anchors_attempted": qwen.get("anchors_attempted"),
            "qwen32_anchors_parsed": qwen.get("anchors_parsed"),
        }
        robot_anchor_geometry = {
            "robot_candidate_anchor_id": selected_id,
            "robot_candidate_anchor_center_px": selected.get("anchor_px"),
            "robot_candidate_selection_mode": final_source,
            "robot_candidate_is_safe": not bool(selected.get("veto")),
            "robot_candidate_for_motion_allowed": not bool(selected.get("veto")),
            "qwen_required": True,
            "qwen_selected_anchor_id": selected_id,
            "deterministic_candidate_source": selected,
            "axis_mode": axis_mode,
            "requested_class": requested_material,
        }
        overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / f"{axis_mode}_selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_source}",
        )
        final_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=selected_id,
            output_path=out / "selected_anchor_overlay.png",
            title=f"v2 anchor selection axis_mode={axis_mode}; source={final_source}",
        )
        _json(out / f"{axis_mode}_anchor_candidates.json", {"axis_search_domain": domain.__dict__, "candidates": candidates})
        _json(out / f"{axis_mode}_anchor_scores.json", decision)
        _json(out / f"{axis_mode}_final_anchor.json", selected)
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
        _json(out / "final_anchor_decision_wide_context.json", final_decision)
        _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)

        payload = {
            "success": bool(selected),
            "axis_mode": axis_mode,
            "final_anchor_id": selected_id,
            "selected_anchor_px": robot_anchor_geometry["robot_candidate_anchor_center_px"],
            "contact_point_a_px": selected.get("contact_point_a_px"),
            "contact_point_b_px": selected.get("contact_point_b_px"),
            "local_cross_axis_vector": selected.get("local_cross_axis_vector"),
            "score": selected.get("score"),
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
                "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
                "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                **clean_contact_artifacts,
                "rotation_metadata": str(out / "rotation_metadata.json"),
                **rotated_artifacts,
                "deterministic_anchor_features_wide_context": str(out / "deterministic_anchor_features_wide_context.json"),
                "hybrid_decision_trace_wide_context": str(out / "hybrid_decision_trace_wide_context.json"),
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                "qwen32_single_anchor_decision": str(out / "qwen32_single_anchor_decision.json"),
                "qwen32_single_anchor_score_table": str(out / "qwen32_single_anchor_score_table.csv"),
                "qwen32_single_anchor_inputs_dir": str(out / "clean_single_anchor_inputs"),
                "qwen32_single_anchor_responses_dir": str(out / "qwen32_single_anchor_responses"),
                "selected_anchor_overlay": final_overlay,
            },
            "overlay_paths": {
                "selected_anchor_overlay": final_overlay,
                "axis_selected_anchor_overlay": overlay,
                "anchor_candidates_overlay": overlay,
            },
            "failure_reason": None,
            "diagnostics": {
                "anchor_backend": "v2a_qwen32_single_anchor_contact_scoring",
                "qwen_required": True,
                "qwen_selected_anchor_id": selected_id,
                "final_anchor_source": final_source,
                "final_selected_anchor_source": final_source,
                "robot_anchor_geometry": robot_anchor_geometry,
                "wide_contact_grid": grid_path,
                "wide_contact_grid_2col": grid_2col_path,
                "clean_contact_artifacts": clean_contact_artifacts,
                "qwen_grid_source": "clean_single_anchor_inputs",
                "rotation_metadata": rotation_metadata,
                "deterministic_survivors": survivors,
                "deterministic_decision": decision,
                "final_decision": final_decision,
                "qwen_decision": qwen,
                "anchor_count_metadata": anchor_count_metadata,
                "source": "v2a_qwen32_single_anchor_contact_scoring",
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload

    qwen_input_image_name = str(cfg.get("qwen_input_image", "combined_wide_contact_guided_grid"))
    if qwen_input_image_name == "combined_wide_contact_guided_grid":
        qwen_grid_image_path = grid_path
    elif qwen_input_image_name == "combined_wide_contact_guided_grid_2col":
        qwen_grid_image_path = grid_2col_path
    else:
        raise ValueError(f"Unsupported qwen_input_image: {qwen_input_image_name}")

    rotation_metadata["qwen_input_image_config"] = qwen_input_image_name
    rotation_metadata["qwen_input_image_path"] = qwen_grid_image_path
    _json(out / "rotation_metadata.json", rotation_metadata)

    qwen = run_required_qwen_anchor_selection(
        config=config,
        grid_image_path=qwen_grid_image_path,
        requested_material=requested_material,
        axis_mode=axis_mode,
        candidates=candidates,
        deterministic_features=deterministic_features,
        output_dir=out,
    )
    if qwen.get("success") and qwen.get("safe_no_anchor"):
        final_failure = _failure_decision(
            requested_material=requested_material,
            axis_mode=axis_mode,
            deterministic_selected=deterministic_selected_id,
            survivors=survivors,
            reason="NO_SAFE_ANCHOR",
        )
        final_failure.update(
            {
                "qwen_required": True,
                "vlm_rerank_used": True,
                "qwen_selected_anchor": "NO_SAFE_ANCHOR",
                "selected_anchor": None,
                "final_decision": "NO_SAFE_ANCHOR",
                "final_selected_anchor_source": "qwen_required_no_safe_anchor",
                "qwen_reasoning_short": qwen.get("reasoning_short"),
            }
        )
        robot_anchor_geometry = {
            "robot_candidate_anchor_id": None,
            "robot_candidate_anchor_center_px": None,
            "robot_candidate_selection_mode": "qwen_required_no_safe_anchor",
            "robot_candidate_is_safe": False,
            "robot_candidate_for_motion_allowed": False,
            "qwen_required": True,
            "qwen_selected_anchor_id": "NO_SAFE_ANCHOR",
            "axis_mode": axis_mode,
            "requested_class": requested_material,
            "failure_reason": "NO_SAFE_ANCHOR",
        }
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
        _json(out / "final_anchor_decision_wide_context.json", final_failure)
        _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)
        no_safe_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=None,
            output_path=out / "no_safe_anchor_overlay.png",
            title="v2 anchor selection: NO_SAFE_ANCHOR",
        )
        payload = {
            "success": False,
            "axis_mode": axis_mode,
            "failure_reason": "NO_SAFE_ANCHOR",
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                **clean_contact_artifacts,
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
                "qwen_anchor_decision": str(out / "qwen_anchor_decision.json"),
                "no_safe_anchor_overlay": no_safe_overlay,
            },
            "overlay_paths": {"no_safe_anchor_overlay": no_safe_overlay},
            "diagnostics": {
                "qwen": qwen,
                "decision": final_failure,
                "robot_anchor_geometry": robot_anchor_geometry,
                "source": "v2a_wide_contact_qwen_required",
            },
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload
    selected_id = qwen.get("selected_anchor_id") if qwen.get("success") else None
    selected = next((item for item in candidates if item["anchor_id"] == selected_id), None)
    if not selected:
        failure_reason = str(qwen.get("failure_reason") or "QWEN_REQUIRED_BUT_FAILED")
        final_failure = _failure_decision(
            requested_material=requested_material,
            axis_mode=axis_mode,
            deterministic_selected=deterministic_selected_id,
            survivors=survivors,
            reason=failure_reason,
        )
        no_safe_overlay = save_anchor_overlay(
            rgb_path=rgb_path,
            mask_path=mask_path,
            anchors=candidates,
            selected_anchor_id=None,
            output_path=out / "no_safe_anchor_overlay.png",
            title=f"v2 anchor selection failed: {failure_reason}",
        )
        _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
        _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
        _json(out / "final_anchor_decision_wide_context.json", final_failure)
        payload = {
            "success": False,
            "axis_mode": axis_mode,
            "failure_reason": failure_reason,
            "candidate_count": len(candidates),
            "candidate_artifacts": {
                "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
                "combined_wide_contact_guided_grid": grid_path,
                "combined_wide_contact_guided_grid_2col": grid_2col_path,
                **clean_contact_artifacts,
                "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
                "no_safe_anchor_overlay": no_safe_overlay,
            },
            "overlay_paths": {"no_safe_anchor_overlay": no_safe_overlay},
            "diagnostics": {"qwen": qwen, "decision": final_failure, "source": "v2a_wide_contact_qwen_required"},
        }
        _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
        return payload

    final_source = qwen.get("final_selected_anchor_source") or "qwen_required"
    final_decision = {
        "phase": "upv_vlm_v2_v2a_wide_contact_qwen_required",
        "anchor_stage": "v2a_wide_contact_qwen_required",
        "requested_class": requested_material,
        "axis_mode": axis_mode,
        "qwen_required": True,
        "vlm_rerank_used": True,
        "deterministic_anchor_decision_is_final": False,
        "deterministic_selected_anchor": deterministic_selected_id,
        "qwen_selected_anchor": selected_id,
        "selected_anchor": selected_id,
        "final_decision": selected_id,
        "final_selected_anchor_source": final_source,
        "survivor_list": survivors,
        "qwen_reasoning_short": qwen.get("reasoning_short"),
        "qwen_response_repaired": bool(qwen.get("qwen_response_repaired", False)),
        "qwen_missing_assessment_ids": qwen.get("qwen_missing_assessment_ids", []),
        "no_safe_anchor_inconsistent": bool(qwen.get("no_safe_anchor_inconsistent", False)),
        "no_safe_anchor_repaired_to_anchor_id": qwen.get("no_safe_anchor_repaired_to_anchor_id"),
        "qwen_repair_reason": qwen.get("qwen_repair_reason"),
        "selected_anchor_before_repair": qwen.get("selected_anchor_before_repair"),
        "selected_anchor_after_repair": qwen.get("selected_anchor_after_repair"),
        "failure_reason": None,
        "anchor_count_generated": len(candidates),
        "anchor_count_rule_name": ANCHOR_COUNT_RULE_NAME,
    }
    robot_anchor_geometry = {
        "robot_candidate_anchor_id": selected_id,
        "robot_candidate_anchor_center_px": selected.get("anchor_px"),
        "robot_candidate_selection_mode": final_source,
        "robot_candidate_is_safe": not bool(selected.get("veto")),
        "robot_candidate_for_motion_allowed": not bool(selected.get("veto")),
        "qwen_required": True,
        "qwen_selected_anchor_id": selected_id,
        "deterministic_candidate_source": selected,
        "axis_mode": axis_mode,
        "requested_class": requested_material,
    }
    overlay = save_anchor_overlay(
        rgb_path=rgb_path,
        mask_path=mask_path,
        anchors=candidates,
        selected_anchor_id=selected_id,
        output_path=out / f"{axis_mode}_selected_anchor_overlay.png",
        title=f"v2 anchor selection axis_mode={axis_mode}",
    )
    final_overlay = save_anchor_overlay(
        rgb_path=rgb_path,
        mask_path=mask_path,
        anchors=candidates,
        selected_anchor_id=selected_id,
        output_path=out / "selected_anchor_overlay.png",
        title=(
            f"v2 anchor selection axis_mode={axis_mode}; source={final_source}"
            + ("; Qwen response repaired using deterministic features" if qwen.get("qwen_response_repaired") else "")
        ),
    )
    _json(out / f"{axis_mode}_anchor_candidates.json", {"axis_search_domain": domain.__dict__, "candidates": candidates})
    _json(out / f"{axis_mode}_anchor_scores.json", decision)
    _json(out / f"{axis_mode}_final_anchor.json", selected)
    _json(out / "deterministic_anchor_features_wide_context.json", deterministic_features)
    _json(out / "hybrid_decision_trace_wide_context.json", decision_trace)
    _json(out / "final_anchor_decision_wide_context.json", final_decision)
    _json(out / "robot_anchor_geometry.json", robot_anchor_geometry)
    payload = {
        "success": bool(selected),
        "axis_mode": axis_mode,
        "final_anchor_id": selected_id,
        "selected_anchor_px": robot_anchor_geometry["robot_candidate_anchor_center_px"],
        "contact_point_a_px": selected.get("contact_point_a_px"),
        "contact_point_b_px": selected.get("contact_point_b_px"),
        "local_cross_axis_vector": selected.get("local_cross_axis_vector"),
        "score": selected.get("score"),
        "candidate_count": len(candidates),
        "candidate_artifacts": {
            "anchor_candidates_path": str(out / f"{axis_mode}_anchor_candidates.json"),
            "anchor_scores_path": str(out / f"{axis_mode}_anchor_scores.json"),
            "final_anchor_path": str(out / f"{axis_mode}_final_anchor.json"),
            "boxed_wide_contact_guided_tiles_dir": str(tile_dir),
            "combined_wide_contact_guided_grid": grid_path,
            "combined_wide_contact_guided_grid_2col": grid_2col_path,
            **clean_contact_artifacts,
            "rotation_metadata": str(out / "rotation_metadata.json"),
            **rotated_artifacts,
            "deterministic_anchor_features_wide_context": str(out / "deterministic_anchor_features_wide_context.json"),
            "hybrid_decision_trace_wide_context": str(out / "hybrid_decision_trace_wide_context.json"),
            "final_anchor_decision_wide_context": str(out / "final_anchor_decision_wide_context.json"),
            "robot_anchor_geometry": str(out / "robot_anchor_geometry.json"),
            "qwen_anchor_decision": str(out / "qwen_anchor_decision.json"),
            "selected_anchor_overlay": final_overlay,
        },
        "overlay_paths": {"selected_anchor_overlay": final_overlay, "axis_selected_anchor_overlay": overlay, "anchor_candidates_overlay": overlay},
        "failure_reason": None,
        "diagnostics": {
            "anchor_backend": "v2a_wide_contact_qwen_required",
            "qwen_required": True,
            "qwen_selected_anchor_id": selected_id,
            "final_anchor_source": final_source,
            "qwen_response_repaired": bool(qwen.get("qwen_response_repaired", False)),
            "qwen_missing_assessment_ids": qwen.get("qwen_missing_assessment_ids", []),
            "selected_anchor_before_repair": qwen.get("selected_anchor_before_repair"),
            "selected_anchor_after_repair": qwen.get("selected_anchor_after_repair"),
            "final_selected_anchor_source": final_source,
            "robot_anchor_geometry": robot_anchor_geometry,
            "wide_contact_grid": grid_path,
            "wide_contact_grid_2col": grid_2col_path,
            "qwen_grid_source": "cropped_rotated_object_view",
            "rotation_metadata": rotation_metadata,
            "deterministic_survivors": survivors,
            "deterministic_decision": decision,
            "final_decision": final_decision,
            "qwen_decision": qwen,
            "anchor_count_metadata": anchor_count_metadata,
            "source": "v2a_wide_contact_qwen_required",
        },
    }
    _json(out / f"anchor_selection_{axis_mode}_summary.json", payload)
    return payload
