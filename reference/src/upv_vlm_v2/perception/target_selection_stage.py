"""Requested-material target selection stage for v2."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from upv_vlm_v2.geometry.overlays import save_selected_mask_overlay
from upv_vlm_v2.perception.candidate_pool import _save_mask  # internal validation helper
from upv_vlm_v2.perception.candidate_pool import CROP_PRIORITY
from upv_vlm_v2.perception.candidate_pool import EXPERIMENT_MODE
from upv_vlm_v2.perception.candidate_pool import build_candidate_pool
from upv_vlm_v2.perception.candidate_pool import crop_panel_candidate_boxes
from upv_vlm_v2.perception.candidate_pool import crop_panel_layout_metadata
from upv_vlm_v2.perception.candidate_pool import render_crop_verification_panel
from upv_vlm_v2.perception.candidate_pool import save_candidate_detection_overlay
from upv_vlm_v2.perception.candidate_pool import save_candidate_pool_manifest
from upv_vlm_v2.perception.candidate_pool import save_empty_candidate_pool_failure
from upv_vlm_v2.perception.clip_verifier import verify_candidate_crops_with_clip
from upv_vlm_v2.perception.grounded_sam2 import run_grounded_sam2_target_detection
from upv_vlm_v2.perception.target_prompt_contract import save_target_prompt_contract


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _copy_selected_mask(candidate: dict[str, Any], output_path: Path) -> str:
    source = Path(str(candidate["mask_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return str(output_path)


def _write_material_verification_artifacts(
    *,
    output_dir: Path,
    requested_material: str,
    verification: dict[str, Any],
) -> dict[str, str | None]:
    material_dir = output_dir / "material_verification"
    material_scores_path = material_dir / "material_scores.json"
    _write_json(material_scores_path, verification)
    selected_id = str(verification.get("selected_candidate_id") or "NO_VERIFIED_MATCH")
    candidates = list(verification.get("candidates", []))
    selected = next((item for item in candidates if str(item.get("candidate_id")) == selected_id), None)
    selected_summary_path: Path | None = None
    if selected:
        selected_summary = {
            "candidate_id": selected.get("candidate_id"),
            "selected": True,
            "requested_material": requested_material,
            "decision_backend": "clip_crop_verifier",
            "official_selector": "clip_crop_verifier",
            "selection_rule": selected.get("selection_rule", verification.get("selection_rule", "t5_v2_source_prior_color_texture_guard")),
            "final_selection_reason": verification.get("final_selection_reason", selected.get("final_selection_reason")),
            "selected_candidate_id_before_prior": verification.get("selected_candidate_id_before_prior"),
            "selected_candidate_id_after_prior": verification.get("selected_candidate_id_after_prior"),
            "selected_candidate_id_after_guards": verification.get("selected_candidate_id_after_guards"),
            "final_selected_candidate_id": verification.get("final_selected_candidate_id", selected.get("candidate_id")),
            "source_class_prior_bonus": verification.get("source_class_prior_bonus"),
            "cross_source_override_margin": verification.get("cross_source_override_margin"),
            "same_source_preferred": verification.get("same_source_preferred"),
            "source_class_prior_used": verification.get("source_class_prior_used"),
            "cross_source_override_used": verification.get("cross_source_override_used"),
            "beige_brick_guard_triggered": verification.get("beige_brick_guard_triggered"),
            "concrete_color_texture_guard_triggered": verification.get("concrete_color_texture_guard_triggered"),
            "timber_guard_triggered": verification.get("timber_guard_triggered"),
            "timber_guard_reason": verification.get("timber_guard_reason"),
            "crop_path_used_for_scoring": selected.get("crop_path_used_for_scoring") or selected.get("crop_used_for_verification"),
            "crop_type_used_for_scoring": selected.get("crop_type_used_for_scoring") or selected.get("crop_type_used_for_verification"),
            "material_scores": selected.get("clip_scores", {}),
            "source_detection_query": selected.get("source_detection_query", requested_material),
            "detection_score": selected.get("detection_score"),
            "clip_top_label": selected.get("clip_top_label"),
            "clip_top_score": selected.get("clip_top_score"),
            "requested_material_score": selected.get("requested_material_score", selected.get("clip_score_for_requested")),
            "allow_no_verified_match": verification.get("allow_no_verified_match"),
            "absent_class_rejection_enabled": verification.get("absent_class_rejection_enabled"),
            "target_absent_rejection_triggered": verification.get("target_absent_rejection_triggered"),
            "target_absent_rejection_reason": verification.get("target_absent_rejection_reason"),
            "selected_candidate_id_before_no_match_gate": verification.get("selected_candidate_id_before_no_match_gate"),
            "selected_candidate_id_after_no_match_gate": verification.get("selected_candidate_id_after_no_match_gate"),
            "no_verified_match": verification.get("no_verified_match", False),
        }
    else:
        selected_summary = {
            "candidate_id": "NO_VERIFIED_MATCH",
            "selected": False,
            "requested_material": requested_material,
            "decision_backend": "clip_crop_verifier",
            "official_selector": "clip_crop_verifier",
            "selection_rule": verification.get("selection_rule", "t5_v2_source_prior_color_texture_guard_with_no_match_gate"),
            "target_absent_rejection_triggered": verification.get("target_absent_rejection_triggered", True),
            "target_absent_rejection_reason": verification.get("target_absent_rejection_reason", verification.get("final_selection_reason")),
            "final_selection_reason": verification.get("final_selection_reason"),
            "final_selected_candidate_id": "NO_VERIFIED_MATCH",
            "selected_candidate_id_before_prior": verification.get("selected_candidate_id_before_prior"),
            "selected_candidate_id_after_prior": verification.get("selected_candidate_id_after_prior"),
            "selected_candidate_id_after_guards": verification.get("selected_candidate_id_after_guards"),
            "selected_candidate_id_before_no_match_gate": verification.get("selected_candidate_id_before_no_match_gate"),
            "selected_candidate_id_after_no_match_gate": verification.get("selected_candidate_id_after_no_match_gate"),
            "allow_no_verified_match": verification.get("allow_no_verified_match"),
            "absent_class_rejection_enabled": verification.get("absent_class_rejection_enabled"),
            "no_verified_match": True,
        }
    selected_summary_path = material_dir / "selected_candidate_summary.json"
    _write_json(selected_summary_path, selected_summary)
    panel_candidates: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        panel_candidates.append(
            {
                "panel_label": f"C{idx}",
                "candidate_id": candidate.get("candidate_id"),
                "source_detection_query": candidate.get("source_detection_query", requested_material),
                "bbox_crop_path": candidate.get("bbox_crop_path"),
                "masked_texture_crop_path": candidate.get("masked_texture_crop_path"),
                "inner_texture_crop_path": candidate.get("inner_texture_crop_path"),
                "inner_crop_valid": bool(candidate.get("inner_crop_valid", candidate.get("inner_texture_crop_usable", False))),
                "inner_crop_erode_px": candidate.get("inner_crop_erode_px"),
                "inner_crop_pixel_count": candidate.get("inner_crop_pixel_count", candidate.get("inner_texture_foreground_pixels")),
                "fallback_reason": candidate.get("fallback_reason"),
                "crop_used_for_verification": candidate.get("crop_used_for_verification") or candidate.get("crop_path_used_for_scoring"),
                "crop_used_for_verification_path": candidate.get("crop_used_for_verification_path") or candidate.get("crop_used_for_verification") or candidate.get("crop_path_used_for_scoring"),
                "crop_used_for_verification_resolved": candidate.get("crop_used_for_verification") or candidate.get("crop_path_used_for_scoring"),
                "crop_type_used_for_verification": candidate.get("crop_type_used_for_verification") or candidate.get("crop_type_used_for_scoring"),
                "verification_scores": candidate.get("clip_scores", {}),
                "clip_labels": candidate.get("clip_labels", []),
                "clip_scores": candidate.get("clip_scores", {}),
                "clip_aggregate_scores": candidate.get("clip_aggregate_scores", candidate.get("clip_scores", {})),
                "clip_per_prompt_scores": candidate.get("clip_per_prompt_scores", {}),
                "clip_top_label": candidate.get("clip_top_label"),
                "clip_top_score": candidate.get("clip_top_score"),
                "requested_material_score": candidate.get("requested_material_score", candidate.get("clip_score_for_requested")),
                "selection_rule": candidate.get("selection_rule", verification.get("selection_rule", "t5_v2_source_prior_color_texture_guard")),
                "requested_score": candidate.get("requested_score"),
                "score_margin": candidate.get("score_margin"),
                "final_score": candidate.get("final_score"),
                "color_texture_features": candidate.get("color_texture_features", {}),
                "grey_score": candidate.get("grey_score"),
                "warm_beige_score": candidate.get("warm_beige_score"),
                "cement_texture_score": candidate.get("cement_texture_score"),
                "wood_like_score": candidate.get("wood_like_score"),
                "final_selected": bool(candidate.get("final_selected")),
                "match_assessment": (verification.get("candidate_match_assessments") or {}).get(str(candidate.get("candidate_id")), {}),
            }
        )
    columns = 2 if len(panel_candidates) > 1 else 1
    panel_layout = crop_panel_layout_metadata(columns=columns)
    crop_panel_manifest = {
        "experiment_mode": EXPERIMENT_MODE,
        "candidate_count": len(panel_candidates),
        "oracle_known_classes_used": False,
        "active_prompt_stages": ["open_vocab_detection", "material_crop_verification"],
        "panel_layout": panel_layout,
        "candidate_panel_boxes": crop_panel_candidate_boxes(candidates=candidates, columns=columns),
        "crop_priority": CROP_PRIORITY,
        "decision_backend": "clip_crop_verifier",
        "official_selector": "clip_crop_verifier",
        "vlm_decision_attempted": False,
        "qwen_vlm_server_enabled": False,
        "selection_rule": verification.get("selection_rule", "t5_v2_source_prior_color_texture_guard"),
        "final_selected_candidate_id": verification.get("final_selected_candidate_id", selected_id),
        "no_verified_match": bool(verification.get("no_verified_match", selected_id == "NO_VERIFIED_MATCH")),
        "target_absent_rejection_triggered": bool(verification.get("target_absent_rejection_triggered", False)),
        "target_absent_rejection_reason": verification.get("target_absent_rejection_reason"),
        "selected_candidate_id_before_no_match_gate": verification.get("selected_candidate_id_before_no_match_gate"),
        "selected_candidate_id_after_no_match_gate": verification.get("selected_candidate_id_after_no_match_gate"),
        "candidate_match_assessments": verification.get("candidate_match_assessments", {}),
        "final_selection_reason": verification.get("final_selection_reason"),
        "selected_candidate_id_before_prior": verification.get("selected_candidate_id_before_prior"),
        "selected_candidate_id_after_prior": verification.get("selected_candidate_id_after_prior"),
        "selected_candidate_id_after_guards": verification.get("selected_candidate_id_after_guards"),
        "source_class_prior_bonus": verification.get("source_class_prior_bonus"),
        "cross_source_override_margin": verification.get("cross_source_override_margin"),
        "same_source_preferred": verification.get("same_source_preferred"),
        "beige_brick_guard_triggered": verification.get("beige_brick_guard_triggered"),
        "concrete_color_texture_guard_triggered": verification.get("concrete_color_texture_guard_triggered"),
        "timber_guard_triggered": verification.get("timber_guard_triggered"),
        "candidates": panel_candidates,
    }
    crop_panel_manifest_path = material_dir / "crop_panel_manifest.json"
    _write_json(crop_panel_manifest_path, crop_panel_manifest)
    panel_path = render_crop_verification_panel(
        candidates=candidates,
        selected_candidate_id=selected_id,
        requested_material=requested_material,
        output_path=material_dir / "crop_verification_panel.png",
        columns=columns,
    )
    return {
        "material_verification_dir": str(material_dir),
        "material_scores_path": str(material_scores_path),
        "selected_candidate_summary_path": str(selected_summary_path) if selected_summary_path else None,
        "crop_panel_manifest_path": str(crop_panel_manifest_path),
        "crop_verification_panel_path": panel_path,
    }


def _write_empty_material_failure(*, output_dir: Path, requested_material: str, failure_reason: str) -> dict[str, str]:
    material_dir = output_dir / "material_verification"
    payload = {
        "requested_material": requested_material,
        "decision_backend": "clip_crop_verifier",
        "official_selector": "clip_crop_verifier",
        "qwen_vlm_server_enabled": False,
        "require_vlm_decision": False,
        "selected_candidate_id": "NO_VERIFIED_MATCH",
        "final_selected_candidate_id": "NO_VERIFIED_MATCH",
        "no_verified_match": True,
        "allow_no_verified_match": True,
        "absent_class_rejection_enabled": True,
        "target_absent_rejection_triggered": True,
        "target_absent_rejection_reason": failure_reason,
        "failure_reason": failure_reason,
        "candidate_scores": [],
        "candidates": [],
    }
    material_scores_path = material_dir / "material_scores.json"
    _write_json(material_scores_path, payload)
    crop_panel_manifest_path = material_dir / "crop_panel_manifest.json"
    _write_json(
        crop_panel_manifest_path,
        {
            "experiment_mode": EXPERIMENT_MODE,
            "candidate_count": 0,
            "oracle_known_classes_used": False,
            "active_prompt_stages": ["open_vocab_detection", "material_crop_verification"],
            "panel_layout": crop_panel_layout_metadata(columns=1),
            "candidate_panel_boxes": [],
            "crop_priority": CROP_PRIORITY,
            "decision_backend": "clip_crop_verifier",
            "official_selector": "clip_crop_verifier",
            "vlm_decision_attempted": False,
            "qwen_vlm_server_enabled": False,
            "final_selected_candidate_id": "NO_VERIFIED_MATCH",
            "no_verified_match": True,
            "target_absent_rejection_triggered": True,
            "target_absent_rejection_reason": failure_reason,
            "failure_reason": failure_reason,
            "candidates": [],
        },
    )
    selected_candidate_summary_path = material_dir / "selected_candidate_summary.json"
    _write_json(
        selected_candidate_summary_path,
        {
            "candidate_id": "NO_VERIFIED_MATCH",
            "selected": False,
            "requested_material": requested_material,
            "decision_backend": "clip_crop_verifier",
            "official_selector": "clip_crop_verifier",
            "selection_rule": "t5_v2_source_prior_color_texture_guard_with_no_match_gate",
            "target_absent_rejection_triggered": True,
            "target_absent_rejection_reason": failure_reason,
            "final_selected_candidate_id": "NO_VERIFIED_MATCH",
            "no_verified_match": True,
        },
    )
    panel_path = render_crop_verification_panel(
        candidates=[],
        selected_candidate_id=None,
        requested_material=requested_material,
        output_path=material_dir / "crop_verification_panel.png",
    )
    return {
        "material_verification_dir": str(material_dir),
        "material_scores_path": str(material_scores_path),
        "selected_candidate_summary_path": str(selected_candidate_summary_path),
        "crop_panel_manifest_path": str(crop_panel_manifest_path),
        "crop_verification_panel_path": panel_path,
    }


def run_target_selection(
    *,
    rgb_path: str | Path,
    requested_material: str,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied_rgb = out / "raw_rgb.png"
    if Path(rgb_path).resolve() != copied_rgb.resolve():
        shutil.copy2(rgb_path, copied_rgb)
    if bool(config.get("testing", {}).get("synthetic_target_from_saved_input", False)):
        import numpy as np
        from PIL import Image

        image = Image.open(copied_rgb).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        x0, x1 = int(image.width * 0.25), int(image.width * 0.75)
        y0, y1 = int(image.height * 0.35), int(image.height * 0.65)
        mask[y0:y1, x0:x1] = 1
        detections = [{"mask": mask, "bbox": [x0, y0, x1, y1], "score": 0.99, "candidate_generation_source": "synthetic_saved_input"}]
        candidates = build_candidate_pool(
            rgb_path=copied_rgb,
            detections=detections,
            output_dir=out,
            source_detection_query=requested_material,
            pad_px=int(config.get("target_selection", {}).get("crop_pad_px", 8)),
            erode_px=int(config.get("target_selection", {}).get("inner_crop_erode_px", 3)),
            min_inner_crop_pixels=int(config.get("target_selection", {}).get("min_inner_crop_pixels", 1000)),
        )
        prompt_paths = save_target_prompt_contract(
            output_dir=out,
            requested_material=requested_material,
            candidate_ids=[str(item.get("candidate_id")) for item in candidates],
            config=config,
        )
        candidate_pool_manifest = save_candidate_pool_manifest(output_dir=out, requested_material=requested_material, candidates=candidates)
        detection_overlay = save_candidate_detection_overlay(
            rgb_path=copied_rgb,
            output_dir=out,
            requested_material=requested_material,
            candidates=candidates,
        )
        material_scores_path = out / "material_verification" / "material_scores.json"
        verification = verify_candidate_crops_with_clip(
            candidates=candidates,
            requested_material=requested_material,
            config=config,
            output_path=material_scores_path,
        )
        material_paths = _write_material_verification_artifacts(output_dir=out, requested_material=requested_material, verification=verification)
        selected_id = str(verification.get("selected_candidate_id") or "NO_VERIFIED_MATCH")
        selected = next((item for item in verification.get("candidates", candidates) if item.get("candidate_id") == selected_id), None)
        if selected is None:
            payload = {
                "success": False,
                "failure_reason": "NO_VERIFIED_MATCH",
                "requested_material": requested_material,
                "selected_candidate_id": "NO_VERIFIED_MATCH",
                "candidate_count": len(candidates),
                "candidate_artifacts": {item["candidate_id"]: item for item in verification.get("candidates", candidates)},
                "candidate_pool_dir": str(out / "candidate_pool"),
                "candidate_pool_manifest_path": candidate_pool_manifest,
                "detection_overlay_path": detection_overlay,
                **material_paths,
                **prompt_paths,
                "clip_scores_path": material_paths["material_scores_path"],
                "detection_queries_used": [requested_material],
                "classes_existing_used_as_model_input": False,
                "diagnostics": {"source": "validation_synthetic_target_from_saved_input", "no_verified_match": True},
            }
            _write_json(out / "target_selection_result.json", payload)
            return payload
        mask_path = Path(_copy_selected_mask(selected, out / "selected_mask.png"))
        overlay = save_selected_mask_overlay(
            rgb_path=copied_rgb,
            mask_path=mask_path,
            output_path=out / "selected_mask_overlay.png",
            header=["v2 target selection", "validation synthetic target from saved input", f"selected: {selected_id}"],
        )
        payload = {
            "success": True,
            "failure_reason": None,
            "requested_material": requested_material,
            "selected_candidate_id": selected_id,
            "selected_mask_path": str(mask_path),
            "selected_rgb_path": str(copied_rgb),
            "selected_mask_overlay_path": overlay,
            "candidate_count": 1,
            "candidate_artifacts": {item["candidate_id"]: item for item in verification.get("candidates", candidates)},
            "candidate_pool_dir": str(out / "candidate_pool"),
            "candidate_pool_manifest_path": candidate_pool_manifest,
            "detection_overlay_path": detection_overlay,
            "material_verification_dir": material_paths["material_verification_dir"],
            "material_scores_path": material_paths["material_scores_path"],
            "selected_candidate_summary_path": material_paths.get("selected_candidate_summary_path"),
            "crop_panel_manifest_path": material_paths["crop_panel_manifest_path"],
            "crop_verification_panel_path": material_paths["crop_verification_panel_path"],
            **prompt_paths,
            "clip_scores_path": material_paths["material_scores_path"],
            "detection_queries_used": [requested_material],
            "classes_existing_used_as_model_input": False,
            "diagnostics": {"source": "validation_synthetic_target_from_saved_input"},
        }
        _write_json(out / "target_selection_result.json", payload)
        return payload
    gsam2_dir = out / "grounded_sam2"
    gsam2 = run_grounded_sam2_target_detection(
        image_path=copied_rgb,
        requested_material=requested_material,
        config=config,
        output_dir=gsam2_dir,
    )
    if not gsam2.get("success"):
        payload = {
            "success": False,
            "failure_reason": gsam2.get("failure_reason", "grounded_sam2_failed"),
            "requested_material": requested_material,
            "detection_queries_used": [requested_material],
            "grounded_sam2": {key: value for key, value in gsam2.items() if key != "detections"},
        }
        _write_json(out / "target_selection_result.json", payload)
        return payload
    candidates = build_candidate_pool(
        rgb_path=copied_rgb,
        detections=list(gsam2.get("detections", [])),
        output_dir=out,
        source_detection_query=requested_material,
        pad_px=int(config.get("target_selection", {}).get("crop_pad_px", 8)),
        erode_px=int(config.get("target_selection", {}).get("inner_crop_erode_px", 3)),
        min_inner_crop_pixels=int(config.get("target_selection", {}).get("min_inner_crop_pixels", 1000)),
    )
    prompt_paths = save_target_prompt_contract(
        output_dir=out,
        requested_material=requested_material,
        candidate_ids=[str(item.get("candidate_id")) for item in candidates],
        config=config,
    )
    candidate_pool_manifest = save_candidate_pool_manifest(output_dir=out, requested_material=requested_material, candidates=candidates)
    detection_overlay = save_candidate_detection_overlay(rgb_path=copied_rgb, output_dir=out, requested_material=requested_material, candidates=candidates)
    if not candidates:
        save_empty_candidate_pool_failure(output_dir=out, requested_material=requested_material, failure_reason="no_candidate_masks")
        material_paths = _write_empty_material_failure(output_dir=out, requested_material=requested_material, failure_reason="no_candidate_masks")
        payload = {
            "success": False,
            "failure_reason": "no_candidate_masks",
            "requested_material": requested_material,
            "candidate_count": 0,
            "detection_queries_used": [requested_material],
            "candidate_pool_dir": str(out / "candidate_pool"),
            "candidate_pool_manifest_path": candidate_pool_manifest,
            **prompt_paths,
            **material_paths,
        }
        _write_json(out / "target_selection_result.json", payload)
        return payload
    clip_scores_path = out / "material_verification" / "material_scores.json"
    verification = verify_candidate_crops_with_clip(
        candidates=candidates,
        requested_material=requested_material,
        config=config,
        output_path=clip_scores_path,
    )
    material_paths = _write_material_verification_artifacts(output_dir=out, requested_material=requested_material, verification=verification)
    selected_id = str(verification.get("selected_candidate_id") or "NO_VERIFIED_MATCH")
    by_id = {str(item["candidate_id"]): item for item in verification.get("candidates", candidates)}
    selected = by_id.get(selected_id)
    if selected is None and selected_id != "NO_VERIFIED_MATCH":
        selected = next((item for item in candidates if item["candidate_id"] == selected_id), None)
    if selected is None and not bool(config.get("target_selection", {}).get("use_clip_crop_verifier", True)):
        selected = candidates[0]
        selected_id = selected["candidate_id"]
    if selected is None:
        payload = {
            "success": False,
            "failure_reason": "NO_VERIFIED_MATCH",
            "requested_material": requested_material,
            "candidate_count": len(candidates),
            "candidate_artifacts": {item["candidate_id"]: item for item in candidates},
            "candidate_pool_dir": str(out / "candidate_pool"),
            "candidate_pool_manifest_path": candidate_pool_manifest,
            "detection_overlay_path": detection_overlay,
            "material_verification_dir": material_paths["material_verification_dir"],
            "material_scores_path": material_paths["material_scores_path"],
            "selected_candidate_summary_path": material_paths.get("selected_candidate_summary_path"),
            "crop_panel_manifest_path": material_paths["crop_panel_manifest_path"],
            "crop_verification_panel_path": material_paths["crop_verification_panel_path"],
            **prompt_paths,
            "clip_scores_path": str(clip_scores_path),
            "detection_queries_used": [requested_material],
            "grounded_sam2": {key: value for key, value in gsam2.items() if key != "detections"},
        }
        _write_json(out / "target_selection_result.json", payload)
        return payload
    selected_mask = _copy_selected_mask(selected, out / "selected_mask.png")
    overlay = save_selected_mask_overlay(
        rgb_path=copied_rgb,
        mask_path=selected_mask,
        output_path=out / "selected_mask_overlay.png",
        header=[
            "v2 target selection",
            f"requested: {requested_material}",
            f"selected: {selected_id}",
            "official selector: CLIP crop verifier" if config.get("target_selection", {}).get("use_clip_crop_verifier", True) else "official selector: detection score fallback",
        ],
    )
    payload = {
        "success": True,
        "failure_reason": None,
        "requested_material": requested_material,
        "selected_candidate_id": selected_id,
        "selected_mask_path": selected_mask,
        "selected_rgb_path": str(copied_rgb),
        "selected_mask_overlay_path": overlay,
        "candidate_count": len(candidates),
        "candidate_artifacts": {item["candidate_id"]: item for item in verification.get("candidates", candidates)},
        "candidate_pool_dir": str(out / "candidate_pool"),
        "candidate_pool_manifest_path": candidate_pool_manifest,
        "detection_overlay_path": detection_overlay,
        "material_verification_dir": material_paths["material_verification_dir"],
        "material_scores_path": material_paths["material_scores_path"],
        "selected_candidate_summary_path": material_paths.get("selected_candidate_summary_path"),
        "crop_panel_manifest_path": material_paths["crop_panel_manifest_path"],
        "crop_verification_panel_path": material_paths["crop_verification_panel_path"],
        **prompt_paths,
        "clip_scores_path": str(clip_scores_path),
        "detection_queries_used": [requested_material],
        "classes_existing_used_as_model_input": False,
        "grounded_sam2": {key: value for key, value in gsam2.items() if key != "detections"},
    }
    _write_json(out / "target_selection_result.json", payload)
    return payload
