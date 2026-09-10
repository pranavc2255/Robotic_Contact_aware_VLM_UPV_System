"""CLIP crop verifier for v2 target selection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


MATERIAL_LABEL_PROMPTS: dict[str, list[str]] = {
    "brick": [
        "warm red brick surface",
        "reddish brown brick texture",
        "orange brown clay brick surface",
        "warm tan clay brick texture",
        "warm beige clay brick texture",
        "cream beige brick surface",
        "yellowish tan brick surface",
        "warm off white clay brick surface under bright light",
        "earthy warm masonry brick surface",
        "matte warm clay construction material",
    ],
    "concrete block": [
        "cool cement gray concrete block texture",
        "neutral gray CMU surface",
        "ash gray cementitious block texture",
        "light cool gray cement block surface",
        "gray aggregate concrete block surface",
        "chalky cool gray cement surface",
        "matte neutral gray concrete masonry unit",
        "cool gray cast cement surface",
        "pale gray cement block material",
    ],
    "timber": [
        "wood grain surface with long parallel fibers",
        "pale tan lumber with linear grain",
        "sawn timber surface with fiber lines",
        "natural wood texture with visible grain",
        "light pine wood grain",
        "tan brown fibrous wood surface",
        "cut lumber face with subtle grain",
        "smooth pale timber surface with linear fibers",
    ],
}
SELECTION_RULE = "t5_v2_source_prior_color_texture_guard_with_no_match_gate"


def _prompt_rows() -> list[dict[str, str]]:
    return [{"material": material, "prompt": prompt} for material, prompts in MATERIAL_LABEL_PROMPTS.items() for prompt in prompts]


def _aggregate(rows: list[dict[str, str]], labels: list[str], scores: list[float], method: str) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    by_label = {str(label): float(score) for label, score in zip(labels, scores)}
    per_prompt: dict[str, list[dict[str, Any]]] = {key: [] for key in MATERIAL_LABEL_PROMPTS}
    for row in rows:
        score = float(by_label.get(row["prompt"], 0.0))
        per_prompt[row["material"]].append({"label": row["prompt"], "score": round(score, 6)})
    agg: dict[str, float] = {}
    for material, items in per_prompt.items():
        values = [float(item["score"]) for item in items]
        if method == "mean":
            agg[material] = round(sum(values) / max(len(values), 1), 6)
        else:
            agg[material] = round(max(values) if values else 0.0, 6)
    return agg, per_prompt


def _normalize_material_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    if text in {"concrete", "concrete_block", "concrete block", "cmu", "cinder block"}:
        return "concrete block"
    if text in {"wood", "lumber", "timber"}:
        return "timber"
    if text in {"brick", "clay brick", "fired clay brick"}:
        return "brick"
    return text


def analyze_crop_color_texture(crop_path: str | Path | None) -> dict[str, Any]:
    if not crop_path or not Path(str(crop_path)).exists():
        return {
            "available": False,
            "mean_rgb": [0.0, 0.0, 0.0],
            "saturation": 0.0,
            "brightness": 0.0,
            "grey_score": 0.0,
            "warm_color_score": 0.0,
            "warm_beige_score": 0.0,
            "texture_variance": 0.0,
            "edge_variance": 0.0,
            "cement_texture_score": 0.0,
            "wood_like_score": 0.0,
        }
    image = Image.open(crop_path).convert("RGB").resize((96, 96))
    arr = np.asarray(image).astype(np.float32) / 255.0
    nonwhite = np.mean(arr, axis=2) < 0.96
    pixels = arr[nonwhite] if np.any(nonwhite) else arr.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    r, g, b = [float(value) for value in mean]
    maxc = pixels.max(axis=1)
    minc = pixels.min(axis=1)
    sat = float(np.mean((maxc - minc) / np.maximum(maxc, 1e-6)))
    channel_std = float(np.mean(np.std(pixels, axis=1)))
    warmth = float(np.clip((r - b) * 1.7 + (r - g) * 0.7 + sat * 0.15, 0.0, 1.0))
    brightness = float(np.mean(pixels))
    red_dominance = float(np.clip((r - b) * 4.0 + (r - g) * 2.0, 0.0, 1.0))
    cool_grey_bonus = float(np.clip((b - r) * 2.5 + (g - r) * 0.8, 0.0, 0.35))
    gray_channel_balance = float(np.clip(1.0 - channel_std / 0.24, 0.0, 1.0))
    low_saturation = float(np.clip(1.0 - sat / 0.42, 0.0, 1.0))
    grayness = float(np.clip(low_saturation * (1.0 - 0.88 * red_dominance) + cool_grey_bonus, 0.0, 1.0))
    warm_brightness = float(np.clip(1.0 - abs(brightness - 0.62) / 0.46, 0.0, 1.0))
    warm_beige = float(np.clip(red_dominance * warm_brightness * (1.0 - max(0.0, sat - 0.62) * 0.8), 0.0, 1.0))
    gray = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    valid_gray = gray[nonwhite] if np.any(nonwhite) else gray.reshape(-1)
    texture_variance = float(np.clip(np.std(valid_gray) / 0.22, 0.0, 1.0))
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    edge_variance = float(np.clip((np.std(dx) + np.std(dy)) / 0.26, 0.0, 1.0))
    cement_texture = float(np.clip(0.65 * grayness + 0.20 * low_saturation + 0.15 * max(texture_variance, edge_variance), 0.0, 1.0))
    brownness = float(np.clip(warmth * 0.75 + sat * 0.2 + (0.65 - abs(brightness - 0.45)) * 0.1, 0.0, 1.0))
    return {
        "available": True,
        "mean_rgb": [round(r, 4), round(g, 4), round(b, 4)],
        "saturation": round(sat, 4),
        "brightness": round(brightness, 4),
        "grey_score": round(grayness, 4),
        "red_dominance_score": round(red_dominance, 4),
        "cool_grey_bonus": round(cool_grey_bonus, 4),
        "warm_color_score": round(warmth, 4),
        "warm_beige_score": round(warm_beige, 4),
        "texture_variance": round(texture_variance, 4),
        "edge_variance": round(edge_variance, 4),
        "cement_texture_score": round(cement_texture, 4),
        "wood_like_score": round(brownness, 4),
    }


def _mock_per_prompt_scores(aggregate: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    return {
        material: [{"label": prompt, "score": round(float(aggregate.get(material, 0.0)), 6)} for prompt in prompts]
        for material, prompts in MATERIAL_LABEL_PROMPTS.items()
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_match_assessment(item: dict[str, Any], requested: str, target_cfg: dict[str, Any]) -> dict[str, Any]:
    scores = item.get("clip_scores", {}) or {}
    requested_score = _safe_float(scores.get(requested))
    brick_score = _safe_float(scores.get("brick"))
    timber_score = _safe_float(scores.get("timber"))
    concrete_score = _safe_float(scores.get("concrete block"))
    top_label = _normalize_material_name(item.get("clip_top_label"))
    grey_score = _safe_float(item.get("grey_score"))
    cement_score = _safe_float(item.get("cement_texture_score"))
    warm_beige_score = _safe_float(item.get("warm_beige_score"))
    wood_like_score = max(_safe_float(item.get("wood_like_score")), timber_score)
    masonry_support = max(brick_score, concrete_score, grey_score * 0.5, cement_score * 0.5)
    grey_cement_support = max(concrete_score, grey_score, cement_score)
    wood_support = max(timber_score, wood_like_score)
    conflict_material = "none"
    conflict_score = 0.0
    if requested == "brick":
        conflict_material = "timber" if timber_score >= max(concrete_score, 0.0) else "concrete block"
        conflict_score = max(timber_score, concrete_score)
        min_support = float(target_cfg.get("brick_min_masonry_support_score", 0.15))
        close_masonry_margin = float(target_cfg.get("brick_accept_concrete_top_if_close_masonry_margin", 0.08))
        is_match = (
            (requested_score >= float(target_cfg.get("min_requested_material_score", 0.10)) and top_label != "timber")
            or (masonry_support >= min_support and timber_score < max(0.45, masonry_support + 0.08))
            or (
                top_label == "concrete block"
                and max(brick_score, concrete_score) >= min_support
                and concrete_score <= brick_score + close_masonry_margin
                and wood_like_score < float(target_cfg.get("brick_reject_if_wood_like_score_above", 0.45))
            )
        )
        if top_label == "timber" and wood_support >= float(target_cfg.get("brick_reject_if_wood_like_score_above", 0.45)):
            is_match = False
    elif requested == "timber":
        conflict_material = "concrete block" if concrete_score >= brick_score else "brick"
        conflict_score = max(brick_score, concrete_score)
        masonry_score = max(brick_score, concrete_score)
        timber_margin_allowance = float(target_cfg.get("timber_top_label_accept_margin", 0.03))
        strong_masonry_margin = float(target_cfg.get("timber_reject_if_masonry_exceeds_timber_by", 0.08))
        min_wood_support = float(target_cfg.get("timber_min_wood_like_score", 0.25))
        if top_label == "timber" and timber_score >= masonry_score - timber_margin_allowance:
            is_match = True
        elif wood_support >= min_wood_support and top_label not in {"brick", "concrete block"}:
            is_match = True
        elif top_label == "timber" and masonry_score <= timber_score + timber_margin_allowance:
            is_match = True
        else:
            is_match = False
        if masonry_score >= timber_score + strong_masonry_margin and top_label in {"brick", "concrete block"}:
            is_match = False
    elif requested == "concrete block":
        conflict_material = "timber" if timber_score >= brick_score else "brick"
        conflict_score = max(timber_score, brick_score)
        min_grey_cement = float(target_cfg.get("concrete_min_grey_or_cement_score", 0.30))
        warm_veto = float(target_cfg.get("concrete_reject_if_warm_beige_or_brick_like_above", 0.35))
        warm_margin = float(target_cfg.get("concrete_warm_beige_veto_margin", 0.05))
        require_cool_gray = bool(target_cfg.get("concrete_require_cool_gray_not_warm_beige", True))
        is_match = grey_cement_support >= min_grey_cement and top_label != "timber"
        concrete_advantage_required = 0.08
        if top_label == "brick" and not (concrete_score >= brick_score + concrete_advantage_required and grey_cement_support >= min_grey_cement):
            is_match = False
        warm_brick_like = (
            warm_beige_score >= warm_veto
            or (warm_beige_score >= float(target_cfg.get("brick_accept_if_warm_beige_score_above", 0.10)) and concrete_score <= brick_score + concrete_advantage_required)
            or (brick_score >= float(target_cfg.get("min_requested_material_score", 0.10)) and concrete_score <= brick_score + concrete_advantage_required)
        )
        if require_cool_gray and warm_brick_like and concrete_score <= brick_score + max(warm_margin, concrete_advantage_required):
            is_match = False
    else:
        is_match = requested_score >= float(target_cfg.get("min_requested_material_score", 0.10))
    reason = "passes requested-material consistency gates" if is_match else "conflicts with requested material or lacks requested-material support"
    return {
        "requested_material_score": round(requested_score, 6),
        "clip_top_label": top_label or item.get("clip_top_label"),
        "is_confident_requested_match": bool(is_match),
        "conflict_material": conflict_material,
        "conflict_score": round(conflict_score, 6),
        "brick_score": round(brick_score, 6),
        "timber_score": round(timber_score, 6),
        "concrete_block_score": round(concrete_score, 6),
        "masonry_support_score": round(masonry_support, 6),
        "grey_cement_support_score": round(grey_cement_support, 6),
        "wood_support_score": round(wood_support, 6),
        "wood_like_score": round(wood_like_score, 6),
        "grey_score": round(grey_score, 6),
        "cement_texture_score": round(cement_score, 6),
        "warm_beige_score": round(warm_beige_score, 6),
        "match_decision_reason": reason,
    }


def _no_match_gate(
    *,
    selected: dict[str, Any] | None,
    scored: list[dict[str, Any]],
    requested: str,
    target_cfg: dict[str, Any],
    threshold: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    allow_no_match = bool(target_cfg.get("allow_no_verified_match", True))
    enabled = bool(target_cfg.get("absent_class_rejection_enabled", True))
    assessments = {
        str(item.get("candidate_id")): _candidate_match_assessment(item, requested, target_cfg)
        for item in scored
    }
    meta = {
        "allow_no_verified_match": allow_no_match,
        "absent_class_rejection_enabled": enabled,
        "candidate_match_assessments": assessments,
    }
    if not allow_no_match or not enabled:
        return False, None, meta
    if selected is None:
        return True, "no tentative selected candidate", meta

    selected_id = str(selected.get("candidate_id"))
    selected_assessment = assessments.get(selected_id, {})
    scores = selected.get("clip_scores", {}) or {}
    req_score = _safe_float(scores.get(requested))
    brick_score = _safe_float(scores.get("brick"))
    timber_score = _safe_float(scores.get("timber"))
    concrete_score = _safe_float(scores.get("concrete block"))
    top_label = _normalize_material_name(selected.get("clip_top_label"))
    min_req = float(target_cfg.get("min_requested_material_score", max(threshold, 0.10)))
    min_margin = float(target_cfg.get("min_requested_margin_over_reject_material", 0.02))
    other_scores = [_safe_float(value) for key, value in scores.items() if _normalize_material_name(key) != requested]
    margin = req_score - (max(other_scores) if other_scores else 0.0)

    if req_score < max(threshold, min_req) and not selected_assessment.get("is_confident_requested_match"):
        return True, f"requested material score below no-match minimum ({req_score:.4f} < {max(threshold, min_req):.4f})", meta

    if bool(target_cfg.get("reject_if_all_candidates_conflict", True)) and assessments:
        if not any(bool(item.get("is_confident_requested_match")) for item in assessments.values()):
            return True, "all candidates conflict with requested material", meta

    if selected_assessment.get("is_confident_requested_match"):
        return False, None, meta

    if bool(target_cfg.get("reject_if_top_label_conflicts_with_requested", True)) and top_label and top_label != requested and margin < min_margin:
        if requested == "brick" and top_label == "concrete block" and max(brick_score, concrete_score) >= float(
            target_cfg.get("brick_min_masonry_support_score", 0.15)
        ):
            return False, None, meta
        return True, f"top CLIP label conflicts with requested material: {top_label}", meta

    if requested == "brick":
        masonry_support = max(brick_score, concrete_score)
        wood_support = max(timber_score, _safe_float(selected.get("wood_like_score")))
        min_masonry = float(target_cfg.get("brick_min_masonry_support_score", 0.18))
        if bool(target_cfg.get("brick_reject_if_timber_top_label", True)) and top_label == "timber" and wood_support >= float(
            target_cfg.get("brick_reject_if_wood_like_score_above", 0.45)
        ) and masonry_support < min_masonry:
            return True, "requested brick but selected candidate is timber/wood-like without masonry support", meta
        if bool(target_cfg.get("brick_reject_if_timber_score_above_and_no_masonry_support", True)) and timber_score >= float(
            target_cfg.get("brick_reject_if_wood_like_score_above", 0.45)
        ) and masonry_support < min_masonry:
            return True, "requested brick but timber score is high and masonry support is weak", meta
    elif requested == "timber":
        wood_support = max(timber_score, _safe_float(selected.get("wood_like_score")))
        if bool(target_cfg.get("timber_reject_if_masonry_top_label", True)) and top_label in {"brick", "concrete block"} and wood_support < float(
            target_cfg.get("timber_min_wood_like_score", 0.25)
        ):
            return True, "requested timber but selected candidate is masonry-like with weak wood support", meta
    elif requested == "concrete block":
        grey_support = max(concrete_score, _safe_float(selected.get("grey_score")), _safe_float(selected.get("cement_texture_score")))
        if bool(target_cfg.get("concrete_reject_if_timber_top_label", True)) and top_label == "timber":
            return True, "requested concrete block but selected candidate is timber-like", meta
        if _safe_float(selected.get("warm_beige_score")) >= float(target_cfg.get("concrete_reject_if_warm_beige_or_brick_like_above", 0.45)) and grey_support < float(
            target_cfg.get("concrete_min_grey_or_cement_score", 0.30)
        ):
            return True, "requested concrete block but candidate is warm brick-like with weak grey/cement support", meta
    return False, None, meta


def _select_t5_style_candidate(
    scored: list[dict[str, Any]],
    *,
    requested_material: str,
    config: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    target_cfg = config.get("target_selection", {})
    requested = _normalize_material_name(requested_material)
    source_bonus = float(target_cfg.get("source_class_prior_bonus", 0.15))
    same_source_preferred = bool(target_cfg.get("same_source_preferred", True))
    override_margin = float(target_cfg.get("cross_source_override_margin", 0.20))
    min_margin = float(target_cfg.get("min_margin", 0.02))
    brick_timber_guard_enabled = bool(target_cfg.get("brick_timber_guard_enabled", True))
    brick_timber_guard_min_timber = float(target_cfg.get("brick_timber_guard_min_timber_score", 0.10))
    brick_timber_guard_min_alt_masonry = float(target_cfg.get("brick_timber_guard_min_alt_masonry_score", 0.30))
    brick_timber_guard_margin = float(target_cfg.get("brick_timber_guard_masonry_margin", 0.05))
    concrete_guard_enabled = bool(target_cfg.get("concrete_requires_grey_texture_guard", True))
    concrete_min_grey = float(target_cfg.get("concrete_min_grey_score", 0.42))
    concrete_max_warm = float(target_cfg.get("concrete_max_warm_beige_score", 0.38))

    for item in scored:
        scores = item.get("clip_scores", {}) or {}
        requested_score = _safe_float(scores.get(requested))
        other_scores = [_safe_float(value) for key, value in scores.items() if _normalize_material_name(key) != requested]
        margin = requested_score - (max(other_scores) if other_scores else 0.0)
        source_match = _normalize_material_name(item.get("source_detection_query")) == requested
        item["requested_score"] = round(requested_score, 6)
        item["clip_score_for_requested"] = round(requested_score, 6)
        item["requested_material_score"] = round(requested_score, 6)
        item["score_margin"] = round(margin, 6)
        item["source_class_prior_bonus_applied"] = source_bonus if source_match else 0.0
        item["final_score"] = round(requested_score + (source_bonus if source_match else 0.0), 6)

    if not scored:
        return None, {
            "selection_rule": SELECTION_RULE,
            "selected_candidate_id_before_prior": "NO_VERIFIED_MATCH",
            "selected_candidate_id_after_prior": "NO_VERIFIED_MATCH",
            "selected_candidate_id_after_guards": "NO_VERIFIED_MATCH",
            "final_selected_candidate_id": "NO_VERIFIED_MATCH",
            "source_class_prior_bonus": source_bonus,
            "cross_source_override_margin": override_margin,
            "same_source_preferred": same_source_preferred,
            "source_class_prior_used": False,
            "cross_source_override_used": False,
            "beige_brick_guard_triggered": False,
            "concrete_color_texture_guard_triggered": False,
            "timber_guard_triggered": False,
            "final_selection_reason": "no_candidates",
            "no_verified_match": True,
        }

    by_requested = sorted(scored, key=lambda item: (_safe_float(item.get("requested_score")), _safe_float(item.get("score_margin"))), reverse=True)
    selected_before_prior = by_requested[0]
    same_source = [item for item in scored if _normalize_material_name(item.get("source_detection_query")) == requested]
    same_source.sort(key=lambda item: (_safe_float(item.get("requested_score")), _safe_float(item.get("score_margin"))), reverse=True)
    same_best = same_source[0] if same_source else None
    by_final = sorted(scored, key=lambda item: (_safe_float(item.get("final_score")), _safe_float(item.get("score_margin"))), reverse=True)
    top = by_final[0]
    source_prior_used = bool(same_source_preferred and same_best is not None)
    cross_source_override_used = False
    if source_prior_used and same_best is not None:
        other_best = next((item for item in by_final if item.get("candidate_id") != same_best.get("candidate_id")), None)
        if other_best and _normalize_material_name(other_best.get("source_detection_query")) != requested:
            if _safe_float(other_best.get("requested_score")) >= _safe_float(same_best.get("requested_score")) + override_margin:
                top = other_best
                cross_source_override_used = True
            else:
                top = same_best
        else:
            top = same_best

    selected_after_prior = top
    beige_guard_triggered = False
    beige_guard_reason = "NA"
    concrete_guard_triggered = False
    concrete_guard_reason = "NA"
    timber_guard_triggered = False
    timber_guard_reason = "NA"

    if requested == "concrete block" and concrete_guard_enabled and top:
        top_grey = _safe_float(top.get("grey_score"))
        top_warm = _safe_float(top.get("warm_beige_score"))
        grey_candidates = [
            item
            for item in scored
            if _safe_float(item.get("grey_score")) >= concrete_min_grey
            and _safe_float(item.get("warm_beige_score")) <= concrete_max_warm
        ]
        if top_warm > concrete_max_warm and top_grey < concrete_min_grey and grey_candidates:
            replacement = max(
                grey_candidates,
                key=lambda item: (
                    _safe_float(item.get("grey_score")) + _safe_float((item.get("clip_scores") or {}).get("concrete block")),
                    _safe_float(item.get("final_score")),
                ),
            )
            if replacement.get("candidate_id") != top.get("candidate_id"):
                concrete_guard_triggered = True
                concrete_guard_reason = "selected grey/cement-colored candidate over warmer beige/tan concrete false positive"
                top = replacement

    if requested == "brick" and brick_timber_guard_enabled and top:
        top_scores = top.get("clip_scores", {}) or {}
        top_timber = _safe_float(top_scores.get("timber"))
        top_wood_like = _safe_float(top.get("wood_like_score"))
        top_masonry = max(_safe_float(top_scores.get("brick")), _safe_float(top_scores.get("concrete block")))
        masonry_candidates = []
        for item in scored:
            if item.get("candidate_id") == top.get("candidate_id"):
                continue
            scores = item.get("clip_scores", {}) or {}
            masonry_score = max(_safe_float(scores.get("brick")), _safe_float(scores.get("concrete block")))
            timber_score = _safe_float(scores.get("timber"))
            if (
                masonry_score >= brick_timber_guard_min_alt_masonry
                and masonry_score >= top_masonry + brick_timber_guard_margin
                and timber_score < max(top_timber, brick_timber_guard_min_timber)
            ):
                masonry_candidates.append((masonry_score, -timber_score, _safe_float(item.get("mask_score")), item))
        if (top_timber >= brick_timber_guard_min_timber or top_wood_like >= 0.45) and masonry_candidates:
            masonry_candidates.sort(reverse=True, key=lambda row: row[:3])
            replacement = masonry_candidates[0][3]
            timber_guard_triggered = True
            timber_guard_reason = (
                "requested brick: top requested-score crop had timber/wood evidence; "
                "selected stronger masonry-like candidate with lower timber evidence"
            )
            top = replacement

    selected_after_guards = top
    threshold_no_match = _safe_float(top.get("requested_score")) < threshold
    gate_reject, gate_reason, gate_meta = _no_match_gate(
        selected=top,
        scored=scored,
        requested=requested,
        target_cfg=target_cfg,
        threshold=threshold,
    )
    no_verified_match = bool(threshold_no_match or gate_reject)
    selected = None if no_verified_match else top
    final_id = selected.get("candidate_id") if selected else "NO_VERIFIED_MATCH"
    if threshold_no_match:
        final_reason = f"top requested-material score below threshold {threshold:.4f}"
    elif gate_reject:
        final_reason = gate_reason or "requested material not confidently present"
    elif timber_guard_triggered:
        final_reason = timber_guard_reason
    elif concrete_guard_triggered:
        final_reason = concrete_guard_reason
    elif source_prior_used:
        final_reason = "selected by T5_v2-style requested-class source prior and CLIP crop verification"
    else:
        final_reason = "selected by T5_v2-style CLIP crop verification"

    meta = {
        "selection_rule": SELECTION_RULE,
        "selected_candidate_id_before_prior": selected_before_prior.get("candidate_id", "NA"),
        "selected_candidate_id_after_prior": selected_after_prior.get("candidate_id", "NA") if selected_after_prior else "NA",
        "selected_candidate_id_after_guards": selected_after_guards.get("candidate_id", "NA") if selected_after_guards else "NA",
        "selected_candidate_id_before_no_match_gate": selected_after_guards.get("candidate_id", "NA") if selected_after_guards else "NA",
        "selected_candidate_id_after_no_match_gate": final_id,
        "final_selected_candidate_id": final_id,
        "selected_before_source_prior": selected_before_prior.get("candidate_id", "NA"),
        "selected_after_source_prior": selected_after_prior.get("candidate_id", "NA") if selected_after_prior else "NA",
        "source_class_prior_bonus": source_bonus,
        "cross_source_override_margin": override_margin,
        "same_source_preferred": same_source_preferred,
        "same_source_candidate_available": bool(same_best),
        "same_source_candidate_score": _safe_float(same_best.get("requested_score")) if same_best else "NA",
        "source_class_prior_used": source_prior_used,
        "cross_source_override_used": cross_source_override_used,
        "beige_brick_guard_triggered": beige_guard_triggered,
        "beige_brick_guard_reason": beige_guard_reason,
        "concrete_color_texture_guard_triggered": concrete_guard_triggered,
        "concrete_color_texture_guard_reason": concrete_guard_reason,
        "timber_guard_triggered": timber_guard_triggered,
        "timber_guard_reason": timber_guard_reason,
        "final_selection_reason": final_reason,
        "no_verified_match": no_verified_match,
        "target_absent_rejection_triggered": bool(gate_reject),
        "target_absent_rejection_reason": gate_reason,
        **gate_meta,
    }
    return selected, meta


def resolve_clip_python(config: dict[str, Any]) -> dict[str, Any]:
    configured = str(config.get("perception", {}).get("clip_subprocess_python", "auto") or "auto")
    resolved = sys.executable if configured.lower() == "auto" else configured
    return {
        "parent_sys_executable": sys.executable,
        "clip_subprocess_python": resolved,
        "same_as_parent_python": Path(resolved).resolve() == Path(sys.executable).resolve() if Path(resolved).exists() else resolved == sys.executable,
    }


def _run_clip_subprocess(
    *,
    python_executable: str,
    image_path: str,
    labels: list[str],
    model_id: str,
) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        "from transformers import pipeline\n"
        "model_id=sys.argv[1]\n"
        "image_path=sys.argv[2]\n"
        "labels=json.loads(sys.argv[3])\n"
        "classifier=pipeline(task='zero-shot-image-classification', model=model_id)\n"
        "results=classifier(image_path, candidate_labels=labels)\n"
        "print(json.dumps({'results': results, 'python_executable': sys.executable}))\n"
    )
    completed = subprocess.run(
        [python_executable, "-c", code, model_id, image_path, json.dumps(labels)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "CLIP subprocess failed")
    return json.loads(completed.stdout.strip())


def _crop_for_candidate(candidate: dict[str, Any]) -> tuple[str | None, str | None]:
    configured_path = candidate.get("crop_used_for_verification") or candidate.get("crop_used_for_verification_path")
    configured_type = candidate.get("crop_used_for_verification_type")
    if configured_path and configured_path != "NA" and Path(str(configured_path)).exists():
        return str(configured_path), str(configured_type or "unknown")
    for crop_type, key in (
        ("inner_texture_crop", "inner_texture_crop_path"),
        ("masked_texture_crop", "masked_texture_crop_path"),
        ("bbox_crop", "bbox_crop_path"),
    ):
        value = candidate.get(key)
        if value and Path(str(value)).exists():
            return str(value), crop_type
    return None, None


def _mock_scores_for_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, float] | None:
    mock_scores = config.get("testing", {}).get("mock_clip_scores")
    if mock_scores is None:
        mock_scores = config.get("target_selection", {}).get("mock_clip_scores")
    if not isinstance(mock_scores, dict):
        return None
    candidate_id = str(candidate.get("candidate_id"))
    values = mock_scores.get(candidate_id)
    if not isinstance(values, dict):
        return None
    out = {material: float(values.get(material, 0.0)) for material in MATERIAL_LABEL_PROMPTS}
    for key, value in values.items():
        if key not in out:
            out[str(key)] = float(value)
    return out


def verify_candidate_crops_with_clip(
    *,
    candidates: list[dict[str, Any]],
    requested_material: str,
    config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    rows = _prompt_rows()
    labels = [row["prompt"] for row in rows]
    aggregation = str(config.get("target_selection", {}).get("class_score_aggregation", "max")).lower()
    model_id = str(config.get("perception", {}).get("clip_model_id", "openai/clip-vit-base-patch32"))
    python_info = resolve_clip_python(config)
    threshold = float(config.get("target_selection", {}).get("clip_score_threshold", 0.0))
    use_clip = bool(config.get("target_selection", {}).get("use_clip_crop_verifier", True))
    scored: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in candidates:
        crop, crop_type = _crop_for_candidate(candidate)
        if crop is None:
            scored.append(
                {
                    **candidate,
                    "crop_used_for_verification": None,
                    "crop_type_used_for_scoring": None,
                    "clip_scores": {},
                    "clip_error": "missing_crop",
                }
            )
            continue
        try:
            mock_scores = _mock_scores_for_candidate(candidate, config)
            if mock_scores is not None:
                aggregate = mock_scores
                per_prompt = _mock_per_prompt_scores(aggregate)
                score_source = "mock_clip_crop_verifier"
                error = None
            elif use_clip:
                payload = _run_clip_subprocess(
                    python_executable=python_info["clip_subprocess_python"],
                    image_path=crop,
                    labels=labels,
                    model_id=model_id,
                )
                result_items = payload.get("results", [])
                out_labels = [str(item["label"]) for item in result_items]
                out_scores = [float(item["score"]) for item in result_items]
                aggregate, per_prompt = _aggregate(rows, out_labels, out_scores, aggregation)
                score_source = "clip_crop_verifier"
                error = None
            else:
                aggregate = {requested_material: float(candidate.get("mask_score") or candidate.get("detection_score") or 1.0)}
                for material in MATERIAL_LABEL_PROMPTS:
                    aggregate.setdefault(material, 0.0)
                per_prompt = {material: [] for material in MATERIAL_LABEL_PROMPTS}
                score_source = "clip_disabled_detection_score_fallback"
                error = None
        except Exception as exc:
            aggregate = {material: 0.0 for material in MATERIAL_LABEL_PROMPTS}
            per_prompt = {material: [] for material in MATERIAL_LABEL_PROMPTS}
            score_source = "clip_crop_verifier_failed"
            error = str(exc)
            errors.append(str(exc))
        top_label = max(aggregate, key=lambda key: float(aggregate[key])) if aggregate else None
        features = analyze_crop_color_texture(crop)
        scored.append(
            {
                **candidate,
                "crop_used_for_verification": crop,
                "crop_path_used_for_scoring": crop,
                "crop_type_used_for_scoring": crop_type,
                "crop_type_used_for_verification": crop_type,
                "clip_scores": aggregate,
                "clip_aggregate_scores": aggregate,
                "clip_per_prompt_scores": per_prompt,
                "clip_labels": list(MATERIAL_LABEL_PROMPTS.keys()),
                "clip_top_label": top_label,
                "clip_top_score": float(aggregate.get(top_label, 0.0)) if top_label else 0.0,
                "clip_score_for_requested": float(aggregate.get(requested_material, 0.0)),
                "requested_material_score": float(aggregate.get(requested_material, 0.0)),
                "selection_rule": SELECTION_RULE,
                "material_scores": aggregate,
                "color_texture_features": features,
                "grey_score": features.get("grey_score", 0.0),
                "warm_beige_score": features.get("warm_beige_score", 0.0),
                "cement_texture_score": features.get("cement_texture_score", 0.0),
                "wood_like_score": features.get("wood_like_score", 0.0),
                "concrete_color_gate_score": round(float(features.get("grey_score", 0.0)) - float(features.get("warm_beige_score", 0.0)), 4),
                "crop_source_used": crop_type,
                "clip_score_source": score_source,
                "score_source": score_source,
                "clip_error": error,
            }
        )
    selected, selection_meta = _select_t5_style_candidate(
        scored,
        requested_material=requested_material,
        config=config,
        threshold=threshold,
    )
    for item in scored:
        item["final_selected"] = bool(selected and item.get("candidate_id") == selected.get("candidate_id"))
        item["selection_rule"] = SELECTION_RULE
        item["final_selection_reason"] = selection_meta.get("final_selection_reason")
        item["selected_candidate_id_before_prior"] = selection_meta.get("selected_candidate_id_before_prior")
        item["selected_candidate_id_after_prior"] = selection_meta.get("selected_candidate_id_after_prior")
        item["selected_candidate_id_after_guards"] = selection_meta.get("selected_candidate_id_after_guards")
        item["selected_candidate_id_before_no_match_gate"] = selection_meta.get("selected_candidate_id_before_no_match_gate")
        item["selected_candidate_id_after_no_match_gate"] = selection_meta.get("selected_candidate_id_after_no_match_gate")
    candidate_scores = [
        {
            "candidate_id": item.get("candidate_id"),
            "crop_path_used_for_scoring": item.get("crop_path_used_for_scoring") or item.get("crop_used_for_verification"),
            "crop_type_used_for_scoring": item.get("crop_type_used_for_scoring") or item.get("crop_type_used_for_verification"),
            "clip_labels": item.get("clip_labels", list(MATERIAL_LABEL_PROMPTS.keys())),
            "clip_scores": item.get("clip_scores", {}),
            "clip_aggregate_scores": item.get("clip_aggregate_scores", item.get("clip_scores", {})),
            "clip_per_prompt_scores": item.get("clip_per_prompt_scores", {}),
            "clip_top_label": item.get("clip_top_label"),
            "clip_top_score": item.get("clip_top_score"),
            "requested_material_score": item.get("requested_material_score", item.get("clip_score_for_requested")),
            "requested_score": item.get("requested_score"),
            "score_margin": item.get("score_margin"),
            "source_class_prior_bonus_applied": item.get("source_class_prior_bonus_applied"),
            "final_score": item.get("final_score"),
            "selection_rule": SELECTION_RULE,
            "color_texture_features": item.get("color_texture_features", {}),
            "grey_score": item.get("grey_score"),
            "warm_beige_score": item.get("warm_beige_score"),
            "cement_texture_score": item.get("cement_texture_score"),
            "wood_like_score": item.get("wood_like_score"),
            "final_selected": item.get("final_selected", False),
            "match_assessment": selection_meta.get("candidate_match_assessments", {}).get(str(item.get("candidate_id")), {}),
        }
        for item in scored
    ]
    payload = {
        "requested_material": requested_material,
        "decision_backend": "clip_crop_verifier",
        "official_selector": "clip_crop_verifier",
        "qwen_vlm_server_enabled": False,
        "require_vlm_decision": False,
        "clip_verifier_enabled": use_clip,
        "clip_model_id": model_id,
        **python_info,
        "labels": labels,
        "threshold": threshold,
        "selection_rule": SELECTION_RULE,
        **selection_meta,
        "candidate_scores": candidate_scores,
        "candidates": scored,
        "selected_candidate_id": selected.get("candidate_id") if selected else "NO_VERIFIED_MATCH",
        "errors": errors,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
