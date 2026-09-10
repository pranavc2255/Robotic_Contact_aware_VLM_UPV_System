from __future__ import annotations

from typing import Any

import numpy as np


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _extract_patch(base_image_rgb: np.ndarray, bounds: tuple[int, int, int, int] | list[int] | None) -> np.ndarray | None:
    if bounds is None:
        return None
    x_min, y_min, x_max, y_max = [int(v) for v in bounds]
    if x_max <= x_min or y_max <= y_min:
        return None
    return base_image_rgb[y_min:y_max, x_min:x_max].copy()


def _local_edge_y(hit_xy_rotated: list[float] | tuple[float, float] | None, bounds: tuple[int, int, int, int] | list[int] | None) -> float | None:
    if hit_xy_rotated is None or bounds is None:
        return None
    return float(hit_xy_rotated[1]) - float(bounds[1])


def _band(gray: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = max(0, min(gray.shape[0], y0))
    y1 = max(0, min(gray.shape[0], y1))
    if y1 <= y0:
        return gray[0:0, :]
    return gray[y0:y1, :]


def _mortar_like_ratio(patch_rgb: np.ndarray, y0: int, y1: int) -> float:
    rgb = patch_rgb[max(0, y0) : min(patch_rgb.shape[0], y1), :, :].astype(np.float32)
    if rgb.size == 0:
        return 0.0
    brightness = rgb.mean(axis=2)
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)
    mortar_like = (brightness >= 142.0) & (channel_range <= 58.0)
    return float(mortar_like.mean())


def compute_contact_side_features(
    patch_rgb: np.ndarray | None,
    edge_y_px: float | None,
    side_name: str,
) -> dict[str, Any]:
    if patch_rgb is None or edge_y_px is None:
        return {
            "side": side_name,
            "valid": False,
            "edge_y_px": edge_y_px,
            "edge_straightness_score": 0.0,
            "edge_roughness_score": 0.0,
            "edge_roughness_px": None,
            "local_edge_consistency_score": 0.0,
            "contact_context_score": 0.0,
            "mortar_like_anomaly_score": 0.0,
            "side_quality_score": 0.0,
            "notes": ["missing patch or contact edge"],
        }

    height, width = patch_rgb.shape[:2]
    edge_y = int(round(edge_y_px))
    if height < 8 or width < 8 or edge_y < 0 or edge_y >= height:
        return {
            "side": side_name,
            "valid": False,
            "edge_y_px": float(edge_y_px),
            "edge_straightness_score": 0.0,
            "edge_roughness_score": 0.0,
            "edge_roughness_px": None,
            "local_edge_consistency_score": 0.0,
            "contact_context_score": 0.0,
            "mortar_like_anomaly_score": 0.0,
            "side_quality_score": 0.0,
            "notes": ["degenerate patch or contact edge outside patch"],
        }

    gray = patch_rgb.astype(np.float32).mean(axis=2)
    gradient = np.abs(np.diff(gray, axis=0))
    search_y0 = max(0, edge_y - 14)
    search_y1 = min(gradient.shape[0], edge_y + 15)
    if search_y1 <= search_y0:
        detected_y = np.full(width, float(edge_y), dtype=np.float32)
    else:
        search = gradient[search_y0:search_y1, :]
        detected_y = search.argmax(axis=0).astype(np.float32) + float(search_y0)

    median_y = float(np.median(detected_y))
    roughness_px = float(np.std(detected_y))
    mean_abs_dev_px = float(np.mean(np.abs(detected_y - median_y)))
    consistency = float(np.mean(np.abs(detected_y - median_y) <= 3.0))

    edge_straightness_score = _clamp(100.0 - roughness_px * 8.5 - mean_abs_dev_px * 3.0)
    edge_roughness_score = _clamp(100.0 - roughness_px * 9.0 - mean_abs_dev_px * 4.0)
    local_edge_consistency_score = _clamp(consistency * 100.0)

    if side_name == "top":
        outside_ratio = _mortar_like_ratio(patch_rgb, edge_y - 22, edge_y + 2)
        near_edge_ratio = _mortar_like_ratio(patch_rgb, edge_y - 6, edge_y + 10)
    else:
        outside_ratio = _mortar_like_ratio(patch_rgb, edge_y - 2, edge_y + 22)
        near_edge_ratio = _mortar_like_ratio(patch_rgb, edge_y - 10, edge_y + 6)

    mortar_like_anomaly_score = _clamp((0.7 * outside_ratio + 0.3 * near_edge_ratio) * 180.0)
    context_score = _clamp(100.0 - mortar_like_anomaly_score * 1.15)
    side_quality_score = _clamp(
        0.35 * edge_straightness_score
        + 0.25 * edge_roughness_score
        + 0.20 * local_edge_consistency_score
        + 0.20 * context_score
    )

    notes: list[str] = []
    if roughness_px >= 7.0:
        notes.append("high local edge roughness")
    if mortar_like_anomaly_score >= 45.0:
        notes.append("strong gray-white contact-zone anomaly")

    return {
        "side": side_name,
        "valid": True,
        "edge_y_px": float(edge_y_px),
        "edge_straightness_score": round(edge_straightness_score, 2),
        "edge_roughness_score": round(edge_roughness_score, 2),
        "edge_roughness_px": round(roughness_px, 3),
        "edge_mean_abs_deviation_px": round(mean_abs_dev_px, 3),
        "local_edge_consistency_score": round(local_edge_consistency_score, 2),
        "contact_context_score": round(context_score, 2),
        "mortar_like_anomaly_score": round(mortar_like_anomaly_score, 2),
        "mortar_like_outside_ratio": round(outside_ratio, 4),
        "mortar_like_near_edge_ratio": round(near_edge_ratio, 4),
        "side_quality_score": round(side_quality_score, 2),
        "notes": notes,
    }


def compute_anchor_contact_features(
    rotated_object_image_rgb: np.ndarray,
    source_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for source in source_candidates:
        anchor_id = source["source_anchor_id"]
        top_bounds = source.get("top_patch_bounds")
        bottom_bounds = source.get("bottom_patch_bounds")
        top_patch = _extract_patch(rotated_object_image_rgb, top_bounds)
        bottom_patch = _extract_patch(rotated_object_image_rgb, bottom_bounds)
        top_edge_y = _local_edge_y(source.get("top_hit_xy_rotated"), top_bounds)
        bottom_edge_y = _local_edge_y(source.get("bottom_hit_xy_rotated"), bottom_bounds)

        top = compute_contact_side_features(top_patch, top_edge_y, "top")
        bottom = compute_contact_side_features(bottom_patch, bottom_edge_y, "bottom")
        paired_consistency_score = _clamp(100.0 - abs(top["side_quality_score"] - bottom["side_quality_score"]))
        deterministic_score = _clamp(
            0.42 * top["side_quality_score"]
            + 0.42 * bottom["side_quality_score"]
            + 0.16 * paired_consistency_score
        )

        anchors.append(
            {
                "anchor_id": anchor_id,
                "top": top,
                "bottom": bottom,
                "paired_consistency_score": round(paired_consistency_score, 2),
                "deterministic_score": round(deterministic_score, 2),
                "valid": bool(top["valid"] and bottom["valid"]),
                "top_contact_guide_y_px": top_edge_y,
                "bottom_contact_guide_y_px": bottom_edge_y,
            }
        )

    return {
        "feature_version": "v1_simple_patch_edge_heuristics",
        "score_convention": "higher quality scores are better; mortar_like_anomaly_score is higher when gray-white attached material is more likely",
        "anchors": anchors,
    }


def hard_veto(anchor_features: dict[str, Any], veto_profile: str = "v1_simple") -> dict[str, Any]:
    if veto_profile != "v1_simple":
        raise ValueError(f"Unsupported hybrid veto profile: {veto_profile}")

    reasons: list[str] = []
    top = anchor_features["top"]
    bottom = anchor_features["bottom"]
    for side in (top, bottom):
        side_name = side["side"]
        if not side["valid"]:
            reasons.append(f"{side_name}: invalid contact patch")
        roughness_px = side.get("edge_roughness_px")
        if roughness_px is not None and roughness_px >= 6.5:
            reasons.append(f"{side_name}: severe edge roughness ({roughness_px:.2f}px)")
        if side["local_edge_consistency_score"] < 35.0:
            reasons.append(f"{side_name}: poor local edge consistency")
        if side["contact_context_score"] < 35.0:
            reasons.append(f"{side_name}: poor contact context")
        if side["mortar_like_anomaly_score"] >= 55.0:
            reasons.append(f"{side_name}: strong mortar-like gray/white anomaly")
        if side["side_quality_score"] < 30.0:
            reasons.append(f"{side_name}: very low side quality")

    if anchor_features["paired_consistency_score"] < 35.0:
        reasons.append("top/bottom contact quality is poorly paired")

    return {
        "veto": bool(reasons),
        "veto_reasons": reasons,
    }


def build_hybrid_decision_trace(
    deterministic_features: dict[str, Any],
    veto_profile: str = "v1_simple",
    vlm_rerank_used: bool = False,
) -> dict[str, Any]:
    anchors = []
    survivors = []
    for anchor in deterministic_features["anchors"]:
        veto = hard_veto(anchor, veto_profile=veto_profile)
        item = {
            **anchor,
            "veto": veto["veto"],
            "veto_reasons": veto["veto_reasons"],
        }
        anchors.append(item)
        if not veto["veto"]:
            survivors.append(anchor["anchor_id"])

    if not survivors:
        final_decision = "NO_SAFE_ANCHOR"
        selected_anchor = None
        decision_reason = "all candidate anchors were vetoed by deterministic contact checks"
    else:
        survivor_items = [item for item in anchors if item["anchor_id"] in survivors]
        survivor_items.sort(key=lambda item: (-float(item["deterministic_score"]), item["anchor_id"]))
        selected_anchor = survivor_items[0]["anchor_id"]
        final_decision = selected_anchor
        decision_reason = "selected highest deterministic contact score among non-vetoed anchors"

    return {
        "phase": "v1m_contact_guided_hybrid_selector",
        "veto_profile": veto_profile,
        "vlm_rerank_used": bool(vlm_rerank_used),
        "anchors": anchors,
        "survivor_list": survivors,
        "final_decision": final_decision,
        "selected_anchor": selected_anchor,
        "decision_reason": decision_reason,
        "vlm_rerank": None,
    }
