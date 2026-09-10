from __future__ import annotations

from typing import Any

import numpy as np


SEMANTIC_LABELS = [
    "clean_brick_or_target_material",
    "mortar_or_attached_cement",
    "nail_or_metal",
    "wood_or_foreign_material",
    "debris_or_glue_or_contamination",
    "uncertain",
]


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


def _slice_band(patch_rgb: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = max(0, min(patch_rgb.shape[0], y0))
    y1 = max(0, min(patch_rgb.shape[0], y1))
    if y1 <= y0:
        return patch_rgb[0:0, :, :]
    return patch_rgb[y0:y1, :, :]


def _color_cues(rgb_band: np.ndarray) -> dict[str, float]:
    if rgb_band.size == 0:
        return {
            "gray_white_ratio": 0.0,
            "dark_low_saturation_ratio": 0.0,
            "brick_like_ratio": 0.0,
            "wood_like_ratio": 0.0,
            "mean_brightness": 0.0,
        }

    rgb = rgb_band.astype(np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    brightness = rgb.mean(axis=2)
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)

    gray_white = (brightness >= 135.0) & (channel_range <= 62.0)
    dark_low_sat = (brightness <= 72.0) & (channel_range <= 75.0)
    brick_like = (red >= 95.0) & (green >= 45.0) & (green <= 175.0) & (blue <= 155.0) & (red >= green * 1.04)
    wood_like = (red >= 85.0) & (green >= 55.0) & (blue <= 105.0) & (np.abs(red - green) <= 70.0) & (channel_range >= 25.0)

    return {
        "gray_white_ratio": round(float(gray_white.mean()), 4),
        "dark_low_saturation_ratio": round(float(dark_low_sat.mean()), 4),
        "brick_like_ratio": round(float(brick_like.mean()), 4),
        "wood_like_ratio": round(float(wood_like.mean()), 4),
        "mean_brightness": round(float(brightness.mean()), 2),
    }


def _side_bands(patch_rgb: np.ndarray, edge_y_px: float, side_name: str) -> tuple[np.ndarray, np.ndarray]:
    edge_y = int(round(edge_y_px))
    if side_name == "top":
        contact_band = _slice_band(patch_rgb, edge_y - 8, edge_y + 28)
        inward_band = _slice_band(patch_rgb, edge_y + 28, edge_y + 96)
    elif side_name == "bottom":
        contact_band = _slice_band(patch_rgb, edge_y - 28, edge_y + 8)
        inward_band = _slice_band(patch_rgb, edge_y - 96, edge_y - 28)
    else:
        raise ValueError(f"Unsupported side_name: {side_name}")
    return contact_band, inward_band


def verify_contact_side_semantics(
    patch_rgb: np.ndarray | None,
    edge_y_px: float | None,
    side_name: str,
) -> dict[str, Any]:
    flags = {label: False for label in SEMANTIC_LABELS}
    if patch_rgb is None or edge_y_px is None:
        flags["uncertain"] = True
        return {
            "side": side_name,
            "valid": False,
            "semantic_flags": flags,
            "semantic_risk_score": 100.0,
            "semantic_confidence": 0.2,
            "semantic_veto_recommended": True,
            "semantic_veto_reasons": ["missing patch or contact edge"],
            "cue_metrics": {},
        }

    edge_y = int(round(edge_y_px))
    if edge_y < 0 or edge_y >= patch_rgb.shape[0]:
        flags["uncertain"] = True
        return {
            "side": side_name,
            "valid": False,
            "semantic_flags": flags,
            "semantic_risk_score": 100.0,
            "semantic_confidence": 0.25,
            "semantic_veto_recommended": True,
            "semantic_veto_reasons": ["contact edge outside patch"],
            "cue_metrics": {"edge_y_px": edge_y_px},
        }

    contact_band, inward_band = _side_bands(patch_rgb, float(edge_y_px), side_name)
    contact = _color_cues(contact_band)
    inward = _color_cues(inward_band)

    gray_contact = contact["gray_white_ratio"]
    dark_contact = contact["dark_low_saturation_ratio"]
    brick_inward = inward["brick_like_ratio"]
    brick_contact = contact["brick_like_ratio"]
    wood_contact = contact["wood_like_ratio"]
    brightness_jump = abs(contact["mean_brightness"] - inward["mean_brightness"]) / 255.0

    flags["mortar_or_attached_cement"] = gray_contact >= 0.24
    flags["nail_or_metal"] = dark_contact >= 0.16 and gray_contact < 0.35
    flags["wood_or_foreign_material"] = wood_contact >= 0.32 and brick_inward < 0.28
    flags["debris_or_glue_or_contamination"] = brightness_jump >= 0.22 or (gray_contact >= 0.18 and brick_contact < 0.32)
    flags["clean_brick_or_target_material"] = brick_inward >= 0.34 and not (
        flags["mortar_or_attached_cement"] or flags["nail_or_metal"] or flags["wood_or_foreign_material"]
    )
    flags["uncertain"] = not any(
        [
            flags["clean_brick_or_target_material"],
            flags["mortar_or_attached_cement"],
            flags["nail_or_metal"],
            flags["wood_or_foreign_material"],
            flags["debris_or_glue_or_contamination"],
        ]
    )

    risk = _clamp(
        gray_contact * 70.0
        + dark_contact * 42.0
        + max(0.0, 0.30 - brick_inward) * 65.0
        + brightness_jump * 34.0
        + (18.0 if flags["wood_or_foreign_material"] else 0.0)
    )

    reasons: list[str] = []
    if flags["mortar_or_attached_cement"]:
        reasons.append("gray/white cement-like material near contact band")
    if flags["nail_or_metal"]:
        reasons.append("dark low-saturation object-like region near contact band")
    if flags["wood_or_foreign_material"]:
        reasons.append("wood/foreign-material color cue near contact band")
    if flags["debris_or_glue_or_contamination"]:
        reasons.append("large contact-to-inward color discontinuity or contamination cue")
    if brick_inward < 0.18:
        reasons.append("weak clean brick/material evidence in inward context")

    semantic_veto = risk >= 58.0 or flags["mortar_or_attached_cement"] or flags["nail_or_metal"]
    confidence = _clamp(35.0 + max(gray_contact, dark_contact, brick_inward, brightness_jump) * 85.0, 0.0, 95.0) / 100.0

    return {
        "side": side_name,
        "valid": True,
        "semantic_flags": flags,
        "semantic_risk_score": round(risk, 2),
        "semantic_confidence": round(confidence, 3),
        "semantic_veto_recommended": bool(semantic_veto),
        "semantic_veto_reasons": reasons,
        "cue_metrics": {
            "edge_y_px": float(edge_y_px),
            "contact_band": contact,
            "inward_band": inward,
            "brightness_jump_ratio": round(float(brightness_jump), 4),
        },
    }


def run_semantic_heuristic_v1(
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

        top = verify_contact_side_semantics(top_patch, top_edge_y, "top")
        bottom = verify_contact_side_semantics(bottom_patch, bottom_edge_y, "bottom")
        risk_score = max(top["semantic_risk_score"], bottom["semantic_risk_score"])
        confidence = round((top["semantic_confidence"] + bottom["semantic_confidence"]) / 2.0, 3)
        veto_reasons = [f"top: {reason}" for reason in top["semantic_veto_reasons"]]
        veto_reasons.extend([f"bottom: {reason}" for reason in bottom["semantic_veto_reasons"]])
        semantic_veto = bool(top["semantic_veto_recommended"] or bottom["semantic_veto_recommended"])

        summary_parts = []
        if top["semantic_veto_recommended"]:
            summary_parts.append("top semantic hazard")
        if bottom["semantic_veto_recommended"]:
            summary_parts.append("bottom semantic hazard")
        if not summary_parts:
            summary_parts.append("no strong semantic hazard detected by heuristic")

        anchors.append(
            {
                "anchor_id": anchor_id,
                "top_semantic_flags": top["semantic_flags"],
                "bottom_semantic_flags": bottom["semantic_flags"],
                "top_semantic_risk_score": top["semantic_risk_score"],
                "bottom_semantic_risk_score": bottom["semantic_risk_score"],
                "semantic_summary": "; ".join(summary_parts),
                "semantic_confidence": confidence,
                "semantic_veto_recommended": semantic_veto,
                "semantic_veto_reasons": veto_reasons,
                "top_details": top,
                "bottom_details": bottom,
            }
        )

    return {
        "semantic_backend": "heuristic",
        "semantic_method": "semantic_heuristic_v1",
        "method_note": "Deterministic color/context cues only; not true semantic recognition.",
        "labels": SEMANTIC_LABELS,
        "anchors": anchors,
    }


def build_semantic_summary(semantic_results: dict[str, Any]) -> dict[str, Any]:
    warnings_by_anchor: dict[str, list[str]] = {}
    veto_recommended = []
    for anchor in semantic_results["anchors"]:
        warnings = list(anchor["semantic_veto_reasons"])
        if anchor["top_semantic_flags"].get("uncertain") or anchor["bottom_semantic_flags"].get("uncertain"):
            warnings.append("semantic heuristic uncertainty")
        warnings_by_anchor[anchor["anchor_id"]] = warnings
        if anchor["semantic_veto_recommended"]:
            veto_recommended.append(anchor["anchor_id"])
    return {
        "semantic_backend": semantic_results["semantic_backend"],
        "semantic_method": semantic_results["semantic_method"],
        "num_anchors": len(semantic_results["anchors"]),
        "semantic_veto_recommended_anchors": veto_recommended,
        "semantic_warnings_by_anchor": warnings_by_anchor,
    }


def build_semantic_advisory(
    deterministic_final_decision: dict[str, Any],
    semantic_results: dict[str, Any],
) -> dict[str, Any]:
    summary = build_semantic_summary(semantic_results)
    selected_anchor = deterministic_final_decision.get("selected_anchor")
    selected_has_warning = bool(selected_anchor and summary["semantic_warnings_by_anchor"].get(selected_anchor))
    return {
        "phase": "v1o_semantic_verification_layer",
        "advisory_only": True,
        "deterministic_selected_anchor": selected_anchor,
        "deterministic_final_decision": deterministic_final_decision.get("final_decision"),
        "semantic_backend": semantic_results["semantic_backend"],
        "semantic_method": semantic_results["semantic_method"],
        "semantic_warnings_by_anchor": summary["semantic_warnings_by_anchor"],
        "semantic_veto_recommended_anchors": summary["semantic_veto_recommended_anchors"],
        "semantic_agrees_with_deterministic_choice": not selected_has_warning,
        "agreement_reason": (
            "selected anchor has no semantic heuristic warnings"
            if not selected_has_warning
            else "selected anchor has semantic heuristic warnings; advisory disagrees but does not override"
        ),
    }
