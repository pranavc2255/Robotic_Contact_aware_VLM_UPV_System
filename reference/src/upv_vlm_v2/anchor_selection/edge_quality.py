"""Deterministic edge/contact quality features for v2 anchor selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _extract_patch(rgb: np.ndarray | None, point_xy: list[float] | tuple[float, float] | None, radius: int) -> np.ndarray | None:
    if rgb is None or point_xy is None:
        return None
    x, y = int(round(point_xy[0])), int(round(point_xy[1]))
    x0, x1 = max(0, x - radius), min(rgb.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(rgb.shape[0], y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return rgb[y0:y1, x0:x1].copy()


def score_edge_quality(
    *,
    rgb_array: np.ndarray | None,
    contact_point_a_px: list[float] | tuple[float, float] | None,
    contact_point_b_px: list[float] | tuple[float, float] | None,
    local_path_length_px: float,
    object_extent_px: float,
    center_distance_ratio: float,
    patch_radius_px: int = 18,
) -> dict[str, Any]:
    """Score one contact pair using ported deterministic contact principles.

    The old V1m/V2a selector scores local edge quality and rejects highly
    inconsistent contact patches. Phase 2 keeps the deterministic geometry
    portion local to v2: valid paired boundary hits, usable cross-section span,
    local texture consistency, and centrality along the selected object axis.
    """
    valid_pair = contact_point_a_px is not None and contact_point_b_px is not None and local_path_length_px > 0
    if not valid_pair:
        return {
            "valid": False,
            "deterministic_score": 0.0,
            "veto": True,
            "veto_reasons": ["missing paired boundary hits"],
            "side_quality_score": 0.0,
            "paired_consistency_score": 0.0,
        }

    span_score = _clamp((local_path_length_px / max(object_extent_px, 1.0)) * 100.0)
    centrality_score = _clamp((1.0 - abs(center_distance_ratio)) * 100.0)
    texture_scores: list[float] = []
    if rgb_array is not None:
        for point in (contact_point_a_px, contact_point_b_px):
            patch = _extract_patch(rgb_array, point, patch_radius_px)
            if patch is None or patch.size == 0:
                texture_scores.append(50.0)
                continue
            gray = patch.astype("float32").mean(axis=2)
            roughness = float(np.std(gray))
            texture_scores.append(_clamp(100.0 - roughness * 1.1))
    side_quality = float(np.mean(texture_scores)) if texture_scores else 70.0
    paired_consistency = 100.0 - abs(texture_scores[0] - texture_scores[1]) if len(texture_scores) == 2 else 80.0
    score = _clamp(0.42 * span_score + 0.28 * centrality_score + 0.20 * side_quality + 0.10 * paired_consistency)
    veto_reasons: list[str] = []
    if span_score < 20.0:
        veto_reasons.append("local cross-section too short")
    if side_quality < 25.0:
        veto_reasons.append("low local contact patch quality")
    if paired_consistency < 25.0:
        veto_reasons.append("contact sides poorly paired")
    return {
        "valid": not veto_reasons,
        "deterministic_score": round(score, 3),
        "veto": bool(veto_reasons),
        "veto_reasons": veto_reasons,
        "span_score": round(span_score, 3),
        "centrality_score": round(centrality_score, 3),
        "side_quality_score": round(side_quality, 3),
        "paired_consistency_score": round(paired_consistency, 3),
    }
