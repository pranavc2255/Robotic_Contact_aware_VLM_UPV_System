"""v2 clamp planning utilities."""

from __future__ import annotations

from typing import Any


def plan_clamp_opening(local_path_length_mm: float | None, config: dict[str, Any]) -> dict[str, Any]:
    if local_path_length_mm is None:
        return {"success": False, "failure_reason": "local_path_length_missing"}
    clamp = config.get("clamp", {})
    safety_margin = float(clamp.get("safety_margin_mm", 10.0))
    contact_allowance = float(clamp.get("contact_allowance_mm", 2.0))
    opening = float(local_path_length_mm) + safety_margin + contact_allowance
    fully_open = clamp.get("fully_open_probe_spacing_mm")
    fully_closed = clamp.get("fully_closed_probe_spacing_mm")
    warnings: list[str] = []
    valid_range = True
    if fully_open is not None and opening > float(fully_open):
        valid_range = False
        warnings.append("recommended clamp opening exceeds fully open probe spacing")
    if fully_closed is not None and opening < float(fully_closed):
        valid_range = False
        warnings.append("recommended clamp opening is below fully closed probe spacing")
    return {
        "success": valid_range,
        "failure_reason": None if valid_range else "clamp_opening_out_of_range",
        "local_path_length_mm": float(local_path_length_mm),
        "safety_margin_mm": safety_margin,
        "contact_allowance_mm": contact_allowance,
        "recommended_clamp_opening_mm": opening,
        "fully_open_probe_spacing_mm": float(fully_open) if fully_open is not None else None,
        "fully_closed_probe_spacing_mm": float(fully_closed) if fully_closed is not None else None,
        "warnings": warnings,
        "note": "Clamp opening is safety spacing, not UPV velocity path length.",
    }
