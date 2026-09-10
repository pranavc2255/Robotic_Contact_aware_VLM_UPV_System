"""Deterministic hybrid anchor selection for v2."""

from __future__ import annotations

from typing import Any


def select_anchor_hybrid(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {
            "final_decision": "NO_SAFE_ANCHOR",
            "selected_anchor": None,
            "decision_reason": "no candidate anchors were generated",
            "survivor_list": [],
            "anchors": [],
        }
    enriched = []
    survivors = []
    for item in candidates:
        score = float(item.get("score", item.get("deterministic_score", 0.0)))
        veto = bool(item.get("veto", False))
        anchor_id = str(item.get("anchor_id"))
        enriched.append({**item, "score": score, "veto": veto})
        if not veto:
            survivors.append(anchor_id)
    if not survivors:
        return {
            "final_decision": "NO_SAFE_ANCHOR",
            "selected_anchor": None,
            "decision_reason": "all candidate anchors were vetoed by deterministic contact checks",
            "survivor_list": [],
            "anchors": enriched,
        }
    survivor_items = [item for item in enriched if str(item.get("anchor_id")) in survivors]
    survivor_items.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("anchor_id"))))
    selected = survivor_items[0]
    return {
        "final_decision": selected["anchor_id"],
        "selected_anchor": selected["anchor_id"],
        "decision_reason": "selected highest deterministic contact score among non-vetoed anchors",
        "survivor_list": survivors,
        "anchors": enriched,
        "selected_anchor_record": selected,
    }
