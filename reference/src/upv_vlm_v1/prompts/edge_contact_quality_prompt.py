from __future__ import annotations

from typing import Any


EXPECTED_SCHEMA: dict[str, Any] = {
    "selected_anchor_id": "... or NO_SAFE_ANCHOR",
    "contact_region_usable": True,
    "edge_straightness": "poor|acceptable|good",
    "imperfection_present": True,
    "obstruction_present": True,
    "confidence": "low|medium|high",
    "reason": "short reason",
}


def build_edge_contact_quality_prompt(
    requested_material: str,
    axis_mode: str,
    candidate_anchor_ids: list[str],
) -> dict[str, Any]:
    anchors = ", ".join(candidate_anchor_ids) if candidate_anchor_ids else "NA"
    system_prompt = (
        "You evaluate UPV transducer contact regions on construction material edges. "
        "Return only valid JSON."
    )
    user_prompt = (
        f"Requested material: {str(requested_material or '').strip().lower()}\n"
        f"UPV axis mode: {axis_mode}\n"
        f"Candidate anchor IDs: {anchors}\n\n"
        "Select the safest usable anchor/contact region. Penalize cracks, chips, mortar residue, "
        "dirt, holes, nails, debris, jagged edges, obstructions, and visibly non-straight edges."
    )
    return {
        "stage_name": "edge_contact_quality",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "expected_schema": EXPECTED_SCHEMA,
    }

