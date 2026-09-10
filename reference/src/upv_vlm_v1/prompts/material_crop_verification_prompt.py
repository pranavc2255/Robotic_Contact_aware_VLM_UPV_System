from __future__ import annotations

from typing import Any


DEFAULT_MATERIAL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "brick": {
        "description": (
            "Brick may be red, maroon, orange, tan, beige, or cream fired clay masonry. "
            "Beige/tan brick is still brick. Fired-clay brick texture may be smoother, "
            "ceramic/clay-like, sandy, warm-toned, rough, or granular, and the object is "
            "usually a rectangular masonry unit."
        ),
        "avoid": (
            "If material is warm beige/tan/orange/red rather than grey cementitious, prefer "
            "brick over concrete block. Do not confuse brick with grey/light-grey concrete "
            "block or timber."
        ),
    },
    "timber": {
        "description": (
            "Timber is wood or lumber. It usually has brown/tan wood texture, grain, "
            "fibers, or cut wood surfaces."
        ),
        "avoid": (
            "Red/maroon/beige fired-clay brick is NOT timber. Grey cement/concrete/CMU "
            "is NOT timber."
        ),
    },
    "concrete block": {
        "description": (
            "Concrete block is usually grey, light grey, off-white grey, or cement-colored. "
            "It is a cementitious concrete/CMU/cinder-block-like masonry material. The surface "
            "often looks powdery, porous, pitted, aggregate-like, cement-grain-like, rough cast, "
            "chipped cement, or grey granular. It may have holes or pores. It is not simply any "
            "light-colored rectangular object."
        ),
        "avoid": (
            "Do NOT select beige, tan, cream, orange, red, or maroon fired-clay brick as concrete "
            "block. A beige/tan brick can look pale or chalky, but if it has fired-clay brick color, "
            "smoother fired-clay texture, brick-like elongated shape, or warm tan/orange undertone, "
            "classify it as brick, not concrete block. Do not confuse beige brick with concrete "
            "merely because it is light colored. Concrete block should be selected only when both "
            "color and texture support grey/cementitious/CMU-like material."
        ),
    },
}


EXPECTED_SCHEMA: dict[str, Any] = {
    "selected_candidate_id": "candidate_001 or candidate_002 or NO_VERIFIED_MATCH",
    "selected_panel_label": "C1 or C2 or NO_VERIFIED_MATCH",
    "predicted_material": "brick|timber|concrete block|unknown",
    "confidence": "low|medium|high",
    "reason": "one short sentence",
    "candidate_assessments": {
        "candidate_001": {
            "predicted_material": "brick|timber|concrete block|unknown",
            "color_evidence": "short phrase",
            "texture_evidence": "short phrase",
            "matches_requested": True,
        }
    },
}


def build_material_crop_verification_prompt(
    requested_material: str,
    candidate_ids: list[str],
    material_descriptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    descriptions = material_descriptions or DEFAULT_MATERIAL_DESCRIPTIONS
    requested = str(requested_material or "").strip().lower()
    candidates = ", ".join(candidate_ids) if candidate_ids else "NA"
    description_lines = []
    for material in ["brick", "timber", "concrete block"]:
        entry = descriptions.get(material, {})
        if isinstance(entry, dict):
            text = " ".join(part for part in [entry.get("description"), entry.get("avoid")] if part)
        else:
            text = str(entry)
        description_lines.append(f"- {material}: {text}")
    system_prompt = (
        "You compare material candidate crops for a UPV_VLM target-selection experiment. "
        "Return only valid JSON. Use the requested construction material and the candidate crop images."
    )
    user_prompt = (
        f"Requested material: {requested}\n"
        f"Candidate IDs: {candidates}\n\n"
        "Material definitions:\n"
        + "\n".join(description_lines)
        + "\n\nSelect the candidate whose masked texture crop best matches the requested material. "
        "Compare all candidate crops against each other. Choose exactly one candidate or NO_VERIFIED_MATCH. "
        "Do not return NA. "
        "Focus on both color family and surface texture. For concrete block, both should support "
        "grey/light-grey/cementitious concrete; beige, tan, cream, orange, red, maroon fired-clay brick "
        "is NOT concrete block; wood/timber/brown grain/fibrous texture is NOT concrete block. "
        "If none match, return NO_VERIFIED_MATCH."
    )
    return {
        "stage_name": "material_crop_verification",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "expected_schema": EXPECTED_SCHEMA,
    }
