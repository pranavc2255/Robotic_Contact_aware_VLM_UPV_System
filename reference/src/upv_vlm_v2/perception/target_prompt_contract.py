"""Logged target-selection prompt contract for v2.

The v2 target selector remains CLIP-based. This module records the old
material-crop verification contract so run artifacts show the intended
candidate-or-NO_VERIFIED_MATCH behavior without calling Qwen for target
selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MATERIAL_DEFINITIONS = {
    "brick": (
        "Brick material verification should focus on warm red, reddish brown, orange brown, warm tan, "
        "warm beige, cream beige, yellowish tan, or warm off-white clay/brick appearance under bright "
        "lighting. A beige or cream brick may look whitish due to illumination, but it should still be "
        "treated as brick when the color is warm/yellow/tan rather than cool cement gray. Brick is not "
        "wood grain and not cool neutral gray cement block."
    ),
    "timber": (
        "Timber material verification should focus on visible wood grain, long parallel fiber lines, "
        "pale tan lumber, sawn wood texture, knots, and fibrous wood appearance. Low contrast or pale "
        "timber can still be timber if linear grain/fiber texture is visible. Timber is not clay masonry "
        "and not cementitious gray concrete."
    ),
    "concrete block": (
        "Concrete block material verification should focus on cool cement gray, neutral gray, ash gray, "
        "or pale gray cement-block appearance. A concrete block may look lighter due to illumination, but "
        "it should remain cool/neutral gray rather than warm beige, yellowish, cream, tan, red, or brown. "
        "Concrete block is not warm beige/yellow clay brick and not wood grain."
    ),
}


def build_material_crop_verification_prompt(*, requested_material: str, candidate_ids: list[str]) -> str:
    candidates = ", ".join(candidate_ids) if candidate_ids else "NO_CANDIDATES"
    definitions = "\n".join(f"- {name}: {text}" for name, text in MATERIAL_DEFINITIONS.items())
    schema = {
        "selected_candidate_id": "candidate_001_brick or NO_VERIFIED_MATCH",
        "selected_panel_label": "C1 or NO_VERIFIED_MATCH",
        "predicted_material": "brick|timber|concrete block|unknown",
        "confidence": "low|medium|high",
        "reason": "one short sentence",
        "candidate_assessments": {
            "candidate_001_brick": {
                "predicted_material": "brick|timber|concrete block|unknown",
                "matches_requested": True,
                "reason": "short reason",
            }
        },
    }
    return (
        "SYSTEM:\n"
        "You compare material candidate crops for a UPV_VLM target-selection experiment. "
        "Return only valid JSON.\n\n"
        "USER:\n"
        f"Requested material: {requested_material}\n"
        f"Candidate IDs: {candidates}\n\n"
        "For target crop verification, the crop may be an inner texture crop, not a full object view. "
        "Therefore material verification should focus on color family, surface texture, pores, grain, "
        "clay/cement/wood appearance, and not object shape. Do not use pores alone to distinguish brick "
        "from concrete block because both can be porous; prioritize warm clay color versus cool "
        "cement-gray color.\n\n"
        "Material definitions:\n"
        f"{definitions}\n\n"
        "Choose exactly one candidate or NO_VERIFIED_MATCH. If none match the requested material, "
        "return NO_VERIFIED_MATCH. Do not select the best available object when the requested class "
        "is absent. Do not return NA. Use candidate IDs only; do not invent coordinates.\n\n"
        "Return ONLY JSON with this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
    )


def save_target_prompt_contract(
    *,
    output_dir: str | Path,
    requested_material: str,
    candidate_ids: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    prompt_dir = Path(output_dir) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = build_material_crop_verification_prompt(
        requested_material=requested_material,
        candidate_ids=candidate_ids,
    )
    prompt_path = prompt_dir / "material_crop_verification_prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    manifest = {
        "stage_name": "material_crop_verification",
        "qwen_vlm_server_enabled": False,
        "require_vlm_decision": False,
        "prompt_logged_for_contract": True,
        "actual_selector": "clip_crop_verifier",
        "official_selector": "clip_crop_verifier",
        "no_match_allowed": True,
        "selected_candidate_contract": "candidate_id_or_NO_VERIFIED_MATCH",
        "requested_material": requested_material,
        "candidate_ids": candidate_ids,
        "selection_rule": (config or {}).get("target_selection", {}).get(
            "selection_rule",
            "t5_v2_source_prior_color_texture_guard_with_no_match_gate",
        ),
    }
    manifest_path = prompt_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "target_prompt_dir": str(prompt_dir),
        "material_crop_verification_prompt_path": str(prompt_path),
        "target_prompt_manifest_path": str(manifest_path),
    }
