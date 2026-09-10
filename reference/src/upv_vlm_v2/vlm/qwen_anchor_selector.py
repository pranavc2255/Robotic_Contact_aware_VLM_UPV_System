"""Required Qwen/VLM final selector for V2a wide-contact anchors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from upv_vlm_v2.vlm.qwen_client import infer_qwen_anchor, load_mock_qwen_response
from upv_vlm_v2.vlm.structured_output import extract_json_object


QWEN_ANCHOR_PROMPT_VERSION = "qwen_anchor_selection_upv_vlm_v2"
QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION = "qwen_anchor_selection_upv_vlm_v2_legacy_v4_replicated"


def _legacy_v4_schema(candidate_ids: list[str]) -> dict[str, Any]:
    """Placeholder-only legacy V4 schema.

    Do not populate this schema with real candidate IDs such as A1/A2/A3.
    Qwen was observed to copy candidate order from schema examples, assigning
    A1=90, A2=85, A3=80, etc. Placeholders avoid that first-candidate bias.
    """
    return {
        "anchors": [
            {
                "id": "<candidate_id>",
                "top_blocked": 0,
                "bottom_blocked": 0,
                "has_mortar_or_attached_material": 0,
                "has_nail_debris_or_foreign_object": 0,
                "has_broken_or_jagged_edge": 0,
                "overall_score": 0,
                "reason": "short visual reason",
            }
        ],
        "ranking_best_to_worst": ["<best_candidate_id>", "<next_candidate_id>"],
        "best_anchor": "<best_candidate_id>",
        "best_anchor_reason": "short visual reason",
    }


def _build_legacy_v4_replicated_prompt(*, candidate_ids: list[str]) -> str:
    schema = _legacy_v4_schema(candidate_ids)
    return (
        f"Prompt mode: {QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION}\n\n"
        "You are evaluating candidate anchor locations for planar probe contact on a brick or masonry-like rectangular object.\n\n"
        "The image contains multiple candidate anchor tiles labeled A1, A2, A3, etc.\n\n"
        "Each tile is one anchor candidate.\n"
        "Inside each tile:\n"
        "- TOP patch = one visible top-view edge region\n"
        "- BOTTOM patch = the opposite visible top-view edge region\n\n"
        "The TOP and BOTTOM patches in the same tile must be judged together.\n\n"
        "Physical task:\n"
        "A flat planar probe will later press against the object's side surfaces at these two opposite edge regions.\n"
        "The probe needs clean, unobstructed, approximately planar side-contact regions.\n\n"
        "Critical visual rule:\n"
        "In these images, gray/white cement-like material, mortar, attached paste, bulges, blobs, protrusions, or crust stuck to the edge are BAD.\n"
        "This is especially important in the BOTTOM patches.\n"
        "If gray/white material protrudes from, hangs below, sticks to, or interrupts the edge boundary, it should be treated as mortar/attached material and as a planar-contact obstruction.\n\n"
        "Do NOT reward an anchor just because the edge looks bright, white, or continuous.\n"
        "White/gray attached material near the edge is usually mortar or contamination, not clean brick surface.\n\n"
        "Strong penalty / veto rules:\n"
        "- If either TOP or BOTTOM has attached mortar/cement-like material at the contact edge, set has_mortar_or_attached_material = 1.\n"
        "- If either TOP or BOTTOM has a protrusion, blob, crust, nail, debris, or foreign object at the contact edge, set contact_blocked = 1.\n"
        "- If bottom edge has a large gray/white protruding region, the candidate should receive a low score and should not be selected as best unless all candidates are worse.\n"
        "- A candidate with one blocked side should rank below candidates whose TOP and BOTTOM edges are both cleaner and more open.\n"
        "- Prefer clean brick edge regions even if they are not perfect.\n\n"
        "What to evaluate:\n"
        "For every visible anchor, evaluate:\n"
        "1. straightness of TOP and BOTTOM edge regions\n"
        "2. whether the contact edge is clean and unobstructed\n"
        "3. whether mortar/cement-like material is attached near the contact edge\n"
        "4. whether there are protrusions, blobs, nails, debris, chips, broken/jagged edge, or other non-planar obstructions\n"
        "5. whether both opposite edges together are suitable for planar probe contact\n\n"
        "Scoring:\n"
        "Use overall_score from 0 to 100.\n"
        "- 0 to 20 = unusable or blocked contact\n"
        "- 21 to 40 = poor\n"
        "- 41 to 60 = moderate\n"
        "- 61 to 80 = good\n"
        "- 81 to 100 = very good\n\n"
        "Be comparative:\n"
        "Even if all candidates are imperfect, choose the best available candidate.\n"
        "Do not assume the leftmost, rightmost, center, A1, A2, A3, A4, or A5 is best.\n"
        "Do not rank candidates by their order in the image or by their order in the JSON schema.\n"
        "Do not give every candidate the same clean reason unless every patch truly looks identical.\n"
        "Judge only visible TOP and BOTTOM edge/contact quality.\n\n"
        "Return ONLY valid JSON, no markdown fences, no extra text.\n\n"
        "Use exactly this flat schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Important schema rule:\n"
        "- The schema above uses placeholders only. Replace placeholders with actual candidate IDs from the image.\n"
        "- Do NOT copy the placeholder order as the ranking.\n"
        "- Do NOT assign scores by candidate order such as A1=90, A2=85, A3=80.\n"
        "- A1, A2, A3, A4, and A5 have no default priority.\n"
        "- The best_anchor must be chosen from visual TOP/BOTTOM contact quality only.\n\n"
        "Field rules:\n"
        "- top_blocked, bottom_blocked, has_mortar_or_attached_material, has_nail_debris_or_foreign_object, and has_broken_or_jagged_edge must be 0 or 1.\n"
        "- overall_score must be an integer from 0 to 100.\n"
        "- Include every visible anchor exactly once.\n"
        "- Sort anchors from highest overall_score to lowest overall_score.\n"
        "- ranking_best_to_worst must match the same order.\n"
        "- best_anchor must be the highest-scoring anchor.\n"
        "- Keep reasons short.\n"
        "- If mortar/attached material is visible, mention it in the reason.\n"
    )


def build_qwen_anchor_prompt(
    *,
    requested_material: str,
    axis_mode: str,
    candidate_ids: list[str],
    deterministic_features: dict[str, Any],
    vetoed_ids: list[str],
    include_numeric_deterministic_features: bool = False,
    prompt_version: str | None = None,
) -> str:
    if prompt_version == QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION:
        return _build_legacy_v4_replicated_prompt(candidate_ids=candidate_ids)

    veto_summary = _build_qwen_visible_veto_summary(deterministic_features, candidate_ids)
    feature_summary = deterministic_features if include_numeric_deterministic_features else veto_summary
    anchor_template = {
        "id": "A1",
        "top_visible_defects": ["none"],
        "bottom_visible_defects": ["none"],
        "top_blocked": 0,
        "bottom_blocked": 0,
        "has_mortar_or_attached_material": 0,
        "has_nail_debris_or_foreign_object": 0,
        "has_broken_or_jagged_edge": 0,
        "normal_material_texture_not_defect": 1,
        "top_score": 0,
        "bottom_score": 0,
        "overall_score": 0,
        "contact_region_usable": True,
        "reason": "short specific reason",
    }
    schema_anchors = []
    for candidate_id in candidate_ids:
        item = dict(anchor_template)
        item["id"] = candidate_id
        schema_anchors.append(item)
    schema = {
        "anchors": schema_anchors,
        "ranking_best_to_worst": candidate_ids,
        "best_anchor": candidate_ids[0] if candidate_ids else "NO_SAFE_ANCHOR",
        "best_anchor_reason": "short specific reason",
        "no_safe_anchor": False,
        "no_safe_anchor_reason": None,
    }
    return (
        "You are performing qwen_anchor_selection_upv_vlm_v2.\n\n"
        "You are selecting the best physical contact anchor for ultrasonic pulse velocity (UPV) testing on a construction material object.\n\n"
        f"Requested material: {requested_material}\n"
        f"Axis mode: {axis_mode}\n\n"
        "The image contains candidate anchor tiles labeled A1, A2, A3, etc.\n\n"
        "Each tile is one anchor candidate.\n"
        "Inside each tile:\n"
        "- TOP patch = one intended edge/contact region of the object\n"
        "- BOTTOM patch = the opposite intended edge/contact region of the same object\n\n"
        "The TOP and BOTTOM patches in the same tile must be judged together.\n\n"
        "Physical UPV task:\n"
        "A pair of flat planar UPV transducers will later press against the object's two opposite side surfaces at these two contact regions.\n\n"
        "A good anchor requires BOTH opposite contact regions to be:\n"
        "- clean\n"
        "- open\n"
        "- continuous\n"
        "- approximately planar\n"
        "- not locally blocked\n"
        "- not covered by attached material\n"
        "- not interrupted by chips, cracks, jagged breaks, nails, tape, debris, protrusions, or mortar/cement paste\n\n"
        "Your goal:\n"
        "Choose the candidate anchor that gives the best real physical UPV transducer contact.\n\n"
        "Do NOT choose an anchor only because the dashed guide line looks straight.\n"
        "Do NOT choose an anchor only because it is near the center.\n"
        "Do NOT choose an anchor only because deterministic scores are high.\n"
        "Visual evidence of contact blockage or edge damage overrides deterministic scores.\n\n"
        "Important visual inspection rule:\n"
        "Judge the full visible TOP and BOTTOM patch, not only the dashed guide line.\n"
        "A defect near the intended contact edge can make the anchor bad even if the dashed line itself appears straight.\n\n"
        "Material-awareness rule:\n"
        "Do not confuse normal object texture with defects.\n\n"
        "For BRICK:\n"
        "Normal red, brown, orange, tan, beige, cream, yellowish, speckled, porous, dusty, or rough brick texture is not automatically mortar.\n"
        "However, attached gray/white cement-like paste, mortar blobs, crust, protruding material, missing chipped edge, jagged broken edge, or debris at the contact edge is a defect.\n\n"
        "For CONCRETE / CINDER BLOCK:\n"
        "Normal gray concrete color, aggregate texture, pores, rough cementitious texture, and light/dark gray variation are not automatically defects.\n"
        "However, protruding attached paste, loose debris, broken/jagged edge, local obstruction, or material sticking out from the intended contact edge is a defect.\n\n"
        "For TIMBER / WOOD:\n"
        "Normal grain, knots, color bands, saw marks, and wood texture are not defects.\n"
        "However, nails, screws, staples, tape, splinters, broken/jagged edge, large cracks, protruding chips, loose debris, or blocked contact regions are defects.\n\n"
        "Defect / obstruction rules:\n"
        "Strongly penalize an anchor if either TOP or BOTTOM has:\n"
        "- attached mortar, cement paste, glue, crust, or clumped material at the contact edge\n"
        "- nail, screw, staple, wire, tape, stone, loose debris, or foreign object near the contact edge\n"
        "- broken, chipped, jagged, cracked, missing, or highly uneven edge where the transducer would contact\n"
        "- protrusion, blob, bump, or raised material that would prevent flat contact\n"
        "- one side clean but the opposite side blocked or damaged\n\n"
        "Hard contact rule:\n"
        "If either TOP or BOTTOM contact region is blocked, the anchor should normally not be selected.\n"
        "A candidate with one blocked side must rank below candidates where both TOP and BOTTOM are cleaner and more open.\n"
        "A candidate with mortar/attached material/protrusion at the contact edge must rank below a candidate with clean continuous edges.\n"
        "A candidate with a broken/jagged/chipped contact edge must rank below a candidate with straight continuous contact edges.\n\n"
        "Visual-first decision process:\n"
        "You must follow this order:\n"
        "1. Inspect the image tile for each candidate.\n"
        "2. List visible defects separately for the TOP patch and BOTTOM patch.\n"
        "3. Set contact flags from the visible defects.\n"
        "4. Score TOP and BOTTOM contact quality.\n"
        "5. Rank anchors only after the visual defect audit is complete.\n\n"
        "Do not score first.\n"
        "Do not assume the center anchor is best.\n"
        "Do not copy any deterministic or geometric ranking.\n"
        "Do not give the same generic reason to every anchor.\n"
        "Each reason must mention the actual visual condition of that anchor.\n\n"
        "A candidate with visible contact damage, mortar, nail/tape/debris, protrusion, broken edge, or blocked side must score lower than a visually clean candidate even if its edge line looks straight.\n\n"
        "Hard visual override:\n"
        "Visual contact evidence is the primary evidence.\n"
        "Deterministic information is used only to warn about hard vetoes.\n"
        "If a candidate looks damaged or obstructed in the image, mark the relevant defect flags even if there is no deterministic veto.\n"
        "If deterministic information and image evidence disagree, trust the image evidence.\n\n"
        "Relative ranking rule:\n"
        "Compare candidates against each other.\n"
        "Choose the best available anchor among the visible candidates.\n"
        "If all candidates are imperfect, choose the least bad candidate only if it is still physically usable for UPV contact.\n"
        "Return NO_SAFE_ANCHOR only if every candidate is physically unusable for UPV contact.\n\n"
        f"Candidate IDs:\n{candidate_ids}\n\n"
        f"Deterministic vetoed IDs:\n{vetoed_ids}\n\n"
        "Deterministic hard-veto summary:\n"
        "The following JSON only tells you whether the deterministic stage found a hard veto.\n"
        "It does NOT contain scores and must NOT be used as a ranking.\n"
        "If deterministic_hard_veto is true, strongly avoid that anchor unless every other anchor is worse.\n"
        "If deterministic_hard_veto is false, you must still inspect the image yourself.\n\n"
        f"Hard-veto summary JSON:\n{json.dumps(feature_summary, indent=2)}\n\n"
        "Return ONLY valid JSON.\n"
        "Do not include markdown fences.\n"
        "Do not include text outside JSON.\n\n"
        "Use exactly this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n"
        "\nField rules:\n"
        "- Include every visible candidate exactly once.\n"
        "- id must be one of the known candidate IDs.\n"
        "- top_visible_defects and bottom_visible_defects must be lists of short strings.\n"
        "- Use [\"none\"] only if no visible defect is present in that patch.\n"
        "- Valid defect terms include: mortar_or_attached_material, nail_or_foreign_object, debris_or_tape, broken_or_jagged_edge, chip_or_missing_edge, crack_or_split, protrusion_or_bump, blocked_contact_region, uncertain_possible_defect, none.\n"
        "- top_blocked, bottom_blocked, has_mortar_or_attached_material, has_nail_debris_or_foreign_object, has_broken_or_jagged_edge, and normal_material_texture_not_defect must be 0 or 1.\n"
        "- top_score, bottom_score, and overall_score must be integers from 0 to 100.\n"
        "- contact_region_usable must be true or false.\n"
        "- Sort anchors from highest overall_score to lowest overall_score.\n"
        "- ranking_best_to_worst must match the same order.\n"
        "- best_anchor must be the highest-scoring physically usable UPV contact anchor.\n"
        "- Do not select an anchor with contact_region_usable=false.\n"
        "- If all anchors are physically unusable, set best_anchor to \"NO_SAFE_ANCHOR\", no_safe_anchor to true, and explain why every candidate is unusable.\n"
        "- Do not set best_anchor to \"NO_SAFE_ANCHOR\" if any candidate has contact_region_usable=true.\n"
        "- Keep reasons short, visual, and specific.\n"
    )


REQUIRED_CONTACT_FLAGS = {
    "top_visible_defects",
    "bottom_visible_defects",
    "top_blocked",
    "bottom_blocked",
    "has_mortar_or_attached_material",
    "has_nail_debris_or_foreign_object",
    "has_broken_or_jagged_edge",
    "contact_region_usable",
    "reason",
}


def _build_qwen_visible_veto_summary(deterministic_features: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    features = _feature_by_id(deterministic_features)
    summary: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        feature = features.get(str(candidate_id), {})
        summary[str(candidate_id)] = {
            "deterministic_hard_veto": bool(feature.get("veto", False)),
            "veto_reasons": list(feature.get("veto_reasons", []) or []),
        }
    return summary


def _ensure_visual_defect_fields(parsed: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    assessments = parsed.get("candidate_assessments")
    if not isinstance(assessments, dict):
        return parsed
    for candidate_id in candidate_ids:
        entry = assessments.get(candidate_id)
        if not isinstance(entry, dict):
            continue
        entry.setdefault("top_visible_defects", ["none"])
        entry.setdefault("bottom_visible_defects", ["none"])
    return parsed


def _validate_contact_flags(parsed: dict[str, Any], candidate_ids: list[str]) -> tuple[bool, str | None, dict[str, list[str]]]:
    assessments = parsed.get("candidate_assessments")
    if not isinstance(assessments, dict):
        return False, "candidate_assessments missing", {item: sorted(REQUIRED_CONTACT_FLAGS) for item in candidate_ids}
    missing: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        entry = assessments.get(candidate_id)
        if not isinstance(entry, dict):
            missing[candidate_id] = sorted(REQUIRED_CONTACT_FLAGS)
            continue
        fields = sorted(flag for flag in REQUIRED_CONTACT_FLAGS if flag not in entry)
        if fields:
            missing[candidate_id] = fields
    if missing:
        return False, "required contact flags missing", missing
    return True, None, {}


def _score_0_100(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return score * 100.0 if 0.0 <= score <= 1.0 else score


def _quality_label(value: Any) -> str:
    score = _score_0_100(value)
    if score >= 85.0:
        return "good"
    if score >= 65.0:
        return "acceptable"
    return "poor"


def _feature_by_id(deterministic_features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("anchor_id")): dict(item) for item in deterministic_features.get("anchor_features", [])}


def _repair_missing_assessments(
    *,
    parsed: dict[str, Any],
    candidate_ids: list[str],
    deterministic_features: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    repaired = json.loads(json.dumps(parsed))
    assessments = repaired.setdefault("candidate_assessments", {})
    features = _feature_by_id(deterministic_features)
    missing = [candidate_id for candidate_id in candidate_ids if not isinstance(assessments.get(candidate_id), dict)]
    for candidate_id in missing:
        feature = features.get(candidate_id, {})
        veto = bool(feature.get("veto"))
        deterministic_score = _score_0_100(feature.get("deterministic_score"))
        assessments[candidate_id] = {
            "top_visible_defects": ["none"],
            "bottom_visible_defects": ["none"],
            "top_blocked": 0,
            "bottom_blocked": 0,
            "has_mortar_or_attached_material": 0,
            "has_nail_debris_or_foreign_object": 0,
            "has_broken_or_jagged_edge": 0,
            "edge_straightness": _quality_label(feature.get("edge_straightness")),
            "opposite_edge_consistency": _quality_label(feature.get("paired_consistency", feature.get("local_edge_consistency"))),
            "contact_region_usable": False if veto else deterministic_score >= 50.0,
            "confidence": "medium",
            "reason": "Assessment repaired from deterministic features because Qwen omitted this candidate.",
            "assessment_source": "deterministic_repair",
            "deterministic_score": deterministic_score,
            "veto": veto,
            "veto_reasons": feature.get("veto_reasons", []),
        }
    return repaired, missing


def _best_usable_candidate(
    *,
    candidate_ids: list[str],
    parsed: dict[str, Any],
    deterministic_features: dict[str, Any],
    vetoed_ids: list[str],
) -> str | None:
    features = _feature_by_id(deterministic_features)
    assessments = parsed.get("candidate_assessments", {}) if isinstance(parsed.get("candidate_assessments"), dict) else {}
    usable: list[tuple[float, str]] = []
    unsafe_ids = {str(item) for item in parsed.get("unsafe_anchor_ids", [])}
    for candidate_id in candidate_ids:
        assessment = assessments.get(candidate_id, {})
        if not isinstance(assessment, dict):
            continue
        if candidate_id in unsafe_ids or candidate_id in vetoed_ids:
            continue
        if assessment.get("contact_region_usable") is False:
            continue
        feature = features.get(candidate_id, {})
        usable.append((_score_0_100(feature.get("deterministic_score")), candidate_id))
    if not usable:
        return None
    usable.sort(reverse=True)
    return usable[0][1]


def _all_assessed_candidates_unsafe(parsed: dict[str, Any], candidate_ids: list[str]) -> bool:
    assessments = parsed.get("candidate_assessments", {}) if isinstance(parsed.get("candidate_assessments"), dict) else {}
    unsafe_ids = {str(item) for item in parsed.get("unsafe_anchor_ids", [])}
    for candidate_id in candidate_ids:
        assessment = assessments.get(candidate_id, {})
        if isinstance(assessment, dict) and assessment.get("contact_region_usable") is not False and candidate_id not in unsafe_ids:
            return False
    return True


def _as_defect_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value] or list(default)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return list(default)


def _normalize_anchor_schema(
    parsed: dict[str, Any],
    candidate_ids: list[str],
    *,
    prompt_version: str | None = None,
) -> tuple[dict[str, Any], str]:
    if isinstance(parsed.get("anchors"), list):
        legacy_v4 = prompt_version == QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION
        anchors = [item for item in parsed.get("anchors", []) if isinstance(item, dict)]
        assessments: dict[str, dict[str, Any]] = {}
        unsafe_ids: list[str] = []
        best_anchor = str(parsed.get("best_anchor") or "")
        for item in anchors:
            anchor_id = str(item.get("id") or "")
            if not anchor_id:
                continue
            overall_score = int(_score_0_100(item.get("overall_score")))
            nail_flag = int(item.get("has_nail_debris_or_foreign_object", item.get("has_nail_or_foreign_object", 0)) or 0)
            top_blocked = int(item.get("top_blocked", 0) or 0)
            bottom_blocked = int(item.get("bottom_blocked", 0) or 0)
            mortar_flag = int(item.get("has_mortar_or_attached_material", 0) or 0)
            broken_flag = int(item.get("has_broken_or_jagged_edge", 0) or 0)
            if "contact_region_usable" in item:
                contact_region_usable = bool(item.get("contact_region_usable"))
            else:
                contact_region_usable = not any((top_blocked, bottom_blocked, mortar_flag, nail_flag, broken_flag))
            assessment = {
                "top_visible_defects": _as_defect_list(
                    item.get("top_visible_defects"),
                    ["not_reported_legacy_schema"] if legacy_v4 else ["none"],
                ),
                "bottom_visible_defects": _as_defect_list(
                    item.get("bottom_visible_defects"),
                    ["not_reported_legacy_schema"] if legacy_v4 else ["none"],
                ),
                "top_blocked": top_blocked,
                "bottom_blocked": bottom_blocked,
                "has_mortar_or_attached_material": mortar_flag,
                "has_nail_debris_or_foreign_object": nail_flag,
                "has_broken_or_jagged_edge": broken_flag,
                "normal_material_texture_not_defect": int(item.get("normal_material_texture_not_defect", 1 if legacy_v4 else 0) or 0),
                "top_score": int(_score_0_100(item.get("top_score", overall_score))),
                "bottom_score": int(_score_0_100(item.get("bottom_score", overall_score))),
                "overall_score": overall_score,
                "contact_region_usable": contact_region_usable,
                "reason": str(item.get("reason", "")),
                "assessment_source": "qwen_anchor_legacy_v4_replicated" if legacy_v4 else "qwen_anchor_score_ranking_v1",
            }
            assessments[anchor_id] = assessment
            if not assessment["contact_region_usable"]:
                if not (legacy_v4 and anchor_id == best_anchor):
                    unsafe_ids.append(anchor_id)
        normalized = dict(parsed)
        no_safe = bool(parsed.get("no_safe_anchor", False)) or best_anchor == "NO_SAFE_ANCHOR"
        normalized["selected_anchor_id"] = "NO_SAFE_ANCHOR" if no_safe else best_anchor
        normalized["ranked_anchor_ids"] = [str(item) for item in parsed.get("ranking_best_to_worst", [])]
        normalized["unsafe_anchor_ids"] = unsafe_ids
        normalized["candidate_assessments"] = assessments
        normalized["reasoning_short"] = parsed.get("best_anchor_reason") or parsed.get("no_safe_anchor_reason")
        normalized["parsed_original_schema"] = parsed
        normalized_schema = QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION if legacy_v4 else QWEN_ANCHOR_PROMPT_VERSION
        normalized["normalized_from_schema"] = normalized_schema
        return normalized, normalized_schema
    parsed.setdefault("normalized_from_schema", "selected_anchor_id_legacy")
    return parsed, "selected_anchor_id_legacy"


def _generic_repeated_all_unsafe_reason(parsed: dict[str, Any], candidate_ids: list[str], min_repetition: int) -> bool:
    assessments = parsed.get("candidate_assessments", {}) if isinstance(parsed.get("candidate_assessments"), dict) else {}
    reasons = []
    for candidate_id in candidate_ids:
        assessment = assessments.get(candidate_id, {})
        if isinstance(assessment, dict):
            reasons.append(str(assessment.get("reason", "")).strip().lower())
    nonempty = [item for item in reasons if item]
    if len(nonempty) < int(min_repetition):
        return False
    counts: dict[str, int] = {}
    for reason in nonempty:
        counts[reason] = counts.get(reason, 0) + 1
    repeated = max(counts.values()) >= int(min_repetition)
    generic_terms = ("raised mortar blob", "mortar blob", "blocked contact region")
    return repeated and any(term in reason for reason in nonempty for term in generic_terms)


def _generic_visual_audit_response(parsed: dict[str, Any], candidate_ids: list[str]) -> bool:
    assessments = parsed.get("candidate_assessments", {}) if isinstance(parsed.get("candidate_assessments"), dict) else {}
    if not candidate_ids:
        return False
    generic_phrases = (
        "clean and continuous contact region",
        "highest overall score and clean contact region",
        "no apparent defects",
        "clean contact region",
    )
    all_clean = 0
    generic_reasons = 0
    reasons: dict[str, int] = {}
    for candidate_id in candidate_ids:
        assessment = assessments.get(candidate_id, {})
        if not isinstance(assessment, dict):
            continue
        top_defects = [str(item).strip().lower() for item in assessment.get("top_visible_defects", ["none"])]
        bottom_defects = [str(item).strip().lower() for item in assessment.get("bottom_visible_defects", ["none"])]
        flags_clear = all(
            int(assessment.get(key, 0) or 0) == 0
            for key in (
                "top_blocked",
                "bottom_blocked",
                "has_mortar_or_attached_material",
                "has_nail_debris_or_foreign_object",
                "has_broken_or_jagged_edge",
            )
        )
        if top_defects == ["none"] and bottom_defects == ["none"] and flags_clear:
            all_clean += 1
        reason = str(assessment.get("reason", "")).strip().lower()
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            if any(phrase in reason for phrase in generic_phrases):
                generic_reasons += 1
    if all_clean / max(len(candidate_ids), 1) < 0.70:
        return False
    if generic_reasons / max(len(candidate_ids), 1) >= 0.70:
        return True
    return bool(reasons and max(reasons.values()) / max(len(candidate_ids), 1) >= 0.70)


def _deterministic_best_if_strong(
    *,
    deterministic_features: dict[str, Any],
    candidate_ids: list[str],
    vetoed_ids: list[str],
    min_score: float,
) -> str | None:
    best: tuple[float, str] | None = None
    for item in deterministic_features.get("anchor_features", []):
        anchor_id = str(item.get("anchor_id"))
        if anchor_id not in candidate_ids or anchor_id in vetoed_ids or bool(item.get("veto")):
            continue
        score = _score_0_100(item.get("deterministic_score"))
        if score >= float(min_score) and (best is None or score > best[0]):
            best = (score, anchor_id)
    return best[1] if best else None


def _response_text(response: dict[str, Any]) -> str:
    if "selected_anchor_id" in response:
        return json.dumps(response)
    for key in ("text", "response", "generated_text", "raw_text", "content"):
        if response.get(key):
            return str(response[key])
    if response.get("parsed_json"):
        return json.dumps(response["parsed_json"])
    return json.dumps(response)


def run_required_qwen_anchor_selection(
    *,
    config: dict[str, Any],
    grid_image_path: str,
    requested_material: str,
    axis_mode: str,
    candidates: list[dict[str, Any]],
    deterministic_features: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.get("anchor_selection", {})
    candidate_ids = [str(item["anchor_id"]) for item in candidates]
    vetoed_ids = [str(item["anchor_id"]) for item in candidates if item.get("veto")]
    prompt_version = str(cfg.get("qwen_prompt_version") or QWEN_ANCHOR_PROMPT_VERSION)
    prompt = build_qwen_anchor_prompt(
        requested_material=requested_material,
        axis_mode=axis_mode,
        candidate_ids=candidate_ids,
        deterministic_features=deterministic_features,
        vetoed_ids=vetoed_ids,
        include_numeric_deterministic_features=bool(cfg.get("qwen_prompt_include_numeric_deterministic_features", False)),
        prompt_version=prompt_version,
    )
    (out / "qwen_anchor_prompt.txt").write_text(prompt, encoding="utf-8")
    manifest = {
        "qwen_required": True,
        "qwen_input_image": grid_image_path,
        "candidate_ids": candidate_ids,
        "vetoed_anchor_ids": vetoed_ids,
        "qwen_server_url": cfg.get("qwen_server_url"),
        "qwen_prompt_version": prompt_version,
        "mock_qwen_response": config.get("_runtime_mock_qwen_response"),
    }
    (out / "qwen_anchor_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        if config.get("_runtime_mock_qwen_response"):
            response = load_mock_qwen_response(config["_runtime_mock_qwen_response"])
        elif cfg.get("qwen_server_url"):
            response = infer_qwen_anchor(
                server_url=str(cfg["qwen_server_url"]),
                image_path=grid_image_path,
                prompt_text=prompt,
                timeout_sec=float(cfg.get("qwen_timeout_sec", 90)),
                output_path=str(out / "qwen_server_output.json"),
            )
        else:
            return {
                "success": False,
                "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
                "error": "qwen_server_url is null and no mock_qwen_response was provided",
            }
        if response.get("success") is False or response.get("ok") is False:
            return {
                "success": False,
                "failure_reason": response.get("failure_reason", "QWEN_REQUIRED_BUT_FAILED"),
                "error": response.get("error", "qwen_response_not_ok"),
                "response": response,
            }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "failure_reason": "QWEN_REQUIRED_BUT_FAILED", "error": str(exc), "exception_type": type(exc).__name__}

    raw_text = _response_text(response)
    (out / "qwen_anchor_response_raw.txt").write_text(raw_text, encoding="utf-8")
    try:
        parsed = response if "selected_anchor_id" in response else extract_json_object(raw_text)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "failure_reason": "QWEN_INVALID_JSON", "raw_response": raw_text, "error": str(exc)}
    (out / "qwen_anchor_response_parsed.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    parsed_original = json.loads(json.dumps(parsed))
    parsed, normalized_schema = _normalize_anchor_schema(parsed, candidate_ids, prompt_version=prompt_version)
    parsed = _ensure_visual_defect_fields(parsed, candidate_ids)
    if normalized_schema != "selected_anchor_id_legacy":
        (out / "qwen_anchor_response_normalized.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    selected_id = str(parsed.get("selected_anchor_id") or "")
    ranked = [str(item) for item in parsed.get("ranked_anchor_ids", [])]
    allow_no_safe_anchor = bool(cfg.get("qwen_allow_no_safe_anchor", True))
    require_flags = bool(cfg.get("qwen_required_contact_flags", True))
    repair_enabled = bool(cfg.get("qwen_auto_repair_missing_assessments", True))
    repair_from_features = bool(cfg.get("qwen_auto_repair_missing_assessments_from_deterministic_features", True))
    qwen_response_repaired = False
    qwen_missing_assessment_ids: list[str] = []
    validation_warnings: list[str] = []
    selected_before_repair = selected_id
    no_safe_anchor_inconsistent = False
    no_safe_anchor_repaired_to_anchor_id: str | None = None
    qwen_repair_reason: str | None = None
    if require_flags:
        ok_flags, flag_error, missing = _validate_contact_flags(parsed, candidate_ids)
        if not ok_flags:
            qwen_missing_assessment_ids = [candidate_id for candidate_id in candidate_ids if candidate_id in missing]
            if repair_enabled and repair_from_features and qwen_missing_assessment_ids:
                parsed, repaired_ids = _repair_missing_assessments(
                    parsed=parsed,
                    candidate_ids=candidate_ids,
                    deterministic_features=deterministic_features,
                )
                qwen_missing_assessment_ids = repaired_ids
                qwen_response_repaired = True
                validation_warnings.append("QWEN_MISSING_ASSESSMENTS_REPAIRED")
                qwen_repair_reason = "Qwen omitted candidate assessments; missing assessments repaired from deterministic features."
                (out / "qwen_anchor_response_repaired.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                ok_flags, flag_error, missing = _validate_contact_flags(parsed, candidate_ids)
            if not ok_flags and bool(cfg.get("qwen_fail_if_missing_required_flags", True)):
                return {
                    "success": False,
                    "failure_reason": "QWEN_MISSING_REQUIRED_CONTACT_FLAGS",
                    "parsed": parsed,
                    "error": flag_error,
                    "missing_required_contact_flags": missing,
                    "qwen_missing_assessment_ids": qwen_missing_assessment_ids,
                    "qwen_response_repaired": qwen_response_repaired,
                    "validation_warnings": validation_warnings,
                }
    if _generic_visual_audit_response(parsed, candidate_ids):
        validation_warnings.append("QWEN_VISUAL_DEFECT_AUDIT_GENERIC")
    if selected_id == "NO_SAFE_ANCHOR" and bool(cfg.get("qwen_no_safe_anchor_requires_all_unsafe", True)):
        if not _all_assessed_candidates_unsafe(parsed, candidate_ids):
            no_safe_anchor_inconsistent = True
            if repair_enabled:
                repaired = _best_usable_candidate(
                    candidate_ids=candidate_ids,
                    parsed=parsed,
                    deterministic_features=deterministic_features,
                    vetoed_ids=vetoed_ids,
                )
                if repaired:
                    selected_id = repaired
                    no_safe_anchor_repaired_to_anchor_id = repaired
                    qwen_response_repaired = True
                    validation_warnings.append("QWEN_NO_SAFE_ANCHOR_REPAIRED")
                    qwen_repair_reason = (
                        qwen_repair_reason or "Qwen returned NO_SAFE_ANCHOR but at least one assessed/repaired candidate is usable."
                    )
        elif (
            bool(cfg.get("qwen_repair_all_unsafe_if_no_deterministic_veto", True))
            and str(cfg.get("qwen_task_mode", "")) == "measurement_anchor_selection"
            and not vetoed_ids
            and _generic_repeated_all_unsafe_reason(
                parsed,
                candidate_ids,
                int(cfg.get("qwen_generic_repeated_reason_min_repetition", 3)),
            )
        ):
            repaired = _deterministic_best_if_strong(
                deterministic_features=deterministic_features,
                candidate_ids=candidate_ids,
                vetoed_ids=vetoed_ids,
                min_score=float(cfg.get("qwen_measurement_min_deterministic_repair_score", 75.0)),
            )
            if repaired:
                no_safe_anchor_inconsistent = True
                selected_id = repaired
                no_safe_anchor_repaired_to_anchor_id = repaired
                parsed["selected_anchor_id"] = repaired
                parsed["unsafe_anchor_ids"] = [item for item in parsed.get("unsafe_anchor_ids", []) if str(item) != repaired]
                if isinstance(parsed.get("candidate_assessments"), dict) and isinstance(parsed["candidate_assessments"].get(repaired), dict):
                    parsed["candidate_assessments"][repaired]["contact_region_usable"] = True
                    parsed["candidate_assessments"][repaired]["assessment_source"] = "qwen_generic_all_unsafe_repair"
                qwen_response_repaired = True
                validation_warnings.append("QWEN_GENERIC_ALL_UNSAFE_REPAIRED")
                qwen_repair_reason = (
                    "Qwen marked all anchors unsafe with generic repeated obstruction reason despite no deterministic vetoes; "
                    "repaired to deterministic best for measurement mode."
                )
    if selected_id == "NO_SAFE_ANCHOR":
        if not allow_no_safe_anchor:
            return {"success": False, "failure_reason": "QWEN_SELECTED_UNKNOWN_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
        unknown_ranked = [item for item in ranked if item not in candidate_ids]
        if unknown_ranked:
            return {"success": False, "failure_reason": "QWEN_SELECTED_UNKNOWN_ANCHOR", "parsed": parsed, "unknown_ranked_anchor_ids": unknown_ranked}
        decision = {
            "success": True,
            "safe_no_anchor": True,
            "selected_anchor_id": "NO_SAFE_ANCHOR",
            "ranked_anchor_ids": ranked,
            "unsafe_anchor_ids": [str(item) for item in parsed.get("unsafe_anchor_ids", [])],
            "candidate_assessments": parsed.get("candidate_assessments", {}),
            "reasoning_short": parsed.get("reasoning_short"),
            "parsed": parsed,
            "parsed_original_schema": parsed_original,
            "normalized_from_schema": normalized_schema,
            "qwen_response_repaired": qwen_response_repaired,
            "qwen_missing_assessment_ids": qwen_missing_assessment_ids,
            "validation_warnings": validation_warnings,
            "no_safe_anchor_inconsistent": no_safe_anchor_inconsistent,
            "no_safe_anchor_repaired_to_anchor_id": no_safe_anchor_repaired_to_anchor_id,
            "qwen_repair_reason": qwen_repair_reason,
            "selected_anchor_before_repair": selected_before_repair,
            "selected_anchor_after_repair": "NO_SAFE_ANCHOR",
            "final_selected_anchor_source": "qwen_required_no_safe_anchor",
        }
        (out / "qwen_anchor_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        return decision
    if selected_id not in candidate_ids:
        return {"success": False, "failure_reason": "QWEN_SELECTED_UNKNOWN_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
    unknown_ranked = [item for item in ranked if item not in candidate_ids]
    if unknown_ranked:
        return {"success": False, "failure_reason": "QWEN_SELECTED_UNKNOWN_ANCHOR", "parsed": parsed, "unknown_ranked_anchor_ids": unknown_ranked}
    if selected_id in vetoed_ids and not bool(cfg.get("qwen_allow_vetoed_anchor_selection", False)):
        return {"success": False, "failure_reason": "QWEN_SELECTED_VETOED_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
    selected_assessment = (parsed.get("candidate_assessments") or {}).get(selected_id, {})
    legacy_v4_replicated = normalized_schema == QWEN_ANCHOR_LEGACY_V4_REPLICATED_PROMPT_VERSION
    if isinstance(selected_assessment, dict) and selected_assessment.get("contact_region_usable") is False and not legacy_v4_replicated:
        if repair_enabled and str(cfg.get("qwen_task_mode", "")) == "measurement_anchor_selection":
            repaired = _best_usable_candidate(
                candidate_ids=candidate_ids,
                parsed=parsed,
                deterministic_features=deterministic_features,
                vetoed_ids=vetoed_ids,
            )
            if repaired and repaired != selected_id:
                validation_warnings.append("QWEN_SELECTED_UNUSABLE_ANCHOR_REPAIRED")
                qwen_response_repaired = True
                qwen_repair_reason = qwen_repair_reason or "Qwen selected an unusable anchor; repaired to best usable deterministic candidate."
                selected_id = repaired
                selected_assessment = (parsed.get("candidate_assessments") or {}).get(selected_id, {})
            else:
                return {"success": False, "failure_reason": "QWEN_SELECTED_UNUSABLE_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
        else:
            return {"success": False, "failure_reason": "QWEN_SELECTED_UNUSABLE_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
    if isinstance(selected_assessment, dict) and selected_assessment.get("contact_region_usable") is False and not legacy_v4_replicated:
        return {"success": False, "failure_reason": "QWEN_SELECTED_UNUSABLE_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
    unsafe_ids = [str(item) for item in parsed.get("unsafe_anchor_ids", [])]
    if selected_id in unsafe_ids:
        if repair_enabled and str(cfg.get("qwen_task_mode", "")) == "measurement_anchor_selection":
            repaired = _best_usable_candidate(
                candidate_ids=candidate_ids,
                parsed=parsed,
                deterministic_features=deterministic_features,
                vetoed_ids=vetoed_ids,
            )
            if repaired and repaired != selected_id:
                validation_warnings.append("QWEN_SELECTED_UNSAFE_ANCHOR_REPAIRED")
                qwen_response_repaired = True
                qwen_repair_reason = qwen_repair_reason or "Qwen selected an unsafe anchor; repaired to best usable deterministic candidate."
                selected_id = repaired
            else:
                return {"success": False, "failure_reason": "QWEN_SELECTED_UNSAFE_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
        else:
            return {"success": False, "failure_reason": "QWEN_SELECTED_UNSAFE_ANCHOR", "parsed": parsed, "selected_anchor_id": selected_id}
    decision = {
        "success": True,
        "safe_no_anchor": False,
        "selected_anchor_id": selected_id,
        "ranked_anchor_ids": ranked,
        "unsafe_anchor_ids": unsafe_ids,
        "candidate_assessments": parsed.get("candidate_assessments", {}),
        "reasoning_short": parsed.get("reasoning_short"),
        "contact_quality_notes": parsed.get("contact_quality_notes", {}),
        "parsed": parsed,
        "parsed_original_schema": parsed_original,
        "normalized_from_schema": normalized_schema,
        "qwen_response_repaired": qwen_response_repaired,
        "qwen_missing_assessment_ids": qwen_missing_assessment_ids,
        "validation_warnings": validation_warnings,
        "no_safe_anchor_inconsistent": no_safe_anchor_inconsistent,
        "no_safe_anchor_repaired_to_anchor_id": no_safe_anchor_repaired_to_anchor_id,
        "qwen_repair_reason": qwen_repair_reason,
        "selected_anchor_before_repair": selected_before_repair,
        "selected_anchor_after_repair": selected_id,
        "final_selected_anchor_source": "qwen_repaired_with_deterministic_features" if qwen_response_repaired else "qwen_required",
    }
    (out / "qwen_anchor_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision
