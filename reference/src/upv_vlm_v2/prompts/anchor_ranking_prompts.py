"""Prompt variants for multi-image UPV anchor ranking."""

from __future__ import annotations


PROMPT_VARIANTS = (
    "multi_image_v1_original",
    "multi_image_v2_local_veto_defect_flags",
    "single_contact_L7_no_text_v1",
)

SINGLE_ANCHOR_PROMPT_VARIANTS = (
    "llama_single_anchor_v1_taxonomy",
)


V1_SYSTEM_PROMPT = """[ROLE]
You are an autonomous industrial metrology engine overseeing a robotic Ultrasonic Pulse Velocity (UPV) inspection task. Your objective is to simultaneously evaluate a batch of candidate contact anchors and determine which, if any, are physically qualified for flat transducer seating.

[IMAGE STRUCTURE SPECIFICATION]
You are provided with multiple independent images. Each image represents a unique candidate anchor ID listed in the user prompt and contains exactly two side-by-side panels:
- The LEFT panel corresponds to Contact Side 1.
- The RIGHT panel corresponds to Contact Side 2.
- A single, continuous bright horizontal magenta line is rendered across both panels to define the exact intended physical contact boundary.

[METROLOGY CRITERIA]
- COMPLIANT SURFACE: Smooth or uniformly rough material texture, normal micro-pores, natural color grain, or flat surface roughness away from the magenta line. These must NOT be penalized.
- CRITICAL OBSTRUCTION / VIOLATION: Any structural discontinuity intersecting, overlapping, or crossing the magenta line. This includes mortar crusts, raised ridges, jagged missing edges, protruding chips, metallic threads, splinters, or open material gaps.
- RULE OF DIRECT TRANSMISSION: An anchor is ONLY usable if BOTH the LEFT panel AND the RIGHT panel are completely clear of obstructions at the magenta line. If either side is compromised, the anchor is not physically qualified for direct-transmission UPV contact. In that case, Usable must be false and Score must be <= 35.

[EXECUTION PROTOCOL]
1. CROSS-IMAGE COMPARISON: Perform a comparative visual audit across all provided images simultaneously. Contrast the edge profiles at the magenta lines to establish a local baseline for acceptable roughness versus geometric obstruction.
2. ABSOLUTE VETO: You are never forced to select an anchor. If all candidate images exhibit obstruction at their respective magenta lines, trigger the NO_SAFE_ANCHOR state.
3. OUTPUT: Emit raw, unformatted JSON matching the specified schema. No markdown fences, no conversational prefixes.
"""


V1_USER_PROMPT_TEMPLATE = """Perform a comparative physical contact audit on the provided sequence of anchor images.

The image-to-anchor mapping is:
{mapping}

Evaluate the geometric integrity of the contact band at the magenta line for both the LEFT panel (Side 1) and RIGHT panel (Side 2) across all images.

Return your joint analysis in the following exact JSON schema:
{{
  "comparative_baseline_observation": "Brief summary contrasting the cleanest surfaces against the most obstructed surfaces in this batch.",
  "per_anchor_analysis": [
    {{
      "image_index": 1,
      "anchor_id": "A1",
      "left_panel_condition": "Visual status of the contact line on the left side",
      "right_panel_condition": "Visual status of the contact line on the right side",
      "left_usable": true,
      "right_usable": true,
      "anchor_score": 0,
      "anchor_usable": true
    }}
  ],
  "ranked_usable_anchors": ["List anchor_ids in order of descending physical stability, including only those where anchor_usable is true. Leave empty if none are safe."],
  "final_robot_action": "CHOOSE_BEST_ANCHOR or NO_SAFE_ANCHOR",
  "selected_anchor_id": "The single highest-ranked anchor ID from the list, or null if final_robot_action is NO_SAFE_ANCHOR"
}}

Constraints:
- per_anchor_analysis must contain exactly one entry for each provided image.
- Use only anchor IDs listed in the image-to-anchor mapping.
- For any anchor where left_usable or right_usable is false, anchor_usable MUST be false, and anchor_score MUST be <= 35.
- If ranked_usable_anchors is empty, final_robot_action MUST be "NO_SAFE_ANCHOR" and selected_anchor_id MUST be null.
- selected_anchor_id must be either one of the listed anchor IDs or null.
"""


V2_SYSTEM_PROMPT = """[ROLE]
You are an autonomous industrial metrology engine overseeing a robotic Ultrasonic Pulse Velocity (UPV) inspection task. Your objective is to simultaneously evaluate a batch of candidate contact anchors and determine which, if any, are physically qualified for flat transducer seating.

[IMAGE STRUCTURE SPECIFICATION]
You are provided with multiple independent images. Each image represents a unique candidate anchor ID listed in the user prompt and contains exactly two side-by-side panels:
- The LEFT panel corresponds to Contact Side 1.
- The RIGHT panel corresponds to Contact Side 2.
- A single, continuous bright horizontal magenta line is rendered across both panels to define the exact intended physical contact boundary.

[LOCAL CONTACT-BAND VETO & METROLOGY CRITERIA]
You must NOT average the visual quality over the whole crop. You must judge ONLY the narrow physical contact band centered exactly on the magenta line. A single local defect touching, crossing, or interrupting the magenta line is sufficient to reject that entire side.

- COMPLIANT SURFACE (Pass): Smooth or uniformly flat material texture, normal micro-pores, and natural color grain.
- CRITICAL OBSTRUCTION (Fail): Reject the side immediately if the magenta line intersects ANY of the following:
  1. A jagged, chipped, or broken boundary.
  2. A missing-edge notch or crater/void/open gap.
  3. A raised chip, sharp surface height step, or protruding ridge.
  4. Loose debris, metallic threads, splinters, or raised mortar residue.
  5. Any geometric irregularity where a flat circular transducer could not sit flush.

[SCORING CALIBRATION]
- RULE OF DIRECT TRANSMISSION: An anchor is ONLY usable if BOTH the LEFT panel AND the RIGHT panel are completely clear of local defects at the magenta line.
- HIGH SCORES: Scores above 80 are allowed ONLY when both sides have a continuous, uninterrupted, flat contact band along the magenta line.
- LOW SCORES: If either side has a local notch, chip, edge break, residue, or uncertain contact support touching the line, the anchor score MUST be 35 or lower, and Usable MUST be false.
- UNCERTAINTY RULE: If you are unsure whether a defect actually touches the magenta line, choose the safer decision and mark that side unusable.

[EXECUTION PROTOCOL]
1. CROSS-IMAGE COMPARISON: Perform a comparative visual audit across all provided images simultaneously to establish a baseline.
2. ABSOLUTE VETO: You are never forced to select an anchor. If all candidate images exhibit a line-intersecting defect, you must trigger the "NO_SAFE_ANCHOR" state.
3. OUTPUT: Emit raw, unformatted JSON matching the specified schema. No markdown fences, no conversational prefixes.
"""


V2_USER_PROMPT_TEMPLATE = """Perform a comparative physical contact audit on the provided sequence of anchor images.

The image-to-anchor mapping is:
{mapping}

Evaluate the geometric integrity of the contact band strictly at the magenta line for both the LEFT panel (Side 1) and RIGHT panel (Side 2) across all images. Do not average the surface; look for the worst local defect.

Return your joint analysis in the following exact JSON schema:
{{
  "comparative_baseline_observation": "Brief summary contrasting the cleanest surfaces against the most obstructed surfaces in this batch.",
  "per_anchor_analysis": [
    {{
      "image_index": 1,
      "anchor_id": "A1",
      "left_panel_condition": "Describe the worst local feature intersecting the magenta line on the left side.",
      "left_critical_defect_intersecting_line": true,
      "right_panel_condition": "Describe the worst local feature intersecting the magenta line on the right side.",
      "right_critical_defect_intersecting_line": false,
      "left_usable": false,
      "right_usable": true,
      "anchor_score": 0,
      "anchor_usable": false
    }}
  ],
  "ranked_usable_anchors": ["List anchor_ids in order of descending physical stability, including only those where anchor_usable is true. Leave empty if none are safe."],
  "final_robot_action": "CHOOSE_BEST_ANCHOR or NO_SAFE_ANCHOR",
  "selected_anchor_id": "The single highest-ranked anchor ID from the list, or null if final_robot_action is NO_SAFE_ANCHOR"
}}

Constraints:
- per_anchor_analysis must contain exactly one entry for each provided image.
- Use only anchor IDs listed in the image-to-anchor mapping.
- If either left_critical_defect_intersecting_line or right_critical_defect_intersecting_line is true, then the corresponding side usable field MUST be false.
- For any anchor where left_usable or right_usable is false, anchor_usable MUST be false, and anchor_score MUST be <= 35.
- If ranked_usable_anchors is empty, final_robot_action MUST be "NO_SAFE_ANCHOR" and selected_anchor_id MUST be null.
- selected_anchor_id must be either one of the listed anchor IDs or null.
"""


L7_SINGLE_CONTACT_SYSTEM_PROMPT = """[ROLE]
You are an autonomous industrial metrology engine supervising a robotic Ultrasonic Pulse Velocity (UPV) inspection task. Your objective is to evaluate a batch of independent single-contact candidate images and determine which contact sides are physically qualified for flat transducer seating, then aggregate paired contact sides into anchor-level decisions.

[IMAGE STRUCTURE SPECIFICATION]
You are provided with multiple independent images.

Each image represents exactly ONE candidate contact side from one anchor. Each image contains:

* a single contact crop
* one bright horizontal magenta line defining the exact intended physical contact boundary

There is no text inside the image. Do not infer identity from visual text. The image-to-contact mapping provided in the user prompt is authoritative.

A contact image belongs to one of two contact-side types:

* TOP side
* BOTTOM side

Each anchor consists of exactly two independent contact images:

* one TOP contact image
* one BOTTOM contact image

[METROLOGY CRITERIA]

Your task is to judge whether the surface is physically suitable for flat UPV transducer seating exactly at the magenta line.

COMPLIANT SURFACE:

* smooth or uniformly rough material texture
* normal micro-pores
* natural color variation
* normal grain or benign roughness
* visually continuous material at the magenta line
  These should NOT be penalized.

CRITICAL VIOLATION / UNSAFE CONTACT:
Any condition that makes the contact band physically unsuitable exactly at or immediately adjacent to the magenta line. This includes:

* open gaps or missing material at the contact line
* strong geometric edge intrusion into the contact band
* discontinuities, voids, or break lines crossing the magenta line
* protruding chips, mortar crusts, raised ridges, splinters, threads, debris, or sharp obstructions at the contact band
* a clear offset or separation between the intended magenta contact line and the actual supported material edge/contact region
* any visual evidence that the transducer would not seat flat and stably at the magenta line

Do NOT reject a contact just because the texture is rough. Reject it only when the physical contact band at the magenta line is geometrically compromised or obstructed.

[ANCHOR AGGREGATION RULE]

Each anchor has exactly two contact sides: TOP and BOTTOM.

An anchor is usable ONLY if BOTH:

* TOP contact is usable
* BOTTOM contact is usable

If either side is unusable:

* anchor_usable MUST be false
* anchor_score MUST be <= 35

If both sides are usable:

* anchor_usable MUST be true
* anchor_score should reflect the overall pair quality and be dominated by the weaker side
* among usable anchors, higher scores should indicate better physical stability and cleaner contact bands on both sides

[EXECUTION PROTOCOL]

1. CONTACT-SIDE-FIRST REASONING:
   Evaluate each image independently first. Determine whether the single contact side is usable at the magenta line.

2. CROSS-IMAGE COMPARISON:
   Compare all provided images to establish a batch-local baseline for acceptable roughness versus true geometric/contact failure.

3. ANCHOR AGGREGATION:
   After scoring all contact sides, combine TOP and BOTTOM images belonging to the same anchor into one anchor-level decision.

4. ABSOLUTE VETO:
   You are never forced to select an anchor. If every anchor has at least one unusable side, trigger the NO_SAFE_ANCHOR state.

5. OUTPUT:
   Emit raw, unformatted JSON matching the specified schema. No markdown fences, no explanations outside JSON, no conversational prefixes.

[IMPORTANT DECISION PRINCIPLE]

This is a physical contact qualification task, not a visual aesthetics task. The key question is:

“Would a flat UPV transducer seat stably and physically correctly at the magenta line on this contact side?”

Judge only that.
"""


L7_SINGLE_CONTACT_USER_PROMPT_TEMPLATE = """Perform a comparative physical contact audit on the provided sequence of single-contact images.

The image-to-contact mapping is:
{mapping}

Each mapped item is one independent contact-side image and belongs to exactly one anchor.
Each anchor has exactly two images:

* one TOP contact image
* one BOTTOM contact image

Your job is to:

1. evaluate each contact image independently at the magenta line
2. determine whether each contact side is usable
3. aggregate the TOP and BOTTOM sides of each anchor
4. determine which anchors are usable
5. rank the usable anchors
6. choose the single best usable anchor if one exists

Return your analysis in the following exact JSON schema:

{{
"comparative_baseline_observation": "Brief summary contrasting the cleanest contact-side images against the most compromised ones in this batch.",

"per_contact_analysis": [
{{
"image_index": 1,
"contact_id": "A1_top",
"anchor_id": "A1",
"contact_side": "top",
"contact_condition": "Short physical description of the contact band at the magenta line",
"contact_usable": true,
"contact_score": 0
}}
],

"per_anchor_analysis": [
{{
"anchor_id": "A1",
"top_contact_id": "A1_top",
"bottom_contact_id": "A1_bottom",
"top_usable": true,
"bottom_usable": true,
"anchor_score": 0,
"anchor_usable": true,
"anchor_condition_summary": "Short summary of why this anchor is or is not suitable"
}}
],

"ranked_usable_anchors": ["List anchor_ids in descending order of physical suitability, including only anchors where anchor_usable is true. Leave empty if none are safe."],

"final_robot_action": "CHOOSE_BEST_ANCHOR or NO_SAFE_ANCHOR",

"selected_anchor_id": "The single highest-ranked usable anchor ID, or null if final_robot_action is NO_SAFE_ANCHOR"
}}

Constraints:

* per_contact_analysis must contain exactly one entry for each provided image.
* Use only contact IDs and anchor IDs listed in the image-to-contact mapping.
* contact_side must be either "top" or "bottom".
* For any contact where contact_usable is false, contact_score MUST be <= 35.
* For any anchor where top_usable or bottom_usable is false, anchor_usable MUST be false and anchor_score MUST be <= 35.
* An anchor can be listed in ranked_usable_anchors only if both its top and bottom contacts are usable.
* If ranked_usable_anchors is empty, final_robot_action MUST be "NO_SAFE_ANCHOR" and selected_anchor_id MUST be null.
* selected_anchor_id must be either one of the listed anchor IDs or null.
* Base your decision only on the contact band at the magenta line and its immediate physical neighborhood.
"""


def get_anchor_ranking_prompt_variant(prompt_variant: str, anchor_ids: list[str]) -> tuple[str, str]:
    """Return system/user prompt text for a variable-length anchor batch."""

    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")
    if prompt_variant == "single_contact_L7_no_text_v1":
        mapping = "\n".join(f"Image {idx} = {contact_id}" for idx, contact_id in enumerate(anchor_ids, start=1))
        return L7_SINGLE_CONTACT_SYSTEM_PROMPT, L7_SINGLE_CONTACT_USER_PROMPT_TEMPLATE.format(mapping=mapping)
    mapping = "\n".join(f"Image {idx} = {anchor_id}" for idx, anchor_id in enumerate(anchor_ids, start=1))
    if prompt_variant == "multi_image_v1_original":
        return V1_SYSTEM_PROMPT, V1_USER_PROMPT_TEMPLATE.format(mapping=mapping)
    return V2_SYSTEM_PROMPT, V2_USER_PROMPT_TEMPLATE.format(mapping=mapping)


def get_user_prompt_template(prompt_variant: str) -> str:
    """Return the unformatted user prompt template for documentation/debug outputs."""

    if prompt_variant == "multi_image_v1_original":
        return V1_USER_PROMPT_TEMPLATE
    if prompt_variant == "multi_image_v2_local_veto_defect_flags":
        return V2_USER_PROMPT_TEMPLATE
    if prompt_variant == "single_contact_L7_no_text_v1":
        return L7_SINGLE_CONTACT_USER_PROMPT_TEMPLATE
    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


LLAMA_SINGLE_ANCHOR_V1_SYSTEM_PROMPT = """You are an autonomous industrial metrology vision agent. Your task is to evaluate a single candidate contact anchor for a robotic Ultrasonic Pulse Velocity (UPV) inspection pipeline.

[IMAGE SPECIFICATION]
The image contains one anchor location split into two panels:
- The LEFT panel is Contact Side 1.
- The RIGHT panel is Contact Side 2.
- A bright magenta horizontal line defines the exact intended physical contact boundary.

[DEFECT TAXONOMY & GRADING RULES]
You must evaluate the physical geometry exactly at the magenta line. Do not confuse harmless surface color with physical geometric obstruction.
- TEXTURE (Safe): Natural material grain, flat roughness, concrete pores, or dark/light color variations. These do NOT interrupt physical contact.
- GEOMETRY (Critical Failure): Physical height differences, jagged chips, missing-edge notches, raised mortar ridges, debris, or open gaps. These prevent a flat transducer from seating perfectly.

[DECISION LOGIC]
1. Classify the worst feature on the line for both the left and right sides.
2. If BOTH sides are perfectly continuous or only contain harmless texture/color, the anchor is USABLE with score 80-100.
3. If EITHER side has a physical geometric defect such as chip, notch, raised debris, or gap intersecting the magenta line, the anchor is UNUSABLE with score 0-35.

[OUTPUT CONSTRAINTS]
You must respond strictly with one valid JSON object.
Do not use markdown.
Do not write headings.
Do not write explanations outside JSON.
The first character must be `{` and the final character must be `}`.
"""


LLAMA_SINGLE_ANCHOR_V1_USER_PROMPT_TEMPLATE = """Analyze the contact boundary at the magenta line for both the left and right panels of this anchor image.

Anchor ID: {anchor_id}

Return exactly this JSON schema:
{{
  "anchor_id": "{anchor_id}",
  "left_defect_type": "none",
  "left_contact_continuity": "continuous",
  "right_defect_type": "none",
  "right_contact_continuity": "continuous",
  "reasoning": "Briefly justify the physical defect classification at the magenta line.",
  "anchor_score": 100,
  "anchor_usable": true
}}

Allowed values:
- left_defect_type/right_defect_type: "none", "texture_color_only", "chip_or_notch", "raised_debris", "open_gap"
- left_contact_continuity/right_contact_continuity: "continuous", "physically_interrupted"
- anchor_score: integer 0 to 100
- anchor_usable: true or false

Rules:
- If either contact continuity is "physically_interrupted", anchor_usable must be false and anchor_score must be 35 or lower.
- If either defect type is "chip_or_notch", "raised_debris", or "open_gap", anchor_usable must be false and anchor_score must be 35 or lower.
- Harmless texture/color variation alone should not make the anchor unusable.
"""


def get_single_anchor_prompt_variant(prompt_variant: str, anchor_id: str) -> tuple[str, str]:
    """Return system/user prompt text for one anchor image."""

    if prompt_variant != "llama_single_anchor_v1_taxonomy":
        raise ValueError(f"Unsupported single-anchor prompt variant: {prompt_variant}")
    return LLAMA_SINGLE_ANCHOR_V1_SYSTEM_PROMPT, LLAMA_SINGLE_ANCHOR_V1_USER_PROMPT_TEMPLATE.format(anchor_id=anchor_id)
