DEFAULT_PROMPT_VERSION = "v1_baseline"
CURRENT_DEFAULT_PROMPT_VERSION = DEFAULT_PROMPT_VERSION


VLM_PROMPT_V1_BASELINE = """
You are evaluating candidate anchor locations for planar probe contact on a rectangular object such as a brick, wooden block, cinder block, or similar object.

The image contains a grid of candidate anchor tiles.

Each tile is one anchor candidate, labeled A1, A2, A3, etc.
Inside each tile:
- the TOP patch shows one edge region of the object
- the BOTTOM patch shows the opposite edge region of the same object

The TOP and BOTTOM patches inside the same tile must be judged together.

Goal:
Choose the best anchor for a flat planar probe to contact the object's two opposite side surfaces.

Judge each anchor relative to the other anchors in this image.
Even if all candidates are imperfect, choose the best available candidate.

For each anchor, score from 0 to 100 based on:
1. Edge straightness: both visible edges should be as straight and continuous as possible.
2. Edge/contact cleanliness: no obvious nails, mortar, debris, glue, splinters, protrusions, or foreign objects near the edge.
3. Damage/irregularity: penalize chipped, jagged, cracked, broken, or highly uneven edge regions.
4. Pair consistency: both TOP and BOTTOM patches should be good. If one side is poor, reduce the overall score.
5. Planar probe suitability: a flat probe should plausibly make stable contact with both opposite side surfaces.

Important:
- This is a relative comparison task.
- Do not judge against a perfect manufactured object.
- Use visual evidence only.
- Do not over-report defects. If uncertain, write "possible" rather than definite.
- The number of anchors may vary. Score every visible anchor.
- Do not assume A5 or the rightmost tile is best. Judge only by visible edge quality.

Return ONLY valid JSON:

{
  "anchors": [
    {
      "id": "A1",
      "top_score": 0,
      "bottom_score": 0,
      "overall_score": 0,
      "issues": ["short issue list"],
      "reason": "short comparative reason"
    }
  ],
  "ranking_best_to_worst": ["A1", "A2"],
  "best_anchor": "A1",
  "best_anchor_reason": "short reason"
}

Rules:
- Include every visible anchor exactly once.
- Sort anchors from highest overall_score to lowest.
- ranking_best_to_worst must match the same order.
- best_anchor must be the highest-scoring anchor.
- Do not include markdown fences.
- Do not include text outside the JSON.
""".strip()


VLM_PROMPT_V2_CONTACT_FLAGS = """
You are evaluating candidate anchor locations for planar probe contact on a rectangular object such as a brick, wooden block, cinder block, or similar object.

The image contains a grid of candidate anchor tiles labeled A1, A2, A3, etc.
Inside each tile:
- the TOP patch shows one edge/contact region of the object
- the BOTTOM patch shows the opposite edge/contact region of the same object

Judge TOP and BOTTOM together for each anchor. A planar probe needs both opposite contact regions to be suitable.

Goal:
Choose the best available anchor for a flat planar probe to contact the object's two opposite side surfaces.

This is a relative comparison task: even if all anchors are imperfect, choose the best available one. However, hard contact-failure evidence must strongly penalize suitability.

For every visible anchor, evaluate:
1. Edge straightness: both visible contact edges should be straight and continuous.
2. Edge cleanliness: contact regions should be free of attached material, debris, foreign objects, or protrusions.
3. Broken/jagged/chipped edge: chipped, broken, jagged, cracked, or highly uneven contact regions reduce suitability.
4. Mortar or attached material: mortar, glue, cement, clumps, or adhered material at or near the contact edge can block planar contact and should be treated as a severe problem.
5. Nail / foreign object / protrusion / debris: any nail, wire, splinter, stone, raised bump, or protruding debris near the contact edge can block planar contact and should be treated as a severe problem.
6. Planar contact blocked: if a flat probe would hit attached material, a protrusion, a clump, or a highly uneven broken region before contacting a clean planar side surface, mark that side as blocked.

Hard-contact rule:
- If either TOP or BOTTOM has planar_contact_blocked=1, the overall score should usually be low.
- If either side has mortar/attached material or a protrusion at the contact edge, strongly penalize that anchor even if the visible edge line looks straight.
- Prefer anchors where both TOP and BOTTOM have planar_contact_blocked=0 and no attached material/protrusion flags.

Return ONLY valid JSON:

{
  "anchors": [
    {
      "id": "A1",
      "top": {
        "score": 0,
        "has_mortar_or_attached_material": 0,
        "has_nail_or_foreign_object": 0,
        "has_broken_or_jagged_edge": 0,
        "planar_contact_blocked": 0,
        "issues": ["short issue list"]
      },
      "bottom": {
        "score": 0,
        "has_mortar_or_attached_material": 0,
        "has_nail_or_foreign_object": 0,
        "has_broken_or_jagged_edge": 0,
        "planar_contact_blocked": 0,
        "issues": ["short issue list"]
      },
      "overall_score": 0,
      "hard_contact_veto": 0,
      "reason": "short comparative reason"
    }
  ],
  "ranking_best_to_worst": ["A1", "A2"],
  "best_anchor": "A1",
  "best_anchor_reason": "short reason"
}

Rules:
- Include every visible anchor exactly once.
- Use only 0 or 1 for binary flags.
- Set hard_contact_veto=1 when either side has planar_contact_blocked=1 or severe attached material/protrusion blocking contact.
- Sort anchors from highest overall_score to lowest.
- ranking_best_to_worst must match the same order.
- best_anchor must be the highest-scoring anchor.
- Do not include markdown fences.
- Do not include text outside the JSON.
""".strip()


VLM_PROMPT_V3_CONTACT_FLAGS_FLAT = """
Evaluate every visible anchor tile for flat planar probe contact.

Each tile A1, A2, etc. has a TOP edge patch and a BOTTOM opposite edge patch. Judge both together.

Prefer anchors with straight, clean, continuous edges on both sides. Strongly penalize mortar, attached material, protrusions, nails, debris, broken/jagged/chipped edges, or anything blocking flat contact.

Rules:
- Binary flags must be 0 or 1.
- If TOP or BOTTOM has mortar/attached material/protrusion blocking flat contact, set that side blocked and give a low overall_score.
- If top_blocked=1 or bottom_blocked=1, that anchor should not win unless all candidates are worse.
- Even if all anchors are imperfect, choose the best available anchor.
- Reasons must be short.
- Return JSON only. No markdown fences. No text outside JSON.

Use this flat schema:
{
  "anchors": [
    {
      "id": "A1",
      "top_blocked": 0,
      "bottom_blocked": 0,
      "has_mortar_or_attached_material": 0,
      "has_nail_or_foreign_object": 0,
      "has_broken_or_jagged_edge": 0,
      "overall_score": 75,
      "reason": "short reason"
    }
  ],
  "ranking_best_to_worst": ["A1", "A2"],
  "best_anchor": "A1",
  "best_anchor_reason": "short reason"
}
""".strip()


VLM_PROMPT_V4_MORTAR_CONTACT_VETO = """
You are evaluating candidate anchor locations for planar probe contact on a brick or masonry-like rectangular object.

The image contains multiple candidate anchor tiles labeled A1, A2, A3, etc.

Each tile is one anchor candidate.
Inside each tile:
- TOP patch = one visible top-view edge region
- BOTTOM patch = the opposite visible top-view edge region

The TOP and BOTTOM patches in the same tile must be judged together.

Physical task:
A flat planar probe will later press against the object's side surfaces at these two opposite edge regions.
The probe needs clean, unobstructed, approximately planar side-contact regions.

Critical visual rule:
In these images, gray/white cement-like material, mortar, attached paste, bulges, blobs, protrusions, or crust stuck to the edge are BAD.
This is especially important in the BOTTOM patches.
If gray/white material protrudes from, hangs below, sticks to, or interrupts the edge boundary, it should be treated as mortar/attached material and as a planar-contact obstruction.

Do NOT reward an anchor just because the edge looks bright, white, or continuous.
White/gray attached material near the edge is usually mortar or contamination, not clean brick surface.

Strong penalty / veto rules:
- If either TOP or BOTTOM has attached mortar/cement-like material at the contact edge, set has_mortar_or_attached_material = 1.
- If either TOP or BOTTOM has a protrusion, blob, crust, nail, debris, or foreign object at the contact edge, set contact_blocked = 1.
- If bottom edge has a large gray/white protruding region, the candidate should receive a low score and should not be selected as best unless all candidates are worse.
- A candidate with one blocked side should rank below candidates whose TOP and BOTTOM edges are both cleaner and more open.
- Prefer clean brick edge regions even if they are not perfect.

What to evaluate:
For every visible anchor, evaluate:
1. straightness of TOP and BOTTOM edge regions
2. whether the contact edge is clean and unobstructed
3. whether mortar/cement-like material is attached near the contact edge
4. whether there are protrusions, blobs, nails, debris, chips, broken/jagged edge, or other non-planar obstructions
5. whether both opposite edges together are suitable for planar probe contact

Scoring:
Use overall_score from 0 to 100.
- 0 to 20 = unusable or blocked contact
- 21 to 40 = poor
- 41 to 60 = moderate
- 61 to 80 = good
- 81 to 100 = very good

Be comparative:
Even if all candidates are imperfect, choose the best available candidate.
Do not assume the leftmost, rightmost, A1, or A5 is best.
Judge only visible edge/contact quality.

Return ONLY valid JSON, no markdown fences, no extra text.

Use exactly this flat schema:

{
  "anchors": [
    {
      "id": "A1",
      "top_blocked": 0,
      "bottom_blocked": 0,
      "has_mortar_or_attached_material": 0,
      "has_nail_debris_or_foreign_object": 0,
      "has_broken_or_jagged_edge": 0,
      "overall_score": 75,
      "reason": "short reason"
    }
  ],
  "ranking_best_to_worst": ["A1", "A2"],
  "best_anchor": "A1",
  "best_anchor_reason": "short reason"
}

Field rules:
- top_blocked, bottom_blocked, has_mortar_or_attached_material, has_nail_debris_or_foreign_object, and has_broken_or_jagged_edge must be 0 or 1.
- overall_score must be an integer from 0 to 100.
- Include every visible anchor exactly once.
- Sort anchors from highest overall_score to lowest overall_score.
- ranking_best_to_worst must match the same order.
- best_anchor must be the highest-scoring anchor.
- Keep reasons short.
- If mortar/attached material is visible, mention it in the reason.
""".strip()


VLM_PROMPTS = {
    "v1_baseline": VLM_PROMPT_V1_BASELINE,
    "v2_contact_flags": VLM_PROMPT_V2_CONTACT_FLAGS,
    "v3_contact_flags_flat": VLM_PROMPT_V3_CONTACT_FLAGS_FLAT,
    "v4_mortar_contact_veto": VLM_PROMPT_V4_MORTAR_CONTACT_VETO,
}


def normalize_prompt_version(prompt_version: str | None) -> str:
    if prompt_version in (None, "", "current_default"):
        return CURRENT_DEFAULT_PROMPT_VERSION
    return prompt_version


def get_prompt(prompt_version: str | None) -> tuple[str, str]:
    resolved_version = normalize_prompt_version(prompt_version)
    try:
        return resolved_version, VLM_PROMPTS[resolved_version]
    except KeyError as exc:
        valid_versions = ", ".join(list_prompt_versions())
        raise ValueError(f"Unsupported prompt version {prompt_version!r}. Valid versions: {valid_versions}") from exc


def list_prompt_versions() -> list[str]:
    return sorted([*VLM_PROMPTS.keys(), "current_default"])
