"""Adaptive offline Qwen anchor prompt + image-layout search.

This is the closed-loop search the Codex iterative trial was not. Each
iteration's hypothesis, prompt, and layout are explicit. After each iteration
we count per-anchor false-good and false-reject rates against E3 manual labels
plus the E45 visual convention (A2/A3 = Bad, A1/A4 = Good) and pick the next
iteration based on the dominant failure mode.

Safety: only reads saved PNGs/JSONs, only POSTs to ``http://127.0.0.1:8899/infer``
on the already-running local Qwen server. Never touches robot, live camera,
RTDE, Arduino, clamp, or Pundit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from upv_vlm_v2.experiments.prepare_qwen_anchor_layout_variants import (
    LAYOUTS,
    render_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

E3_SCORING_SESSION = Path("outputs/v2_experiments/e3_qwen32_single_anchor_scoring/session_20260522_134303")
E3_LABEL_SESSION = Path("outputs/v2_experiments/e3_anchor_selection/session_20260522_131552")
E45_SESSIONS = [
    Path("outputs/v2_pipeline_runs/session_20260528_035248"),
    Path("outputs/v2_pipeline_runs/session_20260528_035433"),
    Path("outputs/v2_pipeline_runs/session_20260528_035620"),
    Path("outputs/v2_pipeline_runs/session_20260528_035801"),
    Path("outputs/v2_pipeline_runs/session_20260528_035949"),
    Path("outputs/v2_pipeline_runs/session_20260528_041815"),
    Path("outputs/v2_pipeline_runs/session_20260528_042004"),
    Path("outputs/v2_pipeline_runs/session_20260528_042147"),
]

STRICT_GATE = {
    "parse_ok": True,
    "overall_usable": True,
    "top_usable": True,
    "bottom_usable": True,
    "min_score": 70.0,
}


@dataclass(frozen=True)
class AnchorImage:
    dataset: str
    case_id: str
    anchor_id: str
    source_image_path: Path
    geometry_debug_path: Path
    manual_label_raw: str
    manual_is_bad: bool
    manual_is_good: bool


@dataclass
class IterationPlan:
    index: int
    name: str
    hypothesis: str
    prompt_name: str
    prompt_template: str
    layout: str


# ---------- I/O helpers ---------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=float(timeout_sec)) as response:  # noqa: S310 - local server
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Qwen HTTPError {exc.code}: {body}") from exc
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "raw_http_text": text}


def _check_health(server_url: str) -> dict[str, Any]:
    req = Request(server_url.rstrip("/") + "/health", method="GET")
    with urlopen(req, timeout=10.0) as response:  # noqa: S310 - local server
        return json.loads(response.read().decode("utf-8", errors="replace"))


# ---------- Manual label loading ------------------------------------------ #


def load_e3_manual_labels() -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_csv(E3_LABEL_SESSION / "manual_labels" / "manual_anchor_labels.csv")
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        case_id = (row.get("case_id") or "").strip()
        anchor_id = (row.get("anchor_id") or "").strip()
        if not case_id or not anchor_id:
            continue
        label = (row.get("manual_label") or "").strip()
        is_usable = (row.get("is_usable") or "").strip().lower() in {"true", "1", "yes", "y", "good", "acceptable"}
        labels[(case_id, anchor_id)] = {
            "manual_label": label,
            "manual_is_good": is_usable and label.lower() != "bad",
            "manual_is_bad": not is_usable or label.lower() == "bad",
            "is_manual_best": (row.get("is_manual_best") or "").strip().lower() in {"true", "1", "yes"},
        }
    return labels


def _e45_manual_label(anchor_id: str) -> tuple[str, bool, bool]:
    if anchor_id in {"A2", "A3"}:
        return "Bad_user_visual", True, False
    if anchor_id in {"A1", "A4"}:
        return "Good_or_candidate_user_visual", False, True
    return "", False, False


# ---------- Dataset discovery --------------------------------------------- #


def _e3_case_id_short(full_case_dir_name: str) -> str:
    # e.g. case_003_brick_03_chipped_jagged -> brick_03_chipped_jagged
    return re.sub(r"^case_\d+_", "", full_case_dir_name)


def discover_anchor_images() -> list[AnchorImage]:
    images: list[AnchorImage] = []
    e3_labels = load_e3_manual_labels()

    for case_dir in sorted((E3_SCORING_SESSION / "cases").glob("case_*")):
        clean_dir = case_dir / "single_anchor_inputs"
        if not clean_dir.exists():
            continue
        case_id_short = _e3_case_id_short(case_dir.name)
        # geometry debug file is in the OTHER session
        geom_path = E3_LABEL_SESSION / "cases" / case_dir.name / "anchor_crop_geometry_debug.json"
        for image in sorted(clean_dir.glob("source_A*.png")):
            aid = image.stem.replace("source_", "")
            lab = e3_labels.get((case_id_short, aid), {})
            images.append(
                AnchorImage(
                    dataset="E3",
                    case_id=case_dir.name,
                    anchor_id=aid,
                    source_image_path=image,
                    geometry_debug_path=geom_path,
                    manual_label_raw=lab.get("manual_label", ""),
                    manual_is_bad=bool(lab.get("manual_is_bad", False)),
                    manual_is_good=bool(lab.get("manual_is_good", False)),
                )
            )

    for session in E45_SESSIONS:
        clean_dir = session / "artifacts/04_anchor_selection/clean_single_anchor_inputs"
        if not clean_dir.exists():
            continue
        geom_path = session / "artifacts/04_anchor_selection/anchor_crop_geometry_debug.json"
        for image in sorted(clean_dir.glob("source_A*.png")):
            aid = image.stem.replace("source_", "")
            label, is_bad, is_good = _e45_manual_label(aid)
            images.append(
                AnchorImage(
                    dataset="E45",
                    case_id=session.name,
                    anchor_id=aid,
                    source_image_path=image,
                    geometry_debug_path=geom_path,
                    manual_label_raw=label,
                    manual_is_bad=is_bad,
                    manual_is_good=is_good,
                )
            )

    return images


# ---------- Layout rendering ---------------------------------------------- #


def _load_geometry_for(anchor: AnchorImage) -> dict[str, Any]:
    if not anchor.geometry_debug_path.exists():
        return {}
    try:
        d = json.loads(anchor.geometry_debug_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d.get("anchors", {}).get(anchor.anchor_id, {})


def materialise_layout_inputs(
    anchors: list[AnchorImage],
    layout: str,
    cache_root: Path,
) -> dict[tuple[str, str, str], Path]:
    """Render the layout variant for every anchor; return mapping to image path.

    Cached on disk under cache_root/{layout}/{dataset}/{case_id}/{anchor_id}.png.
    L0 returns the original source path without copying.
    """

    mapping: dict[tuple[str, str, str], Path] = {}
    if layout == "L0_baseline":
        for a in anchors:
            mapping[(a.dataset, a.case_id, a.anchor_id)] = a.source_image_path
        return mapping

    for a in anchors:
        out_path = cache_root / layout / a.dataset / a.case_id / f"{a.anchor_id}.png"
        if out_path.exists():
            mapping[(a.dataset, a.case_id, a.anchor_id)] = out_path
            continue
        geom = _load_geometry_for(a)
        if not geom:
            # geometry missing -> fall back to baseline image for this anchor
            mapping[(a.dataset, a.case_id, a.anchor_id)] = a.source_image_path
            continue
        try:
            render_layout(layout, source_path=a.source_image_path, out_path=out_path, geometry=geom)
            mapping[(a.dataset, a.case_id, a.anchor_id)] = out_path
        except Exception:
            mapping[(a.dataset, a.case_id, a.anchor_id)] = a.source_image_path
    return mapping


# ---------- Response parsing ---------------------------------------------- #


def _extract_json_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            if isinstance(obj, dict):
                return obj, ""
        except Exception as exc:
            return None, str(exc)
    return None, "no JSON object found"


def _extract_response(response: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    raw = ""
    for key in ("text", "raw_text", "response", "output", "answer", "generated_text"):
        if isinstance(response.get(key), str):
            raw = str(response[key])
            break
    if not raw:
        raw = json.dumps(response, indent=2)
    parsed, err = _extract_json_object(raw)
    return raw, parsed, err


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "usable", "good", "acceptable", "pass"}:
            return True
        if s in {"false", "0", "no", "n", "bad", "unusable", "fail"}:
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["none"]
    if value is None or value == "":
        return ["none"]
    return [str(value).strip()]


def _normalize(parsed: dict[str, Any] | None, anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(parsed, dict):
        return {"anchor_id": anchor_id, "parse_ok": False}, False, "missing parsed JSON object"
    pid = str(parsed.get("anchor_id") or anchor_id).strip()
    if pid != anchor_id:
        return (
            {"anchor_id": pid, "expected_anchor_id": anchor_id, "parse_ok": False, "raw_parsed": parsed},
            False,
            f"anchor mismatch expected {anchor_id}, got {pid}",
        )
    score = _safe_float(parsed.get("score") or parsed.get("overall_score") or parsed.get("contact_score"))
    if score is None:
        return ({"anchor_id": anchor_id, "parse_ok": False, "raw_parsed": parsed}, False, "missing score")
    score = max(0.0, min(100.0, score))
    top = _as_bool(parsed.get("top_usable") or parsed.get("top_pass"))
    bottom = _as_bool(parsed.get("bottom_usable") or parsed.get("bottom_pass"))
    overall = _as_bool(parsed.get("overall_usable") or parsed.get("overall_pass") or parsed.get("usable"))
    if top is None:
        top = False
    if bottom is None:
        bottom = False
    if overall is None:
        overall = bool(top and bottom and score >= 70.0)
    return (
        {
            "anchor_id": anchor_id,
            "parse_ok": True,
            "score": score,
            "top_usable": bool(top),
            "bottom_usable": bool(bottom),
            "overall_usable": bool(overall),
            "top_defects": _string_list(parsed.get("top_defects") or parsed.get("defects_top")),
            "bottom_defects": _string_list(parsed.get("bottom_defects") or parsed.get("defects_bottom")),
            "reason": str(parsed.get("reason") or "").strip(),
            "raw_parsed": parsed,
        },
        True,
        "",
    )


def _strict_ok(row: dict[str, Any]) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("overall_usable")
        and row.get("top_usable")
        and row.get("bottom_usable")
        and float(row.get("score") or 0.0) >= float(STRICT_GATE["min_score"])
    )


def _anchor_sort_key(anchor_id: str) -> tuple[int, str]:
    match = re.match(r"A(\d+)$", str(anchor_id))
    return (int(match.group(1)), str(anchor_id)) if match else (10**9, str(anchor_id))


# ---------- Prompt templates ---------------------------------------------- #


SCHEMA_BLOCK = (
    "Return JSON only. No prose, no markdown fences. Use exactly this schema and key order:\n"
    "{{\n"
    '  "anchor_id": "{anchor_id}",\n'
    '  "top_usable": true,\n'
    '  "bottom_usable": true,\n'
    '  "overall_usable": true,\n'
    '  "score": 0,\n'
    '  "top_defects": ["none"],\n'
    '  "bottom_defects": ["none"],\n'
    '  "reason": "short physical-contact reason"\n'
    "}}\n"
    "overall_usable MUST equal (top_usable AND bottom_usable). "
    "If top_usable=false OR bottom_usable=false, score MUST be <= 35. "
    "Only return score >= 70 if BOTH sides are usable.\n"
)


PROMPT_CURRENT_CODEX = (
    "You are judging ONE candidate UPV transducer anchor from a saved inspection image.\n\n"
    "The image contains two contact-crop panels:\n"
    "- left/top panel: one side of the material edge\n"
    "- right/bottom panel: the opposite side of the material edge\n\n"
    "The yellow dashed line marks the intended physical contact boundary for a flat circular UPV transducer. "
    "Judge whether BOTH contact regions are physically suitable for stable flat transducer contact.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Use a strict but local physical-contact standard.\n\n"
    "Inspect a contact band around the yellow dashed line in both panels. "
    "Ignore harmless material color, grain, pores, or rough texture that is not at the contact boundary. "
    "Reject any physical obstruction or shape defect at the contact boundary: mortar/crust, raised blob, "
    "debris, protruding chip, missing/jagged edge, ridge, void, crack opening, splinter, or material step "
    "that prevents a flat circular probe from seating.\n\n"
    "Both panels must be clean enough for a flat UPV transducer. If either side is blocked, jagged, "
    "protruding, or uncertain, set that side false, overall_usable=false, and score <= 30. Only assign "
    "score >= 70 when both side booleans are true.\n"
)


PROMPT_TIGHT_CONTACT_BAND = (
    "You are inspecting a UPV transducer anchor. The image is a tight zoom of the contact band on each "
    "side of the material edge.\n\n"
    "- The image stacks two narrow strips: TOP strip is one side of the contact line, BOTTOM strip is the "
    "opposite side. Each strip is centred on a bright yellow dashed line that marks the exact contact line "
    "where the flat circular UPV transducer must seat.\n"
    "- The strip is intentionally narrow so you can inspect at line level.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Decision rules for each strip:\n"
    "- A strip is BAD (set side_usable=false) if you can see ANY of these touching, crossing, or within "
    "approximately 10% of the strip height of the dashed line:\n"
    "  * mortar / crust / gray-white attached material (cementitious residue)\n"
    "  * raised blob, protruding chip, lump, ridge, splinter\n"
    "  * missing/chipped edge (notch where material is absent)\n"
    "  * jagged break, void, crack opening at the line\n"
    "  * debris, foreign object, screw, nail\n"
    "- A strip is GOOD if both above and below the dashed line look continuous, planar, and free of the "
    "above. Ordinary color variation, pores, saw marks, brick speckles, and rough but unbroken texture "
    "are NOT defects.\n"
    "- overall_usable is true ONLY if top_usable AND bottom_usable.\n"
    "- Name the actual visible evidence in top_defects/bottom_defects. Do not give a generic reason.\n"
)


PROMPT_DEFECT_CHECKLIST = (
    "You are inspecting a UPV transducer anchor. Two contact strips are shown, separated by a black gap. "
    "Each strip is centred on a yellow dashed line that is the contact line.\n\n"
    "Run this checklist independently for the TOP strip and the BOTTOM strip. For each strip, check if "
    "ANY of these are touching or within ~10% of the strip height from the dashed line. Mark the strip "
    "BAD if at least one is present:\n"
    "  [Q1] mortar / gray-white crust / attached cementitious material\n"
    "  [Q2] raised blob / protruding chip / lump\n"
    "  [Q3] missing/chipped edge (notch where material is absent at the line)\n"
    "  [Q4] jagged break / crack opening / void at the line\n"
    "  [Q5] ridge / step where the surface is not planar\n"
    "  [Q6] debris / foreign object\n\n"
    "Ordinary color variation, pores, saw marks, brick speckles, and bulk roughness AWAY from the dashed "
    "line are NOT defects.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "If at least one of Q1..Q6 fires on a strip, that side_usable=false. overall_usable is true ONLY if "
    "BOTH sides pass all 6 questions. score must be <=35 if either side fails. score >=80 only if both "
    "sides cleanly pass all 6 questions; 70-79 if both sides pass but the strip is borderline.\n"
    "In top_defects and bottom_defects, list each question that fired (e.g. 'Q1: mortar crust', 'Q3: chip "
    "notch at line').\n"
)


PROMPT_SIDE_FIRST = (
    "You are inspecting a UPV transducer anchor. Two contact strips are shown. Each strip has a yellow "
    "dashed line marking the contact line.\n\n"
    "Reason in this order before writing JSON:\n"
    "1. Look ONLY at the TOP strip. Can a flat circular UPV transducer seat against the contact line "
    "WITHOUT any visible mortar crust, raised blob, chip notch, jagged break, ridge, or debris within "
    "~10% of the strip height of the dashed line? If yes -> top_usable=true, else false.\n"
    "2. Now look ONLY at the BOTTOM strip and answer the same question. Set bottom_usable.\n"
    "3. overall_usable = top_usable AND bottom_usable.\n"
    "4. Pick score: <=30 if either side false, 70-85 if both true and clean, 86-100 only if both sides are "
    "obviously clean and continuous with no marginal evidence.\n\n"
    "Ordinary color variation, brick pores, saw marks, and bulk material roughness AWAY from the dashed "
    "line are NOT defects.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Name visible evidence in top_defects/bottom_defects.\n"
)


PROMPT_SKEPTICAL_DEFAULT = (
    "You are inspecting a UPV transducer anchor. Two contact strips are shown, each with a yellow dashed "
    "line at the contact line.\n\n"
    "DEFAULT POSTURE: assume the anchor is NOT usable until you can name a specific clean visible reason "
    "that BOTH sides are usable. Do not give the benefit of the doubt. If you are uncertain about a side, "
    "set that side false.\n\n"
    "A side passes ONLY when:\n"
    "- the dashed line crosses a continuous, planar material edge with no mortar/crust touching it;\n"
    "- no raised blob, chip notch, jagged break, ridge, debris, or splinter touches or comes within ~10% "
    "of the strip height of the dashed line;\n"
    "- the material edge below/above the line is unbroken and approximately straight.\n\n"
    "Ordinary color variation, brick pores, saw marks, and harmless bulk texture AWAY from the dashed "
    "line are NOT defects.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Score policy:\n"
    "- If either side fails the rule, score <= 30.\n"
    "- If both sides pass and you can name the clean evidence, score 70-85.\n"
    "- Reserve 86-100 for cases where BOTH sides have an obviously crisp, uniform contact line.\n"
    "Cite the specific visible evidence (a phrase, not a generic platitude) in top_defects, "
    "bottom_defects, and reason. Do not write 'looks clean' as the only reason.\n"
)


PROMPT_PHYSICAL_SEATING = (
    "You are inspecting a UPV transducer anchor for a 50 mm diameter flat circular transducer. Two contact "
    "strips are shown, each with a yellow dashed line indicating the contact line.\n\n"
    "Physical question: if the flat 50 mm transducer face is pressed against the material AT the dashed "
    "line on each side, would the face make stable, full-contact, non-rocking contact? It must not be "
    "tilted by a raised blob, ridge, chip, or debris. It must not span a gap or missing-edge notch. It "
    "must not be deflected by an attached mortar/crust pile.\n\n"
    "Apply that physical question separately to top and bottom strips. Ordinary color variation, brick "
    "pores, saw marks, and harmless bulk roughness AWAY from the dashed line are NOT defects.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Score policy:\n"
    "- If the flat face would rock, tilt, or hover on EITHER side, score <= 30 and overall_usable=false.\n"
    "- If both sides clearly support stable flat contact, score 70-85.\n"
    "- Reserve 86-100 only for crisp uniform edges with no marginal evidence.\n"
    "Describe the specific visible reason for any false side in top_defects/bottom_defects.\n"
)


PROMPT_COMBO_BEST = (
    "You are inspecting a UPV transducer anchor for a 50 mm flat circular transducer. The image shows two "
    "tight contact strips. Each strip is centred on a yellow dashed line at the contact line. The strip is "
    "narrow on purpose so you can inspect at line level.\n\n"
    "Process:\n"
    "1) DEFAULT POSTURE: assume the anchor is NOT usable until proven otherwise.\n"
    "2) Look at the TOP strip. Run this checklist:\n"
    "   [Q1] mortar / gray-white crust / attached cementitious material touching or near the dashed line\n"
    "   [Q2] raised blob / protruding chip / lump near the dashed line\n"
    "   [Q3] missing/chipped edge notch at or below/above the dashed line\n"
    "   [Q4] jagged break / crack / void at the line\n"
    "   [Q5] ridge / step where the surface would not be planar\n"
    "   [Q6] debris / foreign object near the line\n"
    "   If ANY of Q1-Q6 fires, top_usable=false.\n"
    "3) Repeat for the BOTTOM strip; set bottom_usable.\n"
    "4) Could a 50 mm flat circular transducer face press at the dashed line on each side without rocking, "
    "tilting, hovering, or spanning a notch? If not, that side fails.\n"
    "5) overall_usable = top_usable AND bottom_usable.\n\n"
    "Ordinary color variation, brick pores, saw marks, and harmless bulk roughness AWAY from the dashed "
    "line are NOT defects.\n\n"
    + SCHEMA_BLOCK
    + "\n"
    "Score policy:\n"
    "- Either side false -> score <= 30.\n"
    "- Both sides pass -> score 70-85.\n"
    "- Score 86-100 reserved for crisp uniform contact lines with no marginal evidence.\n"
    "Cite specific visible evidence in top_defects/bottom_defects (e.g. 'Q1 mortar crust at left of line', "
    "'Q3 chipped notch at right'). Generic phrases like 'looks clean' are not acceptable.\n"
)


# ---------- Iteration runner ---------------------------------------------- #


def run_iteration(
    *,
    plan: IterationPlan,
    images: list[AnchorImage],
    layout_image_map: dict[tuple[str, str, str], Path],
    iter_dir: Path,
    infer_url: str,
    timeout_sec: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "prompt.txt").write_text(plan.prompt_template, encoding="utf-8")
    _write_json(
        iter_dir / "prompt_info.json",
        {
            "index": plan.index,
            "name": plan.name,
            "hypothesis": plan.hypothesis,
            "prompt_name": plan.prompt_name,
            "layout": plan.layout,
            "strict_gate": STRICT_GATE,
        },
    )

    per_anchor_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    elapsed_total = 0.0

    for item in images:
        image_path = layout_image_map.get((item.dataset, item.case_id, item.anchor_id), item.source_image_path)
        case_out = iter_dir / item.dataset / item.case_id / item.anchor_id
        case_out.mkdir(parents=True, exist_ok=True)
        rendered_prompt = plan.prompt_template.format(anchor_id=item.anchor_id)
        (case_out / "prompt.txt").write_text(rendered_prompt, encoding="utf-8")
        payload = {
            "image_path": str(image_path.resolve() if hasattr(image_path, "resolve") else image_path),
            "prompt_text": rendered_prompt,
            "prompt_version": f"iter_{plan.index:02d}_{plan.prompt_name}_{plan.layout}",
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "output_path": str(case_out / "raw_response.txt"),
        }
        t0 = time.perf_counter()
        try:
            server_response = _post_json(infer_url, payload, timeout_sec)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            raw_text, parsed, parse_error = _extract_response(server_response)
            normalized, parse_ok, norm_error = _normalize(parsed, item.anchor_id)
            final_error = "" if parse_ok else (norm_error or parse_error)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            server_response = {"ok": False, "error": str(exc)}
            raw_text = ""
            normalized = {"anchor_id": item.anchor_id, "parse_ok": False, "score": "", "overall_usable": False}
            final_error = str(exc)

        elapsed_total += float(elapsed_ms)

        normalized.update(
            {
                "dataset": item.dataset,
                "case_id": item.case_id,
                "manual_label_raw": item.manual_label_raw,
                "manual_is_bad": item.manual_is_bad,
                "manual_is_good": item.manual_is_good,
                "layout": plan.layout,
                "image_path": str(image_path),
                "source_image_path": str(item.source_image_path),
                "elapsed_ms": elapsed_ms,
                "parse_error": final_error,
                "strict_gate_passed": _strict_ok(normalized),
                "raw_response_path": str(case_out / "raw_response.txt"),
                "parsed_response_path": str(case_out / "parsed_response.json"),
                "server_response_path": str(case_out / "server_response.json"),
            }
        )
        (case_out / "raw_response.txt").write_text(raw_text or "", encoding="utf-8")
        _write_json(case_out / "server_response.json", server_response)
        _write_json(case_out / "parsed_response.json", normalized)
        row = dict(normalized)
        row["top_defects"] = "; ".join(str(v) for v in normalized.get("top_defects", []))
        row["bottom_defects"] = "; ".join(str(v) for v in normalized.get("bottom_defects", []))
        row.pop("raw_parsed", None)
        per_anchor_rows.append(row)
        grouped.setdefault((item.dataset, item.case_id), []).append(row)

    anchor_fields = [
        "dataset",
        "case_id",
        "anchor_id",
        "manual_label_raw",
        "manual_is_bad",
        "manual_is_good",
        "layout",
        "parse_ok",
        "strict_gate_passed",
        "score",
        "overall_usable",
        "top_usable",
        "bottom_usable",
        "top_defects",
        "bottom_defects",
        "reason",
        "parse_error",
        "elapsed_ms",
        "image_path",
        "source_image_path",
        "raw_response_path",
        "parsed_response_path",
        "server_response_path",
    ]
    _write_csv(iter_dir / "per_anchor_results.csv", per_anchor_rows, anchor_fields)

    # Per-case decisions
    per_case_rows: list[dict[str, Any]] = []
    for (dataset, case_id), rows in sorted(grouped.items()):
        survivors = sorted(
            (r for r in rows if _strict_ok(r)),
            key=lambda r: (-float(r.get("score") or -math.inf), _anchor_sort_key(str(r.get("anchor_id") or ""))),
        )
        ranked = sorted(
            rows,
            key=lambda r: (-float(r.get("score") or -math.inf), _anchor_sort_key(str(r.get("anchor_id") or ""))),
        )
        selected = survivors[0] if survivors else None
        per_case_rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "anchors_attempted": len(rows),
                "anchors_parsed": sum(1 for r in rows if r.get("parse_ok")),
                "selected_anchor_id": selected.get("anchor_id") if selected else "NO_SAFE_ANCHOR",
                "selected_score": selected.get("score") if selected else "",
                "ranked_anchors": ",".join(str(r.get("anchor_id")) for r in ranked),
                "survivor_count": len(survivors),
            }
        )
    _write_csv(
        iter_dir / "per_case_decisions.csv",
        per_case_rows,
        [
            "dataset",
            "case_id",
            "anchors_attempted",
            "anchors_parsed",
            "selected_anchor_id",
            "selected_score",
            "ranked_anchors",
            "survivor_count",
        ],
    )

    metrics = compute_iteration_metrics(per_anchor_rows, per_case_rows, plan, elapsed_total)
    _write_json(iter_dir / "iteration_metrics.json", metrics)
    write_iteration_failure_review(iter_dir, per_anchor_rows, metrics)
    return metrics


# ---------- Metrics & failure review -------------------------------------- #


def compute_iteration_metrics(
    per_anchor_rows: list[dict[str, Any]],
    per_case_rows: list[dict[str, Any]],
    plan: IterationPlan,
    elapsed_total_ms: float,
) -> dict[str, Any]:
    def _is_high_score(row: dict[str, Any]) -> bool:
        s = row.get("score")
        if s in ("", None):
            return False
        try:
            return float(s) >= float(STRICT_GATE["min_score"])
        except Exception:
            return False

    def _is_strict_pass(row: dict[str, Any]) -> bool:
        return bool(row.get("strict_gate_passed"))

    e3 = [r for r in per_anchor_rows if r.get("dataset") == "E3"]
    e45 = [r for r in per_anchor_rows if r.get("dataset") == "E45"]
    e3_bad = [r for r in e3 if r.get("manual_is_bad")]
    e3_good = [r for r in e3 if r.get("manual_is_good")]
    e45_bad = [r for r in e45 if r.get("manual_is_bad")]
    e45_good = [r for r in e45 if r.get("manual_is_good")]

    def _rate(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom else 0.0

    e3_bad_high_score = sum(1 for r in e3_bad if _is_high_score(r))
    e45_bad_high_score = sum(1 for r in e45_bad if _is_high_score(r))
    bad_high_score = e3_bad_high_score + e45_bad_high_score
    bad_total = len(e3_bad) + len(e45_bad)

    e3_good_rejected = sum(1 for r in e3_good if not _is_strict_pass(r))
    e45_good_rejected = sum(1 for r in e45_good if not _is_strict_pass(r))

    parse_ok = sum(1 for r in per_anchor_rows if r.get("parse_ok"))
    total = len(per_anchor_rows)
    avg_ms = round(elapsed_total_ms / total, 2) if total else 0.0

    no_safe_cases = [r for r in per_case_rows if r.get("selected_anchor_id") == "NO_SAFE_ANCHOR"]

    return {
        "iteration_index": plan.index,
        "iteration_name": plan.name,
        "hypothesis": plan.hypothesis,
        "prompt_name": plan.prompt_name,
        "layout": plan.layout,
        "total_anchors": total,
        "e3_anchors": len(e3),
        "e45_anchors": len(e45),
        "e3_bad_count": len(e3_bad),
        "e3_bad_scored_high": e3_bad_high_score,
        "e3_false_good_rate": _rate(e3_bad_high_score, len(e3_bad)),
        "e3_good_count": len(e3_good),
        "e3_good_rejected": e3_good_rejected,
        "e3_false_reject_rate": _rate(e3_good_rejected, len(e3_good)),
        "e45_bad_count": len(e45_bad),
        "e45_bad_scored_high": e45_bad_high_score,
        "e45_false_good_rate": _rate(e45_bad_high_score, len(e45_bad)),
        "e45_good_count": len(e45_good),
        "e45_good_rejected": e45_good_rejected,
        "e45_false_reject_rate": _rate(e45_good_rejected, len(e45_good)),
        "combined_bad_count": bad_total,
        "combined_bad_scored_high": bad_high_score,
        "combined_false_good_rate": _rate(bad_high_score, bad_total),
        "combined_good_count": len(e3_good) + len(e45_good),
        "combined_good_rejected": e3_good_rejected + e45_good_rejected,
        "combined_false_reject_rate": _rate(e3_good_rejected + e45_good_rejected, len(e3_good) + len(e45_good)),
        "parse_ok_count": parse_ok,
        "parse_success_rate": _rate(parse_ok, total),
        "avg_runtime_ms_per_image": avg_ms,
        "no_safe_anchor_cases": len(no_safe_cases),
        "selected_anchors_per_case": [
            {"dataset": r["dataset"], "case_id": r["case_id"], "selected_anchor_id": r["selected_anchor_id"]}
            for r in per_case_rows
        ],
    }


def write_iteration_failure_review(
    iter_dir: Path,
    per_anchor_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f"# Failure review — {metrics['iteration_name']} (iter {metrics['iteration_index']:02d})")
    lines.append("")
    lines.append(f"Prompt: `{metrics['prompt_name']}`  •  Layout: `{metrics['layout']}`")
    lines.append("")
    lines.append(f"Hypothesis: {metrics['hypothesis']}")
    lines.append("")
    lines.append(
        f"Combined bad-anchor false-good rate: **{metrics['combined_bad_scored_high']}/"
        f"{metrics['combined_bad_count']} = {metrics['combined_false_good_rate']*100:.1f}%**  "
        f"(target < 25%, acceptable < 40%)"
    )
    lines.append("")
    lines.append(
        f"Combined good-anchor false-reject rate: {metrics['combined_good_rejected']}/"
        f"{metrics['combined_good_count']} = {metrics['combined_false_reject_rate']*100:.1f}%"
    )
    lines.append("")

    def _block(title: str, rows: Iterable[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| dataset | case | anchor | manual | score | top | bot | overall | image |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['dataset']} | {r['case_id']} | {r['anchor_id']} | {r.get('manual_label_raw','')} | "
                f"{r.get('score','')} | {r.get('top_usable','')} | {r.get('bottom_usable','')} | "
                f"{r.get('overall_usable','')} | `{r.get('image_path','')}` |"
            )
        lines.append("")

    false_good = [r for r in per_anchor_rows if r.get("manual_is_bad") and _strict_ok(r) is False and float(r.get("score") or 0) >= 70]
    false_good += [r for r in per_anchor_rows if r.get("manual_is_bad") and _strict_ok(r)]
    # dedup
    seen: set[tuple[str, str, str]] = set()
    dedup_fg: list[dict[str, Any]] = []
    for r in false_good:
        k = (r["dataset"], r["case_id"], r["anchor_id"])
        if k in seen:
            continue
        seen.add(k)
        dedup_fg.append(r)
    false_reject = [r for r in per_anchor_rows if r.get("manual_is_good") and not _strict_ok(r)]

    _block(f"False-good (bad anchors scored >= 70) — n={len(dedup_fg)}", dedup_fg)
    _block(f"False-reject (good anchors not passing strict gate) — n={len(false_reject)}", false_reject)
    (iter_dir / "failure_review.md").write_text("\n".join(lines), encoding="utf-8")


# ---------- Adaptive planner ---------------------------------------------- #


def _baseline_plan() -> IterationPlan:
    return IterationPlan(
        index=1,
        name="iter01_baseline_codex_prompt_L0",
        hypothesis=(
            "Reproduce the Codex iteration_05 false-good rate ~83% with the same prompt and L0 baseline "
            "layout. Confirms the runner matches the user's audit numbers."
        ),
        prompt_name="codex_iter05_final_tuned_hybrid",
        prompt_template=PROMPT_CURRENT_CODEX,
        layout="L0_baseline",
    )


def _layout_sweep_plans() -> list[IterationPlan]:
    return [
        IterationPlan(
            index=2,
            name="iter02_layout_L1_contact_band_zoom",
            hypothesis=(
                "Tight contact-band zoom (no text, no large background) gives Qwen the line at sufficient "
                "scale to actually see mortar / chip / blob defects."
            ),
            prompt_name="codex_iter05_final_tuned_hybrid",
            prompt_template=PROMPT_CURRENT_CODEX,
            layout="L1_contact_band_zoom",
        ),
        IterationPlan(
            index=3,
            name="iter03_layout_L4_thick_line_no_text",
            hypothesis=(
                "Removing title/labels and thickening the dashed line keeps the full panel context but "
                "makes the contact line itself visually dominant."
            ),
            prompt_name="codex_iter05_final_tuned_hybrid",
            prompt_template=PROMPT_CURRENT_CODEX,
            layout="L4_thick_line_no_text",
        ),
        IterationPlan(
            index=4,
            name="iter04_layout_L2_full_plus_zoom_2x2",
            hypothesis=(
                "2x2 (full panel + zoom for each side) gives both context and line-level detail; tests "
                "whether the model can integrate both."
            ),
            prompt_name="codex_iter05_final_tuned_hybrid",
            prompt_template=PROMPT_CURRENT_CODEX,
            layout="L2_full_plus_zoom_2x2",
        ),
        IterationPlan(
            index=5,
            name="iter05_layout_L3_red_inspection_band",
            hypothesis=(
                "Translucent red inspection band overlay around the dashed line points the model's "
                "attention to the contact zone without otherwise changing the image."
            ),
            prompt_name="codex_iter05_final_tuned_hybrid",
            prompt_template=PROMPT_CURRENT_CODEX,
            layout="L3_red_inspection_band",
        ),
    ]


PROMPT_TEMPLATES = {
    "codex_iter05_final_tuned_hybrid": PROMPT_CURRENT_CODEX,
    "tight_contact_band_strict": PROMPT_TIGHT_CONTACT_BAND,
    "defect_checklist": PROMPT_DEFECT_CHECKLIST,
    "side_first_two_stage": PROMPT_SIDE_FIRST,
    "skeptical_default": PROMPT_SKEPTICAL_DEFAULT,
    "physical_seating_50mm": PROMPT_PHYSICAL_SEATING,
    "combo_best": PROMPT_COMBO_BEST,
}


def pick_best_layout_from_history(history: list[dict[str, Any]]) -> str:
    """Pick layout that minimises (combined_false_good_rate, combined_false_reject_rate)."""

    if not history:
        return "L0_baseline"
    candidates = sorted(
        history,
        key=lambda m: (
            float(m.get("combined_false_good_rate") or 0.0),
            float(m.get("combined_false_reject_rate") or 0.0),
            -float(m.get("parse_success_rate") or 0.0),
        ),
    )
    return str(candidates[0].get("layout") or "L0_baseline")


def adaptive_prompt_plans(history: list[dict[str, Any]], best_layout: str) -> list[IterationPlan]:
    """Decide the prompt sweep order from the layout sweep failure modes."""

    last_layout_metric = next((m for m in reversed(history) if m.get("layout") == best_layout), None)
    fg = float((last_layout_metric or {}).get("combined_false_good_rate") or 0.0)
    fr = float((last_layout_metric or {}).get("combined_false_reject_rate") or 0.0)

    # Prompt order: lead with the prompt most likely to fix the dominant error.
    if fg >= fr:
        order = [
            "skeptical_default",
            "defect_checklist",
            "side_first_two_stage",
            "combo_best",
        ]
    else:
        order = [
            "tight_contact_band_strict",
            "physical_seating_50mm",
            "side_first_two_stage",
            "combo_best",
        ]

    plans: list[IterationPlan] = []
    for offset, prompt_name in enumerate(order, start=1):
        plans.append(
            IterationPlan(
                index=5 + offset,
                name=f"iter{5 + offset:02d}_{best_layout}_{prompt_name}",
                hypothesis=(
                    f"On best layout `{best_layout}`, swap prompt to `{prompt_name}` to address the "
                    f"dominant failure (false-good={fg*100:.1f}%, false-reject={fr*100:.1f}%)."
                ),
                prompt_name=prompt_name,
                prompt_template=PROMPT_TEMPLATES[prompt_name],
                layout=best_layout,
            )
        )
    return plans


# ---------- Top-level driver ---------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-url", default="http://127.0.0.1:8899")
    ap.add_argument("--output-root", default="outputs/debug_anchor_prompt_eval")
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument(
        "--max-layout-sweep-iters",
        type=int,
        default=4,
        help="How many layout-sweep iterations after the baseline (default 4 = iters 02..05).",
    )
    ap.add_argument(
        "--max-prompt-sweep-iters",
        type=int,
        default=4,
        help="How many prompt-sweep iterations after the layout sweep (default 4 = iters 06..09).",
    )
    ap.add_argument(
        "--time-budget-sec",
        type=int,
        default=2400,
        help="Soft wall-clock budget. Stops adding iterations after this elapses.",
    )
    ap.add_argument(
        "--limit-anchors",
        type=int,
        default=None,
        help="Optional debug-only cap on number of anchor images.",
    )
    args = ap.parse_args()

    server_url = args.server_url.rstrip("/")
    infer_url = server_url + "/infer"
    try:
        health = _check_health(server_url)
    except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Qwen server is not reachable at {server_url}: {exc}") from exc
    if not health.get("ok") or not health.get("model_loaded"):
        raise SystemExit(f"Qwen server is not ready at {server_url}: {json.dumps(health, indent=2)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_root) / f"claude_iterative_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    layout_cache_dir = session_dir / "_layout_cache"
    layout_cache_dir.mkdir(parents=True, exist_ok=True)

    images = discover_anchor_images()
    if args.limit_anchors:
        images = images[: args.limit_anchors]
    if not images:
        raise SystemExit("No anchor images discovered (E3 + E45).")

    _write_json(
        session_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "server_url": server_url,
            "health": health,
            "image_count": len(images),
            "e3_scoring_session": str(E3_SCORING_SESSION),
            "e3_label_session": str(E3_LABEL_SESSION),
            "e45_sessions": [str(p) for p in E45_SESSIONS],
            "strict_gate": STRICT_GATE,
            "hardware_safety": {
                "robot": "not used",
                "live_camera": "not used",
                "rtde": "not used",
                "arduino": "not used",
                "clamp": "not used",
                "pundit": "not used",
            },
        },
    )
    _write_csv(
        session_dir / "input_image_manifest.csv",
        [
            {
                "dataset": a.dataset,
                "case_id": a.case_id,
                "anchor_id": a.anchor_id,
                "manual_label_raw": a.manual_label_raw,
                "manual_is_bad": a.manual_is_bad,
                "manual_is_good": a.manual_is_good,
                "source_image_path": str(a.source_image_path),
                "geometry_debug_path": str(a.geometry_debug_path),
            }
            for a in images
        ],
        [
            "dataset",
            "case_id",
            "anchor_id",
            "manual_label_raw",
            "manual_is_bad",
            "manual_is_good",
            "source_image_path",
            "geometry_debug_path",
        ],
    )

    history: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    def _time_left() -> bool:
        return (time.perf_counter() - t_start) < float(args.time_budget_sec)

    def _run(plan: IterationPlan) -> dict[str, Any]:
        print(f"[ITER {plan.index:02d}] {plan.name} prompt={plan.prompt_name} layout={plan.layout}")
        layout_map = materialise_layout_inputs(images, plan.layout, layout_cache_dir)
        iter_dir = session_dir / f"iteration_{plan.index:02d}_{plan.layout}_{plan.prompt_name}"
        metrics = run_iteration(
            plan=plan,
            images=images,
            layout_image_map=layout_map,
            iter_dir=iter_dir,
            infer_url=infer_url,
            timeout_sec=int(args.timeout_sec),
            max_new_tokens=int(args.max_new_tokens),
        )
        metrics["iter_dir"] = str(iter_dir)
        history.append(metrics)
        iteration_summaries.append(metrics)
        print(
            f"[ITER {plan.index:02d} done] "
            f"combined_false_good={metrics['combined_false_good_rate']*100:.1f}% "
            f"combined_false_reject={metrics['combined_false_reject_rate']*100:.1f}% "
            f"parse={metrics['parse_success_rate']*100:.1f}% "
            f"avg_ms={metrics['avg_runtime_ms_per_image']:.0f}"
        )
        return metrics

    # Iter 1 baseline
    _run(_baseline_plan())

    # Iters 2..5 layout sweep
    for plan in _layout_sweep_plans()[: args.max_layout_sweep_iters]:
        if not _time_left():
            print("[time budget reached during layout sweep]")
            break
        _run(plan)

    # Iters 6..9 prompt sweep on best layout
    best_layout = pick_best_layout_from_history(history)
    print(f"[picked best layout from history: {best_layout}]")
    for plan in adaptive_prompt_plans(history, best_layout)[: args.max_prompt_sweep_iters]:
        if not _time_left():
            print("[time budget reached during prompt sweep]")
            break
        _run(plan)

    # Aggregate
    summary_rows = [
        {
            "iteration_index": m["iteration_index"],
            "iteration_name": m["iteration_name"],
            "prompt_name": m["prompt_name"],
            "layout": m["layout"],
            "total_anchors": m["total_anchors"],
            "e3_anchors": m["e3_anchors"],
            "e45_anchors": m["e45_anchors"],
            "e3_bad_count": m["e3_bad_count"],
            "e3_bad_scored_high": m["e3_bad_scored_high"],
            "e3_false_good_rate": m["e3_false_good_rate"],
            "e3_good_count": m["e3_good_count"],
            "e3_good_rejected": m["e3_good_rejected"],
            "e3_false_reject_rate": m["e3_false_reject_rate"],
            "e45_bad_count": m["e45_bad_count"],
            "e45_bad_scored_high": m["e45_bad_scored_high"],
            "e45_false_good_rate": m["e45_false_good_rate"],
            "e45_good_count": m["e45_good_count"],
            "e45_good_rejected": m["e45_good_rejected"],
            "e45_false_reject_rate": m["e45_false_reject_rate"],
            "combined_bad_count": m["combined_bad_count"],
            "combined_bad_scored_high": m["combined_bad_scored_high"],
            "combined_false_good_rate": m["combined_false_good_rate"],
            "combined_good_count": m["combined_good_count"],
            "combined_good_rejected": m["combined_good_rejected"],
            "combined_false_reject_rate": m["combined_false_reject_rate"],
            "parse_success_rate": m["parse_success_rate"],
            "avg_runtime_ms_per_image": m["avg_runtime_ms_per_image"],
            "no_safe_anchor_cases": m["no_safe_anchor_cases"],
            "hypothesis": m["hypothesis"],
        }
        for m in iteration_summaries
    ]

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    _write_csv(session_dir / "summary_iteration_metrics.csv", summary_rows, summary_fields)
    _write_json(session_dir / "iteration_history.json", iteration_summaries)

    # best iteration
    if iteration_summaries:
        best = sorted(
            iteration_summaries,
            key=lambda m: (
                float(m.get("combined_false_good_rate") or 0.0),
                float(m.get("combined_false_reject_rate") or 0.0),
                -float(m.get("parse_success_rate") or 0.0),
            ),
        )[0]
        _write_json(session_dir / "best_iteration_summary.json", best)
        best_iter_dir = Path(best["iter_dir"])
        if (best_iter_dir / "per_anchor_results.csv").exists():
            (session_dir / "best_iteration_per_anchor_scores.csv").write_text(
                (best_iter_dir / "per_anchor_results.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if (best_iter_dir / "per_case_decisions.csv").exists():
            (session_dir / "best_iteration_per_case_decisions.csv").write_text(
                (best_iter_dir / "per_case_decisions.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if (best_iter_dir / "prompt.txt").exists():
            (session_dir / "best_prompt.txt").write_text(
                (best_iter_dir / "prompt.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (session_dir / "best_image_layout.md").write_text(
            describe_best_layout(best.get("layout", "L0_baseline"), best),
            encoding="utf-8",
        )

    print(f"\nDONE.\nSession dir: {session_dir}")
    return 0


def describe_best_layout(layout: str, best: dict[str, Any]) -> str:
    txt = [
        f"# Best image layout: `{layout}`",
        "",
        f"- iteration_index: {best.get('iteration_index')}",
        f"- prompt_name: {best.get('prompt_name')}",
        f"- combined_false_good_rate: {best.get('combined_false_good_rate'):.4f}",
        f"- combined_false_reject_rate: {best.get('combined_false_reject_rate'):.4f}",
        f"- e3_false_good_rate: {best.get('e3_false_good_rate'):.4f}",
        f"- e45_false_good_rate: {best.get('e45_false_good_rate'):.4f}",
        "",
        "Renderer: see `prepare_qwen_anchor_layout_variants.py` for the exact function used.",
        "",
        "Build the same image for a new anchor with:",
        "",
        "```",
        f"python -m upv_vlm_v2.experiments.prepare_qwen_anchor_layout_variants \\",
        "  --source <artifact_dir>/clean_single_anchor_inputs/source_A?.png \\",
        "  --geometry-debug <artifact_dir>/anchor_crop_geometry_debug.json \\",
        "  --anchor-id A?  --out-dir <output_dir>  --layouts " + layout,
        "```",
        "",
    ]
    return "\n".join(txt)


if __name__ == "__main__":
    raise SystemExit(main())
