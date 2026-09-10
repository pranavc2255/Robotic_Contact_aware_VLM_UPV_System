#!/usr/bin/env python3
"""
E3 Baseline 2:
Full-object per-anchor Qwen32 scoring baseline.

This baseline uses the SAME full-object neutral anchor-overlay image as the
direct-choice baseline, but instead of asking Qwen to choose one anchor from
A1-A5 directly, it evaluates each source anchor independently:

  Full-object overview + "Evaluate ONLY A1"
  Full-object overview + "Evaluate ONLY A2"
  ...
  Full-object overview + "Evaluate ONLY A5"

It does NOT use:
- clean_single_anchor_inputs/source_A*.png
- paired top/bottom local contact crops
- proposed contact-crop scoring logic

The goal is to remove direct-choice center/default-anchor collapse while still
keeping the baseline representation as the full-object overview only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ANCHOR_IDS = ["A1", "A2", "A3", "A4", "A5"]

PHASE = "e3_qwen32_full_overlay_per_anchor_scoring_baseline"
PROMPT_VERSION = "e3_full_overlay_per_anchor_strong_physical_contact_v1"


# ---------------------------------------------------------------------
# Strong baseline prompt
# ---------------------------------------------------------------------

def build_prompt(anchor_id: str) -> str:
    return f"""You are evaluating one candidate UPV contact anchor from a full-object overview image for robotic ultrasonic pulse velocity testing.

This is a PHYSICAL CONTACT-AFFORDANCE SELECTION task, not a visual-center, object-center, or geometry-only task.

The image shows one construction material object with candidate anchors labeled A1-A5. You must evaluate ONLY anchor {anchor_id}. Do not select, score, rank, or recommend any other anchor.

Anchor {anchor_id} represents a possible pair of opposite UPV probe contact locations across the object. A pair of flat planar UPV transducers will later press on the two opposite edge/contact regions associated with this anchor. The goal is to decide whether anchor {anchor_id} is physically suitable for stable, repeatable UPV probe contact.

Important limitation:
You are seeing only a full-object overview image, not isolated local contact crops. Use only the visible evidence available in this overview image. If the overview does not clearly show local contact quality, be conservative.

All anchor markers, labels, and lines are visual identifiers only. Do not infer contact quality from marker color, marker brightness, label position, line thickness, or anchor order.

A physically usable UPV contact anchor should satisfy ALL of the following as much as can be judged from the overview:
- both opposite contact neighborhoods appear clean and reachable
- the edge/contact regions appear continuous enough for flat probe placement
- the contact regions are not visibly blocked by debris, mortar, dust piles, tape, nails, screws, splinters, or foreign objects
- the contact regions are not on visibly chipped, jagged, broken, missing, cracked, or highly irregular edge sections
- the material near the contact regions does not visibly protrude outward into the probe contact path
- the visible edge neighborhood appears stable and approximately planar enough for repeatable probe coupling

Penalize anchor {anchor_id} if either side of the paired contact region appears risky. One bad side makes the whole anchor risky.

Specific defect cues to penalize:
- chipped or missing edge material
- jagged or broken edge boundary
- mortar-like crust, cement paste, gray/white buildup, or attached material
- nail, screw, tape, splinter, dust pile, debris, or foreign object near the contact region
- abrupt color or texture change at the contact side suggesting attached material or obstruction
- protruding material that could prevent flat transducer contact
- contact region located too close to a damaged corner or incomplete edge
- any visible obstruction that could produce unstable or no UPV signal

Material awareness:
- Brick: pores, speckles, rough fired-clay texture, and normal color variation are not defects by themselves. Penalize attached mortar/cement, white or gray crust, chipped/missing edge, jagged break, protrusion, debris, or obstructed contact region.
- Timber: grain, knots, saw marks, and normal wood texture are not defects by themselves. Penalize nails, screws, tape, splinters, cracks, protruding chips, debris, or blocked/uneven contact regions.
- Concrete/cinder block: aggregate, pores, and normal cementitious roughness are not defects by themselves. Penalize loose debris, protruding paste, broken/jagged edge, missing material, local obstruction, or unstable contact edge.

Scoring rule:
- 90-100: clearly clean, reachable, continuous, and low-risk for both opposite contact sides
- 70-89: mostly usable, minor visible imperfections but likely stable contact
- 50-69: uncertain or marginal; visible limitations but not clearly unusable
- 25-49: risky; one or both contact sides show defects likely to affect coupling
- 0-24: contact-unsuitable; obvious obstruction, broken edge, missing material, or severe irregularity

Set usable=true only if anchor {anchor_id} is likely to provide stable UPV contact on both opposite sides.
Set usable=false if either side is visibly bad, blocked, damaged, or too uncertain from the overview.

Return JSON only. Do not include markdown fences or extra text.

Use exactly this schema:
{{
  "anchor_id": "{anchor_id}",
  "top_or_side_a_assessment": "short visible assessment",
  "bottom_or_side_b_assessment": "short visible assessment",
  "usable": true,
  "score": 0,
  "defects": ["short visible defect list"],
  "uncertainty": "short note if full overview does not clearly reveal local contact quality",
  "reason": "short physical-contact reason based only on anchor {anchor_id}"
}}
"""


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def case_sort_key(p: Path) -> tuple[int, str]:
    m = re.search(r"case_(\d+)_", p.name)
    return (int(m.group(1)) if m else 9999, p.name)


def get_case_id_from_case_name(case_name: str) -> str:
    # case_004_brick_04_debris_obstruction -> brick_04_debris_obstruction
    return re.sub(r"^case_\d+_", "", case_name)


def label_to_contact_class(label: str) -> str:
    t = str(label).strip().lower()
    if t in {"good", "acceptable"}:
        return "contact_usable"
    if t == "bad":
        return "contact_unsuitable"
    return "unknown"


def load_manual_labels(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """
    Expected manual labels CSV from label_e3_anchor_candidates.py.
    Supports both case_id/anchor_id/label style and similar variants.
    """
    rows = read_csv(path)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        case_id = r.get("case_id") or r.get("source_case_id") or r.get("id")
        anchor_id = r.get("anchor_id") or r.get("source_anchor_id") or r.get("candidate_anchor_id")
        label = r.get("label") or r.get("manual_label") or r.get("contact_label")
        if not case_id or not anchor_id:
            continue
        rr = dict(r)
        rr["case_id"] = case_id
        rr["anchor_id"] = anchor_id
        rr["label"] = label or ""
        rr["contact_class"] = label_to_contact_class(label or "")
        out[(case_id, anchor_id)] = rr
    return out


# ---------------------------------------------------------------------
# Reuse full-overlay renderer from direct baseline
# ---------------------------------------------------------------------

def render_full_overlay_from_existing_runner(case_dir: Path, output_path: Path) -> dict[str, Any]:
    """
    Reuse the neutral full-overlay rendering logic already implemented in
    run_e3_qwen32_full_overlay_baseline_from_session.py.

    This keeps Baseline 1 and Baseline 2 image representation identical.
    """
    from upv_vlm_v2.experiments import run_e3_qwen32_full_overlay_baseline_from_session as direct

    if not hasattr(direct, "render_full_anchor_overlay"):
        raise RuntimeError(
            "Existing direct baseline runner does not expose render_full_anchor_overlay()."
        )

    fn = getattr(direct, "render_full_anchor_overlay")
    meta = fn(case_dir, output_path)

    if not isinstance(meta, dict):
        meta = {"render_return_value": str(meta)}

    meta["source_renderer"] = "run_e3_qwen32_full_overlay_baseline_from_session.render_full_anchor_overlay"
    meta["baseline2_reused_direct_baseline_overlay_renderer"] = True
    return meta


# ---------------------------------------------------------------------
# Qwen call and parsing
# ---------------------------------------------------------------------

def post_json(url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as r:
            txt = r.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "error": f"HTTPError {exc.code}",
            "status": exc.code,
            "body_snippet": body[:1000],
            "_raw_http_body": body,
        }
    except URLError as exc:
        return {"ok": False, "error": f"URLError {exc}", "body_snippet": ""}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "body_snippet": ""}

    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            obj.setdefault("_raw_http_body", txt)
            return obj
        return {"ok": True, "raw_http_text": txt, "_raw_http_body": txt}
    except Exception:
        return {"ok": True, "raw_http_text": txt, "_raw_http_body": txt}


def call_qwen(
    *,
    server_url: str,
    server_endpoint: str,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    output_path: Path,
    timeout_sec: float,
) -> tuple[str, dict[str, Any]]:
    """Call the local UPV_VLM_v2 Qwen server using its supported /infer schema."""
    base = server_url.rstrip("/")
    endpoint = base + (server_endpoint if server_endpoint.startswith("/") else f"/{server_endpoint}")
    payload = {
        "image_path": str(image_path),
        "prompt_text": prompt,
        "prompt_version": PROMPT_VERSION,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "output_path": str(output_path),
    }
    obj = post_json(endpoint, payload, timeout=timeout_sec)
    if obj.get("ok") is False:
        raise RuntimeError(
            f"Qwen request failed at {endpoint}: {obj.get('error')}; body={str(obj.get('body_snippet') or '')[:500]}"
        )

    for key in ["text", "raw_text", "response", "output", "answer", "generated_text"]:
        if isinstance(obj.get(key), str):
            return str(obj[key]), obj
    if isinstance(obj.get("result"), dict):
        for key in ["text", "raw_text", "response", "output", "answer", "generated_text"]:
            if isinstance(obj["result"].get(key), str):
                return str(obj["result"][key]), obj
    if isinstance(obj.get("raw_http_text"), str):
        return str(obj["raw_http_text"]), obj
    return json.dumps(obj, indent=2), obj


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract first JSON object from Qwen text, allowing markdown fences.
    """
    raw = text.strip()

    # Remove markdown fence if present.
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.I).strip()
    raw = re.sub(r"```$", "", raw.strip()).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract substring from first { to last }.
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        candidate = raw[s : e + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from response")


def parse_anchor_response(raw: str, expected_anchor_id: str) -> dict[str, Any]:
    try:
        obj = extract_json_object(raw)
        repaired = False

        aid = str(obj.get("anchor_id") or "").strip()
        if aid not in ANCHOR_IDS:
            aid = expected_anchor_id
            repaired = True

        usable = obj.get("usable")
        if isinstance(usable, str):
            usable = usable.strip().lower() in {"true", "yes", "1", "usable", "good", "acceptable"}
        usable = bool(usable)

        score = obj.get("score", 0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))

        defects = obj.get("defects", [])
        if isinstance(defects, str):
            defects = [defects]
        if not isinstance(defects, list):
            defects = []

        return {
            "parse_ok": True,
            "anchor_id": aid,
            "anchor_id_repaired": repaired,
            "usable": usable,
            "score": score,
            "defects": defects,
            "top_or_side_a_assessment": str(obj.get("top_or_side_a_assessment", "")),
            "bottom_or_side_b_assessment": str(obj.get("bottom_or_side_b_assessment", "")),
            "uncertainty": str(obj.get("uncertainty", "")),
            "reason": str(obj.get("reason", "")),
            "raw_parsed_object": obj,
        }
    except Exception as e:
        return {
            "parse_ok": False,
            "anchor_id": expected_anchor_id,
            "anchor_id_repaired": True,
            "usable": False,
            "score": -1.0,
            "defects": [],
            "top_or_side_a_assessment": "",
            "bottom_or_side_b_assessment": "",
            "uncertainty": "",
            "reason": f"PARSE_ERROR: {e}",
            "raw_parsed_object": {},
        }


# ---------------------------------------------------------------------
# Optional deterministic tie-break
# ---------------------------------------------------------------------

def load_deterministic_tiebreak_scores(case_dir: Path) -> dict[str, float]:
    """
    Best-effort extraction of deterministic per-anchor scores.
    If unavailable, returns zeros.
    """
    candidates = [
        case_dir / "deterministic_anchor_features_wide_context.json",
        case_dir / "anchor_selection_summary.json",
        case_dir / "final_anchor_decision_wide_context.json",
    ]

    scores: dict[str, float] = {a: 0.0 for a in ANCHOR_IDS}

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            aid = obj.get("anchor_id") or obj.get("id") or obj.get("source_anchor_id")
            if aid in ANCHOR_IDS:
                for key in [
                    "deterministic_score",
                    "geometry_score",
                    "score",
                    "overall_score",
                    "anchor_score",
                    "combined_score",
                ]:
                    if key in obj:
                        try:
                            scores[str(aid)] = float(obj[key])
                            break
                        except Exception:
                            pass
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for p in candidates:
        if p.exists():
            try:
                visit(read_json(p))
            except Exception:
                pass

    return scores


def select_best_anchor(anchor_rows: list[dict[str, Any]], tiebreak_scores: dict[str, float]) -> dict[str, Any] | None:
    valid = [r for r in anchor_rows if r.get("parse_ok")]
    if not valid:
        return None

    def key(r: dict[str, Any]) -> tuple[int, float, float, str]:
        aid = str(r["anchor_id"])
        return (
            1 if r.get("usable") else 0,
            float(r.get("score", -1.0)),
            float(tiebreak_scores.get(aid, 0.0)),
            # reverse sort will put larger string last if used directly,
            # so use negative ordinal-like behavior by returning id normally
            # and sorting with reverse=False below? Easier: sort manually below.
            aid,
        )

    # We want usable true, high score, high deterministic score, then A1 before A5.
    valid_sorted = sorted(
        valid,
        key=lambda r: (
            -(1 if r.get("usable") else 0),
            -float(r.get("score", -1.0)),
            -float(tiebreak_scores.get(str(r["anchor_id"]), 0.0)),
            str(r["anchor_id"]),
        ),
    )
    return valid_sorted[0]


# ---------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------

@dataclass
class Args:
    session: Path
    manual_labels: Path
    server_url: str
    server_endpoint: str
    max_new_tokens: int
    temperature: float
    timeout_sec: float
    case_id: str | None
    fail_fast: bool


def make_output_session(input_session: Path) -> Path:
    out = Path("outputs/v2_experiments/e3_qwen32_full_overlay_per_anchor_scoring_baseline") / f"session_{now_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(args: Args) -> Path:
    if not args.session.exists():
        raise FileNotFoundError(args.session)
    if not args.manual_labels.exists():
        raise FileNotFoundError(args.manual_labels)

    manual = load_manual_labels(args.manual_labels)
    out_session = make_output_session(args.session)

    write_json(
        out_session / "run_manifest.json",
        {
            "phase": PHASE,
            "prompt_version": PROMPT_VERSION,
            "input_session": str(args.session),
            "manual_labels": str(args.manual_labels),
            "server_url": args.server_url,
            "server_endpoint": args.server_endpoint,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "timeout_sec": args.timeout_sec,
            "case_id_filter": args.case_id,
            "forbidden_inputs": [
                "clean_single_anchor_inputs/source_A*.png",
                "paired local contact crops",
                "proposed contact-crop scoring images",
            ],
            "baseline_definition": "Full-object neutral overlay image, one independent Qwen call per source anchor.",
        },
    )

    case_dirs = sorted((args.session / "cases").glob("case_*"), key=case_sort_key)
    if args.case_id:
        wanted = {x.strip() for x in args.case_id.split(",") if x.strip()}
        case_dirs = [p for p in case_dirs if get_case_id_from_case_name(p.name) in wanted]
        missing = sorted(wanted - {get_case_id_from_case_name(p.name) for p in case_dirs})
        if missing:
            raise ValueError(f"Requested case_id not found in session: {missing}")

    endpoint_for_log = args.server_url.rstrip("/") + (
        args.server_endpoint if args.server_endpoint.startswith("/") else f"/{args.server_endpoint}"
    )
    print(f"[QWEN] endpoint={endpoint_for_log}")

    master_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for i, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        case_id = get_case_id_from_case_name(case_name)
        print(f"[{i}/{len(case_dirs)}] {case_id}")

        out_case = out_session / "cases" / case_name
        out_case.mkdir(parents=True, exist_ok=True)

        try:
            overlay_path = out_case / "input_full_anchor_overlay.png"
            overlay_meta = render_full_overlay_from_existing_runner(case_dir, overlay_path)
            write_json(out_case / "input_full_anchor_overlay_metadata.json", overlay_meta)

            tiebreak_scores = load_deterministic_tiebreak_scores(case_dir)

            prompt_dir = out_case / "per_anchor_prompts"
            raw_dir = out_case / "qwen32_responses"
            parsed_dir = out_case / "parsed"
            prompt_dir.mkdir(exist_ok=True)
            raw_dir.mkdir(exist_ok=True)
            parsed_dir.mkdir(exist_ok=True)

            anchor_rows: list[dict[str, Any]] = []

            for aid in ANCHOR_IDS:
                prompt = build_prompt(aid)
                (prompt_dir / f"source_{aid}_prompt.txt").write_text(prompt, encoding="utf-8")

                raw_path = raw_dir / f"source_{aid}_raw.txt"
                server_response_path = raw_dir / f"source_{aid}_server_response.json"
                raw, server_response = call_qwen(
                    server_url=args.server_url,
                    server_endpoint=args.server_endpoint,
                    image_path=overlay_path,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    output_path=raw_path,
                    timeout_sec=args.timeout_sec,
                )
                raw_path.write_text(raw, encoding="utf-8")
                write_json(server_response_path, server_response)

                parsed = parse_anchor_response(raw, aid)
                write_json(parsed_dir / f"source_{aid}_parsed.json", parsed)

                m = manual.get((case_id, aid), {})
                label = m.get("label", "")
                contact_class = m.get("contact_class", label_to_contact_class(label))

                row = {
                    "case_name": case_name,
                    "case_id": case_id,
                    "anchor_id": aid,
                    "parse_ok": bool(parsed["parse_ok"]),
                    "anchor_id_repaired": bool(parsed.get("anchor_id_repaired", False)),
                    "qwen_anchor_id": parsed.get("anchor_id", aid),
                    "qwen_usable": bool(parsed.get("usable", False)),
                    "qwen_score": float(parsed.get("score", -1.0)),
                    "manual_label": label,
                    "manual_contact_class": contact_class,
                    "deterministic_tiebreak_score": tiebreak_scores.get(aid, 0.0),
                    "defects": "|".join(str(x) for x in parsed.get("defects", [])),
                    "uncertainty": parsed.get("uncertainty", ""),
                    "reason": parsed.get("reason", ""),
                    "raw_response_path": str(raw_path),
                    "server_response_path": str(server_response_path),
                    "parsed_response_path": str(parsed_dir / f"source_{aid}_parsed.json"),
                    "prompt_path": str(prompt_dir / f"source_{aid}_prompt.txt"),
                }
                anchor_rows.append(row)

                # Small delay to avoid hammering local server too aggressively.
                time.sleep(0.05)

            write_csv(
                out_case / "anchor_score_table.csv",
                anchor_rows,
                [
                    "case_name",
                    "case_id",
                    "anchor_id",
                    "parse_ok",
                    "anchor_id_repaired",
                    "qwen_anchor_id",
                    "qwen_usable",
                    "qwen_score",
                    "manual_label",
                    "manual_contact_class",
                    "deterministic_tiebreak_score",
                    "defects",
                    "uncertainty",
                    "reason",
                    "prompt_path",
                    "raw_response_path",
                    "server_response_path",
                    "parsed_response_path",
                ],
            )

            selected = select_best_anchor(anchor_rows, tiebreak_scores)
            if selected is None:
                selected_anchor = "NO_PARSE_RESULT"
                selected_label = ""
                selected_class = "unknown"
                top1_contact_usable = False
                contact_unsuitable_selected = False
                selected_score = -1.0
                selected_reason = "No parsed anchors."
            else:
                selected_anchor = str(selected["anchor_id"])
                selected_label = str(selected.get("manual_label", ""))
                selected_class = str(selected.get("manual_contact_class", "unknown"))
                top1_contact_usable = selected_class == "contact_usable"
                contact_unsuitable_selected = selected_class == "contact_unsuitable"
                selected_score = float(selected.get("qwen_score", -1.0))
                selected_reason = str(selected.get("reason", ""))

            case_summary = {
                "case_name": case_name,
                "case_id": case_id,
                "selected_anchor": selected_anchor,
                "selected_manual_label": selected_label,
                "selected_contact_class": selected_class,
                "top1_contact_usable": top1_contact_usable,
                "contact_unsuitable_selected": contact_unsuitable_selected,
                "selected_qwen_score": selected_score,
                "anchors_attempted": len(anchor_rows),
                "anchors_parsed": sum(1 for r in anchor_rows if r["parse_ok"]),
                "selected_reason": selected_reason,
                "overlay_path": str(overlay_path),
            }
            write_json(out_case / "case_summary.json", case_summary)

            master_rows.append({
                **case_summary,
                "case_output_dir": str(out_case),
            })

            print(
                f"  selected={selected_anchor} "
                f"label={selected_label} "
                f"class={selected_class} "
                f"parsed={case_summary['anchors_parsed']}/{case_summary['anchors_attempted']}"
            )

        except Exception as e:
            err = {
                "case_name": case_name,
                "case_id": case_id,
                "error": repr(e),
                "case_output_dir": str(out_case),
            }
            error_rows.append(err)
            print("  ERROR:", repr(e))
            if args.fail_fast:
                raise

    # Master CSV
    master_fields = [
        "case_name",
        "case_id",
        "selected_anchor",
        "selected_manual_label",
        "selected_contact_class",
        "top1_contact_usable",
        "contact_unsuitable_selected",
        "selected_qwen_score",
        "anchors_attempted",
        "anchors_parsed",
        "selected_reason",
        "overlay_path",
        "case_output_dir",
    ]
    write_csv(out_session / "master_full_overlay_per_anchor_scoring_results.csv", master_rows, master_fields)

    if error_rows:
        write_csv(out_session / "errors.csv", error_rows, ["case_name", "case_id", "error", "case_output_dir"])

    n_cases = len(master_rows)
    anchors_attempted = sum(int(r["anchors_attempted"]) for r in master_rows)
    anchors_parsed = sum(int(r["anchors_parsed"]) for r in master_rows)
    top1_usable = sum(1 for r in master_rows if str(r["top1_contact_usable"]).lower() == "true" or r["top1_contact_usable"] is True)
    bad_selected = sum(1 for r in master_rows if str(r["contact_unsuitable_selected"]).lower() == "true" or r["contact_unsuitable_selected"] is True)

    selected_dist: dict[str, int] = {}
    for r in master_rows:
        selected_dist[str(r["selected_anchor"])] = selected_dist.get(str(r["selected_anchor"]), 0) + 1

    summary = {
        "phase": PHASE,
        "prompt_version": PROMPT_VERSION,
        "input_session": str(args.session),
        "manual_labels": str(args.manual_labels),
        "output_session": str(out_session),
        "n_cases": n_cases,
        "n_errors": len(error_rows),
        "anchors_attempted": anchors_attempted,
        "anchors_parsed": anchors_parsed,
        "anchor_parse_success_rate": (anchors_parsed / anchors_attempted) if anchors_attempted else 0.0,
        "cases_with_all_anchors_parsed": sum(1 for r in master_rows if int(r["anchors_parsed"]) == int(r["anchors_attempted"])),
        "top1_contact_usable_count": top1_usable,
        "top1_contact_usable_rate": (top1_usable / n_cases) if n_cases else 0.0,
        "contact_unsuitable_selected_count": bad_selected,
        "contact_unsuitable_selected_rate": (bad_selected / n_cases) if n_cases else 0.0,
        "selected_anchor_distribution": selected_dist,
    }
    write_json(out_session / "summary_overall.json", summary)

    print()
    print("DONE")
    print("output_session:", out_session)
    print("summary:", out_session / "summary_overall.json")
    print("master:", out_session / "master_full_overlay_per_anchor_scoring_results.csv")
    return out_session


def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--manual-labels", required=True, type=Path)
    ap.add_argument("--server-url", default="http://127.0.0.1:8899")
    ap.add_argument("--server-endpoint", default="/infer")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-sec", type=float, default=240.0)
    ap.add_argument("--case-id", default=None, help="Optional case_id or comma-separated case IDs to run.")
    ap.add_argument("--fail-fast", action="store_true")
    ns = ap.parse_args()
    return Args(
        session=ns.session,
        manual_labels=ns.manual_labels,
        server_url=ns.server_url,
        server_endpoint=ns.server_endpoint,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        timeout_sec=ns.timeout_sec,
        case_id=ns.case_id,
        fail_fast=ns.fail_fast,
    )


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
