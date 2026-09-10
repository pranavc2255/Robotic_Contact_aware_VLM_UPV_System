#!/usr/bin/env python3
"""
E3 Qwen32 full-overlay anchor-selection baseline.

This baseline gives Qwen one full-object/overview anchor image per case and asks
it to select the best UPV contact anchor. It intentionally does not use the
single-anchor contact-crop inputs from the proposed method.

No robot, RTDE, Arduino, clamp, live camera, dataset capture, or E2 path is
touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from upv_vlm_v2.experiments.run_e3_qwen32_shuffle_consensus_from_session import (
    anchor_sort_key,
    discover_cases,
    extract_response,
    load_manual_labels,
    post_json,
    write_csv,
    write_json,
)


OUTPUT_ROOT = Path("outputs/v2_experiments/e3_qwen32_full_overlay_baseline")
PROMPT_VERSION = "e3_qwen32_full_overlay_baseline_neutral_overlay_v2"

PROMPT_TEXT = """You are selecting one UPV contact anchor from a full-object overview image.

This is a physical UPV contact-affordance selection task, not a visual-center or geometry-only task.

The image shows one construction material object with five candidate anchors labeled A1-A5. Each anchor represents a possible pair of opposite UPV probe contact locations across the object. Select the anchor most likely to provide stable, repeatable ultrasonic pulse velocity probe contact.

Judge only from visible evidence in the full-object overview image.

Important visual-marker rule:
All anchor markers, labels, and lines are only visual identifiers. Do not infer contact quality from marker color, marker brightness, label position, or line thickness.

A good UPV anchor should have contact regions that appear:
- clean
- continuous
- open/reachable
- not blocked by debris or attached material
- not located on visibly chipped, jagged, broken, missing, or irregular edge regions
- not near protrusions, mortar-like buildup, crust, nails, tape, splinters, dust piles, or foreign objects
- not located where the material edge appears non-planar or unstable for flat probe contact

Important physical-contact rules:
- Do not choose an anchor only because it is near the object center.
- Do not choose an anchor only because its marker/line looks geometrically neat.
- Prefer anchors whose visible edge neighborhoods appear clean and stable for physical contact.
- Penalize anchors near edge damage, missing material, protruding material, debris, mortar/crust, jagged boundaries, or abrupt material/color changes.
- If all anchors are imperfect, choose the least risky anchor.

Material awareness:
- Brick: normal fired-clay texture, pores, speckles, and roughness are not defects by themselves. Gray/white cement-like attached material, mortar, crust, protrusion, chipped/missing edge, jagged break, abrupt boundary, or debris near the contact region is a defect.
- Timber: normal grain, knots, and saw marks are not defects by themselves. Nails, screws, tape, splinters, cracks, protruding chips, debris, or blocked contact regions are defects.
- Concrete/cinder block: normal gray cementitious texture, aggregate, pores, and roughness are not defects by themselves. Loose debris, protruding paste, broken/jagged edge, missing material, local obstruction, or material sticking out near the contact region is a defect.

Return JSON only. Do not include markdown fences or text outside JSON.

Use exactly this schema:
{
  "selected_anchor": "<A1_or_A2_or_A3_or_A4_or_A5>",
  "ranking_best_to_worst": ["<best>", "<next>", "<next>", "<next>", "<worst>"],
  "rejected_risky_anchors": ["<risky_anchor_id_if_any>"],
  "reason": "short physical-contact reason based on visible evidence"
}
"""

ANCHOR_COLOR = (255, 255, 255)
ANCHOR_OUTLINE_COLOR = (0, 0, 0)
CONTACT_LINE_COLOR = (255, 255, 255)
CONTACT_LINE_OUTLINE_COLOR = (0, 0, 0)
LABEL_FILL_COLOR = (255, 255, 255)
LABEL_TEXT_COLOR = (0, 0, 0)
MASK_TINT_COLOR = (0, 200, 255, 45)
MASK_CONTOUR_COLOR = (0, 210, 210, 180)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_anchor_id(value: Any) -> str:
    aid = str(value or "").strip().upper()
    if aid.startswith("ANCHOR "):
        aid = aid.split()[-1]
    return aid


def is_usable_label(label: str) -> bool:
    return label.strip().lower() in {"good", "acceptable"}


def is_bad_label(label: str) -> bool:
    return label.strip().lower() == "bad"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _latest_pipeline_artifacts(case_dir: Path) -> Path | None:
    candidates = sorted(case_dir.glob("pipeline_session/session_*/artifacts"))
    return candidates[-1] if candidates else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _load_rgb_and_mask(case_dir: Path) -> tuple[Image.Image, Image.Image | None, Path]:
    artifacts = _latest_pipeline_artifacts(case_dir)
    candidates: list[Path] = []
    if artifacts is not None:
        candidates.extend(
            [
                artifacts / "01_capture" / "input_rgb.png",
                artifacts / "02_target_selection" / "raw_rgb.png",
            ]
        )
    candidates.extend([case_dir / "raw_rgb.png", case_dir / "input_rgb.png"])

    rgb_path = next((p for p in candidates if p.exists()), None)
    if rgb_path is None:
        raise FileNotFoundError(f"No raw RGB image found for {case_dir}")

    rgb = Image.open(rgb_path).convert("RGB")

    mask: Image.Image | None = None
    if artifacts is not None:
        mask_path = artifacts / "02_target_selection" / "selected_mask.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            if mask.size != rgb.size:
                mask = mask.resize(rgb.size, Image.Resampling.NEAREST)

    return rgb, mask, rgb_path


def load_anchor_candidates_for_case(case_dir: Path) -> list[dict[str, Any]]:
    feature_path = case_dir / "deterministic_anchor_features_wide_context.json"
    if not feature_path.exists():
        artifacts = _latest_pipeline_artifacts(case_dir)
        if artifacts is not None:
            feature_path = artifacts / "04_anchor_selection" / "deterministic_anchor_features_wide_context.json"

    features = load_json(feature_path)
    anchors = features.get("anchor_features")
    if not isinstance(anchors, list):
        raise FileNotFoundError(f"No anchor_features found at {feature_path}")

    out: list[dict[str, Any]] = []
    for item in anchors:
        if not isinstance(item, dict):
            continue
        aid = normalize_anchor_id(item.get("anchor_id"))
        center = _point(item.get("anchor_px"))
        pa = _point(item.get("contact_point_a_px"))
        pb = _point(item.get("contact_point_b_px"))
        if aid in {"A1", "A2", "A3", "A4", "A5"} and center is not None:
            out.append(
                {
                    "anchor_id": aid,
                    "anchor_px": center,
                    "contact_point_a_px": pa,
                    "contact_point_b_px": pb,
                    "deterministic_score": item.get("deterministic_score"),
                }
            )

    out.sort(key=lambda x: anchor_sort_key(str(x["anchor_id"])))
    ids = [str(a["anchor_id"]) for a in out]
    if ids != ["A1", "A2", "A3", "A4", "A5"]:
        raise ValueError(f"Expected A1-A5 anchor features for {case_dir.name}; found {ids}")
    return out


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=LABEL_FILL_COLOR,
        outline=ANCHOR_OUTLINE_COLOR,
        width=2,
    )
    draw.text((x, y), text, fill=LABEL_TEXT_COLOR, font=font)


def render_full_anchor_overlay(case_dir: Path, output_path: Path) -> dict[str, Any]:
    rgb, mask, rgb_path = _load_rgb_and_mask(case_dir)
    anchors = load_anchor_candidates_for_case(case_dir)

    base = rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if mask is not None:
        mask_overlay = Image.new("RGBA", base.size, MASK_TINT_COLOR)
        base = Image.composite(Image.alpha_composite(base, mask_overlay), base, mask)

        # Draw a lightweight mask contour by marking boundary pixels without
        # depending on OpenCV/scipy.
        try:
            import numpy as np  # type: ignore

            arr = np.array(mask) > 0
            boundary = arr.copy()
            boundary[1:, :] &= arr[:-1, :]
            boundary[:-1, :] &= arr[1:, :]
            boundary[:, 1:] &= arr[:, :-1]
            boundary[:, :-1] &= arr[:, 1:]
            edge = arr & ~boundary
            ys, xs = np.where(edge)
            for x, y in zip(xs[:: max(1, len(xs) // 6000)], ys[:: max(1, len(ys) // 6000)]):
                draw.point((int(x), int(y)), fill=MASK_CONTOUR_COLOR)
        except Exception:
            pass

    line_w = max(3, int(round(min(base.size) * 0.004)))
    marker_r = max(10, int(round(min(base.size) * 0.014)))
    label_font = _font(max(24, int(round(min(base.size) * 0.034))))

    for a in anchors:
        aid = str(a["anchor_id"])
        pa = a.get("contact_point_a_px")
        pb = a.get("contact_point_b_px")
        cx, cy = a["anchor_px"]

        if pa is not None and pb is not None:
            # Neutral contact line: black outline + white core.
            draw.line(
                (pa[0], pa[1], pb[0], pb[1]),
                fill=(*CONTACT_LINE_OUTLINE_COLOR, 235),
                width=line_w + 4,
            )
            draw.line(
                (pa[0], pa[1], pb[0], pb[1]),
                fill=(*CONTACT_LINE_COLOR, 235),
                width=line_w,
            )
            for px, py in (pa, pb):
                rr = max(5, marker_r // 2)
                draw.ellipse(
                    (px - rr, py - rr, px + rr, py + rr),
                    fill=(*ANCHOR_COLOR, 235),
                    outline=(*ANCHOR_OUTLINE_COLOR, 255),
                    width=2,
                )

        # Neutral anchor center marker: identical for all A1-A5.
        draw.ellipse(
            (cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r),
            fill=(*ANCHOR_COLOR, 235),
            outline=(*ANCHOR_OUTLINE_COLOR, 255),
            width=3,
        )
        _draw_label(draw, (cx + marker_r + 5, cy - marker_r - 5), aid, label_font)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

    return {
        "rgb_source_path": str(rgb_path),
        "mask_used": mask is not None,
        "anchor_ids": [str(a["anchor_id"]) for a in anchors],
        "overlay_path": str(output_path),
        "neutral_anchor_style": True,
        "all_anchor_colors_identical": True,
        "anchor_label_style_identical": True,
        "contact_line_style_identical": True,
        "visual_bias_patch": "neutral_black_white_anchor_overlay_v1",
    }


def normalize_ranking(value: Any, selected_anchor: str) -> list[str]:
    ranking: list[str] = []
    if isinstance(value, list):
        for item in value:
            aid = normalize_anchor_id(item)
            if aid and aid not in ranking:
                ranking.append(aid)
    if selected_anchor and selected_anchor not in ranking:
        ranking.insert(0, selected_anchor)
    return sorted(ranking, key=anchor_sort_key) if not ranking else ranking


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_cases = len(rows)
    parse_success_count = sum(1 for r in rows if bool(r.get("parse_ok")))
    usable_count = sum(1 for r in rows if bool(r.get("top1_contact_usable")))
    unsuitable_count = sum(1 for r in rows if bool(r.get("contact_unsuitable_selected")))

    return {
        "phase": "e3_qwen32_full_overlay_baseline_from_existing_session",
        "n_cases": n_cases,
        "parse_success_count": parse_success_count,
        "parse_success_rate": parse_success_count / n_cases if n_cases else None,
        "top1_contact_usable_count": usable_count,
        "top1_contact_usable_rate": usable_count / n_cases if n_cases else None,
        "contact_unsuitable_selected_count": unsuitable_count,
        "contact_unsuitable_selected_rate": unsuitable_count / n_cases if n_cases else None,
    }


def run(args: argparse.Namespace) -> Path:
    session_dir = Path(args.session)
    manual_labels_path = Path(args.manual_labels)
    output_root = Path(args.output_root)
    out_session = output_root / f"session_{now_stamp()}"
    out_session.mkdir(parents=True, exist_ok=False)

    labels, manual_best = load_manual_labels(manual_labels_path)
    cases = discover_cases(session_dir, "case_*")

    infer_url = args.server_url.rstrip("/") + args.server_endpoint
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    write_json(
        out_session / "run_manifest.json",
        {
            "phase": "e3_qwen32_full_overlay_baseline_from_existing_session",
            "input_session": str(session_dir),
            "manual_labels": str(manual_labels_path),
            "server_url": args.server_url,
            "server_endpoint": args.server_endpoint,
            "prompt_version": PROMPT_VERSION,
            "output_session": str(out_session),
            "baseline_input": "generated input_full_anchor_overlay.png from raw RGB, target mask, and A1-A5 anchor geometry",
            "forbidden_inputs": [
                "clean_single_anchor_inputs/source_A*.png",
                "combined_wide_contact_guided_grid_2col.png",
                "per-anchor contact crops",
            ],
        },
    )

    for case in cases:
        case_out = out_session / "cases" / case.case_name
        case_out.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "case_name": case.case_name,
            "case_id": case.case_id,
            "case_index": case.case_index,
            "material": case.material,
            "condition": case.condition,
            "parse_ok": False,
            "parse_error": "",
            "selected_anchor": "",
            "ranking_best_to_worst": "",
            "reason": "",
            "selected_manual_label": "",
            "manual_best_anchor": manual_best.get(case.case_id, ""),
            "top1_contact_usable": False,
            "contact_unsuitable_selected": False,
            "is_manual_best_match": False,
            "input_overlay_source": "generated_full_anchor_overlay",
            "input_overlay_path": "",
            "overlay_render_metadata_path": str(case_out / "input_full_anchor_overlay_metadata.json"),
            "prompt_path": str(case_out / "baseline_prompt.txt"),
            "raw_response_path": str(case_out / "baseline_response_raw.txt"),
            "parsed_response_path": str(case_out / "baseline_response_parsed.json"),
        }

        try:
            input_path = case_out / "input_full_anchor_overlay.png"
            render_meta = render_full_anchor_overlay(case.case_dir, input_path)
            write_json(case_out / "input_full_anchor_overlay_metadata.json", render_meta)

            prompt_path = case_out / "baseline_prompt.txt"
            prompt_path.write_text(PROMPT_TEXT, encoding="utf-8")

            row["input_overlay_path"] = str(input_path)

            payload = {
                "image_path": str(input_path),
                "prompt_text": PROMPT_TEXT,
                "prompt_version": PROMPT_VERSION,
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "output_path": str(case_out / "baseline_response_raw.txt"),
            }
            response = post_json(infer_url, payload, int(args.timeout_sec))
            write_json(case_out / "baseline_server_response.json", response)

            raw_text, parsed, parse_ok, parse_error = extract_response(response)
            (case_out / "baseline_response_raw.txt").write_text(raw_text, encoding="utf-8")

            selected_anchor = ""
            ranking: list[str] = []
            reason = ""
            if parse_ok and isinstance(parsed, dict):
                selected_anchor = normalize_anchor_id(parsed.get("selected_anchor"))
                ranking = normalize_ranking(parsed.get("ranking_best_to_worst"), selected_anchor)
                reason = str(parsed.get("reason") or "")
                if selected_anchor not in {"A1", "A2", "A3", "A4", "A5"}:
                    parse_ok = False
                    parse_error = f"invalid selected_anchor: {selected_anchor!r}"
            else:
                parse_error = parse_error or "parse failed"

            normalized = {
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "selected_anchor": selected_anchor,
                "ranking_best_to_worst": ranking,
                "reason": reason,
                "raw_parsed_json": parsed,
            }
            write_json(case_out / "baseline_response_parsed.json", normalized)

            selected_label = labels.get((case.case_id, selected_anchor), "") if parse_ok else ""
            row.update(
                {
                    "parse_ok": bool(parse_ok),
                    "parse_error": parse_error,
                    "selected_anchor": selected_anchor if parse_ok else "",
                    "ranking_best_to_worst": "|".join(ranking),
                    "reason": reason,
                    "selected_manual_label": selected_label,
                    "top1_contact_usable": bool(parse_ok and is_usable_label(selected_label)),
                    "contact_unsuitable_selected": bool(parse_ok and is_bad_label(selected_label)),
                    "is_manual_best_match": bool(parse_ok and selected_anchor == manual_best.get(case.case_id, "")),
                }
            )

        except Exception as exc:
            row["parse_error"] = repr(exc)
            errors.append({"case_id": case.case_id, "case_name": case.case_name, "error": repr(exc)})
            if args.fail_fast:
                rows.append(row)
                write_csv(out_session / "master_full_overlay_baseline_results.csv", rows)
                write_csv(out_session / "errors.csv", errors)
                raise

        rows.append(row)
        print(
            f"[{case.case_name}] selected={row.get('selected_anchor') or 'NO_PARSE'} "
            f"label={row.get('selected_manual_label') or '-'} parse_ok={row.get('parse_ok')}"
        )

    fields = [
        "case_name",
        "case_id",
        "case_index",
        "material",
        "condition",
        "input_overlay_source",
        "input_overlay_path",
        "overlay_render_metadata_path",
        "parse_ok",
        "parse_error",
        "selected_anchor",
        "ranking_best_to_worst",
        "reason",
        "selected_manual_label",
        "manual_best_anchor",
        "top1_contact_usable",
        "contact_unsuitable_selected",
        "is_manual_best_match",
        "prompt_path",
        "raw_response_path",
        "parsed_response_path",
    ]
    write_csv(out_session / "master_full_overlay_baseline_results.csv", rows, fields)
    write_csv(out_session / "errors.csv", errors)

    summary = summarize(rows)
    summary.update(
        {
            "input_session": str(session_dir),
            "manual_labels": str(manual_labels_path),
            "output_session": str(out_session),
        }
    )
    write_json(out_session / "summary_overall.json", summary)

    print(json.dumps(summary, indent=2))
    return out_session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True, help="Existing E3 anchor-selection session")
    p.add_argument("--manual-labels", required=True, help="manual_anchor_labels.csv")
    p.add_argument("--server-url", default="http://127.0.0.1:8899")
    p.add_argument("--server-endpoint", default="/infer")
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-sec", type=int, default=240)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
