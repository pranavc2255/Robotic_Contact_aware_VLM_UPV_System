#!/usr/bin/env python3
"""
E3 Qwen32 contact-crop grid baseline.

This baseline gives Qwen one existing A1-A5 contact-crop grid image per case
and asks it to choose the best UPV contact anchor. It differs from the proposed
single-anchor method because all five crop tiles are shown together and only
one Qwen call is made per case.

No robot, RTDE, Arduino, clamp, live camera, dataset capture, Qwen server code,
or E2 path is touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from upv_vlm_v2.experiments.run_e3_qwen32_shuffle_consensus_from_session import (
    anchor_sort_key,
    discover_cases,
    extract_response,
    load_manual_labels,
    post_json,
    write_csv,
    write_json,
)


OUTPUT_ROOT = Path("outputs/v2_experiments/e3_qwen32_contact_crop_grid_baseline")
PROMPT_VERSION = "e3_qwen32_contact_crop_grid_baseline_v1"

PROMPT_TEXT = """You are selecting one UPV contact anchor from a contact-crop grid image.

The image contains candidate contact-crop tiles labeled A1-A5. Each tile shows the TOP and BOTTOM contact regions for one source anchor.

Choose the anchor most suitable for stable ultrasonic pulse velocity probe contact.

Both the TOP and BOTTOM contact regions must be physically usable. A candidate is poor if either side is blocked, damaged, jagged, chipped, cracked, obstructed, non-planar, or edge-limited.

Penalize mortar, cement paste, gray/white crust, attached material, debris, protrusions, cracks, tape, nails, screws, splinters, loose material, unstable edges, chipped edges, jagged breaks, missing material, and blocked contact regions.

Normal brick fired-clay texture, timber grain/knots/saw marks, and concrete pores/aggregate are not defects by themselves.

Compare all candidates and choose the best available physically usable anchor. If all candidates are imperfect, choose the least bad usable anchor. Return NO_SAFE_ANCHOR only if every candidate is physically unusable.

Return JSON only. Do not include markdown fences or text outside JSON.

Use this schema:
{
  "anchors": [
    {
      "id": "A1",
      "usable": true,
      "score": 0,
      "reason": "short visible contact reason"
    }
  ],
  "ranking_best_to_worst": ["A1", "A2", "A3", "A4", "A5"],
  "best_anchor": "A1",
  "best_anchor_reason": "short visual reason",
  "no_safe_anchor": false
}
"""


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


def find_grid_image(case_dir: Path) -> Path:
    direct = case_dir / "combined_wide_contact_guided_grid_2col.png"
    if direct.exists():
        return direct
    candidates = sorted(
        case_dir.glob("pipeline_session/session_*/artifacts/04_anchor_selection/combined_wide_contact_guided_grid_2col.png")
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Missing combined_wide_contact_guided_grid_2col.png for {case_dir}")


def _score_from_anchor_list(parsed: dict[str, Any]) -> list[str]:
    anchors = parsed.get("anchors")
    scored: list[tuple[str, float]] = []
    if not isinstance(anchors, list):
        return []
    for item in anchors:
        if not isinstance(item, dict):
            continue
        aid = normalize_anchor_id(item.get("id") or item.get("anchor_id"))
        if aid not in {"A1", "A2", "A3", "A4", "A5"}:
            continue
        try:
            score = float(item.get("overall_score", item.get("score", -1)))
        except Exception:
            score = -1.0
        scored.append((aid, score))
    scored.sort(key=lambda kv: (-kv[1], anchor_sort_key(kv[0])))
    return [aid for aid, _ in scored]


def normalize_grid_response(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "parse_ok": False,
            "parse_error": "parsed response is not object",
            "best_anchor": "",
            "ranking_best_to_worst": [],
            "best_anchor_reason": "",
            "no_safe_anchor": False,
        }

    no_safe = bool(parsed.get("no_safe_anchor")) or normalize_anchor_id(parsed.get("best_anchor")) == "NO_SAFE_ANCHOR"
    best = normalize_anchor_id(parsed.get("best_anchor") or parsed.get("selected_anchor"))
    ranking_raw = parsed.get("ranking_best_to_worst")
    ranking: list[str] = []
    if isinstance(ranking_raw, list):
        for item in ranking_raw:
            aid = normalize_anchor_id(item)
            if aid in {"A1", "A2", "A3", "A4", "A5"} and aid not in ranking:
                ranking.append(aid)
    if not ranking:
        ranking = _score_from_anchor_list(parsed)
    if best in {"A1", "A2", "A3", "A4", "A5"} and best not in ranking:
        ranking.insert(0, best)
    if not best and ranking:
        best = ranking[0]

    ok = no_safe or best in {"A1", "A2", "A3", "A4", "A5"}
    return {
        "parse_ok": ok,
        "parse_error": "" if ok else f"invalid best_anchor: {best!r}",
        "best_anchor": "NO_SAFE_ANCHOR" if no_safe else best,
        "ranking_best_to_worst": ranking,
        "best_anchor_reason": str(parsed.get("best_anchor_reason") or parsed.get("reason") or ""),
        "no_safe_anchor": no_safe,
        "raw_parsed_json": parsed,
    }


def summarize(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parsed = sum(1 for r in rows if bool(r.get("parse_ok")))
    usable = sum(1 for r in rows if bool(r.get("top1_contact_usable")))
    bad = sum(1 for r in rows if bool(r.get("contact_unsuitable_selected")))
    dist: dict[str, int] = {}
    for r in rows:
        a = str(r.get("selected_anchor") or "NO_PARSE_RESULT")
        dist[a] = dist.get(a, 0) + 1
    return {
        "phase": "e3_qwen32_contact_crop_grid_baseline_from_existing_session",
        "n_cases": n,
        "n_errors": len(errors),
        "parse_success_count": parsed,
        "parse_success_rate": parsed / n if n else None,
        "top1_contact_usable_count": usable,
        "top1_contact_usable_rate": usable / n if n else None,
        "contact_unsuitable_selected_count": bad,
        "contact_unsuitable_selected_rate": bad / n if n else None,
        "selected_anchor_distribution": dist,
    }


def run(args: argparse.Namespace) -> Path:
    session_dir = Path(args.session)
    labels, manual_best = load_manual_labels(Path(args.manual_labels))
    cases = discover_cases(session_dir, "case_*")
    if args.case_id:
        wanted = {x.strip() for x in str(args.case_id).split(",") if x.strip()}
        cases = [case for case in cases if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in cases})
        if missing:
            raise ValueError(f"Requested case_id not found: {missing}")

    out_session = Path(args.output_root) / f"session_{now_stamp()}"
    out_session.mkdir(parents=True, exist_ok=False)
    infer_url = args.server_url.rstrip("/") + (args.server_endpoint if args.server_endpoint.startswith("/") else f"/{args.server_endpoint}")
    print(f"[QWEN] endpoint={infer_url}")

    write_json(
        out_session / "run_manifest.json",
        {
            "phase": "e3_qwen32_contact_crop_grid_baseline_from_existing_session",
            "input_session": str(session_dir),
            "manual_labels": str(args.manual_labels),
            "server_url": args.server_url,
            "server_endpoint": args.server_endpoint,
            "prompt_version": PROMPT_VERSION,
            "output_session": str(out_session),
            "input_representation": "combined_wide_contact_guided_grid_2col.png",
        },
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        case_out = out_session / "cases" / case.case_name
        case_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "case_name": case.case_name,
            "case_id": case.case_id,
            "case_index": case.case_index,
            "material": case.material,
            "condition": case.condition,
            "input_grid_path": str(case_out / "input_contact_crop_grid.png"),
            "parse_ok": False,
            "parse_error": "",
            "selected_anchor": "",
            "ranking_best_to_worst": "",
            "best_anchor_reason": "",
            "selected_manual_label": "",
            "manual_best_anchor": manual_best.get(case.case_id, ""),
            "top1_contact_usable": False,
            "contact_unsuitable_selected": False,
            "is_manual_best_match": False,
        }
        try:
            source_grid = find_grid_image(case.case_dir)
            input_grid = case_out / "input_contact_crop_grid.png"
            shutil.copy2(source_grid, input_grid)
            prompt_path = case_out / "grid_prompt.txt"
            prompt_path.write_text(PROMPT_TEXT, encoding="utf-8")
            server_path = case_out / "grid_server_response.json"
            raw_path = case_out / "grid_raw.txt"
            parsed_path = case_out / "grid_parsed.json"
            payload = {
                "image_path": str(input_grid),
                "prompt_text": PROMPT_TEXT,
                "prompt_version": PROMPT_VERSION,
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "output_path": str(raw_path),
            }
            response = post_json(infer_url, payload, int(args.timeout_sec))
            write_json(server_path, response)
            raw_text, parsed, parse_ok, parse_error = extract_response(response)
            raw_path.write_text(raw_text or "", encoding="utf-8")
            normalized = normalize_grid_response(parsed if parse_ok else None)
            if not parse_ok:
                normalized["parse_ok"] = False
                normalized["parse_error"] = parse_error
            write_json(parsed_path, normalized)

            best = str(normalized.get("best_anchor") or "")
            selected_label = labels.get((case.case_id, best), "") if normalized.get("parse_ok") else ""
            row.update(
                {
                    "parse_ok": bool(normalized.get("parse_ok")),
                    "parse_error": str(normalized.get("parse_error") or ""),
                    "selected_anchor": best if normalized.get("parse_ok") else "",
                    "ranking_best_to_worst": "|".join(normalized.get("ranking_best_to_worst") or []),
                    "best_anchor_reason": str(normalized.get("best_anchor_reason") or ""),
                    "selected_manual_label": selected_label,
                    "top1_contact_usable": bool(normalized.get("parse_ok") and is_usable_label(selected_label)),
                    "contact_unsuitable_selected": bool(normalized.get("parse_ok") and is_bad_label(selected_label)),
                    "is_manual_best_match": bool(normalized.get("parse_ok") and best == manual_best.get(case.case_id, "")),
                }
            )
            write_json(case_out / "case_result.json", row)
            print(f"[{idx}/{len(cases)}] {case.case_id}: selected={row['selected_anchor'] or 'NO_PARSE'} label={row['selected_manual_label'] or '-'} parse_ok={row['parse_ok']}")
        except Exception as exc:
            row["parse_error"] = repr(exc)
            errors.append({"case_name": case.case_name, "case_id": case.case_id, "error": repr(exc), "case_output_dir": str(case_out)})
            print(f"[{idx}/{len(cases)}] {case.case_id}: ERROR {exc!r}")
            if args.fail_fast:
                rows.append(row)
                write_csv(out_session / "master_contact_crop_grid_baseline_results.csv", rows)
                write_csv(out_session / "errors.csv", errors)
                raise
        rows.append(row)

    fields = [
        "case_name",
        "case_id",
        "case_index",
        "material",
        "condition",
        "input_grid_path",
        "parse_ok",
        "parse_error",
        "selected_anchor",
        "ranking_best_to_worst",
        "best_anchor_reason",
        "selected_manual_label",
        "manual_best_anchor",
        "top1_contact_usable",
        "contact_unsuitable_selected",
        "is_manual_best_match",
    ]
    write_csv(out_session / "master_contact_crop_grid_baseline_results.csv", rows, fields)
    write_csv(out_session / "errors.csv", errors)
    summary = summarize(rows, errors)
    summary.update({"input_session": str(session_dir), "manual_labels": str(args.manual_labels), "output_session": str(out_session)})
    write_json(out_session / "summary_overall.json", summary)
    print(json.dumps(summary, indent=2))
    return out_session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True)
    p.add_argument("--manual-labels", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8899")
    p.add_argument("--server-endpoint", default="/infer")
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-sec", type=int, default=240)
    p.add_argument("--case-id", default=None)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
