"""Run Qwen3-VL pilot on magenta-line/red-zone contact crop variants.

Offline only: this script transforms saved source_A*.png images, calls an
already-running Qwen3 server, and writes comparison artifacts. It does not run
robot, live camera, RTDE, Arduino, clamp, Pundit/UPV, or E45 collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from upv_vlm_v2.experiments.prepare_qwen_anchor_layout_variants import render_layout
from upv_vlm_v2.experiments.run_qwen3_anchor_pilot_compare import (
    _anchor_sort_key,
    _as_bool,
    _baseline_pilot_metrics,
    _extract_response,
    _get_health,
    _normalize,
    _post_json,
    _safe_float,
    _select_case,
    _strict_gate,
    _write_csv,
    _write_json,
    compute_metrics,
    discover_pilot_anchors,
)


SERVER_URL_DEFAULT = "http://127.0.0.1:8898"
OUTPUT_ROOT_DEFAULT = "outputs/debug_anchor_prompt_eval"
LAYOUT_NAME = "L6_magenta_line_only"


PROMPT_TEMPLATE = """You are an automated visual quality inspector for UPV transducer contact placement.

The image contains two contact-crop panels.
The LEFT panel is contact side 1.
The RIGHT panel is contact side 2.

A bright MAGENTA line marks the intended physical contact boundary.

Judge only the material region near the magenta line.

A contact side FAILS if mortar, crust, raised material, debris, protruding chip, jagged missing edge, ridge, void, crack opening, splinter, or material step touches/crosses the magenta contact line region or blocks flat probe seating near the magenta line.

A contact side PASSES if the marked zone is clear enough for stable flat seating of a circular UPV transducer. Ignore harmless color variation, pores, grain, or flat rough texture away from the marked zone.

Return JSON only. No markdown. No extra text.

Use this exact schema:
{{
"anchor_id": "{anchor_id}",
"visible_contact_observation": "one short sentence describing the marked zone",
"top_usable": true,
"bottom_usable": true,
"overall_usable": true,
"score": 0,
"top_defects": ["none"],
"bottom_defects": ["none"],
"reason": "short physical-contact reason"
}}

Rules:

* overall_usable must equal top_usable AND bottom_usable.
* If either side fails, score must be <= 35.
* If uncertain, score must be <= 60.
* Score >= 70 only when both sides are clearly usable.
* Do not mark an anchor usable just because it is the least bad.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _geometry_for_source(source_path: Path, anchor_id: str) -> dict[str, Any]:
    """Find optional anchor crop geometry metadata for a saved source crop."""
    for parent in [source_path.parent, *source_path.parents]:
        candidate = parent / "anchor_crop_geometry_debug.json"
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return data.get("anchors", {}).get(anchor_id, {})
            except Exception:
                return {}
    # E45 layout: source is under clean_single_anchor_inputs, metadata is one level up.
    candidate = source_path.parent.parent / "anchor_crop_geometry_debug.json"
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return data.get("anchors", {}).get(anchor_id, {})
        except Exception:
            return {}
    return {}


def _transform_images(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transformed_rows: list[dict[str, Any]] = []
    layout_infos: list[dict[str, Any]] = []
    image_root = out_dir / "transformed_images"
    for anchor in discover_pilot_anchors():
        geometry = _geometry_for_source(anchor.image_path, anchor.anchor_id)
        transformed_path = image_root / anchor.dataset / anchor.case_id / f"source_{anchor.anchor_id}_{LAYOUT_NAME}.png"
        info = render_layout(
            LAYOUT_NAME,
            source_path=anchor.image_path,
            out_path=transformed_path,
            geometry=geometry,
        )
        layout_infos.append(
            {
                "dataset": anchor.dataset,
                "case_id": anchor.case_id,
                "anchor_id": anchor.anchor_id,
                "source_image_path": str(anchor.image_path),
                "transformed_image_path": str(transformed_path),
                **info,
            }
        )
        transformed_rows.append(
            {
                "dataset": anchor.dataset,
                "case_id": anchor.case_id,
                "anchor_id": anchor.anchor_id,
                "image_path": transformed_path,
                "source_image_path": anchor.image_path,
                "manual_label_raw": anchor.label_raw,
                "manual_is_good": anchor.manual_is_good,
                "manual_is_bad": anchor.manual_is_bad,
                "is_manual_best": anchor.is_manual_best,
            }
        )
    return transformed_rows, layout_infos


def _run_inference(out_dir: Path, rows: list[dict[str, Any]], server_url: str, timeout_sec: int, max_new_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_anchor: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in rows:
        anchor_dir = out_dir / item["dataset"] / item["case_id"] / item["anchor_id"]
        anchor_dir.mkdir(parents=True, exist_ok=True)
        prompt = PROMPT_TEMPLATE.format(anchor_id=item["anchor_id"])
        (anchor_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        payload = {
            "image_path": str(item["image_path"]),
            "prompt_text": prompt,
            "prompt_version": "qwen3_magenta_zone_pilot_20260528",
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "output_path": str(anchor_dir / "raw_response.txt"),
        }
        t0 = time.perf_counter()
        try:
            response = _post_json(server_url.rstrip() + "/infer", payload, timeout_sec)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            raw_text, parsed, parse_err = _extract_response(response)
            normalized, parse_ok, norm_err = _normalize(parsed, str(item["anchor_id"]))
            parse_error = "" if parse_ok else (norm_err or parse_err)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            response = {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}
            raw_text = ""
            normalized = {"anchor_id": item["anchor_id"], "parse_ok": False, "score": "", "overall_usable": False}
            parse_error = str(exc)
        normalized.update(
            {
                "dataset": item["dataset"],
                "case_id": item["case_id"],
                "anchor_id": item["anchor_id"],
                "manual_label_raw": item["manual_label_raw"],
                "manual_is_good": item["manual_is_good"],
                "manual_is_bad": item["manual_is_bad"],
                "is_manual_best": item["is_manual_best"],
                "strict_gate_passed": _strict_gate(normalized),
                "layout": LAYOUT_NAME,
                "elapsed_ms": elapsed_ms,
                "parse_error": parse_error,
                "image_path": str(item["image_path"]),
                "source_image_path": str(item["source_image_path"]),
                "raw_response_path": str(anchor_dir / "raw_response.txt"),
                "parsed_response_path": str(anchor_dir / "parsed_response.json"),
                "server_response_path": str(anchor_dir / "server_response.json"),
            }
        )
        (anchor_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
        _write_json(anchor_dir / "server_response.json", response)
        _write_json(anchor_dir / "parsed_response.json", normalized)
        row = dict(normalized)
        row["top_defects"] = "; ".join(str(v) for v in normalized.get("top_defects", []))
        row["bottom_defects"] = "; ".join(str(v) for v in normalized.get("bottom_defects", []))
        row.pop("raw_parsed", None)
        per_anchor.append(row)
        grouped.setdefault((str(item["dataset"]), str(item["case_id"])), []).append(row)

    per_case: list[dict[str, Any]] = []
    for (dataset, case_id), case_rows in sorted(grouped.items()):
        decision = _select_case(case_rows)
        per_case.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "anchors_attempted": len(case_rows),
                "anchors_parsed": sum(1 for row in case_rows if row.get("parse_ok")),
                **decision,
            }
        )
    return per_anchor, per_case


def _write_failure_review(path: Path, per_anchor: list[dict[str, Any]]) -> None:
    lines = ["# Failure Review", ""]
    false_good = [
        row for row in per_anchor
        if row.get("manual_is_bad") and (float(row.get("score") or 0.0) >= 70 or bool(row.get("overall_usable")))
    ]
    false_bad = [
        row for row in per_anchor
        if row.get("manual_is_good") and (float(row.get("score") or 0.0) < 70 or not bool(row.get("overall_usable")))
    ]
    lines.append("## False-good anchors")
    if not false_good:
        lines.append("- None")
    for row in false_good:
        lines.append(
            f"- {row['dataset']} {row['case_id']} {row['anchor_id']}: label={row['manual_label_raw']} "
            f"score={row.get('score')} overall={row.get('overall_usable')} reason={row.get('reason')} image={row.get('image_path')}"
        )
    lines += ["", "## False-bad anchors"]
    if not false_bad:
        lines.append("- None")
    for row in false_bad:
        lines.append(
            f"- {row['dataset']} {row['case_id']} {row['anchor_id']}: label={row['manual_label_raw']} "
            f"score={row.get('score')} overall={row.get('overall_usable')} reason={row.get('reason')} image={row.get('image_path')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(out_dir: Path, health: dict[str, Any], metrics: dict[str, Any], layout_infos: list[dict[str, Any]]) -> None:
    report = Path("docs/dev_history/qwen3_magenta_zone_pilot_20260528.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    sample_paths = [str(row["transformed_image_path"]) for row in layout_infos[:4]]
    improved = float(metrics["combined_false_good_rate"]) < 0.8889
    recommend = improved and float(metrics["combined_false_good_rate"]) <= 0.35 and float(metrics["selected_bad_rate"]) == 0.0
    lines = [
        "# Qwen3 Magenta Line Only Pilot - 2026-05-28",
        "",
        "## Scope",
        "",
        "Offline-only saved-image pilot for a visual prompt layout with clean bright magenta contact lines only. No live pipeline or hardware was run.",
        "",
        "## Server health",
        "",
        "```json",
        json.dumps(health, indent=2),
        "```",
        "",
        "## Visual transformation",
        "",
        "- Layout: `L6_magenta_line_only`",
        "- Contact line: bright magenta, approximately RGB `(255, 0, 255)`",
        "- No red tolerance band or red outline is drawn",
        "- Original RGB crop content is preserved; mortar, chips, and debris remain visible.",
        "- If anchor geometry metadata exists it is used; otherwise the existing yellow guide line is detected from the saved crop and overdrawn only over its horizontal support.",
        "",
        "Sample transformed images:",
        *[f"- `{p}`" for p in sample_paths],
        "",
        "## Prompt",
        "",
        "```text",
        PROMPT_TEMPLATE,
        "```",
        "",
        "## Metrics",
        "",
        f"- selected_good_rate: {metrics['selected_good_rate']}",
        f"- selected_bad_rate: {metrics['selected_bad_rate']}",
        f"- E3 selected_good_rate: {metrics['E3_selected_good_rate']}",
        f"- E45 selected_good_rate: {metrics['E45_selected_good_rate']}",
        f"- combined false-good rate: {metrics['combined_false_good_rate']}",
        f"- combined false-bad rate: {metrics['combined_false_bad_rate']}",
        f"- E3 false-good rate: {metrics['E3_false_good_rate']}",
        f"- E45 false-good rate: {metrics['E45_false_good_rate']}",
        f"- E3 false-bad rate: {metrics['E3_false_bad_rate']}",
        f"- E45 false-bad rate: {metrics['E45_false_bad_rate']}",
        "",
        "## Baseline comparison",
        "",
        "Previous Qwen3 best pilot: selected_good_rate `100%`, selected_bad_rate `0%`, combined false-good `88.89%`, combined false-bad `0%`.",
        "",
        "Qwen2.5/Qwen32 best baseline: selected_good_rate `88.9%`, selected_bad_rate `0.0%`, combined false-good `26.7%`, combined false-bad `15.4%`.",
        "",
        f"Magenta-line-only improved false-good behavior over previous Qwen3: `{improved}`.",
        f"Full 18-case test recommended: `{recommend}`.",
        "",
        "## Result files",
        "",
        f"- Output root: `{out_dir}`",
        f"- Per-anchor results: `{out_dir / 'per_anchor_results.csv'}`",
        f"- Per-case decisions: `{out_dir / 'per_case_decisions.csv'}`",
        f"- Failure review: `{out_dir / 'failure_review.md'}`",
        "",
        "## Safety confirmation",
        "",
        "No robot/live camera/RTDE/Arduino/clamp/Pundit was run.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    server_url = str(args.server_url).rstrip("/")
    health = _get_health(server_url)
    if not health.get("ok") or not health.get("model_loaded"):
        raise RuntimeError(f"Qwen3 server is not ready: {json.dumps(health, indent=2)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"qwen3_magenta_zone_pilot_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "server_health.json", health)
    (out_dir / "prompt_used.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")

    transformed_rows, layout_infos = _transform_images(out_dir)
    _write_csv(
        out_dir / "layout_manifest.csv",
        layout_infos,
        [
            "dataset",
            "case_id",
            "anchor_id",
            "source_image_path",
            "transformed_image_path",
            "layout",
            "band_half_height_panel_px",
            "band_alpha",
            "contact_line_y_values",
            "used_geometry_metadata",
            "image_size",
        ],
    )
    per_anchor, per_case = _run_inference(out_dir, transformed_rows, server_url, int(args.timeout_sec), int(args.max_new_tokens))
    anchor_fields = [
        "dataset", "case_id", "anchor_id", "manual_label_raw", "manual_is_good", "manual_is_bad",
        "is_manual_best", "parse_ok", "strict_gate_passed", "score", "overall_usable", "top_usable",
        "bottom_usable", "top_defects", "bottom_defects", "reason", "parse_error", "layout",
        "elapsed_ms", "image_path", "source_image_path", "raw_response_path", "parsed_response_path",
        "server_response_path",
    ]
    _write_csv(out_dir / "per_anchor_results.csv", per_anchor, anchor_fields)
    _write_csv(
        out_dir / "per_case_decisions.csv",
        per_case,
        [
            "dataset", "case_id", "anchors_attempted", "anchors_parsed", "selected_anchor_id",
            "selected_score", "selected_good", "selected_bad", "manual_good_exists",
            "no_safe_when_good_exists", "manual_best_anchor_id", "manual_best_match",
        ],
    )
    metrics = compute_metrics(per_anchor, per_case)
    metrics["qwen3_previous_best_pilot"] = {
        "selected_good_rate": 1.0,
        "selected_bad_rate": 0.0,
        "combined_false_good_rate": 0.8889,
        "combined_false_bad_rate": 0.0,
    }
    metrics["qwen32_best_baseline"] = {
        "selected_good_rate": 0.889,
        "selected_bad_rate": 0.0,
        "combined_false_good_rate": 0.267,
        "combined_false_bad_rate": 0.154,
    }
    metrics["qwen32_pilot_subset"] = _baseline_pilot_metrics()
    _write_json(out_dir / "summary_metrics.json", metrics)
    _write_csv(
        out_dir / "summary_metrics.csv",
        [{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in metrics.items()],
        ["metric", "value"],
    )
    selected_lines = ["# Selected Anchor Metric Report", ""]
    for row in per_case:
        selected_lines.append(
            f"- {row['dataset']} {row['case_id']}: selected={row['selected_anchor_id']} "
            f"score={row['selected_score']} selected_good={row['selected_good']} selected_bad={row['selected_bad']}"
        )
    (out_dir / "selected_anchor_metric_report.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    _write_failure_review(out_dir / "failure_review.md", per_anchor)
    _write_report(out_dir, health, metrics, layout_infos)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8898")
    parser.add_argument("--output-root", default="outputs/debug_anchor_prompt_eval")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    return parser.parse_args()


def main() -> int:
    out_dir = run(parse_args())
    metrics = json.loads((out_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    layout_rows = _read_csv(out_dir / "layout_manifest.csv")
    improved = float(metrics["combined_false_good_rate"]) < 0.8889
    print(f"output_root: {out_dir}")
    print(f"selected_good_rate: {metrics['selected_good_rate']}")
    print(f"selected_bad_rate: {metrics['selected_bad_rate']}")
    print(f"combined_false_good_rate: {metrics['combined_false_good_rate']}")
    print(f"combined_false_bad_rate: {metrics['combined_false_bad_rate']}")
    print(f"improved_over_previous_qwen3: {improved}")
    if layout_rows:
        print(f"sample_transformed_image: {layout_rows[0]['transformed_image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
