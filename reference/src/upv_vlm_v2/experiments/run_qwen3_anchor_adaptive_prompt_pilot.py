"""Adaptive Qwen3-VL prompt pilot for UPV anchor contact scoring.

This is intentionally small and offline: it calls an already-running Qwen3
server on saved source_A*.png crops only. It does not start/load models and
does not touch robot, live camera, RTDE, Arduino, clamp, Pundit, or E45
hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from upv_vlm_v2.experiments.run_qwen3_anchor_pilot_compare import (
    BASELINE_ITERATION_DIR,
    _anchor_sort_key,
    _baseline_pilot_metrics,
    _extract_response,
    _get_health,
    _png_dimensions,
    _post_json,
    _select_case,
    _strict_gate,
    _write_csv,
    _write_json,
    compute_metrics,
    discover_pilot_anchors,
)


SERVER_URL_DEFAULT = "http://127.0.0.1:8898"
OUTPUT_ROOT_DEFAULT = "outputs/debug_anchor_prompt_eval"


PROMPT_SCHEMA = """Return JSON only. No prose, no markdown fences. Use exactly this schema and key order:
{{
"anchor_id": "{anchor_id}",
"top_usable": true,
"bottom_usable": true,
"overall_usable": true,
"score": 0,
"top_defects": ["none"],
"bottom_defects": ["none"],
"reason": "short physical-contact reason"
}}

overall_usable MUST equal top_usable AND bottom_usable.
If top_usable=false OR bottom_usable=false, score MUST be <= 35.
Only return score >= 70 if BOTH sides are usable.
"""


INITIAL_RULES = """Use a strict but local physical-contact standard.

Inspect a contact band around the yellow dashed line in both panels. Ignore harmless material color, grain, pores, or rough texture that is not at the contact boundary. Reject any physical obstruction or shape defect at the contact boundary: mortar/crust, raised blob, debris, protruding chip, missing/jagged edge, ridge, void, crack opening, splinter, or material step that prevents a flat circular probe from seating.

Both panels must be clean enough for a flat UPV transducer. If either side is blocked, jagged, protruding, or uncertain, set that side false, overall_usable=false, and score <= 30. Only assign score >= 70 when both side booleans are true.
"""


def _base_prompt(anchor_id: str, image_path: Path, rules: str) -> tuple[str, str]:
    width, height = _png_dimensions(image_path)
    if height > width:
        panel_text = (
            "The image contains two contact-crop panels.\n"
            "The TOP panel is contact side 1.\n"
            "The BOTTOM panel is contact side 2."
        )
        orientation = "vertical_top_bottom"
    else:
        panel_text = (
            "The image contains two contact-crop panels.\n"
            "The LEFT panel is contact side 1.\n"
            "The RIGHT panel is contact side 2."
        )
        orientation = "side_by_side_left_right"

    prompt = f"""You are judging ONE candidate UPV transducer anchor from a saved inspection image.

{panel_text}

The yellow dashed line marks the intended physical contact boundary for a flat circular UPV transducer. Judge whether BOTH contact regions are physically suitable for stable flat transducer contact.

{PROMPT_SCHEMA}

{rules}
"""
    return prompt.format(anchor_id=anchor_id), orientation


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_response(parsed: dict[str, Any] | None, anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    from upv_vlm_v2.experiments.run_qwen3_anchor_pilot_compare import _normalize

    return _normalize(parsed, anchor_id)


def _false_good_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("manual_is_bad")
        and (float(row.get("score") or 0.0) >= 70.0 or bool(row.get("overall_usable")))
    ]


def _false_bad_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("manual_is_good")
        and (float(row.get("score") or 0.0) < 70.0 or not bool(row.get("overall_usable")))
    ]


def _reason_terms(rows: list[dict[str, Any]]) -> dict[str, int]:
    terms = {
        "clean_or_none": 0,
        "raised_blob": 0,
        "mortar_crust": 0,
        "chip_jagged": 0,
        "debris": 0,
        "uncertain": 0,
    }
    for row in rows:
        text = " ".join(
            str(row.get(key, ""))
            for key in ("reason", "top_defects", "bottom_defects")
        ).lower()
        if any(tok in text for tok in ("none", "clean", "suitable", "usable")):
            terms["clean_or_none"] += 1
        if any(tok in text for tok in ("raised", "blob", "protrud")):
            terms["raised_blob"] += 1
        if any(tok in text for tok in ("mortar", "crust", "residue")):
            terms["mortar_crust"] += 1
        if any(tok in text for tok in ("chip", "jagged", "broken", "missing")):
            terms["chip_jagged"] += 1
        if any(tok in text for tok in ("debris", "dust", "loose")):
            terms["debris"] += 1
        if "uncertain" in text:
            terms["uncertain"] += 1
    return terms


def _make_next_rules(
    *,
    iteration_index: int,
    metrics: dict[str, Any],
    false_good: list[dict[str, Any]],
    false_bad: list[dict[str, Any]],
) -> tuple[str, str]:
    terms = _reason_terms(false_good)
    rationale = [
        f"Iteration {iteration_index} observed {len(false_good)} false-good anchors and {len(false_bad)} false-bad anchors.",
        f"Reason terms among false-good anchors: {terms}.",
    ]

    rules = [
        "Use an adaptive conservative UPV-contact audit standard tuned for previous Qwen3 failures.",
        "",
        "Step 1: Find the yellow dashed line in each panel. Judge only the narrow physical contact band centered on that line, not the whole crop.",
        "Step 2: For each panel, decide whether a flat circular transducer can seat on a continuous, unobstructed, approximately planar material boundary.",
        "Step 3: A single failed side makes the whole anchor unusable.",
        "",
        "Score bands are mandatory:",
        "- 0-30: any physical defect near the dashed line, including one bad side.",
        "- 40-60: uncertain visibility, partial obstruction, or marginal support.",
        "- 70-85: both sides are usable with only harmless flat texture.",
        "- 90-100: exceptionally clean, flat, continuous, and open on both sides.",
        "",
        "Do not give score >=70 unless you can name visible evidence that BOTH contact sides are clear at the dashed line.",
    ]

    if false_good:
        rules += [
            "",
            "Previous Qwen3 outputs were too permissive and called manually bad anchors clean. Correct that failure now.",
            "Do NOT write 'none' for defects merely because most of the crop looks clean. Inspect exactly where the yellow dashed line meets the material.",
            "Reject edge discontinuity: a chipped, notched, jagged, missing, broken, or uneven material boundary at the dashed line is a physical contact failure.",
            "Reject attached material: mortar, crust, residue, raised blob, protrusion, loose debris, or a material step near the dashed line is a physical contact failure.",
        ]
        if terms["clean_or_none"]:
            rules.append("If your first impression is 'clean', actively search for subtle boundary breaks, raised edges, and side-specific obstructions before setting usable=true.")
        if terms["raised_blob"]:
            rules.append("Raised blobs or protrusions touching either dashed contact band must force that side false and score <=30.")
        if terms["mortar_crust"]:
            rules.append("Mortar/crust/residue at the dashed line must force the affected side false even if the brick color elsewhere is normal.")
        if terms["chip_jagged"]:
            rules.append("Jagged/chipped/missing corners or edge steps at the dashed line are not harmless rough texture; they are contact failures.")
    if false_bad:
        rules += [
            "",
            "Some good anchors were over-rejected. Preserve normal material roughness: flat brick pores, timber grain, and color variation are acceptable when they do not interrupt or raise the contact boundary.",
        ]
    else:
        rules += [
            "",
            "No false-bad anchors were observed, so stay conservative against bad-contact defects while still allowing normal flat material texture.",
        ]

    if iteration_index >= 3 and false_good:
        rules += [
            "",
            "Final binary gate: before JSON, ask whether side 1 and side 2 are BOTH definitely suitable. If either answer is not definitely yes, set that side false, overall_usable=false, and score <=30.",
            "The reason must name the visible evidence near the dashed line. Shallow reasons such as 'both sides are clean' are allowed only when no boundary defects are visible.",
        ]

    return "\n".join(rules), "\n".join(rationale)


def _run_iteration(
    *,
    iter_dir: Path,
    iteration_index: int,
    rules: str,
    server_url: str,
    timeout_sec: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    iter_dir.mkdir(parents=True, exist_ok=True)
    anchors = discover_pilot_anchors()
    # Save a representative prompt with A1 substituted; per-anchor prompts hold exact IDs/orientations.
    sample_prompt, _ = _base_prompt("A1", anchors[0].image_path, rules)
    (iter_dir / "prompt_used.txt").write_text(sample_prompt, encoding="utf-8")

    per_anchor: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for anchor in anchors:
        anchor_dir = iter_dir / anchor.dataset / anchor.case_id / anchor.anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)
        prompt, orientation = _base_prompt(anchor.anchor_id, anchor.image_path, rules)
        (anchor_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        payload = {
            "image_path": str(anchor.image_path),
            "prompt_text": prompt,
            "prompt_version": f"qwen3_adaptive_pilot_iter_{iteration_index:02d}",
            "max_new_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "output_path": str(anchor_dir / "raw_response.txt"),
        }
        t0 = time.perf_counter()
        try:
            response = _post_json(server_url.rstrip("/") + "/infer", payload, int(timeout_sec))
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            raw_text, parsed, parse_err = _extract_response(response)
            normalized, parse_ok, norm_err = _normalize_response(parsed, anchor.anchor_id)
            parse_error = "" if parse_ok else (norm_err or parse_err)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            response = {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}
            raw_text = ""
            normalized = {"anchor_id": anchor.anchor_id, "parse_ok": False, "score": "", "overall_usable": False}
            parse_error = str(exc)
        width, height = _png_dimensions(anchor.image_path)
        normalized.update(
            {
                "dataset": anchor.dataset,
                "case_id": anchor.case_id,
                "anchor_id": anchor.anchor_id,
                "manual_label_raw": anchor.label_raw,
                "manual_is_good": anchor.manual_is_good,
                "manual_is_bad": anchor.manual_is_bad,
                "is_manual_best": anchor.is_manual_best,
                "strict_gate_passed": _strict_gate(normalized),
                "orientation_prompt": orientation,
                "image_width": width,
                "image_height": height,
                "elapsed_ms": elapsed_ms,
                "parse_error": parse_error,
                "image_path": str(anchor.image_path),
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
        grouped.setdefault((anchor.dataset, anchor.case_id), []).append(row)

    anchor_fields = [
        "dataset",
        "case_id",
        "anchor_id",
        "manual_label_raw",
        "manual_is_good",
        "manual_is_bad",
        "is_manual_best",
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
        "orientation_prompt",
        "image_width",
        "image_height",
        "elapsed_ms",
        "image_path",
        "raw_response_path",
        "parsed_response_path",
        "server_response_path",
    ]
    _write_csv(iter_dir / "per_anchor_results.csv", per_anchor, anchor_fields)

    per_case: list[dict[str, Any]] = []
    for (dataset, case_id), rows in sorted(grouped.items()):
        decision = _select_case(rows)
        case_row = {
            "dataset": dataset,
            "case_id": case_id,
            "anchors_attempted": len(rows),
            "anchors_parsed": sum(1 for row in rows if row.get("parse_ok")),
            **decision,
        }
        per_case.append(case_row)
        _write_json(iter_dir / dataset / case_id / "case_decision.json", case_row)
    _write_csv(
        iter_dir / "per_case_decisions.csv",
        per_case,
        [
            "dataset",
            "case_id",
            "anchors_attempted",
            "anchors_parsed",
            "selected_anchor_id",
            "selected_score",
            "selected_good",
            "selected_bad",
            "manual_good_exists",
            "no_safe_when_good_exists",
            "manual_best_anchor_id",
            "manual_best_match",
        ],
    )

    metrics = compute_metrics(per_anchor, per_case)
    _write_json(iter_dir / "iteration_metrics.json", metrics)
    _write_failure_review(iter_dir / "failure_review.md", per_anchor, metrics)
    return per_anchor, per_case, metrics


def _write_failure_review(path: Path, per_anchor: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    false_good = _false_good_rows(per_anchor)
    false_bad = _false_bad_rows(per_anchor)
    lines = [
        "# Failure Review",
        "",
        f"combined_false_good_rate: {metrics['combined_false_good_rate']}",
        f"combined_false_bad_rate: {metrics['combined_false_bad_rate']}",
        "",
        "## False-good anchors",
    ]
    if not false_good:
        lines.append("- None")
    for row in false_good:
        lines.append(
            f"- {row['dataset']} {row['case_id']} {row['anchor_id']} label={row['manual_label_raw']} "
            f"score={row.get('score')} overall={row.get('overall_usable')} "
            f"dims={row.get('image_width')}x{row.get('image_height')} image={row.get('image_path')} "
            f"reason={row.get('reason')}"
        )
        lines.append(f"  raw_response: {row.get('raw_response_path')}")
    lines += ["", "## False-bad anchors"]
    if not false_bad:
        lines.append("- None")
    for row in false_bad:
        lines.append(
            f"- {row['dataset']} {row['case_id']} {row['anchor_id']} label={row['manual_label_raw']} "
            f"score={row.get('score')} overall={row.get('overall_usable')} "
            f"dims={row.get('image_width')}x{row.get('image_height')} image={row.get('image_path')} "
            f"reason={row.get('reason')}"
        )
        lines.append(f"  raw_response: {row.get('raw_response_path')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row.get("selected_bad_rate") or 0.0),
        -float(row.get("selected_good_rate") or 0.0),
        float(row.get("combined_false_good_rate") or 0.0),
        float(row.get("combined_false_bad_rate") or 0.0),
        len(row.get("no_safe_when_good_exists") or []),
    )


def _write_final_report(
    *,
    out_dir: Path,
    health: dict[str, Any],
    iteration_rows: list[dict[str, Any]],
    adaptive_log: list[str],
    best_iter: int,
) -> None:
    report_path = Path("docs/dev_history/qwen3_8b_adaptive_prompt_pilot_20260528.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3 8B Adaptive Prompt Pilot - 2026-05-28",
        "",
        "## Scope",
        "",
        "This was a true sequential adaptive prompt pilot on saved baseline `source_A*.png` images only. It used the already-running Qwen3 server and did not start/load a model.",
        "",
        "No robot, live camera, RTDE, Arduino, clamp, Pundit/UPV, full E45 collection, or live pipeline execution was run.",
        "",
        "## Qwen3 server health",
        "",
        "```json",
        json.dumps(health, indent=2),
        "```",
        "",
        "## Pilot size",
        "",
        "- Cases: 4",
        "- Anchor images: 18",
        "- E45: `session_20260528_035248`, `session_20260528_042004`",
        "- E3: `case_003_brick_03_chipped_jagged`, `case_004_brick_04_debris_obstruction`",
        "",
        "## Adaptive prompt log",
        "",
        *adaptive_log,
        "",
        "## Metrics table",
        "",
        "| iteration | selected_good_rate | selected_bad_rate | E3 selected_good_rate | E45 selected_good_rate | combined false-good | combined false-bad | no_safe_when_good_exists |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in iteration_rows:
        lines.append(
            f"| {row['iteration']} | {row['selected_good_rate']} | {row['selected_bad_rate']} | "
            f"{row['E3_selected_good_rate']} | {row['E45_selected_good_rate']} | "
            f"{row['combined_false_good_rate']} | {row['combined_false_bad_rate']} | "
            f"{row['no_safe_when_good_exists_count']} |"
        )
    best_dir = out_dir / f"iteration_{best_iter:02d}"
    best_prompt = (best_dir / "prompt_used.txt").read_text(encoding="utf-8")
    final_failure = (best_dir / "failure_review.md").read_text(encoding="utf-8")
    lines += [
        "",
        f"## Best iteration",
        "",
        f"Best iteration by the requested priority rule: `iteration_{best_iter:02d}`",
        "",
        "## Best prompt",
        "",
        "```text",
        best_prompt,
        "```",
        "",
        "## Remaining failures in best iteration",
        "",
        final_failure,
        "",
        "## Baseline comparison",
        "",
        "Previous Qwen3 corrected hybrid pilot:",
        "",
        "- selected_good_rate: 100%",
        "- selected_bad_rate: 0%",
        "- combined false-good rate: 100%",
        "- combined false-bad rate: 0%",
        "",
        "Current best Qwen2.5/Qwen32 full-set baseline from Claude iteration 1:",
        "",
        "- selected_good_rate: 88.9%",
        "- selected_bad_rate: 0.0%",
        "- E3 selected_good_rate: 100.0%",
        "- E45 selected_good_rate: 75.0%",
        "- per-anchor combined false-good: 26.7%",
        "- per-anchor combined false-bad/false-reject: 15.4%",
        "",
    ]
    best_row = next(row for row in iteration_rows if int(row["iteration"]) == best_iter)
    if float(best_row["combined_false_good_rate"]) < 1.0:
        lines.append("The adaptive Qwen3 prompt improved over the initial Qwen3 prompt on false-good rate.")
    else:
        lines.append("The adaptive Qwen3 prompt did not improve over the initial Qwen3 prompt on false-good rate.")
    if float(best_row["combined_false_good_rate"]) <= 0.267 and float(best_row["combined_false_bad_rate"]) <= 0.154:
        lines.append("Qwen3 is competitive with the current Qwen2.5/Qwen32 baseline on this pilot.")
    else:
        lines.append("Qwen3 is not better than the current Qwen2.5/Qwen32 baseline on this pilot.")
    if float(best_row["combined_false_good_rate"]) <= 0.35 and float(best_row["selected_bad_rate"]) == 0.0:
        lines.append("Full 18-case testing is reasonable as a next offline step.")
    else:
        lines.append("Full 18-case testing is not recommended yet; improve the prompt or visual input first.")
    lines += [
        "",
        "## Output root",
        "",
        f"`{out_dir}`",
        "",
        "## Safety confirmation",
        "",
        "No robot/live camera/RTDE/Arduino/clamp/Pundit was run.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_adaptive(args: argparse.Namespace) -> Path:
    server_url = str(args.server_url).rstrip("/")
    health = _get_health(server_url)
    if not health.get("ok") or not health.get("model_loaded"):
        raise RuntimeError(f"Qwen3 server is not ready: {json.dumps(health, indent=2)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"qwen3_8b_adaptive_prompt_pilot_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "server_health.json", health)

    adaptive_log: list[str] = []
    rules = INITIAL_RULES
    iteration_rows: list[dict[str, Any]] = []
    best_iter = 1

    for iteration in range(1, 6):
        iter_dir = out_dir / f"iteration_{iteration:02d}"
        per_anchor, _per_case, metrics = _run_iteration(
            iter_dir=iter_dir,
            iteration_index=iteration,
            rules=rules,
            server_url=server_url,
            timeout_sec=int(args.timeout_sec),
            max_new_tokens=int(args.max_new_tokens),
        )
        false_good = _false_good_rows(per_anchor)
        false_bad = _false_bad_rows(per_anchor)
        row = {
            "iteration": iteration,
            **metrics,
            "false_good_count": len(false_good),
            "false_bad_count": len(false_bad),
            "no_safe_when_good_exists_count": len(metrics.get("no_safe_when_good_exists") or []),
        }
        iteration_rows.append(row)
        adaptive_log.append(
            f"- Iteration {iteration}: selected_good={metrics['selected_good_rate']}, "
            f"selected_bad={metrics['selected_bad_rate']}, false_good={metrics['combined_false_good_rate']}, "
            f"false_bad={metrics['combined_false_bad_rate']}."
        )
        if iteration < 5:
            rules, rationale = _make_next_rules(
                iteration_index=iteration,
                metrics=metrics,
                false_good=false_good,
                false_bad=false_bad,
            )
            adaptive_log.append(f"  Next prompt rationale: {rationale}")

    metric_fields = [
        "iteration",
        "selected_good_rate",
        "selected_bad_rate",
        "E3_selected_good_rate",
        "E45_selected_good_rate",
        "no_safe_anchor_cases",
        "no_safe_when_good_exists",
        "E3_manual_best_match_rate",
        "E3_false_good_rate",
        "E45_false_good_rate",
        "combined_false_good_rate",
        "E3_false_bad_rate",
        "E45_false_bad_rate",
        "combined_false_bad_rate",
        "false_good_count",
        "false_bad_count",
        "no_safe_when_good_exists_count",
    ]
    serializable_rows: list[dict[str, Any]] = []
    for row in iteration_rows:
        serializable_rows.append(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        )
    _write_csv(out_dir / "summary_iteration_metrics.csv", serializable_rows, metric_fields)
    (out_dir / "adaptive_prompt_log.md").write_text("\n".join(adaptive_log) + "\n", encoding="utf-8")

    best_row = sorted(iteration_rows, key=_best_key)[0]
    best_iter = int(best_row["iteration"])
    best_dir = out_dir / f"iteration_{best_iter:02d}"
    shutil.copyfile(best_dir / "per_anchor_results.csv", out_dir / "best_iteration_per_anchor_results.csv")
    shutil.copyfile(best_dir / "per_case_decisions.csv", out_dir / "best_iteration_per_case_decisions.csv")
    shutil.copyfile(best_dir / "prompt_used.txt", out_dir / "best_prompt.txt")
    shutil.copyfile(best_dir / "failure_review.md", out_dir / "final_failure_review.md")
    _write_json(
        out_dir / "baseline_reference_metrics.json",
        {
            "initial_qwen3_pilot": {
                "selected_good_rate": 1.0,
                "selected_bad_rate": 0.0,
                "combined_false_good_rate": 1.0,
                "combined_false_bad_rate": 0.0,
            },
            "qwen32_claude_iteration_1_full_set": {
                "selected_good_rate": 0.889,
                "selected_bad_rate": 0.0,
                "E3_selected_good_rate": 1.0,
                "E45_selected_good_rate": 0.75,
                "combined_false_good_rate": 0.267,
                "combined_false_bad_rate": 0.154,
            },
            "qwen32_pilot_subset": _baseline_pilot_metrics(),
        },
    )
    _write_final_report(
        out_dir=out_dir,
        health=health,
        iteration_rows=iteration_rows,
        adaptive_log=adaptive_log,
        best_iter=best_iter,
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8898")
    parser.add_argument("--output-root", default="outputs/debug_anchor_prompt_eval")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = run_adaptive(args)
    rows = _read_csv(out_dir / "summary_iteration_metrics.csv")
    best_metrics = _read_csv(out_dir / "best_iteration_per_case_decisions.csv")
    # Re-read best row from summary by matching best prompt folder copy source: sort with same rule.
    raw_rows = []
    for row in rows:
        parsed = dict(row)
        for key in (
            "iteration",
            "selected_good_rate",
            "selected_bad_rate",
            "combined_false_good_rate",
            "combined_false_bad_rate",
            "no_safe_when_good_exists_count",
        ):
            try:
                parsed[key] = float(parsed[key])
            except Exception:
                pass
        raw_rows.append(parsed)
    best_row = sorted(raw_rows, key=_best_key)[0]
    print(f"output_root: {out_dir}")
    print("iteration metrics table:")
    for row in rows:
        print(
            f"  iter {row['iteration']}: selected_good={row['selected_good_rate']} "
            f"selected_bad={row['selected_bad_rate']} false_good={row['combined_false_good_rate']} "
            f"false_bad={row['combined_false_bad_rate']}"
        )
    print(f"best_iteration: iteration_{int(float(best_row['iteration'])):02d}")
    print(f"best_prompt_path: {out_dir / 'best_prompt.txt'}")
    print(f"selected_good_rate: {best_row['selected_good_rate']}")
    print(f"selected_bad_rate: {best_row['selected_bad_rate']}")
    print(f"combined_false_good_rate: {best_row['combined_false_good_rate']}")
    print(f"combined_false_bad_rate: {best_row['combined_false_bad_rate']}")
    rec = "expand" if float(best_row["combined_false_good_rate"]) <= 0.35 and float(best_row["selected_bad_rate"]) == 0.0 else "do not expand"
    print(f"recommendation: {rec}")
    print("final_report: docs/dev_history/qwen3_8b_adaptive_prompt_pilot_20260528.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
