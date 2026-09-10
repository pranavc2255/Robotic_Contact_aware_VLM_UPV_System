"""Analyze offline Qwen anchor-contact prompt trials.

This module intentionally uses only saved per-anchor CSV/JSON artifacts. It
does not call models, cameras, robot APIs, Arduino, clamp, or UPV hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


E3_LABELS_DEFAULT = Path(
    "outputs/v2_experiments/e3_anchor_selection/session_20260522_131552/manual_labels/manual_anchor_labels.csv"
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "good", "acceptable", "usable"}
    return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_e3_manual_labels(path: Path = E3_LABELS_DEFAULT) -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(path):
        case_id = str(row.get("case_id") or "").strip()
        anchor_id = str(row.get("anchor_id") or "").strip()
        if not case_id or not anchor_id:
            continue
        labels[(case_id, anchor_id)] = {
            "manual_label": row.get("manual_label", ""),
            "manual_is_usable": _as_bool(row.get("is_usable")),
            "is_manual_best": _as_bool(row.get("is_manual_best")),
        }
    return labels


def summarize_iteration(iter_dir: Path, labels: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    labels = labels or {}
    per_anchor = _read_csv(iter_dir / "per_anchor_results.csv")
    per_case = _read_csv(iter_dir / "per_case_decisions.csv")

    total_images = len(per_anchor)
    parse_ok_count = sum(1 for r in per_anchor if _as_bool(r.get("parse_ok")))
    selected_cases = [r for r in per_case if r.get("selected_anchor_id") and r.get("selected_anchor_id") != "NO_SAFE_ANCHOR"]
    no_safe_cases = [r for r in per_case if r.get("selected_anchor_id") == "NO_SAFE_ANCHOR"]
    e3_cases = [r for r in per_case if r.get("dataset") == "E3"]
    e45_cases = [r for r in per_case if r.get("dataset") == "E45"]
    e45_anchor_rows = [r for r in per_anchor if r.get("dataset") == "E45"]
    e3_selected = [r for r in e3_cases if r.get("selected_anchor_id") != "NO_SAFE_ANCHOR"]
    e3_strict_usable_selected = [r for r in e3_selected if _as_bool(r.get("selected_strict_usable"))]

    bad_selected_count = 0
    selected_manual_bad: list[str] = []
    over_rejected_e3_count = 0
    no_safe_with_manual_usable: list[str] = []
    for row in per_case:
        if row.get("dataset") != "E3":
            continue
        cid = row.get("case_id", "")
        aid = row.get("selected_anchor_id", "")
        if aid and aid != "NO_SAFE_ANCHOR":
            lab = labels.get((cid, aid))
            if lab and not lab.get("manual_is_usable"):
                bad_selected_count += 1
                selected_manual_bad.append(f"{cid}:{aid}:{lab.get('manual_label')}")
        if aid == "NO_SAFE_ANCHOR":
            usable = [a for (case_id, a), lab in labels.items() if case_id == cid and lab.get("manual_is_usable")]
            if usable:
                over_rejected_e3_count += 1
                no_safe_with_manual_usable.append(f"{cid}: usable={','.join(sorted(usable))}")

    selected_score_below_70 = sum(
        1
        for r in selected_cases
        if r.get("selected_score") not in {"", None} and float(r.get("selected_score") or 0.0) < 70.0
    )
    e45_no_safe_count = sum(1 for r in e45_cases if r.get("selected_anchor_id") == "NO_SAFE_ANCHOR")
    e45_unusable_anchor_count = sum(1 for r in e45_anchor_rows if str(r.get("overall_usable")).strip().lower() == "false")
    e45_low_score_anchor_count = sum(
        1
        for r in e45_anchor_rows
        if r.get("score") not in {"", None} and float(r.get("score") or 0.0) < 70.0
    )

    prompt_info_path = iter_dir / "prompt_info.json"
    prompt_info = json.loads(prompt_info_path.read_text(encoding="utf-8")) if prompt_info_path.exists() else {}
    return {
        "iteration": prompt_info.get("iteration", iter_dir.name),
        "prompt_variant": prompt_info.get("prompt_variant", ""),
        "image_variant": prompt_info.get("image_variant", "saved_clean_source_crop"),
        "total_images": total_images,
        "total_cases": len(per_case),
        "parse_success_count": parse_ok_count,
        "parse_success_rate": round(parse_ok_count / total_images, 4) if total_images else 0.0,
        "strict_selected_usable_count_E3": len(e3_strict_usable_selected),
        "strict_selected_usable_rate_E3": round(len(e3_strict_usable_selected) / len(e3_cases), 4) if e3_cases else 0.0,
        "bad_selected_count": bad_selected_count,
        "no_safe_anchor_count": len(no_safe_cases),
        "e45_no_safe_anchor_count": e45_no_safe_count,
        "e45_unusable_anchor_count": e45_unusable_anchor_count,
        "e45_low_score_anchor_count": e45_low_score_anchor_count,
        "selected_score_below_70_count": selected_score_below_70,
        "over_rejected_e3_count": over_rejected_e3_count,
        "selected_manual_bad": selected_manual_bad,
        "no_safe_with_manual_usable": no_safe_with_manual_usable,
        "notes": prompt_info.get("notes", ""),
    }


def analyze_session(session_dir: Path) -> dict[str, Any]:
    labels = load_e3_manual_labels()
    rows: list[dict[str, Any]] = []
    for iter_dir in sorted(session_dir.glob("iteration_*")):
        if iter_dir.is_dir() and (iter_dir / "per_case_decisions.csv").exists():
            summary = summarize_iteration(iter_dir, labels)
            rows.append(summary)

    fields = [
        "iteration",
        "prompt_variant",
        "image_variant",
        "total_images",
        "total_cases",
        "parse_success_rate",
        "strict_selected_usable_rate_E3",
        "bad_selected_count",
        "no_safe_anchor_count",
        "e45_no_safe_anchor_count",
        "e45_unusable_anchor_count",
        "e45_low_score_anchor_count",
        "selected_score_below_70_count",
        "over_rejected_e3_count",
        "notes",
    ]
    _write_csv(session_dir / "summary_iteration_table.csv", rows, fields)

    best = None
    if rows:
        best = sorted(
            rows,
            key=lambda r: (
                float(r.get("bad_selected_count") or 0),
                -float(r.get("strict_selected_usable_rate_E3") or 0),
                float(r.get("over_rejected_e3_count") or 0),
                -float(r.get("e45_low_score_anchor_count") or 0),
                float(r.get("no_safe_anchor_count") or 0),
                -float(r.get("parse_success_rate") or 0),
            ),
        )[0]
        best_iter = session_dir / str(best["iteration"])
        prompt_path = best_iter / "prompt.txt"
        if prompt_path.exists():
            (session_dir / "best_prompt.txt").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
        _write_json(session_dir / "best_prompt_result_summary.json", best)

    failures = ["# Qwen Anchor Prompt Failure Cases", ""]
    for row in rows:
        failures.append(f"## {row['iteration']} - {row['prompt_variant']}")
        if row.get("selected_manual_bad"):
            failures.append("")
            failures.append("Manual-bad E3 selections:")
            for item in row["selected_manual_bad"]:
                failures.append(f"- {item}")
        if row.get("no_safe_with_manual_usable"):
            failures.append("")
            failures.append("E3 cases over-rejected despite manual usable anchors:")
            for item in row["no_safe_with_manual_usable"]:
                failures.append(f"- {item}")
        if not row.get("selected_manual_bad") and not row.get("no_safe_with_manual_usable"):
            failures.append("- No manual-bad E3 selections or over-rejected E3 cases recorded.")
        failures.append("")
    (session_dir / "failure_cases.md").write_text("\n".join(failures), encoding="utf-8")

    aggregate = {
        "session_dir": str(session_dir),
        "iterations": rows,
        "best": best,
        "manual_label_count": len(labels),
    }
    _write_json(session_dir / "analysis_summary.json", aggregate)
    return aggregate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, help="Prompt trial session directory.")
    args = ap.parse_args()
    result = analyze_session(Path(args.session))
    print(json.dumps(result.get("best", {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
