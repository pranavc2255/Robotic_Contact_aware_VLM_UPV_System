"""Analyze adaptive Qwen anchor prompt+layout search outputs.

Reads a session directory created by ``run_qwen_anchor_adaptive_prompt_search.py``
and produces a session-level ``failure_review.md`` plus a CSV/JSON breakdown of
bad/good anchor score behaviour per iteration.

No hardware. Only reads/writes saved CSV/JSON/MD files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _is_high_score(row: dict[str, str]) -> bool:
    v = row.get("score")
    if v in ("", None):
        return False
    try:
        return float(v) >= 70.0
    except Exception:
        return False


def analyze_session(session_dir: Path) -> dict[str, Any]:
    iter_dirs = sorted(p for p in session_dir.glob("iteration_*") if p.is_dir())
    iteration_rows: list[dict[str, Any]] = []
    failure_lines: list[str] = []
    failure_lines.append("# Adaptive Qwen anchor search — combined failure review")
    failure_lines.append("")

    for iter_dir in iter_dirs:
        anchors = _read_csv(iter_dir / "per_anchor_results.csv")
        metrics_path = iter_dir / "iteration_metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        except Exception:
            metrics = {}
        if not anchors:
            continue

        e3_bad = [r for r in anchors if r["dataset"] == "E3" and _as_bool(r.get("manual_is_bad"))]
        e45_bad = [r for r in anchors if r["dataset"] == "E45" and _as_bool(r.get("manual_is_bad"))]
        e3_good = [r for r in anchors if r["dataset"] == "E3" and _as_bool(r.get("manual_is_good"))]
        e45_good = [r for r in anchors if r["dataset"] == "E45" and _as_bool(r.get("manual_is_good"))]
        bad_total = e3_bad + e45_bad
        good_total = e3_good + e45_good

        false_good = [r for r in bad_total if _is_high_score(r)]
        false_reject = [r for r in good_total if not _as_bool(r.get("strict_gate_passed"))]

        bad_score_stats = _score_distribution([r.get("score") for r in bad_total])
        good_score_stats = _score_distribution([r.get("score") for r in good_total])

        iteration_rows.append(
            {
                "iteration_dir": iter_dir.name,
                "prompt_name": metrics.get("prompt_name", ""),
                "layout": metrics.get("layout", ""),
                "total_anchors": len(anchors),
                "bad_total": len(bad_total),
                "bad_scored_high": len(false_good),
                "bad_false_good_rate": round(len(false_good) / len(bad_total), 4) if bad_total else 0.0,
                "good_total": len(good_total),
                "good_rejected": len(false_reject),
                "good_false_reject_rate": round(len(false_reject) / len(good_total), 4) if good_total else 0.0,
                "bad_score_mean": bad_score_stats["mean"],
                "bad_score_min": bad_score_stats["min"],
                "bad_score_max": bad_score_stats["max"],
                "good_score_mean": good_score_stats["mean"],
                "good_score_min": good_score_stats["min"],
                "good_score_max": good_score_stats["max"],
            }
        )

        failure_lines.append(f"## {iter_dir.name}")
        failure_lines.append("")
        failure_lines.append(
            f"prompt=`{metrics.get('prompt_name','')}` layout=`{metrics.get('layout','')}` "
            f"bad_false_good={len(false_good)}/{len(bad_total)} "
            f"good_false_reject={len(false_reject)}/{len(good_total)} "
            f"bad_score_mean={bad_score_stats['mean']} good_score_mean={good_score_stats['mean']}"
        )
        failure_lines.append("")
        if false_good:
            failure_lines.append("**Bad anchors scored high (false-good):**")
            failure_lines.append("")
            failure_lines.append("| dataset | case | anchor | label | score | top | bot | overall | reason |")
            failure_lines.append("|---|---|---|---|---|---|---|---|---|")
            for r in false_good:
                failure_lines.append(
                    f"| {r['dataset']} | {r['case_id']} | {r['anchor_id']} | "
                    f"{r.get('manual_label_raw','')} | {r.get('score','')} | "
                    f"{r.get('top_usable','')} | {r.get('bottom_usable','')} | "
                    f"{r.get('overall_usable','')} | {r.get('reason','')[:90]} |"
                )
            failure_lines.append("")
        if false_reject:
            failure_lines.append("**Good anchors rejected (false-reject):**")
            failure_lines.append("")
            failure_lines.append("| dataset | case | anchor | label | score | top | bot | overall | reason |")
            failure_lines.append("|---|---|---|---|---|---|---|---|---|")
            for r in false_reject:
                failure_lines.append(
                    f"| {r['dataset']} | {r['case_id']} | {r['anchor_id']} | "
                    f"{r.get('manual_label_raw','')} | {r.get('score','')} | "
                    f"{r.get('top_usable','')} | {r.get('bottom_usable','')} | "
                    f"{r.get('overall_usable','')} | {r.get('reason','')[:90]} |"
                )
            failure_lines.append("")

    _write_csv(
        session_dir / "bad_good_score_distribution_per_iteration.csv",
        iteration_rows,
        list(iteration_rows[0].keys()) if iteration_rows else [],
    )
    (session_dir / "failure_review.md").write_text("\n".join(failure_lines), encoding="utf-8")

    aggregate = {
        "session_dir": str(session_dir),
        "iterations_analyzed": len(iteration_rows),
        "iteration_rows": iteration_rows,
    }
    _write_json(session_dir / "analysis_summary.json", aggregate)
    return aggregate


def _score_distribution(values: list[Any]) -> dict[str, float]:
    nums: list[float] = []
    for v in values:
        if v in ("", None):
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": round(sum(nums) / len(nums), 2),
        "min": round(min(nums), 2),
        "max": round(max(nums), 2),
        "count": len(nums),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, help="Adaptive search session dir.")
    args = ap.parse_args()
    out = analyze_session(Path(args.session))
    print(json.dumps({"session_dir": out["session_dir"], "iterations_analyzed": out["iterations_analyzed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
