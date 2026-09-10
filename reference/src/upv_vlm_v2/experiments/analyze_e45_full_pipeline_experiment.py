from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any


MASTER_FIELDS = [
    "session_id",
    "reading_index",
    "specimen_id",
    "material",
    "condition_label",
    "axis_mode",
    "timestamp_start",
    "timestamp_end",
    "attempt_count",
    "precheck_attempts",
    "pipeline_session_path",
    "requested_material",
    "pipeline_success",
    "failure_reason",
    "selected_object_class",
    "selected_candidate_id",
    "selected_anchor_id",
    "qwen_selected_anchor",
    "anchor_backend",
    "anchor_score",
    "anchor_count_generated",
    "anchor_count_metadata",
    "mask_path_length_mm",
    "depth_path_length_mm",
    "upv_path_length_mm",
    "clamp_opening_mm",
    "depth_valid",
    "depth_mask_disagreement_percent",
    "robot_moved",
    "motion_success",
    "clamp_moved",
    "arduino_available",
    "clamp_attempted",
    "clamp_success",
    "clamp_failure_reason",
    "release_attempted",
    "release_success",
    "home_returned",
    "home_success",
    "home_success_after_execution",
    "execution_success",
    "total_pipeline_timing_ms",
    "target_selection_timing_ms",
    "geometry_timing_ms",
    "anchor_selection_timing_ms",
    "path_length_timing_ms",
    "robot_plan_timing_ms",
    "execution_timing_ms",
    "approach_final_target_pose",
    "approach_final_actual_before",
    "approach_final_actual_after",
    "final_robot_pose_if_available",
    "tcp_position_error_mm",
    "tcp_orientation_error_deg",
    "stage_names",
    "stage_success_flags",
    "pre_capture_ready_for_capture",
    "ur_rtde_available",
    "arduino_available_pre_capture",
    "pre_capture_home_success",
    "precheck_attempt_count",
    "reading_source",
    "imported_from_session",
    "source_specimen_id",
    "source_reading_index",
    "target_specimen_id",
    "target_reading_index",
    "skipped_robot_execution_due_to_import",
    "reading_dir",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "success"}


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": "", "std": "", "min": "", "max": ""}
    if len(values) == 1:
        v = round(values[0], 6)
        return {"mean": v, "std": 0.0, "min": v, "max": v}
    return {
        "mean": round(mean(values), 6),
        "std": round(stdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _group(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return grouped


def _truth_rate(rows: list[dict[str, str]], key: str) -> Any:
    if not rows:
        return ""
    return round(sum(_boolish(row.get(key)) for row in rows) / len(rows), 6)


def _anchor_distribution(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("selected_anchor_id", "") for row in rows if row.get("selected_anchor_id"))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _summarize_group(group: list[dict[str, str]], keys: dict[str, str]) -> dict[str, Any]:
    exec_vals = [_float(r.get("execution_timing_ms")) for r in group]
    total_vals = [_float(r.get("total_pipeline_timing_ms")) for r in group]
    pos_vals = [_float(r.get("tcp_position_error_mm")) for r in group]
    anchor_counts = [_float(r.get("anchor_count_generated")) for r in group]
    exec_s = [v / 1000.0 for v in exec_vals if v is not None]
    total_s = [v / 1000.0 for v in total_vals if v is not None]
    pos = [v for v in pos_vals if v is not None]
    counts = [v for v in anchor_counts if v is not None]
    exec_stats = _stats(exec_s)
    total_stats = _stats(total_s)
    pos_stats = _stats(pos)
    count_stats = _stats(counts)
    return {
        **keys,
        "n_readings": len(group),
        "pipeline_success_rate": _truth_rate(group, "pipeline_success"),
        "robot_motion_success_rate": _truth_rate(group, "motion_success"),
        "clamp_success_rate": _truth_rate(group, "clamp_success"),
        "release_success_rate": _truth_rate(group, "release_success"),
        "home_success_rate": _truth_rate(group, "home_success_after_execution"),
        "mean_execution_time_s": exec_stats["mean"],
        "std_execution_time_s": exec_stats["std"],
        "mean_total_pipeline_time_s": total_stats["mean"],
        "std_total_pipeline_time_s": total_stats["std"],
        "mean_tcp_position_error_mm": pos_stats["mean"],
        "std_tcp_position_error_mm": pos_stats["std"],
        "selected_anchor_distribution": _anchor_distribution(group),
        "anchor_count_mean": count_stats["mean"],
        "anchor_count_min": count_stats["min"],
        "anchor_count_max": count_stats["max"],
    }


def analyze_rows(rows: list[dict[str, str]], session: Path) -> dict[str, Any]:
    tables = session / "02_tables"
    by_specimen: list[dict[str, Any]] = []
    by_material: list[dict[str, Any]] = []
    for (specimen_id, material), group in _group(rows, ["specimen_id", "material"]).items():
        by_specimen.append(_summarize_group(group, {"specimen_id": specimen_id, "material": material}))
    for (material,), group in _group(rows, ["material"]).items():
        by_material.append(_summarize_group(group, {"material": material}))

    summary_fields = [
        "specimen_id",
        "material",
        "n_readings",
        "pipeline_success_rate",
        "robot_motion_success_rate",
        "clamp_success_rate",
        "release_success_rate",
        "home_success_rate",
        "mean_execution_time_s",
        "std_execution_time_s",
        "mean_total_pipeline_time_s",
        "std_total_pipeline_time_s",
        "mean_tcp_position_error_mm",
        "std_tcp_position_error_mm",
        "selected_anchor_distribution",
        "anchor_count_mean",
        "anchor_count_min",
        "anchor_count_max",
    ]
    _write_csv(tables / "e4_robot_summary_by_specimen.csv", by_specimen, summary_fields)
    _write_csv(tables / "e4_robot_summary_by_material.csv", by_material, [f for f in summary_fields if f != "specimen_id"])
    comp_rows = _condition_comparison(rows)
    _write_csv(tables / "e45_condition_comparison.csv", comp_rows, [
        "scope",
        "material",
        "condition_label",
        "n_readings",
        "pipeline_success_rate",
        "robot_motion_success_rate",
        "clamp_success_rate",
        "home_success_rate",
        "selected_anchor_distribution",
    ])
    _write_figures(session, rows, by_specimen, by_material)
    _write_paper_assets(session, rows)
    summary = {
        "n_readings": len(rows),
        "n_specimens": len({r.get("specimen_id", "") for r in rows if r.get("specimen_id")}),
        "n_materials": len({r.get("material", "") for r in rows if r.get("material")}),
        "specimen_summary_rows": len(by_specimen),
        "material_summary_rows": len(by_material),
        "condition_comparison_rows": len(comp_rows),
        "manual_upv_entry": False,
    }
    _write_json(session / "05_summary" / "e45_session_summary.json", summary)
    _write_csv(session / "05_summary" / "e45_session_summary.csv", [{"metric": k, "value": v} for k, v in summary.items()], ["metric", "value"])
    _write_summary_md(session, summary)
    return summary


def _condition_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for keys, scope in [(["condition_label"], "overall"), (["material", "condition_label"], "material")]:
        for group_key, group in _group(rows, keys).items():
            material = group_key[0] if scope == "material" else ""
            condition = group_key[-1]
            out.append({
                "scope": scope,
                "material": material,
                "condition_label": condition,
                "n_readings": len(group),
                "pipeline_success_rate": _truth_rate(group, "pipeline_success"),
                "robot_motion_success_rate": _truth_rate(group, "motion_success"),
                "clamp_success_rate": _truth_rate(group, "clamp_success"),
                "home_success_rate": _truth_rate(group, "home_success_after_execution"),
                "selected_anchor_distribution": _anchor_distribution(group),
            })
    return out


def _write_paper_assets(session: Path, rows: list[dict[str, str]]) -> None:
    root = session / "04_paper_assets"
    rep = root / "representative_cycle_assets"
    rep.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    row = rows[0] if rows else None
    if row:
        reading_dir = Path(row.get("reading_dir", ""))
        for rel in ["pipeline_result.json", "reading_summary.json", "reading_request.json"]:
            src = reading_dir / rel
            copied = rep / rel
            if src.exists():
                copied.write_bytes(src.read_bytes())
                exists = True
            else:
                exists = False
            manifest_rows.append({"asset_role": rel, "source_path": str(src), "copied_path": str(copied), "exists": exists, "notes": ""})
    else:
        manifest_rows.append({"asset_role": "representative_cycle", "source_path": "", "copied_path": "", "exists": False, "notes": "no_readings_available"})
    _write_csv(root / "figure_asset_manifest.csv", manifest_rows, ["asset_role", "source_path", "copied_path", "exists", "notes"])


def _write_figures(session: Path, rows: list[dict[str, str]], by_specimen: list[dict[str, Any]], by_material: list[dict[str, Any]]) -> None:
    fig_dir = session / "03_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "e4_robot_position_error_by_specimen.png",
        "e4_robot_cycle_time_by_specimen.png",
        "e4_home_return_success_rate.png",
        "e4_clamp_success_rate.png",
        "e4_anchor_distribution_by_specimen.png",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _bar_plot(fig_dir / names[0], by_specimen, "specimen_id", "mean_tcp_position_error_mm", "Robot position error by specimen", plt)
        _bar_plot(fig_dir / names[1], by_specimen, "specimen_id", "mean_total_pipeline_time_s", "Full pipeline cycle time by specimen", plt)
        _bar_plot(fig_dir / names[2], by_specimen, "specimen_id", "home_success_rate", "Home return success rate", plt)
        _bar_plot(fig_dir / names[3], by_specimen, "specimen_id", "clamp_success_rate", "Clamp success rate", plt)
        _anchor_count_plot(fig_dir / names[4], rows, plt)
    except Exception:
        for name in names:
            (fig_dir / name).write_bytes(b"")


def _bar_plot(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str, plt: Any) -> None:
    labels = [str(r.get(label_key, "")) for r in rows] or ["no_data"]
    vals = [_float(r.get(value_key)) or 0.0 for r in rows] or [0.0]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _anchor_count_plot(path: Path, rows: list[dict[str, str]], plt: Any) -> None:
    labels: list[str] = []
    vals: list[int] = []
    for specimen, group in _group(rows, ["specimen_id"]).items():
        counts = Counter(r.get("selected_anchor_id", "") for r in group if r.get("selected_anchor_id"))
        for anchor, count in sorted(counts.items()):
            labels.append(f"{specimen[0]}:{anchor}")
            vals.append(count)
    if not labels:
        labels = ["no_data"]
        vals = [0]
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.title("Selected anchor distribution by specimen")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_summary_md(session: Path, summary: dict[str, Any]) -> None:
    text = ["# E45 Session Summary", ""]
    text.append("This summary contains automatic full-pipeline robot/clamp/anchor telemetry only.")
    text.append("Pundit PL-200 time-of-flight and waveform data are recorded externally in this protocol.")
    text.append("")
    for key, value in summary.items():
        text.append(f"- {key}: {value}")
    (session / "05_summary" / "e45_session_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def analyze_session(session: str | Path) -> dict[str, Any]:
    session_path = Path(session)
    master = session_path / "02_tables" / "e45_all_readings_master.csv"
    if not master.exists():
        master = session_path / "02_tables" / "e45_all_cycles_master.csv"
    rows = _read_csv(master)
    return analyze_rows(rows, session_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze E45 full-pipeline experiment session.")
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    summary = analyze_session(args.session)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
