#!/usr/bin/env python3
"""Collect timing metrics from existing UPV_VLM_v2 paper experiment sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from upv_vlm_v2.experiments.run_all_paper_experiments import DEFAULT_SESSION_PROVENANCE


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None


def collect_pipeline_session(session: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage_status = read_json(session / "stage_status.json")
    if not isinstance(stage_status, list):
        warnings.append(f"Missing or invalid stage_status.json: {session}")
        return rows
    for item in stage_status:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "experiment": "main_pipeline",
                "session": str(session),
                "case_id": "",
                "stage": item.get("name", ""),
                "timing_ms": item.get("timing_ms", ""),
                "source_file": str(session / "stage_status.json"),
            }
        )
    timing_summary = read_json(session / "timing_summary.json")
    if isinstance(timing_summary, dict):
        for key, value in timing_summary.items():
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "experiment": "main_pipeline",
                        "session": str(session),
                        "case_id": "",
                        "stage": f"timing_summary.{key}",
                        "timing_ms": value,
                        "source_file": str(session / "timing_summary.json"),
                    }
                )
    return rows


def _read_csv(path: Path, warnings: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        warnings.append(f"Missing CSV: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_e3_session(session: Path, experiment: str, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_name in [
        "master_single_anchor_results.csv",
        "master_contact_crop_grid_baseline_results.csv",
        "master_full_overlay_baseline_results.csv",
        "master_full_overlay_per_anchor_scoring_results.csv",
    ]:
        for row in _read_csv(session / csv_name, warnings=[]):
            case_id = row.get("case_id", "")
            for key, value in row.items():
                if key.endswith("_ms"):
                    rows.append(
                        {
                            "experiment": experiment,
                            "session": str(session),
                            "case_id": case_id,
                            "stage": key,
                            "timing_ms": value,
                            "source_file": str(session / csv_name),
                        }
                    )
    decision_paths = list(session.glob("cases/*/qwen32_responses/*_parsed.json"))
    decision_paths += list(session.glob("cases/*/parsed/*_parsed.json"))
    for path in decision_paths:
        obj = read_json(path)
        if isinstance(obj, dict):
            elapsed = obj.get("elapsed_ms")
            if elapsed is not None:
                rows.append(
                    {
                        "experiment": experiment,
                        "session": str(session),
                        "case_id": path.parents[1].name,
                        "stage": "qwen_response_elapsed_ms",
                        "timing_ms": elapsed,
                        "source_file": str(path),
                    }
                )
    if not rows:
        warnings.append(f"No timing rows found for {experiment}: {session}")
    return rows


def read_session_provenance(paper: Path) -> dict[str, str]:
    obj = read_json(paper / "00_run_manifest" / "session_provenance.json")
    if isinstance(obj, dict) and isinstance(obj.get("session_paths"), dict):
        return {str(k): str(v) for k, v in obj["session_paths"].items() if v}
    return dict(DEFAULT_SESSION_PROVENANCE)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = safe_float(row.get("timing_ms"))
        if value is None:
            continue
        key = (str(row.get("experiment", "")), str(row.get("stage", "")))
        groups.setdefault(key, []).append(value)
    out = []
    for (experiment, stage), values in sorted(groups.items()):
        out.append(
            {
                "experiment": experiment,
                "stage": stage,
                "n": len(values),
                "mean_ms": round(mean(values), 3) if values else "",
                "min_ms": round(min(values), 3) if values else "",
                "max_ms": round(max(values), 3) if values else "",
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    paper = Path(args.paper_session)
    out_dir = paper / "04_timing_analysis"
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    provenance = read_session_provenance(paper)
    pipeline_session = args.pipeline_session or provenance.get("main_pipeline_qwen32_integration_session")
    e3_proposed = args.e3_proposed_session or provenance.get("e3_proposed_session")
    e3_grid = args.e3_grid_baseline_session or provenance.get("e3_grid_baseline_session")
    e3_direct = args.e3_direct_baseline_session or provenance.get("direct_full_object_baseline_session")
    e3_per_anchor = args.e3_per_anchor_baseline_session or provenance.get("per_anchor_full_object_baseline_session")
    if pipeline_session:
        rows.extend(collect_pipeline_session(Path(pipeline_session), warnings))
    if e3_proposed:
        rows.extend(collect_e3_session(Path(e3_proposed), "e3_proposed_single_anchor", warnings))
    if e3_grid:
        rows.extend(collect_e3_session(Path(e3_grid), "e3_contact_crop_grid_baseline", warnings))
    if e3_direct:
        rows.extend(collect_e3_session(Path(e3_direct), "e3_direct_full_object_baseline", warnings))
    if e3_per_anchor:
        rows.extend(collect_e3_session(Path(e3_per_anchor), "e3_per_anchor_full_object_baseline", warnings))
    fields = ["experiment", "session", "case_id", "stage", "timing_ms", "source_file"]
    write_csv(out_dir / "all_stage_timing_by_case.csv", rows, fields)
    summary_rows = summarize(rows)
    write_csv(out_dir / "timing_summary_by_experiment.csv", summary_rows, ["experiment", "stage", "n", "mean_ms", "min_ms", "max_ms"])
    write_json(out_dir / "timing_summary.json", {"rows": summary_rows})
    write_json(out_dir / "timing_warnings.json", {"warnings": warnings})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-session", required=True)
    p.add_argument("--pipeline-session")
    p.add_argument("--e3-proposed-session")
    p.add_argument("--e3-grid-baseline-session")
    p.add_argument("--e3-direct-baseline-session")
    p.add_argument("--e3-per-anchor-baseline-session")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
