#!/usr/bin/env python3
"""Create and orchestrate UPV_VLM_v2 paper experiment rerun sessions.

Current implementation supports only --dry-structure-only. It creates the
paper experiment folder structure and a manifest without running models.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PAPER_SUBDIRS = [
    "00_run_manifest",
    "01_e1_target_selection",
    "02_e2_geometry_path_length",
    "03_e3_anchor_selection",
    "04_timing_analysis",
    "05_paper_tables",
    "06_paper_figure_assets",
    "07_audit_report",
]

DEFAULT_CONFIGS = {
    "e1": "configs/v2/experiments/paper/e1_target_selection.yaml",
    "e2": "configs/v2/experiments/paper/e2_geometry_path_length.yaml",
    "e3": "configs/v2/experiments/paper/e3_anchor_selection_proposed_main_pipeline.yaml",
}

DEFAULT_SESSION_PROVENANCE = {
    "e3_anchor_session": "outputs/v2_experiments/e3_anchor_selection/session_20260522_131552",
    "e3_proposed_session": "outputs/v2_experiments/e3_qwen32_single_anchor_scoring/session_20260522_134303",
    "e3_grid_baseline_session": "outputs/v2_experiments/e3_qwen32_contact_crop_grid_baseline/session_20260524_151740",
    "main_pipeline_qwen32_integration_session": "outputs/v2_pipeline_runs/session_20260524_151406",
    "direct_full_object_baseline_session": "outputs/v2_experiments/e3_qwen32_full_overlay_baseline/session_20260523_171830",
    "per_anchor_full_object_baseline_session": "outputs/v2_experiments/e3_qwen32_full_overlay_per_anchor_scoring_baseline/session_20260523_193421",
}

DEFAULT_E3_DATASET_SESSION = "outputs/v2_datasets/e3_anchor_fixed10_highres/session_20260522_101455"
DEFAULT_E3_CASE_ID = "brick_04_debris_obstruction"
E3_MAIN_PIPELINE_CONFIG = "configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_single_anchor.yaml"
DEFAULT_E1_SOURCE_SESSION = "outputs/T5_v2_experiment_1_target_selection_fixed27/session_20260518_142404"
DEFAULT_E1_TRIAL_ID = "trial_001_TS_F01_brick"
DEFAULT_E2_DATASET_SESSION = "outputs/v2_datasets/width_fixed15/session_20260521_214906"
DEFAULT_E2_CASE_ID = "brick_01"
E2_OFFLINE_CONFIG = "configs/v2/experiments/width_offline_from_dataset.yaml"
DEFAULT_E2_RESULT_SESSION = "outputs/v2_experiments/width_offline_fixed15/session_20260521_231017"
DEFAULT_E3_ANCHOR_SESSION = "outputs/v2_experiments/e3_anchor_selection/session_20260522_131552"
DEFAULT_E3_PROPOSED_SESSION = "outputs/v2_experiments/e3_qwen32_single_anchor_scoring/session_20260522_134303"
DEFAULT_E3_DIRECT_BASELINE_SESSION = "outputs/v2_experiments/e3_qwen32_full_overlay_baseline/session_20260523_171830"
DEFAULT_E3_PER_ANCHOR_BASELINE_SESSION = "outputs/v2_experiments/e3_qwen32_full_overlay_per_anchor_scoring_baseline/session_20260523_193421"
DEFAULT_E3_GRID_BASELINE_SESSION = "outputs/v2_experiments/e3_qwen32_contact_crop_grid_baseline/session_20260524_151740"
DEFAULT_E3_MAIN_PIPELINE_PROOF_SESSION = "outputs/v2_pipeline_runs/session_20260524_151406"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_experiments(value: str) -> list[str]:
    selected = [x.strip().lower() for x in value.split(",") if x.strip()]
    invalid = [x for x in selected if x not in DEFAULT_CONFIGS]
    if invalid:
        raise ValueError(f"Unsupported experiments: {invalid}")
    return selected


def paper_config_paths() -> dict[str, str]:
    return {
        "e1_target_selection": "configs/v2/experiments/paper/e1_target_selection.yaml",
        "e2_geometry_path_length": "configs/v2/experiments/paper/e2_geometry_path_length.yaml",
        "e3_anchor_selection_proposed_main_pipeline": "configs/v2/experiments/paper/e3_anchor_selection_proposed_main_pipeline.yaml",
        "e3_baseline_contact_crop_grid": "configs/v2/experiments/paper/e3_baseline_contact_crop_grid.yaml",
        "paper_experiment_manifest": "configs/v2/experiments/paper/paper_experiment_manifest.yaml",
    }


def create_structure(
    output_root: Path,
    experiments: list[str],
    server_url: str,
    *,
    dry_structure_only: bool,
    one_case_validation: bool = False,
) -> Path:
    session = output_root / f"session_{now_stamp()}"
    for subdir in PAPER_SUBDIRS:
        (session / subdir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paper_session": str(session),
        "git_branch": _git(["branch", "--show-current"]),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "selected_experiments": experiments,
        "config_paths": {name: DEFAULT_CONFIGS[name] for name in experiments},
        "qwen_server_url": server_url,
        "dry_structure_only": dry_structure_only,
        "one_case_validation": one_case_validation,
    }
    write_json(session / "00_run_manifest" / "run_manifest.json", manifest)
    write_json(
        session / "00_run_manifest" / "session_provenance.json",
        {
            "session_paths": DEFAULT_SESSION_PROVENANCE,
            "config_paths": paper_config_paths(),
            "qwen_server_url": server_url,
        },
    )
    return session


def update_run_manifest(session: Path, updates: dict[str, Any]) -> None:
    path = session / "00_run_manifest" / "run_manifest.json"
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.update(updates)
    write_json(path, manifest)


def read_dataset_case(dataset_session: Path, case_id: str) -> dict[str, str]:
    manifest = dataset_session / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("case_id") == case_id:
                return row
    raise ValueError(f"case_id={case_id!r} not found in {manifest}")


def resolve_existing_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def newest_pipeline_session(before: set[Path]) -> Path:
    candidates = sorted(Path("outputs/v2_pipeline_runs").glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate not in before:
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No outputs/v2_pipeline_runs/session_* found")


def newest_session_under(root: Path, before: set[Path]) -> Path:
    candidates = sorted(root.glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate not in before:
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No session_* folders found under {root}")


def stage_timing(stage_status: Any, name: str) -> Any:
    if not isinstance(stage_status, list):
        return None
    for item in stage_status:
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("timing_ms")
    return None


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def write_missing_dataset_placeholder(session: Path, experiment: str, case_id: str | None) -> None:
    subdir = "01_e1_target_selection" if experiment == "e1" else "02_e2_geometry_path_length"
    title = "E1 target selection" if experiment == "e1" else "E2 geometry/path-length"
    config_path = DEFAULT_CONFIGS[experiment]
    text = (
        f"# {title} one-case validation placeholder\n\n"
        "This stage was not run because the paper config does not yet provide a concrete dataset manifest.\n\n"
        f"- Config: `{config_path}`\n"
        f"- Requested case id: `{case_id or 'not specified'}`\n"
        "- Status: `not_run_missing_dataset_config`\n"
        "- No model, hardware, or live camera execution was performed for this placeholder.\n"
    )
    (session / subdir / "README_missing_dataset.md").write_text(text, encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def run_e1_reference_ingest(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_session = Path(args.e1_source_session)
    trial_id = args.e1_trial_id
    case_dir = source_session / "cases" / trial_id
    master_csv = source_session / "T5_v2_master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"E1 master CSV not found: {master_csv}")
    if not case_dir.exists():
        raise FileNotFoundError(f"E1 trial folder not found: {case_dir}")

    rows = []
    with master_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected_row: dict[str, Any] | None = None
    for row in rows:
        raw_path = row.get("raw_rgb_path", "")
        trial_index = str(row.get("trial_index", "")).strip()
        frame_id = str(row.get("frame_id", "")).strip()
        if trial_id in raw_path or (trial_id.startswith(f"trial_{int(trial_index or 0):03d}_") if trial_index.isdigit() else False):
            selected_row = row
            break
        if trial_id == DEFAULT_E1_TRIAL_ID and trial_index == "1" and frame_id == "TS_F01":
            selected_row = row
            break
    if selected_row is None:
        raise ValueError(f"Could not find E1 trial row for {trial_id} in {master_csv}")

    ingest_dir = session / "01_e1_target_selection" / "reference_ingest" / trial_id
    copied: dict[str, str] = {}
    for name in [
        "raw_rgb.png",
        "candidate_masks_overlay.png",
        "candidate_crops_panel.png",
        "selected_mask_overlay.png",
        "manual_label.json",
        "target_selection_summary.json",
        "trial_result.json",
    ]:
        copied[name] = copy_if_exists(case_dir / name, ingest_dir / name)

    total_time_s = _safe_float(selected_row.get("total_time_s"))
    correct = str(selected_row.get("correct_manual", "")).strip().lower() in {"yes", "true", "1"}
    result = {
        "trial_id": trial_id,
        "frame_id": selected_row.get("frame_id"),
        "input_text": selected_row.get("input_text"),
        "expected_selected_class": selected_row.get("expected_selected_class"),
        "final_selected_candidate_id": selected_row.get("final_selected_candidate_id"),
        "final_selected_class_manual": selected_row.get("final_selected_class_manual"),
        "correct_manual": selected_row.get("correct_manual"),
        "success": correct,
        "total_time_s": total_time_s,
        "total_timing_ms": total_time_s * 1000.0 if total_time_s is not None else None,
        "source_session": str(source_session),
        "master_results": str(master_csv),
        "mode": "existing_session_reference_ingest",
        "copied_artifacts": copied,
    }

    out_dir = session / "01_e1_target_selection"
    write_json(out_dir / "one_case_result.json", result)
    fields = [
        "trial_id",
        "frame_id",
        "input_text",
        "expected_selected_class",
        "final_selected_candidate_id",
        "final_selected_class_manual",
        "correct_manual",
        "success",
        "total_time_s",
        "source_session",
        "mode",
    ]
    write_csv(out_dir / "per_case_results.csv", [result], fields)
    write_json(
        out_dir / "summary_metrics.json",
        {
            "n_cases": 1,
            "n_success": 1 if correct else 0,
            "accuracy_manual": 1.0 if correct else 0.0,
            "mode": "existing_session_reference_ingest",
        },
    )
    return result


def run_e2_one_case_validation(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    dataset_session = Path(args.e2_dataset_session)
    dataset_manifest = dataset_session / "dataset_manifest.csv"
    case_id = args.e2_case_id
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"E2 dataset manifest not found: {dataset_manifest}")

    output_root = Path("outputs/v2_experiments/width_offline_fixed15")
    before_sessions = set(output_root.glob("session_*"))
    cmd = [
        sys.executable,
        "-m",
        "upv_vlm_v2.experiments.run_width_offline_from_dataset",
        "--dataset",
        str(dataset_manifest),
        "--config",
        E2_OFFLINE_CONFIG,
        "--case-id",
        case_id,
        "--fail-fast",
    ]
    log_path = session / "02_e2_geometry_path_length" / "offline_validation_run.log"
    run_command(cmd, log_path)
    e2_session = newest_session_under(output_root, before_sessions)
    master_csv = e2_session / "master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"E2 output master_results.csv not found: {master_csv}")

    rows: list[dict[str, Any]] = []
    with master_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("case_id") == case_id:
                manual = _safe_float(row.get("manual_width_mm"))
                depth = _safe_float(row.get("depth_path_length_mm"))
                mask = _safe_float(row.get("mask_path_length_mm"))
                chosen = depth if depth is not None else mask
                if manual is not None and chosen is not None:
                    row["absolute_error_mm"] = abs(chosen - manual)
                    row["percent_error"] = abs(chosen - manual) / manual * 100.0 if manual else None
                else:
                    row["absolute_error_mm"] = ""
                    row["percent_error"] = ""
                rows.append(row)
    if not rows:
        raise ValueError(f"No E2 result rows for case_id={case_id} in {master_csv}")

    out_dir = session / "02_e2_geometry_path_length"
    validation_dir = out_dir / "offline_validation" / case_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    write_json(validation_dir / "e2_session_reference.json", {"e2_result_session": str(e2_session), "case_id": case_id})

    asset_dir = session / "06_paper_figure_assets" / "e2_one_case_validation" / case_id
    copied: dict[str, str] = {}
    for row in rows:
        axis = row.get("axis_mode") or "axis"
        axis_dir = e2_session / "cases" / f"case_{int(row.get('case_index') or 0):03d}_{case_id}" / f"axis_{axis}"
        if not axis_dir.exists():
            # Older/replacement sessions use the same case folder for a case id; locate it if the index was not recorded.
            matches = list((e2_session / "cases").glob(f"*_{case_id}/axis_{axis}"))
            axis_dir = matches[0] if matches else axis_dir
        for name in [
            "selected_mask_overlay.png",
            f"{axis}_local_chord_overlay.png",
            f"{axis}_depth_used_points_overlay.png",
            f"{axis}_depth_width_diagnostics.json",
            "pipeline_result.json",
            "stage_status.json",
        ]:
            copied[f"{axis}/{name}"] = copy_if_exists(axis_dir / name, asset_dir / axis / name)

    result = {
        "case_id": case_id,
        "dataset_session": str(dataset_session),
        "dataset_manifest": str(dataset_manifest),
        "e2_result_session": str(e2_session),
        "master_results": str(master_csv),
        "mode": "saved_rgbd_offline_validation",
        "n_result_rows": len(rows),
        "all_success": all(str(r.get("success", "")).lower() == "true" for r in rows),
        "rows": rows,
        "copied_assets": copied,
    }
    write_json(out_dir / "one_case_result.json", result)
    fields = [
        "case_id",
        "requested_material",
        "axis_mode",
        "success",
        "manual_width_mm",
        "mask_path_length_mm",
        "depth_path_length_mm",
        "absolute_error_mm",
        "percent_error",
        "pipeline_result_path",
        "stage_status_path",
        "total_pipeline_runtime_ms",
    ]
    write_csv(out_dir / "per_case_results.csv", rows, fields)
    abs_errors = [_safe_float(r.get("absolute_error_mm")) for r in rows]
    abs_errors = [x for x in abs_errors if x is not None]
    write_json(
        out_dir / "summary_metrics.json",
        {
            "n_cases": 1,
            "n_measurement_rows": len(rows),
            "n_success": sum(1 for r in rows if str(r.get("success", "")).lower() == "true"),
            "mean_absolute_error_mm": sum(abs_errors) / len(abs_errors) if abs_errors else None,
            "e2_result_session": str(e2_session),
        },
    )
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _material_key(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(k, ""))
        for k in ["input_text", "requested_material", "case_id", "frame_id", "expected_selected_class"]
    ).lower()
    if "timber" in text or "wood" in text:
        return "timber"
    if "concrete" in text or "cinder" in text or "block" in text:
        return "concrete_block"
    if "brick" in text:
        return "brick"
    return text.split()[0] if text.split() else "unknown"


def _copy_many(src_dir: Path, dst_dir: Path, names: list[str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in names:
        copied[name] = copy_if_exists(src_dir / name, dst_dir / name)
    return copied


def ingest_e1_full_current_data(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_session = Path(args.e1_source_session)
    master_csv = source_session / "T5_v2_master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"E1 master CSV not found: {master_csv}")
    rows = read_csv_rows(master_csv)
    out_dir = session / "01_e1_target_selection"
    write_csv(out_dir / "per_case_results.csv", rows, list(rows[0].keys()) if rows else [])

    correct_count = sum(1 for r in rows if str(r.get("correct_manual", "")).lower() in {"yes", "true", "1"})
    completed_count = sum(
        1
        for r in rows
        if str(r.get("status", "")).lower() in {"completed", "done", "success", "saved", ""}
        or str(r.get("perception_backend_status", "")).lower() in {"completed", "done", "success"}
    )
    times = [_safe_float(r.get("total_time_s")) for r in rows]
    times = [x for x in times if x is not None]
    breakdown: dict[str, int] = {}
    for row in rows:
        key = _material_key(row)
        breakdown[key] = breakdown.get(key, 0) + 1
    summary = {
        "mode": "existing_fixed27_session_full_ingest",
        "source_session": str(source_session),
        "master_results": str(master_csv),
        "n_trials": len(rows),
        "completed_count": completed_count,
        "correct_manual_count": correct_count,
        "target_selection_success_rate": correct_count / len(rows) if rows else None,
        "mean_total_time_s": _mean(times),
        "median_total_time_s": _median(times),
        "min_total_time_s": min(times) if times else None,
        "max_total_time_s": max(times) if times else None,
        "class_request_breakdown": breakdown,
    }
    write_json(out_dir / "summary_metrics.json", summary)
    write_json(
        out_dir / "source_session_manifest.json",
        {
            "source_session": str(source_session),
            "master_results": str(master_csv),
            "row_count": len(rows),
            "model_rerun": False,
        },
    )

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = _material_key(row)
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        selected = rows[:3]

    asset_rows: list[dict[str, Any]] = []
    asset_root = session / "06_paper_figure_assets" / "e1_target_selection_examples"
    for row in selected:
        case_dir = Path(row.get("case_output_dir") or "")
        if not case_dir.exists() and row.get("raw_rgb_path"):
            case_dir = Path(row["raw_rgb_path"]).parent
        if not case_dir.exists():
            continue
        trial_id = case_dir.name
        dst = asset_root / trial_id
        copied = _copy_many(
            case_dir,
            dst,
            [
                "raw_rgb.png",
                "candidate_masks_overlay.png",
                "candidate_crops_panel.png",
                "selected_mask_overlay.png",
                "manual_label.json",
                "target_selection_summary.json",
                "trial_result.json",
            ],
        )
        for name, path in copied.items():
            if path:
                asset_rows.append({"trial_id": trial_id, "asset": name, "source": str(case_dir / name), "copied_path": path})
    write_csv(asset_root / "asset_manifest.csv", asset_rows, ["trial_id", "asset", "source", "copied_path"])
    return summary


def ingest_e2_full_current_data(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.rerun_e2:
        dataset_manifest = Path(args.e2_dataset_session) / "dataset_manifest.csv"
        before = set(Path("outputs/v2_experiments/width_offline_fixed15").glob("session_*"))
        run_command(
            [
                sys.executable,
                "-m",
                "upv_vlm_v2.experiments.run_width_offline_from_dataset",
                "--dataset",
                str(dataset_manifest),
                "--config",
                E2_OFFLINE_CONFIG,
                "--fail-fast",
            ],
            session / "02_e2_geometry_path_length" / "full_rerun_e2.log",
        )
        source_session = newest_session_under(Path("outputs/v2_experiments/width_offline_fixed15"), before)
    else:
        source_session = Path(args.e2_result_session)
    master_csv = source_session / "master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"E2 master results not found: {master_csv}")
    rows = read_csv_rows(master_csv)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        manual = _safe_float(row.get("manual_width_mm"))
        mask = _safe_float(row.get("mask_path_length_mm"))
        depth = _safe_float(row.get("depth_path_length_mm"))
        if row.get("mask_abs_error_mm") in (None, "") and manual is not None and mask is not None:
            row["mask_abs_error_mm"] = abs(mask - manual)
        if row.get("depth_abs_error_mm") in (None, "") and manual is not None and depth is not None:
            row["depth_abs_error_mm"] = abs(depth - manual)
        if row.get("depth_percent_error") in (None, "") and manual and depth is not None:
            row["depth_percent_error"] = abs(depth - manual) / manual * 100.0
        out_rows.append(row)

    out_dir = session / "02_e2_geometry_path_length"
    write_csv(out_dir / "per_case_results.csv", out_rows, list(out_rows[0].keys()) if out_rows else [])
    copy_if_exists(source_session / "summary_by_axis.csv", out_dir / "summary_by_axis.csv")
    copy_if_exists(source_session / "summary_overall.json", out_dir / "source_summary_overall.json")

    depth_errors = [_safe_float(r.get("depth_abs_error_mm")) for r in out_rows]
    depth_errors = [x for x in depth_errors if x is not None]
    mask_errors = [_safe_float(r.get("mask_abs_error_mm")) for r in out_rows]
    mask_errors = [x for x in mask_errors if x is not None]
    depth_pct = [_safe_float(r.get("depth_percent_error")) for r in out_rows]
    depth_pct = [x for x in depth_pct if x is not None]
    case_ids = {r.get("case_id") for r in out_rows if r.get("case_id")}
    summary = {
        "mode": "existing_result_session_ingest" if not args.rerun_e2 else "full_saved_rgbd_rerun",
        "source_session": str(source_session),
        "dataset_session": args.e2_dataset_session,
        "n_rows": len(out_rows),
        "n_cases": len(case_ids),
        "n_success": sum(1 for r in out_rows if str(r.get("success", "")).lower() == "true"),
        "success_rate": sum(1 for r in out_rows if str(r.get("success", "")).lower() == "true") / len(out_rows) if out_rows else None,
        "mean_abs_error_mask_mm": _mean(mask_errors),
        "mean_abs_error_depth_mm": _mean(depth_errors),
        "median_abs_error_depth_mm": _median(depth_errors),
        "max_abs_error_depth_mm": max(depth_errors) if depth_errors else None,
        "mean_percent_error_depth": _mean(depth_pct),
    }
    write_json(out_dir / "summary_metrics.json", summary)
    write_json(
        out_dir / "source_session_manifest.json",
        {"source_session": str(source_session), "dataset_session": args.e2_dataset_session, "master_results": str(master_csv), "rerun_e2": args.rerun_e2},
    )

    selected_cases: list[str] = []
    seen: set[str] = set()
    for row in out_rows:
        key = _material_key(row)
        case_id = row.get("case_id", "")
        if key not in seen and case_id:
            selected_cases.append(case_id)
            seen.add(key)
        if len(selected_cases) >= 3:
            break
    if not selected_cases:
        selected_cases = sorted(case_ids)[:3]
    asset_root = session / "06_paper_figure_assets" / "e2_geometry_path_length_examples"
    asset_rows: list[dict[str, Any]] = []
    for row in out_rows:
        if row.get("case_id") not in selected_cases:
            continue
        axis = row.get("axis_mode") or "axis"
        axis_dir = Path(row.get("pipeline_result_path") or "").parent
        dst = asset_root / str(row.get("case_id")) / axis
        for src_name, asset_name in [
            ("selected_mask_overlay_path", "selected_mask_overlay.png"),
            ("local_chord_overlay_path", f"{axis}_local_chord_overlay.png"),
            ("depth_used_points_overlay_path", f"{axis}_depth_used_points_overlay.png"),
            ("depth_width_diagnostics_path", f"{axis}_depth_width_diagnostics.json"),
        ]:
            src = Path(row.get(src_name) or "")
            copied = copy_if_exists(src, dst / asset_name)
            if copied:
                asset_rows.append({"case_id": row.get("case_id"), "axis_mode": axis, "asset": asset_name, "source": str(src), "copied_path": copied})
        for name in ["pipeline_result.json", "stage_status.json"]:
            copied = copy_if_exists(axis_dir / name, dst / name)
            if copied:
                asset_rows.append({"case_id": row.get("case_id"), "axis_mode": axis, "asset": name, "source": str(axis_dir / name), "copied_path": copied})
    write_csv(asset_root / "asset_manifest.csv", asset_rows, ["case_id", "axis_mode", "asset", "source", "copied_path"])
    return summary


def _read_e3_summary(session_path: str) -> dict[str, Any]:
    path = Path(session_path) / "summary_overall.json"
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def ingest_e3_full_current_data(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = session / "03_e3_anchor_selection"
    grid_session = Path(args.e3_grid_baseline_session)
    four_way_src = grid_session / "four_way_ablation" / "four_way_comparison_table.csv"
    if not four_way_src.exists():
        raise FileNotFoundError(f"Missing four-way comparison table: {four_way_src}")
    four_way_rows = read_csv_rows(four_way_src)
    write_csv(out_dir / "four_way_ablation.csv", four_way_rows, list(four_way_rows[0].keys()) if four_way_rows else [])
    write_csv(session / "05_paper_tables" / "table_e3_ablation_results.csv", four_way_rows, list(four_way_rows[0].keys()) if four_way_rows else [])

    sessions = {
        "direct_full_object_baseline": args.e3_direct_baseline_session,
        "per_anchor_full_object_baseline": args.e3_per_anchor_baseline_session,
        "contact_crop_grid_baseline": args.e3_grid_baseline_session,
        "proposed_contact_crop_scoring": args.e3_proposed_session,
    }
    summaries = {name: _read_e3_summary(path) for name, path in sessions.items()}
    summary = {"mode": "existing_e3_sessions_ingest", "sessions": sessions, "method_summaries": summaries, "four_way_table": str(four_way_src)}
    write_json(out_dir / "summary_metrics.json", summary)
    write_json(
        out_dir / "source_session_manifest.json",
        {
            "e3_anchor_session": args.e3_anchor_session,
            "e3_proposed_session": args.e3_proposed_session,
            "e3_direct_baseline_session": args.e3_direct_baseline_session,
            "e3_per_anchor_baseline_session": args.e3_per_anchor_baseline_session,
            "e3_grid_baseline_session": args.e3_grid_baseline_session,
            "e3_main_pipeline_proof_session": args.e3_main_pipeline_proof_session,
        },
    )

    anchor_session = Path(args.e3_anchor_session)
    proposed_session = Path(args.e3_proposed_session)
    wanted = [
        "case_002_brick_02_chipped_jagged",
        "case_004_brick_04_debris_obstruction",
        "case_009_timber_04_debris_obstruction",
    ]
    asset_root = session / "06_paper_figure_assets" / "e3_anchor_selection_examples"
    asset_rows: list[dict[str, Any]] = []
    for case_name in wanted:
        src_case = anchor_session / "cases" / case_name
        prop_case = proposed_session / "cases" / case_name
        dst = asset_root / case_name
        nested_raw = next(src_case.glob("pipeline_session/session_*/artifacts/02_target_selection/raw_rgb.png"), None) if src_case.exists() else None
        if nested_raw:
            copied = copy_if_exists(nested_raw, dst / "raw_rgb.png")
            if copied:
                asset_rows.append({"case_id": case_name, "asset": "raw_rgb.png", "source": str(nested_raw), "copied_path": copied})
        for name in [
            "selected_mask_overlay.png",
            "major_minor_axis_overlay.png",
            "combined_wide_contact_guided_grid_2col.png",
            "clean_anchor_review_grid.png",
            "selected_anchor_overlay.png",
        ]:
            copied = copy_if_exists(src_case / name, dst / name)
            if copied:
                asset_rows.append({"case_id": case_name, "asset": name, "source": str(src_case / name), "copied_path": copied})
        for anchor_id in ["A1", "A2", "A3", "A4", "A5"]:
            name = f"source_{anchor_id}.png"
            copied = copy_if_exists(src_case / "clean_single_anchor_inputs" / name, dst / "clean_single_anchor_inputs" / name)
            if copied:
                asset_rows.append({"case_id": case_name, "asset": name, "source": str(src_case / "clean_single_anchor_inputs" / name), "copied_path": copied})
        for src, name in [
            (prop_case / "anchor_score_table.csv", "proposed_anchor_score_table.csv"),
            (prop_case / "case_result.json", "proposed_case_result.json"),
        ]:
            copied = copy_if_exists(src, dst / name)
            if copied:
                asset_rows.append({"case_id": case_name, "asset": name, "source": str(src), "copied_path": copied})
    write_csv(asset_root / "asset_manifest.csv", asset_rows, ["case_id", "asset", "source", "copied_path"])
    return summary


def run_e3_one_case_validation(session: Path, args: argparse.Namespace) -> dict[str, Any]:
    dataset_session = Path(args.dataset_session)
    case_id = args.e3_case_id
    row = read_dataset_case(dataset_session, case_id)
    case_index = int(row.get("case_index") or 0)
    case_slug = f"case_{case_index:03d}_{case_id}" if case_index else case_id
    material = row.get("material") or "brick"
    axis_mode = "major"

    rgb = resolve_existing_path(row["rgb_path"])
    depth = resolve_existing_path(row["depth_path"])
    camera_info = resolve_existing_path(row["camera_info_path"])

    before_sessions = set(Path("outputs/v2_pipeline_runs").glob("session_*"))
    cmd = [
        sys.executable,
        "-m",
        "upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline",
        "--config",
        E3_MAIN_PIPELINE_CONFIG,
        "--requested-material",
        material,
        "--axis-mode",
        axis_mode,
        "--mode",
        "anchor_only",
        "--input-rgb",
        str(rgb),
        "--input-depth",
        str(depth),
        "--input-camera-info",
        str(camera_info),
        "--execution-backend",
        "none",
        "--summary-only",
        "--print-result-json",
    ]
    log_path = session / "03_e3_anchor_selection" / "pipeline_run.log"
    run_command(cmd, log_path)
    pipeline_session = newest_pipeline_session(before_sessions)

    pipeline_result = read_json(pipeline_session / "pipeline_result.json")
    stage_status = read_json(pipeline_session / "stage_status.json")
    timing_summary = read_json(pipeline_session / "timing_summary.json")
    anchor_dir = pipeline_session / "artifacts" / "04_anchor_selection"
    qwen32_decision = read_json(anchor_dir / "qwen32_single_anchor_decision.json")
    final_decision = read_json(anchor_dir / "final_anchor_decision_wide_context.json")

    result = {
        "case_id": case_id,
        "case_index": case_index,
        "case_slug": case_slug,
        "requested_material": material,
        "axis_mode": axis_mode,
        "source_rgb_path": str(rgb),
        "source_depth_path": str(depth),
        "source_camera_info_path": str(camera_info),
        "pipeline_session": str(pipeline_session),
        "pipeline_log_path": str(log_path),
        "success": bool(isinstance(pipeline_result, dict) and pipeline_result.get("success")),
        "final_anchor_id": final_decision.get("selected_anchor") if isinstance(final_decision, dict) else None,
        "anchor_stage": final_decision.get("anchor_stage") if isinstance(final_decision, dict) else None,
        "final_selected_anchor_source": final_decision.get("final_selected_anchor_source") if isinstance(final_decision, dict) else None,
        "qwen32_anchors_attempted": qwen32_decision.get("anchors_attempted") if isinstance(qwen32_decision, dict) else None,
        "qwen32_anchors_parsed": qwen32_decision.get("anchors_parsed") if isinstance(qwen32_decision, dict) else None,
        "qwen32_ranked_source_anchors": qwen32_decision.get("ranked_source_anchors") if isinstance(qwen32_decision, dict) else None,
        "total_timing_ms": timing_summary.get("total_timing_ms") if isinstance(timing_summary, dict) else None,
        "target_selection_timing_ms": stage_timing(stage_status, "target_selection"),
        "geometry_timing_ms": stage_timing(stage_status, "geometry"),
        "anchor_selection_timing_ms": stage_timing(stage_status, "anchor_selection"),
    }

    out_dir = session / "03_e3_anchor_selection"
    write_json(out_dir / "one_case_result.json", result)
    csv_fields = [
        "case_id",
        "requested_material",
        "axis_mode",
        "source_rgb_path",
        "source_depth_path",
        "pipeline_session",
        "success",
        "final_anchor_id",
        "anchor_stage",
        "final_selected_anchor_source",
        "qwen32_anchors_attempted",
        "qwen32_anchors_parsed",
        "qwen32_ranked_source_anchors",
        "total_timing_ms",
        "target_selection_timing_ms",
        "geometry_timing_ms",
        "anchor_selection_timing_ms",
    ]
    csv_row = dict(result)
    csv_row["qwen32_ranked_source_anchors"] = "|".join(result.get("qwen32_ranked_source_anchors") or [])
    write_csv(out_dir / "per_case_results.csv", [csv_row], csv_fields)
    write_json(
        out_dir / "summary_metrics.json",
        {
            "n_cases": 1,
            "n_success": 1 if result["success"] else 0,
            "qwen32_anchors_attempted": result["qwen32_anchors_attempted"],
            "qwen32_anchors_parsed": result["qwen32_anchors_parsed"],
            "all_qwen32_anchors_parsed": result["qwen32_anchors_attempted"] == result["qwen32_anchors_parsed"] == 5,
            "selected_anchor": result["final_anchor_id"],
            "anchor_stage": result["anchor_stage"],
            "final_selected_anchor_source": result["final_selected_anchor_source"],
        },
    )
    write_csv(
        session / "05_paper_tables" / "table_one_case_validation_summary.csv",
        [csv_row],
        csv_fields,
    )
    timing_fields = ["case_id", "total_timing_ms", "target_selection_timing_ms", "geometry_timing_ms", "anchor_selection_timing_ms"]
    write_csv(session / "04_timing_analysis" / "one_case_timing_summary.csv", [csv_row], timing_fields)

    ref_dir = out_dir / "pipeline_runs" / case_slug
    ref_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        ref_dir / "pipeline_session_reference.json",
        {
            "pipeline_session": str(pipeline_session),
            "case_id": case_id,
            "note": "Referenced rather than copied wholesale to avoid duplicating generated pipeline outputs.",
        },
    )

    asset_dir = session / "06_paper_figure_assets" / "e3_one_case_validation" / case_slug
    copied: dict[str, str] = {}
    copied["raw_rgb.png"] = copy_if_exists(pipeline_session / "artifacts" / "01_capture" / "input_rgb.png", asset_dir / "raw_rgb.png")
    for rel in [
        "artifacts/02_target_selection/selected_mask_overlay.png",
        "artifacts/03_geometry/major_minor_axis_overlay.png",
        "artifacts/04_anchor_selection/clean_anchor_review_grid.png",
        "artifacts/04_anchor_selection/combined_wide_contact_guided_grid_2col.png",
        "artifacts/04_anchor_selection/selected_anchor_overlay.png",
        "artifacts/04_anchor_selection/qwen32_single_anchor_decision.json",
        "artifacts/04_anchor_selection/qwen32_single_anchor_score_table.csv",
        "artifacts/04_anchor_selection/final_anchor_decision_wide_context.json",
    ]:
        src = pipeline_session / rel
        copied[Path(rel).name] = copy_if_exists(src, asset_dir / Path(rel).name)
    for anchor_id in ["A1", "A2", "A3", "A4", "A5"]:
        name = f"source_{anchor_id}.png"
        copied[name] = copy_if_exists(anchor_dir / "clean_single_anchor_inputs" / name, asset_dir / "clean_single_anchor_inputs" / name)
    write_json(asset_dir / "asset_manifest.json", {"case_id": case_id, "pipeline_session": str(pipeline_session), "copied_assets": copied})
    result["figure_asset_dir"] = str(asset_dir)
    return result


def run_post_metadata(session: Path, pipeline_session: str, server_url: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "upv_vlm_v2.experiments.collect_paper_timing_metrics",
            "--paper-session",
            str(session),
            "--pipeline-session",
            pipeline_session,
        ],
        session / "04_timing_analysis" / "collect_paper_timing_metrics.log",
    )
    run_command(
        [
            sys.executable,
            "-m",
            "upv_vlm_v2.experiments.write_reproducibility_report",
            "--paper-session",
            str(session),
            "--server-url",
            server_url,
        ],
        session / "07_audit_report" / "write_reproducibility_report.log",
    )


def run_one_case_validation(args: argparse.Namespace) -> Path:
    experiments = parse_experiments(args.experiments)
    session = create_structure(
        Path(args.output_root),
        experiments,
        args.server_url,
        dry_structure_only=False,
        one_case_validation=True,
    )
    e1_result: dict[str, Any] | None = None
    e2_result: dict[str, Any] | None = None
    e1_status = "not_requested"
    e2_status = "not_requested"
    if "e1" in experiments:
        if args.ingest_e1_reference:
            e1_result = run_e1_reference_ingest(session, args)
            e1_status = "ingested_existing_session_reference"
        else:
            write_missing_dataset_placeholder(session, "e1", args.e1_trial_id)
            e1_status = "not_run_reference_ingest_disabled"
    if "e2" in experiments:
        if args.run_e2_one_case:
            e2_result = run_e2_one_case_validation(session, args)
            e2_status = "ran_one_case_saved_rgbd_offline_validation"
        else:
            write_missing_dataset_placeholder(session, "e2", args.e2_case_id)
            e2_status = "not_run_e2_one_case_disabled"

    e3_result: dict[str, Any] | None = None
    e3_status = "not_requested"
    if "e3" in experiments:
        e3_result = run_e3_one_case_validation(session, args)
        e3_status = "ran_one_case_main_pipeline_anchor_only"

    provenance_path = session / "00_run_manifest" / "session_provenance.json"
    provenance = read_json(provenance_path)
    if not isinstance(provenance, dict):
        provenance = {"session_paths": {}, "config_paths": paper_config_paths(), "qwen_server_url": args.server_url}
    session_paths = provenance.setdefault("session_paths", {})
    if isinstance(session_paths, dict):
        if e1_result:
            session_paths["e1_reference_session"] = args.e1_source_session
            session_paths["e1_master_results"] = e1_result["master_results"]
        if e2_result:
            session_paths["e2_dataset_session"] = args.e2_dataset_session
            session_paths["e2_one_case_result_session"] = e2_result["e2_result_session"]
        session_paths["e3_one_case_dataset_session"] = args.dataset_session
        if e3_result:
            session_paths["e3_one_case_main_pipeline_session"] = e3_result["pipeline_session"]
    write_json(provenance_path, provenance)

    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    if e1_result:
        summary_rows.append(
            {
                "experiment": "E1 target-selection reference ingest",
                "validation_mode": e1_result["mode"],
                "case_or_trial_id": e1_result["trial_id"],
                "success": e1_result["success"],
                "primary_result": e1_result.get("final_selected_candidate_id"),
                "source_session": e1_result.get("source_session"),
                "output_session": "",
                "notes": f"manual_correct={e1_result.get('correct_manual')}",
            }
        )
        timing_rows.append(
            {
                "experiment": "E1",
                "case_or_trial_id": e1_result["trial_id"],
                "timing_source": "T5_v2_master_results.total_time_s",
                "total_time_s": e1_result.get("total_time_s"),
                "total_timing_ms": e1_result.get("total_timing_ms"),
                "target_selection_timing_ms": "",
                "geometry_timing_ms": "",
                "anchor_selection_timing_ms": "",
            }
        )
    if e2_result:
        first_row = e2_result["rows"][0] if e2_result.get("rows") else {}
        summary_rows.append(
            {
                "experiment": "E2 geometry/path-length offline validation",
                "validation_mode": e2_result["mode"],
                "case_or_trial_id": e2_result["case_id"],
                "success": e2_result["all_success"],
                "primary_result": f"{e2_result['n_result_rows']} axis rows",
                "source_session": e2_result.get("dataset_session"),
                "output_session": e2_result.get("e2_result_session"),
                "notes": "saved RGB-D offline run",
            }
        )
        timing_rows.append(
            {
                "experiment": "E2",
                "case_or_trial_id": e2_result["case_id"],
                "timing_source": "width_offline master_results.total_pipeline_runtime_ms",
                "total_time_s": "",
                "total_timing_ms": first_row.get("total_pipeline_runtime_ms", ""),
                "target_selection_timing_ms": first_row.get("target_selection_runtime_ms", ""),
                "geometry_timing_ms": first_row.get("geometry_runtime_ms", ""),
                "anchor_selection_timing_ms": first_row.get("anchor_selection_runtime_ms", ""),
            }
        )
    if e3_result:
        summary_rows.append(
            {
                "experiment": "E3 anchor-selection main-pipeline validation",
                "validation_mode": "main_pipeline_anchor_only",
                "case_or_trial_id": e3_result["case_id"],
                "success": e3_result["success"],
                "primary_result": e3_result.get("final_anchor_id"),
                "source_session": args.dataset_session,
                "output_session": e3_result.get("pipeline_session"),
                "notes": f"anchor_stage={e3_result.get('anchor_stage')}",
            }
        )
        timing_rows.append(
            {
                "experiment": "E3",
                "case_or_trial_id": e3_result["case_id"],
                "timing_source": "main pipeline stage_status/timing_summary",
                "total_time_s": "",
                "total_timing_ms": e3_result.get("total_timing_ms"),
                "target_selection_timing_ms": e3_result.get("target_selection_timing_ms"),
                "geometry_timing_ms": e3_result.get("geometry_timing_ms"),
                "anchor_selection_timing_ms": e3_result.get("anchor_selection_timing_ms"),
            }
        )

    if summary_rows:
        write_csv(
            session / "05_paper_tables" / "table_one_case_validation_summary.csv",
            summary_rows,
            ["experiment", "validation_mode", "case_or_trial_id", "success", "primary_result", "source_session", "output_session", "notes"],
        )
    if timing_rows:
        write_csv(
            session / "04_timing_analysis" / "one_case_timing_summary.csv",
            timing_rows,
            [
                "experiment",
                "case_or_trial_id",
                "timing_source",
                "total_time_s",
                "total_timing_ms",
                "target_selection_timing_ms",
                "geometry_timing_ms",
                "anchor_selection_timing_ms",
            ],
        )

    update_run_manifest(
        session,
        {
            "e1_status": e1_status,
            "e2_status": e2_status,
            "e3_status": e3_status,
            "dataset_session": args.dataset_session,
            "e1_source_session": args.e1_source_session,
            "e1_trial_id": args.e1_trial_id,
            "e2_dataset_session": args.e2_dataset_session,
            "e2_case_id": args.e2_case_id,
            "e3_case_id": args.e3_case_id,
            "e1_one_case_result": e1_result,
            "e2_one_case_result": e2_result,
            "e3_one_case_result": e3_result,
        },
    )
    if e3_result:
        run_post_metadata(session, str(e3_result["pipeline_session"]), args.server_url)
    return session


def write_full_current_data_tables(
    session: Path,
    args: argparse.Namespace,
    e1_summary: dict[str, Any] | None,
    e2_summary: dict[str, Any] | None,
    e3_summary: dict[str, Any] | None,
) -> None:
    setup_rows = [
        {"experiment": "E1", "mode": "existing fixed27 target-selection ingest", "source": args.e1_source_session, "rerun": False},
        {"experiment": "E2", "mode": "saved RGB-D width/path-length", "source": args.e2_result_session, "rerun": bool(args.rerun_e2)},
        {"experiment": "E3", "mode": "existing anchor-selection ablation ingest", "source": args.e3_anchor_session, "rerun": bool(args.rerun_e3)},
    ]
    write_csv(session / "05_paper_tables" / "table_experiment_setup.csv", setup_rows, ["experiment", "mode", "source", "rerun"])

    e1_rows = [
        {
            "metric": key,
            "value": json.dumps(value) if isinstance(value, (dict, list)) else value,
        }
        for key, value in (e1_summary or {}).items()
        if key not in {"source_session", "master_results"}
    ]
    write_csv(session / "05_paper_tables" / "table_e1_results.csv", e1_rows, ["metric", "value"])

    e2_rows = [
        {
            "metric": key,
            "value": json.dumps(value) if isinstance(value, (dict, list)) else value,
        }
        for key, value in (e2_summary or {}).items()
        if key not in {"source_session", "dataset_session"}
    ]
    write_csv(session / "05_paper_tables" / "table_e2_results.csv", e2_rows, ["metric", "value"])

    timing_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    if e1_summary:
        timing_rows.append(
            {
                "experiment": "E1",
                "source": "T5_v2_master_results.csv",
                "n": e1_summary.get("n_trials"),
                "mean_total_time_ms": (e1_summary.get("mean_total_time_s") or 0) * 1000 if e1_summary.get("mean_total_time_s") is not None else "",
                "median_total_time_ms": (e1_summary.get("median_total_time_s") or 0) * 1000 if e1_summary.get("median_total_time_s") is not None else "",
                "note": "existing-session target-selection timings",
            }
        )
    else:
        warnings.append({"experiment": "E1", "warning": "E1 summary unavailable"})
    if e2_summary:
        e2_master = Path(e2_summary["source_session"]) / "master_results.csv"
        e2_rows_src = read_csv_rows(e2_master) if e2_master.exists() else []
        totals = [_safe_float(r.get("total_pipeline_runtime_ms")) for r in e2_rows_src]
        totals = [x for x in totals if x is not None]
        timing_rows.append(
            {
                "experiment": "E2",
                "source": "width_offline master_results.csv",
                "n": len(totals),
                "mean_total_time_ms": _mean(totals),
                "median_total_time_ms": _median(totals),
                "note": "saved RGB-D offline width/path-length timings",
            }
        )
        if not totals:
            warnings.append({"experiment": "E2", "warning": "No total_pipeline_runtime_ms values found"})
    if e3_summary:
        proposed = Path(args.e3_proposed_session) / "master_single_anchor_results.csv"
        proposed_rows = read_csv_rows(proposed) if proposed.exists() else []
        totals = [_safe_float(r.get("total_runtime_ms") or r.get("case_runtime_ms") or r.get("elapsed_ms")) for r in proposed_rows]
        totals = [x for x in totals if x is not None]
        proof_timing = read_json(Path(args.e3_main_pipeline_proof_session) / "timing_summary.json")
        timing_rows.append(
            {
                "experiment": "E3",
                "source": "proposed session and main-pipeline proof",
                "n": len(totals),
                "mean_total_time_ms": _mean(totals),
                "median_total_time_ms": _median(totals),
                "note": f"main_pipeline_proof_total_ms={proof_timing.get('total_timing_ms') if isinstance(proof_timing, dict) else ''}",
            }
        )
        if not totals:
            warnings.append({"experiment": "E3", "warning": "No per-case proposed elapsed timing fields found in master_single_anchor_results.csv"})
    write_csv(session / "04_timing_analysis" / "timing_summary_by_experiment.csv", timing_rows, ["experiment", "source", "n", "mean_total_time_ms", "median_total_time_ms", "note"])
    write_csv(session / "05_paper_tables" / "table_timing_summary.csv", timing_rows, ["experiment", "source", "n", "mean_total_time_ms", "median_total_time_ms", "note"])
    write_json(session / "04_timing_analysis" / "timing_summary.json", {"rows": timing_rows, "warnings": warnings})
    write_json(session / "04_timing_analysis" / "timing_warnings.json", warnings)

    all_timing: list[dict[str, Any]] = []
    e1_master = Path(args.e1_source_session) / "T5_v2_master_results.csv"
    if e1_master.exists():
        for row in read_csv_rows(e1_master):
            all_timing.append({"experiment": "E1", "case_id": row.get("frame_id"), "trial_index": row.get("trial_index"), "total_time_s": row.get("total_time_s"), "total_pipeline_runtime_ms": ""})
    e2_master = Path((e2_summary or {}).get("source_session", "")) / "master_results.csv" if e2_summary else Path()
    if e2_master.exists():
        for row in read_csv_rows(e2_master):
            all_timing.append({"experiment": "E2", "case_id": row.get("case_id"), "trial_index": row.get("axis_mode"), "total_time_s": "", "total_pipeline_runtime_ms": row.get("total_pipeline_runtime_ms")})
    write_csv(session / "04_timing_analysis" / "all_stage_timing_by_case.csv", all_timing, ["experiment", "case_id", "trial_index", "total_time_s", "total_pipeline_runtime_ms"])


def run_full_current_data(args: argparse.Namespace) -> Path:
    experiments = parse_experiments(args.experiments)
    session = create_structure(
        Path(args.output_root),
        experiments,
        args.server_url,
        dry_structure_only=False,
        one_case_validation=False,
    )
    update_run_manifest(session, {"full_current_data": True, "rerun_e2": args.rerun_e2, "rerun_e3": args.rerun_e3})

    e1_summary = ingest_e1_full_current_data(session, args) if "e1" in experiments else None
    e2_summary = ingest_e2_full_current_data(session, args) if "e2" in experiments else None
    if args.rerun_e3:
        raise SystemExit("--rerun-e3 is not implemented for this safe ingest stage.")
    e3_summary = ingest_e3_full_current_data(session, args) if "e3" in experiments else None

    provenance = {
        "session_paths": {
            "e1_reference_session": args.e1_source_session,
            "e1_master_results": str(Path(args.e1_source_session) / "T5_v2_master_results.csv"),
            "e2_dataset_session": args.e2_dataset_session,
            "e2_result_session": (e2_summary or {}).get("source_session", args.e2_result_session),
            "e3_anchor_session": args.e3_anchor_session,
            "e3_proposed_session": args.e3_proposed_session,
            "e3_direct_baseline_session": args.e3_direct_baseline_session,
            "e3_per_anchor_baseline_session": args.e3_per_anchor_baseline_session,
            "e3_grid_baseline_session": args.e3_grid_baseline_session,
            "e3_main_pipeline_proof_session": args.e3_main_pipeline_proof_session,
        },
        "config_paths": paper_config_paths(),
        "qwen_server_url": args.server_url,
    }
    write_json(session / "00_run_manifest" / "session_provenance.json", provenance)
    write_full_current_data_tables(session, args, e1_summary, e2_summary, e3_summary)

    run_command(
        [
            sys.executable,
            "-m",
            "upv_vlm_v2.experiments.write_reproducibility_report",
            "--paper-session",
            str(session),
            "--server-url",
            args.server_url,
        ],
        session / "07_audit_report" / "write_reproducibility_report.log",
    )
    return session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="e1,e2,e3")
    parser.add_argument("--output-root", default="outputs/paper_experiments")
    parser.add_argument("--server-url", default="http://127.0.0.1:8899")
    parser.add_argument("--dry-structure-only", action="store_true")
    parser.add_argument("--one-case-validation", action="store_true")
    parser.add_argument("--full-current-data", action="store_true")
    parser.add_argument("--e1-case-id", help="Deprecated alias; use --e1-trial-id for E1 reference ingest.")
    parser.add_argument("--e2-case-id", default=DEFAULT_E2_CASE_ID)
    parser.add_argument("--e1-source-session", default=DEFAULT_E1_SOURCE_SESSION)
    parser.add_argument("--e1-trial-id", default=DEFAULT_E1_TRIAL_ID)
    parser.add_argument("--e2-dataset-session", default=DEFAULT_E2_DATASET_SESSION)
    parser.add_argument("--e2-result-session", default=DEFAULT_E2_RESULT_SESSION)
    parser.add_argument("--e3-anchor-session", default=DEFAULT_E3_ANCHOR_SESSION)
    parser.add_argument("--e3-proposed-session", default=DEFAULT_E3_PROPOSED_SESSION)
    parser.add_argument("--e3-direct-baseline-session", default=DEFAULT_E3_DIRECT_BASELINE_SESSION)
    parser.add_argument("--e3-per-anchor-baseline-session", default=DEFAULT_E3_PER_ANCHOR_BASELINE_SESSION)
    parser.add_argument("--e3-grid-baseline-session", default=DEFAULT_E3_GRID_BASELINE_SESSION)
    parser.add_argument("--e3-main-pipeline-proof-session", default=DEFAULT_E3_MAIN_PIPELINE_PROOF_SESSION)
    parser.add_argument("--rerun-e2", action="store_true")
    parser.add_argument("--rerun-e3", action="store_true")
    parser.add_argument("--run-e2-one-case", dest="run_e2_one_case", action="store_true", default=True)
    parser.add_argument("--no-run-e2-one-case", dest="run_e2_one_case", action="store_false")
    parser.add_argument("--ingest-e1-reference", dest="ingest_e1_reference", action="store_true", default=True)
    parser.add_argument("--no-ingest-e1-reference", dest="ingest_e1_reference", action="store_false")
    parser.add_argument("--e3-case-id", default=DEFAULT_E3_CASE_ID)
    parser.add_argument("--dataset-session", default=DEFAULT_E3_DATASET_SESSION)
    args = parser.parse_args(argv)
    if args.e1_case_id and args.e1_trial_id == DEFAULT_E1_TRIAL_ID:
        args.e1_trial_id = args.e1_case_id
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = [args.dry_structure_only, args.one_case_validation, args.full_current_data]
    if sum(1 for mode in modes if mode) > 1:
        raise SystemExit("Choose only one of --dry-structure-only, --one-case-validation, or --full-current-data.")
    if args.one_case_validation:
        session = run_one_case_validation(args)
        print(f"paper_session={session}")
        return 0
    if args.full_current_data:
        session = run_full_current_data(args)
        print(f"paper_session={session}")
        return 0
    if not args.dry_structure_only:
        raise SystemExit("Use --dry-structure-only, --one-case-validation, or --full-current-data.")
    experiments = parse_experiments(args.experiments)
    session = create_structure(Path(args.output_root), experiments, args.server_url, dry_structure_only=True)
    print(f"paper_session={session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
