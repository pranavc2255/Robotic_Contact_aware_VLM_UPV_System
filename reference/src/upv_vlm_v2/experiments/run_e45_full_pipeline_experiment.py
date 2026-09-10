from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from upv_vlm_v2.experiments.analyze_e45_full_pipeline_experiment import MASTER_FIELDS, analyze_session


DEFAULT_CONFIG = "configs/v2/experiments/paper/e45_full_pipeline_experiment.yaml"
DEFAULT_PIPELINE_CONFIG = "configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_single_anchor_clamp_enabled.yaml"
DEFAULT_SPECIMEN_PLAN = "configs/v2/experiments/paper/e45_specimen_plan.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/v2_experiments/e45_robot_upv_repeatability"
CONFIRM_TOKEN = "RUN_V2_EXECUTE"
RETRYABLE_PRECHECK_FAILURES = {
    "pre_capture_ur_rtde_unavailable",
    "pre_capture_arduino_unavailable",
    "pre_capture_home_failed",
}


def _now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("session_%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def _make_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / _now_stamp()
    candidate = base
    idx = 2
    while candidate.exists():
        candidate = root / f"{base.name}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True)
    return candidate


def _safe_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _axis_payload(result: dict[str, Any], key: str, axis: str) -> dict[str, Any]:
    payload = result.get(key) or {}
    if isinstance(payload, dict):
        item = payload.get(axis)
        return item if isinstance(item, dict) else {}
    return {}


def _pipeline_command(args: argparse.Namespace, material: str, axis: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline",
        "--config",
        str(args.pipeline_config),
        "--requested-material",
        material,
        "--axis-mode",
        axis,
        "--mode",
        "execute",
        "--live-ros2",
        "--execution-backend",
        "real",
        "--allow-real-hardware",
        "--confirm",
        CONFIRM_TOKEN,
        "--summary-only",
        "--print-result-json",
    ]


def _parse_pipeline_path(stdout: str) -> Path | None:
    match = re.search(r"Full JSON saved at:\s*(.+pipeline_result\.json)", stdout)
    if match:
        path = Path(match.group(1).strip())
        if path.exists():
            return path.parent
    return None


def _latest_pipeline_session_after(start_time: float) -> Path | None:
    root = Path("outputs/v2_pipeline_runs")
    if not root.exists():
        return None
    candidates = sorted(root.glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for session in candidates:
        result = session / "pipeline_result.json"
        if result.exists() and result.stat().st_mtime >= start_time:
            return session
    return None


def _init_dirs(session: Path) -> None:
    for rel in ["00_manifest", "cases", "02_tables", "03_figures", "04_paper_assets", "05_summary"]:
        (session / rel).mkdir(parents=True, exist_ok=True)


def _write_operator_protocol(session: Path, external_note: str) -> None:
    text = f"""# E45 Operator Protocol

This E45 session runs the complete current main pipeline for every reading:

capture -> target selection -> geometry -> anchor selection -> path length -> robot plan -> robot execute/clamp/release/home

The runner does not ask for manual UPV fields after each reading. The Pundit PL-200 records time-of-flight, waveform, and signal quality externally during each clamp hold.

External UPV note: {external_note or 'not provided'}

For each specimen:

1. Place the specimen and confirm the requested material/axis.
2. Start the specimen block.
3. The runner executes all planned readings back-to-back unless `--prompt-between-readings` is used.
4. Change to the next specimen only when prompted.

No Pundit automation, clamp bypass, perception shortcut, or robot planning shortcut is implemented here.
"""
    (session / "00_manifest" / "operator_protocol.md").write_text(text, encoding="utf-8")


def _write_master(session: Path, rows: list[dict[str, Any]]) -> None:
    for name in ["e45_all_readings_master.csv", "e4_robot_trials_raw.csv"]:
        with (session / "02_tables" / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MASTER_FIELDS})


def _reading_folder(case_dir: Path, reading_index: int) -> Path:
    return case_dir / f"reading_{int(reading_index):03d}"


def _reading_is_complete(reading_dir: Path) -> bool:
    return any(
        path.exists()
        for path in [
            reading_dir / "pipeline_result.json",
            reading_dir / "pipeline_full_session" / "pipeline_result.json",
            reading_dir / "imported_reading_metadata.json",
        ]
    )


def _planned_missing_readings(case_dir: Path, planned_repeats: int) -> list[int]:
    missing: list[int] = []
    for idx in range(1, int(planned_repeats) + 1):
        if not _reading_is_complete(_reading_folder(case_dir, idx)):
            missing.append(idx)
    return missing


def _build_collection_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if args.specimen_plan and Path(args.specimen_plan).exists() and not args.specimen_id:
        plan = _load_yaml(args.specimen_plan)
        for specimen in plan.get("specimens", []) or []:
            for condition in specimen.get("conditions", []) or []:
                repeats = int(condition.get("repeats", args.planned_repeats))
                for reading in range(1, repeats + 1):
                    items.append({
                        "specimen_id": specimen.get("specimen_id"),
                        "material": specimen.get("material"),
                        "condition_label": condition.get("condition_label"),
                        "axis_mode": condition.get("axis_mode", args.axis_mode),
                        "reading_index_within_specimen": reading,
                        "notes": specimen.get("notes", ""),
                    })
    else:
        for reading in range(1, int(args.planned_repeats) + 1):
            items.append({
                "specimen_id": args.specimen_id,
                "material": args.material,
                "condition_label": args.condition_label,
                "axis_mode": args.axis_mode,
                "reading_index_within_specimen": reading,
                "notes": "",
            })
    if args.max_cycles:
        items = items[: int(args.max_cycles)]
    return items


def _expand_import_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for block in plan.get("imports", []) or []:
        from_session = block.get("from_session")
        include = block.get("include") or {}
        for specimen_id, readings in include.items():
            for reading in readings or []:
                expanded.append({
                    "from_session": from_session,
                    "source_specimen_id": specimen_id,
                    "source_reading_index": int(reading),
                    "target_specimen_id": specimen_id,
                    "target_reading_index": int(reading),
                })
        for mapping in block.get("mappings", []) or []:
            source_readings = list(mapping.get("source_readings") or [])
            target_readings = list(mapping.get("target_readings") or [])
            if len(source_readings) != len(target_readings):
                raise ValueError("source_readings and target_readings must have the same length")
            for source_reading, target_reading in zip(source_readings, target_readings):
                expanded.append({
                    "from_session": from_session,
                    "source_specimen_id": mapping.get("source_specimen_id"),
                    "source_reading_index": int(source_reading),
                    "target_specimen_id": mapping.get("target_specimen_id") or mapping.get("source_specimen_id"),
                    "target_reading_index": int(target_reading),
                })
    return expanded


def _import_selected_readings(session: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.import_readings_plan:
        return []
    plan_path = Path(args.import_readings_plan)
    plan = _load_yaml(plan_path)
    mappings = _expand_import_plan(plan)
    _write_yaml(session / "00_manifest" / "import_readings_plan_used.yaml", plan)
    missing: list[str] = []
    for mapping in mappings:
        source = (
            Path(str(mapping["from_session"]))
            / "cases"
            / str(mapping["source_specimen_id"])
            / f"reading_{int(mapping['source_reading_index']):03d}"
        )
        if not source.exists():
            missing.append(str(source))
    if missing and args.fail_on_import_missing:
        raise FileNotFoundError("missing import readings: " + ", ".join(missing))

    manifest: list[dict[str, Any]] = []
    for mapping in mappings:
        source = (
            Path(str(mapping["from_session"]))
            / "cases"
            / str(mapping["source_specimen_id"])
            / f"reading_{int(mapping['source_reading_index']):03d}"
        )
        target = session / "cases" / str(mapping["target_specimen_id"]) / f"reading_{int(mapping['target_reading_index']):03d}"
        row = {**mapping, "source_path": str(source), "target_path": str(target), "status": "missing"}
        if not source.exists():
            manifest.append(row)
            continue
        if target.exists():
            raise FileExistsError(f"target import reading already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        metadata = {
            "reading_source": "imported",
            "imported_from_session": str(mapping["from_session"]),
            "source_specimen_id": str(mapping["source_specimen_id"]),
            "source_reading_index": int(mapping["source_reading_index"]),
            "target_specimen_id": str(mapping["target_specimen_id"]),
            "target_reading_index": int(mapping["target_reading_index"]),
            "import_timestamp": _now_iso(),
            "copy_mode": "full_reading_folder",
        }
        _write_json(target / "imported_reading_metadata.json", metadata)
        summary = _read_json(target / "reading_summary.json")
        summary.update(metadata)
        summary["skipped_robot_execution_due_to_import"] = True
        _write_json(target / "reading_summary.json", summary)
        row["status"] = "imported"
        manifest.append(row)
    _write_json(session / "00_manifest" / "import_readings_manifest.json", manifest)
    return manifest


def _fake_pipeline_session(reading_dir: Path, item: dict[str, Any], global_index: int, *, failure_reason: str | None = None) -> Path:
    pipeline = reading_dir / "_fake_pipeline_source"
    artifacts = pipeline / "artifacts" / "07_execution"
    tables = pipeline / "tables"
    artifacts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    axis = item["axis_mode"]
    selected_anchor = "A4"
    pre_capture_payload = {
        "pre_capture_hardware_check_performed": True,
        "ur_rtde_check_performed": True,
        "ur_rtde_available": failure_reason != "pre_capture_ur_rtde_unavailable",
        "ur_rtde_failure_reason": "dry_run_simulated_failure" if failure_reason == "pre_capture_ur_rtde_unavailable" else None,
        "arduino_check_performed": True,
        "arduino_available": failure_reason != "pre_capture_arduino_unavailable",
        "arduino_failure_reason": "dry_run_simulated_failure" if failure_reason == "pre_capture_arduino_unavailable" else None,
        "pre_capture_home_check_performed": True,
        "pre_capture_home_within_threshold": True,
        "pre_capture_home_move_attempted": False,
        "pre_capture_home_success": failure_reason != "pre_capture_home_failed",
        "pre_capture_ready_for_capture": failure_reason is None,
        "capture_started": failure_reason is None,
    }
    result = {
        "success": failure_reason is None,
        "mode": "execute",
        "requested_material": item["material"],
        "axis_mode": axis,
        "session_dir": str(pipeline),
        "target": {"requested_material": item["material"], "selected_candidate_id": "candidate_001"},
        "anchors": {
            axis: {
                "final_anchor_id": selected_anchor,
                "score": 0.92,
                "candidate_count": 4,
                "diagnostics": {
                    "qwen_selected_anchor_id": selected_anchor,
                    "anchor_backend": "v2a_qwen32_single_anchor_contact_scoring",
                    "anchor_count_metadata": {
                        "anchor_count_generated": 4,
                        "rule": "physical_axis_length_floor_transducer_diameter",
                    },
                },
            }
        },
        "path_lengths": {
            axis: {
                "mask_path_length_mm": 118.0,
                "depth_path_length_mm": 119.0,
                "depth_valid": True,
                "depth_mask_disagreement_ratio": 0.02,
            }
        },
        "robot_plan": {"upv_path_length_mm": 119.0, "clamp_opening_mm": 124.0},
        "execution": {
            "success": True,
            "robot_moved": True,
            "clamp_moved": True,
            "home_returned": True,
            "timing_ms": 35000 + global_index * 100,
            "diagnostics": {
                "motion_success": True,
                "clamp_success": True,
                "arduino_available": True,
                "clamp_attempted": True,
                "clamp_failure_reason": None,
                "release_attempted": True,
                "release_success": True,
                "home_success": True,
                "home_success_after_execution": True,
                "final_robot_pose_if_available": [0, 0, 0.4, 0, 3.14, 0],
                "execution_payload": {
                    "stages": [
                        {
                            "stage": "approach_final",
                            "target_pose": [0.1, 0.1, 0.1, 0, 3.14, 0],
                            "actual_before": [0.1, 0.1, 0.2, 0, 3.14, 0],
                            "actual_after": [0.101, 0.1, 0.1005, 0, 3.14, 0],
                            "success": True,
                        },
                        {"stage": "home", "target_pose": [0, 0, 0.4, 0, 3.14, 0], "actual_after": [0, 0, 0.4, 0, 3.14, 0], "success": True},
                    ]
                },
            },
        },
        "stage_status": [
            {"name": "pre_capture_home", "timing_ms": 50, "success": failure_reason is None, "failure_reason": failure_reason},
            {"name": "target_selection", "timing_ms": 1000, "success": failure_reason is None, "skipped": failure_reason is not None},
            {"name": "geometry", "timing_ms": 500, "success": True},
            {"name": "anchor_selection", "timing_ms": 1500, "success": True},
            {"name": "path_length", "timing_ms": 400, "success": True},
            {"name": "robot_plan", "timing_ms": 300, "success": True},
            {"name": "execution", "timing_ms": 35000 + global_index * 100, "success": True},
        ],
        "timing_ms": 39000 + global_index * 100,
        "failure_reason": failure_reason,
    }
    if failure_reason is not None:
        result["target"] = None
        result["anchors"] = {}
        result["path_lengths"] = {}
        result["robot_plan"] = None
        result["execution"] = None
    _write_json(pipeline / "pipeline_result.json", result)
    _write_json(pipeline / "stage_status.json", result["stage_status"])
    _write_json(pipeline / "timing_summary.json", {"total_timing_ms": result["timing_ms"], "stages": result["stage_status"]})
    _write_json(pipeline / "run_manifest.json", {"pre_capture_home_check": pre_capture_payload})
    (tables / "flat_result.csv").write_text("success\ntrue\n", encoding="utf-8")
    return pipeline


def _stage_time(result: dict[str, Any], name: str) -> Any:
    for status in result.get("stage_status") or []:
        if status.get("name") == name:
            return status.get("timing_ms")
    return ""


def _stage_by_name(stages: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for stage in stages:
        if stage.get("stage") == name:
            return stage
    return {}


def _pose_error(target: Any, actual: Any) -> tuple[Any, Any]:
    if not isinstance(target, list) or not isinstance(actual, list) or len(target) < 6 or len(actual) < 6:
        return "", ""
    pos = math.sqrt(sum((float(actual[i]) - float(target[i])) ** 2 for i in range(3))) * 1000.0
    ori = math.sqrt(sum((float(actual[i + 3]) - float(target[i + 3])) ** 2 for i in range(3))) * 180.0 / math.pi
    return round(pos, 6), round(ori, 6)


def _json_field(value: Any) -> str:
    return json.dumps(value if value is not None else "")


def _extract_anchor_count(anchor: dict[str, Any]) -> tuple[Any, Any]:
    diag = anchor.get("diagnostics") or {}
    meta = diag.get("anchor_count_metadata") or anchor.get("anchor_count_metadata") or {}
    generated = anchor.get("anchor_count_generated") or anchor.get("candidate_count") or meta.get("anchor_count_generated")
    return generated, meta


def _extract_row(
    session: Path,
    global_index: int,
    item: dict[str, Any],
    reading_dir: Path,
    pipeline_session: Path,
    result: dict[str, Any],
    *,
    attempt_count: int = 1,
    precheck_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    axis = item["axis_mode"]
    target = result.get("target") or {}
    anchor = _axis_payload(result, "anchors", axis)
    path = _axis_payload(result, "path_lengths", axis)
    plan = result.get("robot_plan") or {}
    execution = result.get("execution") or {}
    diag = execution.get("diagnostics") or {}
    payload = diag.get("execution_payload") or {}
    stages = payload.get("stages") or []
    approach = _stage_by_name(stages, "approach_final")
    pos_err, ori_err = _pose_error(approach.get("target_pose"), approach.get("actual_after"))
    anchor_count, anchor_count_meta = _extract_anchor_count(anchor)
    run_manifest = _read_json(pipeline_session / "run_manifest.json")
    pre_capture = run_manifest.get("pre_capture_home_check") or {}
    row = {
        "session_id": session.name,
        "reading_index": global_index,
        "specimen_id": item.get("specimen_id"),
        "material": item.get("material"),
        "condition_label": item.get("condition_label"),
        "axis_mode": axis,
        "attempt_count": attempt_count,
        "precheck_attempts": _json_field(precheck_attempts or []),
        "pipeline_session_path": str(pipeline_session),
        "requested_material": result.get("requested_material") or item.get("material"),
        "pipeline_success": result.get("success"),
        "failure_reason": result.get("failure_reason"),
        "selected_object_class": target.get("requested_material"),
        "selected_candidate_id": target.get("selected_candidate_id"),
        "selected_anchor_id": anchor.get("final_anchor_id"),
        "qwen_selected_anchor": (anchor.get("diagnostics") or {}).get("qwen_selected_anchor_id"),
        "anchor_backend": (anchor.get("diagnostics") or {}).get("anchor_backend"),
        "anchor_score": anchor.get("score"),
        "anchor_count_generated": anchor_count,
        "anchor_count_metadata": _json_field(anchor_count_meta),
        "mask_path_length_mm": path.get("mask_path_length_mm"),
        "depth_path_length_mm": path.get("depth_path_length_mm"),
        "upv_path_length_mm": plan.get("upv_path_length_mm"),
        "clamp_opening_mm": plan.get("clamp_opening_mm"),
        "depth_valid": path.get("depth_valid"),
        "depth_mask_disagreement_percent": (100.0 * float(path.get("depth_mask_disagreement_ratio"))) if path.get("depth_mask_disagreement_ratio") is not None else "",
        "robot_moved": execution.get("robot_moved"),
        "motion_success": diag.get("motion_success"),
        "clamp_moved": execution.get("clamp_moved"),
        "arduino_available": diag.get("arduino_available"),
        "clamp_attempted": diag.get("clamp_attempted"),
        "clamp_success": diag.get("clamp_success"),
        "clamp_failure_reason": diag.get("clamp_failure_reason"),
        "release_attempted": diag.get("release_attempted"),
        "release_success": diag.get("release_success"),
        "home_returned": execution.get("home_returned"),
        "home_success": diag.get("home_success"),
        "home_success_after_execution": diag.get("home_success_after_execution"),
        "execution_success": execution.get("success"),
        "total_pipeline_timing_ms": result.get("timing_ms"),
        "target_selection_timing_ms": _stage_time(result, "target_selection"),
        "geometry_timing_ms": _stage_time(result, "geometry"),
        "anchor_selection_timing_ms": _stage_time(result, "anchor_selection"),
        "path_length_timing_ms": _stage_time(result, "path_length"),
        "robot_plan_timing_ms": _stage_time(result, "robot_plan"),
        "execution_timing_ms": _stage_time(result, "execution") or execution.get("timing_ms"),
        "approach_final_target_pose": _json_field(approach.get("target_pose")),
        "approach_final_actual_before": _json_field(approach.get("actual_before")),
        "approach_final_actual_after": _json_field(approach.get("actual_after")),
        "final_robot_pose_if_available": _json_field(diag.get("final_robot_pose_if_available")),
        "tcp_position_error_mm": pos_err,
        "tcp_orientation_error_deg": ori_err,
        "stage_names": _json_field([stage.get("stage") or stage.get("name") for stage in stages]),
        "stage_success_flags": _json_field([stage.get("success") for stage in stages]),
        "pre_capture_ready_for_capture": pre_capture.get("pre_capture_ready_for_capture"),
        "ur_rtde_available": pre_capture.get("ur_rtde_available"),
        "arduino_available_pre_capture": pre_capture.get("arduino_available"),
        "pre_capture_home_success": pre_capture.get("pre_capture_home_success"),
        "precheck_attempt_count": attempt_count,
        "reading_source": item.get("reading_source", "newly_collected"),
        "imported_from_session": item.get("imported_from_session", ""),
        "source_specimen_id": item.get("source_specimen_id", ""),
        "source_reading_index": item.get("source_reading_index", ""),
        "target_specimen_id": item.get("target_specimen_id", item.get("specimen_id")),
        "target_reading_index": item.get("target_reading_index", item.get("reading_index_within_specimen")),
        "skipped_robot_execution_due_to_import": bool(item.get("skipped_robot_execution_due_to_import", False)),
        "reading_dir": str(reading_dir),
    }
    return row


def _copy_pipeline_session(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _pipeline_failure_reason(pipeline_src: Path | None) -> str | None:
    if pipeline_src is None:
        return None
    result = _read_json(pipeline_src / "pipeline_result.json")
    reason = result.get("failure_reason")
    return str(reason) if reason else None


def _save_failed_attempt(
    *,
    reading_dir: Path,
    attempt_index: int,
    stdout: str,
    stderr: str,
    pipeline_src: Path | None,
    failure_reason: str,
) -> dict[str, Any]:
    attempt_dir = reading_dir / f"attempt_{attempt_index:03d}_precheck_failed"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "pipeline_stdout.txt").write_text(stdout, encoding="utf-8")
    (attempt_dir / "pipeline_stderr.txt").write_text(stderr, encoding="utf-8")
    if pipeline_src is not None:
        (attempt_dir / "pipeline_session_path.txt").write_text(str(pipeline_src), encoding="utf-8")
        _copy_pipeline_session(pipeline_src, attempt_dir / "pipeline_full_session")
        result_src = attempt_dir / "pipeline_full_session" / "pipeline_result.json"
        if result_src.exists():
            shutil.copy2(result_src, attempt_dir / "pipeline_result.json")
    payload = {
        "attempt_index": attempt_index,
        "failure_reason": failure_reason,
        "attempt_dir": str(attempt_dir),
        "timestamp": _now_iso(),
    }
    _write_json(attempt_dir / "attempt_summary.json", payload)
    return payload


def _write_case_manifest(case_dir: Path, item: dict[str, Any], args: argparse.Namespace) -> None:
    manifest = {
        "specimen_id": item.get("specimen_id"),
        "material": item.get("material"),
        "condition_label": item.get("condition_label"),
        "axis_mode": item.get("axis_mode"),
        "specimen_note": item.get("notes", ""),
        "visible_condition_note": "",
        "external_upv_file_id": "",
        "operator_note": "",
        "no_manual_upv_entry": bool(args.no_manual_upv_entry),
        "external_upv_recording_note": args.external_upv_recording_note,
    }
    _write_json(case_dir / "case_manifest.json", manifest)


def _row_from_existing_reading(session: Path, global_index: int, item: dict[str, Any], reading_dir: Path) -> dict[str, Any]:
    metadata = _read_json(reading_dir / "imported_reading_metadata.json")
    pipeline_dest = reading_dir / "pipeline_full_session"
    if not (pipeline_dest / "pipeline_result.json").exists():
        pipeline_dest = reading_dir
    result = _read_json(reading_dir / "pipeline_result.json")
    if not result:
        result = _read_json(pipeline_dest / "pipeline_result.json")
    row_item = dict(item)
    if metadata:
        row_item.update({
            "reading_source": "imported",
            "imported_from_session": metadata.get("imported_from_session", ""),
            "source_specimen_id": metadata.get("source_specimen_id", ""),
            "source_reading_index": metadata.get("source_reading_index", ""),
            "target_specimen_id": metadata.get("target_specimen_id", item.get("specimen_id")),
            "target_reading_index": metadata.get("target_reading_index", item.get("reading_index_within_specimen")),
            "skipped_robot_execution_due_to_import": True,
        })
    else:
        row_item.update({
            "reading_source": "newly_collected",
            "target_specimen_id": item.get("specimen_id"),
            "target_reading_index": item.get("reading_index_within_specimen"),
            "skipped_robot_execution_due_to_import": False,
        })
    row_item.setdefault("timestamp_start", _now_iso())
    row_item.setdefault("timestamp_end", _now_iso())
    row = _extract_row(session, global_index, row_item, reading_dir, pipeline_dest, result)
    row["timestamp_start"] = row_item["timestamp_start"]
    row["timestamp_end"] = row_item["timestamp_end"]
    row["pipeline_returncode"] = ""
    row["reading_source"] = row_item.get("reading_source", "newly_collected")
    row["skipped_robot_execution_due_to_import"] = bool(row_item.get("skipped_robot_execution_due_to_import", False))
    summary = _read_json(reading_dir / "reading_summary.json")
    summary.update(row)
    _write_json(reading_dir / "reading_summary.json", summary)
    return row


def _maybe_prompt_specimen(args: argparse.Namespace, item: dict[str, Any]) -> None:
    if args.dry_run or not args.prompt_between_specimens:
        return
    print("\nPrepare specimen block")
    print(f"SPECIMEN: {item.get('specimen_id')}")
    print(f"MATERIAL: {item.get('material')}")
    print(f"CONDITION: {item.get('condition_label')}")
    print(f"AXIS: {item.get('axis_mode')}")
    action = input("Place/change specimen, then press ENTER to run all planned readings, or q to quit: ").strip().lower()
    if action == "q":
        raise KeyboardInterrupt


def _maybe_prompt_reading(args: argparse.Namespace, item: dict[str, Any], global_index: int) -> None:
    if args.dry_run or not args.prompt_between_readings:
        return
    print(f"\nReading {global_index}: {item.get('specimen_id')} {item.get('condition_label')} reading {item.get('reading_index_within_specimen')}")
    action = input("Press ENTER to run this reading, or q to quit: ").strip().lower()
    if action == "q":
        raise KeyboardInterrupt


def _block_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("specimen_id")),
        str(item.get("material")),
        str(item.get("condition_label")),
        str(item.get("axis_mode")),
    )


def _block_missing_readings(session: Path, block_items: list[dict[str, Any]]) -> list[int]:
    if not block_items:
        return []
    specimen = str(block_items[0].get("specimen_id"))
    case_dir = session / "cases" / specimen
    planned = sorted(int(item.get("reading_index_within_specimen") or 0) for item in block_items)
    return [idx for idx in planned if idx > 0 and not _reading_is_complete(_reading_folder(case_dir, idx))]


def _group_items_by_block(items: list[dict[str, Any]]) -> list[tuple[tuple[str, str, str, str], list[dict[str, Any]]]]:
    blocks: list[tuple[tuple[str, str, str, str], list[dict[str, Any]]]] = []
    current_key: tuple[str, str, str, str] | None = None
    current_items: list[dict[str, Any]] = []
    for item in items:
        key = _block_key(item)
        if current_key is None:
            current_key = key
        if key != current_key:
            blocks.append((current_key, current_items))
            current_key = key
            current_items = []
        current_items.append(item)
    if current_key is not None:
        blocks.append((current_key, current_items))
    return blocks


def _run_one_reading(args: argparse.Namespace, session: Path, global_index: int, item: dict[str, Any]) -> dict[str, Any]:
    specimen_id = str(item.get("specimen_id") or f"specimen_{global_index:03d}")
    case_dir = session / "cases" / specimen_id
    reading_dir = _reading_folder(case_dir, int(item.get("reading_index_within_specimen") or global_index))
    if reading_dir.exists() and ((reading_dir / "imported_reading_metadata.json").exists() or (reading_dir / "pipeline_result.json").exists()):
        print(f"Reading {global_index} already exists, skipping robot execution: {reading_dir}")
        return _row_from_existing_reading(session, global_index, item, reading_dir)
    reading_dir.mkdir(parents=True, exist_ok=True)
    if not (case_dir / "case_manifest.json").exists():
        _write_case_manifest(case_dir, item, args)
    request = dict(item)
    request["global_reading_index"] = global_index
    request["timestamp_start"] = _now_iso()
    _write_json(reading_dir / "reading_request.json", request)
    _maybe_prompt_reading(args, item, global_index)

    stdout = ""
    stderr = ""
    returncode = 0
    precheck_attempts: list[dict[str, Any]] = []
    attempt_index = 0
    max_precheck_retries = None if args.max_precheck_retries is None else int(args.max_precheck_retries)
    while True:
        attempt_index += 1
        simulated_failure = None
        if args.dry_run_precheck_fail_once and args.dry_run and global_index == 1 and attempt_index == 1:
            simulated_failure = f"pre_capture_{args.dry_run_precheck_fail_once}_unavailable" if args.dry_run_precheck_fail_once != "home" else "pre_capture_home_failed"
        if args.dry_run:
            pipeline_src = _fake_pipeline_session(reading_dir, item, global_index, failure_reason=simulated_failure)
        else:
            command = _pipeline_command(args, item["material"], item["axis_mode"])
            print("Running:", " ".join(command))
            started = datetime.now().timestamp()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            pipeline_src = _parse_pipeline_path(stdout) or _latest_pipeline_session_after(started)
            if pipeline_src is None:
                raise RuntimeError("pipeline_session_not_found_after_reading")
        failure_reason = _pipeline_failure_reason(pipeline_src)
        if failure_reason not in RETRYABLE_PRECHECK_FAILURES:
            break
        attempt = _save_failed_attempt(
            reading_dir=reading_dir,
            attempt_index=attempt_index,
            stdout=stdout,
            stderr=stderr,
            pipeline_src=pipeline_src,
            failure_reason=failure_reason,
        )
        precheck_attempts.append(attempt)
        if max_precheck_retries is not None and len(precheck_attempts) > max_precheck_retries:
            break
        if args.dry_run:
            continue
        if failure_reason == "pre_capture_ur_rtde_unavailable":
            prompt = "UR3e RTDE not connected. Connect/fix robot, then press ENTER to retry this same reading, or q to quit: "
        elif failure_reason == "pre_capture_arduino_unavailable":
            prompt = "Arduino/clamp not connected. Connect/fix Arduino, then press ENTER to retry this same reading, or q to quit: "
        else:
            prompt = "Robot did not reach home before capture. Fix hardware and press ENTER to retry this same reading, or q to quit: "
        action = input(prompt).strip().lower()
        if action == "q":
            raise KeyboardInterrupt

    (reading_dir / "pipeline_stdout.txt").write_text(stdout, encoding="utf-8")
    (reading_dir / "pipeline_stderr.txt").write_text(stderr, encoding="utf-8")
    (reading_dir / "pipeline_session_path.txt").write_text(str(pipeline_src), encoding="utf-8")
    pipeline_dest = reading_dir / "pipeline_full_session"
    _copy_pipeline_session(pipeline_src, pipeline_dest)
    result = _read_json(pipeline_dest / "pipeline_result.json")
    shutil.copy2(pipeline_dest / "pipeline_result.json", reading_dir / "pipeline_result.json")
    request["timestamp_end"] = _now_iso()
    row = _extract_row(
        session,
        global_index,
        request,
        reading_dir,
        pipeline_dest,
        result,
        attempt_count=attempt_index,
        precheck_attempts=precheck_attempts,
    )
    row["timestamp_start"] = request["timestamp_start"]
    row["timestamp_end"] = request["timestamp_end"]
    row["pipeline_returncode"] = returncode
    _write_json(reading_dir / "reading_summary.json", row)
    return row


def _write_manifest(session: Path, args: argparse.Namespace, config: dict[str, Any], plan: dict[str, Any]) -> None:
    _write_json(session / "00_manifest" / "e45_session_manifest.json", {
        "session_id": session.name,
        "created_at": _now_iso(),
        "config": str(args.config),
        "pipeline_config": str(args.pipeline_config),
        "specimen_plan": str(args.specimen_plan or ""),
        "dry_run": bool(args.dry_run),
        "auto_confirm_pipeline": bool(args.auto_confirm_pipeline),
        "calls_existing_full_pipeline_cli": True,
        "manual_upv_entry": False,
        "no_manual_upv_entry": bool(args.no_manual_upv_entry),
        "prompt_between_readings": bool(args.prompt_between_readings),
        "prompt_between_specimens": bool(args.prompt_between_specimens),
        "external_upv_recording_note": args.external_upv_recording_note,
        "retry_same_reading_on_precheck_failure": True,
        "max_precheck_retries": args.max_precheck_retries,
        "import_readings_plan": str(args.import_readings_plan or ""),
    })
    _write_yaml(session / "00_manifest" / "e45_config_resolved.yaml", config)
    _write_yaml(session / "00_manifest" / "specimen_plan_used.yaml", plan)
    _write_operator_protocol(session, args.external_upv_recording_note)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E45 full-pipeline robot repeatability experiment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline-config", default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--specimen-plan", default=DEFAULT_SPECIMEN_PLAN)
    parser.add_argument("--auto-confirm-pipeline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--condition-label", default="proposed_contact_usable")
    parser.add_argument("--specimen-id")
    parser.add_argument("--material", default="brick")
    parser.add_argument("--axis-mode", choices=["major", "minor"], default="major")
    parser.add_argument("--planned-repeats", type=int, default=5)
    parser.add_argument("--resume-session")
    parser.add_argument("--no-manual-upv-entry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-between-readings", action="store_true")
    parser.add_argument("--prompt-between-specimens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-upv-recording-note", default="")
    parser.add_argument("--max-precheck-retries", type=int)
    parser.add_argument("--dry-run-precheck-fail-once", choices=["ur_rtde", "arduino", "home"])
    parser.add_argument("--import-readings-plan")
    parser.add_argument("--fail-on-import-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun-existing", action="store_true")
    parser.add_argument("--skip-manual-entry", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_manual_upv_entry:
        print("[INFO] Manual UPV entry is disabled in this E45 runner version; continuing with no manual UPV entry.")
        args.no_manual_upv_entry = True
    config = _load_yaml(args.config)
    plan = _load_yaml(args.specimen_plan) if args.specimen_plan else {}
    session = Path(args.resume_session) if args.resume_session else _make_session(Path(args.output_root))
    _init_dirs(session)
    _write_manifest(session, args, config, plan)
    if args.force_rerun_existing:
        raise SystemExit("force_rerun_existing_is_not_implemented_use_a_new_session_or_remove_import")
    _import_selected_readings(session, args)
    if not args.dry_run and not args.auto_confirm_pipeline:
        print("This E45 session can execute repeated full robot pipeline cycles.")
        token = input(f"Type {CONFIRM_TOKEN} once to arm this session: ").strip()
        if token != CONFIRM_TOKEN:
            raise SystemExit("confirmation_token_mismatch")

    items = _build_collection_items(args)
    rows: list[dict[str, Any]] = []
    skipped_blocks: list[dict[str, Any]] = []
    global_idx = 0
    try:
        for _key, block_items in _group_items_by_block(items):
            if not block_items:
                continue
            missing = _block_missing_readings(session, block_items)
            first = block_items[0]
            if not missing:
                message = (
                    f"Skipping {first.get('specimen_id')} {first.get('condition_label')}: "
                    f"all {len(block_items)} readings already imported/completed"
                )
                print(message)
                skipped_blocks.append({
                    "specimen_id": first.get("specimen_id"),
                    "material": first.get("material"),
                    "condition_label": first.get("condition_label"),
                    "axis_mode": first.get("axis_mode"),
                    "planned_readings": [int(item.get("reading_index_within_specimen") or 0) for item in block_items],
                    "missing_readings": [],
                    "message": message,
                })
                _write_json(session / "00_manifest" / "skipped_complete_blocks.json", skipped_blocks)
            else:
                _maybe_prompt_specimen(args, first)
            for item in block_items:
                global_idx += 1
                row = _run_one_reading(args, session, global_idx, item)
                rows.append(row)
                _write_master(session, rows)
                analyze_session(session)
                print(
                    f"Reading {global_idx} complete: specimen={row.get('specimen_id')} "
                    f"source={row.get('reading_source')} anchor={row.get('selected_anchor_id')} "
                    f"home={row.get('home_success_after_execution')}"
                )
    except KeyboardInterrupt:
        print("E45 session stopped by operator.")
    if skipped_blocks:
        _write_json(session / "00_manifest" / "skipped_complete_blocks.json", skipped_blocks)
    _write_master(session, rows)
    summary = analyze_session(session)
    print(f"E45 session: {session}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
