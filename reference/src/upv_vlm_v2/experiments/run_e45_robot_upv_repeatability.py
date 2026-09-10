"""E45 robot-UPV repeatability experiment orchestrator.

This runner does not implement a new perception pipeline. It either creates
dry/manual logging structure, reuses an existing main-pipeline session, or
optionally calls the existing main v2 pipeline CLI when explicitly allowed.
Robot motion and automated UPV triggering are blocked by default.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to read E45 YAML config.") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/v2/experiments/paper/e45_robot_upv_repeatability.yaml"
DEFAULT_CASE_PLAN = REPO_ROOT / "configs/v2/experiments/paper/e45_case_plan_pilot.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/v2_experiments/e45_robot_upv_repeatability"

RAW_FIELDS = [
    "session_id",
    "case_id",
    "material",
    "condition",
    "anchor_type",
    "anchor_id",
    "repeat_index",
    "pipeline_case_dir",
    "selected_anchor_id",
    "bad_anchor_reason",
    "planned_tcp_x_m",
    "planned_tcp_y_m",
    "planned_tcp_z_m",
    "planned_tcp_rx_rad",
    "planned_tcp_ry_rad",
    "planned_tcp_rz_rad",
    "actual_tcp_x_m",
    "actual_tcp_y_m",
    "actual_tcp_z_m",
    "actual_tcp_rx_rad",
    "actual_tcp_ry_rad",
    "actual_tcp_rz_rad",
    "position_error_mm",
    "orientation_error_deg",
    "robot_motion_success",
    "robot_mode",
    "safety_mode",
    "failure_stage",
    "failure_reason",
    "path_length_mm",
    "upv_velocity_m_s",
    "arrival_time_us",
    "signal_detected",
    "signal_quality",
    "operator_note",
    "photo_overview_path",
    "photo_contact_path",
    "timestamp",
]

MANUAL_UPV_FIELDS = [
    "case_id",
    "anchor_type",
    "anchor_id",
    "repeat_index",
    "path_length_mm",
    "upv_velocity_m_s",
    "arrival_time_us",
    "signal_detected",
    "signal_quality",
    "operator_note",
    "photo_contact_path",
]


def _now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _session_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("session_%Y%m%d_%H%M%S")


def _resolve(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_session(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = _session_stamp()
    for i in range(100):
        suffix = "" if i == 0 else f"_{i:02d}"
        session = output_root / f"{stamp}{suffix}"
        try:
            session.mkdir(parents=True, exist_ok=False)
            return session
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique session under {output_root}")


def _case_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases = ((config.get("case_plan") or {}).get("cases") or [])
    return [c for c in cases if isinstance(c, dict)]


def _apply_case_plan(config: dict[str, Any], case_plan: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    merged["case_plan"] = {"cases": case_plan.get("cases") or []}
    if case_plan.get("repeats_per_anchor") is not None:
        experiment = dict(merged.get("experiment") or {})
        experiment["repeats_per_anchor"] = int(case_plan["repeats_per_anchor"])
        merged["experiment"] = experiment
    merged["case_plan_metadata"] = {
        "session_description": case_plan.get("session_description", ""),
    }
    return merged


def _repeat_count(config: dict[str, Any]) -> int:
    return max(1, int((config.get("experiment") or {}).get("repeats_per_anchor", 5)))


def _anchor_types(config: dict[str, Any]) -> list[str]:
    values = (config.get("experiment") or {}).get("anchor_types") or ["contact_usable", "contact_unsuitable"]
    return [str(v) for v in values]


def _write_run_log(session: Path) -> None:
    text = """# E45 Robot-UPV Repeatability Session

This session supports two paper experiments using the same raw repeated trials.

- Experiment 4 uses robot pose fields to quantify repeatability of selected UPV contact poses.
- Experiment 5 uses manual UPV reading fields to quantify repeatability at contact-usable and contact-unsuitable anchors.
- Manual UPV CSV entry is expected initially.
- Robot motion is blocked unless explicit command-line confirmation and config acknowledgement are provided.
- Contact motion, clamp actuation, and automated UPV triggering are not implemented in this scaffold.
"""
    (session / "run_log.md").write_text(text, encoding="utf-8")


def _write_operator_instructions(session: Path) -> None:
    text = """# E45 Operator Quick Instructions

For each trial:

1. Confirm case and anchor ID.
2. Move robot using an external tested script, or allow guided mode only after reviewed motion gating is enabled.
3. Wait until robot reaches the pose.
4. Read the Pundit PL-200.
5. Enter:
   - velocity_m_s
   - arrival_time_us if available
   - signal_detected yes/no
   - signal_quality stable/unstable/no_signal/rejected
   - operator_note
6. Save photo path if available.
7. Continue to next repeat.

This runner does not actuate the clamp and does not trigger UPV hardware automatically.
"""
    (session / "operator_quick_instructions.md").write_text(text, encoding="utf-8")


def _base_trial_row(session: Path, case: dict[str, Any], anchor_type: str, anchor_id: str, repeat_index: int) -> dict[str, Any]:
    return {
        "session_id": session.name,
        "case_id": case.get("case_id", ""),
        "material": case.get("material", ""),
        "condition": case.get("condition", ""),
        "anchor_type": anchor_type,
        "anchor_id": anchor_id,
        "repeat_index": f"{repeat_index:03d}",
        "bad_anchor_reason": case.get("bad_anchor_reason", "") if anchor_type == "contact_unsuitable" else "",
        "path_length_mm": case.get("path_length_mm") or "",
        "robot_motion_success": "",
        "timestamp": _now_iso(),
    }


def _blank_trial_rows(session: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repeats = _repeat_count(config)
    for case in _case_entries(config):
        for anchor_type in _anchor_types(config):
            anchor_id = ""
            if anchor_type == "contact_usable":
                anchor_id = str(case.get("good_anchor_id") or "")
            elif anchor_type == "contact_unsuitable":
                anchor_id = str(case.get("bad_anchor_id") or "")
            for repeat_index in range(1, repeats + 1):
                rows.append(_base_trial_row(session, case, anchor_type, anchor_id, repeat_index))
    return rows


def _write_manual_template(session: Path, raw_rows: list[dict[str, Any]]) -> None:
    rows = [{field: row.get(field, "") for field in MANUAL_UPV_FIELDS} for row in raw_rows]
    _write_csv(session / "manual_upv_entry_template.csv", rows, MANUAL_UPV_FIELDS)


def _write_empty_summaries(session: Path) -> None:
    _write_csv(session / "e4_robot_motion_summary_by_anchor.csv", [], [
        "case_id", "material", "condition", "anchor_type", "anchor_id", "n_attempted", "n_motion_success",
        "motion_success_rate", "mean_position_error_mm", "std_position_error_mm", "std_actual_tcp_x_mm",
        "std_actual_tcp_y_mm", "std_actual_tcp_z_mm", "mean_orientation_error_deg", "std_orientation_error_deg",
        "safety_failure_count", "failure_count", "notes",
    ])
    _write_json(session / "e4_robot_motion_summary_overall.json", {
        "n_attempted": 0,
        "n_motion_success": 0,
        "motion_success_rate": None,
        "mean_position_error_mm": None,
        "mean_orientation_error_deg": None,
        "safety_failure_count": 0,
        "failure_stage_counts": {},
    })
    _write_csv(session / "e5_upv_repeatability_summary_by_anchor.csv", [], [
        "case_id", "material", "condition", "anchor_type", "anchor_id", "n_attempted", "n_valid",
        "valid_signal_rate", "n_failed", "failed_reading_rate", "mean_velocity_m_s", "std_velocity_m_s",
        "cv_velocity_percent", "mean_arrival_time_us", "std_arrival_time_us", "cv_arrival_time_percent", "notes",
    ])
    _write_json(session / "e5_upv_repeatability_summary_overall.json", {
        "usable_valid_signal_rate": None,
        "unsuitable_valid_signal_rate": None,
        "usable_mean_cv_velocity_percent": None,
        "unsuitable_mean_cv_velocity_percent": None,
        "usable_failed_reading_rate": None,
        "unsuitable_failed_reading_rate": None,
        "n_cases": 0,
        "n_attempts": 0,
    })


def _initialize_session(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    root = _resolve(args.output_root) or DEFAULT_OUTPUT_ROOT
    session = _make_session(root)
    _write_yaml(session / "run_config.yaml", config)
    if getattr(args, "case_plan", None):
        case_plan_path = _resolve(args.case_plan)
        if case_plan_path and case_plan_path.exists():
            shutil.copy2(case_plan_path, session / "case_plan_used.yaml")
    _write_run_log(session)
    _write_operator_instructions(session)
    for sub in ["cases", "figures"]:
        (session / sub).mkdir(parents=True, exist_ok=True)
    _write_json(session / "run_manifest.json", {
        "session_id": session.name,
        "timestamp": _now_iso(),
        "mode": args.mode,
        "config_path": str(_resolve(args.config)),
        "output_root": str(root),
        "reuse_pipeline_session": args.reuse_pipeline_session,
        "allow_qwen_calls": bool(args.allow_qwen_calls),
        "allow_robot_motion": bool(args.allow_robot_motion),
        "robot_mode": getattr(args, "robot_mode", ""),
        "fake_data": bool(getattr(args, "fake_data", False)),
        "not_for_paper_results": bool(getattr(args, "fake_data", False)),
        "automated_upv_trigger_enabled": False,
        "main_pipeline_source_modified": False,
    })
    return session


def _write_case_plans(session: Path, config: dict[str, Any], mode: str) -> None:
    for case in _case_entries(config):
        case_id = str(case.get("case_id") or "case_unknown")
        case_dir = session / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "case_id": case_id,
            "mode": mode,
            "case_config": case,
            "repeats_per_anchor": _repeat_count(config),
            "anchor_types": _anchor_types(config),
            "robot_motion_allowed": False,
            "upv_automated_trigger_allowed": False,
        }
        _write_json(case_dir / "case_plan.json", plan)
        for anchor_type in _anchor_types(config):
            aid = case.get("good_anchor_id") if anchor_type == "contact_usable" else case.get("bad_anchor_id")
            label = "good_anchor" if anchor_type == "contact_usable" else "bad_anchor"
            (case_dir / f"{label}_{aid or 'manual_to_fill'}").mkdir(parents=True, exist_ok=True)


def _extract_pipeline_anchor_info(pipeline_session: Path) -> dict[str, Any]:
    anchor_dir = pipeline_session / "artifacts" / "04_anchor_selection"
    qwen = _read_json(anchor_dir / "qwen32_single_anchor_decision.json")
    final = _read_json(anchor_dir / "final_anchor_decision_wide_context.json")
    robot_geom = _read_json(anchor_dir / "robot_anchor_geometry.json")
    pipeline_result = _read_json(pipeline_session / "pipeline_result.json")
    selected = qwen.get("selected_anchor_id") or final.get("selected_anchor") or robot_geom.get("robot_candidate_anchor_id")
    deterministic = robot_geom.get("deterministic_candidate_source") if isinstance(robot_geom.get("deterministic_candidate_source"), dict) else {}
    return {
        "pipeline_session": str(pipeline_session),
        "pipeline_success": pipeline_result.get("success"),
        "selected_anchor_id": selected,
        "ranked_source_anchors": qwen.get("ranked_source_anchors", []),
        "anchors_attempted": qwen.get("anchors_attempted"),
        "anchors_parsed": qwen.get("anchors_parsed"),
        "final_selected_anchor_source": final.get("final_selected_anchor_source") or qwen.get("final_selected_anchor_source"),
        "anchor_stage": final.get("anchor_stage"),
        "selected_anchor_px": robot_geom.get("robot_candidate_anchor_center_px"),
        "contact_point_a_px": deterministic.get("contact_point_a_px"),
        "contact_point_b_px": deterministic.get("contact_point_b_px"),
        "local_path_length_px": deterministic.get("local_path_length_px"),
        "qwen32_decision_path": str(anchor_dir / "qwen32_single_anchor_decision.json"),
        "qwen32_score_table_path": str(anchor_dir / "qwen32_single_anchor_score_table.csv"),
        "selected_anchor_overlay_path": str(anchor_dir / "selected_anchor_overlay.png"),
        "clean_anchor_review_grid_path": str(anchor_dir / "clean_anchor_review_grid.png"),
        "combined_contact_grid_path": str(anchor_dir / "combined_wide_contact_guided_grid_2col.png"),
        "clean_single_anchor_inputs_dir": str(anchor_dir / "clean_single_anchor_inputs"),
    }


def _pipeline_artifact_index(pipeline_session: Path, selected_anchor_id: str = "") -> dict[str, Any]:
    anchor_dir = pipeline_session / "artifacts" / "04_anchor_selection"
    clean_dir = anchor_dir / "clean_single_anchor_inputs"
    source_inputs = {f"A{i}": str(clean_dir / f"source_A{i}.png") for i in range(1, 6) if (clean_dir / f"source_A{i}.png").exists()}
    return {
        "pipeline_session": str(pipeline_session),
        "anchor_dir": str(anchor_dir),
        "clean_anchor_review_grid": str(anchor_dir / "clean_anchor_review_grid.png") if (anchor_dir / "clean_anchor_review_grid.png").exists() else "",
        "combined_wide_contact_guided_grid_2col": str(anchor_dir / "combined_wide_contact_guided_grid_2col.png") if (anchor_dir / "combined_wide_contact_guided_grid_2col.png").exists() else "",
        "selected_anchor_overlay": str(anchor_dir / "selected_anchor_overlay.png") if (anchor_dir / "selected_anchor_overlay.png").exists() else "",
        "qwen32_single_anchor_score_table": str(anchor_dir / "qwen32_single_anchor_score_table.csv") if (anchor_dir / "qwen32_single_anchor_score_table.csv").exists() else "",
        "qwen32_single_anchor_decision": str(anchor_dir / "qwen32_single_anchor_decision.json") if (anchor_dir / "qwen32_single_anchor_decision.json").exists() else "",
        "source_anchor_inputs": source_inputs,
        "selected_anchor_input": source_inputs.get(selected_anchor_id, ""),
    }


def _anchor_dir_name(anchor_type: str, anchor_id: str) -> str:
    return f"{anchor_type}_{anchor_id or 'manual_to_fill'}"


def _write_case_plan_guided(session: Path, case: dict[str, Any], pipeline_info: dict[str, Any], artifact_index: dict[str, Any], good_anchor: str, bad_anchor: str) -> Path:
    case_id = str(case.get("case_id") or "case_unknown")
    case_dir = session / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "case_id": case_id,
        "mode": "guided_collect",
        "case_config": case,
        "good_anchor_id": good_anchor,
        "bad_anchor_id": bad_anchor,
        "pipeline_info": pipeline_info,
        "repeats_per_anchor": case.get("repeats_per_anchor"),
        "robot_motion_allowed": False,
        "upv_automated_trigger_allowed": False,
    }
    _write_json(case_dir / "case_plan.json", plan)
    _write_json(case_dir / "pipeline_artifacts_index.json", artifact_index)
    for anchor_type, aid in [("contact_usable", good_anchor), ("contact_unsuitable", bad_anchor)]:
        (case_dir / _anchor_dir_name(anchor_type, aid)).mkdir(parents=True, exist_ok=True)
    return case_dir


def _save_rows_immediately(session: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
    _write_manual_template(session, rows)


def _read_optional_pose(value: str) -> list[float] | None:
    value = value.strip()
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 6:
        raise ValueError("TCP pose must have six comma-separated values: x,y,z,rx,ry,rz")
    return [float(p) for p in parts]


def _prompt(text: str, default: str = "") -> str:
    value = input(text)
    return value.strip() if value.strip() else default


def _print_trial_block(case: dict[str, Any], anchor_type: str, anchor_id: str, repeat_index: int, repeats: int, pipeline_session: str, artifact_index: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"CASE: {case.get('case_id', '')}")
    print(f"MATERIAL: {case.get('material', '')}")
    print(f"CONDITION: {case.get('condition', '')}")
    print(f"ANCHOR TYPE: {anchor_type}")
    print(f"ANCHOR ID: {anchor_id}")
    print(f"REPEAT: {repeat_index} / {repeats}")
    print(f"PIPELINE SESSION: {pipeline_session}")
    print("CONTACT CROP / VISUAL ASSETS:")
    for label, key in [
        ("clean_anchor_review_grid", "clean_anchor_review_grid"),
        (f"source_{anchor_id}.png", "source_anchor_inputs"),
        ("score table", "qwen32_single_anchor_score_table"),
    ]:
        if key == "source_anchor_inputs":
            path = (artifact_index.get(key) or {}).get(anchor_id, "")
        else:
            path = artifact_index.get(key, "")
        print(f"- {label}: {path or 'not available'}")
    print("=" * 72)


def _fake_trial_values(row: dict[str, Any], repeat_index: int, preserve_robot_fields: bool = False) -> dict[str, Any]:
    rng = random.Random(f"{row.get('case_id')}:{row.get('anchor_type')}:{row.get('anchor_id')}:{repeat_index}")
    planned = [0.300, -0.200, 0.180, 0.0, 3.14159, 0.0]
    actual = [
        planned[0] + rng.uniform(-0.00025, 0.00025),
        planned[1] + rng.uniform(-0.00025, 0.00025),
        planned[2] + rng.uniform(-0.00020, 0.00020),
        planned[3] + rng.uniform(-0.001, 0.001),
        planned[4] + rng.uniform(-0.001, 0.001),
        planned[5] + rng.uniform(-0.001, 0.001),
    ]
    if not preserve_robot_fields:
        for key, value in zip([
            "planned_tcp_x_m", "planned_tcp_y_m", "planned_tcp_z_m",
            "planned_tcp_rx_rad", "planned_tcp_ry_rad", "planned_tcp_rz_rad",
        ], planned, strict=False):
            row[key] = round(value, 8)
        for key, value in zip([
            "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
            "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
        ], actual, strict=False):
            row[key] = round(value, 8)
        row["robot_motion_success"] = "yes"
        row["robot_mode"] = "fake"
        row["safety_mode"] = "fake"
    if row.get("anchor_type") == "contact_usable":
        row["signal_detected"] = "yes"
        row["signal_quality"] = "stable"
        row["upv_velocity_m_s"] = round(3400.0 + rng.uniform(-20.0, 20.0), 3)
        row["arrival_time_us"] = round(63.0 + rng.uniform(-0.4, 0.4), 3)
    else:
        pattern = ["unstable", "no_signal", "stable", "rejected", "stable"]
        quality = pattern[(repeat_index - 1) % len(pattern)]
        row["signal_quality"] = quality
        row["signal_detected"] = "yes" if quality in {"stable", "unstable"} else "no"
        if quality == "stable":
            row["upv_velocity_m_s"] = round(3350.0 + rng.uniform(-180.0, 180.0), 3)
            row["arrival_time_us"] = round(66.0 + rng.uniform(-5.0, 5.0), 3)
        elif quality == "unstable":
            row["upv_velocity_m_s"] = round(3350.0 + rng.uniform(-260.0, 260.0), 3)
            row["arrival_time_us"] = round(66.0 + rng.uniform(-8.0, 8.0), 3)
        else:
            row["upv_velocity_m_s"] = ""
            row["arrival_time_us"] = ""
    row["operator_note"] = "fake_data_not_for_paper_results"
    return row


def _manual_trial_values(row: dict[str, Any], robot_mode: str, args: argparse.Namespace) -> dict[str, Any]:
    if robot_mode == "none":
        print("Robot mode is none. Place/execute contact manually or using external tested script, then enter readings.")
    elif robot_mode == "external":
        print("Robot mode is external. Use the already-tested external robot script, then enter readings here.")
    elif robot_mode == "read_only":
        print("Robot mode is read_only. This runner can read RTDEReceive state only; it does not move the robot.")
        if args.robot_ip:
            state = _capture_read_only_state(args.robot_ip)
            pose = state.get("actual_tcp_pose") or []
            for key, value in zip([
                "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
                "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
            ], pose, strict=False):
                row[key] = value
            row["robot_mode"] = state.get("robot_mode", "")
            row["safety_mode"] = state.get("safety_mode", "")
    elif robot_mode == "execute_hover_confirmed":
        raise RuntimeError("execute_hover_confirmed remains scaffolded. Use an external reviewed R9-style script for real motion.")

    row["robot_motion_success"] = _prompt("robot_motion_success yes/no/blank: ")
    pose_text = _prompt("actual_tcp_pose x,y,z,rx,ry,rz or blank: ")
    if pose_text:
        pose = _read_optional_pose(pose_text)
        if pose:
            for key, value in zip([
                "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
                "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
            ], pose, strict=False):
                row[key] = value
    row["robot_mode"] = row.get("robot_mode") or _prompt("robot_mode optional: ")
    row["safety_mode"] = row.get("safety_mode") or _prompt("safety_mode optional: ")
    row["path_length_mm"] = _prompt("path_length_mm: ", str(row.get("path_length_mm") or ""))
    row["upv_velocity_m_s"] = _prompt("Pundit PL-200 velocity m/s or blank: ")
    row["arrival_time_us"] = _prompt("Pundit PL-200 arrival/transit time us or blank: ")
    row["signal_detected"] = _prompt("signal_detected yes/no: ")
    row["signal_quality"] = _prompt("signal_quality stable/unstable/no_signal/rejected: ")
    row["operator_note"] = _prompt("operator_note: ")
    row["photo_contact_path"] = _prompt("photo_contact_path optional: ")
    return row


def _apply_robot_state_json(row: dict[str, Any], robot_state_json: Path) -> dict[str, Any]:
    state = _read_json(robot_state_json)
    if not state:
        return row
    row["_robot_command_state"] = state
    pose = state.get("actual_tcp_pose") or state.get("actual_tcp") or state.get("tcp_pose") or []
    if isinstance(pose, list):
        for key, value in zip([
            "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
            "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
        ], pose, strict=False):
            row[key] = value
    if state.get("robot_mode") is not None:
        row["robot_mode"] = state.get("robot_mode")
    if state.get("safety_mode") is not None:
        row["safety_mode"] = state.get("safety_mode")
    return row


def _run_robot_command_for_repeat(
    args: argparse.Namespace,
    case: dict[str, Any],
    row: dict[str, Any],
    session: Path,
    case_dir: Path,
    repeat_dir: Path,
    repeat_name: str,
    selected_anchor_id: str,
    pipeline_session: str,
) -> dict[str, Any]:
    template = args.robot_command_template or case.get("robot_command_template")
    if not template:
        raise RuntimeError("--robot-command-template is required for --robot-mode command")
    robot_state_json = repeat_dir / f"{repeat_name}_robot_state.json"
    values = {
        "case_id": row.get("case_id", ""),
        "anchor_type": row.get("anchor_type", ""),
        "anchor_id": row.get("anchor_id", ""),
        "repeat_index": row.get("repeat_index", ""),
        "session_dir": str(session),
        "case_dir": str(case_dir),
        "robot_state_json": str(robot_state_json),
        "selected_anchor_id": selected_anchor_id,
        "pipeline_session": pipeline_session,
    }
    command = str(template)
    for key, value in values.items():
        command = command.replace("{" + key + "}", str(value))
    print("\nROBOT COMMAND TO EXECUTE:")
    print(command)
    if not args.yes_run_robot_command:
        answer = input("Run this robot command? Type y then Enter to execute: ").strip().lower()
        if answer not in {"y", "yes"}:
            row["robot_motion_success"] = "no"
            row["failure_stage"] = "robot_command"
            row["failure_reason"] = "operator_declined_robot_command"
            return row
    completed = subprocess.run(command, cwd=REPO_ROOT, shell=True, text=True, capture_output=True, check=False)
    (repeat_dir / f"{repeat_name}_robot_command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (repeat_dir / f"{repeat_name}_robot_command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    row["robot_motion_success"] = "yes" if completed.returncode == 0 else "no"
    if completed.returncode != 0:
        row["failure_stage"] = "robot_command"
        row["failure_reason"] = f"returncode={completed.returncode}"
    if robot_state_json.exists():
        row = _apply_robot_state_json(row, robot_state_json)
    return row


def _capture_read_only_state(robot_ip: str) -> dict[str, Any]:
    from rtde_receive import RTDEReceiveInterface  # type: ignore  # noqa: PLC0415

    receive = RTDEReceiveInterface(robot_ip)
    state = {
        "timestamp": _now_iso(),
        "robot_ip": robot_ip,
        "read_only": True,
        "robot_motion_command_sent": False,
        "actual_tcp_pose": list(receive.getActualTCPPose()),
        "actual_q": list(receive.getActualQ()),
        "robot_mode": receive.getRobotMode(),
        "safety_mode": receive.getSafetyMode(),
    }
    disconnect = getattr(receive, "disconnect", None)
    if disconnect:
        disconnect()
    return state


def _run_analysis(session: Path) -> None:
    from upv_vlm_v2.experiments.analyze_e45_robot_upv_repeatability import analyze_robot, analyze_upv, write_figures

    rows = _read_csv(session / "e45_trials_raw.csv")
    analyze_robot(rows, session)
    analyze_upv(rows, session)
    write_figures(session)


def run_guided_collect(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    if args.robot_mode == "execute_hover_confirmed":
        run_execute_hover_confirmed(args, config, session)
        return
    if args.robot_mode == "read_only" and not args.robot_ip and not args.fake_data:
        raise RuntimeError("--robot-ip is required for --robot-mode read_only")
    if args.robot_mode == "command" and not (args.robot_command_template or any(c.get("robot_command_template") for c in _case_entries(config))):
        raise RuntimeError("--robot-command-template or case.robot_command_template is required for --robot-mode command")

    rows: list[dict[str, Any]] = []
    repeats = _repeat_count(config)
    cases = _case_entries(config)
    if not cases:
        raise RuntimeError("guided_collect requires at least one case in --case-plan or config.case_plan")
    for case in cases:
        case_id = str(case.get("case_id") or "case_unknown")
        pipeline_session_value = case.get("reuse_pipeline_session") or args.reuse_pipeline_session
        pipeline_info: dict[str, Any] = {}
        artifact_index: dict[str, Any] = {}
        if case.get("input_source_type") == "reuse_pipeline_session" or pipeline_session_value:
            pipeline_session = _resolve(pipeline_session_value)
            if pipeline_session is None or not pipeline_session.exists():
                raise FileNotFoundError(f"reuse pipeline session not found for {case_id}: {pipeline_session_value}")
            pipeline_info = _extract_pipeline_anchor_info(pipeline_session)
            artifact_index = _pipeline_artifact_index(pipeline_session, str(pipeline_info.get("selected_anchor_id") or ""))
            _copy_pipeline_refs(session, case_id, pipeline_session, pipeline_info)
        elif not args.allow_qwen_calls:
            raise RuntimeError(f"case {case_id} has no reused pipeline session and Qwen/main-pipeline calls are blocked")
        else:
            raise RuntimeError("guided_collect pipeline execution is not implemented here; reuse a main-pipeline session or run pipeline_only separately")

        good_anchor = str(case.get("good_anchor_id") or "")
        if str(case.get("good_anchor_policy", "")) == "proposed_selected":
            good_anchor = str(pipeline_info.get("selected_anchor_id") or good_anchor)
        bad_anchor = str(case.get("bad_anchor_id") or "")
        if not good_anchor:
            raise RuntimeError(f"case {case_id} could not resolve contact_usable anchor")
        if not bad_anchor:
            raise RuntimeError(f"case {case_id} requires bad_anchor_id")
        case_dir = _write_case_plan_guided(session, case, pipeline_info, artifact_index, good_anchor, bad_anchor)

        for anchor_type, anchor_id in [("contact_usable", good_anchor), ("contact_unsuitable", bad_anchor)]:
            repeat_dir = case_dir / _anchor_dir_name(anchor_type, anchor_id)
            for repeat_index in range(1, repeats + 1):
                _print_trial_block(case, anchor_type, anchor_id, repeat_index, repeats, pipeline_info.get("pipeline_session", ""), artifact_index)
                row = _base_trial_row(session, case, anchor_type, anchor_id, repeat_index)
                row["pipeline_case_dir"] = pipeline_info.get("pipeline_session", "")
                row["selected_anchor_id"] = good_anchor
                row["photo_overview_path"] = artifact_index.get("selected_anchor_overlay", "")
                repeat_name = f"repeat_{repeat_index:03d}"
                if args.robot_mode == "command":
                    row = _run_robot_command_for_repeat(
                        args=args,
                        case=case,
                        row=row,
                        session=session,
                        case_dir=case_dir,
                        repeat_dir=repeat_dir,
                        repeat_name=repeat_name,
                        selected_anchor_id=good_anchor,
                        pipeline_session=pipeline_info.get("pipeline_session", ""),
                    )
                if args.fake_data:
                    row = _fake_trial_values(row, repeat_index, preserve_robot_fields=args.robot_mode == "command")
                else:
                    input("Press Enter when ready to log this repeat, or Ctrl+C to stop.")
                    if args.robot_mode != "command":
                        row = _manual_trial_values(row, args.robot_mode, args)
                    else:
                        row["path_length_mm"] = _prompt("path_length_mm: ", str(row.get("path_length_mm") or ""))
                        row["upv_velocity_m_s"] = _prompt("Pundit PL-200 velocity m/s or blank: ")
                        row["arrival_time_us"] = _prompt("Pundit PL-200 arrival/transit time us or blank: ")
                        row["signal_detected"] = _prompt("signal_detected yes/no: ")
                        row["signal_quality"] = _prompt("signal_quality stable/unstable/no_signal/rejected: ")
                        row["operator_note"] = _prompt("operator_note: ")
                        row["photo_contact_path"] = _prompt("photo_contact_path optional: ")
                rows.append(row)
                _write_json(repeat_dir / f"{repeat_name}_robot_state.json", {
                    "planned_tcp_pose": [row.get(k, "") for k in [
                        "planned_tcp_x_m", "planned_tcp_y_m", "planned_tcp_z_m",
                        "planned_tcp_rx_rad", "planned_tcp_ry_rad", "planned_tcp_rz_rad",
                    ]],
                    "actual_tcp_pose": [row.get(k, "") for k in [
                        "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
                        "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
                    ]],
                    "robot_motion_success": row.get("robot_motion_success", ""),
                    "robot_mode": row.get("robot_mode", ""),
                    "safety_mode": row.get("safety_mode", ""),
                    "actual_q": (row.get("_robot_command_state") or {}).get("actual_q", ""),
                    "command_robot_state": row.get("_robot_command_state", {}),
                    "fake_data": bool(args.fake_data),
                })
                _write_json(repeat_dir / f"{repeat_name}_upv_entry.json", {
                    "path_length_mm": row.get("path_length_mm", ""),
                    "upv_velocity_m_s": row.get("upv_velocity_m_s", ""),
                    "arrival_time_us": row.get("arrival_time_us", ""),
                    "signal_detected": row.get("signal_detected", ""),
                    "signal_quality": row.get("signal_quality", ""),
                    "operator_note": row.get("operator_note", ""),
                    "photo_contact_path": row.get("photo_contact_path", ""),
                    "fake_data": bool(args.fake_data),
                })
                _save_rows_immediately(session, rows)
        _write_json(case_dir / "case_summary.json", {
            "case_id": case_id,
            "good_anchor_id": good_anchor,
            "bad_anchor_id": bad_anchor,
            "repeats_per_anchor": repeats,
            "pipeline_session": pipeline_info.get("pipeline_session", ""),
            "fake_data": bool(args.fake_data),
        })
    _save_rows_immediately(session, rows)
    _run_analysis(session)


def _copy_pipeline_refs(session: Path, case_id: str, pipeline_session: Path, info: dict[str, Any]) -> None:
    out = session / "cases" / case_id / "pipeline_result"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "reused_pipeline_session_reference.json", info)
    for rel in [
        "pipeline_result.json",
        "stage_status.json",
        "timing_summary.json",
        "artifacts/04_anchor_selection/qwen32_single_anchor_decision.json",
        "artifacts/04_anchor_selection/qwen32_single_anchor_score_table.csv",
        "artifacts/04_anchor_selection/final_anchor_decision_wide_context.json",
        "artifacts/04_anchor_selection/robot_anchor_geometry.json",
        "artifacts/04_anchor_selection/selected_anchor_overlay.png",
    ]:
        src = pipeline_session / rel
        if src.exists():
            dst = out / rel.replace("/", "__")
            shutil.copy2(src, dst)


def run_dry_or_pipeline(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    _write_case_plans(session, config, args.mode)
    rows = _blank_trial_rows(session, config)
    if args.mode == "pipeline_only":
        if args.reuse_pipeline_session:
            pipeline_session = _resolve(args.reuse_pipeline_session)
            if pipeline_session is None or not pipeline_session.exists():
                raise FileNotFoundError(f"reuse pipeline session not found: {args.reuse_pipeline_session}")
            info = _extract_pipeline_anchor_info(pipeline_session)
            cases = _case_entries(config)
            case = cases[0] if cases else {"case_id": "case_unknown", "material": "", "condition": ""}
            case_id = str(case.get("case_id") or "case_unknown")
            _copy_pipeline_refs(session, case_id, pipeline_session, info)
            _write_json(session / "cases" / case_id / "case_summary.json", info)
            for row in rows:
                if row.get("case_id") != case_id:
                    continue
                if row.get("anchor_type") == "contact_usable" and not row.get("anchor_id"):
                    row["anchor_id"] = info.get("selected_anchor_id", "")
                row["pipeline_case_dir"] = str(pipeline_session)
                row["selected_anchor_id"] = info.get("selected_anchor_id", "")
                row["photo_overview_path"] = info.get("selected_anchor_overlay_path", "")
        elif not args.allow_qwen_calls:
            raise RuntimeError("pipeline_only without --reuse-pipeline-session is blocked unless --allow-qwen-calls is true")
        else:
            run_existing_main_pipeline(args, config, session)

    _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
    _write_manual_template(session, rows)
    _write_empty_summaries(session)


def run_existing_main_pipeline(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    cases = _case_entries(config)
    if not cases:
        raise RuntimeError("No cases configured for pipeline run")
    case = cases[0]
    input_path = case.get("input_path")
    if not input_path:
        raise RuntimeError("pipeline run requires case_plan.cases[0].input_path or --reuse-pipeline-session")
    raise RuntimeError(
        "Qwen/main-pipeline execution is intentionally not implemented for this scaffold without explicit saved RGB-D fields. "
        "Use --reuse-pipeline-session for now."
    )


def run_read_robot_only(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    if not bool((config.get("robot") or {}).get("read_only_allowed", False)):
        raise RuntimeError("robot.read_only_allowed is false in config")
    robot_ip = args.robot_ip or (config.get("robot") or {}).get("robot_ip")
    if not robot_ip:
        raise RuntimeError("--robot-ip is required for read_robot_only")
    state = _capture_read_only_state(robot_ip)
    _write_json(session / "robot_read_only_state.json", state)
    rows = _blank_trial_rows(session, config)
    for row in rows:
        pose = state["actual_tcp_pose"]
        for key, value in zip([
            "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
            "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
        ], pose, strict=False):
            row[key] = value
        row["robot_mode"] = state["robot_mode"]
        row["safety_mode"] = state["safety_mode"]
    _write_case_plans(session, config, args.mode)
    _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
    _write_manual_template(session, rows)
    _write_empty_summaries(session)


def run_hover_plan_only(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    _write_case_plans(session, config, args.mode)
    rows = _blank_trial_rows(session, config)
    note = {
        "success": False,
        "failure_reason": "missing_calibration_or_pose",
        "message": "Base-frame transform/final pose is not available in anchor_only proof sessions. Generate plan_only outputs after calibration validation.",
    }
    for case in _case_entries(config):
        _write_json(session / "cases" / str(case.get("case_id", "case_unknown")) / "hover_plan_only_result.json", note)
    for row in rows:
        row["failure_stage"] = "hover_plan_only"
        row["failure_reason"] = "missing_calibration_or_pose"
    _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
    _write_manual_template(session, rows)
    _write_empty_summaries(session)


def run_manual_upv_entry(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    _write_case_plans(session, config, args.mode)
    rows = _blank_trial_rows(session, config)
    if args.manual_upv_csv:
        manual_rows = _read_csv(_resolve(args.manual_upv_csv) or Path(args.manual_upv_csv))
        by_key = {
            (r.get("case_id", ""), r.get("anchor_type", ""), r.get("anchor_id", ""), r.get("repeat_index", "")): r
            for r in manual_rows
        }
        for row in rows:
            key = (str(row.get("case_id", "")), str(row.get("anchor_type", "")), str(row.get("anchor_id", "")), str(row.get("repeat_index", "")))
            if key in by_key:
                for field in MANUAL_UPV_FIELDS:
                    if field in row:
                        row[field] = by_key[key].get(field, row.get(field, ""))
                row["upv_velocity_m_s"] = by_key[key].get("upv_velocity_m_s", "")
                row["arrival_time_us"] = by_key[key].get("arrival_time_us", "")
                row["signal_detected"] = by_key[key].get("signal_detected", "")
                row["signal_quality"] = by_key[key].get("signal_quality", "")
                row["operator_note"] = by_key[key].get("operator_note", "")
                row["photo_contact_path"] = by_key[key].get("photo_contact_path", "")
    _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
    _write_manual_template(session, rows)
    _write_empty_summaries(session)


def run_execute_hover_confirmed(args: argparse.Namespace, config: dict[str, Any], session: Path) -> None:
    robot = config.get("robot") or {}
    required = str(robot.get("required_motion_confirmation", "MOVE_TO_E45_HOVER"))
    blocked = []
    if not args.allow_robot_motion:
        blocked.append("missing --allow-robot-motion")
    if args.confirm_text != required:
        blocked.append(f"--confirm-text must be {required}")
    if not bool(robot.get("calibration_validation_acknowledged", False)):
        blocked.append("robot.calibration_validation_acknowledged is false")
    if not bool(robot.get("motion_allowed", False)):
        blocked.append("robot.motion_allowed is false")
    if blocked:
        _write_case_plans(session, config, args.mode)
        _write_json(session / "execute_hover_confirmed_blocked.json", {
            "success": False,
            "blocked": True,
            "reasons": blocked,
            "robot_motion_command_sent": False,
            "contact_motion_implemented": False,
            "clamp_motion_implemented": False,
            "automated_upv_trigger_implemented": False,
        })
        rows = _blank_trial_rows(session, config)
        for row in rows:
            row["failure_stage"] = "execute_hover_confirmed_precheck"
            row["failure_reason"] = "; ".join(blocked)
        _write_csv(session / "e45_trials_raw.csv", rows, RAW_FIELDS)
        _write_manual_template(session, rows)
        _write_empty_summaries(session)
        return
    raise RuntimeError("Hover execution is intentionally scaffolded only. Reuse R9 after adding an explicit reviewed adapter.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--mode", required=True, choices=[
        "dry_run", "pipeline_only", "read_robot_only", "hover_plan_only", "manual_upv_entry", "execute_hover_confirmed",
        "guided_collect",
    ])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)))
    parser.add_argument("--case-plan")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--robot-mode", choices=["none", "read_only", "external", "execute_hover_confirmed", "command"], default="none")
    parser.add_argument("--robot-command-template")
    parser.add_argument("--yes-run-robot-command", action="store_true")
    parser.add_argument("--reuse-pipeline-session")
    parser.add_argument("--allow-qwen-calls", action="store_true")
    parser.add_argument("--robot-ip")
    parser.add_argument("--manual-upv-csv")
    parser.add_argument("--allow-robot-motion", action="store_true")
    parser.add_argument("--confirm-text", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    config = _load_yaml(config_path)
    if args.case_plan:
        case_plan_path = _resolve(args.case_plan)
        if case_plan_path is None or not case_plan_path.exists():
            raise FileNotFoundError(f"Case plan not found: {args.case_plan}")
        config = _apply_case_plan(config, _load_yaml(case_plan_path))
    session = _initialize_session(args, config)

    try:
        if args.mode in {"dry_run", "pipeline_only"}:
            run_dry_or_pipeline(args, config, session)
        elif args.mode == "read_robot_only":
            run_read_robot_only(args, config, session)
        elif args.mode == "hover_plan_only":
            run_hover_plan_only(args, config, session)
        elif args.mode == "manual_upv_entry":
            run_manual_upv_entry(args, config, session)
        elif args.mode == "execute_hover_confirmed":
            run_execute_hover_confirmed(args, config, session)
        elif args.mode == "guided_collect":
            run_guided_collect(args, config, session)
        else:  # pragma: no cover
            raise RuntimeError(f"unsupported mode: {args.mode}")
        print(f"e45_session: {session}")
        print(f"e45_trials_raw_csv: {session / 'e45_trials_raw.csv'}")
        print(f"manual_upv_entry_template_csv: {session / 'manual_upv_entry_template.csv'}")
        return 0
    except Exception as exc:
        _write_json(session / "error.json", {"success": False, "error": str(exc), "mode": args.mode})
        print(f"e45_session: {session}")
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
