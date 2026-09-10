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


CSV_COLUMNS = [
    "session_id",
    "cycle_index",
    "timestamp_start",
    "timestamp_end",
    "requested_material",
    "axis_mode",
    "pipeline_config",
    "pipeline_returncode",
    "pipeline_session_path",
    "pipeline_result_path",
    "pipeline_success",
    "failure_reason",
    "selected_object_class",
    "selected_candidate_id",
    "selected_anchor_id",
    "qwen_selected_anchor",
    "anchor_score",
    "mask_path_length_mm",
    "depth_path_length_mm",
    "upv_path_length_mm",
    "clamp_opening_mm",
    "robot_moved",
    "clamp_moved",
    "home_returned",
    "arduino_available",
    "clamp_attempted",
    "clamp_success",
    "release_attempted",
    "release_success",
    "home_success",
    "upv_velocity_m_s",
    "arrival_time_us",
    "signal_detected",
    "signal_quality",
    "operator_note",
    "photo_path",
    "upv_device_reading_id",
    "path_length_mm_operator_override",
    "cycle_dir",
]


def _now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("session_%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / _now_stamp()
    idx = 2
    while candidate.exists():
        candidate = output_root / f"{_now_stamp()}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True)
    return candidate


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


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
        str(args.confirm_token),
        "--summary-only",
        "--print-result-json",
    ]


def _parse_pipeline_result_path(stdout: str) -> Path | None:
    match = re.search(r"Full JSON saved at:\s*(.+pipeline_result\.json)", stdout)
    if not match:
        return None
    path = Path(match.group(1).strip())
    return path if path.exists() else None


def _latest_pipeline_result_after(start_time: float) -> Path | None:
    root = Path("outputs/v2_pipeline_runs")
    if not root.exists():
        return None
    candidates = sorted(root.glob("session_*/pipeline_result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.stat().st_mtime >= start_time:
            return path
    return None


def _get_axis_payload(result: dict[str, Any], key: str, axis: str) -> dict[str, Any]:
    payload = result.get(key) or {}
    if isinstance(payload, dict):
        item = payload.get(axis)
        return item if isinstance(item, dict) else {}
    return {}


def _extract_cycle_fields(result: dict[str, Any], axis: str) -> dict[str, Any]:
    target = result.get("target") or {}
    anchor = _get_axis_payload(result, "anchors", axis)
    path_length = _get_axis_payload(result, "path_lengths", axis)
    plan = result.get("robot_plan") or {}
    execution = result.get("execution") or {}
    execution_diag = execution.get("diagnostics") or {}
    anchor_diag = anchor.get("diagnostics") or {}
    return {
        "pipeline_success": result.get("success"),
        "failure_reason": result.get("failure_reason"),
        "selected_object_class": target.get("requested_material"),
        "selected_candidate_id": target.get("selected_candidate_id"),
        "selected_anchor_id": anchor.get("final_anchor_id"),
        "qwen_selected_anchor": anchor_diag.get("qwen_selected_anchor_id"),
        "anchor_score": anchor.get("score"),
        "mask_path_length_mm": path_length.get("mask_path_length_mm"),
        "depth_path_length_mm": path_length.get("depth_path_length_mm"),
        "upv_path_length_mm": plan.get("upv_path_length_mm"),
        "clamp_opening_mm": plan.get("clamp_opening_mm"),
        "robot_moved": execution.get("robot_moved"),
        "clamp_moved": execution.get("clamp_moved"),
        "home_returned": execution.get("home_returned"),
        "arduino_available": execution_diag.get("arduino_available"),
        "clamp_attempted": execution_diag.get("clamp_attempted"),
        "clamp_success": execution_diag.get("clamp_success"),
        "release_attempted": execution_diag.get("release_attempted"),
        "release_success": execution_diag.get("release_success"),
        "home_success": execution_diag.get("home_success_after_execution", execution_diag.get("home_success")),
    }


def _fake_pipeline_result(cycle_index: int, material: str, axis: str, session_dir: Path) -> dict[str, Any]:
    fake_pipeline_dir = session_dir / "fake_pipeline_runs" / f"cycle_{cycle_index:03d}"
    fake_pipeline_dir.mkdir(parents=True, exist_ok=True)
    return {
        "success": True,
        "mode": "execute",
        "requested_material": material,
        "axis_mode": axis,
        "session_dir": str(fake_pipeline_dir),
        "target": {"requested_material": material, "selected_candidate_id": "candidate_001"},
        "anchors": {axis: {"final_anchor_id": "A4", "score": 0.91, "diagnostics": {"qwen_selected_anchor_id": "A4"}}},
        "path_lengths": {axis: {"mask_path_length_mm": 120.0, "depth_path_length_mm": 119.4}},
        "robot_plan": {"upv_path_length_mm": 119.4, "clamp_opening_mm": 124.4},
        "execution": {
            "success": True,
            "robot_moved": True,
            "clamp_moved": True,
            "home_returned": True,
            "diagnostics": {
                "arduino_available": True,
                "clamp_attempted": True,
                "clamp_success": True,
                "release_attempted": True,
                "release_success": True,
                "home_success_after_execution": True,
            },
        },
    }


def _prompt_upv(skip: bool, dry_run: bool, cycle_index: int) -> dict[str, Any]:
    if dry_run:
        return {
            "upv_velocity_m_s": 3400.0 if cycle_index == 1 else 3405.0,
            "arrival_time_us": 63.0 if cycle_index == 1 else 62.9,
            "signal_detected": "yes",
            "signal_quality": "stable",
            "operator_note": "dry_run_fake_upv_entry",
            "photo_path": "",
            "upv_device_reading_id": f"DRY{cycle_index:03d}",
            "path_length_mm_operator_override": "",
        }
    if skip:
        return {
            "upv_velocity_m_s": "",
            "arrival_time_us": "",
            "signal_detected": "",
            "signal_quality": "",
            "operator_note": "",
            "photo_path": "",
            "upv_device_reading_id": "",
            "path_length_mm_operator_override": "",
        }
    print("\nEnter Pundit PL-200 reading for this cycle.")
    return {
        "upv_velocity_m_s": _prompt("velocity_m_s"),
        "arrival_time_us": _prompt("arrival_time_us"),
        "signal_detected": _prompt("signal_detected yes/no"),
        "signal_quality": _prompt("signal_quality stable/unstable/no_signal/rejected"),
        "operator_note": _prompt("operator_note"),
        "photo_path": _prompt("photo_path optional"),
        "upv_device_reading_id": _prompt("upv_device_reading_id optional"),
        "path_length_mm_operator_override": _prompt("path_length_mm override optional"),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    velocities_by_key: dict[str, list[float]] = {}
    for row in rows:
        velocity = _safe_float(row.get("upv_velocity_m_s"))
        if velocity is not None:
            key = f"{row.get('requested_material')}|{row.get('axis_mode')}"
            velocities_by_key.setdefault(key, []).append(velocity)
    velocity_stats: dict[str, dict[str, Any]] = {}
    for key, values in velocities_by_key.items():
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) >= 2 else 0.0
        velocity_stats[key] = {
            "n": len(values),
            "mean_velocity_m_s": mean,
            "std_velocity_m_s": std,
            "cv_velocity_percent": (100.0 * std / mean) if mean and len(values) >= 2 else None,
        }
    return {
        "total_cycles": len(rows),
        "successful_pipeline_cycles": sum(str(row.get("pipeline_success")).lower() == "true" for row in rows),
        "failed_pipeline_cycles": sum(str(row.get("pipeline_success")).lower() != "true" for row in rows),
        "clamp_attempted_count": sum(str(row.get("clamp_attempted")).lower() == "true" for row in rows),
        "clamp_success_count": sum(str(row.get("clamp_success")).lower() == "true" for row in rows),
        "valid_upv_reading_count": sum(_safe_float(row.get("upv_velocity_m_s")) is not None for row in rows),
        "stable_signal_count": sum(str(row.get("signal_quality")).lower() == "stable" for row in rows),
        "velocity_by_material_axis": velocity_stats,
        "cycle_dirs": [row.get("cycle_dir") for row in rows],
    }


def _write_summary(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = _summarize(rows)
    summary_dir = session_dir / "summary"
    _write_json(summary_dir / "session_summary.json", summary)
    csv_path = summary_dir / "session_summary.csv"
    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        flat_rows.append({"metric": key, "value": value})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(flat_rows)


def _copy_pipeline_result(source: Path | None, cycle_dir: Path) -> tuple[str, str, dict[str, Any] | None]:
    if not source or not source.exists():
        return "", "", None
    result = _read_json(source)
    copied = cycle_dir / "pipeline_result.json"
    shutil.copy2(source, copied)
    return str(source.parent), str(copied), result


def _run_cycle(args: argparse.Namespace, session_dir: Path, cycle_index: int, material: str, axis: str) -> dict[str, Any]:
    cycle_dir = session_dir / "cycles" / f"cycle_{cycle_index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    timestamp_start = _now_iso()
    request = {
        "cycle_index": cycle_index,
        "requested_material": material,
        "axis_mode": axis,
        "pipeline_config": str(args.pipeline_config),
        "dry_run": bool(args.dry_run),
    }
    _write_json(cycle_dir / "cycle_request.json", request)

    returncode = 0
    pipeline_result_source: Path | None = None
    pipeline_result: dict[str, Any] | None = None
    if args.dry_run:
        pipeline_result = _fake_pipeline_result(cycle_index, material, axis, session_dir)
        pipeline_result_source = cycle_dir / "pipeline_result.json"
        _write_json(pipeline_result_source, pipeline_result)
        (cycle_dir / "pipeline_session_path.txt").write_text(str(Path(pipeline_result["session_dir"])), encoding="utf-8")
        (cycle_dir / "pipeline_stdout.txt").write_text("dry_run fake pipeline stdout\n", encoding="utf-8")
        (cycle_dir / "pipeline_stderr.txt").write_text("", encoding="utf-8")
    else:
        if bool(args.confirm_each_cycle) and not args.auto_confirm_pipeline:
            token = input(f"Type {args.confirm_token} to run real pipeline cycle, or q to quit: ").strip()
            if token.lower() == "q":
                raise KeyboardInterrupt
            if token != args.confirm_token:
                raise RuntimeError("confirmation_token_mismatch")
        command = _pipeline_command(args, material, axis)
        print("\nRunning full pipeline command:")
        print(" ".join(command))
        start_mtime = datetime.now().timestamp()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        returncode = int(completed.returncode)
        (cycle_dir / "pipeline_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (cycle_dir / "pipeline_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        pipeline_result_source = _parse_pipeline_result_path(completed.stdout) or _latest_pipeline_result_after(start_mtime)
        pipeline_session, copied_result, pipeline_result = _copy_pipeline_result(pipeline_result_source, cycle_dir)
        (cycle_dir / "pipeline_session_path.txt").write_text(pipeline_session, encoding="utf-8")
        if copied_result:
            pipeline_result_source = Path(copied_result)

    if pipeline_result is None and pipeline_result_source and pipeline_result_source.exists():
        pipeline_result = _read_json(pipeline_result_source)
    extracted = _extract_cycle_fields(pipeline_result or {}, axis)
    upv = _prompt_upv(bool(args.skip_upv_entry), bool(args.dry_run), cycle_index)
    _write_json(cycle_dir / "upv_reading.json", upv)
    timestamp_end = _now_iso()
    row = {
        "session_id": session_dir.name,
        "cycle_index": cycle_index,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "requested_material": material,
        "axis_mode": axis,
        "pipeline_config": str(args.pipeline_config),
        "pipeline_returncode": returncode,
        "pipeline_session_path": (pipeline_result or {}).get("session_dir", ""),
        "pipeline_result_path": str(pipeline_result_source or ""),
        "cycle_dir": str(cycle_dir),
        **extracted,
        **upv,
    }
    _write_json(cycle_dir / "cycle_summary.json", row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive repeated UPV robot session runner.")
    parser.add_argument("--pipeline-config", default="configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_single_anchor_clamp_enabled.yaml")
    parser.add_argument("--output-root", default="outputs/v2_interactive_upv_sessions")
    parser.add_argument("--default-material", default="brick")
    parser.add_argument("--default-axis", choices=["major", "minor"], default="major")
    parser.add_argument("--confirm-token", default="RUN_V2_EXECUTE")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-confirm-pipeline", action="store_true")
    parser.add_argument("--confirm-once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-each-cycle", action="store_true")
    parser.add_argument("--skip-upv-entry", action="store_true")
    parser.add_argument("--server-url", default=None)
    return parser.parse_args()


def _confirmation_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.auto_confirm_pipeline:
        return "auto_confirm"
    if args.confirm_each_cycle:
        return "confirm_each_cycle"
    return "confirm_once" if args.confirm_once else "confirm_each_cycle"


def _startup_confirmation(args: argparse.Namespace) -> None:
    mode = _confirmation_mode(args)
    if mode != "confirm_once":
        return
    print("")
    print("This interactive session can execute real robot motion repeatedly.")
    print(f"Type {args.confirm_token} once to arm this interactive session.")
    token = input("Confirmation token: ").strip()
    if token != args.confirm_token:
        raise SystemExit("confirmation_token_mismatch: interactive session was not armed")


def main() -> int:
    args = parse_args()
    session_dir = _make_session_dir(Path(args.output_root))
    material = str(args.default_material)
    axis = str(args.default_axis)
    session_log_csv = session_dir / "interactive_session_log.csv"
    session_log_jsonl = session_dir / "interactive_session_log.jsonl"
    rows: list[dict[str, Any]] = []
    confirmation_mode = _confirmation_mode(args)
    manifest = {
        "session_id": session_dir.name,
        "created_at": _now_iso(),
        "pipeline_config": str(args.pipeline_config),
        "output_root": str(args.output_root),
        "default_material": material,
        "default_axis": axis,
        "dry_run": bool(args.dry_run),
        "auto_confirm_pipeline": bool(args.auto_confirm_pipeline),
        "confirm_once": bool(args.confirm_once),
        "confirm_each_cycle": bool(args.confirm_each_cycle),
        "confirmation_mode": confirmation_mode,
        "confirm_token_used": str(args.confirm_token),
        "skip_upv_entry": bool(args.skip_upv_entry),
        "server_url": args.server_url,
        "safety": {
            "orchestrates_existing_full_pipeline_cli": True,
            "does_not_import_rtde_or_arduino": True,
            "does_not_trigger_pundit": True,
        },
    }
    _write_json(session_dir / "interactive_session_manifest.json", manifest)
    (session_dir / "operator_notes.md").write_text(
        "# Interactive UPV Robot Session Notes\n\n"
        "- Keep workspace clear.\n"
        "- Keep E-stop ready.\n"
        "- Confirm Arduino is connected if clamp actuation is desired.\n"
        "- Confirm Qwen server and ROS2 RealSense are active before real cycles.\n",
        encoding="utf-8",
    )
    print("Interactive UPV robot session")
    print(f"Pipeline config: {args.pipeline_config}")
    print(f"Output session: {session_dir}")
    print("Safety reminder: workspace clear, E-stop ready, Arduino connected if clamp desired, Qwen/ROS2 active for real mode.")
    if not args.dry_run:
        material = _prompt("Requested material", material)
        axis = _prompt("Axis mode major/minor", axis)
        _startup_confirmation(args)

    cycle_index = 1
    try:
        while True:
            if args.max_cycles is not None and cycle_index > args.max_cycles:
                break
            if not args.dry_run:
                print(f"\nCycle {cycle_index}: material={material}, axis={axis}")
                action = input("Press ENTER to run cycle, 'c' to change material/axis/object, 'q' to quit: ").strip().lower()
                if action == "q":
                    break
                if action == "c":
                    _append_jsonl(session_log_jsonl, {"event": "object_change_event", "cycle_index": cycle_index, "timestamp": _now_iso()})
                    print("Change object/material as needed, then enter the next request.")
                    material = _prompt("Requested material", material)
                    axis = _prompt("Axis mode major/minor", axis)
                    continue
            row = _run_cycle(args, session_dir, cycle_index, material, axis)
            rows.append(row)
            _append_csv(session_log_csv, row)
            _append_jsonl(session_log_jsonl, row)
            print(f"Cycle {cycle_index} complete: anchor={row.get('selected_anchor_id')} pipeline_success={row.get('pipeline_success')} clamp={row.get('clamp_success')} home={row.get('home_success')}")
            cycle_index += 1
    except KeyboardInterrupt:
        print("\nInteractive session stopped by operator.")
    finally:
        _write_summary(session_dir, rows)
        print(f"Session summary: {session_dir / 'summary' / 'session_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
