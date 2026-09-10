"""T4-style robot + Arduino clamp execution sequence for v2.

The real backend uses RTDE only after external safety gates have passed. The
sequence mirrors T4 full_final_home: Arduino check, mid-hover, orient, XY,
preview approach, final approach, clamp travel, hold, release/open, home.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from upv_vlm_v2.planning.base_frame_planner import build_real_robot_motion_plan_from_current_tcp


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _motion_config(config: dict[str, Any], stage: str) -> tuple[float, float]:
    motion = (config.get("robot") or {}).get("motion") or {}
    defaults = config.get("robot") or {}
    speed = float(defaults.get("speed_m_s", 0.008))
    accel = float(defaults.get("accel_m_s2", 0.015))
    if stage == "move_midhover":
        return float(motion.get("speed_midhover_m_s", 0.04)), float(motion.get("accel_midhover_m_s2", 0.08))
    if stage == "orient":
        return float(motion.get("speed_rotate_m_s", 0.05)), float(motion.get("accel_rotate_m_s2", 0.1))
    if stage == "xy":
        return float(motion.get("speed_xy_m_s", 0.03)), float(motion.get("accel_xy_m_s2", 0.06))
    if stage == "approach_preview":
        return float(motion.get("speed_preview_z_m_s", 0.008)), float(motion.get("accel_preview_z_m_s2", 0.015))
    if stage == "approach_final":
        return float(motion.get("speed_final_z_m_s", 0.004)), float(motion.get("accel_final_z_m_s2", 0.01))
    if stage == "home":
        return float(motion.get("speed_home_m_s", 0.08)), float(motion.get("accel_home_m_s2", 0.15))
    return speed, accel


def _move_stage(robot: Any, plan: dict[str, Any], config: dict[str, Any], stage: str, pose_key: str) -> dict[str, Any]:
    pose = plan.get(pose_key)
    if pose is None:
        return {
            "stage": stage,
            "pose_key": pose_key,
            "target_pose": None,
            "success": False,
            "failure_reason": f"{pose_key}_missing",
        }
    speed, accel = _motion_config(config, stage)
    before = robot.snapshot()
    robot.move_l(pose, speed, accel, stage)
    after = robot.snapshot()
    return {
        "stage": stage,
        "pose_key": pose_key,
        "target_pose": pose,
        "speed_m_s": speed,
        "accel_m_s2": accel,
        "actual_before": before.get("actual_tcp_pose"),
        "actual_after": after.get("actual_tcp_pose"),
        "robot_mode_after": after.get("robot_mode"),
        "safety_mode_after": after.get("safety_mode"),
        "success": True,
    }


def _stage_succeeded(row: dict[str, Any]) -> bool:
    return bool(row.get("success"))


def _clamp_to_planned_width(clamp: Any, robot_plan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    clamp_cfg = config.get("clamp", {})
    clamp.configure_from_config(config)
    if bool(clamp_cfg.get("configure_before_clamp", True)):
        clamp.set_steps_per_mm(float(clamp_cfg.get("steps_per_mm", 100.0)))
        clamp.set_step_delay_us(int(clamp_cfg.get("step_delay_us", 600)))
    opening = float(robot_plan["clamp_opening_mm"])
    fully_open = float(clamp_cfg.get("fully_open_probe_spacing_mm", 257.0))
    fully_closed = float(clamp_cfg.get("fully_closed_probe_spacing_mm", 40.0))
    if not (fully_closed <= opening <= fully_open):
        return {"stage": "clamp_to_planned_width", "success": False, "failure_reason": "clamp_opening_out_of_range", "opening_mm": opening}
    responses: list[str] = []
    if bool(clamp_cfg.get("open_before_clamp", True)):
        responses.extend(clamp.open_full())
    mode = str(clamp_cfg.get("clamp_command_mode", "clamp_travel"))
    if mode == "clamp_travel":
        travel = fully_open - opening
        responses.extend(clamp.clamp_travel_mm(travel))
        command_sent = f"CLAMP_TRAVEL_MM {travel:.3f}"
    else:
        responses.extend(clamp.move_to_spacing_mm(opening))
        command_sent = f"MOVE_TO_SPACING_MM {opening:.3f}"
    return {
        "stage": "clamp_to_planned_width",
        "success": True,
        "recommended_clamp_opening_mm": opening,
        "fully_open_probe_spacing_mm": fully_open,
        "total_clamp_closing_mm": fully_open - opening,
        "clamp_command_mode": mode,
        "actual_serial_command_sent": command_sent,
        "responses": responses,
    }


def run_t4_style_execution(
    *,
    backend: str,
    robot: Any,
    clamp: Any,
    robot_plan_payload: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "backend": backend,
        "sequence": "t4_full_final_home",
        "success": False,
        "overall_success": False,
        "motion_success": False,
        "clamp_success": "skipped",
        "home_success": False,
        "arduino_available": None,
        "clamp_required": bool((config.get("clamp") or {}).get("required", False)),
        "clamp_attempted": False,
        "clamp_failure_reason": None,
        "release_attempted": False,
        "release_success": "skipped",
        "home_attempted_after_execution": False,
        "home_success_after_execution": False,
        "final_robot_pose_if_available": None,
        "stages": [],
        "failure_reason": None,
    }
    executable_plan: dict[str, Any] = {}
    motion_stage_success = False
    hard_failure = False
    clamp_cfg = config.get("clamp", {}) or {}
    clamp_enabled = bool(clamp_cfg.get("enabled", True))
    clamp_required = bool(clamp_cfg.get("required", False))
    allow_missing_arduino = bool(clamp_cfg.get("allow_missing_arduino", False))
    home_requested = bool(config.get("execution", {}).get("return_home_after_execution", True))
    if not clamp_enabled:
        summary["clamp_success"] = "disabled"
        summary["arduino_available"] = "not_checked_clamp_disabled"
        summary["clamp_failure_reason"] = "clamp_disabled_in_config"
    preflight: dict[str, Any] = {
        "backend": backend,
        "sequence": "t4_full_final_home",
        "plan_has_clamp_opening": robot_plan_payload.get("clamp_opening_mm") is not None,
        "plan_has_upv_path_length": robot_plan_payload.get("upv_path_length_mm") is not None,
    }
    try:
        current = robot.snapshot()
        preflight["initial_robot_state"] = current
        if backend == "real":
            base_plan = build_real_robot_motion_plan_from_current_tcp(
                current_tcp_pose=current.get("actual_tcp_pose") or [],
                robot_plan_payload=robot_plan_payload,
                config=config,
            )
            preflight["base_frame_plan"] = base_plan
            if not base_plan.get("success"):
                summary["failure_reason"] = base_plan.get("failure_reason", "real_base_frame_plan_failed")
                _write_json(out / "real_preflight_check.json", preflight)
                return summary
            executable_plan = {**robot_plan_payload, **base_plan}
        else:
            executable_plan = {
                **robot_plan_payload,
                "move_midhover_pose": robot_plan_payload.get("hover_pose_base"),
                "orient_pose": robot_plan_payload.get("hover_pose_base"),
                "xy_pose": robot_plan_payload.get("hover_pose_base"),
                "approach_preview_pose": robot_plan_payload.get("approach_pose_base"),
                "approach_final_pose": robot_plan_payload.get("final_pose_base"),
                "pose_frame": robot_plan_payload.get("pose_frame", "simulated_or_tool0_relative"),
            }
            preflight["base_frame_plan"] = {"success": True, "simulated": True, "pose_frame": executable_plan.get("pose_frame")}
        _write_json(out / "real_preflight_check.json", preflight)

        arduino_ready = False
        if clamp_enabled:
            try:
                clamp.connect()
                clamp.configure_from_config(config)
                arduino_check = {"stage": "arduino_check", "success": True, "responses": clamp.ping()}
                summary["arduino_available"] = True
                arduino_ready = True
            except Exception as exc:  # noqa: BLE001
                arduino_check = {
                    "stage": "arduino_check",
                    "success": False,
                    "failure_reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "clamp_required": clamp_required,
                    "allow_missing_arduino": allow_missing_arduino,
                }
                summary["arduino_available"] = False
                summary["clamp_failure_reason"] = f"arduino_unavailable: {exc}"
                if clamp_required or not allow_missing_arduino:
                    hard_failure = True
                    summary["failure_reason"] = "arduino_unavailable_for_required_clamp"
                else:
                    arduino_check["skipped_downstream_clamp"] = True
                    summary["clamp_success"] = "skipped"
                    summary["clamp_failure_reason"] = "arduino_unavailable_optional_clamp"
            summary["stages"].append(arduino_check)
            _write_json(out / "arduino_check.json", arduino_check)

        if not hard_failure:
            motion_rows: list[dict[str, Any]] = []
            for stage, pose_key in [
                ("move_midhover", "move_midhover_pose"),
                ("orient", "orient_pose"),
                ("xy", "xy_pose"),
                ("approach_preview", "approach_preview_pose"),
                ("approach_final", "approach_final_pose"),
            ]:
                row = _move_stage(robot, executable_plan, config, stage, pose_key)
                summary["stages"].append(row)
                motion_rows.append(row)
                if not _stage_succeeded(row):
                    hard_failure = True
                    summary["failure_reason"] = row.get("failure_reason") or f"{stage}_failed"
                    break
            motion_stage_success = bool(motion_rows) and all(_stage_succeeded(row) for row in motion_rows)

        if clamp_enabled and arduino_ready and not hard_failure:
            summary["clamp_attempted"] = True
            clamp_result = _clamp_to_planned_width(clamp, executable_plan, config)
            summary["stages"].append(clamp_result)
            _write_json(out / "clamp_to_planned_width.json", clamp_result)
            if not clamp_result.get("success"):
                summary["clamp_success"] = False
                summary["clamp_failure_reason"] = clamp_result.get("failure_reason")
                try:
                    summary["release_attempted"] = True
                    summary["emergency_release_responses"] = clamp.emergency_release()
                    summary["release_success"] = True
                except Exception as exc:  # noqa: BLE001
                    summary["release_success"] = False
                    summary["release_failure_reason"] = str(exc)
                hard_failure = bool(clamp_required)
                if hard_failure:
                    summary["failure_reason"] = clamp_result.get("failure_reason")
            else:
                summary["clamp_success"] = True
                hold_sec = float(config.get("execution", {}).get("clamp_hold_sec", 5.0))
                clamp_hold = {"stage": "clamp_hold", "success": True, "responses": clamp.hold_ms(int(hold_sec * 1000.0))}
                summary["stages"].append(clamp_hold)
                _write_json(out / "clamp_hold.json", clamp_hold)
                summary["release_attempted"] = True
                clamp_release = {"stage": "clamp_release", "success": True, "responses": clamp.open_full()}
                summary["release_success"] = bool(clamp_release.get("success", True))
                summary["stages"].append(clamp_release)
                _write_json(out / "clamp_release.json", clamp_release)
        elif clamp_enabled and not arduino_ready:
            summary["clamp_attempted"] = False
            summary["clamp_success"] = "skipped"
            if not summary.get("clamp_failure_reason"):
                summary["clamp_failure_reason"] = "arduino_unavailable_optional_clamp"

        summary["motion_success"] = motion_stage_success
        return summary
    except Exception as exc:  # noqa: BLE001
        summary["failure_reason"] = str(exc)
        summary["exception_type"] = type(exc).__name__
        return summary
    finally:
        return_home_on_failure = bool((config.get("robot") or {}).get("return_home_on_failure", True))
        should_return_home = home_requested or (return_home_on_failure and bool(summary.get("failure_reason")))
        home_pose = (config.get("robot") or {}).get("home_pose")
        if should_return_home and home_pose:
            summary["home_attempted_after_execution"] = True
            try:
                executable_plan["home_pose"] = home_pose
                home_row = _move_stage(robot, executable_plan, config, "home", "home_pose")
                summary["stages"].append(home_row)
                summary["home_success_after_execution"] = _stage_succeeded(home_row)
                summary["home_success"] = summary["home_success_after_execution"]
                if not summary["home_success_after_execution"] and not summary.get("failure_reason"):
                    summary["failure_reason"] = home_row.get("failure_reason") or "home_return_failed"
            except Exception as exc:  # noqa: BLE001
                summary["home_success_after_execution"] = False
                summary["home_success"] = False
                summary["home_failure_reason"] = str(exc)
                if not summary.get("failure_reason"):
                    summary["failure_reason"] = f"home_return_failed: {exc}"
        elif should_return_home and not home_pose:
            summary["home_attempted_after_execution"] = False
            summary["home_success_after_execution"] = False
            summary["home_failure_reason"] = "home_pose_missing"
            if not summary.get("failure_reason"):
                summary["failure_reason"] = "home_pose_missing"

        try:
            final_state = robot.snapshot()
            summary["final_robot_pose_if_available"] = final_state.get("actual_tcp_pose")
            summary["final_robot_state_if_available"] = final_state
        except Exception as exc:  # noqa: BLE001
            summary["final_robot_pose_if_available"] = None
            summary["final_robot_state_failure_reason"] = str(exc)

        clamp_ok = summary.get("clamp_success") is True or (summary.get("clamp_success") in {"skipped", "disabled"} and not clamp_required)
        home_ok = bool(summary.get("home_success_after_execution")) if should_return_home else True
        overall = (
            bool(summary.get("motion_success"))
            and bool(clamp_ok)
            and bool(home_ok)
            and not bool(hard_failure)
            and not bool(summary.get("failure_reason"))
        )
        summary["overall_success"] = overall
        summary["success"] = overall
        _write_json(out / "actual_tcp_trace.json", {"backend": backend, "robot_commands": getattr(robot, "commands", [])})
        _write_json(out / "real_robot_execution_log.json" if backend == "real" else out / "simulated_robot_execution_log.json", {"backend": backend, "commands": getattr(robot, "commands", [])})
        _write_json(out / "real_clamp_execution_log.json" if backend == "real" else out / "simulated_clamp_execution_log.json", {"backend": backend, "commands": getattr(clamp, "commands", [])})
        _write_json(out / "execution_summary.json", summary)
