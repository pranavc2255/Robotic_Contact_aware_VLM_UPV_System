"""Operator-confirmed staged safe-gap execution for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r14_rtde_axis_aware_gap_motion_operator_confirmed.py
and terminal_scripts/run_r21_move_to_safe_observation_pose_operator_confirmed.py.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
from zoneinfo import ZoneInfo

from upv_vlm_orient.debug_robot_motion_rotation.robot.safe_observation import load_safe_observation_pose


class ExecutionError(RuntimeError):
    pass


def _timestamp_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _load_rtde_interfaces():
    try:
        from rtde_control import RTDEControlInterface  # noqa: PLC0415
        from rtde_receive import RTDEReceiveInterface  # noqa: PLC0415
    except ImportError as exc:
        raise ExecutionError("ur_rtde is required for staged gap execution.") from exc
    return RTDEReceiveInterface, RTDEControlInterface


def _write_substage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["stage", "timestamp", "command_sent", "command_pose", "actual_tcp_pose", "speed_m_s", "accel_m_s2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), list) else row.get(field) for field in fields})


def _move(control, receive, log: list[dict[str, Any]], stage: str, pose: list[float], speed: float, accel: float, settle_s: float) -> bool:
    ok = bool(control.moveL(pose, speed, accel))
    time.sleep(settle_s)
    log.append(
        {
            "stage": stage,
            "timestamp": _timestamp_now(),
            "command_sent": True,
            "command_pose": pose,
            "actual_tcp_pose": receive.getActualTCPPose(),
            "speed_m_s": speed,
            "accel_m_s2": accel,
        }
    )
    return ok


def execute_staged_gap_motion(plan: dict[str, Any], config: dict[str, Any], operator_confirmed: bool = False) -> dict[str, Any]:
    motion = config.get("motion", {})
    if not bool(motion.get("move_robot", False)):
        raise ExecutionError("motion.move_robot=false; refusing physical execution.")
    if bool(motion.get("require_operator_confirmation", True)) and not operator_confirmed:
        raise ExecutionError("Operator confirmation is required for physical execution.")
    orientation_check = plan.get("orientation_constant_check") or {}
    if not orientation_check.get("ok"):
        raise ExecutionError("Plan orientation is not constant through descent/retract.")

    RTDEReceiveInterface, RTDEControlInterface = _load_rtde_interfaces()
    receive = RTDEReceiveInterface(config["robot_ip"])
    control = RTDEControlInterface(config["robot_ip"])
    log: list[dict[str, Any]] = []
    output_dir = Path(plan.get("output_dir", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": _timestamp_now(),
        "robot_motion_command_sent": False,
        "contact_motion_sent": False,
        "force_mode_sent": False,
        "servo_sent": False,
        "actuator_command_sent": False,
        "command_sequence": [],
        "success": False,
    }
    try:
        log.append({"stage": "before_any_motion", "timestamp": _timestamp_now(), "command_sent": False, "actual_tcp_pose": receive.getActualTCPPose()})
        safe_pose = load_safe_observation_pose(config)
        settle_s = float(motion.get("readback_settle_s", 0.2))
        sequence = [
            ("safe_observation_start", safe_pose, motion.get("speed_safe_observation_m_s", 0.08), motion.get("accel_safe_observation_m_s2", 0.15)),
            ("uncorrected_prehover", plan["uncorrected_prehover_tcp_pose"], motion.get("speed_move_to_prehover_m_s", 0.08), motion.get("accel_move_to_prehover_m_s2", 0.15)),
            ("prehover_oriented", plan["prehover_oriented_pose"], motion.get("speed_rotate_at_prehover_m_s", 0.01), motion.get("accel_rotate_at_prehover_m_s2", 0.02)),
            ("corrected_prehover", plan["corrected_prehover_tcp_pose"], motion.get("speed_lateral_correction_m_s", 0.06), motion.get("accel_lateral_correction_m_s2", 0.12)),
            ("gap", plan["gap_tcp_pose"], motion.get("speed_descent_m_s", 0.025), motion.get("accel_descent_m_s2", 0.05)),
            ("retract", plan["retract_tcp_pose"], motion.get("speed_retract_m_s", 0.08), motion.get("accel_retract_m_s2", 0.15)),
            ("safe_observation_end", safe_pose, motion.get("speed_safe_observation_m_s", 0.08), motion.get("accel_safe_observation_m_s2", 0.15)),
        ]
        for stage, pose, speed, accel in sequence:
            ok = _move(control, receive, log, stage, pose, float(speed), float(accel), settle_s)
            result["robot_motion_command_sent"] = True
            result["command_sequence"].append(stage)
            if not ok:
                raise ExecutionError(f"moveL({stage}) returned false.")
            if stage == "gap":
                time.sleep(float(motion.get("hold_at_gap_s", 1.5)))
                log.append({"stage": "after_hold_at_gap", "timestamp": _timestamp_now(), "command_sent": False, "actual_tcp_pose": receive.getActualTCPPose()})
        result["success"] = True
    finally:
        result["substage_tcp_log"] = log
        csv_path = output_dir / "substage_tcp_log.csv"
        _write_substage_csv(csv_path, log)
        result["substage_tcp_log_csv"] = str(csv_path)
        for interface in (control, receive):
            disconnect = getattr(interface, "disconnect", None)
            if disconnect is not None:
                disconnect()
    return result

