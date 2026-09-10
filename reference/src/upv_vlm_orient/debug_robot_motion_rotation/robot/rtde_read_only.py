"""RTDE read-only robot state for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r4_rtde_read_only_state_once.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RobotStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class RobotState:
    connected: bool
    actual_tcp_pose: list[float] | None
    actual_q: list[float] | None
    robot_mode: Any
    safety_mode: Any
    raw: dict[str, Any]


def read_robot_state(robot_ip: str) -> RobotState:
    try:
        from rtde_receive import RTDEReceiveInterface  # noqa: PLC0415
    except ImportError as exc:
        raise RobotStateError("ur_rtde is required for RTDE read-only state.") from exc
    receive = None
    try:
        receive = RTDEReceiveInterface(robot_ip)
        raw = {
            "actual_tcp_pose": receive.getActualTCPPose(),
            "actual_q": receive.getActualQ(),
            "robot_mode": receive.getRobotMode(),
            "safety_mode": receive.getSafetyMode(),
        }
        return RobotState(
            connected=True,
            actual_tcp_pose=raw["actual_tcp_pose"],
            actual_q=raw["actual_q"],
            robot_mode=raw["robot_mode"],
            safety_mode=raw["safety_mode"],
            raw=raw,
        )
    finally:
        disconnect = getattr(receive, "disconnect", None)
        if disconnect is not None:
            disconnect()

