"""Lazy UR RTDE client with safe simulation backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class URRTDEClient:
    def __init__(self, *, backend: str = "simulated", robot_ip: str | None = None, allow_real_hardware: bool = False) -> None:
        self.backend = backend
        self.robot_ip = robot_ip
        self.allow_real_hardware = allow_real_hardware
        self.commands: list[dict[str, Any]] = []
        self._control = None
        self._receive = None

    def connect(self) -> None:
        if self.backend == "simulated":
            self.commands.append({"event": "connect_simulated", "robot_ip": self.robot_ip})
            return
        if not self.allow_real_hardware:
            raise PermissionError("Real RTDE connection requires allow_real_hardware=True")
        from rtde_control import RTDEControlInterface  # type: ignore
        from rtde_receive import RTDEReceiveInterface  # type: ignore

        self._control = RTDEControlInterface(self.robot_ip)
        self._receive = RTDEReceiveInterface(self.robot_ip)
        self.commands.append({"event": "connect_real", "robot_ip": self.robot_ip})

    def snapshot(self) -> dict[str, Any]:
        if self.backend == "simulated":
            pose = [-0.04451708887209529, -0.30996360185460287, 0.44, 1.6641551719613543e-05, -3.1411789271540913, -8.944466236178852e-05]
            state = {
                "backend": "simulated",
                "actual_tcp_pose": pose,
                "actual_q": [0.0] * 6,
                "robot_mode": "SIMULATED",
                "safety_mode": "SIMULATED",
            }
            self.commands.append({"event": "snapshot_simulated", **state})
            return state
        if self._receive is None:
            raise RuntimeError("RTDE receive client is not connected")
        state = {
            "backend": "real",
            "actual_tcp_pose": [float(v) for v in self._receive.getActualTCPPose()],
            "actual_q": [float(v) for v in self._receive.getActualQ()],
            "robot_mode": self._receive.getRobotMode(),
            "safety_mode": self._receive.getSafetyMode(),
        }
        self.commands.append({"event": "snapshot_real", **state})
        return state

    def move_l(self, pose: list[float], speed: float, accel: float, label: str) -> None:
        if self.backend == "simulated":
            self.commands.append({"event": "moveL_simulated", "label": label, "pose": pose, "speed": speed, "accel": accel})
            return
        if self._control is None:
            raise RuntimeError("RTDE client is not connected")
        before = self.snapshot()
        ok = bool(self._control.moveL(pose, speed, accel))
        after = self.snapshot()
        self.commands.append({
            "event": "moveL_real",
            "label": label,
            "pose": pose,
            "speed": speed,
            "accel": accel,
            "moveL_return_value": ok,
            "actual_before": before.get("actual_tcp_pose"),
            "actual_after": after.get("actual_tcp_pose"),
        })

    def stop(self) -> None:
        if self.backend == "simulated":
            self.commands.append({"event": "stop_simulated"})
            return
        if self._control is not None:
            self._control.stopScript()

    def write_log(self, path: str | Path) -> str:
        import json

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"backend": self.backend, "commands": self.commands}, indent=2), encoding="utf-8")
        return str(out)


def connect_rtde(*, backend: str = "simulated", robot_ip: str | None = None, allow_real_hardware: bool = False) -> URRTDEClient:
    client = URRTDEClient(backend=backend, robot_ip=robot_ip, allow_real_hardware=allow_real_hardware)
    client.connect()
    return client


def read_only_rtde_status(*, robot_ip: str, allow_real_hardware: bool = False) -> dict[str, Any]:
    if not allow_real_hardware:
        raise PermissionError("Read-only RTDE status requires allow_real_hardware=True")
    from rtde_receive import RTDEReceiveInterface  # type: ignore

    receive = RTDEReceiveInterface(robot_ip)
    return {
        "robot_ip": robot_ip,
        "actual_tcp_pose": [float(v) for v in receive.getActualTCPPose()],
        "actual_q": [float(v) for v in receive.getActualQ()],
        "robot_mode": receive.getRobotMode(),
        "safety_mode": receive.getSafetyMode(),
        "read_only": True,
    }
