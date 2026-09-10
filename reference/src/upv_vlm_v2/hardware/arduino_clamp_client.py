"""Lazy Arduino clamp client with safe simulation backend."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any


class ArduinoClampClient:
    def __init__(self, *, backend: str = "simulated", port: str | None = None, baud: int = 115200, allow_real_hardware: bool = False) -> None:
        self.backend = backend
        self.port = port
        self.baud = baud
        self.allow_real_hardware = allow_real_hardware
        self.commands: list[dict[str, Any]] = []
        self._serial = None
        self.timeout_s = 5.0
        self.fully_open_probe_spacing_mm = 257.0
        self.steps_per_mm = 100.0
        self.step_delay_us = 600
        self.emergency_release_jog_ms = 3000

    def connect(self) -> None:
        if self.backend == "simulated":
            self.commands.append({"event": "connect_simulated", "port": self.port})
            return
        if not self.allow_real_hardware:
            raise PermissionError("Real Arduino connection requires allow_real_hardware=True")
        import serial  # type: ignore

        self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2.0)
        self._drain()
        self.commands.append({"event": "connect_real", "port": self.port, "baud": self.baud})

    def configure_from_config(self, config: dict[str, Any]) -> None:
        clamp = config.get("clamp", {})
        self.timeout_s = float(clamp.get("timeout_s", self.timeout_s))
        self.fully_open_probe_spacing_mm = float(clamp.get("fully_open_probe_spacing_mm", self.fully_open_probe_spacing_mm))
        self.steps_per_mm = float(clamp.get("steps_per_mm", self.steps_per_mm))
        self.step_delay_us = int(clamp.get("step_delay_us", self.step_delay_us))
        self.emergency_release_jog_ms = int(clamp.get("emergency_release_jog_ms", self.emergency_release_jog_ms))

    def _drain(self) -> None:
        if self._serial is None:
            return
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if not self._serial.readline():
                break

    def _estimated_timeout_for_steps(self, steps: float) -> tuple[float, float]:
        estimated = max(0.0, float(steps)) * 2.0 * float(self.step_delay_us) / 1_000_000.0
        return estimated, max(float(self.timeout_s), estimated + 10.0)

    def _send_command(self, command: str, expected_prefixes: tuple[str, ...], *, timeout_s: float | None = None, metadata: dict[str, Any] | None = None) -> list[str]:
        payload = command.strip()
        timeout_used = float(self.timeout_s if timeout_s is None else timeout_s)
        entry: dict[str, Any] = {
            "event": "clamp_command_simulated" if self.backend == "simulated" else "clamp_command_real",
            "command": payload,
            "expected_prefixes": list(expected_prefixes),
            "timeout_used_s": timeout_used,
            "responses": [],
        }
        if metadata:
            entry.update(metadata)
        self.commands.append(entry)
        if self.backend == "simulated":
            response = f"OK DRY_RUN {payload}"
            entry["responses"].append(response)
            entry["matched"] = response
            return [response]
        if self._serial is None:
            raise RuntimeError("Clamp serial client is not connected")
        self._serial.write((payload + "\n").encode("utf-8"))
        self._serial.flush()
        deadline = time.monotonic() + timeout_used
        lines: list[str] = []
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            entry["responses"].append(line)
            if line.startswith("ERR"):
                entry["error"] = line
                raise RuntimeError(f"Arduino rejected {payload}: {line}")
            if any(line.startswith(prefix) for prefix in expected_prefixes):
                entry["matched"] = line
                return lines
        entry["timeout"] = True
        raise TimeoutError(f"Timed out waiting for {expected_prefixes} after {payload}; lines={lines}")

    def command(self, name: str, value: float | None = None) -> None:
        if name == "OPEN":
            self.open_full()
        elif name == "CLOSE":
            if value is None:
                raise ValueError("CLOSE requires target opening mm for v2 T4-style protocol")
            self.move_to_spacing_mm(float(value))
        elif name == "CLAMP_TRAVEL_MM":
            if value is None:
                raise ValueError("CLAMP_TRAVEL_MM requires total travel mm")
            self.clamp_travel_mm(float(value))
        elif name == "HOLD":
            self.hold_ms(int(float(value or 0.0) * 1000.0))
        elif name == "RELEASE":
            self.open_full()
        else:
            payload = f"{name}" if value is None else f"{name} {value:.3f}"
            self._send_command(payload, ("OK",))

    def ping(self) -> list[str]:
        return self._send_command("PING", ("OK PONG",))

    def status(self) -> list[str]:
        return self._send_command("STATUS", ("OK STATUS",))

    def stop(self) -> list[str]:
        return self._send_command("STOP", ("OK STOPPED",))

    def set_steps_per_mm(self, value: float) -> list[str]:
        self.steps_per_mm = float(value)
        return self._send_command(f"SET_STEPS_PER_MM {float(value):.6f}", ("OK SET_STEPS_PER_MM",))

    def set_step_delay_us(self, value: int) -> list[str]:
        self.step_delay_us = int(value)
        return self._send_command(f"SET_STEP_DELAY_US {int(value)}", ("OK SET_STEP_DELAY_US",))

    def open_full(self) -> list[str]:
        max_steps = (self.fully_open_probe_spacing_mm / 2.0) * self.steps_per_mm
        estimated, timeout = self._estimated_timeout_for_steps(max_steps)
        return self._send_command(
            "OPEN_FULL",
            ("OK OPEN_FULL_DONE",),
            timeout_s=timeout,
            metadata={"estimated_worst_case_steps": max_steps, "estimated_move_time_s": estimated},
        )

    def move_to_spacing_mm(self, spacing_mm: float) -> list[str]:
        total_closing = self.fully_open_probe_spacing_mm - float(spacing_mm)
        one_side = total_closing / 2.0
        steps = one_side * self.steps_per_mm
        estimated, timeout = self._estimated_timeout_for_steps(steps)
        return self._send_command(
            f"MOVE_TO_SPACING_MM {float(spacing_mm):.3f}",
            ("OK MOVE_TO_SPACING_DONE",),
            timeout_s=timeout,
            metadata={
                "spacing_mm": float(spacing_mm),
                "total_closing_mm": total_closing,
                "one_side_motion_mm": one_side,
                "estimated_steps": steps,
                "estimated_move_time_s": estimated,
            },
        )

    def clamp_travel_mm(self, total_closing_mm: float) -> list[str]:
        one_side = float(total_closing_mm) / 2.0
        steps = one_side * self.steps_per_mm
        estimated, timeout = self._estimated_timeout_for_steps(steps)
        return self._send_command(
            f"CLAMP_TRAVEL_MM {float(total_closing_mm):.3f}",
            ("OK CLAMP_TRAVEL_DONE",),
            timeout_s=timeout,
            metadata={
                "total_closing_mm": float(total_closing_mm),
                "one_side_motion_mm": one_side,
                "estimated_steps": steps,
                "estimated_move_time_s": estimated,
            },
        )

    def hold_ms(self, ms: int) -> list[str]:
        timeout = max(float(self.timeout_s), float(ms) / 1000.0 + 5.0)
        return self._send_command(f"HOLD_MS {int(ms)}", ("OK HOLD_DONE",), timeout_s=timeout, metadata={"hold_ms": int(ms)})

    def jog_open_ms(self, ms: int) -> list[str]:
        timeout = max(float(self.timeout_s), float(ms) / 1000.0 + 5.0)
        return self._send_command(f"JOG_OPEN_MS {int(ms)}", ("OK JOG_OPEN_MS_DONE",), timeout_s=timeout, metadata={"jog_ms": int(ms)})

    def emergency_release(self) -> list[str]:
        responses: list[str] = []
        try:
            responses.extend(self.stop())
        except Exception:
            if self.backend != "simulated":
                raise
        responses.extend(self.jog_open_ms(int(self.emergency_release_jog_ms)))
        responses.extend(self.stop())
        responses.extend(self.status())
        return responses

    def write_log(self, path: str | Path) -> str:
        import json

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"backend": self.backend, "commands": self.commands}, indent=2), encoding="utf-8")
        return str(out)


def connect_clamp(*, backend: str = "simulated", port: str | None = None, baud: int = 115200, allow_real_hardware: bool = False) -> ArduinoClampClient:
    client = ArduinoClampClient(backend=backend, port=port, baud=baud, allow_real_hardware=allow_real_hardware)
    client.connect()
    return client
