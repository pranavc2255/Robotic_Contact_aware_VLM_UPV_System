from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


class ClampSerialError(RuntimeError):
    """Raised when the Arduino clamp controller rejects or misses a command."""


@dataclass
class ArduinoClampController:
    port: str
    baud: int = 9600
    timeout_s: float = 5.0
    dry_run: bool = False
    fully_open_probe_spacing_mm: float = 245.0
    steps_per_mm: float = 100.0
    step_delay_us: int = 600
    emergency_release_jog_ms: int = 3000
    serial_handle: Any | None = None
    command_log: list[dict[str, Any]] = field(default_factory=list)

    def connect(self) -> None:
        if self.dry_run:
            self.command_log.append({"event": "connect", "dry_run": True, "port": self.port, "baud": self.baud})
            return
        if self.serial_handle is not None:
            return
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise ClampSerialError("pyserial is required for Arduino clamp control.") from exc
        self.serial_handle = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2.0)
        self._drain()
        self.command_log.append({"event": "connect", "dry_run": False, "port": self.port, "baud": self.baud})

    def close(self) -> None:
        if self.serial_handle is not None:
            self.serial_handle.close()
            self.serial_handle = None
        self.command_log.append({"event": "close"})

    def _drain(self) -> None:
        if self.serial_handle is None:
            return
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self.serial_handle.readline()
            if not line:
                break

    def _send_command(
        self,
        command: str,
        expected_prefixes: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        command = command.strip()
        timeout_used_s = float(self.timeout_s if timeout_s is None else timeout_s)
        entry: dict[str, Any] = {
            "command": command,
            "sent_at": time.time(),
            "dry_run": self.dry_run,
            "timeout_used_s": timeout_used_s,
            "responses": [],
        }
        if metadata:
            entry.update(metadata)
        self.command_log.append(entry)
        if self.dry_run:
            response = f"OK DRY_RUN {command}"
            entry["responses"].append(response)
            entry["matched"] = response
            return [response]
        if self.serial_handle is None:
            self.connect()
        assert self.serial_handle is not None
        self.serial_handle.write((command + "\n").encode("utf-8"))
        self.serial_handle.flush()
        deadline = time.monotonic() + timeout_used_s
        lines: list[str] = []
        while time.monotonic() < deadline:
            raw = self.serial_handle.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            entry["responses"].append(line)
            if line.startswith("ERR"):
                entry["error"] = line
                raise ClampSerialError(f"Arduino rejected {command}: {line}")
            if any(line.startswith(prefix) for prefix in expected_prefixes):
                entry["matched"] = line
                return lines
        entry["timeout"] = True
        raise ClampSerialError(f"Timed out waiting for {expected_prefixes} after command {command}; lines={lines}")

    def _estimated_timeout_for_steps(self, steps: float) -> tuple[float, float]:
        estimated_seconds = max(0.0, float(steps)) * 2.0 * float(self.step_delay_us) / 1_000_000.0
        return estimated_seconds, max(float(self.timeout_s), estimated_seconds + 10.0)

    def _spacing_move_metadata(self, spacing_mm: float) -> dict[str, Any]:
        total_closing_mm = float(self.fully_open_probe_spacing_mm) - float(spacing_mm)
        one_side_motion_mm = total_closing_mm / 2.0
        estimated_steps = one_side_motion_mm * float(self.steps_per_mm)
        estimated_move_time_s, timeout_used_s = self._estimated_timeout_for_steps(estimated_steps)
        return {
            "spacing_mm": float(spacing_mm),
            "total_closing_mm": total_closing_mm,
            "one_side_motion_mm": one_side_motion_mm,
            "estimated_steps": estimated_steps,
            "estimated_move_time_s": estimated_move_time_s,
            "computed_timeout_used_s": timeout_used_s,
        }

    def _total_closing_metadata(self, total_closing_mm: float) -> dict[str, Any]:
        one_side_motion_mm = float(total_closing_mm) / 2.0
        estimated_steps = one_side_motion_mm * float(self.steps_per_mm)
        estimated_move_time_s, timeout_used_s = self._estimated_timeout_for_steps(estimated_steps)
        return {
            "total_closing_mm": float(total_closing_mm),
            "one_side_motion_mm": one_side_motion_mm,
            "estimated_steps": estimated_steps,
            "estimated_move_time_s": estimated_move_time_s,
            "computed_timeout_used_s": timeout_used_s,
        }

    def ping(self) -> list[str]:
        return self._send_command("PING", ("OK PONG",))

    def status(self) -> list[str]:
        return self._send_command("STATUS", ("OK STATUS",))

    def stop(self) -> list[str]:
        return self._send_command("STOP", ("OK STOPPED",))

    def zero_open(self) -> list[str]:
        return self._send_command("ZERO_OPEN", ("OK ZERO_OPEN",))

    def open_full(self) -> list[str]:
        max_one_side_motion_mm = float(self.fully_open_probe_spacing_mm) / 2.0
        max_steps = max_one_side_motion_mm * float(self.steps_per_mm)
        estimated_move_time_s, timeout_used_s = self._estimated_timeout_for_steps(max_steps)
        return self._send_command(
            "OPEN_FULL",
            ("OK OPEN_FULL_DONE",),
            timeout_s=timeout_used_s,
            metadata={
                "estimated_worst_case_steps": max_steps,
                "estimated_move_time_s": estimated_move_time_s,
            },
        )

    def move_to_spacing_mm(self, spacing_mm: float) -> list[str]:
        metadata = self._spacing_move_metadata(float(spacing_mm))
        return self._send_command(
            f"MOVE_TO_SPACING_MM {float(spacing_mm):.3f}",
            ("OK MOVE_TO_SPACING_DONE",),
            timeout_s=float(metadata["computed_timeout_used_s"]),
            metadata=metadata,
        )

    def clamp_travel_mm(self, total_closing_mm: float) -> list[str]:
        metadata = self._total_closing_metadata(float(total_closing_mm))
        return self._send_command(
            f"CLAMP_TRAVEL_MM {float(total_closing_mm):.3f}",
            ("OK CLAMP_TRAVEL_DONE",),
            timeout_s=float(metadata["computed_timeout_used_s"]),
            metadata=metadata,
        )

    def hold_ms(self, ms: int) -> list[str]:
        timeout_s = max(float(self.timeout_s), float(ms) / 1000.0 + 5.0)
        return self._send_command(f"HOLD_MS {int(ms)}", ("OK HOLD_DONE",), timeout_s=timeout_s, metadata={"hold_ms": int(ms)})

    def close_steps(self, steps: int) -> list[str]:
        return self._send_command(f"CLOSE_STEPS {int(steps)}", ("OK CLOSE_STEPS_DONE",))

    def open_steps(self, steps: int) -> list[str]:
        return self._send_command(f"OPEN_STEPS {int(steps)}", ("OK OPEN_STEPS_DONE",))

    def force_open_steps(self, steps: int) -> list[str]:
        estimated_move_time_s, timeout_used_s = self._estimated_timeout_for_steps(int(steps))
        return self._send_command(
            f"FORCE_OPEN_STEPS {int(steps)}",
            ("OK FORCE_OPEN_STEPS_DONE",),
            timeout_s=timeout_used_s,
            metadata={"estimated_steps": int(steps), "estimated_move_time_s": estimated_move_time_s},
        )

    def jog_open_ms(self, ms: int) -> list[str]:
        timeout_s = max(float(self.timeout_s), float(ms) / 1000.0 + 5.0)
        return self._send_command(f"JOG_OPEN_MS {int(ms)}", ("OK JOG_OPEN_MS_DONE",), timeout_s=timeout_s, metadata={"jog_ms": int(ms)})

    def jog_close_ms(self, ms: int) -> list[str]:
        timeout_s = max(float(self.timeout_s), float(ms) / 1000.0 + 5.0)
        return self._send_command(f"JOG_CLOSE_MS {int(ms)}", ("OK JOG_CLOSE_MS_DONE",), timeout_s=timeout_s, metadata={"jog_ms": int(ms)})

    def emergency_release(self) -> list[str]:
        responses: list[str] = []
        try:
            responses.extend(self.stop())
        except ClampSerialError:
            if not self.dry_run:
                raise
        responses.extend(self.jog_open_ms(int(self.emergency_release_jog_ms)))
        responses.extend(self.stop())
        responses.extend(self.status())
        return responses

    def set_steps_per_mm(self, value: float) -> list[str]:
        self.steps_per_mm = float(value)
        return self._send_command(f"SET_STEPS_PER_MM {float(value):.6f}", ("OK SET_STEPS_PER_MM",))

    def set_step_delay_us(self, value: int) -> list[str]:
        self.step_delay_us = int(value)
        return self._send_command(f"SET_STEP_DELAY_US {int(value)}", ("OK SET_STEP_DELAY_US",))
