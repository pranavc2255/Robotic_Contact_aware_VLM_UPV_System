from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo


def _stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds").replace(":", "").replace("-", "").replace("T", "_").split("-")[0]


def _parse_last_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx in range(len(stdout) - 1, -1, -1):
        if stdout[idx] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not stdout[idx + end :].strip():
            return obj
    return {}


@dataclass
class T4RunnerAdapter:
    repo_root: Path
    t4_config_path: Path
    python_executable: str = sys.executable
    default_prompt_config: Path | None = None

    def __init__(
        self,
        repo_root: str | Path,
        t4_config_path: str | Path,
        python_executable: str | None = None,
        default_prompt_config: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.t4_config_path = self._resolve(t4_config_path)
        self.python_executable = python_executable or sys.executable
        self.default_prompt_config = self._resolve(default_prompt_config) if default_prompt_config else None

    def _resolve(self, path_text: str | Path | None) -> Path:
        if path_text is None:
            raise ValueError("path_text cannot be None")
        path = Path(path_text)
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def run_oneshot(
        self,
        action: str,
        confirmation: str = "",
        e1_run_dir: str | Path | None = None,
        prompt_config: str | Path | None = None,
        prompt_overrides_json: str | Path | None = None,
        extra_args: list[str] | None = None,
        progress_file: str | Path | None = None,
        live_progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        output_parent = Path(e1_run_dir) / "t4_linked_outputs" if e1_run_dir else None
        stamp = _stamp()
        cmd = [
            self.python_executable,
            str(self.repo_root / "terminal_scripts" / "run_T4_full_robot_clamp_test.py"),
            "--config",
            str(self.t4_config_path),
            "--one-shot",
            action,
        ]
        if output_parent is not None:
            cmd.extend(["--output-parent", str(output_parent), "--session-name", f"t4_{action}_{stamp}"])
        if action not in {"status", "home_debug", "snap", "plan", "plan_clamp", "perception_only", "plan_only"}:
            cmd.append("--i-understand-this-moves-robot-and-clamp")
        if confirmation:
            cmd.extend(["--confirm", confirmation])
        effective_prompt_config = prompt_config or self.default_prompt_config
        if effective_prompt_config:
            cmd.extend(["--prompt-config", str(self._resolve(effective_prompt_config))])
        if prompt_overrides_json:
            cmd.extend(["--prompt-overrides-json", str(self._resolve(prompt_overrides_json))])
        if progress_file:
            cmd.extend(["--progress-file", str(self._resolve(progress_file))])
        if extra_args:
            cmd.extend(extra_args)
        if progress_file and live_progress_callback:
            process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.repo_root)
            progress_path = self._resolve(progress_file)
            last_payload = ""
            while process.poll() is None:
                if progress_path.exists():
                    try:
                        payload = progress_path.read_text(encoding="utf-8")
                        if payload and payload != last_payload:
                            live_progress_callback(json.loads(payload))
                            last_payload = payload
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(0.5)
            stdout, stderr = process.communicate()
            if progress_path.exists():
                try:
                    live_progress_callback(json.loads(progress_path.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    pass
            returncode = process.returncode
        else:
            result = subprocess.run(cmd, text=True, capture_output=True, cwd=self.repo_root, timeout=None, check=False)
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        parsed = _parse_last_json_object(stdout)
        if not parsed:
            parsed = {"success": False, "action": action, "failure_reason": "T4_ONE_SHOT_JSON_SUMMARY_MISSING"}
        parsed.update({"returncode": returncode, "stdout": stdout, "stderr": stderr, "command": cmd})
        return parsed

    def status(self) -> dict[str, Any]:
        return self.run_oneshot("status")

    def home(self, confirmation: str = "HOME_ROBOT") -> dict[str, Any]:
        return self.run_oneshot("home", confirmation=confirmation)

    def arduino_check(self, confirmation: str = "ARDUINO_CHECK") -> dict[str, Any]:
        return self.run_oneshot("arduino_check", confirmation=confirmation)

    def clamp_zero_open(self, confirmation: str = "CLAMP_ZERO_OPEN") -> dict[str, Any]:
        return self.run_oneshot("clamp_zero_open", confirmation=confirmation)

    def clamp_open_full(self, confirmation: str = "CLAMP_OPEN_FULL") -> dict[str, Any]:
        return self.run_oneshot("clamp_open_full", confirmation=confirmation)

    def clamp_release(self, confirmation: str = "CLAMP_RELEASE") -> dict[str, Any]:
        return self.run_oneshot("clamp_release", confirmation=confirmation)

    def emergency_release(self, confirmation: str = "EMERGENCY_RELEASE") -> dict[str, Any]:
        return self.run_oneshot("emergency_release", confirmation=confirmation)

    def jog_stop(self, confirmation: str = "JOG_STOP") -> dict[str, Any]:
        return self.run_oneshot("jog_stop", confirmation=confirmation)

    def plan_only(self, prompt_config: str | Path | None = None, prompt_overrides_json: str | Path | None = None) -> dict[str, Any]:
        return self.run_oneshot("plan_only", prompt_config=prompt_config, prompt_overrides_json=prompt_overrides_json)

    def perception_only(self, prompt_config: str | Path | None = None, prompt_overrides_json: str | Path | None = None) -> dict[str, Any]:
        return self.run_oneshot("perception_only", prompt_config=prompt_config, prompt_overrides_json=prompt_overrides_json)

    def clamp_only(self, confirmation: str, manual_opening_mm: float | None = None) -> dict[str, Any]:
        extra = ["--manual-opening-mm", f"{manual_opening_mm:.3f}"] if manual_opening_mm is not None else None
        return self.run_oneshot("clamp_only", confirmation=confirmation, extra_args=extra)

    def full_final(self, confirmation: str, prompt_config: str | Path | None = None, prompt_overrides_json: str | Path | None = None) -> dict[str, Any]:
        return self.run_oneshot("full_final", confirmation=confirmation, prompt_config=prompt_config, prompt_overrides_json=prompt_overrides_json)

    def full_final_home(self, confirmation: str = "RUN_FINAL_HOME", prompt_config: str | Path | None = None, prompt_overrides_json: str | Path | None = None) -> dict[str, Any]:
        return self.run_oneshot("full_final_home", confirmation=confirmation, prompt_config=prompt_config, prompt_overrides_json=prompt_overrides_json)
