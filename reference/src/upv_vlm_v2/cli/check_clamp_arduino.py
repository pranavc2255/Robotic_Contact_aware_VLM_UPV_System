from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from upv_vlm_v2.config import load_config, validate_config_schema
from upv_vlm_v2.hardware.arduino_clamp_client import ArduinoClampClient


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def check_clamp_arduino(
    *,
    config_path: str | Path,
    test_open_close: bool = False,
    confirm_test_open_close: str | None = None,
) -> dict[str, Any]:
    config = validate_config_schema(load_config(config_path))
    clamp_cfg = config.get("clamp") or {}
    port = str(clamp_cfg.get("serial_port") or "")
    baud = int(clamp_cfg.get("baud", 115200))
    port_exists = bool(port) and Path(port).exists()
    result: dict[str, Any] = {
        "config_path": str(config_path),
        "serial_port": port,
        "baud": baud,
        "port_exists": port_exists,
        "clamp_enabled_in_config": bool(clamp_cfg.get("enabled", False)),
        "clamp_required": bool(clamp_cfg.get("required", False)),
        "allow_missing_arduino": bool(clamp_cfg.get("allow_missing_arduino", False)),
        "arduino_available": False,
        "firmware_response": None,
        "status_response": None,
        "safe_to_enable_clamp": False,
        "test_open_close_requested": bool(test_open_close),
        "test_open_close_performed": False,
        "failure_reason": None,
    }
    if not port_exists:
        result["failure_reason"] = "serial_port_missing"
        return result

    client = ArduinoClampClient(backend="real", port=port, baud=baud, allow_real_hardware=True)
    try:
        client.connect()
        client.configure_from_config(config)
        ping = client.ping()
        status = client.status()
        result["arduino_available"] = True
        result["firmware_response"] = ping
        result["status_response"] = status
        result["safe_to_enable_clamp"] = True
        if test_open_close:
            if confirm_test_open_close != "TEST_CLAMP_OPEN_CLOSE":
                result["failure_reason"] = "test_open_close_requires_confirm_TEST_CLAMP_OPEN_CLOSE"
                return result
            open_response = client.open_full()
            result["test_open_close_performed"] = True
            result["test_open_close_response"] = open_response
    except Exception as exc:  # noqa: BLE001
        result["failure_reason"] = str(exc)
        result["exception_type"] = type(exc).__name__
    finally:
        result["command_log"] = client.commands
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Arduino clamp availability without moving the clamp by default.")
    parser.add_argument("--config", required=True, help="Pipeline config containing the clamp section.")
    parser.add_argument("--test-open-close", action="store_true", help="Actuate OPEN_FULL only when explicitly confirmed.")
    parser.add_argument("--confirm-test-open-close", help="Must be TEST_CLAMP_OPEN_CLOSE when --test-open-close is used.")
    args = parser.parse_args()
    payload = check_clamp_arduino(
        config_path=args.config,
        test_open_close=bool(args.test_open_close),
        confirm_test_open_close=args.confirm_test_open_close,
    )
    _json_print(payload)
    return 0 if payload.get("safe_to_enable_clamp") or not payload.get("port_exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
