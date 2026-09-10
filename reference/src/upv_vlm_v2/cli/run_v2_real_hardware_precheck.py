from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from upv_vlm_v2.config import load_config, validate_config_schema  # noqa: E402
from upv_vlm_v2.planning.calibration import validate_calibration_files  # noqa: E402
from upv_vlm_v2.hardware.ur_rtde_client import read_only_rtde_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only v2 real hardware precheck. No robot or clamp motion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-real-hardware", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--check-rtde", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.confirm != "STATUS_ONLY" or not args.allow_real_hardware:
        print("real hardware precheck requires --allow-real-hardware --confirm STATUS_ONLY")
        return 2
    config = validate_config_schema(load_config(args.config))
    records, warnings = validate_calibration_files(config, required=True)
    payload = {
        "mode": "real_status_only",
        "calibration_files": records,
        "calibration_warnings": warnings,
        "robot_status": None,
        "clamp_status": "not_checked_no_serial_writes",
        "no_motion_commands_sent": True,
    }
    if args.check_rtde:
        payload["robot_status"] = read_only_rtde_status(
            robot_ip=str((config.get("robot") or {}).get("robot_ip")),
            allow_real_hardware=True,
        )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
