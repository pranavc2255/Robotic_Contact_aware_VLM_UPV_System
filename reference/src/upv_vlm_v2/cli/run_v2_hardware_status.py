from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 hardware status placeholder. No motion or clamp commands.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    _args = parser.parse_args()
    print(json.dumps({
        "implemented": False,
        "phase": "phase1",
        "robot_connected": False,
        "clamp_connected": False,
        "motion_command_sent": False,
        "clamp_command_sent": False,
        "note": "Hardware status will be added in a later phase with lazy RTDE/serial imports.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

