from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[4] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from upv_vlm_v2.pipeline.modes import MODE_ORDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Print v2 stage order for a mode.")
    parser.add_argument("--mode", choices=sorted(MODE_ORDER), default="dry_run")
    args = parser.parse_args()
    print(f"mode: {args.mode}")
    print("stages:")
    for stage in MODE_ORDER[args.mode]:
        print(f"- {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

