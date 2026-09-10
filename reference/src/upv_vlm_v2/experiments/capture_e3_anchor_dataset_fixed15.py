"""Stage A capture-only dataset builder for E3 anchor-selection validation.

This module captures RGB-D scenes and scene metadata only. It intentionally
does not run target selection, CLIP, Qwen, anchor selection, path-length
measurement, RTDE, Arduino, robot motion, or clamp control.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from upv_vlm_v2.acquisition.ros2_realsense import capture_ros2_realsense_snapshot
from upv_vlm_v2.config.loader import load_config, snapshot_yaml
from upv_vlm_v2.experiments.capture_width_dataset_fixed15 import (
    _copy_capture_payload,
    _read_existing_rows,
)


MANIFEST_COLUMNS = [
    "case_index",
    "case_id",
    "material",
    "condition_type",
    "condition_notes",
    "status",
    "case_dir",
    "rgb_path",
    "depth_path",
    "depth_visualization_path",
    "camera_info_path",
    "camera_intrinsics_path",
    "pointcloud_npz_path",
    "pointcloud_ply_path",
    "scene_metadata_path",
    "capture_timestamp",
    "requested_rgb_width",
    "requested_rgb_height",
    "actual_rgb_width",
    "actual_rgb_height",
    "actual_depth_width",
    "actual_depth_height",
    "capture_resolution_warning",
]


@dataclass
class E3CaptureSession:
    config: dict[str, Any]
    config_path: Path
    session_dir: Path
    rows: list[dict[str, Any]]


def _now_id() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _load_cases(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = session_dir / "dataset_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in MANIFEST_COLUMNS})
    (session_dir / "dataset_manifest.json").write_text(
        json.dumps({"rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )


def _requested_capture_settings(config: dict[str, Any]) -> dict[str, Any]:
    camera = config.get("camera") or {}
    capture = config.get("capture") or {}
    return {
        "color_width": camera.get("color_width") or capture.get("color_width"),
        "color_height": camera.get("color_height") or capture.get("color_height"),
        "color_fps": camera.get("color_fps") or capture.get("color_fps"),
        "depth_width": camera.get("depth_width") or capture.get("depth_width"),
        "depth_height": camera.get("depth_height") or capture.get("depth_height"),
        "depth_fps": camera.get("depth_fps") or capture.get("depth_fps"),
        "align_depth_to_color": camera.get("align_depth_to_color", capture.get("align_depth_to_color")),
        "allow_lower_resolution": bool(camera.get("allow_lower_resolution", capture.get("allow_lower_resolution", False))),
        "notes": (
            "ROS topic capture records existing stream resolution; launch RealSense/ROS "
            "at the requested resolution before running capture if direct reconfiguration is unavailable."
        ),
    }


def _actual_capture_resolution(copied: dict[str, str]) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    rgb_path = copied.get("rgb_path")
    if rgb_path and Path(rgb_path).exists():
        with Image.open(rgb_path) as im:
            actual["actual_rgb_width"] = int(im.width)
            actual["actual_rgb_height"] = int(im.height)
    depth_path = copied.get("depth_path")
    if depth_path and Path(depth_path).exists():
        depth = np.load(depth_path, mmap_mode="r")
        if depth.ndim >= 2:
            actual["actual_depth_height"] = int(depth.shape[0])
            actual["actual_depth_width"] = int(depth.shape[1])
    camera_info_path = copied.get("camera_info_path")
    if camera_info_path and Path(camera_info_path).exists():
        info = json.loads(Path(camera_info_path).read_text(encoding="utf-8"))
        actual["camera_info_width"] = info.get("width")
        actual["camera_info_height"] = info.get("height")
    return actual


def _resolution_warnings(requested: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    allow_lower = bool(requested.get("allow_lower_resolution"))
    req_w = requested.get("color_width")
    req_h = requested.get("color_height")
    act_w = actual.get("actual_rgb_width")
    act_h = actual.get("actual_rgb_height")
    if req_w and req_h and act_w and act_h and (int(act_w) < int(req_w) or int(act_h) < int(req_h)):
        msg = f"captured RGB resolution {act_w}x{act_h} is below requested {req_w}x{req_h}"
        warnings.append(msg if allow_lower else msg + "; relaunch camera/ROS stream at requested resolution")
    req_dw = requested.get("depth_width")
    req_dh = requested.get("depth_height")
    act_dw = actual.get("actual_depth_width")
    act_dh = actual.get("actual_depth_height")
    if req_dw and req_dh and act_dw and act_dh and (int(act_dw) < int(req_dw) or int(act_dh) < int(req_dh)):
        warnings.append(f"captured depth resolution {act_dw}x{act_dh} is below requested {req_dw}x{req_dh}")
    return warnings


def _update_capture_metadata(case_dir: Path, requested: dict[str, Any], actual: dict[str, Any], warnings: list[str]) -> None:
    metadata_path = case_dir / "capture_metadata.json"
    payload = {}
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(
        {
            "requested_capture_resolution": requested,
            "actual_capture_resolution": actual,
            "capture_resolution_warnings": warnings,
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _session(config_path: str | Path, resume_session: str | Path | None = None) -> E3CaptureSession:
    config = load_config(config_path)
    if resume_session:
        session_dir = Path(resume_session)
        rows = _read_existing_rows(session_dir)
    else:
        session_dir = Path(config.get("output_root", "outputs/v2_datasets/e3_anchor_fixed15")) / f"session_{_now_id()}"
        rows = []
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_yaml(config, session_dir / "run_config_snapshot.yaml")
    (session_dir / "requested_capture_resolution.json").write_text(
        json.dumps(_requested_capture_settings(config), indent=2, default=str),
        encoding="utf-8",
    )
    _write_manifest(session_dir, rows)
    return E3CaptureSession(config=config, config_path=Path(config_path), session_dir=session_dir, rows=rows)


def capture_case(
    *,
    session: E3CaptureSession,
    case: dict[str, Any],
    capture_payload: dict[str, Any],
) -> dict[str, Any]:
    case_dir = session.session_dir / "cases" / f"case_{int(case['case_index']):03d}_{case['case_id']}"
    copied = _copy_capture_payload(capture_payload, case_dir, session.config)
    requested_resolution = _requested_capture_settings(session.config)
    actual_resolution = _actual_capture_resolution(copied)
    resolution_warnings = _resolution_warnings(requested_resolution, actual_resolution)
    _update_capture_metadata(case_dir, requested_resolution, actual_resolution, resolution_warnings)
    metadata = {
        "case_index": int(case["case_index"]),
        "case_id": case["case_id"],
        "material": case["material"],
        "condition_type": case["condition_type"],
        "condition_notes": case.get("condition_notes", ""),
        "timestamp": _timestamp(),
        "operator": None,
        "requested_capture_resolution": requested_resolution,
        "actual_capture_resolution": actual_resolution,
        "capture_resolution_warnings": resolution_warnings,
    }
    metadata_path = case_dir / "scene_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    row = {
        "case_index": case["case_index"],
        "case_id": case["case_id"],
        "material": case["material"],
        "condition_type": case["condition_type"],
        "condition_notes": case.get("condition_notes", ""),
        "status": "saved",
        "case_dir": str(case_dir),
        "rgb_path": copied.get("rgb_path"),
        "depth_path": copied.get("depth_path"),
        "depth_visualization_path": copied.get("depth_visualization_path"),
        "camera_info_path": copied.get("camera_info_path"),
        "camera_intrinsics_path": copied.get("camera_intrinsics_path"),
        "pointcloud_npz_path": copied.get("pointcloud_npz_path"),
        "pointcloud_ply_path": copied.get("pointcloud_ply_path"),
        "scene_metadata_path": str(metadata_path),
        "capture_timestamp": metadata["timestamp"],
        "requested_rgb_width": requested_resolution.get("color_width"),
        "requested_rgb_height": requested_resolution.get("color_height"),
        "actual_rgb_width": actual_resolution.get("actual_rgb_width"),
        "actual_rgb_height": actual_resolution.get("actual_rgb_height"),
        "actual_depth_width": actual_resolution.get("actual_depth_width"),
        "actual_depth_height": actual_resolution.get("actual_depth_height"),
        "capture_resolution_warning": "; ".join(resolution_warnings),
    }
    session.rows = [existing for existing in session.rows if str(existing.get("case_id")) != str(case["case_id"])]
    session.rows.append(row)
    session.rows.sort(key=lambda item: int(item.get("case_index") or 0))
    _write_manifest(session.session_dir, session.rows)
    return row


def _skip_case(session: E3CaptureSession, case: dict[str, Any]) -> None:
    row = {
        "case_index": case["case_index"],
        "case_id": case["case_id"],
        "material": case["material"],
        "condition_type": case["condition_type"],
        "condition_notes": case.get("condition_notes", ""),
        "status": "skipped",
    }
    session.rows = [existing for existing in session.rows if str(existing.get("case_id")) != str(case["case_id"])]
    session.rows.append(row)
    session.rows.sort(key=lambda item: int(item.get("case_index") or 0))
    _write_manifest(session.session_dir, session.rows)


def run_interactive(
    *,
    config_path: str | Path,
    live_ros2: bool,
    resume_session: str | Path | None = None,
    case_index: int | None = None,
    capture_func: Callable[..., dict[str, Any]] = capture_ros2_realsense_snapshot,
) -> Path:
    if not live_ros2:
        raise RuntimeError("E3 dataset capture requires --live-ros2 unless a test capture_func is supplied")
    session = _session(config_path, resume_session)
    cases = _load_cases(session.config["cases_csv"])
    if case_index is not None:
        cases = [case for case in cases if int(case["case_index"]) == int(case_index)]
    for case in cases:
        while True:
            print("=" * 60)
            print(f"E3 Anchor Dataset Capture Case {case['case_index']}/15")
            print(f"Case ID: {case['case_id']}")
            print(f"Material: {case['material']}")
            print(f"Condition: {case['condition_type']}")
            print(f"Condition notes: {case.get('condition_notes', '')}")
            action = input("Place object under camera. Press ENTER to capture, type retry, skip, or quit: ").strip().lower()
            if action == "quit":
                _write_manifest(session.session_dir, session.rows)
                return session.session_dir
            if action == "skip":
                _skip_case(session, case)
                break
            tmp_capture = session.session_dir / "tmp_capture" / f"case_{int(case['case_index']):03d}"
            payload = capture_func(config=session.config, output_dir=tmp_capture)
            if not payload.get("success"):
                print(f"Capture failed: {payload.get('failure_reason')}")
                continue
            save_action = input("Save this case? [save/retry/skip/quit]: ").strip().lower() or "save"
            if save_action == "quit":
                _write_manifest(session.session_dir, session.rows)
                return session.session_dir
            if save_action == "retry":
                continue
            if save_action == "skip":
                _skip_case(session, case)
                break
            row = capture_case(session=session, case=case, capture_payload=payload)
            print(f"Saved case: {row['case_dir']}")
            break
    return session.session_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the E3 fixed15 RGB-D anchor-selection dataset without model inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--live-ros2", action="store_true")
    parser.add_argument("--resume-session")
    parser.add_argument("--case-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = run_interactive(
        config_path=args.config,
        live_ros2=args.live_ros2,
        resume_session=args.resume_session,
        case_index=args.case_index,
    )
    print(f"dataset_session: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
