"""Stage A capture-only dataset builder for E2 width validation.

This module captures RGB-D samples and manual global major/minor width labels.
It intentionally does not import or run perception, Qwen, RTDE, Arduino, robot
motion, clamp control, or path-length measurement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from upv_vlm_v2.acquisition.ros2_realsense import capture_ros2_realsense_snapshot
from upv_vlm_v2.config.loader import load_config, snapshot_yaml


MANIFEST_COLUMNS = [
    "case_index",
    "case_id",
    "requested_material",
    "status",
    "case_dir",
    "rgb_path",
    "depth_path",
    "depth_visualization_path",
    "camera_info_path",
    "camera_intrinsics_path",
    "pointcloud_npz_path",
    "pointcloud_ply_path",
    "manual_measurements_path",
    "manual_major_width_mm",
    "manual_minor_width_mm",
    "measurement_tool",
    "notes",
    "capture_timestamp",
]


@dataclass
class CaptureDatasetSession:
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
    (session_dir / "dataset_manifest.json").write_text(json.dumps({"rows": rows}, indent=2, default=str), encoding="utf-8")


def _read_existing_rows(session_dir: Path) -> list[dict[str, Any]]:
    manifest = session_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _camera_intrinsics(info: dict[str, Any]) -> dict[str, float]:
    if all(key in info for key in ("fx", "fy", "cx", "cy")):
        return {key: float(info[key]) for key in ("fx", "fy", "cx", "cy")}
    k = info.get("K") or info.get("k") or []
    if len(k) >= 6:
        return {"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])}
    raise ValueError("camera_info missing fx/fy/cx/cy")


def _depth_scale(config: dict[str, Any], metadata: dict[str, Any] | None = None) -> float:
    capture_cfg = config.get("capture", {})
    value = capture_cfg.get("depth_scale_m_per_unit", "inherit_from_camera")
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    if metadata and isinstance(metadata.get("depth_scale_m_per_unit"), (int, float)):
        return float(metadata["depth_scale_m_per_unit"])
    camera_value = (config.get("camera") or {}).get("depth_scale_m_per_unit")
    if isinstance(camera_value, (int, float)) and float(camera_value) > 0:
        return float(camera_value)
    return 0.001


def _save_pointcloud(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    camera_info_path: str | Path,
    output_npz: str | Path,
    output_ply: str | Path,
    depth_scale_m_per_unit: float,
) -> tuple[str, str]:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    info = json.loads(Path(camera_info_path).read_text(encoding="utf-8"))
    intr = _camera_intrinsics(info)
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[yy, xx].astype(float) * float(depth_scale_m_per_unit)
    valid = np.isfinite(z) & (z > 0)
    xx = xx[valid]
    yy = yy[valid]
    z = z[valid]
    x = (xx.astype(float) - intr["cx"]) * z / intr["fx"]
    y = (yy.astype(float) - intr["cy"]) * z / intr["fy"]
    xyz = np.column_stack((x, y, z))
    uv = np.column_stack((xx.astype(float), yy.astype(float)))
    colors = rgb[yy, xx] if rgb.size else np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    npz_path = Path(output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        xyz_m=xyz,
        uv_px=uv,
        depth_m=z,
        rgb=colors,
        intrinsics=json.dumps(intr),
        depth_scale_m_per_unit=float(depth_scale_m_per_unit),
    )
    ply_path = Path(output_ply)
    with ply_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, colors):
            handle.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    return str(npz_path), str(ply_path)


def _copy_capture_payload(payload: dict[str, Any], case_dir: Path, config: dict[str, Any]) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "rgb_path": "raw_rgb.png",
        "depth_path": "depth_raw.npy",
        "depth_visualization_path": "depth_visualization.png",
        "camera_info_path": "camera_info.json",
        "camera_intrinsics_path": "camera_intrinsics.json",
        "capture_metadata_path": "capture_metadata.json",
    }
    copied: dict[str, str] = {}
    for key, name in mapping.items():
        src = payload.get(key)
        if src and Path(src).exists():
            dst = case_dir / name
            if Path(src).resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied[key] = str(dst)
    if "camera_intrinsics_path" not in copied and copied.get("camera_info_path"):
        info = json.loads(Path(copied["camera_info_path"]).read_text(encoding="utf-8"))
        intr_path = case_dir / "camera_intrinsics.json"
        intr_path.write_text(json.dumps(_camera_intrinsics(info), indent=2), encoding="utf-8")
        copied["camera_intrinsics_path"] = str(intr_path)
    metadata = payload.get("diagnostics") or {}
    scale = _depth_scale(config, metadata)
    npz, ply = _save_pointcloud(
        rgb_path=copied["rgb_path"],
        depth_path=copied["depth_path"],
        camera_info_path=copied["camera_info_path"],
        output_npz=case_dir / "pointcloud_raw.npz",
        output_ply=case_dir / "pointcloud_raw.ply",
        depth_scale_m_per_unit=scale,
    )
    copied["pointcloud_npz_path"] = npz
    copied["pointcloud_ply_path"] = ply
    return copied


def _session(config_path: str | Path, resume_session: str | Path | None = None) -> CaptureDatasetSession:
    config = load_config(config_path)
    if resume_session:
        session_dir = Path(resume_session)
        rows = _read_existing_rows(session_dir)
    else:
        session_dir = Path(config.get("output_root", "outputs/v2_datasets/width_fixed15")) / f"session_{_now_id()}"
        rows = []
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_yaml(config, session_dir / "run_config_snapshot.yaml")
    _write_manifest(session_dir, rows)
    return CaptureDatasetSession(config=config, config_path=Path(config_path), session_dir=session_dir, rows=rows)


def capture_case(
    *,
    session: CaptureDatasetSession,
    case: dict[str, Any],
    manual: dict[str, Any],
    capture_payload: dict[str, Any],
) -> dict[str, Any]:
    case_dir = session.session_dir / "cases" / f"case_{int(case['case_index']):03d}_{case['case_id']}"
    copied = _copy_capture_payload(capture_payload, case_dir, session.config)
    manual_payload = {
        "case_index": int(case["case_index"]),
        "case_id": case["case_id"],
        "requested_material": case["requested_material"],
        "manual_major_width_mm": _safe_float(manual.get("manual_major_width_mm")),
        "manual_minor_width_mm": _safe_float(manual.get("manual_minor_width_mm")),
        "measurement_tool": manual.get("measurement_tool"),
        "notes": manual.get("notes", ""),
        "timestamp": _timestamp(),
        "operator": manual.get("operator"),
    }
    manual_path = case_dir / "manual_measurements.json"
    manual_path.write_text(json.dumps(manual_payload, indent=2), encoding="utf-8")
    metadata_path = case_dir / "capture_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(capture_payload.get("diagnostics", {}), indent=2), encoding="utf-8")
    row = {
        "case_index": case["case_index"],
        "case_id": case["case_id"],
        "requested_material": case["requested_material"],
        "status": "saved",
        "case_dir": str(case_dir),
        "rgb_path": copied.get("rgb_path"),
        "depth_path": copied.get("depth_path"),
        "depth_visualization_path": copied.get("depth_visualization_path"),
        "camera_info_path": copied.get("camera_info_path"),
        "camera_intrinsics_path": copied.get("camera_intrinsics_path"),
        "pointcloud_npz_path": copied.get("pointcloud_npz_path"),
        "pointcloud_ply_path": copied.get("pointcloud_ply_path"),
        "manual_measurements_path": str(manual_path),
        "manual_major_width_mm": manual_payload["manual_major_width_mm"],
        "manual_minor_width_mm": manual_payload["manual_minor_width_mm"],
        "measurement_tool": manual_payload["measurement_tool"],
        "notes": manual_payload["notes"],
        "capture_timestamp": manual_payload["timestamp"],
    }
    session.rows = [existing for existing in session.rows if str(existing.get("case_id")) != str(case["case_id"])]
    session.rows.append(row)
    session.rows.sort(key=lambda item: int(item.get("case_index") or 0))
    _write_manifest(session.session_dir, session.rows)
    return row


def _prompt_manual(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "manual_major_width_mm": input("Manual major-axis width in mm: ").strip(),
        "manual_minor_width_mm": input("Manual minor-axis width in mm: ").strip(),
        "measurement_tool": input("Measurement tool [ruler/vernier_caliper/digital_caliper/tape/other]: ").strip() or "vernier_caliper",
        "notes": input("Manual measurement notes: ").strip(),
    }


def run_interactive(
    *,
    config_path: str | Path,
    live_ros2: bool,
    resume_session: str | Path | None = None,
    case_index: int | None = None,
    capture_func: Callable[..., dict[str, Any]] = capture_ros2_realsense_snapshot,
) -> Path:
    if not live_ros2:
        raise RuntimeError("dataset capture requires --live-ros2 unless a test capture_func is supplied")
    session = _session(config_path, resume_session)
    cases = _load_cases(session.config["cases_csv"])
    if case_index is not None:
        cases = [case for case in cases if int(case["case_index"]) == int(case_index)]
    for case in cases:
        while True:
            print("=" * 60)
            print(f"E2 Dataset Capture Case {case['case_index']}/15: {case['case_id']}")
            print(f"Requested material: {case['requested_material']}")
            action = input("Place object under camera. Press ENTER to capture, type retry, skip, or quit: ").strip().lower()
            if action == "quit":
                _write_manifest(session.session_dir, session.rows)
                return session.session_dir
            if action == "skip":
                row = {"case_index": case["case_index"], "case_id": case["case_id"], "requested_material": case["requested_material"], "status": "skipped"}
                session.rows.append(row)
                _write_manifest(session.session_dir, session.rows)
                break
            tmp_capture = session.session_dir / "tmp_capture" / f"case_{int(case['case_index']):03d}"
            payload = capture_func(config=session.config, output_dir=tmp_capture)
            if not payload.get("success"):
                print(f"Capture failed: {payload.get('failure_reason')}")
                continue
            manual = _prompt_manual(case)
            save_action = input("Save this case? [save/retry/skip/quit]: ").strip().lower() or "save"
            if save_action == "quit":
                _write_manifest(session.session_dir, session.rows)
                return session.session_dir
            if save_action == "retry":
                continue
            if save_action == "skip":
                break
            row = capture_case(session=session, case=case, manual=manual, capture_payload=payload)
            print(f"Saved case: {row['case_dir']}")
            break
    return session.session_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the E2 fixed15 RGB-D width dataset without model inference.")
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
