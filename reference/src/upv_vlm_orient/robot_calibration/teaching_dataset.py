"""Persistence helpers for manual teaching calibration datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def next_run_folder(session_dir: str | Path) -> Path:
    root = Path(session_dir)
    manual_dir = root / "manual_locations"
    manual_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(manual_dir.glob("run_*"))
    next_idx = len(existing) + 1
    run_dir = manual_dir / f"run_{next_idx:03d}"
    while run_dir.exists():
        next_idx += 1
        run_dir = manual_dir / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_run_record(record: dict[str, Any], run_dir: str | Path | None = None) -> Path:
    run_path = Path(run_dir) if run_dir is not None else Path(record["run_dir"])
    run_index = int(record["run_index"])
    path = run_path / f"run_{run_index:03d}_teaching_record.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    pred = record.get("predicted_gap_pose") or [None] * 6
    auto = record.get("auto_reached_gap_pose") or [None] * 6
    manual = record.get("manual_corrected_pose") or [None] * 6
    delta_mm = record.get("manual_minus_predicted_gap_pose_mm") or [None, None, None]
    return {
        "run_index": record.get("run_index"),
        "timestamp": record.get("timestamp"),
        "centroid_u": (record.get("target_centroid_px") or [None, None])[0],
        "centroid_v": (record.get("target_centroid_px") or [None, None])[1],
        "axis_angle_deg": record.get("selected_axis_angle_deg_image"),
        "predicted_dx_m": record.get("predicted_dx_base_m"),
        "predicted_dy_m": record.get("predicted_dy_base_m"),
        "predicted_x": pred[0],
        "predicted_y": pred[1],
        "predicted_z": pred[2],
        "predicted_rx": pred[3],
        "predicted_ry": pred[4],
        "predicted_rz": pred[5],
        "auto_x": auto[0],
        "auto_y": auto[1],
        "auto_z": auto[2],
        "auto_rx": auto[3],
        "auto_ry": auto[4],
        "auto_rz": auto[5],
        "manual_x": manual[0],
        "manual_y": manual[1],
        "manual_z": manual[2],
        "manual_rx": manual[3],
        "manual_ry": manual[4],
        "manual_rz": manual[5],
        "manual_minus_pred_x_mm": delta_mm[0],
        "manual_minus_pred_y_mm": delta_mm[1],
        "manual_minus_pred_z_mm": delta_mm[2],
        "manual_minus_pred_rot_angle_deg": record.get("manual_minus_predicted_rot_angle_deg"),
        "return_home_success": record.get("return_home_success"),
        "success": record.get("success"),
    }


def append_session_csv(record: dict[str, Any], session_dir: str | Path) -> Path:
    path = Path(session_dir) / "manual_locations_summary.csv"
    row = _csv_row(record)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def load_all_teaching_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_dir)
    records = []
    for path in sorted(root.glob("manual_locations/run_*/run_*_teaching_record.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def compute_summary_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in records if r.get("manual_minus_predicted_gap_pose_mm") is not None]
    if not usable:
        return {"count": len(records), "usable_count": 0}
    xyz = np.array([r["manual_minus_predicted_gap_pose_mm"] for r in usable], dtype=float)
    rot = np.array([r.get("manual_minus_predicted_rot_angle_deg", np.nan) for r in usable], dtype=float)
    return {
        "count": len(records),
        "usable_count": len(usable),
        "mean_xyz_mm": xyz.mean(axis=0).tolist(),
        "median_xyz_mm": np.median(xyz, axis=0).tolist(),
        "std_xyz_mm": xyz.std(axis=0).tolist(),
        "mean_rotation_delta_deg": float(np.nanmean(rot)),
        "median_rotation_delta_deg": float(np.nanmedian(rot)),
    }

