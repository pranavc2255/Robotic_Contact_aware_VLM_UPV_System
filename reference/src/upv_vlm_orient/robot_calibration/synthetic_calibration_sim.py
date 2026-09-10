"""Synthetic data generation for R34 teaching dataset validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .teaching_dataset import append_session_csv, next_run_folder, save_run_record
from .transform_utils import pose_delta


def synthetic_record(index: int, bias_m: tuple[float, float, float] = (0.012, -0.006, 0.002)) -> dict[str, Any]:
    predicted = [
        -0.05 + 0.002 * index,
        -0.31 + 0.001 * index,
        0.282,
        0.0,
        -3.14159,
        0.01 * index,
    ]
    manual = [
        predicted[0] + bias_m[0],
        predicted[1] + bias_m[1],
        predicted[2] + bias_m[2],
        predicted[3],
        predicted[4],
        predicted[5] + 0.01,
    ]
    delta = pose_delta(predicted, manual)
    return {
        "run_index": index,
        "timestamp": f"synthetic_{index:03d}",
        "target_centroid_px": [600.0 + index, 410.0 - index],
        "selected_axis_angle_deg_image": -22.0 + index,
        "predicted_dx_base_m": 0.01,
        "predicted_dy_base_m": -0.004,
        "predicted_gap_pose": predicted,
        "auto_reached_gap_pose": predicted,
        "manual_corrected_pose": manual,
        "manual_minus_predicted_gap_pose_m": delta["delta_xyz_m"],
        "manual_minus_predicted_gap_pose_mm": delta["delta_xyz_mm"],
        "manual_minus_auto_reached_gap_pose_m": delta["delta_xyz_m"],
        "manual_minus_auto_reached_gap_pose_mm": delta["delta_xyz_mm"],
        "manual_minus_predicted_rot_angle_deg": delta["rotation_delta_deg"],
        "return_home_success": True,
        "success": True,
    }


def generate_synthetic_dataset(session_dir: str | Path, num_samples: int) -> list[dict[str, Any]]:
    records = []
    for idx in range(1, num_samples + 1):
        run_dir = next_run_folder(session_dir)
        record = synthetic_record(idx)
        record["run_dir"] = str(run_dir)
        save_run_record(record, run_dir)
        append_session_csv(record, session_dir)
        records.append(record)
    return records

