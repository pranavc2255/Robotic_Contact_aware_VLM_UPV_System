"""Analyze E45 robot pose and manual UPV repeatability CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROBOT_FIELDS = [
    "case_id", "material", "condition", "anchor_type", "anchor_id", "n_attempted", "n_motion_success",
    "motion_success_rate", "mean_position_error_mm", "std_position_error_mm", "std_actual_tcp_x_mm",
    "std_actual_tcp_y_mm", "std_actual_tcp_z_mm", "mean_orientation_error_deg", "std_orientation_error_deg",
    "safety_failure_count", "failure_count", "notes",
]

UPV_FIELDS = [
    "case_id", "material", "condition", "anchor_type", "anchor_id", "n_attempted", "n_valid",
    "valid_signal_rate", "n_failed", "failed_reading_rate", "mean_velocity_m_s", "std_velocity_m_s",
    "cv_velocity_percent", "mean_arrival_time_us", "std_arrival_time_us", "cv_arrival_time_percent", "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_placeholder_png(path: Path, title: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 1180, 680], outline="black", width=3)
        draw.text((60, 80), title, fill="black")
        draw.text((60, 140), note, fill="black")
        img.save(path)
    except Exception:
        path.write_bytes(b"")


def write_figures(session: Path) -> None:
    figures = session / "figures"
    specs = [
        ("e4_robot_position_error_by_anchor.png", "E4 robot position error by anchor", "No measured robot error values are available in this scaffold/dry session."),
        ("e4_tcp_repeatability.png", "E4 TCP repeatability", "Populate actual TCP poses before interpreting this figure."),
        ("e5_good_vs_bad_cv.png", "E5 good vs bad UPV coefficient of variation", "Populate valid manual UPV readings before interpreting this figure."),
        ("e5_valid_signal_rate.png", "E5 valid signal rate", "Populate signal_detected=yes and signal_quality=stable entries first."),
        ("e5_failed_reading_rate.png", "E5 failed reading rate", "Populate failed/no-signal/rejected entries first."),
    ]
    for filename, title, note in specs:
        _write_placeholder_png(figures / filename, title, note)


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "success", "stable"}


def _has_robot_attempt(row: dict[str, str]) -> bool:
    if str(row.get("robot_motion_success", "")).strip():
        return True
    if str(row.get("failure_stage", "")).strip() or str(row.get("failure_reason", "")).strip():
        return True
    return any(str(row.get(k, "")).strip() for k in [
        "actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m",
        "actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad",
    ])


def _has_upv_attempt(row: dict[str, str]) -> bool:
    return any(str(row.get(k, "")).strip() for k in [
        "upv_velocity_m_s", "arrival_time_us", "signal_detected", "signal_quality",
    ])


def _group(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("material", ""),
            row.get("condition", ""),
            row.get("anchor_type", ""),
            row.get("anchor_id", ""),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _stats(values: list[float]) -> tuple[Any, Any]:
    if not values:
        return "", ""
    if len(values) == 1:
        return round(values[0], 6), 0.0
    return round(mean(values), 6), round(stdev(values), 6)


def _compute_pose_error(row: dict[str, str]) -> tuple[float | None, float | None]:
    existing_pos = _float(row.get("position_error_mm"))
    existing_ori = _float(row.get("orientation_error_deg"))
    if existing_pos is not None or existing_ori is not None:
        return existing_pos, existing_ori
    planned = [_float(row.get(k)) for k in ["planned_tcp_x_m", "planned_tcp_y_m", "planned_tcp_z_m"]]
    actual = [_float(row.get(k)) for k in ["actual_tcp_x_m", "actual_tcp_y_m", "actual_tcp_z_m"]]
    pos = None
    if all(v is not None for v in planned + actual):
        pos = math.sqrt(sum((float(a) - float(p)) ** 2 for p, a in zip(planned, actual, strict=False))) * 1000.0
    planned_r = [_float(row.get(k)) for k in ["planned_tcp_rx_rad", "planned_tcp_ry_rad", "planned_tcp_rz_rad"]]
    actual_r = [_float(row.get(k)) for k in ["actual_tcp_rx_rad", "actual_tcp_ry_rad", "actual_tcp_rz_rad"]]
    ori = None
    if all(v is not None for v in planned_r + actual_r):
        ori = math.sqrt(sum((float(a) - float(p)) ** 2 for p, a in zip(planned_r, actual_r, strict=False))) * 180.0 / math.pi
    return pos, ori


def analyze_robot(rows: list[dict[str, str]], session: Path) -> None:
    out_rows: list[dict[str, Any]] = []
    all_pos: list[float] = []
    all_ori: list[float] = []
    n_success = 0
    n_attempted_total = 0
    failure_stage_counts: dict[str, int] = {}
    safety_failure_count = 0
    for key, group in _group(rows).items():
        case_id, material, condition, anchor_type, anchor_id = key
        pos_values: list[float] = []
        ori_values: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        success_count = 0
        failures = 0
        safety_failures = 0
        notes: list[str] = []
        attempted_count = 0
        for row in group:
            attempted = _has_robot_attempt(row)
            if attempted:
                attempted_count += 1
            if _truthy(row.get("robot_motion_success")):
                success_count += 1
            if row.get("failure_reason"):
                failures += 1
                failure_stage_counts[row.get("failure_stage") or "unknown"] = failure_stage_counts.get(row.get("failure_stage") or "unknown", 0) + 1
                if "safety" in (row.get("failure_stage", "") + row.get("failure_reason", "")).lower():
                    safety_failures += 1
            pos, ori = _compute_pose_error(row)
            if pos is not None:
                pos_values.append(pos)
                all_pos.append(pos)
            if ori is not None:
                ori_values.append(ori)
                all_ori.append(ori)
            for key_name, dest in [("actual_tcp_x_m", xs), ("actual_tcp_y_m", ys), ("actual_tcp_z_m", zs)]:
                v = _float(row.get(key_name))
                if v is not None:
                    dest.append(v * 1000.0)
        n = attempted_count
        n_attempted_total += attempted_count
        n_success += success_count
        safety_failure_count += safety_failures
        pos_mean, pos_std = _stats(pos_values)
        ori_mean, ori_std = _stats(ori_values)
        _, x_std = _stats(xs)
        _, y_std = _stats(ys)
        _, z_std = _stats(zs)
        if not pos_values:
            notes.append("position_error_unavailable")
        if attempted_count == 0:
            notes.append("not_attempted")
        out_rows.append({
            "case_id": case_id,
            "material": material,
            "condition": condition,
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "n_attempted": n,
            "n_motion_success": success_count,
            "motion_success_rate": round(success_count / n, 6) if n else "",
            "mean_position_error_mm": pos_mean,
            "std_position_error_mm": pos_std,
            "std_actual_tcp_x_mm": x_std,
            "std_actual_tcp_y_mm": y_std,
            "std_actual_tcp_z_mm": z_std,
            "mean_orientation_error_deg": ori_mean,
            "std_orientation_error_deg": ori_std,
            "safety_failure_count": safety_failures,
            "failure_count": failures,
            "notes": "; ".join(notes),
        })
    _write_csv(session / "e4_robot_motion_summary_by_anchor.csv", out_rows, ROBOT_FIELDS)
    _write_json(session / "e4_robot_motion_summary_overall.json", {
        "n_attempted": n_attempted_total,
        "n_motion_success": n_success,
        "motion_success_rate": round(n_success / n_attempted_total, 6) if n_attempted_total else None,
        "mean_position_error_mm": _stats(all_pos)[0],
        "mean_orientation_error_deg": _stats(all_ori)[0],
        "safety_failure_count": safety_failure_count,
        "failure_stage_counts": failure_stage_counts,
    })


def _is_valid_upv(row: dict[str, str]) -> bool:
    return str(row.get("signal_detected", "")).strip().lower() == "yes" and str(row.get("signal_quality", "")).strip().lower() == "stable"


def _is_failed_upv(row: dict[str, str]) -> bool:
    if not _has_upv_attempt(row):
        return False
    quality = str(row.get("signal_quality", "")).strip().lower()
    detected = str(row.get("signal_detected", "")).strip().lower()
    return quality in {"unstable", "no_signal", "rejected"} or detected == "no"


def analyze_upv(rows: list[dict[str, str]], session: Path) -> None:
    out_rows: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, list[float] | int]] = {
        "contact_usable": {"attempts": 0, "valid": 0, "failed": 0, "cv": []},
        "contact_unsuitable": {"attempts": 0, "valid": 0, "failed": 0, "cv": []},
    }
    for key, group in _group(rows).items():
        case_id, material, condition, anchor_type, anchor_id = key
        attempted_group = [r for r in group if _has_upv_attempt(r)]
        velocities = [_float(r.get("upv_velocity_m_s")) for r in attempted_group if _is_valid_upv(r)]
        arrivals = [_float(r.get("arrival_time_us")) for r in attempted_group if _is_valid_upv(r)]
        velocities_f = [float(v) for v in velocities if v is not None]
        arrivals_f = [float(v) for v in arrivals if v is not None]
        valid_count = sum(1 for r in attempted_group if _is_valid_upv(r))
        failed_count = sum(1 for r in attempted_group if _is_failed_upv(r))
        v_mean, v_std = _stats(velocities_f)
        a_mean, a_std = _stats(arrivals_f)
        cv_v = ""
        if velocities_f and isinstance(v_mean, (float, int)) and float(v_mean) != 0:
            cv_v = round(float(v_std) / float(v_mean) * 100.0, 6)
        cv_a = ""
        if arrivals_f and isinstance(a_mean, (float, int)) and float(a_mean) != 0:
            cv_a = round(float(a_std) / float(a_mean) * 100.0, 6)
        n = len(attempted_group)
        notes = []
        if n == 0:
            notes.append("not_entered")
        if not velocities_f:
            notes.append("no_valid_numeric_velocity")
        out_rows.append({
            "case_id": case_id,
            "material": material,
            "condition": condition,
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "n_attempted": n,
            "n_valid": valid_count,
            "valid_signal_rate": round(valid_count / n, 6) if n else "",
            "n_failed": failed_count,
            "failed_reading_rate": round(failed_count / n, 6) if n else "",
            "mean_velocity_m_s": v_mean,
            "std_velocity_m_s": v_std,
            "cv_velocity_percent": cv_v,
            "mean_arrival_time_us": a_mean,
            "std_arrival_time_us": a_std,
            "cv_arrival_time_percent": cv_a,
            "notes": "; ".join(notes),
        })
        if anchor_type in by_type:
            bucket = by_type[anchor_type]
            bucket["attempts"] = int(bucket["attempts"]) + n
            bucket["valid"] = int(bucket["valid"]) + valid_count
            bucket["failed"] = int(bucket["failed"]) + failed_count
            if isinstance(cv_v, (int, float)):
                bucket["cv"].append(float(cv_v))  # type: ignore[union-attr]
    _write_csv(session / "e5_upv_repeatability_summary_by_anchor.csv", out_rows, UPV_FIELDS)

    def rate(bucket: dict[str, Any], key: str) -> float | None:
        attempts = int(bucket.get("attempts") or 0)
        return round(int(bucket.get(key) or 0) / attempts, 6) if attempts else None

    def mean_cv(bucket: dict[str, Any]) -> float | None:
        values = bucket.get("cv") or []
        return round(mean(values), 6) if values else None

    usable = by_type["contact_usable"]
    unsuitable = by_type["contact_unsuitable"]
    _write_json(session / "e5_upv_repeatability_summary_overall.json", {
        "usable_valid_signal_rate": rate(usable, "valid"),
        "unsuitable_valid_signal_rate": rate(unsuitable, "valid"),
        "usable_mean_cv_velocity_percent": mean_cv(usable),
        "unsuitable_mean_cv_velocity_percent": mean_cv(unsuitable),
        "usable_failed_reading_rate": rate(usable, "failed"),
        "unsuitable_failed_reading_rate": rate(unsuitable, "failed"),
        "n_cases": len({r.get("case_id", "") for r in rows if r.get("case_id")}),
        "n_attempts": len(rows),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--trials-csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = Path(args.session)
    trials_csv = Path(args.trials_csv) if args.trials_csv else session / "e45_trials_raw.csv"
    rows = _read_csv(trials_csv)
    analyze_robot(rows, session)
    analyze_upv(rows, session)
    write_figures(session)
    print(f"wrote {session / 'e4_robot_motion_summary_by_anchor.csv'}")
    print(f"wrote {session / 'e4_robot_motion_summary_overall.json'}")
    print(f"wrote {session / 'e5_upv_repeatability_summary_by_anchor.csv'}")
    print(f"wrote {session / 'e5_upv_repeatability_summary_overall.json'}")
    print(f"wrote figures under {session / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
