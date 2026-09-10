from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from upv_vlm_v1.experiment.t5_target_selection_runner import T5Case, resolve_repo_path, run_real_perception
from upv_vlm_v1.experiment.t5_v2_target_selection_fixed27_runner import capture_ros2_snapshot


REPO_ROOT = Path(__file__).resolve().parents[3]
T6_COLUMNS = [
    "trial_index", "frame_id", "input_text", "expected_selected_class", "axis_mode",
    "orientation_case", "status", "centroid_u_px", "centroid_v_px", "major_axis_angle_deg",
    "minor_axis_angle_deg", "mask_area_px", "mask_usable_manual", "axis_correct_manual",
    "manual_major_axis_angle_deg", "manual_minor_axis_angle_deg", "manual_notes",
    "raw_rgb_path", "selected_mask_overlay_path", "geometry_axis_overlay_path",
    "mask_only_path", "case_output_dir", "total_time_s", "created_at", "completed_at",
]


@dataclass
class T6Trial:
    trial_index: int
    frame_id: str
    input_text: str
    expected_selected_class: str
    axis_mode: str
    orientation_case: str
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "T6Trial":
        return cls(
            trial_index=int(row["trial_index"]),
            frame_id=str(row["frame_id"]),
            input_text=str(row["input_text"]),
            expected_selected_class=str(row["expected_selected_class"]),
            axis_mode=str(row["axis_mode"]),
            orientation_case=str(row["orientation_case"]),
            enabled=str(row.get("enabled", "true")).lower() not in {"0", "false", "no"},
            notes=str(row.get("notes") or ""),
        )

    def folder_name(self) -> str:
        return f"trial_{self.trial_index:03d}_{self.frame_id}_{_slug(self.input_text)}_{_slug(self.orientation_case)}"


def _slug(text: Any) -> str:
    return "_".join("".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).split("_"))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def read_cases(path: str | Path) -> list[T6Trial]:
    with resolve_repo_path(path).open("r", newline="", encoding="utf-8") as handle:
        return [T6Trial.from_row(row) for row in csv.DictReader(handle)]


def validate_cases(cases: list[T6Trial]) -> None:
    enabled = [case for case in cases if case.enabled]
    if len(enabled) != 15:
        raise ValueError(f"T6 requires 15 enabled cases, found {len(enabled)}")


def create_session_dir(output_root: str | Path, session_dir: str | Path | None = None) -> Path:
    root = resolve_repo_path(session_dir) if session_dir else resolve_repo_path(output_root) / f"session_{now_stamp()}"
    root.mkdir(parents=True, exist_ok=True)
    for sub in ["cases", "paper_figures", "paper_tables", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return {"mask_area_px": 0, "centroid_u_px": "NA", "centroid_v_px": "NA", "major_axis_angle_deg": "NA", "minor_axis_angle_deg": "NA"}
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return {
        "mask_area_px": int(xs.size),
        "centroid_u_px": round(float(mean[0]), 2),
        "centroid_v_px": round(float(mean[1]), 2),
        "major_axis_unit_px": [round(float(major[0]), 5), round(float(major[1]), 5)],
        "minor_axis_angle_deg": round(angle + 90.0, 2),
        "major_axis_angle_deg": round(angle, 2),
    }


def _save_geometry_artifacts(case_dir: Path, trial: T6Trial, rgb_path: Path, mask: np.ndarray) -> dict[str, Any]:
    geom = _mask_geometry(mask)
    raw = Image.open(rgb_path).convert("RGB")
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(case_dir / "mask_only.png")
    overlay = raw.convert("RGBA")
    color = Image.new("RGBA", raw.size, (0, 160, 255, 95))
    overlay.paste(color, (0, 0), Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L"))
    draw = ImageDraw.Draw(overlay)
    if geom["centroid_u_px"] != "NA":
        cx, cy = float(geom["centroid_u_px"]), float(geom["centroid_v_px"])
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 220, 0, 255), outline=(0, 0, 0, 255), width=2)
        angle = math.radians(float(geom["major_axis_angle_deg"]))
        for theta, fill in [(angle, (255, 60, 60, 255)), (angle + math.pi / 2, (60, 230, 120, 255))]:
            dx, dy = math.cos(theta) * 90, math.sin(theta) * 90
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=fill, width=4)
    draw.rectangle((0, 0, 430, 78), fill=(245, 245, 238, 235), outline=(25, 25, 25, 255))
    draw.text((10, 8), f"T6 geometry/axis Trial {trial.trial_index} {trial.frame_id}", fill=(20, 20, 20), font=_font(16, True))
    draw.text((10, 34), f"requested={trial.input_text} axis_mode={trial.axis_mode} orientation={trial.orientation_case}", fill=(20, 20, 20), font=_font(13))
    draw.text((10, 54), f"centroid=({geom['centroid_u_px']},{geom['centroid_v_px']}) major={geom['major_axis_angle_deg']} deg", fill=(20, 20, 20), font=_font(13))
    overlay.convert("RGB").save(case_dir / "geometry_axis_overlay.png")
    shutil.copyfile(case_dir / "geometry_axis_overlay.png", case_dir / "selected_mask_overlay.png")
    geom.update({
        "raw_rgb_path": str(rgb_path),
        "mask_only_path": str(case_dir / "mask_only.png"),
        "geometry_axis_overlay_path": str(case_dir / "geometry_axis_overlay.png"),
        "selected_mask_overlay_path": str(case_dir / "selected_mask_overlay.png"),
    })
    write_json(case_dir / "geometry_summary.json", geom)
    return geom


def _fake_trial_image(case_dir: Path, trial: T6Trial) -> tuple[Path, np.ndarray]:
    img = Image.new("RGB", (720, 480), (238, 239, 233))
    draw = ImageDraw.Draw(img)
    color = {"brick": (190, 92, 60), "timber": (150, 100, 55)}.get(trial.input_text, (150, 155, 150))
    draw.rectangle((210, 160, 510, 285), fill=color, outline=(40, 40, 40), width=4)
    draw.text((24, 22), "T6 DRY RUN", fill=(20, 22, 25), font=_font(28, True))
    path = case_dir / "raw_rgb.png"
    img.save(path)
    mask = np.zeros((480, 720), dtype=np.uint8)
    mask[160:285, 210:510] = 255
    return path, mask


def prompt_manual(no_manual: bool) -> dict[str, Any]:
    if no_manual:
        return {"centroid_correct_manual": "NA", "manual_major_axis_angle_deg": "NA", "manual_minor_axis_angle_deg": "NA", "axis_correct_manual": "NA", "mask_usable_manual": "NA", "manual_notes": "NA"}
    return {
        "centroid_correct_manual": input("Centroid correct? [y/n/skip]: ").strip() or "NA",
        "manual_major_axis_angle_deg": input("Manual major-axis angle deg: ").strip() or "NA",
        "manual_minor_axis_angle_deg": input("Manual minor-axis angle deg: ").strip() or "NA",
        "axis_correct_manual": input("Axis correct? [y/n/skip]: ").strip() or "NA",
        "mask_usable_manual": input("Mask usable? [y/n/skip]: ").strip() or "NA",
        "manual_notes": input("Notes: ").strip() or "NA",
    }


def append_row(session_dir: Path, row: dict[str, Any]) -> None:
    path = session_dir / "T6_master_results.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=T6_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "NA") for key in T6_COLUMNS})
    shutil.copyfile(path, session_dir / "paper_tables" / "table_experiment_2_geometry_axis.csv")


def run_trial(session_dir: Path, trial: T6Trial, config: dict[str, Any], *, dry_run: bool, live: bool, image_path: Path | None, no_manual: bool) -> dict[str, Any]:
    start = time.perf_counter()
    case_dir = session_dir / "cases" / trial.folder_name()
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "trial_metadata.json", trial.__dict__)
    if dry_run:
        rgb_path, mask = _fake_trial_image(case_dir, trial)
    elif live:
        snap = capture_ros2_snapshot(case_dir, config)
        rgb_path = Path(snap["color_path"])
        result = run_real_perception(case_dir / "backend_outputs", rgb_path, None, T5Case(f"trial_{trial.trial_index:03d}", trial.frame_id, trial.frame_id, trial.input_text, trial.input_text, trial.expected_selected_class), config)
        mask_path = result.get("selected_target_mask_path")
        mask = np.asarray(Image.open(mask_path).convert("L")) if mask_path else np.zeros((Image.open(rgb_path).height, Image.open(rgb_path).width), dtype=np.uint8)
    elif image_path:
        rgb_path = image_path
        shutil.copyfile(rgb_path, case_dir / "raw_rgb.png")
        rgb_path = case_dir / "raw_rgb.png"
        result = run_real_perception(case_dir / "backend_outputs", rgb_path, None, T5Case(f"trial_{trial.trial_index:03d}", trial.frame_id, trial.frame_id, trial.input_text, trial.input_text, trial.expected_selected_class), config)
        mask_path = result.get("selected_target_mask_path")
        mask = np.asarray(Image.open(mask_path).convert("L")) if mask_path else np.zeros((Image.open(rgb_path).height, Image.open(rgb_path).width), dtype=np.uint8)
    else:
        raise ValueError("T6 non-dry run requires --live or --image")
    geom = _save_geometry_artifacts(case_dir, trial, rgb_path, mask)
    manual = prompt_manual(no_manual)
    write_json(case_dir / "manual_label.json", manual)
    timing = {"created_at": now_iso(), "completed_at": now_iso(), "total_time_s": round(time.perf_counter() - start, 4)}
    write_json(case_dir / "timing_summary.json", timing)
    row = {
        **trial.__dict__,
        **geom,
        **manual,
        "status": "saved",
        "case_output_dir": str(case_dir),
        **timing,
    }
    append_row(session_dir, row)
    return row


def write_session_summary(session_dir: Path, cases: list[T6Trial]) -> None:
    write_json(session_dir / "T6_session_summary.json", {"session_dir": str(session_dir), "enabled_cases": len([c for c in cases if c.enabled]), "robot_used": False, "clamp_used": False, "rtde_used": False, "arduino_used": False})
