#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upv_vlm_v2.anchor_selection.axis_parameterization import build_axis_search_domain
from upv_vlm_v2.anchor_selection.cross_section_sampler import compute_anchor_cross_section_hits
from upv_vlm_v2.anchor_selection.edge_quality import score_edge_quality

DEFAULT_SESSION = REPO_ROOT / "outputs/path_length_ros2_stream_capture_test/session_20260702_153357"
DEFAULT_OUTPUT_ROOT = DEFAULT_SESSION / "manual_anchor_selection"

CANDIDATE_COLUMNS = [
    "case_id",
    "candidate_id",
    "candidate_source",
    "path_family",
    "axis_position_index",
    "axis_fraction",
    "offset_px",
    "p0_x",
    "p0_y",
    "direction_x",
    "direction_y",
    "cross_axis_x",
    "cross_axis_y",
    "contact1_x",
    "contact1_y",
    "contact2_x",
    "contact2_y",
    "endpoint1_x",
    "endpoint1_y",
    "endpoint2_x",
    "endpoint2_y",
    "path_length_px",
    "path_length_mm_if_available",
    "mask_intersection_count",
    "valid_candidate",
    "failure_reason",
    "score",
    "edge_quality",
]

MANIFEST_COLUMNS = [
    "case_id",
    "case_index",
    "material",
    "object_id",
    "manual_major_mm",
    "manual_minor_mm",
    "major_selected_anchor_id",
    "major_selection_source",
    "major_p0_x",
    "major_p0_y",
    "major_direction_x",
    "major_direction_y",
    "major_endpoint1_x",
    "major_endpoint1_y",
    "major_endpoint2_x",
    "major_endpoint2_y",
    "major_path_length_px",
    "minor_selected_anchor_id",
    "minor_selection_source",
    "minor_p0_x",
    "minor_p0_y",
    "minor_direction_x",
    "minor_direction_y",
    "minor_endpoint1_x",
    "minor_endpoint1_y",
    "minor_endpoint2_x",
    "minor_endpoint2_y",
    "minor_path_length_px",
    "mask_path",
    "rgb_path",
    "selection_timestamp",
    "notes",
]


@dataclass
class CaseInput:
    case_id: str
    case_index: int
    material: str
    object_id: str
    case_dir: Path
    rgb_path: Path
    depth_path: Path
    manual_path: Path
    capture_metadata_path: Path
    camera_info_path: Path
    mask_path: Path
    mask_npy_path: Path


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip() or None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def open_image_file(path: Path) -> bool:
    if not path.exists():
        print(f"Could not auto-open image because it does not exist. Manually open: {path}")
        return False
    for command in (["xdg-open", str(path)], ["gio", "open", str(path)]):
        if shutil.which(command[0]) is None:
            continue
        try:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            if proc.poll() is None or proc.returncode == 0:
                return True
        except Exception:
            continue
    print(f"Could not auto-open image. Manually open: {path}")
    return False


def _case_index(case_id: str, fallback: int) -> int:
    try:
        return int(case_id.split("_")[1])
    except Exception:
        return fallback


def _is_backup_case_name(name: str) -> bool:
    low = str(name).lower()
    return "backup" in low or "_backup_" in low or "_before_" in low


def _material_from_case(case_id: str, material: str) -> str:
    text = (material or "").strip().lower()
    if text in {"t", "timber", "wood", "lumber"}:
        return "timber"
    if text in {"b", "brick"}:
        return "brick"
    if text in {"c", "concrete", "concrete block", "block"}:
        return "concrete block"
    idx = _case_index(case_id, 0)
    if 1 <= idx <= 5:
        return "timber"
    if 6 <= idx <= 10:
        return "brick"
    if 11 <= idx <= 15:
        return "concrete block"
    return material or ""


def _load_cases(capture_session: Path, mask_root: Path, args: argparse.Namespace) -> list[CaseInput]:
    manifest = capture_session / "session_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing session manifest: {manifest}")
    if args.case_id:
        case_dir = capture_session / "cases" / args.case_id
        backup_matches = sorted(
            path for path in (capture_session / "cases").glob(f"{args.case_id}*")
            if path.is_dir() and path.name != args.case_id and _is_backup_case_name(path.name)
        )
        if args.debug_case_flow:
            print(f"[case-flow] Requested case_id: {args.case_id}")
            print(f"[case-flow] Resolved exact case directory: {case_dir.resolve()}")
            print("[case-flow] Backup folders ignored:")
            for path in backup_matches:
                print(f"[case-flow]   {path.resolve()}")
        if not case_dir.exists():
            raise FileNotFoundError(f"Case folder not found: {case_dir}")
        row_match: dict[str, Any] = {}
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("case_id")) == args.case_id:
                    row_match = row
                    break
        saved_status = str(row_match.get("saved_status") or "saved")
        if not saved_status.startswith("saved"):
            raise RuntimeError(f"Case {args.case_id} has non-saved status in manifest: {saved_status!r}")
        manual_path = case_dir / "manual_measurements.json"
        material = str(row_match.get("material") or "")
        object_id = str(row_match.get("object_id") or args.case_id)
        if manual_path.exists():
            try:
                manual = _read_json(manual_path)
                material = str(manual.get("material") or material)
                object_id = str(manual.get("object_id") or object_id)
            except Exception:
                pass
        return [
            CaseInput(
                case_id=args.case_id,
                case_index=int(row_match.get("case_index") or _case_index(args.case_id, 0)),
                material=_material_from_case(args.case_id, material),
                object_id=object_id,
                case_dir=case_dir.resolve(),
                rgb_path=(case_dir / "raw_rgb.png").resolve(),
                depth_path=(case_dir / "raw_depth_aligned_z16.png").resolve(),
                manual_path=manual_path.resolve(),
                capture_metadata_path=(case_dir / "capture_metadata.json").resolve(),
                camera_info_path=(case_dir / "camera_info_aligned_depth.json").resolve(),
                mask_path=mask_root / "cases" / args.case_id / "selected_mask.png",
                mask_npy_path=mask_root / "cases" / args.case_id / "selected_mask.npy",
            )
        ]
    cases: list[CaseInput] = []
    matched_case_id = False
    with manifest.open(newline="", encoding="utf-8") as handle:
        for fallback_idx, row in enumerate(csv.DictReader(handle), start=1):
            case_id = row["case_id"]
            if _is_backup_case_name(case_id):
                if args.debug_case_flow:
                    print(f"[case-flow] skip backup-like manifest row {case_id}")
                continue
            if args.case_id and case_id != args.case_id:
                if args.debug_case_flow:
                    print(f"[case-flow] skip manifest row {case_id}: does not match --case-id {args.case_id}")
                continue
            matched_case_id = bool(args.case_id)
            saved_status = str(row.get("saved_status") or "saved")
            if not saved_status.startswith("saved"):
                if args.debug_case_flow:
                    print(f"[case-flow] skip {case_id}: saved_status={saved_status!r}")
                continue
            case_index = int(row.get("case_index") or _case_index(case_id, fallback_idx))
            if case_index < int(args.start_index):
                if args.debug_case_flow:
                    print(f"[case-flow] skip {case_id}: case_index {case_index} < start_index {args.start_index}")
                continue
            case_dir = _resolve(Path(row.get("rgb_path") or capture_session / "cases" / case_id / "raw_rgb.png").parent)
            cases.append(
                CaseInput(
                    case_id=case_id,
                    case_index=case_index,
                    material=_material_from_case(case_id, row.get("material", "")),
                    object_id=str(row.get("object_id") or case_id),
                    case_dir=case_dir,
                    rgb_path=_resolve(row.get("rgb_path") or case_dir / "raw_rgb.png"),
                    depth_path=_resolve(row.get("aligned_depth_z16_path") or case_dir / "raw_depth_aligned_z16.png"),
                    manual_path=_resolve(row.get("manual_measurements_path") or case_dir / "manual_measurements.json"),
                    capture_metadata_path=_resolve(row.get("capture_metadata_path") or case_dir / "capture_metadata.json"),
                    camera_info_path=_resolve(row.get("aligned_depth_camera_info_path") or case_dir / "camera_info_aligned_depth.json"),
                    mask_path=mask_root / "cases" / case_id / "selected_mask.png",
                    mask_npy_path=mask_root / "cases" / case_id / "selected_mask.npy",
                )
            )
    if args.case_id and not cases:
        case_dir = capture_session / "cases" / args.case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"--case-id {args.case_id!r} was requested but case folder is missing: {case_dir}")
        if not matched_case_id and args.debug_case_flow:
            print(f"[case-flow] --case-id {args.case_id} not found in manifest; falling back to case folder")
        cases = [
            CaseInput(
                case_id=args.case_id,
                case_index=_case_index(args.case_id, 0),
                material=_material_from_case(args.case_id, ""),
                object_id=args.case_id,
                case_dir=case_dir.resolve(),
                rgb_path=(case_dir / "raw_rgb.png").resolve(),
                depth_path=(case_dir / "raw_depth_aligned_z16.png").resolve(),
                manual_path=(case_dir / "manual_measurements.json").resolve(),
                capture_metadata_path=(case_dir / "capture_metadata.json").resolve(),
                camera_info_path=(case_dir / "camera_info_aligned_depth.json").resolve(),
                mask_path=mask_root / "cases" / args.case_id / "selected_mask.png",
                mask_npy_path=mask_root / "cases" / args.case_id / "selected_mask.npy",
            )
        ]
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    return cases


def _manual_major_minor(manual: dict[str, Any], case_id: str) -> tuple[float, float]:
    major = manual.get("major_axis_mm", manual.get("manual_major_mm"))
    minor = manual.get("minor_axis_mm", manual.get("manual_minor_mm"))
    if major is None or minor is None:
        raise ValueError(
            f"{case_id} manual_measurements.json must contain major_axis_mm/minor_axis_mm "
            "or manual_major_mm/manual_minor_mm"
        )
    return float(major), float(minor)


def _validate_case_inputs(case: CaseInput, args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "case_dir": case.case_dir,
        "rgb_path": case.rgb_path,
        "depth_path": case.depth_path,
        "manual_path": case.manual_path,
        "camera_info_path": case.camera_info_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if not case.mask_path.exists() and not case.mask_npy_path.exists():
        missing.append(f"selected_mask: {case.mask_npy_path} or {case.mask_path}")
    if missing:
        raise FileNotFoundError(f"Missing required inputs for {case.case_id}:\n" + "\n".join(f"  {item}" for item in missing))
    manual = _read_json(case.manual_path)
    major, minor = _manual_major_minor(manual, case.case_id)
    if args.debug_case_flow or args.case_id:
        print(f"[case-flow] capture_session: {_resolve(args.capture_session)}")
        print(f"[case-flow] mask_root: {_resolve(args.mask_root)}")
        print(f"[case-flow] output_root: {_resolve(args.output_root)}")
        print(f"[case-flow] candidate_mode: {args.candidate_mode}")
        print(f"[case-flow] case_id: {case.case_id}")
        print(f"[case-flow] open_overlays: {args.open_overlays}")
        print(f"[case-flow] force_regenerate_candidates: {args.force_regenerate_candidates}")
        print(f"[case-flow] resolved case directory exists? {case.case_dir.exists()} -> {case.case_dir}")
        print(f"[case-flow] resolved RGB path exists? {case.rgb_path.exists()} -> {case.rgb_path}")
        print(f"[case-flow] resolved selected mask path exists? {case.mask_path.exists()} -> {case.mask_path}")
        print(f"[case-flow] resolved selected mask npy exists? {case.mask_npy_path.exists()} -> {case.mask_npy_path}")
        print(f"[case-flow] resolved manual_measurements.json exists? {case.manual_path.exists()} -> {case.manual_path}")
        print(f"[case-flow] parsed manual major: {major}")
        print(f"[case-flow] parsed manual minor: {minor}")
    manual["major_axis_mm"] = major
    manual["minor_axis_mm"] = minor
    return manual


def _wait_for_mask(case: CaseInput, args: argparse.Namespace) -> bool:
    if case.mask_path.exists() or case.mask_npy_path.exists():
        return True
    if not args.wait_for_masks:
        return False
    deadline = time.time() + float(args.mask_wait_timeout_sec)
    while time.time() < deadline:
        if case.mask_path.exists() or case.mask_npy_path.exists():
            return True
        time.sleep(2.0)
    return False


def _load_mask(case: CaseInput) -> np.ndarray:
    if case.mask_npy_path.exists():
        arr = np.load(case.mask_npy_path)
        return (arr > 0).astype(np.uint8)
    mask = cv2.imread(str(case.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask for {case.case_id}: {case.mask_path}")
    return (mask > 0).astype(np.uint8)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("zero vector")
    return vec / norm


def _mask_rect(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty selected mask")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.array(rect[0], dtype=float)
    edges = [box[(idx + 1) % 4] - box[idx] for idx in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_dir = _normalize(edges[int(np.argmax(lengths))])
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=float)
    return center, long_dir, short_dir, box


def _axis_lengths_from_box(box: np.ndarray) -> tuple[float, float]:
    edges = [box[(idx + 1) % 4] - box[idx] for idx in range(4)]
    lengths = sorted([float(np.linalg.norm(edge)) for edge in edges], reverse=True)
    if not lengths:
        return 0.0, 0.0
    return float(lengths[0]), float(lengths[-1])


def _load_intrinsics(camera_info_path: Path) -> dict[str, float]:
    try:
        info = _read_json(camera_info_path)
    except Exception:
        return {}
    fx = info.get("fx")
    fy = info.get("fy")
    if fx is None or fy is None:
        k = info.get("K") or info.get("k") or []
        if len(k) >= 5:
            fx = k[0]
            fy = k[4]
    try:
        return {"fx": float(fx), "fy": float(fy)}
    except Exception:
        return {}


def _median_mask_depth_mm(case: CaseInput, mask: np.ndarray) -> float | None:
    if not case.depth_path.exists():
        return None
    depth = cv2.imread(str(case.depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        return None
    valid = (depth > 0) & (mask > 0)
    if int(np.count_nonzero(valid)) < 20:
        valid = depth > 0
    values = depth[valid]
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float64)))


def _axis_length_mm(axis_length_px: float, axis_unit: np.ndarray, case: CaseInput, mask: np.ndarray) -> float | None:
    intr = _load_intrinsics(case.camera_info_path)
    z_mm = _median_mask_depth_mm(case, mask)
    if z_mm is None or not intr:
        return None
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    if fx <= 0 or fy <= 0:
        return None
    unit = _normalize(axis_unit)
    mm_per_px = z_mm * math.sqrt((float(unit[0]) / fx) ** 2 + (float(unit[1]) / fy) ** 2)
    return float(axis_length_px) * mm_per_px


def _candidate_count_for_axis(axis_length_px: float, axis_unit: np.ndarray, case: CaseInput, mask: np.ndarray, args: argparse.Namespace) -> tuple[int, float | None, str]:
    requested = int(args.num_anchor_samples)
    if args.candidate_count_mode != "transducer_diameter_floor":
        return max(int(args.min_candidates), min(int(args.max_candidates), requested)), None, "requested_count_clamped"
    length_mm = _axis_length_mm(axis_length_px, axis_unit, case, mask)
    if length_mm is None or length_mm <= 0 or float(args.transducer_diameter_mm) <= 0:
        count = requested
        source = "fallback_requested_count_missing_physical_axis_length"
    else:
        count = int(math.floor(length_mm / float(args.transducer_diameter_mm)))
        source = "transducer_diameter_floor"
    count = max(int(args.min_candidates), count)
    count = min(int(args.max_candidates), count)
    return max(1, count), length_mm, source


def _base_candidate_row(case_id: str, candidate_id: str, candidate_source: str, path_family: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "candidate_source": candidate_source,
        "path_family": path_family,
        "axis_position_index": None,
        "axis_fraction": None,
        "offset_px": None,
        "p0_x": None,
        "p0_y": None,
        "direction_x": None,
        "direction_y": None,
        "cross_axis_x": None,
        "cross_axis_y": None,
        "contact1_x": None,
        "contact1_y": None,
        "contact2_x": None,
        "contact2_y": None,
        "endpoint1_x": None,
        "endpoint1_y": None,
        "endpoint2_x": None,
        "endpoint2_y": None,
        "path_length_px": None,
        "path_length_mm_if_available": None,
        "mask_intersection_count": 0,
        "valid_candidate": False,
        "failure_reason": None,
        "score": None,
        "edge_quality": None,
    }


def _generate_pipeline_candidates(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    center, long_dir, short_dir, box = _mask_rect(mask)
    major_len_px, minor_len_px = _axis_lengths_from_box(box)
    rows: list[dict[str, Any]] = []
    next_idx = 1
    axis_specs = [
        ("major", long_dir, short_dir, major_len_px),
        ("minor", short_dir, long_dir, minor_len_px),
    ]
    for axis_mode, chosen_axis, measured_axis, axis_length_px in axis_specs:
        candidate_count, axis_length_mm, count_source = _candidate_count_for_axis(axis_length_px, chosen_axis, case, mask, args)
        try:
            domain = build_axis_search_domain(
                object_center_xy=(float(center[0]), float(center[1])),
                chosen_axis_unit_vector_xy=(float(chosen_axis[0]), float(chosen_axis[1])),
                axis_extent_px=float(axis_length_px),
                usable_axis_margin_ratio=float(args.usable_axis_margin_ratio),
                num_anchor_samples=int(candidate_count),
                axis_name=axis_mode,
            )
        except Exception as exc:
            row = _base_candidate_row(case.case_id, f"A{next_idx}", "pipeline_axis_cross_section_hits", f"pipeline_{axis_mode}")
            row["failure_reason"] = f"axis_domain_failed: {exc}"
            rows.append(row)
            next_idx += 1
            continue
        denom = max(abs(domain.s_min), abs(domain.s_max), 1.0)
        for local_idx, s in enumerate(domain.candidate_s_values, start=1):
            candidate_id = f"A{next_idx}"
            row = _base_candidate_row(case.case_id, candidate_id, "pipeline_axis_cross_section_hits", f"pipeline_{axis_mode}")
            row["axis_position_index"] = local_idx
            row["axis_fraction"] = float(s / denom)
            try:
                anchor_xy, left, right = compute_anchor_cross_section_hits(
                    axis_search_domain=domain,
                    candidate_s=float(s),
                    mask=mask,
                    max_search_distance=float(max(mask.shape) * 1.5),
                    step_size_px=float(args.cross_section_step_px),
                )
                a = left.hit_xy
                b = right.hit_xy
                if not a or not b:
                    row["p0_x"] = float(anchor_xy[0])
                    row["p0_y"] = float(anchor_xy[1])
                    row["direction_x"] = float(measured_axis[0])
                    row["direction_y"] = float(measured_axis[1])
                    row["cross_axis_x"] = float(chosen_axis[0])
                    row["cross_axis_y"] = float(chosen_axis[1])
                    row["failure_reason"] = "missing_opposing_boundary_hit"
                    rows.append(row)
                    next_idx += 1
                    continue
                p1 = np.asarray(a, dtype=float)
                p2 = np.asarray(b, dtype=float)
                delta = p2 - p1
                length_px = float(np.linalg.norm(delta))
                if length_px <= 1e-9:
                    row["failure_reason"] = "zero_contact_path_length"
                    rows.append(row)
                    next_idx += 1
                    continue
                direction = delta / length_px
                p0 = (p1 + p2) / 2.0
                length_mm = _axis_length_mm(length_px, direction, case, mask)
                center_ratio = float(s / denom)
                quality = score_edge_quality(
                    rgb_array=rgb,
                    contact_point_a_px=[float(p1[0]), float(p1[1])],
                    contact_point_b_px=[float(p2[0]), float(p2[1])],
                    local_path_length_px=length_px,
                    object_extent_px=float(major_len_px if axis_mode == "minor" else minor_len_px),
                    center_distance_ratio=center_ratio,
                    patch_radius_px=int(args.contact_patch_radius_px),
                )
                row.update(
                    {
                        "p0_x": float(p0[0]),
                        "p0_y": float(p0[1]),
                        "direction_x": float(direction[0]),
                        "direction_y": float(direction[1]),
                        "cross_axis_x": float(chosen_axis[0]),
                        "cross_axis_y": float(chosen_axis[1]),
                        "contact1_x": float(p1[0]),
                        "contact1_y": float(p1[1]),
                        "contact2_x": float(p2[0]),
                        "contact2_y": float(p2[1]),
                        "endpoint1_x": float(p1[0]),
                        "endpoint1_y": float(p1[1]),
                        "endpoint2_x": float(p2[0]),
                        "endpoint2_y": float(p2[1]),
                        "path_length_px": length_px,
                        "path_length_mm_if_available": length_mm,
                        "mask_intersection_count": int(
                            round(((left.distance_from_anchor or 0.0) + (right.distance_from_anchor or 0.0)) / max(float(args.cross_section_step_px), 1e-6))
                        ),
                        "valid_candidate": True,
                        "failure_reason": None,
                        "score": quality.get("deterministic_score"),
                        "edge_quality": json.dumps(
                            {
                                **quality,
                                "axis_mode": axis_mode,
                                "axis_length_px": axis_length_px,
                                "axis_length_mm_if_available": axis_length_mm,
                                "candidate_count_source": count_source,
                            },
                            default=str,
                        ),
                    }
                )
            except Exception as exc:
                row["failure_reason"] = str(exc)
            rows.append(row)
            next_idx += 1
    return rows, box


def _candidate_offsets(mask: np.ndarray, direction: np.ndarray, perpendicular: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return []
    uv = np.column_stack([xs.astype(float), ys.astype(float)])
    center = uv.mean(axis=0)
    r = (uv - center[None, :]) @ perpendicular
    extent = max(abs(float(np.percentile(r, 2))), abs(float(np.percentile(r, 98))))
    base = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    return [offset for offset in base if abs(offset) <= extent + 4 or offset == 0]


def _make_candidate(mask: np.ndarray, case_id: str, family: str, offset: int, direction: np.ndarray, perpendicular: np.ndarray, center: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask > 0)
    uv = np.column_stack([xs.astype(float), ys.astype(float)])
    line_p0 = center + perpendicular * float(offset)
    rel = uv - line_p0[None, :]
    r = np.abs(rel @ perpendicular)
    band = r <= 2.5
    used = uv[band]
    candidate_id = f"{family}_{offset}" if offset < 0 else f"{family}_{offset}"
    row: dict[str, Any] = {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "candidate_source": "minAreaRect_chord_candidates",
        "path_family": family,
        "axis_position_index": None,
        "axis_fraction": None,
        "offset_px": offset,
        "p0_x": float(line_p0[0]),
        "p0_y": float(line_p0[1]),
        "direction_x": float(direction[0]),
        "direction_y": float(direction[1]),
        "cross_axis_x": float(perpendicular[0]),
        "cross_axis_y": float(perpendicular[1]),
        "contact1_x": None,
        "contact1_y": None,
        "contact2_x": None,
        "contact2_y": None,
        "endpoint1_x": None,
        "endpoint1_y": None,
        "endpoint2_x": None,
        "endpoint2_y": None,
        "path_length_px": None,
        "path_length_mm_if_available": None,
        "mask_intersection_count": int(used.shape[0]),
        "valid_candidate": False,
        "failure_reason": None,
        "score": None,
        "edge_quality": None,
    }
    if used.shape[0] < 5:
        row["failure_reason"] = "insufficient_mask_intersection"
        return row
    t = (used - line_p0[None, :]) @ direction
    t0 = float(np.min(t))
    t1 = float(np.max(t))
    p1 = line_p0 + direction * t0
    p2 = line_p0 + direction * t1
    row.update(
        {
            "endpoint1_x": float(p1[0]),
            "endpoint1_y": float(p1[1]),
            "endpoint2_x": float(p2[0]),
            "endpoint2_y": float(p2[1]),
            "path_length_px": float(abs(t1 - t0)),
            "contact1_x": float(p1[0]),
            "contact1_y": float(p1[1]),
            "contact2_x": float(p2[0]),
            "contact2_y": float(p2[1]),
            "valid_candidate": True,
            "failure_reason": None,
        }
    )
    return row


def _generate_fallback_ls_candidates(case_id: str, mask: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    center, long_dir, short_dir, box = _mask_rect(mask)
    rows: list[dict[str, Any]] = []
    for family, direction, perpendicular in [
        ("L", long_dir, short_dir),
        ("S", short_dir, long_dir),
    ]:
        for offset in _candidate_offsets(mask, direction, perpendicular):
            rows.append(_make_candidate(mask, case_id, family, int(offset), direction, perpendicular, center))
    return rows, box


def _generate_candidates(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    if args.candidate_mode == "fallback_ls":
        return _generate_fallback_ls_candidates(case.case_id, mask)
    return _generate_pipeline_candidates(case, rgb, mask, args)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_base(ax: Any, rgb: np.ndarray, mask: np.ndarray, box: np.ndarray) -> None:
    overlay = rgb.copy()
    overlay[mask > 0] = (0.70 * overlay[mask > 0] + 0.30 * np.array([30, 120, 255])).astype(np.uint8)
    ax.imshow(overlay)
    ax.add_patch(Polygon(box, fill=False, edgecolor="white", linewidth=2))
    ax.axis("off")


def _draw_candidate(ax: Any, row: dict[str, Any], *, selected: bool = False) -> None:
    if str(row.get("valid_candidate")) not in {"True", "true", "1"} and not row.get("endpoint1_x"):
        return
    family = str(row.get("path_family") or "")
    if family in {"L", "pipeline_major"}:
        color = "#00d1ff"
    elif family in {"S", "pipeline_minor"}:
        color = "#ffcc00"
    else:
        color = "#ff8c00"
    if selected:
        color = "#ff00ff" if row.get("path_label") == "major" else "#00ff55"
    x1, y1 = float(row["endpoint1_x"]), float(row["endpoint1_y"])
    x2, y2 = float(row["endpoint2_x"]), float(row["endpoint2_y"])
    px, py = float(row["p0_x"]), float(row["p0_y"])
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=3 if selected else 1.6, alpha=0.95)
    ax.scatter([px], [py], s=25 if selected else 14, c=color, edgecolors="black", linewidths=0.5)
    ax.text(px + 5, py - 5, str(row["candidate_id"]), color="white", fontsize=10 if selected else 8, weight="bold", bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})


def _save_candidate_overlay(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, box: np.ndarray, candidates: list[dict[str, Any]], out_png: Path, out_svg: Path, manual: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_base(ax, rgb, mask, box)
    for row in candidates:
        if row.get("valid_candidate"):
            _draw_candidate(ax, row)
    ax.set_title(
        f"{case.case_id} | {case.material} | major/path-1={manual.get('major_axis_mm')} mm | minor/path-2={manual.get('minor_axis_mm')} mm",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)


def _save_candidate_sheet(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, box: np.ndarray, candidates: list[dict[str, Any]], out_png: Path, out_svg: Path) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax_img = fig.add_subplot(gs[0, 0])
    _draw_base(ax_img, rgb, mask, box)
    for row in candidates:
        if row.get("valid_candidate"):
            _draw_candidate(ax_img, row)
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")
    valid = [row for row in candidates if row.get("valid_candidate")]
    lines = ["ID   family          px_len   mm_len  score", ""]
    for row in valid:
        mm = row.get("path_length_mm_if_available")
        mm_text = f"{float(mm):.1f}" if mm not in {None, ""} else ""
        score = row.get("score")
        score_text = f"{float(score):.1f}" if score not in {None, ""} else ""
        lines.append(f"{row['candidate_id']:<4} {str(row['path_family']):<15} {float(row['path_length_px']):>7.1f} {mm_text:>8} {score_text:>6}")
    ax_tbl.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=11)
    fig.suptitle(f"{case.case_id} candidate path sheet", fontsize=14)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)


def _candidate_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]).lower(): row for row in candidates if row.get("valid_candidate")}


def _valid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in candidates if row.get("valid_candidate")]


def _print_candidate_list(candidates: list[dict[str, Any]]) -> None:
    valid = _valid_candidates(candidates)
    if not valid:
        print("No valid candidates.")
        return
    print("Available candidate IDs:")
    print("  ID       family            length_px  length_mm  score")
    for row in valid:
        mm = row.get("path_length_mm_if_available")
        mm_text = f"{float(mm):.1f}" if mm not in {None, ""} else ""
        score = row.get("score")
        score_text = f"{float(score):.1f}" if score not in {None, ""} else ""
        print(f"  {row['candidate_id']:<8} {str(row['path_family']):<16} {float(row['path_length_px']):>9.1f} {mm_text:>10} {score_text:>7}")


def _selection_from_candidate(path_label: str, manual_mm: float | None, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_label": path_label,
        "manual_mm": manual_mm,
        "selected_anchor_id": candidate["candidate_id"],
        "selection_source": "candidate_id",
        "p0_xy": [float(candidate["p0_x"]), float(candidate["p0_y"])],
        "direction_xy": [float(candidate["direction_x"]), float(candidate["direction_y"])],
        "endpoint1_xy": [float(candidate["endpoint1_x"]), float(candidate["endpoint1_y"])],
        "endpoint2_xy": [float(candidate["endpoint2_x"]), float(candidate["endpoint2_y"])],
        "path_length_px": float(candidate["path_length_px"]),
        "candidate_metadata": dict(candidate),
    }


def _selection_from_custom(path_label: str, manual_mm: float | None, text: str) -> dict[str, Any]:
    vals = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(vals) != 4:
        raise ValueError("custom endpoints must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("custom endpoints are identical")
    return {
        "path_label": path_label,
        "manual_mm": manual_mm,
        "selected_anchor_id": "custom_pixels",
        "selection_source": "custom_pixels",
        "p0_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "direction_xy": [dx / length, dy / length],
        "endpoint1_xy": [x1, y1],
        "endpoint2_xy": [x2, y2],
        "path_length_px": length,
        "candidate_metadata": {},
    }


def _selection_from_click(
    case: CaseInput,
    path_label: str,
    manual_mm: float | None,
    rgb: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_base(ax, rgb, mask, box)
    ax.set_title(f"{case.case_id}: click endpoint 1 and endpoint 2 for manual_{path_label}_mm")
    try:
        points = plt.ginput(2, timeout=0)
    finally:
        plt.close(fig)
    if len(points) != 2:
        raise ValueError("click selection did not receive two points")
    (x1, y1), (x2, y2) = points
    selection = _selection_from_custom(path_label, manual_mm, f"{x1},{y1},{x2},{y2}")
    selection["selected_anchor_id"] = "custom_click"
    selection["selection_source"] = "custom_click"
    return selection


def _prompt_selection(
    case: CaseInput,
    path_label: str,
    manual_mm: float | None,
    candidates: list[dict[str, Any]],
    sheet_path: Path,
    overlay_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    by_id = _candidate_by_id(candidates)
    valid_ids = ", ".join(row["candidate_id"] for row in _valid_candidates(candidates))
    while True:
        value = input(
            f"Enter candidate ID for manual_{path_label}_mm / {path_label}-path ({manual_mm} mm)\n"
            "Example: A1, A2, A3. Or type show / list / custom / skip / q: "
        ).strip()
        low = value.lower()
        if low == "q":
            raise KeyboardInterrupt
        if low == "skip":
            return None
        if low == "show":
            preferred = sheet_path if sheet_path.exists() else overlay_path
            print(f"Candidate sheet: {sheet_path}")
            print(f"Candidate overlay: {overlay_path}")
            if args.open_overlays:
                open_image_file(preferred)
            continue
        if low == "list":
            _print_candidate_list(candidates)
            continue
        if low == "custom":
            if args.selection_mode == "click":
                try:
                    return _selection_from_click(case, path_label, manual_mm, rgb, mask, box)
                except Exception as exc:
                    print(f"Click selection failed: {exc}")
                    print("Falling back to typed endpoint pixels.")
            endpoint_text = input("Enter endpoint pixels as x1,y1,x2,y2: ").strip()
            try:
                return _selection_from_custom(path_label, manual_mm, endpoint_text)
            except Exception as exc:
                print(f"Invalid custom endpoint input: {exc}")
                continue
        candidate = by_id.get(low)
        if candidate is None:
            print(f"Unknown candidate ID: {value}")
            print(f"Valid candidate IDs: {valid_ids}")
            continue
        return _selection_from_candidate(path_label, manual_mm, candidate)


def _save_selected_overlay(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, box: np.ndarray, selected: dict[str, Any], out_png: Path, out_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_base(ax, rgb, mask, box)
    for label in ["major", "minor"]:
        item = selected.get(label)
        if not item:
            continue
        row = {
            "candidate_id": item.get("selected_anchor_id"),
            "path_family": "L" if label == "major" else "S",
            "path_label": label,
            "p0_x": item["p0_xy"][0],
            "p0_y": item["p0_xy"][1],
            "endpoint1_x": item["endpoint1_xy"][0],
            "endpoint1_y": item["endpoint1_xy"][1],
            "endpoint2_x": item["endpoint2_xy"][0],
            "endpoint2_y": item["endpoint2_xy"][1],
        }
        _draw_candidate(ax, row, selected=True)
    ax.set_title(
        f"{case.case_id} selected paths | major/path-1={selected.get('major', {}).get('manual_mm')} | minor/path-2={selected.get('minor', {}).get('manual_mm')}",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)


def _manifest_row(case: CaseInput, manual: dict[str, Any], selected: dict[str, Any], notes: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "case_index": case.case_index,
        "material": case.material,
        "object_id": case.object_id,
        "manual_major_mm": manual.get("major_axis_mm"),
        "manual_minor_mm": manual.get("minor_axis_mm"),
        "mask_path": str(case.mask_path),
        "rgb_path": str(case.rgb_path),
        "selection_timestamp": _now(),
        "notes": notes,
    }
    for label in ["major", "minor"]:
        item = selected.get(label) or {}
        prefix = label
        row[f"{prefix}_selected_anchor_id"] = item.get("selected_anchor_id")
        row[f"{prefix}_selection_source"] = item.get("selection_source")
        p0 = item.get("p0_xy") or [None, None]
        d = item.get("direction_xy") or [None, None]
        e1 = item.get("endpoint1_xy") or [None, None]
        e2 = item.get("endpoint2_xy") or [None, None]
        row[f"{prefix}_p0_x"], row[f"{prefix}_p0_y"] = p0[0], p0[1]
        row[f"{prefix}_direction_x"], row[f"{prefix}_direction_y"] = d[0], d[1]
        row[f"{prefix}_endpoint1_x"], row[f"{prefix}_endpoint1_y"] = e1[0], e1[1]
        row[f"{prefix}_endpoint2_x"], row[f"{prefix}_endpoint2_y"] = e2[0], e2[1]
        row[f"{prefix}_path_length_px"] = item.get("path_length_px")
    return row


def _write_integration_notes(path: Path) -> None:
    path.write_text(
        "# Manual Anchor/Path Selection Integration Notes\n\n"
        "1. Previous evaluations using minAreaRect fallback are not final because they did not know the actual manual measurement paths.\n"
        "2. Future evaluation should use `selected_anchors.json` for each case/path.\n"
        "3. For path_label `major`, use `selected_anchors.json[\"major\"][\"p0_xy\"]`, `direction_xy`, and `manual_mm`.\n"
        "4. For path_label `minor`, use `selected_anchors.json[\"minor\"][\"p0_xy\"]`, `direction_xy`, and `manual_mm`.\n"
        "5. Mask-based method should compute chord/path length using this selected path.\n"
        "6. Strip-depth method should compute depth support around this selected path.\n"
        "7. Whole-ROI results should remain diagnostic only unless explicitly requested.\n"
        "8. Do not use minAreaRect major/minor fallback once manual anchor selection exists.\n",
        encoding="utf-8",
    )


def _write_debug_candidate_mode_comparison(case: CaseInput, rgb: np.ndarray, mask: np.ndarray, manual: dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    debug_dir = output_root / "debug_candidate_mode_comparison" / case.case_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    pipeline_candidates, pipeline_box = _generate_pipeline_candidates(case, rgb, mask, args)
    fallback_candidates, fallback_box = _generate_fallback_ls_candidates(case.case_id, mask)
    _write_csv(debug_dir / "pipeline_anchor_candidates.csv", CANDIDATE_COLUMNS, pipeline_candidates)
    _write_csv(debug_dir / "fallback_ls_candidates.csv", CANDIDATE_COLUMNS, fallback_candidates)
    _save_candidate_overlay(
        case,
        rgb,
        mask,
        pipeline_box,
        pipeline_candidates,
        debug_dir / "pipeline_anchor_candidates_overlay.png",
        debug_dir / "pipeline_anchor_candidates_overlay.svg",
        manual,
    )
    _save_candidate_overlay(
        case,
        rgb,
        mask,
        fallback_box,
        fallback_candidates,
        debug_dir / "fallback_ls_candidates_overlay.png",
        debug_dir / "fallback_ls_candidates_overlay.svg",
        manual,
    )
    (debug_dir / "README.md").write_text(
        "# Candidate Mode Comparison\n\n"
        "`pipeline` mode is the default and should be used for manual path selection. It creates A1/A2/A3-style contact-pair candidates by spacing anchor centers along the object axis and finding opposing boundary hits across the selected mask.\n\n"
        "`fallback_ls` is emergency-only. It creates dense minAreaRect-parallel chords such as L_-40 and S_20, which are visually close together and are not the main UPV/VLM contact-pair anchor candidates.\n",
        encoding="utf-8",
    )


def _make_grid(image_paths: list[Path], out_png: Path, out_svg: Path, title: str) -> None:
    if not image_paths:
        return
    cols = min(5, len(image_paths))
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, image_paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(path.parent.name, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_svg)
    plt.close(fig)


def _read_existing_manifest(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "manual_anchor_selection_manifest.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if not _is_backup_case_name(str(row.get("case_id") or ""))]


def _selected_payload(case: CaseInput, manual: dict[str, Any], major: dict[str, Any] | None, minor: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "material": case.material,
        "object_id": case.object_id,
        "manual_measurements": {"major_axis_mm": manual.get("major_axis_mm"), "minor_axis_mm": manual.get("minor_axis_mm")},
        "major": major,
        "minor": minor,
    }


def run(args: argparse.Namespace) -> Path:
    capture_session = _resolve(args.capture_session)
    mask_root = _resolve(args.mask_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases").mkdir(exist_ok=True)
    (output_root / "grids").mkdir(exist_ok=True)
    _write_integration_notes(output_root / "INTEGRATION_NOTES.md")
    _write_json(
        output_root / "manual_anchor_selection_run_config.json",
        {
            "created_at": _now(),
            "git_branch": _git_value(["git", "branch", "--show-current"]),
            "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
            "capture_session": str(capture_session),
            "mask_root": str(mask_root),
            "candidate_mode": args.candidate_mode,
            "candidate_source": "pipeline_axis_cross_section_hits" if args.candidate_mode == "pipeline" else "minAreaRect_chord_candidates",
            "candidate_count_mode": args.candidate_count_mode,
            "transducer_diameter_mm": args.transducer_diameter_mm,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "usable_axis_margin_ratio": args.usable_axis_margin_ratio,
            "dry_run": bool(args.dry_run),
            "dry_run_open": bool(args.dry_run_open),
            "dry_run_case_flow": bool(args.dry_run_case_flow),
            "debug_case_flow": bool(args.debug_case_flow),
            "case_id": args.case_id,
            "selection_mode": args.selection_mode,
            "notes": "No final path-length metrics are computed by this selector.",
        },
    )

    manifest_rows: list[dict[str, Any]] = _read_existing_manifest(output_root) if args.resume else []
    manifest_rows = [row for row in manifest_rows if row.get("case_id")]
    done_cases = {str(row.get("case_id")) for row in manifest_rows if row.get("case_id")}
    candidate_grid_paths: list[Path] = []
    selected_grid_paths: list[Path] = []
    cases = _load_cases(capture_session, mask_root, args)
    if args.case_id:
        print(f"Processing exactly one case: {args.case_id}")
        if len(cases) != 1:
            raise RuntimeError(f"--case-id {args.case_id!r} resolved to {len(cases)} cases; expected exactly one")
    if args.dry_run_open and args.case_id is None:
        cases = cases[:1]
    for case in cases:
        case_out = output_root / "cases" / case.case_id
        selected_path = case_out / "selected_anchors.json"
        already_selected = case.case_id in done_cases or selected_path.exists()
        if args.case_id and already_selected and not args.force_regenerate_candidates:
            if args.resume or args.skip_existing_selections:
                if args.debug_case_flow:
                    print(f"[case-flow] resume/skip_existing selected state for {case.case_id}; continuing because --case-id should not silently skip")
            else:
                answer = input(f"{selected_path} already exists. Overwrite selection for {case.case_id}? [y/N]: ").strip().lower()
                if answer not in {"y", "yes"}:
                    print(f"Not overwriting existing selection for {case.case_id}.")
                    continue
        elif (args.resume or args.skip_existing_selections) and already_selected and not args.force_regenerate_candidates:
            print(f"Skipping already selected case: {case.case_id}")
            if args.debug_case_flow:
                print(f"[case-flow] skip {case.case_id}: already selected and resume/skip_existing enabled")
            continue
        case_out.mkdir(parents=True, exist_ok=True)
        if not _wait_for_mask(case, args):
            if args.case_id:
                raise FileNotFoundError(f"Missing selected mask for {case.case_id}: {case.mask_npy_path} or {case.mask_path}")
            _write_json(case_out / "selection_failure.json", {"case_id": case.case_id, "failure_reason": "missing_selected_mask", "mask_path": str(case.mask_path)})
            if args.debug_case_flow:
                print(f"[case-flow] skip {case.case_id}: missing selected mask")
            continue
        manual = _validate_case_inputs(case, args)
        rgb = _read_rgb(case.rgb_path)
        mask = _load_mask(case)
        if mask.shape[:2] != rgb.shape[:2]:
            raise RuntimeError(
                f"Mask/image shape mismatch for {case.case_id}: mask={mask.shape[:2]} rgb={rgb.shape[:2]}. "
                "This usually means the mask was generated from a stale manifest path or backup/original case. "
                "Regenerate GSAM2 for this exact case after the manifest/path fix."
            )
        candidates, box = _generate_candidates(case, rgb, mask, args)
        _write_csv(case_out / "anchor_candidates.csv", CANDIDATE_COLUMNS, candidates)
        _write_json(
            case_out / "anchor_candidates.json",
            {
                "case_id": case.case_id,
                "candidate_mode": args.candidate_mode,
                "candidate_source": "pipeline_axis_cross_section_hits" if args.candidate_mode == "pipeline" else "minAreaRect_chord_candidates",
                "candidates": candidates,
            },
        )
        if case.case_id == "case_001_1":
            _write_debug_candidate_mode_comparison(case, rgb, mask, manual, output_root, args)
        overlay_png = case_out / "anchor_candidates_overlay.png"
        overlay_svg = case_out / "anchor_candidates_overlay.svg"
        sheet_png = case_out / "anchor_candidates_sheet.png"
        sheet_svg = case_out / "anchor_candidates_sheet.svg"
        _save_candidate_overlay(case, rgb, mask, box, candidates, overlay_png, overlay_svg, manual)
        _save_candidate_sheet(case, rgb, mask, box, candidates, sheet_png, sheet_svg)
        candidate_grid_paths.append(overlay_png)
        preferred_open_path = sheet_png if sheet_png.exists() else overlay_png
        if args.dry_run_case_flow:
            valid = _valid_candidates(candidates)
            print(f"\nCase: {case.case_id}")
            print(f"Material: {case.material}")
            print(f"Manual major/path-1 measurement: {manual.get('major_axis_mm')} mm")
            print(f"Manual minor/path-2 measurement: {manual.get('minor_axis_mm')} mm")
            print(f"Candidate CSV: {case_out / 'anchor_candidates.csv'}")
            print(f"Candidate sheet: {sheet_png}")
            print(f"Candidate overlay: {overlay_png}")
            print(f"Candidate count: {len(valid)} valid / {len(candidates)} total")
            print("Candidate IDs:", ", ".join(str(row["candidate_id"]) for row in valid))
            _print_candidate_list(candidates)
            continue
        if args.dry_run_open:
            print(f"\nCase: {case.case_id}")
            print(f"Opening candidate sheet:\n{preferred_open_path}")
            opened = open_image_file(preferred_open_path) if args.open_overlays else False
            print(f"open_image_file succeeded: {opened}")
            continue
        if args.dry_run:
            continue
        print(f"\nCase: {case.case_id}")
        print(f"Material: {case.material}")
        print(f"Manual major/path-1 measurement: {manual.get('major_axis_mm')} mm")
        print(f"Manual minor/path-2 measurement: {manual.get('minor_axis_mm')} mm")
        print(f"Candidate sheet: {sheet_png}")
        print(f"Candidate overlay: {overlay_png}")
        if args.open_overlays:
            print(f"Opening candidate sheet:\n{preferred_open_path}")
            open_image_file(preferred_open_path)
        print("Candidate sheet opened. Use the visible candidate IDs.")
        _print_candidate_list(candidates)

        selected: dict[str, Any] | None = None
        while True:
            major = _prompt_selection(case, "major", manual.get("major_axis_mm"), candidates, sheet_png, overlay_png, rgb, mask, box, args)
            minor = _prompt_selection(case, "minor", manual.get("minor_axis_mm"), candidates, sheet_png, overlay_png, rgb, mask, box, args)
            selected = _selected_payload(case, manual, major, minor)
            selected_png = case_out / "selected_anchors_overlay.png"
            selected_svg = case_out / "selected_anchors_overlay.svg"
            _save_selected_overlay(case, rgb, mask, box, selected, selected_png, selected_svg)
            if args.open_overlays:
                print(f"Opening selected overlay:\n{selected_png}")
                open_image_file(selected_png)
            while True:
                action = input("Save this selection? [s] save / [e] edit / [show] show selected overlay / [skip] skip case / [q] quit: ").strip().lower()
                if action == "q":
                    raise KeyboardInterrupt
                if action == "skip":
                    selected = None
                    break
                if action == "show":
                    print(f"Selected overlay: {selected_png}")
                    if args.open_overlays:
                        open_image_file(selected_png)
                    continue
                if action == "e":
                    break
                if action in {"", "s", "save"}:
                    break
                print("Unknown action. Use s, e, show, skip, or q.")
            if action in {"", "s", "save", "skip"}:
                break
        if not selected:
            continue
        _write_json(case_out / "selected_anchors.json", selected)
        selected_grid_paths.append(selected_png)
        manifest_rows = [row for row in manifest_rows if row.get("case_id") != case.case_id]
        manifest_rows.append(_manifest_row(case, manual, selected))
        _write_csv(output_root / "manual_anchor_selection_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
        _write_json(output_root / "manual_anchor_selection_manifest.json", {"rows": manifest_rows, "row_count": len(manifest_rows), "updated_at": _now()})

    if args.dry_run_open or args.dry_run_case_flow:
        pass
    elif args.dry_run:
        _write_json(output_root / "manual_anchor_selection_manifest.json", {"rows": [], "row_count": 0, "dry_run": True, "updated_at": _now()})
        _write_csv(output_root / "manual_anchor_selection_manifest.csv", MANIFEST_COLUMNS, [])
    else:
        _write_csv(output_root / "manual_anchor_selection_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
        _write_json(output_root / "manual_anchor_selection_manifest.json", {"rows": manifest_rows, "row_count": len(manifest_rows), "updated_at": _now()})
    _make_grid(candidate_grid_paths, output_root / "grids" / "all_cases_anchor_candidates_grid.png", output_root / "grids" / "all_cases_anchor_candidates_grid.svg", "Anchor/path candidates")
    _make_grid(selected_grid_paths, output_root / "grids" / "all_cases_selected_anchors_grid.png", output_root / "grids" / "all_cases_selected_anchors_grid.svg", "Selected manual anchor paths")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select manual path anchors for the new ROS2 path-length capture dataset.")
    parser.add_argument("--capture-session", default=str(DEFAULT_SESSION))
    parser.add_argument("--mask-root", default=str(DEFAULT_SESSION / "gsam2_masks"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-open", action="store_true", help="Generate candidates for the first matching case, try to open the sheet, then exit without prompting.")
    parser.add_argument("--dry-run-case-flow", action="store_true", help="Resolve one case, validate inputs, generate candidates/sheets, print candidate IDs, and exit without prompting.")
    parser.add_argument("--debug-case-flow", action="store_true", help="Print case resolution, skip decisions, input paths, and parsed manual values.")
    parser.add_argument("--candidate-mode", choices=["pipeline", "fallback_ls"], default="pipeline", help="Candidate generator to use. Default uses main-pipeline-style A* cross-section anchors.")
    parser.add_argument("--candidate-count-mode", choices=["transducer_diameter_floor", "requested_count"], default="transducer_diameter_floor")
    parser.add_argument("--num-anchor-samples", type=int, default=5)
    parser.add_argument("--transducer-diameter-mm", type=float, default=50.0)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--usable-axis-margin-ratio", type=float, default=0.18)
    parser.add_argument("--cross-section-step-px", type=float, default=1.0)
    parser.add_argument("--contact-patch-radius-px", type=int, default=18)
    parser.add_argument("--wait-for-masks", action="store_true")
    parser.add_argument("--mask-wait-timeout-sec", type=float, default=600.0)
    parser.add_argument("--open-overlays", dest="open_overlays", action="store_true", default=False)
    parser.add_argument("--no-open-overlays", dest="open_overlays", action="store_false")
    parser.add_argument("--selection-mode", choices=["terminal", "click"], default="terminal")
    parser.add_argument("--force-regenerate-candidates", action="store_true")
    parser.add_argument("--skip-existing-selections", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        out = run(parse_args())
    except KeyboardInterrupt:
        print("\nSelection interrupted by user.")
        return 130
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
