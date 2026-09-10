#!/usr/bin/env python3
"""
Experiment 1 ablation replay:
GroundingDINO+SAM2 top-detection baseline vs.
GroundingDINO+SAM2+CLIP crop-verification pipeline.

This script does NOT rerun GroundingDINO, SAM2, or CLIP.
It only replays the saved artifacts from the completed T5_v2 Experiment 1 session.

Baseline:
    GSAM-only = candidate with highest detection_score in saved candidate pool.

Reference/proposed:
    +CLIP = final_selected_candidate_id from target_selection_summary.json.
    This is valid here because the completed +CLIP run was manually verified
    correct for all 27 trials.

Expected command:

python -m upv_vlm_v2.experiments.experiment_target_selection_ablation \
  --source-session outputs/T5_v2_experiment_1_target_selection_fixed27/session_20260518_142404 \
  --mode run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_copy(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def to_float(value: Any, default: float = float("-inf")) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().upper() in {"NA", "NONE", ""}:
            return default
        return float(value)
    except Exception:
        return default


def normalize_material_name(s: Any) -> str:
    return str(s or "").strip().lower().replace("_", " ")


def split_scene_classes(classes_existing: str) -> List[str]:
    if not classes_existing:
        return []
    return [x.strip() for x in str(classes_existing).split(",") if x.strip()]


def compact_json(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"))
    except Exception:
        return str(value)


def resolve_path(path_like: Any, repo_root: Path) -> Optional[Path]:
    if not path_like or str(path_like).upper() == "NA":
        return None
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return repo_root / p


def find_case_dirs(source_session: Path) -> List[Path]:
    cases_dir = source_session / "cases"
    return sorted([p for p in cases_dir.glob("trial_*") if p.is_dir()])


def find_attempt_dir(case_dir: Path) -> Path:
    # Current experiment uses attempt_001. Keep this flexible.
    attempts_dir = case_dir / "attempts"
    if not attempts_dir.exists():
        return case_dir
    attempts = sorted([p for p in attempts_dir.glob("attempt_*") if p.is_dir()])
    if attempts:
        return attempts[0]
    return case_dir


# -------------------------------------------------------------------------
# Candidate extraction
# -------------------------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: str
    detection_score: float
    bbox: Optional[List[float]]
    mask_path: Optional[Path]
    bbox_crop_path: Optional[Path]
    inner_texture_crop_path: Optional[Path]
    masked_texture_crop_path: Optional[Path]
    source_detection_query: str
    annotation_index: Optional[int]
    raw: Dict[str, Any]


def extract_candidates_from_manifest(
    manifest: Any,
    repo_root: Path,
) -> List[Candidate]:
    """
    Supports multiple possible saved schema shapes.

    Expected ideal shapes:
    - {"candidates": [...]}
    - {"candidate_pool": [...]}
    - [...]
    """
    if manifest is None:
        return []

    if isinstance(manifest, list):
        raw_candidates = manifest
    elif isinstance(manifest, dict):
        if isinstance(manifest.get("candidates"), list):
            raw_candidates = manifest["candidates"]
        elif isinstance(manifest.get("candidate_pool"), list):
            raw_candidates = manifest["candidate_pool"]
        elif isinstance(manifest.get("items"), list):
            raw_candidates = manifest["items"]
        else:
            raw_candidates = []
    else:
        raw_candidates = []

    return [candidate_from_dict(c, repo_root) for c in raw_candidates if isinstance(c, dict)]


def extract_candidates_from_target_summary(
    target_summary: Dict[str, Any],
    repo_root: Path,
) -> List[Candidate]:
    """
    The uploaded bundle showed that target_selection_summary.json contains:
        artifacts -> material_verification -> candidates
    Each candidate includes detection_score, bbox, mask_path, crop paths, etc.
    """
    paths = [
        ["artifacts", "material_verification", "candidates"],
        ["material_verification", "candidates"],
        ["result", "artifacts", "material_verification", "candidates"],
        ["result", "candidates"],
        ["candidates"],
    ]

    for chain in paths:
        node: Any = target_summary
        for key in chain:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, list):
            return [candidate_from_dict(c, repo_root) for c in node if isinstance(c, dict)]

    return []


def candidate_from_dict(c: Dict[str, Any], repo_root: Path) -> Candidate:
    candidate_id = str(
        c.get("candidate_id")
        or c.get("id")
        or c.get("name")
        or c.get("candidate_name")
        or "UNKNOWN_CANDIDATE"
    )

    detection_score = to_float(
        c.get("detection_score", c.get("score", c.get("confidence", float("-inf"))))
    )

    bbox = (
        c.get("bbox")
        or c.get("bbox_xyxy")
        or c.get("box")
        or c.get("crop_info", {}).get("bbox_xyxy")
    )
    if bbox is not None:
        try:
            bbox = [float(x) for x in bbox]
        except Exception:
            bbox = None

    annotation_index = c.get("annotation_index")
    try:
        annotation_index = int(annotation_index) if annotation_index is not None else None
    except Exception:
        annotation_index = None

    return Candidate(
        candidate_id=candidate_id,
        detection_score=detection_score,
        bbox=bbox,
        mask_path=resolve_path(c.get("mask_path"), repo_root),
        bbox_crop_path=resolve_path(
            c.get("bbox_crop_path") or c.get("crop_path"), repo_root
        ),
        inner_texture_crop_path=resolve_path(c.get("inner_texture_crop_path"), repo_root),
        masked_texture_crop_path=resolve_path(c.get("masked_texture_crop_path"), repo_root),
        source_detection_query=str(c.get("source_detection_query") or ""),
        annotation_index=annotation_index,
        raw=c,
    )


def load_candidates(
    attempt_dir: Path,
    target_summary: Dict[str, Any],
    repo_root: Path,
) -> Tuple[List[Candidate], str]:
    manifest_path = attempt_dir / "candidate_pool" / "candidate_pool_manifest.json"
    manifest = load_json(manifest_path, default=None)
    candidates = extract_candidates_from_manifest(manifest, repo_root)
    if candidates:
        return candidates, str(manifest_path)

    candidates = extract_candidates_from_target_summary(target_summary, repo_root)
    if candidates:
        return candidates, "target_selection_summary.json::artifacts.material_verification.candidates"

    return [], "NONE_FOUND"


def choose_gsam_only_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None

    # Primary rule: highest GroundingDINO detection score.
    # Tie-breaker: lower annotation index, then stable candidate_id.
    def key(c: Candidate) -> Tuple[float, float, str]:
        idx = c.annotation_index if c.annotation_index is not None else 999999
        return (c.detection_score, -idx, c.candidate_id)

    return max(candidates, key=key)


def find_candidate_by_id(candidates: List[Candidate], candidate_id: str) -> Optional[Candidate]:
    for c in candidates:
        if c.candidate_id == candidate_id:
            return c
    return None


def candidate_to_record(prefix: str, cand: Optional[Candidate]) -> Dict[str, Any]:
    if cand is None:
        return {
            f"{prefix}_candidate_id": "NA",
            f"{prefix}_detection_score": "NA",
            f"{prefix}_bbox": "NA",
            f"{prefix}_mask_path": "NA",
            f"{prefix}_source_detection_query": "NA",
            f"{prefix}_annotation_index": "NA",
        }

    return {
        f"{prefix}_candidate_id": cand.candidate_id,
        f"{prefix}_detection_score": cand.detection_score,
        f"{prefix}_bbox": compact_json(cand.bbox),
        f"{prefix}_mask_path": str(cand.mask_path) if cand.mask_path else "NA",
        f"{prefix}_source_detection_query": cand.source_detection_query or "NA",
        f"{prefix}_annotation_index": cand.annotation_index if cand.annotation_index is not None else "NA",
    }


# -------------------------------------------------------------------------
# Image generation
# -------------------------------------------------------------------------


def get_font(size: int = 16):
    # DejaVu is common on Ubuntu, but fallback cleanly.
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in font_candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def image_with_header(img: Image.Image, title_lines: List[str], header_h: int = 84) -> Image.Image:
    img = img.convert("RGB")
    out = Image.new("RGB", (img.width, img.height + header_h), "white")
    out.paste(img, (0, header_h))
    draw = ImageDraw.Draw(out)
    font_big = get_font(17)
    font_small = get_font(13)

    y = 8
    for i, line in enumerate(title_lines):
        font = font_big if i == 0 else font_small
        draw.text((12, y), str(line), fill=(0, 0, 0), font=font)
        y += 24 if i == 0 else 18
    return out


def create_selected_overlay(
    raw_rgb_path: Optional[Path],
    mask_path: Optional[Path],
    output_path: Path,
    title_lines: List[str],
    bbox: Optional[List[float]] = None,
    overlay_rgb: Tuple[int, int, int] = (255, 0, 0),
) -> Optional[str]:
    if raw_rgb_path is None or not raw_rgb_path.exists():
        return None

    raw = Image.open(raw_rgb_path).convert("RGB")
    overlay = Image.new("RGBA", raw.size, (0, 0, 0, 0))

    if mask_path is not None and mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize(raw.size)
        mask_bin = mask.point(lambda p: 255 if p > 0 else 0)

        color_layer = Image.new("RGBA", raw.size, (*overlay_rgb, 92))
        overlay.paste(color_layer, (0, 0), mask_bin)

        edges = mask_bin.filter(ImageFilter.FIND_EDGES)
        edge_layer = Image.new("RGBA", raw.size, (*overlay_rgb, 255))
        overlay.paste(edge_layer, (0, 0), edges)

    composed = Image.alpha_composite(raw.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    font = get_font(14)

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=overlay_rgb, width=3)

    composed = image_with_header(composed, title_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)
    return str(output_path)


def resize_to_panel(img: Image.Image, panel_size: Tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    target_w, target_h = panel_size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h))
    canvas = Image.new("RGB", panel_size, "white")
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(img_resized, (x, y))
    return canvas


def add_panel_label(img: Image.Image, label: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = get_font(18)
    draw.rectangle([0, 0, out.width, 32], fill=(255, 255, 255))
    draw.text((8, 6), label, fill=(0, 0, 0), font=font)
    return out


def create_side_by_side_ablation(
    raw_rgb_path: Optional[Path],
    candidate_masks_overlay_path: Optional[Path],
    gsam_overlay_path: Optional[Path],
    clip_overlay_path: Optional[Path],
    output_path: Path,
    title: str,
) -> Optional[str]:
    paths = [
        raw_rgb_path,
        candidate_masks_overlay_path,
        gsam_overlay_path,
        clip_overlay_path,
    ]
    labels = [
        "A. Raw RGB",
        "B. Candidate masks",
        "C. GSAM-only top detection",
        "D. +CLIP crop-verified",
    ]

    images: List[Image.Image] = []
    for p, label in zip(paths, labels):
        if p is not None and p.exists():
            img = Image.open(p).convert("RGB")
        else:
            img = Image.new("RGB", (640, 480), "white")
            d = ImageDraw.Draw(img)
            d.text((20, 20), "Missing image", fill=(0, 0, 0), font=get_font(18))
        images.append(add_panel_label(resize_to_panel(img, (480, 380)), label))

    header_h = 54
    margin = 14
    grid_w = 2 * 480 + 3 * margin
    grid_h = header_h + 2 * 380 + 3 * margin
    out = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(out)
    draw.text((margin, 14), title, fill=(0, 0, 0), font=get_font(20))

    positions = [
        (margin, header_h + margin),
        (2 * margin + 480, header_h + margin),
        (margin, header_h + 2 * margin + 380),
        (2 * margin + 480, header_h + 2 * margin + 380),
    ]
    for img, pos in zip(images, positions):
        out.paste(img, pos)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return str(output_path)


# -------------------------------------------------------------------------
# Per-case processing
# -------------------------------------------------------------------------


def process_case(
    case_dir: Path,
    out_case_dir: Path,
    source_session: Path,
    ablation_session: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    attempt_dir = find_attempt_dir(case_dir)

    trial_metadata = load_json(case_dir / "trial_metadata.json", default={}) or {}
    manual_label = load_json(case_dir / "manual_label.json", default={}) or {}
    target_summary = load_json(case_dir / "target_selection_summary.json", default={}) or {}

    # Prefer attempt-level summary if case-level summary is missing.
    if not target_summary:
        target_summary = load_json(attempt_dir / "target_selection_summary.json", default={}) or {}

    candidates, candidate_source = load_candidates(attempt_dir, target_summary, repo_root)

    gsam_candidate = choose_gsam_only_candidate(candidates)

    clip_candidate_id = (
        target_summary.get("final_selected_candidate_id")
        or target_summary.get("selected_candidate_id")
        or target_summary.get("result", {}).get("selected_candidate_id")
        or target_summary.get("result", {}).get("selected_label_or_NONE")
        or "NA"
    )
    clip_candidate_id = str(clip_candidate_id)

    clip_candidate = find_candidate_by_id(candidates, clip_candidate_id)

    raw_rgb_path = resolve_path(
        target_summary.get("artifacts", {}).get("raw_rgb_path")
        or target_summary.get("result", {}).get("image_path")
        or attempt_dir / "raw_rgb.png",
        repo_root,
    )
    candidate_masks_overlay_path = resolve_path(
        target_summary.get("artifacts", {}).get("candidate_masks_overlay_path")
        or attempt_dir / "candidate_masks_overlay.png",
        repo_root,
    )
    candidate_crops_panel_path = resolve_path(
        target_summary.get("artifacts", {}).get("candidate_crops_panel_path")
        or attempt_dir / "candidate_crops_panel.png",
        repo_root,
    )
    material_verification_panel_path = resolve_path(
        target_summary.get("artifacts", {}).get("material_verification_panel_path")
        or attempt_dir / "material_verification_panel.png",
        repo_root,
    )
    clip_selected_overlay_original = resolve_path(
        target_summary.get("artifacts", {}).get("selected_mask_overlay_path")
        or attempt_dir / "selected_mask_overlay.png",
        repo_root,
    )

    out_case_dir.mkdir(parents=True, exist_ok=True)

    safe_copy(raw_rgb_path, out_case_dir / "raw_rgb.png")
    safe_copy(candidate_masks_overlay_path, out_case_dir / "candidate_masks_overlay.png")
    safe_copy(candidate_crops_panel_path, out_case_dir / "candidate_crops_panel.png")
    safe_copy(material_verification_panel_path, out_case_dir / "material_verification_panel.png")
    safe_copy(clip_selected_overlay_original, out_case_dir / "clip_selected_overlay_original.png")

    trial_index = trial_metadata.get("trial_index", "NA")
    frame_id = trial_metadata.get("frame_id", "NA")
    input_text = trial_metadata.get("input_text", "NA")
    expected_selected_class = trial_metadata.get("expected_selected_class", "NA")
    classes_existing = trial_metadata.get("classes_existing_in_frame", "NA")

    clip_manual_correct = normalize_material_name(manual_label.get("correct_manual")) == "yes"
    gsam_candidate_id = gsam_candidate.candidate_id if gsam_candidate else "NA"

    candidate_id_match = gsam_candidate_id == clip_candidate_id
    # In this specific completed experiment, +CLIP selected candidate is the reference
    # because all final selections were manually verified as correct.
    gsam_only_correct_auto = bool(candidate_id_match)
    clip_corrected_gsam_only = bool(clip_manual_correct and not gsam_only_correct_auto)

    scene_class_count = len(split_scene_classes(str(classes_existing)))
    candidate_count = len(candidates)
    candidate_count_matches_scene_class_count = (
        scene_class_count > 0 and candidate_count == scene_class_count
    )

    material_scores = target_summary.get("material_scores", {}) or {}
    if not material_scores:
        material_scores = (
            target_summary.get("artifacts", {})
            .get("material_verification", {})
            .get("material_scores", {})
            or {}
        )

    gsam_overlay_path = out_case_dir / "gsam_only_selected_overlay.png"
    clip_overlay_path = out_case_dir / "clip_selected_overlay.png"
    side_by_side_path = out_case_dir / "side_by_side_ablation.png"

    create_selected_overlay(
        raw_rgb_path=raw_rgb_path,
        mask_path=gsam_candidate.mask_path if gsam_candidate else None,
        output_path=gsam_overlay_path,
        title_lines=[
            "GroundingDINO + SAM2 only",
            f"Selected: {gsam_candidate_id}",
            f"Detection score: {gsam_candidate.detection_score:.4f}" if gsam_candidate else "Detection score: NA",
            f"Matches +CLIP reference: {'YES' if candidate_id_match else 'NO'}",
        ],
        bbox=gsam_candidate.bbox if gsam_candidate else None,
        overlay_rgb=(220, 30, 30),
    )

    create_selected_overlay(
        raw_rgb_path=raw_rgb_path,
        mask_path=clip_candidate.mask_path if clip_candidate else None,
        output_path=clip_overlay_path,
        title_lines=[
            "Proposed: GroundingDINO + SAM2 + CLIP",
            f"Selected: {clip_candidate_id}",
            f"Manual correctness: {'YES' if clip_manual_correct else 'NO'}",
            f"Candidate changed from GSAM-only: {'YES' if not candidate_id_match else 'NO'}",
        ],
        bbox=clip_candidate.bbox if clip_candidate else None,
        overlay_rgb=(30, 130, 60),
    )

    create_side_by_side_ablation(
        raw_rgb_path=raw_rgb_path,
        candidate_masks_overlay_path=out_case_dir / "candidate_masks_overlay.png",
        gsam_overlay_path=gsam_overlay_path,
        clip_overlay_path=clip_overlay_path,
        output_path=side_by_side_path,
        title=f"Experiment 1 Ablation | Trial {trial_index} | Query: {input_text}",
    )

    row: Dict[str, Any] = {
        "trial_index": trial_index,
        "frame_id": frame_id,
        "classes_existing_in_frame": classes_existing,
        "scene_class_count": scene_class_count,
        "input_text": input_text,
        "expected_selected_class": expected_selected_class,
        "candidate_count": candidate_count,
        "candidate_count_matches_scene_class_count": candidate_count_matches_scene_class_count,
        "candidate_source": candidate_source,
        "clip_reference_definition": "final_selected_candidate_id_from_manually_verified_clip_pipeline",
        "clip_manual_correct": "yes" if clip_manual_correct else "no",
        "candidate_id_match": "yes" if candidate_id_match else "no",
        "gsam_only_correct_auto": "yes" if gsam_only_correct_auto else "no",
        "clip_corrected_gsam_only": "yes" if clip_corrected_gsam_only else "no",
        "no_verified_match_original": target_summary.get("no_verified_match", "NA"),
        "target_selected_original": target_summary.get("target_selected", "NA"),
        "material_verification_method": target_summary.get("material_verification_method", "NA"),
        "clip_material_score_brick": material_scores.get("brick", "NA"),
        "clip_material_score_timber": material_scores.get("timber", "NA"),
        "clip_material_score_concrete_block": material_scores.get("concrete block", "NA"),
        "source_case_dir": str(case_dir),
        "source_attempt_dir": str(attempt_dir),
        "output_case_dir": str(out_case_dir),
        "raw_rgb_path": str(out_case_dir / "raw_rgb.png"),
        "candidate_masks_overlay_path": str(out_case_dir / "candidate_masks_overlay.png"),
        "candidate_crops_panel_path": str(out_case_dir / "candidate_crops_panel.png"),
        "material_verification_panel_path": str(out_case_dir / "material_verification_panel.png"),
        "gsam_only_selected_overlay_path": str(gsam_overlay_path),
        "clip_selected_overlay_path": str(clip_overlay_path),
        "side_by_side_ablation_path": str(side_by_side_path),
    }

    row.update(candidate_to_record("gsam_only", gsam_candidate))
    row.update(candidate_to_record("clip", clip_candidate))

    case_result = {
        "row": row,
        "trial_metadata": trial_metadata,
        "manual_label": manual_label,
        "candidate_ids": [c.candidate_id for c in candidates],
        "candidate_detection_scores": {
            c.candidate_id: c.detection_score for c in candidates
        },
        "gsam_only_rule": "highest_detection_score",
        "gsam_only_candidate_id": gsam_candidate_id,
        "clip_candidate_id": clip_candidate_id,
        "candidate_id_match": candidate_id_match,
        "gsam_only_correct_auto": gsam_only_correct_auto,
        "clip_corrected_gsam_only": clip_corrected_gsam_only,
    }
    save_json(case_result, out_case_dir / "ablation_case_result.json")

    return row


# -------------------------------------------------------------------------
# Tables and summaries
# -------------------------------------------------------------------------


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    gsam_correct = sum(1 for r in rows if r.get("gsam_only_correct_auto") == "yes")
    clip_correct = sum(1 for r in rows if r.get("clip_manual_correct") == "yes")
    changed = sum(1 for r in rows if r.get("candidate_id_match") == "no")
    corrections = sum(1 for r in rows if r.get("clip_corrected_gsam_only") == "yes")

    return {
        "experiment_name": "experiment_1_target_selection_clip_ablation",
        "n_trials": n,
        "baseline_method": "GroundingDINO+SAM2 top-detection baseline",
        "proposed_method": "GroundingDINO+SAM2+CLIP crop verification",
        "reference_definition": "final_selected_candidate_id from manually verified +CLIP pipeline",
        "gsam_only_correct": gsam_correct,
        "gsam_only_incorrect": n - gsam_correct,
        "gsam_only_accuracy": gsam_correct / n if n else None,
        "clip_correct": clip_correct,
        "clip_incorrect": n - clip_correct,
        "clip_accuracy": clip_correct / n if n else None,
        "clip_candidate_changes": changed,
        "clip_corrections": corrections,
    }


def write_paper_tables(rows: List[Dict[str, Any]], paper_tables_dir: Path) -> None:
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_rows(rows)
    main_table = [
        {
            "method": "GroundingDINO+SAM2 top-detection baseline",
            "trials": summary["n_trials"],
            "correct": summary["gsam_only_correct"],
            "incorrect": summary["gsam_only_incorrect"],
            "accuracy_percent": round(100.0 * summary["gsam_only_accuracy"], 2)
            if summary["gsam_only_accuracy"] is not None
            else "NA",
        },
        {
            "method": "GroundingDINO+SAM2+CLIP crop verification",
            "trials": summary["n_trials"],
            "correct": summary["clip_correct"],
            "incorrect": summary["clip_incorrect"],
            "accuracy_percent": round(100.0 * summary["clip_accuracy"], 2)
            if summary["clip_accuracy"] is not None
            else "NA",
        },
    ]
    write_csv(main_table, paper_tables_dir / "table_experiment_1_ablation_summary.csv")

    per_material: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "material": "",
            "trials": 0,
            "gsam_only_correct": 0,
            "gsam_only_incorrect": 0,
            "clip_correct": 0,
            "clip_incorrect": 0,
            "clip_corrections": 0,
        }
    )

    for r in rows:
        mat = normalize_material_name(r.get("input_text"))
        d = per_material[mat]
        d["material"] = mat
        d["trials"] += 1
        d["gsam_only_correct"] += 1 if r.get("gsam_only_correct_auto") == "yes" else 0
        d["gsam_only_incorrect"] += 1 if r.get("gsam_only_correct_auto") != "yes" else 0
        d["clip_correct"] += 1 if r.get("clip_manual_correct") == "yes" else 0
        d["clip_incorrect"] += 1 if r.get("clip_manual_correct") != "yes" else 0
        d["clip_corrections"] += 1 if r.get("clip_corrected_gsam_only") == "yes" else 0

    per_material_rows = []
    for mat in sorted(per_material.keys()):
        d = per_material[mat]
        trials = d["trials"]
        per_material_rows.append(
            {
                **d,
                "gsam_only_accuracy_percent": round(
                    100.0 * d["gsam_only_correct"] / trials, 2
                )
                if trials
                else "NA",
                "clip_accuracy_percent": round(100.0 * d["clip_correct"] / trials, 2)
                if trials
                else "NA",
            }
        )
    write_csv(per_material_rows, paper_tables_dir / "table_experiment_1_ablation_per_material.csv")

    change_rows = [
        r
        for r in rows
        if r.get("candidate_id_match") == "no"
    ]
    change_cols = []
    for r in change_rows:
        change_cols.append(
            {
                "trial_index": r.get("trial_index"),
                "frame_id": r.get("frame_id"),
                "input_text": r.get("input_text"),
                "classes_existing_in_frame": r.get("classes_existing_in_frame"),
                "gsam_only_candidate_id": r.get("gsam_only_candidate_id"),
                "gsam_only_detection_score": r.get("gsam_only_detection_score"),
                "clip_candidate_id": r.get("clip_candidate_id"),
                "clip_detection_score": r.get("clip_detection_score"),
                "clip_corrected_gsam_only": r.get("clip_corrected_gsam_only"),
                "side_by_side_ablation_path": r.get("side_by_side_ablation_path"),
            }
        )

    write_csv(change_cols, paper_tables_dir / "table_experiment_1_candidate_change_cases.csv")
    write_csv(rows, paper_tables_dir / "table_experiment_1_ablation_per_trial.csv")


def write_paper_figures(rows: List[Dict[str, Any]], paper_figures_dir: Path) -> None:
    if plt is None:
        return

    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)

    # Accuracy bar
    methods = ["GSAM-only", "+CLIP"]
    accuracies = [
        100.0 * summary["gsam_only_accuracy"] if summary["gsam_only_accuracy"] is not None else 0,
        100.0 * summary["clip_accuracy"] if summary["clip_accuracy"] is not None else 0,
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(methods, accuracies)
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment 1 Target-Selection Ablation")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(paper_figures_dir / "figure_experiment_1_ablation_accuracy_bar.png", dpi=300)
    plt.close()

    # Per-material accuracy
    mats = sorted(set(normalize_material_name(r.get("input_text")) for r in rows))
    gsam_vals = []
    clip_vals = []
    for mat in mats:
        mat_rows = [r for r in rows if normalize_material_name(r.get("input_text")) == mat]
        n = len(mat_rows)
        g = sum(1 for r in mat_rows if r.get("gsam_only_correct_auto") == "yes")
        c = sum(1 for r in mat_rows if r.get("clip_manual_correct") == "yes")
        gsam_vals.append(100.0 * g / n if n else 0)
        clip_vals.append(100.0 * c / n if n else 0)

    x = list(range(len(mats)))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], gsam_vals, width, label="GSAM-only")
    plt.bar([i + width / 2 for i in x], clip_vals, width, label="+CLIP")
    plt.xticks(x, mats)
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-material Target-Selection Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        paper_figures_dir / "figure_experiment_1_ablation_per_material_accuracy.png",
        dpi=300,
    )
    plt.close()

    # Candidate change bar
    unchanged = sum(1 for r in rows if r.get("candidate_id_match") == "yes")
    changed = sum(1 for r in rows if r.get("candidate_id_match") == "no")

    plt.figure(figsize=(6, 4))
    plt.bar(["Same candidate", "Changed by +CLIP"], [unchanged, changed])
    plt.ylabel("Number of trials")
    plt.title("Effect of CLIP Crop Verification on Candidate Choice")
    for i, v in enumerate([unchanged, changed]):
        plt.text(i, v + 0.2, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(
        paper_figures_dir / "figure_experiment_1_ablation_candidate_change_bar.png",
        dpi=300,
    )
    plt.close()

    # Copy a few useful example side-by-side figures.
    changed_rows = [r for r in rows if r.get("candidate_id_match") == "no"]
    same_rows = [r for r in rows if r.get("candidate_id_match") == "yes"]

    examples = []
    if changed_rows:
        examples.append(("example_clip_changed_case", changed_rows[0]))
    if len(changed_rows) > 1:
        examples.append(("example_clip_changed_case_2", changed_rows[1]))
    if same_rows:
        examples.append(("example_same_candidate_case", same_rows[0]))

    for name, row in examples:
        src = Path(str(row.get("side_by_side_ablation_path", "")))
        if src.exists():
            shutil.copy2(src, paper_figures_dir / f"{name}.png")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def run_ablation(args: argparse.Namespace) -> None:
    repo_root = Path.cwd()
    source_session = Path(args.source_session).resolve()

    if not source_session.exists():
        raise FileNotFoundError(f"Source session does not exist: {source_session}")

    output_root = Path(args.output_root)
    ablation_session = output_root / f"session_{now_stamp()}"

    cases_out = ablation_session / "cases"
    paper_tables_dir = ablation_session / "paper_tables"
    paper_figures_dir = ablation_session / "paper_figures"

    ablation_session.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": "run",
        "source_session": str(source_session),
        "output_root": str(output_root),
        "ablation_session": str(ablation_session),
        "baseline_method": "GroundingDINO+SAM2 top-detection baseline",
        "baseline_rule": "candidate with highest detection_score in saved candidate pool",
        "proposed_method": "GroundingDINO+SAM2+CLIP crop verification",
        "reference_candidate": "final_selected_candidate_id from manually verified +CLIP pipeline",
        "models_rerun": False,
        "created_at": datetime.now().isoformat(),
    }
    save_json(config, ablation_session / "ablation_config.json")

    case_dirs = find_case_dirs(source_session)
    if not case_dirs:
        raise RuntimeError(f"No case directories found under: {source_session / 'cases'}")

    rows: List[Dict[str, Any]] = []

    for case_dir in case_dirs:
        out_case_dir = cases_out / case_dir.name
        print(f"[INFO] Processing {case_dir.name}")
        row = process_case(
            case_dir=case_dir,
            out_case_dir=out_case_dir,
            source_session=source_session,
            ablation_session=ablation_session,
            repo_root=repo_root,
        )
        rows.append(row)

    write_csv(rows, ablation_session / "ablation_master_results.csv")
    write_paper_tables(rows, paper_tables_dir)
    write_paper_figures(rows, paper_figures_dir)

    summary = summarize_rows(rows)
    summary.update(
        {
            "source_session": str(source_session),
            "ablation_session": str(ablation_session),
            "paper_tables_dir": str(paper_tables_dir),
            "paper_figures_dir": str(paper_figures_dir),
            "completed_at": datetime.now().isoformat(),
        }
    )
    save_json(summary, ablation_session / "ablation_session_summary.json")

    print("\n============================================================")
    print("Experiment 1 ablation complete")
    print("============================================================")
    print(f"Output session: {ablation_session}")
    print(f"Trials: {summary['n_trials']}")
    print(
        f"GSAM-only correct: {summary['gsam_only_correct']}/"
        f"{summary['n_trials']} "
        f"({100.0 * summary['gsam_only_accuracy']:.2f}%)"
    )
    print(
        f"+CLIP correct: {summary['clip_correct']}/"
        f"{summary['n_trials']} "
        f"({100.0 * summary['clip_accuracy']:.2f}%)"
    )
    print(f"+CLIP candidate changes/corrections: {summary['clip_corrections']}")
    print("============================================================")


def summarize_existing(args: argparse.Namespace) -> None:
    ablation_session = Path(args.ablation_session)
    master_csv = ablation_session / "ablation_master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_csv}")

    with master_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary = summarize_rows(rows)
    save_json(summary, ablation_session / "ablation_session_summary_resummarized.json")
    write_paper_tables(rows, ablation_session / "paper_tables")
    write_paper_figures(rows, ablation_session / "paper_figures")

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Experiment 1 target-selection ablation from saved artifacts."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "summarize"],
        default="run",
        help="run = create ablation from source session; summarize = regenerate summaries from ablation session.",
    )
    parser.add_argument(
        "--source-session",
        type=str,
        default="outputs/T5_v2_experiment_1_target_selection_fixed27/session_20260518_142404",
        help="Path to completed T5_v2 Experiment 1 session.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/v2_experiments/experiment_1_target_selection_ablation",
        help="Output root for new ablation session.",
    )
    parser.add_argument(
        "--ablation-session",
        type=str,
        default="",
        help="Existing ablation session to summarize when --mode summarize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_ablation(args)
    elif args.mode == "summarize":
        if not args.ablation_session:
            raise ValueError("--ablation-session is required for --mode summarize")
        summarize_existing(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
