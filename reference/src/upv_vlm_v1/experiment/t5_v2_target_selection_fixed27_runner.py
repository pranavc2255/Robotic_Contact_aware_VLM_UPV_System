from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from upv_vlm_v1.experiment.t5_target_selection_runner import (
    T5Case,
    _candidate_annotations,
    _copy_if_exists,
    _decode_mask,
    _json_safe_result,
    _score,
    _save_depth_visualization,
    resolve_repo_path,
    run_real_perception,
    save_candidate_masks_overlay,
    save_material_verification_panel,
)
from upv_vlm_v1.prompts.prompt_registry import build_experiment1_prompt_manifest
from upv_vlm_orient.integration.crop_verify import (
    DEFAULT_CROP_VERIFIER_MODEL_ID,
    verify_crop_with_vlm,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
_CLIP_ENVIRONMENT_CHECK_CACHE: dict[str, dict[str, Any]] = {}

REALSENSE_ROS2_LAUNCH_COMMAND = """source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py \\
  enable_color:=true \\
  enable_depth:=true \\
  align_depth.enable:=true \\
  enable_sync:=true \\
  enable_rgbd:=false \\
  rgb_camera.color_profile:=640x480x30 \\
  depth_module.depth_profile:=640x480x30"""

MATERIAL_VERIFICATION_TEXT_LABELS = {
    "brick": [
        "a beige fired clay brick",
        "a tan fired clay masonry brick",
        "a red or maroon clay brick",
        "a rectangular fired clay brick with warm tan or orange color",
        "a rough fired clay brick surface",
    ],
    "concrete block": [
        "a grey concrete block",
        "a light grey cement block",
        "a cinder block or concrete masonry unit",
        "a rough pitted cementitious concrete surface",
        "a grey porous CMU block with cement texture",
        "a rectangular grey concrete masonry block",
    ],
    "timber": [
        "a brown wood block",
        "a timber lumber piece with visible grain",
        "a cut wooden construction material",
    ],
}

V2_COLUMNS = [
    "trial_index",
    "frame_id",
    "classes_existing_in_frame",
    "input_text",
    "original_input_text",
    "detection_query_used",
    "expanded_backend_query",
    "prompt_expansion_injection_status",
    "material_color_hint",
    "expanded_query_used_for_detection",
    "expanded_material_description",
    "expected_selected_class",
    "experiment_mode",
    "oracle_known_classes_used",
    "classes_existing_used_as_model_input",
    "active_prompt_stages",
    "final_overlay_label_source",
    "candidate_pool_classes",
    "detection_queries_used",
    "candidate_generation_source",
    "material_verification_method",
    "material_score_brick",
    "material_score_timber",
    "material_score_concrete_block",
    "beige_brick_guard_triggered",
    "no_verified_match",
    "selected_source_detection_query",
    "source_class_prior_used",
    "crop_source_used_for_material_verification",
    "concrete_color_texture_guard_triggered",
    "grey_score",
    "warm_beige_score",
    "final_selected_candidate_id",
    "final_selected_class_manual",
    "correct_manual",
    "failure_type_manual",
    "manual_notes",
    "status",
    "perception_backend_status",
    "target_selected",
    "final_selected_material_backend",
    "final_selected_object_id_backend",
    "grounding_mask_available",
    "candidate_count",
    "crop_verification_prediction",
    "crop_verification_confidence",
    "raw_rgb_path",
    "candidate_masks_overlay_path",
    "candidate_crops_panel_path",
    "selected_mask_overlay_path",
    "case_output_dir",
    "backend_output_dir",
    "total_time_s",
    "camera_capture_time_s",
    "open_vocab_detection_time_s",
    "segmentation_time_s",
    "crop_verification_time_s",
    "final_selection_time_s",
    "visualization_save_time_s",
    "created_at",
    "completed_at",
]

ATTEMPT_COLUMNS = [
    "trial_index",
    "frame_id",
    "input_text",
    "attempt_index",
    "attempt_status",
    "case_attempt_dir",
    "backend_status",
    "perception_outputs_found",
    "discard_reason",
    "created_at",
]


@dataclass
class T5V2Trial:
    trial_index: int
    frame_id: str
    classes_existing_in_frame: str
    input_text: str
    expected_selected_class: str
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "T5V2Trial":
        return cls(
            trial_index=int(row["trial_index"]),
            frame_id=str(row["frame_id"]),
            classes_existing_in_frame=str(row["classes_existing_in_frame"]),
            input_text=str(row["input_text"]),
            expected_selected_class=str(row["expected_selected_class"]),
            enabled=str(row.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "n"},
            notes=str(row.get("notes") or ""),
        )

    def case_folder_name(self) -> str:
        return f"trial_{self.trial_index:03d}_{_safe_component(self.frame_id)}_{_slug(self.input_text)}"

    def to_t5_case(self) -> T5Case:
        return T5Case(
            case_id=f"trial_{self.trial_index:03d}",
            scene_id=self.frame_id,
            layout_id=self.frame_id,
            material_query=self.input_text,
            objects_present=self.classes_existing_in_frame,
            expected_material=self.expected_selected_class,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "frame_id": self.frame_id,
            "classes_existing_in_frame": self.classes_existing_in_frame,
            "input_text": self.input_text,
            "expected_selected_class": self.expected_selected_class,
            "enabled": self.enabled,
            "notes": self.notes,
        }


def now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _slug(text: Any) -> str:
    value = str(text or "NA").strip().lower()
    out = [ch if ch.isalnum() else "_" for ch in value]
    return "_".join("".join(out).split("_")) or "item"


def _safe_component(text: Any) -> str:
    value = str(text or "NA").strip()
    out = [ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value]
    return "_".join("".join(out).split("_")) or "item"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _write_placeholder_image(path: Path, title: str, subtitle: str = "") -> str:
    image = Image.new("RGB", (720, 420), (238, 239, 233))
    draw = ImageDraw.Draw(image)
    draw.text((24, 28), title, fill=(22, 24, 28), font=_font(28, True))
    if subtitle:
        draw.text((24, 78), subtitle, fill=(55, 58, 62), font=_font(18))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def normalize_material_name(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").split())


def parse_existing_classes(text: Any) -> list[str]:
    classes: list[str] = []
    for item in str(text or "").replace(";", ",").split(","):
        normalized = normalize_material_name(item)
        if normalized and normalized not in classes:
            classes.append(normalized)
    return classes


def experiment_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("experiment_mode") or "primary_requested_class_only").strip()
    return mode or "primary_requested_class_only"


def is_oracle_mode(config: dict[str, Any]) -> bool:
    mode = experiment_mode(config)
    pool_cfg = config.get("candidate_pool", {})
    return mode == str(pool_cfg.get("oracle_mode_name", "oracle_known_classes_candidate_pool"))


def resolve_detection_queries_for_trial(trial: T5V2Trial, config: dict[str, Any]) -> dict[str, Any]:
    requested = normalize_material_name(trial.input_text)
    pool_cfg = config.get("candidate_pool", {})
    mode = experiment_mode(config)
    oracle = is_oracle_mode(config)
    allow_oracle = bool(pool_cfg.get("allow_oracle_known_classes", False))
    if oracle:
        if not allow_oracle:
            raise RuntimeError(
                "oracle_known_classes_candidate_pool requested but candidate_pool.allow_oracle_known_classes=false. "
                "This mode uses ground-truth classes_existing_in_frame as model input and is ablation/debug only."
            )
        classes = parse_existing_classes(trial.classes_existing_in_frame)
        detection_queries = classes or [requested]
        if requested not in detection_queries:
            detection_queries.append(requested)
        print(
            "WARNING: oracle_known_classes_candidate_pool uses ground-truth classes_existing_in_frame "
            "as model input. This is for ablation/debug only, not the primary target-selection experiment."
        )
        return {
            "experiment_mode": mode,
            "oracle_known_classes_used": True,
            "classes_existing_used_as_model_input": True,
            "detection_queries": detection_queries,
            "requested_detection_query": requested,
            "candidate_generation_source": "oracle_known_classes_candidate_pool",
        }
    return {
        "experiment_mode": "primary_requested_class_only",
        "oracle_known_classes_used": False,
        "classes_existing_used_as_model_input": False,
        "detection_queries": [requested],
        "requested_detection_query": requested,
        "candidate_generation_source": "requested_class_detection_instances",
    }


def material_color_hint(material_name: str) -> str:
    normalized = normalize_material_name(material_name)
    if normalized == "concrete block":
        return "grey/light-grey cementitious material; avoid red/maroon/beige brick"
    if normalized == "brick":
        return "red/maroon/orange/tan/beige fired clay masonry; avoid grey concrete block"
    if normalized == "timber":
        return "brown/tan wood grain material; avoid brick/concrete"
    return "NA"


def material_prompt_info(config: dict[str, Any], trial: T5V2Trial) -> dict[str, Any]:
    prompt_cfg = config.get("material_prompt_engineering", {})
    enabled = bool(prompt_cfg.get("enabled", False))
    use_expanded_for_detection = bool(prompt_cfg.get("use_expanded_query_for_detection", False))
    allow_long_detection = bool(prompt_cfg.get("allow_long_detection_queries", False))
    normalized = normalize_material_name(trial.input_text)
    aliases = prompt_cfg.get("material_aliases", {}) if isinstance(prompt_cfg.get("material_aliases"), dict) else {}
    material_entry = aliases.get(normalized, {}) if isinstance(aliases.get(normalized, {}), dict) else {}
    expanded_query = material_entry.get("backend_query") or trial.input_text
    detection_query = trial.input_text
    expanded_query_used_for_detection = False
    detection_query_warning = "NA"
    if enabled and use_expanded_for_detection and expanded_query:
        proposed = str(expanded_query)
        if ("," in proposed or len(proposed) > 60) and not allow_long_detection:
            detection_query_warning = "Long expanded detection query blocked; using simple query for Grounded-SAM2."
        else:
            detection_query = proposed
            expanded_query_used_for_detection = detection_query != trial.input_text
    if enabled and expanded_query_used_for_detection:
        injection_status = "expanded_query_used_for_detection"
    elif enabled and material_entry:
        injection_status = "description_logged_only_detection_query_simple"
    else:
        injection_status = "not_used_detection_query_simple"
    positive = material_entry.get("positive_description", "NA")
    negative = material_entry.get("negative_description", "NA")
    expanded_description = " ".join(part for part in [positive, negative] if part and part != "NA") or "NA"
    return {
        "original_input_text": trial.input_text,
        "normalized_requested_class": normalized,
        "detection_query_used": detection_query,
        "expanded_backend_query": expanded_query if enabled else trial.input_text,
        "expanded_aliases": [item.strip() for item in str(expanded_query).split(",") if item.strip()] if enabled else [],
        "positive_description": positive,
        "negative_description": negative,
        "expanded_material_description": expanded_description,
        "expected_selected_class": trial.expected_selected_class,
        "color_semantic_hint": material_color_hint(normalized),
        "prompt_expansion_injection_status": injection_status,
        "material_prompt_engineering_enabled": enabled,
        "expanded_query_used_for_detection": expanded_query_used_for_detection,
        "expanded_description_use": "post-detection verification/logging only" if not expanded_query_used_for_detection else "detection query",
        "detection_query_warning": detection_query_warning,
    }


def print_material_prompt_info(info: dict[str, Any]) -> None:
    print(f"Requested class for paper: {info.get('original_input_text', 'NA')}")
    print(f"Detection query sent to Grounded-SAM2: {info.get('detection_query_used', 'NA')}")
    print(f"Expanded material description: {info.get('color_semantic_hint', 'NA')}")
    print(f"Expanded description use: {info.get('expanded_description_use', 'post-detection verification/logging only')}")
    warning = info.get("detection_query_warning")
    if warning and warning != "NA":
        print(f"WARNING: {warning}")
    hint = info.get("color_semantic_hint") or "NA"
    if hint != "NA":
        print(f"Important distinction: {hint}")


def write_prompt_manifest(
    case_dir: Path,
    config: dict[str, Any],
    material_query: str,
    candidate_ids: list[str] | None = None,
    prompt_injection_status: str = "prompt_logged_only_clip_fixed_labels",
) -> str:
    manifest = build_experiment1_prompt_manifest(
        material_query=material_query,
        candidate_ids=candidate_ids or [],
        prompt_config_path=config.get("canonical_prompt_config", "src/upv_vlm_v1/prompts"),
        prompt_injection_status=prompt_injection_status,
    )
    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, stage in manifest.get("stages", {}).items():
        if stage_name == "open_vocab_detection":
            text = str(stage.get("prompt", ""))
        else:
            text = "\n".join(
                part
                for part in [stage.get("system_prompt", ""), stage.get("user_prompt", "")]
                if part
            ).strip()
        (prompts_dir / f"{stage_name}_prompt.txt").write_text(text + "\n", encoding="utf-8")
        stage["prompt_text_path"] = str(prompts_dir / f"{stage_name}_prompt.txt")
    path = prompts_dir / "prompt_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (case_dir / "prompt_engineering_summary.json").write_text(
        json.dumps(
            {
                "prompt_registry_version": manifest["prompt_registry_version"],
                "active_prompt_stages": manifest["active_prompt_stages"],
                "inactive_prompt_stages": manifest["inactive_prompt_stages"],
                "model_calls_expected": manifest["model_calls_expected"],
                "stages": {
                    name: {
                        "enabled": stage.get("enabled"),
                        "model_role": stage.get("model_role"),
                        "prompt_hash": stage.get("prompt_hash"),
                        "prompt_text_path": stage.get("prompt_text_path"),
                        "prompt_injection_status": stage.get("prompt_injection_status", "NA"),
                        "disabled_reason": stage.get("disabled_reason", "NA"),
                    }
                    for name, stage in manifest.get("stages", {}).items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(path)


def _capture_config(config: dict[str, Any]) -> dict[str, Any]:
    capture = dict(config.get("capture") or {})
    legacy_camera = dict(config.get("camera") or {})
    if not capture:
        capture = legacy_camera
    mode = str(capture.get("mode") or config.get("capture_mode") or "ros2_topics")
    color_topic = capture.get("color_topic") or legacy_camera.get("color_topic") or "/camera/camera/color/image_raw"
    depth_topic = capture.get("depth_topic") or legacy_camera.get("depth_topic") or "/camera/camera/aligned_depth_to_color/image_raw"
    camera_info_topic = capture.get("camera_info_topic") or legacy_camera.get("camera_info_topic") or "/camera/camera/color/camera_info"
    aligned_depth_info_topic = (
        capture.get("aligned_depth_camera_info_topic")
        or legacy_camera.get("aligned_depth_camera_info_topic")
        or "/camera/camera/aligned_depth_to_color/camera_info"
    )
    return {
        "mode": mode,
        "color_topic": color_topic,
        "depth_topic": depth_topic,
        "camera_info_topic": camera_info_topic,
        "aligned_depth_camera_info_topic": aligned_depth_info_topic,
        "timeout_s": float(capture.get("timeout_s", legacy_camera.get("timeout_s", 10.0))),
        "sync_slop_s": float(capture.get("sync_slop_s", 0.08)),
        "queue_size": int(capture.get("queue_size", 10)),
        "save_snapshot": bool(capture.get("save_snapshot", True)),
    }


def check_ros2_topics_available(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ROS2 topic preflight plan used by the synchronized snapshot wait.

    The actual "at least one message" check is performed by
    capture_ros2_rgbd_snapshot(), which waits for a synchronized color/depth/info
    message set and fails before perception starts if the ROS2 camera is not live.
    """

    capture = _capture_config(config)
    return {
        "capture_mode": capture["mode"],
        "color_topic": {"topic": capture["color_topic"], "status": "checked_by_synchronized_snapshot_wait"},
        "aligned_depth_topic": {"topic": capture["depth_topic"], "status": "checked_by_synchronized_snapshot_wait"},
        "camera_info_topic": {"topic": capture["camera_info_topic"], "status": "checked_by_synchronized_snapshot_wait"},
        "aligned_depth_camera_info_topic": {
            "topic": capture["aligned_depth_camera_info_topic"],
            "status": "metadata_only_not_required_for_aligned_depth_snapshot",
        },
        "timeout_s": capture["timeout_s"],
    }


def print_capture_preflight(config: dict[str, Any]) -> None:
    capture = _capture_config(config)
    print("Capture mode: ROS2 topics")
    print(f"Color topic: {capture['color_topic']}")
    print(f"Depth topic: {capture['depth_topic']}")
    print(f"Camera info topic: {capture['camera_info_topic']}")
    print(f"Aligned depth camera info topic: {capture['aligned_depth_camera_info_topic']}")
    print("Backend input: saved RGB image file")
    print("Direct RealSense access: disabled")
    print("ROS2 camera preflight:")
    print(f"color topic: waiting for synchronized message on {capture['color_topic']}")
    print(f"aligned depth topic: waiting for synchronized message on {capture['depth_topic']}")
    print(f"camera info: waiting for synchronized message on {capture['camera_info_topic']}")


def _copy_snapshot_file(source: str | Path | None, destination: Path) -> str:
    if not source:
        return "NA"
    source_path = Path(source)
    if not source_path.exists():
        return "NA"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return str(destination)


def _copy_backend_images(case_dir: Path) -> list[str]:
    backend_root = case_dir / "backend_outputs"
    images_dir = case_dir / "backend_outputs" / "images_all"
    if not backend_root.exists():
        return []
    images_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    suffixes = {".png", ".jpg", ".jpeg"}
    for source in sorted(path for path in backend_root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes):
        relative_name = "_".join(source.relative_to(backend_root).parts)
        destination = images_dir / relative_name
        try:
            if source.resolve() == destination.resolve():
                copied.append(str(destination))
                continue
            shutil.copyfile(source, destination)
            copied.append(str(destination))
        except shutil.SameFileError:
            copied.append(str(destination))
        except Exception:
            continue
    return copied


def _load_selected_mask(mask_path: Any, image_size: tuple[int, int]) -> np.ndarray | None:
    if not mask_path or mask_path == "NA":
        return None
    path = Path(str(mask_path))
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".npy":
            mask = np.load(path)
        else:
            mask = np.asarray(Image.open(path).convert("L"))
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = mask > 0
        width, height = image_size
        if mask_bool.shape != (height, width):
            mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
            mask_bool = np.asarray(mask_img) > 0
        if not np.any(mask_bool):
            return None
        return mask_bool
    except Exception:
        return None


def _resolve_existing_path(path_value: Any, *roots: Path) -> Path | None:
    if not path_value or path_value == "NA":
        return None
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else [root / path for root in roots] + [REPO_ROOT / path]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _candidate_bbox(candidate: dict[str, Any]) -> list[int] | None:
    for value in [
        candidate.get("bbox"),
        candidate.get("bbox_xyxy"),
        (candidate.get("masked_texture_crop_metadata") or {}).get("bbox_xyxy")
        if isinstance(candidate.get("masked_texture_crop_metadata"), dict)
        else None,
    ]:
        if isinstance(value, list) and len(value) == 4:
            try:
                return [int(float(item)) for item in value]
            except Exception:
                continue
    return None


def resolve_selected_candidate_mask(
    candidate: dict[str, Any],
    attempt_dir: Path,
    backend_output_dir: Path | None = None,
) -> Path | None:
    backend_output_dir = backend_output_dir or (attempt_dir / "backend_outputs")
    for key in ["mask_path", "selected_mask_path", "selected_target_mask_path", "selected_candidate_mask_path"]:
        resolved = _resolve_existing_path(candidate.get(key), attempt_dir, backend_output_dir, attempt_dir / "candidate_pool")
        if resolved is not None:
            return resolved
    candidate_id = str(candidate.get("candidate_id", "") or "")
    if candidate_id:
        for root in [attempt_dir / "candidate_pool", backend_output_dir, attempt_dir]:
            if root.exists():
                for path in sorted(root.rglob(f"*{candidate_id}*mask*")):
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}:
                        return path.resolve()
    return None


def _candidate_from_manifest(case_dir: Path, candidate_id: str) -> dict[str, Any]:
    if not candidate_id or candidate_id in {"NA", "NONE", "NO_VERIFIED_MATCH"}:
        return {}
    for manifest_path in [
        case_dir / "candidate_pool" / "candidate_pool_manifest.json",
        case_dir / "material_verification" / "material_scores.json",
    ]:
        payload = read_json(manifest_path)
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        for candidate in candidates:
            if str(candidate.get("candidate_id")) == candidate_id:
                return candidate
    return {}


def _find_selected_mask_path(result: dict[str, Any], case_dir: Path) -> str:
    candidates = [
        result.get("selected_target_mask_path"),
        result.get("selected_candidate_mask_path"),
        result.get("selected_mask_path"),
        result.get("mask_path"),
    ]
    selected_id = str(result.get("selected_candidate_id", "") or "")
    manifest_candidate = _candidate_from_manifest(case_dir, selected_id)
    resolved = resolve_selected_candidate_mask(manifest_candidate, case_dir, case_dir / "backend_outputs") if manifest_candidate else None
    if resolved is not None:
        return str(resolved)
    backend_root = case_dir / "backend_outputs" / "run_one_image"
    if backend_root.exists():
        candidates.extend(sorted(str(path) for path in backend_root.rglob("*selected*mask*.png")))
        candidates.extend(sorted(str(path) for path in backend_root.rglob("*selected*mask*.npy")))
        candidates.extend(sorted(str(path) for path in backend_root.rglob("*mask*.png")))
    for candidate in candidates:
        if candidate and candidate != "NA" and Path(str(candidate)).exists():
            return str(candidate)
    return "NA"


def _find_selected_bbox(result: dict[str, Any], case_dir: Path) -> list[int] | None:
    selected_id = str(result.get("selected_candidate_id", "") or "")
    candidate = _candidate_from_manifest(case_dir, selected_id)
    return _candidate_bbox(candidate) if candidate else None


def create_masked_material_crop(
    raw_rgb_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    mode: str = "mask_texture",
    erode_px: int = 3,
    background: str = "white",
    pad_px: int = 8,
) -> dict[str, Any]:
    rgb_path = Path(raw_rgb_path)
    output = Path(output_path)
    image = Image.open(rgb_path).convert("RGB")
    mask = _load_selected_mask(mask_path, image.size)
    if mask is None:
        return {"success": False, "failure_reason": "mask_not_available", "crop_path": "NA"}
    original_mask = mask.copy()
    if erode_px > 0:
        try:
            import cv2  # noqa: PLC0415

            kernel = np.ones((erode_px, erode_px), np.uint8)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            if np.count_nonzero(eroded) >= 25:
                mask = eroded
        except Exception:
            pass
    ys, xs = np.where(mask)
    if xs.size == 0:
        ys, xs = np.where(original_mask)
        mask = original_mask
    if xs.size == 0:
        return {"success": False, "failure_reason": "empty_mask", "crop_path": "NA"}
    left = max(0, int(xs.min()) - pad_px)
    right = min(image.width, int(xs.max()) + pad_px + 1)
    top = max(0, int(ys.min()) - pad_px)
    bottom = min(image.height, int(ys.max()) + pad_px + 1)
    rgb_crop = np.asarray(image.crop((left, top, right, bottom))).copy()
    mask_crop = mask[top:bottom, left:right]
    bg_color = np.array([255, 255, 255], dtype=np.uint8) if background == "white" else np.array([0, 0, 0], dtype=np.uint8)
    masked = rgb_crop.copy()
    masked[~mask_crop] = bg_color
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked).save(output)
    alpha_path = output.with_name(output.stem + "_alpha.png")
    alpha = np.dstack([rgb_crop, (mask_crop.astype(np.uint8) * 255)])
    Image.fromarray(alpha, mode="RGBA").save(alpha_path)
    inner_path = output.with_name(output.stem.replace("_masked_texture_crop", "_inner_texture_crop") + ".png")
    inner_success = False
    if np.count_nonzero(mask_crop) >= 25:
        ys2, xs2 = np.where(mask_crop)
        li = max(0, int(np.percentile(xs2, 20)) - 2)
        ri = min(masked.shape[1], int(np.percentile(xs2, 80)) + 3)
        ti = max(0, int(np.percentile(ys2, 20)) - 2)
        bi = min(masked.shape[0], int(np.percentile(ys2, 80)) + 3)
        if ri > li and bi > ti:
            inner = masked[ti:bi, li:ri]
            Image.fromarray(inner).save(inner_path)
            inner_success = True
    return {
        "success": True,
        "mode": mode,
        "crop_path": str(output),
        "alpha_crop_path": str(alpha_path),
        "inner_texture_crop_path": str(inner_path) if inner_success else "NA",
        "mask_pixels": int(np.count_nonzero(mask)),
        "bbox_xyxy": [left, top, right, bottom],
        "erode_px": erode_px,
        "background": background,
    }


def create_bbox_crop_from_mask(
    raw_rgb_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    pad_px: int = 18,
) -> str:
    image = Image.open(raw_rgb_path).convert("RGB")
    mask = _load_selected_mask(mask_path, image.size)
    if mask is None:
        return "NA"
    ys, xs = np.where(mask)
    if xs.size == 0:
        return "NA"
    left = max(0, int(xs.min()) - pad_px)
    right = min(image.width, int(xs.max()) + pad_px + 1)
    top = max(0, int(ys.min()) - pad_px)
    bottom = min(image.height, int(ys.max()) + pad_px + 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).save(output)
    return str(output)


def generate_selected_mask_overlay(
    rgb_path: Path,
    result: dict[str, Any],
    output_path: Path,
    trial: T5V2Trial,
    prompt_info: dict[str, Any],
    backend_status: str,
) -> tuple[str, bool, str]:
    if not rgb_path.exists():
        return "NA", False, "raw_rgb_not_found"
    image = Image.open(rgb_path).convert("RGB")
    header_h = 158

    def _compose_with_header(content: Image.Image, selected_mask_available: bool, reason: str) -> tuple[str, bool, str]:
        header = Image.new("RGB", (content.width, header_h), (245, 245, 238))
        draw_header = ImageDraw.Draw(header)
        draw_header.rectangle((0, 0, content.width - 1, header_h - 1), outline=(45, 48, 52), width=1)
        lines = [
            f"trial/frame: {trial.trial_index} / {trial.frame_id}",
            f"mode: {result.get('experiment_mode_label', result.get('experiment_mode', 'primary_requested_class_only'))}",
            f"requested: {prompt_info.get('original_input_text', trial.input_text)}",
            f"detection query: {result.get('requested_detection_query', prompt_info.get('detection_query_used', trial.input_text))}",
            f"selected candidate: {result.get('selected_candidate_id', 'NO_VERIFIED_MATCH')}",
            f"selected source query: {result.get('selected_source_detection_query', 'NA')}",
            f"status: {backend_status}",
        ]
        y = 8
        for idx, line in enumerate(lines):
            draw_header.text((14, y), line[:96], fill=(20, 22, 25), font=_font(16, idx == 0))
            y += 21
        canvas = Image.new("RGB", (content.width, header_h + content.height), (245, 245, 238))
        canvas.paste(header, (0, 0))
        canvas.paste(content.convert("RGB"), (0, header_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return str(output_path), selected_mask_available, reason

    mask_path = _find_selected_mask_path(result, output_path.parent)
    mask = _load_selected_mask(mask_path, image.size)
    if mask is None:
        bbox = _find_selected_bbox(result, output_path.parent)
        if bbox:
            fallback = image.convert("RGBA")
            draw = ImageDraw.Draw(fallback)
            draw.rectangle(tuple(bbox), outline=(255, 210, 0, 255), width=5)
            draw.rectangle((12, 12, min(image.width - 12, 520), 52), fill=(255, 245, 190, 235), outline=(130, 95, 0, 255), width=2)
            draw.text((24, 22), "mask unavailable; bbox fallback used", fill=(35, 35, 20), font=_font(16, True))
            return _compose_with_header(fallback, False, "mask unavailable; bbox fallback used")
        return _compose_with_header(image, False, "selected candidate exists but mask unavailable")

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_u8 = (mask.astype(np.uint8) * 255)
    color_layer = Image.new("RGBA", image.size, (0, 160, 255, 105))
    overlay.paste(color_layer, (0, 0), Image.fromarray(mask_u8, mode="L"))
    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)
    ys, xs = np.where(mask)
    if xs.size:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 230, 0, 255), outline=(20, 20, 20, 255), width=2)
        try:
            import cv2  # noqa: PLC0415

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [(int(p[0][0]), int(p[0][1])) for p in contour]
                if len(points) > 1:
                    draw.line(points + [points[0]], fill=(255, 255, 0, 255), width=3)
            coords = np.column_stack([xs, ys]).astype(np.float32)
            if coords.shape[0] >= 3:
                mean = coords.mean(axis=0)
                centered = coords - mean
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                major = vt[0]
                minor = vt[1]
                scale = max(35.0, min(float(np.ptp(xs)), float(np.ptp(ys))) * 0.35)
                for axis, color in [(major, (255, 64, 64, 255)), (minor, (64, 255, 120, 255))]:
                    x0, y0 = mean - axis * scale
                    x1, y1 = mean + axis * scale
                    draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=3)
        except Exception:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            draw.rectangle(bbox, outline=(255, 255, 0, 255), width=3)

    return _compose_with_header(composed, True, "generated_transparent_mask_overlay_with_header_above_image")


def _copy_candidate_crop(result: dict[str, Any], candidate_dir: Path, candidate_id: str) -> str:
    crop_info = result.get("crop_info") if isinstance(result.get("crop_info"), dict) else {}
    for source in [crop_info.get("crop_path"), result.get("crop_path")]:
        copied = _copy_if_exists(source, candidate_dir / f"{candidate_id}_bbox_crop.png")
        if copied != "NA":
            return copied
    overlay = _copy_if_exists(result.get("overlay_path"), candidate_dir / f"{candidate_id}_bbox_crop.png")
    return overlay


def analyze_crop_color_texture(crop_path: str | Path | None) -> dict[str, Any]:
    if not crop_path or crop_path == "NA" or not Path(str(crop_path)).exists():
        return {
            "available": False,
            "mean_rgb": [0.0, 0.0, 0.0],
            "saturation": 0.0,
            "brightness": 0.0,
            "grey_score": 0.0,
            "warm_color_score": 0.0,
            "warm_beige_score": 0.0,
            "texture_variance": 0.0,
            "edge_variance": 0.0,
            "cement_texture_score": 0.0,
        }
    image = Image.open(crop_path).convert("RGB").resize((96, 96))
    arr = np.asarray(image).astype(np.float32) / 255.0
    # Ignore near-white padding from masked crops where possible.
    nonwhite = np.mean(arr, axis=2) < 0.96
    pixels = arr[nonwhite] if np.any(nonwhite) else arr.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    r, g, b = [float(value) for value in mean]
    maxc = pixels.max(axis=1)
    minc = pixels.min(axis=1)
    sat = float(np.mean((maxc - minc) / np.maximum(maxc, 1e-6)))
    channel_std = float(np.mean(np.std(pixels, axis=1)))
    warmth = float(np.clip((r - b) * 1.7 + (r - g) * 0.7 + sat * 0.15, 0.0, 1.0))
    brightness = float(np.mean(pixels))
    red_dominance = float(np.clip((r - b) * 4.0 + (r - g) * 2.0, 0.0, 1.0))
    cool_grey_bonus = float(np.clip((b - r) * 2.5 + (g - r) * 0.8, 0.0, 0.35))
    gray_channel_balance = float(np.clip(1.0 - channel_std / 0.24, 0.0, 1.0))
    low_saturation = float(np.clip(1.0 - sat / 0.42, 0.0, 1.0))
    grayness = float(np.clip(low_saturation * (1.0 - 0.88 * red_dominance) + cool_grey_bonus, 0.0, 1.0))
    warm_brightness = float(np.clip(1.0 - abs(brightness - 0.62) / 0.46, 0.0, 1.0))
    warm_beige = float(np.clip(red_dominance * warm_brightness * (1.0 - max(0.0, sat - 0.62) * 0.8), 0.0, 1.0))
    gray = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    valid_gray = gray[nonwhite] if np.any(nonwhite) else gray.reshape(-1)
    texture_variance = float(np.clip(np.std(valid_gray) / 0.22, 0.0, 1.0))
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    edge_variance = float(np.clip((np.std(dx) + np.std(dy)) / 0.26, 0.0, 1.0))
    cement_texture = float(np.clip(0.65 * grayness + 0.20 * low_saturation + 0.15 * max(texture_variance, edge_variance), 0.0, 1.0))
    return {
        "available": True,
        "mean_rgb": [round(r, 4), round(g, 4), round(b, 4)],
        "saturation": round(sat, 4),
        "brightness": round(brightness, 4),
        "grey_score": round(grayness, 4),
        "red_dominance_score": round(red_dominance, 4),
        "cool_grey_bonus": round(cool_grey_bonus, 4),
        "warm_color_score": round(warmth, 4),
        "warm_beige_score": round(warm_beige, 4),
        "texture_variance": round(texture_variance, 4),
        "edge_variance": round(edge_variance, 4),
        "cement_texture_score": round(cement_texture, 4),
    }


def _heuristic_prompt_scores(features: dict[str, Any]) -> dict[str, list[dict[str, float | str]]]:
    grey = float(features.get("grey_score", 0.0))
    warm = float(features.get("warm_color_score", 0.0))
    warm_beige = float(features.get("warm_beige_score", 0.0))
    sat = float(features.get("saturation", 0.0))
    cement = float(features.get("cement_texture_score", 0.0))
    texture = float(max(float(features.get("texture_variance", 0.0)), float(features.get("edge_variance", 0.0))))
    brightness = float(features.get("brightness", 0.0))
    brownness = float(np.clip(warm * 0.75 + sat * 0.2 + (0.65 - abs(brightness - 0.45)) * 0.1, 0.0, 1.0))
    prompt_scores = {
        "brick": [
            0.20 + 0.70 * warm_beige,
            0.22 + 0.60 * warm_beige + 0.12 * texture,
            0.18 + 0.70 * warm + 0.10 * sat,
            0.24 + 0.55 * warm + 0.12 * warm_beige,
            0.18 + 0.40 * warm + 0.30 * texture,
        ],
        "concrete block": [
            0.18 + 0.68 * grey + 0.10 * cement - 0.22 * warm_beige,
            0.20 + 0.58 * grey + 0.22 * cement - 0.22 * warm_beige,
            0.18 + 0.50 * grey + 0.30 * cement - 0.20 * warm,
            0.16 + 0.58 * cement + 0.18 * texture - 0.24 * warm_beige,
            0.18 + 0.62 * grey + 0.16 * texture - 0.26 * warm_beige,
            0.18 + 0.60 * grey + 0.18 * cement - 0.22 * warm_beige,
        ],
        "timber": [
            0.16 + 0.58 * brownness + 0.12 * texture,
            0.16 + 0.46 * brownness + 0.22 * texture,
            0.14 + 0.42 * brownness + 0.18 * warm + 0.10 * texture,
        ],
    }
    output: dict[str, list[dict[str, float | str]]] = {}
    for material, labels in MATERIAL_VERIFICATION_TEXT_LABELS.items():
        values = prompt_scores.get(material, [])
        output[material] = [
            {"label": label, "score": round(float(np.clip(values[idx], 0.0, 1.0)), 4)}
            for idx, label in enumerate(labels)
        ]
    return output


def _aggregate_prompt_scores(
    per_prompt_scores: dict[str, list[dict[str, float | str]]],
    aggregation: str = "max",
) -> dict[str, float]:
    output: dict[str, float] = {}
    for material, rows in per_prompt_scores.items():
        scores = [float(row.get("score", 0.0)) for row in rows]
        if not scores:
            output[material] = 0.0
        elif aggregation == "mean":
            output[material] = round(float(np.mean(scores)), 4)
        else:
            output[material] = round(float(max(scores)), 4)
    return output


def score_crop_material(crop_path: str | Path | None, *, aggregation: str = "max") -> dict[str, float]:
    features = analyze_crop_color_texture(crop_path)
    if not features.get("available"):
        return {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}
    prompt_scores = _heuristic_prompt_scores(features)
    scores = _aggregate_prompt_scores(prompt_scores, aggregation)
    r, g, b = [float(value) for value in features.get("mean_rgb", [0.0, 0.0, 0.0])]
    sat = float(features.get("saturation", 0.0))
    warmth = float(features.get("warm_color_score", 0.0))
    brightness = float(features.get("brightness", 0.0))
    brownness = float(np.clip((r - b) * 1.2 + (g - b) * 0.5 - abs(r - g) * 0.25, 0.0, 1.0))
    return {
        "brick": round(float(np.clip(max(scores.get("brick", 0.0), 0.18 + 0.48 * warmth + 0.22 * sat), 0.0, 1.0)), 4),
        "timber": round(float(np.clip(max(scores.get("timber", 0.0), 0.14 + 0.40 * brownness + 0.28 * warmth), 0.0, 1.0)), 4),
        "concrete block": round(float(np.clip(scores.get("concrete block", 0.0), 0.0, 1.0)), 4),
    }


def _clip_prompt_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for material, prompts in MATERIAL_VERIFICATION_TEXT_LABELS.items():
        for prompt in prompts:
            rows.append({"material": material, "prompt": prompt})
    return rows


def _aggregate_clip_results(
    prompt_rows: list[dict[str, str]],
    labels: list[str],
    scores: list[float],
    aggregation: str,
) -> tuple[dict[str, float], dict[str, list[dict[str, float | str]]]]:
    score_by_label = {str(label): float(score) for label, score in zip(labels, scores)}
    per_prompt: dict[str, list[dict[str, float | str]]] = {material: [] for material in MATERIAL_VERIFICATION_TEXT_LABELS}
    for row in prompt_rows:
        material = row["material"]
        prompt = row["prompt"]
        per_prompt.setdefault(material, []).append(
            {"label": prompt, "score": round(float(score_by_label.get(prompt, 0.0)), 6)}
        )
    return _aggregate_prompt_scores(per_prompt, aggregation), per_prompt


def resolve_clip_subprocess_python(config: dict[str, Any]) -> dict[str, Any]:
    verification_cfg = config.get("material_verification", {})
    configured = str(
        verification_cfg.get(
            "clip_subprocess_python",
            verification_cfg.get("clip_python_executable", "auto"),
        )
        or "auto"
    )
    parent = sys.executable
    if configured.strip().lower() == "auto":
        resolved = parent
        explicit_path_exists = True
    else:
        path = resolve_repo_path(configured)
        explicit_path_exists = path.exists()
        resolved = str(path) if explicit_path_exists else str(path)
    return {
        "parent_sys_executable": parent,
        "configured_clip_subprocess_python": configured,
        "clip_subprocess_python": resolved,
        "explicit_path_exists": explicit_path_exists,
        "same_as_parent_python": str(Path(resolved).resolve()) == str(Path(parent).resolve()) if Path(resolved).exists() else resolved == parent,
    }


def check_clip_subprocess_environment(
    python_executable: str,
    output_dir: str | Path | None = None,
    *,
    mock_missing_module: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir) / "clip_environment_check.json" if output_dir else None
    if mock_missing_module:
        payload = {
            "ok": False,
            "status": "clip_environment_failed",
            "python_executable": str(python_executable),
            "missing_module": str(mock_missing_module),
            "stdout": "",
            "stderr": f"ModuleNotFoundError: No module named '{mock_missing_module}'",
            "returncode": 1,
            "error": f"missing module: {mock_missing_module}",
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, payload)
        return payload
    command = [
        str(python_executable),
        "-c",
        "import transformers; import PIL; import torch; print('OK')",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        ok = completed.returncode == 0 and "OK" in completed.stdout
        missing = "NA"
        stderr = completed.stderr or ""
        if not ok and "No module named" in stderr:
            missing = stderr.split("No module named", 1)[1].strip().strip("'\"")
        payload = {
            "ok": ok,
            "status": "ok" if ok else "clip_environment_failed",
            "python_executable": str(python_executable),
            "command": command,
            "missing_module": missing,
            "stdout": completed.stdout,
            "stderr": stderr,
            "returncode": completed.returncode,
            "error": "NA" if ok else stderr.strip() or "CLIP dependency import check failed",
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "clip_environment_failed",
            "python_executable": str(python_executable),
            "command": command,
            "missing_module": "unknown",
            "stdout": "",
            "stderr": "",
            "returncode": "NA",
            "error": str(exc),
        }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
    return payload


def run_clip_crop_verifier_for_candidate(
    crop_path: str | Path | None,
    config: dict[str, Any],
    *,
    aggregation: str = "max",
    override_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_rows = _clip_prompt_rows()
    labels = [row["prompt"] for row in prompt_rows]
    python_info = resolve_clip_subprocess_python(config)
    env_check = config.get("_clip_environment_check")
    if not isinstance(env_check, dict):
        cache_key = python_info["clip_subprocess_python"]
        env_check = _CLIP_ENVIRONMENT_CHECK_CACHE.get(cache_key)
        if env_check is None and not bool(config.get("_dry_run", False)):
            env_check = check_clip_subprocess_environment(cache_key)
            _CLIP_ENVIRONMENT_CHECK_CACHE[cache_key] = env_check
    if isinstance(env_check, dict) and env_check.get("ok") is False:
        return {
            "ok": False,
            "score_source": "clip_environment_failed",
            **python_info,
            "clip_environment_check": env_check,
            "clip_labels": labels,
            "clip_scores": {},
            "clip_per_prompt_scores": {material: [] for material in MATERIAL_VERIFICATION_TEXT_LABELS},
            "clip_aggregate_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0},
            "clip_top_label": "NONE",
            "clip_top_score": 0.0,
            "clip_error": env_check.get("error", "CLIP subprocess environment failed"),
        }
    if not crop_path or crop_path == "NA" or not Path(str(crop_path)).exists():
        return {
            "ok": False,
            "score_source": "clip_crop_verifier_missing_crop",
            **python_info,
            "clip_environment_check": env_check if isinstance(env_check, dict) else {},
            "clip_labels": labels,
            "clip_scores": {},
            "clip_per_prompt_scores": {material: [] for material in MATERIAL_VERIFICATION_TEXT_LABELS},
            "clip_aggregate_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0},
            "clip_top_label": "NA",
            "clip_top_score": 0.0,
            "clip_error": "missing_crop",
        }

    verification_cfg = config.get("material_verification", {})
    if bool(config.get("_dry_run", False)):
        if isinstance(override_scores, dict):
            aggregate = {normalize_material_name(key): float(value) for key, value in override_scores.items()}
            per_prompt = {
                material: [
                    {"label": prompt, "score": round(float(aggregate.get(material, 0.0)), 6)}
                    for prompt in prompts
                ]
                for material, prompts in MATERIAL_VERIFICATION_TEXT_LABELS.items()
            }
        else:
            per_prompt = _heuristic_prompt_scores(analyze_crop_color_texture(crop_path))
            aggregate = _aggregate_prompt_scores(per_prompt, aggregation)
        top_material = max(aggregate, key=lambda key: float(aggregate.get(key, 0.0))) if aggregate else "NA"
        return {
            "ok": True,
            "score_source": "dry_run_mock_clip_crop_verifier",
            **python_info,
            "clip_environment_check": {
                "ok": True,
                "status": "dry_run_skipped",
                "python_executable": python_info["clip_subprocess_python"],
            },
            "clip_labels": labels,
            "clip_scores": aggregate,
            "clip_per_prompt_scores": per_prompt,
            "clip_aggregate_scores": aggregate,
            "clip_top_label": top_material,
            "clip_top_score": float(aggregate.get(top_material, 0.0)) if top_material != "NA" else 0.0,
            "clip_error": "NA",
        }

    try:
        python_executable = python_info["clip_subprocess_python"]
        model_id = str(verification_cfg.get("clip_model_id", DEFAULT_CROP_VERIFIER_MODEL_ID))
        model_name = verification_cfg.get("clip_model_name")
        pretrained = verification_cfg.get("clip_pretrained")
        payload = verify_crop_with_vlm(
            python_executable=python_executable,
            image_path=str(crop_path),
            candidate_labels=labels,
            model_id=model_id,
            model_name=str(model_name) if model_name else None,
            pretrained=str(pretrained) if pretrained else None,
        )
        aggregate, per_prompt = _aggregate_clip_results(
            prompt_rows,
            [str(label) for label in payload.get("labels", [])],
            [float(score) for score in payload.get("scores", [])],
            aggregation,
        )
        top_material = max(aggregate, key=lambda key: float(aggregate.get(key, 0.0))) if aggregate else "NA"
        return {
            "ok": True,
            "score_source": "clip_crop_verifier",
            **python_info,
            "clip_environment_check": env_check if isinstance(env_check, dict) else {},
            "clip_model_id": payload.get("model_id", model_id),
            "clip_model_name": payload.get("model_name", model_name or model_id),
            "clip_pretrained": payload.get("pretrained", pretrained),
            "clip_labels": labels,
            "clip_scores": aggregate,
            "clip_per_prompt_scores": per_prompt,
            "clip_aggregate_scores": aggregate,
            "clip_top_label": top_material,
            "clip_top_score": float(aggregate.get(top_material, 0.0)) if top_material != "NA" else 0.0,
            "clip_runtime_sec": payload.get("runtime_sec", "NA"),
            "clip_error": "NA",
        }
    except Exception as exc:
        return {
            "ok": False,
            "score_source": "clip_crop_verifier_failed",
            **python_info,
            "clip_environment_check": env_check if isinstance(env_check, dict) else {},
            "clip_labels": labels,
            "clip_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0},
            "clip_per_prompt_scores": {material: [] for material in MATERIAL_VERIFICATION_TEXT_LABELS},
            "clip_aggregate_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0},
            "clip_top_label": "NA",
            "clip_top_score": 0.0,
            "clip_error": str(exc),
        }


def save_material_vlm_decision_input(
    candidates: list[dict[str, Any]],
    output_path: Path,
    requested_class: str,
) -> str:
    tiles: list[Image.Image] = []
    for idx, candidate in enumerate(candidates, start=1):
        neutral_id = f"candidate_{idx:03d}"
        candidate["vlm_panel_label"] = f"C{idx}"
        candidate["vlm_neutral_candidate_id"] = neutral_id
        crop_path = (
            candidate.get("crop_path_used_for_scoring")
            or candidate.get("inner_texture_crop_path")
            or candidate.get("masked_texture_crop_path")
            or candidate.get("bbox_crop_path")
        )
        tile = Image.new("RGB", (330, 300), (244, 245, 240))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 329, 74), fill=(32, 35, 39))
        draw.text((10, 8), f"C{idx}", fill=(255, 255, 255), font=_font(18, True))
        draw.text((10, 34), f"candidate_id: {neutral_id}", fill=(235, 235, 235), font=_font(12))
        draw.text((10, 54), "source: requested-class detection", fill=(235, 235, 235), font=_font(11))
        try:
            if not crop_path or crop_path == "NA":
                raise FileNotFoundError("NA")
            crop = Image.open(str(crop_path)).convert("RGB")
            crop.thumbnail((292, 180))
            tile.paste(crop, ((330 - crop.width) // 2, 88 + (180 - crop.height) // 2))
        except Exception:
            draw.rectangle((28, 98, 302, 248), outline=(130, 130, 130), width=2)
            draw.text((70, 156), "No valid crop", fill=(70, 70, 70), font=_font(16, True))
        draw.text((10, 276), f"crop: {candidate.get('crop_source_used', 'NA')}", fill=(25, 28, 32), font=_font(11))
        tiles.append(tile)
    if not tiles:
        return "NA"
    header_h = 106
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * 330, header_h + rows * 300), (232, 234, 228))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 8), "VLM material crop verification", fill=(20, 22, 25), font=_font(20, True))
    draw.text((14, 34), f"Requested material: {requested_class}", fill=(20, 22, 25), font=_font(15, True))
    draw.text((14, 56), "Task: choose exactly one crop: C1, C2, ..., or NO_VERIFIED_MATCH.", fill=(20, 22, 25), font=_font(13))
    draw.text((14, 75), "Concrete block = grey/light-grey cementitious/CMU/cinder-block-like material.", fill=(20, 22, 25), font=_font(13))
    draw.text((14, 92), "Do NOT choose beige/tan/cream/orange/red/maroon fired-clay brick as concrete block.", fill=(20, 22, 25), font=_font(13))
    for idx, tile in enumerate(tiles):
        panel.paste(tile, ((idx % cols) * 330, header_h + (idx // cols) * 300))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return str(output_path)


def build_material_vlm_prompt(trial: T5V2Trial, candidates: list[dict[str, Any]]) -> str:
    candidate_lines = "\n".join(
        f"- C{idx} = candidate_{idx:03d}; source=requested-class detection; crop={candidate.get('crop_source_used', 'inner_texture_crop')}"
        for idx, candidate in enumerate(candidates, start=1)
    )
    return (
        "You are the material crop verification VLM for UPV_VLM Experiment 1.\n"
        "Use only the candidate crop image panel and the requested material. Do not use ground-truth scene classes.\n\n"
        f"Requested material: {normalize_material_name(trial.input_text)}\n"
        f"Candidate IDs:\n{candidate_lines}\n\n"
        "Concrete block positive examples:\n"
        "- grey, light-grey, off-white-grey, cement-colored\n"
        "- cementitious / concrete / CMU / cinder block / masonry block\n"
        "- rough, pitted, porous, aggregate-like, cement-grain texture, grey cast concrete surface\n\n"
        "Concrete block negative examples:\n"
        "- beige, tan, cream, orange, red, maroon fired-clay brick is NOT concrete block\n"
        "- warm-colored clay/fired brick texture is NOT concrete\n"
        "- wood/timber/brown grain/fibrous texture is NOT concrete\n"
        "- do not select a candidate only because it is rectangular\n"
        "- do not select beige brick even if it looks pale or dusty\n\n"
        "Decision rule for brick:\n"
        "- Positive: red/maroon/orange/tan/beige/cream fired-clay masonry unit; warm clay color; fired-clay surface; brick texture.\n"
        "- Negative: grey cement/CMU/cinder block is NOT brick; brown fibrous wood/timber is NOT brick.\n\n"
        "Decision rule for timber:\n"
        "- Positive: wood, lumber, timber, grain, fibers, brown/tan wood texture.\n"
        "- Negative: red/maroon/beige fired-clay brick is NOT timber; grey cement/concrete/CMU is NOT timber.\n\n"
        "Compare all candidate crops against each other. Select the best match for the requested material. "
        "If none match, return NO_VERIFIED_MATCH. Do not return NA.\n\n"
        "Return only JSON with this schema:\n"
        "{\n"
        '  "selected_candidate_id": "candidate_001 or candidate_002 or NO_VERIFIED_MATCH",\n'
        '  "selected_panel_label": "C1 or C2 or NO_VERIFIED_MATCH",\n'
        '  "predicted_material": "brick|timber|concrete block|unknown",\n'
        '  "confidence": "low|medium|high",\n'
        '  "reason": "short reason mentioning color and texture",\n'
        '  "candidate_assessments": {\n'
        '    "candidate_001": {"predicted_material": "brick|timber|concrete block|unknown", "color_evidence": "short phrase", "texture_evidence": "short phrase", "matches_requested": true}\n'
        "  }\n"
        "}\n"
    )


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib_request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
    return {}


def parse_material_vlm_selection(raw_or_parsed: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = raw_or_parsed if isinstance(raw_or_parsed, dict) else _extract_first_json_object(str(raw_or_parsed or ""))
    candidate_ids = [str(c.get("candidate_id", "")) for c in candidates]
    neutral_to_real = {f"candidate_{idx:03d}": candidate_id for idx, candidate_id in enumerate(candidate_ids, start=1)}
    panel_to_real = {f"C{idx}": candidate_id for idx, candidate_id in enumerate(candidate_ids, start=1)}
    text = json.dumps(parsed) if parsed else str(raw_or_parsed or "")
    upper_text = text.upper()
    if "NO_VERIFIED_MATCH" in upper_text:
        return {"parse_ok": True, "selection_state": "NO_VERIFIED_MATCH", "selected_candidate_id": "NO_VERIFIED_MATCH", "parsed_json": parsed}
    values = [
        parsed.get("selected_candidate_id") if isinstance(parsed, dict) else None,
        parsed.get("selected_panel_label") if isinstance(parsed, dict) else None,
        parsed.get("selected") if isinstance(parsed, dict) else None,
        parsed.get("answer") if isinstance(parsed, dict) else None,
    ]
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        if token in candidate_ids:
            return {"parse_ok": True, "selection_state": "SELECTED", "selected_candidate_id": token, "parsed_json": parsed}
        if token in neutral_to_real:
            return {"parse_ok": True, "selection_state": "SELECTED", "selected_candidate_id": neutral_to_real[token], "parsed_json": parsed}
        token_up = token.upper().replace(" ", "")
        if token_up in panel_to_real:
            return {"parse_ok": True, "selection_state": "SELECTED", "selected_candidate_id": panel_to_real[token_up], "parsed_json": parsed}
        if token_up.isdigit():
            key = f"C{int(token_up)}"
            if key in panel_to_real:
                return {"parse_ok": True, "selection_state": "SELECTED", "selected_candidate_id": panel_to_real[key], "parsed_json": parsed}
    for idx, candidate_id in enumerate(candidate_ids, start=1):
        if f"C{idx}" in upper_text or f"CANDIDATE_{idx:03d}".upper() in upper_text or f" {idx} " in f" {upper_text} ":
            return {"parse_ok": True, "selection_state": "SELECTED", "selected_candidate_id": candidate_id, "parsed_json": parsed}
    return {
        "parse_ok": False,
        "selection_state": "VLM_PARSE_FAILED",
        "selected_candidate_id": "VLM_PARSE_FAILED",
        "parsed_json": parsed,
        "parse_error": "Could not map VLM response to candidate_001/C1, candidate_002/C2, or NO_VERIFIED_MATCH.",
    }


def call_material_verification_vlm(
    trial: T5V2Trial,
    candidates: list[dict[str, Any]],
    material_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    requested = normalize_material_name(trial.input_text)
    prompt_text = build_material_vlm_prompt(trial, candidates)
    prompt_path = material_dir / "material_vlm_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    input_image = save_material_vlm_decision_input(
        candidates,
        material_dir / "vlm_material_decision_input.png",
        requested,
    )
    cfg = config.get("material_verification", {})
    if bool(config.get("_dry_run", False)) and bool(cfg.get("dry_run_mock_vlm_decision", True)):
        chosen = "NO_VERIFIED_MATCH"
        if requested == "concrete block":
            eligible = [
                item for item in candidates
                if float(item.get("grey_score", 0.0)) >= float(cfg.get("concrete_min_grey_score", 0.45))
                and float(item.get("warm_beige_score", 0.0)) <= float(cfg.get("concrete_max_warm_beige_score", 0.35))
            ]
            if eligible:
                chosen = max(eligible, key=lambda item: (float(item.get("grey_score", 0.0)), float(item.get("concrete_color_gate_score", 0.0)))).get("candidate_id", "NO_VERIFIED_MATCH")
        elif candidates:
            chosen = max(candidates, key=lambda item: float(item.get("material_scores", {}).get(requested, 0.0))).get("candidate_id", "NO_VERIFIED_MATCH")
        parsed = {
            "selected_candidate_id": chosen,
            "predicted_material": requested if chosen != "NO_VERIFIED_MATCH" else "unknown",
            "confidence": "high" if chosen != "NO_VERIFIED_MATCH" else "low",
            "reason": "dry-run mock VLM applied the requested material crop prompt",
            "candidate_scores": {item.get("candidate_id", f"candidate_{idx:03d}"): item.get("material_scores", {}) for idx, item in enumerate(candidates, start=1)},
        }
        payload = {
            "vlm_decision_attempted": True,
            "vlm_decision_completed": True,
            "vlm_decision_source": "dry_run_mock_vlm",
            "vlm_input_image_path": input_image,
            "vlm_prompt_path": str(prompt_path),
            "vlm_response_path": str(material_dir / "material_vlm_response.json"),
            "parsed_json": parsed,
            "parse_result": parse_material_vlm_selection(parsed, candidates),
            "raw_response": json.dumps(parsed),
            "error": None,
        }
        write_json(material_dir / "material_vlm_response.json", payload)
        write_json(material_dir / "material_vlm_parse_result.json", payload["parse_result"])
        (material_dir / "material_vlm_response_raw.txt").write_text(payload["raw_response"] + "\n", encoding="utf-8")
        return payload

    server_url = str(cfg.get("vlm_server_url", "http://127.0.0.1:8008")).rstrip("/")
    output_path = material_dir / "material_vlm_response.json"
    payload = {
        "image_path": input_image,
        "prompt_version": "t5_v2_material_crop_verification",
        "prompt_text": prompt_text,
        "max_new_tokens": int(cfg.get("vlm_max_new_tokens", 700)),
        "temperature": float(cfg.get("vlm_temperature", 0.0)),
        "output_path": str(output_path),
    }
    try:
        response = _post_json(f"{server_url}/infer", payload, timeout_s=float(cfg.get("vlm_timeout_s", 180.0)))
        response_payload = {
            "vlm_decision_attempted": True,
            "vlm_decision_completed": bool(response.get("ok") and response.get("parse_ok")),
            "vlm_decision_source": "qwen_vlm_server",
            "vlm_server_url": server_url,
            "vlm_input_image_path": input_image,
            "vlm_prompt_path": str(prompt_path),
            "vlm_response_path": str(output_path),
            "parsed_json": response.get("parsed_json") or {},
            "raw_response": response.get("raw_response"),
            "server_response": response,
            "error": response.get("error") or response.get("parse_error"),
        }
        response_payload["parse_result"] = parse_material_vlm_selection(
            response_payload["parsed_json"] or response_payload["raw_response"],
            candidates,
        )
    except Exception as exc:
        response_payload = {
            "vlm_decision_attempted": True,
            "vlm_decision_completed": False,
            "vlm_decision_source": "qwen_vlm_server",
            "vlm_server_url": server_url,
            "vlm_input_image_path": input_image,
            "vlm_prompt_path": str(prompt_path),
            "vlm_response_path": str(output_path),
            "parsed_json": {},
            "raw_response": None,
            "server_response": {},
            "error": str(exc),
        }
        response_payload["parse_result"] = {
            "parse_ok": False,
            "selection_state": "VLM_SERVER_UNAVAILABLE",
            "selected_candidate_id": "VLM_SERVER_UNAVAILABLE",
            "parse_error": str(exc),
        }
    write_json(output_path, response_payload)
    write_json(material_dir / "material_vlm_parse_result.json", response_payload["parse_result"])
    (material_dir / "material_vlm_response_raw.txt").write_text(str(response_payload.get("raw_response") or response_payload.get("error") or "") + "\n", encoding="utf-8")
    return response_payload


def build_material_verification(
    trial: T5V2Trial,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    case_dir: Path | None = None,
) -> dict[str, Any]:
    requested = normalize_material_name(trial.input_text)
    verification_cfg = config.get("material_verification", {})
    decision_backend = str(verification_cfg.get("decision_backend", "clip_crop_verifier")).strip().lower()
    official_selector = str(verification_cfg.get("official_selector", decision_backend)).strip().lower()
    require_vlm_decision = bool(verification_cfg.get("require_vlm_decision", False))
    qwen_enabled = bool(verification_cfg.get("qwen_vlm_server_enabled", False))
    min_margin = float(verification_cfg.get("min_confidence_margin", 0.10))
    allow_no_match = bool(verification_cfg.get("allow_no_verified_match", True))
    beige_guard_enabled = bool(verification_cfg.get("beige_brick_guard_enabled", True))
    min_accept_score = float(verification_cfg.get("min_accept_score", 0.45))
    source_bonus = float(verification_cfg.get("source_class_prior_bonus", 0.15))
    override_margin = float(verification_cfg.get("cross_source_override_margin", 0.20))
    same_source_preferred = bool(verification_cfg.get("same_source_preferred", True))
    configured_crop_source = str(verification_cfg.get("crop_source", "masked_texture_crop"))
    use_inner = bool(verification_cfg.get("use_inner_texture_crop", True))
    min_inner_pixels = int(verification_cfg.get("min_inner_crop_pixels", 1000))
    class_score_aggregation = str(verification_cfg.get("class_score_aggregation", "max")).strip().lower() or "max"
    use_multi_prompt_labels = bool(verification_cfg.get("use_multi_prompt_clip_labels", True))
    concrete_guard_enabled = bool(verification_cfg.get("concrete_requires_grey_texture_guard", True))
    beige_negative_guard = bool(verification_cfg.get("beige_brick_negative_guard", True))
    concrete_min_grey_score = float(verification_cfg.get("concrete_min_grey_score", 0.45))
    concrete_max_warm_beige_score = float(verification_cfg.get("concrete_max_warm_beige_score", 0.35))
    concrete_grey_preference_margin = float(verification_cfg.get("concrete_grey_preference_margin", 0.08))
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        crop_used = candidate.get("masked_texture_crop_path") or candidate.get("crop_path")
        crop_source_used = "masked_texture_crop" if candidate.get("masked_texture_crop_path") else "bbox_crop"
        inner = candidate.get("inner_texture_crop_path")
        if use_inner and inner and inner != "NA" and Path(str(inner)).exists():
            try:
                inner_pixels = int(np.prod(Image.open(inner).size))
            except Exception:
                inner_pixels = 0
            if inner_pixels >= min_inner_pixels:
                crop_used = inner
                crop_source_used = "inner_texture_crop"
        if not crop_used or crop_used == "NA" or not Path(str(crop_used)).exists():
            crop_used = candidate.get("bbox_crop_path") or candidate.get("crop_path")
            crop_source_used = "bbox_crop"
        features = analyze_crop_color_texture(crop_used)
        per_prompt_scores = _heuristic_prompt_scores(features)
        heuristic_scores = _aggregate_prompt_scores(per_prompt_scores, class_score_aggregation)
        override_scores = candidate.get("material_scores_override")
        clip_result: dict[str, Any] = {}
        if decision_backend == "clip_crop_verifier":
            clip_result = run_clip_crop_verifier_for_candidate(
                crop_used,
                config,
                aggregation=class_score_aggregation,
                override_scores=override_scores if isinstance(override_scores, dict) else None,
            )
            scores_base = dict(clip_result.get("clip_aggregate_scores") or clip_result.get("clip_scores") or {})
        else:
            scores_base = (
                {normalize_material_name(key): float(value) for key, value in override_scores.items()}
                if isinstance(override_scores, dict)
                else heuristic_scores
            )
        if not scores_base:
            scores_base = {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}
        scores = dict(scores_base)
        beige_penalty_applied = False
        concrete_guard_candidate_reason = "NA"
        if requested == "concrete block" and decision_backend != "clip_crop_verifier":
            grey_score = float(features.get("grey_score", 0.0))
            warm_beige_score = float(features.get("warm_beige_score", 0.0))
            cement_score = float(features.get("cement_texture_score", 0.0))
            if beige_negative_guard and warm_beige_score > concrete_max_warm_beige_score and grey_score < concrete_min_grey_score:
                penalty = min(0.42, 0.18 + 0.35 * (warm_beige_score - concrete_max_warm_beige_score))
                scores["concrete block"] = round(float(max(0.0, scores.get("concrete block", 0.0) - penalty)), 4)
                scores["brick"] = round(float(min(1.0, max(scores.get("brick", 0.0), 0.42 + 0.40 * warm_beige_score))), 4)
                beige_penalty_applied = True
                concrete_guard_candidate_reason = "warm beige/tan crop with weak grey cement evidence"
            elif concrete_guard_enabled and cement_score < concrete_min_grey_score and warm_beige_score > grey_score + 0.18:
                penalty = min(0.30, 0.12 + 0.22 * (warm_beige_score - grey_score))
                scores["concrete block"] = round(float(max(0.0, scores.get("concrete block", 0.0) - penalty)), 4)
                beige_penalty_applied = True
                concrete_guard_candidate_reason = "concrete score reduced because color/texture evidence is warmer than grey/cementitious"
        requested_score = scores.get(requested, 0.0)
        other_scores = [score for label, score in scores.items() if label != requested]
        margin = requested_score - max(other_scores or [0.0])
        source_match = normalize_material_name(candidate.get("source_detection_query")) == requested
        final_score = requested_score + (source_bonus if source_match else 0.0)
        scored.append({
            **candidate,
            "material_scores": scores,
            "base_material_scores": scores_base,
            "material_prompt_labels": MATERIAL_VERIFICATION_TEXT_LABELS,
            "per_prompt_scores": (
                clip_result.get("clip_per_prompt_scores", {})
                if decision_backend == "clip_crop_verifier"
                else per_prompt_scores if use_multi_prompt_labels else {}
            ),
            "clip_labels": clip_result.get("clip_labels", []),
            "clip_scores": clip_result.get("clip_scores", {}),
            "clip_aggregate_scores": clip_result.get("clip_aggregate_scores", {}),
            "clip_per_prompt_scores": clip_result.get("clip_per_prompt_scores", {}),
            "clip_top_label": clip_result.get("clip_top_label", "NA"),
            "clip_top_score": clip_result.get("clip_top_score", "NA"),
            "clip_error": clip_result.get("clip_error", "NA"),
            "parent_sys_executable": clip_result.get("parent_sys_executable", sys.executable),
            "clip_subprocess_python": clip_result.get("clip_subprocess_python", sys.executable),
            "same_as_parent_python": clip_result.get("same_as_parent_python", False),
            "clip_environment_check": clip_result.get("clip_environment_check", {}),
            "class_score_aggregation": class_score_aggregation,
            "color_texture_features": features,
            "grey_score": features.get("grey_score", 0.0),
            "warm_beige_score": features.get("warm_beige_score", 0.0),
            "cement_texture_score": features.get("cement_texture_score", 0.0),
            "concrete_color_gate_score": round(
                float(features.get("grey_score", 0.0)) - float(features.get("warm_beige_score", 0.0)),
                4,
            ),
            "beige_brick_penalty_applied": beige_penalty_applied,
            "concrete_color_texture_guard_reason": concrete_guard_candidate_reason,
            "requested_score": requested_score,
            "source_class_prior_bonus_applied": source_bonus if source_match else 0.0,
            "final_score": round(final_score, 4),
            "score_margin": round(margin, 4),
            "crop_source_used": crop_source_used,
            "crop_path_used_for_scoring": str(crop_used),
            "score_source": (
                clip_result.get("score_source", "clip_crop_verifier")
                if decision_backend == "clip_crop_verifier"
                else "material_scores_override" if isinstance(override_scores, dict) else "color_texture_heuristic_crop_rerank"
            ),
        })
    scored.sort(key=lambda item: (item.get("requested_score", 0.0), item.get("score_margin", 0.0)), reverse=True)
    selected_before_prior = scored[0] if scored else {}
    same_source = [item for item in scored if normalize_material_name(item.get("source_detection_query")) == requested]
    same_source.sort(key=lambda item: (item.get("requested_score", 0.0), item.get("score_margin", 0.0)), reverse=True)
    same_best = same_source[0] if same_source else {}
    scored_after_prior = sorted(scored, key=lambda item: (item.get("final_score", 0.0), item.get("score_margin", 0.0)), reverse=True)
    top = scored_after_prior[0] if scored_after_prior else {}
    source_prior_used = bool(same_source_preferred and same_best and float(same_best.get("requested_score", 0.0)) >= min_accept_score)
    cross_source_override_used = False
    if source_prior_used:
        other_best = next((item for item in scored_after_prior if item.get("candidate_id") != same_best.get("candidate_id")), {})
        if other_best and normalize_material_name(other_best.get("source_detection_query")) != requested:
            if float(other_best.get("requested_score", 0.0)) >= float(same_best.get("requested_score", 0.0)) + override_margin:
                top = other_best
                cross_source_override_used = True
            else:
                top = same_best
        else:
            top = same_best
    top_scores = top.get("material_scores", {}) if top else {}
    beige_guard_triggered = False
    beige_guard_reason = "NA"
    concrete_color_texture_guard_triggered = False
    concrete_color_texture_guard_reason = "NA"
    no_verified_match = False
    if requested == "concrete block" and top and decision_backend != "clip_crop_verifier":
        grey_candidates = [
            item for item in scored_after_prior
            if float(item.get("grey_score", 0.0)) >= concrete_min_grey_score
            and float(item.get("warm_beige_score", 0.0)) <= concrete_max_warm_beige_score
        ]
        concrete_color_best = max(
            scored_after_prior,
            key=lambda item: (
                float(item.get("concrete_color_gate_score", 0.0)),
                float(item.get("grey_score", 0.0)),
                float(item.get("material_scores", {}).get("concrete block", 0.0)),
            ),
            default={},
        )
        if (
            concrete_guard_enabled
            and concrete_color_best
            and concrete_color_best.get("candidate_id") != top.get("candidate_id")
            and float(concrete_color_best.get("concrete_color_gate_score", -1.0))
            > float(top.get("concrete_color_gate_score", -1.0)) + concrete_grey_preference_margin
            and float(concrete_color_best.get("grey_score", 0.0)) >= concrete_min_grey_score
            and float(concrete_color_best.get("warm_beige_score", 0.0)) <= concrete_max_warm_beige_score
        ):
            concrete_color_texture_guard_triggered = True
            concrete_color_texture_guard_reason = (
                "requested concrete block: selected grey/cement-colored candidate over warmer beige/tan candidate"
            )
            top = concrete_color_best
            top_scores = top.get("material_scores", {})
        grey_candidates = [
            item for item in grey_candidates
            if float(item.get("material_scores", {}).get("concrete block", 0.0)) >= min_accept_score
        ]
        if concrete_guard_enabled:
            top_grey = float(top.get("grey_score", 0.0))
            top_warm_beige = float(top.get("warm_beige_score", 0.0))
            top_penalized = bool(top.get("beige_brick_penalty_applied", False))
            if (top_penalized or (top_warm_beige > concrete_max_warm_beige_score and top_grey < concrete_min_grey_score)) and grey_candidates:
                replacement = max(
                    grey_candidates,
                    key=lambda item: (
                        float(item.get("grey_score", 0.0)) + float(item.get("material_scores", {}).get("concrete block", 0.0)),
                        float(item.get("final_score", 0.0)),
                    ),
                )
                if replacement.get("candidate_id") != top.get("candidate_id"):
                    concrete_color_texture_guard_triggered = True
                    concrete_color_texture_guard_reason = (
                        "top concrete candidate looked warm beige/tan with weak grey cement evidence; "
                        "selected best grey/cementitious candidate instead"
                    )
                    top = replacement
                    top_scores = top.get("material_scores", {})
            elif top_grey < concrete_min_grey_score and top_warm_beige > top_grey + concrete_grey_preference_margin and not grey_candidates:
                concrete_color_texture_guard_triggered = True
                concrete_color_texture_guard_reason = (
                    "no candidate had enough grey/cementitious evidence for concrete block; blocked warm beige/tan false positive"
                )
                no_verified_match = allow_no_match
        before_source = normalize_material_name(selected_before_prior.get("source_detection_query")) if selected_before_prior else ""
        before_score = float(selected_before_prior.get("requested_score", 0.0)) if selected_before_prior else 0.0
        same_score = float(same_best.get("requested_score", 0.0)) if same_best else 0.0
        if (
            beige_guard_enabled
            and before_source == "brick"
            and same_best
            and before_score < same_score + override_margin
        ):
            beige_guard_triggered = True
            beige_guard_reason = "requested concrete block but selected candidate came from brick detector while concrete-block candidate existed"
            if same_score >= min_accept_score:
                top = same_best
                top_scores = top.get("material_scores", {})
            else:
                no_verified_match = allow_no_match
        else:
            brick_score = float(top_scores.get("brick", 0.0))
            concrete_score = float(top_scores.get("concrete block", 0.0))
            if beige_guard_enabled and brick_score >= concrete_score + min_margin:
                beige_guard_triggered = True
                beige_guard_reason = (
                    "top crop scored more brick-like than concrete-like; beige/tan fired-clay brick guard blocked concrete selection"
                )
                no_verified_match = allow_no_match
    if requested == "concrete block" and decision_backend != "clip_crop_verifier" and any(bool(item.get("beige_brick_penalty_applied", False)) for item in scored):
        if not concrete_color_texture_guard_triggered:
            concrete_color_texture_guard_triggered = True
            concrete_color_texture_guard_reason = (
                "one or more requested-concrete candidates received a warm beige/tan fired-clay penalty before final selection"
            )
    if top and float(top.get("score_margin", 0.0)) < -min_margin and allow_no_match:
        no_verified_match = True
    clip_environment_failed = bool(
        decision_backend == "clip_crop_verifier"
        and candidates
        and any(item.get("score_source") == "clip_environment_failed" for item in scored)
    )
    clip_verifier_failed = bool(
        decision_backend == "clip_crop_verifier"
        and candidates
        and not clip_environment_failed
        and not any(item.get("score_source") in {"clip_crop_verifier", "dry_run_mock_clip_crop_verifier"} for item in scored)
    )
    if decision_backend == "clip_crop_verifier":
        threshold = float(verification_cfg.get("clip_score_threshold", verification_cfg.get("min_accept_score", 0.0)))
        if not scored:
            no_verified_match = allow_no_match
            top = {"candidate_id": "NO_VERIFIED_MATCH", "source_detection_query": "NO_VERIFIED_MATCH", "material_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}}
            top_scores = top.get("material_scores", {})
        elif clip_environment_failed:
            no_verified_match = allow_no_match
            top = {"candidate_id": "NONE", "source_detection_query": "NONE", "material_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}}
            top_scores = top.get("material_scores", {})
        elif clip_verifier_failed:
            no_verified_match = allow_no_match
            top = {"candidate_id": "NONE", "source_detection_query": "NONE", "material_scores": {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}}
            top_scores = top.get("material_scores", {})
        elif float(top.get("requested_score", 0.0)) < threshold and allow_no_match:
            no_verified_match = True
            top = {"candidate_id": "NO_VERIFIED_MATCH", "source_detection_query": "NO_VERIFIED_MATCH", "material_scores": top_scores}
    vlm_decision: dict[str, Any] = {
        "vlm_decision_attempted": False,
        "vlm_decision_completed": False,
        "vlm_decision_source": "not_required_official_selector_is_clip_crop_verifier",
        "parsed_json": {},
        "error": None,
    }
    if decision_backend == "qwen_vlm_server" and qwen_enabled:
        if case_dir is None:
            vlm_decision = {
                "vlm_decision_attempted": False,
                "vlm_decision_completed": False,
                "vlm_decision_source": "qwen_vlm_server",
                "parsed_json": {},
                "error": "case_dir_required_for_vlm_material_decision",
            }
        else:
            vlm_decision = call_material_verification_vlm(trial, scored, case_dir / "material_verification", config)
        parse_result = vlm_decision.get("parse_result") if isinstance(vlm_decision.get("parse_result"), dict) else {}
        selected_id = str(parse_result.get("selected_candidate_id", "")).strip()
        scored_by_id = {str(item.get("candidate_id")): item for item in scored}
        if vlm_decision.get("vlm_decision_completed") and selected_id in scored_by_id:
            top = scored_by_id[selected_id]
            top_scores = top.get("material_scores", {})
            no_verified_match = False
            concrete_color_texture_guard_triggered = bool(concrete_color_texture_guard_triggered)
            concrete_color_texture_guard_reason = concrete_color_texture_guard_reason
        elif vlm_decision.get("vlm_decision_completed") and selected_id == "NO_VERIFIED_MATCH":
            no_verified_match = allow_no_match
            top = {"candidate_id": "NO_VERIFIED_MATCH", "source_detection_query": "NO_VERIFIED_MATCH", "material_scores": top_scores}
        elif selected_id in {"VLM_PARSE_FAILED", "VLM_SERVER_UNAVAILABLE"}:
            no_verified_match = allow_no_match
            top = {"candidate_id": selected_id, "source_detection_query": selected_id, "material_scores": top_scores}
        elif require_vlm_decision:
            no_verified_match = allow_no_match
            top = {"candidate_id": "VLM_PARSE_FAILED", "source_detection_query": "VLM_PARSE_FAILED", "material_scores": top_scores}
            top_scores = {"brick": 0.0, "timber": 0.0, "concrete block": 0.0}
    selected = {} if no_verified_match else top
    path_candidate = selected
    if not path_candidate and top and str(top.get("candidate_id", "")) not in {
        "NO_VERIFIED_MATCH",
        "NONE",
        "clip_verifier_failed",
        "VLM_PARSE_FAILED",
        "VLM_SERVER_UNAVAILABLE",
    }:
        path_candidate = top
    selected_id_for_output = selected.get("candidate_id", top.get("candidate_id", "NO_VERIFIED_MATCH") if top else "NO_VERIFIED_MATCH")
    selected_source_for_output = selected.get("source_detection_query", top.get("source_detection_query", selected_id_for_output) if top else selected_id_for_output)
    selected_crop_source = path_candidate.get("crop_source_used", "NA") if path_candidate else top.get("crop_source_used", "NA") if top else "NA"
    selected_crop_path = path_candidate.get("crop_path_used_for_scoring", "NA") if path_candidate else top.get("crop_path_used_for_scoring", "NA") if top else "NA"
    return {
        "material_verification_method": (
            "qwen_vlm_material_crop_verification"
            if decision_backend == "qwen_vlm_server" and qwen_enabled
            else "clip_crop_verifier"
            if decision_backend == "clip_crop_verifier"
            else "color_texture_heuristic_crop_rerank"
        ),
        "decision_backend": decision_backend,
        "official_selector": official_selector,
        "qwen_vlm_server_enabled": qwen_enabled,
        "clip_environment_failed": clip_environment_failed,
        "clip_verifier_failed": clip_verifier_failed,
        "failure_reason": (
            "CLIP subprocess Python missing transformers"
            if clip_environment_failed
            else "CLIP crop verifier failed"
            if clip_verifier_failed
            else "NA"
        ),
        "require_vlm_decision": require_vlm_decision,
        "vlm_decision_attempted": vlm_decision.get("vlm_decision_attempted", False),
        "vlm_decision_completed": vlm_decision.get("vlm_decision_completed", False),
        "vlm_decision_source": vlm_decision.get("vlm_decision_source", "NA"),
        "vlm_input_image_path": vlm_decision.get("vlm_input_image_path", "NA"),
        "vlm_prompt_path": vlm_decision.get("vlm_prompt_path", "NA"),
        "vlm_response_path": vlm_decision.get("vlm_response_path", "NA"),
        "vlm_selected_candidate_id": (vlm_decision.get("parse_result") or {}).get("selected_candidate_id", "NA")
        if isinstance(vlm_decision.get("parse_result"), dict)
        else "NA",
        "vlm_selection_state": (vlm_decision.get("parse_result") or {}).get("selection_state", "NA")
        if isinstance(vlm_decision.get("parse_result"), dict)
        else "NA",
        "vlm_predicted_material": (vlm_decision.get("parsed_json") or {}).get("predicted_material", "NA")
        if isinstance(vlm_decision.get("parsed_json"), dict)
        else "NA",
        "vlm_confidence": (vlm_decision.get("parsed_json") or {}).get("confidence", "NA")
        if isinstance(vlm_decision.get("parsed_json"), dict)
        else "NA",
        "vlm_reason": (vlm_decision.get("parsed_json") or {}).get("reason", "NA")
        if isinstance(vlm_decision.get("parsed_json"), dict)
        else vlm_decision.get("error", "NA"),
        "vlm_decision_error": vlm_decision.get("error", "NA"),
        "parent_sys_executable": (scored[0].get("parent_sys_executable", sys.executable) if scored else sys.executable),
        "clip_subprocess_python": (scored[0].get("clip_subprocess_python", sys.executable) if scored else sys.executable),
        "same_as_parent_python": (scored[0].get("same_as_parent_python", False) if scored else False),
        "clip_environment_check": (scored[0].get("clip_environment_check", {}) if scored else {}),
        "requested_class": requested,
        "candidates": scored,
        "selected_candidate": selected,
        "selected_candidate_id": selected_id_for_output,
        "selected_source_detection_query": selected_source_for_output,
        "selected_candidate_mask_path": path_candidate.get("mask_path", "NA") if path_candidate else "NA",
        "selected_candidate_crop_path": path_candidate.get("crop_path_used_for_scoring", "NA") if path_candidate else "NA",
        "selected_candidate_masked_texture_crop_path": path_candidate.get("masked_texture_crop_path", "NA") if path_candidate else "NA",
        "selected_candidate_bbox": _candidate_bbox(path_candidate) if path_candidate else "NA",
        "material_scores": top_scores if top else {"brick": 0.0, "timber": 0.0, "concrete block": 0.0},
        "material_prompt_labels": MATERIAL_VERIFICATION_TEXT_LABELS,
        "use_multi_prompt_clip_labels": use_multi_prompt_labels,
        "class_score_aggregation": class_score_aggregation,
        "color_texture_features": {
            item.get("candidate_id", f"candidate_{idx:03d}"): item.get("color_texture_features", {})
            for idx, item in enumerate(scored, start=1)
        },
        "crop_source_configured": configured_crop_source,
        "crop_source_used_for_material_verification": selected_crop_source,
        "crop_path_used_for_scoring": selected_crop_path,
        "masked_texture_crops_generated": all(
            bool(item.get("masked_texture_crop_path")) and item.get("masked_texture_crop_path") != "NA"
            for item in candidates
        ) if candidates else False,
        "source_class_prior_used": source_prior_used,
        "source_class_prior_bonus": source_bonus,
        "same_source_preferred": same_source_preferred,
        "selected_candidate_id_before_prior": selected_before_prior.get("candidate_id", "NA") if selected_before_prior else "NA",
        "selected_candidate_id_after_prior": top.get("candidate_id", "NA") if top else "NA",
        "selected_before_source_prior": selected_before_prior.get("candidate_id", "NA") if selected_before_prior else "NA",
        "selected_after_source_prior": top.get("candidate_id", "NA") if top else "NA",
        "same_source_candidate_available": bool(same_best),
        "same_source_candidate_score": same_best.get("requested_score", "NA") if same_best else "NA",
        "cross_source_override_used": cross_source_override_used,
        "cross_source_override_margin": override_margin,
        "beige_brick_guard_triggered": beige_guard_triggered,
        "beige_brick_guard_reason": beige_guard_reason,
        "concrete_color_texture_guard_triggered": concrete_color_texture_guard_triggered,
        "concrete_color_texture_guard_reason": concrete_color_texture_guard_reason,
        "beige_brick_penalty_applied": any(bool(item.get("beige_brick_penalty_applied", False)) for item in scored),
        "grey_score": top.get("grey_score", "NA") if top else "NA",
        "warm_beige_score": top.get("warm_beige_score", "NA") if top else "NA",
        "concrete_requires_grey_texture_guard": concrete_guard_enabled,
        "no_verified_match": no_verified_match,
    }


def save_candidate_pool_outputs(case_dir: Path, candidates: list[dict[str, Any]], verification: dict[str, Any]) -> None:
    pool_dir = case_dir / "candidate_pool"
    material_dir = case_dir / "material_verification"
    pool_dir.mkdir(parents=True, exist_ok=True)
    material_dir.mkdir(parents=True, exist_ok=True)
    write_json(pool_dir / "candidate_pool_manifest.json", {"candidates": candidates})
    write_json(material_dir / "material_scores.json", verification)
    write_json(material_dir / "selected_candidate_summary.json", verification.get("selected_candidate", {}))
    selected_id = verification.get("selected_candidate_id")
    guard = bool(verification.get("beige_brick_guard_triggered", False))
    scored_by_id = {item.get("candidate_id"): item for item in verification.get("candidates", [])}
    manifest_rows: list[dict[str, Any]] = []
    resolved_crop_to_candidates: dict[str, list[str]] = {}
    score_dict_to_candidates: dict[str, list[str]] = {}
    thumbs: list[Image.Image] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.get("candidate_id", f"candidate_{idx:03d}")
        scored = scored_by_id.get(candidate_id, {})
        path = (
            scored.get("crop_path_used_for_scoring")
            or candidate.get("inner_texture_crop_path")
            or candidate.get("masked_texture_crop_path")
            or candidate.get("bbox_crop_path")
        )
        path_str = str(path) if path and path != "NA" else "NA"
        displayed = False
        resolved_path = "NA"
        if path_str != "NA" and Path(path_str).exists():
            try:
                resolved_path = str(Path(path_str).resolve())
                resolved_crop_to_candidates.setdefault(resolved_path, []).append(str(candidate_id))
            except Exception:
                resolved_path = path_str
        scores = scored.get("material_scores", {})
        score_key = json.dumps(scores, sort_keys=True)
        score_dict_to_candidates.setdefault(score_key, []).append(str(candidate_id))
        is_selected = candidate_id == selected_id
        wrong_source_guard = (
            guard
            and verification.get("requested_class") == "concrete block"
            and normalize_material_name(candidate.get("source_detection_query")) == "brick"
        )
        border = (45, 150, 70) if is_selected else ((215, 88, 44) if wrong_source_guard else (100, 100, 100))
        canvas = Image.new("RGB", (260, 260), (245, 245, 240))
        draw = ImageDraw.Draw(canvas)
        try:
            if path_str == "NA" or not Path(path_str).exists():
                raise FileNotFoundError(path_str)
            thumb = Image.open(path_str).convert("RGB")
            thumb.thumbnail((230, 150))
            canvas.paste(thumb, ((260 - thumb.width) // 2, 54 + (150 - thumb.height) // 2))
            displayed = True
        except Exception:
            draw.rectangle((20, 58, 240, 200), fill=(232, 230, 222), outline=(130, 130, 130), width=2)
            draw.text((34, 116), "No valid crop", fill=(70, 70, 70), font=_font(16, True))
        draw.rectangle((3, 3, 256, 256), outline=border, width=4)
        draw.text((10, 8), f"C{idx}: {candidate.get('source_detection_query')}", fill=(20, 22, 25), font=_font(13, True))
        draw.text((10, 28), str(candidate_id)[:34], fill=(20, 22, 25), font=_font(11))
        draw.text((10, 213), f"brick={scores.get('brick', 'NA')} conc={scores.get('concrete block', 'NA')}", fill=(20, 22, 25), font=_font(11))
        draw.text((10, 230), f"grey={scored.get('grey_score', 'NA')} warm={scored.get('warm_beige_score', 'NA')}", fill=(20, 22, 25), font=_font(11))
        draw.text((10, 246), f"final={scored.get('final_score', 'NA')} crop={scored.get('crop_source_used', 'NA')}", fill=(20, 22, 25), font=_font(10))
        thumbs.append(canvas)
        manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "source_detection_query": candidate.get("source_detection_query", "NA"),
                "displayed_in_crop_panel": displayed,
                "bbox_crop_path": candidate.get("bbox_crop_path", "NA"),
                "masked_texture_crop_path": candidate.get("masked_texture_crop_path", "NA"),
                "inner_texture_crop_path": candidate.get("inner_texture_crop_path", "NA"),
                "crop_used_for_verification": path_str,
                "crop_used_for_verification_resolved": resolved_path,
                "verification_scores": scores,
                "clip_labels": scored.get("clip_labels", []),
                "clip_scores": scored.get("clip_scores", {}),
                "clip_aggregate_scores": scored.get("clip_aggregate_scores", {}),
                "clip_per_prompt_scores": scored.get("clip_per_prompt_scores", {}),
                "clip_top_label": scored.get("clip_top_label", "NA"),
                "clip_top_score": scored.get("clip_top_score", "NA"),
                "clip_error": scored.get("clip_error", "NA"),
                "parent_sys_executable": scored.get("parent_sys_executable", "NA"),
                "clip_subprocess_python": scored.get("clip_subprocess_python", "NA"),
                "same_as_parent_python": scored.get("same_as_parent_python", "NA"),
                "clip_environment_check": scored.get("clip_environment_check", {}),
                "grey_score": scored.get("grey_score", "NA"),
                "warm_beige_score": scored.get("warm_beige_score", "NA"),
                "color_texture_features": scored.get("color_texture_features", {}),
                "concrete_color_texture_guard_reason": scored.get("concrete_color_texture_guard_reason", "NA"),
                "beige_brick_penalty_applied": scored.get("beige_brick_penalty_applied", False),
                "score_source": scored.get("score_source", "material_crop_verification_reranker"),
                "final_selected": is_selected,
            }
        )
    duplicate_warnings = [
        {
            "crop_path": path,
            "candidate_ids": ids,
            "warning": "different candidate ids resolve to the same crop path",
        }
        for path, ids in resolved_crop_to_candidates.items()
        if len(set(ids)) > 1
    ]
    duplicate_score_warnings = [
        {
            "score_dict": json.loads(score_key) if score_key else {},
            "candidate_ids": ids,
            "warning": "different candidate ids have identical verification score dictionaries",
        }
        for score_key, ids in score_dict_to_candidates.items()
        if len(set(ids)) > 1
    ]
    crop_panel_manifest = {
        "experiment_mode": verification.get("experiment_mode", "primary_requested_class_only"),
        "oracle_known_classes_used": verification.get("oracle_known_classes_used", False),
        "active_prompt_stages": ["open_vocab_detection", "material_crop_verification"],
        "panel_layout": "all_candidates_material_verification_crops",
        "crop_priority": "crop_path_used_for_scoring_from_material_verification",
        "candidates": manifest_rows,
        "duplicate_crop_path_detected": bool(duplicate_warnings),
        "duplicate_score_dict_detected": bool(duplicate_score_warnings),
        "duplicate_crop_path_warnings": duplicate_warnings,
        "duplicate_score_dict_warnings": duplicate_score_warnings,
        "concrete_color_texture_guard_triggered": verification.get("concrete_color_texture_guard_triggered", False),
        "concrete_color_texture_guard_reason": verification.get("concrete_color_texture_guard_reason", "NA"),
        "beige_brick_penalty_applied": verification.get("beige_brick_penalty_applied", False),
        "decision_backend": verification.get("decision_backend", "NA"),
        "vlm_decision_attempted": verification.get("vlm_decision_attempted", False),
        "vlm_decision_completed": verification.get("vlm_decision_completed", False),
        "vlm_decision_source": verification.get("vlm_decision_source", "NA"),
        "vlm_selected_candidate_id": verification.get("vlm_selected_candidate_id", "NA"),
        "vlm_reason": verification.get("vlm_reason", "NA"),
        "official_selector": verification.get("official_selector", "NA"),
        "clip_verifier_failed": verification.get("clip_verifier_failed", False),
    }
    write_json(material_dir / "crop_panel_manifest.json", crop_panel_manifest)
    if duplicate_warnings:
        (material_dir / "crop_panel_warnings.txt").write_text(
            "\n".join(f"{item['warning']}: {item['crop_path']} -> {', '.join(item['candidate_ids'])}" for item in duplicate_warnings)
            + "\n",
            encoding="utf-8",
        )
    if duplicate_score_warnings:
        with (material_dir / "crop_panel_warnings.txt").open("a", encoding="utf-8") as handle:
            for item in duplicate_score_warnings:
                handle.write(f"{item['warning']}: {item['candidate_ids']}\n")
    if thumbs:
        header_h = 78
        panel = Image.new("RGB", (max(560, len(thumbs) * 270 + 10), 270 + header_h), (230, 232, 226))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 10), "CLIP material crop verification", fill=(20, 22, 25), font=_font(18, True))
        draw.text((12, 36), f"requested={verification.get('requested_class')} selected={selected_id} selector={verification.get('official_selector', 'clip_crop_verifier')}", fill=(20, 22, 25), font=_font(13))
        if verification.get("requested_class") == "concrete block":
            rule = "rule: concrete should be grey/cementitious; beige/tan fired-clay brick is negative"
        else:
            rule = f"active stages: open_vocab_detection -> material_crop_verification | beige_guard={verification.get('beige_brick_guard_triggered')}"
        draw.text((12, 56), rule[:118], fill=(20, 22, 25), font=_font(13))
        x = 10
        for thumb in thumbs:
            panel.paste(thumb, (x, header_h))
            x += 270
        panel.save(material_dir / "crop_verification_panel.png")


def save_t5v2_candidate_masks_overlay(
    rgb_path: Path,
    candidates: list[dict[str, Any]],
    output_path: Path,
    requested_class: str,
) -> str:
    if not candidates or not rgb_path.exists():
        return "NA"
    image = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    palette = [
        (235, 82, 70, 95),
        (68, 155, 230, 95),
        (76, 190, 118, 95),
        (235, 185, 55, 95),
        (165, 110, 225, 95),
    ]
    for idx, candidate in enumerate(candidates, start=1):
        mask = _load_selected_mask(candidate.get("mask_path"), image.size)
        if mask is None:
            continue
        color = palette[(idx - 1) % len(palette)]
        mask_img = Image.fromarray((mask.astype(np.uint8) * color[3]), mode="L")
        color_layer = Image.new("RGBA", image.size, color)
        overlay.paste(color_layer, (0, 0), mask_img)
        ys, xs = np.where(mask)
        if xs.size:
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            draw_overlay.rectangle((x0, y0, x1, y1), outline=color[:3] + (255,), width=3)
            label = f"C{idx}: {candidate.get('source_detection_query', 'NA')}"
            draw_overlay.rectangle((x0, max(0, y0 - 24), min(image.width, x0 + 190), max(22, y0)), fill=(0, 0, 0, 180))
            draw_overlay.text((x0 + 4, max(1, y0 - 21)), label, fill=(255, 255, 255, 255), font=_font(14, True))
    composed = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(composed)
    panel_h = 54
    draw.rectangle((0, 0, min(430, image.width), panel_h), fill=(245, 245, 238, 235), outline=(30, 30, 30, 255), width=1)
    draw.text((10, 8), "T5_v2 candidate masks", fill=(20, 22, 25, 255), font=_font(16, True))
    draw.text((10, 30), f"requested: {requested_class}", fill=(20, 22, 25, 255), font=_font(14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output_path)
    return str(output_path)


def save_t5v2_candidate_crops_panel(
    candidates: list[dict[str, Any]],
    output_path: Path,
    experiment_mode_value: str,
) -> str:
    if not candidates:
        return "NA"
    tiles: list[Image.Image] = []
    for idx, candidate in enumerate(candidates, start=1):
        crop_path = (
            candidate.get("inner_texture_crop_path")
            if candidate.get("inner_texture_crop_path") and candidate.get("inner_texture_crop_path") != "NA"
            else candidate.get("masked_texture_crop_path")
            if candidate.get("masked_texture_crop_path") and candidate.get("masked_texture_crop_path") != "NA"
            else candidate.get("bbox_crop_path")
        )
        tile = Image.new("RGB", (300, 285), (242, 243, 238))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 299, 66), fill=(36, 39, 43))
        draw.text((10, 8), f"C{idx}: {candidate.get('candidate_id', 'NA')}"[:32], fill=(255, 255, 255), font=_font(14, True))
        draw.text((10, 32), f"source={candidate.get('source_detection_query', 'NA')}"[:34], fill=(235, 235, 235), font=_font(12))
        draw.text((10, 50), f"score={candidate.get('detection_score', 'NA')}", fill=(235, 235, 235), font=_font(11))
        try:
            if not crop_path or crop_path == "NA":
                raise FileNotFoundError("NA")
            crop = Image.open(crop_path).convert("RGB")
            crop.thumbnail((275, 160))
            tile.paste(crop, ((300 - crop.width) // 2, 76 + (160 - crop.height) // 2))
        except Exception:
            draw.rectangle((22, 86, 278, 226), outline=(120, 120, 120), width=2)
            draw.text((72, 145), "No valid crop", fill=(70, 70, 70), font=_font(16, True))
        crop_type = (
            "inner_texture_crop"
            if crop_path == candidate.get("inner_texture_crop_path")
            else "masked_texture_crop"
            if crop_path == candidate.get("masked_texture_crop_path")
            else "bbox_crop"
        )
        draw.text((10, 246), f"crop={crop_type}", fill=(30, 32, 35), font=_font(12))
        draw.text((10, 264), f"mode={experiment_mode_value}"[:40], fill=(30, 32, 35), font=_font(11))
        tiles.append(tile)
    cols = min(3, max(1, len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * 300, rows * 285), (225, 226, 220))
    for idx, tile in enumerate(tiles):
        panel.paste(tile, ((idx % cols) * 300, (idx // cols) * 285))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return str(output_path)


def run_candidate_pool_perception(
    case_dir: Path,
    raw_rgb: str,
    depth_path: Path | None,
    trial: T5V2Trial,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query_plan = resolve_detection_queries_for_trial(trial, config)
    detection_queries = list(query_plan["detection_queries"])
    detection_queries = [query for query in detection_queries if query]
    max_candidates = int(config.get("candidate_pool", {}).get("max_candidates_per_class", 5))
    candidates: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for query in detection_queries:
        if "," in query or len(query) > 60:
            raise RuntimeError(f"Unsafe non-simple detection query blocked for Grounded-SAM2: {query}")
        query_dir = case_dir / "backend_outputs" / f"detection_{_slug(query)}"
        query_case = T5Case(
            case_id=f"trial_{trial.trial_index:03d}_{_slug(query)}",
            scene_id=trial.frame_id,
            layout_id=trial.frame_id,
            material_query=query,
            objects_present=trial.classes_existing_in_frame if query_plan["oracle_known_classes_used"] else query,
            expected_material=trial.expected_selected_class,
        )
        try:
            result = run_real_perception(query_dir, Path(raw_rgb), depth_path, query_case, config)
        except Exception as exc:  # noqa: BLE001
            errors[query] = str(exc)
            continue
        pool_dir = case_dir / "candidate_pool"
        _copy_if_exists(result.get("overlay_path"), pool_dir / f"detection_{_slug(query)}_overlay.png")
        annotations = sorted(_candidate_annotations(result), key=_score, reverse=True)[:max_candidates]
        candidate_specs: list[dict[str, Any]] = []
        for ann_idx, ann in enumerate(annotations, start=1):
            decoded = _decode_mask(ann)
            if decoded is None or not np.any(decoded):
                continue
            candidate_specs.append(
                {
                    "annotation": ann,
                    "mask_array": decoded.astype(bool),
                    "detection_score": _score(ann),
                    "source": "requested_class_detection_annotation",
                    "annotation_index": ann_idx,
                }
            )
        if not candidate_specs and result.get("selected_target_mask_path"):
            selected_mask = _load_selected_mask(result.get("selected_target_mask_path"), Image.open(raw_rgb).size)
            if selected_mask is not None:
                candidate_specs.append(
                    {
                        "annotation": {},
                        "mask_array": selected_mask,
                        "detection_score": result.get("selected_detection_score", result.get("score", "NA")),
                        "source": "selected_backend_mask_fallback",
                        "annotation_index": 1,
                    }
                )
        for spec in candidate_specs:
            candidate_id = f"candidate_{len(candidates) + 1:03d}_{_slug(query)}"
            mask_copy_path = pool_dir / f"{candidate_id}_mask.png"
            Image.fromarray(spec["mask_array"].astype(np.uint8) * 255).save(mask_copy_path)
            crop_path = create_bbox_crop_from_mask(raw_rgb, mask_copy_path, pool_dir / f"{candidate_id}_bbox_crop.png")
            mv_cfg = config.get("material_verification", {})
            masked_crop = create_masked_material_crop(
                raw_rgb,
                mask_copy_path,
                pool_dir / f"{candidate_id}_masked_texture_crop.png",
                erode_px=int(mv_cfg.get("mask_erode_px", 3)),
                pad_px=8,
            )
            candidate_result = dict(_json_safe_result(result))
            candidate_result.update(
                {
                    "selected_target_mask_path": str(mask_copy_path),
                    "selected_candidate_id": candidate_id,
                    "selected_source_detection_query": query,
                    "selected_label": query,
                    "selected_label_or_NONE": query,
                    "valid_target": True,
                }
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_detection_query": query,
                    "candidate_generation_source": spec["source"],
                    "annotation_index": spec["annotation_index"],
                    "detection_score": spec["detection_score"],
                    "bbox": masked_crop.get("bbox_xyxy", "NA"),
                    "bbox_crop_path": crop_path,
                    "crop_path": crop_path,
                    "masked_texture_crop_path": masked_crop.get("crop_path", "NA"),
                    "inner_texture_crop_path": masked_crop.get("inner_texture_crop_path", "NA"),
                    "masked_texture_crop_alpha_path": masked_crop.get("alpha_crop_path", "NA"),
                    "masked_texture_crop_metadata": masked_crop,
                    "mask_path": str(mask_copy_path),
                    "overlay_path": result.get("overlay_path", "NA"),
                    "backend_result": candidate_result,
                }
            )
    verification = build_material_verification(trial, candidates, config, case_dir=case_dir)
    verification.update(
        {
            "experiment_mode": query_plan["experiment_mode"],
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
        }
    )
    save_candidate_pool_outputs(case_dir, candidates, verification)
    selected = verification.get("selected_candidate") or {}
    selected_result = selected.get("backend_result") if isinstance(selected.get("backend_result"), dict) else {}
    if verification.get("no_verified_match"):
        state = verification.get("selected_candidate_id", "NO_VERIFIED_MATCH")
        final_result = {
            "valid_target": False,
            "selected_label": state,
            "selected_label_or_NONE": state,
            "selected_candidate_id": state,
            "selected_source_detection_query": state,
        }
    elif selected_result:
        final_result = dict(selected_result)
        final_result.update(
            {
                "valid_target": True,
                "selected_label": verification.get("selected_source_detection_query", "NA"),
                "selected_label_or_NONE": verification.get("selected_source_detection_query", "NA"),
                "selected_candidate_id": verification.get("selected_candidate_id", "NA"),
                "selected_source_detection_query": verification.get("selected_source_detection_query", "NA"),
                "selected_candidate_mask_path": verification.get("selected_candidate_mask_path", "NA"),
                "selected_candidate_crop_path": verification.get("selected_candidate_crop_path", "NA"),
                "selected_candidate_masked_texture_crop_path": verification.get("selected_candidate_masked_texture_crop_path", "NA"),
            }
        )
    elif candidates:
        final_result = candidates[0].get("backend_result", {})
    else:
        final_result = {"valid_target": False, "selected_label": "NO_VERIFIED_MATCH", "selected_label_or_NONE": "NO_VERIFIED_MATCH"}
    final_result.update(
        {
            "experiment_mode": query_plan["experiment_mode"],
            "experiment_mode_label": (
                "ORACLE known-classes candidate pool"
                if query_plan["oracle_known_classes_used"]
                else "primary requested-class only"
            ),
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
        }
    )
    pool_summary = {
        "candidate_pool_enabled": True,
        **query_plan,
        "candidate_pool_classes": detection_queries,
        "detection_queries_used": detection_queries,
        "candidate_count": len(candidates),
        "candidate_errors": errors,
        "candidates": candidates,
    }
    return final_result, pool_summary, verification


def reevaluate_attempt(attempt_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Rerun only T5_v2 material verification/selection on saved attempt crops."""

    case_dir = resolve_repo_path(attempt_dir)
    metadata = read_json(case_dir / "trial_metadata.json")
    if not metadata:
        raise FileNotFoundError(f"Missing trial_metadata.json in {case_dir}")
    trial = T5V2Trial(
        trial_index=int(metadata["trial_index"]),
        frame_id=str(metadata["frame_id"]),
        classes_existing_in_frame=str(metadata.get("classes_existing_in_frame", "")),
        input_text=str(metadata["input_text"]),
        expected_selected_class=str(metadata.get("expected_selected_class", metadata.get("input_text", ""))),
        enabled=bool(metadata.get("enabled", True)),
        notes=str(metadata.get("notes", "")),
    )
    old_summary = read_json(case_dir / "target_selection_summary.json")
    old_selected = old_summary.get("final_selected_candidate_id", old_summary.get("selected_candidate_id", "NA"))
    candidate_manifest = read_json(case_dir / "candidate_pool" / "candidate_pool_manifest.json")
    candidates = candidate_manifest.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"No saved candidates found in {case_dir / 'candidate_pool' / 'candidate_pool_manifest.json'}")
    query_plan = resolve_detection_queries_for_trial(trial, config)
    material_verification = build_material_verification(trial, candidates, config, case_dir=case_dir)
    material_verification.update(
        {
            "experiment_mode": query_plan["experiment_mode"],
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
            "reevaluated_offline": True,
            "reevaluated_at": now_iso(),
        }
    )
    save_candidate_pool_outputs(case_dir, candidates, material_verification)
    selected = material_verification.get("selected_candidate") or {}
    selected_result = selected.get("backend_result") if isinstance(selected.get("backend_result"), dict) else {}
    if material_verification.get("no_verified_match"):
        state = material_verification.get("selected_candidate_id", "NO_VERIFIED_MATCH")
        result = {
            "valid_target": False,
            "selected_label": state,
            "selected_label_or_NONE": state,
            "selected_candidate_id": state,
            "selected_source_detection_query": state,
        }
    elif selected_result:
        result = dict(selected_result)
    else:
        result = {"valid_target": bool(selected), "selected_label": material_verification.get("selected_source_detection_query", "NA")}
    result.update(
        {
            "selected_candidate_id": material_verification.get("selected_candidate_id", "NA"),
            "selected_source_detection_query": material_verification.get("selected_source_detection_query", "NA"),
            "selected_candidate_mask_path": material_verification.get("selected_candidate_mask_path", "NA"),
            "selected_candidate_crop_path": material_verification.get("selected_candidate_crop_path", "NA"),
            "selected_candidate_masked_texture_crop_path": material_verification.get("selected_candidate_masked_texture_crop_path", "NA"),
            "selected_label": material_verification.get("selected_candidate_id", "NO_VERIFIED_MATCH") if material_verification.get("no_verified_match") else material_verification.get("selected_source_detection_query", trial.input_text),
            "selected_label_or_NONE": material_verification.get("selected_candidate_id", "NO_VERIFIED_MATCH") if material_verification.get("no_verified_match") else material_verification.get("selected_source_detection_query", trial.input_text),
            "experiment_mode": query_plan["experiment_mode"],
            "experiment_mode_label": (
                "ORACLE known-classes candidate pool"
                if query_plan["oracle_known_classes_used"]
                else "primary requested-class only"
            ),
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
        }
    )
    prompt_info = material_prompt_info(config, trial)
    selected_overlay, overlay_generated, overlay_reason = generate_selected_mask_overlay(
        case_dir / "raw_rgb.png",
        result,
        case_dir / "selected_mask_overlay.png",
        trial,
        prompt_info,
        str(old_summary.get("status", "reevaluated_offline")),
    )
    candidate_crops_panel = save_t5v2_candidate_crops_panel(
        candidates,
        case_dir / "candidate_crops_panel.png",
        query_plan["experiment_mode"],
    )
    new_selected = material_verification.get("selected_candidate_id", "NA")
    merged_summary = dict(old_summary)
    merged_summary.update(
        {
            "reevaluated_offline": True,
            "reevaluated_at": now_iso(),
            "old_final_selected_candidate_id": old_selected,
            "final_selected_candidate_id": new_selected,
            "final_selected_class_backend": result.get("selected_label", "NA"),
            "selected_source_detection_query": material_verification.get("selected_source_detection_query", "NA"),
            "selected_mask_overlay_path": selected_overlay,
            "candidate_crops_panel_path": candidate_crops_panel,
            "material_verification_panel_path": str(case_dir / "material_verification" / "crop_verification_panel.png"),
            "selected_mask_overlay_generated": overlay_generated,
            "selected_mask_overlay_has_transparent_mask": overlay_generated,
            "selected_mask_overlay_mask_path": _find_selected_mask_path(result, case_dir),
            "selected_mask_overlay_failure_reason": "NA" if overlay_generated else overlay_reason,
            "material_verification": material_verification,
            "material_scores": material_verification.get("material_scores", {}),
            "crop_source_used_for_material_verification": material_verification.get("crop_source_used_for_material_verification", "NA"),
            "concrete_color_texture_guard_triggered": material_verification.get("concrete_color_texture_guard_triggered", False),
            "concrete_color_texture_guard_reason": material_verification.get("concrete_color_texture_guard_reason", "NA"),
            "beige_brick_penalty_applied": material_verification.get("beige_brick_penalty_applied", False),
            "grey_score": material_verification.get("grey_score", "NA"),
            "warm_beige_score": material_verification.get("warm_beige_score", "NA"),
            "no_verified_match": material_verification.get("no_verified_match", False),
            "beige_brick_guard_triggered": material_verification.get("beige_brick_guard_triggered", False),
            "beige_brick_guard_reason": material_verification.get("beige_brick_guard_reason", "NA"),
            "detection_queries_used": query_plan["detection_queries"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "experiment_mode": query_plan["experiment_mode"],
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
        }
    )
    write_json(case_dir / "target_selection_summary.json", merged_summary)
    payload = {
        "attempt_dir": str(case_dir),
        "old_final_selected_candidate_id": old_selected,
        "new_final_selected_candidate_id": new_selected,
        "new_final_selected_class_backend": result.get("selected_label", "NA"),
        "no_verified_match": material_verification.get("no_verified_match", False),
        "concrete_color_texture_guard_triggered": material_verification.get("concrete_color_texture_guard_triggered", False),
        "concrete_color_texture_guard_reason": material_verification.get("concrete_color_texture_guard_reason", "NA"),
        "beige_brick_penalty_applied": material_verification.get("beige_brick_penalty_applied", False),
        "grey_score": material_verification.get("grey_score", "NA"),
        "warm_beige_score": material_verification.get("warm_beige_score", "NA"),
        "material_scores": material_verification.get("material_scores", {}),
        "reevaluated_at": now_iso(),
    }
    write_json(case_dir / "reevaluation_summary.json", payload)
    return payload


def regenerate_attempt_overlays(attempt_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Regenerate selected/candidate overlays from saved candidate masks only."""

    case_dir = resolve_repo_path(attempt_dir)
    metadata = read_json(case_dir / "trial_metadata.json")
    if not metadata:
        raise FileNotFoundError(f"Missing trial_metadata.json in {case_dir}")
    trial = T5V2Trial(
        trial_index=int(metadata["trial_index"]),
        frame_id=str(metadata["frame_id"]),
        classes_existing_in_frame=str(metadata.get("classes_existing_in_frame", "")),
        input_text=str(metadata["input_text"]),
        expected_selected_class=str(metadata.get("expected_selected_class", metadata.get("input_text", ""))),
        enabled=bool(metadata.get("enabled", True)),
        notes=str(metadata.get("notes", "")),
    )
    raw_rgb = case_dir / "raw_rgb.png"
    if not raw_rgb.exists():
        raise FileNotFoundError(f"Missing raw_rgb.png in {case_dir}")

    summary = read_json(case_dir / "target_selection_summary.json")
    material_verification = read_json(case_dir / "material_verification" / "material_scores.json")
    manifest = read_json(case_dir / "candidate_pool" / "candidate_pool_manifest.json")
    candidates = manifest.get("candidates", []) if isinstance(manifest, dict) else []
    selected_id = (
        summary.get("final_selected_candidate_id")
        or summary.get("selected_candidate_id")
        or material_verification.get("selected_candidate_id")
        or "NO_VERIFIED_MATCH"
    )
    selected_candidate = _candidate_from_manifest(case_dir, str(selected_id))
    selected_mask_path = resolve_selected_candidate_mask(selected_candidate, case_dir, case_dir / "backend_outputs") if selected_candidate else None
    result = {
        "valid_target": bool(selected_candidate and selected_mask_path),
        "selected_candidate_id": selected_id,
        "selected_source_detection_query": (
            selected_candidate.get("source_detection_query")
            if selected_candidate
            else material_verification.get("selected_source_detection_query", "NA")
        ),
        "selected_candidate_mask_path": str(selected_mask_path) if selected_mask_path else "NA",
        "selected_target_mask_path": str(selected_mask_path) if selected_mask_path else "NA",
        "selected_mask_path": str(selected_mask_path) if selected_mask_path else "NA",
        "selected_candidate_bbox": _candidate_bbox(selected_candidate) if selected_candidate else "NA",
        "selected_label": selected_candidate.get("source_detection_query", trial.input_text) if selected_candidate else "NO_VERIFIED_MATCH",
        "selected_label_or_NONE": selected_candidate.get("source_detection_query", trial.input_text) if selected_candidate else "NO_VERIFIED_MATCH",
        "experiment_mode": summary.get("experiment_mode", "primary_requested_class_only"),
        "experiment_mode_label": "primary requested-class only",
        "requested_detection_query": summary.get("requested_detection_query", normalize_material_name(trial.input_text)),
    }
    prompt_info = material_prompt_info(config, trial)
    selected_overlay, overlay_generated, overlay_reason = generate_selected_mask_overlay(
        raw_rgb,
        result,
        case_dir / "selected_mask_overlay.png",
        trial,
        prompt_info,
        str(summary.get("status", "overlay_regenerated")),
    )
    candidate_masks = save_t5v2_candidate_masks_overlay(
        raw_rgb,
        candidates,
        case_dir / "candidate_masks_overlay.png",
        normalize_material_name(trial.input_text),
    )
    payload = {
        "regenerated_at": now_iso(),
        "attempt_dir": str(case_dir),
        "selected_candidate_id": selected_id,
        "selected_candidate_mask_path": str(selected_mask_path) if selected_mask_path else "NA",
        "selected_mask_overlay_path": selected_overlay,
        "selected_mask_overlay_generated": overlay_generated,
        "selected_mask_overlay_has_transparent_mask": overlay_generated,
        "selected_mask_overlay_mask_path": str(selected_mask_path) if selected_mask_path else "NA",
        "selected_mask_overlay_failure_reason": "NA" if overlay_generated else overlay_reason,
        "candidate_masks_overlay_path": candidate_masks,
        "candidate_count": len(candidates),
        "used_camera": False,
        "reran_grounding_sam2": False,
        "reran_clip": False,
    }
    write_json(case_dir / "overlay_regeneration_summary.json", payload)
    merged = dict(summary)
    merged.update(payload)
    write_json(case_dir / "target_selection_summary.json", merged)
    if case_dir.parent.name == "attempts":
        base_case_dir = case_dir.parents[1]
        saved_attempt = base_case_dir / "saved_attempt.txt"
        if saved_attempt.exists():
            try:
                if Path(saved_attempt.read_text(encoding="utf-8")).resolve() == case_dir.resolve():
                    _copy_if_exists(selected_overlay, base_case_dir / "selected_mask_overlay.png")
                    _copy_if_exists(candidate_masks, base_case_dir / "candidate_masks_overlay.png")
                    write_json(base_case_dir / "target_selection_summary.json", merged)
            except Exception:
                pass
    return payload


def capture_ros2_snapshot(case_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    capture = _capture_config(config)
    if capture["mode"] != "ros2_topics":
        if capture["mode"] == "direct_realsense" and not bool(config.get("allow_direct_realsense", False)):
            raise RuntimeError(
                "T5_v2 safety error: direct RealSense capture is disabled. "
                "Use capture.mode='ros2_topics' with the RealSense ROS2 node running."
            )
        raise RuntimeError(f"T5_v2 unsupported live capture mode for this runner: {capture['mode']}")

    from upv_vlm_orient.ros2_snapshot.rgbd_snapshot_node import (  # noqa: PLC0415
        Ros2SnapshotUnavailable,
        capture_ros2_rgbd_snapshot,
    )

    snapshot_dir = case_dir / "ros2_snapshot"
    ros2_config = {
        "ros2": {
            "color_topic": capture["color_topic"],
            "depth_topic": capture["depth_topic"],
            "camera_info_topic": capture["camera_info_topic"],
            "sync_slop_s": capture["sync_slop_s"],
            "timeout_s": capture["timeout_s"],
            "queue_size": capture["queue_size"],
        }
    }
    try:
        result = capture_ros2_rgbd_snapshot(snapshot_dir, ros2_config, timeout_s=capture["timeout_s"])
    except Ros2SnapshotUnavailable as exc:
        raise RuntimeError(
            "ROS2 snapshot capture is unavailable. Source ROS2 and run T5_v2 from an environment with rclpy, "
            "sensor_msgs, and message_filters.\n\n"
            f"Expected RealSense launch:\n{REALSENSE_ROS2_LAUNCH_COMMAND}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "ROS2 RealSense snapshot failed before perception started. T5_v2 did not open the RealSense device directly.\n\n"
            f"Expected RealSense launch:\n{REALSENSE_ROS2_LAUNCH_COMMAND}\n\n"
            f"Original error: {exc}"
        ) from exc

    raw_rgb = snapshot_dir / "raw_rgb.png"
    depth_raw = snapshot_dir / "depth_raw.npy"
    depth_vis = snapshot_dir / "depth_visualization.png"
    camera_info = snapshot_dir / "camera_info.json"
    metadata = snapshot_dir / "rgbd_snapshot_metadata.json"
    _copy_snapshot_file(result.get("color_path"), raw_rgb)
    _copy_snapshot_file(result.get("depth_npy_path"), depth_raw)
    _copy_snapshot_file(result.get("depth_png_path"), depth_vis)
    _copy_snapshot_file(result.get("intrinsics_path"), camera_info)
    _copy_snapshot_file(result.get("metadata_path") or result.get("capture_metadata_path"), metadata)

    metadata_payload = read_json(metadata)
    metadata_payload.update(
        {
            "capture_mode": "ros2_topics",
            "aligned_depth_camera_info_topic": capture["aligned_depth_camera_info_topic"],
            "backend_input_mode": config.get("backend_input_mode", "saved_rgb_image"),
            "direct_realsense_access": False,
            "message_availability_check": "capture_ros2_rgbd_snapshot_received_synchronized_color_depth_camera_info",
            "source_paths": {
                "color_path": result.get("color_path", "NA"),
                "depth_png_path": result.get("depth_png_path", "NA"),
                "depth_npy_path": result.get("depth_npy_path", "NA"),
                "intrinsics_path": result.get("intrinsics_path", "NA"),
                "metadata_path": result.get("metadata_path") or result.get("capture_metadata_path", "NA"),
            },
        }
    )
    write_json(metadata, metadata_payload)

    root_rgb = case_dir / "raw_rgb.png"
    root_depth = case_dir / "depth_visualization.png"
    _copy_snapshot_file(raw_rgb, root_rgb)
    _copy_snapshot_file(depth_vis, root_depth)
    print("ROS2 camera preflight:")
    print("color topic: OK")
    print("aligned depth topic: OK")
    print("camera info: OK")
    print("Saved ROS2 snapshot:")
    print(f"raw_rgb.png: {raw_rgb}")
    print(f"depth_raw.npy: {depth_raw}")
    print(f"depth_visualization.png: {depth_vis}")
    print(f"camera_info.json: {camera_info}")
    return {
        "capture_dir": str(snapshot_dir),
        "color_path": str(root_rgb),
        "depth_png_path": str(root_depth),
        "depth_npy_path": str(depth_raw),
        "intrinsics_path": str(camera_info),
        "metadata_path": str(metadata),
        "raw_rgb_snapshot_path": str(raw_rgb),
        "direct_realsense_access": False,
    }


def read_fixed27_cases(path: str | Path) -> list[T5V2Trial]:
    path = resolve_repo_path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [T5V2Trial.from_row(row) for row in rows]


def validate_fixed27_cases(cases: list[T5V2Trial]) -> None:
    enabled = [case for case in cases if case.enabled]
    if len(enabled) != 27:
        raise ValueError(f"T5_v2 requires exactly 27 enabled rows, found {len(enabled)}")
    first, last = enabled[0], enabled[-1]
    if (first.trial_index, first.frame_id, first.input_text, first.expected_selected_class) != (1, "TS_F01", "brick", "Brick"):
        raise ValueError(f"Unexpected first T5_v2 trial: {first}")
    if (last.trial_index, last.frame_id, last.input_text, last.expected_selected_class) != (27, "TS_F12", "concrete block", "Concrete block"):
        raise ValueError(f"Unexpected last T5_v2 trial: {last}")


def create_session_dir(output_root: str | Path, session_dir: str | Path | None = None) -> Path:
    if session_dir:
        path = resolve_repo_path(session_dir)
        path.mkdir(parents=True, exist_ok=True)
    else:
        root = resolve_repo_path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        for suffix in range(100):
            name = f"session_{now_stamp()}" if suffix == 0 else f"session_{now_stamp()}_{suffix:02d}"
            path = root / name
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"Could not create unique T5_v2 session under {root}")
    for sub in ["cases", "paper_figures", "paper_tables", "logs"]:
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_master_rows(session_dir: Path) -> list[dict[str, str]]:
    path = session_dir / "T5_v2_master_results.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_master_row(session_dir: Path, row: dict[str, Any]) -> None:
    path = session_dir / "T5_v2_master_results.csv"
    existing = _read_master_rows(session_dir)
    existing = [old for old in existing if str(old.get("trial_index")) != str(row.get("trial_index"))]
    fieldnames = list(V2_COLUMNS)
    for old in existing:
        fieldnames = list(dict.fromkeys([*fieldnames, *old.keys()]))
    fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for old in sorted(existing, key=lambda item: int(item.get("trial_index") or 0)):
            writer.writerow({key: old.get(key, "NA") for key in fieldnames})
        writer.writerow({key: row.get(key, "NA") for key in fieldnames})
    update_case_progress(session_dir)
    update_summary_tables(session_dir)


def update_case_progress(session_dir: Path) -> None:
    rows = _read_master_rows(session_dir)
    progress_path = session_dir / "T5_v2_case_progress.csv"
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["trial_index", "frame_id", "input_text", "status", "correct_manual", "completed_at"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item.get("trial_index") or 0)):
            writer.writerow({key: row.get(key, "NA") for key in fieldnames})


def update_summary_tables(session_dir: Path) -> None:
    rows = _read_master_rows(session_dir)
    paper_tables = session_dir / "paper_tables"
    _write_rows(paper_tables / "table_experiment_1_target_selection.csv", rows, V2_COLUMNS)

    labels = sorted(
        {
            *(row.get("expected_selected_class", "NA") for row in rows),
            *(row.get("final_selected_class_manual", "NA") for row in rows),
        }
    )
    matrix: dict[tuple[str, str], int] = {}
    for row in rows:
        expected = row.get("expected_selected_class", "NA") or "NA"
        selected = row.get("final_selected_class_manual", "NA") or "NA"
        matrix[(expected, selected)] = matrix.get((expected, selected), 0) + 1
    cm_rows = [{"expected_selected_class": exp, "final_selected_class_manual": sel, "count": count} for (exp, sel), count in sorted(matrix.items())]
    if not cm_rows:
        cm_rows = [{"expected_selected_class": "NA", "final_selected_class_manual": "NA", "count": 0}]
    _write_rows(paper_tables / "table_experiment_1_confusion_matrix.csv", cm_rows)

    failure_counts: dict[str, int] = {}
    for row in rows:
        failure = row.get("failure_type_manual", "NA") or "NA"
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
    failure_rows = [{"failure_type_manual": key, "count": value} for key, value in sorted(failure_counts.items())]
    if not failure_rows:
        failure_rows = [{"failure_type_manual": "NA", "count": 0}]
    _write_rows(paper_tables / "table_experiment_1_failure_types.csv", failure_rows)


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "NA") for key in fieldnames})


def case_dir_for(session_dir: Path, trial: T5V2Trial) -> Path:
    return session_dir / "cases" / trial.case_folder_name()


def is_trial_complete(session_dir: Path, trial: T5V2Trial) -> bool:
    return (case_dir_for(session_dir, trial) / "saved_attempt.txt").exists()


def next_attempt_index(base_case_dir: Path) -> int:
    attempts_dir = base_case_dir / "attempts"
    existing: list[int] = []
    if attempts_dir.exists():
        for path in attempts_dir.glob("attempt_*"):
            if path.is_dir():
                try:
                    existing.append(int(path.name.split("_", 1)[1]))
                except Exception:
                    continue
    return max(existing, default=0) + 1


def attempt_dir_for(base_case_dir: Path, attempt_index: int) -> Path:
    return base_case_dir / "attempts" / f"attempt_{attempt_index:03d}"


def append_attempt_log(
    session_dir: Path,
    trial: T5V2Trial,
    *,
    attempt_index: int,
    attempt_status: str,
    attempt_dir: Path,
    backend_status: str = "NA",
    perception_outputs_found: bool | str = "NA",
    discard_reason: str = "NA",
) -> None:
    path = session_dir / "T5_v2_attempt_log.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "trial_index": trial.trial_index,
                "frame_id": trial.frame_id,
                "input_text": trial.input_text,
                "attempt_index": attempt_index,
                "attempt_status": attempt_status,
                "case_attempt_dir": str(attempt_dir),
                "backend_status": backend_status,
                "perception_outputs_found": perception_outputs_found,
                "discard_reason": discard_reason,
                "created_at": now_iso(),
            }
        )


def print_trial_block(trial: T5V2Trial, ordinal: int, total: int) -> None:
    print("\n" + "=" * 60)
    print("Experiment 1 / Target Selection")
    print(f"Trial {ordinal} / {total}")
    print(f"Frame ID: {trial.frame_id}")
    print(f"Place these classes in frame: {trial.classes_existing_in_frame}")
    print(f"Input text / requested class: {trial.input_text}")
    print(f"Expected selected class: {trial.expected_selected_class}")
    print("=" * 60)


def prompt_manual_label(config: dict[str, Any], no_manual: bool) -> dict[str, Any]:
    if no_manual:
        return {
            "final_selected_class_manual": "NA",
            "correct_manual": "NA",
            "failure_type_manual": "NA",
            "manual_notes": "NA",
        }
    print("\nInspect the output folder and enter manual result.")
    final_selected = input("Final selected class: ").strip() or "NA"
    correct_answer = input("Correct? [y/n/skip]: ").strip().lower()
    if correct_answer in {"y", "yes"}:
        correct = "yes"
    elif correct_answer in {"n", "no"}:
        correct = "no"
    else:
        correct = "NA"
    options = list(config.get("failure_type_options", [])) or [
        "none",
        "wrong object selected",
        "wrong material class selected",
        "no mask detected",
        "mask detected but unusable",
        "crop verification failed",
        "ambiguous output",
    ]
    print("Failure type:")
    for idx, option in enumerate(options, start=1):
        print(f"  {idx} {option}")
    selected = input("Select failure type number: ").strip()
    try:
        failure = options[int(selected) - 1]
    except Exception:
        failure = "none" if correct == "yes" else "NA"
    if correct == "no" and failure == "none":
        confirm = input("Correct is no but failure type is none. Keep failure type none? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            failure = "NA"
    notes = input("Manual notes: ").strip() or "NA"
    return {
        "final_selected_class_manual": final_selected,
        "correct_manual": correct,
        "failure_type_manual": failure,
        "manual_notes": notes,
    }


def build_v2_row(
    trial: T5V2Trial,
    case_dir: Path,
    *,
    status: str,
    backend_status: str,
    result: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    manual: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    prompt_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = result or {}
    artifacts = artifacts or {}
    manual = manual or {}
    timings = timings or {}
    prompt_info = prompt_info or {}
    crop = result.get("crop_verification") if isinstance(result.get("crop_verification"), dict) else {}
    return {
        "trial_index": trial.trial_index,
        "frame_id": trial.frame_id,
        "classes_existing_in_frame": trial.classes_existing_in_frame,
        "input_text": trial.input_text,
        "original_input_text": prompt_info.get("original_input_text", trial.input_text),
        "detection_query_used": prompt_info.get("detection_query_used", trial.input_text),
        "expanded_backend_query": prompt_info.get("expanded_backend_query", trial.input_text),
        "prompt_expansion_injection_status": prompt_info.get("prompt_expansion_injection_status", "not_used"),
        "material_color_hint": prompt_info.get("color_semantic_hint", "NA"),
        "expanded_query_used_for_detection": prompt_info.get("expanded_query_used_for_detection", False),
        "expanded_material_description": prompt_info.get("expanded_material_description", "NA"),
        "expected_selected_class": trial.expected_selected_class,
        "experiment_mode": artifacts.get("experiment_mode", "primary_requested_class_only"),
        "oracle_known_classes_used": artifacts.get("oracle_known_classes_used", False),
        "classes_existing_used_as_model_input": artifacts.get("classes_existing_used_as_model_input", False),
        "active_prompt_stages": "open_vocab_detection, material_crop_verification",
        "final_overlay_label_source": "t5_v2_final_decision",
        "candidate_pool_classes": ", ".join(artifacts.get("candidate_pool_classes", [])) if isinstance(artifacts.get("candidate_pool_classes"), list) else artifacts.get("candidate_pool_classes", "NA"),
        "detection_queries_used": ", ".join(artifacts.get("detection_queries_used", [])) if isinstance(artifacts.get("detection_queries_used"), list) else artifacts.get("detection_queries_used", "NA"),
        "candidate_generation_source": artifacts.get("candidate_generation_source", "NA"),
        "material_verification_method": (artifacts.get("material_verification") or {}).get("material_verification_method", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "material_score_brick": ((artifacts.get("material_verification") or {}).get("material_scores") or {}).get("brick", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "material_score_timber": ((artifacts.get("material_verification") or {}).get("material_scores") or {}).get("timber", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "material_score_concrete_block": ((artifacts.get("material_verification") or {}).get("material_scores") or {}).get("concrete block", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "beige_brick_guard_triggered": (artifacts.get("material_verification") or {}).get("beige_brick_guard_triggered", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "no_verified_match": (artifacts.get("material_verification") or {}).get("no_verified_match", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "selected_source_detection_query": (artifacts.get("material_verification") or {}).get("selected_source_detection_query", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "source_class_prior_used": (artifacts.get("material_verification") or {}).get("source_class_prior_used", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "crop_source_used_for_material_verification": (artifacts.get("material_verification") or {}).get("crop_source_used_for_material_verification", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "concrete_color_texture_guard_triggered": (artifacts.get("material_verification") or {}).get("concrete_color_texture_guard_triggered", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "grey_score": (artifacts.get("material_verification") or {}).get("grey_score", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "warm_beige_score": (artifacts.get("material_verification") or {}).get("warm_beige_score", "NA") if isinstance(artifacts.get("material_verification"), dict) else "NA",
        "final_selected_candidate_id": (artifacts.get("material_verification") or {}).get("selected_candidate_id", result.get("selected_candidate_id", "NA")) if isinstance(artifacts.get("material_verification"), dict) else result.get("selected_candidate_id", "NA"),
        "final_selected_class_manual": manual.get("final_selected_class_manual", "NA"),
        "correct_manual": manual.get("correct_manual", "NA"),
        "failure_type_manual": manual.get("failure_type_manual", "NA"),
        "manual_notes": manual.get("manual_notes", "NA"),
        "status": status,
        "perception_backend_status": backend_status,
        "target_selected": bool(result.get("valid_target", False)),
        "final_selected_material_backend": result.get("selected_label", result.get("selected_label_or_NONE", "NA")),
        "final_selected_object_id_backend": result.get("selected_candidate_id", result.get("winning_candidate", "NA")),
        "grounding_mask_available": bool(result.get("selected_target_mask_path")),
        "candidate_count": result.get("requested_class_annotation_count", "NA"),
        "crop_verification_prediction": crop.get("top_label", result.get("crop_verifier_top_label", "NA")),
        "crop_verification_confidence": crop.get("top_score", result.get("crop_verifier_top_score", "NA")),
        "raw_rgb_path": artifacts.get("raw_rgb_path", "NA"),
        "candidate_masks_overlay_path": artifacts.get("candidate_masks_overlay_path", "NA"),
        "candidate_crops_panel_path": artifacts.get("candidate_crops_panel_path", "NA"),
        "selected_mask_overlay_path": artifacts.get("selected_mask_overlay_path", "NA"),
        "case_output_dir": str(case_dir),
        "backend_output_dir": str(case_dir / "backend_outputs"),
        "total_time_s": timings.get("total_time_s", result.get("total_runtime_sec", "NA")),
        "camera_capture_time_s": timings.get("camera_capture_time_s", "NA"),
        "open_vocab_detection_time_s": result.get("gsam2_runtime_sec", "NA"),
        "segmentation_time_s": result.get("pipeline_runtime_sec", "NA"),
        "crop_verification_time_s": result.get("crop_verifier_runtime_sec", "NA"),
        "final_selection_time_s": timings.get("final_selection_time_s", "NA"),
        "visualization_save_time_s": timings.get("visualization_save_time_s", "NA"),
        "created_at": timings.get("created_at", now_iso()),
        "completed_at": timings.get("completed_at", now_iso()),
    }


def prompt_save_decision(trial: T5V2Trial, attempt_dir: Path, artifacts: dict[str, Any], *, no_manual: bool) -> str:
    if no_manual:
        return "save"
    print(f"\nInference completed for Trial {trial.trial_index}.")
    print(f"Case output folder: {attempt_dir}")
    print("Important images:")
    print(f"- raw_rgb: {artifacts.get('raw_rgb_path', 'NA')}")
    print(f"- candidate masks: {artifacts.get('candidate_masks_overlay_path', 'NA')}")
    print(f"- selected mask: {artifacts.get('selected_mask_overlay_path', 'NA')}")
    print(f"- candidate crops: {artifacts.get('candidate_crops_panel_path', 'NA')}")
    while True:
        answer = input(
            f"Do you want to save this inference as the official result for Trial {trial.trial_index}?\n"
            "[s] save / [d] discard / [r] rerun / [q] quit: "
        ).strip().lower()
        if answer in {"s", "save"}:
            return "save"
        if answer in {"d", "discard"}:
            return "discard"
        if answer in {"r", "rerun"}:
            return "rerun"
        if answer in {"q", "quit"}:
            return "quit"
        print("Choose s, d, r, or q.")


def run_trial(
    session_dir: Path,
    trial: T5V2Trial,
    config: dict[str, Any],
    *,
    live: bool = False,
    image_path: Path | None = None,
    no_manual: bool = False,
    dry_run: bool = False,
    skip_scene_confirmation: bool = False,
) -> dict[str, Any]:
    base_case_dir = case_dir_for(session_dir, trial)
    for sub in ["attempts", "logs"]:
        (base_case_dir / sub).mkdir(parents=True, exist_ok=True)
    write_json(base_case_dir / "config_snapshot.json", config)
    write_json(base_case_dir / "trial_metadata.json", trial.to_dict())
    if dry_run:
        return run_dry_trial(session_dir, trial, config)
    if not skip_scene_confirmation and bool(config.get("interactive_scene_confirmation", True)):
        answer = input("Arrange the scene so the listed classes are visible.\nPress ENTER to run perception for this trial, or type skip: ").strip().lower()
        if answer == "skip":
            manual = {
                "final_selected_class_manual": "NA",
                "correct_manual": "NA",
                "failure_type_manual": "NA",
                "manual_notes": "skipped_by_user",
            }
            prompt_info = material_prompt_info(config, trial)
            row = build_v2_row(trial, base_case_dir, status="skipped", backend_status="not_run_user_skipped", manual=manual, prompt_info=prompt_info)
            write_json(base_case_dir / "manual_label.json", manual)
            write_json(
                base_case_dir / "target_selection_summary.json",
                {
                    "success": False,
                    "status": "skipped",
                    "backend_call_attempted": False,
                    "backend_call_completed": False,
                    "perception_outputs_found": False,
                    "perception_backend_status": "not_run_user_skipped",
                    "material_prompt_engineering": prompt_info,
                    "trial": trial.to_dict(),
                    "row": row,
                },
            )
            write_master_row(session_dir, row)
            return {"success": False, "status": "skipped", "row": row}

    while True:
        attempt_index = next_attempt_index(base_case_dir)
        case_dir = attempt_dir_for(base_case_dir, attempt_index)
        for sub in ["logs", "backend_outputs", "candidate_crops"]:
            (case_dir / sub).mkdir(parents=True, exist_ok=True)
        (case_dir / "logs" / "backend_stdout_stderr.log").write_text(
            "T5_v2 calls scripts.run_one_image.run_integrated_case in-process; no subprocess stdout/stderr is produced.\n",
            encoding="utf-8",
        )
        write_json(case_dir / "config_snapshot.json", config)
        write_json(case_dir / "trial_metadata.json", {**trial.to_dict(), "attempt_index": attempt_index})
        prompt_info = material_prompt_info(config, trial)
        write_json(case_dir / "material_prompt_engineering.json", prompt_info)
        print_material_prompt_info(prompt_info)
        manifest_path = write_prompt_manifest(case_dir, config, trial.input_text)
        if manifest_path != "NA" and Path(manifest_path).exists():
            shutil.copyfile(manifest_path, case_dir / "prompt_manifest.json")

        total_start = time.perf_counter()
        capture_time: float | str = "NA"
        depth_path: Path | None = None
        backend_call_attempted = False
        backend_call_completed = False
        backend_status = "not_attempted"
        perception_outputs_found = False
        backend_retry_attempted = False
        backend_retry_reason = "NA"
        backend_retry_query = "NA"
        backend_retry_success = False
        try:
            if live:
                capture_start = time.perf_counter()
                capture_mode = _capture_config(config)["mode"]
                if capture_mode != "ros2_topics":
                    if capture_mode == "direct_realsense" and not bool(config.get("allow_direct_realsense", False)):
                        raise RuntimeError(
                            "T5_v2 safety error: direct RealSense capture is disabled by default. "
                            "Start the RealSense ROS2 node and use capture.mode='ros2_topics'."
                        )
                    raise RuntimeError(f"T5_v2 live mode only supports ROS2 topic capture here, got {capture_mode!r}.")
                print_capture_preflight(config)
                capture = capture_ros2_snapshot(case_dir, config)
                capture_time = time.perf_counter() - capture_start
                rgb_path = Path(capture["color_path"])
                depth_path = Path(capture["depth_png_path"])
            elif image_path is not None:
                rgb_path = image_path
            else:
                raise ValueError("T5_v2 non-dry execution requires --live or --image.")

            raw_rgb = _copy_if_exists(rgb_path, case_dir / "raw_rgb.png")
            depth_vis = _save_depth_visualization(depth_path, case_dir / "depth_visualization.png")
            if depth_vis == "NA":
                depth_vis = _write_placeholder_image(case_dir / "depth_visualization.png", "Depth visualization", "NA: depth image not available")
            if bool(config.get("force_backend_image_file_mode", True)) and not Path(raw_rgb).exists():
                raise RuntimeError(
                    "T5_v2 safety error: backend image-file mode was requested, but no saved RGB image exists. "
                    "Refusing to run a backend path that could attempt direct RealSense capture."
                )
            backend_output_dir = case_dir / "backend_outputs"
            query_plan = resolve_detection_queries_for_trial(trial, config)
            print("Running perception backend:")
            print("backend = scripts.run_one_image.run_integrated_case")
            print(f"image = {raw_rgb}")
            print(f"experiment_mode = {query_plan['experiment_mode']}")
            print(f"detection_queries = {query_plan['detection_queries']}")
            print(f"output_dir = {backend_output_dir}")
            if prompt_info.get("expanded_query_used_for_detection"):
                print("WARNING: use_expanded_query_for_detection=true may break Grounded-SAM2. Recommended false.")
            backend_call_attempted = True
            result, candidate_pool_summary, material_verification = run_candidate_pool_perception(
                case_dir,
                raw_rgb,
                depth_path,
                trial,
                config,
            )
            if material_verification.get("clip_environment_failed"):
                backend_status = "clip_environment_failed"
            candidate_ids = [
                item.get("candidate_id")
                for item in material_verification.get("candidates", [])
                if item.get("candidate_id")
            ]
            manifest_path = write_prompt_manifest(
                case_dir,
                config,
                trial.input_text,
                candidate_ids=candidate_ids,
                prompt_injection_status="prompt_logged_only_clip_fixed_labels",
            )
            if manifest_path != "NA" and Path(manifest_path).exists():
                _copy_if_exists(manifest_path, case_dir / "prompt_manifest.json")
            backend_call_completed = bool(candidate_pool_summary.get("candidates"))
            if backend_status != "clip_environment_failed":
                backend_status = "completed" if backend_call_completed else "failed_no_candidates"
            backend_query = ", ".join(candidate_pool_summary.get("detection_queries_used", []))
            vis_start = time.perf_counter()
            backend_images = _copy_backend_images(case_dir)
            candidate_masks = save_t5v2_candidate_masks_overlay(
                Path(raw_rgb),
                candidate_pool_summary.get("candidates", []),
                case_dir / "candidate_masks_overlay.png",
                normalize_material_name(trial.input_text),
            )
            if candidate_masks == "NA":
                candidate_masks = save_candidate_masks_overlay(Path(raw_rgb), result, case_dir / "candidate_masks_overlay.png")
            candidate_crops = save_t5v2_candidate_crops_panel(
                candidate_pool_summary.get("candidates", []),
                case_dir / "candidate_crops_panel.png",
                candidate_pool_summary.get("experiment_mode", "primary_requested_class_only"),
            )
            legacy_backend_candidate_masks = save_candidate_masks_overlay(Path(raw_rgb), result, case_dir / "legacy_backend_candidate_masks_overlay.png")
            selected_overlay, selected_overlay_transparent, selected_overlay_reason = generate_selected_mask_overlay(
                Path(raw_rgb),
                result,
                case_dir / "selected_mask_overlay.png",
                trial,
                prompt_info,
                backend_status,
            )
            material_panel = save_material_verification_panel(result, case_dir / "material_verification_panel.png")
            verification_panel = case_dir / "material_verification" / "crop_verification_panel.png"
            if verification_panel.exists():
                material_panel = str(verification_panel)
            visualization_time = time.perf_counter() - vis_start
            perception_outputs_found = bool(
                backend_images
                or (selected_overlay != "NA" and Path(selected_overlay).exists())
                or bool(result.get("selected_target_mask_path"))
                or bool(result.get("requested_class_annotations"))
            )
            if not perception_outputs_found:
                backend_status = "failed_no_outputs"
            artifacts = {
                "raw_rgb_path": raw_rgb,
                "depth_visualization_path": depth_vis,
                "candidate_masks_overlay_path": candidate_masks,
                "candidate_crops_panel_path": candidate_crops,
                "candidate_crops_panel_source": "t5_v2_final_candidate_set",
                "legacy_backend_candidate_crops_panel_path": "NA",
                "legacy_backend_candidate_masks_overlay_path": legacy_backend_candidate_masks,
                "selected_mask_overlay_path": selected_overlay,
                "material_verification_panel_path": material_panel,
                "backend_images_all_dir": str(case_dir / "backend_outputs" / "images_all"),
                "backend_image_count": len(backend_images),
                "selected_mask_overlay_generated": selected_overlay_transparent,
                "selected_mask_overlay_has_transparent_mask": selected_overlay_transparent,
                "selected_mask_overlay_mask_path": _find_selected_mask_path(result, case_dir),
                "selected_mask_overlay_failure_reason": "NA" if selected_overlay_transparent else selected_overlay_reason,
                "selected_mask_overlay_layout": "header_panel_above_rgb_image",
                "experiment_mode": candidate_pool_summary.get("experiment_mode", "primary_requested_class_only"),
                "oracle_known_classes_used": candidate_pool_summary.get("oracle_known_classes_used", False),
                "classes_existing_used_as_model_input": candidate_pool_summary.get("classes_existing_used_as_model_input", False),
                "requested_detection_query": candidate_pool_summary.get("requested_detection_query", normalize_material_name(trial.input_text)),
                "candidate_generation_source": candidate_pool_summary.get("candidate_generation_source", "requested_class_detection_instances"),
                "candidate_pool_classes": candidate_pool_summary.get("candidate_pool_classes", []),
                "detection_queries_used": candidate_pool_summary.get("detection_queries_used", []),
                "candidate_pool_enabled": candidate_pool_summary.get("candidate_pool_enabled", False),
                "material_verification": material_verification,
            }
            print(f"Backend call attempted: {str(backend_call_attempted).lower()}")
            print(f"Backend call completed: {str(backend_call_completed).lower()}")
            print(f"Perception outputs found: {str(perception_outputs_found).lower()}")
            if not perception_outputs_found:
                print("WARNING: No perception mask/overlay outputs found. Do not mark this as successful target selection unless manually verified.")
            print(f"Output folder: {case_dir}")
            print(f"selected_mask_overlay: {selected_overlay}")
            print(f"candidate_crops_panel: {candidate_crops}")

            timings = {
                "created_at": now_iso(),
                "completed_at": now_iso(),
                "total_time_s": time.perf_counter() - total_start,
                "camera_capture_time_s": capture_time,
                "visualization_save_time_s": visualization_time,
                "final_selection_time_s": result.get("pipeline_runtime_sec", "NA"),
            }
            prompt_manifest = read_json(case_dir / "prompts" / "prompt_manifest.json")
            predecision_summary = {
                "success": bool(perception_outputs_found),
                "status": "clip_environment_failed"
                if material_verification.get("clip_environment_failed")
                else "inference_completed_pending_save_decision",
                "attempt_index": attempt_index,
                "backend_call_attempted": backend_call_attempted,
                "backend_call_completed": backend_call_completed,
                "perception_outputs_found": perception_outputs_found,
                "perception_backend_status": backend_status,
                "backend_retry_attempted": backend_retry_attempted,
                "backend_retry_reason": backend_retry_reason,
                "backend_retry_query": backend_retry_query,
                "backend_retry_success": backend_retry_success,
                "prompt_registry_version": prompt_manifest.get("prompt_registry_version", "NA"),
                "active_prompt_stages": prompt_manifest.get("active_prompt_stages", []),
                "inactive_prompt_stages": prompt_manifest.get("inactive_prompt_stages", []),
                "open_vocab_detection_prompt": (prompt_manifest.get("stages", {}).get("open_vocab_detection", {}) or {}).get("prompt", "NA"),
                "material_crop_verification_prompt_path": (prompt_manifest.get("stages", {}).get("material_crop_verification", {}) or {}).get("prompt_text_path", "NA"),
                "edge_contact_quality_prompt_path": (prompt_manifest.get("stages", {}).get("edge_contact_quality", {}) or {}).get("prompt_text_path", "NA"),
                "final_overlay_label_source": "t5_v2_final_decision",
                "legacy_backend_label_ignored_for_overlay": True,
                "prompt_expansion_injection_status": prompt_info["prompt_expansion_injection_status"],
                "material_prompt_engineering": prompt_info,
                "experiment_mode": candidate_pool_summary.get("experiment_mode", "primary_requested_class_only"),
                "primary_requested_class_only": candidate_pool_summary.get("experiment_mode") == "primary_requested_class_only",
                "oracle_known_classes_used": candidate_pool_summary.get("oracle_known_classes_used", False),
                "classes_existing_used_as_model_input": candidate_pool_summary.get("classes_existing_used_as_model_input", False),
                "requested_detection_query": candidate_pool_summary.get("requested_detection_query", normalize_material_name(trial.input_text)),
                "candidate_generation_source": candidate_pool_summary.get("candidate_generation_source", "requested_class_detection_instances"),
                "material_verification_candidate_count": len(material_verification.get("candidates", [])),
                "legacy_backend_artifacts_copied": {
                    "legacy_backend_candidate_crops_panel": False,
                    "legacy_backend_candidate_masks_overlay": legacy_backend_candidate_masks != "NA",
                },
                "candidate_crops_panel_source": "t5_v2_final_candidate_set",
                "candidate_pool_enabled": candidate_pool_summary.get("candidate_pool_enabled", False),
                "candidate_pool_classes": candidate_pool_summary.get("candidate_pool_classes", []),
                "requested_class": normalize_material_name(trial.input_text),
                "detection_queries_used": candidate_pool_summary.get("detection_queries_used", []),
                "final_selected_candidate_id": material_verification.get("selected_candidate_id", "NA"),
                "final_selected_class_backend": result.get("selected_label", result.get("selected_label_or_NONE", "NA")),
                "material_verification_method": material_verification.get("material_verification_method", "NA"),
                "clip_environment_failed": material_verification.get("clip_environment_failed", False),
                "failure_reason": material_verification.get("failure_reason", "NA"),
                "parent_sys_executable": material_verification.get("parent_sys_executable", "NA"),
                "clip_subprocess_python": material_verification.get("clip_subprocess_python", "NA"),
                "same_as_parent_python": material_verification.get("same_as_parent_python", "NA"),
                "material_scores": material_verification.get("material_scores", {}),
                "crop_source_used_for_material_verification": material_verification.get("crop_source_used_for_material_verification", "NA"),
                "masked_texture_crops_generated": material_verification.get("masked_texture_crops_generated", False),
                "source_class_prior_used": material_verification.get("source_class_prior_used", False),
                "same_source_preferred": material_verification.get("same_source_preferred", False),
                "selected_candidate_id_before_prior": material_verification.get("selected_candidate_id_before_prior", "NA"),
                "selected_candidate_id_after_prior": material_verification.get("selected_candidate_id_after_prior", "NA"),
                "selected_source_detection_query": material_verification.get("selected_source_detection_query", "NA"),
                "same_source_candidate_available": material_verification.get("same_source_candidate_available", False),
                "same_source_candidate_score": material_verification.get("same_source_candidate_score", "NA"),
                "cross_source_override_used": material_verification.get("cross_source_override_used", False),
                "cross_source_override_margin": material_verification.get("cross_source_override_margin", "NA"),
                "no_verified_match": material_verification.get("no_verified_match", False),
                "beige_brick_guard_triggered": material_verification.get("beige_brick_guard_triggered", False),
                "beige_brick_guard_reason": material_verification.get("beige_brick_guard_reason", "NA"),
                "concrete_color_texture_guard_triggered": material_verification.get("concrete_color_texture_guard_triggered", False),
                "concrete_color_texture_guard_reason": material_verification.get("concrete_color_texture_guard_reason", "NA"),
                "beige_brick_penalty_applied": material_verification.get("beige_brick_penalty_applied", False),
                "grey_score": material_verification.get("grey_score", "NA"),
                "warm_beige_score": material_verification.get("warm_beige_score", "NA"),
                "selected_candidate_mask_path": material_verification.get("selected_candidate_mask_path", "NA"),
                "selected_candidate_crop_path": material_verification.get("selected_candidate_crop_path", "NA"),
                "selected_candidate_masked_texture_crop_path": material_verification.get("selected_candidate_masked_texture_crop_path", "NA"),
                "selected_mask_overlay_generated": selected_overlay_transparent,
                "selected_mask_overlay_has_transparent_mask": selected_overlay_transparent,
                "selected_mask_overlay_mask_path": artifacts["selected_mask_overlay_mask_path"],
                "selected_mask_overlay_failure_reason": artifacts["selected_mask_overlay_failure_reason"],
                "selected_mask_overlay_layout": artifacts["selected_mask_overlay_layout"],
                "trial": trial.to_dict(),
                "result": _json_safe_result(result),
                "artifacts": artifacts,
            }
            write_json(
                case_dir / "perception_backend_summary.json",
                {
                    "backend_call_attempted": backend_call_attempted,
                    "backend_call_completed": backend_call_completed,
                    "perception_outputs_found": perception_outputs_found,
                    "perception_backend_status": backend_status,
                    "backend": "scripts.run_one_image.run_integrated_case",
                    "image": raw_rgb,
                    "detection_query": backend_query,
                    "output_dir": str(backend_output_dir),
                    "backend_retry_attempted": backend_retry_attempted,
                    "backend_retry_reason": backend_retry_reason,
                    "backend_retry_query": backend_retry_query,
                    "backend_retry_success": backend_retry_success,
                    "backend_images_copied": backend_images,
                    "material_prompt_engineering": prompt_info,
                    "candidate_pool": candidate_pool_summary,
                    "material_verification": material_verification,
                    "result": _json_safe_result(result),
                },
            )
            write_json(case_dir / "timing_summary.json", timings)
            write_json(case_dir / "target_selection_summary.json", predecision_summary)

            decision = prompt_save_decision(trial, case_dir, artifacts, no_manual=no_manual)
            if decision == "quit":
                write_json(case_dir / "target_selection_summary.json", {**predecision_summary, "status": "quit_before_save"})
                append_attempt_log(
                    session_dir,
                    trial,
                    attempt_index=attempt_index,
                    attempt_status="quit_before_save",
                    attempt_dir=case_dir,
                    backend_status=backend_status,
                    perception_outputs_found=perception_outputs_found,
                    discard_reason="operator_quit",
                )
                raise KeyboardInterrupt("operator_quit_before_save")
            if decision in {"discard", "rerun"}:
                discard_reason = "operator_requested_rerun" if decision == "rerun" else "operator_discarded"
                write_json(case_dir / "discarded.json", {"discarded": True, "discard_reason": discard_reason, "discarded_at": now_iso()})
                write_json(
                    case_dir / "target_selection_summary.json",
                    {**predecision_summary, "status": "discarded", "discard_reason": discard_reason},
                )
                append_attempt_log(
                    session_dir,
                    trial,
                    attempt_index=attempt_index,
                    attempt_status="discarded",
                    attempt_dir=case_dir,
                    backend_status=backend_status,
                    perception_outputs_found=perception_outputs_found,
                    discard_reason=discard_reason,
                )
                print("Discarded. This trial is not counted.")
                if decision == "rerun":
                    continue
                if not no_manual:
                    retry = input("Retry this trial now? [y/n]: ").strip().lower()
                    if retry in {"y", "yes"}:
                        continue
                return {"success": False, "status": "discarded", "attempt_dir": str(case_dir)}

            manual = prompt_manual_label(config, no_manual)
            row = build_v2_row(
                trial,
                case_dir,
                status="saved" if perception_outputs_found else "saved_failed_no_outputs",
                backend_status=backend_status,
                result=result,
                artifacts=artifacts,
                manual=manual,
                timings=timings,
                prompt_info=prompt_info,
            )
            write_json(case_dir / "manual_label.json", manual)
            write_json(
                case_dir / "target_selection_summary.json",
                {**predecision_summary, "status": "saved", "row": row, "manual_label": manual},
            )
            write_json(base_case_dir / "manual_label.json", manual)
            write_json(base_case_dir / "trial_result.json", {"saved_attempt_dir": str(case_dir), "row": row})
            write_json(base_case_dir / "target_selection_summary.json", {**predecision_summary, "status": "saved", "row": row})
            for source_key, filename in [
                ("raw_rgb_path", "raw_rgb.png"),
                ("depth_visualization_path", "depth_visualization.png"),
                ("candidate_masks_overlay_path", "candidate_masks_overlay.png"),
                ("candidate_crops_panel_path", "candidate_crops_panel.png"),
                ("selected_mask_overlay_path", "selected_mask_overlay.png"),
                ("material_verification_panel_path", "material_verification_panel.png"),
            ]:
                _copy_if_exists(artifacts.get(source_key), base_case_dir / filename)
            (base_case_dir / "saved_attempt.txt").write_text(str(case_dir), encoding="utf-8")
            append_attempt_log(
                session_dir,
                trial,
                attempt_index=attempt_index,
                attempt_status="saved",
                attempt_dir=case_dir,
                backend_status=backend_status,
                perception_outputs_found=perception_outputs_found,
                discard_reason="NA",
            )
            _copy_paper_figures(session_dir, trial, artifacts)
            write_master_row(session_dir, row)
            return {"success": bool(perception_outputs_found), "row": row, "artifacts": artifacts, "attempt_dir": str(case_dir)}
        except Exception as exc:
            trace = traceback.format_exc()
            (case_dir / "logs" / "error_trace.txt").write_text(trace, encoding="utf-8")
            timings = {"created_at": now_iso(), "completed_at": now_iso(), "total_time_s": time.perf_counter() - total_start, "camera_capture_time_s": capture_time}
            row = build_v2_row(
                trial,
                case_dir,
                status="failed",
                backend_status=backend_status if backend_call_attempted else str(exc),
                manual={"manual_notes": "backend_failed_before_save_decision"},
                timings=timings,
                prompt_info=prompt_info,
            )
            write_json(
                case_dir / "perception_backend_summary.json",
                {
                    "backend_call_attempted": backend_call_attempted,
                    "backend_call_completed": backend_call_completed,
                    "perception_outputs_found": perception_outputs_found,
                    "perception_backend_status": backend_status if backend_call_attempted else "not_attempted",
                    "backend_retry_attempted": backend_retry_attempted,
                    "backend_retry_reason": backend_retry_reason,
                    "backend_retry_query": backend_retry_query,
                    "backend_retry_success": backend_retry_success,
                    "material_prompt_engineering": prompt_info,
                    "error": str(exc),
                },
            )
            write_json(
                case_dir / "target_selection_summary.json",
                {
                    "success": False,
                    "status": "failed",
                    "backend_call_attempted": backend_call_attempted,
                    "backend_call_completed": backend_call_completed,
                    "perception_outputs_found": perception_outputs_found,
                    "perception_backend_status": backend_status if backend_call_attempted else "not_attempted",
                    "backend_retry_attempted": backend_retry_attempted,
                    "backend_retry_reason": backend_retry_reason,
                    "backend_retry_query": backend_retry_query,
                    "backend_retry_success": backend_retry_success,
                    "material_prompt_engineering": prompt_info,
                    "error": str(exc),
                    "traceback": trace,
                    "trial": trial.to_dict(),
                    "row": row,
                },
            )
            append_attempt_log(
                session_dir,
                trial,
                attempt_index=attempt_index,
                attempt_status="failed",
                attempt_dir=case_dir,
                backend_status=backend_status if backend_call_attempted else str(exc),
                perception_outputs_found=perception_outputs_found,
                discard_reason=str(exc),
            )
            raise


def run_dry_trial(session_dir: Path, trial: T5V2Trial, config: dict[str, Any]) -> dict[str, Any]:
    base_case_dir = case_dir_for(session_dir, trial)
    for sub in ["attempts", "logs"]:
        (base_case_dir / sub).mkdir(parents=True, exist_ok=True)
    write_json(base_case_dir / "config_snapshot.json", config)
    write_json(base_case_dir / "trial_metadata.json", trial.to_dict())
    attempt_plan = ["discarded", "saved"] if trial.trial_index == 1 else ["saved"]
    saved_payload: dict[str, Any] | None = None
    prompt_info = material_prompt_info(config, trial)

    def write_fake_attempt(attempt_status: str, attempt_index: int) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
        case_dir = attempt_dir_for(base_case_dir, attempt_index)
        for sub in ["logs", "backend_outputs", "candidate_crops", "ros2_snapshot"]:
            (case_dir / sub).mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "config_snapshot.json", config)
        write_json(case_dir / "trial_metadata.json", {**trial.to_dict(), "attempt_index": attempt_index})
        write_json(case_dir / "material_prompt_engineering.json", prompt_info)
        manifest_path = write_prompt_manifest(case_dir, config, trial.input_text)
        if manifest_path != "NA" and Path(manifest_path).exists():
            shutil.copyfile(manifest_path, case_dir / "prompt_manifest.json")

        image = Image.new("RGB", (720, 420), (238, 239, 233))
        draw = ImageDraw.Draw(image)
        draw.text((24, 22), "T5_v2 DRY RUN", fill=(22, 24, 28), font=_font(30, True))
        draw.text((24, 70), f"Trial {trial.trial_index}: {trial.frame_id}", fill=(45, 48, 52), font=_font(18, True))
        draw.text((24, 102), f"Frame classes: {trial.classes_existing_in_frame}", fill=(45, 48, 52), font=_font(18))
        draw.text((24, 134), f"Input text: {trial.input_text}", fill=(45, 48, 52), font=_font(18))
        draw.rectangle((90, 190, 300, 330), fill=(202, 166, 104), outline=(95, 70, 38), width=4)
        draw.rectangle((370, 185, 590, 330), fill=(142, 145, 140), outline=(70, 74, 72), width=4)
        image.save(case_dir / "raw_rgb.png")
        image.save(case_dir / "depth_visualization.png")
        image.save(case_dir / "candidate_masks_overlay.png")
        image.save(case_dir / "candidate_crops_panel.png")
        image.save(case_dir / "material_verification_panel.png")
        brick_mask = np.zeros((420, 720), dtype=np.uint8)
        brick_mask[190:330, 90:300] = 255
        concrete_mask = np.zeros((420, 720), dtype=np.uint8)
        concrete_mask[185:330, 370:590] = 255
        timber_mask = np.zeros((420, 720), dtype=np.uint8)
        timber_mask[188:330, 355:575] = 255
        default_mask = concrete_mask if normalize_material_name(trial.input_text) == "concrete block" else brick_mask
        Image.fromarray(default_mask).save(case_dir / "backend_outputs" / "run_one_image_selected_target_mask.png")
        result = {
            "valid_target": True,
            "selected_label": trial.input_text,
            "requested_class_annotation_count": 1,
            "total_runtime_sec": 0.0,
            "selected_target_mask_path": str(case_dir / "backend_outputs" / "run_one_image_selected_target_mask.png"),
        }
        query_plan = resolve_detection_queries_for_trial(trial, config)
        detection_queries = list(query_plan["detection_queries"])
        candidates: list[dict[str, Any]] = []
        pool_dir = case_dir / "candidate_pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        for query in detection_queries:
            candidate_id = f"candidate_{len(candidates) + 1:03d}_{_slug(query)}"
            crop = Image.new("RGB", (180, 140), (202, 166, 104))
            candidate_mask = brick_mask
            scores_override = {"brick": 0.72, "timber": 0.14, "concrete block": 0.78}
            if query == "concrete block":
                crop = Image.new("RGB", (180, 140), (142, 145, 140))
                candidate_mask = concrete_mask
                scores_override = {"brick": 0.2, "timber": 0.1, "concrete block": 0.7617}
            elif query == "timber":
                crop = Image.new("RGB", (180, 140), (150, 105, 58))
                candidate_mask = timber_mask
                scores_override = {"brick": 0.28, "timber": 0.79, "concrete block": 0.18}
            bbox_crop_path = pool_dir / f"{candidate_id}_bbox_crop.png"
            mask_path = pool_dir / f"{candidate_id}_mask.png"
            crop_path = pool_dir / f"{candidate_id}_masked_texture_crop.png"
            crop.save(bbox_crop_path)
            Image.fromarray(candidate_mask).save(mask_path)
            masked_meta = create_masked_material_crop(case_dir / "raw_rgb.png", mask_path, crop_path)
            candidate_result = {
                **result,
                "selected_label": query,
                "selected_target_mask_path": str(mask_path),
                "selected_candidate_id": candidate_id,
                "selected_source_detection_query": query,
            }
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_detection_query": query,
                    "bbox_crop_path": str(bbox_crop_path),
                    "crop_path": str(bbox_crop_path),
                    "masked_texture_crop_path": masked_meta.get("crop_path", "NA"),
                    "inner_texture_crop_path": masked_meta.get("inner_texture_crop_path", "NA"),
                    "masked_texture_crop_alpha_path": masked_meta.get("alpha_crop_path", "NA"),
                    "masked_texture_crop_metadata": masked_meta,
                    "material_scores_override": scores_override,
                    "mask_path": str(mask_path),
                    "overlay_path": str(case_dir / "selected_mask_overlay.png"),
                    "backend_result": candidate_result,
                }
            )
        dry_config = {**config, "_dry_run": True}
        material_verification = build_material_verification(trial, candidates, dry_config, case_dir=case_dir)
        material_verification.update(
            {
                "experiment_mode": query_plan["experiment_mode"],
                "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
                "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
                "requested_detection_query": query_plan["requested_detection_query"],
                "candidate_generation_source": query_plan["candidate_generation_source"],
            }
        )
        manifest_path = write_prompt_manifest(
            case_dir,
            config,
            trial.input_text,
            candidate_ids=[candidate["candidate_id"] for candidate in candidates],
            prompt_injection_status="prompt_logged_only_clip_fixed_labels",
        )
        if manifest_path != "NA" and Path(manifest_path).exists():
            _copy_if_exists(manifest_path, case_dir / "prompt_manifest.json")
        prompt_manifest = read_json(case_dir / "prompts" / "prompt_manifest.json")
        selected_candidate = material_verification.get("selected_candidate") or {}
        if material_verification.get("no_verified_match"):
            result = {
                "valid_target": False,
                "selected_label": "NO_VERIFIED_MATCH",
                "selected_label_or_NONE": "NO_VERIFIED_MATCH",
                "requested_class_annotation_count": len(candidates),
            }
        elif selected_candidate:
            result = dict(selected_candidate.get("backend_result", result))
            result.update(
                {
                    "valid_target": True,
                    "selected_label": material_verification.get("selected_source_detection_query", trial.input_text),
                    "selected_label_or_NONE": material_verification.get("selected_source_detection_query", trial.input_text),
                    "selected_candidate_id": material_verification.get("selected_candidate_id", "NA"),
                    "selected_source_detection_query": material_verification.get("selected_source_detection_query", "NA"),
                    "selected_candidate_mask_path": material_verification.get("selected_candidate_mask_path", "NA"),
                    "selected_candidate_crop_path": material_verification.get("selected_candidate_crop_path", "NA"),
                    "selected_candidate_masked_texture_crop_path": material_verification.get("selected_candidate_masked_texture_crop_path", "NA"),
                }
            )
        result.update(
            {
                "experiment_mode": query_plan["experiment_mode"],
                "experiment_mode_label": (
                    "ORACLE known-classes candidate pool"
                    if query_plan["oracle_known_classes_used"]
                    else "primary requested-class only"
                ),
                "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
                "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
                "requested_detection_query": query_plan["requested_detection_query"],
                "candidate_generation_source": query_plan["candidate_generation_source"],
            }
        )
        save_candidate_pool_outputs(
            case_dir,
            candidates,
            {
                **material_verification,
                "candidate_pool_enabled": bool(config.get("candidate_pool", {}).get("enabled", False)),
                "candidate_pool_classes": detection_queries,
                "detection_queries_used": detection_queries,
            },
        )
        candidate_crops_panel = save_t5v2_candidate_crops_panel(
            candidates,
            case_dir / "candidate_crops_panel.png",
            query_plan["experiment_mode"],
        )
        selected_overlay, overlay_generated, overlay_reason = generate_selected_mask_overlay(
            case_dir / "raw_rgb.png",
            result,
            case_dir / "selected_mask_overlay.png",
            trial,
            prompt_info,
            "dry_run_backend_not_called",
        )
        backend_image_dir = case_dir / "backend_outputs" / "run_one_image"
        backend_image_dir.mkdir(parents=True, exist_ok=True)
        image.save(backend_image_dir / "dry_run_selected_mask_overlay.png")
        images_all = case_dir / "backend_outputs" / "images_all"
        images_all.mkdir(parents=True, exist_ok=True)
        image.save(images_all / "dry_run_selected_mask_overlay.png")
        snapshot_dir = case_dir / "ros2_snapshot"
        image.save(snapshot_dir / "raw_rgb.png")
        image.save(snapshot_dir / "depth_visualization.png")
        np.save(snapshot_dir / "depth_raw.npy", np.zeros((64, 64), dtype=np.float32))
        write_json(snapshot_dir / "camera_info.json", {"width": 640, "height": 480, "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "dry_run": True})
        write_json(
            snapshot_dir / "rgbd_snapshot_metadata.json",
            {
                "dry_run": True,
                "capture_mode": "ros2_topics",
                "backend_input_mode": "saved_rgb_image",
                "direct_realsense_access": False,
                "color_topic": _capture_config(config)["color_topic"],
                "depth_topic": _capture_config(config)["depth_topic"],
                "camera_info_topic": _capture_config(config)["camera_info_topic"],
            },
        )
        manual = {
            "final_selected_class_manual": "NA",
            "correct_manual": "NA",
            "failure_type_manual": "NA",
            "manual_notes": "dry_run",
        }
        artifacts = {
            "raw_rgb_path": str(case_dir / "raw_rgb.png"),
            "depth_visualization_path": str(case_dir / "depth_visualization.png"),
            "candidate_masks_overlay_path": str(case_dir / "candidate_masks_overlay.png"),
            "candidate_crops_panel_path": candidate_crops_panel,
            "candidate_crops_panel_source": "t5_v2_final_candidate_set",
            "selected_mask_overlay_path": selected_overlay,
            "material_verification_panel_path": str(
                case_dir / "material_verification" / "crop_verification_panel.png"
                if (case_dir / "material_verification" / "crop_verification_panel.png").exists()
                else case_dir / "material_verification_panel.png"
            ),
            "selected_mask_overlay_generated": overlay_generated,
            "selected_mask_overlay_has_transparent_mask": overlay_generated,
            "selected_mask_overlay_mask_path": _find_selected_mask_path(result, case_dir),
            "selected_mask_overlay_failure_reason": "NA" if overlay_generated else overlay_reason,
            "selected_mask_overlay_layout": "header_panel_above_rgb_image",
            "experiment_mode": query_plan["experiment_mode"],
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
            "candidate_pool_classes": detection_queries,
            "detection_queries_used": detection_queries,
            "candidate_pool_enabled": bool(config.get("candidate_pool", {}).get("enabled", False)),
            "material_verification": material_verification,
        }
        timings = {"created_at": now_iso(), "completed_at": now_iso(), "total_time_s": 0.0, "visualization_save_time_s": 0.0}
        row = build_v2_row(
            trial,
            case_dir,
            status="saved" if attempt_status == "saved" else "discarded",
            backend_status="dry_run_backend_not_called",
            result=result,
            artifacts=artifacts,
            manual=manual,
            timings=timings,
            prompt_info=prompt_info,
        )
        write_json(case_dir / "manual_label.json", manual)
        write_json(
            case_dir / "perception_backend_summary.json",
            {
                "dry_run": True,
                "backend_call_attempted": True,
                "backend_call_completed": True,
                "perception_outputs_found": True,
                "perception_backend_status": "dry_run_backend_not_called",
                "backend": "scripts.run_one_image.run_integrated_case",
                "backend_not_called": True,
                "backend_images_copied": [str(images_all / "dry_run_selected_mask_overlay.png")],
                "material_prompt_engineering": prompt_info,
                "candidate_pool": {
                    "candidate_pool_enabled": bool(config.get("candidate_pool", {}).get("enabled", False)),
                    **query_plan,
                    "candidate_pool_classes": detection_queries,
                    "detection_queries_used": detection_queries,
                    "candidate_count": len(candidates),
                },
                "material_verification": material_verification,
            },
        )
        write_json(case_dir / "timing_summary.json", timings)
        summary = {
            "success": True,
            "dry_run": True,
            "status": attempt_status,
            "backend_call_attempted": True,
            "backend_call_completed": True,
            "perception_outputs_found": True,
            "perception_backend_status": "dry_run_backend_not_called",
            "prompt_registry_version": prompt_manifest.get("prompt_registry_version", "NA"),
            "active_prompt_stages": prompt_manifest.get("active_prompt_stages", []),
            "inactive_prompt_stages": prompt_manifest.get("inactive_prompt_stages", []),
            "open_vocab_detection_prompt": (prompt_manifest.get("stages", {}).get("open_vocab_detection", {}) or {}).get("prompt", "NA"),
            "material_crop_verification_prompt_path": (prompt_manifest.get("stages", {}).get("material_crop_verification", {}) or {}).get("prompt_text_path", "NA"),
            "edge_contact_quality_prompt_path": (prompt_manifest.get("stages", {}).get("edge_contact_quality", {}) or {}).get("prompt_text_path", "NA"),
            "final_overlay_label_source": "t5_v2_final_decision",
            "legacy_backend_label_ignored_for_overlay": True,
            "material_prompt_engineering": prompt_info,
            "experiment_mode": query_plan["experiment_mode"],
            "primary_requested_class_only": query_plan["experiment_mode"] == "primary_requested_class_only",
            "oracle_known_classes_used": query_plan["oracle_known_classes_used"],
            "classes_existing_used_as_model_input": query_plan["classes_existing_used_as_model_input"],
            "requested_detection_query": query_plan["requested_detection_query"],
            "candidate_generation_source": query_plan["candidate_generation_source"],
            "material_verification_candidate_count": len(material_verification.get("candidates", [])),
            "legacy_backend_artifacts_copied": {
                "legacy_backend_candidate_crops_panel": False,
                "legacy_backend_candidate_masks_overlay": False,
            },
            "candidate_crops_panel_source": "t5_v2_final_candidate_set",
            "candidate_pool_enabled": bool(config.get("candidate_pool", {}).get("enabled", False)),
            "candidate_pool_classes": detection_queries,
            "requested_class": normalize_material_name(trial.input_text),
            "detection_queries_used": detection_queries,
            "final_selected_candidate_id": material_verification.get("selected_candidate_id", "NA"),
            "final_selected_class_backend": result.get("selected_label", "NA"),
            "material_verification_method": material_verification.get("material_verification_method", "NA"),
            "material_scores": material_verification.get("material_scores", {}),
            "crop_source_used_for_material_verification": material_verification.get("crop_source_used_for_material_verification", "NA"),
            "masked_texture_crops_generated": material_verification.get("masked_texture_crops_generated", False),
            "source_class_prior_used": material_verification.get("source_class_prior_used", False),
            "same_source_preferred": material_verification.get("same_source_preferred", False),
            "selected_candidate_id_before_prior": material_verification.get("selected_candidate_id_before_prior", "NA"),
            "selected_candidate_id_after_prior": material_verification.get("selected_candidate_id_after_prior", "NA"),
            "selected_source_detection_query": material_verification.get("selected_source_detection_query", "NA"),
            "same_source_candidate_available": material_verification.get("same_source_candidate_available", False),
            "same_source_candidate_score": material_verification.get("same_source_candidate_score", "NA"),
            "cross_source_override_used": material_verification.get("cross_source_override_used", False),
            "cross_source_override_margin": material_verification.get("cross_source_override_margin", "NA"),
            "no_verified_match": material_verification.get("no_verified_match", False),
            "beige_brick_guard_triggered": material_verification.get("beige_brick_guard_triggered", False),
            "beige_brick_guard_reason": material_verification.get("beige_brick_guard_reason", "NA"),
            "concrete_color_texture_guard_triggered": material_verification.get("concrete_color_texture_guard_triggered", False),
            "concrete_color_texture_guard_reason": material_verification.get("concrete_color_texture_guard_reason", "NA"),
            "beige_brick_penalty_applied": material_verification.get("beige_brick_penalty_applied", False),
            "grey_score": material_verification.get("grey_score", "NA"),
            "warm_beige_score": material_verification.get("warm_beige_score", "NA"),
            "selected_candidate_mask_path": material_verification.get("selected_candidate_mask_path", "NA"),
            "selected_candidate_crop_path": material_verification.get("selected_candidate_crop_path", "NA"),
            "selected_candidate_masked_texture_crop_path": material_verification.get("selected_candidate_masked_texture_crop_path", "NA"),
            "selected_mask_overlay_generated": overlay_generated,
            "selected_mask_overlay_has_transparent_mask": overlay_generated,
            "selected_mask_overlay_failure_reason": "NA" if overlay_generated else overlay_reason,
            "selected_mask_overlay_layout": artifacts["selected_mask_overlay_layout"],
            "trial": trial.to_dict(),
            "row": row,
            "artifacts": artifacts,
        }
        if attempt_status == "discarded":
            write_json(case_dir / "discarded.json", {"discarded": True, "discard_reason": "dry_run_simulated_discard", "discarded_at": now_iso()})
            summary["discard_reason"] = "dry_run_simulated_discard"
        write_json(case_dir / "target_selection_summary.json", summary)
        (case_dir / "logs" / "backend_stdout_stderr.log").write_text("T5_v2 dry-run; perception backend was not called.\n", encoding="utf-8")
        append_attempt_log(
            session_dir,
            trial,
            attempt_index=attempt_index,
            attempt_status=attempt_status,
            attempt_dir=case_dir,
            backend_status="dry_run_backend_not_called",
            perception_outputs_found=True,
            discard_reason="dry_run_simulated_discard" if attempt_status == "discarded" else "NA",
        )
        return case_dir, row, artifacts, manual

    for idx, status in enumerate(attempt_plan, start=1):
        attempt_dir, row, artifacts, manual = write_fake_attempt(status, idx)
        if status == "saved":
            _copy_paper_figures(session_dir, trial, artifacts)
            write_json(base_case_dir / "manual_label.json", manual)
            write_json(base_case_dir / "trial_result.json", {"saved_attempt_dir": str(attempt_dir), "row": row})
            write_json(base_case_dir / "target_selection_summary.json", read_json(attempt_dir / "target_selection_summary.json"))
            _copy_if_exists(artifacts.get("raw_rgb_path"), base_case_dir / "raw_rgb.png")
            _copy_if_exists(artifacts.get("selected_mask_overlay_path"), base_case_dir / "selected_mask_overlay.png")
            (base_case_dir / "saved_attempt.txt").write_text(str(attempt_dir), encoding="utf-8")
            write_master_row(session_dir, row)
            saved_payload = {"success": True, "dry_run": True, "row": row, "artifacts": artifacts, "attempt_dir": str(attempt_dir)}
    return saved_payload or {"success": False, "dry_run": True}


def _copy_paper_figures(session_dir: Path, trial: T5V2Trial, artifacts: dict[str, Any]) -> None:
    paper_dir = session_dir / "paper_figures"
    prefix = f"trial_{trial.trial_index:03d}_{_safe_component(trial.frame_id)}_{_slug(trial.input_text)}"
    mapping = {
        "selected_mask_overlay_path": f"{prefix}_selected_mask_overlay.png",
        "candidate_crops_panel_path": f"{prefix}_candidate_crops_panel.png",
        "candidate_masks_overlay_path": f"{prefix}_candidate_masks_overlay.png",
        "material_verification_panel_path": f"{prefix}_material_verification_panel.png",
    }
    for key, name in mapping.items():
        path = artifacts.get(key)
        if path and path != "NA" and Path(path).exists():
            shutil.copyfile(path, paper_dir / name)


def write_session_summary(session_dir: Path, config: dict[str, Any], cases: list[T5V2Trial]) -> None:
    rows = _read_master_rows(session_dir)
    completed = [row for row in rows if row.get("status") in {"completed", "saved"}]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    correct = [row for row in rows if row.get("correct_manual") == "yes"]
    incorrect = [row for row in rows if row.get("correct_manual") == "no"]
    summary = {
        "experiment_name": config.get("experiment_name", "experiment_1_open_vocab_multi_material_target_selection_fixed27"),
        "session_dir": str(session_dir),
        "created_at": now_iso(),
        "enabled_trials_expected": len([case for case in cases if case.enabled]),
        "completed_trials": len(completed),
        "skipped_trials": len(skipped),
        "correct_manual_count": len(correct),
        "incorrect_manual_count": len(incorrect),
        "target_selection_success_rate_manual": "NA" if not (correct or incorrect) else len(correct) / max(1, len(correct) + len(incorrect)),
        "backend": "scripts.run_one_image.run_integrated_case",
        "robot_used": False,
        "clamp_used": False,
        "rtde_used": False,
        "arduino_used": False,
        "upv_measurement_used": False,
    }
    write_json(session_dir / "T5_v2_session_summary.json", summary)
