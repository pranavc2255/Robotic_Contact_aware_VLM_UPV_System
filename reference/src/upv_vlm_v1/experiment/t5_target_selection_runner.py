from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]


T5_COLUMNS = [
    "case_id",
    "scene_id",
    "layout_id",
    "material_query",
    "objects_present",
    "expected_material",
    "final_selected_material",
    "final_selected_object_id",
    "target_selected",
    "target_selection_success",
    "grounding_mask_available",
    "grounding_mask_correct_manual",
    "crop_verification_prediction",
    "crop_verification_success_manual",
    "final_selected_correct_manual",
    "failure_type_manual",
    "manual_notes",
    "raw_rgb_path",
    "candidate_masks_overlay_path",
    "candidate_crops_panel_path",
    "selected_mask_overlay_path",
    "backend_output_dir",
    "total_time_s",
    "camera_capture_time_s",
    "open_vocab_detection_time_s",
    "segmentation_time_s",
    "crop_verification_time_s",
    "final_selection_time_s",
    "visualization_save_time_s",
]


@dataclass
class T5Case:
    case_id: str
    scene_id: str
    layout_id: str
    material_query: str
    objects_present: str
    specimen_ids: str = "NA"
    placement_note: str = "NA"
    expected_material: str = "NA"
    enabled: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "T5Case":
        return cls(
            case_id=str(row.get("case_id") or "case_001"),
            scene_id=str(row.get("scene_id") or "scene_01"),
            layout_id=str(row.get("layout_id") or "layout_01"),
            material_query=str(row.get("material_query") or "brick"),
            objects_present=str(row.get("objects_present") or row.get("material_query") or "brick"),
            specimen_ids=str(row.get("specimen_ids") or "NA"),
            placement_note=str(row.get("placement_note") or "NA"),
            expected_material=str(row.get("expected_material") or row.get("material_query") or "NA"),
            enabled=str(row.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "n"},
        )

    def folder_name(self) -> str:
        return f"{_slug(self.case_id)}_{_slug(self.material_query)}_{_slug(self.scene_id)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scene_id": self.scene_id,
            "layout_id": self.layout_id,
            "material_query": self.material_query,
            "objects_present": self.objects_present,
            "specimen_ids": self.specimen_ids,
            "placement_note": self.placement_note,
            "expected_material": self.expected_material,
            "enabled": self.enabled,
        }


def now_stamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _slug(text: Any) -> str:
    value = str(text or "NA").strip().lower()
    out = [ch if ch.isalnum() else "_" for ch in value]
    return "_".join("".join(out).split("_")) or "item"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_session_dir(output_root: str | Path) -> Path:
    root = resolve_repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(100):
        name = f"session_{now_stamp()}" if suffix == 0 else f"session_{now_stamp()}_{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            (candidate / "cases").mkdir()
            (candidate / "paper_figures").mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique T5 session under {root}")


def read_cases_csv(path: str | Path) -> list[T5Case]:
    path = resolve_repo_path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [T5Case.from_row(row) for row in csv.DictReader(handle)]


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    fieldnames = list(T5_COLUMNS)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        for old in existing:
            fieldnames = list(dict.fromkeys([*fieldnames, *old.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing:
            writer.writerow({key: old.get(key, "NA") for key in fieldnames})
        writer.writerow({key: row.get(key, "NA") for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_if_exists(src: Any, dst: Path) -> str:
    if not src:
        return "NA"
    path = Path(str(src))
    if not path.exists() or not path.is_file():
        return "NA"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.resolve() == dst.resolve():
            return str(dst)
        shutil.copyfile(path, dst)
    except shutil.SameFileError:
        return str(dst)
    return str(dst)


def _save_depth_visualization(depth_path: Path | None, output_path: Path) -> str:
    if depth_path is None or not depth_path.exists():
        return "NA"
    try:
        import cv2  # noqa: PLC0415

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return "NA"
        valid = depth[depth > 0]
        if valid.size == 0:
            return "NA"
        low, high = np.percentile(valid, [2, 98])
        scaled = np.clip((depth.astype(np.float32) - low) / max(high - low, 1.0), 0, 1)
        vis = (scaled * 255).astype(np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_path), color)
        return str(output_path)
    except Exception:
        return "NA"


def _candidate_annotations(result: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from upv_vlm_orient.integration.gsam2_results import load_gsam2_results  # noqa: PLC0415

        results_path = Path(str(result.get("gsam2_results_path", "")))
        if not results_path.exists():
            return []
        annotations = load_gsam2_results(str(results_path)).get("annotations")
        return annotations if isinstance(annotations, list) else []
    except Exception:
        return []


def _decode_mask(annotation: dict[str, Any]) -> np.ndarray | None:
    try:
        from upv_vlm_orient.integration.gsam2_results import decode_coco_rle_mask  # noqa: PLC0415

        if "segmentation" not in annotation:
            return None
        return decode_coco_rle_mask(annotation["segmentation"])
    except Exception:
        return None


def _score(annotation: dict[str, Any]) -> float:
    value = annotation.get("detection_score", annotation.get("score", 0.0))
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def save_candidate_masks_overlay(rgb_path: Path, result: dict[str, Any], output_path: Path) -> str:
    annotations = sorted(_candidate_annotations(result), key=_score, reverse=True)[:8]
    if not annotations:
        return "NA"
    base = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    palette = [
        (230, 70, 70, 80),
        (60, 170, 240, 80),
        (75, 205, 120, 80),
        (240, 190, 55, 80),
        (180, 105, 230, 80),
        (245, 120, 50, 80),
    ]
    draw = ImageDraw.Draw(overlay)
    for idx, ann in enumerate(annotations, start=1):
        mask = _decode_mask(ann)
        if mask is None:
            continue
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask > 0] = np.array(palette[(idx - 1) % len(palette)], dtype=np.uint8)
        mask_img = Image.fromarray(rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_img)
        ys, xs = np.nonzero(mask > 0)
        if xs.size:
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            draw.rectangle((x0, y0, x1, y1), outline=palette[(idx - 1) % len(palette)][:3] + (255,), width=4)
            draw.text((x0 + 6, max(0, y0 - 26)), f"C{idx} {ann.get('class_name', '')}", fill=(255, 255, 255, 255), font=_font(18, True))
    canvas = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def save_candidate_crops_panel(rgb_path: Path, result: dict[str, Any], case_dir: Path, output_path: Path) -> str:
    annotations = sorted(_candidate_annotations(result), key=_score, reverse=True)[:8]
    if not annotations:
        return "NA"
    image = Image.open(rgb_path).convert("RGB")
    tiles: list[Image.Image] = []
    crops_dir = case_dir / "candidate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for idx, ann in enumerate(annotations, start=1):
        mask = _decode_mask(ann)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            continue
        pad = 24
        x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        x1, y1 = min(image.width, int(xs.max()) + pad), min(image.height, int(ys.max()) + pad)
        crop = image.crop((x0, y0, x1, y1))
        crop_path = crops_dir / f"candidate_C{idx:02d}_crop.png"
        crop.save(crop_path)
        tile = Image.new("RGB", (300, 260), (242, 243, 238))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 299, 58), fill=(36, 39, 43))
        draw.text((10, 8), f"C{idx}: {ann.get('class_name', 'NA')}", fill=(255, 255, 255), font=_font(16, True))
        draw.text((10, 34), f"score={_score(ann):.3f}", fill=(235, 235, 235), font=_font(13))
        crop.thumbnail((280, 182))
        tile.paste(crop, ((300 - crop.width) // 2, 68 + (182 - crop.height) // 2))
        tiles.append(tile)
    if not tiles:
        return "NA"
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * 300, rows * 260), (225, 226, 220))
    for idx, tile in enumerate(tiles):
        panel.paste(tile, ((idx % cols) * 300, (idx // cols) * 260))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return str(output_path)


def save_material_verification_panel(result: dict[str, Any], output_path: Path) -> str:
    crop_info = result.get("crop_info") if isinstance(result.get("crop_info"), dict) else {}
    crop_path = crop_info.get("crop_path") or result.get("crop_path")
    if not crop_path or not Path(str(crop_path)).exists():
        return "NA"
    crop = Image.open(crop_path).convert("RGB")
    canvas = Image.new("RGB", (720, 360), (245, 246, 241))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Material / crop verification", fill=(20, 22, 24), font=_font(24, True))
    crop.thumbnail((300, 260))
    canvas.paste(crop, (28, 78))
    verification = result.get("crop_verification") if isinstance(result.get("crop_verification"), dict) else {}
    lines = [
        f"selected_label: {result.get('selected_label', 'NA')}",
        f"valid_target: {result.get('valid_target', 'NA')}",
        f"crop_top_label: {verification.get('top_label', result.get('crop_verifier_top_label', 'NA'))}",
        f"crop_top_score: {verification.get('top_score', result.get('crop_verifier_top_score', 'NA'))}",
        f"requested_annotations: {result.get('requested_class_annotation_count', 'NA')}",
    ]
    y = 84
    for line in lines:
        draw.text((360, y), line, fill=(35, 38, 42), font=_font(16))
        y += 34
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def write_prompt_manifest(case_dir: Path, config: dict[str, Any], material_query: str) -> str:
    prompt_config = config.get("canonical_prompt_config")
    if not prompt_config:
        return "NA"
    try:
        from upv_vlm_v1.prompts.prompt_registry import load_prompt_bundle, write_prompt_run_package  # noqa: PLC0415

        bundle = load_prompt_bundle(resolve_repo_path(prompt_config))
        package = write_prompt_run_package(
            run_dir=case_dir,
            bundle=bundle,
            material_query=material_query,
            axis_mode="major",
            pipeline_mode="t5_target_selection",
            operator_name=config.get("default_operator", "Pranav"),
        )
        return str(case_dir / "prompts" / "prompt_manifest.json") if package else "NA"
    except Exception:
        return "NA"


def capture_live_rgbd(case_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    from terminal_scripts.run_r1a_realsense_capture_once import capture_once  # noqa: PLC0415

    camera = config.get("camera", {})
    capture_root = case_dir / "capture"
    args = SimpleNamespace(
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        fps=int(camera.get("fps", 30)),
        warmup=int(camera.get("warmup", 30)),
        frame_timeout_ms=int(float(camera.get("timeout_s", 10.0)) * 1000),
        frame_wait_attempts=int(camera.get("frame_wait_attempts", 3)),
        startup_retry_count=int(camera.get("startup_retry_count", 2)),
        out_root=str(capture_root),
    )
    run_dir = capture_once(args)
    return {
        "capture_dir": str(run_dir),
        "color_path": str(run_dir / "color.png"),
        "depth_png_path": str(run_dir / "depth_aligned.png"),
        "depth_npy_path": str(run_dir / "depth_aligned.npy"),
        "intrinsics_path": str(run_dir / "camera_intrinsics.json"),
    }


def run_real_perception(case_dir: Path, image_path: Path, depth_path: Path | None, case: T5Case, config: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_one_image import run_integrated_case  # noqa: PLC0415

    backend_dir = case_dir / "backend_outputs" / "run_one_image"
    candidates = case.objects_present.replace(";", ",")
    mode = config.get("perception_backend", {}).get("run_one_image_mode", "requested_class_best_instance_then_crop_verify")
    crop_verify = bool(config.get("perception_backend", {}).get("crop_verify", True))
    return run_integrated_case(
        image_path_value=str(image_path),
        requested_class_name=case.material_query,
        mode=mode,
        candidates=candidates,
        output_dir=str(backend_dir),
        depth_path_value=str(depth_path or "NA"),
        crop_verify=crop_verify,
    )


def build_result_row(
    case: T5Case,
    case_dir: Path,
    result: dict[str, Any],
    artifacts: dict[str, Any],
    manual: dict[str, Any],
    timings: dict[str, Any],
) -> dict[str, Any]:
    selected_material = result.get("selected_label") or result.get("selected_label_or_NONE") or "NA"
    selected_id = result.get("selected_candidate_id") or result.get("winning_candidate") or selected_material
    target_selected = bool(result.get("valid_target", False))
    manual_correct = manual.get("final_selected_correct_manual", "NA")
    if manual_correct in {"yes", "y", True}:
        target_success: Any = True
    elif manual_correct in {"no", "n", False}:
        target_success = False
    else:
        target_success = target_selected and selected_material not in {"NONE", "NA", None}
    crop_verification = result.get("crop_verification") if isinstance(result.get("crop_verification"), dict) else {}
    return {
        "case_id": case.case_id,
        "scene_id": case.scene_id,
        "layout_id": case.layout_id,
        "material_query": case.material_query,
        "objects_present": case.objects_present,
        "expected_material": case.expected_material,
        "final_selected_material": selected_material,
        "final_selected_object_id": selected_id,
        "target_selected": target_selected,
        "target_selection_success": target_success,
        "grounding_mask_available": bool(result.get("selected_target_mask_path")),
        "grounding_mask_correct_manual": manual.get("grounding_mask_correct_manual", "NA"),
        "crop_verification_prediction": crop_verification.get("top_label", result.get("crop_verifier_top_label", "NA")),
        "crop_verification_success_manual": manual.get("crop_verification_success_manual", "NA"),
        "final_selected_correct_manual": manual.get("final_selected_correct_manual", "NA"),
        "failure_type_manual": manual.get("failure_type_manual", "NA"),
        "manual_notes": manual.get("manual_notes", "NA"),
        "raw_rgb_path": artifacts.get("raw_rgb_path", "NA"),
        "candidate_masks_overlay_path": artifacts.get("candidate_masks_overlay_path", "NA"),
        "candidate_crops_panel_path": artifacts.get("candidate_crops_panel_path", "NA"),
        "selected_mask_overlay_path": artifacts.get("selected_mask_overlay_path", "NA"),
        "backend_output_dir": str(case_dir / "backend_outputs"),
        "total_time_s": timings.get("total_time_s", result.get("total_runtime_sec", "NA")),
        "camera_capture_time_s": timings.get("camera_capture_time_s", "NA"),
        "open_vocab_detection_time_s": result.get("gsam2_runtime_sec", "NA"),
        "segmentation_time_s": result.get("pipeline_runtime_sec", "NA"),
        "crop_verification_time_s": result.get("crop_verifier_runtime_sec", "NA"),
        "final_selection_time_s": timings.get("final_selection_time_s", "NA"),
        "visualization_save_time_s": timings.get("visualization_save_time_s", "NA"),
    }


def prompt_manual_label(no_manual: bool) -> dict[str, Any]:
    if no_manual:
        return {
            "final_selected_correct_manual": "NA",
            "grounding_mask_correct_manual": "NA",
            "crop_verification_success_manual": "NA",
            "failure_type_manual": "NA",
            "manual_selected_material": "NA",
            "manual_notes": "NA",
        }
    answer = input("Manual final selected object correct? [y/n/skip]: ").strip().lower()
    if answer in {"y", "yes"}:
        correct = "yes"
    elif answer in {"n", "no"}:
        correct = "no"
    else:
        correct = "NA"
    manual_material = input("Manual selected material/class: ").strip() or "NA"
    failure = input("Failure type [none/wrong_mask/wrong_class/no_detection/ambiguous/other]: ").strip() or "NA"
    notes = input("Notes: ").strip() or "NA"
    return {
        "final_selected_correct_manual": correct,
        "grounding_mask_correct_manual": "NA",
        "crop_verification_success_manual": "NA",
        "failure_type_manual": failure,
        "manual_selected_material": manual_material,
        "manual_notes": notes,
    }


def run_case(
    session_dir: Path,
    case: T5Case,
    config: dict[str, Any],
    *,
    live: bool = False,
    image_path: Path | None = None,
    no_manual: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    case_dir = session_dir / "cases" / case.folder_name()
    for sub in ["logs", "backend_outputs", "candidate_crops"]:
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
    (case_dir / "logs" / "backend_stdout_stderr.log").write_text(
        "T5 calls scripts.run_one_image.run_integrated_case in-process; no subprocess stdout/stderr is produced.\n",
        encoding="utf-8",
    )
    _write_json(case_dir / "config_snapshot.json", config)
    _write_json(case_dir / "case_metadata.json", case.to_dict())
    write_prompt_manifest(case_dir, config, case.material_query)

    if dry_run:
        return run_dry_case(session_dir, case, config)

    total_start = time.perf_counter()
    capture_time = 0.0
    depth_path: Path | None = None
    try:
        if live:
            capture_start = time.perf_counter()
            capture = capture_live_rgbd(case_dir, config)
            capture_time = time.perf_counter() - capture_start
            rgb_path = Path(capture["color_path"])
            depth_path = Path(capture["depth_png_path"])
        elif image_path is not None:
            rgb_path = image_path
        else:
            raise ValueError("T5 needs either --live or --image for non-dry execution.")

        raw_rgb_copy = _copy_if_exists(rgb_path, case_dir / "raw_rgb.png")
        depth_vis = _save_depth_visualization(depth_path, case_dir / "depth_visualization.png")
        result = run_real_perception(case_dir, rgb_path, depth_path, case, config)

        vis_start = time.perf_counter()
        candidate_masks = save_candidate_masks_overlay(rgb_path, result, case_dir / "candidate_masks_overlay.png")
        candidate_crops = save_candidate_crops_panel(rgb_path, result, case_dir, case_dir / "candidate_crops_panel.png")
        selected_overlay = _copy_if_exists(result.get("overlay_path"), case_dir / "selected_mask_overlay.png")
        material_panel = save_material_verification_panel(result, case_dir / "material_verification_panel.png")
        visualization_time = time.perf_counter() - vis_start

        artifacts = {
            "raw_rgb_path": raw_rgb_copy,
            "depth_visualization_path": depth_vis,
            "candidate_masks_overlay_path": candidate_masks,
            "candidate_crops_panel_path": candidate_crops,
            "selected_mask_overlay_path": selected_overlay,
            "material_verification_panel_path": material_panel,
        }
        manual = prompt_manual_label(no_manual)
        timings = {
            "total_time_s": time.perf_counter() - total_start,
            "camera_capture_time_s": capture_time if live else "NA",
            "visualization_save_time_s": visualization_time,
            "final_selection_time_s": result.get("pipeline_runtime_sec", "NA"),
        }
        row = build_result_row(case, case_dir, result, artifacts, manual, timings)
        summary = {
            "case": case.to_dict(),
            "success": True,
            "perception_backend": "scripts.run_one_image.run_integrated_case",
            "backend_output_dir": str(case_dir / "backend_outputs" / "run_one_image"),
            "result": _json_safe_result(result),
            "artifacts": artifacts,
            "row": row,
        }
        _write_json(case_dir / "perception_backend_summary.json", _json_safe_result(result))
        _write_json(case_dir / "target_selection_summary.json", summary)
        _write_json(case_dir / "timing_summary.json", timings)
        _write_json(case_dir / "manual_label.json", manual)
        _copy_paper_figures(session_dir, case, artifacts)
        write_csv_row(session_dir / "T5_master_results.csv", row)
        return summary
    except Exception as exc:
        trace = traceback.format_exc()
        (case_dir / "logs" / "error_trace.txt").write_text(trace, encoding="utf-8")
        manual = prompt_manual_label(no_manual)
        row = build_result_row(
            case,
            case_dir,
            {"valid_target": False, "selected_label": "NA"},
            {},
            manual,
            {"total_time_s": time.perf_counter() - total_start, "camera_capture_time_s": capture_time},
        )
        row["target_selection_success"] = False
        row["failure_type_manual"] = manual.get("failure_type_manual", "runtime_error")
        _write_json(case_dir / "manual_label.json", manual)
        _write_json(case_dir / "target_selection_summary.json", {"success": False, "error": str(exc), "traceback": trace, "case": case.to_dict(), "row": row})
        write_csv_row(session_dir / "T5_master_results.csv", row)
        raise


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, Path):
            safe[key] = str(value)
        elif isinstance(value, np.ndarray):
            safe[key] = f"ndarray shape={value.shape}"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                safe[key] = value
            except TypeError:
                safe[key] = str(value)
        else:
            safe[key] = str(value)
    return safe


def _copy_paper_figures(session_dir: Path, case: T5Case, artifacts: dict[str, Any]) -> None:
    paper_dir = session_dir / "paper_figures"
    mapping = {
        "selected_mask_overlay_path": f"{case.case_id}_selected_mask_overlay.png",
        "candidate_crops_panel_path": f"{case.case_id}_candidate_crops_panel.png",
        "candidate_masks_overlay_path": f"{case.case_id}_candidate_masks_overlay.png",
        "material_verification_panel_path": f"{case.case_id}_material_verification_panel.png",
    }
    for key, name in mapping.items():
        path = artifacts.get(key)
        if path and path != "NA" and Path(path).exists():
            shutil.copyfile(path, paper_dir / name)


def run_dry_case(session_dir: Path, case: T5Case, config: dict[str, Any]) -> dict[str, Any]:
    case_dir = session_dir / "cases" / case.folder_name()
    for sub in ["logs", "backend_outputs", "candidate_crops"]:
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
    (case_dir / "logs" / "backend_stdout_stderr.log").write_text(
        "T5 dry-run placeholder; perception backend was not called.\n",
        encoding="utf-8",
    )
    image = Image.new("RGB", (640, 420), (235, 237, 232))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "T5 DRY RUN PLACEHOLDER", fill=(20, 22, 24), font=_font(28, True))
    draw.rectangle((120, 130, 380, 280), fill=(170, 90, 55), outline=(80, 45, 30), width=4)
    draw.text((130, 290), f"query={case.material_query}", fill=(20, 22, 24), font=_font(18))
    raw_path = case_dir / "raw_rgb.png"
    image.save(raw_path)
    for name in ["candidate_masks_overlay.png", "candidate_crops_panel.png", "selected_mask_overlay.png", "material_verification_panel.png"]:
        image.save(case_dir / name)
    manual = {
        "final_selected_correct_manual": "NA",
        "grounding_mask_correct_manual": "NA",
        "crop_verification_success_manual": "NA",
        "failure_type_manual": "NA",
        "manual_notes": "dry_run",
    }
    result = {
        "valid_target": True,
        "selected_label": case.material_query,
        "total_runtime_sec": 0.0,
        "gsam2_runtime_sec": "NA",
        "pipeline_runtime_sec": "NA",
    }
    artifacts = {
        "raw_rgb_path": str(raw_path),
        "candidate_masks_overlay_path": str(case_dir / "candidate_masks_overlay.png"),
        "candidate_crops_panel_path": str(case_dir / "candidate_crops_panel.png"),
        "selected_mask_overlay_path": str(case_dir / "selected_mask_overlay.png"),
        "material_verification_panel_path": str(case_dir / "material_verification_panel.png"),
    }
    timings = {"total_time_s": 0.0, "camera_capture_time_s": "NA", "visualization_save_time_s": 0.0, "final_selection_time_s": "NA"}
    row = build_result_row(case, case_dir, result, artifacts, manual, timings)
    _write_json(case_dir / "config_snapshot.json", config)
    _write_json(case_dir / "manual_label.json", manual)
    _write_json(case_dir / "perception_backend_summary.json", {"dry_run": True, "backend_not_called": True})
    _write_json(case_dir / "timing_summary.json", timings)
    _write_json(case_dir / "target_selection_summary.json", {"success": True, "dry_run": True, "case": case.to_dict(), "row": row, "artifacts": artifacts})
    _copy_paper_figures(session_dir, case, artifacts)
    write_csv_row(session_dir / "T5_master_results.csv", row)
    return {"success": True, "dry_run": True, "case": case.to_dict(), "row": row, "artifacts": artifacts}


def write_session_summary(session_dir: Path, config: dict[str, Any], cases: list[T5Case]) -> None:
    rows = []
    csv_path = session_dir / "T5_master_results.csv"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    summary = {
        "experiment_name": config.get("experiment_name", "open_vocab_multi_material_target_selection"),
        "session_dir": str(session_dir),
        "created_at": now_iso(),
        "case_count_planned": len(cases),
        "case_count_written": len(rows),
        "backend": "scripts.run_one_image.run_integrated_case",
        "robot_used": False,
        "clamp_used": False,
        "rtde_used": False,
        "arduino_used": False,
        "upv_measurement_used": False,
    }
    _write_json(session_dir / "T5_session_summary.json", summary)
