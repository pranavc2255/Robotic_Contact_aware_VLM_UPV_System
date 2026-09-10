"""GroundingDINO + SAM2 integration for v2 target selection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _resolve_existing_path(path_value: str | Path, *, primary_base: Path, fallback_base: Path | None = None) -> Path:
    primary = _resolve_path(path_value, base=primary_base)
    if primary.exists() or fallback_base is None:
        return primary
    fallback = _resolve_path(path_value, base=fallback_base)
    return fallback if fallback.exists() else primary


def _sam2_hydra_config_name(value: str) -> str:
    raw = str(value).strip()
    marker = "configs/sam2."
    if marker in raw:
        return raw[raw.index(marker) :]
    return raw


def _score_value(value: Any) -> float | None:
    if isinstance(value, list):
        return _score_value(value[0]) if value else None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _decode_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError("pycocotools is required to decode Grounded-SAM2 mask RLE results") from exc
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype("uint8")


def _build_wrapped_command(
    *,
    python_executable: str,
    demo_script: Path,
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    detector_model_id: str,
    sam2_checkpoint: str,
    sam2_config: str,
) -> list[str]:
    wrapper = """
import json
from pathlib import Path
import runpy
import sys
demo_script, detector_model_id, image_path, text_prompt, output_dir, sam2_checkpoint, sam2_model_config = sys.argv[1:8]
sys.argv = [
    demo_script,
    "--grounding-model", detector_model_id,
    "--img-path", image_path,
    "--text-prompt", text_prompt,
    "--output-dir", output_dir,
    "--sam2-checkpoint", sam2_checkpoint,
    "--sam2-model-config", sam2_model_config,
]
globals_dict = runpy.run_path(demo_script, run_name="__main__")
metadata = {
    "class_names": [str(x) for x in list(globals_dict.get("class_names", []))],
    "confidences": [float(x) for x in list(globals_dict.get("confidences", []))],
}
Path(output_dir).mkdir(parents=True, exist_ok=True)
(Path(output_dir) / "v2_gsam2_score_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
"""
    return [
        python_executable,
        "-c",
        wrapper,
        str(demo_script),
        detector_model_id,
        str(image_path),
        text_prompt,
        str(output_dir),
        sam2_checkpoint,
        sam2_config,
    ]


def _load_results(results_json: Path) -> list[dict[str, Any]]:
    if not results_json.exists():
        raise FileNotFoundError(f"Grounded-SAM2 results JSON not found: {results_json}")
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        return []
    detections: list[dict[str, Any]] = []
    for idx, ann in enumerate(annotations, start=1):
        seg = ann.get("segmentation")
        if not isinstance(seg, dict):
            continue
        mask = _decode_coco_rle(seg)
        detections.append(
            {
                "annotation_index": idx,
                "class_name": ann.get("class_name"),
                "bbox": ann.get("bbox"),
                "score": _score_value(ann.get("score")),
                "detection_score": _score_value(ann.get("detection_score")),
                "mask_score": _score_value(ann.get("mask_score", ann.get("score"))),
                "mask": mask,
                "raw_annotation": ann,
            }
        )
    detections.sort(key=lambda item: float(item.get("mask_score") or item.get("score") or 0.0), reverse=True)
    return detections


def run_grounded_sam2_target_detection(
    *,
    image_path: str | Path,
    requested_material: str,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = _resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    perception = config.get("perception", {})
    repo = _resolve_path(str(perception.get("gsam2_repo_dir", "third_party/Grounded-SAM-2")))
    detector = str(perception.get("detector_model_id", "IDEA-Research/grounding-dino-tiny"))
    checkpoint = _resolve_existing_path(
        str(perception.get("sam2_checkpoint", "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")),
        primary_base=REPO_ROOT,
        fallback_base=repo,
    )
    sam2_config_hydra_name = _sam2_hydra_config_name(
        str(perception.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    )
    sam2_config_preflight_path = (repo / "sam2" / sam2_config_hydra_name).resolve()
    image_file = _resolve_path(image_path)
    demo_script = (repo / "grounded_sam2_hf_model_demo.py").resolve()
    configured_python = str(perception.get("gsam2_python", sys.executable) or sys.executable)
    python_executable = sys.executable if configured_python.lower() == "auto" else configured_python
    prompt = requested_material.strip().lower()
    if not prompt.endswith("."):
        prompt = f"{prompt}."
    preflight = {
        "repo_root": str(REPO_ROOT),
        "gsam2_repo_dir": str(repo),
        "demo_script": str(demo_script),
        "sam2_checkpoint": str(checkpoint),
        "sam2_config_hydra_name": sam2_config_hydra_name,
        "sam2_config_preflight_path": str(sam2_config_preflight_path),
        "image_path": str(image_file),
        "output_dir": str(out),
    }
    for label, path in [
        ("Grounded-SAM2 repo dir", repo),
        ("Grounded-SAM2 demo script", demo_script),
        ("SAM2 checkpoint", checkpoint),
        ("SAM2 config", sam2_config_preflight_path),
        ("input image", image_file),
    ]:
        if not path.exists():
            return {
                "success": False,
                "failure_reason": f"{label} not found: {path}",
                "detections": [],
                "detection_query_used": prompt.rstrip("."),
                "preflight": preflight,
            }
    if not demo_script.exists():
        return {
            "success": False,
            "failure_reason": f"Grounded-SAM2 demo script not found: {demo_script}",
            "detections": [],
            "detection_query_used": prompt.rstrip("."),
        }
    command = _build_wrapped_command(
        python_executable=python_executable,
        demo_script=demo_script,
        image_path=image_file,
        text_prompt=prompt,
        output_dir=out,
        detector_model_id=detector,
        sam2_checkpoint=str(checkpoint),
        sam2_config=sam2_config_hydra_name,
    )
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    runtime_s = time.perf_counter() - start
    summary = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "runtime_s": runtime_s,
        "detection_query_used": prompt.rstrip("."),
        "detector_model_id": detector,
        "sam2_checkpoint": str(checkpoint),
        "sam2_config": sam2_config_hydra_name,
        "sam2_config_hydra_name": sam2_config_hydra_name,
        "sam2_config_preflight_path": str(sam2_config_preflight_path),
        "preflight": preflight,
        "results_json_path": str(out / "grounded_sam2_hf_model_demo_results.json"),
    }
    (out / "grounded_sam2_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        return {**summary, "success": False, "failure_reason": "grounded_sam2_subprocess_failed", "detections": []}
    try:
        detections = _load_results(out / "grounded_sam2_hf_model_demo_results.json")
    except Exception as exc:
        return {**summary, "success": False, "failure_reason": str(exc), "detections": []}
    return {**summary, "success": True, "failure_reason": None, "detections": detections}
