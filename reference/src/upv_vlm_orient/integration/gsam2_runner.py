import json
from pathlib import Path
import shutil
import subprocess
import time


SCORE_METADATA_FILENAME = "grounded_sam2_score_metadata.json"


def safe_name(text: str) -> str:
    return "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in text.strip().lower().replace(" ", "_")
    )


def _score_value(value) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
        return _score_value(value)

    if value is None:
        return None

    return float(value)


def _best_annotation_summary(results_json_path: Path) -> dict:
    if not results_json_path.exists():
        return {
            "best_class_name_from_json": None,
            "detection_score_from_json": None,
            "mask_score_from_json": None,
        }

    with results_json_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    annotations = results.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return {
            "best_class_name_from_json": None,
            "detection_score_from_json": None,
            "mask_score_from_json": None,
        }

    best_annotation = max(
        annotations,
        key=lambda annotation: _score_value(annotation.get("mask_score", annotation.get("score"))) or -1.0,
    )
    return {
        "best_class_name_from_json": best_annotation.get("class_name"),
        "detection_score_from_json": _score_value(best_annotation.get("detection_score")),
        "mask_score_from_json": _score_value(best_annotation.get("mask_score", best_annotation.get("score"))),
    }


def _copy_if_exists(source_path: Path, destination_path: Path) -> str | None:
    if not source_path.exists():
        return None

    shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _augment_results_with_dual_scores(
    original_results_json_path: Path,
    score_metadata_path: Path,
) -> dict | None:
    results = _load_json_if_exists(original_results_json_path)
    if results is None:
        return None

    metadata = _load_json_if_exists(score_metadata_path) or {}
    metadata_annotations = metadata.get("annotations", [])
    result_annotations = results.get("annotations", [])

    enhanced_annotations = []
    for index, annotation in enumerate(result_annotations):
        enhanced_annotation = dict(annotation)
        metadata_annotation = metadata_annotations[index] if index < len(metadata_annotations) else {}

        mask_score = _score_value(annotation.get("score"))
        detection_score = _score_value(metadata_annotation.get("detection_score"))

        enhanced_annotation["detection_score"] = detection_score
        enhanced_annotation["mask_score"] = mask_score
        enhanced_annotations.append(enhanced_annotation)

    enhanced_results = dict(results)
    enhanced_results["annotations"] = enhanced_annotations
    return enhanced_results


def create_paired_artifacts(
    candidate_class: str,
    image_path: str,
    prompt_text: str,
    run_output_dir: str,
) -> dict:
    run_output_path = Path(run_output_dir)
    safe_candidate_name = safe_name(candidate_class)

    original_results_json_path = run_output_path / "grounded_sam2_hf_model_demo_results.json"
    original_detection_image_path = run_output_path / "groundingdino_annotated_image.jpg"
    original_mask_image_path = run_output_path / "grounded_sam2_annotated_image_with_mask.jpg"
    score_metadata_path = run_output_path / SCORE_METADATA_FILENAME

    paired_results_path = run_output_path / f"{safe_candidate_name}_results.json"
    paired_results_json_path = None
    enhanced_results = _augment_results_with_dual_scores(original_results_json_path, score_metadata_path)
    if enhanced_results is not None:
        paired_results_json_path = _write_json(paired_results_path, enhanced_results)

    paired_detection_image_path = _copy_if_exists(
        original_detection_image_path,
        run_output_path / f"{safe_candidate_name}_annotated_detection.jpg",
    )
    paired_mask_image_path = _copy_if_exists(
        original_mask_image_path,
        run_output_path / f"{safe_candidate_name}_annotated_mask.jpg",
    )

    summary = _best_annotation_summary(paired_results_path if paired_results_json_path else original_results_json_path)

    candidate_info = {
        "candidate_class": candidate_class,
        "safe_candidate_name": safe_candidate_name,
        "source_image_path": image_path,
        "prompt_text": prompt_text,
        "run_output_dir": str(run_output_path),
        "original_results_json_path": str(original_results_json_path),
        "original_detection_image_path": str(original_detection_image_path),
        "original_mask_image_path": str(original_mask_image_path),
        "paired_results_json_path": paired_results_json_path,
        "paired_detection_image_path": paired_detection_image_path,
        "paired_mask_image_path": paired_mask_image_path,
        "best_class_name_from_json": summary["best_class_name_from_json"],
        "detection_score_from_json": summary["detection_score_from_json"],
        "mask_score_from_json": summary["mask_score_from_json"],
    }

    candidate_info_path = run_output_path / "candidate_info.json"
    _write_json(candidate_info_path, candidate_info)

    candidate_info["candidate_info_path"] = str(candidate_info_path)
    return candidate_info


def build_gsam2_command(
    python_executable: str,
    gsam2_repo_dir: str,
    image_path: str,
    text_prompt: str,
    output_dir: str,
    detector_model_id: str,
    sam2_checkpoint: str,
    sam2_model_config: str,
) -> list[str]:
    demo_script = str(Path(gsam2_repo_dir) / "grounded_sam2_hf_model_demo.py")

    return [
        python_executable,
        demo_script,
        "--grounding-model",
        detector_model_id,
        "--img-path",
        image_path,
        "--text-prompt",
        text_prompt,
        "--output-dir",
        output_dir,
        "--sam2-checkpoint",
        sam2_checkpoint,
        "--sam2-model-config",
        sam2_model_config,
    ]


def _build_wrapped_gsam2_command(
    python_executable: str,
    gsam2_repo_dir: str,
    image_path: str,
    text_prompt: str,
    output_dir: str,
    detector_model_id: str,
    sam2_checkpoint: str,
    sam2_model_config: str,
) -> list[str]:
    demo_script = str(Path(gsam2_repo_dir) / "grounded_sam2_hf_model_demo.py")
    wrapper_code = """
import json
from pathlib import Path
import runpy
import sys

demo_script, detector_model_id, image_path, text_prompt, output_dir, sam2_checkpoint, sam2_model_config = sys.argv[1:8]
sys.argv = [
    demo_script,
    "--grounding-model",
    detector_model_id,
    "--img-path",
    image_path,
    "--text-prompt",
    text_prompt,
    "--output-dir",
    output_dir,
    "--sam2-checkpoint",
    sam2_checkpoint,
    "--sam2-model-config",
    sam2_model_config,
]
globals_dict = runpy.run_path(demo_script, run_name="__main__")

class_names = list(globals_dict.get("class_names", []))
confidences = [float(value) for value in list(globals_dict.get("confidences", []))]
mask_scores = globals_dict.get("scores", [])
if hasattr(mask_scores, "tolist"):
    mask_scores = mask_scores.tolist()

def normalize_score(value):
    while isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return float(value)

mask_scores = [normalize_score(value) for value in list(mask_scores)]

metadata = {
    "annotations": [
        {
            "class_name": class_name,
            "detection_score": detection_score,
            "mask_score": mask_score,
        }
        for class_name, detection_score, mask_score in zip(class_names, confidences, mask_scores)
    ]
}

output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
(output_path / "grounded_sam2_score_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
"""

    return [
        python_executable,
        "-c",
        wrapper_code,
        demo_script,
        detector_model_id,
        image_path,
        text_prompt,
        output_dir,
        sam2_checkpoint,
        sam2_model_config,
    ]


def run_gsam2_hf_demo(
    python_executable: str,
    gsam2_repo_dir: str,
    image_path: str,
    text_prompt: str,
    output_dir: str,
    detector_model_id: str,
    sam2_checkpoint: str,
    sam2_model_config: str,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command = _build_wrapped_gsam2_command(
        python_executable=python_executable,
        gsam2_repo_dir=gsam2_repo_dir,
        image_path=image_path,
        text_prompt=text_prompt,
        output_dir=output_dir,
        detector_model_id=detector_model_id,
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
    )

    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=gsam2_repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_sec = time.perf_counter() - start_time

    result = {
        "command": command,
        "returncode": completed.returncode,
        "runtime_sec": runtime_sec,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "results_json_path": str(output_path / "grounded_sam2_hf_model_demo_results.json"),
        "score_metadata_path": str(output_path / SCORE_METADATA_FILENAME),
    }

    if completed.returncode != 0:
        raise RuntimeError(
            "Grounded SAM 2 demo failed.\n"
            f"Command: {command}\n"
            f"Return code: {completed.returncode}\n"
            f"stderr:\n{completed.stderr}"
        )

    return result


def run_gsam2_for_candidate_set(
    python_executable: str,
    gsam2_repo_dir: str,
    image_path: str,
    candidate_list: list[str],
    output_dir: str,
    detector_model_id: str,
    sam2_checkpoint: str,
    sam2_model_config: str,
) -> dict:
    output_path = Path(output_dir)
    raw_root = output_path / "gsam2_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    results = {}

    for candidate_class in candidate_list:
        prompt_text = candidate_class.strip().lower()
        if not prompt_text.endswith("."):
            prompt_text = f"{prompt_text}."

        candidate_output_dir = raw_root / safe_name(candidate_class)
        run_result = run_gsam2_hf_demo(
            python_executable=python_executable,
            gsam2_repo_dir=gsam2_repo_dir,
            image_path=image_path,
            text_prompt=prompt_text,
            output_dir=str(candidate_output_dir),
            detector_model_id=detector_model_id,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_config=sam2_model_config,
        )
        paired_artifacts = create_paired_artifacts(
            candidate_class=candidate_class,
            image_path=image_path,
            prompt_text=prompt_text,
            run_output_dir=str(candidate_output_dir),
        )

        results[candidate_class] = {
            "prompt_text": prompt_text,
            "returncode": run_result["returncode"],
            "runtime_sec": run_result["runtime_sec"],
            "results_json_path": run_result["results_json_path"],
            "score_metadata_path": run_result["score_metadata_path"],
            "stdout": run_result["stdout"],
            "stderr": run_result["stderr"],
            "artifact_dir": str(candidate_output_dir),
            "candidate_info_path": paired_artifacts["candidate_info_path"],
            "paired_results_json_path": paired_artifacts["paired_results_json_path"],
            "paired_detection_image_path": paired_artifacts["paired_detection_image_path"],
            "paired_mask_image_path": paired_artifacts["paired_mask_image_path"],
        }

    return results
