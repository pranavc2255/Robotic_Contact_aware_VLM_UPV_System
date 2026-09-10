import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def _score_value(annotation: dict) -> float:
    score = annotation.get("score", 0.0)

    if isinstance(score, list):
        if not score:
            return 0.0

        score = score[0]

    return float(score)


def _load_mask_utils():
    try:
        from pycocotools import mask as mask_utils

        return mask_utils
    except ModuleNotFoundError:
        fallback_path = "/tmp/upv_vlm_orient_pydeps_nodeps"

        if fallback_path not in sys.path:
            sys.path.insert(0, fallback_path)

        try:
            from pycocotools import mask as mask_utils

            return mask_utils
        except ModuleNotFoundError as exc:
            raise ImportError(
                "pycocotools is required to decode Grounded SAM 2 masks."
            ) from exc


def load_gsam2_results(path: str) -> dict:
    results_path = Path(path)

    if not results_path.exists():
        raise FileNotFoundError(f"Grounded SAM 2 results file not found: {results_path}")

    try:
        with results_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Grounded SAM 2 JSON file: {results_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Grounded SAM 2 results must be a JSON object: {results_path}")

    return data


def select_top_annotation(results: dict) -> dict:
    annotations = results.get("annotations")

    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Grounded SAM 2 results contain no annotations.")

    selected = max(annotations, key=_score_value)
    selected_annotation = dict(selected)
    selected_annotation["score"] = _score_value(selected)
    return selected_annotation


def list_available_class_names(results: dict) -> list[str]:
    annotations = results.get("annotations")

    if not isinstance(annotations, list):
        return []

    return sorted(
        {
            annotation["class_name"]
            for annotation in annotations
            if isinstance(annotation, dict) and "class_name" in annotation
        }
    )


def select_best_annotation_for_class(results: dict, target_class_name: str) -> dict:
    annotations = results.get("annotations")

    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Grounded SAM 2 results contain no annotations.")

    matching_annotations = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and annotation.get("class_name") == target_class_name
    ]

    if not matching_annotations:
        available_names = list_available_class_names(results)
        raise ValueError(
            f"No Grounded SAM 2 annotation found for class '{target_class_name}'. "
            f"Available class names: {available_names}"
        )

    selected = max(matching_annotations, key=_score_value)
    selected_annotation = dict(selected)
    selected_annotation["score"] = _score_value(selected)
    return selected_annotation


def decode_coco_rle_mask(segmentation: dict) -> np.ndarray:
    mask_utils = _load_mask_utils()
    decoded = mask_utils.decode(segmentation)

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]

    binary_mask = (decoded > 0).astype(np.uint8) * 255
    return binary_mask


def save_decoded_mask(mask: np.ndarray, output_path: str) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask).save(output_file)
    return str(output_file)
