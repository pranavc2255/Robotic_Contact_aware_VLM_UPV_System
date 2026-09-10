import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


HF_HOME = "workspace/.hf_cache"
HF_HUB_CACHE = "workspace/.hf_cache/hub"
TRANSFORMERS_CACHE = "workspace/.hf_cache/transformers"
DEFAULT_CROP_VERIFIER_MODEL_ID = "openai/clip-vit-base-patch32"
LITE_CROP_VERIFIER_MODEL_NAME = "MobileCLIP2-S0"
LITE_CROP_VERIFIER_PRETRAINED = "dfndr2b"
LITE_CROP_VERIFIER_MODEL_ID = f"{LITE_CROP_VERIFIER_MODEL_NAME}/{LITE_CROP_VERIFIER_PRETRAINED}"


def extract_masked_crop(
    image_path: str,
    mask_path: str,
    output_path: str,
    padding: int = 10,
) -> dict:
    rgb_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)

    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    foreground = np.argwhere(mask_array > 0)
    if foreground.size == 0:
        raise ValueError(f"Mask has no foreground pixels: {mask_path}")

    y_min, x_min = foreground.min(axis=0)
    y_max, x_max = foreground.max(axis=0)

    left = max(0, int(x_min) - padding)
    top = max(0, int(y_min) - padding)
    right = min(rgb_image.width, int(x_max) + 1 + padding)
    bottom = min(rgb_image.height, int(y_max) + 1 + padding)

    crop = rgb_image.crop((left, top, right, bottom))
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_file)

    return {
        "crop_path": str(output_file),
        "bbox_xyxy": [left, top, right, bottom],
        "crop_width": crop.width,
        "crop_height": crop.height,
    }


def verify_crop_with_vlm(
    python_executable: str,
    image_path: str,
    candidate_labels: list[str],
    model_id: str = DEFAULT_CROP_VERIFIER_MODEL_ID,
    model_name: str | None = None,
    pretrained: str | None = None,
) -> dict:
    if model_name is None and model_id == LITE_CROP_VERIFIER_MODEL_ID:
        model_name = LITE_CROP_VERIFIER_MODEL_NAME
    if pretrained is None and model_id == LITE_CROP_VERIFIER_MODEL_ID:
        pretrained = LITE_CROP_VERIFIER_PRETRAINED

    if model_name is not None and pretrained is not None:
        command = [
            python_executable,
            "-c",
            (
                "import json, os, sys\n"
                "python_executable = sys.executable\n"
                "model_name = sys.argv[1]\n"
                "pretrained = sys.argv[2]\n"
                "image_path = sys.argv[3]\n"
                "candidate_labels = json.loads(sys.argv[4])\n"
                "debug = {\n"
                "    'python_executable': python_executable,\n"
                "    'model_name': model_name,\n"
                "    'pretrained': pretrained,\n"
                "    'HF_HOME': os.environ.get('HF_HOME'),\n"
                "    'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),\n"
                "    'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),\n"
                "    'open_clip_import_succeeded': False,\n"
                "    'exception_text': None,\n"
                "}\n"
                "try:\n"
                "    import open_clip\n"
                "    import torch\n"
                "    from PIL import Image\n"
                "    debug['open_clip_import_succeeded'] = True\n"
                "    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)\n"
                "    tokenizer = open_clip.get_tokenizer(model_name)\n"
                "    model.eval()\n"
                "    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)\n"
                "    text_tokens = tokenizer([f'a photo of {label}' for label in candidate_labels])\n"
                "    with torch.no_grad():\n"
                "        image_features = model.encode_image(image)\n"
                "        text_features = model.encode_text(text_tokens)\n"
                "        image_features = image_features / image_features.norm(dim=-1, keepdim=True)\n"
                "        text_features = text_features / text_features.norm(dim=-1, keepdim=True)\n"
                "        logits = 100.0 * image_features @ text_features.T\n"
                "        probabilities = logits.softmax(dim=-1)[0].tolist()\n"
                "    pairs = sorted(zip(candidate_labels, probabilities), key=lambda item: item[1], reverse=True)\n"
                "    payload = {\n"
                "        'results': [{'label': label, 'score': float(score)} for label, score in pairs],\n"
                "        'debug': debug,\n"
                "    }\n"
                "    print(json.dumps(payload))\n"
                "except Exception as exc:\n"
                "    debug['exception_text'] = str(exc)\n"
                "    print(json.dumps({'results': [], 'debug': debug}), file=sys.stderr)\n"
                "    raise RuntimeError(\n"
                "        'Unable to load lightweight crop verifier with open_clip. '\n"
                "        'Install open_clip_torch in workspace/.venv_gsam2 and ensure '\n"
                "        'the MobileCLIP2-S0 / dfndr2b weights are reachable through the local HF cache or network. '\n"
                "        'Original error: ' + str(exc)\n"
                "    ) from exc\n"
            ),
            model_name,
            pretrained,
            image_path,
            json.dumps(candidate_labels),
        ]
    else:
        command = [
            python_executable,
            "-c",
            (
                "import json, os, sys\n"
                "model_id = sys.argv[1]\n"
                "image_path = sys.argv[2]\n"
                "candidate_labels = json.loads(sys.argv[3])\n"
                "from transformers import pipeline\n"
                "debug = {\n"
                "    'python_executable': sys.executable,\n"
                "    'model_name': model_id,\n"
                "    'pretrained': None,\n"
                "    'HF_HOME': os.environ.get('HF_HOME'),\n"
                "    'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),\n"
                "    'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),\n"
                "    'open_clip_import_succeeded': False,\n"
                "    'exception_text': None,\n"
                "}\n"
                "try:\n"
                "    classifier = pipeline(\n"
                "        task='zero-shot-image-classification',\n"
                "        model=model_id,\n"
                "    )\n"
                "    results = classifier(image_path, candidate_labels=candidate_labels)\n"
                "    print(json.dumps({'results': results, 'debug': debug}))\n"
                "except Exception as exc:\n"
                "    debug['exception_text'] = str(exc)\n"
                "    print(json.dumps({'results': [], 'debug': debug}), file=sys.stderr)\n"
                "    raise\n"
            ),
            model_id,
            image_path,
            json.dumps(candidate_labels),
        ]
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HF_HUB_CACHE"] = HF_HUB_CACHE
    env["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMERS_CACHE).mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    runtime_sec = time.perf_counter() - start_time

    if completed.returncode != 0:
        raise RuntimeError(
            "Crop verification subprocess failed.\n"
            f"Python executable: {python_executable}\n"
            f"Model id: {model_id}\n"
            f"Model name: {model_name}\n"
            f"Pretrained tag: {pretrained}\n"
            f"HF_HOME: {env['HF_HOME']}\n"
            f"HF_HUB_CACHE: {env['HF_HUB_CACHE']}\n"
            f"TRANSFORMERS_CACHE: {env['TRANSFORMERS_CACHE']}\n"
            f"Command: {command}\n"
            f"Return code: {completed.returncode}\n"
            f"stderr:\n{completed.stderr}"
        )

    stdout_text = completed.stdout.strip()
    payload = json.loads(stdout_text) if stdout_text else {"results": [], "debug": {}}
    results = payload.get("results", [])
    debug = payload.get("debug", {})

    labels = [item["label"] for item in results]
    scores = [float(item["score"]) for item in results]
    top_label = labels[0] if labels else None
    top_score = scores[0] if scores else None

    return {
        "labels": labels,
        "scores": scores,
        "top_label": top_label,
        "top_score": top_score,
        "model_id": model_id,
        "model_name": model_name or model_id,
        "pretrained": pretrained,
        "debug": debug,
        "runtime_sec": runtime_sec,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def verify_crop_pairwise(
    python_executable: str,
    image_path: str,
    requested_class_name: str,
    candidate_list: list[str],
    model_id: str = DEFAULT_CROP_VERIFIER_MODEL_ID,
    model_name: str | None = None,
    pretrained: str | None = None,
) -> dict:
    verification = verify_crop_with_vlm(
        python_executable=python_executable,
        image_path=image_path,
        candidate_labels=candidate_list,
        model_id=model_id,
        model_name=model_name,
        pretrained=pretrained,
    )

    return {
        "requested_class_name": requested_class_name,
        "candidate_list": candidate_list,
        "model_id": verification["model_id"],
        "model_name": verification["model_name"],
        "pretrained": verification["pretrained"],
        "debug": verification["debug"],
        "top_label": verification["top_label"],
        "top_score": verification["top_score"],
        "labels": verification["labels"],
        "scores": verification["scores"],
        "runtime_sec": verification["runtime_sec"],
        "stdout": verification["stdout"],
        "stderr": verification["stderr"],
    }
