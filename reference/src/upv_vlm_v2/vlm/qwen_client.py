"""Small Qwen VLM client for required v2 anchor selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


def load_mock_qwen_response(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_qwen_server_health(server_url: str, timeout_sec: float = 5.0) -> dict[str, Any]:
    try:
        req = urllib_request.Request(server_url.rstrip("/") + "/health", method="GET")
        with urllib_request.urlopen(req, timeout=float(timeout_sec)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib_error.URLError, TimeoutError) as exc:
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": f"qwen_server_unreachable: {exc}",
        }
    if not payload.get("ok"):
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": f"qwen_server_health_not_ok: {payload}",
            "health": payload,
        }
    if not payload.get("model_loaded"):
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": "qwen_server_model_not_loaded",
            "health": payload,
        }
    return {"success": True, "health": payload}


def infer_qwen_anchor(
    *,
    server_url: str,
    image_path: str,
    prompt_text: str,
    timeout_sec: float,
    output_path: str | None = None,
) -> dict[str, Any]:
    health = check_qwen_server_health(server_url, timeout_sec=min(float(timeout_sec), 5.0))
    if not health.get("success"):
        return health
    payload = {
        "image_path": image_path,
        "prompt_text": prompt_text,
        "prompt_version": "upv_vlm_v2_anchor_required_json",
        "max_new_tokens": 900,
        "temperature": 0.0,
        "output_path": output_path,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        server_url.rstrip("/") + "/infer",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=float(timeout_sec)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib_error.URLError, TimeoutError) as exc:
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": f"qwen_infer_failed: {exc}",
        }
    if not result.get("ok"):
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": result.get("error", "qwen_infer_not_ok"),
            "response": result,
        }
    result["health"] = health.get("health")
    return result


def infer_qwen_multi_anchor(
    *,
    server_url: str,
    image_paths: list[str],
    system_prompt: str | None,
    user_prompt: str,
    timeout_sec: float,
    endpoint: str = "/infer_multi",
    max_new_tokens: int = 1600,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call the local Qwen server multi-image endpoint for anchor ranking."""

    if not image_paths:
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": "infer_qwen_multi_anchor requires at least one image path",
        }
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": "missing multi-image Qwen input files",
            "missing_image_paths": missing,
        }

    health = check_qwen_server_health(server_url, timeout_sec=min(float(timeout_sec), 5.0))
    if not health.get("success"):
        return health

    payload = {
        "image_paths": image_paths,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    data = json.dumps(payload).encode("utf-8")
    infer_url = server_url.rstrip("/") + (endpoint if endpoint.startswith("/") else f"/{endpoint}")
    req = urllib_request.Request(
        infer_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=float(timeout_sec)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib_error.URLError, TimeoutError) as exc:
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": f"qwen_infer_multi_failed: {exc}",
            "endpoint": endpoint,
        }
    if not result.get("ok"):
        return {
            "success": False,
            "failure_reason": "QWEN_REQUIRED_BUT_FAILED",
            "error": result.get("error", "qwen_infer_multi_not_ok"),
            "endpoint": endpoint,
            "response": result,
        }
    result["success"] = True
    result["health"] = health.get("health")
    result["endpoint"] = endpoint
    result["num_images"] = result.get("num_images", len(image_paths))
    return result
