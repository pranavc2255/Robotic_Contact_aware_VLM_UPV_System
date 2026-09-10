"""Main-pipeline Llama single-image L6 taxonomy scoring backend."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from upv_vlm_v2.anchor_selection.qwen32_multi_anchor_l6_batch_ranker import (
    ANCHOR_SELECTION_STAGE_PATH,
    L6_LAYOUT_VARIANT,
    prepare_l6_qwen_inputs,
)
from upv_vlm_v2.prompts.anchor_ranking_prompts import get_single_anchor_prompt_variant
from upv_vlm_v2.vlm.structured_output import extract_json_object


BACKEND_NAME = "llama32_single_anchor_l6_taxonomy_scoring"
DEFAULT_PROMPT_VARIANT = "llama_single_anchor_v1_taxonomy"
RANKING_METHOD = "python_select_highest_usable_score_with_threshold"
DEFAULT_MIN_SELECTABLE_SCORE = 85.0


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=float(timeout_sec)) as response:  # noqa: S310 - local server URL from config
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Llama HTTPError {exc.code}: {body}") from exc
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "raw_http_text": text}


def _get_json(url: str, timeout_sec: int) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(req, timeout=float(timeout_sec)) as response:  # noqa: S310 - local server URL from config
        text = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "raw_http_text": text}


def _check_llama_health(server_url: str, timeout_sec: int, attempts: int = 3, delay_sec: float = 0.75) -> dict[str, Any]:
    health_url = server_url.rstrip("/") + "/health"
    errors: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        try:
            health = _get_json(health_url, timeout_sec=min(timeout_sec, 10))
            ok = bool(health.get("ok")) and bool(health.get("model_loaded", True))
            family = str(health.get("model_family") or "").lower()
            model_type = str(health.get("model_type") or "").lower()
            model_class = str(health.get("model_class") or "").lower()
            model_path = str(health.get("model_path") or "").lower()
            llama_evidence = (
                family == "llama_vision"
                or "llama" in family
                or model_type == "mllama"
                or "mllama" in model_type
                or "mllama" in model_class
                or "llama" in model_path
            )
            not_qwen = "qwen" not in model_class and "qwen" not in model_path and "qwen" not in model_type
            if ok and llama_evidence and not_qwen:
                health["health_preflight_ok"] = True
                health["health_url"] = health_url
                health["attempt"] = attempt
                return health
            errors.append(
                {
                    "attempt": attempt,
                    "health": health,
                    "reason": "health fields did not match llama_vision/mllama expectations",
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"attempt": attempt, "error": str(exc), "exception_type": type(exc).__name__})
        if attempt < attempts:
            time.sleep(delay_sec)
    return {
        "ok": False,
        "health_preflight_ok": False,
        "health_url": health_url,
        "attempts": attempts,
        "errors": errors,
        "failure_reason": "llama_server_health_preflight_failed",
    }


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "usable", "good"}:
            return True
        if lowered in {"false", "0", "no", "n", "unusable", "bad"}:
            return False
    return None


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    return max(0.0, min(100.0, score))


def _normalize(parsed: dict[str, Any] | None, expected_anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    if parsed is None:
        return {"anchor_id": expected_anchor_id, "parse_ok": False, "schema_ok": False}, False, "missing parsed object"
    anchor_id = str(parsed.get("anchor_id") or "").strip()
    if anchor_id != expected_anchor_id:
        return {"anchor_id": anchor_id, "parse_ok": True, "schema_ok": False, "raw_parsed": parsed}, False, f"anchor_id mismatch: {anchor_id}"
    left_defect = str(parsed.get("left_defect_type") or "").strip()
    right_defect = str(parsed.get("right_defect_type") or "").strip()
    left_cont = str(parsed.get("left_contact_continuity") or "").strip()
    right_cont = str(parsed.get("right_contact_continuity") or "").strip()
    score = _safe_score(parsed.get("anchor_score"))
    usable = _as_bool(parsed.get("anchor_usable"))
    allowed_defects = {"none", "texture_color_only", "chip_or_notch", "raised_debris", "open_gap"}
    allowed_cont = {"continuous", "physically_interrupted"}
    errors = []
    if left_defect not in allowed_defects:
        errors.append("invalid left_defect_type")
    if right_defect not in allowed_defects:
        errors.append("invalid right_defect_type")
    if left_cont not in allowed_cont:
        errors.append("invalid left_contact_continuity")
    if right_cont not in allowed_cont:
        errors.append("invalid right_contact_continuity")
    if score is None:
        errors.append("missing numeric anchor_score")
        score = 0.0
    if usable is None:
        errors.append("missing boolean anchor_usable")
        usable = False
    critical = left_defect in {"chip_or_notch", "raised_debris", "open_gap"} or right_defect in {"chip_or_notch", "raised_debris", "open_gap"}
    interrupted = left_cont == "physically_interrupted" or right_cont == "physically_interrupted"
    if (critical or interrupted) and (usable or score > 35.0):
        errors.append("constraint violation: critical/interrupted but usable or score > 35")
    norm = {
        "anchor_id": expected_anchor_id,
        "left_defect_type": left_defect,
        "left_contact_continuity": left_cont,
        "right_defect_type": right_defect,
        "right_contact_continuity": right_cont,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "anchor_score": score,
        "anchor_usable": bool(usable),
        "parse_ok": True,
        "schema_ok": not errors,
        "schema_error": "; ".join(errors),
        "raw_parsed": parsed,
    }
    return norm, not errors, norm["schema_error"]


def llama_single_anchor_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_selection_backend": BACKEND_NAME,
        "vlm_backend": "llama_single_image_loop",
        "anchor_layout_variant": L6_LAYOUT_VARIANT,
        "prompt_variant": str(cfg.get("prompt_variant") or cfg.get("qwen_prompt_variant") or DEFAULT_PROMPT_VARIANT),
        "vlm_endpoint": str(cfg.get("vlm_server_endpoint") or cfg.get("qwen_server_endpoint") or "/infer"),
        "vlm_server_endpoint": str(cfg.get("vlm_server_endpoint") or cfg.get("qwen_server_endpoint") or "/infer"),
        "vlm_mode": "single_image_loop",
        "vlm_input_dir": "artifacts/04_anchor_selection/clean_single_anchor_inputs",
        "ranking_method": RANKING_METHOD,
        "llama_min_selectable_score": float(cfg.get("llama_min_selectable_score", cfg.get("min_selectable_anchor_score", DEFAULT_MIN_SELECTABLE_SCORE))),
        "anchor_selection_stage": ANCHOR_SELECTION_STAGE_PATH,
    }


def run_llama32_single_anchor_l6_taxonomy_scoring(
    *,
    config: dict[str, Any],
    artifact_dir: str | Path,
    candidates: list[dict[str, Any]],
    requested_material: str,
    axis_mode: str,
) -> dict[str, Any]:
    cfg = config.get("anchor_selection", {}) or {}
    out = Path(artifact_dir)
    provenance = llama_single_anchor_provenance(cfg)
    conversion = prepare_l6_qwen_inputs(artifact_dir=out, cfg=cfg)
    if not conversion.get("success"):
        return {
            "success": False,
            "failure_reason": str(conversion.get("failure_reason") or "L6_INPUT_PREPARATION_FAILED"),
            "selected_anchor_id": None,
            "provenance": provenance,
        }

    server_url = str(cfg.get("vlm_server_url") or cfg.get("qwen_server_url") or "http://127.0.0.1:8899").rstrip("/")
    endpoint = str(cfg.get("vlm_server_endpoint") or cfg.get("qwen_server_endpoint") or "/infer")
    infer_url = server_url + (endpoint if endpoint.startswith("/") else f"/{endpoint}")
    prompt_variant = str(cfg.get("prompt_variant") or cfg.get("qwen_prompt_variant") or DEFAULT_PROMPT_VARIANT)
    timeout_sec = int(cfg.get("vlm_timeout_sec", cfg.get("qwen_timeout_sec", 240)))
    max_new_tokens = int(cfg.get("max_new_tokens", 384))
    temperature = float(cfg.get("temperature", cfg.get("qwen_temperature", 0.0)))
    min_selectable_score = float(cfg.get("llama_min_selectable_score", cfg.get("min_selectable_anchor_score", DEFAULT_MIN_SELECTABLE_SCORE)))
    health = _check_llama_health(server_url, timeout_sec=timeout_sec)
    _write_json(out / "llama32_single_anchor_server_health.json", health)
    if not health.get("health_preflight_ok"):
        result = {
            "success": False,
            "phase": "main_pipeline_llama32_single_anchor_l6_taxonomy_scoring",
            "vlm_backend": "llama_single_image_loop",
            "backend": BACKEND_NAME,
            "prompt_variant": prompt_variant,
            "ranking_method": RANKING_METHOD,
            "llama_min_selectable_score": min_selectable_score,
            "requested_material": requested_material,
            "axis_mode": axis_mode,
            "selected_anchor_id": None,
            "selected_anchor": None,
            "final_robot_action": "NO_SAFE_ANCHOR",
            "ranked_usable_anchors": [],
            "ranked_selectable_anchors": [],
            "anchors_attempted": len(candidates),
            "server_ok_count": 0,
            "parse_success_count": 0,
            "schema_success_count": 0,
            "anchor_results": {},
            "provenance": provenance,
            "l6_conversion": conversion,
            "vlm_server_health": health,
            "failure_reason": "LLAMA_SERVER_HEALTH_PREFLIGHT_FAILED",
        }
        _write_json(out / "llama32_single_anchor_decision.json", result)
        _write_csv(
            out / "llama32_single_anchor_score_table.csv",
            [],
            [
                "image_index",
                "anchor_id",
                "server_ok",
                "parse_ok",
                "schema_ok",
                "left_defect_type",
                "left_contact_continuity",
                "right_defect_type",
                "right_contact_continuity",
                "reasoning",
                "anchor_score",
                "anchor_usable",
                "error",
                "image_path",
            ],
        )
        return result

    clean_dir = out / str(cfg.get("qwen_input_dir_name") or "clean_single_anchor_inputs")
    prompt_dir = out / "llama32_single_anchor_prompts"
    response_dir = out / "llama32_single_anchor_responses"
    parsed_dir = out / "llama32_single_anchor_parsed"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        aid = str(candidate["anchor_id"])
        image_path = clean_dir / f"source_{aid}.png"
        system_prompt, user_prompt = get_single_anchor_prompt_variant(prompt_variant, aid)
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        (prompt_dir / f"source_{aid}_prompt.txt").write_text(prompt_text, encoding="utf-8")
        payload = {
            "image_path": str(image_path),
            "prompt_text": prompt_text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        t0 = time.perf_counter()
        response: dict[str, Any]
        raw_text = ""
        parsed = None
        error = ""
        server_ok = False
        try:
            response = _post_json(infer_url, payload, timeout_sec)
            server_ok = bool(response.get("ok"))
            raw_text = str(response.get("text") or response.get("raw_text") or "")
            if server_ok:
                parsed = extract_json_object(raw_text)
            else:
                error = str(response.get("error") or "server ok false")
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        _write_json(response_dir / f"source_{aid}_server_response.json", response)
        (response_dir / f"source_{aid}_raw.txt").write_text(raw_text, encoding="utf-8")
        norm, schema_ok, schema_error = _normalize(parsed, aid)
        if error and not norm.get("schema_error"):
            norm["schema_error"] = error
        norm.update(
            {
                "server_ok": server_ok,
                "elapsed_ms": elapsed_ms,
                "image_index": idx,
                "image_path": str(image_path),
                "prompt_path": str(prompt_dir / f"source_{aid}_prompt.txt"),
                "raw_response_path": str(response_dir / f"source_{aid}_raw.txt"),
                "parsed_response_path": str(parsed_dir / f"source_{aid}_parsed.json"),
                "server_response_path": str(response_dir / f"source_{aid}_server_response.json"),
            }
        )
        _write_json(parsed_dir / f"source_{aid}_parsed.json", norm)
        results[aid] = norm
        rows.append(
            {
                "image_index": idx,
                "anchor_id": aid,
                "server_ok": server_ok,
                "parse_ok": norm.get("parse_ok", False),
                "schema_ok": schema_ok,
                "left_defect_type": norm.get("left_defect_type", ""),
                "left_contact_continuity": norm.get("left_contact_continuity", ""),
                "right_defect_type": norm.get("right_defect_type", ""),
                "right_contact_continuity": norm.get("right_contact_continuity", ""),
                "reasoning": norm.get("reasoning", ""),
                "anchor_score": norm.get("anchor_score", ""),
                "anchor_usable": norm.get("anchor_usable", False),
                "error": norm.get("schema_error", ""),
                "image_path": str(image_path),
            }
        )

    ranked = sorted(
        [
            aid
            for aid, item in results.items()
            if item.get("schema_ok")
            and item.get("anchor_usable")
            and float(item.get("anchor_score") or 0.0) >= min_selectable_score
        ],
        key=lambda aid: (-float(results[aid].get("anchor_score") or 0.0), int(aid[1:]) if aid[1:].isdigit() else 10**9),
    )
    selected = ranked[0] if ranked else None
    score_table = _write_csv(
        out / "llama32_single_anchor_score_table.csv",
        rows,
        [
            "image_index",
            "anchor_id",
            "server_ok",
            "parse_ok",
            "schema_ok",
            "left_defect_type",
            "left_contact_continuity",
            "right_defect_type",
            "right_contact_continuity",
            "reasoning",
            "anchor_score",
            "anchor_usable",
            "error",
            "image_path",
        ],
    )
    result = {
        "success": bool(selected),
        "phase": "main_pipeline_llama32_single_anchor_l6_taxonomy_scoring",
        "vlm_backend": "llama_single_image_loop",
        "backend": BACKEND_NAME,
        "prompt_variant": prompt_variant,
        "ranking_method": RANKING_METHOD,
        "llama_min_selectable_score": min_selectable_score,
        "requested_material": requested_material,
        "axis_mode": axis_mode,
        "selected_anchor_id": selected,
        "selected_anchor": selected,
        "final_robot_action": "CHOOSE_BEST_ANCHOR" if selected else "NO_SAFE_ANCHOR",
        "ranked_usable_anchors": ranked,
        "ranked_selectable_anchors": ranked,
        "anchors_attempted": len(candidates),
        "server_ok_count": sum(1 for item in results.values() if item.get("server_ok")),
        "parse_success_count": sum(1 for item in results.values() if item.get("parse_ok")),
        "schema_success_count": sum(1 for item in results.values() if item.get("schema_ok")),
        "anchor_results": results,
        "score_table_path": score_table,
        "provenance": provenance,
        "l6_conversion": conversion,
        "vlm_server_health": health,
        "failure_reason": None if selected else "NO_SAFE_ANCHOR_OR_NO_SCHEMA_VALID_USABLE_ANCHOR",
    }
    _write_json(out / "llama32_single_anchor_decision.json", result)
    return result
