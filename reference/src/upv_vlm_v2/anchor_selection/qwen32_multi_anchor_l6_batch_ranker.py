"""Main-pipeline Qwen32 multi-image L6 anchor ranking backend."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from upv_vlm_v2.anchor_selection.partition_contact_crop_builder import build_clean_anchor_review_grid
from upv_vlm_v2.experiments.prepare_qwen_anchor_layout_variants import render_l6_magenta_line_only
from upv_vlm_v2.prompts.anchor_ranking_prompts import get_anchor_ranking_prompt_variant
from upv_vlm_v2.vlm.qwen_client import infer_qwen_multi_anchor
from upv_vlm_v2.vlm.structured_output import extract_json_object


BACKEND_NAME = "qwen32_multi_anchor_l6_batch_ranking"
L6_LAYOUT_VARIANT = "L6_magenta_line_only_fixed"
DEFAULT_PROMPT_VARIANT = "multi_image_v1_original"
ANCHOR_SELECTION_STAGE_PATH = "src.upv_vlm_v2.anchor_selection.anchor_selection_stage.run_anchor_selection"


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


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


def multi_anchor_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return manifest/result provenance for the multi-image L6 backend."""

    return {
        "anchor_selection_backend": BACKEND_NAME,
        "anchor_layout_variant": L6_LAYOUT_VARIANT,
        "qwen_prompt_variant": str(cfg.get("qwen_prompt_variant") or DEFAULT_PROMPT_VARIANT),
        "qwen_endpoint": str(cfg.get("qwen_server_endpoint") or "/infer_multi"),
        "qwen_mode": "multi_image",
        "qwen_input_dir": "artifacts/04_anchor_selection/clean_single_anchor_inputs",
        "anchor_selection_stage": ANCHOR_SELECTION_STAGE_PATH,
    }


def prepare_l6_qwen_inputs(*, artifact_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Preserve original crops and replace canonical clean inputs with L6 images."""

    out = Path(artifact_dir)
    clean_dir = out / str(cfg.get("qwen_input_dir_name") or "clean_single_anchor_inputs")
    original_dir = out / str(cfg.get("original_anchor_input_dir_name") or "original_clean_single_anchor_inputs")
    debug_path = out / "anchor_crop_geometry_debug.json"
    if not clean_dir.exists():
        return {
            "success": False,
            "failure_reason": f"missing canonical clean input directory: {clean_dir}",
            "clean_single_anchor_inputs_dir": str(clean_dir),
        }
    try:
        debug = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else {}
    except Exception:
        debug = {}
    anchors_meta = debug.get("anchors") if isinstance(debug.get("anchors"), dict) else {}
    original_dir.mkdir(parents=True, exist_ok=True)

    converted: dict[str, Any] = {}
    image_paths: dict[str, str] = {}
    for source in sorted(clean_dir.glob("source_A*.png")):
        anchor_id = source.stem.replace("source_", "")
        original = original_dir / source.name
        if not original.exists():
            shutil.copy2(source, original)
        meta = anchors_meta.get(anchor_id, {}) if isinstance(anchors_meta, dict) else {}
        result = render_l6_magenta_line_only(source_path=original, out_path=source, geometry=meta)
        converted[anchor_id] = {
            "anchor_id": anchor_id,
            "original_clean_input_path": str(original),
            "l6_clean_input_path": str(source),
            "layout_variant": L6_LAYOUT_VARIANT,
            "render_result": result,
        }
        image_paths[anchor_id] = str(source)

    review_grid = None
    if image_paths:
        review_grid = build_clean_anchor_review_grid(image_paths=image_paths, output_path=out / "clean_anchor_review_grid.png")
    payload = {
        "success": bool(converted),
        "backend": BACKEND_NAME,
        "layout_variant": L6_LAYOUT_VARIANT,
        "canonical_qwen_input_dir": str(clean_dir),
        "original_clean_input_dir": str(original_dir),
        "converted_anchor_count": len(converted),
        "converted_anchors": converted,
        "clean_anchor_review_grid": review_grid,
    }
    _write_json(out / "l6_clean_input_conversion.json", payload)
    return payload


def _normalize_per_anchor(parsed: dict[str, Any] | None, anchor_ids: list[str]) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[str]]:
    per_anchor: dict[str, dict[str, Any]] = {}
    missing = list(anchor_ids)
    extra: list[str] = []
    violations: list[str] = []
    if not parsed:
        return per_anchor, missing, extra, ["missing parsed object"]

    items = parsed.get("per_anchor_analysis")
    if not isinstance(items, list):
        return per_anchor, missing, extra, ["missing per_anchor_analysis list"]

    expected = set(anchor_ids)
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            violations.append(f"item {index} is not an object")
            continue
        anchor_id = str(item.get("anchor_id") or "").strip()
        if anchor_id not in expected:
            extra.append(anchor_id or f"<missing_anchor_id_{index}>")
            continue
        left_usable = _as_bool(item.get("left_usable"))
        right_usable = _as_bool(item.get("right_usable"))
        anchor_usable = _as_bool(item.get("anchor_usable"))
        score = _safe_float(item.get("anchor_score"))
        if left_usable is None:
            left_usable = False
        if right_usable is None:
            right_usable = False
        if anchor_usable is None:
            anchor_usable = bool(left_usable and right_usable and (score or 0.0) >= 70.0)
        if score is None:
            score = 0.0
            violations.append(f"{anchor_id}: missing numeric anchor_score")
        score = max(0.0, min(100.0, score))
        if (not left_usable or not right_usable) and (anchor_usable or score > 35.0):
            violations.append(f"{anchor_id}: side unusable but anchor_usable true or score > 35")

        per_anchor[anchor_id] = {
            "anchor_id": anchor_id,
            "image_index": item.get("image_index", anchor_ids.index(anchor_id) + 1),
            "left_panel_condition": str(item.get("left_panel_condition") or ""),
            "right_panel_condition": str(item.get("right_panel_condition") or ""),
            "left_critical_defect_intersecting_line": _as_bool(item.get("left_critical_defect_intersecting_line")),
            "right_critical_defect_intersecting_line": _as_bool(item.get("right_critical_defect_intersecting_line")),
            "left_usable": bool(left_usable),
            "right_usable": bool(right_usable),
            "anchor_score": score,
            "anchor_usable": bool(anchor_usable),
            "raw_model_item": item,
        }
    missing = [anchor_id for anchor_id in anchor_ids if anchor_id not in per_anchor]
    return per_anchor, missing, extra, violations


def _rank_from_response(parsed: dict[str, Any] | None, per_anchor: dict[str, dict[str, Any]], anchor_ids: list[str]) -> list[str]:
    if parsed and isinstance(parsed.get("ranked_usable_anchors"), list):
        ranked = [str(item) for item in parsed["ranked_usable_anchors"] if str(item) in per_anchor]
        ranked = [aid for aid in ranked if per_anchor.get(aid, {}).get("anchor_usable")]
        if ranked:
            return ranked
    return sorted(
        [aid for aid in anchor_ids if per_anchor.get(aid, {}).get("anchor_usable")],
        key=lambda aid: (-float(per_anchor[aid].get("anchor_score") or 0.0), anchor_ids.index(aid)),
    )


def _strict_score_rank(parsed, anchor_ids):
    """No inferred usability, score clipping, missing candidates, or model ranking."""
    import math
    items = (parsed or {}).get("per_anchor_analysis")
    if not isinstance(items, list) or len(items) != len(anchor_ids):
        raise ValueError("incomplete per_anchor_analysis")
    seen = set()
    usable = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("non-object candidate")
        aid = item.get("anchor_id")
        score = item.get("anchor_score")
        if aid not in anchor_ids or aid in seen:
            raise ValueError("unexpected or duplicate anchor ID")
        seen.add(aid)
        if type(item.get("anchor_usable")) is not bool or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("invalid usability or score")
        if item["anchor_usable"]:
            usable.append((aid, score))
    return [aid for aid, score in sorted(usable, key=lambda row: (-row[1], int(row[0].lstrip("A"))))]


def run_qwen32_multi_anchor_l6_batch_ranking(
    *,
    config: dict[str, Any],
    artifact_dir: str | Path,
    candidates: list[dict[str, Any]],
    requested_material: str,
    axis_mode: str,
) -> dict[str, Any]:
    """Run multi-image Qwen ranking against canonical L6 clean inputs."""

    cfg = config.get("anchor_selection", {}) or {}
    out = Path(artifact_dir)
    prompt_variant = str(cfg.get("qwen_prompt_variant") or DEFAULT_PROMPT_VARIANT)
    provenance = multi_anchor_provenance(cfg)

    conversion = prepare_l6_qwen_inputs(artifact_dir=out, cfg=cfg)
    if not conversion.get("success"):
        return {
            "success": False,
            "failure_reason": str(conversion.get("failure_reason") or "L6_INPUT_PREPARATION_FAILED"),
            "selected_anchor_id": None,
            "provenance": provenance,
            "l6_conversion": conversion,
        }

    anchor_ids = [str(candidate["anchor_id"]) for candidate in candidates]
    clean_dir = out / str(cfg.get("qwen_input_dir_name") or "clean_single_anchor_inputs")
    image_paths = [str(clean_dir / f"source_{anchor_id}.png") for anchor_id in anchor_ids]
    system_prompt, user_prompt = get_anchor_ranking_prompt_variant(prompt_variant, anchor_ids)

    prompt_dir = out / "qwen32_multi_anchor_prompts"
    response_dir = out / "qwen32_multi_anchor_responses"
    parsed_dir = out / "qwen32_multi_anchor_parsed"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (prompt_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")

    run_id = f"{axis_mode}_multi_anchor"
    server_url = str(cfg.get("qwen_server_url") or cfg.get("server_url") or "http://127.0.0.1:8899").rstrip("/")
    endpoint = str(cfg.get("qwen_server_endpoint") or "/infer_multi")
    timeout_sec = float(cfg.get("qwen_timeout_sec", cfg.get("timeout_sec", 240)))
    max_new_tokens = int(cfg.get("max_new_tokens", cfg.get("qwen_multi_anchor_max_new_tokens", 1600)))
    temperature = float(cfg.get("temperature", cfg.get("qwen_temperature", 0.0)))

    request_payload = {
        "image_paths": image_paths,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "prompt_variant": prompt_variant,
        "provenance": provenance,
    }
    _write_json(response_dir / f"{run_id}_request_payload.json", request_payload)

    t0 = time.perf_counter()
    response = infer_qwen_multi_anchor(
        server_url=server_url,
        image_paths=image_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        endpoint=endpoint,
        timeout_sec=timeout_sec,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    _write_json(response_dir / f"{run_id}_server_response.json", response)

    raw_text = ""
    for key in ("text", "raw_text", "response", "output", "answer", "generated_text"):
        if isinstance(response.get(key), str):
            raw_text = str(response[key])
            break
    if not raw_text:
        raw_text = json.dumps(response, indent=2, default=str)
    raw_path = response_dir / f"{run_id}_raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    parsed: dict[str, Any] | None = None
    parse_error = ""
    if response.get("success") or response.get("ok"):
        try:
            parsed = extract_json_object(raw_text)
        except Exception as exc:
            parse_error = str(exc)
    else:
        parse_error = str(response.get("error") or response.get("failure_reason") or "qwen request failed")
    parsed_path = parsed_dir / f"{run_id}_parsed.json"
    _write_json(parsed_path, parsed if parsed is not None else {"parse_error": parse_error})

    per_anchor, missing, extra, violations = _normalize_per_anchor(parsed, anchor_ids)
    ranked = _rank_from_response(parsed, per_anchor, anchor_ids)
    if config.get("anchor_selection", {}).get("selection_policy") == "highest_scoring_usable_strict":
        try:
            ranked = _strict_score_rank(parsed, anchor_ids)
        except (ValueError, TypeError) as exc:
            ranked = []
            parse_error = f"INPUT_FAILURE: {exc}"
    selected_anchor_id = ranked[0] if ranked else None
    final_action = "CHOOSE_BEST_ANCHOR" if selected_anchor_id else "NO_SAFE_ANCHOR"

    rows: list[dict[str, Any]] = []
    for idx, anchor_id in enumerate(anchor_ids, start=1):
        item = per_anchor.get(anchor_id, {})
        rows.append(
            {
                "image_index": idx,
                "anchor_id": anchor_id,
                "qwen_input_path": image_paths[idx - 1],
                "parse_ok": bool(parsed is not None and anchor_id in per_anchor),
                "left_usable": item.get("left_usable", ""),
                "right_usable": item.get("right_usable", ""),
                "anchor_usable": item.get("anchor_usable", ""),
                "anchor_score": item.get("anchor_score", ""),
                "left_critical_defect_intersecting_line": item.get("left_critical_defect_intersecting_line", ""),
                "right_critical_defect_intersecting_line": item.get("right_critical_defect_intersecting_line", ""),
                "left_panel_condition": item.get("left_panel_condition", ""),
                "right_panel_condition": item.get("right_panel_condition", ""),
            }
        )
    score_table = _write_csv(
        out / "qwen32_multi_anchor_score_table.csv",
        rows,
        [
            "image_index",
            "anchor_id",
            "qwen_input_path",
            "parse_ok",
            "left_usable",
            "right_usable",
            "anchor_usable",
            "anchor_score",
            "left_critical_defect_intersecting_line",
            "right_critical_defect_intersecting_line",
            "left_panel_condition",
            "right_panel_condition",
        ],
    )

    result = {
        "success": bool(selected_anchor_id),
        "phase": "main_pipeline_qwen32_multi_anchor_l6_batch_ranking",
        "backend": BACKEND_NAME,
        "requested_material": requested_material,
        "axis_mode": axis_mode,
        "selected_anchor_id": selected_anchor_id,
        "selected_anchor": selected_anchor_id,
        "final_robot_action": final_action,
        "safe_no_anchor": selected_anchor_id is None,
        "ranked_usable_anchors": ranked,
        "anchors_attempted": len(anchor_ids),
        "anchors_parsed": len(per_anchor),
        "all_anchors_parsed": len(per_anchor) == len(anchor_ids),
        "parse_ok": parsed is not None,
        "parse_error": parse_error,
        "schema_missing_anchor_ids": missing,
        "schema_extra_anchor_ids": extra,
        "constraint_violations": violations,
        "elapsed_ms": elapsed_ms,
        "prompt_variant": prompt_variant,
        "provenance": provenance,
        "l6_conversion": conversion,
        "image_paths": image_paths,
        "raw_response_path": str(raw_path),
        "parsed_response_path": str(parsed_path),
        "server_response_path": str(response_dir / f"{run_id}_server_response.json"),
        "score_table_path": score_table,
        "per_anchor_analysis": per_anchor,
        "raw_parsed_response": parsed,
        "failure_reason": None if selected_anchor_id else (parse_error or "NO_SAFE_ANCHOR"),
    }
    _write_json(out / "qwen32_multi_anchor_decision.json", result)
    return result
