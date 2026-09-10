from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from upv_vlm_orient.experiment_logging.experiment_schema import (
    DEFAULT_PROMPTS,
    MODEL_STAGE_DEFAULTS,
    NA,
    PROMPT_DISABLED_BY_MODE,
    PROMPT_FILE_NAMES,
    PROMPT_STAGES,
    now_iso,
)
from upv_vlm_v1.prompts.prompt_hash import sha256_text
from upv_vlm_v1.prompts.prompt_registry import (
    CANONICAL_TO_LEGACY_STAGE,
    LEGACY_TO_CANONICAL_STAGE,
    apply_ablation_mode,
    apply_prompt_overrides,
    load_prompt_bundle,
    write_prompt_run_package,
)


def canonical_prompt_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "upv_vlm_v1" / "prompts" / "default_prompts.yaml"


def disabled_stages_for_mode(pipeline_mode: str) -> list[str]:
    return list(PROMPT_DISABLED_BY_MODE.get(pipeline_mode, []))


def prompt_stage_enabled(stage: str, pipeline_mode: str) -> bool:
    return stage not in disabled_stages_for_mode(pipeline_mode)


def default_prompt_text(stage: str, material_query: str) -> str:
    template = DEFAULT_PROMPTS.get(stage)
    if template is None:
        return f"NA: this stage does not use a text prompt ({stage})"
    return template.format(material_query=material_query)


def build_prompt_records(
    *,
    pipeline_mode: str,
    material_query: str,
    prompt_set_name: str,
    prompt_set_version: str,
    operator_name: str,
    prompt_overrides: dict[str, str] | None = None,
    preset_path: str | None = None,
) -> dict[str, Any]:
    prompt_overrides = prompt_overrides or {}
    records: dict[str, Any] = {}
    for stage in PROMPT_STAGES:
        enabled = prompt_stage_enabled(stage, pipeline_mode)
        default_text = default_prompt_text(stage, material_query)
        edited = stage in prompt_overrides and prompt_overrides[stage] != default_text
        disabled_reason = None
        if not enabled:
            disabled_reason = f"disabled_by_pipeline_mode:{pipeline_mode}"
            text = f"NA: this prompt stage was not used in this run ({disabled_reason})"
        else:
            text = prompt_overrides.get(stage, default_text)
        model_defaults = MODEL_STAGE_DEFAULTS.get(stage, {})
        records[stage] = {
            "stage": stage,
            "enabled": enabled,
            "prompt_stage_enabled": enabled,
            "disabled_reason": disabled_reason,
            "model": model_defaults.get("model", NA),
            "model_path": model_defaults.get("model_path", NA),
            "input_image_or_crop_path": NA,
            "system_prompt": NA,
            "user_prompt": text,
            "prompt_text": text,
            "temperature": NA,
            "max_tokens": NA,
            "expected_output_schema": "JSON if VLM stage is enabled; otherwise NA",
            "actual_raw_model_response": NA,
            "parsed_structured_response": NA,
            "parser_success": NA,
            "fallback_behavior_if_parse_fails": "Save raw response, mark parse_success=false, deterministic geometry continues only if safe.",
            "prompt_hash": sha256_text(text),
            "edited_prompt_used": edited,
            "default_prompt_path": NA,
            "prompt_preset_path": preset_path or NA,
            "timestamp": now_iso(),
            "operator_name": operator_name,
            "prompt_set_name": prompt_set_name,
            "prompt_set_version": prompt_set_version,
        }
    return records


def write_prompt_package(
    run_dir: Path,
    *,
    pipeline_mode: str,
    material_query: str,
    prompt_set_name: str,
    prompt_set_version: str,
    operator_name: str,
    prompt_overrides: dict[str, str] | None = None,
    preset_path: str | None = None,
) -> dict[str, Any]:
    bundle = load_prompt_bundle(canonical_prompt_path(), prompt_set_name=prompt_set_name, prompt_set_version=prompt_set_version)
    canonical_overrides = {LEGACY_TO_CANONICAL_STAGE.get(key, key): value for key, value in (prompt_overrides or {}).items()}
    bundle = apply_prompt_overrides(bundle, canonical_overrides)
    bundle = apply_ablation_mode(bundle, pipeline_mode)
    package = write_prompt_run_package(
        run_dir,
        bundle,
        material_query=material_query,
        axis_mode="major",
        pipeline_mode=pipeline_mode,
        operator_name=operator_name,
    )
    summary = package["summary"]
    summary["edited_prompts_used"] = bool(prompt_overrides)
    (Path(run_dir) / "prompt_engineering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return package


def prompt_summary_csv_row(run_id: str, pipeline_mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    stages = summary.get("stages", {})
    def stage(name: str) -> dict[str, Any]:
        canonical = LEGACY_TO_CANONICAL_STAGE.get(name, name)
        return stages.get(name) or stages.get(canonical) or {}
    return {
        "run_id": run_id,
        "pipeline_mode": pipeline_mode,
        "prompt_set_name": summary.get("prompt_set_name", NA),
        "prompt_set_version": summary.get("prompt_set_version", NA),
        "open_vocab_prompt_hash": stage("open_vocab").get("prompt_hash", NA),
        "material_prompt_hash": stage("material_crop_verification").get("prompt_hash", NA),
        "edge_prompt_hash": stage("edge_contact_evaluation").get("prompt_hash", NA),
        "material_vlm_parse_success": stage("material_crop_verification").get("parse_success", NA),
        "edge_vlm_parse_success": stage("edge_contact_evaluation").get("parse_success", NA),
        "stages_disabled_by_ablation": ";".join(summary.get("ablation_disabled_stages") or []),
        "edited_prompt_used": summary.get("edited_prompts_used", False),
    }
