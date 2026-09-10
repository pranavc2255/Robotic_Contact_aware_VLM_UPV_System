from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import yaml

from upv_vlm_v1.prompts.edge_contact_quality_prompt import build_edge_contact_quality_prompt
from upv_vlm_v1.prompts.material_crop_verification_prompt import (
    DEFAULT_MATERIAL_DESCRIPTIONS,
    build_material_crop_verification_prompt,
)
from upv_vlm_v1.prompts.open_vocab_detection_prompt import build_open_vocab_detection_prompt
from upv_vlm_v1.prompts.prompt_bundle import PromptBundle, PromptStage
from upv_vlm_v1.prompts.prompt_hash import sha256_text


REAL_PROMPT_STAGE_NAMES = [
    "open_vocab_detection",
    "material_crop_verification",
    "edge_contact_quality",
]


CANONICAL_STAGE_FILE_NAMES = {
    "open_vocab_detection": "open_vocab_detection_prompt.txt",
    "material_crop_verification": "material_crop_verification_prompt.txt",
    "edge_contact_quality": "edge_contact_quality_prompt.txt",
}

LEGACY_TO_CANONICAL_STAGE = {
    "open_vocab": "open_vocab_detection",
    "material_verification": "material_crop_verification",
    "candidate_crop_comparison": "material_crop_verification",
    "crop_comparison": "material_crop_verification",
    "edge_contact": "edge_contact_quality",
    "edge_contact_evaluation": "edge_contact_quality",
    "structured_output": "material_crop_verification",
}

CANONICAL_TO_LEGACY_STAGE = {value: key for key, value in LEGACY_TO_CANONICAL_STAGE.items()}
CANONICAL_TO_LEGACY_STAGE.update({name: name for name in REAL_PROMPT_STAGE_NAMES})

PROMPT_DISABLED_BY_MODE = {
    "baseline_detector_only_no_material_vlm": ["material_crop_verification", "edge_contact_quality"],
    "ablation_no_material_crop_vlm": ["material_crop_verification"],
    "ablation_no_edge_crop_contact_vlm": ["edge_contact_quality"],
    "ablation_no_anchor_contact_quality_check": ["edge_contact_quality"],
}


def _stage_key(stage: str) -> str:
    return LEGACY_TO_CANONICAL_STAGE.get(stage, stage)


def canonical_prompt_stage_names() -> list[str]:
    return list(REAL_PROMPT_STAGE_NAMES)


def load_prompt_bundle(path: str | Path, prompt_set_name: str = "default_v1", prompt_set_version: str = "v1") -> PromptBundle:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stages: dict[str, PromptStage] = {}
    for name, values in data.items():
        name = _stage_key(str(name))
        if name not in REAL_PROMPT_STAGE_NAMES:
            continue
        values = values or {}
        stages[name] = PromptStage(
            name=name,
            enabled=bool(values.get("enabled", True)),
            model_role=str(values.get("model_role", "")),
            description=str(values.get("description", "")),
            system_prompt=str(values.get("system_prompt") or ""),
            user_prompt_template=str(values.get("user_prompt_template") or ""),
            expected_output_schema=values.get("expected_output_schema"),
            disabled_reason=values.get("disabled_reason"),
        )
    return PromptBundle(
        prompt_set_name=prompt_set_name,
        prompt_set_version=prompt_set_version,
        source_path=str(path),
        stages=stages,
    )


def _prompt_hash(prompt: dict[str, Any] | str) -> str:
    if isinstance(prompt, str):
        return sha256_text(prompt)
    return sha256_text("\n".join(str(prompt.get(key, "")) for key in ["system_prompt", "user_prompt"]))


def build_experiment1_prompt_manifest(
    material_query: str,
    candidate_ids: list[str] | None = None,
    prompt_config_path: str | Path | None = None,
    prompt_injection_status: str = "prompt_logged_only_clip_fixed_labels",
) -> dict[str, Any]:
    detection_prompt = build_open_vocab_detection_prompt(material_query)
    material_prompt = build_material_crop_verification_prompt(
        requested_material=material_query,
        candidate_ids=candidate_ids or [],
        material_descriptions=DEFAULT_MATERIAL_DESCRIPTIONS,
    )
    edge_prompt = build_edge_contact_quality_prompt(
        requested_material=material_query,
        axis_mode="NA",
        candidate_anchor_ids=[],
    )
    return {
        "prompt_registry_version": "upv_vlm_v1_real_prompt_stages_v1",
        "source_path": str(prompt_config_path or "src/upv_vlm_v1/prompts"),
        "active_prompt_stages": ["open_vocab_detection", "material_crop_verification"],
        "inactive_prompt_stages": [
            {
                "name": "edge_contact_quality",
                "reason": "inactive_for_experiment_1: used later for anchor/contact experiments, not Experiment 1 target selection",
            }
        ],
        "model_calls_expected": {
            "open_vocab_detection": "GroundingDINO/SAM2",
            "material_crop_verification": "CLIP crop verifier",
            "edge_contact_quality": "future VLM",
        },
        "stages": {
            "open_vocab_detection": {
                "enabled": True,
                "model_role": "GroundingDINO/SAM2 open-vocabulary detector",
                "prompt": detection_prompt,
                "prompt_hash": _prompt_hash(detection_prompt),
                "prompt_injection_status": "simple_text_query_used",
            },
            "material_crop_verification": {
                "enabled": True,
                "model_role": "CLIP crop verifier",
                **material_prompt,
                "prompt_hash": _prompt_hash(material_prompt),
                "prompt_injection_status": prompt_injection_status,
            },
            "edge_contact_quality": {
                "enabled": False,
                "model_role": "future edge/contact quality VLM",
                **edge_prompt,
                "prompt_hash": _prompt_hash(edge_prompt),
                "disabled_reason": "inactive_for_experiment_1",
            },
        },
    }


def apply_prompt_overrides(bundle: PromptBundle, overrides: dict[str, str]) -> PromptBundle:
    result = bundle
    for raw_name, text in (overrides or {}).items():
        name = _stage_key(raw_name)
        if name not in result.stages:
            continue
        result = result.with_stage(result.stages[name].with_user_prompt_template(str(text)))
    return result


def disabled_stages_for_pipeline_mode(pipeline_mode: str) -> list[str]:
    return list(PROMPT_DISABLED_BY_MODE.get(pipeline_mode, []))


def apply_ablation_mode(bundle: PromptBundle, pipeline_mode: str) -> PromptBundle:
    result = bundle
    for stage_name in disabled_stages_for_pipeline_mode(pipeline_mode):
        if stage_name in result.stages:
            reason = f"disabled_by_pipeline_mode:{pipeline_mode}"
            result = result.with_stage(result.stages[stage_name].with_disabled(reason))
    return result


def write_prompt_run_package(
    run_dir: str | Path,
    bundle: PromptBundle,
    material_query: str,
    axis_mode: str,
    pipeline_mode: str,
    operator_name: str,
    raw_responses: dict[str, Any] | None = None,
    parsed_responses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest = bundle.to_manifest(material_query=material_query, axis_mode=axis_mode)
    manifest.update({"pipeline_mode": pipeline_mode, "operator_name": operator_name})
    raw_payload: dict[str, Any] = {}
    parsed_payload: dict[str, Any] = {}
    raw_responses = raw_responses or {}
    parsed_responses = parsed_responses or {}
    for stage_name, rendered in manifest["stages"].items():
        file_name = CANONICAL_STAGE_FILE_NAMES.get(stage_name, f"{stage_name}_prompt.txt")
        text_path = prompts_dir / file_name
        if rendered["enabled"]:
            text = "\n".join(part for part in [rendered.get("system_prompt", ""), rendered.get("user_prompt", "")] if part).strip()
        else:
            text = f"NA: this prompt stage was disabled. Reason: {rendered.get('disabled_reason') or 'disabled'}"
        text_path.write_text(text + "\n", encoding="utf-8")
        rendered["prompt_text_path"] = str(text_path)
        rendered["raw_response_path"] = str(prompts_dir / "vlm_raw_responses.json")
        rendered["parsed_response_path"] = str(prompts_dir / "vlm_parsed_responses.json")
        raw_payload[stage_name] = raw_responses.get(stage_name, {"raw_response": "NA", "prompt_stage_enabled": rendered["enabled"], "disabled_reason": rendered.get("disabled_reason")})
        parsed_payload[stage_name] = parsed_responses.get(
            stage_name,
            {
                "requested_schema": rendered.get("expected_output_schema"),
                "raw_response": "NA",
                "cleaned_response": "NA",
                "parsed_json": "NA",
                "parse_success": "NA",
                "parse_error": "NA",
                "fields_used_by_pipeline": [],
                "fields_ignored_by_pipeline": [],
                "disabled_reason": rendered.get("disabled_reason"),
            },
        )
    (prompts_dir / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (prompts_dir / "vlm_raw_responses.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    (prompts_dir / "vlm_parsed_responses.json").write_text(json.dumps(parsed_payload, indent=2), encoding="utf-8")
    disabled = [name for name, stage in bundle.stages.items() if not stage.enabled]
    summary = {
        "prompt_set_name": bundle.prompt_set_name,
        "prompt_set_version": bundle.prompt_set_version,
        "source_path": bundle.source_path,
        "pipeline_mode": pipeline_mode,
        "operator_name": operator_name,
        "stages": {
            name: {
                "enabled": rendered["enabled"],
                "model": rendered.get("model_role"),
                "model_role": rendered.get("model_role"),
                "prompt_hash": rendered.get("prompt_hash"),
                "prompt_text_path": rendered.get("prompt_text_path"),
                "raw_response_path": rendered.get("raw_response_path"),
                "parsed_response_path": rendered.get("parsed_response_path"),
                "parse_success": "NA",
                "disabled_reason": rendered.get("disabled_reason"),
            }
            for name, rendered in manifest["stages"].items()
        },
        "edited_prompts_used": False,
        "ablation_disabled_stages": disabled,
    }
    (run_dir / "prompt_engineering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"manifest": manifest, "summary": summary, "raw_responses": raw_payload, "parsed_responses": parsed_payload}


def bundle_to_jsonable(bundle: PromptBundle) -> dict[str, Any]:
    return {
        "prompt_set_name": bundle.prompt_set_name,
        "prompt_set_version": bundle.prompt_set_version,
        "source_path": bundle.source_path,
        "stages": {
            name: {
                "name": stage.name,
                "enabled": stage.enabled,
                "model_role": stage.model_role,
                "description": stage.description,
                "system_prompt": stage.system_prompt,
                "user_prompt_template": stage.user_prompt_template,
                "expected_output_schema": stage.expected_output_schema,
                "disabled_reason": stage.disabled_reason,
            }
            for name, stage in bundle.stages.items()
        },
    }


def bundle_from_jsonable(data: dict[str, Any]) -> PromptBundle:
    return PromptBundle(
        prompt_set_name=str(data.get("prompt_set_name", "default_v1")),
        prompt_set_version=str(data.get("prompt_set_version", "v1")),
        source_path=str(data.get("source_path", "")),
        stages={
            name: PromptStage(
                name=str(values.get("name", name)),
                enabled=bool(values.get("enabled", True)),
                model_role=str(values.get("model_role", "")),
                description=str(values.get("description", "")),
                system_prompt=str(values.get("system_prompt") or ""),
                user_prompt_template=str(values.get("user_prompt_template") or ""),
                expected_output_schema=values.get("expected_output_schema"),
                disabled_reason=values.get("disabled_reason"),
            )
            for name, values in (data.get("stages") or {}).items()
        },
    )
