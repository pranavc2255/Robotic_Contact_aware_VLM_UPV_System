from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from upv_vlm_v1.prompts.prompt_hash import sha256_text


def _render_template(text: str, context: dict[str, Any]) -> str:
    rendered = text
    for key, value in context.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


@dataclass(frozen=True)
class PromptStage:
    name: str
    enabled: bool
    model_role: str
    description: str
    system_prompt: str
    user_prompt_template: str
    expected_output_schema: Any = None
    disabled_reason: str | None = None

    def with_disabled(self, reason: str) -> "PromptStage":
        return replace(self, enabled=False, disabled_reason=reason)

    def with_user_prompt_template(self, text: str) -> "PromptStage":
        return replace(self, user_prompt_template=text)


@dataclass(frozen=True)
class PromptBundle:
    prompt_set_name: str
    prompt_set_version: str
    source_path: str
    stages: dict[str, PromptStage]

    def render_stage(
        self,
        stage_name: str,
        material_query: str,
        axis_mode: str,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage_name not in self.stages:
            raise KeyError(f"Unknown prompt stage: {stage_name}")
        stage = self.stages[stage_name]
        context = {"material_query": material_query, "axis_mode": axis_mode}
        context.update(extra_context or {})
        if stage.enabled:
            user_prompt = _render_template(stage.user_prompt_template, context)
            system_prompt = stage.system_prompt
        else:
            user_prompt = f"NA: this prompt stage was disabled. Reason: {stage.disabled_reason or 'disabled'}"
            system_prompt = ""
        full_text = f"{system_prompt}\n{user_prompt}".strip()
        return {
            "stage_name": stage.name,
            "enabled": stage.enabled,
            "disabled_reason": stage.disabled_reason,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_hash": sha256_text(full_text),
            "expected_output_schema": stage.expected_output_schema,
            "model_role": stage.model_role,
            "description": stage.description,
        }

    def to_manifest(self, material_query: str, axis_mode: str) -> dict[str, Any]:
        return {
            "prompt_set_name": self.prompt_set_name,
            "prompt_set_version": self.prompt_set_version,
            "source_path": self.source_path,
            "material_query": material_query,
            "axis_mode": axis_mode,
            "stages": {
                name: self.render_stage(name, material_query=material_query, axis_mode=axis_mode)
                for name in self.stages
            },
        }

    def with_stage(self, stage: PromptStage) -> "PromptBundle":
        stages = dict(self.stages)
        stages[stage.name] = stage
        return replace(self, stages=stages)

    def source_as_path(self) -> Path:
        return Path(self.source_path)
