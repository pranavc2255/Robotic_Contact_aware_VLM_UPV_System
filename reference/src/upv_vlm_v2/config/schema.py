from __future__ import annotations

from typing import Any

from upv_vlm_v2.config.loader import validate_required_sections


def validate_config_schema(config: dict[str, Any]) -> dict[str, Any]:
    """Lightweight Phase 1 schema validation.

    Later phases should replace this with stricter typed validation once real
    perception, planning, and hardware stages are ported.
    """
    validate_required_sections(config)
    safety = config.get("safety", {})
    if safety.get("dry_run_default") is None:
        raise ValueError("safety.dry_run_default is required")
    logging = config.get("logging", {})
    if not logging.get("output_root"):
        raise ValueError("logging.output_root is required")
    return config

