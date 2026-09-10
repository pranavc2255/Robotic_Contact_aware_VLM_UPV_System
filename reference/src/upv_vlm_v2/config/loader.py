from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from upv_vlm_v2.config.defaults import REQUIRED_TOP_LEVEL_SECTIONS


class ConfigError(RuntimeError):
    pass


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load v2 YAML config files.") from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    yaml = _load_yaml_module()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    config = deepcopy(payload)
    config["_config_path"] = str(config_path)
    return config


def validate_required_sections(config: dict[str, Any]) -> list[str]:
    missing = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Missing required v2 config sections: {missing}")
    return missing


def snapshot_yaml(config: dict[str, Any], output_path: str | Path) -> Path:
    yaml = _load_yaml_module()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path

