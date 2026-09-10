from pathlib import Path

import yaml


def load_yaml_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {config_path}") from exc

    if not isinstance(data, dict) or not data:
        raise ValueError(f"YAML config is empty or invalid: {config_path}")

    return data
