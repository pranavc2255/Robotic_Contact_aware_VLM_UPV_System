"""Load and validate explicit UPV tool/camera transform configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .transform_utils import make_transform


TRANSFORM_FIELDS = [
    "T_tcp_camera_color_optical",
    "T_tcp_upv_contact_midplane",
    "T_upv_contact_midplane_left_contact_center",
    "T_upv_contact_midplane_right_contact_center",
]


def _rpy_deg_to_rad(values: list[float]) -> list[float]:
    return [math.radians(float(v)) for v in values]


def load_tool_frame_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_path"] = str(path.resolve())
    validate_tool_frame_config(config)
    return config


def validate_tool_frame_config(config: dict[str, Any]) -> None:
    if config.get("frame_schema_version") != "v1":
        raise ValueError("frame_schema_version must be v1.")
    if not config.get("calibration_status"):
        raise ValueError("calibration_status is required.")
    for field in TRANSFORM_FIELDS:
        spec = config.get(field)
        if not isinstance(spec, dict):
            raise ValueError(f"{field} must be an object.")
        translation = spec.get("translation_m")
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError(f"{field}.translation_m must be three values.")
        if not all(math.isfinite(float(v)) for v in translation):
            raise ValueError(f"{field}.translation_m must be finite.")
        rpy = spec.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
        if not isinstance(rpy, list) or len(rpy) != 3:
            raise ValueError(f"{field}.rotation_rpy_deg must be three values.")
        if "confidence" not in spec:
            raise ValueError(f"{field}.confidence is required.")


def transform_matrix_from_spec(spec: dict[str, Any]) -> np.ndarray:
    translation = [float(v) for v in spec["translation_m"]]
    rpy = _rpy_deg_to_rad(spec.get("rotation_rpy_deg", [0.0, 0.0, 0.0]))
    return make_transform(translation, rpy=rpy)


def transforms_from_config(config: dict[str, Any]) -> dict[str, np.ndarray]:
    return {field: transform_matrix_from_spec(config[field]) for field in TRANSFORM_FIELDS}


def transform_chain_json(config: dict[str, Any]) -> dict[str, Any]:
    matrices = transforms_from_config(config)
    return {
        "tool_frame_config_path": config.get("_path"),
        "frame_schema_version": config.get("frame_schema_version"),
        "calibration_status": config.get("calibration_status"),
        "transforms": {name: matrix.tolist() for name, matrix in matrices.items()},
        "transform_metadata": {
            name: {
                "source": config[name].get("source"),
                "confidence": config[name].get("confidence"),
                "translation_m": config[name].get("translation_m"),
                "rotation_rpy_deg": config[name].get("rotation_rpy_deg"),
            }
            for name in TRANSFORM_FIELDS
        },
        "warnings": [
            "These transforms are bookkeeping assumptions, not final hand-eye calibration.",
            "Do not treat identity or approximate transforms as calibrated.",
        ],
    }

