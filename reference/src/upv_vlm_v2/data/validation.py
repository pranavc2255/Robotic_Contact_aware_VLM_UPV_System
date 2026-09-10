from __future__ import annotations

VALID_AXIS_MODES = {"major", "minor"}


def validate_requested_material(value: str) -> str:
    material = value.strip().lower()
    if not material:
        raise ValueError("requested_material must not be empty")
    return material


def validate_axis_mode(value: str) -> str:
    axis = value.strip().lower()
    if axis not in VALID_AXIS_MODES:
        raise ValueError(f"axis_mode must be one of {sorted(VALID_AXIS_MODES)}")
    return axis

