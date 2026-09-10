"""Axis geometry helpers for rotation debugging.

Provenance: copied/adapted from V2a robot-anchor geometry outputs and R12 axis planning.
"""

from __future__ import annotations

import math
from typing import Any


class AxisGeometryError(RuntimeError):
    pass


def normalize_xy(vector_xy: list[float], label: str = "axis") -> list[float]:
    norm = math.hypot(float(vector_xy[0]), float(vector_xy[1]))
    if norm <= 1e-12:
        raise AxisGeometryError(f"{label} has zero-length XY vector.")
    return [float(vector_xy[0]) / norm, float(vector_xy[1]) / norm]


def axis_angle_deg_image(axis_unit_px: list[float]) -> float:
    axis = normalize_xy(axis_unit_px, "axis_unit_px")
    return math.degrees(math.atan2(axis[1], axis[0]))


def selected_axis_from_geometry(robot_anchor_geometry: dict[str, Any]) -> list[float]:
    target = robot_anchor_geometry.get("target_geometry") or {}
    axis = target.get("selected_axis_unit_px")
    if isinstance(axis, list) and len(axis) == 2:
        return normalize_xy([float(axis[0]), float(axis[1])], "selected_axis_unit_px")
    raise AxisGeometryError("robot_anchor_geometry target_geometry is missing selected_axis_unit_px.")

