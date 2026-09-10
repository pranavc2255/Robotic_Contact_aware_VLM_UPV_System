"""v2 UPV tool geometry utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

from upv_vlm_v2.planning.transforms import load_transform


def load_tool_geometry(path: str | Path | None) -> dict[str, Any] | None:
    return load_transform(path)


def selected_path_length_mm(path_result: Any) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    depth_value = getattr(path_result, "depth_path_length_mm", None)
    mask_value = getattr(path_result, "mask_path_length_mm", None)
    if depth_value is not None:
        try:
            depth_float = float(depth_value)
        except (TypeError, ValueError):
            depth_float = float("nan")
        if math.isfinite(depth_float) and depth_float > 0.0:
            return depth_float, warnings
    if mask_value is not None:
        warnings.append("depth path length unavailable; using mask local path length for planning")
        return float(mask_value), warnings
    warnings.append("no local path length available for planning")
    return None, warnings


def tool_point(tool_geometry: dict[str, Any] | None, name: str) -> list[float] | None:
    if not tool_geometry:
        return None
    points = tool_geometry.get("tool_points_m") or {}
    point = points.get(name)
    if isinstance(point, (list, tuple)) and len(point) >= 3:
        return [float(point[0]), float(point[1]), float(point[2])]
    return None


def tool_vector(tool_geometry: dict[str, Any] | None, name: str, default: list[float] | None = None) -> list[float] | None:
    if not tool_geometry:
        return default
    value = tool_geometry.get(name)
    if isinstance(value, dict):
        return [float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return default
