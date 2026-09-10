"""Visual XY correction helpers for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r16_plan_visual_centering_correction.py.
"""

from __future__ import annotations

import math
from typing import Any


def _project_tool_center_px(image_width: int, image_height: int, config: dict[str, Any]) -> list[float]:
    offset = config.get("camera_to_tool_center_offset_m", {"x": 0.036, "y": 0.055, "z": 0.0})
    visual = config.get("visual_correction", {})
    return [
        float(image_width) / 2.0 + float(offset.get("x", 0.036)) / float(visual.get("meters_per_pixel_x", 0.0004)),
        float(image_height) / 2.0 + float(offset.get("y", 0.055)) / float(visual.get("meters_per_pixel_y", 0.0004)),
    ]


def compute_no_xy_correction() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "none",
        "delta_x_base_m": 0.0,
        "delta_y_base_m": 0.0,
        "correction_norm_applied_m": 0.0,
        "clipped": False,
    }


def compute_legacy_pixel_scale_correction(
    *,
    desired_pixel: list[float],
    image_width: int,
    image_height: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    visual = config.get("visual_correction", {})
    tool_center = _project_tool_center_px(image_width, image_height, config)
    image_center = [float(image_width) / 2.0, float(image_height) / 2.0]
    delta_u = float(desired_pixel[0]) - float(tool_center[0])
    delta_v = float(desired_pixel[1]) - float(tool_center[1])
    dx = delta_u * float(visual.get("meters_per_pixel_x", 0.0004)) * float(visual.get("robot_dx_sign_from_err_u", -1.0))
    dy = delta_v * float(visual.get("meters_per_pixel_y", 0.0004)) * float(visual.get("robot_dy_sign_from_err_v", 1.0))
    max_norm = float(visual.get("max_lateral_correction_m", 0.020))
    norm = math.hypot(dx, dy)
    clipped = norm > max_norm > 0.0
    if clipped:
        scale = max_norm / norm
        dx *= scale
        dy *= scale
    return {
        "enabled": True,
        "method": "fixed_pixel_scale_legacy",
        "desired_pixel_px": [float(desired_pixel[0]), float(desired_pixel[1])],
        "tool_center_pixel_px": tool_center,
        "image_center_px": image_center,
        "delta_u_px": delta_u,
        "delta_v_px": delta_v,
        "delta_x_base_m": dx,
        "delta_y_base_m": dy,
        "correction_norm_applied_m": math.hypot(dx, dy),
        "clipped": clipped,
        "camera_to_tool_center_offset_m": config.get("camera_to_tool_center_offset_m", {"x": 0.036, "y": 0.055, "z": 0.0}),
    }

