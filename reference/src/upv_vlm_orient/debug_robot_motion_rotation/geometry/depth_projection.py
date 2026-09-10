"""Depth projection helpers for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r2_anchor_depth_debug_from_r1b.py
and terminal_scripts/run_r3_camera_frame_robot_candidate_from_r1b.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DepthProjectionError(RuntimeError):
    pass


def lookup_depth_with_window(
    depth_image: np.ndarray,
    pixel_xy: list[float],
    depth_scale: float,
    window_radius_px: int = 5,
) -> dict[str, Any]:
    u, v = int(round(pixel_xy[0])), int(round(pixel_xy[1]))
    h, w = depth_image.shape[:2]
    x0, x1 = max(0, u - window_radius_px), min(w, u + window_radius_px + 1)
    y0, y1 = max(0, v - window_radius_px), min(h, v + window_radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        raise DepthProjectionError(f"Pixel {pixel_xy} is outside depth image bounds {w}x{h}.")
    window = depth_image[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size == 0:
        return {"valid": False, "depth_m": None, "valid_count": 0, "window_radius_px": window_radius_px}
    depth_m = float(np.median(valid.astype(np.float32)) * float(depth_scale))
    return {"valid": True, "depth_m": depth_m, "valid_count": int(valid.size), "window_radius_px": window_radius_px}


def project_pixel_to_camera_xyz(pixel_xy: list[float], depth_m: float, intrinsics: dict[str, Any]) -> list[float]:
    if depth_m <= 0.0:
        raise DepthProjectionError("depth_m must be positive.")
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    x = (u - float(intrinsics["ppx"])) * depth_m / float(intrinsics["fx"])
    y = (v - float(intrinsics["ppy"])) * depth_m / float(intrinsics["fy"])
    return [float(x), float(y), float(depth_m)]


def create_robot_point_candidate(
    *,
    anchor_id: str,
    anchor_center_px: list[float],
    selected_axis_unit_px: list[float],
    depth_image: np.ndarray,
    intrinsics: dict[str, Any],
    window_radius_px: int = 5,
) -> dict[str, Any]:
    depth_lookup = lookup_depth_with_window(
        depth_image,
        anchor_center_px,
        float(intrinsics["depth_scale"]),
        window_radius_px,
    )
    camera_xyz = None
    if depth_lookup["valid"]:
        camera_xyz = project_pixel_to_camera_xyz(anchor_center_px, float(depth_lookup["depth_m"]), intrinsics)
    return {
        "anchor_id": anchor_id,
        "anchor_center_px": anchor_center_px,
        "selected_axis_unit_px": selected_axis_unit_px,
        "depth_lookup": depth_lookup,
        "camera_xyz_m": camera_xyz,
        "valid": camera_xyz is not None,
        "robot_motion_allowed": False,
    }

