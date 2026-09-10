"""Vision geometry helpers for mask-centered robot teaching."""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np


def centroid_from_mask(mask: np.ndarray) -> list[float]:
    binary = (mask > 0).astype(np.uint8)
    m = cv2.moments(binary)
    if abs(m["m00"]) <= 1e-12:
        raise ValueError("Mask has zero area.")
    return [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])]


def pca_axes_from_mask(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 3:
        raise ValueError("Mask needs at least three nonzero pixels for PCA.")
    pts = np.column_stack([xs, ys]).astype(float)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eig(cov)
    order = np.argsort(vals)[::-1]
    major = vecs[:, order[0]]
    minor = vecs[:, order[1]]
    return {
        "centroid_px": [float(mean[0]), float(mean[1])],
        "major_axis_unit_px": [float(major[0]), float(major[1])],
        "minor_axis_unit_px": [float(minor[0]), float(minor[1])],
        "eigenvalues": [float(vals[order[0]]), float(vals[order[1]])],
    }


def selected_axis_angle_deg(axis_px: Iterable[float]) -> float:
    axis = np.asarray(list(axis_px), dtype=float)
    if axis.shape != (2,):
        raise ValueError("axis_px must contain two values.")
    return float(math.degrees(math.atan2(axis[1], axis[0])))


def estimate_tool_center_pixel(
    image_width: int,
    image_height: int,
    camera_to_tool_center_offset_m: dict[str, float],
    meters_per_pixel_x: float,
    meters_per_pixel_y: float,
) -> list[float]:
    return [
        float(image_width) / 2.0 + float(camera_to_tool_center_offset_m["x"]) / float(meters_per_pixel_x),
        float(image_height) / 2.0 + float(camera_to_tool_center_offset_m["y"]) / float(meters_per_pixel_y),
    ]


def pixel_error(target_px: Iterable[float], tool_center_px: Iterable[float]) -> list[float]:
    target = list(target_px)
    tool = list(tool_center_px)
    return [float(target[0]) - float(tool[0]), float(target[1]) - float(tool[1])]


def legacy_pixel_scale_xy_correction(
    err_uv: Iterable[float],
    meters_per_pixel_x: float,
    meters_per_pixel_y: float,
    signs: dict[str, float],
    max_step: float,
) -> dict[str, Any]:
    err = list(err_uv)
    dx = float(signs.get("robot_dx_sign_from_err_u", -1.0)) * float(err[0]) * float(meters_per_pixel_x)
    dy = float(signs.get("robot_dy_sign_from_err_v", 1.0)) * float(err[1]) * float(meters_per_pixel_y)
    norm = math.hypot(dx, dy)
    clipped = False
    if norm > float(max_step):
        scale = float(max_step) / norm
        dx *= scale
        dy *= scale
        clipped = True
    return {"dx_m": dx, "dy_m": dy, "norm_m": math.hypot(dx, dy), "clipped": clipped}


def deproject_pixel_to_camera(pixel_uv: Iterable[float], depth_m: float, intrinsics: dict[str, float]) -> list[float]:
    u, v = [float(x) for x in pixel_uv]
    z = float(depth_m)
    x = (u - float(intrinsics["ppx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["ppy"])) * z / float(intrinsics["fy"])
    return [x, y, z]


def project_camera_to_pixel(point_xyz: Iterable[float], intrinsics: dict[str, float]) -> list[float]:
    x, y, z = [float(v) for v in point_xyz]
    if z <= 0.0:
        raise ValueError("z must be positive to project.")
    u = x * float(intrinsics["fx"]) / z + float(intrinsics["ppx"])
    v = y * float(intrinsics["fy"]) / z + float(intrinsics["ppy"])
    return [u, v]


def build_target_object_frame_from_mask_depth(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "status": "placeholder_depth_model_not_mature",
        "note": "R34 records teaching corrections before relying on a full 3D target object frame.",
    }

