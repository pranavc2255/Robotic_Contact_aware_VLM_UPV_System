"""v2 transform utilities. Calibration is loaded from config, never hardcoded."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_yaml_or_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text) or {}
    except Exception:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Transform file must contain a mapping: {p}")
    return payload


def load_transform(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _load_yaml_or_json(p)


def matrix_from_transform(payload: dict[str, Any] | None) -> list[list[float]] | None:
    if not payload:
        return None
    matrix = payload.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 4:
        try:
            rows = [[float(value) for value in row] for row in matrix]
        except (TypeError, ValueError):
            return None
        if all(len(row) == 4 for row in rows):
            return rows
    translation = payload.get("translation_m")
    if isinstance(translation, dict):
        try:
            tx = float(translation.get("x", 0.0))
            ty = float(translation.get("y", 0.0))
            tz = float(translation.get("z", 0.0))
        except (TypeError, ValueError):
            return None
        return [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return None


def apply_transform(matrix: list[list[float]] | None, point_xyz: list[float] | tuple[float, float, float] | None) -> list[float] | None:
    if matrix is None or point_xyz is None or len(point_xyz) < 3:
        return None
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def _matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return _matmul4(a, b)


def _rotvec_to_matrix(rotvec: list[float] | tuple[float, float, float]) -> list[list[float]]:
    rx, ry, rz = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z = rx / theta, ry / theta, rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ]


def rotvec_to_matrix3(rotvec: list[float] | tuple[float, float, float]) -> list[list[float]]:
    return _rotvec_to_matrix(rotvec)


def _matrix_to_rotvec(r: list[list[float]]) -> list[float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return [0.0, 0.0, 0.0]
    denom = 2.0 * math.sin(theta)
    if abs(denom) < 1e-12:
        return [theta, 0.0, 0.0]
    axis = [
        (r[2][1] - r[1][2]) / denom,
        (r[0][2] - r[2][0]) / denom,
        (r[1][0] - r[0][1]) / denom,
    ]
    return [axis[0] * theta, axis[1] * theta, axis[2] * theta]


def matrix3_to_rotvec(r: list[list[float]]) -> list[float]:
    return _matrix_to_rotvec(r)


def matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def z_rotation_matrix(angle_rad: float) -> list[list[float]]:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def pose_vector_to_matrix_ur(pose: list[float] | tuple[float, ...]) -> list[list[float]]:
    if len(pose) != 6:
        raise ValueError("UR pose vector must contain 6 values [x,y,z,rx,ry,rz].")
    r = _rotvec_to_matrix([float(pose[3]), float(pose[4]), float(pose[5])])
    return [
        [r[0][0], r[0][1], r[0][2], float(pose[0])],
        [r[1][0], r[1][1], r[1][2], float(pose[1])],
        [r[2][0], r[2][1], r[2][2], float(pose[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_pose_vector_ur(matrix: list[list[float]]) -> list[float]:
    rot = [row[:3] for row in matrix[:3]]
    rotvec = _matrix_to_rotvec(rot)
    return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]), *rotvec]


def rotate_vector(matrix_or_pose: list[list[float]] | list[float], vector_xyz: list[float]) -> list[float]:
    matrix = pose_vector_to_matrix_ur(matrix_or_pose) if len(matrix_or_pose) == 6 and not isinstance(matrix_or_pose[0], list) else matrix_or_pose  # type: ignore[arg-type]
    return [
        float(matrix[0][0]) * vector_xyz[0] + float(matrix[0][1]) * vector_xyz[1] + float(matrix[0][2]) * vector_xyz[2],
        float(matrix[1][0]) * vector_xyz[0] + float(matrix[1][1]) * vector_xyz[1] + float(matrix[1][2]) * vector_xyz[2],
        float(matrix[2][0]) * vector_xyz[0] + float(matrix[2][1]) * vector_xyz[1] + float(matrix[2][2]) * vector_xyz[2],
    ]


def vector3_from_mapping(value: Any, default: list[float] | None = None) -> list[float] | None:
    if isinstance(value, dict):
        try:
            return [float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))]
        except (TypeError, ValueError):
            return default
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return default
    return default


def camera_pixel_depth_to_xyz_m(
    *,
    x_px: float,
    y_px: float,
    depth_m: float,
    intrinsics: dict[str, float] | None,
) -> list[float] | None:
    if not intrinsics:
        return None
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0 or fy <= 0 or depth_m <= 0:
        return None
    x_m = (float(x_px) - cx) * depth_m / fx
    y_m = (float(y_px) - cy) * depth_m / fy
    return [x_m, y_m, depth_m]
