"""Transform math utilities for explicit robot/camera/UPV frame bookkeeping."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _rotation_class():
    try:
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("scipy is required for robot_calibration transform utilities.") from exc
    return R


def normalize_vector(v: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(v), dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Vector must contain finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / norm


def rotvec_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    return _rotation_class().from_rotvec([rx, ry, rz]).as_matrix()


def matrix_to_rotvec(R_matrix: np.ndarray) -> np.ndarray:
    return _rotation_class().from_matrix(np.asarray(R_matrix, dtype=float)).as_rotvec()


def pose_vec_to_matrix(pose6: Iterable[float]) -> np.ndarray:
    pose = ensure_finite_pose(pose6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotvec_to_matrix(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose_vec(T: np.ndarray) -> list[float]:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Transform matrix must be finite with shape 4x4.")
    rotvec = matrix_to_rotvec(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), *[float(v) for v in rotvec]]


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("Transform matrix must have shape 4x4.")
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def compose_transforms(*Ts: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    for T in Ts:
        arr = np.asarray(T, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError("All transforms must have shape 4x4.")
        result = result @ arr
    return result


def make_transform(
    xyz: Iterable[float],
    rpy: Iterable[float] | None = None,
    rotvec: Iterable[float] | None = None,
    matrix: Iterable[Iterable[float]] | None = None,
) -> np.ndarray:
    xyz_arr = np.asarray(list(xyz), dtype=float)
    if xyz_arr.shape != (3,) or not np.isfinite(xyz_arr).all():
        raise ValueError("xyz must be three finite values.")
    supplied = sum(value is not None for value in (rpy, rotvec, matrix))
    if supplied > 1:
        raise ValueError("Provide only one of rpy, rotvec, or matrix.")
    R_cls = _rotation_class()
    if matrix is not None:
        R_matrix = np.asarray(matrix, dtype=float)
        if R_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must have shape 3x3.")
    elif rotvec is not None:
        R_matrix = R_cls.from_rotvec(np.asarray(list(rotvec), dtype=float)).as_matrix()
    elif rpy is not None:
        R_matrix = R_cls.from_euler("xyz", np.asarray(list(rpy), dtype=float)).as_matrix()
    else:
        R_matrix = np.eye(3, dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_matrix
    T[:3, 3] = xyz_arr
    return T


def transform_point(T: np.ndarray, p: Iterable[float]) -> np.ndarray:
    point = np.asarray(list(p), dtype=float)
    if point.shape != (3,):
        raise ValueError("Point must have three values.")
    hom = np.ones(4, dtype=float)
    hom[:3] = point
    return (np.asarray(T, dtype=float) @ hom)[:3]


def angle_between_unit_vectors_2d(a: Iterable[float], b: Iterable[float]) -> float:
    av = normalize_vector(a)[:2]
    bv = normalize_vector(b)[:2]
    dot = float(np.clip(np.dot(av, bv), -1.0, 1.0))
    cross = float(av[0] * bv[1] - av[1] * bv[0])
    return math.atan2(cross, dot)


def yaw_from_axis_xy(axis_xy: Iterable[float]) -> float:
    axis = normalize_vector(axis_xy)[:2]
    return float(math.atan2(axis[1], axis[0]))


def ensure_finite_pose(pose: Iterable[float]) -> list[float]:
    values = [float(v) for v in pose]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("Pose must contain six finite values.")
    return values


def pose_delta(predicted: Iterable[float], corrected: Iterable[float]) -> dict[str, list[float] | float]:
    pred = ensure_finite_pose(predicted)
    corr = ensure_finite_pose(corrected)
    delta = [corr[i] - pred[i] for i in range(6)]
    return {
        "delta_pose": delta,
        "delta_xyz_m": delta[:3],
        "delta_xyz_mm": [1000.0 * v for v in delta[:3]],
        "rotation_delta_deg": rotation_angle_between_rotvecs(pred[3:], corr[3:]),
    }


def rotation_angle_between_rotvecs(rotvec_a: Iterable[float], rotvec_b: Iterable[float]) -> float:
    R_cls = _rotation_class()
    Ra = R_cls.from_rotvec(np.asarray(list(rotvec_a), dtype=float))
    Rb = R_cls.from_rotvec(np.asarray(list(rotvec_b), dtype=float))
    rel = Rb * Ra.inv()
    return float(math.degrees(rel.magnitude()))

