"""Axis parameterization for deterministic anchor search."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class AxisSearchDomain:
    axis_name: str
    axis_center_xy: tuple[float, float]
    axis_dir_xy: tuple[float, float]
    axis_perp_xy: tuple[float, float]
    s_min: float
    s_max: float
    usable_margin_ratio: float
    candidate_s_values: list[float]


def normalize_xy(vector_xy: tuple[float, float] | list[float]) -> tuple[float, float]:
    x, y = float(vector_xy[0]), float(vector_xy[1])
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError("Axis direction must have non-zero length.")
    return (x / norm, y / norm)


def _build_evenly_spaced_samples(s_min: float, s_max: float, num_anchor_samples: int) -> list[float]:
    if num_anchor_samples <= 0:
        raise ValueError("num_anchor_samples must be positive.")
    if num_anchor_samples == 1:
        return [(s_min + s_max) / 2.0]
    step = (s_max - s_min) / float(num_anchor_samples - 1)
    return [s_min + step * idx for idx in range(num_anchor_samples)]


def build_axis_search_domain(
    object_center_xy: tuple[float, float] | list[float],
    chosen_axis_unit_vector_xy: tuple[float, float] | list[float],
    axis_extent_px: float,
    usable_axis_margin_ratio: float,
    num_anchor_samples: int,
    axis_name: str = "chosen_axis",
) -> AxisSearchDomain:
    if axis_extent_px <= 0.0:
        raise ValueError("axis_extent_px must be positive.")
    if not 0.0 <= usable_axis_margin_ratio < 0.5:
        raise ValueError("usable_axis_margin_ratio must be in [0.0, 0.5).")
    axis_dir_xy = normalize_xy(chosen_axis_unit_vector_xy)
    axis_perp_xy = (-axis_dir_xy[1], axis_dir_xy[0])
    half_extent = axis_extent_px / 2.0
    margin_px = axis_extent_px * usable_axis_margin_ratio
    s_min = -half_extent + margin_px
    s_max = half_extent - margin_px
    return AxisSearchDomain(
        axis_name=axis_name,
        axis_center_xy=(float(object_center_xy[0]), float(object_center_xy[1])),
        axis_dir_xy=axis_dir_xy,
        axis_perp_xy=axis_perp_xy,
        s_min=float(s_min),
        s_max=float(s_max),
        usable_margin_ratio=float(usable_axis_margin_ratio),
        candidate_s_values=_build_evenly_spaced_samples(s_min, s_max, num_anchor_samples),
    )


def map_axis_scalar_to_point(
    axis_center_xy: tuple[float, float],
    axis_dir_xy: tuple[float, float],
    s: float,
) -> tuple[float, float]:
    return (
        float(axis_center_xy[0]) + float(s) * float(axis_dir_xy[0]),
        float(axis_center_xy[1]) + float(s) * float(axis_dir_xy[1]),
    )
