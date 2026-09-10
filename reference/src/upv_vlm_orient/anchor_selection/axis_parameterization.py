import math

from upv_vlm_orient.anchor_selection.result_models import AxisSearchDomain


def _normalize_xy(vector_xy: tuple[float, float]) -> tuple[float, float]:
    x, y = vector_xy
    norm = math.hypot(x, y)

    if norm <= 0.0:
        raise ValueError("Axis direction must have non-zero length.")

    return (x / norm, y / norm)


def _build_evenly_spaced_samples(
    s_min: float,
    s_max: float,
    num_anchor_samples: int,
) -> list[float]:
    if num_anchor_samples <= 0:
        raise ValueError("num_anchor_samples must be positive.")

    if num_anchor_samples == 1:
        return [(s_min + s_max) / 2.0]

    step = (s_max - s_min) / float(num_anchor_samples - 1)
    return [s_min + step * idx for idx in range(num_anchor_samples)]


def build_axis_search_domain(
    object_center_xy: tuple[float, float],
    chosen_axis_unit_vector_xy: tuple[float, float],
    axis_extent_px: float,
    usable_axis_margin_ratio: float,
    num_anchor_samples: int,
    axis_name: str = "chosen_axis",
) -> AxisSearchDomain:
    """
    Build a deterministic scalar search domain along a chosen image-plane axis.

    The axis is centered on ``object_center_xy`` and spans ``axis_extent_px``.
    A symmetric margin is removed from both ends before sampling candidate
    scalar positions along the usable interval.
    """
    if axis_extent_px <= 0.0:
        raise ValueError("axis_extent_px must be positive.")

    if not 0.0 <= usable_axis_margin_ratio < 0.5:
        raise ValueError("usable_axis_margin_ratio must be in [0.0, 0.5).")

    axis_dir_xy = _normalize_xy(chosen_axis_unit_vector_xy)
    axis_perp_xy = (-axis_dir_xy[1], axis_dir_xy[0])

    half_extent = axis_extent_px / 2.0
    margin_px = axis_extent_px * usable_axis_margin_ratio
    s_min = -half_extent + margin_px
    s_max = half_extent - margin_px

    if s_min > s_max:
        raise ValueError("Usable axis interval is invalid after trimming.")

    candidate_s_values = _build_evenly_spaced_samples(
        s_min=s_min,
        s_max=s_max,
        num_anchor_samples=num_anchor_samples,
    )

    return AxisSearchDomain(
        axis_name=axis_name,
        axis_center_xy=(float(object_center_xy[0]), float(object_center_xy[1])),
        axis_dir_xy=axis_dir_xy,
        axis_perp_xy=axis_perp_xy,
        s_min=float(s_min),
        s_max=float(s_max),
        usable_margin_ratio=float(usable_axis_margin_ratio),
        candidate_s_values=candidate_s_values,
    )


def map_axis_scalar_to_point(
    axis_center_xy: tuple[float, float],
    axis_dir_xy: tuple[float, float],
    s: float,
) -> tuple[float, float]:
    """
    Map scalar position ``s`` on an axis to a 2D image point.
    """
    center_x, center_y = axis_center_xy
    axis_dx, axis_dy = axis_dir_xy
    return (center_x + s * axis_dx, center_y + s * axis_dy)
