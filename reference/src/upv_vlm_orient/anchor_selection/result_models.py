from dataclasses import dataclass


@dataclass(frozen=True)
class AxisSearchDomain:
    axis_name: str
    axis_center_xy: tuple[float, float]
    axis_dir_xy: tuple[float, float]
    axis_perp_xy: tuple[float, float]
    s_min: float
    s_max: float
    usable_margin_ratio: float
    candidate_s_values: list[float]


@dataclass(frozen=True)
class BoundaryHit:
    side: str
    hit_xy: tuple[float, float] | None
    distance_from_anchor: float | None
    valid: bool


@dataclass(frozen=True)
class LocalEdgeSupport:
    side: str
    hit_xy: tuple[float, float]
    support_points_xy: list[tuple[float, float]]
    support_size: int
    valid: bool
    note: str | None


@dataclass(frozen=True)
class PairObservation:
    candidate_id: str
    anchor_xy: tuple[float, float]
    left_hit_xy: tuple[float, float] | None
    right_hit_xy: tuple[float, float] | None
    crop_center_xy: tuple[float, float]
    rotation_angle_deg: float
    crop_width_px: int
    crop_height_px: int
    image_path: str | None
    valid: bool
    note: str | None


@dataclass(frozen=True)
class RotatedObjectView:
    image_path: str | None
    rotated_image_path: str | None
    rotated_overlay_path: str | None
    rotation_angle_deg: float
    crop_x_min: int
    crop_y_min: int
    crop_x_max: int
    crop_y_max: int
    crop_width_px: int
    crop_height_px: int
    valid: bool
    note: str | None


@dataclass(frozen=True)
class EdgePairCrop:
    candidate_id: str
    anchor_xy_rotated: tuple[float, float]
    top_hit_xy_rotated: tuple[float, float] | None
    bottom_hit_xy_rotated: tuple[float, float] | None
    crop_x_min: int
    crop_y_min: int
    crop_x_max: int
    crop_y_max: int
    crop_width_px: int
    crop_height_px: int
    image_path: str | None
    valid: bool
    note: str | None


@dataclass(frozen=True)
class TrueEdgePairPatch:
    candidate_id: str
    anchor_xy_rotated: tuple[float, float]
    top_hit_xy_rotated: tuple[float, float] | None
    bottom_hit_xy_rotated: tuple[float, float] | None
    top_patch_bounds: tuple[int, int, int, int] | None
    bottom_patch_bounds: tuple[int, int, int, int] | None
    combined_pair_image_path: str | None
    valid: bool
    note: str | None


@dataclass(frozen=True)
class CleanEdgePairPatch:
    candidate_id: str
    top_patch_image_path: str | None
    bottom_patch_image_path: str | None
    combined_clean_pair_image_path: str | None
    valid: bool
    note: str | None
