from dataclasses import asdict, dataclass


@dataclass
class ImageTargetResult:
    prompt: str
    label: str
    confidence: float
    rgb_path: str
    depth_path: str
    center_px: list[float] | None
    major_axis_angle_deg: float | None
    major_axis_vector: list[float] | None
    minor_axis_vector: list[float] | None
    major_axis_length_px: float | None
    minor_axis_length_px: float | None
    depth_m: float | None
    valid_target: bool
    notes: str
    gsam2_runtime_sec: float | None = None
    pipeline_runtime_sec: float | None = None
    total_runtime_sec: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def dummy_from_config(cls, config: dict) -> "ImageTargetResult":
        io_config = config.get("io", {})

        return cls(
            prompt=config.get("prompt", "unknown"),
            label="scaffold_only",
            confidence=0.0,
            rgb_path=io_config["rgb_path"],
            depth_path=io_config["depth_path"],
            center_px=None,
            major_axis_angle_deg=None,
            major_axis_vector=None,
            minor_axis_vector=None,
            major_axis_length_px=None,
            minor_axis_length_px=None,
            depth_m=None,
            valid_target=False,
            notes="Scaffold result only. Perception not implemented yet.",
        )

    @classmethod
    def from_geometry(
        cls,
        config: dict,
        geometry: dict,
        notes: str,
    ) -> "ImageTargetResult":
        io_config = config.get("io", {})

        return cls(
            prompt=config.get("prompt", "unknown"),
            label="manual_mask_validation",
            confidence=1.0,
            rgb_path=io_config["rgb_path"],
            depth_path=io_config["depth_path"],
            center_px=geometry["center_px"],
            major_axis_angle_deg=geometry["major_axis_angle_deg"],
            major_axis_vector=geometry["major_axis_vector"],
            minor_axis_vector=geometry["minor_axis_vector"],
            major_axis_length_px=geometry["major_axis_length_px"],
            minor_axis_length_px=geometry["minor_axis_length_px"],
            depth_m=None,
            valid_target=True,
            notes=notes,
        )

    @classmethod
    def from_gsam2_annotation(
        cls,
        config: dict,
        annotation: dict,
        geometry: dict,
        notes: str,
        gsam2_runtime_sec: float | None = None,
        pipeline_runtime_sec: float | None = None,
        total_runtime_sec: float | None = None,
    ) -> "ImageTargetResult":
        io_config = config.get("io", {})

        return cls(
            prompt=config.get("prompt", "unknown"),
            label=annotation["class_name"],
            confidence=annotation["score"],
            rgb_path=io_config["rgb_path"],
            depth_path=io_config["depth_path"],
            center_px=geometry["center_px"],
            major_axis_angle_deg=geometry["major_axis_angle_deg"],
            major_axis_vector=geometry["major_axis_vector"],
            minor_axis_vector=geometry["minor_axis_vector"],
            major_axis_length_px=geometry["major_axis_length_px"],
            minor_axis_length_px=geometry["minor_axis_length_px"],
            depth_m=None,
            valid_target=True,
            notes=notes,
            gsam2_runtime_sec=gsam2_runtime_sec,
            pipeline_runtime_sec=pipeline_runtime_sec,
            total_runtime_sec=total_runtime_sec,
        )
