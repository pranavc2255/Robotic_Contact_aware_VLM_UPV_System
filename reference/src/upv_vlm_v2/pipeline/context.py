from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upv_vlm_v2.logging.artifact_writer import ArtifactWriter


@dataclass
class PipelineContext:
    config: dict[str, Any]
    mode: str
    requested_material: str
    axis_mode: str
    input_rgb: Path | None
    input_depth: Path | None
    output_root: Path
    session_dir: Path
    dry_run: bool
    confirm: str | None
    artifact_writer: ArtifactWriter
    input_camera_info: Path | None = None
    live_ros2: bool = False
    execution_backend: str = "none"
    allow_real_hardware: bool = False
