"""File-based RGB-D acquisition helpers for v2.

This module is intentionally independent of ROS2 and model stacks. It is used
by Phase 2 saved-input modes and by checks that must not touch live hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in the repo env.
    np = None  # type: ignore[assignment]


def validate_image_file(path: str | Path) -> Path:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    return image_path


def _copy_file(source: str | Path | None, destination: Path) -> str | None:
    if source is None:
        return None
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    return str(destination)


def _write_depth_visualization(depth_path: Path | None, output_path: Path) -> str | None:
    if depth_path is None or not depth_path.exists() or np is None:
        return None
    try:
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
        depth = depth.astype("float32")
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return None
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        image = (norm * 255.0).astype("uint8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)
        return str(output_path)
    except Exception:
        return None


def load_camera_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    info_path = Path(path)
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def copy_saved_rgbd_inputs(
    *,
    rgb_path: str | Path,
    output_dir: str | Path,
    depth_path: str | Path | None = None,
    camera_info_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy saved RGB-D inputs into a v2 capture artifact directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb_src = validate_image_file(rgb_path)
    copied_rgb = _copy_file(rgb_src, out / "input_rgb.png")
    copied_depth = _copy_file(depth_path, out / f"input_depth{Path(depth_path).suffix}" if depth_path else out / "input_depth.npy")
    copied_info = _copy_file(camera_info_path, out / "camera_info.json") if camera_info_path else None
    depth_vis = _write_depth_visualization(Path(copied_depth) if copied_depth else None, out / "depth_visualization.png")
    return {
        "rgb_path": copied_rgb,
        "depth_path": copied_depth,
        "camera_info_path": copied_info,
        "depth_visualization_path": depth_vis,
        "source_rgb_path": str(rgb_src),
        "source_depth_path": str(depth_path) if depth_path else None,
        "source_camera_info_path": str(camera_info_path) if camera_info_path else None,
    }
