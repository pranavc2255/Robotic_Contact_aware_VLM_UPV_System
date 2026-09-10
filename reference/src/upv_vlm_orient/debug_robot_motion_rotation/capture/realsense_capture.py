"""Robust RealSense RGB-D capture for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r1a_realsense_capture_once.py,
terminal_scripts/run_r1b_realsense_v2a_perception_once.py, and the robust
capture style in terminal_scripts/run_r28_fastsam_mask_object_pose_once.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from upv_vlm_orient.debug_robot_motion_rotation import REPO_ROOT


CAMERA_NAME = "Intel RealSense D435i"


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureResult:
    run_dir: Path
    color_path: Path
    depth_png_path: Path
    depth_npy_path: Path
    intrinsics_path: Path
    metadata_path: Path
    color_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    metadata: dict[str, Any]


def _timestamp_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _timestamp_folder() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _timestamped_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_folder()
    for idx in range(100):
        suffix = "" if idx == 0 else f"_{idx:02d}"
        run_dir = root / f"{stamp}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise CaptureError(f"Could not create capture output folder under {root}")


def _import_realsense():
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        raise CaptureError("pyrealsense2 is required for RealSense capture.") from exc
    return rs


def _intrinsics_to_dict(intrinsics, depth_scale: float) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "coeffs": [float(value) for value in intrinsics.coeffs],
        "model": str(intrinsics.model),
        "depth_scale": float(depth_scale),
    }


def _stream_profile_details(profile, rs) -> dict[str, int]:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrinsics = color_profile.intrinsics
    depth_intrinsics = depth_profile.intrinsics
    return {
        "actual_width": int(color_intrinsics.width),
        "actual_height": int(color_intrinsics.height),
        "actual_fps": int(color_profile.fps()),
        "actual_depth_width": int(depth_intrinsics.width),
        "actual_depth_height": int(depth_intrinsics.height),
        "actual_depth_fps": int(depth_profile.fps()),
    }


def _wait_for_aligned_frame(pipeline, align, timeout_ms: int, max_attempts: int):
    last_error: Exception | None = None
    for attempt_idx in range(max_attempts):
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            last_error = exc
            print(
                f"Warning: RealSense wait_for_frames attempt {attempt_idx + 1}/{max_attempts} failed: {exc}",
                file=sys.stderr,
            )
            continue
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            last_error = CaptureError("Aligned RealSense frames did not include both color and depth.")
            continue
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        return color_frame, color, depth
    if last_error is not None:
        raise CaptureError(f"Timed out waiting for aligned RealSense color/depth frame: {last_error}") from last_error
    raise CaptureError("Timed out waiting for aligned RealSense color/depth frame.")


def _capture_attempt(
    *,
    run_dir: Path,
    rs,
    camera_config: dict[str, Any],
    requested: dict[str, int],
    startup_attempt: int,
    error_recovered: bool,
) -> CaptureResult:
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, requested["width"], requested["height"], rs.format.bgr8, requested["fps"])
    rs_config.enable_stream(rs.stream.depth, requested["width"], requested["height"], rs.format.z16, requested["fps"])
    align = rs.align(rs.stream.color)
    started = False
    try:
        profile = pipeline.start(rs_config)
        started = True
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        stream_details = _stream_profile_details(profile, rs)
        timeout_ms = int(camera_config.get("frame_timeout_ms", 15000))
        frame_attempts = int(camera_config.get("frame_wait_attempts", 3))
        for _ in range(int(camera_config.get("warmup", 30))):
            _wait_for_aligned_frame(pipeline, align, timeout_ms, frame_attempts)
        color_frame, color, depth = _wait_for_aligned_frame(pipeline, align, timeout_ms, frame_attempts)
        intrinsics = _intrinsics_to_dict(color_frame.profile.as_video_stream_profile().intrinsics, depth_scale)
        paths = {
            "color": run_dir / "color.png",
            "depth_png": run_dir / "depth_aligned.png",
            "depth_npy": run_dir / "depth_aligned.npy",
            "intrinsics": run_dir / "camera_intrinsics.json",
            "metadata": run_dir / "capture_metadata.json",
        }
        cv2.imwrite(str(paths["color"]), color)
        cv2.imwrite(str(paths["depth_png"]), depth)
        np.save(paths["depth_npy"], depth)
        paths["intrinsics"].write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
        metadata = {
            "timestamp": _timestamp_now(),
            "camera_name": CAMERA_NAME,
            "requested_width": requested["width"],
            "requested_height": requested["height"],
            "requested_fps": requested["fps"],
            **stream_details,
            "used_fallback": False,
            "frame_timeout_ms": timeout_ms,
            "startup_attempts": startup_attempt,
            "error_recovered": error_recovered,
            "color_path": str(paths["color"]),
            "depth_png_path": str(paths["depth_png"]),
            "depth_npy_path": str(paths["depth_npy"]),
            "intrinsics_path": str(paths["intrinsics"]),
            "depth_units_explanation": "Multiply raw uint16 depth by camera_intrinsics.depth_scale for meters.",
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return CaptureResult(
            run_dir=run_dir,
            color_path=paths["color"],
            depth_png_path=paths["depth_png"],
            depth_npy_path=paths["depth_npy"],
            intrinsics_path=paths["intrinsics"],
            metadata_path=paths["metadata"],
            color_shape=tuple(color.shape),
            depth_shape=tuple(depth.shape),
            metadata=metadata,
        )
    finally:
        if started:
            try:
                pipeline.stop()
            except RuntimeError as exc:
                print(f"Warning: failed to stop RealSense pipeline cleanly: {exc}", file=sys.stderr)


def capture_realsense_rgbd(config: dict[str, Any]) -> CaptureResult:
    camera = config.get("camera", {})
    output_root = _resolve_repo_path(config.get("output_root", "outputs/debug_robot_motion_rotation")) / "capture"
    run_dir = _timestamped_run_dir(output_root)
    rs = _import_realsense()
    requested = {
        "width": int(camera.get("width", 1280)),
        "height": int(camera.get("height", 720)),
        "fps": int(camera.get("fps", 30)),
    }
    retry_count = int(camera.get("startup_retry_count", 2))
    print(
        f"Rotation debug requested RealSense stream: {requested['width']}x{requested['height']} "
        f"@ {requested['fps']} FPS"
    )
    last_error: Exception | None = None
    for attempt_idx in range(1, retry_count + 1):
        try:
            result = _capture_attempt(
                run_dir=run_dir,
                rs=rs,
                camera_config=camera,
                requested=requested,
                startup_attempt=attempt_idx,
                error_recovered=last_error is not None,
            )
            print(f"capture_output_folder: {run_dir}")
            print(f"capture_color_shape: {result.color_shape}")
            return result
        except CaptureError as exc:
            last_error = exc
            print(f"Warning: capture attempt {attempt_idx}/{retry_count} failed: {exc}", file=sys.stderr)
    raise CaptureError(f"RealSense capture failed after {retry_count} startup attempts: {last_error}")

