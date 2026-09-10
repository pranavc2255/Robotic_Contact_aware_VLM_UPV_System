"""Image conversion and persistence utilities for ROS2 RGB-D snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _header_to_dict(header: Any) -> dict[str, Any]:
    if header is None:
        return {}
    stamp = getattr(header, "stamp", None)
    return {
        "frame_id": getattr(header, "frame_id", None),
        "stamp": {
            "sec": getattr(stamp, "sec", None),
            "nanosec": getattr(stamp, "nanosec", None),
        },
    }


def stamp_to_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) * 1e-9


def camera_info_to_dict(camera_info: Any) -> dict[str, Any]:
    if isinstance(camera_info, dict):
        return dict(camera_info)
    d = getattr(camera_info, "d", [])
    k = getattr(camera_info, "k", [])
    r = getattr(camera_info, "r", [])
    p = getattr(camera_info, "p", [])
    return {
        "header": _header_to_dict(getattr(camera_info, "header", None)),
        "height": getattr(camera_info, "height", None),
        "width": getattr(camera_info, "width", None),
        "distortion_model": getattr(camera_info, "distortion_model", None),
        "d": list(d) if d is not None else [],
        "k": list(k) if k is not None else [],
        "r": list(r) if r is not None else [],
        "p": list(p) if p is not None else [],
    }


def intrinsics_from_camera_info(camera_info: Any, source_topic: str | None = None) -> dict[str, Any]:
    raw = camera_info_to_dict(camera_info)
    k = raw.get("k")
    if k is None or len(k) == 0:
        k = raw.get("K")
    if k is None:
        k = []
    if len(k) < 6:
        raise ValueError("CameraInfo does not contain a valid 3x3 K matrix.")
    header = raw.get("header") or {}
    distortion = raw.get("d")
    if distortion is None or len(distortion) == 0:
        distortion = raw.get("D")
    if distortion is None:
        distortion = []
    return {
        "width": int(raw.get("width")),
        "height": int(raw.get("height")),
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
        "ppx": float(k[2]),
        "ppy": float(k[5]),
        "distortion_model": raw.get("distortion_model"),
        "distortion_coefficients": list(distortion),
        "frame_id": header.get("frame_id"),
        "source_topic": source_topic,
        "depth_scale": 0.001,
        "depth_units": "meters",
    }


def _message_data_to_uint8(data: Any) -> np.ndarray:
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except TypeError:
        return np.asarray(data, dtype=np.uint8)


def _encoding_spec(encoding: str) -> tuple[np.dtype, int]:
    enc = encoding.upper()
    if enc in {"RGB8", "BGR8"}:
        return np.dtype(np.uint8), 3
    if enc in {"RGBA8", "BGRA8"}:
        return np.dtype(np.uint8), 4
    if enc == "MONO8":
        return np.dtype(np.uint8), 1
    if enc in {"16UC1", "MONO16"}:
        return np.dtype(np.uint16), 1
    if enc == "32FC1":
        return np.dtype(np.float32), 1
    raise ValueError(f"Unsupported ROS image encoding: {encoding}")


def ros_image_to_numpy(image_msg: Any) -> np.ndarray:
    """Decode a ROS sensor_msgs/Image message directly from its byte buffer.

    The decoder respects row stride (`step`) so padded rows are accepted. It
    returns the image in the message's native channel order.
    """

    height = int(getattr(image_msg, "height"))
    width = int(getattr(image_msg, "width"))
    encoding = str(getattr(image_msg, "encoding"))
    step = int(getattr(image_msg, "step"))
    dtype, channels = _encoding_spec(encoding)
    bytes_per_pixel = dtype.itemsize * channels
    minimum_step = width * bytes_per_pixel
    if step < minimum_step:
        raise ValueError(f"Image step {step} is too small for {width} px and encoding {encoding}.")
    raw = _message_data_to_uint8(getattr(image_msg, "data"))
    required = height * step
    if raw.size < required:
        raise ValueError(f"Image data has {raw.size} bytes, expected at least {required}.")
    rows = raw[:required].reshape(height, step)
    cropped = rows[:, :minimum_step].copy()
    arr = cropped.view(dtype).reshape(height, width, channels)
    is_bigendian = bool(getattr(image_msg, "is_bigendian", False))
    native_bigendian = sys.byteorder == "big"
    if dtype.itemsize > 1 and is_bigendian != native_bigendian:
        arr = arr.byteswap()
    if channels == 1:
        return arr[:, :, 0]
    return arr


def ros_color_image_to_rgb_numpy(image_msg: Any) -> np.ndarray:
    arr = ros_image_to_numpy(image_msg)
    enc = str(getattr(image_msg, "encoding")).lower()
    if enc == "rgb8":
        return arr.copy()
    if enc == "bgr8":
        return arr[:, :, ::-1].copy()
    if enc == "rgba8":
        return arr[:, :, :3].copy()
    if enc == "bgra8":
        return arr[:, :, :3][:, :, ::-1].copy()
    if enc == "mono8":
        return np.repeat(arr[:, :, None], 3, axis=2)
    raise ValueError(f"Unsupported color image encoding: {enc}")


def ros_depth_image_to_meters_numpy(image_msg: Any) -> np.ndarray:
    return depth_image_to_meters(ros_image_to_numpy(image_msg), getattr(image_msg, "encoding", None))


def color_to_bgr(color: np.ndarray, encoding: str | None = None) -> np.ndarray:
    arr = np.asarray(color)
    enc = (encoding or "").lower()
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported color image shape: {arr.shape}")
    if enc in {"rgb8", "rgba8"}:
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    if enc in {"bgr8", "bgra8", ""}:
        return arr[:, :, :3].copy()
    return arr[:, :, :3].copy()


def depth_image_to_meters(depth: np.ndarray, encoding: str | None = None) -> np.ndarray:
    arr = np.asarray(depth)
    enc = (encoding or "").upper()
    if enc in {"16UC1", "MONO16"} or arr.dtype == np.uint16:
        return arr.astype(np.float32) * 0.001
    if enc == "32FC1" or np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported depth encoding/dtype: encoding={encoding}, dtype={arr.dtype}")


def save_rgbd_arrays(
    output_dir: str | Path,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    metadata: dict[str, Any],
    camera_info_raw: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    color_path = out / "color.png"
    depth_png_path = out / "depth_aligned.png"
    depth_npy_path = out / "depth_aligned.npy"
    intrinsics_path = out / "camera_intrinsics.json"
    metadata_path = out / "capture_metadata.json"
    camera_info_path = out / "camera_info_raw.json"

    if not cv2.imwrite(str(color_path), color_bgr):
        raise RuntimeError(f"Failed to write {color_path}")
    depth_m = np.asarray(depth_m, dtype=np.float32)
    np.save(depth_npy_path, depth_m)
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if finite.size:
        upper = float(np.percentile(finite, 99.0))
        if upper <= 0.0:
            upper = float(finite.max())
        depth_vis = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = np.clip(depth_vis / upper, 0.0, 1.0)
        depth_png = (depth_vis * 65535.0).astype(np.uint16)
    else:
        depth_png = np.zeros(depth_m.shape, dtype=np.uint16)
    if not cv2.imwrite(str(depth_png_path), depth_png):
        raise RuntimeError(f"Failed to write {depth_png_path}")
    intrinsics_path.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    if camera_info_raw is not None:
        camera_info_path.write_text(json.dumps(camera_info_raw, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "color_path": str(color_path),
        "depth_png_path": str(depth_png_path),
        "depth_npy_path": str(depth_npy_path),
        "intrinsics_path": str(intrinsics_path),
        "camera_info_raw_path": str(camera_info_path) if camera_info_raw is not None else "",
        "capture_metadata_path": str(metadata_path),
    }
