"""Live ROS2 RealSense RGB-D snapshot acquisition for v2.

ROS2 imports are intentionally lazy so importing v2 never requires ROS2.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image


def _image_msg_to_array(msg: Any) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(msg.height)
    width = int(msg.width)
    data = bytes(msg.data)
    if encoding in {"rgb8", "bgr8"}:
        arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return arr.copy()
    if encoding in {"mono16", "16uc1"}:
        return np.frombuffer(data, dtype=np.uint16).reshape((height, width)).copy()
    if encoding in {"32fc1"}:
        return np.frombuffer(data, dtype=np.float32).reshape((height, width)).copy()
    raise ValueError(f"unsupported ROS2 image encoding: {encoding}")


def _save_depth_visualization(depth: np.ndarray, output_path: Path) -> str:
    d = depth.astype("float32")
    valid = d[np.isfinite(d) & (d > 0)]
    if valid.size == 0:
        Image.fromarray(np.zeros(d.shape, dtype=np.uint8)).save(output_path)
        return str(output_path)
    lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(output_path)
    return str(output_path)


def _camera_info_payload(msg: Any) -> dict[str, Any]:
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        try:
            return list(value)
        except TypeError:
            return []

    k = _as_list(getattr(msg, "k", None))
    payload = {
        "width": int(getattr(msg, "width", 0)),
        "height": int(getattr(msg, "height", 0)),
        "distortion_model": str(getattr(msg, "distortion_model", "")),
        "d": _as_list(getattr(msg, "d", None)),
        "k": k,
        "K": k,
        "r": _as_list(getattr(msg, "r", None)),
        "p": _as_list(getattr(msg, "p", None)),
    }
    if len(k) >= 6:
        payload.update({"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])})
    return payload


def capture_ros2_realsense_snapshot(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
        from sensor_msgs.msg import CameraInfo, Image as RosImage  # type: ignore
    except Exception as exc:
        return {"success": False, "failure_reason": f"ROS2 import unavailable: {exc}"}

    camera = config.get("camera", {})
    color_topic = str(camera.get("color_topic", "/camera/color/image_raw"))
    depth_topic = str(camera.get("depth_topic", "/camera/aligned_depth_to_color/image_raw"))
    info_topic = str(camera.get("camera_info_topic", "/camera/color/camera_info"))
    timeout_sec = float(camera.get("live_timeout_sec", 5.0))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(out / "ros_logs"))
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    class SnapshotNode(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__("upv_vlm_v2_single_rgbd_snapshot")
            self.color_msg = None
            self.depth_msg = None
            self.info_msg = None
            self.create_subscription(RosImage, color_topic, lambda msg: setattr(self, "color_msg", msg), 10)
            self.create_subscription(RosImage, depth_topic, lambda msg: setattr(self, "depth_msg", msg), 10)
            self.create_subscription(CameraInfo, info_topic, lambda msg: setattr(self, "info_msg", msg), 10)

    started_here = False
    if not rclpy.ok():
        try:
            rclpy.init(args=None)
            started_here = True
        except Exception as exc:
            return {"success": False, "failure_reason": f"ROS2 init failed: {exc}"}
    node = SnapshotNode()
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.color_msg is not None and node.depth_msg is not None and node.info_msg is not None:
                break
        missing = []
        if node.color_msg is None:
            missing.append(f"color topic timeout: {color_topic}")
        if node.depth_msg is None:
            missing.append(f"depth topic timeout: {depth_topic}")
        if node.info_msg is None:
            missing.append(f"camera_info topic timeout: {info_topic}")
        if missing:
            return {"success": False, "failure_reason": "; ".join(missing)}
        rgb = _image_msg_to_array(node.color_msg)
        depth = _image_msg_to_array(node.depth_msg)
        info = _camera_info_payload(node.info_msg)
        rgb_path = out / "raw_rgb.png"
        depth_path = out / "depth_raw.npy"
        depth_vis_path = out / "depth_visualization.png"
        info_path = out / "camera_info.json"
        intr_path = out / "camera_intrinsics.json"
        meta_path = out / "capture_metadata.json"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        _save_depth_visualization(depth, depth_vis_path)
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        intr_path.write_text(json.dumps({key: info.get(key) for key in ("fx", "fy", "cx", "cy")}, indent=2), encoding="utf-8")
        metadata = {
            "source": "ros2_realsense",
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": info_topic,
            "color_encoding": str(node.color_msg.encoding),
            "depth_encoding": str(node.depth_msg.encoding),
            "depth_scale_m_per_unit": float(camera.get("depth_scale_m_per_unit", 0.001)),
            "timeout_sec": timeout_sec,
            "elapsed_sec": time.perf_counter() - start,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "success": True,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera_info_path": str(info_path),
            "depth_visualization_path": str(depth_vis_path),
            "camera_intrinsics_path": str(intr_path),
            "capture_metadata_path": str(meta_path),
            "diagnostics": metadata,
        }
    finally:
        node.destroy_node()
        if started_here:
            rclpy.shutdown()
