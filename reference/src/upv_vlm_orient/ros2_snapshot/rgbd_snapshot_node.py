"""ROS2 RealSense RGB-D snapshot node/function.

This module imports ROS2 dependencies lazily so non-ROS dry-runs and tests can
import the package on normal Python environments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .image_utils import (
    camera_info_to_dict,
    color_to_bgr,
    intrinsics_from_camera_info,
    ros_color_image_to_rgb_numpy,
    ros_depth_image_to_meters_numpy,
    save_rgbd_arrays,
    stamp_to_seconds,
)


class Ros2SnapshotUnavailable(RuntimeError):
    """Raised when ROS2 snapshot capture cannot run in the current environment."""


def _load_ros2_modules() -> dict[str, Any]:
    try:
        import message_filters  # noqa: PLC0415
        import rclpy  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415
        from sensor_msgs.msg import CameraInfo, Image  # noqa: PLC0415
    except ImportError as exc:
        raise Ros2SnapshotUnavailable(
            "ROS2 snapshot capture requires rclpy, sensor_msgs, and message_filters. "
            "Dry-run and synthetic tests do not require them."
        ) from exc
    return {
        "message_filters": message_filters,
        "rclpy": rclpy,
        "Node": Node,
        "CameraInfo": CameraInfo,
        "Image": Image,
    }


class _RgbdSnapshotNode:
    def __init__(self, modules: dict[str, Any], config: dict[str, Any]) -> None:
        self.modules = modules
        self.config = config
        self.node = modules["Node"]("upv_rgbd_snapshot_once")
        self.latest: tuple[Any, Any, Any] | None = None
        ros_cfg = config.get("ros2", config)
        queue_size = int(ros_cfg.get("queue_size", 10))
        slop = float(ros_cfg.get("sync_slop_s", 0.08))
        color_topic = ros_cfg["color_topic"]
        depth_topic = ros_cfg["depth_topic"]
        info_topic = ros_cfg["camera_info_topic"]
        mf = modules["message_filters"]
        self.color_sub = mf.Subscriber(self.node, modules["Image"], color_topic)
        self.depth_sub = mf.Subscriber(self.node, modules["Image"], depth_topic)
        self.info_sub = mf.Subscriber(self.node, modules["CameraInfo"], info_topic)
        self.sync = mf.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self.sync.registerCallback(self._callback)

    def _callback(self, color_msg: Any, depth_msg: Any, camera_info_msg: Any) -> None:
        self.latest = (color_msg, depth_msg, camera_info_msg)


def capture_ros2_rgbd_snapshot(output_dir: str | Path, config: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    modules = _load_ros2_modules()
    rclpy = modules["rclpy"]
    ros_cfg = config.get("ros2", config)
    timeout = float(timeout_s if timeout_s is not None else ros_cfg.get("timeout_s", 10.0))
    output_path = Path(output_dir)
    metadata_base = {
        "capture_backend": "ros2_realsense",
        "color_topic": ros_cfg["color_topic"],
        "depth_topic": ros_cfg["depth_topic"],
        "camera_info_topic": ros_cfg["camera_info_topic"],
        "sync_slop_s": float(ros_cfg.get("sync_slop_s", 0.08)),
        "timeout_s": timeout,
    }

    rclpy.init(args=None)
    snap_node: _RgbdSnapshotNode | None = None
    try:
        snap_node = _RgbdSnapshotNode(modules, config)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and snap_node.latest is None:
            rclpy.spin_once(snap_node.node, timeout_sec=0.1)
        if snap_node.latest is None:
            raise TimeoutError(f"No synchronized RGB-D frame arrived within {timeout:.1f} s.")

        color_msg, depth_msg, camera_info_msg = snap_node.latest
        color_rgb = ros_color_image_to_rgb_numpy(color_msg)
        depth_m = ros_depth_image_to_meters_numpy(depth_msg)
        color_bgr = color_to_bgr(color_rgb, "rgb8")
        intrinsics = intrinsics_from_camera_info(camera_info_msg, source_topic=ros_cfg["camera_info_topic"])
        color_stamp = stamp_to_seconds(getattr(getattr(color_msg, "header", None), "stamp", None))
        depth_stamp = stamp_to_seconds(getattr(getattr(depth_msg, "header", None), "stamp", None))
        sync_delta = None if color_stamp is None or depth_stamp is None else abs(color_stamp - depth_stamp)
        metadata = {
            **metadata_base,
            "color_stamp": color_stamp,
            "depth_stamp": depth_stamp,
            "sync_delta_s": sync_delta,
            "image_shape": list(color_bgr.shape),
            "depth_shape": list(depth_m.shape),
            "depth_dtype": str(depth_m.dtype),
            "color_encoding": getattr(color_msg, "encoding", None),
            "depth_encoding": getattr(depth_msg, "encoding", None),
            "success": True,
        }
        paths = save_rgbd_arrays(
            output_path,
            color_bgr,
            depth_m,
            intrinsics,
            metadata,
            camera_info_raw=camera_info_to_dict(camera_info_msg),
        )
        return {**metadata, **paths, "output_dir": str(output_path)}
    except Exception:
        output_path.mkdir(parents=True, exist_ok=True)
        failed = {**metadata_base, "success": False}
        (output_path / "capture_metadata.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise
    finally:
        if snap_node is not None:
            snap_node.node.destroy_node()
        rclpy.shutdown()


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one ROS2 RealSense RGB-D snapshot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    if args.dry_run:
        ros_cfg = config.get("ros2", config)
        print("ROS2 RGB-D snapshot dry-run")
        print(f"color_topic: {ros_cfg.get('color_topic')}")
        print(f"depth_topic: {ros_cfg.get('depth_topic')}")
        print(f"camera_info_topic: {ros_cfg.get('camera_info_topic')}")
        print("ros2_required_now: false")
        return 0
    try:
        result = capture_ros2_rgbd_snapshot(args.output_dir, config)
    except Ros2SnapshotUnavailable as exc:
        print(f"ROS2 unavailable: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ROS2 snapshot failed: {exc}")
        return 1
    print("ROS2 RGB-D snapshot saved")
    print(f"output_dir: {result['output_dir']}")
    print(f"sync_delta_s: {result.get('sync_delta_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
