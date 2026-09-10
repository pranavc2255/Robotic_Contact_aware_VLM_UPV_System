"""ROS2-backed RGB-D snapshot helpers for RealSense teaching pipelines."""

from .rgbd_snapshot_client import capture_snapshot_from_config
from .rgbd_snapshot_node import Ros2SnapshotUnavailable, capture_ros2_rgbd_snapshot

__all__ = [
    "Ros2SnapshotUnavailable",
    "capture_ros2_rgbd_snapshot",
    "capture_snapshot_from_config",
]
