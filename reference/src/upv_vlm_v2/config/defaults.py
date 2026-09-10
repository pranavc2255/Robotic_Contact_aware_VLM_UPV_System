from __future__ import annotations

DEFAULT_STAGE_DIRS = {
    "capture": "01_capture",
    "target_selection": "02_target_selection",
    "geometry": "03_geometry",
    "anchor_selection": "04_anchor_selection",
    "path_length": "05_path_length",
    "robot_plan": "06_robot_plan",
    "execution": "07_execution",
}

REQUIRED_TOP_LEVEL_SECTIONS = [
    "camera",
    "input",
    "perception",
    "target_selection",
    "geometry",
    "anchor_selection",
    "path_length",
    "planning",
    "execution",
    "robot",
    "tool",
    "clamp",
    "safety",
    "logging",
]
