"""Parse V2a/R1B perception outputs for rotation debugging.

Provenance: copied/adapted from terminal_scripts/run_r1b_realsense_v2a_perception_once.py,
terminal_scripts/run_r3_camera_frame_robot_candidate_from_r1b.py, and
src/upv_vlm_orient/anchor_selection/robot_anchor_geometry.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PerceptionOutputError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PerceptionOutputError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PerceptionOutputError(f"Invalid JSON in {path}: {exc}") from exc


def find_perception_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        data = _load_json(path)
        if data.get("perception_output_dir"):
            return Path(data["perception_output_dir"]).resolve()
        return path.parent
    if (path / "perception_run").exists():
        children = [item for item in (path / "perception_run").iterdir() if item.is_dir()]
        if children:
            return max(children, key=lambda item: item.stat().st_mtime).resolve()
    if (path / "final_decision.json").exists():
        return path
    raise PerceptionOutputError(f"Could not find perception run dir from {path}")


def load_robot_anchor_geometry(perception_run_dir: Path) -> dict[str, Any]:
    return _load_json(perception_run_dir / "robot_anchor_geometry.json")


def load_final_decision(perception_run_dir: Path) -> dict[str, Any]:
    return _load_json(perception_run_dir / "final_decision.json")


def load_anchor_count_metadata(perception_run_dir: Path) -> dict[str, Any] | None:
    path = perception_run_dir / "anchor_count_metadata.json"
    return _load_json(path) if path.exists() else None


def get_selected_anchor(perception_run_dir: Path) -> dict[str, Any]:
    geometry = load_robot_anchor_geometry(perception_run_dir)
    selected_id = geometry.get("final_anchor_id")
    if not selected_id:
        decision = load_final_decision(perception_run_dir)
        selected_id = decision.get("selected_anchor") or decision.get("final_selected_anchor")
    for anchor in geometry.get("anchors", []) or []:
        if anchor.get("anchor_id") == selected_id:
            return anchor
    raise PerceptionOutputError(f"Selected anchor {selected_id!r} not found in robot_anchor_geometry.json")


def get_selected_axis_unit_px(perception_run_dir: Path) -> list[float]:
    geometry = load_robot_anchor_geometry(perception_run_dir)
    target_geometry = geometry.get("target_geometry") or {}
    axis = target_geometry.get("selected_axis_unit_px")
    if isinstance(axis, list) and len(axis) == 2:
        return [float(axis[0]), float(axis[1])]
    anchor = get_selected_anchor(perception_run_dir)
    axis = anchor.get("selected_axis_unit_px")
    if isinstance(axis, list) and len(axis) == 2:
        return [float(axis[0]), float(axis[1])]
    raise PerceptionOutputError("Could not resolve selected axis unit vector.")


def get_selected_anchor_pixel(perception_run_dir: Path) -> list[float]:
    anchor = get_selected_anchor(perception_run_dir)
    pixel = anchor.get("anchor_center_px")
    if isinstance(pixel, list) and len(pixel) == 2:
        return [float(pixel[0]), float(pixel[1])]
    raise PerceptionOutputError("Selected anchor is missing anchor_center_px.")

