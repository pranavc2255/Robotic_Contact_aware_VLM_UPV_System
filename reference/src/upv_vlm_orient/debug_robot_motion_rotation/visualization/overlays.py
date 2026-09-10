"""Overlay drawing helpers for rotation debugging.

Provenance: copied/adapted from overlay logic in R3/R16/R23/R28 scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _pt(pixel: list[float]) -> tuple[int, int]:
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def draw_anchor_pixel(image: np.ndarray, anchor_px: list[float], color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    out = image.copy()
    cv2.circle(out, _pt(anchor_px), 9, color, -1, cv2.LINE_AA)
    cv2.putText(out, "anchor", (_pt(anchor_px)[0] + 12, _pt(anchor_px)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def draw_tool_center_pixel(image: np.ndarray, tool_center_px: list[float], color: tuple[int, int, int] = (255, 0, 180)) -> np.ndarray:
    out = image.copy()
    cv2.drawMarker(out, _pt(tool_center_px), color, cv2.MARKER_CROSS, 28, 3)
    cv2.putText(out, "tool", (_pt(tool_center_px)[0] + 12, _pt(tool_center_px)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def draw_image_center(image: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), color, cv2.MARKER_TILTED_CROSS, 28, 3)
    return out


def draw_selected_axis(image: np.ndarray, center_px: list[float], axis_unit_px: list[float], length_px: float = 180.0) -> np.ndarray:
    out = image.copy()
    cx, cy = float(center_px[0]), float(center_px[1])
    ax, ay = float(axis_unit_px[0]), float(axis_unit_px[1])
    p0 = [cx - ax * length_px / 2.0, cy - ay * length_px / 2.0]
    p1 = [cx + ax * length_px / 2.0, cy + ay * length_px / 2.0]
    cv2.line(out, _pt(p0), _pt(p1), (255, 0, 255), 4, cv2.LINE_AA)
    return out


def draw_robot_correction_arrow(image: np.ndarray, tool_center_px: list[float], target_px: list[float]) -> np.ndarray:
    out = image.copy()
    cv2.arrowedLine(out, _pt(tool_center_px), _pt(target_px), (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.08)
    return out


def draw_text_panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    width = min(out.shape[1] - 20, 980)
    height = 24 + 28 * len(lines)
    cv2.rectangle(out, (10, 10), (width, height), (245, 245, 245), -1)
    cv2.rectangle(out, (10, 10), (width, height), (30, 30, 30), 2)
    for idx, line in enumerate(lines):
        cv2.putText(out, line, (24, 40 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def save_rotation_debug_overlay(
    *,
    color_path: Path,
    output_path: Path,
    anchor_px: list[float],
    tool_center_px: list[float],
    axis_unit_px: list[float],
    correction: dict[str, Any],
) -> Path:
    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {color_path}")
    out = draw_image_center(image)
    out = draw_selected_axis(out, anchor_px, axis_unit_px)
    out = draw_anchor_pixel(out, anchor_px)
    out = draw_tool_center_pixel(out, tool_center_px)
    out = draw_robot_correction_arrow(out, tool_center_px, anchor_px)
    out = draw_text_panel(
        out,
        [
            f"du={correction.get('delta_u_px')} dv={correction.get('delta_v_px')}",
            f"dx={correction.get('delta_x_base_m')} dy={correction.get('delta_y_base_m')}",
            f"method={correction.get('method')} clipped={correction.get('clipped')}",
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return output_path

