"""True edge-pair patch extraction in canonical rotated object frame."""

from __future__ import annotations

import cv2
import numpy as np


def _clip_bounds(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _extract_patch_with_marker(
    *,
    base_overlay_rgb: np.ndarray,
    candidate_id: str,
    hit_xy_rotated: tuple[float, float] | None,
    anchor_xy_rotated: tuple[float, float],
    patch_half_width_px: int,
    inside_margin_px: int,
    outside_margin_px: int,
    patch_kind: str,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, str | None]:
    h, w = base_overlay_rgb.shape[:2]
    if hit_xy_rotated is None:
        return None, None, f"Missing {patch_kind} hit point."
    hit_x, hit_y = hit_xy_rotated
    anchor_x = anchor_xy_rotated[0]
    x0 = int(round(anchor_x - patch_half_width_px))
    x1 = int(round(anchor_x + patch_half_width_px))
    if patch_kind == "top":
        y0 = int(round(hit_y - outside_margin_px))
        y1 = int(round(hit_y + inside_margin_px))
        color = (255, 70, 70)
    elif patch_kind == "bottom":
        y0 = int(round(hit_y - inside_margin_px))
        y1 = int(round(hit_y + outside_margin_px))
        color = (50, 200, 255)
    else:
        raise ValueError(f"Unsupported patch kind: {patch_kind}")
    x0, y0, x1, y1 = _clip_bounds(x0, y0, x1, y1, w, h)
    if x1 <= x0 or y1 <= y0:
        return None, None, f"Degenerate {patch_kind} patch bounds."
    patch = base_overlay_rgb[y0:y1, x0:x1].copy()
    local = (hit_x - x0, hit_y - y0)
    cv2.circle(patch, (int(round(local[0])), int(round(local[1]))), 5, color, -1)
    cv2.line(patch, (int(round(local[0] - 12)), int(round(local[1]))), (int(round(local[0] + 12)), int(round(local[1]))), color, 2)
    cv2.putText(patch, f"{candidate_id} {patch_kind}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return (x0, y0, x1, y1), patch, None


def build_true_edge_pair_patch(
    *,
    candidate_id: str,
    rotated_object_overlay_rgb: np.ndarray,
    anchor_xy_rotated: tuple[float, float],
    top_hit_xy_rotated: tuple[float, float] | None,
    bottom_hit_xy_rotated: tuple[float, float] | None,
    patch_half_width_px: int,
    edge_inside_margin_px: int,
    edge_outside_margin_px: int,
    patch_gap_px: int,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    top_bounds, top_patch, top_note = _extract_patch_with_marker(
        base_overlay_rgb=rotated_object_overlay_rgb,
        candidate_id=candidate_id,
        hit_xy_rotated=top_hit_xy_rotated,
        anchor_xy_rotated=anchor_xy_rotated,
        patch_half_width_px=patch_half_width_px,
        inside_margin_px=edge_inside_margin_px,
        outside_margin_px=edge_outside_margin_px,
        patch_kind="top",
    )
    bottom_bounds, bottom_patch, bottom_note = _extract_patch_with_marker(
        base_overlay_rgb=rotated_object_overlay_rgb,
        candidate_id=candidate_id,
        hit_xy_rotated=bottom_hit_xy_rotated,
        anchor_xy_rotated=anchor_xy_rotated,
        patch_half_width_px=patch_half_width_px,
        inside_margin_px=edge_inside_margin_px,
        outside_margin_px=edge_outside_margin_px,
        patch_kind="bottom",
    )
    valid = top_patch is not None and bottom_patch is not None
    patch = {
        "candidate_id": candidate_id,
        "anchor_xy_rotated": anchor_xy_rotated,
        "top_hit_xy_rotated": top_hit_xy_rotated,
        "bottom_hit_xy_rotated": bottom_hit_xy_rotated,
        "top_patch_bounds": top_bounds,
        "bottom_patch_bounds": bottom_bounds,
        "combined_pair_image_path": None,
        "valid": valid,
        "note": top_note or bottom_note,
    }
    if not valid:
        return patch, top_patch, bottom_patch, None
    width = max(top_patch.shape[1], bottom_patch.shape[1])
    height = top_patch.shape[0] + patch_gap_px + bottom_patch.shape[0]
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    combined[:, :] = np.array([20, 20, 22], dtype=np.uint8)
    tx = (width - top_patch.shape[1]) // 2
    bx = (width - bottom_patch.shape[1]) // 2
    combined[: top_patch.shape[0], tx : tx + top_patch.shape[1]] = top_patch
    by = top_patch.shape[0] + patch_gap_px
    combined[by : by + bottom_patch.shape[0], bx : bx + bottom_patch.shape[1]] = bottom_patch
    cv2.putText(combined, candidate_id, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return patch, top_patch, bottom_patch, combined
