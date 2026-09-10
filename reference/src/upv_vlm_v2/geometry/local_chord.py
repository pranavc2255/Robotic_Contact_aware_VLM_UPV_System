"""Selected-anchor local chord/path-length calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from upv_vlm_v2.geometry.mask_geometry import (
    load_camera_intrinsics,
    load_depth_array,
    load_mask,
    normalize_vector,
    pixel_span_to_mm,
    representative_depth_m,
)
from upv_vlm_v2.geometry.depth_local_pointcloud import compute_true_3d_local_pointcloud_width, finalize_edge_bin_depth_validity
from upv_vlm_v2.geometry.overlays import save_selected_mask_overlay


def _is_in_mask(mask: np.ndarray, point: np.ndarray) -> bool:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x] > 0)


def _search_endpoint(mask: np.ndarray, anchor: np.ndarray, direction: np.ndarray, max_distance: float, step_px: float) -> np.ndarray | None:
    last = None
    distance = 0.0
    while distance <= max_distance:
        p = anchor + direction * distance
        if _is_in_mask(mask, p):
            last = p.copy()
            distance += step_px
            continue
        break
    return last


def compute_chord_endpoints(
    mask: np.ndarray,
    anchor_px: list[float] | tuple[float, float],
    direction_xy: list[float] | tuple[float, float],
    *,
    step_px: float = 1.0,
) -> dict[str, Any]:
    direction = np.asarray(normalize_vector(direction_xy), dtype=float)
    anchor = np.asarray(anchor_px, dtype=float)
    max_distance = float(max(mask.shape) * 1.5)
    a = _search_endpoint(mask, anchor, -direction, max_distance, step_px)
    b = _search_endpoint(mask, anchor, direction, max_distance, step_px)
    if a is None or b is None:
        return {"valid": False, "failure_reason": "anchor line did not intersect mask on both sides"}
    length_px = float(np.linalg.norm(b - a))
    return {
        "valid": True,
        "endpoint_a_px": [float(a[0]), float(a[1])],
        "endpoint_b_px": [float(b[0]), float(b[1])],
        "length_px": length_px,
        "direction_xy": [float(direction[0]), float(direction[1])],
    }


def _depth_refined_local_span(
    *,
    mask: np.ndarray,
    depth: np.ndarray | None,
    anchor_px: list[float],
    direction_xy: list[float],
    intrinsics: dict[str, float] | None,
    config: dict[str, Any],
    rgb_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    axis_mode: str = "major",
    endpoint_a_px: list[float] | None = None,
    endpoint_b_px: list[float] | None = None,
    mask_path_length_mm: float | None = None,
    mask_chord_length_px: float | None = None,
) -> dict[str, Any]:
    if depth is None:
        return {"depth_valid": False, "depth_failure_reason": "depth_not_available"}
    if bool(config.get("path_length", {}).get("use_true_3d_local_pointcloud_width", True)):
        if intrinsics is None:
            return {"depth_valid": False, "depth_failure_reason": "missing_intrinsics"}
        if rgb_path is None or output_dir is None or endpoint_a_px is None or endpoint_b_px is None:
            return {"depth_valid": False, "depth_failure_reason": "missing_true_3d_inputs"}
        diagnostics = compute_true_3d_local_pointcloud_width(
            mask=mask,
            depth=depth,
            intrinsics=intrinsics,
            rgb_path=rgb_path,
            anchor_px=anchor_px,
            direction_xy=direction_xy,
            endpoint_a_px=endpoint_a_px,
            endpoint_b_px=endpoint_b_px,
            mask_path_length_mm=mask_path_length_mm,
            mask_chord_length_px=float(mask_chord_length_px or 0.0),
            config=config,
            output_dir=output_dir,
            axis_mode=axis_mode,
        )
        diagnostics.update(finalize_edge_bin_depth_validity(diagnostics, config))
        return {
            "depth_valid": bool(diagnostics.get("depth_valid", False)),
            "depth_failure_reason": diagnostics.get("depth_failure_reason"),
            "depth_path_length_mm": diagnostics.get("final_depth_path_length_mm"),
            "top_face_point_count": diagnostics.get("used_point_count_after_downsampling"),
            "depth_median_m": diagnostics.get("depth_median_m"),
            "depth_mad_m": diagnostics.get("depth_mad_m"),
            "depth_diagnostics": diagnostics,
            "depth_artifacts": diagnostics.get("artifacts", {}),
        }
    direction = np.asarray(normalize_vector(direction_xy), dtype=float)
    anchor = np.asarray(anchor_px, dtype=float)
    yy, xx = np.nonzero(mask > 0)
    if len(xx) == 0:
        return {"depth_valid": False, "depth_failure_reason": "empty_mask"}
    pts = np.column_stack((xx.astype(float), yy.astype(float)))
    rel = pts - anchor
    perp = np.array([-direction[1], direction[0]], dtype=float)
    band_px = float(config.get("path_length", {}).get("local_depth_band_px", 6.0))
    in_band = np.abs(rel @ perp) <= band_px
    band_pts = pts[in_band]
    if band_pts.shape[0] < int(config.get("path_length", {}).get("min_top_face_points", 1000)):
        min_points = min(30, int(config.get("path_length", {}).get("min_top_face_points", 1000)))
    else:
        min_points = int(config.get("path_length", {}).get("min_top_face_points", 1000))
    values = []
    projections = []
    d = depth.astype("float32")
    for x, y in band_pts:
        xi, yi = int(round(x)), int(round(y))
        if yi < 0 or yi >= d.shape[0] or xi < 0 or xi >= d.shape[1]:
            continue
        z = float(d[yi, xi])
        if not np.isfinite(z) or z <= 0:
            continue
        values.append(z / 1000.0 if z > 20 else z)
        projections.append(float((np.array([x, y]) - anchor) @ direction))
    if len(values) < min_points:
        return {
            "depth_valid": False,
            "depth_failure_reason": f"insufficient_depth_points:{len(values)}",
            "top_face_point_count": len(values),
        }
    z_m = float(np.median(values))
    lo, hi = np.percentile(np.asarray(projections, dtype=float), [2, 98])
    span_px = float(hi - lo)
    span_mm = pixel_span_to_mm(span_px, [float(direction[0]), float(direction[1])], z_m, intrinsics)
    mad = float(np.median(np.abs(np.asarray(values) - z_m)))
    return {
        "depth_valid": span_mm is not None,
        "depth_failure_reason": None if span_mm is not None else "missing_intrinsics",
        "depth_path_length_mm": span_mm,
        "top_face_point_count": len(values),
        "depth_median_m": z_m,
        "depth_mad_m": mad,
        "depth_span_px": span_px,
    }


def compute_local_chord(
    *,
    mask_path: str | Path,
    rgb_path: str | Path,
    anchor_px: list[float],
    direction_xy: list[float],
    output_dir: str | Path,
    axis_mode: str,
    depth_path: str | Path | None = None,
    camera_info_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    mask = load_mask(mask_path)
    chord = compute_chord_endpoints(mask, anchor_px, direction_xy)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not chord["valid"]:
        return {**chord, "axis_mode": axis_mode}
    depth = load_depth_array(depth_path)
    intrinsics = load_camera_intrinsics(camera_info_path)
    z_m = representative_depth_m(mask, depth)
    mask_mm = pixel_span_to_mm(chord["length_px"], chord["direction_xy"], z_m, intrinsics)
    depth_result = _depth_refined_local_span(
        mask=mask,
        depth=depth,
        anchor_px=anchor_px,
        direction_xy=chord["direction_xy"],
        intrinsics=intrinsics,
        config=cfg,
        rgb_path=rgb_path,
        output_dir=out,
        axis_mode=axis_mode,
        endpoint_a_px=chord["endpoint_a_px"],
        endpoint_b_px=chord["endpoint_b_px"],
        mask_path_length_mm=mask_mm,
        mask_chord_length_px=chord["length_px"],
    )
    depth_mm = depth_result.get("depth_path_length_mm")
    depth_diag = depth_result.get("depth_diagnostics") or depth_result
    if depth_diag.get("final_depth_width_method") == "edge_bin_3d_endpoint_distance":
        validity = finalize_edge_bin_depth_validity(depth_diag, cfg)
        depth_diag.update(validity)
        depth_result.update(
            {
                "depth_valid": validity["depth_valid"],
                "depth_failure_reason": validity["depth_failure_reason"],
                "depth_path_length_mm": validity["final_depth_path_length_mm"],
            }
        )
        depth_mm = validity["final_depth_path_length_mm"]
    depth_artifacts = depth_result.get("depth_artifacts") or depth_diag.get("artifacts", {})
    disagreement = None
    sanity_pass = None
    if mask_mm and depth_mm:
        disagreement = abs(float(depth_mm) - float(mask_mm)) / max(float(mask_mm), 1e-6)
        sanity_pass = disagreement <= float(cfg.get("path_length", {}).get("max_depth_mask_disagreement_ratio", 0.50))
    overlay = save_selected_mask_overlay(
        rgb_path=rgb_path,
        mask_path=mask_path,
        output_path=out / f"{axis_mode}_local_chord_overlay.png",
        header=[
            f"v2 local path length: axis_mode={axis_mode}",
            f"mask={mask_mm if mask_mm is not None else 'NA'} mm, depth={depth_mm if depth_mm is not None else 'NA'} mm",
            f"depth method={depth_diag.get('final_depth_width_method', 'legacy_pixel_percentile')}, used={depth_diag.get('used_point_count_after_downsampling', depth_diag.get('top_face_point_count', 'NA'))}",
        ],
        points={"anchor": anchor_px, "A": chord["endpoint_a_px"], "B": chord["endpoint_b_px"]},
        lines=[(chord["endpoint_a_px"], chord["endpoint_b_px"], (255, 40, 40), 3)],
    )
    return {
        **chord,
        "axis_mode": axis_mode,
        "mask_path_length_mm": mask_mm,
        "depth_path_length_mm": depth_mm,
        "depth_valid": bool(depth_result.get("depth_valid", False)),
        "depth_failure_reason": depth_result.get("depth_failure_reason"),
        "depth_mask_disagreement_ratio": disagreement,
        "depth_mask_sanity_pass": sanity_pass,
        "path_length_source": "depth_edge_bin_3d" if depth_mm is not None else "mask_fallback",
        "depth_warnings": depth_diag.get("depth_warnings", []),
        "depth_validity_policy": depth_diag.get("depth_validity_policy"),
        "min_valid_depth_points_enforced": depth_diag.get("min_valid_depth_points_enforced"),
        "depth_diagnostics": depth_result,
        "depth_width_diagnostics": depth_diag,
        "depth_artifacts": depth_artifacts,
        "overlay_path": overlay,
        "depth_used_points_overlay_path": depth_artifacts.get("depth_used_points_overlay"),
        "depth_edge_bins_overlay_path": depth_artifacts.get("depth_edge_bins_overlay"),
        "depth_projection_histogram_path": depth_artifacts.get("depth_projection_histogram"),
        "depth_local_pointcloud_npz_path": depth_artifacts.get("depth_local_pointcloud_npz"),
        "depth_local_pointcloud_ply_path": depth_artifacts.get("depth_local_pointcloud_ply"),
        "depth_width_diagnostics_path": depth_artifacts.get("depth_width_diagnostics"),
    }
