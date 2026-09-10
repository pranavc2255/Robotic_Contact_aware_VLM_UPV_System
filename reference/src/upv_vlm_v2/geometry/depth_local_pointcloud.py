"""True 3D selected-anchor local depth width measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from upv_vlm_v2.geometry.mask_geometry import normalize_vector, pixel_span_to_mm


def _json_write(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _finite_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0.0)


def finalize_edge_bin_depth_validity(diagnostics: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the v2 depth-width validity policy to edge-bin diagnostics.

    Total local point count is diagnostic only. For the edge-bin 3D endpoint
    method, a finite endpoint distance with non-empty A/B bins is the validity
    criterion.
    """
    cfg = config or {}
    path_cfg = cfg.get("path_length", cfg)
    edge_bin = diagnostics.get("edge_bin") if isinstance(diagnostics.get("edge_bin"), dict) else {}
    span = diagnostics.get("span_mm_3d_edge_bin")
    edge_a_count = int(edge_bin.get("edge_bin_a_count") or 0)
    edge_b_count = int(edge_bin.get("edge_bin_b_count") or 0)
    edge_bin_width_computed = _finite_positive(span)
    threshold = path_cfg.get("min_valid_depth_points", path_cfg.get("min_top_face_points", 1000))
    try:
        threshold_value = int(threshold) if threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None
    used_count = int(diagnostics.get("used_point_count_after_downsampling") or 0)
    below_threshold = bool(threshold_value is not None and used_count < threshold_value)
    warnings = list(diagnostics.get("depth_warnings") or [])
    if below_threshold and edge_bin_width_computed:
        msg = "total point count below configured diagnostic threshold, but edge-bin width was computed and accepted"
        if msg not in warnings:
            warnings.append(msg)
    edge_min_points = int(edge_bin.get("edge_bin_min_points") or path_cfg.get("edge_bin_min_points", 0) or 0)
    if edge_bin_width_computed and edge_min_points > 0 and (0 < edge_a_count < edge_min_points or 0 < edge_b_count < edge_min_points):
        msg = "edge bin count below configured diagnostic threshold, but endpoint distance was computed and accepted"
        if msg not in warnings:
            warnings.append(msg)

    valid = bool(edge_bin_width_computed and edge_a_count > 0 and edge_b_count > 0)
    if valid:
        failure = None
        final_mm = float(span)
    else:
        final_mm = None
        if edge_a_count <= 0 or edge_b_count <= 0:
            failure = "edge_bin_insufficient_points_for_endpoint_distance"
        else:
            failure = "edge_bin_width_not_computed"

    return {
        "depth_valid": valid,
        "depth_failure_reason": failure,
        "final_depth_path_length_mm": final_mm,
        "depth_validity_policy": "computed_edge_bin_width_is_valid",
        "min_valid_depth_points_enforced": False,
        "min_valid_depth_points_warning_only": True,
        "total_point_count_below_config_warning": below_threshold,
        "total_point_count_threshold_configured": threshold_value,
        "edge_bin_width_computed": edge_bin_width_computed,
        "depth_warnings": warnings,
    }


def _png_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
    return str(path)


def load_depth_scale_m_per_unit(config: dict[str, Any], capture_diagnostics: dict[str, Any] | None = None) -> tuple[float, str]:
    path_cfg = config.get("path_length", {})
    value = path_cfg.get("depth_scale_m_per_unit", "inherit_from_camera")
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value), "path_length.depth_scale_m_per_unit"
    if str(value).strip().lower() == "inherit_from_camera":
        capture = capture_diagnostics or config.get("_runtime_capture", {}).get("diagnostics", {})
        if isinstance(capture, dict):
            cap_value = capture.get("depth_scale_m_per_unit") or capture.get("depth_scale")
            if isinstance(cap_value, (int, float)) and float(cap_value) > 0:
                return float(cap_value), "capture_diagnostics.depth_scale_m_per_unit"
        camera_value = config.get("camera", {}).get("depth_scale_m_per_unit")
        if isinstance(camera_value, (int, float)) and float(camera_value) > 0:
            return float(camera_value), "camera.depth_scale_m_per_unit"
    fallback = path_cfg.get("fallback_depth_scale_m_per_unit", 0.001)
    return float(fallback), "path_length.fallback_depth_scale_m_per_unit"


def backproject_pixels_to_xyz(
    pixel_uv: np.ndarray,
    depth_raw: np.ndarray,
    intrinsics: dict[str, float],
    depth_scale_m_per_unit: float,
) -> np.ndarray:
    uv = np.asarray(pixel_uv, dtype=float)
    z = np.asarray(depth_raw, dtype=float) * float(depth_scale_m_per_unit)
    fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.column_stack((x, y, z))


def _depth_values_m(depth: np.ndarray, uv: np.ndarray, scale: float) -> np.ndarray:
    x = uv[:, 0].astype(int)
    y = uv[:, 1].astype(int)
    return depth[y, x].astype(float) * float(scale)


def _downsample_sorted(indices: np.ndarray, projections: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    path_cfg = cfg.get("path_length", {})
    if indices.size == 0:
        return indices
    order = np.argsort(projections[indices])
    ordered = indices[order]
    mode = str(path_cfg.get("pointcloud_downsample_mode", "stride") or "stride")
    stride = max(1, int(path_cfg.get("pointcloud_stride", 1) or 1))
    if mode == "stride" and stride > 1:
        ordered = ordered[::stride]
    max_points = path_cfg.get("max_points")
    if max_points is not None and int(max_points) > 0 and ordered.size > int(max_points):
        keep = np.linspace(0, ordered.size - 1, int(max_points)).round().astype(int)
        ordered = ordered[keep]
    return ordered


def _point_on_ray_from_pixel(pixel_xy: np.ndarray, z_m: float, intrinsics: dict[str, float]) -> np.ndarray:
    fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    return np.array([(float(pixel_xy[0]) - cx) * z_m / fx, (float(pixel_xy[1]) - cy) * z_m / fy, z_m], dtype=float)


def _save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return str(path)


def _draw_points(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: tuple[int, int, int], radius: int = 1) -> None:
    for x, y in uv:
        draw.ellipse((float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius), fill=color)


def _save_overlays(
    *,
    rgb_path: str | Path,
    mask: np.ndarray,
    axis_mode: str,
    output_dir: Path,
    anchor: np.ndarray,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    direction: np.ndarray,
    band_px: float,
    band_uv: np.ndarray,
    used_uv: np.ndarray,
    rejected_uv: np.ndarray,
    edge_a_uv: np.ndarray,
    edge_b_uv: np.ndarray,
    final_mm: float | None,
    diagnostics_text: str,
) -> dict[str, str]:
    image = Image.open(rgb_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 120, mode="L")
    tint = Image.new("RGBA", image.size, (40, 180, 80, 80))
    overlay.paste(tint, (0, 0), mask_img)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    for offset in (-band_px, band_px):
        a = endpoint_a + perp * offset
        b = endpoint_b + perp * offset
        draw.line((tuple(a), tuple(b)), fill=(255, 220, 0, 180), width=2)
    draw.line((tuple(endpoint_a), tuple(endpoint_b)), fill=(255, 40, 40, 255), width=3)
    _draw_points(draw, rejected_uv, (255, 80, 80, 120), radius=1)
    _draw_points(draw, used_uv, (40, 120, 255, 200), radius=1)
    draw.ellipse((anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5), fill=(255, 255, 0, 255))
    for p, label in ((endpoint_a, "A"), (endpoint_b, "B")):
        draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), outline=(0, 0, 0, 255), width=2)
        draw.text((p[0] + 6, p[1] + 4), label, fill=(0, 0, 0, 255))
    draw.text((12, 12), diagnostics_text, fill=(0, 0, 0, 255))
    used_path = output_dir / f"{axis_mode}_depth_used_points_overlay.png"
    overlay.save(used_path)

    edge = image.copy()
    edraw = ImageDraw.Draw(edge, "RGBA")
    edge.paste(tint, (0, 0), mask_img)
    edraw.line((tuple(endpoint_a), tuple(endpoint_b)), fill=(255, 40, 40, 255), width=3)
    _draw_points(edraw, edge_a_uv, (0, 180, 255, 220), radius=2)
    _draw_points(edraw, edge_b_uv, (255, 120, 0, 220), radius=2)
    edraw.text((12, 12), f"edge-bin final={final_mm if final_mm is not None else 'NA'} mm", fill=(0, 0, 0, 255))
    edge_path = output_dir / f"{axis_mode}_depth_edge_bins_overlay.png"
    edge.save(edge_path)
    return {"depth_used_points_overlay": str(used_path), "depth_edge_bins_overlay": str(edge_path)}


def _save_histogram(path: Path, s3: np.ndarray, edge_a_range: tuple[float, float] | None, edge_b_range: tuple[float, float] | None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if s3.size:
        plt.hist(s3 * 1000.0, bins=40, color="#4e79a7", alpha=0.85)
    if edge_a_range:
        plt.axvspan(edge_a_range[0] * 1000.0, edge_a_range[1] * 1000.0, color="#59a14f", alpha=0.25)
    if edge_b_range:
        plt.axvspan(edge_b_range[0] * 1000.0, edge_b_range[1] * 1000.0, color="#f28e2b", alpha=0.25)
    plt.xlabel("3D local projection (mm)")
    plt.ylabel("point count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def compute_true_3d_local_pointcloud_width(
    *,
    mask: np.ndarray,
    depth: np.ndarray | None,
    intrinsics: dict[str, float] | None,
    rgb_path: str | Path,
    anchor_px: list[float],
    direction_xy: list[float],
    endpoint_a_px: list[float],
    endpoint_b_px: list[float],
    mask_path_length_mm: float | None,
    mask_chord_length_px: float,
    config: dict[str, Any],
    output_dir: str | Path,
    axis_mode: str,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    path_cfg = config.get("path_length", {})
    failure = None
    if depth is None:
        failure = "depth_not_available"
    if intrinsics is None:
        failure = "missing_intrinsics"

    direction = np.asarray(normalize_vector(direction_xy), dtype=float)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    anchor = np.asarray(anchor_px, dtype=float)
    endpoint_a = np.asarray(endpoint_a_px, dtype=float)
    endpoint_b = np.asarray(endpoint_b_px, dtype=float)
    t_a = float((endpoint_a - anchor) @ direction)
    t_b = float((endpoint_b - anchor) @ direction)
    t_min, t_max = sorted((t_a, t_b))
    band_px = float(path_cfg.get("local_depth_band_px", 6.0))
    margin_px = float(path_cfg.get("local_depth_projection_margin_px", 2.0))

    yy, xx = np.nonzero(mask > 0)
    pts = np.column_stack((xx.astype(float), yy.astype(float)))
    rel = pts - anchor
    projection_px = rel @ direction
    perp_dist = np.abs(rel @ perp)
    band_flags = (perp_dist <= band_px) & (projection_px >= t_min - margin_px) & (projection_px <= t_max + margin_px)
    band_uv = pts[band_flags]
    local_band_mask = np.zeros(mask.shape, dtype=np.uint8)
    local_band_mask[band_uv[:, 1].astype(int), band_uv[:, 0].astype(int)] = 1 if band_uv.size else 0
    artifacts["depth_local_band_mask"] = _png_mask(local_band_mask, out / f"{axis_mode}_depth_local_band_mask.png")

    depth_scale, depth_scale_source = load_depth_scale_m_per_unit(config)
    min_z = float(path_cfg.get("min_valid_depth_m", 0.05))
    max_z = float(path_cfg.get("max_valid_depth_m", 2.0))
    valid_before_z = np.zeros((band_uv.shape[0],), dtype=bool)
    depth_m = np.zeros((band_uv.shape[0],), dtype=float)
    if failure is None and band_uv.size:
        depth_m = _depth_values_m(depth, band_uv, depth_scale)
        valid_before_z = np.isfinite(depth_m) & (depth_m > 0) & (depth_m >= min_z) & (depth_m <= max_z)

    median_z = float(np.median(depth_m[valid_before_z])) if np.any(valid_before_z) else None
    mad_z = float(np.median(np.abs(depth_m[valid_before_z] - median_z))) if median_z is not None and np.any(valid_before_z) else None
    z_keep = valid_before_z.copy()
    if bool(path_cfg.get("z_filter_enabled", True)) and median_z is not None:
        tolerance = max(
            float(path_cfg.get("z_filter_abs_tolerance_m", 0.015)),
            float(path_cfg.get("z_filter_mad_multiplier", 3.0)) * float(mad_z or 0.0),
        )
        z_keep = valid_before_z & (np.abs(depth_m - median_z) <= tolerance)
    valid_indices = np.nonzero(z_keep)[0]
    used_indices = _downsample_sorted(valid_indices, projection_px[band_flags], config)
    used_uv = band_uv[used_indices] if used_indices.size else np.empty((0, 2), dtype=float)
    rejected_flags = np.ones((band_uv.shape[0],), dtype=bool)
    rejected_flags[used_indices] = False
    rejected_uv = band_uv[rejected_flags]

    valid_points_mask = np.zeros(mask.shape, dtype=np.uint8)
    rejected_points_mask = np.zeros(mask.shape, dtype=np.uint8)
    if used_uv.size:
        valid_points_mask[used_uv[:, 1].astype(int), used_uv[:, 0].astype(int)] = 1
    if rejected_uv.size:
        rejected_points_mask[rejected_uv[:, 1].astype(int), rejected_uv[:, 0].astype(int)] = 1
    artifacts["depth_valid_points_mask"] = _png_mask(valid_points_mask, out / f"{axis_mode}_depth_valid_points_mask.png")
    artifacts["depth_rejected_points_mask"] = _png_mask(rejected_points_mask, out / f"{axis_mode}_depth_rejected_points_mask.png")

    xyz = np.empty((0, 3), dtype=float)
    s3 = np.empty((0,), dtype=float)
    used_projection_px = np.empty((0,), dtype=float)
    if failure is None and used_indices.size:
        raw_depth = depth[used_uv[:, 1].astype(int), used_uv[:, 0].astype(int)]
        xyz = backproject_pixels_to_xyz(used_uv, raw_depth, intrinsics, depth_scale)
        z_ref = float(np.median(xyz[:, 2]))
        p0 = _point_on_ray_from_pixel(anchor, z_ref, intrinsics)
        p1 = _point_on_ray_from_pixel(anchor + direction, z_ref, intrinsics)
        dir3 = p1 - p0
        norm = float(np.linalg.norm(dir3))
        dir3 = dir3 / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        s3 = xyz @ dir3
        used_projection_px = projection_px[band_flags][used_indices]
    else:
        dir3 = np.array([1.0, 0.0, 0.0])

    span_px_diag = None
    span_mm_pixel = None
    span_minmax = None
    span_pct = None
    span_edge = None
    edge_payload: dict[str, Any] = {
        "edge_bin_width_px_requested": float(path_cfg.get("edge_bin_width_px", 8.0)),
        "edge_bin_min_points": int(path_cfg.get("edge_bin_min_points", 20)),
    }
    edge_a_uv = np.empty((0, 2), dtype=float)
    edge_b_uv = np.empty((0, 2), dtype=float)
    edge_a_range = None
    edge_b_range = None
    if xyz.shape[0] > 0:
        lo, hi = np.percentile(used_projection_px, [2, 98])
        span_px_diag = float(hi - lo)
        span_mm_pixel = pixel_span_to_mm(span_px_diag, [float(direction[0]), float(direction[1])], median_z, intrinsics)
        span_minmax = float((np.max(s3) - np.min(s3)) * 1000.0)
        s_lo, s_hi = np.percentile(s3, [2, 98])
        span_pct = float((s_hi - s_lo) * 1000.0)
        edge_width = float(path_cfg.get("edge_bin_width_px", 8.0))
        edge_max = float(path_cfg.get("edge_bin_max_width_px", 30.0))
        edge_step = float(path_cfg.get("edge_bin_expand_step_px", 2.0))
        edge_min_points = int(path_cfg.get("edge_bin_min_points", 20))
        width_a = edge_width
        width_b = edge_width
        mask_a = used_projection_px <= t_min + width_a
        while np.count_nonzero(mask_a) < edge_min_points and width_a < edge_max:
            width_a += edge_step
            mask_a = used_projection_px <= t_min + width_a
        mask_b = used_projection_px >= t_max - width_b
        while np.count_nonzero(mask_b) < edge_min_points and width_b < edge_max:
            width_b += edge_step
            mask_b = used_projection_px >= t_max - width_b
        edge_a_uv = used_uv[mask_a]
        edge_b_uv = used_uv[mask_b]
        edge_payload.update(
            {
                "edge_bin_width_px_used_a": float(width_a),
                "edge_bin_width_px_used_b": float(width_b),
                "edge_bin_a_count": int(np.count_nonzero(mask_a)),
                "edge_bin_b_count": int(np.count_nonzero(mask_b)),
            }
        )
        if np.count_nonzero(mask_a) > 0 and np.count_nonzero(mask_b) > 0:
            edge_a_range = (float(np.min(s3[mask_a])), float(np.max(s3[mask_a])))
            edge_b_range = (float(np.min(s3[mask_b])), float(np.max(s3[mask_b])))
            # Recover true endpoint positions from the edge bins: use endpoint pixel
            # projections with median edge-bin depth instead of the old interior percentile.
            z_a = float(np.median(xyz[mask_a, 2]))
            z_b = float(np.median(xyz[mask_b, 2]))
            edge_a_xyz = _point_on_ray_from_pixel(endpoint_a, z_a, intrinsics)
            edge_b_xyz = _point_on_ray_from_pixel(endpoint_b, z_b, intrinsics)
            span_edge = float(np.linalg.norm(edge_b_xyz - edge_a_xyz) * 1000.0)
            edge_payload.update(
                {
                    "edge_a_xyz_m": [float(v) for v in edge_a_xyz],
                    "edge_b_xyz_m": [float(v) for v in edge_b_xyz],
                    "edge_endpoint_distance_mm": span_edge,
                }
            )
        else:
            failure = failure or "edge_bin_insufficient_points_for_endpoint_distance"

    rgb_arr = None
    try:
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_arr_all = np.asarray(rgb_img)
        rgb_arr = rgb_arr_all[used_uv[:, 1].astype(int), used_uv[:, 0].astype(int)] if used_uv.size else None
    except Exception:
        rgb_arr = None
    artifacts["depth_local_pointcloud_npz"] = str(out / f"{axis_mode}_depth_local_pointcloud_crop.npz")
    np.savez(
        artifacts["depth_local_pointcloud_npz"],
        xyz_m=xyz,
        uv_px=used_uv,
        depth_m=xyz[:, 2] if xyz.size else np.empty((0,), dtype=float),
        projection_3d_m=s3,
        projection_px=used_projection_px,
        endpoint_a_px=endpoint_a,
        endpoint_b_px=endpoint_b,
        anchor_px=anchor,
        direction_xy=direction,
        intrinsics_json=json.dumps(intrinsics or {}),
        depth_scale_m_per_unit=float(depth_scale),
    )
    artifacts["depth_local_pointcloud_ply"] = _save_ply(out / f"{axis_mode}_depth_local_pointcloud_crop.ply", xyz, rgb_arr)
    artifacts["depth_projection_histogram"] = _save_histogram(out / f"{axis_mode}_depth_projection_histogram.png", s3, edge_a_range, edge_b_range)
    overlay_paths = _save_overlays(
        rgb_path=rgb_path,
        mask=mask,
        axis_mode=axis_mode,
        output_dir=out,
        anchor=anchor,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        direction=direction,
        band_px=band_px,
        band_uv=band_uv,
        used_uv=used_uv,
        rejected_uv=rejected_uv,
        edge_a_uv=edge_a_uv,
        edge_b_uv=edge_b_uv,
        final_mm=span_edge,
        diagnostics_text=f"used={xyz.shape[0]} depth={span_edge if span_edge is not None else 'NA'}mm",
    )
    artifacts.update(overlay_paths)
    diagnostics = {
        "method": "true_3d_local_pointcloud_edge_bin",
        "final_depth_width_method": "edge_bin_3d_endpoint_distance",
        "depth_valid": False,
        "depth_failure_reason": failure,
        "camera_intrinsics": intrinsics,
        "depth_scale_m_per_unit": float(depth_scale),
        "depth_scale_source": depth_scale_source,
        "axis_mode": axis_mode,
        "anchor_px": [float(v) for v in anchor],
        "endpoint_a_px": [float(v) for v in endpoint_a],
        "endpoint_b_px": [float(v) for v in endpoint_b],
        "direction_xy": [float(v) for v in direction],
        "perpendicular_xy": [float(v) for v in perp],
        "local_depth_band_px": band_px,
        "local_depth_projection_margin_px": margin_px,
        "pointcloud_downsample_mode": path_cfg.get("pointcloud_downsample_mode", "stride"),
        "pointcloud_stride": int(path_cfg.get("pointcloud_stride", 1) or 1),
        "max_points": path_cfg.get("max_points"),
        "no_aggressive_downsampling": bool(path_cfg.get("no_aggressive_downsampling", True)),
        "raw_mask_pixel_count": int(np.count_nonzero(mask > 0)),
        "raw_band_pixel_count": int(band_uv.shape[0]),
        "valid_depth_point_count_before_z_filter": int(np.count_nonzero(valid_before_z)),
        "valid_depth_point_count_after_z_filter": int(valid_indices.size),
        "used_point_count_after_downsampling": int(xyz.shape[0]),
        "rejected_point_count": int(band_uv.shape[0] - xyz.shape[0]),
        "depth_median_m": median_z,
        "depth_mad_m": mad_z,
        "depth_min_m": float(np.min(depth_m[valid_before_z])) if np.any(valid_before_z) else None,
        "depth_max_m": float(np.max(depth_m[valid_before_z])) if np.any(valid_before_z) else None,
        "mask_chord_length_px": float(mask_chord_length_px),
        "mask_path_length_mm": mask_path_length_mm,
        "span_px_percentile_2_98": span_px_diag,
        "span_mm_pixel_percentile_2_98": span_mm_pixel,
        "span_mm_3d_minmax": span_minmax,
        "span_mm_3d_percentile_2_98": span_pct,
        "span_mm_3d_edge_bin": span_edge,
        "edge_bin": edge_payload,
        "final_depth_path_length_mm": None,
        "artifacts": artifacts,
    }
    validity = finalize_edge_bin_depth_validity(diagnostics, config)
    if failure is not None:
        # Keep pre-edge failures such as missing depth/intrinsics as failures.
        validity["depth_valid"] = False
        validity["depth_failure_reason"] = failure
        validity["final_depth_path_length_mm"] = None
    diagnostics.update(validity)
    artifacts["depth_width_diagnostics"] = _json_write(out / f"{axis_mode}_depth_width_diagnostics.json", diagnostics)
    diagnostics["artifacts"] = artifacts
    return diagnostics
