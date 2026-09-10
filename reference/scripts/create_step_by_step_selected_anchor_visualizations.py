#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRIP_OFFSETS_PX = [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0]


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        raise ValueError("zero direction vector")
    return vec / n


def _depth_rgb(depth: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    valid = depth > 0
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = (matplotlib.colormaps[cmap_name](norm(depth.astype(float))) * 255).astype(np.uint8)
    rgba[~valid] = np.array([0, 0, 0, 255], dtype=np.uint8)
    return rgba[:, :, :3]


def _save_img(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, rgb)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def _rect_box(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty selected mask")
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return cv2.boxPoints(rect).astype(np.float32)


def _expand_box(box: np.ndarray, buffer_percent: float) -> np.ndarray:
    center = box.mean(axis=0)
    return center + (box - center) * (1.0 + buffer_percent / 100.0)


def _pick_ref_depth(depth: np.ndarray, p0: np.ndarray, radius: float) -> tuple[float | None, int]:
    h, w = depth.shape
    x0 = max(0, int(math.floor(p0[0] - radius)))
    x1 = min(w, int(math.ceil(p0[0] + radius + 1)))
    y0 = max(0, int(math.floor(p0[1] - radius)))
    y1 = min(h, int(math.ceil(p0[1] + radius + 1)))
    vals = depth[y0:y1, x0:x1]
    valid = vals[vals > 0]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def _strip_support(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    p0: np.ndarray,
    direction: np.ndarray,
    buffer_percent: float,
    strip_half_width_px: float,
    strip_offsets_px: list[float],
    center_region_radius_px: float,
    depth_tolerance_mm: float,
) -> dict[str, Any]:
    box = _rect_box(mask)
    expanded = _expand_box(box, buffer_percent)
    roi = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi, np.round(expanded).astype(np.int32), 1)
    ys, xs = np.nonzero(roi > 0)
    roi_uv = np.column_stack([xs.astype(float), ys.astype(float)])
    ref, ref_count = _pick_ref_depth(depth, p0, center_region_radius_px)
    perp = np.array([-direction[1], direction[0]], dtype=float)
    retained: list[np.ndarray] = []
    rejected: list[np.ndarray] = []
    strip_polys: list[np.ndarray] = []
    strip_results: list[dict[str, Any]] = []
    if ref is None:
        return {
            "valid": False,
            "failure_reason": "missing_reference_depth",
            "expanded_box": expanded,
            "retained_uv": np.empty((0, 2)),
            "rejected_uv": np.empty((0, 2)),
            "strip_polys": [],
            "strip_results": [],
            "center_depth_mm": None,
            "center_region_valid_count": ref_count,
        }
    max_t = float(np.linalg.norm(np.array(depth.shape[::-1], dtype=float))) * 1.2
    for offset in strip_offsets_px:
        line_p0 = p0 + perp * float(offset)
        corners = np.array(
            [
                line_p0 - direction * max_t - perp * strip_half_width_px,
                line_p0 + direction * max_t - perp * strip_half_width_px,
                line_p0 + direction * max_t + perp * strip_half_width_px,
                line_p0 - direction * max_t + perp * strip_half_width_px,
            ],
            dtype=float,
        )
        strip_polys.append(corners)
        rel = roi_uv - line_p0[None, :]
        in_strip = np.abs(rel @ perp) <= strip_half_width_px
        uv = roi_uv[in_strip]
        result: dict[str, Any] = {"strip_offset_px": float(offset), "candidate_pixel_count": int(uv.shape[0])}
        if uv.size == 0:
            result.update({"valid": False, "failure_reason": "empty_strip", "valid_depth_count": 0, "supported_point_count": 0})
            strip_results.append(result)
            continue
        xi = uv[:, 0].astype(int)
        yi = uv[:, 1].astype(int)
        depths = depth[yi, xi].astype(float)
        valid = depths > 0
        supported = valid & (np.abs(depths - ref) <= depth_tolerance_mm)
        support_uv = uv[supported]
        reject_uv = uv[~supported]
        if support_uv.size:
            retained.append(support_uv)
        if reject_uv.size:
            rejected.append(reject_uv)
        result.update(
            {
                "valid": bool(support_uv.shape[0] > 0),
                "failure_reason": None if support_uv.shape[0] > 0 else "no_supported_points",
                "valid_depth_count": int(np.count_nonzero(valid)),
                "supported_point_count": int(support_uv.shape[0]),
            }
        )
        strip_results.append(result)
    retained_uv = np.vstack(retained) if retained else np.empty((0, 2))
    rejected_uv = np.vstack(rejected) if rejected else np.empty((0, 2))
    return {
        "valid": retained_uv.shape[0] > 0,
        "failure_reason": None if retained_uv.shape[0] > 0 else "no_supported_points",
        "expanded_box": expanded,
        "retained_uv": retained_uv,
        "rejected_uv": rejected_uv,
        "strip_polys": strip_polys,
        "strip_results": strip_results,
        "center_depth_mm": ref,
        "center_region_valid_count": ref_count,
    }


def _row_for(rows: list[dict[str, str]], case_id: str, path_label: str) -> dict[str, str]:
    for row in rows:
        if row.get("case_id") == case_id and row.get("path_label") == path_label:
            return row
    raise KeyError((case_id, path_label))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _overlay_selected_path(rgb: np.ndarray, mask: np.ndarray, selected: dict[str, Any]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = [0.0, 0.35, 1.0, 0.22]
    ax.imshow(overlay)
    colors_by_label = {"major": "#ff7f0e", "minor": "#9467bd"}
    for label in ["major", "minor"]:
        item = selected[label]
        p1 = np.asarray(item["endpoint1_xy"], dtype=float)
        p2 = np.asarray(item["endpoint2_xy"], dtype=float)
        p0 = np.asarray(item["p0_xy"], dtype=float)
        c = colors_by_label[label]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=3, label=f"{label} {item.get('selected_anchor_id')}")
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c=c, s=28)
        ax.scatter([p0[0]], [p0[1]], c="black", s=25)
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def _support_figure(
    *,
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    mask: np.ndarray,
    selected_item: dict[str, Any],
    row: dict[str, str],
    support: dict[str, Any],
    out_path: Path,
    path_label: str,
    endpoint_label: str,
) -> None:
    p1 = np.asarray(selected_item["endpoint1_xy"], dtype=float)
    p2 = np.asarray(selected_item["endpoint2_xy"], dtype=float)
    p0 = np.asarray(selected_item["p0_xy"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.imshow(depth_rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = [0.0, 0.35, 1.0, 0.12]
    ax.imshow(overlay)
    ax.add_patch(Polygon(support["expanded_box"], fill=False, edgecolor="#ff7f0e", linewidth=2, linestyle="--"))
    for poly in support["strip_polys"]:
        ax.add_patch(Polygon(poly, fill=True, facecolor="#ffffff", edgecolor="#666666", alpha=0.08, linewidth=0.4))
    rejected = support["rejected_uv"]
    retained = support["retained_uv"]
    if rejected.size:
        sample = rejected[:: max(1, rejected.shape[0] // 1800)]
        ax.scatter(sample[:, 0], sample[:, 1], s=1, c="#d95f02", alpha=0.18, label="rejected/invalid")
    if retained.size:
        sample = retained[:: max(1, retained.shape[0] // 2500)]
        ax.scatter(sample[:, 0], sample[:, 1], s=2, c="#2ca02c", alpha=0.38, label="retained support")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#111111", linewidth=2.5)
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c="#111111", s=30)
    ax.scatter([p0[0]], [p0[1]], c="#ff0000", s=30)
    text = (
        f"{row['case_id']} {path_label}  anchor {row.get('selected_anchor_id')}\n"
        f"manual {_fmt(row.get('manual_mm'))} mm | mask {_fmt(row.get('mask_path_length_mm'))} mm | "
        f"depth point-cloud {_fmt(row.get('strip_depth_path_length_mm'))} mm\n"
        f"depth abs err {_fmt(row.get('strip_depth_abs_error_mm'))} mm | valid {row.get('strip_depth_valid')}\n"
        f"{endpoint_label}"
    )
    ax.text(0.01, 0.99, text, transform=ax.transAxes, va="top", ha="left", fontsize=10, color="white", bbox=dict(facecolor="black", alpha=0.62, pad=5))
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _mask_measurement_figure(
    rgb: np.ndarray,
    mask: np.ndarray,
    selected_item: dict[str, Any],
    row: dict[str, str],
    out_path: Path,
    path_label: str,
) -> None:
    p1 = np.asarray(selected_item["endpoint1_xy"], dtype=float)
    p2 = np.asarray(selected_item["endpoint2_xy"], dtype=float)
    p0 = np.asarray(selected_item["p0_xy"], dtype=float)
    manual = _float(row, "manual_mm")
    mask_len = _float(row, "mask_path_length_mm")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = [0.0, 0.35, 1.0, 0.18]
    ax.imshow(overlay)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1f77b4", linewidth=3.0, label="mask-based selected path")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c="#1f77b4", s=30)
    ax.scatter([p0[0]], [p0[1]], c="black", s=28)
    lines = [
        f"{row['case_id']} {path_label} ({row.get('selected_anchor_id')})",
        "Mask-based path-length measurement",
        f"manual: {_fmt(manual)} mm",
        f"mask: {_fmt(mask_len)} mm, abs err {_fmt(row.get('mask_abs_error_mm'))} mm",
    ]
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.78, pad=5))
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _result_table_figure(case_id: str, major_row: dict[str, str], minor_row: dict[str, str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axis("off")
    ax.set_title(f"{case_id} selected-anchor results", fontsize=14, fontweight="bold", loc="left")
    headers = ["Path", "A ID", "Manual", "Mask est.", "Mask err.", "Depth est.", "Depth err."]
    body = []
    for row in [major_row, minor_row]:
        body.append(
            [
                row.get("path_label", ""),
                row.get("selected_anchor_id", ""),
                f"{_fmt(row.get('manual_mm'))} mm",
                f"{_fmt(row.get('mask_path_length_mm'))} mm",
                f"{_fmt(row.get('mask_abs_error_mm'))} mm",
                f"{_fmt(row.get('strip_depth_path_length_mm'))} mm",
                f"{_fmt(row.get('strip_depth_abs_error_mm'))} mm",
            ]
        )
    table = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    ax.text(
        0.01,
        0.08,
        "CSV column name `strip_depth` corresponds to the depth point-cloud-based estimate\n"
        "using a local depth sampling band around the selected A* path.",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _panel_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _case_panel(case_id: str, paths: dict[str, Path], out_png: Path, out_svg: Path | None = None) -> None:
    items = [
        ("(a) Raw RGB", paths["raw"]),
        ("(b) GSAM2 mask", paths["mask"]),
        ("(c) A* candidates", paths["candidates"]),
        ("(d) Selected A* paths", paths["selected"]),
        ("(e) Fixed depth", paths["depth"]),
        ("(f) Major depth support", paths["major_support"]),
        ("(g) Minor depth support", paths["minor_support"]),
        ("(h) Major mask measurement", paths["major_mask"]),
        ("(i) Minor mask measurement", paths["minor_mask"]),
        ("(j) Result table", paths["result_table"]),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.8))
    for ax, (title, path) in zip(axes.ravel(), items):
        ax.imshow(_panel_image(path))
        ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
        ax.axis("off")
    fig.suptitle(case_id, fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


def _grid(image_paths: list[Path], out_path: Path, title: str, *, cols: int = 5) -> None:
    existing = [p for p in image_paths if p.exists()]
    if not existing:
        return
    rows = int(math.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, existing):
        ax.imshow(_panel_image(path))
        ax.set_title(path.parent.name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    capture_session = _resolve(args.capture_session)
    mask_run = _resolve(args.mask_run)
    manual_root = _resolve(args.manual_anchor_selection_root)
    eval_session = _resolve(args.eval_session)
    paper_folder = _resolve(args.paper_folder)
    out_root = paper_folder / "08_step_by_step_visualizations"
    per_case_root = out_root / "per_case"
    example_root = out_root / "example_panels"
    grids_root = out_root / "grids"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(eval_session / "selected_anchor_mask_strip_depth_vs_manual.csv")
    run_config_path = eval_session / "run_config.json"
    run_config = _read_json(run_config_path) if run_config_path.exists() else {}
    params = run_config.get("parameters") if isinstance(run_config.get("parameters"), dict) else {}
    endpoint_mode = run_config.get("depth_endpoint_mode") or params.get("depth_endpoint_mode") or "current"
    lower_q = run_config.get("depth_lower_quantile") or params.get("lower_quantile")
    upper_q = run_config.get("depth_upper_quantile") or params.get("upper_quantile")
    endpoint_label = (
        f"Depth point-cloud estimate, percentile endpoints ({lower_q}/{upper_q})"
        if endpoint_mode == "percentile"
        else "Depth point-cloud estimate"
    )
    case_ids = sorted({row["case_id"] for row in rows}, key=lambda case_id: (int(case_id.split("_")[1]), case_id))
    panel_paths: dict[str, dict[str, Path]] = {}

    for case_id in case_ids:
        case_dir = capture_session / "cases" / case_id
        out_dir = per_case_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb = _load_rgb(case_dir / "raw_rgb.png")
        depth = cv2.imread(str(case_dir / "raw_depth_aligned_z16.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(case_dir / "raw_depth_aligned_z16.png")
        depth_rgb = _depth_rgb(depth, vmin=args.depth_vmin_mm, vmax=args.depth_vmax_mm, cmap_name=args.depth_colormap)
        mask = _load_mask(mask_run / "cases" / case_id / "selected_mask.npy")
        selected = _read_json(manual_root / "cases" / case_id / "selected_anchors.json")

        raw_out = out_dir / "1_raw_rgb.png"
        depth_out = out_dir / "5_fixed_depth_viridis_268_375_black_invalid.png"
        _save_img(raw_out, rgb)
        _save_img(depth_out, depth_rgb)
        _copy(mask_run / "cases" / case_id / "selected_mask_overlay.png", out_dir / "2_gsam2_mask_overlay.png")
        _copy(manual_root / "cases" / case_id / "anchor_candidates_overlay.png", out_dir / "3_anchor_candidates_overlay.png")
        selected_overlay = _overlay_selected_path(rgb, mask, selected)
        _save_img(out_dir / "4_selected_anchor_paths_overlay.png", selected_overlay)

        result_paths = {}
        for path_label, filename_idx in [("major", "6"), ("minor", "7")]:
            item = selected[path_label]
            p0 = np.asarray(item["p0_xy"], dtype=float)
            direction = _normalize(np.asarray(item["direction_xy"], dtype=float))
            support = _strip_support(
                depth=depth,
                mask=mask,
                p0=p0,
                direction=direction,
                buffer_percent=args.buffer_percent,
                strip_half_width_px=args.strip_half_width_px,
                strip_offsets_px=args.strip_offsets_px,
                center_region_radius_px=args.center_region_radius_px,
                depth_tolerance_mm=args.depth_tolerance_mm,
            )
            row = _row_for(rows, case_id, path_label)
            support_path = out_dir / f"{filename_idx}_{path_label}_depth_point_support.png"
            _support_figure(
                rgb=rgb,
                depth_rgb=depth_rgb,
                mask=mask,
                selected_item=item,
                row=row,
                support=support,
                out_path=support_path,
                path_label=path_label,
                endpoint_label=endpoint_label,
            )
            result_idx = "8" if path_label == "major" else "9"
            result_path = out_dir / f"{result_idx}_{path_label}_mask_based_measurement.png"
            _mask_measurement_figure(rgb, mask, item, row, result_path, path_label)
            result_paths[path_label] = (support_path, result_path)
        result_table_path = out_dir / "11_result_text_table.png"
        _result_table_figure(case_id, _row_for(rows, case_id, "major"), _row_for(rows, case_id, "minor"), result_table_path)
        paths = {
            "raw": raw_out,
            "mask": out_dir / "2_gsam2_mask_overlay.png",
            "candidates": out_dir / "3_anchor_candidates_overlay.png",
            "selected": out_dir / "4_selected_anchor_paths_overlay.png",
            "depth": depth_out,
            "major_support": result_paths["major"][0],
            "minor_support": result_paths["minor"][0],
            "major_mask": result_paths["major"][1],
            "minor_mask": result_paths["minor"][1],
            "result_table": result_table_path,
        }
        _case_panel(case_id, paths, out_dir / "10_case_step_by_step_panel.png", out_dir / "10_case_step_by_step_panel.svg")
        panel_paths[case_id] = paths | {"panel": out_dir / "10_case_step_by_step_panel.png"}

    examples = {
        "case_002_2": "example_timber_case_002_2_step_by_step.png",
        "case_007_2": "example_brick_case_007_2_step_by_step.png",
        "case_015_5": "example_concrete_block_case_015_5_step_by_step.png",
    }
    for case_id, name in examples.items():
        if case_id in panel_paths:
            _copy(per_case_root / case_id / "10_case_step_by_step_panel.png", example_root / name)
    _grid([example_root / name for name in examples.values()], example_root / "example_all_materials_step_by_step_grid.png", "Representative selected-anchor workflow examples", cols=1)

    _grid([panel_paths[c]["raw"] for c in case_ids], grids_root / "all_raw_rgb_grid.png", "Raw RGB")
    _grid([panel_paths[c]["mask"] for c in case_ids], grids_root / "all_mask_overlay_grid.png", "GSAM2 mask overlays")
    _grid([panel_paths[c]["candidates"] for c in case_ids], grids_root / "all_anchor_candidates_grid.png", "A* anchor candidates")
    _grid([panel_paths[c]["selected"] for c in case_ids], grids_root / "all_selected_anchor_paths_grid.png", "Selected A* paths")
    _grid([panel_paths[c]["depth"] for c in case_ids], grids_root / "all_fixed_depth_grid.png", "Fixed viridis depth 268-375 mm")
    _grid([panel_paths[c]["major_support"] for c in case_ids], grids_root / "all_major_depth_point_support_grid.png", "Major/path-1 depth point-cloud support")
    _grid([panel_paths[c]["minor_support"] for c in case_ids], grids_root / "all_minor_depth_point_support_grid.png", "Minor/path-2 depth point-cloud support")
    _grid([panel_paths[c]["major_mask"] for c in case_ids], grids_root / "all_major_mask_based_measurement_grid.png", "Major/path-1 mask-based measurement")
    _grid([panel_paths[c]["minor_mask"] for c in case_ids], grids_root / "all_minor_mask_based_measurement_grid.png", "Minor/path-2 mask-based measurement")
    _grid([panel_paths[c]["panel"] for c in case_ids], grids_root / "all_case_step_panels_grid.png", "Per-case step-by-step panels")

    readme = [
        "# Step-by-Step Selected-Anchor Path-Length Visualizations",
        "",
        "These figures explain the final selected-anchor path-length evaluation workflow.",
        "",
        "## Steps",
        "- `1_raw_rgb.png`: saved RGB frame.",
        "- `2_gsam2_mask_overlay.png`: GSAM2 selected object mask.",
        "- `3_anchor_candidates_overlay.png`: real A* anchor/contact candidates.",
        "- `4_selected_anchor_paths_overlay.png`: manually selected major/path-1 and minor/path-2 A* paths.",
        "- `5_fixed_depth_viridis_268_375_black_invalid.png`: aligned raw depth visualized with fixed viridis scale.",
        "- `6_major_depth_point_support.png`: depth point-cloud support for selected major/path-1.",
        "- `7_minor_depth_point_support.png`: depth point-cloud support for selected minor/path-2.",
        "- `8_major_mask_based_measurement.png`: mask-based measurement for selected major/path-1.",
        "- `9_minor_mask_based_measurement.png`: mask-based measurement for selected minor/path-2.",
        "- `10_case_step_by_step_panel.png`: compact per-case workflow panel.",
        "",
        "## Depth Settings",
        f"- Colormap: `{args.depth_colormap}`",
        f"- Fixed limits: `{args.depth_vmin_mm:g}` to `{args.depth_vmax_mm:g}` mm",
        "- Clipping enabled.",
        "- Invalid zero depth is black.",
        "- No global depth limits are recomputed.",
        f"- Depth endpoint mode: `{endpoint_mode}`.",
        f"- Projected endpoint quantiles: `{lower_q}` / `{upper_q}`.",
        "",
        "## Method Scope",
        "- Selected A* anchors from `selected_anchors.json` are used.",
        "- Whole-ROI depth is not used.",
        "- MinAreaRect fallback axes are not used.",
        "- Depth point-cloud support points are reconstructed from saved aligned depth using the same selected path metadata and evaluator parameters.",
        "- CSV column name `strip_depth` corresponds to the depth point-cloud-based estimate using a local depth sampling band around the selected A* path.",
        "",
        "No GSAM2, manual annotation, final metric evaluation, RealSense capture, robot motion, or RTDE command was run to create these figures.",
    ]
    (out_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"wrote {out_root}")
    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create step-by-step selected-anchor path-length visualization figures.")
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--mask-run", required=True)
    parser.add_argument("--manual-anchor-selection-root", required=True)
    parser.add_argument("--eval-session", required=True)
    parser.add_argument("--paper-folder", required=True)
    parser.add_argument("--depth-vmin-mm", type=float, default=268.0)
    parser.add_argument("--depth-vmax-mm", type=float, default=375.0)
    parser.add_argument("--depth-colormap", default="viridis")
    parser.add_argument("--buffer-percent", type=float, default=4.0)
    parser.add_argument("--strip-half-width-px", type=float, default=6.0)
    parser.add_argument("--strip-offsets-px", default="-18,-12,-6,0,6,12,18")
    parser.add_argument("--center-region-radius-px", type=float, default=15.0)
    parser.add_argument("--depth-tolerance-mm", type=float, default=8.0)
    args = parser.parse_args()
    args.strip_offsets_px = [float(v.strip()) for v in str(args.strip_offsets_px).split(",") if v.strip()]
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
