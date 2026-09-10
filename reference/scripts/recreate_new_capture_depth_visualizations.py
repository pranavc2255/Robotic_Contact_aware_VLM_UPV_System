#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO_ROOT / "outputs/path_length_ros2_stream_capture_test/session_20260702_153357"


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip() or None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_manifest(session_dir: Path) -> list[dict[str, Any]]:
    manifest = session_dir / "session_manifest.csv"
    if not manifest.exists():
        case_rows = []
        for idx, case_dir in enumerate(sorted((session_dir / "cases").glob("case_*")), start=1):
            case_rows.append(
                {
                    "case_index": idx,
                    "case_id": case_dir.name,
                    "rgb_path": str(case_dir / "raw_rgb.png"),
                    "aligned_depth_z16_path": str(case_dir / "raw_depth_aligned_z16.png"),
                    "saved_status": "saved",
                }
            )
        return case_rows
    rows: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("saved_status", "saved") == "saved":
                rows.append(row)
    return rows


def _case_index(case_id: str, fallback: int) -> int:
    try:
        return int(case_id.split("_")[1])
    except Exception:
        return fallback


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def _depth_rgba(depth_mm: np.ndarray, *, vmin: float, vmax: float, cmap_name: str, invalid: str) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm(depth_mm)) * 255).astype(np.uint8)
    if invalid == "black":
        rgba[~valid] = np.array([0, 0, 0, 255], dtype=np.uint8)
    elif invalid == "white":
        rgba[~valid] = np.array([255, 255, 255, 255], dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported invalid color: {invalid}")
    return rgba


def _save_rgba_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_image_svg(path: Path, image: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(4.0, image.shape[1] / 180.0)
    fig_h = max(3.0, image.shape[0] / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_preview(rgb: np.ndarray, depth_rgba: np.ndarray, png_path: Path, svg_path: Path, title: str) -> None:
    depth_rgb = depth_rgba[:, :, :3]
    if depth_rgb.shape[:2] != rgb.shape[:2]:
        depth_rgb = cv2.resize(depth_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    preview = np.hstack([rgb, depth_rgb])
    _save_rgba_png(png_path, np.dstack([preview, np.full(preview.shape[:2], 255, dtype=np.uint8)]))
    _save_image_svg(svg_path, preview, title)


def _stats(depth_mm: np.ndarray, vmin: float, vmax: float) -> dict[str, Any]:
    valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
    total = int(depth_mm.size)
    in_range = int(np.count_nonzero((depth_mm >= vmin) & (depth_mm <= vmax)))
    return {
        "valid_depth_pixel_count": int(valid.size),
        "valid_depth_fraction": float(valid.size / total) if total else 0.0,
        "in_range_268_375_pixel_count": in_range,
        "in_range_268_375_fraction": float(in_range / total) if total else 0.0,
        "min_depth_mm": float(valid.min()) if valid.size else None,
        "median_depth_mm": float(np.median(valid)) if valid.size else None,
        "max_depth_mm": float(valid.max()) if valid.size else None,
        "percentile_1_mm": float(np.percentile(valid, 1)) if valid.size else None,
        "percentile_99_mm": float(np.percentile(valid, 99)) if valid.size else None,
    }


def _save_colorbar(out_dir: Path, *, vmin: float, vmax: float, cmap_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.4, 4.8))
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps[cmap_name])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_label("Depth (mm)")
    fig.savefig(out_dir / "depth_viridis_268_375_colorbar.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "depth_viridis_268_375_colorbar.svg", bbox_inches="tight")
    plt.close(fig)


def _make_grid(images: list[np.ndarray], titles: list[str], out_png: Path, out_svg: Path, *, cols: int = 5) -> None:
    if not images:
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    session_dir = _resolve(args.capture_session)
    out_dir = session_dir / "depth_visualization_recreated"
    cases_out = out_dir / "cases"
    grids_out = out_dir / "grids"
    rows = _read_manifest(session_dir)

    summary_rows: list[dict[str, Any]] = []
    depth_grids: dict[str, list[np.ndarray]] = {"black": [], "white": []}
    preview_grids: dict[str, list[np.ndarray]] = {"black": [], "white": []}
    titles: list[str] = []

    for fallback_idx, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or Path(row["rgb_path"]).parent.name)
        rgb_path = _resolve(row.get("rgb_path") or (session_dir / "cases" / case_id / "raw_rgb.png"))
        depth_path = _resolve(row.get("aligned_depth_z16_path") or (session_dir / "cases" / case_id / "raw_depth_aligned_z16.png"))
        rgb = _read_rgb(rgb_path)
        depth_mm = _read_depth(depth_path)
        case_out = cases_out / case_id
        case_stats = _stats(depth_mm, args.vmin_mm, args.vmax_mm)

        for invalid in ("black", "white"):
            rgba = _depth_rgba(depth_mm, vmin=args.vmin_mm, vmax=args.vmax_mm, cmap_name=args.colormap, invalid=invalid)
            png = case_out / f"depth_viridis_268_375_{invalid}_invalid.png"
            svg = case_out / f"depth_viridis_268_375_{invalid}_invalid.svg"
            _save_rgba_png(png, rgba)
            _save_image_svg(svg, rgba[:, :, :3], f"{case_id}: viridis 268-375 mm, invalid {invalid}")
            preview_png = case_out / f"rgb_depth_viridis_268_375_{invalid}_invalid_preview.png"
            preview_svg = case_out / f"rgb_depth_viridis_268_375_{invalid}_invalid_preview.svg"
            _save_preview(rgb, rgba, preview_png, preview_svg, f"{case_id}: RGB + depth, invalid {invalid}")
            depth_grids[invalid].append(rgba[:, :, :3])
            preview_rgb = np.hstack([rgb, cv2.resize(rgba[:, :, :3], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)])
            preview_grids[invalid].append(preview_rgb)

        idx = _case_index(case_id, fallback_idx)
        titles.append(f"{idx:02d} {case_id}")
        summary_rows.append(
            {
                "case_index": idx,
                "case_id": case_id,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                **case_stats,
            }
        )

    for invalid in ("black", "white"):
        _make_grid(
            depth_grids[invalid],
            titles,
            grids_out / f"all_cases_depth_viridis_268_375_{invalid}_invalid_grid.png",
            grids_out / f"all_cases_depth_viridis_268_375_{invalid}_invalid_grid.svg",
        )
        _make_grid(
            preview_grids[invalid],
            titles,
            grids_out / f"all_cases_rgb_depth_viridis_268_375_{invalid}_invalid_grid.png",
            grids_out / f"all_cases_rgb_depth_viridis_268_375_{invalid}_invalid_grid.svg",
        )

    _save_colorbar(grids_out, vmin=args.vmin_mm, vmax=args.vmax_mm, cmap_name=args.colormap)
    _write_json(
        out_dir / "depth_visualization_mapping.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_branch": _git_value(["git", "branch", "--show-current"]),
            "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
            "capture_session": str(session_dir),
            "colormap": args.colormap,
            "vmin_mm": float(args.vmin_mm),
            "vmax_mm": float(args.vmax_mm),
            "clip": True,
            "invalid_variants": ["black", "white"],
            "note": "E2-style fixed viridis depth visualization; raw depth files are not modified.",
        },
    )
    with (out_dir / "depth_visualization_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["case_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate E2-style fixed viridis depth visualizations for a ROS2 capture session.")
    parser.add_argument("--capture-session", default=str(DEFAULT_SESSION))
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--vmin-mm", type=float, default=268.0)
    parser.add_argument("--vmax-mm", type=float, default=375.0)
    return parser.parse_args()


def main() -> int:
    out_dir = run(parse_args())
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
