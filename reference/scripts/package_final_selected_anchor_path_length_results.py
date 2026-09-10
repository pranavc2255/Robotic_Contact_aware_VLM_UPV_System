#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip() or None
    except Exception:
        return None


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _material_for_case(case_id: str, fallback: str | None = None) -> str:
    text = str(fallback or "").strip().lower().replace("_", " ")
    if text in {"timber", "wood", "t"}:
        return "timber"
    if text in {"brick", "b"}:
        return "brick"
    if text in {"concrete", "concrete block", "block", "c"}:
        return "concrete block"
    try:
        idx = int(case_id.split("_")[1])
    except Exception:
        idx = 0
    if 1 <= idx <= 5:
        return "timber"
    if 6 <= idx <= 10:
        return "brick"
    if 11 <= idx <= 15:
        return "concrete block"
    return fallback or ""


def _depth_rgba(depth: np.ndarray, *, vmin: float, vmax: float, cmap_name: str, invalid: str) -> np.ndarray:
    depth_float = depth.astype(float)
    valid = np.isfinite(depth_float) & (depth_float > 0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm(depth_float)) * 255).astype(np.uint8)
    if invalid == "black":
        rgba[~valid] = np.array([0, 0, 0, 255], dtype=np.uint8)
    else:
        rgba[~valid] = np.array([255, 255, 255, 255], dtype=np.uint8)
    return rgba


def _save_depth_visuals(case_dir: Path, out_dir: Path, *, vmin: float, vmax: float, cmap_name: str) -> dict[str, Any]:
    depth_path = case_dir / "raw_depth_aligned_z16.png"
    rgb_path = case_dir / "raw_rgb.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return {"error": f"unreadable_depth:{depth_path}"}
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB) if rgb_bgr is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {
        "depth_path": str(depth_path),
        "vmin_mm": vmin,
        "vmax_mm": vmax,
        "colormap": cmap_name,
        "clip": True,
    }
    for invalid in ["black", "white"]:
        rgba = _depth_rgba(depth, vmin=vmin, vmax=vmax, cmap_name=cmap_name, invalid=invalid)
        png = out_dir / f"depth_aligned_fixed_scale_{invalid}_invalid.png"
        plt.imsave(png, rgba)
        saved[f"{invalid}_invalid_png"] = str(png)
        if rgb is not None and invalid == "black":
            depth_rgb = rgba[:, :, :3]
            if rgb.shape[:2] != depth_rgb.shape[:2]:
                rgb_show = cv2.resize(rgb, (depth_rgb.shape[1], depth_rgb.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                rgb_show = rgb
            preview = np.concatenate([rgb_show, depth_rgb], axis=1)
            preview_path = out_dir / "rgb_depth_side_by_side.png"
            plt.imsave(preview_path, preview)
            saved["rgb_depth_side_by_side_png"] = str(preview_path)
    valid = depth[depth > 0].astype(float)
    saved["valid_depth_count"] = int(valid.size)
    saved["valid_depth_fraction"] = float(valid.size / depth.size) if depth.size else None
    if valid.size:
        saved["min_depth_mm"] = float(np.min(valid))
        saved["median_depth_mm"] = float(np.median(valid))
        saved["max_depth_mm"] = float(np.max(valid))
        saved["in_range_fraction"] = float(np.count_nonzero((valid >= vmin) & (valid <= vmax)) / valid.size)
    _write_json(out_dir / "depth_visualization_metadata.json", saved)
    return saved


def _copy_case_inputs(capture_session: Path, case_id: str, out_dir: Path) -> dict[str, Any]:
    case_dir = capture_session / "cases" / case_id
    copied = {}
    for name in [
        "raw_rgb.png",
        "raw_depth_aligned_z16.png",
        "manual_measurements.json",
        "capture_metadata.json",
        "camera_info_aligned_depth.json",
    ]:
        copied[name] = _copy_if_exists(case_dir / name, out_dir / name)
    return copied


def _selected_anchor_ids(manual_root: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case_id in case_ids:
        path = manual_root / "cases" / case_id / "selected_anchors.json"
        payload = _read_json(path)
        manual = payload.get("manual_measurements", {})
        rows.append(
            {
                "case_id": case_id,
                "material": payload.get("material") or _material_for_case(case_id),
                "source": manual.get("source") or manual.get("source_case") or "",
                "manual_major_mm": payload.get("major", {}).get("manual_mm"),
                "major_selected_anchor_id": payload.get("major", {}).get("selected_anchor_id"),
                "manual_minor_mm": payload.get("minor", {}).get("manual_mm"),
                "minor_selected_anchor_id": payload.get("minor", {}).get("selected_anchor_id"),
            }
        )
    return rows


def _summary_markdown(summary_rows: list[dict[str, str]]) -> str:
    headers = [
        "group",
        "n_total",
        "mask_MAE_mm",
        "mask_RMSE_mm",
        "mask_MAPE_percent",
        "strip_depth_n_valid",
        "strip_depth_MAE_mm",
        "strip_depth_RMSE_mm",
        "strip_depth_MAPE_percent",
        "improvement_percent",
        "best_method_by_MAE",
    ]
    lines = [
        "# Final Paper Summary Table",
        "",
        "CSV column name `strip_depth` corresponds to the depth point-cloud-based estimate using a local depth sampling band around the selected A* path.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _make_figures(fig_dir: Path, selected_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['case_id']}\n{r['path_label']}" for r in selected_rows]
    x = np.arange(len(selected_rows))
    mask_err = np.array([_float(r, "mask_abs_error_mm") or np.nan for r in selected_rows], dtype=float)
    strip_err = np.array([_float(r, "strip_depth_abs_error_mm") or np.nan for r in selected_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.45), 5))
    ax.bar(x - 0.18, mask_err, width=0.36, label="mask", color="#1f77b4")
    ax.bar(x + 0.18, strip_err, width=0.36, label="strip-depth", color="#9467bd")
    ax.set_ylabel("absolute error (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "final_error_barplot_mask_vs_strip.png", dpi=220)
    plt.close(fig)

    def scatter(path: Path, key: str, title: str, color: str) -> None:
        xs, ys = [], []
        for row in selected_rows:
            manual = _float(row, "manual_mm")
            pred = _float(row, key)
            if manual is not None and pred is not None:
                xs.append(manual)
                ys.append(pred)
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        ax.scatter(xs, ys, c=color)
        if xs and ys:
            lo = min(xs + ys)
            hi = max(xs + ys)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("manual path length (mm)")
        ax.set_ylabel("estimated path length (mm)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    scatter(fig_dir / "final_manual_vs_mask_scatter.png", "mask_path_length_mm", "Manual vs mask-based", "#1f77b4")
    scatter(fig_dir / "final_manual_vs_strip_depth_scatter.png", "strip_depth_path_length_mm", "Manual vs strip-depth", "#9467bd")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    data = [mask_err[~np.isnan(mask_err)], strip_err[~np.isnan(strip_err)]]
    ax.boxplot(data, labels=["mask", "strip-depth"])
    ax.set_ylabel("absolute error (mm)")
    ax.set_title("Selected-anchor absolute error")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_error_boxplot_by_method.png", dpi=220)
    plt.close(fig)

    material_rows = [r for r in summary_rows if r.get("group") in {"timber_all", "brick_all", "concrete_block_all"}]
    labels = [r["group"].replace("_all", "").replace("_", " ") for r in material_rows]
    mx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(mx - 0.18, [_float(r, "mask_MAE_mm") or np.nan for r in material_rows], width=0.36, label="mask", color="#1f77b4")
    ax.bar(mx + 0.18, [_float(r, "strip_depth_MAE_mm") or np.nan for r in material_rows], width=0.36, label="strip-depth", color="#9467bd")
    ax.set_xticks(mx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "final_material_breakdown_barplot.png", dpi=220)
    plt.close(fig)


def _make_image_grid(image_paths: list[Path], out_path: Path, *, title: str) -> None:
    existing = [p for p in image_paths if p.exists()]
    if not existing:
        return
    cols = 5
    rows = int(math.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, existing):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.parent.name, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    capture_session = _resolve(args.capture_session)
    mask_run = _resolve(args.mask_run)
    manual_root = _resolve(args.manual_anchor_selection_root)
    eval_session = _resolve(args.eval_session)
    output_root = _resolve(args.output_root)
    out_dir = output_root.parent / f"{output_root.name}_{_now_id()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    selected_csv = eval_session / "selected_anchor_mask_strip_depth_vs_manual.csv"
    summary_csv = eval_session / "selected_anchor_mask_strip_depth_summary.csv"
    all_anchor_csv = eval_session / "all_anchor_path_measurements.csv"
    selected_rows = _read_csv(selected_csv)
    summary_rows = _read_csv(summary_csv)
    all_anchor_rows = _read_csv(all_anchor_csv)
    eval_config = _read_json(eval_session / "run_config.json") if (eval_session / "run_config.json").exists() else {}
    eval_params = eval_config.get("parameters") if isinstance(eval_config.get("parameters"), dict) else {}
    endpoint_mode = eval_config.get("depth_endpoint_mode") or eval_params.get("depth_endpoint_mode") or "current"
    endpoint_low = eval_config.get("depth_lower_quantile") or eval_params.get("lower_quantile")
    endpoint_high = eval_config.get("depth_upper_quantile") or eval_params.get("upper_quantile")
    case_ids = sorted({row["case_id"] for row in selected_rows}, key=lambda x: (int(x.split("_")[1]), x))

    tables = out_dir / "01_tables"
    _copy_if_exists(selected_csv, tables / "final_selected_anchor_30path_comparison.csv")
    _copy_if_exists(summary_csv, tables / "final_selected_anchor_summary.csv")
    _copy_if_exists(all_anchor_csv, tables / "final_all_anchor_measurements.csv")
    selected_id_rows = _selected_anchor_ids(manual_root, case_ids)
    _write_csv(tables / "final_selected_anchor_ids.csv", selected_id_rows)
    paper_summary = _summary_markdown(summary_rows)
    (tables / "final_paper_summary_table.md").write_text(paper_summary, encoding="utf-8")
    invalid = [r for r in selected_rows if str(r.get("strip_depth_valid")).lower() != "true"]
    if invalid:
        lines = ["# Invalid Depth Point-Cloud Rows", "", "| case_id | path_label | selected_anchor_id | reason |", "| --- | --- | --- | --- |"]
        for row in invalid:
            lines.append(f"| {row['case_id']} | {row['path_label']} | {row.get('selected_anchor_id')} | {row.get('strip_depth_failure_reason')} |")
    else:
        lines = ["# Invalid Depth Point-Cloud Rows", "", "All 30 selected depth point-cloud measurements were valid."]
    (tables / "final_invalid_strip_depth_rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    completeness_rows = []
    replacement_notes = {}
    for case_id in case_ids:
        case_dir = capture_session / "cases" / case_id
        input_dir = out_dir / "02_per_case_inputs" / case_id
        copied = _copy_case_inputs(capture_session, case_id, input_dir)
        manual = _read_json(case_dir / "manual_measurements.json")
        capture_meta = _read_json(case_dir / "capture_metadata.json") if (case_dir / "capture_metadata.json").exists() else {}
        selected_payload = _read_json(manual_root / "cases" / case_id / "selected_anchors.json")
        case_readme = {
            "case_id": case_id,
            "material": manual.get("material") or _material_for_case(case_id),
            "manual_major_mm": selected_payload.get("major", {}).get("manual_mm"),
            "manual_minor_mm": selected_payload.get("minor", {}).get("manual_mm"),
            "major_selected_anchor_id": selected_payload.get("major", {}).get("selected_anchor_id"),
            "minor_selected_anchor_id": selected_payload.get("minor", {}).get("selected_anchor_id"),
            "replacement_source": capture_meta.get("replacement_source") or manual.get("source"),
        }
        _write_json(input_dir / "case_summary.json", case_readme)
        if case_readme["replacement_source"]:
            replacement_notes[case_id] = case_readme
        completeness_rows.append(
            {
                "case_id": case_id,
                **{k: bool(v) for k, v in copied.items()},
                "selected_mask_png": (mask_run / "cases" / case_id / "selected_mask.png").exists(),
                "selected_mask_npy": (mask_run / "cases" / case_id / "selected_mask.npy").exists(),
                "selected_anchors_json": (manual_root / "cases" / case_id / "selected_anchors.json").exists(),
            }
        )

        sel_case = manual_root / "cases" / case_id
        anchor_out = out_dir / "03_selected_anchor_overlays" / case_id
        for name in [
            "selected_anchors_overlay.png",
            "anchor_candidates_sheet.png",
            "anchor_candidates_overlay.png",
            "selected_anchors.json",
            "anchor_candidates.csv",
        ]:
            _copy_if_exists(sel_case / name, anchor_out / name)

        mask_case = mask_run / "cases" / case_id
        mask_out = out_dir / "05_masks" / case_id
        for name in [
            "selected_mask.png",
            "selected_mask_overlay.png",
            "selected_box_overlay.png",
            "all_masks_overlay.png",
            "gsam2_result.json",
        ]:
            _copy_if_exists(mask_case / name, mask_out / name)

        _save_depth_visuals(
            case_dir,
            out_dir / "04_depth_visualizations" / case_id,
            vmin=args.depth_vmin_mm,
            vmax=args.depth_vmax_mm,
            cmap_name=args.depth_colormap,
        )

    _make_image_grid(
        [out_dir / "03_selected_anchor_overlays" / case_id / "selected_anchors_overlay.png" for case_id in case_ids],
        out_dir / "03_selected_anchor_overlays" / "all_selected_anchor_overlays_grid.png",
        title="Selected A* anchor overlays",
    )
    _make_image_grid(
        [out_dir / "03_selected_anchor_overlays" / case_id / "anchor_candidates_sheet.png" for case_id in case_ids],
        out_dir / "03_selected_anchor_overlays" / "all_anchor_candidate_sheets_grid.png",
        title="A* anchor candidate sheets",
    )

    _make_figures(out_dir / "06_figures", selected_rows, summary_rows)

    all_group = next((r for r in summary_rows if r.get("group") == "all"), {})
    final_metrics = {
        "n_cases": len(case_ids),
        "n_selected_paths": len(selected_rows),
        "n_all_anchors": len(all_anchor_rows),
        "strip_depth_valid_selected_rows": sum(1 for r in selected_rows if str(r.get("strip_depth_valid")).lower() == "true"),
        "all_group": all_group,
    }
    machine = out_dir / "07_machine_readable"
    _write_json(
        machine / "run_metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_branch": _git_value(["git", "branch", "--show-current"]),
            "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
            "capture_session": str(capture_session),
            "mask_run": str(mask_run),
            "manual_anchor_selection_root": str(manual_root),
            "eval_session": str(eval_session),
            "whole_roi_depth_used_in_main_result": False,
            "minAreaRect_fallback_used": False,
            "selected_A_star_anchors_used": True,
            "final_path_results_version": args.final_path_results_version,
            "depth_endpoint_mode": endpoint_mode,
            "depth_lower_quantile": endpoint_low,
            "depth_upper_quantile": endpoint_high,
            "robot_motion_used": False,
            "rtde_used": False,
            "realsense_started_or_configured": False,
        },
    )
    _write_csv(machine / "input_completeness_check.csv", completeness_rows)
    _write_json(machine / "selected_anchor_case_list.json", selected_id_rows)
    _write_json(machine / "replacement_case_notes.json", replacement_notes)
    _write_json(machine / "evaluator_args.json", _read_json(eval_session / "run_config.json") if (eval_session / "run_config.json").exists() else {})
    _write_json(machine / "final_metrics.json", final_metrics)

    readme = [
        f"# Final Selected-Anchor Path-Length Results {args.final_path_results_version}",
        "",
        f"Dataset source: `{capture_session}`",
        f"Mask source: `{mask_run}`",
        f"Manual anchor selection source: `{manual_root}`",
        f"Final evaluation session: `{eval_session}`",
        "",
        "## Method Scope",
        "- Metrics use selected A* anchors from `selected_anchors.json`.",
        "- MinAreaRect fallback is not used.",
        "- Whole-ROI depth is not used in the main result.",
        "- The main methods are mask-based path-length estimation and depth point-cloud-based path-length estimation.",
        "- CSV column name `strip_depth` corresponds to the depth point-cloud-based estimate using a local depth sampling band around the selected A* path.",
        f"- Depth endpoint mode: `{endpoint_mode}`.",
        f"- Depth projected endpoint quantiles: `{endpoint_low}` / `{endpoint_high}`.",
        "",
        "## Replacements",
        "- `case_002_2` replaced by old E2 fixed15 case 10.",
        "- `case_007_2` replaced by old E2 fixed15 case 1.",
        "- `case_010_5` replaced by old E2 fixed15 case 5.",
        "- `case_012_2` replaced by old E2 fixed15 case 14.",
        "- `case_015_5` replaced by old E2 fixed15 case 12.",
        "",
        "## Headline Result",
        f"- Cases: {final_metrics['n_cases']}",
        f"- Selected paths: {final_metrics['n_selected_paths']}",
        f"- All A* anchors measured: {final_metrics['n_all_anchors']}",
        f"- Depth point-cloud valid selected rows: {final_metrics['strip_depth_valid_selected_rows']}/{final_metrics['n_selected_paths']}",
        f"- Mask MAE/RMSE/MAPE: {_fmt(all_group.get('mask_MAE_mm'))} mm / {_fmt(all_group.get('mask_RMSE_mm'))} mm / {_fmt(all_group.get('mask_MAPE_percent'))}%",
        f"- Depth point-cloud MAE/RMSE/MAPE: {_fmt(all_group.get('strip_depth_MAE_mm'))} mm / {_fmt(all_group.get('strip_depth_RMSE_mm'))} mm / {_fmt(all_group.get('strip_depth_MAPE_percent'))}%",
        f"- Improvement: {_fmt(all_group.get('improvement_mm'))} mm, {_fmt(all_group.get('improvement_percent'))}%",
        "",
        "## Summary Table",
        "",
        paper_summary,
        "",
        "## Key CSVs",
        "- `01_tables/final_selected_anchor_30path_comparison.csv`",
        "- `01_tables/final_selected_anchor_summary.csv`",
        "- `01_tables/final_all_anchor_measurements.csv`",
        "- `01_tables/final_selected_anchor_ids.csv`",
        "",
        "## Folder Layout",
        "- `02_per_case_inputs/`: raw RGB/depth, manual measurements, capture metadata, camera info.",
        "- `03_selected_anchor_overlays/`: selected-anchor JSON/CSV and overlays.",
        "- `04_depth_visualizations/`: viridis fixed 268-375 mm depth visualizations.",
        "- `05_masks/`: selected GSAM2 masks and overlays.",
        "- `06_figures/`: diagnostic/paper-ready summary figures.",
        "- `07_machine_readable/`: metadata and final metrics JSON.",
    ]
    (out_dir / "00_README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final selected-anchor mask-vs-strip-depth path-length results.")
    parser.add_argument("--capture-session", required=True)
    parser.add_argument("--mask-run", required=True)
    parser.add_argument("--manual-anchor-selection-root", required=True)
    parser.add_argument("--eval-session", required=True)
    parser.add_argument("--output-root", default="outputs/paper_final_path_length_selected_anchor_15case")
    parser.add_argument("--depth-vmin-mm", type=float, default=268.0)
    parser.add_argument("--depth-vmax-mm", type=float, default=375.0)
    parser.add_argument("--depth-colormap", default="viridis")
    parser.add_argument("--final-path-results-version", default="v1")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
