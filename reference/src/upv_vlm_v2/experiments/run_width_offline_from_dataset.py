"""Stage B offline E2 width/path-length evaluation from a captured dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

from upv_vlm_v2.config.loader import load_config, snapshot_yaml
from upv_vlm_v2.data.models import json_ready
from upv_vlm_v2.pipeline.main_pipeline import run_full_main_upv_vlm_v2_pipeline


MASTER_COLUMNS = [
    "success",
    "failed_stage",
    "failure_reason",
    "case_index",
    "case_id",
    "requested_material",
    "axis_mode",
    "manual_width_mm",
    "mask_path_length_mm",
    "depth_path_length_mm",
    "depth_valid",
    "depth_validity_policy",
    "path_length_source",
    "mask_abs_error_mm",
    "depth_abs_error_mm",
    "mask_percent_error",
    "depth_percent_error",
    "selected_candidate_id",
    "selected_anchor_id",
    "qwen_selected_anchor",
    "qwen_repaired",
    "final_selected_anchor_source",
    "anchor_score",
    "mask_compute_ms",
    "depth_compute_ms",
    "target_selection_runtime_ms",
    "anchor_selection_runtime_ms",
    "path_length_runtime_ms",
    "total_pipeline_runtime_ms",
    "rgb_path",
    "depth_path",
    "camera_info_path",
    "selected_mask_overlay_path",
    "crop_verification_panel_path",
    "selected_anchor_overlay_path",
    "wide_contact_grid_path",
    "local_chord_overlay_path",
    "depth_used_points_overlay_path",
    "depth_edge_bins_overlay_path",
    "depth_projection_histogram_path",
    "depth_width_diagnostics_path",
    "depth_pointcloud_npz_path",
    "depth_pointcloud_ply_path",
    "pipeline_result_path",
    "stage_status_path",
]


@dataclass
class OfflineSession:
    config: dict[str, Any]
    config_path: Path
    dataset_path: Path
    session_dir: Path


def _now_id() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_rows_if_exists(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return _read_rows(p)


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _abs_err(value: Any, manual: Any) -> float | None:
    v = _safe_float(value)
    m = _safe_float(manual)
    if v is None or m is None:
        return None
    return abs(v - m)


def _pct_err(value: Any, manual: Any) -> float | None:
    err = _abs_err(value, manual)
    m = _safe_float(manual)
    if err is None or not m:
        return None
    return 100.0 * err / abs(m)


def _rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def _numeric_values(values: Any) -> list[float]:
    vals: list[float] = []
    for value in values:
        v = _safe_float(value)
        if v is None:
            continue
        try:
            numeric = float(v)
        except Exception:
            continue
        if math.isfinite(numeric):
            vals.append(numeric)
    return vals


def _median_or_none(values: Any) -> float | None:
    vals = _numeric_values(values)
    return statistics.median(vals) if vals else None


def _mean_or_none(values: Any) -> float | None:
    vals = _numeric_values(values)
    return statistics.mean(vals) if vals else None


def _rmse_or_none(values: Any) -> float | None:
    vals = _numeric_values(values)
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else None


def _copy(src: str | Path | None, dst_dir: Path, name: str | None = None) -> str | None:
    if not src:
        return None
    path = Path(src)
    if not path.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or path.name)
    if path.resolve() != dst.resolve():
        shutil.copy2(path, dst)
    return str(dst)


def _stage_ms(result: Any, name: str) -> float | None:
    for status in getattr(result, "stage_status", []) or []:
        if getattr(status, "name", None) == name:
            return getattr(status, "timing_ms", None)
    return None


def _failed_stage(result: Any) -> str | None:
    for status in getattr(result, "stage_status", []) or []:
        if not getattr(status, "success", False) and not getattr(status, "skipped", False):
            return getattr(status, "name", None)
    return None


def _health_check(config: dict[str, Any]) -> None:
    qwen = config.get("qwen", {})
    if not bool(qwen.get("require_server", False)):
        return
    url = qwen.get("health_url")
    if not url:
        raise RuntimeError("qwen.require_server is true but qwen.health_url is missing")
    with urlopen(str(url), timeout=3.0) as response:  # noqa: S310 - local health URL from config.
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"Qwen health check failed: {payload}")


def _session(config_path: str | Path, dataset: str | Path, resume_session: str | Path | None = None) -> OfflineSession:
    config = load_config(config_path)
    session_dir = Path(resume_session) if resume_session else Path(config.get("output_root", "outputs/v2_experiments/width_offline_fixed15")) / f"session_{_now_id()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    snapshot_yaml(config, session_dir / "config_snapshot.yaml")
    shutil.copy2(dataset, session_dir / "input_dataset_manifest.csv")
    return OfflineSession(config=config, config_path=Path(config_path), dataset_path=Path(dataset), session_dir=session_dir)


def _parse_case_ids(case_id: str | None = None, case_ids: str | None = None) -> list[str]:
    values: list[str] = []
    for raw in (case_id, case_ids):
        if not raw:
            continue
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return list(dict.fromkeys(values))


def _case_folder_name(row: dict[str, Any]) -> str:
    return f"case_{int(row['case_index']):03d}_{row['case_id']}"


def _archive_case_folder(session_dir: Path, dataset_row: dict[str, Any]) -> str | None:
    case_dir = session_dir / "cases" / _case_folder_name(dataset_row)
    if not case_dir.exists():
        return None
    archive_root = session_dir / "replaced_cases_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    dst = archive_root / f"{case_dir.name}_{_now_id()}"
    suffix = 1
    while dst.exists():
        dst = archive_root / f"{case_dir.name}_{_now_id()}_{suffix}"
        suffix += 1
    shutil.move(str(case_dir), str(dst))
    return str(dst)


def _artifact_paths(result: Any, axis: str, case_axis_dir: Path) -> dict[str, str | None]:
    session_dir = Path(getattr(result, "session_dir", ""))
    target = getattr(result, "target", None)
    anchor = (getattr(result, "anchors", {}) or {}).get(axis)
    path_length = (getattr(result, "path_lengths", {}) or {}).get(axis)
    target_diag = getattr(target, "diagnostics", {}) or {}
    anchor_artifacts = getattr(anchor, "candidate_artifacts", {}) or {}
    anchor_overlays = getattr(anchor, "overlay_paths", {}) or {}
    path_overlays = getattr(path_length, "overlay_paths", {}) or {}
    copied = {
        "selected_mask_overlay_path": _copy(getattr(target, "selected_mask_overlay_path", None), case_axis_dir),
        "crop_verification_panel_path": _copy(target_diag.get("crop_verification_panel_path"), case_axis_dir),
        "selected_anchor_overlay_path": _copy(anchor_overlays.get("selected_anchor_overlay"), case_axis_dir),
        "wide_contact_grid_path": _copy(anchor_artifacts.get("combined_wide_contact_guided_grid_2col") or anchor_artifacts.get("combined_wide_contact_guided_grid"), case_axis_dir),
        "local_chord_overlay_path": _copy(path_overlays.get("local_path_length_overlay"), case_axis_dir),
        "depth_used_points_overlay_path": _copy(path_overlays.get("depth_used_points_overlay"), case_axis_dir),
        "depth_edge_bins_overlay_path": _copy(path_overlays.get("depth_edge_bins_overlay"), case_axis_dir),
        "depth_projection_histogram_path": _copy(path_overlays.get("depth_projection_histogram"), case_axis_dir),
        "depth_width_diagnostics_path": _copy(path_overlays.get("depth_width_diagnostics"), case_axis_dir),
        "depth_pointcloud_npz_path": _copy(path_overlays.get("depth_local_pointcloud_npz"), case_axis_dir),
        "depth_pointcloud_ply_path": _copy(path_overlays.get("depth_local_pointcloud_ply"), case_axis_dir),
        "pipeline_result_path": _copy(session_dir / "pipeline_result.json", case_axis_dir),
        "stage_status_path": _copy(session_dir / "stage_status.json", case_axis_dir),
    }
    return copied


def _row_from_result(dataset_row: dict[str, Any], axis: str, result: Any, case_axis_dir: Path) -> dict[str, Any]:
    manual = dataset_row.get("manual_major_width_mm") if axis == "major" else dataset_row.get("manual_minor_width_mm")
    target = getattr(result, "target", None)
    anchor = (getattr(result, "anchors", {}) or {}).get(axis)
    path_length = (getattr(result, "path_lengths", {}) or {}).get(axis)
    anchor_diag = getattr(anchor, "diagnostics", {}) or {}
    path_diag = getattr(path_length, "diagnostics", {}) or {}
    depth_diag = path_diag.get("depth_width_diagnostics", {}) if isinstance(path_diag, dict) else {}
    qwen = anchor_diag.get("qwen", {}) if isinstance(anchor_diag, dict) else {}
    artifacts = _artifact_paths(result, axis, case_axis_dir)
    mask = getattr(path_length, "mask_path_length_mm", None)
    depth = getattr(path_length, "depth_path_length_mm", None)
    row = {
        "success": bool(getattr(result, "success", False)),
        "failed_stage": _failed_stage(result),
        "failure_reason": getattr(result, "failure_reason", None),
        "case_index": dataset_row.get("case_index"),
        "case_id": dataset_row.get("case_id"),
        "requested_material": dataset_row.get("requested_material"),
        "axis_mode": axis,
        "manual_width_mm": manual,
        "mask_path_length_mm": mask,
        "depth_path_length_mm": depth,
        "depth_valid": getattr(path_length, "depth_valid", None),
        "depth_validity_policy": depth_diag.get("depth_validity_policy"),
        "path_length_source": path_diag.get("path_length_source"),
        "mask_abs_error_mm": _abs_err(mask, manual),
        "depth_abs_error_mm": _abs_err(depth, manual),
        "mask_percent_error": _pct_err(mask, manual),
        "depth_percent_error": _pct_err(depth, manual),
        "selected_candidate_id": getattr(target, "selected_candidate_id", None),
        "selected_anchor_id": getattr(anchor, "final_anchor_id", None),
        "qwen_selected_anchor": anchor_diag.get("qwen_selected_anchor_id") if isinstance(anchor_diag, dict) else None,
        "qwen_repaired": qwen.get("qwen_response_repaired") if isinstance(qwen, dict) else anchor_diag.get("qwen_response_repaired") if isinstance(anchor_diag, dict) else None,
        "final_selected_anchor_source": anchor_diag.get("final_anchor_source") if isinstance(anchor_diag, dict) else None,
        "anchor_score": getattr(anchor, "score", None),
        "mask_compute_ms": path_diag.get("local_mask_compute_ms") if isinstance(path_diag, dict) else None,
        "depth_compute_ms": path_diag.get("local_depth_compute_ms") if isinstance(path_diag, dict) else None,
        "target_selection_runtime_ms": _stage_ms(result, "target_selection"),
        "anchor_selection_runtime_ms": _stage_ms(result, "anchor_selection"),
        "path_length_runtime_ms": _stage_ms(result, "path_length"),
        "total_pipeline_runtime_ms": getattr(result, "timing_ms", None),
        "rgb_path": dataset_row.get("rgb_path"),
        "depth_path": dataset_row.get("depth_path"),
        "camera_info_path": dataset_row.get("camera_info_path"),
        **artifacts,
    }
    return row


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def _write_failure_summary(rows: list[dict[str, Any]], session_dir: Path) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if str(row.get("success")).lower() == "true" or row.get("success") is True:
            continue
        key = (str(row.get("failed_stage") or "unknown"), str(row.get("failure_reason") or "unknown"))
        counts[key] = counts.get(key, 0) + 1
    failure_rows = [
        {"failed_stage": stage, "failure_reason": reason, "count": count}
        for (stage, reason), count in sorted(counts.items())
    ]
    _write_rows(session_dir / "failure_summary.csv", failure_rows, ["failed_stage", "failure_reason", "count"])


def _summary(rows: list[dict[str, Any]], session_dir: Path, extra_warnings: list[str] | None = None) -> list[str]:
    successes = [row for row in rows if str(row.get("success")).lower() == "true" or row.get("success") is True]
    warnings = list(extra_warnings or [])
    if not successes:
        warnings.append("No successful measurements found")
    if len(successes) != len(rows):
        warnings.append("Some measurements failed; see failure_summary.csv")
    if not _numeric_values(row.get("mask_compute_ms") for row in successes):
        warnings.append("No numeric mask_compute_ms values found")
    if not _numeric_values(row.get("depth_compute_ms") for row in successes):
        warnings.append("No numeric depth_compute_ms values found")
    mask_errors = _numeric_values(row.get("mask_abs_error_mm") for row in successes)
    depth_errors = _numeric_values(row.get("depth_abs_error_mm") for row in successes)
    summary = {
        "total_cases": len({row.get("case_id") for row in rows}),
        "total_expected_measurements": len(rows),
        "total_successful_measurements": len(successes),
        "total_failed_measurements": len(rows) - len(successes),
        "mask_mae_mm": _mean_or_none(mask_errors),
        "mask_rmse_mm": _rmse_or_none(mask_errors),
        "mask_mean_percent_error": _mean_or_none(row.get("mask_percent_error") for row in successes),
        "depth_mae_mm": _mean_or_none(depth_errors),
        "depth_rmse_mm": _rmse_or_none(depth_errors),
        "depth_mean_percent_error": _mean_or_none(row.get("depth_percent_error") for row in successes),
        "median_mask_runtime_ms": _median_or_none(row.get("mask_compute_ms") for row in successes),
        "median_depth_runtime_ms": _median_or_none(row.get("depth_compute_ms") for row in successes),
        "median_total_pipeline_runtime_ms": _median_or_none(row.get("total_pipeline_runtime_ms") for row in successes),
        "summary_warnings": sorted(set(warnings)),
    }
    _write_json(session_dir / "summary_overall.json", summary)
    _write_failure_summary(rows, session_dir)

    def grouped_csv(path: Path, key: str) -> None:
        groups = sorted({row.get(key) for row in successes if row.get(key)})
        out = []
        for group in groups:
            subset = [row for row in successes if row.get(key) == group]
            me = _numeric_values(row.get("mask_abs_error_mm") for row in subset)
            de = _numeric_values(row.get("depth_abs_error_mm") for row in subset)
            out.append({
                key: group,
                "n_measurements": len(subset),
                "mask_mae_mm": _mean_or_none(me),
                "depth_mae_mm": _mean_or_none(de),
                "mask_rmse_mm": _rmse_or_none(me),
                "depth_rmse_mm": _rmse_or_none(de),
                "mask_mean_percent_error": _mean_or_none(row.get("mask_percent_error") for row in subset),
                "depth_mean_percent_error": _mean_or_none(row.get("depth_percent_error") for row in subset),
            })
        cols = list(out[0].keys()) if out else [key, "n_measurements", "mask_mae_mm", "depth_mae_mm", "mask_rmse_mm", "depth_rmse_mm", "mask_mean_percent_error", "depth_mean_percent_error"]
        _write_rows(path, out, cols)

    grouped_csv(session_dir / "summary_by_material.csv", "requested_material")
    grouped_csv(session_dir / "summary_by_axis.csv", "axis_mode")
    method_rows = []
    for method, err_key, pct_key, ms_key in [
        ("mask", "mask_abs_error_mm", "mask_percent_error", "mask_compute_ms"),
        ("depth", "depth_abs_error_mm", "depth_percent_error", "depth_compute_ms"),
    ]:
        errs = _numeric_values(row.get(err_key) for row in successes)
        pcts = _numeric_values(row.get(pct_key) for row in successes)
        runtimes = _numeric_values(row.get(ms_key) for row in successes)
        method_rows.append({
            "method": method,
            "n_measurements": len(errs),
            "mae_mm": _mean_or_none(errs),
            "rmse_mm": _rmse_or_none(errs),
            "mean_percent_error": _mean_or_none(pcts),
            "median_compute_ms": _median_or_none(runtimes),
        })
    _write_rows(session_dir / "summary_by_method.csv", method_rows, ["method", "n_measurements", "mae_mm", "rmse_mm", "mean_percent_error", "median_compute_ms"])
    return sorted(set(warnings))


def _figures(rows: list[dict[str, Any]], session_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = session_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    successes = [row for row in rows if str(row.get("success")).lower() == "true" or row.get("success") is True]
    warnings: list[str] = []

    def scatter(path: str, value_key: str, title: str) -> None:
        x = [_safe_float(row.get("manual_width_mm")) for row in successes]
        y = [_safe_float(row.get(value_key)) for row in successes]
        pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if not pts:
            warnings.append(f"Skipped {path}: no numeric data")
            return
        plt.figure(figsize=(5, 5))
        xs, ys = zip(*pts)
        plt.scatter(xs, ys, s=28)
        lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("manual width (mm)")
        plt.ylabel(title)
        plt.tight_layout()
        plt.savefig(fig_dir / path, dpi=180)
        plt.close()

    scatter("error_scatter_mask_vs_manual.png", "mask_path_length_mm", "mask path length (mm)")
    scatter("error_scatter_depth_vs_manual.png", "depth_path_length_mm", "depth path length (mm)")
    for name, labels, series in [
        ("boxplot_absolute_error_by_method.png", ["mask", "depth"], [[_safe_float(r.get("mask_abs_error_mm")) for r in successes], [_safe_float(r.get("depth_abs_error_mm")) for r in successes]]),
        ("runtime_boxplot.png", ["mask", "depth"], [[_safe_float(r.get("mask_compute_ms")) for r in successes], [_safe_float(r.get("depth_compute_ms")) for r in successes]]),
    ]:
        if not any(_numeric_values(values) for values in series):
            warnings.append(f"Skipped {name}: no numeric data")
            continue
        plt.figure(figsize=(6, 4))
        clean = [_numeric_values(values) or [0.0] for values in series]
        plt.boxplot(clean, labels=labels)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()
    for key, name in [("requested_material", "boxplot_absolute_error_by_material.png"), ("axis_mode", "boxplot_absolute_error_by_axis.png")]:
        labels = sorted({row.get(key) for row in successes if row.get(key)})
        if not labels:
            warnings.append(f"Skipped {name}: no successful grouped data")
            continue
        data = [_numeric_values(row.get("depth_abs_error_mm") for row in successes if row.get(key) == label) or [0.0] for label in labels]
        plt.figure(figsize=(7, 4))
        plt.boxplot(data, labels=labels or ["none"])
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()
    _example_panel(successes, fig_dir / "example_success_panel.png")
    return warnings


def _example_panel(rows: list[dict[str, Any]], output: Path) -> None:
    image_keys = [
        "rgb_path",
        "selected_mask_overlay_path",
        "selected_anchor_overlay_path",
        "local_chord_overlay_path",
        "depth_used_points_overlay_path",
        "depth_edge_bins_overlay_path",
    ]
    row = next((r for r in rows if all(r.get(k) and Path(str(r.get(k))).exists() for k in image_keys[:2])), None)
    panels = []
    labels = ["raw RGB", "mask", "anchor", "local chord", "depth points", "edge bins"]
    for key in image_keys:
        if row and row.get(key) and Path(str(row[key])).exists():
            panels.append(Image.open(str(row[key])).convert("RGB").resize((260, 190)))
        else:
            img = Image.new("RGB", (260, 190), (240, 240, 240))
            ImageDraw.Draw(img).text((20, 80), key, fill=(40, 40, 40))
            panels.append(img)
    canvas = Image.new("RGB", (3 * 260, 2 * 220), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, panel in enumerate(panels):
        x = (idx % 3) * 260
        y = (idx // 3) * 220 + 25
        draw.text((x + 8, y - 20), labels[idx], fill=(0, 0, 0))
        canvas.paste(panel, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_offline(
    *,
    dataset: str | Path,
    config_path: str | Path,
    case_id: str | None = None,
    case_ids: str | None = None,
    resume_session: str | Path | None = None,
    anchor_backend: str | None = None,
    fail_fast: bool | None = None,
    replace_existing_case: bool = False,
    pipeline_runner: Callable[..., Any] = run_full_main_upv_vlm_v2_pipeline,
    health_check: bool = True,
) -> Path:
    session = _session(config_path, dataset, resume_session)
    if health_check:
        _health_check(session.config)
    requested_case_ids = _parse_case_ids(case_id=case_id, case_ids=case_ids)
    rows_in = [row for row in _read_rows(dataset) if str(row.get("status", "saved")).lower() == "saved"]
    if requested_case_ids:
        requested = set(requested_case_ids)
        rows_in = [row for row in rows_in if row.get("case_id") in requested]
    if replace_existing_case and not resume_session:
        raise ValueError("--replace-existing-case requires --resume-session")
    if replace_existing_case and not requested_case_ids:
        raise ValueError("--replace-existing-case requires --case-id or --case-ids")
    axes = list(session.config.get("axis_modes", ["major", "minor"]))
    existing_rows = _read_rows_if_exists(session.session_dir / "master_results.csv") if resume_session else []
    removed_count = 0
    archived: list[str] = []
    if replace_existing_case:
        replace_set = set(requested_case_ids)
        print(f"Replacing existing case rows: {','.join(requested_case_ids)}")
        kept_rows = [row for row in existing_rows if row.get("case_id") not in replace_set]
        removed_count = len(existing_rows) - len(kept_rows)
        for item in rows_in:
            archived_path = _archive_case_folder(session.session_dir, item)
            if archived_path:
                archived.append(archived_path)
        rows = kept_rows
    else:
        rows = []
    added_count = 0
    for item in rows_in:
        for axis in axes:
            axis_dir = session.session_dir / "cases" / f"case_{int(item['case_index']):03d}_{item['case_id']}" / f"axis_{axis}"
            axis_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = pipeline_runner(
                    config_path=session.config["main_pipeline_config"],
                    requested_material=item["requested_material"],
                    axis_mode=axis,
                    mode=session.config.get("mode", "path_length_only"),
                    input_rgb=item["rgb_path"],
                    input_depth=item["depth_path"],
                    input_camera_info=item["camera_info_path"],
                    live_ros2=False,
                    execution_backend=session.config.get("execution_backend", "none"),
                    output_root=str(axis_dir / "pipeline_session"),
                )
                row = _row_from_result(item, axis, result, axis_dir)
                _write_json(axis_dir / "pipeline_result.json", result)
                _write_json(axis_dir / "stage_status.json", getattr(result, "stage_status", []))
            except Exception as exc:
                if fail_fast if fail_fast is not None else bool((session.config.get("failure_policy") or {}).get("fail_fast", False)):
                    raise
                manual = item.get("manual_major_width_mm") if axis == "major" else item.get("manual_minor_width_mm")
                row = {
                    "success": False,
                    "failed_stage": "offline_runner",
                    "failure_reason": str(exc),
                    "case_index": item.get("case_index"),
                    "case_id": item.get("case_id"),
                    "requested_material": item.get("requested_material"),
                    "axis_mode": axis,
                    "manual_width_mm": manual,
                    "rgb_path": item.get("rgb_path"),
                    "depth_path": item.get("depth_path"),
                    "camera_info_path": item.get("camera_info_path"),
                }
            rows.append(row)
            added_count += 1
            _write_rows(session.session_dir / "master_results.csv", rows, MASTER_COLUMNS)
    summary_warnings = _summary(rows, session.session_dir)
    if bool((session.config.get("figures") or {}).get("generate", True)):
        figure_warnings = _figures(rows, session.session_dir)
        if figure_warnings:
            _summary(rows, session.session_dir, extra_warnings=summary_warnings + figure_warnings)
    manifest = {
        "dataset": str(dataset),
        "config": str(config_path),
        "session_dir": str(session.session_dir),
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "row_count": len(rows),
        "anchor_backend_override": anchor_backend,
        "replace_existing_case": bool(replace_existing_case),
        "replaced_case_ids": requested_case_ids if replace_existing_case else [],
        "removed_old_rows": removed_count,
        "added_new_rows": added_count,
        "archived_case_dirs": archived,
    }
    _write_json(session.session_dir / "offline_run_manifest.json", manifest)
    if replace_existing_case:
        print(f"Removed old rows: {removed_count}")
        print(f"Added new rows: {added_count}")
        print(f"Final row count: {len(rows)}")
    return session.session_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 offline width/path-length evaluation from a captured dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--case-ids")
    parser.add_argument("--resume-session")
    parser.add_argument("--anchor-backend", choices=["deterministic", "qwen"])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--replace-existing-case", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = run_offline(
        dataset=args.dataset,
        config_path=args.config,
        case_id=args.case_id,
        case_ids=args.case_ids,
        resume_session=args.resume_session,
        anchor_backend=args.anchor_backend,
        fail_fast=args.fail_fast,
        replace_existing_case=args.replace_existing_case,
    )
    print(f"offline_session: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
