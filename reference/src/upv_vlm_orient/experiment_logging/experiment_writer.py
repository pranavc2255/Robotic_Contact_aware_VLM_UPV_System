from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

from upv_vlm_orient.experiment_logging.experiment_schema import (
    EXPERIMENT_CATEGORIES,
    FAILURE_CLASSES,
    NA,
    RUN_MODES,
    all_metric_keys,
    default_manual_labels,
    default_upv_results,
    na_metrics,
    now_iso,
)
from upv_vlm_orient.experiment_logging.git_snapshot import write_git_snapshot
from upv_vlm_orient.experiment_logging.prompt_logging import prompt_summary_csv_row, write_prompt_package


class ExperimentWriterError(RuntimeError):
    pass


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            fieldnames = list(dict.fromkeys([*(reader.fieldnames or []), *fieldnames]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


class ExperimentRunWriter:
    def __init__(self, repo_root: Path, config: dict[str, Any], config_path: Path | None = None) -> None:
        self.repo_root = repo_root
        self.config = config
        self.config_path = config_path
        output_root = Path(config.get("output_root", "outputs/E1_autonomous_upv_experiment_manager"))
        if not output_root.is_absolute():
            output_root = repo_root / output_root
        self.output_root = output_root

    def make_run_dir(self, short_run_id: str) -> Path:
        stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_").split("-")[0]
        root = self.output_root / f"{stamp}_{short_run_id}"
        candidate = root
        idx = 2
        while candidate.exists():
            candidate = self.output_root / f"{root.name}_{idx:02d}"
            idx += 1
        for subdir in ["figures", "prompts", "crops", "logs", "video", "rosbag", "t4_linked_outputs", "configs", "raw"]:
            (candidate / subdir).mkdir(parents=True, exist_ok=True)
        return candidate

    def snapshot_configs(self, run_dir: Path) -> None:
        write_json(run_dir / "config_snapshot.json", self.config)
        write_json(run_dir / "configs" / "E1_config_snapshot.json", self.config)
        if self.config_path and self.config_path.exists():
            shutil.copyfile(self.config_path, run_dir / "configs" / self.config_path.name)
        t4_path = self.config.get("t4_config_path")
        if t4_path:
            path = Path(t4_path)
            if not path.is_absolute():
                path = self.repo_root / path
            if path.exists():
                shutil.copyfile(path, run_dir / "configs" / path.name)
        handeye = self.repo_root / "configs" / "handeye_tool0_camera_color_optical_VERIFIED.json"
        if handeye.exists():
            shutil.copyfile(handeye, run_dir / "configs" / handeye.name)

    def create_run(
        self,
        *,
        metadata: dict[str, Any],
        pipeline_mode: str,
        selected_metrics: list[str] | None = None,
        manual_labels: dict[str, Any] | None = None,
        upv_results: dict[str, Any] | None = None,
        prompt_overrides: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> Path:
        run_id = str(metadata.get("run_id") or f"{metadata.get('specimen_id', 'specimen')}_{metadata.get('trial_index', 'trial')}")
        run_dir = self.make_run_dir(run_id)
        categories = metadata.get("experiment_category_tags") or []
        if not categories:
            categories = [category for category in EXPERIMENT_CATEGORIES if metadata.get(category)]
        metadata = {
            "timestamp": now_iso(),
            "run_id": run_dir.name,
            "dry_run": dry_run,
            "operator_name": metadata.get("operator_name", self.config.get("default_operator", "Pranav")),
            "project_name": metadata.get("project_name", "UPV_VLM_orient"),
            "specimen_id": metadata.get("specimen_id", "specimen_001"),
            "scene_id": metadata.get("scene_id", NA),
            "trial_index": metadata.get("trial_index", NA),
            "notes": metadata.get("notes", ""),
            "material_query": metadata.get("material_query", self.config.get("default_material_query", "brick")),
            "axis_mode": metadata.get("axis_mode", self.config.get("default_axis_mode", "major")),
            "run_mode": metadata.get("run_mode", self.config.get("default_run_mode", "full_autonomous_home")),
            "pipeline_mode": pipeline_mode,
            "experiment_category_tags": categories or [NA],
            "model_selections": metadata.get("model_selections", {}),
            "valid_run_modes": RUN_MODES,
        }
        if metadata["run_mode"] not in RUN_MODES:
            raise ExperimentWriterError(f"Unsupported E1 run mode: {metadata['run_mode']}")

        self.snapshot_configs(run_dir)
        git_snapshot = write_git_snapshot(run_dir / "git_snapshot.txt", self.repo_root)
        write_json(run_dir / "software_environment.json", git_snapshot)
        (run_dir / "software_environment.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in git_snapshot.items()) + "\n",
            encoding="utf-8",
        )

        prompt_cfg = self.config.get("prompt_engineering", {})
        prompt_package = write_prompt_package(
            run_dir,
            pipeline_mode=pipeline_mode,
            material_query=str(metadata["material_query"]),
            prompt_set_name=str(prompt_cfg.get("default_prompt_set_name", "default_v1")),
            prompt_set_version=str(prompt_cfg.get("default_prompt_set_version", "v1")),
            operator_name=str(metadata["operator_name"]),
            prompt_overrides=prompt_overrides,
        )

        metrics = na_metrics(selected_metrics or all_metric_keys())
        labels = default_manual_labels()
        labels.update(manual_labels or {})
        upv = default_upv_results()
        upv.update(upv_results or {})
        timing = {
            key: NA for key in [
                "total_wall_time_s",
                "total_core_pipeline_time_s",
                "total_io_visualization_time_s",
                "total_robot_motion_time_s",
                "total_clamp_time_s",
            ]
        }
        for key in list(timing.keys()):
            timing[f"{key}_NA_reason"] = "dry_run_or_not_available_from_underlying_T4_logs"

        result_skeletons = {
            "perception_results.json": {"status": "not_run" if dry_run else NA, "metrics": metrics},
            "geometry_results.json": {"status": "not_run" if dry_run else NA, "metrics": metrics},
            "robot_results.json": {"status": "not_run", "t4_session_dir": metadata.get("t4_session_dir", NA), "robot_motion_command_sent": False},
            "clamp_results.json": {"status": "not_run", "arduino_command_sent": False, "clamp_command_sent": False},
            "upv_results.json": upv,
        }
        write_json(run_dir / "run_metadata.json", metadata)
        write_json(run_dir / "selected_metrics.json", {"selected_metrics": selected_metrics or all_metric_keys(), "metrics": metrics})
        write_json(run_dir / "manual_labels.json", labels)
        write_json(run_dir / "timing_summary.json", timing)
        for filename, payload in result_skeletons.items():
            write_json(run_dir / filename, payload)
        failure = {
            "failure_class": "none" if dry_run else NA,
            "valid_failure_classes": FAILURE_CLASSES,
            "failure_stage": NA,
            "failure_reason_auto": NA,
            "failure_reason_manual": NA,
            "manual_override_allowed": True,
        }
        write_json(run_dir / "failure_taxonomy.json", failure)
        full_task = {
            "run_id": run_dir.name,
            "run_mode": metadata["run_mode"],
            "pipeline_mode": pipeline_mode,
            "full_autonomous_success": NA,
            "t4_session_dir": metadata.get("t4_session_dir", NA),
            "t4_run_ids": [],
            "full_final_home_json_path": NA,
            "no_physical_hardware_used_in_dry_run": dry_run,
            "upv_measurement_triggered": False,
            "prompt_engineering_summary_path": str(run_dir / "prompt_engineering_summary.json"),
        }
        write_json(run_dir / "full_task_summary.json", full_task)
        summary_lines = [
            "E1 Autonomous UPV Experiment Manager Run",
            f"run_id: {run_dir.name}",
            f"dry_run: {dry_run}",
            f"operator: {metadata['operator_name']}",
            f"material_query: {metadata['material_query']}",
            f"run_mode: {metadata['run_mode']}",
            f"pipeline_mode: {pipeline_mode}",
            f"prompt_summary: {run_dir / 'prompt_engineering_summary.json'}",
            "hardware_used: false" if dry_run else "hardware_used: see T4 logs",
        ]
        (run_dir / "E1_run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        master_row = {
            "run_id": run_dir.name,
            **metadata,
            **metrics,
            "failure_class": failure["failure_class"],
            "t4_session_dir": full_task["t4_session_dir"],
            "upv_measurement_triggered": False,
        }
        append_csv(self.output_root / "E1_master_results.csv", master_row)
        prompt_row = prompt_summary_csv_row(run_dir.name, pipeline_mode, prompt_package["summary"])
        append_csv(self.output_root / "E1_prompt_engineering_summary.csv", prompt_row)
        append_csv(run_dir / "E1_run_summary.csv", master_row)
        return run_dir
