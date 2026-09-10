from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from upv_vlm_v1.experiment.experiment_manager import ExperimentManager
from upv_vlm_v1.experiment.paper_results_writer import PaperResultsWriter, load_json_if_exists, write_json
from upv_vlm_v1.experiment.paper_trial_schema import PaperTrialMetadata, PaperTrialResult, now_iso
from upv_vlm_v1.experiment.progress_tracker import (
    make_progress_template,
    update_stage,
    write_progress,
    write_progress_summary_csv,
)


class PaperExperimentRunner:
    def __init__(self, repo_root: str | Path, paper_config_path: str | Path, campaign_id: str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.paper_config_path = Path(paper_config_path)
        if not self.paper_config_path.is_absolute():
            self.paper_config_path = self.repo_root / self.paper_config_path
        self.config = json.loads(self.paper_config_path.read_text(encoding="utf-8"))
        self.writer = PaperResultsWriter(self.repo_root, self.config, campaign_id=campaign_id)
        self.manager = ExperimentManager(self.repo_root, self.config.get("default_backend_config", "configs/E1_autonomous_upv_experiment_manager.json"))

    def make_metadata(self, values: dict[str, Any]) -> PaperTrialMetadata:
        trial_id = str(values.get("trial_id") or "")
        if not trial_id:
            trial_id = f"{values.get('experiment_group', 'trial')}_{now_iso().replace(':', '').replace('-', '').replace('T', '_').split('-')[0]}"
        return PaperTrialMetadata(
            trial_id=trial_id,
            experiment_group=str(values.get("experiment_group", "target_selection")),
            material_query=str(values.get("material_query", "brick")),
            specimen_id=str(values.get("specimen_id", "brick_01")),
            scene_id=str(values.get("scene_id", "scene_01")),
            trial_index=int(values.get("trial_index", 1)),
            axis_mode=str(values.get("axis_mode", "major")),
            edge_condition=str(values.get("edge_condition", "clean")),
            orientation_case=str(values.get("orientation_case", "0")),
            objects_present=str(values.get("objects_present", "NA")),
            notes=str(values.get("notes", "")),
            operator_name=str(values.get("operator_name", self.config.get("default_operator", "Pranav"))),
        )

    def create_trial(self, metadata: PaperTrialMetadata) -> Path:
        return self.writer.create_trial(metadata, config_snapshot=self.config)

    def run_target_selection_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        return self._run_t4_safe_trial(Path(trial_dir), metadata, "target_selection", "perception_only", dry_run=dry_run)

    def run_geometry_axis_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        return self._run_t4_safe_trial(Path(trial_dir), metadata, "geometry_axis", "plan_only", dry_run=dry_run)

    def run_anchor_selection_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        result = self._run_t4_safe_trial(Path(trial_dir), metadata, "anchor_selection", "plan_only", dry_run=dry_run)
        result.setdefault("metrics", {})["anchor_backend_status"] = (
            "backend_not_fully_wired_for_anchor_selection_mode" if not dry_run else "dry_run_no_anchor_backend_called"
        )
        result["failure_reason"] = result.get("failure_reason") or "NA"
        self.writer.save_trial_result(Path(trial_dir), result)
        return result

    def run_path_length_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        result = self._run_t4_safe_trial(Path(trial_dir), metadata, "path_length", "plan_only", dry_run=dry_run)
        result.setdefault("metrics", {}).update(
            {
                "depth_refined_probe_spacing_mm": "NA",
                "recommended_upv_path_length_mm": "NA",
                "path_length_backend_status": "T3/depth refinement requires saved RGB-D input; no live hardware is used in validation.",
            }
        )
        self.writer.save_trial_result(Path(trial_dir), result)
        return result

    def run_contact_ablation_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        trial_path = Path(trial_dir)
        progress_file = trial_path / "run_progress.json"
        progress = update_stage(make_progress_template("contact_ablation"), "create_trial", "succeeded", success=True)
        for stage in ["load_saved_anchor_data", "run_ablation_replay", "save_outputs", "complete"]:
            progress = update_stage(progress, stage, "succeeded", success=True)
        write_progress(progress, progress_file)
        write_progress_summary_csv(progress, trial_path / "run_progress_summary.csv")
        result = self._dry_result(
            "contact_ablation",
            "backend_not_fully_wired_for_contact_ablation_mode" if not dry_run else "dry_run_contact_ablation_no_backend_called",
        )
        result["success"] = bool(dry_run)
        result["failure_stage"] = "backend_availability" if not dry_run else "NA"
        result["failure_reason"] = "backend_not_fully_wired_for_contact_ablation_mode" if not dry_run else "NA"
        result.setdefault("metrics", {})["contact_ablation_backend_status"] = result["failure_reason"] if not dry_run else "dry_run_no_hardware_called"
        self.writer.save_trial_result(trial_path, result)
        return result

    def run_end_to_end_upv_trial(self, trial_dir: str | Path, metadata: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        trial_path = Path(trial_dir)
        progress_file = trial_path / "run_progress.json"
        progress = update_stage(make_progress_template("end_to_end_upv"), "create_run_folder", "succeeded", success=True)
        write_progress(progress, progress_file)
        if dry_run:
            for stage in [
                "snap_capture_rgbd",
                "perception_and_target_selection",
                "robot_and_clamp_planning",
                "move_midhover_initial_height",
                "orient_tool",
                "xy_alignment",
                "approach_preview",
                "approach_final",
                "clamp_close_to_planned_width",
                "upv_hold_reading_window",
                "clamp_release",
                "return_home",
                "complete",
            ]:
                progress = update_stage(progress, stage, "succeeded", success=True)
            write_progress(progress, progress_file)
            write_progress_summary_csv(progress, trial_path / "run_progress_summary.csv")
            result = self._dry_result("end_to_end_upv", "dry_run_no_hardware_called")
            self.writer.save_trial_result(trial_path, result)
            return result
        adapter = self.manager.make_t4_adapter(self.config["t4_config_path"], self.config.get("canonical_prompt_config"))
        backend = adapter.run_oneshot(
            "full_final_home",
            confirmation="RUN_FINAL_HOME",
            e1_run_dir=trial_path,
            prompt_config=self.config.get("canonical_prompt_config"),
            prompt_overrides_json=None,
            progress_file=progress_file,
        )
        self._save_backend_result(trial_path, backend)
        result = self._result_from_backend("end_to_end_upv", backend)
        result["linked_e1_or_t4_dir"] = backend.get("t4_session_dir", "NA")
        result["metrics"]["backend_action"] = "full_final_home"
        self.writer.save_trial_result(trial_path, result)
        return result

    def _run_t4_safe_trial(self, trial_dir: Path, metadata: dict[str, Any], group: str, action: str, dry_run: bool = False) -> dict[str, Any]:
        progress_file = trial_dir / "run_progress.json"
        progress = update_stage(make_progress_template(group), "create_trial", "succeeded", success=True)
        write_progress(progress, progress_file)
        if dry_run:
            for stage in [stage.name for stage in progress.stages if stage.name != "create_trial"]:
                progress = update_stage(progress, stage, "succeeded", success=True)
            write_progress(progress, progress_file)
            write_progress_summary_csv(progress, trial_dir / "run_progress_summary.csv")
            result = self._dry_result(group, f"dry_run_{action}_not_called")
            self.writer.save_trial_result(trial_dir, result)
            return result
        adapter = self.manager.make_t4_adapter(self.config["t4_config_path"], self.config.get("canonical_prompt_config"))
        backend = adapter.run_oneshot(
            action,
            e1_run_dir=trial_dir,
            prompt_config=self.config.get("canonical_prompt_config"),
            progress_file=progress_file,
        )
        self._save_backend_result(trial_dir, backend)
        result = self._result_from_backend(group, backend)
        if str(backend.get("stdout", "")).find("no_hardware_validation_mode") >= 0:
            result.setdefault("metrics", {})["backend_availability_note"] = "T4 one-shot returned no-hardware validation summary for this mode."
        self.writer.save_trial_result(trial_dir, result)
        return result

    def _dry_result(self, group: str, note: str) -> dict[str, Any]:
        return PaperTrialResult(
            success=True,
            failure_stage="NA",
            failure_reason="NA",
            backend_run_dir="NA",
            linked_e1_or_t4_dir="NA",
            key_artifacts={},
            metrics={
                "experiment_group": group,
                "dry_run": True,
                "backend_status": note,
                "hardware_used": False,
            },
            manual_ground_truth={},
            upv_reading={},
            timing={"completed_at": now_iso()},
        ).to_dict()

    def _result_from_backend(self, group: str, backend: dict[str, Any]) -> dict[str, Any]:
        return PaperTrialResult(
            success=bool(backend.get("success")),
            failure_stage=str(backend.get("abort_stage", "NA")),
            failure_reason=str(backend.get("failure_reason", "NA")),
            backend_run_dir=str(backend.get("t4_session_dir", "NA")),
            linked_e1_or_t4_dir=str(backend.get("t4_session_dir", "NA")),
            key_artifacts={"overlay_files": backend.get("overlay_files", []), "summary_files": backend.get("summary_files", [])},
            metrics={
                "experiment_group": group,
                "backend_action": backend.get("action", "NA"),
                "robot_motion_command_sent": backend.get("robot_motion_command_sent", False),
                "arduino_command_sent": backend.get("arduino_command_sent", False),
                "clamp_command_sent": backend.get("clamp_command_sent", False),
            },
            manual_ground_truth={},
            upv_reading={},
            timing={"completed_at": now_iso()},
        ).to_dict()

    def _save_backend_result(self, trial_dir: Path, backend: dict[str, Any]) -> None:
        write_json(trial_dir / "backend_outputs" / "backend_result.json", backend)
        for idx, path_text in enumerate(backend.get("overlay_files", [])[:8], start=1):
            src = Path(path_text)
            if src.exists() and src.is_file():
                dst = trial_dir / "figures" / f"{idx:02d}_{src.name}"
                try:
                    shutil.copyfile(src, dst)
                except Exception:  # noqa: BLE001
                    pass
        manifest = {
            "backend_run_dir": backend.get("t4_session_dir", "NA"),
            "summary_files": backend.get("summary_files", []),
            "overlay_files": backend.get("overlay_files", []),
        }
        write_json(trial_dir / "backend_outputs" / "linked_outputs_manifest.json", manifest)

    def save_manual_ground_truth(self, trial_dir: str | Path, manual: dict[str, Any]) -> None:
        self.writer.save_manual_ground_truth(Path(trial_dir), manual)

    def save_completed_trial(self, trial_dir: str | Path, manual: dict[str, Any]) -> None:
        self.writer.save_completed_trial(Path(trial_dir), manual=manual)

    def generate_summaries(self) -> dict[str, str]:
        return self.writer.generate_summaries(self.config.get("target_counts", {}))

    def latest_trial_result(self, trial_dir: str | Path) -> dict[str, Any]:
        return load_json_if_exists(Path(trial_dir) / "trial_result.json")
