from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo

from upv_vlm_v1.experiment.paper_trial_schema import PaperTrialMetadata, flatten_trial_row, now_iso, slug_text


GROUP_CSV = {
    "target_selection": "E2_target_selection_trials.csv",
    "geometry_axis": "E2_geometry_axis_trials.csv",
    "anchor_selection": "E2_anchor_selection_trials.csv",
    "path_length": "E2_path_length_trials.csv",
    "contact_ablation": "E2_contact_ablation_trials.csv",
    "end_to_end_upv": "E2_end_to_end_upv_trials.csv",
}

SUMMARY_TABLE_CSV = {
    "target_selection": "table_target_selection.csv",
    "geometry_axis": "table_geometry_axis.csv",
    "anchor_selection": "table_anchor_selection.csv",
    "path_length": "table_path_length.csv",
    "contact_ablation": "table_contact_ablation.csv",
    "end_to_end_upv": "table_end_to_end.csv",
}

BASE_TRIAL_FIELDS = [
    "trial_id",
    "experiment_group",
    "material_query",
    "specimen_id",
    "scene_id",
    "trial_index",
    "axis_mode",
    "edge_condition",
    "orientation_case",
    "success",
    "failure_stage",
    "failure_reason",
    "trial_saved",
    "saved_at",
]

EXPERIMENT_FOLDERS = {
    "target_selection": "01_target_selection",
    "geometry_axis": "02_geometry_axis",
    "anchor_selection": "03_anchor_selection",
    "path_length": "04_path_length",
    "contact_ablation": "05_contact_ablation",
    "end_to_end_upv": "06_end_to_end_upv",
}

PAPER_DATA_FIGURE_FOLDERS = {
    "target_selection": "target_selection",
    "geometry_axis": "geometry_axis",
    "anchor_selection": "anchor_selection",
    "path_length": "path_length",
    "contact_ablation": "ablation",
    "end_to_end_upv": "end_to_end_upv",
}

REQUIRED_TOP_LEVEL_FOLDERS = [
    "00_campaign_metadata",
    "01_target_selection",
    "02_geometry_axis",
    "03_anchor_selection",
    "04_path_length",
    "05_contact_ablation",
    "06_end_to_end_upv",
    "90_paper_data",
    "91_exports",
    "92_logs",
    "99_debug",
]


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        for old in existing:
            fieldnames = list(dict.fromkeys([*fieldnames, *old.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing:
            writer.writerow({key: old.get(key, "NA") for key in fieldnames})
        writer.writerow({key: row.get(key, "NA") for key in fieldnames})


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "NA") for key in fieldnames})


def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    if not path.exists():
        write_csv_rows(path, [], fieldnames=fieldnames)


def campaign_id_from_name(name: str) -> str:
    stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_").split("-")[0]
    clean = slug_text(name or "paper_core_v1")
    return f"campaign_{stamp}_{clean}"


def _timestamp_compact() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def create_paper_experiment_root(output_root: str | Path, experiment_number: int | str, label: str | None = None) -> Path:
    root = Path(output_root)
    number = str(experiment_number).strip() or "1"
    candidate = root / f"Paper_experiment_{number}_{_timestamp_compact()}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"Paper_experiment_{number}_{_timestamp_compact()}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def create_paper_experiment_structure(paper_experiment_root: Path) -> dict[str, str]:
    folders: dict[str, str] = {}
    for name in REQUIRED_TOP_LEVEL_FOLDERS:
        path = paper_experiment_root / name
        path.mkdir(parents=True, exist_ok=True)
        folders[name] = str(path)
    paper = paper_experiment_root / "90_paper_data"
    for group in ["target_selection", "geometry_axis", "anchor_selection", "path_length", "ablation", "end_to_end_upv"]:
        (paper / "figures_source" / group).mkdir(parents=True, exist_ok=True)
    for sub in ["tables_source", "plots_source"]:
        (paper / sub).mkdir(parents=True, exist_ok=True)
    for name in [
        "axis_error_data.csv",
        "width_error_data.csv",
        "path_length_error_data.csv",
        "timing_data.csv",
        "upv_velocity_data.csv",
        "success_rate_data.csv",
    ]:
        ensure_csv(paper / "plots_source" / name, ["status"])
    for table_name in SUMMARY_TABLE_CSV.values():
        ensure_csv(paper / "tables_source" / table_name, BASE_TRIAL_FIELDS)
    ensure_csv(paper / "tables_source" / "table_stagewise_success.csv", ["experiment_group", "stage", "success_count", "total_count"])
    ensure_csv(paper / "tables_source" / "table_upv_readings.csv", ["trial_id", "time_of_flight", "path_length_mm", "velocity_m_per_s"])
    for name in ["hardware_control_logs", "app_logs", "backend_logs"]:
        (paper_experiment_root / "92_logs" / name).mkdir(parents=True, exist_ok=True)
    for name in ["dry_runs", "raw_backend_outputs", "temporary_files", "validation_reports"]:
        (paper_experiment_root / "99_debug" / name).mkdir(parents=True, exist_ok=True)
    ensure_csv(paper / "figure_manifest.csv", ["trial_id", "experiment_group", "source_path", "paper_path"])
    ensure_csv(paper / "table_manifest.csv", ["table_name", "relative_path", "updated_at"])
    exports = paper_experiment_root / "91_exports"
    ensure_csv(exports / "E2_master_trials.csv", BASE_TRIAL_FIELDS)
    for csv_name in GROUP_CSV.values():
        ensure_csv(exports / csv_name, BASE_TRIAL_FIELDS)
    ensure_csv(exports / "E2_progress_summary.csv", ["experiment_group", "completed", "target", "missing", "percent_complete"])
    ensure_csv(exports / "figure_manifest.csv", ["trial_id", "experiment_group", "source_path", "paper_path"])
    ensure_csv(exports / "table_manifest.csv", ["table_name", "relative_path", "updated_at"])
    report = exports / "E2_summary_report.md"
    if not report.exists():
        report.write_text("# E2 Paper Results Summary\n\nNo saved trials yet.\n", encoding="utf-8")
    return folders


def get_experiment_folder(paper_experiment_root: Path, experiment_group: str) -> Path:
    if experiment_group not in EXPERIMENT_FOLDERS:
        raise ValueError(f"Unsupported experiment_group: {experiment_group}")
    path = paper_experiment_root / EXPERIMENT_FOLDERS[experiment_group]
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_trial_run_folder(paper_experiment_root: Path, experiment_group: str, metadata: PaperTrialMetadata) -> Path:
    folder = get_experiment_folder(paper_experiment_root, experiment_group)
    material = slug_text(metadata.material_query)
    scene = slug_text(metadata.scene_id)
    edge = slug_text(metadata.edge_condition)
    axis = slug_text(metadata.axis_mode)
    orient = slug_text(metadata.orientation_case)
    idx = int(metadata.trial_index)
    if experiment_group == "target_selection":
        name = f"target_selection_run_{idx:03d}_{material}_{scene}"
    elif experiment_group == "geometry_axis":
        name = f"geometry_axis_run_{idx:03d}_{material}_{orient}deg"
    elif experiment_group == "anchor_selection":
        name = f"anchor_selection_run_{idx:03d}_{material}_{edge}"
    elif experiment_group == "path_length":
        name = f"path_length_run_{idx:03d}_{material}_{axis}"
    elif experiment_group == "contact_ablation":
        name = f"ablation_run_{idx:03d}_{scene}"
    elif experiment_group == "end_to_end_upv":
        name = f"robotic_upv_run_{idx:03d}_{material}_{edge}"
    else:
        name = f"{experiment_group}_run_{idx:03d}_{material}"
    candidate = folder / name
    suffix = 2
    while candidate.exists():
        candidate = folder / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def update_paper_data_sources(paper_experiment_root: Path, row: dict[str, Any]) -> None:
    group = str(row.get("experiment_group", ""))
    if group not in SUMMARY_TABLE_CSV:
        return
    paper_tables = paper_experiment_root / "90_paper_data" / "tables_source"
    append_csv(paper_tables / SUMMARY_TABLE_CSV[group], row)
    upv_fields = {
        "trial_id": row.get("trial_id", "NA"),
        "time_of_flight": row.get("manual_time_of_flight_value", row.get("upv_time_of_flight_us", "NA")),
        "path_length_mm": row.get("manual_manual_path_length_mm", row.get("metric_recommended_upv_path_length_mm", "NA")),
        "velocity_m_per_s": row.get("manual_computed_velocity_m_per_s", row.get("upv_velocity_m_per_s", "NA")),
    }
    if group == "end_to_end_upv":
        append_csv(paper_tables / "table_upv_readings.csv", upv_fields)


def refresh_exports(paper_experiment_root: Path, target_counts: dict[str, int] | None = None) -> dict[str, str]:
    writer = PaperResultsWriter(Path.cwd(), {"output_root": str(paper_experiment_root.parent)}, paper_experiment_root=paper_experiment_root)
    return writer.generate_summaries(target_counts or {})


class PaperResultsWriter:
    def __init__(self, repo_root: str | Path, config: dict[str, Any], campaign_id: str | None = None, paper_experiment_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        output_root = Path(config.get("output_root", "outputs/E2_paper_results_runner"))
        if not output_root.is_absolute():
            output_root = self.repo_root / output_root
        self.base_output_root = output_root.resolve()
        self.campaign_id = campaign_id
        if paper_experiment_root is not None:
            root = Path(paper_experiment_root)
            self.output_root = (root if root.is_absolute() else self.base_output_root / root).resolve()
            self.campaign_id = self.output_root.name
        elif campaign_id and str(campaign_id).startswith("Paper_experiment_"):
            self.output_root = (self.base_output_root / str(campaign_id)).resolve()
        elif campaign_id:
            self.output_root = (self.base_output_root / "campaigns" / campaign_id).resolve()
        else:
            self.output_root = self.base_output_root
        self.is_paper_experiment = self.output_root.name.startswith("Paper_experiment_")
        if self.is_paper_experiment:
            create_paper_experiment_structure(self.output_root)
        self.trial_root = self.output_root
        self.summary_root = self.output_root / "91_exports"
        self.figure_root = self.output_root / "90_paper_data" / "figures_source"
        for path in (self.output_root, self.trial_root, self.summary_root, self.figure_root):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_campaign(cls, repo_root: str | Path, config: dict[str, Any], campaign_name: str, operator_name: str = "Pranav") -> "PaperResultsWriter":
        writer = cls.create_paper_experiment(repo_root, config, experiment_number=1, label=campaign_name, operator_name=operator_name)
        return writer

    @classmethod
    def create_paper_experiment(
        cls,
        repo_root: str | Path,
        config: dict[str, Any],
        experiment_number: int | str,
        label: str = "paper_core",
        operator_name: str = "Pranav",
    ) -> "PaperResultsWriter":
        repo = Path(repo_root).resolve()
        root = Path(config.get("output_root", "outputs/E2_paper_results_runner"))
        if not root.is_absolute():
            root = repo / root
        paper_root = create_paper_experiment_root(root, experiment_number=experiment_number, label=label)
        create_paper_experiment_structure(paper_root)
        writer = cls(repo, config, paper_experiment_root=paper_root)
        writer.initialize_campaign(campaign_name=label, operator_name=operator_name, experiment_number=experiment_number)
        return writer

    @classmethod
    def list_campaigns(cls, repo_root: str | Path, config: dict[str, Any]) -> list[Path]:
        root = Path(config.get("output_root", "outputs/E2_paper_results_runner"))
        repo = Path(repo_root).resolve()
        if not root.is_absolute():
            root = repo / root
        old_campaigns = root / "campaigns"
        found = [path for path in sorted(root.glob("Paper_experiment_*"), reverse=True) if path.is_dir()]
        if old_campaigns.exists():
            found.extend(path for path in sorted(old_campaigns.glob("campaign_*"), reverse=True) if path.is_dir())
        return found

    def initialize_campaign(self, campaign_name: str, operator_name: str = "Pranav", experiment_number: int | str | None = None) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        create_paper_experiment_structure(self.output_root)
        metadata_dir = self.output_root / "00_campaign_metadata"
        write_json(
            metadata_dir / "campaign_metadata.json",
            {
                "campaign_id": self.output_root.name,
                "paper_experiment_number": experiment_number,
                "paper_experiment_label": campaign_name,
                "operator_name": operator_name,
                "created_at": now_iso(),
                "output_root": str(self.output_root),
            },
        )
        write_json(metadata_dir / "target_counts.json", {"target_counts_initialized": True})
        write_json(metadata_dir / "active_config_snapshot.json", {"output_root": str(self.output_root)})
        write_csv_rows(self.summary_root / "E2_master_trials.csv", [], fieldnames=BASE_TRIAL_FIELDS)
        for csv_name in GROUP_CSV.values():
            write_csv_rows(self.summary_root / csv_name, [], fieldnames=BASE_TRIAL_FIELDS)
        paper_tables = self.output_root / "90_paper_data" / "tables_source"
        for table_name in SUMMARY_TABLE_CSV.values():
            write_csv_rows(paper_tables / table_name, [], fieldnames=BASE_TRIAL_FIELDS)
        write_csv_rows(paper_tables / "table_stagewise_success.csv", [], fieldnames=["experiment_group", "stage", "success_count", "total_count"])
        write_csv_rows(paper_tables / "table_upv_readings.csv", [], fieldnames=["trial_id", "time_of_flight", "path_length_mm", "velocity_m_per_s"])

    def trial_dir_for(self, metadata: PaperTrialMetadata) -> Path:
        if self.is_paper_experiment:
            return create_trial_run_folder(self.output_root, metadata.experiment_group, metadata)
        stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_").split("-")[0]
        material = slug_text(metadata.material_query)
        specimen = slug_text(metadata.specimen_id)
        name = f"{metadata.experiment_group}_{stamp}_{material}_{specimen}_{int(metadata.trial_index):03d}"
        candidate = self.trial_root / name
        suffix = 2
        while candidate.exists():
            candidate = self.trial_root / f"{name}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def create_trial(self, metadata: PaperTrialMetadata, config_snapshot: dict[str, Any] | None = None) -> Path:
        trial_dir = self.trial_dir_for(metadata)
        for sub in ("figures", "data", "backend_outputs", "logs"):
            (trial_dir / sub).mkdir(parents=True, exist_ok=True)
        write_json(trial_dir / "run_metadata.json", metadata.to_dict())
        write_json(trial_dir / "trial_metadata.json", metadata.to_dict())
        initial_result = {"success": False, "failure_stage": "not_run", "failure_reason": "trial_created_not_run", "metrics": {}}
        write_json(trial_dir / "run_result.json", initial_result)
        write_json(trial_dir / "trial_result.json", initial_result)
        write_json(trial_dir / "manual_ground_truth.json", {})
        if config_snapshot is not None:
            write_json(trial_dir / "data" / "config_snapshot.json", config_snapshot)
        return trial_dir

    def save_manual_ground_truth(self, trial_dir: Path, manual: dict[str, Any]) -> None:
        write_json(trial_dir / "manual_ground_truth.json", manual)

    def write_trial_result(self, trial_dir: Path, result: dict[str, Any]) -> None:
        write_json(trial_dir / "run_result.json", result)
        write_json(trial_dir / "trial_result.json", result)
        write_json(trial_dir / "data" / "raw_backend_summary.json", result)

    def save_completed_trial(self, trial_dir: Path, manual: dict[str, Any] | None = None, result: dict[str, Any] | None = None) -> None:
        if manual is not None:
            write_json(trial_dir / "manual_ground_truth.json", manual)
        if result is not None:
            write_json(trial_dir / "trial_result.json", result)
        result = load_json_if_exists(trial_dir / "trial_result.json") or {"metrics": {}}
        result["trial_saved"] = True
        result["saved_at"] = now_iso()
        write_json(trial_dir / "run_result.json", result)
        write_json(trial_dir / "trial_result.json", result)
        write_json(trial_dir / "data" / "parsed_metrics.json", result.get("metrics", {}))
        metadata = load_json_if_exists(trial_dir / "trial_metadata.json")
        metadata["_trial_dir"] = str(trial_dir)
        manual = load_json_if_exists(trial_dir / "manual_ground_truth.json")
        self.append_trial_row(metadata, result, manual)

    def save_trial_result(self, trial_dir: Path, result: dict[str, Any]) -> None:
        self.write_trial_result(trial_dir, result)

    def append_trial_row(self, metadata: dict[str, Any], result: dict[str, Any], manual: dict[str, Any] | None = None) -> None:
        if not metadata:
            return
        row = flatten_trial_row(metadata, result, manual)
        row["trial_saved"] = result.get("trial_saved", False)
        row["saved_at"] = result.get("saved_at", "NA")
        row.pop("_trial_dir", None)
        if metadata.get("_trial_dir"):
            write_json(Path(metadata["_trial_dir"]) / "data" / "csv_row_snapshot.json", row)
        append_csv(self.summary_root / "E2_master_trials.csv", row)
        group = str(metadata.get("experiment_group", ""))
        if group in GROUP_CSV:
            append_csv(self.summary_root / GROUP_CSV[group], row)
            update_paper_data_sources(self.output_root, row)

    def copy_key_artifacts(self, trial_dir: Path, paths: list[str | Path]) -> dict[str, Any]:
        copied: list[dict[str, str]] = []
        missing: list[str] = []
        for source in paths:
            src = Path(source)
            if not src.exists() or not src.is_file():
                missing.append(str(source))
                continue
            dst = trial_dir / "figures" / src.name
            try:
                shutil.copyfile(src, dst)
                copied.append({"source": str(src), "copy": str(dst)})
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{source}: {exc}")
        manifest = {"copied": copied, "missing": missing}
        write_json(trial_dir / "backend_outputs" / "artifact_manifest.json", manifest)
        return manifest

    def counts_by_group(self) -> dict[str, int]:
        counts = {group: 0 for group in GROUP_CSV}
        path = self.summary_root / "E2_master_trials.csv"
        if not path.exists():
            return counts
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                group = row.get("experiment_group")
                if group in counts and str(row.get("trial_saved", "")).lower() == "true":
                    counts[group] += 1
        return counts

    def generate_summaries(self, target_counts: dict[str, int]) -> dict[str, str]:
        counts = self.counts_by_group()
        progress_rows = []
        for group, target in target_counts.items():
            completed = counts.get(group, 0)
            progress_rows.append(
                {
                    "experiment_group": group,
                    "completed": completed,
                    "target": target,
                    "missing": max(0, int(target) - int(completed)),
                    "percent_complete": round((completed / int(target)) * 100.0, 2) if int(target) else 0,
                }
            )
        self._write_csv_rows(self.summary_root / "E2_progress_summary.csv", progress_rows)
        table_paths: dict[str, str] = {}
        for group, csv_name in GROUP_CSV.items():
            src = self.summary_root / csv_name
            dst = self.summary_root / SUMMARY_TABLE_CSV[group]
            if src.exists():
                shutil.copyfile(src, dst)
            else:
                self._write_csv_rows(dst, [])
            table_paths[group] = str(dst)
        figure_manifest = self._build_figure_manifest()
        self._write_csv_rows(self.summary_root / "figure_manifest.csv", figure_manifest)
        self._write_csv_rows(self.output_root / "90_paper_data" / "figure_manifest.csv", figure_manifest)
        table_manifest = [
            {"table_name": path.name, "relative_path": str(path.relative_to(self.output_root)), "updated_at": now_iso()}
            for path in sorted((self.output_root / "90_paper_data" / "tables_source").glob("*.csv"))
        ]
        self._write_csv_rows(self.summary_root / "table_manifest.csv", table_manifest)
        self._write_csv_rows(self.output_root / "90_paper_data" / "table_manifest.csv", table_manifest)
        report = self.summary_root / "E2_summary_report.md"
        report.write_text(self._summary_markdown(progress_rows), encoding="utf-8")
        return {"progress_summary": str(self.summary_root / "E2_progress_summary.csv"), "summary_report": str(report), **table_paths}

    def _build_figure_manifest(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for folder_name in EXPERIMENT_FOLDERS.values():
            search_root = self.output_root / folder_name
            for path in sorted(search_root.glob("*/figures/*")):
                if path.is_file():
                    rows.append(
                        {
                            "trial_id": path.parent.parent.name,
                            "experiment_group": folder_name,
                            "source_path": str(path.relative_to(self.output_root)),
                            "paper_path": "NA",
                        }
                    )
        if rows:
            return rows
        for path in sorted(self.trial_root.glob("*/figures/*")):
            if path.is_file():
                rows.append({"trial_dir": path.parent.parent.name, "figure": path.name, "path": str(path)})
        return rows

    def _summary_markdown(self, progress_rows: list[dict[str, Any]]) -> str:
        lines = ["# E2 Paper Results Summary", "", f"Generated: {now_iso()}", "", "## Progress", ""]
        for row in progress_rows:
            lines.append(f"- {row['experiment_group']}: {row['completed']} / {row['target']} ({row['percent_complete']}%)")
        return "\n".join(lines) + "\n"

    def _write_csv_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        write_csv_rows(path, rows)
