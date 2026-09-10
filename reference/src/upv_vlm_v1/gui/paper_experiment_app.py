from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "upv_vlm_v1" / "paper_experiments.json"

for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from upv_vlm_v1.experiment.paper_experiment_runner import PaperExperimentRunner  # noqa: E402
from upv_vlm_v1.experiment.paper_results_writer import PaperResultsWriter, load_json_if_exists, write_json  # noqa: E402
from upv_vlm_v1.experiment.paper_trial_schema import (  # noqa: E402
    EXPERIMENT_GROUPS,
    PaperTrialMetadata,
    default_manual_ground_truth,
    now_iso,
    slug_text,
)
from upv_vlm_v1.experiment.progress_tracker import progress_to_dict, read_progress  # noqa: E402
from upv_vlm_v1.experiment.t4_runner_adapter import T4RunnerAdapter  # noqa: E402


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_velocity(tof_value: Any, units: str, path_length_mm: Any) -> float | str:
    try:
        tof = float(tof_value)
        length_mm = float(path_length_mm)
    except Exception:  # noqa: BLE001
        return "NA"
    if tof <= 0 or length_mm <= 0:
        return "NA"
    tof_s = tof * 1e-6 if units == "microseconds" else tof
    return (length_mm / 1000.0) / tof_s


def ss_get(st: Any, key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def init_session_state(config: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore

    defaults = {
        "operator_name": config.get("default_operator", "Pranav"),
        "campaign_name": "paper_core_v1",
        "paper_experiment_number": 1,
        "paper_experiment_label": "paper_core",
        "active_campaign_dir": "",
        "active_campaign_id": "",
        "active_trial_dir": "",
        "active_trial_id": "",
        "active_trial_identity": {},
        "latest_trial_dir": "",
        "latest_output_dir": "",
        "latest_backend_result": {},
        "latest_action_result": {},
        "status": "READY",
        "current_stage": "ready",
        "is_running": False,
        "run_started_at": "",
        "event_log": [],
        "experiment_group": "target_selection",
        "material_query_mode": "preset",
        "material_query": "brick",
        "material_query_custom": "",
        "custom_material_text": "",
        "specimen_id": "brick_01",
        "scene_id": "scene_01",
        "trial_index": 1,
        "axis_mode": "major",
        "edge_condition": "clean",
        "orientation_case": "random",
        "objects_present": "",
        "notes": "",
        "manual_ground_truth": {},
        "tof_value": "",
        "tof_units": "microseconds",
        "path_length_source": "recommended_upv_path_length",
        "manual_path_length_mm": "",
        "reading_quality": "NA",
        "last_progress": {},
        "hardware_send_jog_stop": False,
    }
    for key, value in defaults.items():
        ss_get(st, key, value)


def base_output_root(config: dict[str, Any]) -> Path:
    root = Path(config.get("output_root", "outputs/E2_paper_results_runner"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


def writer_for_campaign(config: dict[str, Any], campaign_id: str | None) -> PaperResultsWriter | None:
    if not campaign_id:
        return None
    return PaperResultsWriter(REPO_ROOT, config, campaign_id=campaign_id)


def runner_for_campaign(config_path: Path, campaign_id: str | None) -> PaperExperimentRunner | None:
    if not campaign_id:
        return None
    return PaperExperimentRunner(REPO_ROOT, config_path, campaign_id=campaign_id)


def progress_counts(writer: PaperResultsWriter | None, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    completed = writer.counts_by_group() if writer else {group: 0 for group in EXPERIMENT_GROUPS}
    targets = config.get("target_counts", {})
    return {
        group: {
            "completed": int(completed.get(group, 0)),
            "target": int(targets.get(group, 0)),
            "minimum": int(config.get("minimum_end_to_end_runs", 12)) if group == "end_to_end_upv" else int(targets.get(group, 0)),
        }
        for group in EXPERIMENT_GROUPS
    }


def material_from_state(st: Any) -> str:
    return st.session_state.material_query_custom.strip() if st.session_state.material_query == "custom" else st.session_state.material_query


def make_metadata_from_state(st: Any, config: dict[str, Any]) -> PaperTrialMetadata:
    return PaperTrialMetadata(
        trial_id="",
        experiment_group=st.session_state.get("experiment_group", "target_selection"),
        material_query=material_from_state(st),
        specimen_id=st.session_state.get("specimen_id", "brick_01"),
        scene_id=st.session_state.get("scene_id", "scene_01"),
        trial_index=int(st.session_state.get("trial_index", 1)),
        axis_mode=st.session_state.get("axis_mode", "major"),
        edge_condition=st.session_state.get("edge_condition", "clean"),
        orientation_case=st.session_state.get("orientation_case", "random"),
        objects_present=st.session_state.get("objects_present", ""),
        notes=st.session_state.get("notes", ""),
        operator_name=st.session_state.get("operator_name", config.get("default_operator", "Pranav")) or config.get("default_operator", "Pranav"),
    )


def trial_identity(campaign_id: str | None, metadata: PaperTrialMetadata) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id or "NA",
        "experiment_group": metadata.experiment_group,
        "material_query": metadata.material_query,
        "specimen_id": metadata.specimen_id,
        "scene_id": metadata.scene_id,
        "trial_index": int(metadata.trial_index),
        "axis_mode": metadata.axis_mode,
        "edge_condition": metadata.edge_condition,
        "orientation_case": metadata.orientation_case,
    }


def required_checks(group: str) -> list[str]:
    if group == "target_selection":
        return ["Correct material query entered", "Correct specimen/scene ID entered", "Objects present field is filled", "Camera frame or saved input is ready"]
    if group == "geometry_axis":
        return ["Correct material query entered", "Correct axis mode selected", "Object is visible and not heavily occluded", "Camera frame or saved input is ready"]
    if group == "anchor_selection":
        return ["Correct material query entered", "Correct edge condition selected", "Object edges/contact regions are visible", "Camera frame or saved input is ready"]
    if group == "path_length":
        return ["Correct axis mode selected", "Manual measurement is ready if available", "Depth frame or saved RGB-D input is available", "Object is visible"]
    if group == "contact_ablation":
        return ["Saved anchor-selection data is available", "Ablation condition is selected", "Replay mode will not move hardware"]
    return [
        "Workspace is clear",
        "Robot path is clear",
        "Clamp is open or safe to move",
        "RealSense is running",
        "Arduino clamp controller is connected",
        "UPV device is ready",
        "Emergency stop is accessible",
        "I understand this will move the robot and clamp",
    ]


def run_button_label(group: str) -> str:
    return {
        "target_selection": "RUN TARGET-SELECTION TRIAL",
        "geometry_axis": "RUN GEOMETRY / AXIS TRIAL",
        "anchor_selection": "RUN ANCHOR-SELECTION TRIAL",
        "path_length": "RUN PATH-LENGTH TRIAL",
        "contact_ablation": "RUN CONTACT ABLATION TRIAL",
        "end_to_end_upv": "RUN FULL ROBOTIC UPV TRIAL",
    }[group]


def run_selected_trial(runner: PaperExperimentRunner, trial_dir: Path, metadata: PaperTrialMetadata, dry_run: bool = False) -> dict[str, Any]:
    data = metadata.to_dict()
    if metadata.experiment_group == "target_selection":
        return runner.run_target_selection_trial(trial_dir, data, dry_run=dry_run)
    if metadata.experiment_group == "geometry_axis":
        return runner.run_geometry_axis_trial(trial_dir, data, dry_run=dry_run)
    if metadata.experiment_group == "anchor_selection":
        return runner.run_anchor_selection_trial(trial_dir, data, dry_run=dry_run)
    if metadata.experiment_group == "path_length":
        return runner.run_path_length_trial(trial_dir, data, dry_run=dry_run)
    if metadata.experiment_group == "contact_ablation":
        return runner.run_contact_ablation_trial(trial_dir, data, dry_run=dry_run)
    return runner.run_end_to_end_upv_trial(trial_dir, data, dry_run=dry_run)


def save_minimal_gui_validation_report(output_root: Path, reason: str = "Browser/PDF export not available in CLI validation.") -> Path:
    report_dir = output_root / "gui_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "E2_minimal_gui_validation_report.md"
    report.write_text(
        "\n".join(
            [
                "# E2 Minimal GUI Validation Report",
                "",
                f"Generated: {now_iso()}",
                f"PDF/screenshot status: {reason}",
                "",
                "## Checks",
                "",
                "- no tabs are used",
                "- new campaign starts all counters at zero",
                "- latest output is empty until CREATE TRIAL",
                "- CREATE TRIAL folder matches current experiment group",
                "- run button is large and disabled until checklist complete",
                "- manual ground truth section is visible",
                "- results preview section is visible",
                "- export summary section is visible",
                "- debug is hidden in an expander",
                "- no raw JSON is shown by default",
                "- active campaign is shown",
                "- progress counts come only from active campaign CSVs",
                "- active Paper_experiment folder owns all counters",
                "- six numbered experiment folders are created",
                "- 90_paper_data and 91_exports are present",
                "- session_state is initialized before metadata reads",
                "- no AttributeError for operator_name",
                "- hardware controls are present",
                "- HOME UR3E is present",
                "- ZERO / MARK CLAMP OPEN is present",
                "- OPEN CLAMP FULL is present",
                "- EMERGENCY CLAMP RELEASE is present",
                "- ABORT / JOG STOP is present",
                "- hardware controls do not increment paper trial counts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def hardware_log_root(config: dict[str, Any], campaign_id: str | None) -> Path:
    writer = writer_for_campaign(config, campaign_id) if campaign_id else None
    root = writer.output_root if writer else base_output_root(config)
    path = root / "92_logs" / "hardware_control_logs" if writer else root / "92_logs" / "hardware_control_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_hardware_control(config: dict[str, Any], action: str, campaign_id: str | None = None) -> dict[str, Any]:
    adapter = T4RunnerAdapter(REPO_ROOT, config.get("t4_config_path", "configs/T4_full_robot_clamp_test.json"), default_prompt_config=config.get("canonical_prompt_config"))
    confirmations = {
        "status": "",
        "home": "HOME_ROBOT",
        "arduino_check": "ARDUINO_CHECK",
        "clamp_zero_open": "CLAMP_ZERO_OPEN",
        "clamp_open_full": "CLAMP_OPEN_FULL",
        "clamp_release": "CLAMP_RELEASE",
        "emergency_release": "EMERGENCY_RELEASE",
        "jog_stop": "JOG_STOP",
    }
    log_root = hardware_log_root(config, campaign_id)
    result = adapter.run_oneshot(action, confirmation=confirmations.get(action, ""), e1_run_dir=log_root)
    stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_").split("-")[0]
    write_json(log_root / f"{action}_{stamp}.json", result)
    return result


def create_campaign(config: dict[str, Any], campaign_name: str, operator_name: str) -> PaperResultsWriter:
    return PaperResultsWriter.create_campaign(REPO_ROOT, config, campaign_name=campaign_name, operator_name=operator_name)


def create_paper_experiment(config: dict[str, Any], experiment_number: int | str, label: str, operator_name: str) -> PaperResultsWriter:
    return PaperResultsWriter.create_paper_experiment(
        REPO_ROOT,
        config,
        experiment_number=experiment_number,
        label=label,
        operator_name=operator_name,
    )


def run_dry(config_path: Path, config: dict[str, Any]) -> Path:
    writer = create_paper_experiment(config, 1, "paper_core_v1_dry", config.get("default_operator", "Pranav"))
    campaign_id = writer.campaign_id or writer.output_root.name
    runner = PaperExperimentRunner(REPO_ROOT, config_path, campaign_id=campaign_id)
    initial_counts = writer.counts_by_group()
    expected_counts = dict(initial_counts)
    created: list[str] = []
    for idx, group in enumerate(EXPERIMENT_GROUPS, start=1):
        metadata = PaperTrialMetadata(
            trial_id=f"dry_{group}_{idx}",
            experiment_group=group,
            material_query="brick",
            specimen_id=f"dry_{group}_01",
            scene_id="dry_scene",
            trial_index=idx,
            axis_mode="major",
            edge_condition="clean",
            orientation_case="0",
            objects_present="brick",
            notes="E2 dry-run sample trial. No hardware used.",
            operator_name=config.get("default_operator", "Pranav"),
        )
        trial_dir = runner.create_trial(metadata)
        created.append(str(trial_dir))
        assert writer.counts_by_group() == expected_counts
        run_selected_trial(runner, trial_dir, metadata, dry_run=True)
        assert writer.counts_by_group() == expected_counts
        manual = default_manual_ground_truth(group)
        runner.save_completed_trial(trial_dir, manual)
        expected_counts[group] += 1
    summaries = runner.generate_summaries()
    report = save_minimal_gui_validation_report(writer.base_output_root)
    write_json(
        writer.output_root / "99_debug" / "dry_runs" / "dry_run_validation.json",
        {
            "timestamp": now_iso(),
            "campaign_id": campaign_id,
            "created_trials": created,
            "summaries": summaries,
            "gui_validation_report": str(report),
            "new_campaign_initial_counts": initial_counts,
            "final_counts": writer.counts_by_group(),
            "hardware_used": False,
            "camera_opened": False,
            "rtde_connected": False,
            "arduino_connected": False,
            "clamp_moved": False,
            "rosbag_started": False,
            "real_vlm_called": False,
        },
    )
    print(f"E2 dry-run OK. Campaign folder: {writer.output_root}")
    print(f"GUI validation report: {report}")
    print("No camera, RTDE, robot, Arduino, clamp, rosbag, or real VLM was used.")
    return writer.output_root


def render_progress(progress_payload: dict[str, Any]) -> None:
    percent = int(progress_payload.get("percent_complete", 0) or 0)
    st = sys.modules.get("streamlit")
    if st is None:
        return
    st.progress(percent / 100.0, text=f"{percent}%")
    st.write("Current stage:", progress_payload.get("current_stage", "NA"))
    rows = [
        {"stage": stage.get("label"), "status": stage.get("status"), "percent": stage.get("percent"), "failure": stage.get("failure_reason") or ""}
        for stage in progress_payload.get("stages", [])
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def launch_app(config_path: Path, config: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="E2 Paper Results Runner", layout="wide")
    init_session_state(config)

    st.title("E2 Paper Results Runner")
    st.caption("Minimal final data-collection GUI for UPV_VLM_v1 paper")

    campaign_id = st.session_state.get("active_campaign_id", "") or None
    writer = writer_for_campaign(config, campaign_id)
    runner = runner_for_campaign(config_path, campaign_id)

    st.subheader("Campaign + Progress")
    campaign_cols = st.columns([0.18, 0.22, 0.24, 0.2, 0.16])
    campaign_cols[0].markdown(f"**Status:** `{st.session_state.status}`")
    campaign_cols[1].markdown(f"**Active paper experiment:** `{campaign_id or 'None'}`")
    campaign_cols[2].markdown(f"**Active trial folder:** `{st.session_state.get('latest_trial_dir', '') or 'None'}`")
    campaign_cols[3].markdown(f"**Experiment:** `{st.session_state.get('experiment_group', 'target_selection')}`")
    campaign_cols[4].markdown(f"**Stage:** `{st.session_state.current_stage}`")

    c1, c2, c3, c4 = st.columns([0.18, 0.24, 0.22, 0.36])
    with c1:
        experiment_number = st.number_input("Paper experiment number", min_value=1, step=1, key="paper_experiment_number")
    with c2:
        experiment_label = st.text_input("Paper experiment label", key="paper_experiment_label")
    with c3:
        operator_name = st.text_input("Operator", key="operator_name")
    with c4:
        existing = PaperResultsWriter.list_campaigns(REPO_ROOT, config)
        choices = [""] + [path.name for path in existing]
        selected_campaign = st.selectbox("Load existing paper experiment", choices, format_func=lambda x: "Select paper experiment" if not x else x)
    b1, b2 = st.columns(2)
    if b1.button("CREATE NEW PAPER EXPERIMENT", use_container_width=True):
        new_writer = create_paper_experiment(config, experiment_number, experiment_label, operator_name)
        st.session_state.active_campaign_id = new_writer.campaign_id or new_writer.output_root.name
        st.session_state.active_campaign_dir = str(new_writer.output_root)
        st.session_state.latest_trial_dir = ""
        st.session_state.active_trial_dir = ""
        st.session_state.latest_output_dir = ""
        st.session_state.active_trial_identity = {}
        st.session_state.current_stage = "campaign_created"
        st.session_state.status = "READY"
        st.rerun()
    if b2.button("LOAD EXISTING PAPER EXPERIMENT", use_container_width=True, disabled=not bool(selected_campaign)):
        st.session_state.active_campaign_id = selected_campaign
        loaded_writer = writer_for_campaign(config, selected_campaign)
        st.session_state.active_campaign_dir = str(loaded_writer.output_root) if loaded_writer else ""
        st.session_state.latest_trial_dir = ""
        st.session_state.active_trial_dir = ""
        st.session_state.latest_output_dir = ""
        st.session_state.active_trial_identity = {}
        st.session_state.current_stage = "campaign_loaded"
        st.session_state.status = "READY"
        st.rerun()
    st.caption(f"Active paper experiment folder: {st.session_state.get('active_campaign_dir', '') or 'None'}")

    writer = writer_for_campaign(config, st.session_state.active_campaign_id or None)
    runner = runner_for_campaign(config_path, st.session_state.active_campaign_id or None)
    counts = progress_counts(writer, config)
    labels = {
        "target_selection": "Target Selection",
        "geometry_axis": "Geometry / Axis",
        "anchor_selection": "Anchor Selection",
        "path_length": "Path-Length",
        "contact_ablation": "Contact Ablation",
        "end_to_end_upv": "End-to-End Robotic UPV",
    }
    bar_cols = st.columns(len(EXPERIMENT_GROUPS))
    for idx, group in enumerate(EXPERIMENT_GROUPS):
        item = counts[group]
        denom = max(item["target"], item["minimum"], 1)
        display_target = f"{item['minimum']} min / {item['target']} target" if group == "end_to_end_upv" else str(item["target"])
        bar_cols[idx].progress(min(item["completed"] / denom, 1.0), text=f"{labels[group]}: {item['completed']} / {display_target}")
    total_done = sum(v["completed"] for v in counts.values())
    min_total = sum(v["minimum"] for v in counts.values())
    target_total = sum(v["target"] for v in counts.values())
    st.progress(min(total_done / max(min_total, 1), 1.0), text=f"Total completed: {total_done} / {min_total} minimum ({target_total} target)")

    st.subheader("Trial Setup")
    left, right = st.columns(2)
    with left:
        st.selectbox("Experiment group", EXPERIMENT_GROUPS, key="experiment_group")
        material_options = list(config.get("materials", ["brick", "timber", "concrete block"])) + ["custom"]
        st.selectbox("Material query", material_options, key="material_query")
        if st.session_state.material_query == "custom":
            st.text_input("Custom material text", value="", key="material_query_custom")
        else:
            st.session_state.material_query_custom = ""
        st.text_input("Specimen ID", value="brick_01", key="specimen_id")
        st.text_input("Scene ID", value="scene_01", key="scene_id")
    with right:
        st.number_input("Trial index", min_value=1, step=1, key="trial_index")
        st.radio("Axis mode", config.get("axis_modes", ["major", "minor"]), horizontal=True, key="axis_mode")
        st.selectbox("Edge condition", config.get("edge_conditions", ["clean"]), key="edge_condition")
        st.selectbox("Orientation case", config.get("orientation_cases", ["0"]), key="orientation_case")
        st.text_input("Objects present", value="", key="objects_present")
        st.text_area("Notes", key="notes", height=70)

    metadata = make_metadata_from_state(st, config)
    current_identity = trial_identity(st.session_state.active_campaign_id or None, metadata)
    active_identity = st.session_state.active_trial_identity or {}
    stale_trial = bool(st.session_state.latest_trial_dir and active_identity and active_identity != current_identity)
    if stale_trial:
        st.warning("Current inputs changed. Create a new trial before running.")
    t1, t2 = st.columns(2)
    if t1.button("CREATE TRIAL", use_container_width=True, disabled=not bool(st.session_state.active_campaign_id)):
        if runner is None:
            st.error("Create or load a campaign first.")
        else:
            trial_dir = runner.create_trial(metadata)
            st.session_state.latest_trial_dir = str(trial_dir)
            st.session_state.active_trial_dir = str(trial_dir)
            st.session_state.latest_output_dir = str(trial_dir)
            st.session_state.active_trial_identity = current_identity
            st.session_state.current_stage = "trial_created"
            st.session_state.status = "READY"
            st.rerun()
    if t2.button("CLEAR CURRENT TRIAL", use_container_width=True):
        st.session_state.latest_trial_dir = ""
        st.session_state.active_trial_dir = ""
        st.session_state.latest_output_dir = ""
        st.session_state.active_trial_id = ""
        st.session_state.active_trial_identity = {}
        st.session_state.current_stage = "ready"
        st.rerun()
    if st.session_state.latest_trial_dir:
        st.success(f"Active trial: {Path(st.session_state.latest_trial_dir).name}")
        st.caption(f"Trial folder: {st.session_state.latest_trial_dir}")

    st.subheader("Hardware Controls")
    st.caption("These utility actions never increment paper trial counts. Dangerous actions use checklist gates; E2 passes T4 confirmations internally.")
    hw_status_cols = st.columns([0.16, 0.16, 0.16, 0.16, 0.18, 0.18])
    if hw_status_cols[0].button("ROBOT STATUS", use_container_width=True):
        with st.spinner("Reading robot status through T4 no-hardware/status path..."):
            result = run_hardware_control(config, "status", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "robot_status"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
        st.success("Robot status action complete." if result.get("success") else "Robot status action failed.")
    arduino_ready = st.checkbox("Arduino USB is connected", key="hw_arduino_ready")
    if hw_status_cols[1].button("ARDUINO CHECK", use_container_width=True, disabled=not arduino_ready):
        with st.spinner("Running Arduino check..."):
            result = run_hardware_control(config, "arduino_check", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "arduino_check"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    home_ok = all(
        [
            st.checkbox("Clamp is released/open", key="hw_home_clamp_open"),
            st.checkbox("Robot path to home is clear", key="hw_home_path_clear"),
            st.checkbox("Emergency stop is accessible", key="hw_home_estop"),
            st.checkbox("I understand HOME UR3E moves the robot", key="hw_home_understand"),
        ]
    )
    if hw_status_cols[2].button("HOME UR3E", use_container_width=True, disabled=not home_ok):
        with st.spinner("Homing UR3e through T4..."):
            result = run_hardware_control(config, "home", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "home_ur3e"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    zero_ok = all(
        [
            st.checkbox("Clamp is physically fully open", key="hw_zero_phys_open"),
            st.checkbox("I understand this marks the open reference", key="hw_zero_understand"),
        ]
    )
    if hw_status_cols[3].button("ZERO / MARK CLAMP OPEN", use_container_width=True, disabled=not zero_ok):
        with st.spinner("Marking clamp open reference..."):
            result = run_hardware_control(config, "clamp_zero_open", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "clamp_zero_open"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    clamp_safe = all(
        [
            st.checkbox("Hands are clear", key="hw_clamp_hands_clear"),
            st.checkbox("Clamp path is clear", key="hw_clamp_path_clear"),
        ]
    )
    copen, crelease, emergency, abort = st.columns(4)
    if copen.button("OPEN CLAMP FULL", use_container_width=True, disabled=not clamp_safe):
        with st.spinner("Opening clamp full..."):
            result = run_hardware_control(config, "clamp_open_full", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "clamp_open_full"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    if crelease.button("CLAMP RELEASE", use_container_width=True, disabled=not clamp_safe):
        with st.spinner("Releasing clamp..."):
            result = run_hardware_control(config, "clamp_release", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "clamp_release"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    emergency_ok = all(
        [
            st.checkbox("Clamp/object area is safe", key="hw_emergency_area_safe"),
            st.checkbox("I understand emergency release moves the clamp", key="hw_emergency_understand"),
        ]
    )
    if emergency.button("EMERGENCY CLAMP RELEASE", use_container_width=True, disabled=not emergency_ok):
        with st.spinner("Running emergency clamp release..."):
            result = run_hardware_control(config, "emergency_release", st.session_state.get("active_campaign_id") or None)
        st.session_state.latest_action_result = result
        st.session_state.current_stage = "emergency_release"
        st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
    st.checkbox("Also send clamp/Arduino jog_stop if available", key="hardware_send_jog_stop")
    if abort.button("ABORT / JOG STOP", use_container_width=True):
        if st.session_state.get("hardware_send_jog_stop", False):
            with st.spinner("Sending jog_stop..."):
                result = run_hardware_control(config, "jog_stop", st.session_state.get("active_campaign_id") or None)
            st.session_state.latest_action_result = result
            st.session_state.status = "PAUSED" if result.get("success") else "FAILED"
        else:
            result = {"success": True, "action": "abort_local_only", "failure_reason": None, "note": "No jog_stop sent. Use physical E-stop if needed."}
            st.session_state.latest_action_result = result
            st.session_state.status = "PAUSED"
        st.session_state.current_stage = "abort_jog_stop"

    st.subheader("Required Checks + Run")
    checks_complete = True
    for idx, item in enumerate(required_checks(metadata.experiment_group)):
        checks_complete = st.checkbox(item, key=f"e2_check_{metadata.experiment_group}_{idx}") and checks_complete
    run_disabled = not checks_complete or not st.session_state.active_campaign_id or not st.session_state.latest_trial_dir or stale_trial
    if run_disabled:
        st.info("Run button is enabled only after campaign, trial, and checklist are complete.")
    if st.button(run_button_label(metadata.experiment_group), type="primary", use_container_width=True, disabled=run_disabled):
        try:
            assert runner is not None
            trial_dir = Path(st.session_state.latest_trial_dir)
            st.session_state.status = "RUNNING"
            st.session_state.current_stage = metadata.experiment_group
            with st.spinner(f"RUNNING {metadata.experiment_group}..."):
                result = run_selected_trial(runner, trial_dir, metadata, dry_run=False)
            st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
            st.session_state.current_stage = "backend_complete_unsaved"
            st.success("Backend run complete. Click SAVE TRIAL RESULT to count this trial.")
        except Exception as exc:  # noqa: BLE001
            st.session_state.status = "FAILED"
            st.session_state.current_stage = "exception"
            if st.session_state.latest_trial_dir:
                log_path = Path(st.session_state.latest_trial_dir) / "logs" / "error_trace.txt"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
            st.exception(exc)

    progress_data = read_progress(Path(st.session_state.latest_trial_dir) / "run_progress.json") if st.session_state.latest_trial_dir else {}
    if progress_data:
        st.subheader("Running Progress")
        render_progress(progress_to_dict(progress_data))

    st.subheader("Manual Ground Truth / UPV Entry")
    manual = default_manual_ground_truth(metadata.experiment_group)
    if metadata.experiment_group == "target_selection":
        manual["target_correct"] = st.selectbox("Final selected object correct?", ["NA", "yes", "no"])
        manual["manual_selected_class"] = st.text_input("Manual selected class", value="NA")
        manual["failure_type"] = st.selectbox("Failure type", ["none", "wrong_mask", "wrong_class", "no_detection", "ambiguous", "other", "NA"])
    elif metadata.experiment_group == "geometry_axis":
        manual["manual_major_axis_angle_deg"] = st.text_input("Manual major-axis angle deg", value="NA")
        manual["manual_centroid_u"] = st.text_input("Manual centroid u", value="NA")
        manual["manual_centroid_v"] = st.text_input("Manual centroid v", value="NA")
        manual["mask_usable"] = st.selectbox("Mask usable?", ["NA", "yes", "no"])
        manual["axis_correct"] = st.selectbox("Axis correct?", ["NA", "yes", "no"])
    elif metadata.experiment_group == "anchor_selection":
        manual["selected_anchor_label"] = st.selectbox("Selected anchor label", ["Good", "Acceptable", "Bad", "NA"])
        manual["manual_best_anchor_id"] = st.text_input("Manual best anchor ID", value="NA")
        manual["imperfection_present"] = st.selectbox("Imperfection present?", ["NA", "yes", "no"])
        manual["bad_anchor_reason"] = st.selectbox("Bad-anchor reason", ["chip", "mortar", "nail", "jagged", "depth", "clamp infeasible", "other", "NA"])
    elif metadata.experiment_group == "path_length":
        manual["manual_clamp_width_mm"] = st.text_input("Manual clamp width mm", value="NA")
        manual["manual_upv_path_length_mm"] = st.text_input("Manual UPV path length mm", value="NA")
        manual["measurement_tool"] = st.selectbox("Measurement tool", ["ruler", "caliper", "other"])
        manual["measurement_notes"] = st.text_area("Measurement notes", value="")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if metadata.experiment_group == "contact_ablation":
                manual["source_anchor_trial_id"] = st.text_input("Source anchor trial ID", value="NA")
                manual["ablation_condition"] = st.text_input("Ablation condition", value="NA")
                manual["anchor_acceptable"] = st.selectbox("Anchor acceptable after ablation?", ["NA", "yes", "no"])
                manual["manual_best_anchor_id"] = st.text_input("Manual best anchor ID", value="NA")
            else:
                manual["target_correct"] = st.selectbox("Target correct?", ["NA", "yes", "no"])
                manual["anchor_acceptable"] = st.selectbox("Anchor acceptable?", ["NA", "yes", "no"])
                manual["robot_success"] = st.selectbox("Robot success?", ["NA", "yes", "no"])
                manual["clamp_success"] = st.selectbox("Clamp success?", ["NA", "yes", "no"])
                manual["valid_upv_tof"] = st.selectbox("Valid UPV TOF?", ["NA", "yes", "no"])
                manual["returned_home"] = st.selectbox("Returned home?", ["NA", "yes", "no"])
        with c2:
            if metadata.experiment_group == "contact_ablation":
                manual["ablation_notes"] = st.text_area("Ablation notes", value="")
            else:
                manual["human_intervention_count"] = st.number_input("Human intervention count", min_value=0, step=1)
                manual["failure_stage"] = st.text_input("Failure stage", value="NA")
                manual["time_of_flight_value"] = st.text_input("Time of flight value", value="")
                manual["time_of_flight_units"] = st.selectbox("TOF units", ["microseconds", "seconds"])
                manual["path_length_source"] = st.selectbox("Path length source", ["recommended_upv_path_length", "selected_probe_spacing", "manual_path_length"])
                manual["manual_path_length_mm"] = st.text_input("Manual path length mm", value="")
                manual["reading_quality"] = st.selectbox("Reading quality", ["good", "noisy", "failed", "NA"])
                path_for_preview = manual["manual_path_length_mm"] if manual["path_length_source"] == "manual_path_length" else "NA"
                manual["computed_velocity_m_per_s"] = compute_velocity(manual["time_of_flight_value"], manual["time_of_flight_units"], path_for_preview)
                st.write("Computed velocity preview:", manual["computed_velocity_m_per_s"])
    manual["manual_notes"] = st.text_area("Manual notes", value=manual.get("manual_notes", ""))
    if st.button("SAVE TRIAL RESULT", use_container_width=True, disabled=not bool(runner and st.session_state.latest_trial_dir)):
        assert runner is not None
        runner.save_completed_trial(st.session_state.latest_trial_dir, manual)
        st.session_state.current_stage = "trial_saved"
        st.session_state.status = "COMPLETE"
        st.success("Trial saved and campaign counters updated.")
        st.rerun()

    st.subheader("Results Preview")
    if not st.session_state.latest_trial_dir:
        st.info("No artifact saved for this trial yet.")
    else:
        trial_dir = Path(st.session_state.latest_trial_dir)
        figures = sorted((trial_dir / "figures").glob("*")) if (trial_dir / "figures").exists() else []
        if not figures:
            st.info("No artifact saved for this trial yet.")
        for path in figures[:8]:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(path), caption=path.name, use_container_width=True)
        result = load_json_if_exists(trial_dir / "trial_result.json")
        if result:
            st.write("Trial status:", result.get("success", "NA"), "Saved:", result.get("trial_saved", False))
            st.write("Failure:", result.get("failure_stage", "NA"), result.get("failure_reason", "NA"))

    st.subheader("Export Summary")
    e1, e2, e3 = st.columns(3)
    if e1.button("EXPORT / REFRESH SUMMARY TABLES", use_container_width=True, disabled=runner is None):
        assert runner is not None
        st.json(runner.generate_summaries())
    if e2.button("GENERATE FIGURE MANIFEST", use_container_width=True, disabled=runner is None):
        assert runner is not None
        paths = runner.generate_summaries()
        st.write(paths.get("summary_report", "NA"))
    if e3.button("SHOW CAMPAIGN OUTPUT PATH", use_container_width=True, disabled=writer is None):
        assert writer is not None
        st.info(str(writer.output_root))

    with st.expander("Advanced / Debug", expanded=False):
        st.write("Backend E1 config path:", config.get("default_backend_config"))
        st.write("T4 config path:", config.get("t4_config_path"))
        st.write("Canonical prompt path:", config.get("canonical_prompt_config"))
        if st.session_state.latest_trial_dir:
            trial_dir = Path(st.session_state.latest_trial_dir)
            st.write("Latest backend result JSON:", str(trial_dir / "backend_outputs" / "backend_result.json"))
            st.write("Trial folder tree")
            st.text("\n".join(str(path.relative_to(trial_dir)) for path in sorted(trial_dir.rglob("*"))[:200]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2 paper results runner GUI.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to E2 paper experiments JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Create dry campaign and sample E2 trials without hardware.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.dry_run:
        run_dry(config_path, config)
        return 0
    launch_app(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
