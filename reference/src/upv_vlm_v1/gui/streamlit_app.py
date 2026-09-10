import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "E1_autonomous_upv_experiment_manager.json"

for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from upv_vlm_orient.experiment_logging.experiment_schema import (  # noqa: E402
    DEFAULT_PROMPTS,
    EXPERIMENT_CATEGORIES,
    METRIC_GROUPS,
    PIPELINE_MODES,
    RUN_MODES,
    default_manual_labels,
    default_upv_results,
    compute_upv_velocity,
    now_iso,
)
from upv_vlm_orient.experiment_logging.experiment_writer import ExperimentRunWriter, append_csv, write_json  # noqa: E402
from upv_vlm_orient.experiment_logging.media_recorder import RosbagRecorder, VideoRecorder  # noqa: E402
from upv_vlm_orient.experiment_logging.prompt_logging import disabled_stages_for_mode, write_prompt_package  # noqa: E402
from upv_vlm_orient.experiment_logging.result_flattening import (  # noqa: E402
    discover_t4_files,
    flatten_clamp_results,
    flatten_perception_geometry_results,
    flatten_robot_results,
    flatten_t4_summary,
    flatten_timing_summary,
)
from upv_vlm_v1.experiment.t4_runner_adapter import T4RunnerAdapter  # noqa: E402
from upv_vlm_v1.experiment.progress_tracker import (  # noqa: E402
    failed_stage as progress_failed_stage,
    make_progress_template,
    progress_to_dict,
    read_progress,
    update_stage,
    write_progress,
    write_progress_summary_csv,
)
from upv_vlm_v1.prompts.prompt_registry import (  # noqa: E402
    apply_ablation_mode,
    apply_prompt_overrides,
    bundle_to_jsonable,
    load_prompt_bundle,
    write_prompt_run_package,
)


class E1Error(RuntimeError):
    pass


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise E1Error(f"Missing E1 config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise E1Error(f"Invalid E1 config {path}: {exc}") from exc


def canonical_prompt_config_path(config: dict[str, Any]) -> Path:
    prompt_cfg = config.get("prompt_engineering", {})
    return resolve_repo_path(prompt_cfg.get("canonical_prompt_config", "configs/upv_vlm_v1/prompts/default_prompts.yaml"))


def default_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_name": config.get("default_operator", "Pranav"),
        "project_name": "UPV_VLM_orient_AIC_paper",
        "specimen_id": "brick_01",
        "scene_id": "scene_01",
        "trial_index": 1,
        "notes": "",
        "material_query": config.get("default_material_query", "brick"),
        "axis_mode": config.get("default_axis_mode", "major"),
        "run_mode": config.get("default_run_mode", "full_autonomous_home"),
        "experiment_category_tags": ["normal_single_object"],
        "model_selections": {
            "open_vocab_detector": "repo current detector or T4/T2 default",
            "segmentation_model": "repo current VLM/V2a mask segmentation",
            "material_verification_model": "available if configured; otherwise NA",
            "edge_contact_vlm_model": "Qwen2.5-VL local model if enabled",
        },
    }


def create_dry_run(config_path: Path, config: dict[str, Any]) -> Path:
    writer = ExperimentRunWriter(REPO_ROOT, config, config_path)
    metadata = default_metadata(config)
    metadata["run_id"] = "dry_run"
    run_dir = writer.create_run(
        metadata=metadata,
        pipeline_mode="proposed_full_system",
        selected_metrics=None,
        manual_labels=default_manual_labels(),
        upv_results=default_upv_results(),
        dry_run=True,
    )
    progress = update_stage(make_progress_template("plan_only"), "complete", "succeeded", success=True)
    write_progress(progress, run_dir / "run_progress.json")
    write_progress_summary_csv(progress, run_dir / "run_progress_summary.csv")
    write_json(
        run_dir / "logs" / "dry_run_validation.json",
        {
            "camera_opened": False,
            "rtde_connected": False,
            "robot_motion_command_sent": False,
            "arduino_connected": False,
            "clamp_command_sent": False,
            "rosbag_started": False,
            "real_vlm_called": False,
            "prompts_folder_created": (run_dir / "prompts").exists(),
            "prompt_manifest_created": (run_dir / "prompts" / "prompt_manifest.json").exists(),
            "prompt_engineering_summary_created": (run_dir / "prompt_engineering_summary.json").exists(),
        },
    )
    print(f"E1 dry-run OK. Output folder: {run_dir}")
    print("No camera, RTDE, robot, Arduino, clamp, rosbag, or real VLM was used.")
    return run_dir


def validate_config(config_path: Path, config: dict[str, Any]) -> Path:
    return create_dry_run(config_path, config)


def gui_state_dry_run(config_path: Path, config: dict[str, Any]) -> Path:
    writer = ExperimentRunWriter(REPO_ROOT, config, config_path)
    timber_meta = default_metadata(config)
    timber_meta.update(
        {
            "material_query": "timber",
            "specimen_id": "timber_01",
            "scene_id": "scene_01",
            "trial_index": 1,
            "axis_mode": "major",
            "run_mode": "full_autonomous_home",
        }
    )
    timber_identity = run_identity_from_metadata(timber_meta)
    timber_meta["run_id"] = run_identity_slug(timber_identity)
    run_dir = writer.create_run(
        metadata=timber_meta,
        pipeline_mode="proposed_full_system",
        selected_metrics=[],
        manual_labels=default_manual_labels(),
        upv_results=default_upv_results(),
        dry_run=True,
    )
    progress = update_stage(make_progress_template("full_autonomous_home"), "complete", "succeeded", success=True)
    write_progress(progress, run_dir / "run_progress.json")
    write_progress_summary_csv(progress, run_dir / "run_progress_summary.csv")
    loaded_identity = load_run_identity(run_dir)
    brick_meta = dict(timber_meta)
    brick_meta.update({"material_query": "brick", "specimen_id": "brick_01"})
    brick_identity = run_identity_from_metadata(brick_meta)
    summary = {
        "timestamp": now_iso(),
        "mode": "gui_state_dry_run",
        "created_run_dir": str(run_dir),
        "timber_identity": timber_identity,
        "loaded_identity": loaded_identity,
        "folder_name_contains_timber": "timber" in run_dir.name,
        "brick_identity": brick_identity,
        "mismatch_detected_after_metadata_change": not identities_match(brick_identity, loaded_identity),
        "hardware_used": False,
    }
    if not summary["folder_name_contains_timber"] or not summary["mismatch_detected_after_metadata_change"]:
        raise E1Error(f"GUI state dry-run failed: {summary}")
    write_json(run_dir / "logs" / "gui_state_dry_run.json", summary)
    print(f"E1 GUI-state dry-run OK. Output folder: {run_dir}")
    print(json.dumps(summary, indent=2))
    return run_dir


def load_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {}


def parse_last_json_object(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx in range(len(stdout) - 1, -1, -1):
        if stdout[idx] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not stdout[idx + end :].strip():
            return obj
    best: dict[str, Any] = {}
    for idx, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            best = obj
    return best


RUN_IDENTITY_KEYS = ["specimen_id", "material_query", "scene_id", "trial_index", "axis_mode", "run_mode"]


def slugify(text: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "na"


def run_identity_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {key: str(metadata.get(key, "NA")) for key in RUN_IDENTITY_KEYS}


def run_identity_slug(identity: dict[str, Any]) -> str:
    return f"{slugify(identity.get('material_query'))}_{slugify(identity.get('specimen_id'))}_{slugify(identity.get('trial_index'))}"


def load_run_identity(run_dir: str | Path | None) -> dict[str, str] | None:
    if not run_dir:
        return None
    meta = load_json_if_exists(Path(run_dir) / "run_metadata.json")
    if not meta:
        return None
    return run_identity_from_metadata(meta)


def identities_match(gui_identity: dict[str, Any], loaded_identity: dict[str, Any] | None) -> bool:
    if not loaded_identity:
        return False
    return all(str(gui_identity.get(key, "NA")) == str(loaded_identity.get(key, "NA")) for key in RUN_IDENTITY_KEYS)


def identity_diff(gui_identity: dict[str, Any], loaded_identity: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if not loaded_identity:
        return {}
    return {
        key: {"gui": str(gui_identity.get(key, "NA")), "loaded": str(loaded_identity.get(key, "NA"))}
        for key in RUN_IDENTITY_KEYS
        if str(gui_identity.get(key, "NA")) != str(loaded_identity.get(key, "NA"))
    }


def ensure_e1_run(
    writer: ExperimentRunWriter,
    config: dict[str, Any],
    config_path: Path,
    metadata: dict[str, Any],
    pipeline_mode: str,
    selected_metrics: list[str],
    labels: dict[str, Any],
    upv: dict[str, Any],
    prompt_overrides: dict[str, str],
    latest_run_dir: str | None,
    *,
    allow_existing_mismatch: bool = False,
) -> Path:
    gui_identity = run_identity_from_metadata(metadata)
    if latest_run_dir:
        run_dir = Path(latest_run_dir)
        loaded_identity = load_run_identity(run_dir)
        if loaded_identity and identities_match(gui_identity, loaded_identity):
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        if allow_existing_mismatch:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "logs" / "identity_mismatch_continue_warning.json",
                {
                    "timestamp": now_iso(),
                    "warning": "Operator intentionally continued an existing run folder whose metadata differs from current GUI fields.",
                    "gui_identity": gui_identity,
                    "loaded_identity": loaded_identity,
                    "diff": identity_diff(gui_identity, loaded_identity),
                },
            )
            return run_dir
    metadata = dict(metadata)
    metadata["run_id"] = run_identity_slug(gui_identity)
    metadata["run_identity"] = gui_identity
    return writer.create_run(
        metadata=metadata,
        pipeline_mode=pipeline_mode,
        selected_metrics=selected_metrics,
        manual_labels=labels,
        upv_results=upv,
        prompt_overrides=prompt_overrides,
        dry_run=False,
    )


def save_current_gui_state(run_dir: Path, config: dict[str, Any], metadata: dict[str, Any], pipeline_mode: str, selected_metrics: list[str], labels: dict[str, Any], upv: dict[str, Any], prompt_overrides: dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_meta = load_json_if_exists(run_dir / "run_metadata.json")
    existing_meta.update({"timestamp_last_saved": now_iso(), **metadata, "pipeline_mode": pipeline_mode})
    write_json(run_dir / "run_metadata.json", existing_meta)
    write_json(run_dir / "selected_metrics.json", {"selected_metrics": selected_metrics})
    write_json(run_dir / "manual_labels.json", labels)
    write_json(run_dir / "upv_results.json", upv)
    package = write_effective_prompt_package(run_dir, config, metadata, pipeline_mode, prompt_overrides)
    history_path = run_dir / "prompts" / "prompt_edit_history.json"
    history = load_json_if_exists(history_path).get("events", [])
    history.append({"timestamp": now_iso(), "event": "save_current_gui_state", "edited_prompts_used": package["summary"].get("edited_prompts_used", False)})
    write_json(history_path, {"events": history})


def write_effective_prompt_package(run_dir: Path, config: dict[str, Any], metadata: dict[str, Any], pipeline_mode: str, prompt_overrides: dict[str, str]) -> dict[str, Any]:
    prompt_cfg = config.get("prompt_engineering", {})
    source_path = canonical_prompt_config_path(config)
    bundle = load_prompt_bundle(
        source_path,
        prompt_set_name=str(prompt_cfg.get("default_prompt_set_name", "default_v1")),
        prompt_set_version=str(prompt_cfg.get("default_prompt_set_version", "v1")),
    )
    bundle = apply_prompt_overrides(bundle, prompt_overrides)
    bundle = apply_ablation_mode(bundle, pipeline_mode)
    run_dir.joinpath("configs").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "configs" / "e1_effective_prompt_bundle.json", bundle_to_jsonable(bundle))
    write_json(run_dir / "configs" / "e1_prompt_overrides.json", prompt_overrides)
    package = write_prompt_run_package(
        run_dir,
        bundle,
        material_query=str(metadata.get("material_query", config.get("default_material_query", "brick"))),
        axis_mode=str(metadata.get("axis_mode", config.get("default_axis_mode", "major"))),
        pipeline_mode=pipeline_mode,
        operator_name=str(metadata.get("operator_name", config.get("default_operator", "Pranav"))),
    )
    package["summary"]["edited_prompts_used"] = bool(prompt_overrides)
    package["summary"]["canonical_prompt_config"] = str(source_path)
    write_json(run_dir / "prompt_engineering_summary.json", package["summary"])
    return package


def start_requested_recorders(run_dir: Path, config: dict[str, Any], recording: dict[str, Any], dry_run: bool = False) -> tuple[VideoRecorder | None, RosbagRecorder | None]:
    video: VideoRecorder | None = None
    rosbag: RosbagRecorder | None = None
    if recording.get("save_video"):
        video = VideoRecorder(run_dir / "video" / "video.mp4", fps=float(recording.get("video_fps", config.get("video_fps", 10))), dry_run=True)
        video.start()
        write_json(
            run_dir / "video" / "video_recording.json",
            {
                "video_recording_requested": True,
                "video_recording_started": False,
                "reason": "live_video_capture_not_wired",
                "events": video.events,
            },
        )
    rosbag_cfg = config.get("rosbag", {})
    if recording.get("save_rosbag") or rosbag_cfg.get("enabled"):
        rosbag = RosbagRecorder(
            output_dir=run_dir / "rosbag" / "e1_bag",
            topics=list(rosbag_cfg.get("topics", [])),
            max_duration_s=int(rosbag_cfg.get("max_duration_s", 180)),
            dry_run=dry_run,
        )
        try:
            rosbag.start()
            write_json(run_dir / "rosbag" / "rosbag_recording.json", {"started": not dry_run, "topics": rosbag.topics, "events": rosbag.events, "failure_reason": None})
        except Exception as exc:  # noqa: BLE001
            write_json(run_dir / "rosbag" / "rosbag_recording.json", {"started": False, "topics": rosbag.topics, "events": rosbag.events, "failure_reason": str(exc)})
    return video, rosbag


def stop_recorders(run_dir: Path, video: VideoRecorder | None, rosbag: RosbagRecorder | None) -> None:
    if video is not None:
        video.stop()
        write_json(run_dir / "video" / "video_recording.json", {"video_recording_requested": True, "video_recording_started": False, "reason": "live_video_capture_not_wired", "events": video.events})
    if rosbag is not None:
        rosbag.stop()
        write_json(run_dir / "rosbag" / "rosbag_recording.json", {"started": not rosbag.dry_run, "topics": rosbag.topics, "events": rosbag.events, "stop_status": "stopped"})


def run_t4_oneshot(
    config: dict[str, Any],
    action: str,
    confirmation: str,
    e1_run_dir: Path,
    extra_args: list[str] | None = None,
    progress_file: Path | None = None,
    live_progress_callback: Any | None = None,
) -> dict[str, Any]:
    stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_").split("-")[0]
    log_path = e1_run_dir / "logs" / f"t4_oneshot_{action}_{stamp}.log"
    out_parent = e1_run_dir / "t4_linked_outputs"
    adapter = T4RunnerAdapter(
        REPO_ROOT,
        resolve_repo_path(config["t4_config_path"]),
        python_executable=sys.executable,
        default_prompt_config=canonical_prompt_config_path(config),
    )
    prompt_config = e1_run_dir / "configs" / "e1_effective_prompt_bundle.json"
    prompt_overrides = e1_run_dir / "configs" / "e1_prompt_overrides.json"
    result = adapter.run_oneshot(
        action,
        confirmation=confirmation,
        e1_run_dir=e1_run_dir,
        prompt_config=prompt_config if prompt_config.exists() else canonical_prompt_config_path(config),
        prompt_overrides_json=prompt_overrides if prompt_overrides.exists() else None,
        extra_args=extra_args,
        progress_file=progress_file,
        live_progress_callback=live_progress_callback,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n" + " ".join(str(part) for part in result.get("command", [])) + "\n\nSTDOUT:\n" + str(result.get("stdout", "")) + "\n\nSTDERR:\n" + str(result.get("stderr", "")) + "\n",
        encoding="utf-8",
    )
    parsed = dict(result)
    parsed["log_path"] = str(log_path)
    summary_path = out_parent / f"t4_oneshot_{action}_{stamp}.json"
    write_json(summary_path, parsed)
    collect_t4_outputs_to_e1(parsed, e1_run_dir)
    update_e1_results_from_t4(parsed, e1_run_dir)
    return parsed


def collect_t4_outputs_to_e1(t4_summary: dict[str, Any], e1_run_dir: Path) -> dict[str, Any]:
    manifest = discover_t4_files(t4_summary.get("t4_session_dir"))
    linked = e1_run_dir / "t4_linked_outputs"
    copied_dir = linked / "copied"
    copied_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    expected = [
        manifest.get("latest_full_run_json"),
        manifest.get("latest_run_record_json"),
        manifest.get("latest_clamp_plan_json"),
        manifest.get("latest_summary_csv"),
    ]
    for source in expected:
        if not source or source == "NA":
            continue
        src = Path(str(source))
        if src.exists():
            dst = copied_dir / src.name
            shutil.copyfile(src, dst)
            copied.append({"source": str(src), "copy": str(dst)})
    figure_names = [
        "T1_plan_overlay.png",
        "T2_plan_overlay.png",
        "T3_width_overlay.png",
        "T4_plan_overlay.png",
        "T4_width_overlay.png",
        "selected_mask_overlay.png",
        "color.png",
        "depth_visualization.png",
    ]
    latest_overlays = manifest.get("latest_overlay_paths") or []
    for source in latest_overlays:
        src = Path(source)
        if not src.exists():
            continue
        if src.name in figure_names or "overlay" in src.name.lower() or "width" in src.name.lower() or "plan" in src.name.lower():
            dst = e1_run_dir / "figures" / src.name
            try:
                shutil.copyfile(src, dst)
                copied.append({"source": str(src), "copy": str(dst)})
            except Exception:  # noqa: BLE001
                pass
    t4_prompt_manifest = t4_summary.get("prompt_manifest_path")
    t4_prompt_summary = t4_summary.get("prompt_engineering_summary_path")
    prompt_copy_dir = copied_dir / "prompts"
    prompt_copy_dir.mkdir(parents=True, exist_ok=True)
    for source in [t4_prompt_manifest, t4_prompt_summary]:
        if source and Path(str(source)).exists():
            src = Path(str(source))
            dst = prompt_copy_dir / f"t4_{src.name}"
            shutil.copyfile(src, dst)
            copied.append({"source": str(src), "copy": str(dst)})
    t4_prompt_dir = Path(str(t4_prompt_manifest)).parent if t4_prompt_manifest else None
    if t4_prompt_dir and t4_prompt_dir.exists():
        for src in sorted(t4_prompt_dir.glob("*.txt")) + sorted(t4_prompt_dir.glob("vlm_*.json")):
            dst = prompt_copy_dir / f"t4_{src.name}"
            shutil.copyfile(src, dst)
            copied.append({"source": str(src), "copy": str(dst)})
    manifest["copied_files"] = copied
    write_json(linked / "t4_output_manifest.json", manifest)
    write_actual_prompt_source(e1_run_dir, t4_summary)
    return manifest


def write_actual_prompt_source(e1_run_dir: Path, t4_summary: dict[str, Any]) -> dict[str, Any]:
    gui_summary = load_json_if_exists(e1_run_dir / "prompt_engineering_summary.json")
    t4_prompt_summary = load_json_if_exists(Path(str(t4_summary.get("prompt_engineering_summary_path", ""))))
    gui_hashes = {
        name: stage.get("prompt_hash")
        for name, stage in (gui_summary.get("stages") or {}).items()
        if isinstance(stage, dict)
    }
    t4_hashes = {
        name: stage.get("prompt_hash")
        for name, stage in (t4_prompt_summary.get("stages") or {}).items()
        if isinstance(stage, dict)
    }
    match: bool | str
    if not gui_hashes or not t4_hashes:
        match = "NA"
    else:
        match = gui_hashes == t4_hashes
    payload = {
        "gui_prompt_config": str(e1_run_dir / "configs" / "e1_effective_prompt_bundle.json"),
        "gui_prompt_hashes": gui_hashes,
        "t4_prompt_manifest": t4_summary.get("prompt_manifest_path", "NA"),
        "t4_prompt_hashes": t4_hashes,
        "gui_t4_prompt_hash_match": match,
        "notes": "T4 currently logs supplied prompts. Full prompt injection depends on future perception hooks.",
    }
    write_json(e1_run_dir / "actual_prompt_source.json", payload)
    return payload


def progress_fields_for_run(e1_run_dir: Path) -> dict[str, Any]:
    progress_path = e1_run_dir / "run_progress.json"
    progress = read_progress(progress_path)
    if not progress:
        return {
            "progress_status": "NA",
            "final_percent_complete": "NA",
            "final_stage": "NA",
            "failed_stage": "NA",
            "progress_file": str(progress_path),
        }
    data = progress_to_dict(progress)
    write_progress_summary_csv(progress, e1_run_dir / "run_progress_summary.csv")
    return {
        "progress_status": data.get("status", "NA"),
        "final_percent_complete": data.get("percent_complete", "NA"),
        "final_stage": data.get("current_stage", "NA"),
        "failed_stage": progress_failed_stage(progress) or "NA",
        "progress_file": str(progress_path),
    }


def update_e1_results_from_t4(t4_summary: dict[str, Any], e1_run_dir: Path) -> None:
    manifest = discover_t4_files(t4_summary.get("t4_session_dir"))
    if not manifest.get("copied_files"):
        existing_manifest = load_json_if_exists(e1_run_dir / "t4_linked_outputs" / "t4_output_manifest.json")
        if existing_manifest:
            manifest.update(existing_manifest)
    flat = flatten_t4_summary(t4_summary, manifest)
    geometry = flatten_perception_geometry_results(manifest)
    robot = flatten_robot_results(t4_summary, manifest)
    clamp = flatten_clamp_results(t4_summary, manifest)
    timing = flatten_timing_summary(manifest)
    perception = {
        "status": "available_from_T4" if manifest.get("latest_run_record_json") != "NA" else "NA",
        "t4_action": t4_summary.get("action", "NA"),
    }
    write_json(e1_run_dir / "perception_results.json", perception)
    write_json(e1_run_dir / "geometry_results.json", geometry)
    write_json(e1_run_dir / "robot_results.json", robot)
    write_json(e1_run_dir / "clamp_results.json", clamp)
    write_json(e1_run_dir / "timing_summary.json", timing)
    failure_class = "none" if t4_summary.get("success") else ("manual_abort" if t4_summary.get("action") == "jog_stop" else "unknown")
    progress_fields = progress_fields_for_run(e1_run_dir)
    failure = {
        "failure_class": failure_class,
        "failure_stage": flat.get("abort_stage", progress_fields.get("failed_stage", "NA")),
        "failure_reason_auto": flat.get("abort_reason", t4_summary.get("failure_reason", "NA")),
        "failure_reason_manual": "NA",
        "manual_override_allowed": True,
    }
    write_json(e1_run_dir / "failure_taxonomy.json", failure)
    full_task = {
        **flat,
        **progress_fields,
        "run_id": e1_run_dir.name,
        "t4_run_ids": [],
        "t4_output_manifest_path": str(e1_run_dir / "t4_linked_outputs" / "t4_output_manifest.json"),
    }
    write_json(e1_run_dir / "full_task_summary.json", full_task)
    row = {"run_id": e1_run_dir.name, **flat, **geometry, **robot, **clamp, **timing, **progress_fields, "failure_class": failure_class}
    append_csv(e1_run_dir / "E1_run_summary.csv", row)
    append_csv(e1_run_dir.parent / "E1_master_results.csv", row)


def compute_and_save_upv_results(e1_run_dir: Path, tof_value: str, tof_units: str, path_source: str, manual_path_length: str, quality: str, notes: str = "") -> dict[str, Any]:
    def as_float(text: Any) -> float | None:
        try:
            if text in (None, "", "NA"):
                return None
            return float(text)
        except Exception:  # noqa: BLE001
            return None

    tof_raw = as_float(tof_value)
    tof_s = None
    tof_us = "NA"
    if tof_raw is not None and tof_raw > 0:
        tof_s = tof_raw * 1e-6 if tof_units == "microseconds" else tof_raw
        tof_us = tof_raw if tof_units == "microseconds" else tof_raw * 1e6
    geometry = load_json_if_exists(e1_run_dir / "geometry_results.json")
    clamp = load_json_if_exists(e1_run_dir / "clamp_results.json")
    path_length = None
    if path_source == "manual_path_length":
        path_length = as_float(manual_path_length)
    elif path_source == "recommended_upv_path_length":
        path_length = as_float(geometry.get("recommended_upv_path_length_mm")) or as_float(clamp.get("recommended_upv_path_length_mm"))
    elif path_source == "recommended_clamp_opening":
        path_length = as_float(clamp.get("recommended_clamp_opening_mm"))
    elif path_source == "selected_probe_spacing":
        path_length = as_float(geometry.get("selected_probe_spacing_mm"))
    velocity = compute_upv_velocity(tof_s, "seconds", path_length)
    result = {
        "upv_time_of_flight_us": tof_us if tof_s is not None else "NA",
        "time_of_flight_s": tof_s if tof_s is not None else "NA",
        "upv_path_length_mm": path_length if path_length is not None else "NA",
        "path_length_source": path_source,
        "path_length_NA_reason": "recommended_upv_path_length_unavailable" if path_source == "recommended_upv_path_length" and path_length is None else "NA",
        "upv_velocity_m_per_s": velocity,
        "upv_reading_quality": quality,
        "upv_notes": notes,
    }
    write_json(e1_run_dir / "upv_results.json", result)
    append_csv(e1_run_dir.parent / "E1_master_results.csv", {"run_id": e1_run_dir.name, **result})
    return result


def launch_streamlit_gui(config_path: Path, config: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="E1 Autonomous UPV Experiment Manager", layout="wide")
    st.title("E1 Autonomous UPV Experiment Manager")
    st.caption("Paper experiment manager for T4. Hardware actions require checklist confirmation.")

    if "latest_run_dir" not in st.session_state:
        st.session_state.latest_run_dir = None
    if "status" not in st.session_state:
        st.session_state.status = "SAFE"
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "live_process" not in st.session_state:
        st.session_state.live_process = None
    if "safety_save_requested" not in st.session_state:
        st.session_state.safety_save_requested = False

    def log(message: str) -> None:
        st.session_state.logs.append(f"{now_iso()} {message}")

    def save_error(exc: BaseException) -> None:
        if st.session_state.latest_run_dir:
            log_path = Path(st.session_state.latest_run_dir) / "logs" / "error_trace.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        log(f"ERROR: {exc}")

    def safety_checklist(prefix: str, warning: str, items: list[str]) -> bool:
        st.warning(warning)
        values = [st.checkbox(item, key=f"{prefix}_{idx}") for idx, item in enumerate(items)]
        complete = all(values)
        if complete:
            st.success("Safety checklist complete.")
        else:
            st.info("Complete checklist to enable this action.")
        return complete

    output_root = ExperimentRunWriter(REPO_ROOT, config, config_path).output_root
    recent_runs = [path for path in sorted(output_root.glob("20*"), reverse=True) if path.is_dir()]
    resume_options = ["Start new run"] + [str(path) for path in recent_runs[:10]]
    resume_choice = st.selectbox("Resume incomplete run / Start new run", resume_options, index=0)
    if resume_choice != "Start new run" and st.button("Load selected E1 run", use_container_width=True):
        st.session_state.latest_run_dir = resume_choice
        st.session_state.status = "READY"
        log(f"Loaded E1 run {resume_choice}")

    status_color = {
        "SAFE": "blue",
        "READY": "green",
        "RUNNING": "orange",
        "PAUSED": "orange",
        "FAILED": "red",
        "COMPLETE": "green",
    }.get(st.session_state.status, "gray")
    st.markdown(f"### Status: :{status_color}[{st.session_state.status}]")
    if st.session_state.latest_run_dir:
        st.info(f"Latest E1 output folder: {st.session_state.latest_run_dir}")

    emergency = st.container(border=True)
    with emergency:
        st.subheader("Always-visible Safety Controls")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            send_jog_stop = st.checkbox("Send clamp/Arduino jog_stop if available.", key="abort_send_jog_stop")
            if st.button("ABORT / STOP NOW", use_container_width=True):
                try:
                    if st.session_state.live_process is not None:
                        st.session_state.live_process.terminate()
                        log("Terminated live E1 subprocess.")
                    if st.session_state.latest_run_dir:
                        run_dir = Path(st.session_state.latest_run_dir)
                        write_json(run_dir / "failure_taxonomy.json", {"failure_class": "manual_abort", "failure_stage": "E1_ABORT_STOP", "failure_reason_auto": "operator_pressed_abort_stop", "failure_reason_manual": "NA"})
                        if send_jog_stop:
                            run_t4_oneshot(config, "jog_stop", "JOG_STOP", run_dir)
                    st.session_state.status = "PAUSED"
                except Exception as exc:  # noqa: BLE001
                    st.session_state.status = "FAILED"
                    save_error(exc)
                    st.exception(exc)
        with c2:
            emergency_ok = safety_checklist(
                "emergency_release",
                "This will command the clamp release/emergency-release sequence. Keep hands clear.",
                ["Hands are clear.", "Object/tool area is safe.", "I understand this will move the clamp."],
            )
            if st.button("EMERGENCY CLAMP RELEASE NOW", use_container_width=True, disabled=not emergency_ok):
                try:
                    if not st.session_state.latest_run_dir:
                        raise E1Error("Create or resume an E1 run folder before emergency release.")
                    result = run_t4_oneshot(config, "emergency_release", "EMERGENCY_RELEASE", Path(st.session_state.latest_run_dir))
                    st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
                    st.json(result)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.status = "FAILED"
                    save_error(exc)
                    st.exception(exc)
        with c3:
            home_ok = safety_checklist(
                "home_robot",
                "This will move the robot to the configured home pose. Do not run if the clamp is still attached to the object.",
                ["Clamp is released/open.", "Robot path to home is clear.", "I understand this will move the robot."],
            )
            if st.button("HOME ROBOT NOW", use_container_width=True, disabled=not home_ok):
                try:
                    if not st.session_state.latest_run_dir:
                        raise E1Error("Create or resume an E1 run folder before homing.")
                    result = run_t4_oneshot(config, "home", "HOME_ROBOT", Path(st.session_state.latest_run_dir))
                    st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
                    st.json(result)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.status = "FAILED"
                    save_error(exc)
                    st.exception(exc)
        if c4.button("SAVE CURRENT RUN", use_container_width=True):
            st.session_state.safety_save_requested = True
            log("Save current run requested.")

    left, right = st.columns([0.42, 0.58])
    with left:
        st.header("1. Experiment Metadata")
        operator = st.text_input("Operator name", value=config.get("default_operator", "Pranav"))
        project_name = st.text_input("Project name", value="UPV_VLM_orient_AIC_paper")
        specimen_id = st.text_input("Specimen ID", value="brick_01")
        scene_id = st.text_input("Scene ID", value="scene_01")
        trial_index = st.number_input("Trial index", min_value=1, value=1, step=1)
        notes = st.text_area("Notes", value="", height=90)
        material_query = st.text_input("Material query / open-vocabulary text", value=config.get("default_material_query", "brick"))
        axis_mode = st.radio("Axis mode", ["major", "minor"], horizontal=True, index=0 if config.get("default_axis_mode", "major") == "major" else 1)
        run_mode = st.selectbox("Run mode", RUN_MODES, index=RUN_MODES.index(config.get("default_run_mode", "full_autonomous_home")))
        categories = st.multiselect("Experiment category tags", EXPERIMENT_CATEGORIES, default=["normal_single_object"])

        st.header("2. Model / Baseline / Ablation")
        pipeline_mode = st.selectbox("Pipeline mode", PIPELINE_MODES, index=0)
        st.write("Disabled prompt stages:", disabled_stages_for_mode(pipeline_mode) or "none")
        model_info = {
            "open_vocab_detector": "repo current detector or T4/T2 default",
            "segmentation_model": "repo current VLM/V2a mask segmentation",
            "material_verification_model": "available if configured; otherwise NA",
            "edge_contact_vlm_model": "Qwen2.5-VL local model if enabled",
        }
        open_vocab_detector = st.text_input("open_vocab_detector", value=model_info["open_vocab_detector"])
        segmentation_model = st.text_input("segmentation_model", value=model_info["segmentation_model"])
        material_verification_model = st.text_input("material_verification_model", value=model_info["material_verification_model"])
        edge_contact_vlm_model = st.text_input("edge_contact_vlm_model", value=model_info["edge_contact_vlm_model"])

        st.header("3. Metrics")
        select_all = st.checkbox("Select all metrics", value=bool(config.get("metric_defaults", {}).get("select_all", True)))
        selected_metrics: list[str] = []
        for group, metrics in METRIC_GROUPS.items():
            with st.expander(group, expanded=False):
                for metric in metrics:
                    if st.checkbox(metric, value=select_all, key=f"metric_{metric}"):
                        selected_metrics.append(metric)

    with right:
        st.header("2B. Prompt Engineering and VLM Inputs")
        prompt_cfg = config.get("prompt_engineering", {})
        canonical_prompt_path = canonical_prompt_config_path(config)
        base_bundle = load_prompt_bundle(
            canonical_prompt_path,
            prompt_set_name=str(prompt_cfg.get("default_prompt_set_name", "default_v1")),
            prompt_set_version=str(prompt_cfg.get("default_prompt_set_version", "v1")),
        )
        display_bundle = apply_ablation_mode(base_bundle, pipeline_mode)
        st.caption(f"Canonical prompt source: {canonical_prompt_path}")
        prompt_set_name = st.text_input("Prompt set name", value=prompt_cfg.get("default_prompt_set_name", "default_v1"))
        prompt_set_version = st.text_input("Prompt set version", value=prompt_cfg.get("default_prompt_set_version", "v1"))
        preset_dir = resolve_repo_path(prompt_cfg.get("prompt_presets_dir", "configs/prompt_presets"))
        preset_files = sorted(preset_dir.glob("*.json")) if preset_dir.exists() else []
        preset_choice = st.selectbox("Load prompt preset", ["None"] + [path.name for path in preset_files])
        loaded_prompts: dict[str, str] = {}
        if preset_choice != "None":
            loaded = load_json_if_exists(preset_dir / preset_choice)
            loaded_prompts = dict(loaded.get("prompts", {}))
            st.info(f"Loaded prompt preset metadata from {preset_choice}.")
        if st.button("Reset prompts to defaults", use_container_width=True):
            for stage in display_bundle.stages:
                st.session_state.pop(f"prompt_{stage}", None)
            log("Prompt text areas reset to defaults on next rerun.")
            st.rerun()
        prompt_overrides: dict[str, str] = {}
        for stage, stage_def in display_bundle.stages.items():
            rendered = display_bundle.render_stage(stage, material_query=material_query, axis_mode=axis_mode)
            enabled = bool(rendered["enabled"])
            label = f"{stage} prompt" if enabled else f"{stage} prompt (disabled by mode)"
            default_text = str(rendered["user_prompt"])
            loaded_text = loaded_prompts.get(stage)
            prompt_overrides[stage] = st.text_area(label, value=loaded_text or (default_text if enabled else f"NA: disabled by {pipeline_mode}"), height=120, key=f"prompt_{stage}")
        preset_name = st.text_input("Save prompt preset name", value="")
        if st.button("Save prompt preset", use_container_width=True):
            preset_dir.mkdir(parents=True, exist_ok=True)
            name = preset_name or f"{prompt_set_name}_{prompt_set_version}"
            write_json(preset_dir / f"{name}.json", {"prompt_set_name": prompt_set_name, "prompt_set_version": prompt_set_version, "prompts": prompt_overrides})
            log(f"Saved prompt preset {preset_dir / f'{name}.json'}")

        st.header("4. Manual Labels")
        labels = default_manual_labels()
        labels["target_selection_correct"] = st.selectbox("Correct requested object?", ["NA", "yes", "no"])
        labels["manual_selected_class"] = st.text_input("Manual selected class", value="")
        labels["wrong_failure_type"] = st.selectbox("If wrong, what happened?", ["NA", "wrong_class", "wrong_instance", "bad_mask", "bad_axis", "bad_depth", "bad_anchor", "robot_pose_bad", "clamp_width_bad", "UPV_reading_failed", "other"])
        labels["mask_usable"] = st.selectbox("Mask usable?", ["NA", "yes", "no"])
        labels["axis_correct"] = st.selectbox("Axis correct?", ["NA", "yes", "no"])
        labels["manual_major_dimension_mm"] = st.text_input("Manual major dimension mm", value="NA")
        labels["manual_minor_dimension_mm"] = st.text_input("Manual minor dimension mm", value="NA")
        labels["manual_actual_clamp_opening_mm"] = st.text_input("Manual actual clamp opening mm", value="NA")
        labels["human_intervention_required"] = st.selectbox("Human intervention required?", ["NA", "yes", "no"])
        labels["free_text_notes"] = st.text_area("Manual notes", value="")

        st.header("6. UPV Time-of-Flight Entry")
        tof_value = st.text_input("Time of flight", value="")
        tof_units = st.selectbox("TOF units", ["microseconds", "seconds"])
        path_source = st.selectbox("Path length source", ["recommended_upv_path_length", "recommended_clamp_opening", "selected_probe_spacing", "manual_path_length"])
        manual_path_length = st.text_input("Manual path length mm", value="")
        quality = st.selectbox("Reading quality", ["NA", "good", "noisy", "failed"])
        upv = default_upv_results()
        upv.update({"time_of_flight_entry": tof_value or "NA", "time_of_flight_units": tof_units, "path_length_source": path_source, "manual_path_length_mm": manual_path_length or "NA", "upv_reading_quality": quality})

        st.header("5/7. Pause and Recording Options")
        pause_defaults = config.get("manual_pause_defaults", {})
        pauses = {key: st.checkbox(key, value=bool(value)) for key, value in pause_defaults.items()}
        recording = {
            "save_images": st.checkbox("save_images", value=True),
            "save_overlays": st.checkbox("save_overlays", value=True),
            "save_video": st.checkbox("save_video", value=False),
            "save_rosbag": st.checkbox("save_rosbag", value=False),
            "video_fps": st.number_input("video_fps", value=int(config.get("video_fps", 10))),
            "max_video_duration_s": st.number_input("max_video_duration_s", value=int(config.get("max_video_duration_s", 180))),
        }
        st.write("Pause points:", pauses)
        st.write("Recording:", recording)

    st.divider()
    st.header("Run / Save")
    metadata = {
        "operator_name": operator,
        "project_name": project_name,
        "specimen_id": specimen_id,
        "scene_id": scene_id,
        "trial_index": trial_index,
        "notes": notes,
        "material_query": material_query,
        "axis_mode": axis_mode,
        "run_mode": run_mode,
        "experiment_category_tags": categories,
        "pause_points": pauses,
        "recording_options": recording,
        "model_selections": {
            "open_vocab_detector": open_vocab_detector,
            "segmentation_model": segmentation_model,
            "material_verification_model": material_verification_model,
            "edge_contact_vlm_model": edge_contact_vlm_model,
        },
    }
    writer = ExperimentRunWriter(REPO_ROOT, config, config_path)
    if st.session_state.safety_save_requested:
        try:
            run_dir = ensure_e1_run(writer, config, config_path, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides, st.session_state.latest_run_dir)
            save_current_gui_state(run_dir, config, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides)
            compute_and_save_upv_results(run_dir, tof_value, tof_units, path_source, manual_path_length, quality, notes="")
            st.session_state.latest_run_dir = str(run_dir)
            st.session_state.status = "READY"
            log(f"Saved current E1 run state {run_dir}")
            st.session_state.safety_save_requested = False
        except Exception as exc:  # noqa: BLE001
            st.session_state.status = "FAILED"
            st.session_state.safety_save_requested = False
            save_error(exc)
            st.exception(exc)
    save_col, hw_col = st.columns(2)
    if save_col.button("Create / Save E1 Run Folder", use_container_width=True):
        try:
            run_dir = ensure_e1_run(writer, config, config_path, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides, st.session_state.latest_run_dir)
            save_current_gui_state(run_dir, config, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides)
            compute_and_save_upv_results(run_dir, tof_value, tof_units, path_source, manual_path_length, quality, notes="")
            st.session_state.latest_run_dir = str(run_dir)
            st.session_state.status = "READY"
            log(f"Saved E1 run folder {run_dir}")
            st.success(f"Saved E1 run folder: {run_dir}")
        except Exception as exc:  # noqa: BLE001
            st.session_state.status = "FAILED"
            st.exception(exc)
            save_error(exc)

    manual_clamp_opening = hw_col.text_input("Manual clamp opening for clamp_only mm", value="")
    run_label = {
        "perception_only": "RUN PERCEPTION ONLY",
        "plan_only": "RUN PLAN ONLY",
        "clamp_only": "RUN CLAMP ONLY",
        "full_autonomous": "RUN FULL AUTONOMOUS",
        "full_autonomous_home": "RUN FULL AUTONOMOUS + HOME",
    }.get(run_mode, "RUN SELECTED MODE")
    with hw_col:
        if run_mode == "full_autonomous_home":
            selected_mode_ok = safety_checklist(
                "full_autonomous_home",
                "This will move the UR3e, approach the object, clamp the UPV tool, hold for the UPV reading window, release the clamp, and return home. Verify the workspace is clear, the clamp is open/released, the object is correctly placed, and the emergency stop is accessible.",
                [
                    "Workspace is clear.",
                    "Robot path is clear.",
                    "Clamp is open or safe to move.",
                    "RealSense and Arduino are ready if needed.",
                    "Emergency stop is accessible.",
                    "I understand this will move the robot and clamp.",
                ],
            )
        elif run_mode == "full_autonomous":
            selected_mode_ok = safety_checklist(
                "full_autonomous",
                "This will move the UR3e, approach the object, clamp the UPV tool, hold for the UPV reading window, and release the clamp.",
                [
                    "Workspace is clear.",
                    "Robot path is clear.",
                    "Clamp is open/safe.",
                    "Emergency stop is accessible.",
                    "I understand this will move the robot and clamp.",
                ],
            )
        elif run_mode == "clamp_only":
            selected_mode_ok = safety_checklist(
                "clamp_only",
                "This will move the clamp only. Keep hands clear and verify the requested opening.",
                [
                    "Clamp path is clear.",
                    "Manual/planned opening is correct.",
                    "I understand this will move the clamp.",
                ],
            )
        else:
            selected_mode_ok = True
            st.info("This selected mode is no-motion from E1 validation paths.")
    if hw_col.button(run_label, use_container_width=True, disabled=not selected_mode_ok):
        try:
            run_dir = ensure_e1_run(writer, config, config_path, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides, st.session_state.latest_run_dir)
            save_current_gui_state(run_dir, config, metadata, pipeline_mode, selected_metrics, labels, upv, prompt_overrides)
            st.session_state.latest_run_dir = str(run_dir)
            st.session_state.status = "RUNNING"
            video_recorder, rosbag_recorder = start_requested_recorders(run_dir, config, recording, dry_run=False)
            try:
                if run_mode == "perception_only":
                    result = run_t4_oneshot(config, "perception_only", "", run_dir)
                elif run_mode == "plan_only":
                    result = run_t4_oneshot(config, "plan_only", "", run_dir)
                elif run_mode == "clamp_only":
                    extra_args: list[str] = []
                    if manual_clamp_opening.strip():
                        extra_args = ["--manual-opening-mm", manual_clamp_opening.strip()]
                    result = run_t4_oneshot(config, "clamp_only", "RUN_CLAMP_ONLY", run_dir, extra_args=extra_args)
                elif run_mode == "full_autonomous":
                    result = run_t4_oneshot(config, "full_final", "RUN_FINAL", run_dir)
                elif run_mode == "full_autonomous_home":
                    result = run_t4_oneshot(config, "full_final_home", "RUN_FINAL_HOME", run_dir)
                else:
                    raise E1Error(f"Unsupported E1 run mode: {run_mode}")
            finally:
                stop_recorders(run_dir, video_recorder, rosbag_recorder)
            compute_and_save_upv_results(run_dir, tof_value, tof_units, path_source, manual_path_length, quality, notes="")
            st.session_state.status = "COMPLETE" if result.get("success") else "FAILED"
            st.json(result)
        except Exception as exc:  # noqa: BLE001
            st.session_state.status = "FAILED"
            st.exception(exc)
            save_error(exc)

    st.header("Latest Run Artifacts")
    if st.session_state.latest_run_dir:
        run_dir = Path(st.session_state.latest_run_dir)
        manifest = load_json_if_exists(run_dir / "t4_linked_outputs" / "t4_output_manifest.json")
        prompt_source = load_json_if_exists(run_dir / "actual_prompt_source.json")
        if prompt_source.get("gui_t4_prompt_hash_match") is False:
            st.warning("WARNING: GUI prompts and T4 prompts do not match.")
        images = sorted(run_dir.glob("figures/*.png"))
        tabs = st.tabs(["Perception", "Geometry", "Robot Plan", "Clamp", "Prompt/VLM", "Logs"])
        names_by_tab = {
            0: ["raw", "mask", "selected", "rgb"],
            1: ["axis", "geometry", "width"],
            2: ["plan", "pose", "robot"],
            3: ["clamp", "width"],
        }
        for idx, tab in enumerate(tabs[:4]):
            with tab:
                matches = [path for path in images if any(token in path.name.lower() for token in names_by_tab.get(idx, []))]
                if matches:
                    for path in matches[:4]:
                        st.image(str(path), caption=path.name, use_container_width=True)
                else:
                    st.write("NA: image not available")
        with tabs[4]:
            prompt_summary = load_json_if_exists(run_dir / "prompt_engineering_summary.json")
            st.json(prompt_summary or {"status": "NA: prompt summary not available"})
        with tabs[5]:
            st.json(manifest or {"status": "NA: T4 output manifest not available"})
    else:
        st.write("NA: no E1 run folder selected yet.")

    st.header("Logs")
    st.text_area("Event log", value="\n".join(st.session_state.logs[-200:]), height=220)


def launch_streamlit_gui_v2(config_path: Path, config: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="E1 Autonomous UPV Experiment Manager", layout="wide")

    defaults = default_metadata(config)
    state_defaults = {
        "operator_name": defaults["operator_name"],
        "project_name": defaults["project_name"],
        "specimen_id": defaults["specimen_id"],
        "scene_id": defaults["scene_id"],
        "trial_index": int(defaults["trial_index"]),
        "notes": defaults["notes"],
        "material_query": defaults["material_query"],
        "axis_mode": defaults["axis_mode"],
        "run_mode": defaults["run_mode"],
        "pipeline_mode": PIPELINE_MODES[0],
        "experiment_category_tags": ["normal_single_object"],
        "latest_run_dir": None,
        "current_run_identity": None,
        "continue_existing_mismatch": False,
        "status": "SAFE",
        "logs": [],
        "live_process": None,
        "is_running": False,
        "current_action": "idle",
        "run_started_at": None,
        "last_action_result": {},
        "safety_save_requested": False,
        "recording_options": {
            "save_images": True,
            "save_overlays": True,
            "save_video": False,
            "save_rosbag": False,
            "video_fps": int(config.get("video_fps", 10)),
            "max_video_duration_s": int(config.get("max_video_duration_s", 180)),
        },
        "pause_points": dict(config.get("manual_pause_defaults", {})),
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def log(message: str) -> None:
        st.session_state.logs.append(f"{now_iso()} {message}")

    def save_error(exc: BaseException) -> None:
        if st.session_state.latest_run_dir:
            log_path = Path(st.session_state.latest_run_dir) / "logs" / "error_trace.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        log(f"ERROR: {exc}")

    def current_metadata() -> dict[str, Any]:
        return {
            "operator_name": st.session_state.operator_name,
            "project_name": st.session_state.project_name,
            "specimen_id": st.session_state.specimen_id,
            "scene_id": st.session_state.scene_id,
            "trial_index": int(st.session_state.trial_index),
            "notes": st.session_state.notes,
            "material_query": st.session_state.material_query,
            "axis_mode": st.session_state.axis_mode,
            "run_mode": st.session_state.run_mode,
            "experiment_category_tags": list(st.session_state.experiment_category_tags),
            "pause_points": dict(st.session_state.get("pause_points", {})),
            "recording_options": dict(st.session_state.get("recording_options", {})),
            "model_selections": {
                "open_vocab_detector": st.session_state.get("open_vocab_detector", "repo current detector or T4/T2 default"),
                "segmentation_model": st.session_state.get("segmentation_model", "repo current VLM/V2a mask segmentation"),
                "material_verification_model": st.session_state.get("material_verification_model", "available if configured; otherwise NA"),
                "edge_contact_vlm_model": st.session_state.get("edge_contact_vlm_model", "Qwen2.5-VL local model if enabled"),
            },
        }

    def selected_metrics_from_state() -> list[str]:
        metrics: list[str] = []
        for group_metrics in METRIC_GROUPS.values():
            for metric in group_metrics:
                if st.session_state.get(f"metric_{metric}", bool(config.get("metric_defaults", {}).get("select_all", True))):
                    metrics.append(metric)
        return metrics

    def labels_from_state() -> dict[str, Any]:
        labels = default_manual_labels()
        for key in [
            "target_selection_correct",
            "manual_selected_class",
            "wrong_failure_type",
            "mask_usable",
            "axis_correct",
            "manual_major_dimension_mm",
            "manual_minor_dimension_mm",
            "manual_actual_clamp_opening_mm",
            "human_intervention_required",
            "free_text_notes",
        ]:
            labels[key] = st.session_state.get(key, labels.get(key, "NA"))
        return labels

    def upv_from_state() -> dict[str, Any]:
        upv = default_upv_results()
        upv.update(
            {
                "time_of_flight_entry": st.session_state.get("tof_value", "") or "NA",
                "time_of_flight_units": st.session_state.get("tof_units", "microseconds"),
                "path_length_source": st.session_state.get("path_source", "recommended_upv_path_length"),
                "manual_path_length_mm": st.session_state.get("manual_path_length", "") or "NA",
                "upv_reading_quality": st.session_state.get("reading_quality", "NA"),
            }
        )
        return upv

    def collect_prompt_overrides(display_bundle: Any) -> dict[str, str]:
        return {stage: str(st.session_state.get(f"prompt_{stage}", "")) for stage in display_bundle.stages}

    def clear_loaded_run_state() -> None:
        st.session_state.latest_run_dir = None
        st.session_state.current_run_identity = None
        st.session_state.continue_existing_mismatch = False
        st.session_state.last_action_result = {}
        st.session_state.logs = []

    def reset_gui_defaults() -> None:
        clear_loaded_run_state()
        for key, value in state_defaults.items():
            if key not in {"logs", "latest_run_dir", "current_run_identity", "last_action_result"}:
                st.session_state[key] = value
        for key in list(st.session_state.keys()):
            if str(key).startswith("prompt_") or str(key).startswith("metric_") or str(key).startswith("safety_"):
                del st.session_state[key]

    def status_from_result(action: str, result: dict[str, Any]) -> str:
        if result.get("success"):
            return "COMPLETE"
        reason = str(result.get("failure_reason", "") or result.get("stderr", ""))
        if action in {"perception_only", "plan_only", "plan", "snap"} or "NO_VERIFIED_MATCH" in reason or "perception" in reason.lower():
            return "FAILED_PERCEPTION"
        if action in {"clamp_only", "emergency_release", "clamp_release", "jog_stop"} or "clamp" in reason.lower() or "arduino" in reason.lower():
            return "FAILED_CLAMP"
        if action in {"home", "full_final", "full_final_home"} or "robot" in reason.lower() or "rtde" in reason.lower():
            return "FAILED_ROBOT"
        return "FAILED"

    def save_or_create_run(*, force_new: bool = False, allow_mismatch: bool = False) -> Path:
        writer = ExperimentRunWriter(REPO_ROOT, config, config_path)
        metadata = current_metadata()
        pipeline_mode = st.session_state.pipeline_mode
        prompt_cfg = config.get("prompt_engineering", {})
        canonical_prompt_path = canonical_prompt_config_path(config)
        base_bundle = load_prompt_bundle(
            canonical_prompt_path,
            prompt_set_name=str(prompt_cfg.get("default_prompt_set_name", "default_v1")),
            prompt_set_version=str(prompt_cfg.get("default_prompt_set_version", "v1")),
        )
        display_bundle = apply_ablation_mode(base_bundle, pipeline_mode)
        prompt_overrides = collect_prompt_overrides(display_bundle)
        latest = None if force_new else st.session_state.latest_run_dir
        run_dir = ensure_e1_run(
            writer,
            config,
            config_path,
            metadata,
            pipeline_mode,
            selected_metrics_from_state(),
            labels_from_state(),
            upv_from_state(),
            prompt_overrides,
            latest,
            allow_existing_mismatch=allow_mismatch,
        )
        save_current_gui_state(run_dir, config, metadata, pipeline_mode, selected_metrics_from_state(), labels_from_state(), upv_from_state(), prompt_overrides)
        st.session_state.latest_run_dir = str(run_dir)
        st.session_state.current_run_identity = run_identity_from_metadata(metadata)
        st.session_state.continue_existing_mismatch = False
        log(f"Saved E1 run folder {run_dir}")
        return run_dir

    def render_progress_panel(progress_payload: dict[str, Any] | None = None) -> None:
        payload = progress_payload or st.session_state.get("last_progress") or {}
        if not payload and st.session_state.latest_run_dir:
            loaded = read_progress(Path(st.session_state.latest_run_dir) / "run_progress.json")
            payload = progress_to_dict(loaded) if loaded else {}
        if not payload:
            st.progress(0, text="Progress: waiting / ready")
            st.caption("No run progress file is available yet.")
            return
        percent = int(payload.get("percent_complete", 0) or 0)
        current_stage = str(payload.get("current_stage", "waiting"))
        stages = payload.get("stages") or []
        current_label = current_stage.replace("_", " ")
        for stage in stages:
            if isinstance(stage, dict) and stage.get("name") == current_stage:
                current_label = str(stage.get("label") or current_label)
                break
        st.progress(max(0, min(100, percent)) / 100.0, text=f"Progress: {percent}%")
        st.markdown(f"**Current stage:** {current_label}")
        st.caption(f"Progress status: {payload.get('status', 'NA')} | Last update: {payload.get('updated_at', 'NA')}")
        rows: list[dict[str, Any]] = []
        icon_map = {"succeeded": "done", "running": "running", "failed": "failed", "skipped": "skipped", "pending": "pending"}
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            rows.append(
                {
                    "state": icon_map.get(str(stage.get("status", "pending")), str(stage.get("status", "pending"))),
                    "stage": stage.get("label", stage.get("name")),
                    "percent": stage.get("percent"),
                    "failure_reason": stage.get("failure_reason") or "",
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)

    def execute_action(action: str, confirmation: str = "", extra_args: list[str] | None = None, *, force_new: bool = False) -> dict[str, Any]:
        run_dir = save_or_create_run(force_new=force_new, allow_mismatch=bool(st.session_state.continue_existing_mismatch))
        progress_mode = {
            "full_final": "full_autonomous",
            "full_final_home": "full_autonomous_home",
            "perception_only": "perception_only",
            "plan_only": "plan_only",
            "clamp_only": "clamp_only",
        }.get(action, st.session_state.run_mode)
        progress_file = run_dir / "run_progress.json"
        progress = update_stage(make_progress_template(progress_mode), "create_run_folder", "succeeded", success=True)
        write_progress(progress, progress_file)
        st.session_state.is_running = True
        st.session_state.current_action = action
        st.session_state.run_started_at = now_iso()
        st.session_state.status = "RUNNING"
        log(f"Starting T4 one-shot action {action}")
        video_recorder, rosbag_recorder = start_requested_recorders(run_dir, config, st.session_state.get("recording_options", {}), dry_run=False)
        try:
            progress_slot = st.empty()

            def live_progress_callback(progress_payload: dict[str, Any]) -> None:
                st.session_state.last_progress = progress_payload
                with progress_slot.container():
                    render_progress_panel(progress_payload)

            with st.spinner(f"RUNNING {action}. Do not close this page..."):
                result = run_t4_oneshot(
                    config,
                    action,
                    confirmation,
                    run_dir,
                    extra_args=extra_args,
                    progress_file=progress_file,
                    live_progress_callback=live_progress_callback,
                )
            compute_and_save_upv_results(
                run_dir,
                st.session_state.get("tof_value", ""),
                st.session_state.get("tof_units", "microseconds"),
                st.session_state.get("path_source", "recommended_upv_path_length"),
                st.session_state.get("manual_path_length", ""),
                st.session_state.get("reading_quality", "NA"),
                notes="",
            )
            st.session_state.status = status_from_result(action, result)
            st.session_state.last_action_result = result
            final_progress = read_progress(progress_file)
            if final_progress:
                st.session_state.last_progress = progress_to_dict(final_progress)
            log(f"Completed {action}: success={result.get('success')} status={st.session_state.status}")
            return result
        finally:
            stop_recorders(run_dir, video_recorder, rosbag_recorder)
            st.session_state.is_running = False
            st.session_state.current_action = "idle"

    def safety_checklist(prefix: str, items: list[str]) -> bool:
        values = [st.checkbox(item, key=f"safety_{prefix}_{idx}") for idx, item in enumerate(items)]
        complete = all(values)
        st.success("Safety checklist complete.") if complete else st.info("Complete checklist to enable this action.")
        return complete

    gui_identity = run_identity_from_metadata(current_metadata())
    loaded_identity = load_run_identity(st.session_state.latest_run_dir)
    identity_ok = identities_match(gui_identity, loaded_identity)
    mismatch = bool(st.session_state.latest_run_dir and loaded_identity and not identity_ok)

    st.title("E1 Autonomous UPV Experiment Manager")
    status_color = {
        "SAFE": "blue",
        "READY": "green",
        "RUNNING": "orange",
        "PAUSED": "orange",
        "FAILED": "red",
        "FAILED_PERCEPTION": "red",
        "FAILED_ROBOT": "red",
        "FAILED_CLAMP": "red",
        "COMPLETE": "green",
    }.get(st.session_state.status, "gray")
    header = st.container(border=True)
    with header:
        cols = st.columns([0.16, 0.26, 0.34, 0.24])
        cols[0].markdown(f"**Status:** :{status_color}[{st.session_state.status}]")
        cols[1].markdown(f"**GUI identity:** `{run_identity_slug(gui_identity)}`")
        cols[2].markdown(f"**Loaded folder:** `{st.session_state.latest_run_dir or 'No run loaded'}`")
        badge = "MATCH" if identity_ok else ("NO RUN LOADED" if not st.session_state.latest_run_dir else "MISMATCH")
        cols[3].markdown(f"**Identity:** `{badge}`")
        if st.session_state.is_running:
            st.info(f"RUNNING {st.session_state.current_action} since {st.session_state.run_started_at}")
            render_progress_panel()
        if mismatch:
            st.error(
                "IDENTITY MISMATCH: Current GUI fields do not match the loaded run folder. "
                "Start a new run or intentionally continue the existing run folder."
            )
            st.json(identity_diff(gui_identity, loaded_identity))

    emergency = st.container(border=True)
    with emergency:
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("ABORT / STOP NOW", use_container_width=True):
            try:
                if st.session_state.live_process is not None:
                    st.session_state.live_process.terminate()
                if st.session_state.latest_run_dir:
                    run_dir = Path(st.session_state.latest_run_dir)
                    write_json(run_dir / "failure_taxonomy.json", {"failure_class": "manual_abort", "failure_stage": "E1_ABORT_STOP", "failure_reason_auto": "operator_pressed_abort_stop", "failure_reason_manual": "NA"})
                    if st.session_state.get("safety_abort_send_jog_stop", False):
                        run_t4_oneshot(config, "jog_stop", "JOG_STOP", run_dir)
                st.session_state.status = "PAUSED"
                log("ABORT / STOP pressed.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED"
                save_error(exc)
                st.exception(exc)
        if c2.button("EMERGENCY CLAMP RELEASE NOW", use_container_width=True, disabled=not st.session_state.get("safety_emergency_ready", False)):
            try:
                execute_action("emergency_release", "EMERGENCY_RELEASE")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED_CLAMP"
                save_error(exc)
                st.exception(exc)
        if c3.button("HOME ROBOT NOW", use_container_width=True, disabled=not st.session_state.get("safety_home_ready", False)):
            try:
                execute_action("home", "HOME_ROBOT")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED_ROBOT"
                save_error(exc)
                st.exception(exc)
        if c4.button("SAVE CURRENT RUN", use_container_width=True):
            try:
                run_dir = save_or_create_run(allow_mismatch=bool(st.session_state.continue_existing_mismatch))
                compute_and_save_upv_results(run_dir, st.session_state.get("tof_value", ""), st.session_state.get("tof_units", "microseconds"), st.session_state.get("path_source", "recommended_upv_path_length"), st.session_state.get("manual_path_length", ""), st.session_state.get("reading_quality", "NA"), notes="")
                st.session_state.status = "READY"
                st.success(f"Saved {run_dir}")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED"
                save_error(exc)
                st.exception(exc)

    tab_setup, tab_prompts, tab_execute, tab_labels, tab_results, tab_logs = st.tabs(
        ["Setup", "Prompts / Models", "Safety + Execute", "Manual Labels + UPV", "Results / Artifacts", "Logs / Debug"]
    )

    with tab_setup:
        st.subheader("Experiment Metadata")
        a, b = st.columns(2)
        with a:
            st.text_input("Operator", key="operator_name")
            st.text_input("Project name", key="project_name")
            st.text_input("Material query", key="material_query")
            st.text_input("Specimen ID", key="specimen_id")
            st.text_input("Scene ID", key="scene_id")
        with b:
            st.number_input("Trial index", min_value=1, step=1, key="trial_index")
            st.radio("Axis mode", ["major", "minor"], horizontal=True, key="axis_mode")
            st.selectbox("Run mode", RUN_MODES, key="run_mode")
            if "category_quick" not in st.session_state:
                st.session_state.category_quick = ""
            q1, q2, q3, q4 = st.columns(4)
            if q1.button("orientation test"):
                st.session_state.experiment_category_tags = ["orientation_variation"]
            if q2.button("imperfection test"):
                st.session_state.experiment_category_tags = ["imperfection_variation"]
            if q3.button("baseline test"):
                st.session_state.experiment_category_tags = ["baseline_comparison"]
            if q4.button("repeatability test"):
                st.session_state.experiment_category_tags = ["repeatability"]
            st.multiselect("Experiment category tags", EXPERIMENT_CATEGORIES, key="experiment_category_tags")
        st.text_area("Notes", key="notes", height=90)
        gui_identity = run_identity_from_metadata(current_metadata())
        loaded_identity = load_run_identity(st.session_state.latest_run_dir)
        st.markdown("#### Run Identity")
        c1, c2 = st.columns(2)
        c1.json({"current_gui_identity": gui_identity, "run_identity_slug": run_identity_slug(gui_identity)})
        c2.json({"loaded_run_folder_identity": loaded_identity or "No run loaded", "loaded_run_folder": st.session_state.latest_run_dir or "No run loaded"})
        if st.button("Start New Run With Current Metadata", use_container_width=True):
            try:
                run_dir = save_or_create_run(force_new=True)
                st.session_state.status = "READY"
                st.success(f"Started new E1 run: {run_dir}")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED"
                save_error(exc)
                st.exception(exc)
        c1, c2, c3 = st.columns(3)
        if c1.button("Save Current Setup", use_container_width=True):
            try:
                run_dir = save_or_create_run(allow_mismatch=bool(st.session_state.continue_existing_mismatch))
                st.session_state.status = "READY"
                st.success(f"Saved setup: {run_dir}")
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = "FAILED"
                save_error(exc)
                st.exception(exc)
        if c2.button("Continue Existing Run Folder", use_container_width=True, disabled=not bool(st.session_state.latest_run_dir)):
            st.session_state.continue_existing_mismatch = True
            log("Operator selected Continue Existing Run Folder despite identity mismatch.")
            st.warning("Existing run folder reuse enabled for this session. A warning will be logged.")
        if c3.button("Reset GUI to Defaults", use_container_width=True):
            reset_gui_defaults()
            st.rerun()
        if st.button("Start New Run", use_container_width=True):
            clear_loaded_run_state()
            st.rerun()

        with st.expander("Metric selection", expanded=False):
            select_all = st.checkbox("Select all metrics", value=bool(config.get("metric_defaults", {}).get("select_all", True)))
            for group, metrics in METRIC_GROUPS.items():
                with st.expander(group, expanded=False):
                    for metric in metrics:
                        st.checkbox(metric, value=select_all, key=f"metric_{metric}")

        with st.expander("Pause points", expanded=False):
            pause_state = dict(st.session_state.get("pause_points", {}))
            for key, value in config.get("manual_pause_defaults", {}).items():
                pause_state[key] = st.checkbox(key, value=bool(pause_state.get(key, value)), key=f"pause_{key}")
            st.session_state.pause_points = pause_state

        writer = ExperimentRunWriter(REPO_ROOT, config, config_path)
        recent_runs = [path for path in sorted(writer.output_root.glob("20*"), reverse=True) if path.is_dir()]
        resume_options = ["No run selected"] + [str(path) for path in recent_runs[:10]]
        choice = st.selectbox("Resume latest run / Start new run", resume_options)
        if st.button("Load selected run folder", disabled=choice == "No run selected"):
            st.session_state.latest_run_dir = choice
            st.session_state.current_run_identity = load_run_identity(choice)
            st.session_state.status = "READY"
            log(f"Loaded E1 run {choice}")
            st.rerun()

    pipeline_mode = st.session_state.pipeline_mode
    prompt_identity = f"{st.session_state.material_query}|{st.session_state.axis_mode}|{pipeline_mode}"
    if st.session_state.get("last_prompt_render_identity") != prompt_identity:
        for key in list(st.session_state.keys()):
            if str(key).startswith("prompt_"):
                del st.session_state[key]
        st.session_state.last_prompt_render_identity = prompt_identity

    with tab_prompts:
        st.subheader("Models")
        st.selectbox("Pipeline mode", PIPELINE_MODES, key="pipeline_mode")
        st.caption(f"Disabled prompt stages: {disabled_stages_for_mode(st.session_state.pipeline_mode) or 'none'}")
        st.text_input("open_vocab_detector", value="repo current detector or T4/T2 default", key="open_vocab_detector")
        st.text_input("segmentation_model", value="repo current VLM/V2a mask segmentation", key="segmentation_model")
        st.text_input("material_verification_model", value="available if configured; otherwise NA", key="material_verification_model")
        st.text_input("edge_contact_vlm_model", value="Qwen2.5-VL local model if enabled", key="edge_contact_vlm_model")

        st.subheader("Canonical Prompt Registry")
        prompt_cfg = config.get("prompt_engineering", {})
        canonical_prompt_path = canonical_prompt_config_path(config)
        base_bundle = load_prompt_bundle(
            canonical_prompt_path,
            prompt_set_name=str(prompt_cfg.get("default_prompt_set_name", "default_v1")),
            prompt_set_version=str(prompt_cfg.get("default_prompt_set_version", "v1")),
        )
        display_bundle = apply_ablation_mode(base_bundle, st.session_state.pipeline_mode)
        st.caption(f"Canonical prompt source: {canonical_prompt_path}")
        prompt_source = load_json_if_exists(Path(st.session_state.latest_run_dir) / "actual_prompt_source.json") if st.session_state.latest_run_dir else {}
        st.write("Prompt hash match status:", prompt_source.get("gui_t4_prompt_hash_match", "NA"))
        p1, p2 = st.columns(2)
        p1.text_input("Prompt set name", value=prompt_cfg.get("default_prompt_set_name", "default_v1"), key="prompt_set_name")
        p2.text_input("Prompt set version", value=prompt_cfg.get("default_prompt_set_version", "v1"), key="prompt_set_version")
        preset_dir = resolve_repo_path(prompt_cfg.get("prompt_presets_dir", "configs/prompt_presets"))
        preset_files = sorted(preset_dir.glob("*.json")) if preset_dir.exists() else []
        preset_choice = st.selectbox("Load prompt preset", ["None"] + [path.name for path in preset_files])
        loaded_prompts: dict[str, str] = {}
        if preset_choice != "None":
            loaded = load_json_if_exists(preset_dir / preset_choice)
            loaded_prompts = dict(loaded.get("prompts", {}))
        if st.button("Reset prompts to defaults", use_container_width=True):
            for stage in display_bundle.stages:
                st.session_state.pop(f"prompt_{stage}", None)
            st.rerun()
        for stage, _stage_def in display_bundle.stages.items():
            rendered = display_bundle.render_stage(stage, material_query=st.session_state.material_query, axis_mode=st.session_state.axis_mode)
            enabled = bool(rendered["enabled"])
            with st.expander(f"{stage} prompt - {'enabled' if enabled else 'disabled'}", expanded=False):
                st.caption(f"Prompt hash: {rendered.get('prompt_hash', 'NA')}")
                default_text = str(rendered["user_prompt"])
                if f"prompt_{stage}" not in st.session_state:
                    st.session_state[f"prompt_{stage}"] = loaded_prompts.get(stage) or (default_text if enabled else f"NA: disabled by {st.session_state.pipeline_mode}")
                st.text_area("Prompt text", height=170, key=f"prompt_{stage}")
                if st.button(f"Reset {stage}", key=f"reset_{stage}"):
                    st.session_state[f"prompt_{stage}"] = default_text if enabled else f"NA: disabled by {st.session_state.pipeline_mode}"
                    st.rerun()
        preset_name = st.text_input("Save prompt preset name", value="")
        if st.button("Save prompt preset", use_container_width=True):
            preset_dir.mkdir(parents=True, exist_ok=True)
            prompts = {stage: st.session_state.get(f"prompt_{stage}", "") for stage in display_bundle.stages}
            name = preset_name or f"{st.session_state.prompt_set_name}_{st.session_state.prompt_set_version}"
            write_json(preset_dir / f"{name}.json", {"prompt_set_name": st.session_state.prompt_set_name, "prompt_set_version": st.session_state.prompt_set_version, "prompts": prompts})
            st.success(f"Saved {preset_dir / f'{name}.json'}")

    with tab_execute:
        st.subheader("Safety + Execute")
        gui_identity = run_identity_from_metadata(current_metadata())
        loaded_identity = load_run_identity(st.session_state.latest_run_dir)
        identity_ok = identities_match(gui_identity, loaded_identity)
        if st.session_state.latest_run_dir and loaded_identity and not identity_ok and not st.session_state.continue_existing_mismatch:
            st.error("Current GUI fields do not match the loaded run folder. Start a new run or intentionally resume the old run.")
            st.json(identity_diff(gui_identity, loaded_identity))
        st.write("Current run mode:", st.session_state.run_mode)
        st.write("Current run folder:", st.session_state.latest_run_dir or "No run loaded")
        st.write("Current action:", st.session_state.current_action)
        render_progress_panel()
        st.checkbox("Send clamp/Arduino jog_stop if available on ABORT", key="safety_abort_send_jog_stop")
        st.markdown("#### Emergency/Home enable checklists")
        st.session_state.safety_emergency_ready = safety_checklist("emergency_enable", ["Hands are clear.", "Object/tool area is safe.", "I understand this will move the clamp."])
        st.session_state.safety_home_ready = safety_checklist("home_enable", ["Clamp is released/open.", "Robot path to home is clear.", "I understand this will move the robot."])
        st.markdown("#### Selected Run Mode Checklist")
        if st.session_state.run_mode == "full_autonomous_home":
            run_ok = safety_checklist("full_autonomous_home", ["Workspace is clear.", "Robot path is clear.", "Clamp is open or safe to move.", "RealSense is running.", "Arduino clamp controller is connected.", "Emergency stop is accessible.", "I understand this will move robot and clamp."])
            action, confirmation, label = "full_final_home", "RUN_FINAL_HOME", "RUN FULL AUTONOMOUS + HOME"
        elif st.session_state.run_mode == "full_autonomous":
            run_ok = safety_checklist("full_autonomous", ["Workspace is clear.", "Robot path is clear.", "Clamp is open/safe.", "Emergency stop is accessible.", "I understand this will move the robot and clamp."])
            action, confirmation, label = "full_final", "RUN_FINAL", "RUN FULL AUTONOMOUS"
        elif st.session_state.run_mode == "clamp_only":
            run_ok = safety_checklist("clamp_only", ["Clamp path is clear.", "Manual/planned opening is correct.", "I understand this will move the clamp."])
            action, confirmation, label = "clamp_only", "RUN_CLAMP_ONLY", "RUN CLAMP ONLY"
            st.text_input("Manual clamp opening for clamp_only mm", value="", key="manual_clamp_opening")
        else:
            run_ok = safety_checklist("plan_perception", ["Correct material query.", "Correct specimen ID.", "Camera is ready if live frame is required."])
            action = "perception_only" if st.session_state.run_mode == "perception_only" else "plan_only"
            confirmation = ""
            label = "RUN PERCEPTION ONLY" if action == "perception_only" else "RUN PLAN ONLY"
        if mismatch and not st.session_state.continue_existing_mismatch:
            run_ok = False
        extra_args: list[str] = []
        if st.session_state.run_mode == "clamp_only" and st.session_state.get("manual_clamp_opening", "").strip():
            extra_args = ["--manual-opening-mm", st.session_state.manual_clamp_opening.strip()]
        if st.button(label, use_container_width=True, disabled=not run_ok or bool(st.session_state.is_running)):
            try:
                result = execute_action(action, confirmation, extra_args=extra_args)
                st.json(result)
                stages = load_json_if_exists(Path(st.session_state.latest_run_dir) / "t4_linked_outputs" / "copied" / "full_final_home_001.json").get("stages", [])
                if not stages:
                    stages = load_json_if_exists(Path(st.session_state.latest_run_dir) / "t4_linked_outputs" / "copied" / "full_final_001.json").get("stages", [])
                if stages:
                    st.write("Stage results")
                    st.table([{"stage": s.get("stage"), "success": s.get("success"), "failure_reason": s.get("failure_reason")} for s in stages])
            except Exception as exc:  # noqa: BLE001
                st.session_state.status = status_from_result(action, {"success": False, "failure_reason": str(exc)})
                save_error(exc)
                st.exception(exc)

    with tab_labels:
        st.subheader("Manual Labels")
        st.selectbox("Correct requested object?", ["NA", "yes", "no"], key="target_selection_correct")
        st.text_input("Manual selected class", value="", key="manual_selected_class")
        st.selectbox("If wrong, what happened?", ["NA", "wrong_class", "wrong_instance", "bad_mask", "bad_axis", "bad_depth", "bad_anchor", "robot_pose_bad", "clamp_width_bad", "UPV_reading_failed", "other"], key="wrong_failure_type")
        st.selectbox("Mask usable?", ["NA", "yes", "no"], key="mask_usable")
        st.selectbox("Axis correct?", ["NA", "yes", "no"], key="axis_correct")
        st.text_input("Manual major dimension mm", value="NA", key="manual_major_dimension_mm")
        st.text_input("Manual minor dimension mm", value="NA", key="manual_minor_dimension_mm")
        st.text_input("Manual actual clamp opening mm", value="NA", key="manual_actual_clamp_opening_mm")
        st.selectbox("Human intervention required?", ["NA", "yes", "no"], key="human_intervention_required")
        st.text_area("Manual notes", value="", key="free_text_notes")
        st.subheader("UPV Time-of-Flight")
        st.text_input("Time of flight", value="", key="tof_value")
        st.selectbox("TOF units", ["microseconds", "seconds"], key="tof_units")
        st.selectbox("Path length source", ["recommended_upv_path_length", "recommended_clamp_opening", "selected_probe_spacing", "manual_path_length"], key="path_source")
        st.text_input("Manual path length mm", value="", key="manual_path_length")
        st.selectbox("Reading quality", ["NA", "good", "noisy", "failed"], key="reading_quality")
        preview_run = Path(st.session_state.latest_run_dir) if st.session_state.latest_run_dir else None
        if preview_run:
            geometry = load_json_if_exists(preview_run / "geometry_results.json")
            clamp = load_json_if_exists(preview_run / "clamp_results.json")
            def _as_float(value: Any) -> float | None:
                try:
                    return None if value in (None, "", "NA") else float(value)
                except Exception:  # noqa: BLE001
                    return None
            path_len = None
            if st.session_state.path_source == "recommended_upv_path_length":
                path_len = _as_float(geometry.get("recommended_upv_path_length_mm")) or _as_float(clamp.get("recommended_upv_path_length_mm"))
            elif st.session_state.path_source == "recommended_clamp_opening":
                path_len = _as_float(clamp.get("recommended_clamp_opening_mm"))
            elif st.session_state.path_source == "selected_probe_spacing":
                path_len = _as_float(geometry.get("selected_probe_spacing_mm"))
            elif st.session_state.path_source == "manual_path_length":
                path_len = _as_float(st.session_state.manual_path_length)
            tof_raw = _as_float(st.session_state.tof_value)
            tof_s = None if tof_raw is None else (tof_raw * 1e-6 if st.session_state.tof_units == "microseconds" else tof_raw)
            st.info(f"path_length_mm used = {path_len if path_len is not None else 'NA'} | velocity_m_per_s = {compute_upv_velocity(tof_s, 'seconds', path_len)}")
        if st.button("Save UPV Reading", use_container_width=True):
            try:
                run_dir = save_or_create_run(allow_mismatch=bool(st.session_state.continue_existing_mismatch))
                result = compute_and_save_upv_results(run_dir, st.session_state.tof_value, st.session_state.tof_units, st.session_state.path_source, st.session_state.manual_path_length, st.session_state.reading_quality, notes="")
                st.success("Saved UPV reading.")
                st.json(result)
            except Exception as exc:  # noqa: BLE001
                save_error(exc)
                st.exception(exc)

    with tab_results:
        st.subheader("Latest Run Artifacts")
        run_dir = Path(st.session_state.latest_run_dir) if st.session_state.latest_run_dir else None
        if not run_dir:
            st.write("NA: no E1 run folder selected yet.")
        else:
            loaded_identity = load_run_identity(run_dir)
            if loaded_identity and not identities_match(run_identity_from_metadata(current_metadata()), loaded_identity):
                st.warning("Artifacts shown are from previous run.")
            manifest = load_json_if_exists(run_dir / "t4_linked_outputs" / "t4_output_manifest.json")
            run_record = load_json_if_exists(Path(str(manifest.get("latest_run_record_json", "")))) if manifest else {}
            copied_records = sorted((run_dir / "t4_linked_outputs" / "copied").glob("run_*_record.json")) if (run_dir / "t4_linked_outputs" / "copied").exists() else []
            if not run_record and copied_records:
                run_record = load_json_if_exists(copied_records[-1])
            artifact_class = run_record.get("requested_class") or run_record.get("material_query")
            if artifact_class and str(artifact_class) != str(st.session_state.material_query):
                st.warning(f"Artifact requested_class/material `{artifact_class}` does not match current GUI material query `{st.session_state.material_query}`.")
            prompt_source = load_json_if_exists(run_dir / "actual_prompt_source.json")
            if prompt_source.get("gui_t4_prompt_hash_match") is False:
                st.warning("WARNING: GUI prompts and T4 prompts do not match.")
            result_tabs = st.tabs(["Perception", "Geometry", "Robot Plan", "Clamp", "Prompt/VLM", "Depth Width", "Final Summary"])
            images = sorted(run_dir.glob("figures/*.png"))
            filters = {
                0: ["raw", "mask", "selected", "rgb"],
                1: ["axis", "geometry", "width"],
                2: ["plan", "pose", "robot"],
                3: ["clamp", "width"],
                5: ["depth_width", "depth", "width"],
            }
            for idx in [0, 1, 2, 3, 5]:
                with result_tabs[idx]:
                    matches = [path for path in images if any(token in path.name.lower() for token in filters.get(idx, []))]
                    if matches:
                        for path in matches[:5]:
                            st.image(str(path), caption=path.name, use_container_width=True)
                    else:
                        st.write("NA: image not available")
            with result_tabs[4]:
                st.json(load_json_if_exists(run_dir / "prompt_engineering_summary.json") or {"status": "NA: prompt summary not available"})
            with result_tabs[6]:
                st.json(load_json_if_exists(run_dir / "full_task_summary.json") or {"status": "NA: final summary not available"})

    with tab_logs:
        st.subheader("Logs / Debug")
        with st.expander("Recording options", expanded=False):
            recording_state = dict(st.session_state.get("recording_options", {}))
            recording_state["save_images"] = st.checkbox("save_images", value=bool(recording_state.get("save_images", True)))
            recording_state["save_overlays"] = st.checkbox("save_overlays", value=bool(recording_state.get("save_overlays", True)))
            recording_state["save_video"] = st.checkbox("save_video", value=bool(recording_state.get("save_video", False)))
            recording_state["save_rosbag"] = st.checkbox("save_rosbag", value=bool(recording_state.get("save_rosbag", False)))
            recording_state["video_fps"] = st.number_input("video_fps", value=int(recording_state.get("video_fps", config.get("video_fps", 10))))
            recording_state["max_video_duration_s"] = st.number_input("max_video_duration_s", value=int(recording_state.get("max_video_duration_s", config.get("max_video_duration_s", 180))))
            st.session_state.recording_options = recording_state
            st.caption("Video recording is honest-safe: if live video is not wired, E1 writes a requested/not-started JSON record. Rosbag is not started in dry-run.")
        if st.session_state.latest_run_dir:
            run_dir = Path(st.session_state.latest_run_dir)
            st.write("Run folder:", str(run_dir))
            st.json(load_json_if_exists(run_dir / "t4_linked_outputs" / "t4_output_manifest.json") or {"status": "NA: T4 output manifest not available"})
            err_path = run_dir / "logs" / "error_trace.txt"
            if err_path.exists():
                st.text_area("Latest error trace", value=err_path.read_text(encoding="utf-8"), height=220)
            log_files = sorted((run_dir / "logs").glob("t4_oneshot_*.log")) if (run_dir / "logs").exists() else []
            st.write("T4 stdout/stderr logs:", [str(path) for path in log_files[-10:]])
            st.write("Prompt manifest:", str(run_dir / "prompts" / "prompt_manifest.json"))
            st.write("Config snapshots:", [str(path) for path in sorted((run_dir / "configs").glob("*"))[:20]] if (run_dir / "configs").exists() else [])
        st.text_area("Event log", value="\n".join(st.session_state.logs[-250:]), height=260)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 Streamlit GUI and dry-run validator for autonomous UPV experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to E1 config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Create a dry-run E1 output folder without Streamlit or hardware.")
    parser.add_argument("--validate-config", help="Validate config and create a dry-run E1 output folder.")
    parser.add_argument("--gui-state-dry-run", action="store_true", help="Create a timber run and verify identity mismatch detection without hardware.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.validate_config or args.config)
    try:
        config = load_config(config_path)
        if args.gui_state_dry_run:
            gui_state_dry_run(config_path, config)
            return 0
        if args.dry_run or args.validate_config:
            validate_config(config_path, config)
            return 0
        launch_streamlit_gui_v2(config_path, config)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
