#!/usr/bin/env python3
"""Opt-in staged Qwen NF4 complete UPV cycle with existing manual PL-200 entry."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_multi_anchor_l6.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE)
    parser.add_argument("--model-path", type=Path, default=ROOT / "local_models/Qwen2.5-VL-32B-Instruct-bnb-4bit")
    parser.add_argument("--qwen-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--material", choices=["brick", "timber", "concrete block"], default="brick")
    parser.add_argument("--axis", choices=["major", "minor"], default="major")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument("--prefetch-weights", action="store_true", help="Warm Qwen CPU file cache during perception; no concurrent GPU models")
    parser.add_argument("--hold-sec", type=float, default=5.0)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/prequantized_upv_cycles")
    parser.add_argument("--case-dir", type=Path, default=ROOT / "datasets/gpu_smoke/concrete")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Real full cycle; requires terminal confirmation each cycle")
    mode.add_argument("--plan-check", action="store_true", help="Live camera/inference/planning, no robot or actuator commands")
    mode.add_argument("--dry-run", action="store_true", help="Validate only (also the default)")
    mode.add_argument("--gpu-simulation", action="store_true", help="Real GPU inference on saved RGB-D; simulated robot/clamp, no ROS2 or hardware")
    args = parser.parse_args()
    if not 0 < args.hold_sec <= 30 or args.max_cycles < 1:
        parser.error("hold-sec must be in (0,30]; max-cycles must be positive")
    import yaml
    config = yaml.safe_load(args.config.read_text())
    os.chdir(ROOT)
    if config["anchor_selection"]["backend"] != "qwen32_multi_anchor_l6_batch_ranking":
        parser.error("Use the multi-anchor L6 main pipeline config")
    for path in (args.qwen_python, args.model_path / "config.json",
                 ROOT / "scripts/start_qwen25_32b_bnb4_contact_server.py",
                 Path(config["tool"]["camera_to_tcp_transform_file"]),
                 Path(config["tool"]["upv_tool_geometry_file"]),
                 Path(config["planning"]["workspace_limits_file"])):
        if not path.exists():
            parser.error(f"Required file missing: {path}")
    model_config = json.loads((args.model_path / "config.json").read_text())
    if model_config.get("model_type") != "qwen2_5_vl" or not model_config.get("quantization_config", {}).get("load_in_4bit"):
        parser.error("Expected prequantized Qwen2.5-VL 4-bit checkpoint")
    index = json.loads((args.model_path / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (args.model_path / shard).is_file():
            parser.error(f"Missing model shard: {shard}")
    source = ROOT / "reference/src" if (ROOT / "reference/src/upv_vlm_v2").exists() else ROOT / "src"
    env = os.environ.copy()
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env.pop("HF_HUB_CACHE", None)
    env.pop("TRANSFORMERS_CACHE", None)
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    # Do not resolve a virtualenv Python symlink to the system interpreter.
    config["managed_qwen"] = dict(enabled=True, port=args.port, python=str(args.qwen_python.absolute()),
        launcher=str(ROOT / "scripts/start_qwen25_32b_bnb4_contact_server.py"),
        model_path=str(args.model_path.resolve()), startup_timeout_sec=180,
        prefetch_weights=args.prefetch_weights, prefetch_reserve_gib=8)
    config["anchor_selection"].update(qwen_server_url=f"http://127.0.0.1:{args.port}",
        qwen_server_endpoint="/infer_multi", qwen_prompt_variant="multi_image_v1_original",
        selection_policy="highest_scoring_usable_strict")
    config["perception"]["gsam2_repo_dir"] = str((ROOT / config["perception"]["gsam2_repo_dir"]).resolve())
    config["perception"]["sam2_checkpoint"] = str((ROOT / config["perception"]["sam2_checkpoint"]).resolve())
    config["execution"].update(clamp_hold_sec=args.hold_sec, return_home_after_execution=True)
    config["clamp"].update(enabled=True, required=True, allow_missing_arduino=False)
    # Retain the rig's calibration and conservative speeds; never enable hardware in the saved config.
    config["execution"].update(backend="none", allow_real_hardware=False)
    config["robot"]["enabled"] = False
    if args.plan_check or args.gpu_simulation:
        config["robot"]["home_before_capture_enabled"] = False
    if args.gpu_simulation:
        case = args.case_dir.resolve()
        for name in ("raw_rgb.png", "raw_depth_aligned_z16.png", "camera_info_aligned_depth.json"):
            if not (case / name).is_file():
                parser.error(f"Missing saved input: {case / name}")
        config["camera"]["allow_live_capture"] = False
        config["execution"]["backend"] = "simulated"
        config["planning"]["simulation_enabled"] = True
        config["robot"]["robot_ip"] = None
        config["clamp"]["serial_port"] = None
        print("Saved RGB-D -> real DINO/SAM2/CLIP -> real Qwen -> path/plan -> SIMULATED approach/clamp/hold/release/home")
        print("No ROS2 capture, real robot, Arduino or UPV acquisition. No fabricated UPV reading.")
    else:
        print("Sequence: home (execute only) -> RGB-D -> DINO/SAM2 -> CLIP -> geometry ->")
        print("start NF4 Qwen -> /infer_multi -> stop Qwen -> path/plan ->")
        print(f"approach/clamp/hold {args.hold_sec}s/release/home (execute only) -> manual PL-200 terminal entry")
    print("No automatic UPV acquisition. Stop other GPU servers first; no external process is killed.")
    print(json.dumps(config["managed_qwen"], indent=2))
    if not args.execute and not args.plan_check and not args.gpu_simulation:
        print("Validation only: no models, camera, network or robot were started.")
        return
    if args.gpu_simulation:
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this terminal; no inference or hardware started")
    if args.execute and not sys.stdin.isatty():
        parser.error("Real execution requires an interactive terminal")
    out = args.output_root.resolve() / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config["logging"]["output_root"] = str(out / "pipeline")
    snapshot = out / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.execute:
        command = [sys.executable, "-m", "upv_vlm_v2.cli.run_interactive_upv_robot_session",
            "--pipeline-config", str(snapshot), "--output-root", str(out / "manual_readings"),
            "--default-material", args.material, "--default-axis", args.axis,
            "--max-cycles", str(args.max_cycles), "--confirm-each-cycle", "--no-confirm-once"]
    elif args.gpu_simulation:
        command = [sys.executable, "-m", "upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline",
            "--config", str(snapshot), "--requested-material", args.material, "--axis-mode", args.axis,
            "--input-rgb", str(case / "raw_rgb.png"),
            "--input-depth", str(case / "raw_depth_aligned_z16.png"),
            "--input-camera-info", str(case / "camera_info_aligned_depth.json"),
            "--mode", "execute", "--execution-backend", "simulated",
            "--confirm", config["safety"]["execute_confirm_text"], "--summary-only"]
    else:
        command = [sys.executable, "-m", "upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline",
            "--config", str(snapshot), "--requested-material", args.material, "--axis-mode", args.axis,
            "--live-ros2", "--mode", "plan_only", "--execution-backend", "none", "--summary-only"]
    print("Artifacts:", out, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
