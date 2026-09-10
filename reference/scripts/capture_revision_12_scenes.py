#!/usr/bin/env python3
"""Interactive revision scene capture; no perception or robot execution."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


# Brick/timber/concrete order. Minimum total condition changes from MMM: 16.
SCENES = ["MMM", "MOP", "MMP", "OMP", "OOO", "MPO",
          "PPP", "PPO", "PMO", "POM", "OOM", "OPM"]
MATERIALS = ("brick", "timber", "concrete")
CONDITIONS = {"M": "muddy", "O": "oily", "P": "partially occluded"}
ROOT = Path(__file__).resolve().parents[1]


def open_image(path):
    print(f"Captured image: {path}", flush=True)
    for command in (["xdg-open", str(path)], ["gio", "open", str(path)]):
        if not shutil.which(command[0]):
            continue
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                if process.wait(timeout=0.5) != 0:
                    continue
            except subprocess.TimeoutExpired:
                pass
            return
        except OSError:
            continue
    print(f"Could not open automatically. Open manually: {path}")


def prompt(message, allowed):
    while True:
        value = input(message).strip().lower()
        if value in allowed:
            return value
        print("Choose: " + ", ".join(allowed))


def write_manifests(session):
    rows = []
    trials = []
    for index, codes in enumerate(json.loads((session / "scene_plan.json").read_text()), 1):
        case_id = f"case_{index:03d}"
        case = session / case_id
        metadata = case / "scene_metadata.json"
        if not metadata.exists():
            continue
        row = json.loads(metadata.read_text())
        rows.append(row)
        for material, code in zip(MATERIALS, codes):
            trials.append(dict(scene_id=case_id, requested_material=material,
                               condition=CONDITIONS[code], raw_case_dir=str(case),
                               rgb_path=str(case / "raw_rgb.png")))
    for name, data in (("scene_manifest", rows), ("perception_trials", trials)):
        (session / f"{name}.json").write_text(json.dumps(data, indent=2))
        with (session / f"{name}.csv").open("w", newline="") as stream:
            if data:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_multi_anchor_l6.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/v2_datasets/revision_12_scenes")
    parser.add_argument("--backend", choices=["ros2", "direct"], default="ros2",
                        help="ROS2 uses an already-running camera; direct requests 1280x720 at 30 FPS.")
    parser.add_argument("--capture-retries", type=int, default=2)
    parser.add_argument("--resume-session", type=Path)
    parser.add_argument("--no-open-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print scene plan without imports or camera access.")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Config not found: {args.config}")
    if args.capture_retries < 1:
        parser.error("--capture-retries must be positive")
    if args.dry_run:
        for index, codes in enumerate(SCENES, 1):
            print(f"Scene {index:02}: " + ", ".join(f"{m}: {CONDITIONS[c]}" for m, c in zip(MATERIALS, codes)))
        print("12 unique scenes; 36 queries. No camera opened.")
        return

    from capture_10_live_realsense_snapshots_no_robot_motion import (
        _capture_case_direct, _capture_case_ros2, _session_dir,
        _timestamp, load_config, snapshot_yaml,
    )
    from PIL import Image
    import numpy as np

    config = load_config(args.config.resolve())
    if args.resume_session:
        session = args.resume_session.resolve()
        plan_file = session / "scene_plan.json"
        if not plan_file.exists():
            parser.error("Resume session must contain this script's scene_plan.json")
        scenes = json.loads(plan_file.read_text())
        if not isinstance(scenes, list) or sorted(scenes) != sorted(SCENES):
            parser.error("Resume scene plan does not contain the expected 12 scenes")
        print("Resuming the original saved scene order.")
    else:
        session = _session_dir(args.output_root.resolve())
        (session / "scene_plan.json").write_text(json.dumps(SCENES, indent=2))
        snapshot_yaml(config, session / "run_config_snapshot.yaml")
        scenes = SCENES
    print(f"Session: {session}", flush=True)
    print("ROS2 resolution follows the camera node; intended RGB-D resolution is 1280x720.")
    try:
        for index, codes in enumerate(scenes, 1):
            case_id = f"case_{index:03d}"
            case = session / case_id
            if (case / "scene_metadata.json").exists():
                print(f"Already saved: {case_id}")
                continue
            if case.exists():
                raise RuntimeError(f"Unrecognized existing case directory: {case}")
            while True:
                print(f"\nScene {index}/12 ({case_id})")
                for material, code in zip(MATERIALS, codes):
                    print(f"  {material.title()}: {CONDITIONS[code]}")
                if index > 1:
                    changes = [f"{m}: {CONDITIONS[old]} -> {CONDITIONS[new]}"
                               for m, old, new in zip(MATERIALS, scenes[index - 2], codes)
                               if old != new]
                    print("Changes from previous scene: " + "; ".join(changes))
                if prompt("Set up scene. Enter to capture / q to quit: ", ["", "q"]) == "q":
                    return
                attempts_root = session / "capture_attempts" / case_id
                attempts_root.mkdir(parents=True, exist_ok=True)
                attempt = attempts_root / f"attempt_{len(list(attempts_root.iterdir())) + 1:03d}"
                attempt.mkdir()
                try:
                    if args.backend == "ros2":
                        result = _capture_case_ros2(config, attempt, args.capture_retries)
                    else:
                        result = _capture_case_direct(attempt, args.capture_retries)
                    if not result.get("success"):
                        raise RuntimeError(result.get("failure_reason", "Capture failed"))
                    with Image.open(attempt / "raw_rgb.png") as image:
                        width, height = image.size
                    depth = np.load(attempt / "depth_raw.npy", allow_pickle=False)
                    if depth.shape != (height, width) or not np.any(np.isfinite(depth) & (depth > 0)):
                        raise RuntimeError("Depth must be aligned to RGB and contain valid points")
                    print(f"Captured RGB {width}x{height}; depth {depth.shape}, {depth.dtype}")
                    if (width, height) != (1280, 720):
                        print("Resolution differs from intended 1280x720; retake after adjusting the camera node if needed.")
                except Exception as exc:
                    print(f"Capture failed: {exc}. Attempt kept at {attempt}")
                    continue
                if not args.no_open_images:
                    open_image(attempt / "raw_rgb.png")
                while True:
                    action = prompt("[s] Save and next / [r] Retake / [v] View / [q] Quit: ", ["s", "r", "v", "q"])
                    if action == "v":
                        open_image(attempt / "raw_rgb.png")
                        continue
                    break
                if action == "q":
                    return
                if action == "r":
                    continue
                shutil.copytree(attempt, case)
                metadata = dict(scene_id=case_id, scene_index=index,
                                brick_condition=CONDITIONS[codes[0]],
                                timber_condition=CONDITIONS[codes[1]],
                                concrete_condition=CONDITIONS[codes[2]],
                                rgb_width=width, rgb_height=height,
                                backend=args.backend, saved_at=_timestamp(), raw_case_dir=str(case))
                (case / "scene_metadata.json").write_text(json.dumps(metadata, indent=2))
                write_manifests(session)
                print(f"Saved: {case}", flush=True)
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Saved scenes retained.")
    finally:
        write_manifests(session)
        print(f"Output: {session}\nResume with --resume-session {session}")


if __name__ == "__main__":
    main()
