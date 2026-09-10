#!/usr/bin/env python3
"""Saved RGB target selection and manual review for the 12 revision scenes."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from capture_revision_12_scenes import ROOT, open_image


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def summarize(output):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("trials/*/manual_label.json"))]
    with (output / "manual_results.csv").open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summaries = []
    for group in ["all", "brick", "timber", "concrete block", "muddy", "oily", "partially occluded"]:
        subset = [r for r in rows if group == "all" or group in (r["requested_material"], r["condition"])]
        summaries.append(dict(group=group, n_labeled=len(subset),
                              n_correct=sum(r["correct"] for r in subset),
                              accuracy=sum(r["correct"] for r in subset) / len(subset) if subset else None))
    save_json(output / "summary.json", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-session", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2/deploy_ur3e_realsense_upv_real_plan_check_qwen32_multi_anchor_l6.yaml")
    parser.add_argument("--output-dir", type=Path, help="Reuse this directory to resume without repeating completed inference/labels.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-open-images", action="store_true")
    parser.add_argument("--label-only", action="store_true", help="Review existing inference; never load models.")
    parser.add_argument("--max-trials", type=int)
    args = parser.parse_args()
    source = args.capture_session.resolve()
    manifest = source / "scene_manifest.json"
    if not manifest.is_file() or not args.config.is_file():
        parser.error("Capture scene_manifest.json and config must exist")
    scenes = json.loads(manifest.read_text())
    ids = [s["scene_id"] for s in scenes]
    if len(ids) != 12 or len(set(ids)) != 12:
        parser.error(f"Expected 12 distinct saved scenes; found {len(ids)}")
    trials = []
    for scene in scenes:
        rgb = source / scene["scene_id"] / "raw_rgb.png"
        if not rgb.is_file():
            parser.error(f"Missing RGB: {rgb}")
        for material in ("brick", "timber", "concrete block"):
            key = "concrete" if material == "concrete block" else material
            trials.append(dict(scene_id=scene["scene_id"], requested_material=material,
                               condition=scene[f"{key}_condition"], rgb_path=str(rgb)))
    if args.max_trials is not None:
        if args.max_trials < 1:
            parser.error("--max-trials must be positive")
        trials = trials[:args.max_trials]
    if args.dry_run:
        for trial in trials:
            print(trial)
        print(f"Validated {len(trials)} trials. No model imports or inference.")
        return
    if args.label_only and not args.output_dir:
        parser.error("--label-only requires --output-dir")
    output = (args.output_dir or ROOT / "outputs/revision_36_perception" / datetime.now().strftime("session_%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from upv_vlm_v2.config.loader import load_config, snapshot_yaml
    config = load_config(args.config.resolve())
    config.setdefault("target_selection", {})["use_clip_crop_verifier"] = True
    config.setdefault("testing", {})["synthetic_target_from_saved_input"] = False
    signature = dict(capture_session=str(source), config=config)
    run_file = output / "run_config.json"
    if run_file.exists() and json.loads(run_file.read_text()) != signature:
        parser.error("Output belongs to a different input/config; choose a new output directory")
    save_json(run_file, signature)
    snapshot_yaml(config, output / "config_snapshot.yaml")
    print(f"Output: {output}", flush=True)
    try:
        for number, trial in enumerate(trials, 1):
            name = trial["scene_id"] + "_" + trial["requested_material"].replace(" ", "_")
            folder = output / "trials" / name
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "manual_label.json").exists():
                continue
            print(f"\n[{number}/{len(trials)}] {name} ({trial['condition']})", flush=True)
            result_file = folder / "review_result.json"
            if result_file.exists():
                saved = json.loads(result_file.read_text())
            elif args.label_only:
                print("No saved inference; skipped")
                continue
            else:
                from upv_vlm_v2.perception.target_selection_stage import run_target_selection
                started = time.perf_counter()
                try:
                    result = run_target_selection(rgb_path=trial["rgb_path"], requested_material=trial["requested_material"], config=config, output_dir=folder / "perception")
                except Exception as exc:
                    save_json(folder / "inference_error.json", dict(error=str(exc)))
                    print(f"Inference error: {exc}. Rerun to retry.")
                    continue
                saved = dict(result=result, elapsed_sec=time.perf_counter() - started)
                save_json(result_file, saved)
            result = saved["result"]
            print(f"Selected candidate: {result.get('selected_candidate_id')}; reason: {result.get('failure_reason')}")
            overlay = Path(result.get("selected_mask_overlay_path") or trial["rgb_path"])
            print(f"Review image: {overlay}")
            if not result.get("selected_mask_path"):
                print("No final selected mask. The review image is the raw scene.")
            if not args.no_open_images:
                open_image(overlay)
            choices = {"b": "brick", "t": "timber", "c": "concrete block", "n": "none", "a": "ambiguous"}
            while True:
                answer = input("Actual selected object: [b] brick / [t] timber / [c] concrete / [n] none / [a] ambiguous / [v] view / [skip] later / [q] quit: ").strip().lower()
                if answer == "q":
                    return
                if answer == "v":
                    open_image(overlay)
                    continue
                if answer == "skip":
                    break
                if answer not in choices:
                    continue
                label = choices[answer]
                notes = input("Notes (Enter to leave blank): ").strip()
                row = dict(**trial, selected_candidate_id=result.get("selected_candidate_id"),
                           pipeline_success=result.get("success"), failure_reason=result.get("failure_reason"),
                           actual_selected_class=label, correct=label == trial["requested_material"],
                           elapsed_sec=saved["elapsed_sec"], notes=notes)
                save_json(folder / "manual_label.json", row)
                summarize(output)
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; progress saved.")
    finally:
        summarize(output)
        print(f"Results: {output}\nResume with the same --output-dir.")


if __name__ == "__main__":
    main()
