from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from upv_vlm_v2.data.models import json_ready  # noqa: E402
from upv_vlm_v2.pipeline.main_pipeline import run_full_main_upv_vlm_v2_pipeline  # noqa: E402
from upv_vlm_v2.pipeline.modes import MODE_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UPV_VLM_v2 main pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--requested-material", required=True)
    parser.add_argument("--axis-mode", choices=["major", "minor"], default="major")
    parser.add_argument("--mode", choices=sorted(MODE_ORDER), default="dry_run")
    parser.add_argument("--input-rgb")
    parser.add_argument("--input-depth")
    parser.add_argument("--input-camera-info")
    parser.add_argument("--live-ros2", action="store_true")
    parser.add_argument("--execution-backend", choices=["none", "simulated", "real"])
    parser.add_argument("--allow-real-hardware", action="store_true")
    parser.add_argument("--mock-qwen-response")
    parser.add_argument("--output-root")
    parser.add_argument("--confirm")
    parser.add_argument(
        "--anchor-vlm-backend",
        choices=[
            "llama32_single_anchor_l6_taxonomy_scoring",
            "qwen32_multi_anchor_l6_batch_ranking",
            "v2a_qwen32_single_anchor_contact_scoring",
        ],
    )
    parser.add_argument("--vlm-server-url")
    parser.add_argument("--vlm-server-endpoint")
    parser.add_argument("--anchor-prompt-variant")
    parser.add_argument("--anchor-layout-variant")
    parser.add_argument("--llama-min-selectable-score", type=float)
    parser.add_argument("--anchor-vlm-max-new-tokens", type=int)
    parser.add_argument("--anchor-vlm-temperature", type=float)
    parser.add_argument("--print-result-json", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_summary(result, *, summary_only: bool = False) -> None:
    session = Path(result.session_dir)
    anchor = result.anchors.get(result.axis_mode) if result.anchors else None
    path_length = result.path_lengths.get(result.axis_mode) if result.path_lengths else None
    plan = result.robot_plan
    execution = result.execution
    print("UPV_VLM_v2 run summary")
    print(f"session: {result.session_dir}")
    print(f"mode: {result.mode}")
    print(f"backend: {(execution.diagnostics or {}).get('backend') if execution else 'none'}")
    print(f"success: {result.success}")
    print(f"failure_reason: {result.failure_reason}")
    if result.target:
        target_diag = result.target.diagnostics or {}
        selected_id = result.target.selected_candidate_id
        selected_score = None
        if selected_id:
            selected_candidate = (result.target.candidate_artifacts or {}).get(selected_id, {})
            selected_score = selected_candidate.get("requested_material_score", selected_candidate.get("clip_score_for_requested"))
        print("")
        print("Perception:")
        if not result.target.success or selected_id == "NO_VERIFIED_MATCH":
            print("Target:")
            print("status: NO_VERIFIED_MATCH")
            print(f"requested_material: {result.target.requested_material}")
            print(f"reason: {target_diag.get('target_absent_rejection_reason') or result.target.failure_reason or target_diag.get('failure_reason')}")
            print(f"candidate_count: {result.target.candidate_count}")
            print(f"crop_verification_panel: {target_diag.get('crop_verification_panel_path')}")
            return
        print(f"selected object: {result.target.requested_material}")
        print(f"selected candidate: {result.target.selected_candidate_id}")
        print(f"mask: {'OK' if result.target.selected_mask_path else 'NA'}")
        print("Target verification:")
        print(f"candidate_count: {result.target.candidate_count}")
        print(f"selected_material_score: {_fmt(selected_score)}")
        print(f"crop_verification_panel: {target_diag.get('crop_verification_panel_path')}")
    if result.geometry:
        print(f"centroid_px: {result.geometry.centroid_px}")
        print(f"major_axis_deg: {_fmt(result.geometry.major_axis_angle_deg)}")
        print(f"minor_axis_deg: {_fmt(result.geometry.minor_axis_angle_deg)}")
    if anchor:
        diag = anchor.diagnostics or {}
        artifacts = anchor.candidate_artifacts or {}
        robot_geom = diag.get("robot_anchor_geometry") if isinstance(diag, dict) else None
        print("")
        print("Anchor:")
        print(f"anchor_backend: {diag.get('anchor_backend') if isinstance(diag, dict) else 'NA'}")
        print(f"qwen_required: {diag.get('qwen_required') if isinstance(diag, dict) else 'NA'}")
        print(f"axis_mode: {result.axis_mode}")
        print(f"anchor_count_generated: {anchor.candidate_count}")
        print(f"deterministic_survivors: {diag.get('deterministic_survivors') if isinstance(diag, dict) else 'NA'}")
        print(f"qwen_selected_anchor: {diag.get('qwen_selected_anchor_id') if isinstance(diag, dict) else 'NA'}")
        print(f"selected_anchor: {anchor.final_anchor_id}")
        print(f"final_anchor_source: {diag.get('final_anchor_source') if isinstance(diag, dict) else 'NA'}")
        print(f"anchor_px: {anchor.selected_anchor_px}")
        if isinstance(robot_geom, dict):
            print(f"robot_candidate_anchor_center_px: {robot_geom.get('robot_candidate_anchor_center_px')}")
        print(f"contact_a_px: {anchor.contact_point_a_px}")
        print(f"contact_b_px: {anchor.contact_point_b_px}")
        print(f"anchor_score: {_fmt(anchor.score)}")
        print(f"wide_contact_grid: {artifacts.get('combined_wide_contact_guided_grid_2col') or artifacts.get('combined_wide_contact_guided_grid')}")
        print(f"qwen_response: {artifacts.get('qwen_anchor_decision')}")
        print(f"robot_anchor_geometry: {artifacts.get('robot_anchor_geometry')}")
    if path_length:
        print("")
        print("Path length:")
        print(f"mask_path_length_mm: {_fmt(path_length.mask_path_length_mm)}")
        print(f"depth_path_length_mm: {_fmt(path_length.depth_path_length_mm)}")
        print(f"depth_valid: {path_length.depth_valid}")
        if path_length.depth_mask_disagreement_ratio is not None:
            print(f"depth/mask disagreement: {100.0 * path_length.depth_mask_disagreement_ratio:.2f}%")
    if plan:
        payload = plan.diagnostics or {}
        print("")
        print("Robot plan:")
        print(f"motion_target_source: {payload.get('motion_target_source')}")
        print(f"anchor_camera_xyz_m: {payload.get('anchor_camera_xyz_m')}")
        print(f"anchor_base_xyz_m: {payload.get('anchor_base_xyz_m', 'deferred_until_real_execute')}")
        print(f"hover_pose_base: {plan.hover_pose_base}")
        print(f"approach_pose_base: {plan.approach_pose_base}")
        print(f"final_pose_base: {plan.final_pose_base}")
        print(f"requires_real_executor_rebuild: {payload.get('requires_real_executor_rebuild', payload.get('diagnostics', {}).get('requires_real_executor_rebuild'))}")
        print(f"clamp_opening_mm: {_fmt(plan.clamp_opening_mm)}")
        print(f"upv_path_length_mm: {_fmt(plan.upv_path_length_mm)}")
    if execution:
        print("")
        print("Execution:")
        print("sequence: arduino_check -> move_midhover -> orient -> xy -> approach_preview -> approach_final -> clamp -> hold -> release -> home")
        print(f"status: {'SUCCESS' if execution.success else 'FAILED'}")
        print(f"failure_reason: {execution.failure_reason}")
    if not summary_only:
        print("")
        print("Artifacts:")
        print(f"pipeline_result: {session / 'pipeline_result.json'}")
        print(f"stage_status: {session / 'stage_status.json'}")
        if result.target and result.target.selected_mask_overlay_path:
            print(f"selected_mask_overlay: {result.target.selected_mask_overlay_path}")
        if anchor and anchor.overlay_paths:
            print(f"anchor_overlay: {anchor.overlay_paths.get('selected_anchor_overlay') or anchor.overlay_paths}")
        if path_length and path_length.overlay_paths:
            print(f"local_chord_overlay: {path_length.overlay_paths.get('local_path_length_overlay') or path_length.overlay_paths}")
        if execution:
            print(f"execution_log: {execution.execution_log_path}")


def main() -> int:
    args = parse_args()
    anchor_overrides = {
        "backend": args.anchor_vlm_backend,
        "vlm_server_url": args.vlm_server_url,
        "vlm_server_endpoint": args.vlm_server_endpoint,
        "prompt_variant": args.anchor_prompt_variant,
        "anchor_layout_variant": args.anchor_layout_variant,
        "llama_min_selectable_score": args.llama_min_selectable_score,
        "max_new_tokens": args.anchor_vlm_max_new_tokens,
        "temperature": args.anchor_vlm_temperature,
    }
    result = run_full_main_upv_vlm_v2_pipeline(
        config_path=args.config,
        requested_material=args.requested_material,
        axis_mode=args.axis_mode,
        mode=args.mode,
        input_rgb=args.input_rgb,
        input_depth=args.input_depth,
        input_camera_info=args.input_camera_info,
        live_ros2=args.live_ros2,
        execution_backend=args.execution_backend,
        allow_real_hardware=args.allow_real_hardware,
        mock_qwen_response=args.mock_qwen_response,
        output_root=args.output_root,
        confirm=args.confirm,
        anchor_selection_overrides=anchor_overrides,
    )
    if not args.quiet:
        _print_summary(result, summary_only=args.summary_only)
    if args.print_result_json and args.verbose_json:
        print(json.dumps(json_ready(result), indent=2))
    elif args.print_result_json and not args.quiet:
        print(f"Full JSON saved at: {Path(result.session_dir) / 'pipeline_result.json'}")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
