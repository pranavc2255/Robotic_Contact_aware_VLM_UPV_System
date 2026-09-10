# Runtime Substage Split Audit

Date: 2026-07-09  
Branch: `robot_upv_interactive_20260527`

## Purpose

This audit checks whether existing E45/final UPV-VLM runtime artifacts already split the execution and inference runtime into the requested paper subcomponents. It is a documentation/data audit only. No model inference, RealSense capture, Qwen/VLM server, GSAM2/SAM2/GroundingDINO/CLIP, RTDE, Arduino, UPV hardware, or robot motion was run.

## Executive Conclusion

The repository contains reliable measured timings for the full pipeline and the main pipeline stages:

- `total_pipeline_timing_ms`
- `target_selection_timing_ms`
- `geometry_timing_ms`
- `anchor_selection_timing_ms`
- `path_length_timing_ms`
- `robot_plan_timing_ms`
- `execution_timing_ms`

The repository also contains structured robot execution event logs showing the execution sequence: mid-hover, orientation, XY approach, preview approach, final approach, clamp, hold, release, and return home. However, those per-step execution rows do **not** contain measured start/stop timestamps or elapsed times. Therefore, approach motion, final approach, clamping, holding, release, and return-home durations are not directly logged as separate measured values.

For paper reporting, the safe timing table should use measured full-pipeline and major-stage timings. Robot execution should be reported as one combined measured stage unless new instrumentation is added.

## Requested Subcomponent Availability

| Requested subcomponent | Availability | Exact fields found | Representative artifact/source | Paper-safe conclusion |
|---|---:|---|---|---|
| Robot approach motion to selected UPV contact pose | Structured event exists, measured duration not available | Execution stages: `move_midhover`, `orient`, `xy`, `approach_preview`, `approach_final`; speed/accel and before/after TCP poses | `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164704/01_cycles/cycle_001/pipeline_full_session/artifacts/07_execution/execution_summary.json`; implementation `src/upv_vlm_v2/execution/t4_style_executor.py:L216-L232` | Do not report measured approach time separately. It is included inside combined `execution_timing_ms`. |
| Return-to-home motion after measurement | Structured event exists, measured duration not available | Execution stage `home`; `home_success`, `home_returned`, `home_success_after_execution` | Same execution summary; implementation `src/upv_vlm_v2/execution/t4_style_executor.py:L275-L305` | Do not report measured home-return time separately. It is included inside combined `execution_timing_ms`. |
| UPV transducer actuation / clamping time | Command logs exist; measured elapsed time not available | `clamp_to_planned_width`; clamp log fields `estimated_move_time_s`, `timeout_used_s`, command responses | `.../artifacts/07_execution/real_clamp_execution_log.json`; implementation `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L135-L177` | Only estimated clamp move time is present. Do not present it as measured runtime. |
| UPV transducer release / unclamping time | Command logs exist; measured elapsed time not available | `clamp_release`; `OPEN_FULL`; `estimated_move_time_s`, `timeout_used_s`, command responses | `.../real_clamp_execution_log.json`; implementation `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L135-L143` and `src/upv_vlm_v2/execution/t4_style_executor.py:L258-L262` | Only estimated release/open time is present. Do not present it as measured runtime. |
| UPV holding time during signal acquisition | Requested hold duration exists; measured acquisition duration not available | `clamp_hold`; `HOLD_MS 5000`; `hold_ms`; `time_of_flight_us` in UPV tables | `.../real_clamp_execution_log.json`; implementation `src/upv_vlm_v2/execution/t4_style_executor.py:L253-L257`, `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L179-L181` | Requested hold time is logged, but not a measured elapsed acquisition time. Time-of-flight is signal content, not runtime duration. |
| VLM inference time | Available in two forms: full anchor stage and nested per-request latency for some backends | Full stage: `anchor_selection_timing_ms`; nested request: `elapsed_ms` in VLM decision JSONs | E45 master CSVs; `qwen32_single_anchor_decision.json`; prior audit `docs/dev_history/contact_selection_vlm_runtime_methods_audit_20260706_031048.md:L22-L23`, `L45-L62` | Use `anchor_selection_timing_ms` for full stage. Use nested `elapsed_ms` only as per-request HTTP/VLM latency, not full stage time. |
| GroundingDINO/SAM2 + CLIP target-selection time | Combined target stage available; partial Grounded-SAM2 runtime diagnostics often available; CLIP separate elapsed time not consistently logged in E45 | `target_selection_timing_ms`; diagnostics `grounded_sam2.runtime_s`; CLIP score artifacts identify verifier/model but no consistent per-run elapsed split | `timing_summary.json`; `target_selection_result.json`; `material_verification/material_scores.json`; source `src/upv_vlm_v2/pipeline/stages.py:L115-L168`; prior audit `docs/dev_history/final_pipeline_runtime_summary_model_residency_audit_20260706_011350.md:L116-L140` | Report combined target selection as measured. Do not split GroundingDINO, SAM2, and CLIP unless instrumented. |
| Remaining code time: geometry, path length, planning, file handling, overhead | Partly logged and partly residual | Direct: `geometry_timing_ms`, `path_length_timing_ms`, `robot_plan_timing_ms`; residual can be computed from total minus logged stages | E45 master CSVs and `timing_summary.json`; source `src/upv_vlm_v2/pipeline/main_pipeline.py:L524-L633` | Geometry/path/planning are safe direct rows. File handling/other overhead is only a residual estimate, not a separately logged stage. |

## Fields Found In E45 Runtime Artifacts

The E45 master CSV files and per-cycle timing summaries contain the major stage timings. Representative CSV columns include:

```text
total_pipeline_timing_ms
target_selection_timing_ms
geometry_timing_ms
anchor_selection_timing_ms
path_length_timing_ms
robot_plan_timing_ms
execution_timing_ms
```

Representative source:

- `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164704/02_tables/e45_all_cycles_master.csv`
- `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164704/01_cycles/cycle_001/pipeline_full_session/timing_summary.json`

Representative `timing_summary.json` has the following stage entries:

```text
pre_capture_home: 1090.199 ms
capture: 483.385 ms
target_selection: 10092.419 ms
geometry: 224.618 ms
anchor_selection: 29636.150 ms
path_length: 877.195 ms
robot_plan: 6.919 ms
execution: 63128.767 ms
total: 106571.298 ms
```

The previous final runtime summary aggregated the final E45 rows as 94 successful readings:

- `target_selection_timing_ms`: mean `8.789 s`
- `geometry_timing_ms`: mean `0.277 s`
- `anchor_selection_timing_ms`: mean `25.316 s`
- `path_length_timing_ms`: mean `0.913 s`
- `robot_plan_timing_ms`: mean `0.007 s`
- `execution_timing_ms`: mean `60.037 s`
- `total_pipeline_timing_ms`: mean `98.877 s`

Source: `docs/dev_history/final_pipeline_runtime_summary_20260706_005311.md:L63-L69`.

## Code Timing Boundaries

### Total pipeline runtime

The total pipeline timer starts before stage execution and stops after all stages complete:

- `src/upv_vlm_v2/pipeline/main_pipeline.py:L433`: `start = now_perf()`
- `src/upv_vlm_v2/pipeline/main_pipeline.py:L524-L608`: ordered stage execution
- `src/upv_vlm_v2/pipeline/main_pipeline.py:L629-L633`: writes `total_timing_ms`

This means `total_pipeline_timing_ms` includes the timed pipeline stages and wrapper overhead inside the pipeline function. It does not include starting external servers before the run.

### Target selection

Target selection is timed as a single stage:

- `src/upv_vlm_v2/pipeline/stages.py:L115-L168`, `run_target_selection_stage()`

The model-residency runtime audit found that this stage includes the Grounded-SAM2/GroundingDINO/SAM2 subprocess and CLIP verifier subprocess where configured, but does not split their internal loading/inference/postprocessing times:

- `docs/dev_history/final_pipeline_runtime_summary_model_residency_audit_20260706_011350.md:L116-L140`

Grounded-SAM2 diagnostics often include `grounded_sam2.runtime_s`, but this is a wrapper/subprocess-level diagnostic, not a full target-selection decomposition. CLIP artifacts such as `material_verification/material_scores.json` record `clip_model_id` and scores, but the representative E45 artifacts inspected here do not provide a consistent measured CLIP elapsed field.

### Anchor/VLM selection

Anchor selection is timed as a single full stage:

- `src/upv_vlm_v2/pipeline/stages.py:L236-L298`, `run_anchor_selection_stage()`

The stage includes candidate/crop preparation, prompt construction, repeated VLM HTTP requests where used, parsing, ranking, and final selection. Nested VLM decision JSONs can also include per-request `elapsed_ms`; those are not equivalent to the full anchor-selection stage.

Evidence:

- `docs/dev_history/contact_selection_vlm_runtime_methods_audit_20260706_031048.md:L22-L23`
- `docs/dev_history/contact_selection_vlm_runtime_methods_audit_20260706_031048.md:L45-L62`
- Example E45 artifact: `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164704/01_cycles/cycle_001/pipeline_full_session/artifacts/04_anchor_selection/qwen32_single_anchor_decision.json`

### Geometry, path length, and planning

These are directly timed as separate stages:

- Geometry: `src/upv_vlm_v2/pipeline/stages.py:L171-L233`
- Path length: `src/upv_vlm_v2/pipeline/stages.py:L301-L362`
- Robot planning: `src/upv_vlm_v2/pipeline/stages.py:L365-L495`

These fields are safe to report directly as `geometry_timing_ms`, `path_length_timing_ms`, and `robot_plan_timing_ms`.

### Robot execution

Robot execution is timed as one combined stage:

- `src/upv_vlm_v2/pipeline/stages.py:L498-L603`, `run_hardware_execution_stage()`
- `src/upv_vlm_v2/pipeline/stages.py:L499`: starts one timer before hardware connection/execution
- `src/upv_vlm_v2/pipeline/stages.py:L554-L561`: calls `run_t4_style_execution(...)`
- `src/upv_vlm_v2/pipeline/stages.py:L568-L600`: records one `ExecutionResult.timing_ms`

The execution function logs the substage sequence but not per-substage elapsed time:

- Approach/motion sequence: `src/upv_vlm_v2/execution/t4_style_executor.py:L216-L232`
- Clamp, hold, release: `src/upv_vlm_v2/execution/t4_style_executor.py:L234-L262`
- Return home: `src/upv_vlm_v2/execution/t4_style_executor.py:L275-L305`

The RTDE client logs before/after robot poses around `moveL`, but not elapsed time:

- `src/upv_vlm_v2/hardware/ur_rtde_client.py:L55-L73`

The Arduino clamp client logs commands, responses, timeouts, estimated move time, and requested hold time, but not measured elapsed time:

- Command loop: `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L57-L97`
- Open/clamp estimated times: `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L135-L177`
- Hold command: `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L179-L181`

## Answers To The Specific Questions

### 1. Does `robot_execution_timing_ms` exist as a single combined value?

The exact field name `robot_execution_timing_ms` was not found as the standard E45 master-column name. The equivalent combined execution timing exists as:

- `execution_timing_ms` in E45 CSV tables.
- Stage `execution` with `timing_ms` in `timing_summary.json`.
- `ExecutionResult.timing_ms` from `run_hardware_execution_stage()`.

### 2. Is robot execution split into approach, contact placement, clamping, holding, release, and return-home timing?

No measured split was found. The execution logs list those substage events and statuses, but not elapsed times per event.

### 3. Is actuator/clamping timing logged separately anywhere?

Measured actuator elapsed time is not logged separately. The clamp logs include command responses, `timeout_used_s`, and estimated values such as `estimated_move_time_s`, for example in `real_clamp_execution_log.json`. These are not measured durations.

### 4. Is UPV holding/acquisition time logged separately anywhere?

The requested hold command is logged, e.g. `HOLD_MS 5000` and `hold_ms: 5000`. A measured acquisition duration was not found. `time_of_flight_us` is logged for UPV signal interpretation, but it is not runtime/acquisition duration.

### 5. Is target-selection timing split into GroundingDINO/SAM2 and CLIP separately?

Not as a consistent final/E45 measured split. `target_selection_timing_ms` is the measured combined target-selection stage. Grounded-SAM2 wrapper diagnostics include `runtime_s` in many JSONs; CLIP artifacts record model/scoring information, but a consistent measured CLIP elapsed field was not found in the representative E45 target-selection artifacts. The prior model-residency report concluded that exact GroundingDINO-only, SAM2-only, CLIP-load, and CLIP-inference splits are not logged.

### 6. Is VLM request latency logged separately from the full anchor-selection stage?

Yes, for some VLM backends/artifacts. E45 decision JSONs include nested `elapsed_ms` per VLM request, for example in `qwen32_single_anchor_decision.json`. This is separate from full `anchor_selection_timing_ms`, which includes all candidate preparation, repeated requests, parsing, and selection logic.

### 7. Is geometry/path-length/planning timing logged separately?

Yes:

- `geometry_timing_ms`
- `path_length_timing_ms`
- `robot_plan_timing_ms`

These fields are safe major-stage timings.

### 8. Can remaining-code time be computed as a residual?

Yes, but only as an indirect residual:

\[
T_{\mathrm{residual}}
=
T_{\mathrm{total}}
-
\left(
T_{\mathrm{target}}
+T_{\mathrm{geometry}}
+T_{\mathrm{anchor}}
+T_{\mathrm{path}}
+T_{\mathrm{robot\_plan}}
+T_{\mathrm{execution}}
\right).
\]

If `pre_capture_home` and `capture` are present in `timing_summary.json`, they should be included explicitly before computing residual:

\[
T_{\mathrm{residual}}
=
T_{\mathrm{total}}
-
\sum_{\mathrm{logged\ stages}} T_i.
\]

This residual should be labeled as wrapper/file-handling/other overhead. It is not a directly instrumented code section.

### 9. Which fields are safe to use in the paper?

Safe measured fields:

- `total_pipeline_timing_ms`
- `target_selection_timing_ms`
- `geometry_timing_ms`
- `anchor_selection_timing_ms`
- `path_length_timing_ms`
- `robot_plan_timing_ms`
- `execution_timing_ms`
- Stage-level `pre_capture_home` and `capture` from per-run `timing_summary.json` where present
- Nested VLM `elapsed_ms` only when reported as per-request HTTP/VLM latency
- `grounded_sam2.runtime_s` only when reported as Grounded-SAM2 wrapper/runtime diagnostic, not full target-selection timing

Not safe as measured runtime fields:

- Approach-only duration
- Final contact approach-only duration
- Return-home-only duration
- Clamp actuation measured duration
- Release measured duration
- UPV acquisition measured duration
- GroundingDINO-only duration
- SAM2-only duration
- CLIP-only duration in E45 final artifacts
- Exact file-handling/overhead duration except as residual

### 10. Which fields require new instrumentation?

New instrumentation is required for:

- Start/stop elapsed time for each robot `moveL` stage.
- Start/stop elapsed time for `clamp_to_planned_width`, `clamp_hold`, and `clamp_release`.
- UPV acquisition/export elapsed time.
- Separate GroundingDINO model load/inference time.
- Separate SAM2 model load/inference time.
- Separate CLIP load/inference time.
- Anchor-selection breakdown into candidate generation, crop rendering, prompt construction, VLM request(s), parsing, and ranking.
- Explicit residual/IO timing if paper needs non-model overhead detail.

## Recommended Minimal Manuscript Runtime Table

Use a table like this for the manuscript:

| Runtime row | Reported value source | Notes |
|---|---|---|
| Full pipeline total | `total_pipeline_timing_ms` | Measured E45 end-to-end pipeline with intentionally conservative robot execution. |
| Target perception and verification | `target_selection_timing_ms` | Combined GroundingDINO/SAM2 + CLIP target-selection stage; no reliable submodel split. |
| Contact-anchor selection / VLM | `anchor_selection_timing_ms` | Full anchor-selection stage; includes VLM request/response and selection overhead. |
| Geometry and path-length computation | `geometry_timing_ms` + `path_length_timing_ms` | Directly measured code stages. |
| Robot pose planning | `robot_plan_timing_ms` | Directly measured, typically negligible. |
| Robot execution, clamp, hold, release, home | `execution_timing_ms` | One combined measured stage. Do not split into subcomponents unless newly instrumented. |
| Other overhead | residual from total minus logged stages | Optional; label as indirect residual only. |

Paper-safe wording:

> The final UPV-VLM pipeline logs reliable end-to-end and major-stage runtimes. Robot execution is measured as a single combined stage that includes the conservative approach sequence, clamp/hold/release behavior, and return-home motion. The execution logs record the ordered robot and clamp events, but not separate elapsed times for each robot or actuator substep; therefore, substage robot timings are not reported separately. Similarly, target-selection timing is measured as a combined GroundingDINO/SAM2 plus CLIP verification stage, while VLM request latency is available separately only in nested decision artifacts and should not be confused with the full anchor-selection stage.

## Future Logging Recommendations

Minimal changes for a future run:

1. Add `start_monotonic`, `end_monotonic`, and `elapsed_ms` around every `_move_stage()` call in `t4_style_executor.py`.
2. Add elapsed timing in `ArduinoClampClient._send_command()` for each clamp/release/hold command.
3. Add a dedicated UPV acquisition/export timer around the PL-200 or signal logging call.
4. Split target selection into GroundingDINO, SAM2, CLIP, mask cleanup, crop generation, and selection-decision timers.
5. Split anchor selection into candidate generation, crop/sheet rendering, prompt construction, VLM HTTP calls, response parsing, and ranking timers.
6. Add a residual/IO timer or explicit `other_overhead_timing_ms` to avoid post-hoc residual ambiguity.

