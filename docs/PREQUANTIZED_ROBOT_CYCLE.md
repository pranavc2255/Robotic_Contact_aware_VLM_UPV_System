# Staged prequantized Qwen complete UPV cycle

The main entrypoint uses Qwen2.5-VL-32B-Instruct-bnb-4bit (BitsAndBytes NF4,
double quantization, BF16 compute) with BF16 visual-merger exclusions. It stages
GPU residency on a single RTX 4090 (24 GB). Real perception and Qwen have completed
a saved-RGB-D run with simulated hardware; physical deployment needs a rig acceptance test.

## Sequence and manual reading

1. Explicit per-cycle `RUN_V2_EXECUTE` confirmation and existing hardware gates.
2. Existing home-before-capture check, ROS2 RGB-D capture.
3. Existing GroundingDINO-Tiny/SAM2.1 Hiera-Small subprocess, then CLIP ViT-B/32
   verification subprocesses. Each completes before Qwen starts.
4. Geometry, managed Qwen startup, L6 multi-image contact ranking via `/infer_multi`.
5. Stop and wait for the owned Qwen process before path estimation and planning.
6. Existing approach, clamp, timed hold, release, return home.
7. Read the PL-200 display during the hold; enter velocity, arrival time, signal
   quality and notes at the terminal **after return home**, as in the existing
   interactive runner. There is no PL-200 trigger/acquisition/export command.

Default hold is 5 seconds. `--hold-sec 10` is available if more observation time
is needed (maximum 30 seconds). This is not an indefinite hold waiting for input.
Do not increase motion speeds for this first integration test.

The new selector uses the highest-scoring model-labeled usable pair with numeric
ID tie-break, matching the recent comparison policy. Incomplete/invalid fields
abort selection, rather than infer usability from a score. All-poor responses
produce no selected anchor and do not reach contact execution. Old selector
configs retain their previous behavior. The prompt is `multi_image_v1_original`.

## Commands (repository root)

Leave the existing ROS2 RealSense publisher running. Stop your separately started
Qwen/perception processes yourself first. The runner refuses an occupied managed
port and never kills an unrelated server. Other GPU processes on other ports are
not detected or managed. Run from the repository root in the tested environment.

Validation only, no models or hardware:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py --dry-run
```

First perform an operator-run no-motion plan check (camera and inference DO run):

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --material brick --axis major --plan-check
```

Inspect the selected mask, contact crops, selected anchor, dimensions and planned
poses in the printed output directory. Revalidate hand-eye calibration, TCP/tool
offsets, home pose, workspace limits, robot address and Arduino port against the
actual rig. The bundled settings are rig-specific, not universal safe defaults.

Then, only with the rig checked and an operator at the emergency stop:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --material brick --axis major --execute --max-cycles 1
```

Use `--material 'concrete block'` or `--material timber` as appropriate. Each cycle
requires terminal confirmation. No separate Qwen startup command is needed.
Use `--qwen-python /absolute/path/to/venv/bin/python` for a separate server env.

## Artifacts and timing

`outputs/prequantized_upv_cycles/session_<timestamp>/` contains a config snapshot,
pipeline outputs, and interactive reading JSON/CSV files. Anchor-stage artifacts
include `managed_qwen_server.log` and `managed_qwen_lifecycle.json`, with startup
to health, shutdown, total server lifecycle, model path and process ID. The
pipeline total includes managed startup/shutdown; anchor-stage timing remains
the existing candidate/crop/request timing. Do not compare the total to a
warm-server timing without accounting for those boundaries. Manual reading entry
is outside the pipeline child timing, in the interactive session timing.

## Public checkout

The same launcher and narrowly changed reference modules are mirrored under
`UPV_VLM_Contact/`; it does not depend on the parent workspace. Install perception,
Qwen, ROS2, `ur-rtde` and `pyserial` in the operator's environment; download weights
and SAM2 separately, following its installation docs. Run the same script from
that checkout using that environment's `python`. Supply `--model-path` if the
checkpoint is stored elsewhere. Supplied calibration files describe the original
rig and MUST be replaced/revalidated for another installation.

The separate publishing clone and online GitHub repository were not edited or
pushed by this change. Sync/review these changes before publishing. Existing
`upv-pipeline` and result-reproduction commands remain computational-only.

## Validation limits

Offline checks cover syntax, CLI validation and mocked lifecycle/selection.
The recorded saved-input GPU simulation additionally exercised real perception
and NF4 inference. These checks do not replace a physical rig acceptance test.
Motion speeds, operator confirmations and safety gates remain unchanged.
