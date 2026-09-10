# Saved-RGB-D Pipeline and Stage Interface

`upv-pipeline` runs the computational workflow, **not physical robot execution**.
It imports no RTDE, RealSense, Arduino or UPV-acquisition modules. Each stage runs
in a separate subprocess. Use `python -m upv_vlm_contact.pipeline` as an equivalent
when running with `PYTHONPATH=src`.

| Stage | Inputs | Outputs / boundary |
|---|---|---|
| `perception` | RGB + requested material | GroundingDINO/SAM2 + CLIP selected mask; optional saved-mask reuse |
| `anchors` | Selected mask, RGB-D, intrinsics | Object geometry, length-based A* candidates, L6 pair crops |
| `contact` | Ordered candidate crops | `/infer_multi` request/response, classification, Python selected usable anchor |
| `paths` | Selected A* endpoints, mask and depth | Mask endpoint length and local depth point-cloud percentile length |

The default configuration preserves the bundled main contact-preparation setup:
orientation-fixed core-support rectangle, transducer-diameter candidate count,
and original multi-image prompt. Its legacy CLIP source-prior/color-guard policy
is **not** the later 63-trial CLIP-only ablation. These should not be presented as
the same experiment. Use saved-result reproduction for the exact published cohort
policy and numbers.

## Input contract

Use a directory of your own files; filenames are explicit arguments:

- RGB PNG: one saved image.
- Aligned depth: uint16 PNG or `.npy`, same width/height as RGB; default unit 1 mm.
- Camera info JSON: the intrinsics for that aligned RGB grid, not unaligned depth.
- Material: `brick`, `timber`, or `concrete block`.
- Optional `--mask`: a selected mask on the same grid. This skips detector/CLIP
  inference and records `user_supplied_saved_mask` provenance.

Input hashes and effective config are stored on the first execution. Resumed
stages reject changed inputs. Do not move an unfinished run or its input files:
the working manifest uses absolute paths. Existing stages are never overwritten.

## Recommended single-GPU execution

First, **stop any VLM server manually**, activate the perception environment, and
prepare candidates (paths below are placeholders for your saved RGB-D input):

```bash
upv-pipeline --run-dir runs/example \
  --rgb /path/to/raw_rgb.png --depth /path/to/depth_raw.npy \
  --camera-info /path/to/camera_info.json --material 'concrete block' \
  --stages perception anchors
```

This validates file paths and prints the plan only. Append `--run` to execute.
Inspect `runs/example/perception/selected_mask_overlay.png` and
`runs/example/anchors/clean_anchor_review_grid.png`. If a target mask is wrong,
do not treat subsequent contact results as successful automatic perception.
Start a new run with an explicitly corrected `--mask` if needed.

The perception subprocess exits before the next stage. Now start Qwen separately
in the VLM environment:

```bash
python scripts/start_qwen25_32b_bnb4_contact_server.py
```

In another terminal with the installed package and vision dependencies:

```bash
upv-pipeline --run-dir runs/example --stages contact paths --run --run-inference
```

No manual labels are needed for inference. `NO_SAFE_ANCHOR` is retained as a
decision, and the path stage writes `SKIPPED_NO_SAFE_ANCHOR`. Invalid responses
fail the stage; they are not converted into bad-contact labels. Ground-truth
annotations are only used by benchmark evaluation commands.

## One-command computational pipeline

For automatic sequential residency on one RTX 4090, use the main NF4 launcher:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --gpu-simulation --case-dir /path/to/saved_rgbd --material brick --axis major
```

This starts and stops Qwen itself; do not start a separate VLM server. It uses
the standard checkpoint processor settings, not the reduced-pixel experiment.
See [complete cycle](PREQUANTIZED_ROBOT_CYCLE.md) for live-camera planning and
operator-confirmed physical execution with manual PL-200 terminal entry.

On a host where the separate VLM server can remain resident (for example a second
GPU/host with a shared filesystem), all stages can be invoked together:

```bash
upv-pipeline --run-dir runs/full_example \
  --rgb /path/to/raw_rgb.png --depth /path/to/depth_raw.npy \
  --camera-info /path/to/camera_info.json --material 'concrete block' \
  --run --run-inference
```

Execution calls are sequential, but an external server remains resident. This
command does **not** guarantee sequential GPU residency. The script never starts
or stops a server automatically. `/infer_multi` sends local file paths, not image
uploads: the server must see the identical filesystem paths.

## Configuration and outputs

Copy `configs/pipeline.json` and pass `--config /path/to/config.json` on the first
invocation to change model paths, server URL or algorithm settings. Fresh-run
results need not equal saved historical results. Server model labels are not
proof of loaded model identity; retain the server startup log.

Each stage writes `result.json`, `stage.log`, and `timing.json`, plus scientific
artifacts. Wall time includes worker startup; Qwen server weight loading is
outside that timer. Interrupted/failed stages must be inspected and a new run
used; there is no automatic retry that might mix stale outputs.

The depth method uses the selected contact endpoints, not guessed major/minor
manual labels. Its rectangle is a support ROI, not an alternate measurement axis.
It uses local bands, depth tolerance and percentile endpoints; no whole-ROI depth
estimate is computed. Invalid depth remains explicit inside `depth_point_cloud`.
This fresh contact-path measurement is distinct from the 30 manually selected
path-length study, reproduced with `upv-reproduce recompute-paths`.

## Validation status

- Saved-result reproduction and standard-library planning tests: passed.
- Synthetic saved-mask -> candidate crops -> mocked contact -> path smoke: passed.
- The main NF4 launcher has a saved successful real-perception/Qwen computational
  run with simulated hardware; fresh installations require local validation.
- Robot motion, calibration deployment, clamping and UPV acquisition: **not supported
  by this public runner**; historical sources are references only.
