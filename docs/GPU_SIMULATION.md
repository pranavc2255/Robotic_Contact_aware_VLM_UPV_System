# Real GPU inference, simulated hardware

## Optional CPU weight prefetch experiment

Append `--prefetch-weights` to overlap sequential checkpoint file reads with
perception. This warms the OS page cache using an 8 MiB buffer; it does not load
Qwen onto the GPU or pin 18 GiB in RAM. It starts only if available memory exceeds
checkpoint size plus 8 GiB, and stops if available memory drops below 8 GiB.
The cache may still be evicted. This is an experiment, not a guaranteed speedup.

`artifacts/04_anchor_selection/weight_prefetch.json` records bytes read, read time,
memory check and any additional wait after perception. Compare
`timing_summary.json` total time as well as `managed_qwen_lifecycle.json` startup
time: disk contention or additional prefetch waiting can offset faster loading.
Do not add prefetch read time to total time; it overlaps perception. Repeated
runs without prefetch may also benefit from an already warm OS cache. No caches
are forcibly cleared by this option. Existing runs are unchanged.

Run from this checkout (no ROS2 setup or camera publisher needed):

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --gpu-simulation --material 'concrete block' --axis major
```

This is NOT the existing `--dry-run` placeholder mode. It reads saved RGB-D,
runs actual DINO/SAM2 and CLIP, starts the local NF4 Qwen server, calls
`/infer_multi`, stops Qwen, estimates path length and plans the cycle. Robot and
clamp execution use existing simulated backends only. No real hardware permission,
ROS2 capture, robot IP or serial port is supplied. A simulation confirmation token
is supplied only to satisfy the existing simulated-execution gate.

Default saved input is `datasets/gpu_smoke/concrete/`, copied from the original
capture session `session_20260702_153357/cases/case_012_2` (retained old E2 case 14
replacement). RGB, aligned uint16 depth and camera-info are local copies, not
symlinks. Use `--case-dir` for another folder with those same three filenames.
This local example is ignored by Git with the other datasets.

Outputs: `outputs/prequantized_upv_cycles/session_<timestamp>/`.
Simulated execution is not a physical success result; no UPV value is fabricated.
If no target or no usable anchor is found, downstream execution is skipped as
in the real pipeline. Stop other GPU servers before running; managed Qwen needs
port 8896. Inference remains sequential, not concurrent model residency.

An initial command check on 2026-09-10 encountered unavailable CUDA in the
validation environment. The preflight aborted before model loading or hardware. Syntax and
mocked command/config checks are not a completed GPU run.
