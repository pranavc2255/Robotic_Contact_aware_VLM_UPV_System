# Isolated fast-engine experiment

This does NOT modify the robot pipeline, existing prompts, weights or results.
89 local saved crops in `datasets/fast_contact89` come from the matched Contact73
plus concrete16 cohort. The manifest retains source paths as provenance only;
runtime reads local copies, verifies SHA256 and never sends labels to the model.

Four conditions: original wording/schema; compact-output override; original with
200704-pixel per-image processor cap; compact plus cap. All use temperature 0,
seed 12345 and max_tokens 1600. Image order is unchanged. Jobs are shuffled with
a fixed seed. This is a single pass per condition, not a statistical latency
benchmark. Compact output deliberately omits explanations; these are not invented.
The cap uses processor resizing, NOT a FastV/VisionZip implementation. Check token
counts to establish whether it actually reduced input size on these crops.

## Install separately (operator commands)

Do not install vLLM into the working `.venv`:

```bash
cd ~/UPV_VLM_orient/UPV_VLM_Contact
python3 -m venv .venv-vllm-bnb
.venv-vllm-bnb/bin/python -m pip install 'vllm==0.19.1' bitsandbytes
.venv-vllm-bnb/bin/python scripts/start_fast_contact_experiment.py --check-engine
```

This installation combination has not been tested here. The launcher records
the actual installed packages. Exact prequantized Qwen-VL + vLLM compatibility,
including mixed-precision merger handling and sleep/wake, remains unproven.
Failure is a valid experiment outcome; no AWQ/GPTQ or checkpoint fallback occurs.
Do not change the working environment to address experimental engine failures.

The original unpinned installation installed vLLM 0.29.0, which rejected
`bitsandbytes` during configuration validation, before weight loading, in
`outputs/fast_contact_servers/20260910_023919_477263/server.log`.
Version 0.19.1 explicitly registers BitsAndBytes in its
[quantization registry](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/layers/quantization/__init__.py).
This pin addresses that particular startup incompatibility, not a guarantee that
this exact multimodal checkpoint or sleep mode will work. The new environment
keeps the failed experimental installation and working `.venv` untouched.

## Server (terminal 1)

The launcher explicitly disables video (`{"image":5,"video":0}`). In the
20260910_025202_156853 log, v0.19.1 loaded weights in 11.202854 seconds using
19.95 GiB, then failed with CUDA OOM while profiling a dummy video. Unspecified
modalities default to 999 in this installed version's
`vllm/config/multimodal.py:get_limit_per_prompt`. This image-only experiment
does not need video. Disabling video does not resize the benchmark images or
change its prompts. Image profiling and subsequent inference still need testing;
this correction does not guarantee that the remaining VRAM is sufficient.

Stop other GPU servers yourself. This server uses localhost port 8895:

```bash
.venv-vllm-bnb/bin/python scripts/start_fast_contact_experiment.py --run
```

Without `--run`, it prints the command only. Logs and startup-to-ready timing go
to `outputs/fast_contact_servers/<timestamp>/`. CUDA graphs are enabled by normal
engine defaults; `--eager` is an explicitly different diagnostic mode if necessary.
Sleep development endpoints bind only to localhost; do not expose this server.

## Benchmark (terminal 2)

### Reduced-resolution startup diagnostic

Run 20260910_025425_483729 loaded weights in 3.961313 s (19.95 GiB),
then exhausted VRAM profiling a maximum-size image. For a distinct capped
experiment, try:

```bash
.venv-vllm-bnb/bin/python scripts/start_fast_contact_experiment.py --run --max-image-pixels 200704 --eager
.venv/bin/python scripts/benchmark_fast_contact_experiment.py --run --max-cases 1 --variants reduced_pixels
```

This caps processor pixels globally and disables CUDA graphs. It is not an
unchanged-resolution or optimized-graph baseline. Keep the launch.json with
the benchmark results. Do not run the original-versus-reduced comparison under
this global cap: both conditions would inherit it. GPU success is unverified.

Run 20260910_025609_955546 passed image profiling, but had only 0.43 GiB
available KV cache against 4 GiB required for 16384 tokens. An explicit next
diagnostic (only with other GPU workloads stopped) is:

```bash
.venv-vllm-bnb/bin/python scripts/start_fast_contact_experiment.py --run --max-image-pixels 200704 --eager --max-model-len 6144 --gpu-memory-utilization 0.96
```

The log's cache ratio implies approximately 1.5 GiB for 6144 tokens. Increasing
the budget from 90% to 96% adds approximately 1.41 GiB on this 23.52 GiB GPU;
this is a budget estimate, not a successful test. The context limit includes
image tokens, text and generated output. Requests exceeding it must fail, not
be silently truncated. Do not run perception or another GPU server alongside it.

```bash
.venv/bin/python scripts/benchmark_fast_contact_experiment.py --run
```

This sends 84 requests (4 variants x 21 scenes). To include a sleep/wake round
trip before each request:

```bash
.venv/bin/python scripts/benchmark_fast_contact_experiment.py --run --sleep-between-cases
```

No GSAM2 or robot runs during this benchmark, including during sleep. Thus this
tests the model engine and API lifecycle, NOT successful GPU sharing with GSAM2.
Use no `--run` for validation only; `--max-cases 1` limits to the first scene.

## Saved outputs

`outputs/fast_contact_experiment/<timestamp>/`: settings, input manifest, server
identity, randomized schedule, complete request JSONs (including images), prompt
texts, raw responses, parsed outputs, HTTP failures, per-job timing, predictions,
selections, metrics CSV/JSON and comparison markdown. Invalid/unattempted pairs
are excluded from classification and explicitly counted; they are not poor labels.
Selection is highest score among usable pairs, numeric ID tie-break. Abstentions
and incomplete scenes are separate from good/bad selections.

Request wall time includes HTTP/inference, not model startup, crop generation,
perception or robot motion. Sleep/wake times are separate. Engine-reported input
and output token counts are saved; vision/prefill/decoding time is not separated.
Historical Qwen32 predictions are context, not a freshly measured same-engine
control or a historical NF4 control. The original vLLM condition provides the
within-engine prompt/pixel comparison. Different kernels/processors can change
predictions even at temperature zero. Do not select a deployment winner solely
on these previously inspected test labels.

Speculative decoding and true visual-token pruning are NOT implemented here:
neither has a verified compatible implementation/draft for this NF4 setup.

References: https://docs.vllm.ai/en/stable/features/sleep_mode/ and
https://docs.vllm.ai/en/stable/features/quantization/bnb/ .

Initial validation covered syntax, argument handling, cohort hashes and offline
parser/metric behavior only. It did not include experimental engine installation,
model loading, inference or hardware operation.
