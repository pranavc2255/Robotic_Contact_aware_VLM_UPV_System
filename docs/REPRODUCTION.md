# Reproduction Commands

Run these commands from the new repository root. The parent workspace is not
needed. Paths under `runs/` must not already exist.

## Paper Tables From Saved Predictions

```bash
PYTHONPATH=src python -m upv_vlm_contact reproduce --output runs/paper_tables
```

No models, archives or hardware dependencies are needed for this command.
The output has contact73/contact82/perception/path_length/runtime JSON and a
contact/perception Markdown summary. `results/` preserves original numeric files.

## Data and CPU RGB+Mask Baseline

```bash
python -m pip install -e '.[vision]'
upv-reproduce install-data contact --archive release_assets/contact.zip
upv-reproduce baseline-rgb --dataset contact82 --case-id case_001_brick_01_clean_regular --output runs/rgb_pilot
# After reviewing the printed input validation, explicitly compute:
upv-reproduce baseline-rgb --dataset contact82 --case-id case_001_brick_01_clean_regular --output runs/rgb_pilot --run
upv-reproduce baseline-rgb --dataset contact73 --output runs/rgb73 --run
upv-reproduce baseline-rgb --dataset contact82 --output runs/rgb82 --run
upv-reproduce figures --output runs/contact_figures
```

Failures remain explicit, excluded from valid-prediction metrics with their count
reported. The baseline cannot produce a safe scene selection when an input is
missing without further handling; do not silently ignore failed inputs.

## Optional New VLM Inference

Models are not bundled or downloaded automatically. Obtain the appropriate
model under its own license and place it under an ignored `local_models/` folder:
`local_models/Qwen2.5-VL-32B-Instruct-bnb-4bit` for the main model; `Qwen/Qwen2.5-VL-3B-Instruct` or
`Qwen/Qwen3-VL-2B-Instruct`. Verify the exact model path and supported Transformers
version, and allocate a GPU environment separately from the offline environment.
`configs/build_environment.json` records the importer environment, not all historic
model servers' environments. GPU dependency/version compatibility must be checked
on the deployment host.

```bash
python -m pip install -e '.[vlm]'
# Dry-run command print only:
python scripts/start_qwen25_32b_bnb4_contact_server.py --dry-run
# Explicitly start ONE server; this loads weights, so run manually:
python scripts/start_qwen25_32b_bnb4_contact_server.py
```

In a second terminal using the same filesystem:

```bash
upv-reproduce infer-contact --dataset contact89 --model Qwen2.5-32B-NF4 --case-id case_001_brick_01_clean_regular --server-url http://127.0.0.1:8896 --output runs/qwen_pilot
# Explicit HTTP inference, never hardware execution:
upv-reproduce infer-contact --dataset contact89 --model Qwen2.5-32B-NF4 --case-id case_001_brick_01_clean_regular --server-url http://127.0.0.1:8896 --output runs/qwen_pilot --run-inference
upv-reproduce infer-contact --dataset contact89 --model Qwen2.5-32B-NF4 --server-url http://127.0.0.1:8896 --output runs/qwen89_nf4 --run-inference
```

Stop that server manually before starting another model. Substitute server
`--model qwen25-3b --model-path local_models/Qwen2.5-VL-3B-Instruct` or
`--model qwen3-2b --model-path local_models/Qwen3-VL-2B-Instruct`; use the matching
`--model` label in the client. Labels are user-supplied; the server response is
saved for identity audit. New generations need not match saved outputs exactly.

The client uses `/infer_multi`, original `multi_image_v1_original` prompts,
ordered L6 images, temperature 0, seed 12345 and max_new_tokens 1600. It sends
only image paths and prompts, never ground-truth labels. The API requires a
shared local filesystem; it is not a remote image-upload service. Failed/invalid
responses are recorded, not turned into bad-contact predictions. Model loading
is outside request timing. No service is started by `infer-contact`.

## Optional Raw-Depth Path Evaluation

This is an explicit new CPU evaluation, not required for reproducing the saved
paper numbers. Install the path archive and vision dependencies first:

```bash
upv-reproduce install-data path_length --archive release_assets/path_length.zip
upv-reproduce recompute-paths --version v2 --case-id case_002_2 --output runs/path_pilot
upv-reproduce recompute-paths --version v2 --case-id case_002_2 --output runs/path_pilot --run
upv-reproduce recompute-paths --version v2 --output runs/path_v2 --run
```

The adapter makes a separate input copy, uses the saved evaluator parameters and
selected-anchor root, disables fallback, and writes a new evaluation below the
requested output. It never modifies archive objects or `results/`. V1 is also
available via `--version v1`. The raw-depth adapter is syntax/CLI checked; it was
not run as a new experiment during release assembly.

## CPU RGB-D Contact Baseline

```bash
upv-reproduce baseline-rgbd --dataset contact82 --output runs/rgbd82
upv-reproduce baseline-rgbd --dataset contact82 --output runs/rgbd82 --run
upv-reproduce baseline-rgbd --dataset contact73 --output runs/rgbd73 --run
```

Without `--run` this validates archived paths only. Contact73 follows the saved
comparison protocol: recompute the seven new10 scenes and reuse the 45 original
E3 predictions. Reused rows are marked explicitly. This avoids changing the
RANSAC seed sequence by rerunning a different combined group ordering. Contact82
recomputes all 18 original groups. No model inference or hardware is involved.
This adapter was input/CLI validated, not run as a new full experiment at assembly.

## Historical Code and Remaining Porting Work

`reference/scripts/` and `reference/src/` retain the evaluator, RGB-D baseline,
perception, candidate generation, plotting, and model-server implementations.
They are scientific reference snapshots, not a claim that every old research
CLI is standalone: old workspace paths need archive-index adaptation. The
supported public entrypoints are the ones above and `upv-pipeline`, documented in
[PIPELINE.md](PIPELINE.md). CPU RGB+mask, RGB-D and selected-anchor path evaluation
have portable adapters. The raw-image pipeline has a synthetic CPU smoke test;
fresh GPU perception/VLM validation remains separate from saved-result reproduction.

Do not run hardware modules in the reference tree. Offline readers never import
RTDE, RealSense, Arduino or acquisition libraries. Hardware deployment needs
fresh calibration, configuration, a site safety review and explicit operator
approval; none is provided by this release.
