# UPV-VLM Contact

Research software for RGB-D material perception, candidate contact-pair generation, vision-language contact assessment, and selected-path length estimation for ultrasonic pulse velocity (UPV) inspection.

## Associated Research

**A Vision-Language Artificial Intelligence System for Contact-Aware Robotic Ultrasonic Inspection of Reclaimed Construction Materials**

Pranav Chougule and Arvin Ebrahimkhanlou.

See [`CITATION.cff`](CITATION.cff) for software citation metadata. Publication venue and DOI will be added when available.

This source checkout runs independently of the original research workspace. It separates **reproducing recorded results** from **running new inference**.

Reproduction and `upv-pipeline` commands do not execute robot motion or measurement hardware. A separate, opt-in lab robot-cycle launcher is described below.

> **Release scope:** saved-result reproduction and staged computational tools.
>
> Offline tests, detached-copy reproduction, and a synthetic CPU pipeline smoke test pass. A fresh GPU installation has not been independently revalidated.
>
> Code: [PolyForm Noncommercial License 1.0.0](LICENSE).
> Data: see [Dataset Rights Notice](DATA_LICENSE.md).

---

## Quick Start

Reproduce saved metrics without models, CUDA, ROS, or hardware:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
upv-reproduce reproduce --output runs/reproduced
python -m unittest discover -s tests -v
```

Run these commands from the repository root.

Without installation, use:

```bash
PYTHONPATH=src python -m upv_vlm_contact
```

in place of `upv-reproduce`.

A wheel alone does not contain the datasets and reference code; use the complete source checkout for full reproduction.

---

## Computational Pipeline

```text
Saved RGB-D + material query
  -> GroundingDINO-Tiny + SAM2.1 Hiera-Small
  -> CLIP ViT-B/32 material verification / selected mask
  -> object geometry + length-dependent A* contact candidates
  -> L6 two-contact crop images
  -> Qwen2.5-VL-32B-Instruct-bnb-4bit /infer_multi
  -> highest-scoring model-labeled usable pair, or NO_SAFE_ANCHOR
  -> mask-based and local depth point-cloud path lengths
```

`upv-pipeline` runs the computational stages together or as ordered subsets.

It defaults to validation only. The paths in the following example are placeholders:

```bash
upv-pipeline --run-dir runs/example \
  --rgb /path/to/raw_rgb.png \
  --depth /path/to/depth_raw.npy \
  --camera-info /path/to/camera_info.json \
  --material 'concrete block' \
  --stages perception anchors
```

Append `--run` to prepare candidates.

Start Qwen separately, then run:

```bash
upv-pipeline \
  --run-dir runs/example \
  --stages contact paths \
  --run \
  --run-inference
```

Each stage runs in a separate process and records artifacts, input provenance, and timing.

On a constrained GPU, finish perception **before** starting Qwen. An externally running server stays resident even when pipeline calls are sequential.

No automatic server startup or hardware action is performed.

See:

* [`docs/INSTALL.md`](docs/INSTALL.md) for environment and model dependencies.
* [`docs/PIPELINE.md`](docs/PIPELINE.md) for full and staged operation, saved-mask reuse, input formats, and validation limits.

---

## Main Single-GPU Cycle

The main deployment uses prequantized BitsAndBytes NF4 weights with BF16 compute on one NVIDIA RTX 4090 with 24 GB VRAM.

Perception and Qwen use the GPU sequentially:

1. GroundingDINO/SAM2 and CLIP complete perception.
2. Qwen starts and ranks candidate contact pairs.
3. Qwen terminates after contact assessment.

There is no reduced-pixel or vLLM experiment enabled by this launcher.

Validate the installed files without inference or hardware:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py --dry-run
```

For an end-to-end computational test using saved RGB-D inputs and simulated hardware:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --gpu-simulation \
  --material 'concrete block' \
  --axis major
```

The saved-input directory defaults to:

```text
datasets/gpu_smoke/concrete
```

Use `--case-dir` for another directory containing:

```text
raw_rgb.png
raw_depth_aligned_z16.png
camera_info_aligned_depth.json
```

This route has a recorded successful real-GPU/simulated-hardware run.

Physical operation still requires rig calibration, hardware checks, and operator confirmation.

`scripts/run_prequantized_qwen_upv_cycle.py` supports the original UR3e rig using staged perception, prequantized Qwen2.5-32B NF4 startup/inference/shutdown, and the existing operator-confirmed approach/clamp/hold/release/home sequence.

PL-200 values are entered manually after the cycle and are not acquired directly by the software.

The launcher defaults to validation only.

Real-hardware operation requires:

```text
--execute
```

and typed confirmation for each cycle.

The supplied rig-specific calibration, home, and workspace settings must be revalidated before use and should not be treated as portable robot defaults.

The hardware integration has offline tests but has not undergone a fresh physical validation for this release.

See:

[`docs/PREQUANTIZED_ROBOT_CYCLE.md`](docs/PREQUANTIZED_ROBOT_CYCLE.md)

for complete-cycle commands, prerequisites, and safety requirements.

---

## Experiment Reproduction

| Experiment        | Cohort                                                      | Supported analysis                                                                  |
| ----------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Contact89         | 89 candidate pairs; 21 cases, including five concrete cases | Classification and selected-pair usability; three VLMs and classical RGB+mask       |
| Perception        | 63 present + 9 absent queries; 24 scene configurations      | CLIP prompt ensemble: 58/63 present, 7/9 absent; **65/72 correct (90.28%)** at 0.35 |
| Contact73         | 45 E3 + 28 new10 pairs; 16 scenes                           | Classification and selected-pair usability; five methods                            |
| Contact82         | 50 E3 + 32 E45 pairs; 18 scenes                             | Classification and selected-pair usability; five methods                            |
| Path length V1/V2 | 15 cases, 30 manual paths each                              | MAE, RMSE, MAPE, and material summaries                                             |
| Runtime           | E45 saved master rows                                       | Stage summaries with imported-row caveats                                           |

Contact73 is not Contact82 minus nine pairs; their scene cohorts differ.

The five evaluated contact methods are:

1. Qwen2.5-VL-32B
2. Qwen2.5-VL-3B
3. Qwen3-VL-2B
4. Classical RGB + mask
5. Classical RGB-D

The classical contact classifiers are untrained, although their upstream Grounded-SAM2/CLIP perception components are learned.

### Perception Configuration

The main perception configuration uses:

* eight text descriptions per material,
* normalized mean CLIP text embeddings,
* three-class softmax, and
* an inclusive confidence threshold of `0.35`.

It does not use source priors, color or texture guards, or a margin-based rejection veto.

The prompts and threshold were selected using the same set of 72 queries reported here; therefore, these results should not be interpreted as an independent validation set.

See:

[`docs/CLIP_ENSEMBLE.md`](docs/CLIP_ENSEMBLE.md)

for the mechanism and threshold analysis.

Saved contact, path-length, and runtime experiments retain their original inputs and settings. Changes to the perception configuration do not retroactively rerun previously recorded experiments.

Selection metrics use Python score-based selection among model-labeled usable pairs and do not necessarily correspond to the VLM's original final-action field.

Raw responses retain both fields for audit.

### Reproduction Documentation

* [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md): baselines, VLM inference, paths, and plots.
* [`docs/METHODS.md`](docs/METHODS.md): classification and selection conventions.
* [`results/prompt_sensitivity/README.md`](results/prompt_sensitivity/README.md): original system prompt and four wording variants.
* [`docs/DATA.md`](docs/DATA.md): raw assets, checksums, and provenance.

---

## Data Distribution

Code and dataset rights are handled separately.

The source code is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Dataset reuse permission is governed separately by [`DATA_LICENSE.md`](DATA_LICENSE.md). Public availability of a dataset does not itself grant reuse rights beyond the applicable data-rights notice.

Small result tables and manifests are included directly in the repository.

Raw RGB-D data, masks, image crops, waveform exports, and detailed experiment artifacts are distributed as release assets.

The released archives include:

```text
contact.zip
perception.zip
path_length.zip
runtime.zip
contact89.zip
pl200.zip
```

The archives can be installed using:

```bash
upv-reproduce install-data contact \
  --archive release_assets/contact.zip

upv-reproduce install-data perception \
  --archive release_assets/perception.zip

upv-reproduce install-data path_length \
  --archive release_assets/path_length.zip

upv-reproduce install-data runtime \
  --archive release_assets/runtime.zip

upv-reproduce install-data contact89 \
  --archive release_assets/contact89.zip

upv-reproduce install-data pl200 \
  --archive release_assets/pl200.zip
```

Archive sizes, SHA256 digests, and download information are recorded in:

```text
data/manifests/archives.json
```

The release assets provide the data required for the corresponding image-based and waveform-based reproduction workflows described in this repository.

Git alone supports saved-table reproduction, while image-based and waveform-based reruns use the corresponding released archives.

Do not commit:

* dataset ZIP archives,
* model weights,
* Python environments, or
* generated run directories.

Historical `workspace/...` paths are retained as provenance identifiers and are resolved through archive indexes rather than requiring access to the original research workspace.

### Contact89 and PL-200

Contact89 is registered in the archive catalog and saved-table reproduction.

It combines Contact73 with 16 additional candidate pairs from five concrete cases and is distinct from Contact82.

The released `contact89.zip` and `pl200.zip` archives can be installed with:

```bash
upv-reproduce install-data contact89 \
  --archive release_assets/contact89.zip

upv-reproduce install-data pl200 \
  --archive release_assets/pl200.zip

upv-reproduce reproduce \
  --output runs/contact89_reproduced
```

See:

[`docs/CONTACT89_RELEASE.md`](docs/CONTACT89_RELEASE.md)

for documentation of the Contact89 and PL-200 release additions.

---

## Repository Layout

```text
src/upv_vlm_contact/  Public CLI, staged pipeline, and reproduction adapters
configs/              Computational configuration and recorded environment provenance
results/              Saved predictions, prompts, and metric evidence
data/                 Dataset manifests, labels, and archive checksums
reference/            Bundled research implementations used by explicit adapters
hardware/             Rig-specific hardware references and firmware
docs/                 Installation, methods, datasets, and release limitations
tests/                Offline regression and planning tests
tools/                Release checks and synthetic CPU integration test
scripts/              Explicit launchers, computational tools, and experiments
release_assets/       Local dataset ZIPs (Git-ignored)
datasets/             Installed raw assets (Git-ignored)
runs/                 New outputs (Git-ignored)
```

---

## Validation and Release Checks

Run the primary release validation with:

```bash
python tools/check_release.py
```

Run the CPU pipeline smoke test with:

```bash
python tools/check_pipeline_cpu.py
```

This test requires vision dependencies but does not perform neural-network inference.

For exhaustive local archive verification:

```bash
python tools/check_release.py --check-archives
```

The release checker copies the repository into a detached temporary directory and reproduces the saved tables from that copy.

The CPU smoke test mocks the VLM decision and therefore does not establish GPU or model inference correctness.

Hardware references require:

* fresh calibration,
* operator safeguards, and
* a separate deployment review.

The PL-200 archive contains saved waveform exports and historical analyses.

Exact correspondence between every waveform record, robot session, and selected contact anchor, as well as full end-to-end waveform reprocessing, has not been independently validated for this release.

See [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) for the release checklist.

---

## License

The source code in this repository is licensed under the **PolyForm Noncommercial License 1.0.0**.

Commercial use is not permitted under this license.

See [`LICENSE`](LICENSE) for the complete license terms.

Third-party software, models, datasets, and other components remain subject to their respective licenses and terms.

Available third-party notices are provided under:

```text
docs/third_party/
```

---

## Citation

If you use this software in academic work, please cite the associated research article and software record when available.

Citation metadata is provided in:

[`CITATION.cff`](CITATION.cff)

The final publication DOI and software-release citation will be added when available.
