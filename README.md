# Robotic Contact aware VLM Based UPV System

Research software for RGB-D material perception, candidate contact-pair generation,
vision-language contact assessment, and selected-path length estimation for
ultrasonic pulse velocity (UPV) inspection.

## Associated Research

**A Vision-Language Artificial Intelligence System for Contact-Aware Robotic Ultrasonic Inspection of Reclaimed Construction Materials**

[Pranav Chougule](https://orcid.org/0009-0008-1596-0027) and
[Arvin Ebrahimkhanlou](https://orcid.org/0000-0002-0740-5807).
See [CITATION.cff](CITATION.cff) for software citation metadata.
Publication venue and DOI will be added when available.

This source checkout runs independently of the original research workspace.
It separates **reproducing recorded results** from **running new inference**.
Reproduction and `upv-pipeline` commands do not execute robot motion or measurement
hardware. A separate, opt-in lab robot-cycle launcher is described below.

> **Release scope:** saved-result reproduction and staged computational tools.
> Offline tests, detached-copy reproduction and a synthetic CPU pipeline smoke
> test pass. A fresh GPU installation has not been validated.
> Code: [rights reserved; licensing pending](LICENSE). Data: [rights notice](DATA_LICENSE.md).
> Datasets: [v0.1.0 release](https://github.com/pranavc2255/Robotic_Contact_aware_VLM_UPV_System/releases/tag/v0.1.0).

## Quick Start

Reproduce saved metrics without models, CUDA, ROS or hardware:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
upv-reproduce reproduce --output runs/reproduced
python -m unittest discover -s tests -v
```

Run from this repository root. Use a new output directory each time. Without an
installation, use `PYTHONPATH=src python -m upv_vlm_contact` in place of
`upv-reproduce`. A wheel alone does not contain the datasets and reference code:
use the complete source checkout.

## Computational Pipeline

```text
Saved RGB-D + material query
  -> GroundingDINO-Tiny + SAM2.1 Hiera-Small
  -> CLIP ViT-B/32 material verification / selected mask
  -> object geometry + length-dependent A* contact candidates
  -> L6 two-contact crop images
  -> Qwen2.5-VL-32B /infer_multi
  -> highest-scoring model-labeled usable pair, or NO_SAFE_ANCHOR
  -> mask-based and local depth point-cloud path lengths
```

`upv-pipeline` runs the computational stages together or as ordered subsets.
It defaults to validation only. These input paths are placeholders:

```bash
upv-pipeline --run-dir runs/example \
  --rgb /path/to/raw_rgb.png --depth /path/to/depth_raw.npy \
  --camera-info /path/to/camera_info.json --material 'concrete block' \
  --stages perception anchors
```

Append `--run` to prepare candidates. Start Qwen separately, then run:

```bash
upv-pipeline --run-dir runs/example --stages contact paths --run --run-inference
```

Each stage runs in a separate process and records artifacts, input provenance,
and timing. On a constrained GPU, finish perception **before** starting Qwen.
An externally running server stays resident even when pipeline calls are
sequential. No automatic server startup or hardware action is performed.

See [installation](docs/INSTALL.md) for environments/model dependencies and
[pipeline commands](docs/PIPELINE.md) for full and staged operation, saved-mask
reuse, input formats and validation limits.

## Optional Lab Robot Cycle

`scripts/run_prequantized_qwen_upv_cycle.py` supports the original UR3e rig using
staged perception, prequantized Qwen2.5-32B NF4 startup/inference/shutdown, and the
existing operator-confirmed approach/clamp/hold/release/home sequence. PL-200
values are entered manually after the cycle, not acquired by software.

It defaults to validation only. Real hardware requires `--execute` and a typed
confirmation for each cycle. The supplied rig-specific calibration/home/workspace
settings must be revalidated before use; they are not portable robot defaults.
This integration has offline tests only, not a fresh physical validation.
See [complete-cycle commands and prerequisites](docs/PREQUANTIZED_ROBOT_CYCLE.md).

## Experiment Reproduction

| Experiment | Cohort | Supported analysis |
|---|---|---|
| Perception | 27 original + 36 revision trials | Saved 57/63 correct (90.48%) |
| Contact73 | 45 E3 + 28 new10 pairs; 16 scenes | Classification and selected-pair usability; five methods |
| Contact82 | 50 E3 + 32 E45 pairs; 18 scenes | Classification and selected-pair usability; five methods |
| Path length V1/V2 | 15 cases, 30 manual paths each | MAE, RMSE, MAPE and material summaries |
| Runtime | E45 saved master rows | Stage summaries with imported-row caveats |

Contact73 is not Contact82 minus nine pairs: their scene cohorts differ. The five
contact methods are Qwen2.5-VL-32B, Qwen2.5-VL-3B, Qwen3-VL-2B, classical RGB+mask,
and classical RGB-D. The classical contact classifiers are untrained, but their
upstream GSAM2/CLIP perception is learned.

The main candidate-preparation config retains the historical CLIP policy; it is
not interchangeable with the later 63-trial CLIP-only ablation. Saved results
preserve those experiment-specific choices. Selection metrics use Python score
selection among model-labeled usable pairs, not necessarily the VLM's original
final-action field. Raw responses retain both for audit.

- [Reproduction commands](docs/REPRODUCTION.md): baselines, VLM inference, paths and plots.
- [Method definitions](docs/METHODS.md): classification and selection conventions.
- [Prompt-sensitivity system prompts](results/prompt_sensitivity/README.md): original and four wording variants.
- [Data documentation](docs/DATA.md): raw assets, checksums and provenance.

## Data Distribution

Dataset reuse permission is currently withheld pending a licensing decision;
see [Dataset Rights Notice](DATA_LICENSE.md). Code and dataset rights are separate.
Public availability is not a reuse license. Previously granted permissions remain unaffected.

Small result tables and manifests live in Git. Raw RGB-D, masks, crops and
detailed artifacts are separate verified archives:

```bash
upv-reproduce install-data contact --archive release_assets/contact.zip
upv-reproduce install-data perception --archive release_assets/perception.zip
upv-reproduce install-data path_length --archive release_assets/path_length.zip
upv-reproduce install-data runtime --archive release_assets/runtime.zip
```

The four archives have verified sizes and SHA256 digests in
`data/manifests/archives.json`, with download URLs for the public v0.1.0 release.
Git alone supports saved-table reproduction;
image-based reruns need the archives. Do not commit dataset ZIPs, model weights,
environments or generated runs. Historical `workspace/...` paths are provenance
identifiers resolved through archive indexes, not parent-workspace dependencies.

The later five-concrete / Contact89 extension is not yet imported into this
public snapshot's archive catalog. Do not claim it is included until that
additional release-data import is completed.

## Repository Layout

```text
src/upv_vlm_contact/  Public CLI, staged pipeline and reproduction adapters
configs/             Computational config and recorded environment provenance
results/             Saved predictions, prompts and metric evidence
data/                Dataset manifests, labels and archive checksums
reference/           Bundled research implementations used by explicit adapters
hardware/            Rig-specific hardware references and firmware
docs/                Installation, methods, datasets and release limitations
tests/               Offline regression and planning tests
tools/               Release checks and synthetic CPU integration test
scripts/             Explicit launchers, computational tools and experiments
release_assets/      Local dataset ZIPs (Git-ignored)
datasets/            Installed raw assets (Git-ignored)
runs/                New outputs (Git-ignored)
```

## Validation and Release

```bash
python tools/check_release.py
# Requires vision dependencies; no neural inference:
python tools/check_pipeline_cpu.py
# Optional exhaustive local archive verification:
python tools/check_release.py --check-archives
```

The release checker copies the repository to a detached temporary directory and
reproduces saved tables there. The CPU smoke test mocks the VLM decision; it does
not establish GPU/model correctness. Hardware references need fresh calibration,
operator safeguards and a separate deployment review. Raw PL-200 waveform exports
are not included, so waveform reprocessing is not currently reproducible.

Complete [the release checklist](docs/RELEASE_CHECKLIST.md) before publication.
Project code licensing is pending; see LICENSE. Available
third-party notices are under `docs/third_party/`; redistribution review remains
incomplete. No remote, commit or push is created automatically.
