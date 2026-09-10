# Installation

Run from a **source checkout** of this repository. `reference/`, `configs/` and
`results/` are required resources; the Python wheel alone is not a complete
distribution. The parent development repository is not needed.

## 1. Saved-result reproduction (CPU, no models)

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -e .
upv-reproduce reproduce --output runs/reproduced
python -m unittest discover -s tests -v
```

This tier needs no CUDA, ROS, camera, robot or dataset download. For image-based
CPU stages and plots:

```bash
python -m pip install -e '.[vision]'
python tools/check_pipeline_cpu.py
```

The smoke check uses a synthetic mask and mocked selection; it is not a model
accuracy test or an end-to-end GPU validation.

## 2. Perception environment

Use a separate environment to avoid coupling perception to the VLM server's
Transformers/CUDA requirements. Install a CUDA-compatible PyTorch/torchvision
pair appropriate to your host before installing these extras. The repository's
`configs/build_environment.json` is provenance, **not** a validated universal
lockfile. In particular, fresh GPU installation has not been tested by this
release audit.

```bash
python3 -m venv .venv-perception
source .venv-perception/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vision,perception]'
mkdir -p third_party
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
git -C third_party/Grounded-SAM-2 checkout b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
python -m pip install --no-build-isolation -e third_party/Grounded-SAM-2
```

That revision is the checkout recorded in the development workspace. The
repository uses the Hugging Face GroundingDINO demo, not the optional native
GroundingDINO extension. Follow that checkout's installation requirements if its
CUDA extension build fails. Third-party downloads are never run by the pipeline's
dry run.

Obtain `sam2.1_hiera_small.pt` using the checkpoint links/instructions in
`third_party/Grounded-SAM-2/README.md`, and place it at:

```text
third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt
```

The SAM2 config name is `configs/sam2.1/sam2.1_hiera_s.yaml`. The configured
GroundingDINO and CLIP model IDs are `IDEA-Research/grounding-dino-tiny` and
`openai/clip-vit-base-patch32`; the first explicit perception execution may
download their weights through Hugging Face. Obtain/cache weights beforehand on
an offline deployment host. Respect upstream licenses; no weights are bundled.

## 3. VLM server environment

```bash
python3 -m venv .venv-vlm
source .venv-vlm/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vision,vlm]'
```

Install the model files under `local_models/` after accepting any applicable
upstream terms. Qwen3 requires a Transformers build supporting `qwen3_vl`;
Qwen2.5 requires `qwen2_5_vl`. There is no automatic fallback to another model.
Use a tested dependency set for your selected model, then record it with
`python -m pip freeze > runs/environment.txt` (create `runs/` first).

Print a server command without loading anything:

```bash
python scripts/start_vlm_server.py --model qwen25-32b \
  --model-path local_models/Qwen2.5-VL-32B-Instruct --load-4bit
```

Add `--run` to start it. For a 24 GB GPU, the documented Qwen32 deployment uses
4-bit loading; this is not a guarantee of fit for every input size or dependency
version. Do not leave this server resident while running perception on a GPU
that cannot hold both stacks. Use the staged workflow in [PIPELINE.md](PIPELINE.md).

## 4. Datasets

The Git checkout includes small result tables; raw assets are separate verified
archives. Download the four ZIPs from the repository's v0.1.0 release and
place them in `release_assets/`, then install with checksum verification:

```bash
upv-reproduce install-data contact --archive release_assets/contact.zip
upv-reproduce install-data perception --archive release_assets/perception.zip
upv-reproduce install-data path_length --archive release_assets/path_length.zip
upv-reproduce install-data runtime --archive release_assets/runtime.zip
```

See [DATA.md](DATA.md) for layout and checksums. A Git clone without these archives
can reproduce saved metrics, but cannot rerun image-based experiments.
