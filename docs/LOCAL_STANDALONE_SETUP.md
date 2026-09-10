# Local standalone installation (2026-09-10)

This checkout has a separate `.venv`, local prequantized Qwen2.5-32B weights,
Grounded-SAM-2 checkout/checkpoints, and local Hugging Face cache for DINO-Tiny,
CLIP ViT-B/32 and BERT. Assets were copied from the working development install,
not linked to it. Python packages were copied into a fresh virtual environment
to preserve the working versions; SAM2's editable registration is installed
against this checkout. This is a local installation, not a fresh-download test.

The cycle launcher uses `.cache/huggingface` in offline mode. Models, environments,
cache, third-party checkout and generated outputs are ignored by Git. Publishing
the source therefore does NOT publish this approximately tens-of-GB installation.
CUDA/driver, Python 3.10 and ROS2 Humble are system prerequisites.

From this checkout, in a fresh terminal:

```bash
unset PYTHONPATH
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py --dry-run
```

Keep the ROS2 camera publisher running separately. Stop other GPU model servers.
Run the no-motion camera/perception/planning check first:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --material 'concrete block' --axis major --plan-check
```

After reviewing the plan and verifying the original rig calibration/clearances:

```bash
.venv/bin/python scripts/run_prequantized_qwen_upv_cycle.py \
  --material 'concrete block' --axis major --execute --max-cycles 1
```

Qwen is automatically started after perception and stopped before execution.
The operator reads the PL-200 during the timed hold and enters values after
release/home. No software UPV acquisition is added.

Only offline checks are performed during setup. Successful installation is not
a completed GPU-inference or physical-cycle acceptance test. See
`PREQUANTIZED_ROBOT_CYCLE.md` for safety and timing boundaries. Moving this
checkout requires recreating the virtualenv/editable installs (standard venvs
are not relocatable).

## Completed checks

- Local launcher dry-run passed, including Qwen shard existence checks.
- DINO, CLIP and BERT snapshots resolve from the local cache without networking.
- No external symlinks in model, third-party or cache assets.
- Local Torch 2.11.0+cu126, Transformers 5.6.2, SAM2 and serial imports passed.
- ROS2 import passed after sourcing `/opt/ros/humble/setup.bash` (no node started).
- Four mocked lifecycle/selection tests passed using the new environment.
- Installed-package snapshot: `configs/local_standalone_environment_20260910.txt`.
- Disk use: approximately 7.5G environment, 19G Qwen, 3.2G third-party and 2.2G cache.

`pip check` reports two inherited environment issues: optional ROS parameter
generator package `generate-parameter-library-py` lacks `typeguard`, and `decord`
0.6.0 has an unsupported platform tag. These are not exercised by the image-only
offline tests. The setup is not claimed to have a clean dependency audit or
validated video handling. Model-library versions were deliberately not upgraded.
