# Full GPU pipeline with simulated hardware

This isolated experiment runs the existing simulation pipeline with real saved
RGB-D perception (GroundingDINO/SAM2/CLIP), geometry, newly generated contact
crops, Qwen, path length and planning. Robot and actuator execution is simulated;
no ROS2 camera or automatic/manual UPV acquisition is performed. Simulated
execution time must not be reported as physical motion or holding time.

Start the experimental server in terminal 1 if it is not already running:

```bash
.venv-vllm-bnb/bin/python scripts/start_fast_contact_experiment.py --run --max-image-pixels 200704 --eager --max-model-len 6144 --gpu-memory-utilization 0.96
```

Do not submit other requests while the simulation controls server sleep/wake.
Terminal 2:

```bash
.venv/bin/python scripts/test_vllm_full_gpu_simulation.py
.venv/bin/python scripts/test_vllm_full_gpu_simulation.py --run
```

Optional inputs: `--case-dir PATH --material 'concrete block' --axis major`.
Default input is local `datasets/gpu_smoke/concrete`.

The script attaches to localhost:8895; it does not load or stop that server.
It sleeps the server before perception, verifies sleep state, runs the existing
pipeline, clears unused local Torch cache, wakes Qwen at anchor selection,
then sleeps it before path/planning/simulated execution. Full perception-to-wake
memory compatibility remains experimentally unverified; failures are preserved.
The server remains sleeping after successful completion. Wake it manually before
using another client that does not manage residency.

An isolated subprocess replaces only the managed-model context and Qwen client
call in memory. Production pipeline files and checkpoints are unchanged. The
client translates the existing original multi-image prompts to the OpenAI API;
the 200704 pixel cap is an explicitly different preprocessing condition.

Outputs: `outputs/vllm_full_gpu_simulation/session_*/`. Existing pipeline output
directories retain RGB-D, masks, verification, geometry, anchor crops, responses,
path/plan and simulated execution artifacts and timing summaries. Additional
`vllm_request.json`, `vllm_response.json`, and `residency_timing.json` save full
request/response, sleep/wake durations and GPU memory snapshots, VLM request
duration and worker wall time, including failures. Worker wall time excludes
initial server startup and parent configuration preparation; obtain startup
separately from `outputs/fast_contact_servers/*/startup.json`. Nested pipeline
and request timers overlap and must not be summed blindly.

Initial validation covered syntax and no-run input checks only, not GPU inference.
