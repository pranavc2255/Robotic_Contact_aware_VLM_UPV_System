# Final Pipeline Runtime, Model Identity, and Model Residency Audit

Date: 2026-07-06  
Branch: `robot_upv_interactive_20260527`  
Supersedes/extends: `docs/dev_history/final_pipeline_runtime_summary_20260706_005311.md`

## Purpose

This document strengthens the previous runtime report by defining the timing boundaries, exact model identities, model-loading boundaries, GPU residency evidence, and what can and cannot be claimed for paper reporting. It is a documentation audit only. No experiments, model inference, RealSense capture, robot motion, RTDE commands, GSAM2 runs, or VLM/Qwen calls were executed.

## Executive Answer

- **Measured E45 runtime:** the previous report's end-to-end value remains the primary measured runtime: 94 successful E45 rows, mean total `98.877 s`, median `98.166 s`, with intentionally slow/safe robot motion. Source: `docs/dev_history/final_pipeline_runtime_summary_20260706_005311.md:L57-L83`.
- **Timing boundary:** `total_pipeline_timing_ms` starts inside `run_full_main_upv_vlm_v2_pipeline()` before config snapshot/stage execution and stops after the pipeline stages complete, just before the timing summary/result write. Source: `src/upv_vlm_v2/pipeline/main_pipeline.py:L417-L647`, especially `start = now_perf()` at `L433` and `total_timing_ms = elapsed_ms(start)` at `L629-L634`.
- **Qwen weight loading:** Qwen model weights are loaded by the separate HTTP server at server startup, not inside the per-run pipeline timing. Source: `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L29-L91` and server `main()` at `L389-L418`.
- **Qwen persistent server:** code-supported and documented. The runtime instructions explicitly use a three-terminal setup with a separately running Qwen server, and state that Qwen is loaded once to remain in GPU memory. Source: `docs/UPV_VLM_V2_RUNTIME_TERMINALS.md:L5-L9` and `L41-L64`.
- **Anchor-selection timing:** includes anchor candidate generation, crop/prompt artifact preparation, repeated VLM HTTP request/response calls, response parsing, postprocessing, and selection logic. It excludes Qwen/Llama model weight loading because that happens in the already-running server.
- **Target-selection timing:** includes the GroundingDINO/SAM2 subprocess and CLIP crop-verification subprocesses where configured. Because those subprocesses construct/load their model pipelines internally, target-selection timing likely includes perception model loading for those subprocess calls, but the exact load-vs-inference split is not logged.
- **Exact final/E45 perception models verified from E45 config snapshots:** GroundingDINO `IDEA-Research/grounding-dino-tiny`, SAM2 checkpoint `third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt`, SAM2 config `configs/sam2.1/sam2.1_hiera_s.yaml`, and CLIP `openai/clip-vit-base-patch32`. Source: `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164956/01_cycles/cycle_001/pipeline_full_session/config_snapshot.yaml:L14-L24`.
- **Qwen32 live vs offline:** E45 contains Qwen32-named live backend artifacts/configs and also later live Llama backend artifacts. The offline rerank explicitly records `local_models/Qwen2.5-VL-32B-Instruct` as the loaded Qwen model. The repository does **not** prove that every live E45 Qwen32-labeled run logged the exact Qwen server model path or quantization mode. Paper text should distinguish live lower-memory Llama deployment from offline Qwen32 reranking when discussing the 20260529 robotic analysis.
- **All models concurrent on one 24 GB GPU:** not proven. The repository proves local server/subprocess staging and records an RTX 4090 in some E45 artifacts, but does not directly log a moment where Qwen2.5-VL-32B, GroundingDINO, SAM2, and CLIP were simultaneously resident on the same GPU.

## Exact Model Inventory

| Model role | Exact model/checkpoint/config | Backend or code path | Quantization / precision | Loaded where | Used in E45 measured runtime? | Evidence | Paper-safe statement |
|---|---|---|---|---|---|---|---|
| Qwen/VLM live backend, Qwen32-named E45 runs | Backend name `v2a_qwen32_single_anchor_contact_scoring`; server URL `http://127.0.0.1:8899`, endpoint `/infer`. Exact live server `model_path` not found in representative older E45 server responses. | `src/upv_vlm_v2/anchor_selection/qwen32_single_anchor_contact_scorer.py` | Server supports fp16/bf16, 4-bit, and 8-bit. Live quantization not directly logged in representative E45 Qwen artifacts. | External Qwen HTTP server | Yes, for E45 outputs/configs using the Qwen32 single-anchor backend, but exact loaded model path per live run is not directly logged in those artifacts. | E45 config `config_snapshot.yaml:L37-L46`; scorer HTTP timing in `qwen32_single_anchor_contact_scorer.py:L321-L378`; server response lacks model path in `.../qwen32_single_anchor_responses/source_A1_server_response.json`. | The live E45 configuration used a Qwen32-named single-anchor HTTP backend. Do not state the exact live Qwen32 model path/quantization unless the matching server startup log is recovered. |
| Qwen32 offline/supporting backend | `local_models/Qwen2.5-VL-32B-Instruct` | Qwen server rerun on saved E45 anchor crops | Quantization not recorded in rerank summary; server code supports `--load-4bit`, `--load-8bit`, and `--device-map auto`. | External Qwen HTTP server | No for live robot execution; yes for offline rerank analysis on saved artifacts. | `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260529_062233/analysis_qwen32_multi_anchor_rerank/qwen32_rerank_summary.md:L1-L17`; no live camera/robot/full rerun at `L5`. | Qwen2.5-VL-32B-Instruct was verified for offline reranking of saved E45 crops, not as a proven live model for every E45 robotic run. |
| Llama live lower-memory VLM backend | `local_models/Llama-3.2-11B-Vision-Instruct`; model type `mllama`; class `MllamaForConditionalGeneration`; processor `MllamaProcessor` | `llama32_single_anchor_l6_taxonomy_scoring`, endpoint `/infer` | Server supports 4-bit/8-bit/fp16/bf16; artifact health does not list quantization flag. | External Llama HTTP server | Yes, for 20260529 E45 live artifacts using Llama taxonomy backend. | Health artifact `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260529_061220/.../llama32_single_anchor_server_health.json:L1-L15`; representative `pipeline_result.json:L2498-L2555`; config `configs/v2/deploy_ur3e_realsense_upv_real_plan_check_llama32_single_anchor_l6.yaml:L50-L70`. | The live 20260529 robot-session analysis used a lower-memory Llama-3.2-11B-Vision server backend; Qwen32 was separately rerun offline on saved crops. |
| GroundingDINO | `IDEA-Research/grounding-dino-tiny` | Grounded-SAM2 wrapper subprocess | Not logged | Launched by target-selection subprocess | Yes, in target-selection stage where configured | E45 config `config_snapshot.yaml:L14-L24`; wrapper `src/upv_vlm_v2/perception/grounded_sam2.py:L140-L212`. | Final target perception used GroundingDINO tiny through the Grounded-SAM2 subprocess wrapper. |
| SAM2 | Checkpoint `third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt`; config `configs/sam2.1/sam2.1_hiera_s.yaml` | Grounded-SAM2 wrapper subprocess | Not logged | Launched by target-selection subprocess | Yes, in target-selection stage where configured | E45 config `config_snapshot.yaml:L14-L24`; wrapper `grounded_sam2.py:L149-L177`. | Final target perception used SAM2.1 Hiera small in the E45 config snapshot. |
| CLIP/OpenCLIP crop verifier | `openai/clip-vit-base-patch32` | `transformers.pipeline("zero-shot-image-classification")` subprocess | Not logged | CLIP verifier subprocess | Yes when `use_clip_crop_verifier: true` | E45 config `config_snapshot.yaml:L25-L34`; `src/upv_vlm_v2/perception/clip_verifier.py:L529-L555` and `L590-L632`. | Crop-level verification used the OpenAI CLIP ViT-B/32 Hugging Face model id where enabled. |
| Qwen3 / other exploratory VLMs | Various support code exists, e.g. `run_qwen3_anchor_server.py` | Not final E45 evidence in this audit | Not applicable | Not applicable | No final/E45 measured evidence found | Search hits in `src/upv_vlm_v2/vlm/run_qwen3_anchor_server.py` only. | Do not mix Qwen3 or older exploratory backends into final E45 runtime claims unless a matching run artifact is cited. |

## Timing Boundary

The total pipeline timer is code-supported as follows:

- `src/upv_vlm_v2/pipeline/main_pipeline.py:L417-L433`: `run_full_main_upv_vlm_v2_pipeline(...)` starts and calls `start = now_perf()`.
- `src/upv_vlm_v2/pipeline/main_pipeline.py:L453-L477`: runtime diagnostics, output session directories, config snapshot, and manifest setup occur after this timer starts.
- `src/upv_vlm_v2/pipeline/main_pipeline.py:L524-L608`: the main stages are executed: pre-capture home, capture, target selection, geometry, anchor selection, path length, robot planning, and execution.
- `src/upv_vlm_v2/pipeline/main_pipeline.py:L629-L634`: `total_timing_ms = elapsed_ms(start)` is computed and written to `timing_summary.json`.

Therefore, `total_pipeline_timing_ms` includes the pipeline call overhead and all timed pipeline stages. It excludes anything that happened before the pipeline process/function was called, including starting ROS2, starting RealSense externally, starting the Qwen/Llama server, and loading VLM model weights into that server. It also excludes file writes that occur after the final `elapsed_ms(start)` assignment.

Representative E45 stage timing confirms the measured stage boundaries:

- `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164956/01_cycles/cycle_001/pipeline_full_session/timing_summary.json:L2-L76`
- total `115217.344 ms`, target selection `9424.565 ms`, anchor selection `39579.349 ms`, path length `871.227 ms`, and execution `62780.694 ms`.

## Qwen Server Loading Boundary

Qwen loading happens in the server, outside the pipeline:

- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L29-L39`: `load_model(...)` is documented as loading Qwen once at server startup.
- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L42-L64`: prints loading status and initializes processor/model loading imports.
- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L71-L83`: configures 4-bit/8-bit/dtype and calls `QwenModel.from_pretrained(...)`.
- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L88-L91`: sets `MODEL_PATH` and prints `Model loaded`.
- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L389-L418`: CLI parses model-loading flags, calls `load_model(...)`, then starts `HTTPServer(...).serve_forever()`.

Per-request inference happens later:

- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L126-L179`: `/infer` builds the chat template, processes image inputs, calls `MODEL.generate(...)`, and decodes output.
- `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L199-L260`: `/infer_multi` does the same for multi-image requests.

The runtime setup document separately instructs users to start the Qwen server before the pipeline:

- `docs/UPV_VLM_V2_RUNTIME_TERMINALS.md:L5-L9`: Terminal 2 keeps Qwen running; loading once keeps it in GPU memory.
- `docs/UPV_VLM_V2_RUNTIME_TERMINALS.md:L41-L64`: Qwen server startup command and expected `Model loaded` / `Listening` messages.
- `docs/UPV_VLM_V2_RUNTIME_TERMINALS.md:L72-L85`: pipeline command is run in a separate terminal after the server exists.

**Conclusion:** E45 `total_pipeline_timing_ms` and `anchor_selection_timing_ms` include Qwen/Llama HTTP request-response and inference time, but not Qwen/Llama weight-loading time.

## Anchor-Selection Timing Boundary

`anchor_selection_timing_ms` starts in the pipeline stage wrapper:

- `src/upv_vlm_v2/pipeline/stages.py:L236-L298`, `run_anchor_selection_stage()`
- `start = now_perf()` at `L237`
- `run_anchor_selection(...)` call at `L271-L280`
- returned timing at `L295`

The anchor-selection function prepares geometry and artifacts before dispatching to the VLM backend:

- `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L453-L538`: axis domain, candidate counts, A1/A2/... candidate geometry, and opposing contact endpoints.
- `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L549-L650`: crop/tile artifact generation for the VLM inputs.

For the Qwen32 single-anchor backend:

- `src/upv_vlm_v2/anchor_selection/qwen32_single_anchor_contact_scorer.py:L321-L354`: constructs one request payload per candidate.
- `src/upv_vlm_v2/anchor_selection/qwen32_single_anchor_contact_scorer.py:L356-L358`: records `elapsed_ms` around `_post_json(...)`, so this nested timing includes local HTTP request/response plus server-side inference latency.
- `src/upv_vlm_v2/anchor_selection/qwen32_single_anchor_contact_scorer.py:L359-L378`: writes server response and parsed output.
- Representative E45 decision artifact records per-candidate `elapsed_ms` values, e.g. `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260527_164956/01_cycles/cycle_001/pipeline_full_session/artifacts/04_anchor_selection/qwen32_single_anchor_decision.json:L16-L19` and per-anchor entries around `L55`, `L95`, `L133`, and `L171`.

For the Llama live backend:

- `src/upv_vlm_v2/anchor_selection/llama32_single_anchor_l6_taxonomy_scorer.py:L202-L230`: prepares server URL, endpoint, prompt, and health check.
- `src/upv_vlm_v2/anchor_selection/llama32_single_anchor_l6_taxonomy_scorer.py:L291-L320`: loops over candidates and times each HTTP POST.

Boundary answer:

| Component | Included in `anchor_selection_timing_ms`? | Evidence |
|---|---|---|
| Candidate generation and A* geometry | Yes | `anchor_selection_stage.py:L453-L538` called inside stage timer |
| Crop/tile/image artifact preparation | Yes | `anchor_selection_stage.py:L549-L650` called inside stage timer |
| Prompt construction | Yes | Scorer functions are called inside stage timer |
| Image/crop encoding/path packaging | Yes | Scorer request payload construction inside stage timer |
| Qwen/Llama HTTP request-response | Yes | Per-candidate `elapsed_ms` wraps `_post_json(...)` |
| Qwen/Llama server-side inference | Yes, as part of HTTP latency | Server handles `/infer` by calling `MODEL.generate(...)` |
| Response parsing/postprocessing | Yes | Scorer parsing/writes happen inside stage timer |
| Repeated calls | Yes | Single-anchor backends loop over A* candidates |
| Model weight loading | No | Server loads before `serve_forever()` and before pipeline starts |

## Target-Selection / GSAM2 Timing Boundary

`target_selection_timing_ms` starts and stops in:

- `src/upv_vlm_v2/pipeline/stages.py:L115-L168`, `run_target_selection_stage()`.

Inside that stage, target selection calls:

- `src/upv_vlm_v2/perception/target_selection_stage.py:L279-L285`: target-selection entry point.
- `src/upv_vlm_v2/perception/target_selection_stage.py:L386-L392`: `run_grounded_sam2_target_detection(...)`.
- `src/upv_vlm_v2/perception/target_selection_stage.py:L436-L442`: `verify_candidate_crops_with_clip(...)`.

Grounded-SAM2/GroundingDINO/SAM2 boundary:

- `src/upv_vlm_v2/perception/grounded_sam2.py:L140-L177`: reads detector model id, SAM2 checkpoint, and SAM2 config.
- `src/upv_vlm_v2/perception/grounded_sam2.py:L200-L212`: launches the Grounded-SAM2 demo as a subprocess and times the whole subprocess with `runtime_s`.

CLIP boundary:

- `src/upv_vlm_v2/perception/clip_verifier.py:L529-L555`: the CLIP verifier subprocess imports Transformers and constructs a `zero-shot-image-classification` pipeline for `model_id`.
- `src/upv_vlm_v2/perception/clip_verifier.py:L590-L632`: `verify_candidate_crops_with_clip(...)` calls that subprocess when CLIP verification is enabled.

Interpretation:

- `target_selection_timing_ms` includes Grounded-SAM2 subprocess runtime, including GroundingDINO/SAM2 inference and likely cold model loading inside that subprocess.
- `target_selection_timing_ms` includes CLIP verification subprocess runtime where enabled, including likely CLIP model loading inside the subprocess.
- The repository does not split target selection into GroundingDINO-only, SAM2-only, CLIP-load, CLIP-inference, or mask-postprocessing timings. Those must be logged in future runs for precise paper substage timing.

## Model Residency and GPU Concurrency

| Claim | Evidence found | Evidence strength | Paper-safe wording |
|---|---|---|---|
| Qwen was intended to run as a persistent server | Runtime instructions use a separate Qwen terminal and say loading once keeps it in GPU memory. | Code-supported and documented | The VLM server was designed as a persistent local HTTP service rather than a per-run subprocess. |
| Qwen/Llama weight loading is excluded from per-run E45 timing | Server loads model before `serve_forever()`; pipeline calls server URL during anchor selection. | Code-supported | Per-run timing includes request/inference latency to an already-running server, not VLM cold-start loading. |
| Qwen request time is included | Scorer wraps HTTP POST in `time.perf_counter()` and stores `elapsed_ms`; stage timer wraps scorer. | Code-supported and artifact-supported | Anchor-selection timing includes VLM HTTP request-response/inference latency. |
| GroundingDINO/SAM2/CLIP timing is included in target-selection timing | Target-selection stage calls Grounded-SAM2 and CLIP subprocess wrappers under its timer. | Code-supported | Target-selection timing includes these subprocesses, but submodel load/inference splits are not recorded. |
| GroundingDINO/SAM2/CLIP were resident concurrently with Qwen32 on a 24 GB GPU | No direct concurrent GPU-residency log found. | Unknown/unproven | Do not claim all models were simultaneously resident on one GPU. |
| Staged/sequential execution | Qwen/Llama server is separate; Grounded-SAM2 and CLIP are subprocesses called during target selection; offline Qwen32 rerank states no full pipeline rerun. | Code-supported; some execution details inferred | The workflow used staged model execution and local HTTP VLM serving; exact simultaneous GPU residency was not logged. |
| Qwen32 live vs offline | Offline rerank explicitly logs Qwen32 model path; manuscript script says live robot-session analysis used lower-memory Llama while Qwen32 was rerun offline. | Direct for offline Qwen32; supporting for live/offline distinction | State Qwen32 was used for offline reranking of saved E45 crops, while 20260529 live robot-session artifacts used Llama. |
| Exact model names used in final measured timing | Perception model names in config snapshots; Llama live model in health artifact; Qwen32 offline model in rerank health. | Direct for those artifacts; incomplete for older live Qwen32 server path | Cite exact model names only where logged; label older Qwen32 live exact path as unverified if no server health exists. |

## GPU Memory Evidence

Directly found evidence:

- Current shell `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` failed with `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- Representative E45 Llama live artifact reports `NVIDIA GeForce RTX 4090` with `24564 MiB` total memory: `outputs/v2_experiments/e45_robot_upv_repeatability/session_20260529_061220/cases/brick_01/reading_001/pipeline_result.json:L2498-L2529`.
- The same artifact reports Llama server health with `cuda_memory_total_mib: 24080.875`: `pipeline_result.json:L2545-L2555`.
- Standalone Llama health file confirms `local_models/Llama-3.2-11B-Vision-Instruct`, `device: cuda`, and `cuda_memory_total_mib: 24080.875`: `.../llama32_single_anchor_server_health.json:L1-L15`.
- Qwen server code comment says for Qwen2.5-VL-32B on a 24 GB RTX 4090, use `--load-4bit --device-map auto`: `src/upv_vlm_v2/vlm/run_qwen_anchor_server.py:L29-L39`.
- Qwen server code imports `BitsAndBytesConfig` and implements 4-bit/8-bit loading: `run_qwen_anchor_server.py:L44-L83`.
- Llama server code also imports `BitsAndBytesConfig`, implements 4-bit/8-bit loading, and logs memory before/after load: `src/upv_vlm_v2/vlm/run_llama_vision_anchor_server.py:L66-L127`.

Not found in final/E45 logs during this audit:

- A direct CUDA out-of-memory traceback proving a failed concurrent Qwen32 + GSAM2/SAM2/GroundingDINO run.
- A per-run log showing Qwen32, GroundingDINO, SAM2, and CLIP all simultaneously loaded on one GPU.
- Qwen32 live server startup log with exact model path, quantization mode, load duration, and before/after VRAM for the older Qwen32-named E45 runs.
- Per-submodel target-selection VRAM snapshots.

Paper-safe conclusion: the repository supports that the system was run with staged model execution and an external VLM server because of memory pressure, and it directly records a 24 GB RTX 4090 in representative E45 artifacts. It does not prove an exact VRAM threshold for simultaneous model residency.

## Corrected Runtime Interpretation

The measured E45 full-pipeline total reported in the previous runtime summary, `98.877 s` mean over 94 successful rows, should be interpreted as:

- **with VLM server already loaded:** yes, for server-backed Qwen/Llama anchor selection.
- **Qwen/Llama weight loading excluded:** yes, code-supported.
- **Qwen/Llama request/inference included:** yes, inside anchor-selection timing.
- **perception model loading included:** likely included for the Grounded-SAM2 and CLIP subprocesses launched during target selection, but not split from inference.
- **exact model variants specified:** perception models and Llama live model are directly logged; offline Qwen32 model is directly logged; older live Qwen32 server model path/quantization is not directly logged in representative artifacts.
- **slow robot motion included:** yes. The previous report shows robot execution mean `60.037 s` from E45 successful rows.
- **UPV acquisition/export included:** external Pundit UPV waveform/TOF export was not integrated in the current paper draft; the pipeline config note says external Pundit use was manual/external in at least one E45 config snapshot, not timed as an integrated acquisition stage.

## Paper-Ready Wording

The final robotic UPV/VLM pipeline was timed from E45 robotic repeatability logs using stage timings written by the main pipeline. Across 94 successful readings, the measured end-to-end runtime was 98.9 s on average (median 98.2 s). This timing includes RGB-D capture, GroundingDINO/SAM2 target segmentation, CLIP crop verification where enabled, geometry extraction, VLM-based contact-anchor scoring through a local HTTP vision-language-model server, selected-anchor path-length estimation, robot pose planning, and intentionally conservative robot execution. The VLM server was started before the per-reading pipeline and kept its model resident; therefore, per-reading timing includes VLM request/response and generation latency but excludes VLM weight-loading time. E45 artifacts directly identify the final perception models as GroundingDINO `IDEA-Research/grounding-dino-tiny`, SAM2.1 Hiera small, and `openai/clip-vit-base-patch32`; representative live robot-session artifacts identify the lower-memory Llama-3.2-11B-Vision backend, while Qwen2.5-VL-32B-Instruct was directly logged for offline reranking of saved E45 anchor crops. Because simultaneous residency of Qwen32, GroundingDINO, SAM2, and CLIP on the same GPU was not directly logged, GPU-concurrency claims should be stated as staged execution rather than proven concurrent deployment.

## Paper-Ready Runtime Table Caption

Table X. Runtime breakdown of the final robotic UPV/VLM pipeline from 94 successful E45 robotic repeatability readings. Values include model request/inference latency during target and anchor selection and deliberately slow robot motion used for safe laboratory contact tests. The VLM server was loaded before each per-reading pipeline run, so VLM model weight-loading time is not included. Perception model identities are GroundingDINO `IDEA-Research/grounding-dino-tiny`, SAM2.1 Hiera small, and `openai/clip-vit-base-patch32`; VLM backend identity should be reported separately for live Llama runs and offline Qwen2.5-VL-32B reranking.

## What To Log In Future Runs

Required logging for paper-grade runtime/model-residency claims:

- Qwen/Llama server startup timestamp.
- VLM model name/path, model class, processor class, and endpoint.
- VLM model load duration.
- VLM quantization/precision: 4-bit, 8-bit, fp16, bf16, fp32.
- VLM `torch_dtype`, `device_map`, `attn_implementation`, max image pixels, candidate/image limits, max tokens, and batch settings.
- VLM VRAM before and after load.
- Per-run VLM health snapshot stored in every pipeline result, including model path and quantization.
- GroundingDINO model id/checkpoint/config.
- SAM2 checkpoint/config.
- CLIP/OpenCLIP model id and device.
- Target model load duration and model VRAM before/after load.
- Per-run GPU memory snapshots before/after every model stage and while servers are resident.
- Explicit boolean for whether VLM and perception models are concurrently resident.
- Cold-start timing and warm-inference timing separately for GSAM2, CLIP, and VLM.
- Number of VLM calls per run and per-call HTTP latency, generation time, prompt construction time, image/crop encoding time, parsing time, and retry count.
- Robot segment distances and speed/acceleration settings per run.
- Integrated UPV acquisition/export timing once Pundit waveform and TOF logging are added.

