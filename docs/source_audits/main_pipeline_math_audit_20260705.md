# Main Pipeline Mathematics Audit - 2026-07-05

## Executive Summary

This audit documents the main mathematical operations in the UPV/VLM robotic pipeline and links each operation to the source code that implements it. It is a documentation-only audit: no experiments, GSAM2, VLM/Qwen, RealSense, robot motion, or RTDE commands were run.

The main pipeline call chain is:

1. CLI entry: `src/upv_vlm_v2/cli/run_full_main_upv_vlm_v2_pipeline.py:L19-L53` parses requested material, axis mode, execution mode, input RGB-D paths, and optional VLM backend settings.
2. Orchestrator: `src/upv_vlm_v2/pipeline/main_pipeline.py:L430-L540` validates mode/material/axis/config, creates a session, writes manifests, and runs stages in `MODE_ORDER`.
3. Stage implementations: `src/upv_vlm_v2/pipeline/stages.py:L39-L112`, `L115-L233`, `L236-L360`, and `L365-L490` execute capture, target selection, geometry, anchor selection, local path length, robot planning, and execution.
4. Final selected-anchor path-length evaluation used for V1/V2 paper outputs is implemented offline in `scripts/evaluate_new_capture_mask_and_depth_path_length.py`; this selected-anchor evaluator uses `selected_anchors.json` A* endpoints and directions, not MinAreaRect fallback axes, for final metrics.

Final V2 selected-anchor path-length method note: the depth point-cloud-based estimate uses local sampling bands around selected A* paths and robust projected percentile endpoints at `Q_0.0025` and `Q_0.9975`. Whole-ROI depth and MinAreaRect fallback are exploratory/diagnostic methods, not the final selected-anchor method.

## Source Inspection Commands

Safe read-only inspection commands used for this audit included:

```bash
git branch --show-current
git status --short
rg -n -i "clip|cosine|normalize|logit|similarity|softmax|argmax|threshold" src scripts
rg -n -i "minAreaRect|boxPoints|contour|moments|mask|center|axis|major|minor|orientation" src scripts
rg -n -i "anchor|candidate|A[0-9]|cross_section|axis_search|transducer|diameter|spacing|edge_quality|score_edge_quality|compute_anchor_cross_section_hits|build_axis_search_domain" src scripts
rg -n -i "back.project|backproject|intrinsics|fx|fy|cx|cy|depth|point.cloud|pointcloud|camera_info" src scripts
rg -n -i "path.length|path_length|L_mask|L_depth|L_pc|quantile|percentile|0.0025|0.9975|band_half|depth_tolerance|support_points" src scripts
rg -n -i "camera.to.tool|tool0|transform|homogeneous|T_.*camera|calibration|eye.hand|charuco" src scripts configs docs
rg -n -i "rms|tof|time.of.flight|coefficient|variation|cv|waveform|amplitude" src scripts
rg -n -i "precision|recall|specificity|f1|accuracy|balanced|confusion|mae|rmse|mape|absolute error|signed error" src scripts
```

## Pipeline Call Chain References

- CLI argument parsing: `src/upv_vlm_v2/cli/run_full_main_upv_vlm_v2_pipeline.py:L19-L53`, function `parse_args`.
- CLI summary printing of selected target, anchor, path length, and robot plan: `src/upv_vlm_v2/cli/run_full_main_upv_vlm_v2_pipeline.py:L64-L167`, function `_print_summary`.
- Pipeline initialization, validation, runtime diagnostics, and session setup: `src/upv_vlm_v2/pipeline/main_pipeline.py:L430-L495`, function `run_full_main_upv_vlm_v2_pipeline`.
- Stage dispatch: `src/upv_vlm_v2/pipeline/main_pipeline.py:L513-L540` and following blocks in `run_full_main_upv_vlm_v2_pipeline`.
- Capture stage: `src/upv_vlm_v2/pipeline/stages.py:L39-L112`, function `run_capture_stage`.
- Target selection stage: `src/upv_vlm_v2/pipeline/stages.py:L115-L168`, function `run_target_selection_stage`.
- Geometry stage: `src/upv_vlm_v2/pipeline/stages.py:L171-L233`, function `run_geometry_stage`.
- Anchor selection stage: `src/upv_vlm_v2/pipeline/stages.py:L236-L298`, function `run_anchor_selection_stage`.
- Local path-length stage: `src/upv_vlm_v2/pipeline/stages.py:L301-L360`, function `run_local_path_length_stage`.
- Robot planning stage: `src/upv_vlm_v2/pipeline/stages.py:L365-L490`, function `run_robot_planning_stage`.

---

## M1. Grounded-SAM2 Detection Ordering and Candidate Scores

**Pipeline role:** Open-vocabulary detector/mask proposals are produced by GroundingDINO + SAM2. Model inference is external; repo-side math is score extraction and descending sort by mask/detection score.

**Paper equation:** Text only. If needed: candidates are ordered by

\[
C_{(1)},\ldots,C_{(n)}=\operatorname{sort}_{C_i}(-s_i), \quad s_i=\text{mask\_score}_i \;\text{or}\; \text{score}_i.
\]

**Variables:** `s_i` is the detector/mask confidence for candidate `i`.

**Implementation:** `src/upv_vlm_v2/perception/grounded_sam2.py:L111-L137`, functions `_load_results`, `_score_value`.

**Code excerpt:**

```python
score = _score_value(ann.get("score"))
detection_score = _score_value(ann.get("detection_score"))
mask_score = _score_value(ann.get("mask_score", ann.get("score")))
detections.sort(key=lambda item: float(item.get("mask_score") or item.get("score") or 0.0), reverse=True)
```

**Manuscript recommendation:** Explain in text only. The core detector is learned inference; the repo only records/sorts confidences.

---

## M2. Candidate Mask Bounding Boxes and Verification Crops

**Pipeline role:** For each mask candidate, build a bounding box, masked crop, and eroded interior texture crop used for material verification.

**Paper equation:** Text only. The tight bounding box is

\[
B(M)= [\min x,\min y,\max x+1,\max y+1] \quad \text{over foreground pixels } M(x,y)>0.
\]

Interior crop uses percentile bounds of eroded foreground pixels:

\[
x_L=Q_{0.20}(x)-2,\; x_R=Q_{0.80}(x)+3,\; y_T=Q_{0.20}(y)-2,\; y_B=Q_{0.80}(y)+3.
\]

**Implementation:**
- `_bbox_from_mask`: `src/upv_vlm_v2/perception/candidate_pool.py:L123-L127`.
- `_save_inner_texture_crop`: `src/upv_vlm_v2/perception/candidate_pool.py:L220-L283`.
- `build_candidate_pool`: `src/upv_vlm_v2/perception/candidate_pool.py:L332-L412`.

**Code excerpt:**

```python
ys, xs = np.nonzero(mask > 0)
return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
# ...
left = max(0, int(np.percentile(xs, 20)) - 2)
right = min(rgb.width, int(np.percentile(xs, 80)) + 3)
top = max(0, int(np.percentile(ys, 20)) - 2)
bottom = min(rgb.height, int(np.percentile(ys, 80)) + 3)
```

**Manuscript recommendation:** Explain in text or appendix; do not make it a main displayed equation unless crop construction is central to the target-selection method.

---

## M3. CLIP/Zero-Shot Crop-Text Material Scoring and Aggregation

**Pipeline role:** The verifier scores each candidate crop against multiple text prompts per material and aggregates prompt scores into material scores.

**Paper equation:** If describing material verification:

\[
S_m(C)=\begin{cases}
\max_{p\in P_m} s(C,p), & \text{max aggregation},\\
\frac{1}{|P_m|}\sum_{p\in P_m}s(C,p), & \text{mean aggregation}.
\end{cases}
\]

**Variables:** `C` is a candidate crop, `P_m` is the prompt set for material `m`, and `s(C,p)` is the zero-shot image-classification score returned for prompt `p`.

**Implementation:**
- Prompt sets: `src/upv_vlm_v2/perception/clip_verifier.py:L15-L49`.
- Aggregation: `src/upv_vlm_v2/perception/clip_verifier.py:L57-L70`, function `_aggregate`.
- External classifier call: `src/upv_vlm_v2/perception/clip_verifier.py:L529-L554`, function `_run_clip_subprocess`.
- Verifier loop: `src/upv_vlm_v2/perception/clip_verifier.py:L590-L620` and following lines, function `verify_candidate_crops_with_clip`.

**Code excerpt:**

```python
values = [float(item["score"]) for item in items]
if method == "mean":
    agg[material] = round(sum(values) / max(len(values), 1), 6)
else:
    agg[material] = round(max(values) if values else 0.0, 6)
```

**Manuscript recommendation:** Keep as a compact text equation or appendix equation. Avoid claiming embeddings are computed in this repo; the repo uses the HuggingFace zero-shot pipeline output scores.

---

## M4. Target Candidate Selection With Source Prior, Margins, and No-Match Gate

**Pipeline role:** Select a material-consistent crop/mask candidate, optionally preferring candidates whose original detection query matches the requested material, while rejecting absent/conflicting classes.

**Paper equation:** A simplified form is:

\[
F_i = S_{m^*}(C_i)+b\,\mathbf{1}[q_i=m^*],
\]

then choose the highest `F_i` subject to material-specific guards and a no-match gate. Candidate is rejected if requested score is below threshold or all candidates conflict.

**Implementation:**
- Requested score, margin, source bonus, final score: `src/upv_vlm_v2/perception/clip_verifier.py:L331-L364`, function `_select_t5_style_candidate`.
- Sorting and source-prior override: `src/upv_vlm_v2/perception/clip_verifier.py:L384-L403`.
- Concrete/brick/timber guards: `src/upv_vlm_v2/perception/clip_verifier.py:L412-L460`.
- No-match gate: `src/upv_vlm_v2/perception/clip_verifier.py:L248-L328`, function `_no_match_gate`.
- Artifact writing and `NO_VERIFIED_MATCH`: `src/upv_vlm_v2/perception/target_selection_stage.py:L39-L112`.

**Code excerpt:**

```python
item["requested_score"] = round(requested_score, 6)
item["score_margin"] = round(margin, 6)
item["final_score"] = round(requested_score + (source_bonus if source_match else 0.0), 6)
# ...
by_final = sorted(scored, key=lambda item: (_safe_float(item.get("final_score")), _safe_float(item.get("score_margin"))), reverse=True)
```

**Manuscript recommendation:** Present in text as rule-based verification after CLIP scoring. Put material-specific guard thresholds in appendix/config details.

---

## M5. Color/Texture Guard Features for Material Verification

**Pipeline role:** Compute lightweight RGB texture/color descriptors that guard CLIP against obvious material confusions.

**Paper equation:** Text or appendix. Representative features include saturation

\[
\operatorname{sat}=\mathbb{E}\left[\frac{\max(R,G,B)-\min(R,G,B)}{\max(R,G,B)+\epsilon}\right]
\]

and grayscale texture/edge variation from standard deviations.

**Implementation:** `src/upv_vlm_v2/perception/clip_verifier.py:L84-L140`, function `analyze_crop_color_texture`.

**Code excerpt:**

```python
sat = float(np.mean((maxc - minc) / np.maximum(maxc, 1e-6)))
gray = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
texture_variance = float(np.clip(np.std(valid_gray) / 0.22, 0.0, 1.0))
dx = np.diff(gray, axis=1)
dy = np.diff(gray, axis=0)
edge_variance = float(np.clip((np.std(dx) + np.std(dy)) / 0.26, 0.0, 1.0))
```

**Manuscript recommendation:** Appendix or implementation detail. These are heuristics, not core geometry.

---

## M6. Mask Geometry: Contours, Minimum-Area Rectangle, Center, and Axes

**Pipeline role:** Extract object geometry from the selected mask: foreground contour, rotated minimum-area rectangle, center, major/minor axes, and pixel extents.

**Paper equation:** OpenCV computes the minimum-area enclosing rectangle. Axis vectors are normalized edge vectors:

\[
\hat{a}=\frac{e_{\max}}{\|e_{\max}\|}, \quad \hat{b}=(-a_y,a_x).
\]

**Implementation:**
- Contour extraction: `src/upv_vlm_v2/geometry/mask_geometry.py:L48-L53`, function `_largest_external_contour`.
- Axis normalization: `src/upv_vlm_v2/geometry/mask_geometry.py:L36-L41`, function `normalize_vector`.
- MinAreaRect and box points: `src/upv_vlm_v2/geometry/mask_geometry.py:L83-L101`, function `compute_pixel_geometry`.
- Long/short axis extraction from box edges: `src/upv_vlm_v2/geometry/mask_geometry.py:L61-L80`, function `_long_and_short_rect_axes`.
- Geometry stage call: `src/upv_vlm_v2/pipeline/stages.py:L171-L233`, function `run_geometry_stage`.

**Code excerpt:**

```python
contour = _largest_external_contour(mask)
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect).astype(float)
center = [float(rect[0][0]), float(rect[0][1])]
major_axis_vector, minor_axis_vector, major_len, minor_len, major_angle, minor_angle = _long_and_short_rect_axes(box)
```

**Manuscript recommendation:** Keep as text with one equation for normalized axes if needed.

---

## M7. Pixel-to-Millimeter Scale From Depth and Camera Intrinsics

**Pipeline role:** Convert image-plane pixel spans to metric millimeters using representative depth and pinhole intrinsics.

**Paper equation:** For a local image direction `a=(a_x,a_y)` and depth `Z`, one pixel corresponds to

\[
\alpha(a,Z)=Z\sqrt{\left(\frac{a_x}{f_x}\right)^2+\left(\frac{a_y}{f_y}\right)^2},
\]

so `length_mm = span_px * alpha * 1000`.

**Implementation:**
- Intrinsics loading: `src/upv_vlm_v2/geometry/mask_geometry.py:L217-L232`, function `load_camera_intrinsics`.
- Representative masked depth: `src/upv_vlm_v2/geometry/mask_geometry.py:L235-L244`, function `representative_depth_m`.
- Pixel span conversion: `src/upv_vlm_v2/geometry/mask_geometry.py:L247-L256`, function `pixel_span_to_mm`.
- Geometry aggregation: `src/upv_vlm_v2/geometry/mask_geometry.py:L259-L279`, function `compute_mask_geometry`.

**Code excerpt:**

```python
ax, ay = normalize_vector(axis_vector)
meters_per_px = float(depth_m) * ((ax / float(intrinsics["fx"])) ** 2 + (ay / float(intrinsics["fy"])) ** 2) ** 0.5
return float(span_px) * meters_per_px * 1000.0
```

**Manuscript recommendation:** Keep as a displayed equation; this is central for metric path length and geometry.

---

## M8. Dominant Rectangle Geometry From Robust Rotated Edge Statistics

**Pipeline role:** Optional geometry refinement estimates a dominant body rectangle by rotating the mask into object coordinates and robustly estimating top/bottom/left/right edge positions.

**Paper equation:** Appendix. Robust edge positions are median/MAD-filtered quantiles of supported edge samples.

**Implementation:**
- Rotate mask to object frame: `src/upv_vlm_v2/geometry/mask_geometry.py:L124-L148`, function `estimate_dominant_rectangle_geometry`.
- Edge sample collection: `src/upv_vlm_v2/geometry/mask_geometry.py:L150-L170`.
- Inverse affine transform of corners: `src/upv_vlm_v2/geometry/mask_geometry.py:L174-L182`.
- Attach optional dominant rectangle: `src/upv_vlm_v2/geometry/mask_geometry.py:L282-L305`, function `add_dominant_rectangle_geometry`.

**Code excerpt:**

```python
matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
rotated_mask = cv2.warpAffine(...)
# collect top_edges, bottom_edges, left_edges, right_edges
inv = cv2.invertAffineTransform(matrix)
body_corners = [tuple(float(v) for v in apply_inv(corner)) for corner in local_corners]
```

**Manuscript recommendation:** Appendix only unless the dominant-rectangle mode is emphasized experimentally.

---

## M9. Axis Search Domain for A1/A2/... Anchor Candidate Centers

**Pipeline role:** Generate evenly spaced candidate anchor centers along the selected object axis, after trimming margins near object ends.

**Paper equation:** Let object center be `c`, selected axis unit vector `u`, extent `L_px`, margin ratio `r`, and candidate count `N`.

\[
s_{\min}=-\frac{L}{2}+rL,\quad s_{\max}=\frac{L}{2}-rL,
\]

\[
s_i=s_{\min}+\frac{i}{N-1}(s_{\max}-s_{\min}), \quad p_i=c+s_i u.
\]

**Implementation:**
- Unit vector and perpendicular: `src/upv_vlm_v2/anchor_selection/axis_parameterization.py:L21-L26`, function `normalize_xy`.
- Even samples: `src/upv_vlm_v2/anchor_selection/axis_parameterization.py:L29-L35`, function `_build_evenly_spaced_samples`.
- Search domain: `src/upv_vlm_v2/anchor_selection/axis_parameterization.py:L38-L65`, function `build_axis_search_domain`.
- Scalar-to-point: `src/upv_vlm_v2/anchor_selection/axis_parameterization.py:L68-L76`, function `map_axis_scalar_to_point`.
- Anchor generation call: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L486-L494`.

**Code excerpt:**

```python
half_extent = float(axis_extent_px) / 2.0
margin_px = float(axis_extent_px) * float(usable_axis_margin_ratio)
s_min = -half_extent + margin_px
s_max = half_extent - margin_px
samples = _build_evenly_spaced_samples(s_min, s_max, num_anchor_samples)
```

**Manuscript recommendation:** Keep as a displayed or compact equation; this explains the A* candidates.

---

## M10. Physical Candidate Count From Transducer Diameter

**Pipeline role:** Determine how many A* candidate centers to sample along an object axis, using physical axis length and UPV transducer diameter, then clamp to configured bounds.

**Paper equation:**

\[
N=\operatorname{clip}\left(\left\lfloor\frac{L_{axis,mm}}{D_{transducer,mm}}\right\rfloor, N_{min}, N_{max}\right).
\]

**Implementation:**
- Count rule constant: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L48`.
- Axis length extraction: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L96-L125`, function `_get_axis_length_mm`.
- Candidate count: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L157-L211`, function `_compute_v2a_anchor_count_metadata`.
- Applied in `run_anchor_selection`: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L464-L469`.

**Code excerpt:**

```python
raw_floor = int(math.floor(axis_length_mm / transducer_diameter_mm))
candidate_count = max(min_count, min(max_count, raw_floor))
```

**Manuscript recommendation:** Keep as a main-method equation if physical candidate spacing is discussed.

---

## M11. Opposing Cross-Section Contact Hits

**Pipeline role:** For each A* anchor center, search both directions along the local cross axis until leaving the mask. The last valid pixels are the opposing contact/path endpoints.

**Paper equation:** For center `p_i` and cross-axis unit vector `v`, endpoints are approximately

\[
a_i=p_i-t^-_i v, \quad b_i=p_i+t^+_i v,
\]

where `t^-_i` and `t^+_i` are the largest sampled distances that remain inside the mask.

**Implementation:**
- One-sided search loop: `src/upv_vlm_v2/anchor_selection/cross_section_sampler.py:L36-L55`, function `_search_one_side`.
- Two-sided hits: `src/upv_vlm_v2/anchor_selection/cross_section_sampler.py:L58-L70`, function `sample_cross_section_boundary_hits`.
- Candidate cross-section from axis scalar: `src/upv_vlm_v2/anchor_selection/cross_section_sampler.py:L73-L88`, function `compute_anchor_cross_section_hits`.
- Use in candidate loop: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L495-L538`, function `run_anchor_selection`.

**Code excerpt:**

```python
while distance <= max_search_distance + 1e-9:
    point = (anchor_xy[0] + direction_xy[0] * distance, anchor_xy[1] + direction_xy[1] * distance)
    if not _is_in_mask(mask, point):
        break
    last_xy = point
    last_dist = distance
    distance += step_size_px
```

**Manuscript recommendation:** Keep as a method equation; it is central to contact-pair geometry.

---

## M12. A* Candidate Contact Score and Deterministic Veto

**Pipeline role:** Score each candidate using local path span, centrality, side texture quality, and left/right consistency. Deterministic vetoes reject short spans or poor contact sides.

**Paper equation:**

\[
S=\operatorname{clip}\left(0.42S_{span}+0.28S_{center}+0.20S_{side}+0.10S_{consistency}\right).
\]

`S_side` is the minimum of the two side texture scores, and `S_consistency=100-|S_a-S_b|`.

**Implementation:** `src/upv_vlm_v2/anchor_selection/edge_quality.py:L25-L84`, function `score_edge_quality`.

**Code excerpt:**

```python
span_score = _clamp((local_path_length_px / max(object_extent_px, 1.0)) * 100.0)
centrality_score = _clamp((1.0 - abs(center_distance_ratio)) * 100.0)
texture_scores.append(_clamp(100.0 - roughness * 1.1))
paired_consistency = _clamp(100.0 - abs(texture_scores[0] - texture_scores[1]))
score = _clamp(0.42 * span_score + 0.28 * centrality_score + 0.20 * side_quality + 0.10 * paired_consistency)
```

**Manuscript recommendation:** Appendix or text. It is a heuristic prefilter; do not overstate as learned contact quality.

---

## M13. Deterministic Anchor Selection

**Pipeline role:** If no VLM backend is used or as a deterministic fallback, choose the highest-scoring non-vetoed candidate; if none survive, return `NO_SAFE_ANCHOR`.

**Paper equation:**

\[
A^*=\arg\max_{A_i: \neg veto_i} S_i.
\]

**Implementation:** `src/upv_vlm_v2/anchor_selection/hybrid_selector.py:L8-L43`, function `select_anchor_hybrid`.

**Code excerpt:**

```python
survivor_items = [item for item in enriched if str(item.get("anchor_id")) in survivors]
survivor_items.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("anchor_id"))))
selected = survivor_items[0]
```

**Manuscript recommendation:** Text only, unless deterministic baseline is compared quantitatively.

---

## M14. VLM Contact-Affordance Selection: Boolean Usability and Ranked Anchors

**Pipeline role:** VLM output is parsed into per-anchor left/right usability, scalar anchor score, and `anchor_usable`. The selected anchor is the first ranked usable anchor; otherwise the action is `NO_SAFE_ANCHOR`.

**Paper equation:** Text logic:

\[
usable_i = left_i \land right_i \land (score_i \ge 70) \quad \text{if model omits explicit usability.}
\]

Then rank usable anchors by model-provided ranking or by descending score.

**Implementation:**
- Qwen multi-anchor normalization: `src/upv_vlm_v2/anchor_selection/qwen32_multi_anchor_l6_batch_ranker.py:L132-L184`, function `_normalize_per_anchor`.
- Qwen ranking: `src/upv_vlm_v2/anchor_selection/qwen32_multi_anchor_l6_batch_ranker.py:L187-L196`, function `_rank_from_response`.
- Qwen selected anchor/action: `src/upv_vlm_v2/anchor_selection/qwen32_multi_anchor_l6_batch_ranker.py:L292-L368`, function `run_qwen32_multi_anchor_l6_batch_ranking`.
- Llama schema normalization: `src/upv_vlm_v2/anchor_selection/llama32_single_anchor_l6_taxonomy_scorer.py:L136-L183`, function `_normalize`.
- Llama selected anchor/action: `src/upv_vlm_v2/anchor_selection/llama32_single_anchor_l6_taxonomy_scorer.py:L359-L416`, function `run_llama32_single_anchor_l6_taxonomy_scoring`.

**Code excerpt:**

```python
if anchor_usable is None:
    anchor_usable = bool(left_usable and right_usable and (score or 0.0) >= 70.0)
# ...
selected_anchor_id = ranked[0] if ranked else None
final_action = "CHOOSE_BEST_ANCHOR" if selected_anchor_id else "NO_SAFE_ANCHOR"
```

**Manuscript recommendation:** Text plus a small decision-rule box. Keep prompt/schema details in appendix.

---

## M15. Selected Anchor Geometry Stored for Downstream Planning

**Pipeline role:** The selected A* candidate carries contact endpoints, center, local cross-axis direction, score, and artifact paths. This geometry drives path length and robot planning.

**Implementation:**
- Candidate records created: `src/upv_vlm_v2/anchor_selection/anchor_selection_stage.py:L495-L538`.
- Stage result maps payload to `AnchorSelectionResult`: `src/upv_vlm_v2/pipeline/stages.py:L271-L296`.
- CLI prints selected anchor, contact points, and score: `src/upv_vlm_v2/cli/run_full_main_upv_vlm_v2_pipeline.py:L104-L126`.

**Key variables:**
- `anchor_px`: candidate center / selected path center.
- `contact_point_a_px`, `contact_point_b_px`: opposing contact endpoints.
- `local_cross_axis_vector`: measurement/contact path direction.
- `final_anchor_id`: selected A* identifier.

**Manuscript recommendation:** Include in methods as the bridge between perception and path-length/contact planning.

---

## M16. RGB-D Back-Projection to Camera Coordinates

**Pipeline role:** Convert pixel coordinate and depth to camera-frame XYZ using pinhole intrinsics.

**Paper equation:**

\[
X=\frac{(u-c_x)Z}{f_x}, \quad Y=\frac{(v-c_y)Z}{f_y}, \quad Z=Z.
\]

**Implementation:**
- General transform utility: `src/upv_vlm_v2/planning/transforms.py:L175-L192`, function `camera_pixel_depth_to_xyz_m`.
- Vectorized local point-cloud function: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L110-L121`, function `backproject_pixels_to_xyz`.
- Final offline evaluator vectorized function: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L372-L377`, function `_backproject`.

**Code excerpt:**

```python
x = (uv[:, 0] - intr["cx"]) * z / intr["fx"]
y = (uv[:, 1] - intr["cy"]) * z / intr["fy"]
return np.column_stack([x, y, z])
```

**Manuscript recommendation:** Main displayed equation. This is central to 3D path length and robot contact planning.

---

## M17. Image Direction Lifted Into a 3D Measurement Direction

**Pipeline role:** Convert the selected 2D A* path direction into a camera-frame unit vector at local reference depth.

**Paper equation:** Back-project two neighboring pixels at the same reference depth:

\[
\hat{d}_{3D}=\frac{\pi^{-1}(p_0+d,Z_0)-\pi^{-1}(p_0,Z_0)}{\|\pi^{-1}(p_0+d,Z_0)-\pi^{-1}(p_0,Z_0)\|}.
\]

**Implementation:** `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L443-L448`, function `_direction3`.

**Code excerpt:**

```python
uv = np.vstack([p0, p0 + direction])
xyz = _backproject(uv, np.array([ref_depth_mm, ref_depth_mm], dtype=float), intr)
d3 = _normalize(xyz[1] - xyz[0])
mm_per_px = 1000.0 * float(np.linalg.norm(xyz[1] - xyz[0]))
```

**Manuscript recommendation:** Main methods equation for depth point-cloud projection.

---

## M18. Rigid Transform Composition for Robot Planning

**Pipeline role:** Compose current robot TCP pose with calibrated camera-to-tool transform and transform selected anchor coordinates from camera to robot base or tool frame.

**Paper equation:**

\[
T_{base,camera}=T_{base,tool0}T_{tool0,camera},\quad p_{base}=T_{base,camera}\,p_{camera}.
\]

For pose vectors, UR rotation vectors are converted using Rodrigues' formula.

**Implementation:**
- 4x4 transform multiply: `src/upv_vlm_v2/planning/transforms.py:L75-L80`, functions `_matmul4`, `matmul4`.
- Apply transform: `src/upv_vlm_v2/planning/transforms.py:L64-L72`, function `apply_transform`.
- Rotation vector to matrix: `src/upv_vlm_v2/planning/transforms.py:L83-L100`, functions `_rotvec_to_matrix`, `rotvec_to_matrix3`.
- Matrix to rotation vector: `src/upv_vlm_v2/planning/transforms.py:L103-L121`, functions `_matrix_to_rotvec`, `matrix3_to_rotvec`.
- Camera-to-tool planning: `src/upv_vlm_v2/planning/robot_pose_planner.py:L153-L163`, function `plan_robot_poses`.
- Base-frame real execution rebuild: `src/upv_vlm_v2/planning/base_frame_planner.py:L166-L184`, function `build_real_robot_motion_plan_from_current_tcp`.

**Code excerpt:**

```python
t_base_tool0_current = pose_vector_to_matrix_ur(current_tcp_pose)
t_base_camera = matmul4(t_base_tool0_current, t_tool0_camera)
anchor_base_xyz = apply_transform(t_base_camera, anchor_camera_xyz)
# ...
final_xyz = [anchor_base_xyz[0] - rotated_surface[0], anchor_base_xyz[1] - rotated_surface[1], anchor_base_xyz[2] - rotated_surface[2]]
```

**Manuscript recommendation:** Main methods equation for robot planning. Details of UR rotvec conversion can go to appendix.

---

## M19. Robot Orientation/Yaw Alignment From Selected Image Axis

**Pipeline role:** Choose a yaw correction so the TCP x-axis aligns with the selected image/object axis after a configured offset and candidate wrap-around options.

**Paper equation:**

\[
\psi_{cur}=\operatorname{atan2}(x_y,x_x),\quad \psi_{des}=\operatorname{atan2}(a_y,a_x)+\psi_{offset},
\]

then choose the candidate yaw delta with smallest absolute normalized angle.

**Implementation:** `src/upv_vlm_v2/planning/base_frame_planner.py:L75-L127`, function `plan_t4_style_orientation`.

**Code excerpt:**

```python
current_yaw = math.degrees(math.atan2(current_x[1], current_x[0]))
base_desired = math.degrees(math.atan2(selected_axis_base[1], selected_axis_base[0]))
desired_after_extra = _normalize_deg(base_desired + float(orient.get("extra_yaw_deg", 90.0)))
deltas = [_normalize_deg(candidate - current_yaw) for candidate in candidates]
selected_idx = min(range(len(deltas)), key=lambda idx: abs(deltas[idx]))
```

**Manuscript recommendation:** Text or appendix unless robot orientation is a headline result.

---

## M20. Mask-Based Selected-Anchor Path Length

**Pipeline role:** In final selected-anchor evaluation, the mask-based path length uses selected A* endpoints from `selected_anchors.json` or generated A* candidate endpoints. It does not infer major/minor axes from MinAreaRect.

**Paper equation:** For selected endpoints `p_1,p_2`, reference depth `Z_0`, and local pixel scale `alpha`,

\[
L_{mask}=\|p_2-p_1\|_2\,\alpha(d,Z_0).
\]

**Implementation:**
- Load selected anchor metadata: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L651-L721`, function `_load_manual_anchor_selection`.
- Endpoint length conversion: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L477-L492`, function `_endpoint_length`.
- Selected-path use: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L1318-L1354`, function `run`.
- Selected rows store endpoint coordinates and mask result: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L1450-L1489`.

**Code excerpt:**

```python
delta = p2 - p1
length_px = float(np.linalg.norm(delta))
p0 = 0.5 * (p1 + p2)
direction = delta / length_px
_, _, mm_per_px = _direction3(p0, direction, ref_depth_mm, intr)
"length_mm": length_px * mm_per_px
```

**Manuscript recommendation:** Main displayed equation; emphasize selected A* anchors are the source of truth.

---

## M21. Legacy/Exploratory Mask Chord Fallback

**Pipeline role:** If selected endpoints are absent, the script can intersect a line through `p0` and a direction with the mask. This is explicitly diagnostic unless manual anchor metadata is supplied.

**Paper equation:**

\[
L= (\max T-\min T)\alpha(d,Z_0),\quad T=\{t: M(round(p_0+td))=1\}.
\]

**Implementation:** `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L451-L474`, function `_mask_chord`; manual-metric guard at `L1241-L1252` and fallback invalid reason at `L646-L648`.

**Code excerpt:**

```python
for t in np.linspace(-max_extent, max_extent, 2 * max_extent + 1):
    p = p0 + direction * t
    if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
        ts.append(float(t))
t0, t1 = min(ts), max(ts)
"length_mm": abs(t1 - t0) * mm_per_px
```

**Manuscript recommendation:** Do not include as final method. Mention only to distinguish rejected exploratory analyses from selected-anchor V2.

---

## M22. Final V2 Depth Point-Cloud-Based Selected-Anchor Path Length With Percentile Endpoints

**Pipeline role:** Estimate selected A* path length from aligned depth by sampling local depth bands around the selected path, filtering depth support, back-projecting support pixels to camera-frame 3D points, projecting those points onto the selected 3D path direction, and using robust percentile endpoints. The internal CSV/code prefix is `strip_depth`, but paper prose should call this the **depth point-cloud-based path-length estimate**.

**Final selected-anchor inputs:** The path center `\mathbf{p}^{*}`, path direction `\mathbf{d}^{*}`, and manual target length come from `selected_anchors.json`, not from MinAreaRect fallback axes. The evaluator loads selected anchor metadata in `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L651-L660`, and the selected-anchor evaluation path writes rows with these selected values at `L1440-L1489`.

**Local support equation:** Let `\mathbf{q}_j=[u_j,v_j]^T` be an image pixel in the buffered object ROI, `z_j` its aligned depth in millimetres, `\mathbf{n}^{*}=(-d_y^{*},d_x^{*})` the image-plane normal to the selected path direction, `w_b` the local band half-width, and `Z_{\mathrm{ref}}` the median reference depth near `\mathbf{p}^{*}`. For one parallel band with offset `\delta_k`, the band center is `\mathbf{p}^{*}_k=\mathbf{p}^{*}+\delta_k\mathbf{n}^{*}` and the support set is

\[
\mathcal{P}^{*}_k
=
\left\{
\mathbf{x}^{c}_j
=
\pi^{-1}(\mathbf{q}_j,z_j)
:
\left|
(\mathbf{q}_j-\mathbf{p}^{*}_k)^{T}\mathbf{n}^{*}
\right|
\le
w_b,\ 
|z_j-Z_{\mathrm{ref}}|\le \Delta z,\ 
z_j \text{ valid}
\right\}.
\]

The implementation first clips candidate pixels to the 4% buffered object rectangle, then applies the local band and depth-support tests.

**Projection equation:** For each valid 3D support point, project onto the selected 3D path direction:

\[
\tau_j =
(\mathbf{x}_j^c-\mathbf{x}_0^c)^T\hat{\mathbf{d}}_{3D}^c,
\qquad \mathbf{x}_j^c\in\mathcal{P}^{*}_k.
\]

**Robust percentile endpoint equation:** With `q_l=0.0025` and `q_u=0.9975`, the per-band length is

\[
L_{\mathrm{pc},k}
=
1000
\left|
Q_{0.9975}(\{\tau_j\}_{j=1}^{N_k})
-
Q_{0.0025}(\{\tau_j\}_{j=1}^{N_k})
\right|.
\]

**Band aggregation equation:** The final selected-anchor depth point-cloud estimate is the median over valid local bands:

\[
L_{\mathrm{pc}}
=
\operatorname{median}_{k\in\mathcal{V}} L_{\mathrm{pc},k}.
\]

**Exact final implementation parameters:**
- ROI construction: minAreaRect object box expanded by `buffer_percent = 4.0`.
- Local band half-width: `strip_half_width_px = 6.0`, so each band is 12 px wide.
- Parallel band offsets: `strip_offsets_px = [-18, -12, -6, 0, 6, 12, 18]`, so the default uses seven local bands.
- Reference depth: median valid depth in a square centered at `p0` with `center_region_radius_px = 15.0`.
- Depth validity: `z_j > 0`.
- Depth support filter: `|z_j - Z_ref| <= depth_tolerance_mm`, with `depth_tolerance_mm = 8.0`.
- Minimum support: `min_supported_points = 100` points per local band.
- Endpoint mode CLI: `depth_endpoint_mode` choices are `current` and `percentile`; default is `current`, but the current implementation already uses the robust quantiles. The `percentile` mode records the V2 intent explicitly without changing the endpoint computation.
- Quantiles: `lower_quantile = 0.0025`, `upper_quantile = 0.9975`. Aliases `--depth-lower-quantile` and `--depth-upper-quantile` override the same values.
- Aggregation: each valid band produces one percentile span; final `length_mm` is `median(strip_lengths)`, with `mean_length_mm` also recorded for diagnostics.

**Implementation:**
- ROI expansion helper: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L425-L427`, function `_expand_box`.
- Reference depth near `p0`: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L430-L440`, function `_pick_ref_depth`.
- Pixel direction to 3D direction: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L443-L448`, function `_direction3`.
- Buffered ROI and parallel local bands: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L562-L581`, function `_strip_depth`.
- Valid-depth and depth-tolerance support filtering: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L586-L597`.
- Back-projection and projection onto selected 3D direction: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L599-L600`.
- Percentile endpoints and per-band span: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L601-L618`.
- Median aggregation over valid bands: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L623-L635`.
- CLI/default local band and quantile parameters: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L1617-L1640`, `L1663-L1669`.

**Code excerpt:**

```python
roi = np.zeros(depth_mm.shape, dtype=np.uint8)
cv2.fillConvexPoly(roi, np.round(_expand_box(box, args.buffer_percent)).astype(np.int32), 1)
ys, xs = np.nonzero(roi > 0)
roi_uv = np.column_stack([xs.astype(float), ys.astype(float)])
ref, ref_count = _pick_ref_depth(depth_mm, p0, args.center_region_radius_px)
p0_xyz, d3, _ = _direction3(p0, direction, ref, intr)
perp = np.array([-direction[1], direction[0]], dtype=float)
for offset in args.strip_offsets_px:
    line_p0 = p0 + perp * float(offset)
    rel = roi_uv - line_p0[None, :]
    r = np.abs(rel @ perp)
    in_strip = r <= args.strip_half_width_px
    uv = roi_uv[in_strip]
```

```python
supported = valid & (np.abs(depths - ref) <= args.depth_tolerance_mm)
xyz = _backproject(support_uv, support_depth, intr)
proj = (xyz - p0_xyz) @ d3
lo = float(np.quantile(proj, args.lower_quantile))
hi = float(np.quantile(proj, args.upper_quantile))
length_mm = 1000.0 * abs(hi - lo)
"length_mm": float(np.median(strip_lengths))
```

**Manuscript recommendation:** Main displayed equations. State that final V2 uses selected A* anchors, local depth sampling bands, and percentile projected endpoints. State explicitly that whole-ROI depth and MinAreaRect fallback are not used for final metrics.

---

## M23. Older E2 True 3D Local Point-Cloud Edge-Bin Width

**Pipeline role:** Earlier main-pipeline path-length refinement used local band support, z-filtering, 3D projection, and edge-bin endpoint distance. This is related but not identical to final selected-anchor V2 offline percentile-endpoint evaluation.

**Paper equation:** For edge bins near selected endpoint projections, estimate endpoint depths `Z_a`, `Z_b`, back-project endpoint pixels at median edge-bin depths, and compute endpoint distance:

\[
L_{edge}=1000\,\|\pi^{-1}(p_b,Z_b)-\pi^{-1}(p_a,Z_a)\|_2.
\]

**Implementation:**
- Local band around selected chord: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L280-L295`, function `compute_true_3d_local_pointcloud_width`.
- Depth validity and z filter: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L297-L320`.
- Back-project and project: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L331-L345`.
- Min/max and 2/98 percentile diagnostics: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L361-L367`.
- Edge bins and endpoint distance: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L368-L408`.
- Validity policy: `src/upv_vlm_v2/geometry/depth_local_pointcloud.py:L29-L83`, function `finalize_edge_bin_depth_validity`.

**Code excerpt:**

```python
z_a = float(np.median(xyz[mask_a, 2]))
z_b = float(np.median(xyz[mask_b, 2]))
edge_a_xyz = _point_on_ray_from_pixel(endpoint_a, z_a, intrinsics)
edge_b_xyz = _point_on_ray_from_pixel(endpoint_b, z_b, intrinsics)
span_edge = float(np.linalg.norm(edge_b_xyz - edge_a_xyz) * 1000.0)
```

**Manuscript recommendation:** Appendix or related-method note. Do not describe it as the final V2 selected-anchor percentile-endpoint method.

---

## M24. Whole-ROI Depth Method Is Diagnostic Only

**Pipeline role:** The evaluator still contains a whole-ROI depth method for diagnostics and older comparisons, but it is not used in the final selected-anchor V1/V2 main result.

**Paper equation:** It mirrors the percentile projection equation over all supported ROI points rather than local selected-path strips.

**Implementation:** `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L525-L559`, function `_whole_roi_depth`; main selected report states whole-ROI is not used at `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L1127-L1133` and selected rows note it at `L1487-L1489`.

**Code excerpt:**

```python
supported = valid & (np.abs(depths - ref) <= args.depth_tolerance_mm)
xyz = _backproject(support_uv, support_depth, intr)
proj = (xyz - p0_xyz) @ d3
lo = float(np.quantile(proj, args.lower_quantile))
hi = float(np.quantile(proj, args.upper_quantile))
"length_mm": 1000.0 * abs(hi - lo)
```

**Manuscript recommendation:** Exclude from final method section or mention only as an exploratory diagnostic not used for final selected-anchor metrics.

---

## M25. Clamp Opening and Actuator Step Conversion

**Pipeline role:** Compute a safe clamp opening from selected path length, then convert spacing/travel to symmetric actuator steps for Arduino clamp commands. This is not the UPV velocity path length; it is mechanical spacing.

**Paper equation:** Clamp opening:

\[
O=L_{path}+m_{safety}+m_{contact}.
\]

Move-to-spacing symmetric closing:

\[
\Delta_{close}=O_{fully\ open}-O_{target},\quad \Delta_{side}=\frac{\Delta_{close}}{2},\quad steps=\Delta_{side}\cdot steps/mm.
\]

**Implementation:**
- Clamp opening: `src/upv_vlm_v2/planning/clamp_plan.py:L8-L35`, function `plan_clamp_opening`.
- Arduino simulated/real step conversion: `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L145-L161`, function `move_to_spacing_mm`.
- Direct travel conversion: `src/upv_vlm_v2/hardware/arduino_clamp_client.py:L163-L177`, function `clamp_travel_mm`.
- Robot plan includes clamp opening: `src/upv_vlm_v2/planning/robot_pose_planner.py:L102-L113`, `L210-L214`.

**Code excerpt:**

```python
opening = float(local_path_length_mm) + safety_margin + contact_allowance
# ...
total_closing = self.fully_open_probe_spacing_mm - float(spacing_mm)
one_side = total_closing / 2.0
steps = one_side * self.steps_per_mm
```

**Manuscript recommendation:** Text or appendix. Important to avoid confusing clamp spacing with UPV propagation path length.

---

## M26. Robot Pose Safety Checks and Motion Deltas

**Pipeline role:** Validate planned poses against workspace bounds, max XY step, and max vertical drop. In execution summaries, compute position/orientation error from planned vs actual pose.

**Paper equation:** Workspace membership checks are axis-aligned bounds. Distance metrics:

\[
e_p=1000\sqrt{(x_a-x_p)^2+(y_a-y_p)^2+(z_a-z_p)^2},
\]

\[
e_R=\frac{180}{\pi}\sqrt{(r_{x,a}-r_{x,p})^2+(r_{y,a}-r_{y,p})^2+(r_{z,a}-r_{z,p})^2}.
\]

**Implementation:**
- Workspace bounds: `src/upv_vlm_v2/planning/robot_pose_planner.py:L42-L84`, functions `_pose_within_workspace`, `_workspace_check`.
- Base-frame XY step and Z drop checks: `src/upv_vlm_v2/planning/base_frame_planner.py:L202-L219`, function `build_real_robot_motion_plan_from_current_tcp`.
- Pose error in main pipeline: `src/upv_vlm_v2/pipeline/main_pipeline.py:L41-L44`, function `_pose_distance`.
- E45 pose error analysis: `src/upv_vlm_v2/experiments/analyze_e45_robot_upv_repeatability.py:L130-L145`, function `_compute_pose_error`.

**Code excerpt:**

```python
pos = math.sqrt(sum((float(a) - float(p)) ** 2 for p, a in zip(planned, actual, strict=False))) * 1000.0
ori = math.sqrt(sum((float(a) - float(p)) ** 2 for p, a in zip(planned_r, actual_r, strict=False))) * 180.0 / math.pi
```

**Manuscript recommendation:** Include robot-pose error equations if reporting placement repeatability; otherwise move to appendix.

---

## M27. UPV Signal Repeatability Metrics

**Pipeline role:** For E45/E5-style summaries, compute valid signal rate, failed reading rate, mean/std velocity and arrival time, and coefficient of variation for repeated UPV readings. The audit found placeholders and analysis code, but no live UPV waveform extraction code in the main pipeline path inspected here.

**Paper equation:**

\[
\text{valid rate}=\frac{n_{valid}}{n_{attempted}},\quad \text{failed rate}=\frac{n_{failed}}{n_{attempted}},
\]

\[
CV_v=100\frac{\sigma_v}{\bar{v}},\quad CV_t=100\frac{\sigma_t}{\bar{t}}.
\]

If time of flight `t` and path length `L` are integrated later, UPV velocity should be

\[
v=\frac{L}{t},
\]

with units handled consistently, but the inspected repo mainly expects `upv_velocity_m_s` and `arrival_time_us` fields rather than extracting ToF from raw waveforms.

**Implementation:**
- Valid/failed signal predicates: `src/upv_vlm_v2/experiments/analyze_e45_robot_upv_repeatability.py:L235-L245`, functions `_is_valid_upv`, `_is_failed_upv`.
- UPV summary metrics: `src/upv_vlm_v2/experiments/analyze_e45_robot_upv_repeatability.py:L247-L323`, function `analyze_upv`.
- External UPV placeholder packaging: `scripts/create_paper_E1_to_E5_curated_assets.py:L313-L333`.

**Code excerpt:**

```python
valid_count = sum(1 for r in attempted_group if _is_valid_upv(r))
failed_count = sum(1 for r in attempted_group if _is_failed_upv(r))
cv_v = round(float(v_std) / float(v_mean) * 100.0, 6)
cv_a = round(float(a_std) / float(a_mean) * 100.0, 6)
```

**Manuscript recommendation:** Keep planned/placeholder status clear unless external Pundit waveform/ToF data are integrated. Do not imply waveform ToF extraction exists in the inspected main pipeline.

---

## M28. Classification Metrics for Anchor Usability Benchmarks

**Pipeline role:** Compute TP/FP/TN/FN, precision, recall, specificity, F1, accuracy, balanced accuracy, and selected-anchor usability rate from saved manual labels and model predictions.

**Paper equation:**

\[
precision=\frac{TP}{TP+FP},\quad recall=\frac{TP}{TP+FN},\quad specificity=\frac{TN}{TN+FP},
\]

\[
F1=\frac{2PR}{P+R},\quad accuracy=\frac{TP+TN}{N},\quad balanced=\frac{recall+specificity}{2}.
\]

**Implementation:** `scripts/audit_anchor_selection_model_prompt_results.py:L110-L162`, function body computing metrics from anchor rows.

**Code excerpt:**

```python
precision = safe_float(tp, tp + fp)
recall = safe_float(tp, tp + fn)
specificity = safe_float(tn, tn + fp)
f1 = "" if precision == "" or recall == "" or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
accuracy = safe_float(tp + tn, known)
balanced = "" if recall == "" or specificity == "" else (recall + specificity) / 2
```

**Manuscript recommendation:** Main results equations if reporting anchor classification. Use standard definitions and cite that unknown/non-evaluable rows are excluded.

---

## M29. Path-Length Error Metrics

**Pipeline role:** Compare estimated mask/depth path lengths against manual selected-anchor path labels and summarize by path label/material/all.

**Paper equation:** For estimate `\hat{L}_i` and manual target `L_i`:

\[
e_i=\hat{L}_i-L_i,\quad |e_i|=|\hat{L}_i-L_i|,\quad \%e_i=100\frac{|e_i|}{L_i}.
\]

Summary metrics:

\[
MAE=\frac{1}{N}\sum_i |e_i|,\quad RMSE=\sqrt{\frac{1}{N}\sum_i e_i^2},\quad MAPE=\frac{1}{N}\sum_i \%e_i.
\]

**Implementation:**
- Per-row errors: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L638-L643`, function `_err`.
- Older general summary: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L799-L841`, function `_summary`.
- Final selected-anchor metric block: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L961-L972`, function `_metric_block`.
- Grouped selected summaries and improvement: `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L975-L1007`, function `_selected_summary_rows`.

**Code excerpt:**

```python
signed = float(pred - manual)
abs_err = abs(signed)
return abs_err, 100.0 * abs_err / manual, signed
# ...
f"{prefix}_MAE_mm": float(np.mean(vals)) if vals else None,
f"{prefix}_RMSE_mm": float(np.sqrt(np.mean(np.square(vals)))) if vals else None,
f"{prefix}_MAPE_percent": float(np.mean(pct)) if pct else None,
```

**Manuscript recommendation:** Main results equations. State that final metrics use selected A* anchor metadata and exclude MinAreaRect fallback.

---

## M30. Method Improvement Metric

**Pipeline role:** Report how much the depth point-cloud estimate improves over the mask estimate by MAE.

**Paper equation:**

\[
\Delta MAE = MAE_{mask}-MAE_{depth},\quad improvement(\%)=100\frac{MAE_{mask}-MAE_{depth}}{MAE_{mask}}.
\]

**Implementation:** `scripts/evaluate_new_capture_mask_and_depth_path_length.py:L996-L1005`, function `_selected_summary_rows`.

**Code excerpt:**

```python
row["improvement_mm"] = float(mask_mae) - float(strip_mae)
row["improvement_percent"] = 100.0 * (float(mask_mae) - float(strip_mae)) / float(mask_mae) if float(mask_mae) else None
row["best_method_by_MAE"] = "strip_depth" if float(strip_mae) < float(mask_mae) else "mask"
```

**Manuscript recommendation:** Include in results table, with CSV-column translation: `strip_depth` means depth point-cloud-based selected-anchor estimate.

---

## Final Method Boundaries and Terminology

## Revised manuscript Section 2.3 equation plan

### 2.3 3D robot-coordinate and path-length estimation

This section should begin **after** Section 2.2 has introduced anchor generation and VLM contact-anchor selection. Section 2.3 should assume that the selected anchor `A*` is already known in pixel coordinates. The VLM selects an anchor ID/position from candidate A* contact paths; it does not independently select exact endpoint pixels for path-length measurement.

### 2.3.1 3D robot-coordinate estimation for the selected anchor

**Purpose:** Compute the 3D robot coordinates needed for the UR3e to perform UPV at the selected object and selected anchor.

- **Eq. 1: RGB-D back-projection.** Use camera intrinsics to convert a selected pixel and aligned depth into a camera-frame 3D point:

\[
\mathbf{x}^{c}=\pi^{-1}(\mathbf{q},z).
\]

- **Eq. 2: Camera-to-tool / robot transformation.** Use the calibrated transform chain to map camera-frame coordinates into robot/tool coordinates:

\[
\mathbf{x}^{r}=\mathbf{T}^{r}_{c}\mathbf{x}^{c}
\quad\text{or}\quad
\mathbf{x}^{tool}=\mathbf{T}^{tool}_{c}\mathbf{x}^{c},
\]

depending on which frame is used in the specific manuscript subsection.

### 2.3.2 Selected-anchor path-length measurement

Path-length measurement should be described as a selected-anchor measurement problem, not as a generic object-axis problem. The known inputs are the selected anchor center `\mathbf{p}^{*}`, selected path direction `\mathbf{d}^{*}`, and selected path label/manual target from `selected_anchors.json`.

### 2.3.2.1 Mask-based endpoint method

**Purpose:** Find mask-defined local chord endpoints for path-length measurement. These are not the VLM contact-pair endpoint estimates; they are mask-defined local chord endpoints used to measure the path length associated with the selected anchor.

- **Eq. 3: Mask-defined local chord endpoint extraction.** Intersect the selected object mask with the line through selected anchor center `\mathbf{p}^{*}` in direction `\mathbf{d}^{*}` and take the minimum and maximum valid signed positions:

\[
t^{-},t^{+}=\min/\max\{t:\mathbf{p}^{*}+t\mathbf{d}^{*}\in\mathcal{M}\}.
\]

- **Eq. 4: Mask-based path length.** Convert endpoint distance to metric length using the local RGB-D scale/back-projection:

\[
L_{\mathrm{mask}}
=
1000\left\|
\pi^{-1}(\mathbf{p}^{*}+t^{+}\mathbf{d}^{*}, Z_{\mathrm{ref}})
-
\pi^{-1}(\mathbf{p}^{*}+t^{-}\mathbf{d}^{*}, Z_{\mathrm{ref}})
\right\|_2.
\]

### 2.3.2.2 Depth point-cloud-based endpoint method

**Purpose:** Collect valid depth points in a local sampling band around the selected anchor path and estimate path length from robust projected 3D endpoints.

- **Eq. 5: Local depth-support set / sampling band.** Use the M22 support-set equation over the selected path and local parallel bands.
- **Eq. 6: Projection of valid 3D depth points onto selected path direction.**

\[
\tau_j =
(\mathbf{x}_j^c-\mathbf{x}_0^c)^T\hat{\mathbf{d}}_{3D}^c.
\]

- **Eq. 7: Depth point-cloud percentile endpoint path length.**

\[
L_{\mathrm{pc},k}
=
1000
\left|
Q_{0.9975}(\{\tau_j\})
-
Q_{0.0025}(\{\tau_j\})
\right|.
\]

- **Optional Eq. 8: Median aggregation over valid local bands.** Include this equation because the final implementation uses seven parallel bands and aggregates valid band estimates by median:

\[
L_{\mathrm{pc}}
=
\operatorname{median}_{k\in\mathcal{V}} L_{\mathrm{pc},k}.
\]

**Boundary statement for manuscript:** Whole-ROI depth and MinAreaRect fallback axes are not the final selected-anchor method. They may be mentioned only as exploratory diagnostics. The final reported method uses selected A* anchors, mask-defined local chord endpoints, and local depth point-cloud percentile endpoints.

For paper-ready path-length reporting:

- **Use:** selected A* anchors from `selected_anchors.json`, mask-based endpoint length, and depth point-cloud-based local band projection with robust percentile endpoints.
- **Use:** `manual_mm` stored per selected path label in `selected_anchors.json` as the manual target.
- **Do not use as final:** MinAreaRect major/minor fallback axes, global label swapping, nearest-value pairing, or whole-ROI depth.
- **Terminology:** In prose, call the final depth method “depth point-cloud-based path-length estimation.” The CSV column prefix `strip_depth` corresponds to this local-band point-cloud method.

## Manuscript Equation Priority

Recommended displayed equations in the main manuscript:

1. Pixel/depth back-projection (`M16`).
2. Pixel-to-mm local scale (`M7`) or equivalent derived from back-projection.
3. A* candidate center spacing (`M9`) and candidate count (`M10`) if anchor generation is described.
4. Cross-section opposing contact endpoints (`M11`).
5. Mask-based selected-anchor path length (`M20`).
6. Depth point-cloud projection and percentile endpoint length (`M22`).
7. Classification and path-length evaluation metrics (`M28`, `M29`, `M30`).

Recommended appendix/text-only items:

- CLIP/material aggregation and guard heuristics (`M3`-`M5`).
- Edge/contact heuristic score (`M12`).
- VLM schema/ranking details (`M14`).
- Dominant rectangle refinement (`M8`).
- Clamp and robot workspace checks (`M25`, `M26`) unless robot placement is central in the manuscript section.
- UPV repeatability placeholders (`M27`) until external waveform/ToF data are integrated.

## Safety/Execution Note

This audit did not run RealSense, robot motion, RTDE, GSAM2, Qwen/VLM, or final evaluation. It only inspected source files and created this markdown report.
