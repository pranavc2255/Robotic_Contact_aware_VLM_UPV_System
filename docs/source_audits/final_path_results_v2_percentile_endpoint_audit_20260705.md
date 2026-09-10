# Final Path Results V2 Percentile Endpoint Audit

Generated: 2026-07-05

## Scope

This audit checks the selected-anchor depth point-cloud path-length estimator in
`scripts/evaluate_new_capture_mask_and_depth_path_length.py` before creating the
separate V2 result set.

No GSAM2, manual annotation, RealSense capture, robot motion, RTDE command,
Qwen call, or VLM call is part of this audit.

## Current V1 Endpoint Logic

The selected-anchor evaluator uses the manually selected A* path metadata from
`selected_anchors.json`:

- `p0_xy`
- `direction_xy`
- `endpoint1_xy`
- `endpoint2_xy`
- `manual_mm`

For the depth point-cloud estimate, the evaluator:

1. Builds a buffered ROI from the selected mask min-area rectangle.
2. Uses the selected A* path center and direction.
3. Samples multiple local bands/strips around the selected path.
4. Keeps valid depth pixels whose depth is within `depth_tolerance_mm` of the
   local reference depth near `p0`.
5. Back-projects supported pixels using aligned-depth camera intrinsics.
6. Projects supported 3D points onto the selected path direction.
7. Computes path length from projected endpoints.

The current V1 code already uses robust quantile projected endpoints:

- lower quantile: `0.0025`
- upper quantile: `0.9975`

Relevant functions:

- `_strip_depth(...)`
- `_direction3(...)`
- `_backproject(...)`

## V2 Endpoint Logic

V2 adds explicit endpoint-mode metadata and CLI flags:

- `--depth-endpoint-mode percentile`
- `--depth-lower-quantile 0.0025`
- `--depth-upper-quantile 0.9975`

The projected endpoint math is the same quantile endpoint calculation already
used by V1:

```python
lo = np.quantile(proj, lower_quantile)
hi = np.quantile(proj, upper_quantile)
length_mm = 1000.0 * abs(hi - lo)
```

## What Changed

- V2 output records the depth endpoint mode and quantiles in `run_config.json`.
- V2 selected-path and all-anchor CSVs include explicit endpoint metadata fields:
  - `depth_endpoint_mode`
  - `depth_lower_quantile`
  - `depth_upper_quantile`
  - `depth_support_projected_lower`
  - `depth_support_projected_upper`
  - `depth_support_projected_span_mm`
- V2 paper packaging records the endpoint mode and quantiles.
- V2 step-by-step figures label the method as a depth point-cloud estimate with
  percentile endpoints.

## What Did Not Change

- The source V1 capture session was not modified.
- GSAM2 masks were not regenerated.
- Manual anchor selections were not regenerated.
- Whole-ROI depth is not used in the main selected-anchor result.
- MinAreaRect fallback is not used.
- Major/minor labels are not swapped.
- Selected A* anchors remain the source of truth.

## Expected Interpretation

Because V1 already used percentile projected endpoints at `0.0025/0.9975`, V2
is expected to match V1 numerically unless some other code path or settings differ.
The V2 value is still useful as a clean, explicitly labeled percentile-endpoint
result set.
