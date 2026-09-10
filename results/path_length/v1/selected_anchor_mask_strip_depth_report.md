# Selected-Anchor Mask vs Strip-Depth Path-Length Report

Dataset: `workspace/outputs/final_path_results_v1_capture_session_20260704_222852`
Mask run: `workspace/outputs/final_path_results_v1_capture_session_20260704_222852/gsam2_masks`
Manual anchor selection root: `workspace/outputs/final_path_results_v1_capture_session_20260704_222852/manual_anchor_selection`

## Source of Truth
- Selected anchor files found: 15
- Selected manual paths evaluated: 30
- Evaluation used `selected_anchors.json` A* metadata for p0, direction, endpoints, and manual target.
- No major/minor swapping was performed.
- No minAreaRect fallback axis comparison was used for these selected-path metrics.
- Whole-ROI depth was not used in the main comparison.
- These metrics supersede the previous exploratory minAreaRect/cross-axis results.

## Selected Anchor IDs

| case_id | major selected ID | minor selected ID | manual major mm | manual minor mm |
| --- | --- | --- | --- | --- |
| case_001_1 | A2 | A5 | 38.000 | 156.500 |
| case_002_2 | A2 | A6 | 37.500 | 75.100 |
| case_003_3 | A3 | A7 | 37.500 | 271.500 |
| case_004_4 | A3 | A5 | 37.500 | 229.000 |
| case_005_5 | A1 | A8 | 78.800 | 264.000 |
| case_006_1 | A2 | A5 | 88.000 | 180.000 |
| case_007_2 | A2 | A7 | 59.200 | 208.000 |
| case_008_3 | A3 | A6 | 58.500 | 210.000 |
| case_009_4 | A1 | A7 | 98.900 | 205.000 |
| case_010_5 | A3 | A6 | 65.300 | 215.000 |
| case_011_1 | A2 | A5 | 101.900 | 169.000 |
| case_012_2 | A1 | A7 | 77.000 | 279.500 |
| case_013_3 | A2 | A5 | 76.200 | 118.700 |
| case_014_4 | A3 | A7 | 76.500 | 285.000 |
| case_015_5 | A1 | A5 | 55.000 | 156.500 |

## All A* Anchor Measurement Summary
- Total A* anchors measured: 102
- Anchors with valid mask path length: 102
- Anchors with valid strip-depth path length: 100
- Anchors with invalid strip-depth path length: 2
- Top all-anchor strip-depth failure reasons: `{'missing_reference_depth': 2}`

## Selected 30-Path Summary

| group | n_total | mask_n_valid | strip_depth_n_valid | strip_depth_n_invalid | mask_MAE_mm | mask_RMSE_mm | mask_MAPE_percent | mask_signed_mean_error_mm | mask_median_abs_error_mm | mask_max_abs_error_mm | strip_depth_MAE_mm | strip_depth_RMSE_mm | strip_depth_MAPE_percent | strip_depth_signed_mean_error_mm | strip_depth_median_abs_error_mm | strip_depth_max_abs_error_mm | strip_depth_valid_rate | improvement_mm | improvement_percent | best_method_by_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| major | 15.000 | 15.000 | 15.000 | 0.000 | 3.025 | 5.069 | 5.508 | 1.810 | 0.652 | 12.959 | 0.969 | 1.256 | 1.710 | 0.137 | 0.616 | 2.706 | 1.000 | 2.055 | 67.949 | strip_depth |
| minor | 15.000 | 15.000 | 15.000 | 0.000 | 3.470 | 7.745 | 1.543 | 1.731 | 1.309 | 29.050 | 1.633 | 1.932 | 0.920 | 0.275 | 1.244 | 3.509 | 1.000 | 1.838 | 52.950 | strip_depth |
| all | 30.000 | 30.000 | 30.000 | 0.000 | 3.248 | 6.545 | 3.525 | 1.771 | 1.239 | 29.050 | 1.301 | 1.630 | 1.315 | 0.206 | 1.096 | 3.509 | 1.000 | 1.946 | 59.935 | strip_depth |
| timber_all | 10.000 | 10.000 | 10.000 | 0.000 | 5.739 | 10.604 | 6.284 | 5.486 | 0.873 | 29.050 | 1.288 | 1.528 | 1.736 | 0.748 | 1.096 | 3.108 | 1.000 | 4.451 | 77.557 | strip_depth |
| brick_all | 10.000 | 10.000 | 10.000 | 0.000 | 2.693 | 3.578 | 2.868 | -0.685 | 2.320 | 9.140 | 1.375 | 1.778 | 0.968 | -1.215 | 1.073 | 3.509 | 1.000 | 1.318 | 48.929 | strip_depth |
| concrete_block_all | 10.000 | 10.000 | 10.000 | 0.000 | 1.311 | 1.813 | 1.425 | 0.511 | 0.853 | 3.990 | 1.240 | 1.572 | 1.241 | 1.084 | 0.870 | 3.061 | 1.000 | 0.071 | 5.395 | strip_depth |

## Was Strip-Depth Path Measurement Possible?
Selected paths valid: 30/30.
Answer: yes, for all selected paths.

## Which Method Is More Accurate?
Across all selected paths with valid strip-depth estimates, the lower MAE method is: **strip-depth**.

## Invalid Selected Strip-Depth Rows
None.

Selected-path strip-depth failure reason counts: `{}`

## Output Files
- `all_anchor_path_measurements.csv`
- `selected_anchor_mask_strip_depth_vs_manual.csv`
- `selected_anchor_mask_strip_depth_summary.csv`
- Diagnostic figures: `workspace/outputs/final_path_results_v1_mask_depth_eval/session_20260704_230654/figures`