# Method and Metric Definitions

## Contact Classification

Good/proper contact is the positive class. For every saved pair, compare its
manual label with the saved predicted usability flag. Report TP, FP, TN, FN,
precision TP/(TP+FP), recall TP/(TP+FN), specificity TN/(TN+FP), F1
2TP/(2TP+FP+FN), accuracy and balanced accuracy. Undefined denominators remain null.
Classification never uses the manual label to select an anchor or modify a score.

Selection policies are explicit and separate from classification:

- VLM: highest score among anchors whose saved `anchor_usable` is true.
- RGB+mask: lowest risk among passing anchors.
- RGB-D: highest safety score among passing anchors; central anchor breaks ties.
- VLM/RGB equal scores: ascending numeric anchor ID breaks ties.
- No eligible anchor: abstain. Abstentions remain in the scene denominator.

`selected_good_rate` is selected manually-good pairs divided by all scenes, not
conditional accuracy given selection. Top-score tie IDs are emitted. This is a
Python postprocessing policy; the original VLM final decision may differ. Do not
describe the resulting selections as the original model's own final actions.

Contact73 has 51 good/22 bad pairs (45 original E3 + 28 new10, 16 scenes).
Contact82 has 52 good/30 bad pairs (50 E3 + 32 E45, 18 scenes).
Exclusions, original labels, original response texts and request prompts are
preserved in `data/labels`, `results/contact` and the contact archive.

## Classical Baselines

RGB+mask uses the same L6 rendered contact crops and saved GSAM2 masks. It maps
each mask to the saved crop, excludes the drawn magenta guide, fits the local
mask rim with a robust Huber line, detects mask discontinuities, and uses
CLAHE/Gaussian/Canny plus connected components to detect crossing edge evidence.
Both contacts must pass. The reference implementation retains defect_fraction
0.03 and max_defect_run_fraction 0.10; the old 20% guide-distance veto is absent.
Threshold development is not an independent held-out evaluation or a SOTA claim.

RGB-D uses saved RGB, depth, intrinsics, masks and local contact geometry with
top-view projected support and a 3 mm defect threshold. It has extra sensory
information; do not describe it as an RGB-only comparison. Reference code and
saved per-pair features/configuration are supplied. Its saved 73-pair result
reuses original E3 results and separately evaluates new10; a single reseeded full
run need not be bit-identical because the reference RANSAC seed depends on group
ordering. No depth supports the RGB-only contact classifier.

## Perception

The final 63 trials use saved GSAM2+CLIP decisions, score threshold 0.10,
without a margin veto or manual color/texture veto. Label corrections and prompt
text are preserved in the perception archive. Original27: 27 correct; revision36:
30 correct. No-selection is a failure for these present-material queries.
This release's default reproduction validates the final decision/label table;
it does not infer alternative mask labels or redo annotation.

## Path Length

The manual selected-anchor JSON is the source of truth for each path. No swapping
of major/minor manual labels, no minAreaRect fallback and no whole-ROI replacement
are introduced. The final method uses local depth support and projected endpoint
quantiles. Exact saved run parameters are in each version's run configuration.
See the source mathematics audit and reference evaluator for multiple offset-band
support and median aggregation. The reproduction command calculates errors from
the saved estimates, not from raw depth; raw RGB-D and selected metadata are also
archived for further work. V1 and V2 are preserved even if their numbers coincide.

Signed error = estimate - manual; absolute error = its magnitude; percentage
error = 100*absolute/manual. MAE, RMSE, MAPE, median/max absolute and signed mean
are calculated by path label, material, and all. Invalid depth rows are excluded
from depth metrics and counted, never assigned a zero error.

## Runtime and UPV Limits

Runtime tables contain imported/copied readings. The release emits both the
recorded-row summary (94 successful rows, mean total about 98.88 s) and a
best-effort provenance identity deduplication. Neither proves 94 independent
robot executions. Anchor-selection stage time is not isolated Qwen32 latency;
live backend/model residency and offline Qwen experiments must remain separate.
Persistent-server loading and separately exported UPV acquisition are not part
of the reported per-run pipeline timer. Conservative robot movement was used.
The source audits explain the timing boundaries and unavailable motion splits.

UPV master-table measurements are included, but raw Windows waveform exports
are pending. Do not claim reproduction of ToF extraction/RMS from raw waveforms
from this package alone.
