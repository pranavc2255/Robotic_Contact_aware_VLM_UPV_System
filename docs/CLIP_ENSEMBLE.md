# CLIP Material Prompt Ensemble

## Main Configuration

The defaults in `configs/pipeline.json` and the complete-cycle deployment YAML
use `mean_embedding_class_softmax` and `clip_score_threshold: 0.35` with
`openai/clip-vit-base-patch32`. The verifier dispatches to
`reference/src/upv_vlm_v2/perception/clip_ensemble.py`. Legacy aggregation modes
remain available for reproducing historical configurations, not as the default.

The exact 24 strings are in `results/perception/ensemble_prompts.json` and the
runtime sibling `clip_ensemble_prompts.json`; tests require equality.

For each synonym, use `A photo of {}.`, `A cropped photo of {}.`,
`A close-up photo of {}.`, and `A photo of a piece of {}.`:

| Material | Synonyms |
|---|---|
| Brick | clay brick; fired clay masonry brick |
| Concrete | concrete block; concrete masonry unit |
| Timber | wood; timber lumber |

## Mathematics and Selection

Let v be the normalized crop embedding, t(c,k) each normalized text embedding,
and alpha the model's learned exponential logit scale. Compute:

```text
prototype(c) = normalize(mean_k t(c,k))
score(c) = softmax_over_three_materials(alpha * dot(v, prototype(c)))
```

Rank saved or freshly detected candidates by requested-material score. The
requested-minus-other-material margin breaks exact score ties only. Accept the
winner when its score is at least 0.35; otherwise return no verified match.
There is no source-class prior bonus, color/texture guard, or margin veto.
Missing crops or inference errors are logged as failures and stop selection.
Model inference runs in an isolated process, once for the candidate batch, and
exits afterward. This preserves the sequential model-lifecycle boundary.

The historical 63-query comparison used 14 descriptions (5 brick, 6 concrete,
3 timber), softmax over descriptions, the maximum probability per material, and
threshold 0.10. Averaging embeddings is not averaging probabilities. The new
score normalization changes the meaning of the threshold; this is not a
wording-only ablation. Some other historical deployment configurations also had
guards and priors; those are bypassed in ensemble mode.

## Reviewed Results

| Threshold | Present correct /63 | Absent rejected /9 | Overall accuracy |
|---|---:|---:|---:|
| 0.10 | 60 | 0 | 83.33% |
| 0.20 | 59 | 1 | 83.33% |
| 0.30 | 59 | 4 | 87.50% |
| **0.35** | **58** | **7** | **90.28%** |
| 0.40 | 52 | 7 | 81.94% |
| 0.50 | 42 | 7 | 68.06% |
| 0.60 | 38 | 9 | 65.28% |

All 72 queries are retained and fully labeled, including five candidate-specific
additional human reviews. Historical main scores were 57/63 present and 3/9
absent, totaling 60/72 (83.33%). The new best interval is approximately
`0.3484941721 < threshold <= 0.3558430672`; exact endpoints are saved in CSV.

Prompt families, aggregation and threshold were examined on this dataset. These
are exploratory tuned results, not an untouched holdout or independent validation.
Contact89, path-length and runtime tables are separate saved experiments, not
reruns under this new perception configuration. No fresh model/hardware execution
was performed when integrating the new default into this release.

## Data and Checks

`results/perception/trials72.csv` is sufficient for offline metrics and threshold
reproduction. `candidate_scores.csv` retains the search scores and `inputs.json`
records source provenance. Source paths in these imported records are historical
identifiers, not runtime dependencies. `perception.zip` uses a portable root
`index.json`: `clip72/README.md`, `clip72/results/`, `clip72/thresholds/`,
`clip72/crops/`, and `clip72/evidence/` contain the full evidence package.
Existing archive logical paths are preserved. The ZIP excludes model weights.

```bash
PYTHONPATH=src python -m upv_vlm_contact reproduce --output runs/ensemble_tables
python -m unittest discover -s tests -v
python tools/check_release.py --check-archives
```

The scoring design follows [CLIP prompt-embedding ensembles](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
The full record of tested wording and aggregation variants is included in the
archive. Baselines and deployment should be evaluated on additional independent
scenes before claiming generalization.
