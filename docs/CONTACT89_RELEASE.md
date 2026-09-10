# Contact89 and PL-200 release additions

## Contact89

The committed `results/contact/contact89/predictions.csv` contains the saved
predictions and final manual labels for 89 pairs across 21 cases for each of
Qwen2.5-32B, Qwen2.5-3B, Qwen3-2B and Classical RGB+mask. These are the original
combined comparison, not the later prequantized Qwen or prompt-sensitivity runs.
`reported_metrics.csv` retains the saved comparison used for regression checks.

Run `upv-reproduce reproduce --output runs/contact89_check`. The Contact89
section and `contact89.json` reproduce classification and selected-pair counts.
Classification uses saved usability flags. VLM selection uses the highest score
among usable pairs, numeric-ID tie break; classical selection minimizes risk.
Good selections divided by all scenes and by selections made are different
denominators; abstentions are retained separately. Manuscript table numbering
is not encoded: compare the generated numbers with the final Table 3.

`contact89.zip` is standalone and contains the prior contact archive plus the
five-concrete raw RGB-D, crops, masks, geometry, labels, responses, prompts and
saved comparison artifacts. Its index maps original paths to short hashed files.
Existing Contact73/82 data and cohort exclusions are unchanged. The public CLI
supports `--dataset contact89` for saved-input VLM and RGB+mask reruns. RGB-D
baseline recomputation remains supported only for the older cohorts.

## PL-200

`pl200.zip` organizes the supplied PL-200 export by original source group:
`contact30`, `e45`, `concrete_revision`, and `audit`. Short numbered filenames
are reversible through `FILE_MAP.csv`; hashes verify unchanged source bytes.
The original virtual environment, bytecode caches and editor lock files are
excluded. Native USP files and existing analyses are retained without claiming
that historical scripts are portable or their metrics newly verified.

CSV exports include device metadata, transmission times and waveform samples.
Direct parsing of the PL-Link tab-separated exports found 21 raw CSV files and
190 `Transmission Time` rows: 60 in the earlier contact comparison, 90 in E45,
and 40 in the concrete revision export. These are recorded measurement rows,
not 190 independently verified robot cycles. The concrete export is UTF-16 and
contains PL-Link version 3.0.5.0 and device acquisition settings.
The concrete revision export identifies its instrument folder as `Concrete4`;
this does not establish correspondence to Contact89 case IDs. Do not infer
one-to-one links between UPV readings and anchor labels from folder names alone.

## Publication

Upload `release_assets/contact89.zip` and `release_assets/pl200.zip` as additional
release assets, with their SHA256 files. Push updated code, tables, manifests and
documentation. No existing published ZIP was overwritten. After verifying the
uploaded hashes, set their catalog URLs to the actual release download URLs.
Until then installation accepts the local `--archive` path rather than claiming
these additions are already downloadable.

Validation: `python3 tools/check_release.py --check-archives` performs detached
saved-result reproduction, tests and archive/member hash checks without models
or hardware. It does not push to GitHub.
