# Assembly Validation

The release was assembled from branch `robot_upv_interactive_20260527`, including
its uncommitted working-tree state. The new nested repository uses branch `main`.
No commit, remote, push, model download, GPU inference or hardware run occurred.

Completed checks:

- Original tracked-file hashes were unchanged during assembly; see `assembly_validation.json`.
- All four ZIP CRCs, catalog SHA-256 values and individual object hashes passed.
- Archive internal paths are at most 77 characters; full extraction length still depends on the chosen destination directory.
- A separate temporary copy reproduced metrics without access through parent-relative paths.
- Nine stdlib unit tests passed: contact confusion counts/selections, 63-trial perception, V1/V2 path summaries, runtime rows, invalid labels, ties, schemas, cohort counts and unsafe archive paths.
- Contact and path archives installed successfully through checksum verification.
- CPU-only RGB+mask smoke test: five pairs from `case_001_brick_01_clean_regular`, zero failures; all labels and risk scores matched saved Contact82 results.
- VLM client dry-run validated five images/one request; no HTTP call was made.
- RGB-D adapter dry-run validated 18 Contact82 groups and seven new10 groups used for Contact73.
- Raw-depth adapter dry-run validated 15 selected-anchor records; no new path evaluation was run.
- Model-server launcher dry-run printed its command without importing model libraries or starting a server.
- Syntax checks passed for new Python code. Archive and large installed data folders are ignored by the nested Git repository.

`release_validation.json` contains the latest machine-readable packaging and
detached-copy results. Validation does not certify CUDA/version compatibility,
fresh-generation equivalence, all historical CLIs, legal redistribution rights,
complete data anonymization or physical safety. Raw-image perception portability
and full GPU/raw-depth/RGB-D rerun checks remain explicitly unverified. The
pattern-based secret scan is a precaution, not a substitute for owner review.

Repeat the inexpensive release check:

```bash
python tools/check_release.py --check-archives
```

This reads local archives and runs saved-table tests; it does not run models or
hardware. Omitting `--check-archives` supports testing a Git-only checkout.
