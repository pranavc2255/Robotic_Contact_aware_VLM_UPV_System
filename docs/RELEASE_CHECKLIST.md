# Before Public Release

- Project code: rights reserved pending review; see LICENSE. Previously granted permissions and third-party licenses remain unaffected.
- Review third-party model/code licenses and retain their required notices; copied source is not automatically relicensed.
- Author names, ORCIDs and title are recorded in CITATION.cff; add a publication DOI only when available.
- Review all images and text for publication rights and sensitive laboratory information.
- Upload the four verified ZIPs to the replacement repository and populate the archive URLs. Earlier publication verification does not establish availability at the new address.
- Keep `release_assets/`, `datasets/`, `local_models/`, environments and new runs out of Git.
- Add the missing Windows waveform exports with units, sampling rate, acquisition settings, specimen IDs and checksum provenance.
- Validate `upv-pipeline` on a fresh GPU environment with real perception and Qwen. Synthetic CPU tests do not establish model compatibility or fresh-run equivalence.
- Import the later concrete extension / Contact89 evidence before claiming this snapshot includes the latest 89-pair comparison. The current catalog covers Contact73/82 and the original perception/path/runtime studies.
- Run tests and the standalone portability check, then review the Git file list before committing.
- Publishing repository: https://github.com/pranavc2255/Robotic_Contact_aware_VLM_UPV_System . No push or commit is performed automatically by local release validation.

## Licensing Status

Project code licensing is pending review; see the root LICENSE rights notice.
Dataset reuse is separately governed by DATA_LICENSE.md.
`docs/third_party/` contains available copied license
notices, but is not a complete dependency legal audit. Do not distribute model
weights through this repository. This checklist is an engineering release gate,
not a legal determination.
