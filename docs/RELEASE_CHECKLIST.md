# Before Public Release

- Project code: rights reserved pending review; see LICENSE. Previously granted permissions and third-party licenses remain unaffected.
- Review third-party model/code licenses and retain their required notices; copied source is not automatically relicensed.
- Author names, ORCIDs and title are recorded in CITATION.cff; add a publication DOI only when available.
- Review all images and text for publication rights and sensitive laboratory information.
- Four verified ZIPs are public on the new repository's v0.1.0 release; archive URLs are populated and GitHub sizes/digests match the catalog.
- Keep `release_assets/`, `datasets/`, `local_models/`, environments and new runs out of Git.
- PL-200 exports are packaged locally in `pl200.zip`; upload the asset. Preserve instrument metadata and verify specimen/session associations before waveform reprocessing claims.
- Validate `upv-pipeline` on a fresh GPU environment with real perception and Qwen. Synthetic CPU tests do not establish model compatibility or fresh-run equivalence.
- Contact89 labels, predictions and inputs are imported and tested locally. Upload `contact89.zip` and `pl200.zip`, push updated source/tables/manifests, then configure verified download URLs. Local completion does not mean these new assets are already public.
- Run tests and the standalone portability check, then review the Git file list before committing.
- Publishing repository: https://github.com/pranavc2255/Robotic_Contact_aware_VLM_UPV_System . No push or commit is performed automatically by local release validation.

## Licensing Status

Project code licensing is pending review; see the root LICENSE rights notice.
Dataset reuse is separately governed by DATA_LICENSE.md.
`docs/third_party/` contains available copied license
notices, but is not a complete dependency legal audit. Do not distribute model
weights through this repository. This checklist is an engineering release gate,
not a legal determination.
