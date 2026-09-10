# Local release audit: 2026-09-10

Scope: complete independently verifiable items from the earlier release README,
not import new experiments or publish data. Parent branch:
`robot_upv_interactive_20260527`. No remote operations were performed.

## Completed

Later owner decision: current project code licensing is pending review; root
LICENSE now contains a rights-reserved notice. The initial audit entry below
records historical state, not the current grant. Previously granted permissions
and third-party notices are unaffected.

- Copied the existing owner-selected Apache-2.0 LICENSE from the publishing
  checkout; byte equality verified with `cmp`. Updated project packaging metadata
  and stale README/checklist statements. This does not relicense third-party data.
- Fixed release-checker traversal to prune local model, environment, cache,
  third-party checkout and output directories before scanning/copying.
- Ran `python3 tools/check_release.py --check-archives`: passed. All four
  archive SHA256 hashes and indexed object hashes matched the catalog. Checked
  archive member paths and contact input coverage. Maximum internal path length
  was 77 characters in each archive. These checks do not establish publication
  consent or legal ownership.
- Detached-copy saved-result reproduction and all 13 offline tests passed.
- Ran `.venv/bin/python tools/check_pipeline_cpu.py`: passed synthetic saved-mask,
  anchor and path integration with mocked contact selection. No model inference.
- Release candidate scan found no matches for the checker's narrow secret-token
  patterns. This is not an exhaustive privacy/security review.
- Reviewed `docs/INSTALL.md`: CPU reproduction is separate from image/model
  environments; the Grounded-SAM-2 revision is pinned, but Python GPU dependencies
  remain broad. Existing environment records are provenance, not portable locks.

Machine-readable archive/reproduction evidence: `release_validation.json`.

## Third-party notice inventory

`docs/third_party/LICENSE` and `LICENSE_groundingdino` both contain Apache-2.0
license text. Their filenames alone do not fully establish which copied source
files each covers. Preserve both. A complete source-to-upstream ownership map,
applicable NOTICE files and model-specific terms remain unverified. No model
weights are part of the intended Git release. No permission is inferred for
images or data merely because project code has an Apache license.

## Still pending

Subsequent update: the owner supplied the authors, ORCIDs and title, now in
CITATION.cff. The public v0.1.0 release contains all four archives; GitHub API
sizes/digests match the catalog, whose URLs are now populated. Items 2 and 3
below describe the earlier audit state and are resolved except for a future DOI.

1. Owner/rights-holder selection of a dataset license and publication consent.
2. Archive hosting and actual HTTPS URLs. All four catalog URLs remain null;
   no fictitious links were added. Local archives are validated and ready for
   the owner's publishing decision, not declared legally cleared.
3. Confirmed author order, paper title and citation details. No guessed citation
   file was created; a DOI is not required until one actually exists.
4. Fresh GPU installation/inference validation: not performed. CPU tests do not
   establish CUDA/model compatibility on a new machine.
5. Owner review and synchronization into the separate publishing checkout,
   followed by commit/push. That checkout and GitHub were not modified.

Contact89 import and waveform exports are outside this earlier-release task;
they are only necessary if the release claims those additional capabilities.
No numerical result tables, raw datasets, hardware code or inference code changed.
