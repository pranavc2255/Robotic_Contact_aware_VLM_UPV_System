# Release Readiness Assessment

Updated: 2026-09-10. Parent working branch: `robot_upv_interactive_20260527`.
New repository v0.1.0 archives are public. See RELEASE_AUDIT_20260910.md for local validation.

## Implemented

- Independent saved-metric reproduction from a detached repository copy.
- `upv-pipeline`: explicit perception, anchor/crop preparation, multi-image contact
  inference, and selected-contact path stages, separately or in sequence.
- Saved-mask reuse with provenance, immutable completed stages, input hashes,
  effective config, stage logs and per-stage wall times.
- Explicit `/infer_multi` authorization; no server startup in the runner.
- CPU smoke coverage of saved-mask preparation, crop generation, mocked HTTP
  contact response and percentile path computation. No neural weights loaded.
- Installation instructions, updated README, staged commands and release gates.

## Remaining Gates

1. **Fresh GPU validation:** exercise detector/SAM2/CLIP and Qwen with a clean
   environment and record actual model identities, package versions and VRAM.
   Current synthetic tests do not establish GPU correctness.
2. **Latest evidence import:** the public snapshot covers Contact73/82. Add the
   concrete/Contact89 extension and its manual-mask-override provenance before
   describing the repository as containing the latest full results.
3. **Data hosting complete:** all four new-repository v0.1.0 assets match the
   catalog sizes/digests and download URLs are populated.
4. **Licensing and citation:** pending code licensing notice and author/title/ORCID
   metadata are recorded. Dataset reuse is reserved pending a licensing decision.
   Third-party redistribution review remains incomplete.
5. **Hardware scope:** the optional rig-specific robot-cycle launcher is separate
   from the computational CLI and requires operator confirmation and calibration.
   Simulation is not physical validation. Raw waveform exports remain absent.

## Refactoring Decision

Do not broadly rewrite the validated scientific algorithms just to publish them.
The new adapter reuses bundled implementations with explicit stage boundaries.
Historical scripts under `reference/` are not all supported public entrypoints;
use README/REPRODUCTION/PIPELINE commands only. A later cleanup can reduce legacy
code once fresh GPU equivalence tests exist, without changing paper results.

No existing numeric result tables or parent pipeline source files were modified
by this readiness task. No remote, commit, push, camera operation, hardware
command, real model server request or neural inference was performed.
