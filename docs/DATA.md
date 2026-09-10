# Dataset Archives and Provenance

`data/manifests/archives.json` records archive byte sizes, SHA-256, unique object
counts and logical file counts. Download URLs are unset pending archive upload
to the replacement repository. Local archive hashes remain verified.
See [dataset rights](../DATA_LICENSE.md) before reuse. Local ZIPs are in `release_assets/` and
are ignored by Git. Copy/upload them separately, not through normal Git history.

Every ZIP has `index.json` and `objects/<sha256>.<extension>`. The index retains
the original logical path, release hash, original source hash, and whether text
was sanitized. Binary image/depth bytes are unchanged. Repeated files are stored
once per archive. Extracted filenames are short; original long Windows-incompatible
experiment names are only JSON keys. Use the installer instead of reconstructing
the original directory hierarchy on Windows.

| Archive | Contents |
|---|---|
| contact.zip | 73/82 crops, masks, raw geometry/RGB-D lineage, labels, model evidence, classical diagnostics |
| perception.zip | Final 63-trial evidence, source RGB, candidates, scores, labels, prompt text and annotation provenance |
| path_length.zip | Raw 15-case saved session, masks, selected paths, V1/V2 evaluations and packaged figures |
| runtime.zip | E45 master tables and timing/pipeline/robot execution logs |

The installer checks the whole ZIP checksum, rejects path traversal and symlinks,
checks each object checksum, and refuses an existing destination. Logical paths
in records beginning `workspace/` are resolved by the public `Assets` adapter.
`source_provenance.json` records copied small-file source/release hashes;
`source_state.json` records the original branch/commit and dirty-work inclusion.
This is a working-tree snapshot, not an assertion that every file was committed
under that parent revision. The parent workspace was not changed by assembly.

Text substitutions remove source home paths and private 192.168.* robot hosts.
These are portability/privacy redactions, not proof of complete anonymization.
Inspect faces, labels, lab information, serial numbers, hostnames and third-party
assets before public release. No model checkpoints or virtual environments are
included. Raw UPV waveform files from the Windows machine remain missing.
