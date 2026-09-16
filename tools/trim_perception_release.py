"""Build a current-method-only perception archive without changing source data."""
import csv
import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = ROOT / 'release_assets/perception.zip'
    catalog_path = ROOT / 'data/manifests/archives.json'
    catalog = json.loads(catalog_path.read_text())
    assert digest(archive.read_bytes()) == catalog['perception']['sha256']
    files = {}
    with zipfile.ZipFile(archive) as z:
        old_index = json.loads(z.read('index.json'))
        for name, entry in old_index.items():
            if not name.startswith('clip72/'):
                continue
            relative = name.removeprefix('clip72/')
            if relative.startswith('original/') or relative in {
                'README.md', 'FILE_MAP.csv', 'SHA256SUMS.txt',
                'experiment/prompts.json', 'experiment/report.md',
                'experiment/best_settings.csv', 'experiment/threshold_summary.csv',
            }:
                continue
            data = z.read(entry['file'])
            assert digest(data) == entry['sha256']
            files[name] = data
    key = 'clip72/experiment/candidate_scores.csv'
    reader = csv.DictReader(io.StringIO(files[key].decode()))
    rows = [r for r in reader if r['variant'] == 'object_surface_ensemble'
            and r['aggregation'] == 'mean_embedding_class_softmax']
    assert rows
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    files[key] = out.getvalue().encode()
    # Add the absent-query RGB sources omitted from the earlier compact bundle.
    plan = json.loads((ROOT.parent / 'outputs/absent_material_9/review1/case_plan.json').read_text())
    winners = list(csv.DictReader(io.StringIO(files['clip72/thresholds/winning_candidates72.csv'].decode())))
    absent = [(i, r) for i, r in enumerate(winners, 1) if r['kind'] == 'absent']
    assert len(absent) == len(plan) == 9
    for i, row in absent:
        matches = [p for p in plan if p['scene_id'] in row['trial'] and p['requested_material'] == row['requested']]
        assert len(matches) == 1, row['trial']
        source = matches[0]
        data = Path(source['rgb_path']).read_bytes()
        assert digest(data) == source['rgb_sha256']
        files[f'clip72/evidence/q{i:02d}_rgb.png'] = data
    files['clip72/README.md'] = b'''# GSAM2 + CLIP ensemble perception results

Current method: object_surface_ensemble / mean_embedding_class_softmax.
CLIP ViT-B/32 uses 24 texts, normalized class-average text embeddings,
three-class softmax, highest requested-class candidate score and threshold 0.35.
No source prior, color/texture guard or margin rejection is applied.

Results: 58/63 present-query successes, 7/9 absent rejections, 65/72 overall
(90.28%). results/ contains decisions; thresholds/ contains the sensitivity
analysis; prompts/ contains exact text; annotations/ contains review labels.
experiment/candidate_scores.csv contains only the current method's scores.
crops/ contains the scoring images; evidence/ contains RGB, winning masks and
original source metadata. Original metadata may contain previous selector scores:
it records how the saved crops/masks were obtained, not the current decisions.
Use results/ and thresholds/ for current-method decisions and metrics.

The method and threshold were explored on these inspected data, not an untouched
holdout. Removing redundant old result bundles does not change that limitation.
Source data and annotations have not been relabeled by this packaging step.

This archive uses index.json to map logical names to short content-addressed
objects. Install using the repository data installer. SHA256SUMS.txt lists
logical file checksums; index.json also checks every stored object.
'''
    files['clip72/SHA256SUMS.txt'] = ''.join(
        f'{digest(data)}  {name.removeprefix("clip72/")}\n'
        for name, data in sorted(files.items())).encode()
    pending = archive.with_name('perception_current_pending.zip')
    index, seen = {}, set()
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            sha = digest(data)
            member = 'objects/' + sha + Path(name).suffix
            if member not in seen:
                z.writestr(member, data)
                seen.add(member)
            index[name] = dict(file=member, sha256=sha, size_bytes=len(data))
        z.writestr('index.json', json.dumps(index, indent=2))
    with zipfile.ZipFile(pending) as z:
        assert z.testzip() is None
        for entry in index.values():
            assert digest(z.read(entry['file'])) == entry['sha256']
    history = archive.parent / 'history'
    history.mkdir(exist_ok=True)
    backup = history / f'perception_before_trim_{datetime.now():%Y%m%d_%H%M%S}.zip'
    shutil.copy2(archive, backup)
    pending.replace(archive)
    sha = digest(archive.read_bytes())
    catalog['perception'].update(sha256=sha, size_bytes=archive.stat().st_size,
        logical_files=len(index), unique_objects=len(seen),
        status='Current ensemble archive validated locally; replace GitHub perception.zip asset')
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
    archive.with_suffix('.zip.sha256').write_text(sha + '  perception.zip\n')
    print(json.dumps(dict(archive=str(archive), backup=str(backup),
        logical_files=len(index), bytes=archive.stat().st_size, sha256=sha), indent=2))


if __name__ == '__main__':
    main()
