"""Rerun historical E1 CLIP verification on saved revision candidates only."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/T5_v2_experiment_1_target_selection_fixed27/session_20260518_142404'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision-run', type=Path, default=ROOT/'outputs/revision_36_perception/manual_review2')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    jobs = []
    for f in sorted(a.revision_run.glob('trials/*/manual_label.json')):
        meta = json.loads(f.read_text())
        manifest = f.parent/'perception/candidate_pool/candidate_pool_manifest.json'
        candidates = json.loads(manifest.read_text())['candidates']
        for c in candidates:
            for key in ['mask_path', 'inner_texture_crop_path', 'masked_texture_crop_path']:
                assert Path(c[key]).is_file(), (f, key)
        assert Path(meta['rgb_path']).is_file()
        jobs.append((f.parent.name, meta, candidates))
    assert len(jobs) == 36, f'Expected 36 trials, found {len(jobs)}'
    config = json.loads((OLD/'config_snapshot.json').read_text())
    if a.dry_run:
        print('Validated 36 saved candidate pools and original config. Only CLIP will rerun.')
        return
    sys.path.insert(0, str(ROOT/'src'))
    from upv_vlm_v1.experiment.t5_v2_target_selection_fixed27_runner import (
        T5V2Trial, build_material_verification, _clip_prompt_rows)
    from PIL import Image, ImageDraw
    import numpy as np
    # Require the historical prompt identity, rather than assuming current constants match.
    first = next(csv.DictReader((OLD/'T5_v2_master_results.csv').open()))
    original = json.loads((Path(first['case_output_dir'])/'material_verification/material_scores.json').read_text())
    assert [r['prompt'] for r in _clip_prompt_rows()] == original['candidates'][0]['clip_labels'], 'Historical prompts differ from current implementation'
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(revision_run=str(a.revision_run.resolve()), original_session=str(OLD), config=config)
    snapshot = out/'run_config.json'
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == signature, 'Output/config mismatch'
    snapshot.write_text(json.dumps(signature, indent=2))
    (out/'clip_prompts.json').write_text(json.dumps(_clip_prompt_rows(), indent=2))
    for i, (name, meta, candidates) in enumerate(jobs, 1):
        folder = out/'trials'/name
        folder.mkdir(parents=True, exist_ok=True)
        result_file = folder/'result.json'
        if result_file.exists():
            print(f'[{i}/36] Already completed: {name}', flush=True)
            continue
        print(f'[{i}/36] CLIP: {name}', flush=True)
        trial = T5V2Trial(i, meta['scene_id'], '', meta['requested_material'], meta['requested_material'])
        result = build_material_verification(trial, candidates, config, case_dir=folder)
        (folder/'material_scores.json').write_text(json.dumps(result, indent=2, default=str))
        if result.get('clip_verifier_failed') or result.get('clip_environment_failed'):
            raise RuntimeError(f'CLIP failed: inspect {folder}/material_scores.json')
        selected = result.get('selected_candidate') or {}
        if result.get('no_verified_match'):
            selected = {}
        image = Image.open(meta['rgb_path']).convert('RGB')
        if selected:
            mask = np.asarray(Image.open(selected['mask_path']).convert('L')) > 0
            arr = np.array(image)
            arr[mask] = (arr[mask]*0.55 + np.array([0,255,70])*0.45).astype('uint8')
            image = Image.fromarray(arr)
            crop = selected.get('inner_texture_crop_path') or selected.get('crop_path')
            if crop:
                Image.open(crop).save(folder/'selected_crop.png')
        canvas = Image.new('RGB', (image.width, image.height+55), 'white')
        canvas.paste(image, (0,55))
        ImageDraw.Draw(canvas).text((12,15), f"{name}: selected {selected.get('candidate_id', 'NONE')}", fill='black')
        canvas.save(folder/'review_overlay.png')
        payload = dict(scene_id=meta['scene_id'], requested_material=meta['requested_material'], condition=meta['condition'], selected_candidate_id=selected.get('candidate_id'), source_rgb=meta['rgb_path'])
        result_file.write_text(json.dumps(payload, indent=2))
    print(f'Completed. Run label_revision_clip_original27.py --run-dir {out}')


if __name__ == '__main__':
    main()
