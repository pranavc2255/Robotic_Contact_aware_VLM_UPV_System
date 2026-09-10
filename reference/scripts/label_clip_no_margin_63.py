"""Review changed CLIP selections with score >= 0.10 and no margin gate."""
import argparse
import csv
import json
from pathlib import Path
from capture_revision_12_scenes import ROOT, open_image
from rerun_revision_clip_original27 import OLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/revision_36_perception/clip_no_margin_63')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    revision = ROOT/'outputs/revision_36_perception/original27_prompts_review'
    jobs = []
    for r in csv.DictReader((OLD/'T5_v2_master_results.csv').open()):
        jobs.append(('old_'+r['trial_index'], 'original27', r['input_text'].lower(), r['final_selected_candidate_id'], r['final_selected_class_manual'].lower(), Path(r['case_output_dir'])/'material_verification/material_scores.json', Path(r['raw_rgb_path'])))
    for f in sorted(revision.glob('trials/*/manual_label.json')):
        r = json.loads(f.read_text())
        jobs.append((f.parent.name, 'revision36', r['requested_material'], r['selected_candidate_id'], r['actual_class'], f.parent/'material_scores.json', Path(r['source_rgb'])))
    assert len(jobs) == 63
    rows = []
    pending = []
    prompts = None
    for name, group, requested, previous, label, scorefile, rgb in jobs:
        candidates = json.loads(scorefile.read_text())['candidates']
        for c in candidates:
            assert c.get('score_source') == 'clip_crop_verifier'
            if prompts is None:
                prompts = c['clip_labels']
            assert c['clip_labels'] == prompts
        def rank(c):
            scores = c['clip_scores']
            value = float(scores[requested])
            return value, value-max(float(v) for k,v in scores.items() if k != requested)
        chosen = max(candidates, key=rank)
        score, margin = rank(chosen)
        selected = chosen['candidate_id'] if score >= 0.10 else None
        actual = 'none' if selected is None else label if selected == previous else None
        source = 'no_selection' if selected is None else 'reused_label' if actual is not None else 'pending'
        row = dict(trial=name, dataset=group, requested_material=requested,
                   candidate_id=selected, score=score, margin_diagnostic_only=margin,
                   actual_class=actual, label_source=source,
                   correct=actual == requested if actual is not None else '',
                   scores_file=str(scorefile))
        rows.append(row)
        if actual is None:
            pending.append((row, chosen, rgb))
    print(f'{len(pending)} changed candidates require labels; other labels reused or no-selection.')
    for row, _, _ in pending:
        print(row['trial'], row['candidate_id'])
    if args.dry_run:
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def save():
        with (out/'combined_63_results.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        lines = ['# CLIP-only without margin rejection', '',
                 'Saved GSAM2 candidates and original CLIP prompts. Rank by requested score (margin only breaks ties), accept score >= 0.10. No color/texture rules or source bonuses. No inference rerun.', '']
        for group in ['original27', 'revision36', 'all']:
            subset = [r for r in rows if group == 'all' or r['dataset'] == group]
            n = len(subset); good = sum(r['correct'] is True for r in subset)
            unknown = sum(r['actual_class'] is None for r in subset)
            line = f'{group}: {good}/{n} confirmed correct; {unknown} pending. '
            line += f'Success: {100*good/n:.2f}%' if not unknown else f'Range: {100*good/n:.2f}-{100*(good+unknown)/n:.2f}% (not final)'
            lines.append(line); print(line)
        (out/'summary.md').write_text('\n\n'.join(lines)+'\n')

    # Restore all completed labels before interaction, so quitting retains totals.
    for row, _, _ in pending:
        path = out/row['trial']/'manual_label.json'
        if path.exists():
            label = json.loads(path.read_text())
            assert label['candidate_id'] == row['candidate_id']
            row.update(actual_class=label['actual_class'], correct=label['actual_class'] == row['requested_material'], label_source='manual')
    choices = {'b':'brick', 't':'timber', 'c':'concrete block', 'n':'none', 'a':'ambiguous'}
    try:
        from PIL import Image, ImageDraw
        for row, candidate, rgb in pending:
            if row['actual_class'] is not None:
                continue
            folder = out/row['trial']; folder.mkdir(exist_ok=True)
            image = Image.open(rgb).convert('RGB')
            mask = Image.open(candidate['mask_path']).convert('L').point(lambda v: 105 if v else 0)
            assert image.size == mask.size
            overlay = Image.composite(Image.new('RGB', image.size, (0,255,70)), image, mask)
            crop_path = candidate.get('inner_texture_crop_path') or candidate.get('crop_path')
            crop = Image.open(crop_path).convert('RGB')
            crop.thumbnail((400, image.height))
            panel = Image.new('RGB', (image.width+420, image.height+70), 'white')
            panel.paste(overlay, (0,70)); panel.paste(crop, (image.width+10,70))
            ImageDraw.Draw(panel).text((10,15), f"{row['trial']} | requested: {row['requested_material']} | selected: {row['candidate_id']} | right: scored crop", fill='black')
            preview = folder/'review.png'; panel.save(preview)
            print(f"\n{row['trial']}: label the GREEN selected object, not the requested class.")
            open_image(preview)
            while True:
                answer = input('[b] brick / [t] timber / [c] concrete / [n] none / [a] ambiguous / [v] view / [q] quit: ').strip().lower()
                if answer == 'q':
                    return
                if answer == 'v':
                    open_image(preview); continue
                if answer not in choices:
                    continue
                actual = choices[answer]
                if input(f'Save label {actual}? [y/n]: ').strip().lower() != 'y':
                    continue
                row.update(actual_class=actual, correct=actual == row['requested_material'], label_source='manual')
                (folder/'manual_label.json').write_text(json.dumps(dict(candidate_id=row['candidate_id'], actual_class=actual), indent=2))
                save(); break
    except (KeyboardInterrupt, EOFError):
        print('\nStopped; progress retained.')
    finally:
        save()
        print(f'Output: {out}')


if __name__ == '__main__':
    main()
