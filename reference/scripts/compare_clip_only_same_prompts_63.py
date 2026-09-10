"""Offline CLIP-only decision ablation; no inference or hardware imports."""
import argparse
import csv
import json
from pathlib import Path
from capture_revision_12_scenes import ROOT, open_image
from rerun_revision_clip_original27 import OLD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision-run', type=Path, default=ROOT/'outputs/revision_36_perception/original27_prompts_review')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/revision_36_perception/clip_only_same_prompts_63')
    p.add_argument('--annotate', action='store_true')
    a=p.parse_args()
    jobs=[]
    for r in csv.DictReader((OLD/'T5_v2_master_results.csv').open()):
        folder=Path(r['case_output_dir'])
        jobs.append(('old_'+r['trial_index'], 'original27', r['input_text'].lower(), r['final_selected_candidate_id'], r['final_selected_class_manual'].lower(), folder/'material_verification/material_scores.json', Path(r['raw_rgb_path'])))
    for f in sorted(a.revision_run.glob('trials/*/manual_label.json')):
        r=json.loads(f.read_text())
        jobs.append((f.parent.name,'revision36',r['requested_material'],r['selected_candidate_id'],r['actual_class'],f.parent/'material_scores.json',Path(r['source_rgb'])))
    assert len(jobs)==63, f'Expected 63 labeled trials, found {len(jobs)}'
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    choices={'b':'brick','t':'timber','c':'concrete block','a':'ambiguous','n':'none'}
    rows=[]
    prompts=None
    stop=False
    for name, group, requested, old_id, old_label, scorefile, rgb in jobs:
        data=json.loads(scorefile.read_text()); candidates=data['candidates']
        for c in candidates:
            if prompts is None:prompts=c['clip_labels']
            assert c['clip_labels']==prompts, 'Prompt mismatch'
            assert c.get('score_source')=='clip_crop_verifier', 'Invalid CLIP source'
        def rank(c):
            s=c['clip_scores'];value=float(s[requested]);return value,value-max(float(v) for k,v in s.items() if k!=requested)
        chosen=max(candidates,key=rank)
        score,margin=rank(chosen)
        new_id=chosen['candidate_id'] if score>=0.10 and margin>=0.02 else None
        folder=out/name;folder.mkdir(exist_ok=True)
        label_file=folder/'manual_label.json'
        actual=None
        if new_id is None:actual='none';source='automatic_no_selection'
        elif new_id==old_id:actual=old_label;source='reused_label'
        else:
            source='needs_annotation'
            if label_file.exists():
                saved=json.loads(label_file.read_text())
                assert saved['candidate_id']==new_id
                actual=saved['actual_class'];source='new_manual_label'
            elif a.annotate and not stop:
                from PIL import Image,ImageDraw
                image=Image.open(rgb).convert('RGB')
                mask=Image.open(chosen['mask_path']).convert('L').point(lambda x: 110 if x else 0)
                image=Image.composite(Image.new('RGB',image.size,(0,255,80)),image,mask)
                canvas=Image.new('RGB',(image.width,image.height+45),'white');canvas.paste(image,(0,45))
                ImageDraw.Draw(canvas).text((10,10),f'{name}: requested {requested}; candidate {new_id}',fill='black')
                overlay=folder/'review.png';canvas.save(overlay);open_image(overlay)
                print(f'{name}: {old_id} -> {new_id}',flush=True)
                while True:
                    try:v=input('[b] brick / [t] timber / [c] concrete / [a] ambiguous / [n] none / [q] quit: ').strip().lower()
                    except (EOFError,KeyboardInterrupt):v='q'
                    if v=='q':stop=True;break
                    if v in choices:
                        actual=choices[v];source='new_manual_label'
                        label_file.write_text(json.dumps(dict(candidate_id=new_id,actual_class=actual),indent=2));break
        rows.append(dict(trial=name,dataset=group,requested_material=requested,old_candidate=old_id,new_candidate=new_id,score=score,margin=margin,label_source=source,actual_class=actual,correct=actual==requested if actual is not None else '',scores_file=str(scorefile)))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# GSAM2 + CLIP-only, same saved prompts', '', 'Rank by requested CLIP score, breaking ties by margin. Accept score >= 0.10 and margin >= 0.02; otherwise no selection. No color/texture rules or source bonuses. Saved GSAM2 candidates reused; no inference. This simplified gate is distinct from the historical verifier.', '']
    for group in ['original27','revision36','all']:
        subset=[r for r in rows if group=='all' or r['dataset']==group]
        good=sum(r['correct'] is True for r in subset);pending=sum(r['actual_class'] is None for r in subset);n=len(subset)
        line=f'{group}: {good} confirmed correct / {n}; {pending} pending; success '+(f'{100*good/n:.2f}%' if pending==0 else f'range {100*good/n:.2f}-{100*(good+pending)/n:.2f}%')
        print(line);lines.append(line)
    (out/'summary.md').write_text('\n\n'.join(lines)+'\n')
    print(f'Outputs: {out}')


if __name__=='__main__':main()
