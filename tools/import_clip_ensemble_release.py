"""Merge a reviewed clip72 bundle into perception.zip, preserving prior logical assets."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from upv_vlm_contact.data import sha256


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-directory',type=Path,required=True,help='Extracted reviewed clip72 package')
    args=p.parse_args()
    bundle=args.bundle_directory.resolve()
    for line in (bundle/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle) or sha256(path)!=digest:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    final=rows(bundle/'results/final72_at_035.csv')
    winners={r['trial']:r for r in rows(bundle/'thresholds/winning_candidates72.csv')}
    assert len(final)==72 and len(winners)==72 and sum(r['correct']=='True' for r in final)==65
    exact_prompts=json.loads((bundle/'prompts/winning_prompts.json').read_text())
    runtime_prompts=ROOT/'reference/src/upv_vlm_v2/perception/clip_ensemble_prompts.json'
    assert exact_prompts==json.loads(runtime_prompts.read_text())
    catalog_path=ROOT/'data/manifests/archives.json'
    catalog=json.loads(catalog_path.read_text())
    old=ROOT/'release_assets/perception.zip'
    if sha256(old)!=catalog['perception']['sha256']:
        raise ValueError('Existing perception.zip does not match manifest')
    target=old.with_name('perception_ensemble_pending.zip')
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(old) as previous, zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        index=json.loads(previous.read('index.json'))
        old_count=len(index)
        if any(k.startswith('clip72/') for k in index):
            raise ValueError('Ensemble already imported; refusing ambiguous repeated import')
        existing=set(previous.namelist())
        for item in previous.infolist():
            if item.filename!='index.json':
                z.writestr(item,previous.read(item.filename))
        for source in sorted(bundle.rglob('*')):
            if not source.is_file():
                continue
            data=source.read_bytes()
            digest=hashlib.sha256(data).hexdigest()
            logical='clip72/'+source.relative_to(bundle).as_posix()
            member='objects/'+digest+source.suffix.lower()
            if member not in existing:
                z.writestr(member,data)
                existing.add(member)
            index[logical]=dict(file=member,sha256=digest,size_bytes=len(data))
        z.writestr('index.json',json.dumps(index,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert max(map(len,z.namelist()))<130
        for entry in index.values():
            assert hashlib.sha256(z.read(entry['file'])).hexdigest()==entry['sha256']
    result=ROOT/'results/perception'
    result.mkdir(exist_ok=True)
    with (result/'trials72.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trial','kind','requested_material','candidate_id','score','candidate_good','correct_at_035'],lineterminator='\n')
        writer.writeheader()
        for r in final:
            w=winners[r['trial']]
            assert w['kind']=='absent' or w['good'] in ('True','False')
            writer.writerow(dict(trial=r['trial'],kind=r['kind'],requested_material=r['requested'],
                candidate_id=r['winner_id'],score=r['score'],candidate_good=w['good'],correct_at_035=r['correct']))
    for src,dest in [('results/thresholds_for_paper.csv','threshold_summary.csv'),
                     ('thresholds/exact_intervals.csv','threshold_exact_intervals.csv'),
                     ('prompts/winning_prompts.json','ensemble_prompts.json'),
                     ('annotations/labels.csv','ensemble_labels.csv'),
                     ('annotations/additional_threshold_labels.csv','additional_threshold_labels.csv')]:
        shutil.copy2(bundle/src,result/dest)
    for src,dest in [('candidate_scores.csv','candidate_scores.csv'),('inputs.json','inputs.json')]:
        shutil.copy2(bundle/'experiment'/src,result/dest)
    history=old.parent/'history'
    history.mkdir(exist_ok=True)
    backup=history/f'perception_before_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    shutil.copy2(old,backup)
    target.replace(old)
    digest=sha256(old)
    catalog['perception'].update(sha256=digest,size_bytes=old.stat().st_size,logical_files=len(index),
        unique_objects=len({v['file'] for v in index.values()}),
        status='New ensemble archive validated locally; replace perception.zip on GitHub before remote installation',
        main_results='clip72/results/final72_at_035.csv',threshold=.35)
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    old.with_suffix('.zip.sha256').write_text(digest+'  perception.zip\n')
    print(json.dumps(dict(archive=str(old),backup=str(backup),old_logical_files=old_count,
        new_logical_files=len(index),bytes=old.stat().st_size,sha256=digest),indent=2))


if __name__=='__main__':
    main()
