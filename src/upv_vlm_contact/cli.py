"""Offline reproduction CLI. Heavy/model imports occur only in explicit commands."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from .data import install
from .metrics import boolean, classify, length_metrics, select_usable


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contact(root, dataset):
    rows=read(root/'results/contact'/dataset/'predictions.csv')
    expected=73 if dataset=='contact73' else 82
    models=defaultdict(list)
    for r in rows:models[r['model']].append(r)
    summaries={};selections=[]
    for model,rr in models.items():
        ids={(r['split'],r['case'],r['anchor_id']) for r in rr}
        if len(rr)!=expected or len(ids)!=expected:
            raise ValueError(f'{model}: incomplete/duplicate predictions; expected {expected}')
        stats=classify(rr)
        by_case=defaultdict(list)
        for r in rr:by_case[(r['split'],r['case'])].append(r)
        good=bad=abstain=ties=0
        for (split,case),group in by_case.items():
            if model=='Classical RGB-D':
                eligible=[r for r in group if boolean(r['predicted_good'])]
                selected=max(eligible,key=lambda r:(float(r['score']),-abs(int(r['anchor_id'][1:])-(len(group)+1)/2))) if eligible else None
            else:
                selected=select_usable(group,minimum_risk=model=='Classical RGB+mask')
            eligible=[r for r in group if boolean(r['predicted_good'])]
            tied=[r['anchor_id'] for r in eligible if selected and float(r['score'])==float(selected['score'])]
            ties+=len(tied)>1
            good+=selected is not None and boolean(selected['manual_good'])
            bad+=selected is not None and not boolean(selected['manual_good'])
            abstain+=selected is None
            selections.append(dict(model=model,split=split,case=case,selected=selected['anchor_id'] if selected else None,
                manual_good=boolean(selected['manual_good']) if selected else None,top_tie_ids=tied))
        stats.update(scenes=len(by_case),selected_good=good,selected_bad=bad,abstain=abstain,
                     selected_good_rate=good/len(by_case),top_tied_scenes=ties)
        summaries[model]=stats
    return dict(method='saved flags; Python score selection; not original VLM final action',metrics=summaries,selections=selections)


def perception(root):
    rows=read(root/'results/perception/trials63.csv')
    if len(rows)!=63 or len({r['trial'] for r in rows})!=63:raise ValueError('Expected 63 unique trials')
    correct=[]
    for r in rows:
        success=r['requested_material'].strip().lower()==r['actual_class'].strip().lower()
        if success!=boolean(r['correct']):raise ValueError(f"Saved perception label mismatch: {r['trial']}")
        correct.append(success)
    return dict(n=63,correct=sum(correct),accuracy=sum(correct)/63,
        original27=dict(n=27,correct=sum(c for r,c in zip(rows,correct) if r['dataset']=='original27')),
        revision36=dict(n=36,correct=sum(c for r,c in zip(rows,correct) if r['dataset']=='revision36')),
        policy='saved GSAM2+CLIP decisions: score>=0.10; no margin veto or color/texture rules')


def paths(root):
    result={}
    for version in ('v1','v2'):
        folder=root/'results/path_length'/version
        rows=read(folder/'selected_anchor_mask_strip_depth_vs_manual.csv')
        if len(rows)!=30 or len({(r['case_id'],r['path_label']) for r in rows})!=30:raise ValueError('Expected 30 unique paths')
        reported={r['group']:r for r in read(folder/'selected_anchor_mask_strip_depth_summary.csv')}
        result[version]={}
        for group in ('major','minor','all','timber_all','brick_all','concrete_block_all'):
            subset=[r for r in rows if group=='all' or r['path_label']==group or r['material'].replace(' ','_')+'_all'==group]
            result[version][group]={}
            for label,column,prefix in [('mask','mask_path_length_mm','mask'),('depth_point_cloud','strip_depth_path_length_mm','strip_depth')]:
                values=length_metrics(subset,column)
                for key in ('MAE_mm','RMSE_mm','MAPE_percent'):
                    old=float(reported[group][prefix+'_'+key])
                    if not math.isclose(values[key],old,rel_tol=1e-8,abs_tol=1e-8):
                        raise ValueError(f'Path summary mismatch {version}/{group}/{key}')
                result[version][group][label]=values
    return result


def runtime(root):
    rows=read(root/'results/runtime/master_rows.csv')
    eligible=[]
    for r in rows:
        if r.get('pipeline_success','').lower()!='true':continue
        try:
            total=float(r['total_pipeline_timing_ms']);execution=float(r['execution_timing_ms'])
            if math.isfinite(total) and math.isfinite(execution) and execution>0:eligible.append(r)
        except (ValueError,KeyError):pass
    fields=['total_pipeline_timing_ms','target_selection_timing_ms','anchor_selection_timing_ms',
            'geometry_timing_ms','path_length_timing_ms','robot_plan_timing_ms','execution_timing_ms']
    def stats(rr):
        result={}
        for field in fields:
            values=[]
            for r in rr:
                try:v=float(r[field])/1000
                except (ValueError,KeyError):continue
                if math.isfinite(v):values.append(v)
            result[field]=dict(n=len(values),mean_s=statistics.mean(values) if values else None,
                median_s=statistics.median(values) if values else None,min_s=min(values) if values else None,max_s=max(values) if values else None)
        return result
    # Imported readings are not automatically independent robot executions.
    identities={}
    for r in eligible:
        if r.get('imported_from_session'):
            key=(r['imported_from_session'],r.get('source_specimen_id'),r.get('source_reading_index'))
        else:key=(r.get('pipeline_session_path') or r['source_table'],r.get('reading_index') or r.get('cycle_index'))
        identities.setdefault(key,r)
    return dict(recorded_successful_rows=stats(eligible),source_identity_deduplicated=stats(list(identities.values())),
        warning='Rows can include imported readings. Identity deduplication is best-effort, not proof of independent trials. VLM stage is not Qwen32-only. Server load and external UPV export are not included; robot substeps are not separately timed.')


def reproduce(root,out):
    out.mkdir(parents=True,exist_ok=False)
    report={'contact73':contact(root,'contact73'),'contact82':contact(root,'contact82'),
            'perception':perception(root),'path_length':paths(root),'runtime':runtime(root)}
    for name,value in report.items():write(out/(name+'.json'),value)
    lines=['# Offline reproduction','', 'No model inference or hardware access.','']
    for dataset in ('contact73','contact82'):
        lines += [f'## {dataset}','| Method | N | Precision | Recall | Specificity | F1 | Good selection |','|---|---:|---:|---:|---:|---:|---:|']
        for model,r in report[dataset]['metrics'].items():
            lines.append(f"| {model} | {r['n']} | {100*r['precision']:.2f}% | {100*r['recall']:.2f}% | {100*r['specificity']:.2f}% | {100*r['F1']:.2f}% | {r['selected_good']}/{r['scenes']} |")
    p=report['perception'];lines+=['',f"Perception: {p['correct']}/{p['n']} ({100*p['accuracy']:.2f}%).",'',
        'Path summaries validated against saved per-path estimates (not rerun). Runtime: see runtime.json for raw-row and deduplicated views.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description='Offline-first UPV paper reproduction')
    parser.add_argument('--root',type=Path,default=Path.cwd(),help='Release repository root')
    sub=parser.add_subparsers(dest='command',required=True)
    rep=sub.add_parser('reproduce',help='Recompute metrics from committed saved predictions only')
    rep.add_argument('--output',type=Path)
    ds=sub.add_parser('install-data',help='Install a verified archive; no inference')
    ds.add_argument('dataset',choices=['contact','perception','path_length','runtime'])
    ds.add_argument('--archive',type=Path)
    inf=sub.add_parser('infer-contact',help='Explicit HTTP calls to an already-running model server')
    inf.add_argument('--dataset',choices=['contact73','contact82'],required=True)
    inf.add_argument('--model',required=True)
    inf.add_argument('--server-url',default='http://127.0.0.1:8899')
    inf.add_argument('--case-id')
    inf.add_argument('--output',type=Path,required=True)
    inf.add_argument('--run-inference',action='store_true')
    cv=sub.add_parser('baseline-rgb',help='CPU-only saved RGB/mask algorithm')
    cv.add_argument('--dataset',choices=['contact73','contact82'],required=True)
    cv.add_argument('--case-id')
    cv.add_argument('--output',type=Path,required=True)
    cv.add_argument('--run',action='store_true')
    rgbd=sub.add_parser('baseline-rgbd',help='CPU RGB-D baseline; contact73 reuses saved E3 as in original comparison')
    rgbd.add_argument('--dataset',choices=['contact73','contact82'],required=True)
    rgbd.add_argument('--output',type=Path,required=True)
    rgbd.add_argument('--run',action='store_true')
    raw=sub.add_parser('recompute-paths',help='Optional CPU raw-depth evaluation using archived selected anchors')
    raw.add_argument('--version',choices=['v1','v2'],default='v2')
    raw.add_argument('--case-id')
    raw.add_argument('--output',type=Path,required=True)
    raw.add_argument('--run',action='store_true')
    fig=sub.add_parser('figures',help='Plot saved contact classification summaries; no inference')
    fig.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='reproduce':
        reproduce(root,args.output or root/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    elif args.command=='install-data':print(install(root,args.dataset,args.archive))
    elif args.command=='infer-contact':
        from .experiments import infer_contact
        infer_contact(root,args)
    elif args.command=='baseline-rgb':
        from .experiments import baseline_rgb
        baseline_rgb(root,args)
    elif args.command=='recompute-paths':
        from .experiments import recompute_paths
        recompute_paths(root,args)
    elif args.command=='baseline-rgbd':
        from .experiments import baseline_rgbd
        baseline_rgbd(root,args)
    elif args.command=='figures':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        args.output.mkdir(parents=True,exist_ok=False)
        for dataset in ('contact73','contact82'):
            values=contact(root,dataset)['metrics'];fig,ax=plt.subplots(figsize=(10,5))
            ax.bar(list(values),[100*r['F1'] for r in values.values()]);ax.set_ylim(0,100)
            ax.set_ylabel('F1 (%)');ax.set_title(dataset);ax.tick_params(axis='x',labelrotation=20)
            fig.tight_layout();fig.savefig(args.output/(dataset+'_f1.png'),dpi=180);plt.close(fig)


if __name__=='__main__':main()
