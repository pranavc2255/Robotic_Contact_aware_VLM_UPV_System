"""Maintainer-only import of existing research artifacts. Never runs experiments.

The resulting repository does not need this importer or the source workspace.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

DEST = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if source == DEST or DEST not in source.parents and DEST.parent != source:
        if not (source/'src/upv_vlm_v2').exists():
            raise ValueError('Not a research workspace')
    provenance = []
    original_hashes = {}
    original_tracked = subprocess.check_output(['git','ls-files','-z'],cwd=source).decode().split('\0')
    for name in original_tracked:
        if name and (source/name).is_file():
            original_hashes[name]=sha((source/name).read_bytes())

    def clean(text):
        text = text.replace(str(source)+'/', 'workspace/').replace(str(source), 'workspace')
        text = re.sub(r'/home/[^/\s"\']+/', 'user_home/', text)
        text = re.sub(r'192\.168\.\d+\.\d+', 'ROBOT_HOST_NOT_CONFIGURED', text)
        return text

    def put(path, data):
        path=DEST/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)

    def jsonout(path,obj):
        put(path,(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode())

    def csvout(path,rows):
        if not rows:return
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        put(path,buf.getvalue().encode())

    def copy(src,relative):
        raw=src.read_bytes()
        data=clean(raw.decode()).encode() if src.suffix.lower() in ('.py','.txt','.md','.yaml','.yml','.json','.csv','.ino') else raw
        put(relative,data)
        provenance.append(dict(source=str(src.relative_to(source)),destination=str(relative),
                               source_sha256=sha(raw),release_sha256=sha(data),text_redacted=raw!=data))

    # Keep legacy modules for scientific source traceability, outside installed package.
    for package in ('upv_vlm_v2','upv_vlm_v1','upv_vlm_orient'):
        for src in sorted((source/'src'/package).rglob('*.py')):
            copy(src,Path('reference/src')/src.relative_to(source/'src'))
    scripts=['evaluate_contact_crop_rgb_mask_pilot.py','evaluate_projected_topview_rgbd_contact_baseline.py',
             'evaluate_new_capture_mask_and_depth_path_length.py','recreate_new_capture_depth_visualizations.py',
             'package_final_selected_anchor_path_length_results.py','create_step_by_step_selected_anchor_visualizations.py',
             'run_revision_36_perception_trials.py','rerun_revision_clip_original27.py','label_clip_no_margin_63.py',
             'compare_clip_only_same_prompts_63.py','capture_revision_12_scenes.py',
             'select_manual_path_anchors_for_new_capture.py']
    for name in scripts:
        copy(source/'scripts'/name,Path('reference/scripts')/name)
    for src in sorted((source/'firmware/T4_ClampController').glob('*')):
        if src.is_file():copy(src,Path('hardware/arduino/T4_ClampController')/src.name)
    for directory in ('execution','hardware','robot_planning'):
        for src in sorted((source/'src/upv_vlm_v2'/directory).glob('*.py')):
            copy(src,Path('hardware/robot/reference')/directory/(src.name+'.txt'))
    for name in ['deploy_ur3e_realsense_upv_real_plan_check_qwen32_multi_anchor_l6.yaml']:
        copy(source/'configs/v2'/name,Path('reference/configs')/(name+'.txt'))
    for rel in ['third_party/Grounded-SAM-2/LICENSE','third_party/Grounded-SAM-2/LICENSE_groundingdino']:
        if (source/rel).exists():copy(source/rel,Path('docs/third_party')/Path(rel).name)
    versions={}
    for name in ['numpy','opencv-python','pillow','matplotlib','scipy','pandas','torch','transformers','accelerate','bitsandbytes','qwen-vl-utils','PyYAML']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed in build interpreter'
    jsonout('configs/build_environment.json',versions)

    assets={name:{} for name in ('contact','perception','path_length','runtime')}
    def asset(category,src,logical=None):
        src=Path(src)
        if not src.is_file():raise FileNotFoundError(src)
        if src.suffix.lower() in ('.pt','.safetensors','.bin','.pyc','.zip'):return
        logical=logical or str(src.relative_to(source))
        assets[category][logical]=src

    def tree(category,folder):
        for f in sorted(folder.rglob('*')):
            if f.is_file() and not any('backup' in p.lower() or '__pycache__' in p for p in f.parts) and f.suffix not in ('.py','.pyc','.zip'):
                asset(category,f)

    base=source/'outputs/debug_anchor_prompt_eval'
    runs={'Qwen2.5-32B':'qwen25_32b_multi_image_L6_E3_E45_artifacted_20260529_005107_full',
          'Qwen3-2B':'qwen3_vl_2b_multi_image_L6_E3_E45_seed12345_20260828_151120_full',
          'Qwen2.5-3B':'qwen25_3b_matched_L6_seed12345_20260908_003608_full'}
    study=source/'outputs/16_case_qwen_study_all_data'
    comparison=source/'outputs/contact_73_comparison_20260908'
    rows73=read_csv(comparison/'all_predictions.csv')
    rgbd73=source/'outputs/rgbd_73_comparison_20260908'
    for r in read_csv(rgbd73/'per_anchor_results.csv'):
        rows73.append(dict(model='Classical RGB-D',split=r['dataset_split'],case=r['benchmark_group_id'].split('_',1)[1],
            anchor_id=r['anchor_id'],manual_good=r['manual_is_usable'],predicted_good=r['predicted_usable'],
            score=r['pair_safety_score'],source=str(rgbd73.relative_to(source)),error=''))
    fields=['model','split','case','anchor_id','manual_good','predicted_good','score']
    csvout('results/contact/contact73/predictions.csv',[{k:r[k] for k in fields} for r in rows73])
    for f in [comparison/'metrics.csv',comparison/'selections.csv',rgbd73/'metrics.csv',rgbd73/'per_group_results.csv']:
        copy(f,Path('results/contact/contact73/reference')/(f.parent.name+'_'+f.name))
    copy(study/'filtered_16_case_metrics.json',Path('results/contact/contact73/reference/qwen32_original.json'))
    copy(study/'exclusions/excluded_cases.csv',Path('data/labels/contact73_exclusions.csv'))
    copy(study/'filtered_16_case_per_anchor_predictions.csv',Path('data/labels/contact73_source.csv'))
    tree('contact',comparison);tree('contact',rgbd73)
    for folder in ['original_E3_9_cases','new10_7_cases']:
        tree('contact',study/folder)
    benchmark=source/'data/contact_anchor_selection_benchmark_E3_E45_82'
    tree('contact',benchmark)
    copy(benchmark/'labels_all.csv',Path('data/labels/contact82_source.csv'))
    oldcv=source/'outputs/contact_crop_rgb_mask_baseline/20260908_002405'
    oldd=source/'outputs/projected_topview_rgbd_contact_baseline/final_full82_3mm_20260828'
    tree('contact',oldcv);tree('contact',oldd)
    rows82=[]
    for model,run in runs.items():
        for r in read_csv(base/run/'per_anchor_predictions.csv'):
            rows82.append(dict(model=model,split=r['dataset_split'],case=r['case_or_session_id'],anchor_id=r['anchor_id'],
                manual_good=r['manual_is_usable'],predicted_good=r['anchor_usable'],score=r['anchor_score']))
        for sub in ['raw_responses','parsed_responses','prompts','debug_payloads']:
            tree('contact',base/run/sub)
            for f in sorted((base/run/sub).glob('*')):
                if f.is_file() and f.suffix in ('.json','.txt'):
                    copy(f,Path('results/contact/contact82/responses')/model/sub/f.name)
        for name in ['summary_overall.json','summary_by_group.csv','server_health.json','run_config.json']:
            if (base/run/name).exists():copy(base/run/name,Path('results/contact/contact82/reference')/model/name)
    for model,folder in [('Classical RGB+mask',oldcv),('Classical RGB-D',oldd)]:
        fname='predictions.csv' if model.endswith('mask') else 'per_anchor_results.csv'
        for r in read_csv(folder/fname):
            case=r.get('scene') or r['benchmark_group_id'].split('_',1)[1]
            rows82.append(dict(model=model,split='E3' if case.startswith('case_') else 'E45',case=case,anchor_id=r['anchor_id'],
                manual_good=r.get('manual_good',r.get('manual_is_usable')),predicted_good=r.get('predicted_good',r.get('predicted_usable')),
                score=r.get('risk_score',r.get('pair_safety_score'))))
        copy(folder/'metrics.csv',Path('results/contact/contact82/reference')/model/'metrics.csv')
    csvout('results/contact/contact82/predictions.csv',rows82)
    newrun=base/'qwen25_32b_multi_image_new10_orientation_fixed_core_support_seed12345_20260626_012116_full'
    tree('contact',newrun)
    for model,folder in [('Qwen2.5-32B',newrun),('Qwen3-2B',comparison/'Qwen3-2B'),('Qwen2.5-3B',comparison/'Qwen2.5-3B')]:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.txt'):
                copy(f,Path('results/contact/contact73/responses')/model/f.relative_to(folder))

    # Collect exact inputs needed by CPU baselines; no segmentation is rerun.
    contact_inputs=[]
    datasetrows=read_csv(benchmark/'dataset_manifest.csv')
    layoutmap={r['benchmark_id']:r['layout_image_path'] for r in read_csv(benchmark/'layouts/L6_magenta_line_only_fixed/layout_manifest.csv')}
    def resolve(value):
        p=Path(value);return p if p.is_absolute() else source/p
    for r in datasetrows:
        meta=resolve(r['metadata_json_path']);m=json.loads(meta.read_text());asset('contact',meta)
        folder=resolve(m['support_case_dir']) if m.get('support_case_dir') else resolve(m['source_session'])/'artifacts/04_anchor_selection'
        gfile=folder/'anchor_crop_geometry_debug.json';g=json.loads(gfile.read_text())['anchors'][r['anchor_id']]
        rgb=resolve(g['rotated_image_path']);mask=resolve(g.get('rotated_mask_path') or str(rgb.with_name('rotated_object_mask.png')))
        for f in [gfile,rgb,mask,resolve(layoutmap[r['benchmark_id']])]:asset('contact',f)
        contact_inputs.append(dict(dataset='contact82',split=r['dataset_split'],case=r['case_or_session_id'],anchor_id=r['anchor_id'],
            image=layoutmap[r['benchmark_id']],geometry=str(gfile.relative_to(source)),rgb=str(rgb.relative_to(source)),mask=str(mask.relative_to(source))))
    retained={(r['split'],r['case'],r['anchor_id']) for r in rows73}
    contact_inputs += [{**r,'dataset':'contact73'} for r in contact_inputs if (r['split'],r['case'],r['anchor_id']) in retained]
    prep=source/'outputs/v2_datasets/manual_10_new_anchor_prep/session_20260626_005117'
    for r in read_csv(prep/'new10_manual_anchor_labels.csv'):
        if ('new10',r['case_or_session_id'],r['anchor_id']) not in retained:continue
        folder=resolve(r['pipeline_session_path'])/'artifacts/04_anchor_selection'
        gfile=folder/'anchor_crop_geometry_debug.json';g=json.loads(gfile.read_text())['anchors'][r['anchor_id']]
        rgb=resolve(g['rotated_image_path']);mask=resolve(g.get('rotated_mask_path') or str(rgb.with_name('rotated_object_mask.png')))
        for f in [gfile,rgb,mask,resolve(r['anchor_image_path'])]:asset('contact',f)
        contact_inputs.append(dict(dataset='contact73',split='new10',case=r['case_or_session_id'],anchor_id=r['anchor_id'],
            image=r['anchor_image_path'],geometry=str(gfile.relative_to(source)),rgb=str(rgb.relative_to(source)),mask=str(mask.relative_to(source))))
    jsonout('data/manifests/contact_inputs.json',contact_inputs)
    for lineage in [source/'outputs/genesis_82_pair_benchmark_per_case_lineage.csv',rgbd73/'lineage.csv']:
        asset('contact',lineage)
        for r in read_csv(lineage):
            for field in ['original_raw_rgb_path','original_raw_depth_path','original_camera_info_path','original_selected_mask_path']:
                asset('contact',resolve(r[field]))
            asset('contact',resolve(r['original_anchor_selection_folder'])/'deterministic_anchor_features_wide_context.json')

    perception=source/'outputs/revision_36_clip_no_margin_90_48_windows_compact'
    for r in read_csv(perception/'FILE_MAP.csv'):
        asset('perception',perception/r['new_path'],r['old_path'])
    copy(perception/'final63/f000013.csv',Path('results/perception/trials63.csv'))
    copy(perception/'README.md',Path('results/perception/source_readme.md'))

    capture=source/'outputs/final_path_results_v1_capture_session_20260704_222852'
    tree('path_length',capture/'cases');tree('path_length',capture/'gsam2_masks');tree('path_length',capture/'manual_anchor_selection')
    for name in ['session_manifest.csv','session_manifest.json']:
        asset('path_length',capture/name)
    paths={
        'v1':source/'outputs/final_path_results_v1_mask_depth_eval/session_20260704_230654',
        'v2':source/'outputs/final_path_results_v2_percentile_endpoint_mask_depth_eval/session_20260705_135701'}
    for name,run in paths.items():
        tree('path_length',run)
        for f in sorted(run.glob('*')):
            if f.is_file() and f.suffix in ('.csv','.json','.md'):
                copy(f,Path('results/path_length')/name/f.name)
    for paper in ['final_path_results_v1_paper_ready_20260704_230703','final_path_results_v2_percentile_endpoint_paper_ready_20260705_135715']:
        tree('path_length',source/'outputs'/paper)
    jsonout('data/manifests/path_length.json',dict(capture=str(capture.relative_to(source)),evaluations={k:str(v.relative_to(source)) for k,v in paths.items()}))

    runtime=source/'outputs/v2_experiments/e45_robot_upv_repeatability'
    runtime_rows=[]
    for f in sorted(runtime.rglob('*master.csv')):
        if f.name not in ('e45_all_cycles_master.csv','e45_all_readings_master.csv'):continue
        asset('runtime',f)
        for i,r in enumerate(read_csv(f)):
            runtime_rows.append(dict(source_table=str(f.relative_to(source)),source_row=i+2,**r))
    fieldnames=list(dict.fromkeys(k for r in runtime_rows for k in r))
    csvout('results/runtime/master_rows.csv',[{k:clean(str(r.get(k,''))) for k in fieldnames} for r in runtime_rows])
    for f in sorted(runtime.rglob('*.json')):
        if f.name in ('timing_summary.json','pipeline_result.json','execution_summary.json','real_clamp_execution_log.json','reading_summary.json'):
            asset('runtime',f)
    docs=['final_pipeline_runtime_summary_model_residency_audit_20260706_011350.md',
          'runtime_substage_split_audit_20260709.md','main_pipeline_math_audit_20260705.md',
          'final_path_results_v2_percentile_endpoint_audit_20260705.md']
    for name in docs:
        if (source/'docs/dev_history'/name).exists():copy(source/'docs/dev_history'/name,Path('docs/source_audits')/name)

    # Short content-addressed entries deduplicate repeated experimental copies.
    archive_catalog={}
    for category,members in assets.items():
        archive=DEST/'release_assets'/f'{category}.zip';archive.parent.mkdir(exist_ok=True)
        index={};written=set()
        print(f'Packaging {category}: {len(members)} logical files',flush=True)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for logical,f in sorted(members.items()):
                raw=f.read_bytes();data=raw
                if f.suffix.lower() in ('.json','.csv','.md','.txt','.log','.yaml','.yml'):
                    data=clean(raw.decode(errors='replace')).encode()
                h=sha(data);name=f'objects/{h}{f.suffix.lower()}'
                if name not in written:z.writestr(name,data);written.add(name)
                index[logical]=dict(file=name,sha256=h,size_bytes=len(data),source_sha256=sha(raw),redacted=data!=raw)
            z.writestr('index.json',json.dumps(index,indent=2))
        with zipfile.ZipFile(archive) as z:
            if z.testzip():raise ValueError(f'CRC failure: {archive}')
        archive_catalog[category]=dict(file=archive.name,sha256=sha(archive.read_bytes()),
            size_bytes=archive.stat().st_size,logical_files=len(index),unique_objects=len(written),url=None)
    jsonout('data/manifests/archives.json',archive_catalog)
    jsonout('data/manifests/source_provenance.json',provenance)
    jsonout('data/manifests/source_state.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        uncommitted_work_included=True))
    changed=[name for name,h in original_hashes.items() if not (source/name).exists() or sha((source/name).read_bytes())!=h]
    if changed:raise RuntimeError(f'Parent tracked files changed: {changed}')
    jsonout('docs/assembly_validation.json',dict(parent_tracked_files_checked=len(original_hashes),parent_tracked_files_changed=changed,
        model_inference=False,hardware_execution=False,archive_crc='passed'))
    print('Assembly complete; parent tracked hashes unchanged.',flush=True)


if __name__=='__main__':main()
