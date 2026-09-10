"""Staged saved-RGB-D pipeline. No acquisition or robot modules are imported."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

from .experiments import load, parse_response

STAGES = ('perception', 'anchors', 'contact', 'paths')
REQUIRES = {'perception': (), 'anchors': ('perception',),
            'contact': ('anchors',), 'paths': ('perception', 'anchors', 'contact')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def convert(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def plan(stages, folder):
    if not stages or len(set(stages)) != len(stages):
        raise ValueError('Specify distinct stages in pipeline order')
    if list(stages) != sorted(stages, key=STAGES.index):
        raise ValueError('Stages must follow perception, anchors, contact, paths order')
    completed = set()
    for stage in stages:
        if (folder/stage).exists():
            raise FileExistsError(f'{folder/stage}: use a new run directory; stages are never overwritten')
        for dependency in REQUIRES[stage]:
            if dependency not in completed:
                path = folder/dependency/'result.json'
                if not path.is_file() or not read(path).get('success'):
                    raise ValueError(f'{stage} requires successful {dependency}')
        completed.add(stage)
    return list(stages)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Standalone repository root')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, help='JSON config; defaults to configs/pipeline.json')
    p.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    p.add_argument('--rgb', type=Path)
    p.add_argument('--depth', type=Path, help='Aligned Z16 .npy or uint16 PNG')
    p.add_argument('--camera-info', type=Path)
    p.add_argument('--material', choices=['brick', 'timber', 'concrete block'])
    p.add_argument('--mask', type=Path, help='Reuse a saved selected mask instead of running perception; provenance recorded')
    p.add_argument('--run', action='store_true', help='Execute planned stages; otherwise validate only')
    p.add_argument('--run-inference', action='store_true', help='Also authorize /infer_multi request')
    p.add_argument('--stage-timeout', type=float, default=1800)
    p.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
    a = p.parse_args()
    root, folder = a.root.resolve(), a.run_dir.resolve()
    if a.worker:
        try:
            worker(root, folder, a.worker)
        except Exception as exc:
            write(folder/a.worker/'result.json', {'success':False, 'status':'ERROR',
                  'error_type':type(exc).__name__, 'error':str(exc)})
            raise
        return
    plan(a.stages, folder)
    manifest_path = folder/'input.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if a.config or a.rgb or a.depth or a.camera_info or a.material or a.mask:
            p.error('A resumed run uses input.json and config.json; do not change inputs/config mid-run')
        for record in manifest['files'].values():
            if file_record(record['path']) != record:
                p.error('Input changed since preceding stage; use a new run directory')
        cfg = read(folder/'config.json')
    else:
        if not all([a.rgb, a.depth, a.camera_info, a.material]):
            p.error('New runs require --rgb --depth --camera-info --material')
        cfg = read(a.config or root/'configs/pipeline.json')
        if cfg.get('depth_scale_m_per_unit') != 0.001:
            p.error('This geometry adapter requires Z16 depth at 0.001 m/unit; convert inputs first')
        manifest = {'material': a.material, 'files': {k:file_record(v) for k,v in
                    [('rgb',a.rgb), ('depth',a.depth), ('camera_info',a.camera_info)]},
                    'robot_execution': False, 'acquisition': False}
        if a.mask:
            manifest['files']['mask'] = file_record(a.mask)
    if 'perception' in a.stages and 'mask' not in manifest['files']:
        perception = cfg['perception']
        for key in ('gsam2_repo_dir', 'sam2_checkpoint'):
            path = Path(perception[key])
            perception[key] = str((root/path).resolve()) if not path.is_absolute() else str(path)
        for path in [Path(perception['gsam2_repo_dir'])/'grounded_sam2_hf_model_demo.py', Path(perception['sam2_checkpoint'])]:
            if not path.is_file():
                p.error(f'Missing perception prerequisite: {path}; see docs/INSTALL.md')
    print(json.dumps({'stages':a.stages, 'run':a.run, 'run_dir':str(folder),
                      'input':manifest, 'hardware':False}, indent=2), flush=True)
    if not a.run:
        return
    if 'contact' in a.stages and not a.run_inference:
        p.error('Contact requires --run-inference; or run --stages perception anchors first')
    if a.stage_timeout <= 0:
        p.error('--stage-timeout must be positive')
    folder.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write(manifest_path, manifest)
        write(folder/'config.json', cfg)
    env = {**os.environ, 'PYTHONPATH':str(root/'src'), 'MPLBACKEND':'Agg'}
    for stage in a.stages:
        dest = folder/stage
        dest.mkdir()
        started = time.perf_counter()
        command = [sys.executable, '-m', 'upv_vlm_contact.pipeline', '--root', str(root),
                   '--run-dir', str(folder), '--worker', stage]
        print(f'Starting {stage}', flush=True)
        try:
            with (dest/'stage.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=a.stage_timeout)
        finally:
            write(dest/'timing.json', {'wall_time_ms':1000*(time.perf_counter()-started),
                  'includes_worker_startup':True, 'qwen_server_loading_included':False})
        result = read(dest/'result.json')
        print(f"Finished {stage}: {result.get('status', 'ok')}", flush=True)
        if not result.get('success'):
            print('Pipeline stopped without motion. Inspect result.json and stage.log.', flush=True)
            return


def worker(root, folder, stage):
    # Only the explicit worker imports scientific packages and bundled reference code.
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(root/'reference/src'))
    cfg = read(folder/'config.json')
    manifest = read(folder/'input.json')
    paths = {k:r['path'] for k,r in manifest['files'].items()}
    dest = folder/stage
    cfg['_runtime_requested_material'] = manifest['material']
    cfg['_runtime_capture'] = {'depth_path':paths['depth'], 'camera_info_path':paths['camera_info']}
    # Never permit a configuration to redirect this command into robot/VLM fallback paths.
    cfg.pop('testing', None)
    cfg['anchor_selection']['qwen_required'] = False
    cfg['execution'] = {'backend':'none', 'allow_real_hardware':False}
    rgb = Image.open(paths['rgb'])
    depth = np.load(paths['depth'], allow_pickle=False) if paths['depth'].endswith('.npy') else np.asarray(Image.open(paths['depth']))
    if depth.ndim != 2 or depth.shape != (rgb.height, rgb.width) or depth.dtype != np.uint16:
        raise ValueError('Depth must be uint16 and aligned to the RGB image dimensions')
    if stage == 'perception':
        if 'mask' in paths:
            import shutil
            mask = Image.open(paths['mask']).convert('L')
            if mask.size != rgb.size or not np.asarray(mask).any():
                raise ValueError('Saved mask is empty or has incompatible dimensions')
            shutil.copy2(paths['mask'], dest/'selected_mask.png')
            result = {'success':True, 'selected_mask_path':str(dest/'selected_mask.png'),
                      'selected_rgb_path':paths['rgb'], 'selection_source':'user_supplied_saved_mask'}
        else:
            from upv_vlm_v2.perception.target_selection_stage import run_target_selection
            result = run_target_selection(rgb_path=paths['rgb'], requested_material=manifest['material'], config=cfg, output_dir=dest)
    elif stage == 'anchors':
        from upv_vlm_v2.geometry.mask_geometry import compute_mask_geometry, add_dominant_rectangle_geometry
        from upv_vlm_v2.anchor_selection.anchor_selection_stage import run_anchor_selection
        from upv_vlm_v2.anchor_selection.qwen32_multi_anchor_l6_batch_ranker import prepare_l6_qwen_inputs
        from upv_vlm_v2.geometry.overlays import save_axis_overlay
        mask = read(folder/'perception/result.json')['selected_mask_path']
        geometry = compute_mask_geometry(mask, **cfg['_runtime_capture'])
        geometry = add_dominant_rectangle_geometry(geometry, mask, cfg)
        write(dest/'geometry.json', geometry)
        save_axis_overlay(rgb_path=paths['rgb'], mask_path=mask, geometry=geometry, output_path=dest/'geometry.png')
        payload = run_anchor_selection(mask_path=mask, rgb_path=paths['rgb'], geometry=geometry,
            axis_mode='major', config=cfg, output_dir=dest, **cfg['_runtime_capture'])
        prepare_l6_qwen_inputs(artifact_dir=dest, cfg=cfg['anchor_selection'])
        images = sorted((dest/'clean_single_anchor_inputs').glob('source_A*.png'), key=lambda x:int(x.stem[8:]))
        result = {'success':bool(images), 'images':{p.stem.removeprefix('source_'):str(p) for p in images},
                  'candidate_count':len(images), 'preparation':payload,
                  'note':'Deterministic preparation selection is not the VLM decision.'}
    elif stage == 'contact':
        from upv_vlm_v2.prompts.anchor_ranking_prompts import get_anchor_ranking_prompt_variant
        images = read(folder/'anchors/result.json')['images']
        system, user = get_anchor_ranking_prompt_variant('multi_image_v1_original', list(images))
        payload = dict(image_paths=list(images.values()), system_prompt=system, user_prompt=user,
                       max_new_tokens=1600, temperature=0, seed=12345)
        write(dest/'request.json', payload)
        request = Request(cfg['contact']['server_url'].rstrip('/')+'/infer_multi',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=cfg['contact']['timeout_seconds']) as response:
            raw = json.load(response)
        write(dest/'response.json', raw)
        parsed = parse_response(raw, list(images))
        write(dest/'parsed.json', parsed)
        eligible = [x for x in parsed['per_anchor_analysis'] if x['anchor_usable']]
        best = sorted(eligible, key=lambda x:(-x['anchor_score'],int(x['anchor_id'][1:])))[0] if eligible else None
        result = {'success':True, 'status':'SELECT' if best else 'NO_SAFE_ANCHOR',
                  'selected_anchor_id':best['anchor_id'] if best else None,
                  'policy':'Python highest-scoring model-labeled usable; lowest numeric ID breaks ties',
                  'model_label':cfg['contact']['model_label'], 'robot_motion_allowed':False}
    else:
        selected = read(folder/'contact/result.json')['selected_anchor_id']
        if selected is None:
            write(dest/'result.json', {'success':True, 'status':'SKIPPED_NO_SAFE_ANCHOR'})
            return
        evaluator = load(root/'reference/scripts/evaluate_new_capture_mask_and_depth_path_length.py', '_path_math')
        records = read(folder/'anchors/deterministic_anchor_features_wide_context.json')
        candidates = records['anchor_features']
        candidate = next(x for x in candidates if x['anchor_id']==selected)
        p1,p2 = np.asarray(candidate['contact_point_a_px'],float),np.asarray(candidate['contact_point_b_px'],float)
        p0=(p1+p2)/2
        direction=evaluator._normalize(p2-p1)
        intr=evaluator._load_intrinsics(Path(paths['camera_info']))
        depth_mm=depth.astype(float)*cfg['depth_scale_m_per_unit']*1000
        ref,_=evaluator._pick_ref_depth(depth_mm,p0,cfg['path']['center_region_radius_px'])
        if ref is None:
            write(dest/'result.json', {'success':False,'status':'missing_reference_depth'})
            return
        mask=np.asarray(Image.open(read(folder/'perception/result.json')['selected_mask_path']).convert('L'))>0
        # Rectangle is only a support ROI, never an alternate selected measurement axis.
        _,_,_,box,_=evaluator._rect_axes(mask.astype('uint8'))
        mask_length=evaluator._endpoint_length(p1,p2,ref,intr)
        pc=evaluator._strip_depth(depth_mm,p0,direction,box,intr,SimpleNamespace(**cfg['path']))
        result={'success':True,'selected_anchor_id':selected,'endpoint1_xy':p1,'endpoint2_xy':p2,
                'mask':mask_length,'depth_point_cloud':pc,'whole_roi_depth_used':False,
                'note':'New selected contact path; not a reproduction of manually measured path labels.'}
    write(dest/'result.json', result)


if __name__ == '__main__':
    main()
