"""Optional saved-input experiments; imports and HTTP are explicit, never automatic."""
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

from .data import Assets
from .metrics import boolean, classify, select_usable


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(root, args):
    rows = json.loads((root/'data/manifests/contact_inputs.json').read_text())
    rows = [r for r in rows if r['dataset'] == args.dataset and
            (not args.case_id or r['case'] == args.case_id)]
    if not rows:
        raise ValueError('No exact dataset/case match')
    assets = Assets(root, 'contact')
    for row in rows:
        for field in ('image', 'geometry', 'rgb', 'mask'):
            if not assets.path(row[field]).is_file():
                raise FileNotFoundError(row[field])
    return assets, rows


def labels(root, dataset):
    with (root/'results/contact'/dataset/'predictions.csv').open() as f:
        return {(r['split'], r['case'], r['anchor_id']): r['manual_good']
                for r in csv.DictReader(f)}


def save(out, name, value):
    (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def selection_summary(expected, predictions, minimum_risk=False, center_tie=False):
    groups = defaultdict(list)
    for row in expected:
        groups[row['split'], row['case']].append(row)
    selected = []
    for (split, case), group in groups.items():
        rows = [r for r in predictions if (r['split'], r['case']) == (split, case)]
        complete = len(rows) == len(group)
        best = select_usable(rows, minimum_risk=minimum_risk) if complete else None
        if complete and center_tie:
            eligible = [r for r in rows if boolean(r['predicted_good'])]
            best = max(eligible, key=lambda r: (float(r['score']), -abs(int(r['anchor_id'][1:])-(len(group)+1)/2))) if eligible else None
        selected.append(dict(split=split, case=case, complete=complete,
            action='INPUT_FAILURE' if not complete else ('SELECT' if best else 'NO_SAFE_ANCHOR'),
            anchor_id=best['anchor_id'] if best else None,
            selected_good=boolean(best['manual_good']) if best else None))
    return dict(scenes=len(groups), complete_scenes=sum(r['complete'] for r in selected),
        selected_good=sum(r['selected_good'] is True for r in selected), selections=selected,
        policy='Python best eligible score, not model final-action field')


def baseline_rgb(root, args):
    assets, rows = inputs(root, args)
    print(f'Validated {len(rows)} RGB/mask pairs; CPU only; run={args.run}')
    if not args.run:
        return
    module = load(root/'reference/scripts/evaluate_contact_crop_rgb_mask_pilot.py', '_rgb_reference')
    truth = labels(root, args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions, failures = [], []
    parameters = SimpleNamespace(defect_fraction=.03, max_defect_run_fraction=.10)
    for row in rows:
        try:
            geometry = assets.json(row['geometry'])['anchors'][row['anchor_id']]
            module.geometry = lambda unused: (geometry, assets.path(row['rgb']), assets.path(row['mask']))
            panels, _ = module.extract(row, assets.path(row['image']))
            sides = [module.analyse(rgb, mask, excluded, y, parameters)[0]
                     for _, rgb, mask, excluded, y, _ in panels]
            predictions.append(dict(split=row['split'], case=row['case'], anchor_id=row['anchor_id'],
                manual_good=truth[row['split'], row['case'], row['anchor_id']],
                predicted_good=all(s['good'] for s in sides), score=max(s['score'] for s in sides), sides=sides))
        except (ValueError, KeyError, OSError) as e:
            failures.append(dict(case=row['case'], anchor_id=row['anchor_id'], error=str(e)))
    save(args.output, 'predictions.json', predictions)
    save(args.output, 'failures.json', failures)
    save(args.output, 'selections.json', selection_summary(rows, predictions, minimum_risk=True))
    save(args.output, 'metrics.json', dict(n_requested=len(rows), n_failed=len(failures), metrics=classify(predictions)))
    print(f'Processed={len(predictions)} failures={len(failures)} output={args.output}')


def parse_response(response, ids):
    if response.get('ok') is False:
        raise ValueError(str(response.get('error')))
    text = response.get('response_text') or response.get('text') or response.get('response')
    if isinstance(text, dict):
        result = text
    else:
        text = str(text).strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        result = json.loads(text)
    items = result['per_anchor_analysis']
    if len(items) != len(ids) or {x['anchor_id'] for x in items} != set(ids):
        raise ValueError('Missing, extra or duplicate anchor IDs')
    for item in items:
        if type(item['anchor_usable']) is not bool:
            raise ValueError('anchor_usable must be a JSON boolean')
        score = item['anchor_score']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('anchor_score must be finite and in [0,100]')
    return result


def infer_contact(root, args):
    assets, rows = inputs(root, args)
    prompts = load(root/'reference/src/upv_vlm_v2/prompts/anchor_ranking_prompts.py', '_contact_prompts')
    groups = defaultdict(list)
    for row in rows:
        groups[row['split'], row['case']].append(row)
    print(f'Validated {len(rows)} images; {len(groups)} /infer_multi requests; run={args.run_inference}')
    if not args.run_inference:
        return
    # This API passes file paths, not image bytes: server must share this filesystem.
    args.output.mkdir(parents=True, exist_ok=False)
    truth = labels(root, args.dataset)
    predictions, failures = [], []
    for index, ((split, case), group) in enumerate(groups.items(), 1):
        group.sort(key=lambda r: int(r['anchor_id'][1:]))
        ids = [r['anchor_id'] for r in group]
        system, user = prompts.get_anchor_ranking_prompt_variant('multi_image_v1_original', ids)
        payload = dict(image_paths=[str(assets.path(r['image']).resolve()) for r in group],
            system_prompt=system, user_prompt=user, max_new_tokens=1600, temperature=0, seed=12345)
        save(args.output, f'{index:02d}_request.json', payload)
        print(f'[{index}/{len(groups)}] {case}', flush=True)
        start = time.perf_counter()
        try:
            request = Request(args.server_url.rstrip('/')+'/infer_multi', data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=1800) as response:
                raw = json.load(response)
            save(args.output, f'{index:02d}_response.json', raw)
            parsed = parse_response(raw, ids)
            save(args.output, f'{index:02d}_parsed.json', parsed)
            for item in parsed['per_anchor_analysis']:
                predictions.append(dict(model=args.model, split=split, case=case, anchor_id=item['anchor_id'],
                    manual_good=truth[split, case, item['anchor_id']], predicted_good=item['anchor_usable'], score=item['anchor_score']))
        except (OSError, ValueError, KeyError, TypeError) as e:
            failures.append(dict(case=case, error=str(e), n_pairs=len(group)))
        finally:
            save(args.output, f'{index:02d}_timing.json', dict(request_elapsed_ms=1000*(time.perf_counter()-start),
                 model_label_user_supplied=args.model, model_loading_included=False, robot_execution=False))
    save(args.output, 'predictions.json', predictions)
    save(args.output, 'failures.json', failures)
    save(args.output, 'selections.json', selection_summary(rows, predictions))
    save(args.output, 'metrics.json', dict(n_requested=len(rows), n_failed=sum(r['n_pairs'] for r in failures),
                                        metrics=classify(predictions)))


def recompute_paths(root, args):
    """Materialize only the saved capture; never change installed objects or originals."""
    assets = Assets(root, 'path_length')
    source = json.loads((root/'data/manifests/path_length.json').read_text())['capture']
    prefix = source.rstrip('/')+'/'
    members = {name: value for name, value in assets.index.items() if name.startswith(prefix)}
    selected = [name for name in members if name.endswith('/selected_anchors.json')]
    if len(selected) != 15:
        raise ValueError(f'Expected 15 selected-anchor files, found {len(selected)}')
    if args.case_id and not any('/cases/'+args.case_id+'/' in name for name in selected):
        raise ValueError('Requested exact selected case does not exist')
    print(f'Validated 15 selected-anchor records; version={args.version}; run={args.run}')
    if not args.run:
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    capture = args.output/'input_capture'
    for name in members:
        dest = capture/name[len(prefix):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets.path(name), dest)
    # The evaluator resolves case input files by exact local folder, not stale
    # embedded source paths. Parameters are read from the corresponding saved run.
    module = load(root/'reference/scripts/evaluate_new_capture_mask_and_depth_path_length.py', '_path_reference')
    config = json.loads((root/'results/path_length'/args.version/'run_config.json').read_text())['parameters']
    config.setdefault('depth_endpoint_mode', 'current')
    config.setdefault('depth_lower_quantile', None)
    config.setdefault('depth_upper_quantile', None)
    config.update(capture_session=str(capture), mask_run=str(capture/'gsam2_masks'),
        manual_anchor_selection_root=str(capture/'manual_anchor_selection'), output_root=str(args.output/'evaluation'),
        case_id=args.case_id, max_cases=None, skip_figures=True, no_svg=True, debug_progress=True,
        allow_axis_fallback_error_metrics=False)
    module.REPO_ROOT = root
    result = module.run(SimpleNamespace(**config))
    print(result)


def baseline_rgbd(root, args):
    assets = Assets(root, 'contact')
    lineage = ('outputs/rgbd_73_comparison_20260908/lineage.csv' if args.dataset == 'contact73'
               else 'outputs/genesis_82_pair_benchmark_per_case_lineage.csv')
    label_file = ('outputs/rgbd_73_comparison_20260908/labels.csv' if args.dataset == 'contact73'
                  else 'data/contact_anchor_selection_benchmark_E3_E45_82/labels_all.csv')
    with assets.path(lineage).open() as f:
        groups = list(csv.DictReader(f))
    assets.path(label_file)
    for row in groups:
        for key in ('original_raw_rgb_path','original_raw_depth_path','original_camera_info_path','original_selected_mask_path'):
            assets.path(row[key])
        assets.path(row['original_anchor_selection_folder']+'/deterministic_anchor_features_wide_context.json')
    print(f'Validated {len(groups)} raw RGB-D groups; run={args.run}')
    if not args.run:
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    directories = {}
    for i, row in enumerate(groups):
        logical = row['original_anchor_selection_folder'].removeprefix('workspace/')
        folder = args.output/'input_features'/str(i)
        folder.mkdir(parents=True)
        shutil.copy2(assets.path(logical+'/deterministic_anchor_features_wide_context.json'),
                     folder/'deterministic_anchor_features_wide_context.json')
        directories[logical] = folder
    module = load(root/'reference/scripts/evaluate_projected_topview_rgbd_contact_baseline.py', '_rgbd_reference')
    def resolve(path):
        path = Path(path)
        if path.is_absolute():
            return path
        logical = str(path).removeprefix('workspace/')
        return directories[logical] if logical in directories else assets.path(logical)
    module._resolve = resolve
    old_argv = sys.argv
    try:
        sys.argv = ['rgbd', '--lineage-csv', lineage, '--labels-csv', label_file,
                    '--output-dir', str(args.output/'new_evaluation'), '--no-visuals']
        code = module.main()
        if code not in (None, 0):
            raise RuntimeError(f'RGB-D reference returned {code}; inspect new_evaluation')
    finally:
        sys.argv = old_argv
    with (args.output/'new_evaluation/per_anchor_results.csv').open() as f:
        fresh = list(csv.DictReader(f))
    predictions = [dict(split=r['dataset_split'], case=r['benchmark_group_id'].split('_', 1)[1],
        anchor_id=r['anchor_id'], manual_good=r['manual_is_usable'], predicted_good=r['predicted_usable'],
        score=float(r['pair_safety_score']), reused=False) for r in fresh]
    if args.dataset == 'contact73':
        with (root/'results/contact/contact73/predictions.csv').open() as f:
            predictions += [{**r, 'reused': True} for r in csv.DictReader(f)
                            if r['model']=='Classical RGB-D' and r['split']=='E3']
    expected = [r for r in json.loads((root/'data/manifests/contact_inputs.json').read_text()) if r['dataset']==args.dataset]
    save(args.output, 'predictions.json', predictions)
    save(args.output, 'metrics.json', dict(n_requested=len(expected), n_failed=len(expected)-len(predictions), metrics=classify(predictions)))
    save(args.output, 'selections.json', selection_summary(expected, predictions, center_tie=True))
