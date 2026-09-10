#!/usr/bin/env python3
"""Saved RGB-D, real perception/VLM, simulated hardware. No inference without --run."""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case-dir', type=Path, default=ROOT/'datasets/gpu_smoke/concrete')
    parser.add_argument('--material', choices=['brick', 'timber', 'concrete block'], default='concrete block')
    parser.add_argument('--axis', choices=['major', 'minor'], default='major')
    parser.add_argument('--worker', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker)
    for name in ('raw_rgb.png', 'raw_depth_aligned_z16.png', 'camera_info_aligned_depth.json'):
        if not (args.case_dir/name).is_file():
            parser.error(f'Missing input: {args.case_dir/name}')
    print('Existing localhost:8895 vLLM -> sleep -> saved RGB-D -> real DINO/SAM2/CLIP -> '
          'geometry -> wake -> real Qwen -> sleep -> path/plan -> SIMULATED hardware.')
    if not args.run:
        print('Validation only; no GPU imports, network, models or hardware.')
        return
    # Reuse the established simulation configuration builder, intercepting only
    # its pipeline subprocess to route through this isolated experimental worker.
    original_run = subprocess.run
    def launch(command, **kwargs):
        expected = 'upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline'
        if not isinstance(command, list) or expected not in command:
            raise RuntimeError('Unexpected command from simulation builder')
        import yaml
        config_path = Path(command[command.index('--config')+1])
        cfg = yaml.safe_load(config_path.read_text())
        cfg['managed_qwen']['enabled'] = False
        cfg['experimental_residency'] = 'vllm_sleep_wake_saved_rgbd_simulated_hardware'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return original_run([sys.executable, str(Path(__file__).resolve()), '--worker'] + command[3:], **kwargs)
    subprocess.run = launch
    sys.argv = ['run_prequantized_qwen_upv_cycle.py', '--gpu-simulation',
                '--case-dir', str(args.case_dir.resolve()), '--material', args.material,
                '--axis', args.axis, '--output-root', str(ROOT/'outputs/vllm_full_gpu_simulation')]
    try:
        runpy.run_path(str(ROOT/'scripts/run_prequantized_qwen_upv_cycle.py'), run_name='__main__')
    finally:
        subprocess.run = original_run


def worker(argv):
    import yaml
    config_path = Path(argv[argv.index('--config')+1])
    cfg = yaml.safe_load(config_path.read_text())
    if ('--live-ros2' in argv or cfg['execution']['backend'] != 'simulated'
            or cfg['execution']['allow_real_hardware'] or cfg['robot']['enabled']
            or cfg['camera']['allow_live_capture']):
        raise RuntimeError('Hardware safety check failed')
    out = config_path.parent
    record = {'hardware_simulated': True, 'server_startup_included': False,
              'pixel_cap': 200704, 'events': [], 'success': False}
    start = time.perf_counter()
    def request(path, payload=None):
        req = Request('http://127.0.0.1:8895'+path,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'},
                      method='GET' if payload is None else 'POST')
        with urlopen(req, timeout=300) as response:
            body = response.read()
        return json.loads(body) if body else {}
    def event(action):
        tick = time.perf_counter()
        request('/sleep?level=1' if action == 'sleep' else '/wake_up', {})
        state = request('/is_sleeping')
        if state.get('is_sleeping') is not (action == 'sleep'):
            raise RuntimeError(f'Unexpected residency state: {state}')
        snapshot = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)
        record['events'].append({'action': action, 'elapsed_ms': (time.perf_counter()-tick)*1000,
                                 'gpu_used_free_MiB': snapshot.stdout.strip(), 'state': state})
        print(f'vLLM {action}: {state}; GPU used/free MiB: {snapshot.stdout.strip()}', flush=True)
    try:
        models = request('/v1/models')
        if not any(m['id'] == 'contact-nf4' for m in models['data']):
            raise RuntimeError('Expected contact-nf4 server on port 8895')
        record['server_models'] = models
        event('sleep')
        import torch
        import upv_vlm_v2.pipeline.main_pipeline as pipeline
        import upv_vlm_v2.anchor_selection.qwen32_multi_anchor_l6_batch_ranker as ranker
        @contextmanager
        def residency(*unused, **kwargs):
            gc.collect()
            torch.cuda.empty_cache()
            event('wake')
            try:
                yield
            finally:
                event('sleep')
        def infer(**kw):
            content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
                        base64.b64encode(Path(p).read_bytes()).decode()}} for p in kw['image_paths']]
            content.append({'type': 'text', 'text': kw['user_prompt']})
            payload = {'model': 'contact-nf4', 'messages': [
                {'role': 'system', 'content': kw['system_prompt']}, {'role': 'user', 'content': content}],
                'temperature': kw.get('temperature', 0), 'seed': 12345,
                'max_tokens': kw.get('max_new_tokens', 1600),
                'mm_processor_kwargs': {'min_pixels': 3136, 'max_pixels': 200704}}
            (out/'vllm_request.json').write_text(json.dumps(payload))
            tick = time.perf_counter()
            response = request('/v1/chat/completions', payload)
            record['vlm_request_ms'] = (time.perf_counter()-tick)*1000
            (out/'vllm_response.json').write_text(json.dumps(response, indent=2))
            return {'success': True, 'ok': True, 'text': response['choices'][0]['message']['content'],
                    'elapsed_ms': record['vlm_request_ms'], 'num_images': len(kw['image_paths'])}
        pipeline.managed_qwen = residency
        ranker.infer_qwen_multi_anchor = infer
        sys.argv = ['run_full_main_upv_vlm_v2_pipeline'] + argv
        try:
            runpy.run_module('upv_vlm_v2.cli.run_full_main_upv_vlm_v2_pipeline', run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        record['success'] = True
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['total_simulation_wall_ms'] = (time.perf_counter()-start)*1000
        (out/'residency_timing.json').write_text(json.dumps(record, indent=2))
        print('Full cycle timings:', out/'residency_timing.json', flush=True)


if __name__ == '__main__':
    main()
