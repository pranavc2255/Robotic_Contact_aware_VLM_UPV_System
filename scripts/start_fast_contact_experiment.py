#!/usr/bin/env python3
"""Isolated vLLM experiment. Prints command unless --run is explicitly supplied."""
import argparse
from importlib.metadata import distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def check_engine():
    """Inspect package files without importing vLLM, torch or loading weights."""
    try:
        package = distribution('vllm')
    except PackageNotFoundError as exc:
        raise RuntimeError('vLLM is not installed in this interpreter.') from exc
    implementation = Path(package.locate_file(
        'vllm/model_executor/layers/quantization/bitsandbytes.py'))
    if not implementation.is_file():
        raise RuntimeError(
            f'Installed vLLM {package.version} has no BitsAndBytes implementation. '
            'Use the isolated v0.19.1 environment described in '
            'docs/FAST_CONTACT_EXPERIMENT.md. Do not remove quantization flags '
            'or change the working Transformers environment.')
    print(f'vLLM {package.version}: BitsAndBytes implementation present; '
          'checkpoint/sleep compatibility still requires a runtime test.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--check-engine', action='store_true', help='Inspect installed engine without loading models')
    p.add_argument('--port', type=int, default=8895)
    p.add_argument('--model-path', type=Path, default=ROOT/'local_models/Qwen2.5-VL-32B-Instruct-bnb-4bit')
    p.add_argument('--eager', action='store_true', help='Diagnostic fallback if CUDA graph setup fails; distinct configuration')
    p.add_argument('--max-image-pixels', type=int, help='Explicit processor pixel cap; changes image preprocessing and is a distinct experiment')
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    a = p.parse_args()
    if a.max_model_len <= 0 or not 0 < a.gpu_memory_utilization < 1:
        p.error('Context length must be positive and GPU memory utilization between 0 and 1')
    if a.max_image_pixels is not None and a.max_image_pixels < 3136:
        p.error('--max-image-pixels must be at least 3136')
    if a.run or a.check_engine:
        try:
            check_engine()
        except RuntimeError as exc:
            p.error(str(exc))
        if not a.run:
            return
    model = a.model_path.resolve()
    cfg = json.loads((model/'config.json').read_text())
    if cfg.get('model_type') != 'qwen2_5_vl' or not cfg.get('quantization_config', {}).get('load_in_4bit'):
        p.error('Expected Qwen2.5-VL prequantized 4-bit checkpoint; no model fallback')
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
           '--served-model-name', 'contact-nf4', '--host', '127.0.0.1', '--port', str(a.port),
           '--quantization', 'bitsandbytes', '--load-format', 'bitsandbytes', '--dtype', 'bfloat16',
           '--max-model-len', str(a.max_model_len), '--max-num-seqs', '1',
           '--gpu-memory-utilization', str(a.gpu_memory_utilization),
           '--limit-mm-per-prompt', '{"image":5,"video":0}', '--enable-sleep-mode',
           '--no-enable-prefix-caching', '--mm-processor-cache-gb', '0', '--seed', '12345']
    if a.eager:
        cmd.append('--enforce-eager')
    if a.max_image_pixels is not None:
        cmd.extend(['--mm-processor-kwargs', json.dumps(
            {'min_pixels': 3136, 'max_pixels': a.max_image_pixels})])
    print(shlex.join(cmd), flush=True)
    print('Exact NF4 + vLLM compatibility is experimental. Stop other GPU servers first.')
    if not a.run:
        return
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    out = ROOT/'outputs/fast_contact_servers'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.update(VLLM_SERVER_DEV_MODE='1', HF_HOME=str(ROOT/'.cache/huggingface'), HF_HUB_OFFLINE='1')
    for key in ('HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'):
        env.pop(key,None)
    (out/'launch.json').write_text(json.dumps({'command': cmd, 'checkpoint_config': cfg,
        'boundary': 'Server lifecycle is not included in benchmark request latency'}, indent=2))
    with (out/'environment.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    print('Server log:', out/'server.log', flush=True)
    with (out/'server.log').open('w') as log:
        started=time.perf_counter();proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'ready':False,'command':cmd}
        try:
            while proc.poll() is None:
                if time.perf_counter()-started>1800:raise TimeoutError('vLLM startup exceeded 1800s')
                try:
                    with urlopen(f'http://127.0.0.1:{a.port}/v1/models',timeout=1) as response:
                        models=json.load(response)
                    if any(m['id']=='contact-nf4' for m in models['data']):break
                except (OSError,ValueError):pass
                time.sleep(0.2)
            if proc.poll() is not None:
                print('\n'.join((out/'server.log').read_text(errors='replace').splitlines()[-35:]), file=sys.stderr)
                raise RuntimeError('vLLM exited; inspect server.log. No checkpoint fallback was attempted.')
            record.update(ready=True,startup_ms=(time.perf_counter()-started)*1000)
            (out/'startup.json').write_text(json.dumps(record,indent=2))
            print('Ready; startup seconds:',record['startup_ms']/1000,flush=True)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            record['returncode']=proc.returncode
            (out/'startup.json').write_text(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
