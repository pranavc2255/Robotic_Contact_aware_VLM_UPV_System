#!/usr/bin/env python3
"""Serve the local prequantized Qwen2.5-32B checkpoint without requantizing."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from http.server import HTTPServer

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8896)
    p.add_argument('--model-path', default=str(ROOT/'local_models/Qwen2.5-VL-32B-Instruct-bnb-4bit'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    model = Path(a.model_path).resolve()
    config = json.loads((model/'config.json').read_text())
    quant = config.get('quantization_config', {})
    if config.get('model_type') != 'qwen2_5_vl' or not quant.get('load_in_4bit'):
        p.error('Expected prequantized Qwen2.5-VL BitsAndBytes checkpoint')
    index = json.loads((model/'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (model/shard).is_file():
            p.error(f'Missing shard: {shard}')
    # Transformers reads the stored quantization configuration automatically.
    # Do not pass --load-4bit: that would supply an overriding configuration.
    command = [sys.executable, '-m', 'upv_vlm_v2.vlm.run_qwen_anchor_server',
        '--model-path', str(model), '--host', '127.0.0.1', '--port', str(a.port), '--device-map', 'auto']
    print('Embedded quantization:', json.dumps(quant), flush=True)
    # Older checkpoints used bare "merger" exclusions. Current Transformers
    # does not match that against model.visual.merger.mlp.0 or .2.
    skips = list(quant.get('llm_int8_skip_modules') or [])
    for prefix in ('visual.merger', 'model.visual.merger'):
        if prefix not in skips:
            skips.append(prefix)
    quant = dict(quant, llm_int8_skip_modules=skips)
    print('Effective BF16 exclusions:', skips, flush=True)
    print(f'Launcher endpoint: http://127.0.0.1:{a.port}/infer_multi', flush=True)
    if a.dry_run:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU loading')
    os.environ['HF_HUB_OFFLINE'] = '1'
    source = ROOT/'reference/src' if (ROOT/'reference/src/upv_vlm_v2').is_dir() else ROOT/'src'
    sys.path.insert(0, str(source))
    from upv_vlm_v2.vlm import run_qwen_anchor_server as server
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(str(model), local_files_only=True)
    cfg.quantization_config = quant
    server.PROCESSOR = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    server.MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model), config=cfg, device_map='auto', torch_dtype=torch.bfloat16,
        local_files_only=True)
    server.MODEL.eval()
    server.MODEL_PATH = str(model)
    server.DEVICE = 'cuda'
    from bitsandbytes.nn import Linear4bit
    missing = []
    for name, layer in server.MODEL.named_modules():
        if isinstance(layer, Linear4bit) and getattr(layer.weight, 'quant_state', None) is None:
            missing.append(dict(name=name, shape=list(layer.weight.shape), device=str(layer.weight.device)))
    print('4-bit layers without quant_state:', json.dumps(missing), flush=True)
    if missing:
        raise RuntimeError('Invalid 4-bit layers detected; refusing to serve: '+json.dumps(missing))
    print('Device map:', getattr(server.MODEL, 'hf_device_map', None), flush=True)
    original = server.run_multi_image_inference

    def diagnosed_inference(**kwargs):
        try:
            return original(**kwargs)
        except Exception as exc:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, flush=True)
            raise RuntimeError(f'{type(exc).__name__}: {exc}\n{detail}') from exc

    server.run_multi_image_inference = diagnosed_inference
    print(f'Ready: http://127.0.0.1:{a.port}/infer_multi', flush=True)
    HTTPServer(('127.0.0.1', a.port), server.QwenAnchorRequestHandler).serve_forever()

if __name__ == '__main__':
    main()
