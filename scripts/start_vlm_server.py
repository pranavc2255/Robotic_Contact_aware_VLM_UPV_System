"""Print a local server launch command; --run explicitly loads model weights."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['qwen25-32b', 'qwen25-3b', 'qwen3-2b'], required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--port', type=int, default=8899)
    p.add_argument('--load-4bit', action='store_true')
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = 'run_qwen3_anchor_server.py' if a.model == 'qwen3-2b' else 'run_qwen_anchor_server.py'
    command = [sys.executable, str(root/'reference/src/upv_vlm_v2/vlm'/script),
               '--model-path', str(a.model_path.resolve()), '--host', '127.0.0.1', '--port', str(a.port), '--device-map', 'auto']
    if a.load_4bit:
        command.append('--load-4bit')
    print(shlex.join(command), flush=True)
    if not a.run:
        print('Dry run: no model import, GPU access, download, or server startup.')
        return
    config = json.loads((a.model_path/'config.json').read_text())
    expected = 'qwen3_vl' if a.model == 'qwen3-2b' else 'qwen2_5_vl'
    if config.get('model_type') != expected:
        raise ValueError(f'Model type must be {expected}; verify exact model size/path yourself')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
