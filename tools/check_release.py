"""Validate release artifacts and run metrics from a detached temporary copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from upv_vlm_contact.data import safe_member, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-archives', action='store_true')
    p.add_argument('--check-pipeline', action='store_true', help='Run detached synthetic CPU pipeline test; requires vision dependencies')
    args = p.parse_args()
    report = {'inference': False, 'hardware': False}
    excluded = {'.git', 'release_assets', 'datasets', 'runs', '__pycache__',
                '.cache', 'local_models', 'third_party', 'outputs', '.pytest_cache',
                'build', 'dist'}
    files = []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('.venv')
                   and not d.endswith('.egg-info')]
        for name in names:
            f = Path(directory)/name
            if name == '.env' or f.suffix in ('.pyc', '.pyo', '.pem', '.key'):
                continue
            if f.is_symlink():
                raise ValueError(f'Review release symlink: {f.relative_to(ROOT)}')
            files.append(f)
    large = [str(f.relative_to(ROOT)) for f in files if f.stat().st_size > 10*1024*1024]
    if large:
        raise ValueError(f'Move large files to archives: {large}')
    report['git_candidate_bytes'] = sum(f.stat().st_size for f in files)
    if report['git_candidate_bytes'] >= 100*1024*1024:
        raise ValueError('Git candidate size exceeds project 100 MiB target')
    report['git_candidate_files'] = len(files)
    secret = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for file in files:
        if file.suffix in ('.py','.json','.csv','.md','.txt','.yaml','.yml','.ino') and secret.search(file.read_bytes()):
            raise ValueError(f'Possible secret: {file.relative_to(ROOT)}; review before publication')
    if args.check_archives:
        catalog = json.loads((ROOT/'data/manifests/archives.json').read_text())
        report['archives'] = {}
        for name, spec in catalog.items():
            path = ROOT/'release_assets'/spec['file']
            if sha256(path) != spec['sha256']:
                raise ValueError(f'ZIP hash mismatch: {name}')
            with zipfile.ZipFile(path) as z:
                if any(not safe_member(s) for s in z.namelist()):
                    raise ValueError('Unsafe ZIP member')
                index = json.loads(z.read('index.json'))
                for file, sha in {(v['file'], v['sha256']) for v in index.values()}:
                    data = z.read(file)
                    if hashlib.sha256(data).hexdigest() != sha:
                        raise ValueError(f'Object hash mismatch: {file}')
                    if Path(file).suffix in ('.json','.csv','.md','.txt','.log','.yaml','.yml') and secret.search(data):
                        raise ValueError(f'Possible secret in archive {name}: {file}')
                report['archives'][name] = dict(sha256='passed', logical_files=len(index),
                    max_internal_path_chars=max(map(len, z.namelist())))
                if name == 'contact':
                    entries = json.loads((ROOT/'data/manifests/contact_inputs.json').read_text())
                    for entry in entries:
                        for field in ('image','mask','geometry','rgb'):
                            if entry[field].removeprefix('workspace/') not in index:
                                raise ValueError(f'Missing contact input {entry[field]}')
    with tempfile.TemporaryDirectory(prefix='upv_public_') as temp:
        copy = Path(temp)/'repo'
        for f in files:
            dest = copy/f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        env = {**os.environ, 'PYTHONPATH': str(copy/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
        subprocess.run([sys.executable, '-m', 'upv_vlm_contact', 'reproduce', '--output', str(copy/'runs/check')],
                       cwd=copy, env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=copy, env=env, check=True)
        if args.check_pipeline:
            subprocess.run([sys.executable, 'tools/check_pipeline_cpu.py'], cwd=copy, env=env, check=True)
            report['detached_cpu_pipeline'] = 'passed; saved synthetic mask and mocked HTTP, no neural inference'
    report['detached_copy_reproduction'] = 'passed'
    (ROOT/'docs/release_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
