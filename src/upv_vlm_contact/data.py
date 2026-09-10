"""Checksummed archives with short, content-addressed file names."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name):
    parts = name.replace('\\', '/').split('/')
    return not name.startswith(('/', '\\')) and not any(p in ('', '.', '..') or ':' in p for p in parts)


def install(root, name, archive=None):
    catalog = json.loads((root/'data/manifests/archives.json').read_text())
    spec = catalog[name]
    with tempfile.TemporaryDirectory() as temp:
        if archive is None:
            url = spec.get('url')
            if not url:
                raise ValueError('Public URL not configured yet. Supply --archive with the local release ZIP.')
            if not url.startswith('https://'):
                raise ValueError('Dataset URL must use HTTPS')
            archive = Path(temp)/'download.zip'
            with urlopen(url, timeout=60) as response, archive.open('wb') as f:
                shutil.copyfileobj(response, f)
        archive = Path(archive)
        if sha256(archive) != spec['sha256']:
            raise ValueError('Archive checksum mismatch; nothing installed')
        destination = root/'datasets'/name
        if destination.exists():
            raise FileExistsError(f'{destination} already exists; will not overwrite')
        stage = Path(temp)/'unpacked'
        with zipfile.ZipFile(archive) as z:
            if any(not safe_member(i.filename) or (i.external_attr >> 16) & 0o170000 == 0o120000 for i in z.infolist()):
                raise ValueError('Unsafe archive entry')
            z.extractall(stage)
        index = json.loads((stage/'index.json').read_text())
        for entry in index.values():
            if not safe_member(entry['file']) or sha256(stage/entry['file']) != entry['sha256']:
                raise ValueError('Invalid dataset member or checksum')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, destination)
    return destination


class Assets:
    def __init__(self, root, dataset):
        self.folder = root/'datasets'/dataset
        if not (self.folder/'index.json').exists():
            raise FileNotFoundError(f'Install dataset {dataset} first (see data/manifests/archives.json)')
        self.index = json.loads((self.folder/'index.json').read_text())

    def path(self, logical):
        logical = str(logical).replace('\\', '/')
        if logical.startswith('workspace/'):
            logical = logical[len('workspace/'):]
        if logical not in self.index:
            raise KeyError(f'Unpackaged source reference: {logical}')
        return self.folder/self.index[logical]['file']

    def json(self, logical):
        return json.loads(self.path(logical).read_text())
