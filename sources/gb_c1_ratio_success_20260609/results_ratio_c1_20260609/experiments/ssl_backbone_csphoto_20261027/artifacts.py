"""Confined, atomic experiment artifacts and content-addressed completion records."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import torch

WORKSPACE = Path(r'C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609')
CODE = WORKSPACE / 'experiments/ssl_backbone_csphoto_20261027'
PREP = WORKSPACE / 'research/ssl_backbone_csphoto_20261027'
ROOT = Path('D:/codex_temp/ssl_backbone_csphoto_20261027')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

PROTOCOL = read(PREP / 'protocol.json')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError(f'Output outside experiment root: {path}')
    return path

def atomic(path, writer):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    atomic(path, lambda temporary: Path(temporary).write_text(text, encoding='utf-8'))

def save_tensor(path, value):
    atomic(path, lambda temporary: torch.save(value, temporary))

def signature():
    return {str(p): sha(p) for p in [PREP/'protocol.json', PREP/'input_manifest.json', PREP/'source_manifest.json', *sorted(CODE.glob('*.py'))]}

def complete(folder, expected, required):
    folder = inside(folder)
    record = {'status': 'complete', 'expected': expected,
              'files': {p: sha(folder / p) for p in required}}
    save(folder / 'complete.json', record)
    return record

def reusable(folder, expected):
    folder = inside(folder)
    marker = folder / 'complete.json'
    if not marker.exists():
        return False
    record = read(marker)
    if record['expected'] != expected:
        raise RuntimeError(f'Configuration mismatch: {folder}')
    for name, digest in record['files'].items():
        if sha(folder/name) != digest:
            raise RuntimeError(f'Artifact mismatch: {folder/name}')
    return True
