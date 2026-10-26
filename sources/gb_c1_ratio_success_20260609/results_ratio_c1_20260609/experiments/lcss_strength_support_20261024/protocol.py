"""Read-only historical inputs and output-confined experiment bookkeeping."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
EXP = ROOT.parent
DESIGN = WORKSPACE/'research/lcss_strength_support_design_20261023'
CONFIG = json.loads((DESIGN/'config.scheduled.json').read_text(encoding='utf-8-sig'))
INPUTS = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def output(path):
    path = Path(path).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError('Write outside owned experiment: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(path, value):
    path = output(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def csv_save(path, rows):
    path = output(path)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def append(path, value):
    with output(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def track(path):
    path = Path(path).resolve()
    key = str(path)
    if key not in INPUTS:
        INPUTS[key] = sha(path)
        save(ROOT/'inputs.json', INPUTS)
    return path


def load(path):
    return torch.load(track(path), map_location='cpu', weights_only=True)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(EXP/'representation_supervision_20261021'))
from core import AlignedMLP, state_hash
common = import_file('lcss_common_readonly', EXP/'new_graph_mlp_seed0_20261009/common.py')
sys.path.insert(0, str(ROOT))


def check_time(value=None):
    value = datetime.fromisoformat(value) if value else datetime.now(timezone.utc)
    start = datetime.fromisoformat(CONFIG['not_before_utc'].replace('Z', '+00:00'))
    if value.tzinfo is None or value < start:
        raise RuntimeError('not_before has not elapsed')


def acquire_lock(path=None):
    path = output(path or ROOT/'run.lock')
    with path.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), started=now(), model='gpt-6.1-sol', effort='high'), f)
    return path


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_get():
    return dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def rng_set(state):
    torch.set_rng_state(state['cpu'])
    if state['gpu'] is not None:
        torch.cuda.set_rng_state(state['gpu'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def split_validation(y):
    rng = np.random.RandomState(20261024)
    meta, selection = [], []
    for c in sorted(torch.unique(y).tolist()):
        ids = torch.where(y == c)[0].numpy()
        ids = rng.permutation(ids)
        n = max(1, min(len(ids)-1, len(ids)//2)) if len(ids) >= 2 else 0
        meta.extend(ids[:n].tolist())
        selection.extend(ids[n:].tolist())
    return torch.tensor(sorted(meta), dtype=torch.long), torch.tensor(sorted(selection), dtype=torch.long)


def encoder_path(name):
    if name in ('cora', 'citeseer', 'corafull'):
        return EXP/'representation_supervision_20261021/pretraining'/name/'model.pt'
    return EXP/'r2_support_5dataset_20261022/pretraining'/name/'seed0/model.pt'


def graph_inputs(name):
    path = track(common.DATA_FILES[name])
    stored = torch.load(path, map_location='cpu', weights_only=False)
    raw = stored[0]
    x = raw['x'] if isinstance(raw, dict) else raw.x
    edges = raw['edge_index'] if isinstance(raw, dict) else raw.edge_index
    geo = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt')
    hpath = track(geo['h_path'])
    if INPUTS[str(hpath)] != geo['h_sha256']:
        raise RuntimeError('Historical GRACE hash differs from geometry provenance')
    km = load(EXP/'r2_body_comparison_20261023/geometry'/name/'seed0.pt')
    # No clean train/test labels leave this reader.
    return dict(x=x.float(), edge_index=edges.long(), geometry=geo, km=km)


def cell_inputs(name, condition):
    path = EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0'/f'{condition}.pt'
    pack = load(path)
    return {k: pack[k] for k in ('observed', 'valid_y', 'targets', 'selected_support')}


TRAIN_KEYS = {'x', 'train', 'valid', 'observed', 'valid_y', 'classes', 'meta', 'meta_y'}


def training_data(values):
    extra = set(values)-TRAIN_KEYS
    if extra:
        raise ValueError('Training fields rejected: '+str(sorted(extra)))
    return dict(values)


def freeze_sources():
    files = list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    files += [DESIGN/f for f in ['SCHEDULED_PLAN.md', 'config.scheduled.json', 'PLAN.md',
                                'RUNBOOK.md', 'EVIDENCE.md', 'schedule_receipt.json']]
    files += [EXP/'representation_supervision_20261021/core.py',
              EXP/'new_graph_mlp_seed0_20261009/common.py',
              EXP/'neighbor_preserving_ball_20261022/generator.py',
              EXP/'neighbor_support_ablation_20261022/kernels.py',
              EXP/'r2_support_5dataset_20261022/run.py',
              EXP/'r2_body_comparison_20261023/run.py',
              EXP/'r2_body_comparison_20261023/external_v2.py',
              EXP/'simmlp_tined_audit_seed0_20261009/simmlp_adapter.py']
    files += list((EXP/'simmlp_tined_audit_seed0_20261009/vendor/simmlp').glob('*.py'))
    files += list((WORKSPACE/'GB-LCSS_code/src/gblcss').glob('*.py'))
    manifest = {str(p.resolve()): sha(p) for p in files}
    old = ROOT/'sources.json'
    if old.exists() and read(old) != manifest and (ROOT/'fit_registry.json').exists():
        raise RuntimeError('Frozen sources changed; explicit repair receipt required')
    save(old, manifest)
    for filename in ('SCHEDULED_PLAN.md', 'config.scheduled.json', 'PLAN.md'):
        output(ROOT/'specification'/filename).write_bytes((DESIGN/filename).read_bytes())
    save(ROOT/'config.resolved.json', dict(CONFIG, scientific_stages_independent=True,
        c1_b2_completed_fit_alias=False, training_threads=1, seed_std=None,
        implementation_selection_module='selection.py (avoids stdlib selectors shadowing)'))
