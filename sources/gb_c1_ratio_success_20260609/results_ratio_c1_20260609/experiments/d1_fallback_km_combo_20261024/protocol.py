"""Read-only historical adapters and output-confined provenance."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
WORKSPACE = EXP.parent
PILOT = EXP/'lcss_strength_support_20261024'
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
INPUTS = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def output(path):
    path = Path(path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError('Outside owned output directory: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def canonical(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value):
    path = output(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt==9:
                raise
            time.sleep(.1*(attempt+1))


def append(path, value):
    with output(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def csv_save(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with output(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def track(path):
    path = Path(path).resolve()
    key, digest = str(path), sha(path)
    if key in INPUTS and INPUTS[key] != digest:
        raise RuntimeError('Historical input changed: '+key)
    if key not in INPUTS:
        INPUTS[key] = digest
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
common = import_file('confirmation_common_readonly', EXP/'new_graph_mlp_seed0_20261009/common.py')
sys.path.insert(0, str(ROOT))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(state_hash({'value':item}).encode())
        elif isinstance(item,np.ndarray):
            digest.update(str((item.dtype,item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item,(list,tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def rng_get():
    return dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def rng_set(state):
    torch.set_rng_state(state['cpu'])
    if state['gpu'] is not None:
        torch.cuda.set_rng_state(state['gpu'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


TRAIN_KEYS = {'x', 'train', 'valid', 'observed', 'valid_y', 'classes'}


def training_data(values):
    extra = set(values)-TRAIN_KEYS
    if extra:
        raise ValueError('Training fields rejected: '+str(sorted(extra)))
    return dict(values)


def encoder_path(name, seed):
    if name in ('cora', 'citeseer', 'corafull'):
        if seed == 0:
            return EXP/'representation_supervision_20261021/pretraining'/name/'model.pt'
        return EXP/'gce_lcss_fusion_20261022/encoders'/name/f'seed{seed}/model.pt'
    return EXP/'r2_support_5dataset_20261022/pretraining'/name/f'seed{seed}/model.pt'


def geometry_path(name, seed):
    return EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/f'seed{seed}/geometry.pt'


def normalize_h(stored):
    values = stored['H'] if isinstance(stored,dict) and 'H' in stored else stored
    return F.normalize(values.float(),dim=1,eps=1e-12).detach().cpu()


def graph_inputs(name, seed, verify_km=False):
    stored = torch.load(track(common.DATA_FILES[name]), map_location='cpu', weights_only=False)
    raw = stored[0]
    x = raw['x'] if isinstance(raw, dict) else raw.x
    edges = raw['edge_index'] if isinstance(raw, dict) else raw.edge_index
    raw_y = raw['y'] if isinstance(raw,dict) else raw.y
    del raw, stored
    geo = load(geometry_path(name, seed))
    hpath = track(geo['h_path'])
    if sha(hpath) != geo['h_sha256']:
        raise RuntimeError('Historical H hash differs')
    expected = common.percent_split_indices(len(x), seed)
    for key, ids in zip(('train', 'valid', 'test'), expected):
        if not torch.equal(geo[key], torch.as_tensor(ids).long()):
            raise RuntimeError('Historical split mismatch '+name+str(seed)+key)
    stored_h = load(hpath)
    if isinstance(stored_h,dict) and 'labels' in stored_h and not torch.equal(stored_h['labels'].long(),raw_y.long()):
        raise RuntimeError('Historical H node order differs from raw label alignment')
    h = normalize_h(stored_h)
    original_h = stored_h['H'] if isinstance(stored_h,dict) else stored_h
    h_bridge = dict(container=type(stored_h).__name__,raw_norm_min=float(original_h.float().norm(dim=1).min()),
                   raw_norm_max=float(original_h.float().norm(dim=1).max()),
                   normalized_hash=state_hash({'H':h}),formula='F.normalize(float,dim=1,eps=1e-12)',
                   labels_alignment_checked=isinstance(stored_h,dict) and 'labels' in stored_h)
    if h.ndim!=2 or len(h) != len(x) or not torch.isfinite(h).all():
        raise RuntimeError('H node alignment shape mismatch')
    km_path = EXP/'r2_body_comparison_20261023/geometry'/name/f'seed{seed}.pt'
    km = load(km_path) if km_path.exists() else None
    if verify_km or km is None:
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            exact = torch.from_numpy(KMeans(len(torch.unique(geo['legacy32'])), n_init=10,
                                    random_state=seed).fit_predict(h.numpy())).long()
        if km is not None and not torch.equal(km, exact):
            raise RuntimeError('Historical KM not exact n_init10/seed bridge')
        if km is None:
            km_path = output(ROOT/'geometry'/name/f'seed{seed}_km.pt')
            torch.save(exact, km_path)
            km = exact
    return dict(x=x.float(), edge_index=edges.long(), geometry=geo, km=km,
                h_dim=h.shape[1], h_bridge=h_bridge, encoder=load(encoder_path(name, seed)), km_path=str(km_path))


def verify_clean_labels(pack, geo, raw_y):
    for key, index in [('observed','train'),('valid_y','valid')]:
        if not torch.equal(pack[key].long(),raw_y[geo[index]].long()):
            raise RuntimeError('Clean E1 labels differ from original '+index+' labels')


def cell_inputs(name, seed, condition):
    path = geometry_path(name, seed).parent/f'{condition}.pt'
    pack = load(path)
    oldroot = EXP/('neighbor_support_ablation_20261022' if name in ('cora','citeseer','corafull')
                  else 'r2_support_5dataset_20261022')
    previous_path = oldroot/'inputs'/name/f'seed{seed}'/f'{condition}.pt'
    if condition=='clean':
        raw = torch.load(track(common.DATA_FILES[name]),map_location='cpu',weights_only=False)[0]
        raw_y = raw['y'] if isinstance(raw,dict) else raw.y
        geo = load(geometry_path(name,seed))
        verify_clean_labels(pack,geo,raw_y)
        save(ROOT/'clean_label_bridges'/name/f'seed{seed}.json',dict(
            source=str(path),observed_hash=state_hash({'observed':pack['observed']}),
            valid_hash=state_hash({'valid_y':pack['valid_y']}),train_exact=True,valid_exact=True,
            original_data=str(common.DATA_FILES[name]),legacy_clean_available=previous_path.exists(),
            noise_regenerated=False))
    if previous_path.exists() or condition!='clean':
        previous = load(previous_path)
        for key in ('observed','valid_y'):
            if not torch.equal(pack[key],previous[key]):
                raise RuntimeError('Historical label protocol mismatch '+str(path)+key)
    return {key: pack[key] for key in ('observed', 'valid_y', 'targets', 'selected_support')}


def disk_guard(reserve=0):
    free = shutil.disk_usage(ROOT).free
    if free-reserve < CONFIG['storage']['minimum_free_bytes']:
        save(ROOT/'storage_pause.json', dict(time=now(), free_bytes=free, reserve_bytes=reserve))
        raise RuntimeError('Disk safety reserve reached; resume after storage is available')
    return free


def put_target(target):
    digest = state_hash({'target': target})
    path = output(ROOT/'targets'/f'{digest}.pt')
    if not path.exists():
        torch.save(target.cpu(), path)
    elif state_hash({'target': torch.load(path, weights_only=True)}) != digest:
        raise RuntimeError('Target content address mismatch')
    return canonical(path), digest


def freeze_sources():
    files = [*ROOT.glob('*.py'), *ROOT.glob('tests/*.py'), ROOT/'config.json', ROOT/'PLAN.md',
             PILOT/'targets.py', PILOT/'selection.py',
             Path(sys.modules['core'].__file__), Path(common.__file__),
             EXP/'r2_support_5dataset_20261022/run.py',
             Path(sys.modules['gblcss.models'].__file__), Path(sys.modules['gblcss.data'].__file__)]
    manifest = {str(p.resolve()): sha(p) for p in files}
    path = ROOT/'sources.json'
    if path.exists() and read(path) != manifest:
        raise RuntimeError('Frozen source changed; preserve and version repair explicitly')
    save(path, manifest)
    save(ROOT/'config.resolved.json', CONFIG)
