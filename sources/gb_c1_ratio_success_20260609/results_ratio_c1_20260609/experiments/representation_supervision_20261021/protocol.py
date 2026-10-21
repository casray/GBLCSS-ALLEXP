from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from core import ROOT, WORKSPACE, PROJECT, make_targets, reference_target_functions
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices
from gblcss.targets import build_leave_one_out_targets

DATA_FILES = {
    'cora': PROJECT / 'external/NoisyGL/data/Cora/processed/data.pt',
    'citeseer': PROJECT / 'external/NoisyGL/data/CiteSeer/processed/data.pt',
    'corafull': PROJECT / 'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
}
MATRIX = WORKSPACE / 'experiments/main_protocol_replay_20260922/matrix'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        v = np.asarray(a, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(v.shape, dtype=np.int64).tobytes())
        digest.update(v.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=3)
def raw_data(name):
    from torch_geometric.data import Data
    stored = torch.load(DATA_FILES[name], map_location='cpu', weights_only=False)
    return Data.from_dict(stored[0]) if isinstance(stored[0], dict) else stored[0]


def prepare_inputs(config):
    sources, checks = {}, []
    original = reference_target_functions()
    for name in config['datasets']:
        raw = raw_data(name)
        indices = percent_split_indices(len(raw.x), config['seed'])
        reference_path = MATRIX / name / 'default/seed0.json'
        meta = read(reference_path)
        source = Path(meta['ball_embedding_path'])
        stored = torch.load(source, map_location='cpu', weights_only=False)
        if isinstance(stored, dict) and 'labels' in stored and not torch.equal(stored['labels'].long().cpu(), raw.y.long().cpu()):
            raise RuntimeError('Representation node labels/order differ from raw graph')
        values = stored['H'] if isinstance(stored, dict) else stored
        h = F.normalize(values.float(), dim=1, eps=1e-12).detach().cpu()
        if len(h) != len(raw.x) or not torch.isfinite(h).all():
            raise RuntimeError('Invalid representation')
        with np.load(reference_path.with_suffix('.npz')) as cache:
            assignment = torch.tensor(cache['assignment'].copy(), dtype=torch.long)
            historical_t = torch.tensor(cache['targets'].copy()).float()
            historical_train = torch.tensor(cache['train_index'].copy())
        if not torch.equal(historical_train, torch.tensor(indices[0])):
            raise RuntimeError('Historical training order mismatch')
        assignment_hash = original['_ball_assignment_hash'](assignment.numpy())
        if assignment_hash != meta['ball_assignment_hash'] or fingerprint(*indices) != meta['split_hash']:
            raise RuntimeError('Historical split/partition mismatch')
        folder = ROOT / 'inputs' / name
        folder.mkdir(parents=True, exist_ok=True)
        assets = dict(h=h, assignment=assignment, train=torch.tensor(indices[0]),
                      valid=torch.tensor(indices[1]), test=torch.tensor(indices[2]),
                      classes=int(raw.y.max())+1)
        torch.save(assets, folder / 'assets.pt')
        for condition, (kind, rate) in config['conditions'].items():
            masks = []
            for index in indices:
                mask = torch.zeros(len(raw.x), dtype=torch.bool); mask[index] = True
                masks.append(mask)
            data = GraphData(name, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                             train_index=assets['train'], val_index=assets['valid'], test_index=assets['test'])
            inject_train_label_noise(data, kind, rate, config['seed'])
            if not torch.equal(data.y[~masks[0]], data.noisy_y[~masks[0]]):
                raise RuntimeError('Noise outside training')
            z = data.noisy_y[assets['train']]
            target, alpha = make_targets(z, assignment[assets['train']], data.num_classes)
            legacy, _ = make_targets(z, assignment[assets['train']], data.num_classes,
                                     leave_one_out=False, tie_safe=False)
            legacy_reference, _, _ = original['apply_ball_soft_targets'](
                z, assignment[assets['train']].numpy(), torch.arange(len(z)), data.num_classes,
                alpha_baseline='multiclass', target_mode='distribution', target_beta=.5)
            if float((legacy-legacy_reference).abs().max()) > 3e-7:
                raise RuntimeError('Legacy target reference differs')
            reference = build_leave_one_out_targets(data.noisy_y, assignment.numpy(), assets['train'], data.num_classes)
            maximum_difference = float((target-reference.soft_target[assets['train']]).abs().max())
            if maximum_difference > 3e-7:
                raise RuntimeError('Reference target mismatch')
            record = dict(dataset=name, condition=condition, seed=config['seed'], noise_type=kind, noise_rate=rate,
                          split_hash=fingerprint(*indices), noise_hash=fingerprint(data.y, data.noisy_y),
                          assignment_hash=assignment_hash, num_nodes=len(raw.x), classes=data.num_classes,
                          actual_balls=len(torch.unique(assignment)), raw_feature_dim=raw.x.shape[1], h_dim=h.shape[1],
                          changed_training_labels=int((data.y[masks[0]] != data.noisy_y[masks[0]]).sum()),
                          modified_val_count=0, modified_test_count=0,
                          reference_target_max_difference=maximum_difference,
                          feature_input='raw X', representation_pretrain_seed=stored.get('pretrain_seed') if isinstance(stored, dict) else None,
                          representation_source=str(source), raw_source=str(DATA_FILES[name]),
                          historical_partition_source=str(reference_path.with_suffix('.npz')))
            if condition == 'uniform0.4':
                historical_difference = float((target-historical_t).abs().max())
                if record['noise_hash'] != meta['noise_hash'] or historical_difference > 3e-7:
                    raise RuntimeError('Historical uniform .4 inputs mismatch')
                record['historical_target_max_difference'] = historical_difference
            cache = dict(observed=z, target=target, legacy_target=legacy, alpha=alpha,
                         valid_y=data.y[assets['valid']].clone())
            path = folder / (condition+'.pt')
            torch.save(cache, path)
            write(folder / (condition+'.json'), record)
            checks.append(record)
        for path in (DATA_FILES[name], source, reference_path, reference_path.with_suffix('.npz'), folder/'assets.pt'):
            sources[str(path)] = sha(path)
        for path in folder.glob('*'):
            sources[str(path)] = sha(path)
    return sources, checks


@lru_cache(maxsize=1)
def load_dataset(name, device='cuda'):
    assets = torch.load(ROOT/'inputs'/name/'assets.pt', weights_only=True)
    assets['x'] = raw_data(name).x.float()
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in assets.items()}


def load_condition(name, condition, device='cuda'):
    cache = torch.load(ROOT/'inputs'/name/(condition+'.pt'), weights_only=True)
    return {k: v.to(device) for k, v in cache.items()}
