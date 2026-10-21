from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
from common import DATASETS, DATA_FILES, PROJECT, raw_data, fingerprint, write_json
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=5)
def assets(name):
    data = raw_data(name)
    return data, dict(data_file_sha256=hashlib.sha256(DATA_FILES[name].read_bytes()).hexdigest(),
                     x_sha256=tensor_hash(data.x.float()), edges_sha256=tensor_hash(data.edge_index.long()))


@lru_cache(maxsize=1)
def original_noise_functions():
    path = PROJECT / 'external/NoisyGL/utils/labelnoise.py'
    names = {'uniform_noise_cp', 'add_instance_independent_label_noise'}
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError('Original noise reference functions missing')
    namespace = dict(np=np, assert_array_almost_equal=assert_array_almost_equal)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(name, seed, rate, device='cuda'):
    if name not in DATASETS or seed not in range(5) or rate not in (.4, .6):
        raise ValueError('Only the approved five datasets, seeds 0..4 and rates .4/.6')
    raw, asset_record = assets(name)
    n = len(raw.x)
    indices = percent_split_indices(n, seed)
    masks = []
    for index in indices:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    graph = GraphData(name, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                      train_index=torch.tensor(indices[0]), val_index=torch.tensor(indices[1]),
                      test_index=torch.tensor(indices[2]))
    inject_train_label_noise(graph, 'uniform', rate, seed)
    reference_path = WORKSPACE / f'experiments/main_protocol_replay_20260922/matrix/{name}/default/seed{seed}.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8-sig'))
    split_hash, noise_hash = fingerprint(*indices), fingerprint(graph.y, graph.noisy_y)
    if split_hash != reference['split_hash']:
        raise RuntimeError('Historical split mismatch')
    if rate == .4 and noise_hash != reference['noise_hash']:
        raise RuntimeError('Historical uniform .4 noise mismatch')
    original, source_hash = original_noise_functions()
    matrix = original['uniform_noise_cp'](graph.num_classes, rate)
    original_y = original['add_instance_independent_label_noise'](
        graph.y[graph.train_index].numpy(), matrix, seed)
    if not np.array_equal(original_y, graph.noisy_y[graph.train_index].numpy()):
        raise RuntimeError('Original NoisyGL noise generator mismatch')
    record = dict(dataset=name, seed=seed, split_seed=seed, noise_seed=seed,
                  noise_type='uniform', noise_rate=rate, noise_scope='train',
                  split_hash=split_hash, noise_hash=noise_hash,
                  historical_hashes_verified=rate == .4,
                  historical_split_verified=True, original_noise_generator_verified=True,
                  original_noise_source_sha256=source_hash,
                  num_nodes=n, num_classes=graph.num_classes,
                  modified_train_count=int((graph.noisy_y[masks[0]] != graph.y[masks[0]]).sum()),
                  modified_val_count=int((graph.noisy_y[masks[1]] != graph.y[masks[1]]).sum()),
                  modified_test_count=int((graph.noisy_y[masks[2]] != graph.y[masks[2]]).sum()),
                  feature_input='raw main attributes', split_index_order='original random-choice order',
                  data_file=str(DATA_FILES[name]), **asset_record)
    if record['modified_val_count'] or record['modified_test_count']:
        raise RuntimeError('Noise leaked outside training')
    labels = torch.full_like(graph.y, -1)
    labels[masks[0]] = graph.noisy_y[masks[0]]
    data = Data(x=graph.x, edge_index=graph.edge_index, y=labels,
                train_index=graph.train_index, val_index=graph.val_index,
                test_index=graph.test_index, train_mask=masks[0], val_mask=masks[1],
                test_mask=masks[2], val_y=graph.y[graph.val_index],
                num_classes=graph.num_classes, name=name)
    return data.to(device), record


def final_metrics(name, seed, probabilities):
    raw, _ = assets(name)
    indices = percent_split_indices(len(raw.x), seed)
    clean = raw.y.long()
    if probabilities.shape != (len(clean), int(clean.max()) + 1):
        raise ValueError('Invalid probability shape')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid probabilities')
    torch.testing.assert_close(probabilities.sum(1), torch.ones(len(clean)), atol=1e-5, rtol=0)
    prediction = probabilities.argmax(1)
    return {key: float((prediction[idx] == clean[idx]).double().mean())
            for key, idx in zip(['train_clean', 'valid', 'test'], indices)}
