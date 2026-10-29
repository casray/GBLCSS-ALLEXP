"""Read-only historical data bridge; SSL interface never returns labels."""
import importlib.util
from functools import lru_cache
import torch
from artifacts import WORKSPACE, PREP, read

_path = WORKSPACE / 'experiments/granular_evidence_ssl_20261026/gvec_data.py'
_spec = importlib.util.spec_from_file_location('_ssl_historical_data', _path)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
seed_all, state_hash, import_file = _old.seed_all, _old.state_hash, _old.import_file
class_count, analysis_labels, final_scores = _old.class_count, _old.analysis_labels, _old.final_scores

@lru_cache(maxsize=6)
def load_features(name, seed):
    data = _old.load_features(name, seed)
    expected = next(x for x in read(PREP/'input_manifest.json')['datasets']
                    if x['dataset'] == name and x['seed'] == seed)
    for key, tensor in [('x', data['x']), ('edge', data['edge_index'])]:
        # Preparation uses this exact named-tensor fingerprint convention.
        if state_hash({key: tensor}) != expected[key + '_hash']:
            raise RuntimeError(f'Frozen {key} tensor mismatch: {name}/{seed}')
    if data['metadata']['split_hash'] != expected['split_hash']:
        raise RuntimeError('Frozen split hash mismatch')
    return data

def load_ssl(name, seed):
    data = load_features(name, seed)
    x = data['x']
    x = x / x.abs().sum(1, keepdim=True).clamp_min(1e-12)
    return {'x': x, 'edge_index': data['edge_index'], 'metadata': data['metadata']}

def load_supervised(name, seed, condition):
    return _old.load_supervised(name, seed, condition)
