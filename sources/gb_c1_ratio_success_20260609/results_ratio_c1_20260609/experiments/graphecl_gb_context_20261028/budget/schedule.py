"""Budget selection and strictly checked immutable checkpoint reuse."""
import copy
import json
from pathlib import Path

PLAN = json.loads((Path(__file__).parent / 'plan.json').read_text(encoding='utf-8'))


def arms_for(dataset):
    if dataset not in PLAN['primary_datasets']:
        raise ValueError('Dataset excluded by the budget revision')
    return PLAN['retained_arms'] if dataset in PLAN['full_dose_datasets'] else PLAN['core_arms']


def tasks():
    return [dict(dataset=d, seed=s, arm=a) for d in PLAN['primary_datasets'] for s in PLAN['seeds'] for a in arms_for(d)]


def convert_checkpoint(old, expected):
    for key in ['dataset', 'seed', 'config']:
        if old['expected'][key] != expected[key]:
            raise ValueError('Cached baseline mismatch: ' + key)
    if old['expected'].get('method') != 'graphecl' or old['epoch'] != expected['config']['epochs']:
        raise ValueError('Cached baseline method or epoch mismatch')
    out = copy.deepcopy(old)
    out['expected'] = copy.deepcopy(expected)
    return out
