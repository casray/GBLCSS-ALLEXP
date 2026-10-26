"""Validation-only bounded selectors."""
import numpy as np

SCORE_KEYS = {'arm', 'support', 'strength', 'valid_correct', 'valid_total', 'valid_nll', 'best_epoch', 'epoch'}


def validate_score(row):
    if set(row)-SCORE_KEYS:
        raise ValueError('Selector fields rejected: '+str(set(row)-SCORE_KEYS))
    return row


def rank_key(row):
    validate_score(row)
    return (-row['valid_correct']/row['valid_total'], row['valid_nll'],
            row.get('strength', 0), -row.get('support', 0), row['best_epoch'])


def probe_survivors(rows, count, prefix):
    for r in rows:
        validate_score(r)
        if r['epoch'] != prefix:
            raise ValueError('PROBE score exceeds current prefix')
    ranked = sorted(rows, key=rank_key)
    kept = [r['arm'] for r in ranked[:count]]
    anchor = next((r['arm'] for r in rows if r.get('support') == 32 and r.get('strength') == 1), None)
    if anchor and anchor not in kept:
        kept[-1] = anchor
    return kept


def balanced(scores):
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    top = max(means.values())
    return min(k for k in means if means[k] >= top-.002-1e-12)


def constrained(rows, ce, family):
    feasible = []
    for parameter, values in rows.items():
        clean = values[0]['validation']-ce[0]['validation']
        f1 = np.mean([r['valid_macro_f1']-b['valid_macro_f1'] for r, b in zip(values, ce)])
        if clean >= -.01-1e-12 and f1 >= -.005-1e-12:
            feasible.append(parameter)
    if not feasible:
        return .7 if family == 'GCE' else 0., False
    return min(feasible, key=lambda p: (-np.mean([r['validation'] for r in rows[p]]), p)), True


def dataset_fold(rows, heldout):
    train = [r for r in rows if r['dataset'] != heldout]
    test = [r for r in rows if r['dataset'] == heldout]
    if {r['dataset'] for r in train} & {r['dataset'] for r in test}:
        raise ValueError('Graph leakage')
    return train, test
