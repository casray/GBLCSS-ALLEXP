"""Exact pilot validation ranking, including stable D1 best-epoch ties."""
from protocol import import_file, PILOT

_kernel = import_file('confirmation_pilot_selection', PILOT/'selection.py')
rank_key = _kernel.rank_key
probe_survivors = _kernel.probe_survivors


def score(arm, result, support=0, strength=0):
    return dict(arm=arm, support=support, strength=strength,
                valid_correct=result['valid_correct'], valid_total=result['valid_total'],
                valid_nll=result['valid_nll'], best_epoch=result['best_epoch'], epoch=result['epoch'])
