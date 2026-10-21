import copy

import numpy as np
import torch

from run_experiment import arms, brier_decomposition, fingerprint, validate_pairs


def test_matrix_is_one_factor_at_a_time():
    cfg = {"grids": {"beta": [0, .25, .75, 1], "max_balls": [32, 128],
                      "min_train_per_ball": [5, 10], "min_ball_size": [1, 16, 64],
                      "split_improvement": [.01, .1]}}
    matrix = arms(cfg)
    assert len(matrix) == 18
    assert len({a['name'] for a in matrix}) == 18
    assert all(len(a['override']) <= 1 for a in matrix)


def test_brier_decomposition_accounts_for_correct_and_corrupted_labels():
    y = torch.tensor([0, 1, 0])
    z = torch.tensor([1, 1, 0])
    target = torch.tensor([[.8, .2], [.1, .9], [1., 0.]])
    d = brier_decomposition(target, z, y, torch.arange(3), 2)
    assert abs(d['brier_gain'] - (d['corrupted_gain'] - d['clean_cost'])) < 1e-7
    assert d['corrupted_gain'] > 0 and d['clean_cost'] > 0


def test_hash_preserves_order():
    assert fingerprint(np.array([1, 2])) != fingerprint(np.array([2, 1]))


def test_pair_validation_rejects_mismatched_protocol():
    rows = [{'dataset': 'cora', 'seed': 0, 'arm': a, 'split_hash': 'x', 'noise_hash': 'y'}
            for a in ['default', 'beta_0']]
    validate_pairs(rows)
    rows = copy.deepcopy(rows)
    rows[-1]['noise_hash'] = 'changed'
    try:
        validate_pairs(rows)
    except AssertionError:
        return
    raise AssertionError('protocol drift accepted')
