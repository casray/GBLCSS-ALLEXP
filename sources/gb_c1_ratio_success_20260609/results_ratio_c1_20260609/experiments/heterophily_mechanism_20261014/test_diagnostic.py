import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'GB-LCSS_code/src'))
from gblcss.targets import build_leave_one_out_targets


def module():
    path = HERE / 'diagnostic.py'
    assert path.exists(), 'diagnostic implementation not yet created'
    spec = importlib.util.spec_from_file_location('mechanism_diagnostic', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_formal_target_parity_ties_sparse_and_self_exclusion():
    d = module()
    assignment = np.array([0, 0, 0, 0, 1, 1, 2])
    observed = np.array([0, 0, 1, 2, 1, 1, 2])
    voters = d.group_weights(assignment)
    r, q, a, count, effective = d.vote(voters, observed, 3)
    t = d.mix(q, a, observed)
    formal = build_leave_one_out_targets(torch.tensor(observed), assignment,
                                         torch.ones(7, dtype=torch.bool), 3)
    np.testing.assert_allclose(t, formal.soft_target.numpy(), atol=1e-7)
    assert np.all(voters.diagonal() == 0)
    assert np.all(a[4:] == 0)
    changed = observed.copy()
    changed[0] = 2
    r2, q2, a2, _, _ = d.vote(voters, changed, 3)
    np.testing.assert_allclose(q[0], q2[0])
    assert a[0] == a2[0]
    np.testing.assert_allclose(effective, count)


def test_knn_matches_unique_support_and_no_self():
    d = module()
    h = np.array([[1., 0], [.9, .1], [0, 1], [-1, 0]], dtype=np.float32)
    counts = np.array([2, 1, 0, 3])
    w = d.knn_weights(h, counts, device='cpu')
    np.testing.assert_array_equal(np.diff(w.indptr), counts)
    assert np.all(w.diagonal() == 0)
    assert w[0, 1] == 1


def test_weight_effective_support_hand_computed():
    d = module()
    w = sparse.csr_matrix(np.array([[0., 1., 3.], [1., 0, 1.], [0., 0, 0.]]))
    r, q, alpha, counts, eff = d.vote(w, np.array([0, 1, 0]), 2)
    np.testing.assert_allclose(r[0], [.75, .25])
    np.testing.assert_allclose(eff, [1.6, 2, 0])
    assert alpha[2] == 0
    np.testing.assert_allclose(d.mix(q, alpha, np.array([0, 1, 0])).sum(1), 1)


def test_alpha_controls_preserve_q_match_mean_and_shrink():
    d = module()
    r = np.array([[.8, .2], [.1, .9], [.6, .4]])
    coarse = np.array([[.6, .4], [.5, .5], [.5, .5]])
    a = np.array([.6, .8, .2])
    controls = d.alpha_controls(a, r, coarse, np.array([3, 30, 2]),
                                np.array([3, 30, 2]), np.array([9, 40, 10]), 2)
    assert np.all(controls['full'] <= a)
    np.testing.assert_allclose(controls['full'].mean(), controls['mean_matched'].mean(), atol=1e-12)
    assert set(controls) == {'original', 'half', 'ess', 'js', 'full', 'mean_matched', 'shuffled'}
    np.testing.assert_allclose(d.js_divergence(r, r), 0, atol=1e-12)


def test_metrics_separate_clean_harm_and_noise_recovery():
    d = module()
    clean = np.array([0, 0, 1, 1])
    observed = np.array([0, 1, 1, 0])
    t = np.eye(2)[[1, 0, 1, 1]]
    got = d.target_metrics(t, t, np.ones(4), clean, observed)
    assert got['target_brier'] == .5
    assert got['clean_harm_rate'] == .5
    assert got['noise_recovery_rate'] == 1.


def test_invalid_weights_rejected():
    d = module()
    with pytest.raises(ValueError):
        d.vote(sparse.csr_matrix([[0., -1.], [1., 0.]]), np.array([0, 1]), 2)
