import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'gb_strong_kd_screen_20261007'))
from vendor.dist_official import DIST
from weighted_dist import weighted_dist, sample


def test_full_loss_and_gradient_equal_official():
    torch.manual_seed(4)
    logits = torch.randn(37, 7, dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.randn_like(logits)
    teacher = teacher_logits.softmax(1)
    reference = DIST()(logits, teacher_logits)
    value = weighted_dist(logits, teacher, torch.full((37,), 1/37, dtype=torch.float64), 37)
    torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    b = torch.autograd.grad(value, logits)[0]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_sampler_exact_budget_and_weights():
    assignment = np.repeat(np.arange(4), [3, 7, 13, 21])
    teacher = np.random.default_rng(4).dirichlet(np.ones(3), len(assignment))
    for method in ('uniform', 'size', 'adaptive'):
        for budget in (5, 11, 44):
            ids, weights, allocation = sample(assignment, teacher, budget, method, 1000)
            assert len(ids) == budget and len(np.unique(ids)) == budget
            assert np.all(weights > 0) and np.isclose(weights.sum(), 1)
            if method != 'uniform':
                assert np.array_equal(np.bincount(assignment[ids], minlength=4), allocation)
            if budget == 44:
                np.testing.assert_allclose(weights, 1/44)


def test_permutation_invariance():
    torch.manual_seed(9)
    z = torch.randn(19, 4, dtype=torch.float64)
    t = torch.randn_like(z).softmax(1)
    w = torch.rand(19, dtype=torch.float64)
    w /= w.sum()
    order = torch.randperm(19)
    torch.testing.assert_close(weighted_dist(z, t, w, 100),
                               weighted_dist(z[order], t[order], w[order], 100))


def test_constant_teacher_has_finite_gradient():
    z = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    t = torch.full_like(z, 1/3)
    value = weighted_dist(z, t, torch.full((12,), 1/12, dtype=z.dtype), 60)
    assert torch.isfinite(value)
    assert torch.isfinite(torch.autograd.grad(value, z)[0]).all()


def test_invalid_weights_rejected():
    z = torch.randn(5, 3)
    t = z.softmax(1)
    with pytest.raises(ValueError):
        weighted_dist(z, t, torch.ones(5), 5)
    with pytest.raises(ValueError):
        weighted_dist(z, t, torch.tensor([-1., .5, .5, .5, .5]), 5)
