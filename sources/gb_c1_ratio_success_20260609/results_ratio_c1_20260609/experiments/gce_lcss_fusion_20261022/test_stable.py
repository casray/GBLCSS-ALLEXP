import unittest
import torch
from kernels import objective
from stable_losses import stable_objective


class StableTests(unittest.TestCase):
    def test_extreme_logits_have_finite_gradient(self):
        a = torch.tensor([[0., -200., -1000.]], requires_grad=True)
        t = torch.tensor([[.6, .3, .1]])
        for rule in ('GCE', 'SOFT_GCE'):
            loss, _ = stable_objective(a, torch.tensor([2]), t, rule)
            gradient = torch.autograd.grad(loss, a)[0]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(gradient).all())

    def test_float64_regular_values_agree(self):
        for rule in ('GCE', 'SOFT_GCE'):
            a = torch.tensor([[.3, -.5, .8]], dtype=torch.float64, requires_grad=True)
            target = torch.tensor([[.6, .1, .3]], dtype=torch.float64)
            old, _ = objective(a, torch.tensor([0]), target, rule)
            new, _ = stable_objective(a, torch.tensor([0]), target, rule)
            torch.testing.assert_close(old, new)
            torch.testing.assert_close(torch.autograd.grad(old, a)[0], torch.autograd.grad(new, a)[0])

    def test_other_rules_are_bitwise_unchanged(self):
        a = torch.tensor([[.3, -.5, .8]])
        t = torch.tensor([[.6, .1, .3]])
        for rule in ('CE','LCSS','MASS','W_LCSS','W_MASS'):
            old, w1 = objective(a, torch.tensor([0]), t, rule)
            new, w2 = stable_objective(a, torch.tensor([0]), t, rule)
            self.assertTrue(torch.equal(old,new))
            self.assertTrue(torch.equal(w1,w2))


if __name__ == '__main__':
    unittest.main()
