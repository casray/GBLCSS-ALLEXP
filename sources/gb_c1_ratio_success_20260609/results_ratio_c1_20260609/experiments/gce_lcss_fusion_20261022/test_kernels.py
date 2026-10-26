import unittest
import torch
import torch.nn.functional as F
from kernels import objective, mass_target, target_diagnostic


class LossTests(unittest.TestCase):
    def setUp(self):
        self.a = torch.tensor([[.3, -.5, .8], [-.4, .9, .2]], dtype=torch.float64)
        self.z = torch.tensor([0, 2])
        self.t = torch.tensor([[.6, .1, .3], [.2, .3, .5]], dtype=torch.float64)

    def test_ce(self):
        loss, _ = objective(self.a, self.z, self.t, 'CE')
        torch.testing.assert_close(loss, F.cross_entropy(self.a, self.z))

    def test_gce_gradient(self):
        a = self.a.clone().requires_grad_()
        loss, _ = objective(a, self.z, self.t, 'GCE')
        grad = torch.autograd.grad(loss, a)[0]
        p = a.softmax(1)
        want = (p - F.one_hot(self.z, 3)) * p.gather(1, self.z[:, None]).pow(.7) / 2
        torch.testing.assert_close(grad, want)

    def test_weight_is_detached(self):
        a = self.a.clone().requires_grad_()
        loss, w = objective(a, self.z, self.t, 'W_LCSS')
        self.assertFalse(w.requires_grad)
        want = ((-(self.t * a.log_softmax(1)).sum(1)) *
                (self.t * a.softmax(1)).sum(1).detach().pow(.7)).mean()
        torch.testing.assert_close(torch.autograd.grad(loss, a)[0], torch.autograd.grad(want, a)[0])

    def test_mass_matches_observed_probability(self):
        m = mass_target(self.t, self.z)
        torch.testing.assert_close(m.sum(1), torch.ones(2, dtype=self.t.dtype))
        torch.testing.assert_close(m.gather(1, self.z[:, None]), self.t.gather(1, self.z[:, None]))

    def test_weighted_mass_uses_mass_agreement(self):
        m = mass_target(self.t, self.z)
        loss, w = objective(self.a, self.z, self.t, 'W_MASS')
        want = (m*self.a.softmax(1)).sum(1).pow(.7)
        torch.testing.assert_close(w, want)
        torch.testing.assert_close(loss, (w*(-m*self.a.log_softmax(1)).sum(1)).mean())

    def test_soft_gce_onehot_reduces_to_gce(self):
        one = F.one_hot(self.z, 3).to(self.a.dtype)
        torch.testing.assert_close(objective(self.a, self.z, one, 'SOFT_GCE')[0],
                                   objective(self.a, self.z, one, 'GCE')[0])

    def test_soft_gce_optimum_is_sharpened(self):
        p = self.t.pow(1/(1-.7))
        p = p/p.sum(1, keepdim=True)
        a = p.log().detach().requires_grad_()
        loss, _ = objective(a, self.z, self.t, 'SOFT_GCE')
        torch.testing.assert_close(torch.autograd.grad(loss, a)[0], torch.zeros_like(a), atol=1e-12, rtol=0)

    def test_ties_are_not_counted_as_unique(self):
        t = torch.tensor([[.5, .5, 0.], [.7, .2, .1], [.1, .8, .1]])
        out = target_diagnostic(t, torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1]))
        self.assertEqual(out['count'], 3)
        self.assertEqual(out['observed_unique_count'], 1)
        self.assertEqual(out['clean_unique_count'], 1)
        self.assertEqual(out['tie_count'], 1)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            objective(self.a, self.z, self.t, 'typo')


if __name__ == '__main__':
    unittest.main()
