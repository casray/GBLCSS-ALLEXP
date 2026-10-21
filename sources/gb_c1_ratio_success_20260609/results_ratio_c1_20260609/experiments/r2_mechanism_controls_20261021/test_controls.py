import unittest

import torch
import torch.nn.functional as F

import kernels as k


class ControlTests(unittest.TestCase):
    def test_required_kernels_exist(self):
        for name in ('mass_matched', 'loss_for', 'targets_from_counts', 'neighbor_indices',
                     'support_counts', 'random_partition', 'shuffle_rows'):
            self.assertTrue(callable(getattr(k, name, None)), name)

    def test_mass_preserves_observed_probability(self):
        t = torch.tensor([[.2, .7, .1], [.4, .25, .35]])
        z = torch.tensor([1, 0])
        out = k.mass_matched(t, z)
        self.assertTrue(torch.allclose(out, torch.tensor([[.15, .7, .15], [.4, .3, .3]])))
        self.assertTrue(torch.allclose(out.sum(1), torch.ones(2)))

    def test_mass_onehot_stays_onehot(self):
        z = torch.tensor([0, 1])
        t = F.one_hot(z, 2).float()
        self.assertTrue(torch.equal(k.mass_matched(t, z), t))

    def test_ls_matches_pytorch(self):
        logits = torch.tensor([[1., -1., 2.], [.1, 2., -1.]], requires_grad=True)
        z = torch.tensor([1, 2])
        target = F.one_hot(z, 3).float()
        self.assertTrue(torch.allclose(k.loss_for(logits, z, target, 'LS'),
                                      F.cross_entropy(logits, z, label_smoothing=.1)))

    def test_gce_matches_formula_and_gradient(self):
        logits = torch.tensor([[1., 2., 0.], [-1., 1., 0.]], requires_grad=True)
        z = torch.tensor([0, 1])
        loss = k.loss_for(logits, z, F.one_hot(z, 3).float(), 'GCE')
        expected = ((1-logits.softmax(1)[torch.arange(2), z].pow(.7))/.7).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_boot_prediction_is_detached(self):
        logits = torch.tensor([[.5, -1., 2.]], requires_grad=True)
        z = torch.tensor([0])
        t = .8*F.one_hot(z, 3)+.2*logits.detach().softmax(1)
        expected = -(t*logits.log_softmax(1)).sum()
        got = k.loss_for(logits, z, t, 'BOOT')
        a, = torch.autograd.grad(got, logits, retain_graph=True)
        b, = torch.autograd.grad(expected, logits)
        self.assertTrue(torch.allclose(a, b))

    def test_votes_ties_and_low_support(self):
        z = torch.tensor([2, 0, 1])
        counts = torch.tensor([[2., 2., 0.], [0., 0., 1.], [0., 3., 0.]])
        t, alpha = k.targets_from_counts(z, counts)
        self.assertTrue(torch.allclose(t[0], torch.tensor([.125, .125, .75])))
        self.assertEqual(float(alpha[1]), 0.)
        self.assertTrue(torch.equal(t[1], torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.equal(t[2], torch.tensor([0., 1., 0.])))

    def test_knn_exact_support_self_exclusion_and_ties(self):
        h = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
        support = torch.tensor([2, 0, 1, 3])
        indices = k.neighbor_indices(h, support, batch_size=2)
        for n in range(4):
            chosen = indices[n, :support[n]]
            self.assertNotIn(n, chosen.tolist())
            self.assertEqual(len(chosen.unique()), int(support[n]))
        self.assertEqual(indices[0, :2].tolist(), [1, 2])

    def test_random_partition_matches_sizes_and_training_support(self):
        a = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        train = torch.tensor([0, 2, 3, 5, 7])
        r = k.random_partition(a, train, seed=1701)
        self.assertTrue(torch.equal(a.bincount(), r.bincount()))
        self.assertTrue(torch.equal(a[train].bincount(minlength=3), r[train].bincount(minlength=3)))
        self.assertEqual(sorted(k.support_counts(a[train]).tolist()),
                         sorted(k.support_counts(r[train]).tolist()))

    def test_shuffle_preserves_rows_and_is_reproducible(self):
        h = torch.arange(60).reshape(20, 3)
        v, perm = k.shuffle_rows(h, 1702)
        self.assertTrue(torch.equal(v, h[perm]))
        self.assertEqual(sorted(perm.tolist()), list(range(20)))
        self.assertFalse(torch.equal(h, v))
        self.assertTrue(torch.equal(v, k.shuffle_rows(h, 1702)[0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
