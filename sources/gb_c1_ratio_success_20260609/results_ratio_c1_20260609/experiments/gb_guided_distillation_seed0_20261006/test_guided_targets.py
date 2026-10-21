import unittest

import numpy as np
import torch

from guided_targets import eligible_mask, build_arms, shuffled_assignment


class GuidedTargetTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.tensor([[.6, .4], [.8, .2], [1., 0.], [.3, .7]])
        self.gb = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [1., 0.]])
        self.labels = torch.tensor([0, 1, 1, 0])
        self.alpha = torch.tensor([.8, 0., 1., 1.])
        self.eligible = torch.tensor([True, True, True, False])

    def test_mix_and_shared_weights(self):
        targets, weights = build_arms(self.teacher, self.gb, self.gb.flip(1),
                                     self.labels, self.alpha, self.eligible)
        self.assertAlmostEqual(weights['D3'][0].item(), .16, places=6)
        torch.testing.assert_close(targets['D3'][0], torch.tensor([.504, .496]))
        self.assertTrue(torch.equal(weights['D1'], weights['D3']))
        self.assertTrue(torch.equal(weights['D4'], weights['D3']))
        for arm in targets:
            self.assertTrue(torch.equal(targets[arm][3], self.teacher[3]))
            torch.testing.assert_close(targets[arm].sum(1), torch.ones(4))
        for node in (1, 2):
            self.assertTrue(torch.equal(targets['D3'][node], self.teacher[node]))
        self.assertAlmostEqual(weights['D2'].sum().item(), weights['D3'].sum().item(), places=6)
        self.assertGreater(weights['D2'][1].item(), 0)

    def test_zero_lambda(self):
        targets, _ = build_arms(self.teacher, self.gb, self.gb, self.labels,
                               self.alpha, self.eligible, strength=0.)
        for target in targets.values():
            self.assertTrue(torch.equal(target, self.teacher))

    def test_eligibility_is_not_positive_alpha(self):
        assignment = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
        train = np.array([True, True, True, False, True, True, True, False, False])
        actual = eligible_mask(assignment, train, 2)
        np.testing.assert_array_equal(actual, [True, True, True, False, False, False, False, False, False])

    def test_shuffle_preserves_counts_and_scope(self):
        assignment = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        train = np.array([True, True, True, False] * 2 + [True, False, False, False])
        eligible = eligible_mask(assignment, train, 2)
        shuffled = shuffled_assignment(assignment, eligible)
        np.testing.assert_array_equal(shuffled[~eligible], assignment[~eligible])
        np.testing.assert_array_equal(np.bincount(shuffled), np.bincount(assignment))
        np.testing.assert_array_equal(np.bincount(shuffled[train]), np.bincount(assignment[train]))
        np.testing.assert_array_equal(eligible_mask(shuffled, train, 2), eligible)
        np.testing.assert_array_equal(shuffled_assignment(assignment, eligible), shuffled)
        self.assertFalse(np.array_equal(assignment[eligible], shuffled[eligible]))


if __name__ == '__main__':
    unittest.main()
