from pathlib import Path
import unittest
import numpy as np


def implementation():
    path = Path(__file__).with_name('diagnose.py')
    assert path.exists(), 'diagnostic implementation is not yet present'
    import diagnose
    return diagnose


class DiagnosticTests(unittest.TestCase):
    def test_group_moments_match_direct_computation(self):
        d = implementation()
        x = np.array([[1., 0.], [0., 1.], [.2, .8]])
        a = np.array([0, 0, 1])
        sizes, means, ss = d.moments(x, a)
        np.testing.assert_array_equal(sizes, [2, 1])
        np.testing.assert_allclose(means, [[.5, .5], [.2, .8]])
        np.testing.assert_allclose(ss, [1., 0.], atol=1e-12)

    def test_integer_budget_and_capacity(self):
        d = implementation()
        sizes = np.array([1, 8, 3, 10])
        for budget in range(4, 23):
            for weights in (sizes, np.array([0., 1., 0., 100.]), np.zeros(4)):
                m = d.allocate(sizes, budget, weights)
                self.assertEqual(m.sum(), budget)
                self.assertTrue(np.all((m >= 1) & (m <= sizes)))

    def test_sampling_variance_matches_enumeration(self):
        from itertools import product
        d = implementation()
        x = np.array([[1., 0.], [0., 1.], [.2, .8], [.6, .4]])
        sizes, means, ss = d.moments(x, np.array([0, 0, 1, 1]))
        exact = d.sampling_mse(sizes, ss, np.array([1, 1]))
        errors = [np.sum(((x[a] + x[b]) / 2 - x.mean(0)) ** 2)
                  for a, b in product([0, 1], [2, 3])]
        self.assertAlmostEqual(exact, np.mean(errors), places=12)
        self.assertEqual(d.sampling_mse(sizes, ss, sizes), 0.)

    def test_exception_curve_is_monotone_and_exact_at_full(self):
        d = implementation()
        p = np.array([[.9, .1], [.1, .9], [.4, .6], [.8, .2]])
        a = np.array([0, 0, 1, 1])
        rows = d.compression_curve(p, a, [0., .25, .5, 1.])
        self.assertTrue(all(rows[i]['mse'] >= rows[i+1]['mse'] for i in range(3)))
        self.assertEqual(rows[-1]['mse'], 0.)
        self.assertEqual(rows[-1]['agreement_pct'], 100.)
        self.assertEqual(rows[0]['bytes'], 4*4 + 2*2*4)

    def test_random_groups_preserve_sizes(self):
        d = implementation()
        a = np.array([0, 0, 1, 1, 1, 2])
        b = d.random_groups(a, 100)
        np.testing.assert_array_equal(np.bincount(a), np.bincount(b))
        np.testing.assert_array_equal(b, d.random_groups(a, 100))

    def test_constant_features_have_zero_dispersion(self):
        d = implementation()
        x = np.ones((8, 4)) / 4
        a = np.repeat(np.arange(2), 4)
        _, _, ss = d.moments(x, a)
        np.testing.assert_array_equal(ss, [0., 0.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
