import inspect
import unittest
import numpy as np
from run import build_partition, bundle_from_neighbors, make_balls


class ProtocolTests(unittest.TestCase):
    def test_small_radius_gain_no_longer_stops_all_splits(self):
        # A high dimensional cloud has small relative radius gains, but useful SSE splits.
        h = np.random.RandomState(8).normal(size=(240, 128))
        result = build_partition(h, np.arange(160), 8, 2)
        self.assertEqual(result.summary['num_balls'], 8)
        self.assertGreaterEqual(result.summary['min_train_count'], 16)
        self.assertTrue(all(t['sse_gain'] > 0 for t in result.split_trace if t['accepted']))
        self.assertEqual(set(np.unique(result.assignment)), set(range(8)))

    def test_support_limits_budget_without_labels(self):
        self.assertEqual(list(inspect.signature(build_partition).parameters), ['h', 'train', 'budget', 'seed'])
        h = np.random.RandomState(9).normal(size=(120, 6))
        result = build_partition(h, np.arange(32), 12, 0)
        self.assertLessEqual(result.summary['num_balls'], 2)
        self.assertGreaterEqual(result.summary['min_train_count'], 16)

    def test_identical_points_are_not_forced_into_budget(self):
        result = build_partition(np.ones((80, 5)), np.arange(64), 8, 0)
        self.assertEqual(result.summary['num_balls'], 1)
        self.assertEqual(result.summary['stop_reason'], 'no_feasible_split')

    def test_own_label_cannot_change_peer_q_or_alpha(self):
        groups = np.repeat([0, 1], 4)
        a = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        b = a.copy(); b[0] = 1
        first = make_balls(groups, a, 2); second = make_balls(groups, b, 2)
        np.testing.assert_array_equal(first['q'][0], second['q'][0])
        self.assertEqual(first['alpha'][0], second['alpha'][0])

    def test_knn_matches_ball_vote_count_without_self(self):
        from scipy import sparse
        neighbors = sparse.csr_matrix(np.ones((4, 4)) - np.eye(4))
        bundle = bundle_from_neighbors(neighbors, np.array([0, 0, 1, 1]), 2)
        reference = make_balls(np.zeros(4), np.array([0, 0, 1, 1]), 2)
        for key in ['q', 'alpha', 'target', 'support']:
            np.testing.assert_allclose(bundle[key], reference[key], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
