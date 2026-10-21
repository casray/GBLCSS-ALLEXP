import unittest

import numpy as np
import torch

from utils.reliable_gb import (
    build_reliable_granular_balls,
    build_random_size_matched_assignment,
)


class ReliableGBToolsTest(unittest.TestCase):
    def _two_cluster(self, per=20, dim=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(per, dim, generator=g) + 5.0
        b = torch.randn(per, dim, generator=g) - 5.0
        return torch.cat([a, b], dim=0)

    def test_covers_every_node_no_orphan(self):
        emb = self._two_cluster()
        result = build_reliable_granular_balls(emb, k=10, min_ball_size=3, random_state=7)
        self.assertEqual(result.assignment.shape, (40,))
        self.assertGreaterEqual(int(result.assignment.min()), 0)
        self.assertEqual(sum(ball["size"] for ball in result.balls), 40)
        self.assertEqual(result.summary["num_balls"], int(result.assignment.max()) + 1)
        for key in ["num_balls", "mean_ball_size", "max_ball_size", "singleton_ratio", "mean_radius", "reliable_k"]:
            self.assertIn(key, result.summary)

    def test_reciprocal_membership_keeps_clusters_pure(self):
        emb = self._two_cluster()
        result = build_reliable_granular_balls(emb, k=10, min_ball_size=3, random_state=0)
        assignment = result.assignment
        for ball_id in range(int(assignment.max()) + 1):
            members = np.where(assignment == ball_id)[0]
            clusters = {0 if int(i) < 20 else 1 for i in members}
            self.assertEqual(len(clusters), 1, f"ball {ball_id} crossed the cluster boundary")

    def test_random_size_matched_preserves_ball_count(self):
        emb = self._two_cluster()
        result = build_reliable_granular_balls(emb, k=10, min_ball_size=3, random_state=0)
        matched = build_random_size_matched_assignment(result.assignment, random_state=17)
        self.assertEqual(matched.shape, result.assignment.shape)
        self.assertEqual(len(np.unique(matched)), len(np.unique(result.assignment)))

    def test_deterministic_under_fixed_seed(self):
        emb = self._two_cluster()
        a = build_reliable_granular_balls(emb, k=10, random_state=3).assignment
        b = build_reliable_granular_balls(emb, k=10, random_state=3).assignment
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
