import inspect
import unittest

import numpy as np

from scripts.diagnose_gb_clustering_sagbc_20260709 import (
    cluster_warnings,
    compute_soft_affiliation,
    hungarian_acc,
    sagbc_soft_affiliation_clustering,
)


class SAGBCClusteringDiagnosticTest(unittest.TestCase):
    def test_hungarian_acc_is_one_for_permuted_perfect_clustering(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(hungarian_acc(y_true, y_pred), 1.0)

    def test_soft_affiliation_rows_sum_to_one_and_limit_neighbors(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        centers = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        )
        affiliation, nearest, weights, sigma = compute_soft_affiliation(features, centers, m=2)
        self.assertEqual(affiliation.shape, (4, 4))
        self.assertEqual(nearest.shape, (4, 2))
        self.assertEqual(weights.shape, (4, 2))
        self.assertGreater(sigma, 0.0)
        self.assertTrue(np.allclose(affiliation.sum(axis=1), 1.0))
        self.assertTrue(np.all(np.count_nonzero(affiliation > 0.0, axis=1) <= 2))

    def test_sagbc_clustering_signature_does_not_accept_labels(self):
        signature = inspect.signature(sagbc_soft_affiliation_clustering)
        self.assertNotIn("labels", signature.parameters)
        self.assertNotIn("y_true", signature.parameters)

    def test_sagbc_clustering_is_label_independent(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        assignment = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        result_a = sagbc_soft_affiliation_clustering(
            features,
            assignment,
            adj=None,
            affiliation_m=2,
            graph_k=2,
        )
        result_b = sagbc_soft_affiliation_clustering(
            features,
            assignment,
            adj=None,
            affiliation_m=2,
            graph_k=2,
        )
        self.assertTrue(np.array_equal(result_a["node_cluster"], result_b["node_cluster"]))
        self.assertTrue(np.array_equal(result_a["ball_cluster"], result_b["ball_cluster"]))

    def test_cluster_warnings_marks_degenerate_and_fragmented_k_auto(self):
        self.assertIn("degenerate_k_auto", cluster_warnings(k_auto=1, num_balls=10))
        self.assertIn("fragmented_k_auto", cluster_warnings(k_auto=7, num_balls=10))


if __name__ == "__main__":
    unittest.main()
