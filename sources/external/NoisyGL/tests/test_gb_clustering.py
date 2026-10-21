import unittest

import numpy as np

from scripts.diagnose_gb_clustering import (
    build_gate_rows,
    clustering_metrics,
    granular_ball_clustering,
    hungarian_acc,
)


class GBClusteringDiagnosticTest(unittest.TestCase):
    def test_hungarian_acc_is_one_for_permuted_perfect_clustering(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(hungarian_acc(y_true, y_pred), 1.0)

    def test_metrics_handle_k_auto_not_equal_true_k(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        metrics = clustering_metrics(y_true, y_pred)
        self.assertEqual(metrics["n_clusters"], 3)
        self.assertGreaterEqual(metrics["acc"], 0.0)
        self.assertLessEqual(metrics["acc"], 1.0)
        self.assertTrue(np.isfinite(metrics["nmi"]))
        self.assertTrue(np.isfinite(metrics["ari"]))

    def test_granular_clustering_is_label_independent(self):
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
        labels_a = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        labels_b = np.array([2, 1, 0, 2, 1, 0, 2, 1])

        result_a = granular_ball_clustering(
            features,
            max_balls=4,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=7,
            graph_k=2,
        )
        result_b = granular_ball_clustering(
            features,
            max_balls=4,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=7,
            graph_k=2,
        )
        self.assertTrue(np.array_equal(result_a["assignment"], result_b["assignment"]))
        self.assertTrue(np.array_equal(result_a["node_cluster"], result_b["node_cluster"]))
        self.assertNotEqual(hungarian_acc(labels_a, result_a["node_cluster"]), hungarian_acc(labels_b, result_b["node_cluster"]))

    def test_gate_rejects_when_precision_is_not_close(self):
        metric_rows = [
            {
                "dataset": "toy",
                "seed": 0,
                "split_improvement": 0.05,
                "method": "gb_cluster",
                "status": "ok",
                "true_k": 3,
                "k_auto": 3,
                "num_balls": 6,
                "acc": 0.50,
                "nmi": 0.40,
                "ari": 0.10,
            },
            {
                "dataset": "toy",
                "seed": 0,
                "split_improvement": 0.05,
                "method": "kmeans_H_trueK",
                "status": "ok",
                "true_k": 3,
                "k_auto": 3,
                "num_balls": 6,
                "acc": 0.70,
                "nmi": 0.60,
                "ari": 0.20,
            },
            {
                "dataset": "toy",
                "seed": 0,
                "split_improvement": 0.05,
                "method": "kmeans_H_autoK",
                "status": "ok",
                "true_k": 3,
                "k_auto": 3,
                "num_balls": 6,
                "acc": 0.49,
                "nmi": 0.39,
                "ari": 0.10,
            },
        ]
        wrong_rows = [
            {
                "dataset": "toy",
                "split_improvement": 0.05,
                "acc_advantage": 0.10,
            }
        ]
        gates = build_gate_rows(metric_rows, wrong_rows, main_split=0.05)
        self.assertEqual(gates[0]["g1_precision_close"], 0)
        self.assertEqual(gates[0]["gate"], "FAIL")


if __name__ == "__main__":
    unittest.main()
