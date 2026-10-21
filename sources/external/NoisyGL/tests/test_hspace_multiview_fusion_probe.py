import unittest

import numpy as np


class HSpaceMultiviewFusionProbeTest(unittest.TestCase):
    def test_rank_mean_many_matches_average_percentile_ranks(self):
        from scripts.probe_hspace_multiview_fusion import rank_mean_many

        scores = {
            "a": np.array([0.1, 0.3, 0.2]),
            "b": np.array([0.9, 0.1, 0.5]),
            "c": np.array([2.0, 3.0, 1.0]),
        }

        fused = rank_mean_many(scores, ["a", "b", "c"])

        expected = np.mean(
            np.vstack([
                np.array([0.0, 1.0, 0.5]),
                np.array([1.0, 0.0, 0.5]),
                np.array([0.5, 1.0, 0.0]),
            ]),
            axis=0,
        )
        self.assertTrue(np.allclose(fused, expected))

    def test_rank_mean_many_rejects_missing_detector(self):
        from scripts.probe_hspace_multiview_fusion import rank_mean_many

        with self.assertRaises(KeyError):
            rank_mean_many({"a": np.array([1.0, 2.0])}, ["a", "missing"])

    def test_add_best_single_deltas_uses_per_seed_best_single(self):
        from scripts.probe_hspace_multiview_fusion import add_best_single_deltas

        rows = [
            {"dataset": "cora", "seed": 0, "detector": "s_knn", "family": "single", "auroc": 0.80, "auprc_lift": 0.20, "precision_at_k": 0.70},
            {"dataset": "cora", "seed": 0, "detector": "s_gcn_conf", "family": "single", "auroc": 0.85, "auprc_lift": 0.25, "precision_at_k": 0.72},
            {"dataset": "cora", "seed": 0, "detector": "F1", "family": "fusion", "auroc": 0.87, "auprc_lift": 0.24, "precision_at_k": 0.75},
        ]

        out = add_best_single_deltas(rows, primary_metric="auroc")
        fusion = [row for row in out if row["detector"] == "F1"][0]

        self.assertEqual(fusion["best_single_detector"], "s_gcn_conf")
        self.assertAlmostEqual(fusion["delta_auroc_vs_best_single"], 0.02)
        self.assertAlmostEqual(fusion["delta_auprc_lift_vs_best_single"], -0.01)
        self.assertAlmostEqual(fusion["delta_precision_at_k_vs_best_single"], 0.03)

    def test_gate_decision_requires_two_datasets_with_stable_positive_fusion(self):
        from scripts.probe_hspace_multiview_fusion import gate_decision

        summary = [
            {"dataset": "cora", "detector": "F4_rank_knn_gcn_ball", "family": "fusion", "delta_auroc_vs_best_single_mean": 0.012, "delta_auroc_vs_best_single_pos": 4, "delta_auprc_lift_vs_best_single_mean": 0.001, "delta_precision_at_k_vs_best_single_mean": 0.001},
            {"dataset": "citeseer", "detector": "F4_rank_knn_gcn_ball", "family": "fusion", "delta_auroc_vs_best_single_mean": 0.011, "delta_auroc_vs_best_single_pos": 5, "delta_auprc_lift_vs_best_single_mean": 0.002, "delta_precision_at_k_vs_best_single_mean": 0.000},
            {"dataset": "coauthorcs", "detector": "F4_rank_knn_gcn_ball", "family": "fusion", "delta_auroc_vs_best_single_mean": -0.001, "delta_auroc_vs_best_single_pos": 2, "delta_auprc_lift_vs_best_single_mean": 0.000, "delta_precision_at_k_vs_best_single_mean": 0.000},
        ]

        self.assertTrue(gate_decision(summary).startswith("PASS"))


if __name__ == "__main__":
    unittest.main()
