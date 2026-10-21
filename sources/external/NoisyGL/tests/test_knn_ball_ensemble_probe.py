import unittest

import numpy as np


class KNNBallEnsembleProbeTest(unittest.TestCase):
    def test_percentile_rank_scores_are_scale_invariant_and_higher_is_more_suspicious(self):
        from scripts.probe_knn_ball_ensemble import percentile_rank_scores

        scores = np.array([10.0, 30.0, 20.0])
        scaled = np.array([1.0, 3.0, 2.0])

        self.assertTrue(np.allclose(percentile_rank_scores(scores), percentile_rank_scores(scaled)))
        self.assertGreater(percentile_rank_scores(scores)[1], percentile_rank_scores(scores)[2])
        self.assertGreater(percentile_rank_scores(scores)[2], percentile_rank_scores(scores)[0])

    def test_rank_mean_ensemble_averages_knn_and_ball_ranks(self):
        from scripts.probe_knn_ball_ensemble import rank_mean_ensemble

        knn = np.array([0.1, 0.2, 0.9])
        ball = np.array([0.8, 0.2, 0.3])

        ensemble = rank_mean_ensemble(knn, ball)

        expected = 0.5 * (
            np.array([0.0, 0.5, 1.0]) + np.array([1.0, 0.0, 0.5])
        )
        self.assertTrue(np.allclose(ensemble, expected))

    def test_z_mean_ensemble_handles_constant_inputs(self):
        from scripts.probe_knn_ball_ensemble import z_mean_ensemble

        knn = np.array([1.0, 1.0, 1.0])
        ball = np.array([0.0, 1.0, 2.0])

        ensemble = z_mean_ensemble(knn, ball)

        self.assertTrue(np.isfinite(ensemble).all())
        self.assertAlmostEqual(float(ensemble.mean()), 0.0, places=6)
        self.assertGreater(float(ensemble[2]), float(ensemble[0]))

    def test_summarize_delta_counts_paired_positive_seeds(self):
        from scripts.probe_knn_ball_ensemble import summarize_delta_rows

        rows = [
            {"dataset": "cora", "detector": "rank", "delta_auroc_vs_knn": 0.02},
            {"dataset": "cora", "detector": "rank", "delta_auroc_vs_knn": -0.01},
            {"dataset": "cora", "detector": "rank", "delta_auroc_vs_knn": 0.03},
        ]

        summary = summarize_delta_rows(rows, ["dataset", "detector"])

        self.assertEqual(summary[0]["num_seeds"], 3)
        self.assertEqual(summary[0]["delta_auroc_vs_knn_pos"], 2)
        self.assertAlmostEqual(summary[0]["delta_auroc_vs_knn_mean"], (0.02 - 0.01 + 0.03) / 3)


if __name__ == "__main__":
    unittest.main()
