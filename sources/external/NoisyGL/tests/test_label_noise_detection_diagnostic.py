import math
import unittest

import numpy as np
import torch


class LabelNoiseDetectionDiagnosticTest(unittest.TestCase):
    def test_leave_one_out_scores_remove_self_label_and_fill_undefined(self):
        from scripts.diagnose_label_noise_detection import ball_leave_one_out_scores

        noisy = torch.tensor([0, 0, 1, 1, 0])
        assignment = np.array([0, 0, 0, 1, 2])
        train_idx = np.array([0, 1, 2, 3])

        scores, eligible, jsd = ball_leave_one_out_scores(
            noisy_label=noisy,
            assignment=assignment,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )

        self.assertTrue(np.allclose(scores[:3], np.array([0.5, 0.5, 1.0])))
        self.assertAlmostEqual(float(scores[3]), 0.5)
        self.assertEqual(eligible.tolist(), [True, True, True, False])
        self.assertGreater(float(jsd[2]), float(jsd[0]))

    def test_leave_one_out_scores_are_zero_for_same_label_ball(self):
        from scripts.diagnose_label_noise_detection import ball_leave_one_out_scores

        noisy = torch.tensor([1, 1, 1])
        assignment = np.array([0, 0, 0])
        train_idx = np.array([0, 1, 2])

        scores, eligible, _ = ball_leave_one_out_scores(
            noisy_label=noisy,
            assignment=assignment,
            train_idx=train_idx,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertTrue(np.allclose(scores, np.zeros(3)))
        self.assertTrue(bool(eligible.all()))

    def test_config_loader_uses_softdist_xh_gb_params_for_coauthorcs(self):
        from scripts.diagnose_label_noise_detection import load_softdist_xh_gb_config

        params = load_softdist_xh_gb_config("coauthorcs")

        self.assertEqual(params["max_balls"], 256)
        self.assertEqual(params["min_ball_size"], 32)
        self.assertAlmostEqual(params["split_improvement"], 0.001)
        self.assertIn("CoauthorCS", str(params["embedding_path"]))
        self.assertTrue(params["embedding_path"].exists())

    def test_detection_metrics_report_auprc_lift_and_precision_at_true_noise_count(self):
        from scripts.diagnose_label_noise_detection import detection_metrics

        labels = np.array([1, 0, 1, 0, 0], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)

        metrics = detection_metrics(labels, scores)

        self.assertAlmostEqual(metrics["noise_rate"], 0.4)
        self.assertGreater(metrics["auprc_lift"], 0.0)
        self.assertAlmostEqual(metrics["precision_at_k"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_k"], 0.5)
        self.assertAlmostEqual(metrics["f1_at_k"], 0.5)

    def test_conditional_auc_uses_nan_when_subset_has_too_few_noisy_nodes(self):
        from scripts.diagnose_label_noise_detection import conditional_auc

        labels = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.4], dtype=np.float64)
        subset = np.array([True, True, True, False, False, False])

        value = conditional_auc(labels, scores, subset, min_positive=2)

        self.assertTrue(math.isnan(value))

    def test_graph_density_shuffled_control_uses_same_output_shape_and_eligibility(self):
        from scripts.diagnose_label_noise_detection import graph_density_scores

        noisy = torch.tensor([0, 0, 0, 1, 1, 1])
        assignment = np.array([0, 0, 1, 1, 2, 2])
        train_idx = np.array([0, 1, 2, 3, 4, 5])
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 2],
                [2, 3, 0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [2.0, 0.0],
                [2.1, 0.0],
            ],
            dtype=torch.float32,
        )

        scores, eligible = graph_density_scores(
            embeddings=embeddings,
            noisy_label=noisy,
            assignment=assignment,
            train_idx=train_idx,
            edge_index=edge_index,
            n_classes=2,
            ball_neighbor_k=1,
            min_train_per_ball=2,
        )

        self.assertEqual(scores.shape, (6,))
        self.assertEqual(eligible.shape, (6,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
