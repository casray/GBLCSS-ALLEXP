import tempfile
import unittest
from pathlib import Path

import torch

from utils.frozen_grace_targets import build_knn_soft_targets_from_features
from utils.tools import load_conf


class MLP11KSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.3, 0.7],
            ],
            dtype=torch.float,
        )
        self.noisy = torch.tensor([0, 0, 1, 1, 0, 1])
        self.clean_a = torch.tensor([0, 1, 1, 1, 0, 0])
        self.clean_b = torch.tensor([1, 1, 0, 0, 1, 1])
        self.train_mask = torch.tensor([True, True, True, True, False, False])

    def _build(self, k, clean=None):
        return build_knn_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            clean_label=self.clean_a if clean is None else clean,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_k=k,
            target_mode="distribution",
            target_beta=0.5,
        )

    def test_coauthorcs_knn_config_loads(self):
        conf = load_conf(None, "knn_mlp_softdist_beta05", "coauthorcs")
        self.assertEqual(conf.model["method"], "knn_mlp_softdist_beta05")
        self.assertEqual(conf.gb["soft_target_builder"], "knn")
        self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)
        self.assertIn("CoauthorCS", conf.gb["embedding_path"])

    def test_matched_k_changes_targets(self):
        k1 = self._build(1)
        k3 = self._build(3)
        self.assertEqual(k1.diagnostics["matched_k"], 1)
        self.assertEqual(k3.diagnostics["matched_k"], 3)
        self.assertFalse(torch.allclose(k1.soft_target, k3.soft_target))

    def test_large_k_clamps_to_train_count(self):
        result = self._build(100)
        self.assertEqual(result.diagnostics["matched_k"], int(self.train_mask.sum().item()))

    def test_knn_targets_do_not_depend_on_clean_labels(self):
        a = self._build(3, self.clean_a)
        b = self._build(3, self.clean_b)
        self.assertTrue(torch.allclose(a.soft_target, b.soft_target))
        self.assertTrue(torch.equal(a.alpha, b.alpha))
        self.assertTrue(torch.equal(a.pseudo_label, b.pseudo_label))

    def test_summary_global_k_and_regression_exception(self):
        from scripts import summarize_mlp11_k_sensitivity as summary

        curve = [
            {"dataset": "cora", "k": 10, "test_mean": 0.60, "test_std": 0.01, "valid_mean": 0.59},
            {"dataset": "cora", "k": 50, "test_mean": 0.55, "test_std": 0.01, "valid_mean": 0.58},
            {"dataset": "citeseer", "k": 10, "test_mean": 0.50, "test_std": 0.01, "valid_mean": 0.49},
            {"dataset": "citeseer", "k": 50, "test_mean": 0.57, "test_std": 0.01, "valid_mean": 0.56},
            {"dataset": "pubmed", "k": 10, "test_mean": 0.70, "test_std": 0.01, "valid_mean": 0.70},
            {"dataset": "pubmed", "k": 50, "test_mean": 0.69, "test_std": 0.01, "valid_mean": 0.69},
            {"dataset": "coauthorcs", "k": 10, "test_mean": 0.82, "test_std": 0.01, "valid_mean": 0.82},
            {"dataset": "coauthorcs", "k": 50, "test_mean": 0.81, "test_std": 0.01, "valid_mean": 0.81},
        ]
        best = summary.best_k_rows(curve)
        global_k, losses = summary.global_k_rows(curve, best)
        self.assertEqual(global_k, 10)
        self.assertTrue(any(row["dataset"] == "citeseer" and row["loss_ge_2pp"] == 1 for row in losses))

        new_rows = [
            {"dataset": "citeseer", "seed": 4, "matched_k": 10, "test": 0.51},
        ]
        p0_rows = [
            {"dataset": "citeseer", "seed": 4, "matched_k": 11, "test": 0.52},
        ]
        regression = summary.regression_rows(new_rows, p0_rows)
        self.assertEqual(regression[0]["status"], "skipped_p0_matched_k_11")


if __name__ == "__main__":
    unittest.main()
