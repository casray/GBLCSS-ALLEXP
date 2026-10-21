import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.diagnose_featuregb_mlp import _build_report, evaluate_offline_gate
from scripts.run_standard_downstream import PREDICTOR_MODULES
from utils.frozen_grace_targets import (
    build_frozen_grace_soft_targets,
    build_soft_targets_from_features,
)
from utils.tools import load_conf


def _toy_inputs():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
            [0.0, 0.2, 0.8],
        ]
    )
    noisy = torch.tensor([0, 0, 1, 1, 1, 0])
    clean = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = np.arange(6)
    return features, noisy, clean, train_mask


class FeatureGBMLPTest(unittest.TestCase):
    def test_in_memory_builder_accepts_sparse_features(self):
        features, noisy, clean, train_mask = _toy_inputs()
        sparse = features.to_sparse()

        result = build_soft_targets_from_features(
            ball_features=sparse,
            noisy_label=noisy,
            clean_label=clean,
            train_mask=train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            random_state=0,
        )

        self.assertEqual(result.soft_target.shape, (6, 2))
        self.assertEqual(result.assignment.shape, (6,))
        self.assertEqual(result.diagnostics["ball_feature_source"], "memory")

    def test_frozen_path_wrapper_matches_in_memory_builder(self):
        features, noisy, clean, train_mask = _toy_inputs()
        kwargs = {
            "noisy_label": noisy,
            "clean_label": clean,
            "train_mask": train_mask,
            "n_classes": 2,
            "max_balls": 2,
            "min_ball_size": 2,
            "split_improvement": 0.0,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "random_state": 0,
        }
        direct = build_soft_targets_from_features(ball_features=features, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H.pt"
            torch.save(features, path)
            wrapped = build_frozen_grace_soft_targets(embedding_path=path, **kwargs)

        self.assertTrue(np.array_equal(direct.assignment, wrapped.assignment))
        self.assertTrue(torch.allclose(direct.soft_target, wrapped.soft_target))
        self.assertTrue(torch.allclose(direct.alpha, wrapped.alpha))

    def test_clean_labels_do_not_change_assignment_or_soft_targets(self):
        features, noisy, clean, train_mask = _toy_inputs()
        flipped_clean = 1 - clean
        kwargs = {
            "ball_features": features,
            "noisy_label": noisy,
            "train_mask": train_mask,
            "n_classes": 2,
            "max_balls": 2,
            "min_ball_size": 2,
            "split_improvement": 0.0,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "random_state": 0,
        }

        first = build_soft_targets_from_features(clean_label=clean, **kwargs)
        second = build_soft_targets_from_features(clean_label=flipped_clean, **kwargs)

        self.assertTrue(np.array_equal(first.assignment, second.assignment))
        self.assertTrue(torch.allclose(first.soft_target, second.soft_target))
        self.assertTrue(torch.allclose(first.alpha, second.alpha))

    def test_shuffled_control_preserves_ball_size_distribution(self):
        features, noisy, clean, train_mask = _toy_inputs()
        kwargs = {
            "ball_features": features,
            "noisy_label": noisy,
            "clean_label": clean,
            "train_mask": train_mask,
            "n_classes": 2,
            "max_balls": 2,
            "min_ball_size": 2,
            "split_improvement": 0.0,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "random_state": 0,
        }

        real = build_soft_targets_from_features(random_control=False, **kwargs)
        shuffled = build_soft_targets_from_features(random_control=True, **kwargs)
        real_sizes = sorted(np.bincount(real.assignment).tolist())
        shuffled_sizes = sorted(np.bincount(shuffled.assignment).tolist())

        self.assertEqual(real_sizes, shuffled_sizes)

    def test_x_ball_predictor_does_not_require_embedding_path(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        predictor = object.__new__(module.featuregb_mlp_softdist_Predictor)
        features, noisy, clean, train_mask = _toy_inputs()
        predictor.conf = SimpleNamespace(
            gb={
                "ball_feature_source": "x",
                "max_balls": 2,
                "min_ball_size": 2,
                "split_improvement": 0.0,
                "min_train_per_ball": 2,
                "alpha_baseline": "multiclass",
                "target_mode": "distribution",
            },
            training={"debug": False},
        )
        predictor.feats = features.to_sparse()
        predictor.noisy_label = noisy
        predictor.clean_label = clean
        predictor.train_mask = train_mask
        predictor.n_classes = 2
        predictor.device = "cpu"
        predictor.seed = 0
        predictor.use_xh_features = False
        predictor.xh_embedding_dim = 0

        predictor._make_corrected_labels()

        self.assertEqual(predictor.gb_diagnostics["ball_feature_source"], "x")
        self.assertIsNotNone(predictor.soft_target)

    def test_featuregb_methods_are_registered_and_configs_are_h_free(self):
        methods = (
            "featuregb_mlp_softdist",
            "featuregb_mlp_softdist_shuffled",
        )
        single_exp_text = (
            Path(__file__).resolve().parents[1] / "single_exp.py"
        ).read_text(encoding="utf-8")
        for method in methods:
            self.assertIn(method, PREDICTOR_MODULES)
            self.assertIn(f"'{method}'", single_exp_text)
            for dataset in ("cora", "citeseer"):
                with self.subTest(method=method, dataset=dataset):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.gb["ball_feature_source"], "x")
                    self.assertNotIn("embedding_path", conf.gb)
                    self.assertEqual(conf.gb["max_balls"], 64)
                    self.assertEqual(conf.gb["min_ball_size"], 32)

    def test_offline_gate_requires_both_purity_gaps_and_coverage(self):
        rows = [
            {
                "dataset": dataset,
                "control": control,
                "true_all_purity_weighted": true_all,
                "noisy_train_purity_weighted": noisy_train,
                "eligible_train_coverage": coverage,
            }
            for dataset in ("cora", "citeseer")
            for control, true_all, noisy_train, coverage in (
                ("real", 0.80, 0.70, 0.96),
                ("shuffled", 0.65, 0.60, 0.96),
            )
        ]

        passed, decisions = evaluate_offline_gate(rows)

        self.assertTrue(passed)
        self.assertTrue(all(row["passed"] for row in decisions))

    def test_offline_report_exposes_mechanism_metrics_and_stop_decision(self):
        rows = [
            {
                "dataset": "cora",
                "control": control,
                "num_balls": 1,
                "true_all_purity_weighted": 0.30,
                "noisy_train_purity_weighted": 0.24,
                "soft_pseudo_ncr": 0.33,
                "soft_pseudo_cdr": 0.68,
                "alpha_positive_train_ratio": 1.0,
                "mean_alpha_train": 0.11,
                "eligible_train_coverage": 1.0,
            }
            for control in ("real", "shuffled")
        ]
        decisions = [
            {
                "dataset": "cora",
                "true_all_purity_gap": 0.0,
                "noisy_train_purity_gap": 0.0,
                "eligible_train_coverage": 1.0,
                "passed": False,
            }
        ]

        report = _build_report(rows, decisions, passed=False)

        self.assertIn("| cora | real | 1 |", report)
        self.assertIn("NCR", report)
        self.assertIn("mean alpha", report)
        self.assertIn("single-ball partition", report)
        self.assertIn("Seed-0 downstream: NOT RUN", report)
        self.assertIn("does not establish an H-free SoftDist pipeline", report)


if __name__ == "__main__":
    unittest.main()
