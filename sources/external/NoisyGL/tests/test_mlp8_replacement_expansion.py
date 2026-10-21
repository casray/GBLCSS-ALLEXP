import importlib
import unittest

import torch

from utils.frozen_grace_targets import (
    build_ball_expansion_soft_targets,
    build_knn_expansion_soft_targets_from_features,
    build_kmeans_soft_targets_from_features,
    build_knn_soft_targets_from_features,
)
from utils.tools import load_conf


class MLP8ReplacementExpansionTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )
        self.noisy = torch.tensor([0, 0, 1, 1, 0, 1])
        self.clean_a = torch.tensor([0, 1, 1, 1, 0, 0])
        self.clean_b = torch.tensor([1, 1, 0, 0, 1, 1])
        self.train_mask = torch.tensor([True, True, True, True, False, False])

    def test_predictor_variants_registered(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        self.assertEqual(module.knn_mlp_softdist_beta05_Predictor.backbone, "mlp")
        self.assertEqual(module.knn_mlp_softdist_beta05_Predictor.soft_target_builder, "knn")
        self.assertEqual(module.kmeans_mlp_softdist_beta05_Predictor.backbone, "mlp")
        self.assertEqual(module.kmeans_mlp_softdist_beta05_Predictor.soft_target_builder, "kmeans")
        self.assertTrue(module.frozengrace_gb_mlp_softdist_beta05_expand_Predictor.use_ball_expansion)
        self.assertTrue(module.frozengrace_gb_mlp_softdist_beta05_expand_shuffled_Predictor.random_control)
        self.assertEqual(module.knn_mlp_softdist_beta05_expand_Predictor.soft_target_builder, "knn")
        self.assertEqual(module.knn_mlp_softdist_beta05_expand_Predictor.backbone, "mlp")
        self.assertEqual(module.mlp_distill_gcn_Predictor.backbone, "mlp")
        self.assertTrue(module.mlp_distill_gcn_Predictor.skip_ball_targets)

        standard_runner = importlib.import_module("scripts.run_standard_downstream")
        self.assertIs(standard_runner.predictor_class_for("knn_mlp_softdist_beta05"), module.knn_mlp_softdist_beta05_Predictor)
        self.assertIs(standard_runner.predictor_class_for("kmeans_mlp_softdist_beta05"), module.kmeans_mlp_softdist_beta05_Predictor)
        self.assertIs(
            standard_runner.predictor_class_for("frozengrace_gb_mlp_softdist_beta05_expand"),
            module.frozengrace_gb_mlp_softdist_beta05_expand_Predictor,
        )
        self.assertIs(standard_runner.predictor_class_for("mlp_distill_gcn"), module.mlp_distill_gcn_Predictor)

        ssl_runner = importlib.import_module("scripts.run_ssl_swap_downstream")
        self.assertIs(ssl_runner.predictor_class_for("knn_mlp_softdist_beta05_expand"), module.knn_mlp_softdist_beta05_expand_Predictor)

    def test_configs_load_for_all_p0_datasets(self):
        for method, builder in [
            ("knn_mlp_softdist_beta05", "knn"),
            ("kmeans_mlp_softdist_beta05", "kmeans"),
        ]:
            for dataset in ["cora", "citeseer", "pubmed"]:
                with self.subTest(method=method, dataset=dataset):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.training["n_epochs"], 200)
                    self.assertEqual(conf.gb["soft_target_builder"], builder)
                    self.assertEqual(conf.gb["target_mode"], "distribution")
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_configs_load_for_all_p1_datasets(self):
        for method in [
            "frozengrace_gb_mlp_softdist_beta05_expand",
            "frozengrace_gb_mlp_softdist_beta05_expand_shuffled",
            "knn_mlp_softdist_beta05_expand",
            "mlp_distill_gcn",
        ]:
            for dataset in ["cora", "citeseer", "pubmed"]:
                with self.subTest(method=method, dataset=dataset):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    if method == "mlp_distill_gcn":
                        self.assertEqual(conf.gb["teacher_epochs"], 200)
                        self.assertAlmostEqual(float(conf.gb["distill_temperature"]), 1.0)
                    else:
                        self.assertIn("lambda_u", conf.gb)
                        self.assertIn("expansion_target_beta", conf.gb)
                        self.assertEqual(conf.gb["target_mode"], "distribution")

    def test_knn_targets_do_not_depend_on_clean_labels_and_match_k(self):
        result_a = build_knn_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            clean_label=self.clean_a,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_k=2,
        )
        result_b = build_knn_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            clean_label=self.clean_b,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_k=2,
        )
        self.assertTrue(torch.allclose(result_a.soft_target, result_b.soft_target))
        self.assertTrue(torch.equal(result_a.alpha, result_b.alpha))
        self.assertTrue(torch.equal(result_a.pseudo_label, result_b.pseudo_label))
        self.assertEqual(result_a.diagnostics["target_builder"], "knn")
        self.assertEqual(result_a.diagnostics["matched_k"], 2)

    def test_kmeans_targets_do_not_depend_on_clean_labels_and_match_k(self):
        result_a = build_kmeans_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            clean_label=self.clean_a,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_kmeans_clusters=2,
        )
        result_b = build_kmeans_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            clean_label=self.clean_b,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_kmeans_clusters=2,
        )
        self.assertTrue(torch.allclose(result_a.soft_target, result_b.soft_target))
        self.assertTrue(torch.equal(result_a.alpha, result_b.alpha))
        self.assertTrue(torch.equal(result_a.pseudo_label, result_b.pseudo_label))
        self.assertEqual(result_a.diagnostics["target_builder"], "kmeans")
        self.assertEqual(result_a.diagnostics["kmeans_n_clusters"], 2)

    def test_expansion_target_beta_zero_is_backward_compatible(self):
        assignment = torch.tensor([0, 0, 0, 1, 1, 1]).numpy()
        default_result = build_ball_expansion_soft_targets(
            noisy_label=self.noisy,
            assignment=assignment,
            train_mask=self.train_mask,
            n_classes=2,
            min_train_per_ball=2,
            margin_threshold=0.0,
        )
        beta0_result = build_ball_expansion_soft_targets(
            noisy_label=self.noisy,
            assignment=assignment,
            train_mask=self.train_mask,
            n_classes=2,
            min_train_per_ball=2,
            margin_threshold=0.0,
            target_beta=0.0,
        )
        self.assertTrue(torch.allclose(default_result.soft_target, beta0_result.soft_target))
        self.assertTrue(torch.equal(default_result.candidate_mask, beta0_result.candidate_mask))
        self.assertEqual(beta0_result.diagnostics["expansion_target_beta"], 0.0)

    def test_knn_expansion_selects_only_non_train_candidates(self):
        result = build_knn_expansion_soft_targets_from_features(
            self.features,
            noisy_label=self.noisy,
            train_mask=self.train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=0,
            matched_k=2,
            min_train_per_ball=2,
            margin_threshold=0.0,
            target_beta=0.5,
        )
        self.assertEqual(result.diagnostics["expansion_builder"], "knn")
        self.assertEqual(result.diagnostics["expansion_matched_k"], 2)
        self.assertEqual(int(result.candidate_mask[self.train_mask].sum().item()), 0)
        self.assertGreater(int(result.candidate_mask[~self.train_mask].sum().item()), 0)
        self.assertTrue(torch.allclose(result.soft_target.sum(dim=1), torch.ones_like(self.noisy, dtype=torch.float)))


if __name__ == "__main__":
    unittest.main()
