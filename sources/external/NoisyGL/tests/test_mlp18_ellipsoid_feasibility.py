import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.frozen_grace_targets import (
    apply_ball_ellipsoid_pca_shrinkage_soft_targets,
    build_soft_targets_from_features,
)


class MLP18EllipsoidFeasibilityTest(unittest.TestCase):
    def _toy_inputs(self):
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
            ],
            dtype=torch.float,
        )
        noisy = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)
        clean = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        train_mask = torch.tensor([True, True, True, True, True, True])
        assignment = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        return features, noisy, clean, train_mask, assignment

    def test_clean_labels_do_not_affect_ellipsoid_pca_shrinkage_targets(self):
        features, noisy, clean, train_mask, _ = self._toy_inputs()
        clean_alt = torch.remainder(clean + 1, 2)

        result_a = build_soft_targets_from_features(
            ball_features=features,
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
            target_beta=0.5,
            softdist_weighting="ellipsoid_pca_shrinkage",
            ellipsoid_feature_space="pca128",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_distance_temperature=1.0,
            random_state=0,
        )
        result_b = build_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy,
            clean_label=clean_alt,
            train_mask=train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            softdist_weighting="ellipsoid_pca_shrinkage",
            ellipsoid_feature_space="pca128",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_distance_temperature=1.0,
            random_state=0,
        )

        self.assertTrue(torch.allclose(result_a.soft_target, result_b.soft_target))
        self.assertTrue(torch.allclose(result_a.alpha, result_b.alpha))
        self.assertTrue(torch.equal(result_a.pseudo_label, result_b.pseudo_label))

    def test_pca_dim_is_capped_to_legal_rank(self):
        features, noisy, _, train_mask, assignment = self._toy_inputs()

        _, _, _, diagnostics = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            ellipsoid_feature_space="pca128",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_ridge_scale=0.01,
            ellipsoid_weight_floor=0.0,
            ellipsoid_distance_temperature=1.0,
        )

        self.assertLessEqual(diagnostics["ellipsoid_pca_effective_dim_max"], 2)
        self.assertEqual(diagnostics["ellipsoid_requested_feature_space"], "pca128")

    def test_covariance_failure_records_fallback(self):
        features, noisy, _, train_mask, assignment = self._toy_inputs()

        with patch("utils.frozen_grace_targets.OAS.fit", side_effect=RuntimeError("boom")):
            _, _, _, diagnostics = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
                noisy_label=noisy,
                assignment=assignment,
                train_mask=train_mask,
                n_classes=2,
                ball_features=features,
                min_train_per_ball=2,
                alpha_baseline="multiclass",
                target_mode="distribution",
                target_beta=0.5,
                ellipsoid_feature_space="pca16",
                ellipsoid_covariance_estimator="oas",
                ellipsoid_ridge_scale=0.01,
                ellipsoid_weight_floor=0.0,
                ellipsoid_distance_temperature=1.0,
            )

        self.assertGreater(diagnostics["ellipsoid_covariance_failure_balls"], 0)
        self.assertGreater(diagnostics["ellipsoid_fallback_ball_ratio"], 0.0)

    def test_beta_endpoints_preserve_distribution_and_majority_semantics(self):
        features, noisy, _, train_mask, assignment = self._toy_inputs()

        soft_dist, _, _, _ = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.0,
            ellipsoid_feature_space="fullH",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_ridge_scale=0.01,
            ellipsoid_weight_floor=1.0,
            ellipsoid_distance_temperature=1.0,
        )
        soft_hard, _, _, _ = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=1.0,
            ellipsoid_feature_space="fullH",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_ridge_scale=0.01,
            ellipsoid_weight_floor=1.0,
            ellipsoid_distance_temperature=1.0,
        )

        self.assertGreater(soft_dist[0, 1].item(), 0.0)
        self.assertAlmostEqual(soft_hard[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(soft_hard[3, 1].item(), 1.0, places=6)

    def test_shuffled_control_preserves_ball_size_distribution(self):
        features, noisy, clean, train_mask, _ = self._toy_inputs()

        result = build_soft_targets_from_features(
            ball_features=features,
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
            target_beta=0.5,
            softdist_weighting="ellipsoid_pca_shrinkage",
            ellipsoid_feature_space="fullH",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_distance_temperature=1.0,
            random_state=0,
            random_control=False,
        )
        shuffled = build_soft_targets_from_features(
            ball_features=features,
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
            target_beta=0.5,
            softdist_weighting="ellipsoid_pca_shrinkage",
            ellipsoid_feature_space="fullH",
            ellipsoid_covariance_estimator="ridge",
            ellipsoid_distance_temperature=1.0,
            random_state=0,
            random_control=True,
        )

        self.assertEqual(
            sorted(np.bincount(result.assignment).tolist()),
            sorted(np.bincount(shuffled.assignment).tolist()),
        )


if __name__ == "__main__":
    unittest.main()
