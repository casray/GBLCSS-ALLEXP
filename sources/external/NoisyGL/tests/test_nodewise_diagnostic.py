import unittest

import numpy as np
import torch

from scripts.diagnose_nodewise_targets import (
    apply_nodewise_soft_targets,
    distance_bin_target_metrics,
    mean_kl_to_ball_distribution,
)
from utils.frozen_grace_targets import apply_ball_soft_targets


class NodewiseDiagnosticTest(unittest.TestCase):
    def test_uniform_ball_mode_matches_current_softdist_targets(self):
        h = torch.eye(4)
        noisy_label = torch.tensor([0, 0, 0, 1])
        assignment = np.array([0, 0, 0, 0])
        train_mask = torch.ones(4, dtype=torch.bool)

        result = apply_nodewise_soft_targets(
            embeddings=h,
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            vote_mode="uniform",
            alpha_mode="ball",
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        expected_target, expected_alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        self.assertTrue(torch.allclose(result.soft_target, expected_target))
        self.assertTrue(torch.allclose(result.alpha, expected_alpha))

    def test_similarity_weighted_vote_creates_node_specific_distribution(self):
        h = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        noisy_label = torch.tensor([0, 0, 1])
        assignment = np.array([0, 0, 0])
        train_mask = torch.ones(3, dtype=torch.bool)

        result = apply_nodewise_soft_targets(
            embeddings=h,
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            vote_mode="sim_weighted",
            alpha_mode="ball",
            tau_sim=0.1,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        self.assertGreater(float(result.soft_target[0, 0]), 0.95)
        self.assertGreater(float(result.soft_target[2, 1]), 0.95)
        self.assertGreater(mean_kl_to_ball_distribution(result.soft_target, result.ball_dist, train_mask), 0.1)

    def test_distance_alpha_protects_boundary_clean_label_damage(self):
        h = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
                [0.5, 0.0],
                [10.0, 0.0],
            ]
        )
        noisy_label = torch.tensor([1, 1, 1, 1, 1, 1, 0])
        clean_label = torch.tensor([0, 0, 1, 1, 1, 1, 0])
        assignment = np.array([0, 0, 0, 0, 0, 0, 0])
        train_mask = torch.ones(7, dtype=torch.bool)

        ball = apply_nodewise_soft_targets(
            embeddings=h,
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            vote_mode="uniform",
            alpha_mode="ball",
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        dist = apply_nodewise_soft_targets(
            embeddings=h,
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            vote_mode="uniform",
            alpha_mode="distance",
            gamma=2.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        ball_metrics = distance_bin_target_metrics(
            noisy_label=noisy_label,
            clean_label=clean_label,
            pseudo_label=ball.pseudo_label,
            alpha=ball.alpha,
            train_mask=train_mask,
            distance_ratio=dist.distance_ratio,
            n_bins=2,
        )
        dist_metrics = distance_bin_target_metrics(
            noisy_label=noisy_label,
            clean_label=clean_label,
            pseudo_label=dist.pseudo_label,
            alpha=dist.alpha,
            train_mask=train_mask,
            distance_ratio=dist.distance_ratio,
            n_bins=2,
        )

        self.assertGreater(ball_metrics["all"]["cdr"], dist_metrics["all"]["cdr"])
        self.assertLess(float(dist.alpha[-1]), float(ball.alpha[-1]))


if __name__ == "__main__":
    unittest.main()
