import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.diagnose_orgingb_gate_scan import build_strength_matched_controls, dataset_kwargs_from_conf
from utils.ordinary_gb import (
    build_ordinary_granular_balls,
    build_random_size_matched_assignment,
)
from utils.soft_targets import build_gb_soft_targets


class OrdinaryGBToolsTest(unittest.TestCase):
    def test_build_ordinary_granular_balls_covers_every_node(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
            ]
        )

        result = build_ordinary_granular_balls(
            embeddings,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            random_state=7,
        )

        self.assertEqual(result.assignment.shape, (6,))
        self.assertEqual(set(result.assignment.tolist()), {0, 1})
        self.assertEqual(sum(ball["size"] for ball in result.balls), 6)
        self.assertEqual(result.summary["num_balls"], 2)

    def test_random_size_matched_assignment_preserves_ball_sizes(self):
        assignment = np.array([0, 0, 0, 1, 1, 2])

        shuffled = build_random_size_matched_assignment(assignment, random_state=3)

        original_sizes = sorted(np.bincount(assignment).tolist())
        shuffled_sizes = sorted(np.bincount(shuffled).tolist())
        self.assertEqual(original_sizes, shuffled_sizes)
        self.assertEqual(set(shuffled.tolist()), {0, 1, 2})

    def test_build_gb_soft_targets_returns_normalized_targets_and_weights(self):
        noisy_label = torch.tensor([0, 0, 1, 1, 1, 0])
        clean_label = torch.tensor([0, 1, 1, 1, 0, 0])
        probs = torch.tensor(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.6, 0.4],
            ],
            dtype=torch.float,
        )
        assignment = np.array([0, 0, 0, 1, 1, 1])
        train_mask = np.array([0, 1, 2, 3, 4, 5])

        result = build_gb_soft_targets(
            noisy_label=noisy_label,
            clean_label=clean_label,
            probs=probs,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            alpha_max=0.5,
            gamma=0.5,
            eta_max=0.4,
            min_train_per_ball=2,
            full_train_per_ball=3,
        )

        self.assertTrue(torch.allclose(result.targets.sum(dim=1), torch.ones(6)))
        self.assertTrue(torch.all(result.weights <= 1.0))
        self.assertTrue(torch.all(result.weights > 0.0))
        self.assertGreater(result.diagnostics["alpha_mean"], 0.0)
        self.assertIn("ncr", result.diagnostics)
        self.assertIn("cdr", result.diagnostics)

    def test_reweight_mode_keeps_targets_and_only_downweights_gated_nodes(self):
        noisy_label = torch.tensor([0, 0, 1, 1, 1, 0])
        clean_label = torch.tensor([0, 1, 1, 1, 0, 0])
        probs = torch.tensor(
            [
                [0.95, 0.05],
                [0.10, 0.90],
                [0.20, 0.80],
                [0.15, 0.85],
                [0.85, 0.15],
                [0.80, 0.20],
            ],
            dtype=torch.float,
        )
        assignment = np.array([0, 0, 0, 1, 1, 1])
        train_mask = np.array([0, 1, 2, 3, 4, 5])

        result = build_gb_soft_targets(
            noisy_label=noisy_label,
            clean_label=clean_label,
            probs=probs,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            target_mode="reweight",
            suspicion_mode="gb",
            suspicion_top_ratio=0.5,
            min_ball_conf=0.0,
            beta=0.5,
            min_weight=0.3,
            eta_max=0.0,
            min_train_per_ball=2,
            full_train_per_ball=3,
        )

        self.assertTrue(torch.equal(result.targets.argmax(dim=1), noisy_label))
        self.assertEqual(result.diagnostics["target_argmax_changed_ratio"], 0.0)
        gated = result.alpha > 0
        self.assertGreater(gated.float().mean().item(), 0.0)
        self.assertTrue(torch.all(result.weights[gated] < 1.0))
        self.assertTrue(torch.all(result.weights[~gated] == 1.0))
        self.assertGreaterEqual(result.weights.min().item(), 0.3)
        self.assertIn("gated_noisy_precision", result.diagnostics)
        self.assertIn("weight_mean", result.diagnostics)
        self.assertTrue(torch.equal(result.gate_mask, result.alpha > 0))
        self.assertEqual(result.node_ball_confidence.shape, noisy_label.shape)
        self.assertTrue(torch.all(result.node_ball_confidence >= 0.0))
        self.assertTrue(torch.all(result.node_ball_confidence <= 1.0))

    def test_no_ball_reweight_uses_warmup_label_confidence_for_suspicion(self):
        noisy_label = torch.tensor([0, 0, 1, 1])
        probs = torch.tensor(
            [
                [0.95, 0.05],
                [0.20, 0.80],
                [0.80, 0.20],
                [0.10, 0.90],
            ],
            dtype=torch.float,
        )
        assignment = np.array([0, 0, 1, 1])
        train_mask = np.array([0, 1, 2, 3])

        result = build_gb_soft_targets(
            noisy_label=noisy_label,
            probs=probs,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            target_mode="reweight",
            suspicion_mode="no_ball",
            suspicion_top_ratio=0.5,
            beta=0.5,
            min_weight=0.3,
        )

        expected_suspicion = torch.tensor([0.05, 0.80, 0.80, 0.10])
        self.assertTrue(torch.allclose(result.suspicion, expected_suspicion, atol=1e-6))
        downweighted = torch.nonzero(result.weights < 1.0, as_tuple=False).view(-1).tolist()
        self.assertEqual(set(downweighted), {1, 2})
        self.assertEqual(result.diagnostics["suspicion_mode"], "no_ball")

    def test_strength_matched_controls_match_real_gate_count(self):
        real_mask = np.array([True, False, True, False, False, False])
        train_indices = np.array([0, 1, 2, 3, 4, 5])
        no_ball_scores = np.array([0.10, 0.90, 0.20, 0.80, 0.30, 0.70])
        random_scores = np.array([0.60, 0.20, 0.50, 0.10, 0.90, 0.40])

        controls = build_strength_matched_controls(
            real_mask=real_mask,
            train_indices=train_indices,
            no_ball_scores=no_ball_scores,
            random_scores=random_scores,
        )

        self.assertEqual(int(controls["no_ball_strength_matched"].sum()), 2)
        self.assertEqual(int(controls["random_strength_matched"].sum()), 2)
        self.assertEqual(set(np.where(controls["no_ball_strength_matched"])[0]), {1, 3})
        self.assertEqual(set(np.where(controls["random_strength_matched"])[0]), {0, 4})

    def test_gate_scan_dataset_kwargs_follow_dataset_config(self):
        data_conf = SimpleNamespace(
            split={
                "split_type": "default",
                "train_size": None,
                "val_size": None,
                "test_size": None,
                "train_percent": None,
                "val_percent": None,
                "test_percent": None,
                "train_examples_per_class": None,
                "val_examples_per_class": None,
                "test_examples_per_class": None,
            },
            norm={"feat_norm": False, "adj_norm": False},
            modify={"add_self_loop": False, "from_npz_largest_component": False},
        )

        kwargs = dataset_kwargs_from_conf(data_conf, "cora", "cuda:0")

        self.assertEqual(kwargs["data"], "cora")
        self.assertEqual(kwargs["split_type"], "default")
        self.assertFalse(kwargs["add_self_loop"])
        self.assertFalse(kwargs["from_npz"])
        self.assertEqual(kwargs["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
