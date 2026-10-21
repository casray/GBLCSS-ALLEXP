import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_geaf_bridge import aggregate_geaf_rows, evaluate_geaf_gates, paired_geaf_delta_rows
from utils.frozen_grace_targets import build_frozen_grace_targets
from utils.ordinary_gb import build_random_size_matched_assignment, build_size_adaptive_ellipsoid_balls


class GeafBridgeTest(unittest.TestCase):
    def test_size_adaptive_builder_has_no_label_inputs(self):
        signature = inspect.signature(build_size_adaptive_ellipsoid_balls)

        self.assertNotIn("noisy_label", signature.parameters)
        self.assertNotIn("clean_label", signature.parameters)
        self.assertNotIn("train_mask", signature.parameters)

    def test_sqrt_stop_rule_limits_max_ball_size(self):
        embeddings = torch.arange(16, dtype=torch.float32).reshape(16, 1)

        result = build_size_adaptive_ellipsoid_balls(
            embeddings,
            n_classes=4,
            random_state=0,
        )

        self.assertEqual(result.summary["target_ball_size"], 2)
        self.assertLessEqual(result.summary["max_ball_size"], 2)
        self.assertEqual(result.assignment.shape[0], 16)

    def test_longest_axis_split_produces_two_nonempty_axis_separated_balls(self):
        x = torch.linspace(-6.0, 6.0, steps=12)
        embeddings = torch.stack([x, torch.zeros_like(x)], dim=1)

        result = build_size_adaptive_ellipsoid_balls(
            embeddings,
            n_classes=1,
            max_balls=2,
            random_state=0,
        )

        self.assertEqual(result.summary["num_balls"], 2)
        sizes = sorted(np.bincount(result.assignment).tolist())
        self.assertEqual(sizes, [6, 6])
        means = []
        for ball_id in sorted(np.unique(result.assignment).tolist()):
            means.append(float(embeddings[result.assignment == ball_id, 0].mean()))
        self.assertGreater(abs(means[0] - means[1]), 4.0)

    def test_shuffled_control_preserves_ball_size_distribution(self):
        embeddings = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        result = build_size_adaptive_ellipsoid_balls(embeddings, n_classes=4, random_state=0)

        shuffled = build_random_size_matched_assignment(result.assignment, random_state=11)

        real_sizes = sorted(np.bincount(result.assignment).tolist())
        shuffled_sizes = sorted(np.bincount(shuffled).tolist())
        self.assertEqual(shuffled_sizes, real_sizes)
        self.assertEqual(shuffled.shape, result.assignment.shape)

    def _row(self, dataset, seed, method, confidence, hard_net, ncr, cdr, zero_ratio=0.0):
        return {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "majority_confidence": confidence,
            "hard_net": hard_net,
            "ncr": ncr,
            "cdr": cdr,
            "zero_train_ball_ratio": zero_ratio,
            "num_balls": 8,
            "target_ball_size": 2,
            "changed_train_ratio": 0.1,
            "geaf_gmean": 0.2,
            "geaf_f1": 0.3,
            "status": "ok",
        }

    def test_paired_deltas_do_not_cross_confidence_settings(self):
        rows = [
            self._row("dblp", 0, "real", 0.0, 0.10, 0.12, 0.02),
            self._row("dblp", 0, "shuffled", 0.0, 0.03, 0.05, 0.02),
            self._row("dblp", 0, "real", 0.7, 0.40, 0.42, 0.02),
            self._row("dblp", 0, "shuffled", 0.6, 0.01, 0.03, 0.02),
        ]

        deltas = paired_geaf_delta_rows(rows, "shuffled", "real")

        self.assertEqual(len(deltas), 1)
        self.assertAlmostEqual(deltas[0]["majority_confidence"], 0.0)
        self.assertAlmostEqual(deltas[0]["hard_net_delta"], 0.07)

    def test_gate_is_evaluated_per_dataset_and_confidence(self):
        rows = []
        for seed in range(5):
            rows.extend(
                [
                    self._row("dblp", seed, "real", 0.6, 0.08, 0.10, 0.02),
                    self._row("dblp", seed, "shuffled", 0.6, 0.02, 0.04, 0.02),
                    self._row("dblp", seed, "real", 0.7, 0.01, 0.04, 0.03),
                    self._row("dblp", seed, "shuffled", 0.7, 0.02, 0.04, 0.02),
                ]
            )

        decisions = evaluate_geaf_gates(rows)

        by_conf = {row["majority_confidence"]: row for row in decisions}
        self.assertTrue(by_conf[0.6]["passed"])
        self.assertEqual(by_conf[0.6]["positive_seed_count"], 5)
        self.assertFalse(by_conf[0.7]["passed"])

    def test_aggregate_groups_by_dataset_confidence_and_method(self):
        rows = [
            self._row("dblp", 0, "real", 0.6, 0.08, 0.10, 0.02),
            self._row("dblp", 1, "real", 0.6, 0.10, 0.12, 0.02),
            self._row("dblp", 0, "real", 0.7, 0.01, 0.04, 0.03),
        ]

        aggregate = aggregate_geaf_rows(rows)

        keys = {(row["dataset"], row["method"], row["majority_confidence"]) for row in aggregate}
        self.assertEqual(keys, {("dblp", "real", 0.6), ("dblp", "real", 0.7)})
        real_06 = next(row for row in aggregate if row["majority_confidence"] == 0.6)
        self.assertEqual(real_06["n"], 2)
        self.assertAlmostEqual(real_06["hard_net_mean"], 0.09)

    def test_frozen_grace_hard_targets_can_use_size_adaptive_ellipsoid_balls(self):
        embeddings = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        noisy = torch.tensor([0, 0, 1, 1] * 4)
        clean = noisy.clone()
        train_mask = np.arange(16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H.pt"
            torch.save(embeddings, path)

            result = build_frozen_grace_targets(
                embedding_path=path,
                noisy_label=noisy,
                clean_label=clean,
                train_mask=train_mask,
                n_classes=4,
                min_train_per_ball=2,
                random_state=0,
                ball_mode="size_adaptive_ellipsoid",
            )

        self.assertEqual(result.diagnostics["ball_mode"], "size_adaptive_ellipsoid")
        self.assertEqual(result.diagnostics["target_ball_size"], 2)
        self.assertLessEqual(result.diagnostics["max_ball_size"], 2)


if __name__ == "__main__":
    unittest.main()
