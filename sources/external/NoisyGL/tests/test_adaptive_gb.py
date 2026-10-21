import unittest

import numpy as np
import torch

from utils.frozen_grace_targets import partition_purity_diagnostics
from utils.ordinary_gb import build_purity_adaptive_granular_balls


def _two_blob_embeddings(n_per_blob=40, dim=8, gap=6.0, seed=0):
    rng = np.random.RandomState(seed)
    left = rng.randn(n_per_blob, dim).astype(np.float32)
    right = rng.randn(n_per_blob, dim).astype(np.float32)
    left[:, 0] -= gap
    right[:, 0] += gap
    return np.vstack([left, right])


class AdaptiveGBTest(unittest.TestCase):
    def test_pure_regions_stay_coarse(self):
        # two well-separated blobs, labels follow blobs exactly -> after one split
        # each ball is pure and splitting must stop at 2 balls
        emb = _two_blob_embeddings()
        n = emb.shape[0]
        labels = torch.tensor([0] * (n // 2) + [1] * (n // 2))
        train_mask = np.arange(n)

        result = build_purity_adaptive_granular_balls(
            emb, noisy_label=labels, train_mask=train_mask, n_classes=2,
            purity_threshold=0.9, min_train_per_child=2, min_train_to_split=4,
            min_ball_size=4, max_balls=64, min_purity_gain=0.05, random_state=0,
        )

        self.assertEqual(result.summary["num_balls"], 2)
        left_ids = set(result.assignment[: n // 2].tolist())
        right_ids = set(result.assignment[n // 2:].tolist())
        self.assertEqual(len(left_ids), 1)
        self.assertEqual(len(right_ids), 1)
        self.assertNotEqual(left_ids, right_ids)

    def test_mixed_region_splits_finer(self):
        # left blob mixes two labels in two separable sub-blobs -> must split finer;
        # right blob is pure -> stays one ball
        rng = np.random.RandomState(1)
        sub_a = rng.randn(20, 8).astype(np.float32)
        sub_b = rng.randn(20, 8).astype(np.float32)
        sub_a[:, 0] -= 8.0
        sub_a[:, 1] -= 3.0
        sub_b[:, 0] -= 8.0
        sub_b[:, 1] += 3.0
        right = rng.randn(40, 8).astype(np.float32)
        right[:, 0] += 8.0
        emb = np.vstack([sub_a, sub_b, right])
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 40)
        train_mask = np.arange(80)

        result = build_purity_adaptive_granular_balls(
            emb, noisy_label=labels, train_mask=train_mask, n_classes=3,
            purity_threshold=0.9, min_train_per_child=2, min_train_to_split=4,
            min_ball_size=4, max_balls=64, min_purity_gain=0.05, random_state=0,
        )

        self.assertGreaterEqual(result.summary["num_balls"], 3)
        # right blob stays whole
        right_ids = set(result.assignment[40:].tolist())
        self.assertEqual(len(right_ids), 1)
        # mixed left region got separated by label structure
        a_ids = set(result.assignment[:20].tolist())
        b_ids = set(result.assignment[20:40].tolist())
        self.assertEqual(a_ids & b_ids, set())

    def test_coverage_guard_blocks_split(self):
        # impure ball but too few train nodes to split -> stays one ball
        emb = _two_blob_embeddings()
        n = emb.shape[0]
        labels = torch.tensor([0] * (n // 2) + [1] * (n // 2))
        train_mask = np.array([0, 1, n - 1])  # only 3 train nodes < min_train_to_split

        result = build_purity_adaptive_granular_balls(
            emb, noisy_label=labels, train_mask=train_mask, n_classes=2,
            purity_threshold=0.9, min_train_per_child=2, min_train_to_split=6,
            min_ball_size=4, max_balls=64, random_state=0,
        )

        self.assertEqual(result.summary["num_balls"], 1)

    def test_only_train_labels_affect_assignment(self):
        # leakage guard: changing non-train labels must not change the partition
        emb = _two_blob_embeddings(seed=2)
        n = emb.shape[0]
        train_mask = np.arange(0, n, 2)  # even nodes are train
        base = torch.tensor([0] * (n // 2) + [1] * (n // 2))
        flipped = base.clone()
        odd = torch.arange(1, n, 2)
        flipped[odd] = (flipped[odd] + 1) % 2  # corrupt every non-train label

        r1 = build_purity_adaptive_granular_balls(
            emb, noisy_label=base, train_mask=train_mask, n_classes=2,
            purity_threshold=0.9, min_train_per_child=2, min_train_to_split=4,
            min_ball_size=4, max_balls=64, random_state=0,
        )
        r2 = build_purity_adaptive_granular_balls(
            emb, noisy_label=flipped, train_mask=train_mask, n_classes=2,
            purity_threshold=0.9, min_train_per_child=2, min_train_to_split=4,
            min_ball_size=4, max_balls=64, random_state=0,
        )

        self.assertTrue(np.array_equal(r1.assignment, r2.assignment))

    def test_partition_purity_diagnostics_values(self):
        noisy = torch.tensor([0, 0, 1, 1, 1, 1])
        clean = torch.tensor([0, 0, 0, 1, 1, 1])
        assignment = np.array([0, 0, 0, 1, 1, 1])
        train_mask = np.arange(6)

        diag = partition_purity_diagnostics(
            noisy_label=noisy, clean_label=clean, assignment=assignment,
            train_mask=train_mask, n_classes=2, min_train_per_ball=2,
        )

        # ball0 noisy counts [2,1] -> purity 2/3; ball1 noisy [0,3] -> 1.0
        self.assertAlmostEqual(diag["noisy_train_purity_weighted"], (2 + 3) / 6)
        # ball0 clean [3,0] -> 1.0; ball1 clean [0,3] -> 1.0
        self.assertAlmostEqual(diag["true_train_purity_weighted"], 1.0)
        self.assertAlmostEqual(diag["true_all_purity_weighted"], 1.0)


if __name__ == "__main__":
    unittest.main()
