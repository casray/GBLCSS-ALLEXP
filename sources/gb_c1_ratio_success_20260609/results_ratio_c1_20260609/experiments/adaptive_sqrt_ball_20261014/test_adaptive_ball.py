"""CPU contract tests for the label-free adaptive partition generator."""
from __future__ import annotations

import importlib.util
import inspect
import math
import os
from pathlib import Path
import sys
import unittest

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GB-LCSS_code" / "src"))
SPEC = importlib.util.find_spec("gblcss.granular_ball_adaptive")
if SPEC is not None:
    from gblcss.granular_ball_adaptive import build_adaptive_granular_balls


class AdaptiveBallTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, "adaptive granular-ball implementation is missing")
        self.pool_limit = threadpool_limits(limits=4)
        self.addCleanup(self.pool_limit.restore_original_limits)

    @staticmethod
    def circle(n=100):
        angles = np.arange(n) * (2 * np.pi / n)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def test_default_sqrt_budget_and_complete_disjoint_coverage(self):
        z = self.circle(100)
        result = build_adaptive_granular_balls(z)
        self.assertEqual(result.summary["requested_target"], math.ceil(math.sqrt(len(z))))
        self.assertEqual(result.summary["num_balls"], 10)
        self.assertEqual(result.summary["stop_reason"], "target_reached")
        np.testing.assert_array_equal(np.unique(result.assignment), np.arange(10))
        self.assertEqual(sum(ball["size"] for ball in result.balls), 100)
        for ball in result.balls:
            self.assertEqual(np.count_nonzero(result.assignment == ball["ball_id"]), ball["size"])
            self.assertGreaterEqual(ball["size"], result.summary["min_leaf_size"])
            self.assertNotIn("train_count", ball)

    def test_reproducible_and_no_labels_or_class_input(self):
        signature = inspect.signature(build_adaptive_granular_balls)
        self.assertEqual(list(signature.parameters), ["embeddings", "target_balls", "train_index", "min_train_per_ball", "random_state"])
        z = np.random.RandomState(1).normal(size=(121, 8))
        first = build_adaptive_granular_balls(z, random_state=4)
        second = build_adaptive_granular_balls(z.copy(), random_state=4)
        np.testing.assert_array_equal(first.assignment, second.assignment)
        self.assertEqual(first.split_trace, second.split_trace)
        self.assertEqual(first.summary, second.summary)

    def test_support_budget_cap_and_support_floor(self):
        result = build_adaptive_granular_balls(self.circle(), train_index=np.arange(0, 100, 5))
        self.assertEqual(result.summary["requested_target"], 10)
        self.assertEqual(result.summary["feasible_target"], 5)
        self.assertLessEqual(result.summary["num_balls"], 5)
        self.assertGreaterEqual(result.summary["num_balls"], 1)
        if result.summary["num_balls"] < 5:
            self.assertEqual(result.summary["stop_reason"], "no_feasible_split")
            self.assertGreater(result.summary["rejected_splits"], 0)
        self.assertTrue(all(ball["train_count"] >= 4 for ball in result.balls))

    def test_insufficient_total_training_support_keeps_one_root(self):
        for train in ([], [0, 1, 2]):
            with self.subTest(train=train):
                result = build_adaptive_granular_balls(self.circle(), train_index=train)
                self.assertEqual(result.summary["feasible_target"], 1)
                self.assertEqual(result.summary["num_balls"], 1)
                self.assertEqual(result.summary["stop_reason"], "insufficient_training_support")
                self.assertEqual(result.balls[0]["train_count"], len(train))

    def test_identical_embeddings_are_never_arbitrarily_split(self):
        result = build_adaptive_granular_balls(np.ones((100, 8)))
        self.assertEqual(result.summary["num_balls"], 1)
        self.assertEqual(result.summary["stop_reason"], "no_feasible_split")
        self.assertGreater(result.summary["rejection_counts"].get("zero_geometric_gain", 0), 0)

    def test_duplicate_projection_boundaries_do_not_force_budget(self):
        z = np.repeat([[1., 0.], [-1., 0.]], [40, 40], axis=0)
        result = build_adaptive_granular_balls(z, target_balls=8)
        self.assertEqual(result.summary["num_balls"], 2)
        self.assertEqual(np.unique(result.assignment[:40]).size, 1)
        self.assertEqual(np.unique(result.assignment[40:]).size, 1)

    def test_projection_fallback_finds_feasible_support_split(self):
        angles = np.r_[np.linspace(-.3, .3, 96), np.linspace(2.8, 3., 4)]
        z = np.column_stack((np.cos(angles), np.sin(angles)))
        result = build_adaptive_granular_balls(z, target_balls=2, train_index=np.arange(0, 96, 8))
        self.assertEqual(result.summary["num_balls"], 2)
        accepted = [row for row in result.split_trace if row["accepted"]]
        self.assertEqual(accepted[0]["method"], "projection")
        self.assertTrue(all(ball["train_count"] >= 4 for ball in result.balls))

        # Independent exhaustive cut check: fallback must minimize SSE over every
        # distinct projection boundary satisfying both frozen constraints.
        model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(z)
        direction = model.cluster_centers_[1] - model.cluster_centers_[0]
        projected = z @ (direction / np.linalg.norm(direction))
        order = np.argsort(projected, kind="stable")
        train = np.zeros(100, dtype=bool)
        train[np.arange(0, 96, 8)] = True
        costs = []
        for cut in range(13, 88):
            if projected[order[cut]] - projected[order[cut - 1]] <= 1e-12:
                continue
            left, right = order[:cut], order[cut:]
            if min(train[left].sum(), train[right].sum()) < 4:
                continue
            costs.append(sum(float(np.square(z[group] - z[group].mean(axis=0)).sum()) for group in (left, right)))
        self.assertAlmostEqual(result.summary["total_sse"], min(costs), places=8)

    def test_every_accepted_split_reduces_sse_and_trace_conserves_totals(self):
        result = build_adaptive_granular_balls(self.circle(), target_balls=10)
        total_sse = float(len(self.circle()))
        live_sse = {0: total_sse}
        for row in result.split_trace:
            self.assertAlmostEqual(row["parent_sse"], max(live_sse.values()), places=8)
            live_sse.pop(row["parent_id"])
            if row["accepted"]:
                self.assertGreater(row["sse_gain"], max(1e-12, 1e-8 * row["parent_sse"]))
                self.assertEqual(sum(row["child_sizes"]), row["parent_size"])
                total_sse -= row["sse_gain"]
                live_sse.update(zip(row["child_ids"], row["child_sse"]))
        self.assertAlmostEqual(total_sse, sum(ball["sse"] for ball in result.balls), places=8)

    def test_invalid_embeddings(self):
        for values in ([], np.zeros((0, 2)), np.zeros((2, 0)), [1, 2], [[np.nan, 1]], [[np.inf, 1]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    build_adaptive_granular_balls(values)

    def test_invalid_indices_and_integer_parameters(self):
        for indices in ([0, 0], [-1], [100], [1.5], [[0, 1]], [True, False]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    build_adaptive_granular_balls(self.circle(), train_index=indices)
        for kwargs in ({"target_balls": 0}, {"target_balls": 2.5}, {"min_train_per_ball": 0}, {"random_state": 1.5}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_adaptive_granular_balls(self.circle(), **kwargs)

    def test_single_node_and_zero_vectors_are_valid_degenerate_input(self):
        for z in (np.zeros((1, 2)), np.zeros((10, 2))):
            with self.subTest(n=len(z)):
                result = build_adaptive_granular_balls(z)
                self.assertEqual(result.summary["num_balls"], 1)
                self.assertEqual(result.balls[0]["sse"], 0.)


if __name__ == "__main__":
    unittest.main()
