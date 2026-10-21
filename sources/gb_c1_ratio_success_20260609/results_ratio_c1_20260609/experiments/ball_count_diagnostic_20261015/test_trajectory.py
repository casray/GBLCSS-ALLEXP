"""CPU contract tests for a single nested, label-free ball trajectory."""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import sys
import unittest

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GB-LCSS_code" / "src"))
SPEC = importlib.util.find_spec("gblcss.granular_ball_trajectory")
if SPEC is not None:
    from gblcss.granular_ball_trajectory import build_ball_trajectory


class BallTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(SPEC, "nested ball trajectory implementation is missing")
        self.pool_limit = threadpool_limits(limits=4)
        self.addCleanup(self.pool_limit.restore_original_limits)

    @staticmethod
    def circle(n=1024):
        angles = np.arange(n) * 2 * np.pi / n
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def build(self, z=None, budgets=(1, 8, 16), **kwargs):
        z = self.circle() if z is None else z
        kwargs.setdefault("train_index", np.arange(len(z)))
        return build_ball_trajectory(z, budgets=budgets, **kwargs)

    def test_nested_complete_partitions_with_contiguous_ids(self):
        result = self.build()
        self.assertEqual(list(result.snapshots), [1, 8, 16])
        for budget, assignment in result.snapshots.items():
            self.assertEqual(assignment.shape, (1024,))
            np.testing.assert_array_equal(np.unique(assignment), np.arange(budget))
            self.assertEqual(result.summaries[budget]["requested_target"], budget)
            self.assertEqual(result.summaries[budget]["num_balls"], budget)
            self.assertEqual(result.summaries[budget].get("split_num_threads"), 1)
        for coarse, fine in ((1, 8), (8, 16)):
            for ball in np.unique(result.snapshots[fine]):
                self.assertEqual(np.unique(result.snapshots[coarse][result.snapshots[fine] == ball]).size, 1)

    def test_budget_extension_cannot_change_existing_memberships_or_summaries(self):
        first = self.build(budgets=[1, 8, 16])
        second = self.build(budgets=[32, 16, 8, 1, 8])
        for budget in first.snapshots:
            np.testing.assert_array_equal(first.snapshots[budget], second.snapshots[budget])
            self.assertEqual(first.summaries[budget], second.summaries[budget])
        self.assertEqual(first.split_trace, second.split_trace[:len(first.split_trace)])
        root = first.snapshots[1].copy()
        first.snapshots[16][:] = -1
        np.testing.assert_array_equal(first.snapshots[1], root)

    def test_api_contains_no_class_or_label_input_and_requires_train_index(self):
        signature = inspect.signature(build_ball_trajectory)
        self.assertEqual(list(signature.parameters), ["embeddings", "budgets", "train_index", "min_train_per_ball", "min_leaf_size", "random_state"])
        self.assertIs(signature.parameters["train_index"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            build_ball_trajectory(self.circle(), budgets=[1], train_index=[], labels=np.zeros(1024))

    def test_fixed_support_and_size_floors_with_support_cap(self):
        z = self.circle(256)
        train = np.arange(0, 256, 4)
        result = self.build(z, budgets=[1, 2, 4, 8, 32], train_index=train)
        for budget, assignment in result.snapshots.items():
            summary = result.summaries[budget]
            self.assertEqual(summary["min_leaf_size"], 16)
            self.assertEqual(summary["min_train_per_ball"], 16)
            self.assertEqual(summary["support_count_cap"], 4)
            self.assertLessEqual(summary["num_balls"], 4)
            for ball in np.unique(assignment):
                self.assertGreaterEqual(np.count_nonzero(assignment == ball), 16)
                self.assertGreaterEqual(np.count_nonzero(assignment[train] == ball), 16)
        np.testing.assert_array_equal(result.snapshots[8], result.snapshots[32])
        self.assertEqual(result.summaries[32]["stop_reason"], "support_count_cap_reached")

    def test_sparse_global_support_single_ball_exception(self):
        for train in ([], [0, 2, 4]):
            result = self.build(self.circle(80), budgets=[1, 2, 16], train_index=train)
            for summary in result.summaries.values():
                self.assertEqual(summary["num_balls"], 1)
                self.assertEqual(summary["min_train_count"], len(train))
                self.assertEqual(summary["stop_reason"], "insufficient_training_support")

    def test_local_support_can_stop_below_global_count_ceiling(self):
        train = np.random.RandomState(1).choice(256, 64, replace=False)
        result = self.build(self.circle(256), budgets=[8], train_index=train)
        summary = result.summaries[8]
        self.assertEqual(summary["support_count_cap"], 4)
        self.assertEqual(summary["num_balls"], 3)
        self.assertEqual(summary["stop_reason"], "no_feasible_split")
        rejected = [row for row in result.split_trace if row.get("reason") == "parent_support_constraint"]
        self.assertGreater(len(rejected), 0)
        self.assertTrue(all(row["parent_train_count"] < 32 for row in rejected))

    def test_none_training_membership_disables_only_support_constraint(self):
        result = self.build(self.circle(256), budgets=[1, 4], train_index=None)
        self.assertEqual(result.summaries[4]["num_balls"], 4)
        self.assertIsNone(result.summaries[4]["min_train_count"])
        self.assertIsNone(result.summaries[4]["support_count_cap"])
        self.assertTrue(all(row["child_train_counts"] is None for row in result.split_trace if row["accepted"]))

    def test_identical_embeddings_and_single_node_are_not_forced_to_split(self):
        for z in (np.ones((100, 8)), np.zeros((100, 2)), np.zeros((1, 2))):
            result = self.build(z, budgets=[1, 2, 8])
            self.assertTrue(all(summary["num_balls"] == 1 for summary in result.summaries.values()))
            self.assertTrue(all(summary["total_sse"] < 1e-20 for summary in result.summaries.values()))
            np.testing.assert_array_equal(result.snapshots[1], result.snapshots[8])
        result = self.build(np.ones((100, 8)), budgets=[2])
        self.assertEqual(result.summaries[2]["stop_reason"], "no_feasible_split")
        self.assertEqual(result.split_trace[0]["reason"], "zero_geometric_gain")

    def test_duplicate_groups_and_geometry_cap_are_honest(self):
        result = self.build(np.repeat([[1., 0.], [-1., 0.]], [40, 40], axis=0), budgets=[1, 2, 8])
        self.assertEqual(result.summaries[8]["num_balls"], 2)
        self.assertEqual(result.summaries[8]["stop_reason"], "no_feasible_split")
        result = self.build(self.circle(48), budgets=[1, 2, 8], min_leaf_size=24, min_train_per_ball=1)
        self.assertEqual(result.summaries[8]["geometry_count_cap"], 2)
        self.assertEqual(result.summaries[8]["stop_reason"], "geometry_count_cap_reached")

    def test_projection_fallback_preserves_support(self):
        angles = np.r_[np.linspace(-.3, .3, 96), np.linspace(2.8, 3., 4)]
        z = np.column_stack((np.cos(angles), np.sin(angles)))
        result = self.build(z, budgets=[2], train_index=np.arange(0, 96, 2))
        self.assertEqual(result.summaries[2]["num_balls"], 2)
        accepted = [row for row in result.split_trace if row["accepted"]]
        self.assertEqual(accepted[0]["method"], "projection")
        self.assertGreaterEqual(min(accepted[0]["child_train_counts"]), 16)

    def test_trace_conserves_counts_tracks_best_first_and_monotone_sse(self):
        result = self.build(budgets=[1, 2, 4, 8, 16])
        live = {0: result.summaries[1]["total_sse"]}
        total = live[0]
        count = 1
        for row in result.split_trace:
            self.assertAlmostEqual(row["parent_sse"], max(live.values()), places=8)
            live.pop(row["parent_id"])
            if row["accepted"]:
                count += 1
                self.assertEqual(row["actual_count_after"], count)
                self.assertEqual(sum(row["child_sizes"]), row["parent_size"])
                self.assertEqual(sum(row["child_train_counts"]), row["parent_train_count"])
                self.assertGreater(row["sse_gain"], 0)
                self.assertAlmostEqual(row["sse_gain"], row["parent_sse"] - sum(row["child_sse"]), places=8)
                total -= row["sse_gain"]
                live.update(zip(row["child_ids"], row["child_sse"]))
        self.assertAlmostEqual(total, result.summaries[16]["total_sse"], places=8)
        costs = [summary["total_sse"] for summary in result.summaries.values()]
        self.assertTrue(np.all(np.diff(costs) <= 1e-10))

    def test_determinism_and_training_index_order_invariance(self):
        z = np.random.RandomState(2).normal(size=(400, 8))
        train = np.random.RandomState(8).choice(400, 200, replace=False)
        first = self.build(z, budgets=[1, 4, 8], train_index=train, random_state=3)
        second = self.build(z.copy(), budgets=[1, 4, 8], train_index=train[::-1], random_state=3)
        for budget in first.snapshots:
            np.testing.assert_array_equal(first.snapshots[budget], second.snapshots[budget])
        self.assertEqual(first.split_trace, second.split_trace)
        self.assertEqual(first.summaries, second.summaries)

    def test_invalid_embeddings_indices_and_parameters(self):
        for values in ([], np.zeros((0, 2)), np.zeros((2, 0)), [1, 2], [[np.nan, 1]], [[np.inf, 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_ball_trajectory(values, budgets=[1], train_index=[])
        for indices in ([0, 0], [-1], [1024], [1.5], [[0, 1]], [True, False]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.build(train_index=indices)
        for kwargs in ({"budgets": []}, {"budgets": [0]}, {"budgets": [1.5]}, {"budgets": [True]}, {"budgets": 2}, {"min_leaf_size": 0}, {"min_train_per_ball": 0}, {"random_state": -1}, {"random_state": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_normalization_is_scale_invariant_including_huge_finite_values(self):
        z = np.random.RandomState(4).normal(size=(128, 4))
        first = self.build(z, budgets=[1, 4])
        second = self.build(z * 1e300, budgets=[1, 4])
        np.testing.assert_array_equal(first.snapshots[4], second.snapshots[4])
        self.assertAlmostEqual(first.summaries[4]["total_sse"], second.summaries[4]["total_sse"], places=9)


if __name__ == "__main__":
    unittest.main()
