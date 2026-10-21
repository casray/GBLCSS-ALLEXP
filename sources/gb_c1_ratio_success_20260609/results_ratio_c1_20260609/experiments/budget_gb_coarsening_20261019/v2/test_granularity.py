"""Behavioral checks for unlabeled trees and exact fixed-cardinality cuts."""
import itertools
import unittest
import warnings

import numpy as np

try:
    from granularity import BallTree, build_tree, select_cut
except ImportError:
    BallTree = build_tree = select_cut = None


def synthetic_tree():
    members = [np.arange(6), np.arange(3), np.arange(3, 6),
               np.array([0]), np.array([1, 2]), np.array([3]), np.array([4, 5]),
               np.array([1]), np.array([2]), np.array([4]), np.array([5])]
    children = np.array([[1, 2], [3, 4], [5, 6], [-1, -1], [7, 8],
                         [-1, -1], [9, 10], [-1, -1], [-1, -1], [-1, -1], [-1, -1]])
    parent = np.array([-1, 0, 0, 1, 1, 2, 2, 4, 4, 6, 6])
    costs = np.array([12., 4., 5., 0., 2., 0., 3., 0., 0., 0., 0.])
    return BallTree(members, children, parent, costs)


def enumerate_cuts(tree, node=0):
    yield [node]
    left, right = tree.children[node]
    if left >= 0:
        for a, b in itertools.product(enumerate_cuts(tree, left), enumerate_cuts(tree, right)):
            yield a + b


def independent_label_costs(tree, labels, mask, n_classes, beta):
    total = int(mask.sum())
    impurities = np.empty(len(tree.members))
    cost = np.zeros(len(tree.members))
    for node, members in enumerate(tree.members):
        supported = members[mask[members]]
        n = len(supported)
        if n >= 2:
            counts = np.bincount(labels[supported], minlength=n_classes)
            impurities[node] = 1. - np.sum(counts * (counts - 1)) / (n * (n - 1))
        else:
            impurities[node] = impurities[tree.parent[node]] if node else 1. - 1. / n_classes
        if total:
            cost[node] = beta * n / total * impurities[node] / (1. - 1. / n_classes)
    return cost


class GranularityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(build_tree), "granularity implementation is missing")

    def test_exact_dp_matches_exhaustive_cut_search_r0_r3(self):
        tree = synthetic_tree()
        labels = np.array([0, 1, 0, 1, 1, 0])
        mask = np.array([True, True, False, True, True, False])
        for rule in ["R0", "R3", "O2"]:
            label_costs = independent_label_costs(tree, labels, mask, 2, .7)
            costs = tree.costs + (label_costs if rule != "R0" else 0)
            for budget in range(1, 7):
                with self.subTest(rule=rule, budget=budget):
                    assignment, diag = select_cut(tree, budget, labels, mask, 2, rule, .7)
                    optimum = min(sum(costs[c]) for c in enumerate_cuts(tree) if len(c) == budget)
                    self.assertAlmostEqual(diag["objective_total"], optimum)
                    self.assertEqual(diag["actual_M"], budget)
                    np.testing.assert_array_equal(np.unique(assignment), np.arange(budget))
                    selected = diag["selected_nodes"]
                    self.assertEqual(sorted(np.concatenate([tree.members[n] for n in selected]).tolist()), list(range(6)))

    def test_held_out_labels_never_affect_selection(self):
        tree = synthetic_tree()
        mask = np.array([True, False, True, False, False, True])
        a = np.array([0, 0, 1, 0, 0, 1])
        b = a.copy()
        b[~mask] = 98765
        aa, da = select_cut(tree, 4, a, mask, 2, "R3")
        ab, db = select_cut(tree, 4, b, mask, 2, "R3")
        np.testing.assert_array_equal(aa, ab)
        self.assertEqual(da["objective_total"], db["objective_total"])

    def test_r0_does_not_read_labels_or_mask(self):
        tree = synthetic_tree()
        a, da = select_cut(tree, 3, None, None, 2, "R0")
        b, db = select_cut(tree, 3, np.full(6, 1234), np.ones(6, dtype=bool), 2, "R0")
        np.testing.assert_array_equal(a, b)
        self.assertEqual(da["label_total"], 0.)
        self.assertEqual(da["objective_total"], db["objective_total"])

    def test_singleton_inherits_mixed_parent_without_purity_reward(self):
        tree = BallTree([np.arange(2), np.array([0]), np.array([1])],
                        np.array([[1, 2], [-1, -1], [-1, -1]]),
                        np.array([-1, 0, 0]), np.zeros(3))
        for budget in [1, 2]:
            _, diag = select_cut(tree, budget, np.array([0, 1]), np.ones(2, dtype=bool), 2, "R3")
            self.assertAlmostEqual(diag["label_total"], 2.)
        _, diag = select_cut(tree, 2, np.array([0, 1]), np.ones(2, dtype=bool), 2, "R3")
        self.assertEqual(diag["inherited_label_fraction"], 1.)

    def test_labeled_support_weights_ignore_unlabeled_members(self):
        tree = synthetic_tree()
        mask = np.array([True, True, False, False, False, False])
        _, diag = select_cut(tree, 2, np.array([0, 1, 9, 9, 9, 9]), mask, 2, "R3")
        self.assertAlmostEqual(diag["label_total"], 2.)
        self.assertEqual(diag["label_support_histogram"], {"0": 1, "1": 0, "2plus": 1})

    def test_global_low_support_fallback_is_explicit(self):
        tree = synthetic_tree()
        mask = np.zeros(6, dtype=bool)
        mask[0] = True
        _, diag = select_cut(tree, 6, np.zeros(6, dtype=int), mask, 2, "R3")
        self.assertEqual(diag["global_label_fallback"], "uniform_class_prior")
        self.assertAlmostEqual(diag["label_total"], 1.)
        mask[:] = False
        _, diag = select_cut(tree, 6, np.zeros(6, dtype=int), mask, 2, "R3")
        self.assertEqual(diag["label_total"], 0.)

    def test_build_is_deterministic_unlabeled_and_costs_are_sse(self):
        rng = np.random.default_rng(32)
        z = rng.normal(size=(40, 3))
        a = build_tree(z, 12, min_leaf=1)
        b = build_tree(z, 12, min_leaf=1)
        np.testing.assert_array_equal(a.children, b.children)
        for x, y, cost in zip(a.members, b.members, a.costs):
            np.testing.assert_array_equal(x, y)
            self.assertAlmostEqual(cost, float(np.sum((z[x] - z[x].mean(axis=0)) ** 2)))
        self.assertEqual(np.sum(a.children[:, 0] < 0), 12)
        self.assertTrue(np.all(a.parent[1:] < np.arange(1, len(a.members))))
        snapshots = [x.copy() for x in a.members]
        costs = a.costs.copy()
        select_cut(a, 5, None, None, 2, "R0")
        for x, snapshot in zip(a.members, snapshots):
            np.testing.assert_array_equal(x, snapshot)
        np.testing.assert_array_equal(a.costs, costs)

    def test_degenerate_kmeans_uses_balanced_fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = build_tree(np.zeros((16, 2)), 8, min_leaf=2)
        leaves = [m for m, c in zip(tree.members, tree.children) if c[0] < 0]
        self.assertEqual(len(leaves), 8)
        self.assertTrue(all(len(m) == 2 for m in leaves))
        assignment, diag = select_cut(tree, 8, None, None, 2, "R0")
        self.assertEqual(diag["actual_M"], 8)
        self.assertEqual(len(assignment), 16)

    def test_insufficient_candidate_budget_raises(self):
        tree = synthetic_tree()
        with self.assertRaisesRegex(ValueError, "available|leaves"):
            select_cut(tree, 7, None, None, 2, "R0")
        with self.assertRaises(ValueError):
            select_cut(tree, 0, None, None, 2, "R0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
