"""CPU-only checks against the unchanged candidate-tree generator."""
from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

OLD_FOLDER = Path(__file__).resolve().parent.parent / 'neighbor_preserving_ball_20261022'
sys.path.insert(0, str(OLD_FOLDER))
import generator as old_generator

KERNEL_PATH = Path(__file__).resolve().with_name('kernels.py')
KERNEL_SPEC = importlib.util.spec_from_file_location('neighbor_support_ablation_kernels', KERNEL_PATH)
kernels = importlib.util.module_from_spec(KERNEL_SPEC)
KERNEL_SPEC.loader.exec_module(kernels)


def tiny_tree():
    members = [range(8), range(4), range(4, 8), range(2),
               range(2, 4), range(4, 6), range(6, 8)]
    children = [(1, 2), (3, 4), (5, 6), (), (), (), ()]
    sse = [100., 20., 30., 2., 3., 4., 5.]
    return [dict(members=np.array(list(part), dtype=np.int64),
                 children=child, train_count=len(part), sse=error)
            for part, child, error in zip(members, children, sse)]


def all_cuts(nodes, index=0):
    result = [[index]]
    if nodes[index]['children']:
        options = [all_cuts(nodes, child) for child in nodes[index]['children']]
        result.extend([list(itertools.chain.from_iterable(parts))
                       for parts in itertools.product(*options)])
    return result


class SupportCostTests(unittest.TestCase):
    def test_support_formula_and_dtype(self):
        nodes = [dict(train_count=m) for m in (2, 3, 4, 8)]
        result = kernels.support_costs(nodes, 3)
        np.testing.assert_allclose(result, [4/3, 1., 0., 32/7])
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result.dtype, np.dtype('float64'))

    def test_random_algebra_identity(self):
        rng = np.random.default_rng(20261022)
        for _ in range(1000):
            k, support = map(int, rng.integers(1, 100, size=2))
            overlap = int(rng.integers(0, min(k, support)+1))
            m = support+1
            support_term = kernels.support_costs([dict(train_count=m)], k)[0]/m
            membership_term = (min(k, support)-overlap)/max(k, support)
            self.assertAlmostEqual(support_term+membership_term,
                                   float(old_generator.tv_cost(overlap, k, support)),
                                   places=14)

    def test_support_dp_matches_brute_force(self):
        nodes = tiny_tree()
        for k in range(1, 12):
            with self.subTest(k=k):
                costs = kernels.support_costs(nodes, k)
                leaves, value = old_generator.select_cut(nodes, costs)
                optimum = min(sum(costs[j] for j in cut) for cut in all_cuts(nodes))
                self.assertAlmostEqual(value, optimum, places=12)
                self.assertAlmostEqual(sum(costs[j] for j in leaves), optimum, places=12)
                old_generator.assignment(nodes, leaves, 8)

    def test_support_dp_expected_cuts_and_tie(self):
        nodes = tiny_tree()
        for k, expected in [(1, [3, 4, 5, 6]), (3, [1, 2]), (7, [0])]:
            self.assertEqual(old_generator.select_cut(
                nodes, kernels.support_costs(nodes, k))[0], expected)
        tied = [dict(train_count=4, children=(1, 2)),
                dict(train_count=2, children=()), dict(train_count=2, children=())]
        self.assertEqual(old_generator.select_cut(
            tied, kernels.support_costs(tied, 5))[0], [0])

    def test_invalid_k_and_train_count(self):
        for bad in (0, -1, 1.5, np.nan, np.inf, True, '2', 1+0j, [2]):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                kernels.support_costs([dict(train_count=2)], bad)
        for bad in (0, 1, -2, 2.5, np.nan, np.inf, True, '3', 2+0j, [2]):
            with self.subTest(m=bad), self.assertRaises(ValueError):
                kernels.support_costs([dict(train_count=bad)], 2)
        for bad in ([], [{}], None):
            with self.subTest(nodes=bad), self.assertRaises(ValueError):
                kernels.support_costs(bad, 2)


class ComponentTests(unittest.TestCase):
    def test_components_agree_with_old_costs_and_rows(self):
        rng = np.random.default_rng(913)
        nodes = tiny_tree()
        # Global IDs differ from row indices and training order is deliberately shuffled.
        mapping = np.array([20, 7, 30, 1, 15, 40, 3, 90])
        for node in nodes:
            node['members'] = np.append(mapping[node['members']], 100+len(node['members']))
        train = mapping[[5, 0, 7, 2, 1, 6, 3, 4]]
        for k in (1, 2, 3, 7):
            near = np.array([rng.choice(np.delete(np.arange(8), row), k, replace=False)
                             for row in range(8)])
            support, membership = kernels.cost_components(nodes, train, near)
            np.testing.assert_allclose(support, kernels.support_costs(nodes, k), atol=1e-14)
            np.testing.assert_allclose(support+membership,
                                       old_generator.node_costs(nodes, train, near),
                                       atol=1e-14, rtol=1e-14)
            for idx, node in enumerate(nodes):
                selected = np.isin(train, node['members'])
                rows = np.flatnonzero(selected)
                s = len(rows)-1
                overlaps = selected[near[rows]].sum(axis=1)
                expected = sum((min(s, k)-int(overlap))/max(s, k)
                               for overlap in overlaps)
                self.assertAlmostEqual(membership[idx], expected, places=14)
            self.assertTrue((membership >= 0).all())

    def test_perfect_membership_has_zero_membership_cost(self):
        nodes = tiny_tree()
        near = np.array([[row ^ 1] for row in range(8)])
        support, membership = kernels.cost_components(nodes, np.arange(8), near)
        np.testing.assert_array_equal(membership, np.zeros(7))
        np.testing.assert_allclose(support+membership,
                                   old_generator.node_costs(nodes, np.arange(8), near))

    def test_all_components_match_random_valid_overlap(self):
        rng = np.random.default_rng(329)
        for _ in range(100):
            size = int(rng.integers(4, 25))
            k = int(rng.integers(1, size))
            m = int(rng.integers(2, size+1))
            train = rng.permutation(np.arange(size)*3+10)
            node = dict(members=rng.choice(train, m, replace=False), train_count=m)
            near = np.array([rng.choice(np.delete(np.arange(size), row), k, replace=False)
                             for row in range(size)])
            support, membership = kernels.cost_components([node], train, near)
            np.testing.assert_allclose(support+membership,
                                       old_generator.node_costs([node], train, near),
                                       atol=1e-13, rtol=1e-14)

    def test_invalid_train_or_near(self):
        nodes = tiny_tree()
        train = np.arange(8)
        near = np.array([[row ^ 1] for row in range(8)])
        bad_inputs = [([], near), (train.reshape(2, 4), near),
                      (np.zeros(8, dtype=int), near), (train+.5, near),
                      (train.astype(float)+np.nan, near), (-train-1, near),
                      (train, []), (train, near.ravel()), (train, near[:7]),
                      (train, np.empty((8, 0), dtype=int)), (train, near+.5),
                      (train, near.astype(float)+np.nan),
                      (train, np.full((8, 1), -1)), (train, np.full((8, 1), 8)),
                      (train, np.arange(8)[:, None]), (train, np.repeat(near, 2, axis=1)),
                      (train.astype(str), near), (train, near.astype(str)),
                      (train.astype(bool), near), (train, near.astype(bool))]
        for bad_train, bad_near in bad_inputs:
            with self.subTest(train=bad_train, near=bad_near), self.assertRaises(ValueError):
                kernels.cost_components(nodes, bad_train, bad_near)

    def test_invalid_node_membership_or_count(self):
        for field, value in [('members', []), ('members', [0, 0]),
                             ('members', [0, 1.5]), ('members', [0, np.nan]),
                             ('members', [-1, 0]), ('train_count', 9),
                             ('train_count', 2.5)]:
            nodes = tiny_tree()
            nodes[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                kernels.cost_components(nodes, np.arange(8),
                                        np.array([[row ^ 1] for row in range(8)]))


class CountMatchedCutTests(unittest.TestCase):
    def test_each_feasible_count_is_a_complete_cut(self):
        nodes = tiny_tree()
        original = copy.deepcopy(nodes)
        for count in range(1, 5):
            cut = kernels.count_matched_cut(nodes, count)
            self.assertEqual(len(cut), count)
            self.assertEqual(cut, sorted(cut))
            self.assertIn(cut, [sorted(c) for c in all_cuts(nodes)])
            assignment = old_generator.assignment(nodes, cut, 8)
            self.assertEqual(len(np.unique(assignment)), count)
        for before, after in zip(original, nodes):
            np.testing.assert_array_equal(before['members'], after['members'])
            self.assertEqual(before['children'], after['children'])
            self.assertEqual(before['sse'], after['sse'])

    def test_sse_gain_has_priority_over_size_and_member_id(self):
        # Node 2 has gain 21; node 1 has gain 15 despite its smaller member ID.
        self.assertEqual(kernels.count_matched_cut(tiny_tree(), 3), [1, 5, 6])

    def test_equal_gain_uses_smallest_global_member_deterministically(self):
        nodes = tiny_tree()
        nodes[2]['sse'] = 24.
        for node in nodes:
            node['members'] = 7-node['members']
            node['children'] = tuple(reversed(node['children']))
        # Node 2 now contains global ID 0, although its first member is 3.
        for _ in range(10):
            self.assertEqual(kernels.count_matched_cut(nodes, 3), [1, 5, 6])

    def test_only_available_splits_are_considered(self):
        nodes = tiny_tree()
        nodes[0]['sse'] = 2000.
        nodes[3]['children'] = (7, 8)
        nodes[3]['sse'] = 500.
        nodes.extend([dict(members=np.array([j]), children=(), sse=0., train_count=1)
                      for j in (0, 1)])
        nodes[1]['sse'] = 510.  # Gain 7 versus node 2's gain 21.
        self.assertEqual(kernels.count_matched_cut(nodes, 3), [1, 5, 6])
        self.assertEqual(kernels.count_matched_cut(nodes, 4), [3, 4, 5, 6])
        self.assertEqual(kernels.count_matched_cut(nodes, 5), [4, 5, 6, 7, 8])

    def test_zero_and_negative_gains_still_reach_requested_count(self):
        nodes = tiny_tree()
        for node in nodes:
            node['sse'] = 0.
        self.assertEqual(kernels.count_matched_cut(nodes, 3), [2, 3, 4])
        nodes[3]['sse'] = 2.
        self.assertEqual(kernels.count_matched_cut(nodes, 3), [1, 5, 6])

    def test_invalid_or_infeasible_count(self):
        for bad in (0, -1, 5, 1.5, np.nan, np.inf, True, '2', 2+0j, [2]):
            with self.subTest(count=bad), self.assertRaises(ValueError):
                kernels.count_matched_cut(tiny_tree(), bad)
        with self.assertRaises(ValueError):
            kernels.count_matched_cut([], 1)

    def test_malformed_numeric_tree_inputs(self):
        for field, value in [('sse', np.nan), ('sse', np.inf), ('sse', -1.),
                             ('sse', '1'), ('sse', True), ('sse', 1+0j),
                             ('sse', [1.]), ('members', [0, np.nan]),
                             ('members', [0, .5]), ('members', [0, 0]),
                             ('members', []), ('members', [-1, 0]),
                             ('children', (1.5, 2)), ('children', (1, 100)),
                             ('children', (True, 2)), ('children', (1, 1)),
                             ('children', (0, 2)), ('children', (1,))]:
            nodes = tiny_tree()
            nodes[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                kernels.count_matched_cut(nodes, 1)

    def test_rejects_non_tree_and_invalid_partitions(self):
        bad_trees = []
        nodes = tiny_tree()
        nodes[2]['children'] = (3, 4)
        bad_trees.append(nodes)
        nodes = tiny_tree()
        nodes[3]['members'] = np.array([0, 2])
        bad_trees.append(nodes)
        nodes = tiny_tree()
        nodes.append(dict(members=np.array([99]), children=(), sse=0.))
        bad_trees.append(nodes)
        for nodes in bad_trees:
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                kernels.count_matched_cut(nodes, 1)


if __name__ == '__main__':
    unittest.main()
