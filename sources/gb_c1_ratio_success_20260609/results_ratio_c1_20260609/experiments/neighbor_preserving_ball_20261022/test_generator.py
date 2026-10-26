import itertools
import unittest

import numpy as np
import torch

import generator as g


class GeneratorTests(unittest.TestCase):
    def test_tv_matches_dense_definition(self):
        rng = np.random.default_rng(11)
        for n in range(3, 12):
            for k in range(1, n):
                order = rng.permutation(np.arange(1, n))[:k]
                for s in range(1, n):
                    donors = rng.permutation(np.arange(1, n))[:s]
                    p, w = np.zeros(n), np.zeros(n)
                    p[order], w[donors] = 1/k, 1/s
                    overlap = len(set(order) & set(donors))
                    self.assertAlmostEqual(g.tv_cost(overlap, k, s), .5*np.abs(p-w).sum())

    def test_neighbors_global_id_ties_and_no_self(self):
        x = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
        ids = np.array([80, 2, 6, 1])
        order = g.neighbors(x, ids, 3)
        self.assertEqual(order[0].tolist(), [1, 2, 3])
        self.assertEqual(order[3].tolist(), [1, 2, 0])
        self.assertTrue(all(i not in order[i] for i in range(4)))
        for i in range(4):
            d = ((x-x[i])**2).sum(1)
            expected = sorted((j for j in range(4) if j != i), key=lambda j: (d[j], ids[j]))
            self.assertEqual(order[i].tolist(), expected)

    def test_dp_matches_all_tree_cuts_and_prefers_coarse_ties(self):
        nodes = [dict(children=(1, 2), train_count=8), dict(children=(3, 4), train_count=4),
                 dict(children=(), train_count=4), dict(children=(), train_count=2),
                 dict(children=(), train_count=2)]
        cuts = [[0], [1, 2], [3, 4, 2]]
        for costs in itertools.product([0., 1., 3.], repeat=5):
            chosen, value = g.select_cut(nodes, np.array(costs))
            self.assertAlmostEqual(value, min(sum(costs[i] for i in c) for c in cuts))
            self.assertAlmostEqual(value, sum(costs[i] for i in chosen))
        chosen, _ = g.select_cut(nodes, np.zeros(5))
        self.assertEqual(chosen, [0])

    def test_dp_can_choose_deep_improvement(self):
        nodes = [dict(children=(1, 2), train_count=8), dict(children=(3, 4), train_count=4),
                 dict(children=(), train_count=4), dict(children=(), train_count=2),
                 dict(children=(), train_count=2)]
        chosen, value = g.select_cut(nodes, np.array([8., 9., 2., 1., 1.]))
        self.assertEqual(set(chosen), {2, 3, 4})
        self.assertEqual(value, 4.)

    def test_tree_partition_constraints_and_prefix(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(96, 8))
        train = np.arange(0, 96, 2)
        a = g.build_tree(x, train, max_leaves=8, snapshot_leaves=3, min_nodes=6, min_train=3, seed=0)
        b = g.build_tree(x, train, max_leaves=3, snapshot_leaves=3, min_nodes=6, min_train=3, seed=0)
        np.testing.assert_array_equal(a['snapshot_assignment'], b['fine_assignment'])
        self.assertEqual(len(a['fine_assignment']), 96)
        for leaf in a['fine_leaves']:
            node = a['nodes'][leaf]
            self.assertGreaterEqual(len(node['members']), 6)
            self.assertGreaterEqual(node['train_count'], 3)
        for node in a['nodes']:
            if node['children']:
                l, r = [a['nodes'][j]['members'] for j in node['children']]
                self.assertEqual(len(np.intersect1d(l, r)), 0)
                np.testing.assert_array_equal(np.sort(np.r_[l, r]), node['members'])

    def test_node_cost_independent_dense_reference(self):
        train = np.array([7, 1, 3, 9])
        near = np.array([[1, 2], [0, 3], [1, 3], [0, 2]])
        nodes = [dict(members=np.array([1, 3, 7, 9])), dict(members=np.array([1, 3, 7]))]
        actual = g.node_costs(nodes, train, near)
        for idx, node in enumerate(nodes):
            selected = np.flatnonzero(np.isin(train, node['members']))
            value = 0.
            for row in selected:
                p, w = np.zeros(4), np.zeros(4)
                p[near[row]] = .5
                donors = selected[selected != row]
                w[donors] = 1/len(donors)
                value += .5*np.abs(p-w).sum()
            self.assertAlmostEqual(actual[idx], value)

    def test_targets_ties_insufficient_votes_and_label_permutation(self):
        z = torch.tensor([0, 1, 2, 0])
        counts = torch.tensor([[0., 2., 0.], [1., 0., 1.], [0., 0., 1.], [0., 0., 0.]])
        target, alpha = g.targets_from_counts(z, counts)
        np.testing.assert_allclose(target[0], [0., 1., 0.], atol=1e-7)
        np.testing.assert_allclose(target[1], [.125, .75, .125])
        np.testing.assert_allclose(target[2:], [[0., 0., 1.], [1., 0., 0.]])
        perm = torch.tensor([2, 0, 1])
        swapped = torch.zeros_like(counts)
        swapped[:, perm] = counts
        t2, a2 = g.targets_from_counts(perm[z], swapped)
        torch.testing.assert_close(t2[:, perm], target)
        torch.testing.assert_close(a2, alpha)

    def test_knn_counts_exclude_self_and_match_support(self):
        z = np.array([0, 1, 2, 1])
        near = np.array([[1, 2, 3], [2, 0, 3], [0, 1, 3], [1, 2, 0]])
        support = np.array([2, 1, 3, 0])
        counts = g.neighbor_counts(z, near, support, 3)
        np.testing.assert_array_equal(counts.sum(1), support)
        z2 = z.copy(); z2[0] = 2
        np.testing.assert_array_equal(g.neighbor_counts(z2, near, support, 3)[0], counts[0])
        with self.assertRaises(ValueError):
            bad = near.copy(); bad[0, 0] = 0
            g.neighbor_counts(z, bad, support, 3)

    def test_invalid_inputs_and_zero_geometry(self):
        with self.assertRaises(ValueError):
            g.normalize(np.array([[np.nan, 1.]]))
        with self.assertRaises(ValueError):
            g.neighbors(np.ones((3, 2)), np.arange(3), 3)
        with self.assertRaises(ValueError):
            g.build_tree(np.ones((6, 2)), np.array([0, 0, 1]), min_nodes=2, min_train=2)
        tree = g.build_tree(np.ones((10, 2)), np.arange(6), min_nodes=2, min_train=2)
        self.assertEqual(tree['fine_leaves'], [0])


if __name__ == '__main__':
    unittest.main()
