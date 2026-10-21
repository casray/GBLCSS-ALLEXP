from __future__ import annotations

from collections import deque
from importlib import import_module
import inspect
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GB-LCSS_code" / "src"))


def edges(pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T


def reference_partition(edge_index, n_nodes, capacity):
    """Small exhaustive oracle independent of the production degree heap."""
    neighbors = [set() for _ in range(n_nodes)]
    for left, right in edge_index.T:
        if left != right:
            neighbors[left].add(int(right))
            neighbors[right].add(int(left))
    remaining = set(range(n_nodes))
    assignment = np.full(n_nodes, -1, dtype=np.int64)
    metadata = []
    while remaining:
        degrees = {node: len(neighbors[node] & remaining) for node in remaining}
        center = min(remaining, key=lambda node: (-degrees[node], node))
        queue = deque([(center, 0)])
        enqueued = {center}
        members = []
        max_depth = 0
        while queue and len(members) < capacity:
            node, depth = queue.popleft()
            members.append(node)
            max_depth = depth
            for other in sorted(neighbors[node] & remaining):
                if other not in enqueued:
                    enqueued.add(other)
                    queue.append((other, depth + 1))
        ball_id = len(metadata)
        assignment[members] = ball_id
        remaining.difference_update(members)
        metadata.append(dict(ball_id=ball_id, size=len(members), center=center,
                             center_residual_degree=degrees[center], max_depth=max_depth))
    return assignment, metadata


class TopologyContracts(unittest.TestCase):
    def setUp(self):
        try:
            self.module = import_module("gblcss.granular_ball_topology")
        except ModuleNotFoundError as error:
            self.fail(f"Topology implementation is missing: {error}")

    def test_api_has_no_labels_features_or_train_inputs(self):
        signature = inspect.signature(self.module.build_topology_balls)
        self.assertEqual(list(signature.parameters), ["edge_index", "num_nodes", "max_ball_size"])
        self.assertEqual(signature.parameters["max_ball_size"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(signature.parameters["max_ball_size"].default)
        self.assertEqual(list(inspect.signature(self.module.random_support_matched_assignment).parameters),
                         ["assignment", "train_index", "random_state"])

    def test_residual_degree_centers_and_unassigned_queue_at_capacity(self):
        graph = edges([(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6), (5, 7)])
        result = self.module.build_topology_balls(graph, 8, max_ball_size=3)
        np.testing.assert_array_equal(result.assignment, [0, 0, 0, 2, 1, 1, 1, 3])
        self.assertEqual([ball["center"] for ball in result.balls], [0, 5, 3, 7])
        self.assertEqual([ball["center_residual_degree"] for ball in result.balls], [4, 3, 0, 0])
        self.assertEqual([ball["max_depth"] for ball in result.balls], [1, 1, 0, 0])

    def test_fifo_sorted_neighbors_and_depth(self):
        graph = edges([(0, 4), (0, 2), (0, 1), (1, 5), (2, 3), (4, 6)])
        result = self.module.build_topology_balls(graph, 7, max_ball_size=5)
        self.assertEqual(np.flatnonzero(result.assignment == 0).tolist(), [0, 1, 2, 4, 5])
        self.assertEqual(result.balls[0]["max_depth"], 2)

    def test_disconnected_components_and_isolated_nodes_are_not_merged(self):
        graph = edges([(0, 1), (1, 2), (3, 4), (6, 6)])
        result = self.module.build_topology_balls(graph, 7, max_ball_size=7)
        np.testing.assert_array_equal(result.assignment, [0, 0, 0, 1, 1, 2, 3])
        self.assertEqual(result.summary, dict(num_balls=4, max_ball_size=7, num_nodes=7,
                         canonical_edges=3, num_components=4, isolated_nodes=2,
                         min_size=1, max_size=3, singleton_balls=2))

    def test_duplicate_reverse_self_loop_and_shuffled_edges_are_invariant(self):
        graph = edges([(0, 1), (0, 2), (0, 3), (2, 4), (4, 5), (5, 6)])
        dirty = np.concatenate([graph, graph[::-1], graph, edges([(0, 0), (4, 4)])], axis=1)
        dirty = dirty[:, np.random.RandomState(31).permutation(dirty.shape[1])]
        left = self.module.build_topology_balls(graph, 8)
        right = self.module.build_topology_balls(dirty, 8)
        np.testing.assert_array_equal(left.assignment, right.assignment)
        self.assertEqual(left.balls, right.balls)
        self.assertEqual(left.summary, right.summary)
        self.assertEqual(left.summary["canonical_edges"], 6)

    def test_default_capacity_is_integer_square_root(self):
        result = self.module.build_topology_balls(edges([(0, node) for node in range(1, 10)]), 10)
        self.assertEqual(result.summary["max_ball_size"], math.isqrt(10))
        self.assertTrue(all(ball["size"] <= 3 for ball in result.balls))
        self.assertGreater(result.summary["num_balls"], math.isqrt(10))

    def test_no_edges_and_single_node(self):
        result = self.module.build_topology_balls(np.empty((2, 0), dtype=np.int64), 5)
        np.testing.assert_array_equal(result.assignment, np.arange(5))
        self.assertEqual(result.summary["num_components"], 5)
        self.assertEqual(result.summary["isolated_nodes"], 5)
        self.assertEqual(result.summary["singleton_balls"], 5)
        self.assertTrue(all(ball["center_residual_degree"] == ball["max_depth"] == 0
                            for ball in result.balls))
        single = self.module.build_topology_balls(edges([(0, 0)]), 1)
        self.assertEqual(single.summary["max_ball_size"], 1)
        np.testing.assert_array_equal(single.assignment, [0])

    def test_cpu_torch_matches_numpy(self):
        graph = edges([(0, 1), (1, 2), (2, 3)])
        left = self.module.build_topology_balls(graph, np.int64(5), max_ball_size=np.int64(2))
        right = self.module.build_topology_balls(torch.from_numpy(graph), 5, max_ball_size=2)
        np.testing.assert_array_equal(left.assignment, right.assignment)
        self.assertEqual(left.balls, right.balls)
        self.assertEqual(left.assignment.dtype, np.dtype("int64"))
        self.assertIsInstance(left, self.module.TopologyBallResult)

    def test_rejects_invalid_node_count_and_capacity(self):
        graph = edges([(0, 1)])
        for value in [0, -1, 2.0, True, np.bool_(False), "2"]:
            with self.subTest(num_nodes=value), self.assertRaises(ValueError):
                self.module.build_topology_balls(graph, value)
        for value in [0, -1, 2.0, True, "2"]:
            with self.subTest(max_ball_size=value), self.assertRaises(ValueError):
                self.module.build_topology_balls(graph, 2, max_ball_size=value)

    def test_rejects_malformed_noninteger_or_out_of_bounds_edges(self):
        invalid = [np.array([0, 1]), np.zeros((3, 2), dtype=int),
                   np.array([[0.0], [1.0]]), np.zeros((2, 0), dtype=float),
                   np.array([[False], [True]]), edges([(-1, 0)]), edges([(0, 3)]),
                   np.array([[0], [2**64 - 1]], dtype=np.uint64)]
        for graph in invalid:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                self.module.build_topology_balls(graph, 3)

    def test_sparse_tensor_is_rejected_explicitly(self):
        graph = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1]), (2, 1))
        with self.assertRaises(ValueError):
            self.module.build_topology_balls(graph, 3)

    def test_small_random_graphs_match_exhaustive_oracle_and_connected_coverage(self):
        rng = np.random.RandomState(19)
        for n_nodes in range(1, 22):
            pairs = [(left, right) for left in range(n_nodes) for right in range(left + 1, n_nodes)
                     if rng.rand() < .16]
            graph = edges(pairs)
            adjacency = [set() for _ in range(n_nodes)]
            for left, right in pairs:
                adjacency[left].add(right)
                adjacency[right].add(left)
            for capacity in [1, math.isqrt(n_nodes), n_nodes + 1]:
                with self.subTest(n_nodes=n_nodes, capacity=capacity):
                    expected, metadata = reference_partition(graph, n_nodes, capacity)
                    actual = self.module.build_topology_balls(graph, n_nodes, max_ball_size=capacity)
                    np.testing.assert_array_equal(actual.assignment, expected)
                    self.assertEqual(actual.balls, metadata)
                    self.assertEqual(sum(ball["size"] for ball in actual.balls), n_nodes)
                    for ball in actual.balls:
                        members = set(np.flatnonzero(actual.assignment == ball["ball_id"]).tolist())
                        self.assertEqual(len(members), ball["size"])
                        self.assertLessEqual(len(members), capacity)
                        reached = {ball["center"]}
                        frontier = list(reached)
                        while frontier:
                            node = frontier.pop()
                            added = (adjacency[node] & members) - reached
                            reached.update(added)
                            frontier.extend(added)
                        self.assertEqual(reached, members)

    def test_random_control_preserves_sizes_and_training_support_exactly(self):
        assignment = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=np.int64)
        train = np.array([8, 0, 3, 6, 2], dtype=np.int64)
        original = assignment.copy()
        for seed in range(5):
            randomized = self.module.random_support_matched_assignment(assignment, train, random_state=seed)
            self.assertEqual(randomized.dtype, np.dtype("int64"))
            for ball_id in range(4):
                self.assertEqual(int(np.sum(assignment == ball_id)), int(np.sum(randomized == ball_id)))
                self.assertEqual(int(np.sum(assignment[train] == ball_id)),
                                 int(np.sum(randomized[train] == ball_id)))
            again = self.module.random_support_matched_assignment(assignment, train[::-1], random_state=seed)
            np.testing.assert_array_equal(randomized, again)
        np.testing.assert_array_equal(assignment, original)
        self.assertFalse(np.array_equal(self.module.random_support_matched_assignment(assignment, train, 0),
                                        self.module.random_support_matched_assignment(assignment, train, 1)))

    def test_random_control_empty_or_all_training_and_nondense_ball_ids(self):
        assignment = np.array([4, 4, 9, 9, 9], dtype=np.int64)
        for train in [np.array([], dtype=np.int64), torch.arange(5)]:
            randomized = self.module.random_support_matched_assignment(assignment, train)
            self.assertEqual(sorted(randomized.tolist()), sorted(assignment.tolist()))

    def test_random_control_rejects_invalid_assignment_and_train_positions(self):
        for assignment in [np.array([[0, 1]]), np.array([0., 1.]), np.array([-1, 0]),
                           np.array([True, False])]:
            with self.subTest(assignment=assignment), self.assertRaises(ValueError):
                self.module.random_support_matched_assignment(assignment, np.array([], dtype=np.int64))
        assignment = np.array([0, 0, 1])
        for train in [np.array([0, 0]), np.array([-1]), np.array([3]), np.array([0.]),
                      np.array([[0]]), np.array([True, False, True])]:
            with self.subTest(train=train), self.assertRaises(ValueError):
                self.module.random_support_matched_assignment(assignment, train)


if __name__ == "__main__":
    unittest.main()
