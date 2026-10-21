"""Independent numerical checks for the fixed, label-free structural views."""

import inspect
import unittest

import numpy as np
import torch

from features import structural_views


class StructuralViewsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[3., 0.], [0., 4.], [3., 4.]], dtype=np.float32)
        self.edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    def test_hand_calculated_chain_uses_raw_features_before_residual_normalization(self):
        views = structural_views(self.x, self.edges, block_rows=1)
        expected_u = np.array([[1., 0.], [0., 1.], [.6, .8]], dtype=np.float32)
        expected_v = np.array([[-.6, .8], [3 / np.sqrt(13), -2 / np.sqrt(13)], [-1., 0.]], dtype=np.float32)
        np.testing.assert_allclose(views['X'], expected_u, atol=1e-7)
        np.testing.assert_allclose(views['residual'], expected_v, atol=1e-7)
        np.testing.assert_allclose(views['R'], np.concatenate([expected_u, expected_v], axis=1) / np.sqrt(2), atol=1e-7)

    def test_asymmetric_duplicate_edges_and_selfloops_do_not_change_graph(self):
        dirty = np.array([[0, 0, 1, 1, 2, 2, 0], [1, 1, 0, 2, 1, 2, 0]])
        expected = structural_views(self.x, self.edges)
        actual = structural_views(self.x, dirty)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_zero_features_isolated_nodes_and_zero_residuals_stay_finite(self):
        x = np.array([[0., 0.], [3., 4.], [2., 0.], [2., 0.]], dtype=np.float32)
        result = structural_views(x, np.array([[2], [3]], dtype=np.int64))
        np.testing.assert_array_equal(result['X'][0], [0., 0.])
        np.testing.assert_allclose(result['residual'][1], [-.6, -.8], atol=1e-7)
        np.testing.assert_array_equal(result['residual'][[0, 2, 3]], np.zeros((3, 2)))
        for value in result.values():
            self.assertTrue(np.isfinite(value).all())

    def test_empty_graph_has_negative_normalized_features(self):
        result = structural_views(self.x, np.empty((2, 0), dtype=np.int64))
        np.testing.assert_allclose(result['residual'], -result['X'], atol=1e-7)

    def test_constant_attributes_with_three_neighbors_have_zero_residual(self):
        x = np.tile(np.array([.1, 7., 3.], dtype=np.float32), (4, 1))
        edges = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
        result = structural_views(x, edges, block_rows=1)
        np.testing.assert_array_equal(result['residual'], np.zeros_like(x))

    def test_shuffle_preserves_residual_rows_and_original_feature_alignment(self):
        result = structural_views(self.x, self.edges, shuffle_seed=38)
        np.testing.assert_array_equal(np.sort(result['permutation']), np.arange(3))
        np.testing.assert_allclose(result['SHUFFLE'][:, :2], result['X'] / np.sqrt(2), atol=1e-7)
        np.testing.assert_allclose(result['SHUFFLE'][:, 2:], result['residual'][result['permutation']] / np.sqrt(2), atol=1e-7)
        np.testing.assert_allclose(np.sort(result['SHUFFLE'][:, 2:], axis=0), np.sort(result['R'][:, 2:], axis=0), atol=1e-7)
        repeated = structural_views(self.x, self.edges, shuffle_seed=38)
        np.testing.assert_array_equal(result['SHUFFLE'], repeated['SHUFFLE'])
        expected_permutation = np.random.default_rng(38).permutation(3)
        np.testing.assert_array_equal(result['permutation'], expected_permutation)

    def test_outputs_are_float32_and_permutation_is_int64(self):
        result = structural_views(self.x.astype(np.float64), self.edges)
        self.assertEqual(set(result), {'X', 'R', 'SHUFFLE', 'residual', 'permutation'})
        for key in ['X', 'R', 'SHUFFLE', 'residual']:
            self.assertEqual(result[key].dtype, np.float32)
        self.assertEqual(result['permutation'].dtype, np.int64)
        self.assertEqual(result['R'].shape, (3, 4))

    def test_torch_inputs_detach_without_mutating_inputs(self):
        x = torch.tensor(self.x, requires_grad=True)
        edges = torch.tensor(self.edges)
        result = structural_views(x, edges)
        np.testing.assert_array_equal(x.detach().numpy(), self.x)
        np.testing.assert_array_equal(edges.numpy(), self.edges)
        expected = structural_views(self.x, self.edges)
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])
        result['X'][0, 0] = 7
        self.assertEqual(x[0, 0].item(), 3)

    def test_block_size_does_not_change_views(self):
        reference = structural_views(self.x, self.edges, block_rows=1)
        for block in [2, 10]:
            actual = structural_views(self.x, self.edges, block_rows=block)
            for key in reference:
                np.testing.assert_array_equal(actual[key], reference[key])

    def test_api_has_no_labels_or_masks(self):
        parameters = inspect.signature(structural_views).parameters
        self.assertEqual(list(parameters), ['x', 'edge_index', 'shuffle_seed', 'block_rows'])
        self.assertEqual(parameters['shuffle_seed'].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters['shuffle_seed'].default, 20261015)
        self.assertEqual(parameters['block_rows'].default, 1024)

    def test_rejects_invalid_features(self):
        for x in [np.ones(3), np.ones((0, 2)), np.ones((2, 0)), np.array([[np.nan, 0]]), np.array([[np.inf, 0]]), np.array([[1 + 2j]])]:
            with self.subTest(x=x):
                with self.assertRaises((TypeError, ValueError)):
                    structural_views(x, np.empty((2, 0), dtype=np.int64))

    def test_rejects_invalid_edges(self):
        for edges in [np.array([0, 1]), np.zeros((3, 2), dtype=int), np.array([[0], [3]]), np.array([[-1], [0]]), np.array([[0.5], [1.]]), np.array([[np.nan], [1.]])]:
            with self.subTest(edges=edges):
                with self.assertRaises((TypeError, ValueError)):
                    structural_views(self.x, edges)

    def test_rejects_invalid_block_size_and_shuffle_seed(self):
        for block in [0, -1, 1.2, True]:
            with self.subTest(block=block):
                with self.assertRaises((TypeError, ValueError)):
                    structural_views(self.x, self.edges, block_rows=block)
        for seed in [-1, 1.2, True]:
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    structural_views(self.x, self.edges, shuffle_seed=seed)


if __name__ == '__main__':
    unittest.main()
