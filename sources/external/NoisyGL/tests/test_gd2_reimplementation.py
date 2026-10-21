import unittest

import numpy as np
import torch


class GD2ReimplementationTest(unittest.TestCase):
    def test_build_graph_views_separates_ego_structure_and_mixed_edges(self):
        from predictor.module.GD2 import build_graph_views

        edge_index = torch.tensor([
            [0, 1, 1, 2, 2],
            [1, 0, 2, 1, 2],
        ])
        ego, structure, mixed = build_graph_views(edge_index, n_nodes=3)

        self.assertEqual(set(map(tuple, ego.t().tolist())), {(0, 0), (1, 1), (2, 2)})
        self.assertFalse(any(source == target for source, target in structure.t().tolist()))
        self.assertTrue({(0, 0), (1, 1), (2, 2)}.issubset(set(map(tuple, mixed.t().tolist()))))

    def test_gmm_clean_probability_selects_low_discrepancy_component(self):
        from predictor.module.GD2 import gmm_low_component_probability

        discrepancy = torch.tensor([0.01, 0.04, 0.08, 0.8, 0.9, 1.0])
        probability = gmm_low_component_probability(discrepancy, random_seed=11)

        self.assertGreater(float(probability[:3].mean()), 0.8)
        self.assertLess(float(probability[3:].mean()), 0.2)

    def test_select_purified_nodes_applies_confidence_threshold(self):
        from predictor.module.GD2 import select_purified_nodes

        probabilities = torch.tensor([
            [0.05, 0.95],
            [0.2, 0.8],
            [0.51, 0.49],
        ])
        noisy_indices = torch.tensor([3, 5, 7])

        selected_idx, pseudo_label = select_purified_nodes(
            probabilities, noisy_indices, confidence_threshold=0.9)

        self.assertTrue(torch.equal(selected_idx, torch.tensor([3])))
        self.assertTrue(torch.equal(pseudo_label, torch.tensor([1])))

    def test_select_purified_nodes_accepts_numpy_split_indices(self):
        from predictor.module.GD2 import select_purified_nodes

        selected_idx, pseudo_label = select_purified_nodes(
            torch.tensor([[0.95, 0.05]]),
            np.array([4]),
            confidence_threshold=0.9,
        )

        self.assertTrue(torch.equal(selected_idx, torch.tensor([4])))
        self.assertTrue(torch.equal(pseudo_label, torch.tensor([0])))

    def test_prediction_discrepancy_uses_l2_distance(self):
        from predictor.module.GD2 import prediction_discrepancy

        first = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        second = torch.tensor([[0.0, 1.0], [0.5, 0.5]])

        discrepancy = prediction_discrepancy(first, second)

        self.assertTrue(torch.allclose(discrepancy, torch.tensor([2.0 ** 0.5, 0.0])))

    def test_structure_loss_averages_clean_and_purified_union(self):
        from predictor.module.GD2 import structural_classification_loss

        logits = torch.tensor([
            [3.0, 0.0],
            [0.0, 3.0],
            [1.0, 1.0],
        ])
        noisy_labels = torch.tensor([0, 1, 0])
        clean_idx = torch.tensor([0, 1])
        purified_idx = torch.tensor([2])
        pseudo_label = torch.tensor([1])

        loss = structural_classification_loss(
            logits, clean_idx, noisy_labels, purified_idx, pseudo_label)
        expected = torch.nn.functional.cross_entropy(
            logits, torch.tensor([0, 1, 1]))

        self.assertTrue(torch.allclose(loss, expected))

    def test_predictor_alias_exists(self):
        from predictor.GD2_Predictor import gd2_Predictor

        self.assertIsNotNone(gd2_Predictor)


if __name__ == "__main__":
    unittest.main()
