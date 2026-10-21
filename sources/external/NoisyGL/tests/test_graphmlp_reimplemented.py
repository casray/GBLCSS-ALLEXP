import importlib
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch


class TinyData:
    def __init__(self):
        self.n_nodes = 6
        self.n_classes = 3
        self.dim_feats = 4
        self.feats = torch.randn(self.n_nodes, self.dim_feats)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])
        self.noisy_label = torch.tensor([0, 2, 2, 0, 1, 1])
        self.train_masks = torch.tensor([0, 2, 4])
        self.val_masks = torch.tensor([1])
        self.test_masks = torch.tensor([3, 5])
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        self.adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]),
            (self.n_nodes, self.n_nodes),
        ).coalesce()


def tiny_conf():
    return Namespace(
        model={
            "method": "graphmlp_reimplemented",
            "n_feat": 4,
            "n_classes": 3,
            "n_hidden": 8,
            "dropout": 0.0,
            "graphmlp_alpha": 2.0,
            "graphmlp_tau": 1.0,
            "graphmlp_order": 2,
            "graphmlp_batch_size": 5,
        },
        training={
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "n_epochs": 2,
            "patience": None,
            "criterion": "metric",
            "debug": False,
        },
        dataset={"sparse": True},
        analysis={"flag": False},
    )


class GraphMLPReimplementedTests(unittest.TestCase):
    def test_ncontrast_matches_official_formula(self):
        from predictor.GraphMLPReimplemented_Predictor import graphmlp_ncontrast

        similarity = torch.tensor([[0.0, 0.8, 0.1], [0.8, 0.0, 0.3], [0.1, 0.3, 0.0]])
        adjacency = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        tau = 2.0

        actual = graphmlp_ncontrast(similarity, adjacency, tau=tau)
        exp_similarity = torch.exp(tau * similarity)
        expected = -torch.log(
            (exp_similarity * adjacency).sum(dim=1)
            / exp_similarity.sum(dim=1)
            + 1e-8
        ).mean()

        self.assertTrue(torch.allclose(actual, expected))

    def test_batch_places_every_train_node_once_at_front(self):
        from predictor.GraphMLPReimplemented_Predictor import graphmlp_batch_indices

        train_idx = torch.tensor([4, 1, 5])
        generator = torch.Generator().manual_seed(7)
        batch = graphmlp_batch_indices(train_idx, num_nodes=8, batch_size=6, generator=generator)

        self.assertTrue(torch.equal(batch[:3], train_idx))
        self.assertEqual(len(set(batch.tolist())), batch.numel())
        self.assertTrue(set(train_idx.tolist()).issubset(set(batch.tolist())))

    def test_predictor_is_registered_and_inference_ignores_adjacency(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        method = "graphmlp_reimplemented"

        self.assertEqual(single_exp.PREDICTOR_MODULES[method], "predictor.GraphMLPReimplemented_Predictor")
        self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.GraphMLPReimplemented_Predictor")
        self.assertEqual(ssl_swap.PREDICTOR_MODULES[method], "predictor.GraphMLPReimplemented_Predictor")

        module = importlib.import_module("predictor.GraphMLPReimplemented_Predictor")
        predictor = module.graphmlp_reimplemented_Predictor(tiny_conf(), TinyData(), device="cpu")
        predictor.model.eval()
        output_with_adj, _, _ = predictor.get_prediction(predictor.feats, predictor.adj)
        output_without_adj, _, _ = predictor.get_prediction(predictor.feats, None)
        self.assertTrue(torch.allclose(output_with_adj, output_without_adj))


if __name__ == "__main__":
    unittest.main()
