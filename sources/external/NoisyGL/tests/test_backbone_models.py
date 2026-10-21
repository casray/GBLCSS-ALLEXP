import unittest

import torch

from predictor.module.GNNs import APPNPNet, GATNet


class BackboneModelsTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(5, 4)
        self.edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4],
                [1, 0, 2, 1, 3, 2, 4, 3],
            ],
            dtype=torch.long,
        )
        values = torch.ones(self.edge_index.shape[1])
        self.sparse_adj = torch.sparse_coo_tensor(self.edge_index, values, (5, 5)).coalesce()

    def test_gat_accepts_sparse_adjacency_and_returns_logits(self):
        model = GATNet(in_channels=4, hidden_channels=8, out_channels=3, n_layers=2)

        out, hidden = model.forward_with_hidden(self.x, self.sparse_adj)

        self.assertEqual(tuple(out.shape), (5, 3))
        self.assertEqual(hidden.shape[0], 5)

    def test_appnp_accepts_sparse_adjacency_and_returns_logits(self):
        model = APPNPNet(in_channels=4, hidden_channels=8, out_channels=3, n_layers=2)

        out, hidden = model.forward_with_hidden(self.x, self.sparse_adj)

        self.assertEqual(tuple(out.shape), (5, 3))
        self.assertEqual(hidden.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
