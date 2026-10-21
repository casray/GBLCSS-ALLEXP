"""Focused shape, numerical, and gradient checks for the PiNGDA adapter."""

import unittest
from unittest.mock import patch

import torch

from pingda import build_generators, sample_augmentation


class PiNGDAAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.x = torch.randn(5, 11, requires_grad=True)
        self.edges = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            dtype=torch.long,
        )
        self.edge_gen, self.feat_gen = build_generators(11, torch.device("cpu"))

    def test_shapes_ranges_and_gradients(self) -> None:
        x_aug, edge_weight = sample_augmentation(
            self.x, self.edges, self.edge_gen, self.feat_gen
        )
        self.assertEqual(tuple(x_aug.shape), tuple(self.x.shape))
        self.assertEqual(tuple(edge_weight.shape), (self.edges.size(1),))
        self.assertTrue(torch.isfinite(x_aug).all().item())
        self.assertTrue(torch.isfinite(edge_weight).all().item())
        self.assertTrue(((edge_weight >= 0) & (edge_weight <= 1)).all().item())

        loss = x_aug.square().mean() + edge_weight.square().mean()
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        for module in (self.edge_gen, self.feat_gen):
            grads = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(grad is not None for grad in grads))
            self.assertTrue(all(torch.isfinite(grad).all().item() for grad in grads))
            self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in grads))

    def test_single_edge_keeps_vector_shape(self) -> None:
        one_edge = self.edges[:, :1]
        x_aug, edge_weight = sample_augmentation(
            self.x, one_edge, self.edge_gen, self.feat_gen
        )
        self.assertEqual(tuple(x_aug.shape), tuple(self.x.shape))
        self.assertEqual(tuple(edge_weight.shape), (1,))

    def test_empty_edges_return_empty_weights(self) -> None:
        no_edges = self.edges[:, :0]
        x_aug, edge_weight = sample_augmentation(
            self.x, no_edges, self.edge_gen, self.feat_gen
        )
        self.assertEqual(tuple(x_aug.shape), tuple(self.x.shape))
        self.assertEqual(tuple(edge_weight.shape), (0,))
        x_aug.square().mean().backward()
        self.assertIsNotNone(self.feat_gen.fc_mean[0].weight.grad)

    def test_feature_only_view_skips_edge_generation(self) -> None:
        with patch.object(
            self.edge_gen,
            "apply_gumbel_softmax",
            side_effect=AssertionError("feature-only view must not sample edge gates"),
        ):
            x_aug, edge_weight = sample_augmentation(
                self.x,
                self.edges,
                self.edge_gen,
                self.feat_gen,
                learnable_edge=False,
            )
        self.assertEqual(tuple(x_aug.shape), tuple(self.x.shape))
        self.assertIsNone(edge_weight)
        x_aug.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.edge_gen.parameters()))
        self.assertIsNotNone(self.feat_gen.fc_mean[0].weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device is unavailable")
    def test_device_placement_and_gradients(self) -> None:
        device = torch.device("cuda:0")
        x = self.x.detach().to(device).requires_grad_(True)
        edges = self.edges.to(device)
        edge_gen, feat_gen = build_generators(x.size(-1), device)
        x_aug, edge_weight = sample_augmentation(x, edges, edge_gen, feat_gen)
        self.assertEqual(x_aug.device, device)
        self.assertEqual(edge_weight.device, device)
        (x_aug.square().mean() + edge_weight.mean()).backward()
        self.assertIsNotNone(edge_gen.mlp_edge_model[0].weight.grad)
        self.assertIsNotNone(feat_gen.fc_mean[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
