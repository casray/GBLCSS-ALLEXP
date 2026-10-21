import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pretrain_ssl_candidates import (
    BGRLModel,
    CCASSGModel,
    FixedTauGraceModel,
    GraphMAEModel,
    LearnableTauGraceModel,
    gcn_encoder,
    method_output_name,
    ssl_defaults,
    uniformity_loss,
)


class SSLCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.eye(6)
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 0, 2, 1, 4, 3, 5, 4],
        ], dtype=torch.long)

    def test_learnable_tau_grace_loss_backpropagates_and_clamps_tau(self):
        encoder = gcn_encoder(self.x.size(1), 8, num_layers=2, activation_name="relu")
        model = LearnableTauGraceModel(encoder, hidden_dim=8, proj_dim=8, init_tau=10.0)

        loss = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertLessEqual(float(model.tau.item()), 2.0)
        self.assertIsNotNone(model.log_tau.grad)

    def test_fixed_tau_grace_loss_backpropagates_without_tau_parameter(self):
        encoder = gcn_encoder(self.x.size(1), 8, num_layers=2, activation_name="relu")
        model = FixedTauGraceModel(encoder, hidden_dim=8, proj_dim=8, tau=0.7)

        loss = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(model.tau, 0.7)
        self.assertNotIn("tau", dict(model.named_parameters()))

    def test_fixed_grace_chunked_loss_matches_full_loss(self):
        torch.manual_seed(7)
        encoder = gcn_encoder(self.x.size(1), 8, num_layers=2, activation_name="relu")
        model = FixedTauGraceModel(encoder, hidden_dim=8, proj_dim=8, tau=0.7)

        full = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index)
        chunked = model.loss_on_views(
            self.x,
            self.edge_index,
            self.x,
            self.edge_index,
            contrastive_batch_size=2,
        )

        self.assertTrue(torch.allclose(full, chunked, atol=1e-6, rtol=1e-6))

    def test_coauthor_physics_uses_chunked_contrastive_loss(self):
        cfg = ssl_defaults("grace", "coauthorph")

        self.assertEqual(cfg["config_source_dataset"], "DBLP")
        self.assertEqual(cfg["contrastive_batch_size"], 1024)

    def test_uniformity_loss_is_finite_and_backpropagates(self):
        z = torch.randn(8, 4, requires_grad=True)

        loss = uniformity_loss(z)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_uniformity_loss_supports_deterministic_node_subsampling(self):
        z = torch.randn(16, 4, requires_grad=True)

        loss = uniformity_loss(z, max_nodes=5)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(z.grad)
        self.assertGreater(int((z.grad.abs().sum(dim=1) > 0).sum().item()), 0)
        self.assertLessEqual(int((z.grad.abs().sum(dim=1) > 0).sum().item()), 5)

    def test_uniformity_weight_zero_matches_fixed_grace_loss(self):
        encoder = gcn_encoder(self.x.size(1), 8, num_layers=2, activation_name="relu")
        model = FixedTauGraceModel(encoder, hidden_dim=8, proj_dim=8, tau=0.7)

        base = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index)
        total = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index, uniformity_weight=0.0)

        self.assertTrue(torch.allclose(base, total))

    def test_grace_uniformity_output_name_encodes_weight(self):
        self.assertEqual(method_output_name("grace_uniformity", uniformity_weight=0.1), "grace_unif0p1")
        self.assertEqual(method_output_name("grace_uniformity", uniformity_weight=1.0), "grace_unif1")

    def test_fixed_grace_defaults_use_official_pubmed_tau(self):
        cfg = ssl_defaults("grace", "PubMed")

        self.assertEqual(cfg["epochs"], 1500)
        self.assertEqual(cfg["hidden_dim"], 256)
        self.assertAlmostEqual(cfg["tau"], 0.7)
        self.assertEqual(cfg["config_source_dataset"], "PubMed")

    def test_amazon_photo_grace_defaults_use_preregistered_pubmed_fallback(self):
        cfg = ssl_defaults("grace", "AmazonPhoto")

        self.assertEqual(cfg["epochs"], 1500)
        self.assertEqual(cfg["hidden_dim"], 256)
        self.assertAlmostEqual(cfg["tau"], 0.7)
        self.assertEqual(cfg["config_source_dataset"], "PubMed")
        self.assertIn("fallback uses PubMed", cfg["fallback_note"])

    def test_bgrl_loss_backpropagates(self):
        encoder = gcn_encoder(self.x.size(1), 8, num_layers=2, activation_name="relu")
        model = BGRLModel(encoder, hidden_dim=8, predictor_hidden_dim=16)

        loss = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)

    def test_cca_ssg_loss_backpropagates(self):
        model = CCASSGModel(self.x.size(1), hidden_dim=8, out_dim=8, n_layers=2)

        loss = model.loss_on_views(self.x, self.edge_index, self.x, self.edge_index, lambd=1e-3)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    def test_graphmae_loss_backpropagates(self):
        model = GraphMAEModel(self.x.size(1), hidden_dim=8, num_layers=2, activation_name="prelu")

        loss = model.loss(self.x, self.edge_index, mask_rate=0.5, replace_rate=0.0, alpha_l=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
