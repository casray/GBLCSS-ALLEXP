import unittest

import torch

from predictor.module.GNNs import MLP
from scripts.run_mlp8_p2_advisor_smoke import extract_mlp_hidden, p2a_gate, p2b_gate


class MLP8P2AdvisorSmokeTests(unittest.TestCase):
    def test_p2a_gate_requires_geometry_and_downstream(self):
        rows = [
            {
                "phase": "p2a",
                "variant": "anchor_grace",
                "num_balls": 60,
                "true_all_purity_weighted": 0.75,
                "downstream_test": 0.60,
            },
            {
                "phase": "p2a",
                "variant": "grace_ce_l0p1",
                "num_balls": 40,
                "true_all_purity_weighted": 0.746,
                "downstream_test": 0.61,
            },
        ]
        decision = p2a_gate(rows)
        self.assertEqual(decision["p2a_gate"], "pass")
        self.assertTrue(decision["num_balls_ok"])
        self.assertTrue(decision["purity_ok"])
        self.assertTrue(decision["downstream_ok"])

        rows[1]["downstream_test"] = 0.59
        self.assertEqual(p2a_gate(rows)["p2a_gate"], "fail")

    def test_p2b_gate_requires_both_dataset_positive(self):
        rows = [
            {"phase": "p2b", "variant": "F1_loop_Z1", "loop_delta": 0.01},
            {"phase": "p2b", "variant": "F1_loop_Z1", "loop_delta": 0.02},
        ]
        self.assertEqual(p2b_gate(rows)["p2b_gate"], "pass")
        rows[1]["loop_delta"] = -0.01
        self.assertEqual(p2b_gate(rows)["p2b_gate"], "fail")

    def test_extract_mlp_hidden_returns_penultimate_layer(self):
        model = MLP(in_channels=4, hidden_channels=8, out_channels=3, n_layers=2, dropout=0.0)
        x = torch.randn(5, 4)
        hidden = extract_mlp_hidden(model, x)
        self.assertEqual(tuple(hidden.shape), (5, 8))


if __name__ == "__main__":
    unittest.main()
