import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from predictor.module.GNNs import MLP
from scripts.summarize_mlp_dualrobust import (
    expected_label_keys,
    normalize_label_row,
    paired_label_deltas,
    judge_c1,
)
from utils.structure_noise import perturb_test_adj
from utils.tools import load_conf


class MLPDualRobustTests(unittest.TestCase):
    def test_hard_mlp_method_is_registered(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")

        self.assertEqual(
            single_exp.PREDICTOR_MODULES["frozengrace_gb_mlp_hard"],
            "predictor.FrozenGraceGB_GCN_Predictor",
        )
        self.assertEqual(
            standard.PREDICTOR_MODULES["frozengrace_gb_mlp_hard"],
            "predictor.FrozenGraceGB_GCN_Predictor",
        )
        self.assertEqual(
            ssl_swap.PREDICTOR_MODULES["frozengrace_gb_mlp_hard"],
            "predictor.FrozenGraceGB_GCN_Predictor",
        )
        self.assertTrue(hasattr(module, "frozengrace_gb_mlp_hard_Predictor"))

    def test_coauthorcs_mlp_configs_exist_and_match_backbone(self):
        for method in [
            "mlp_ce",
            "mlp_small_loss",
            "mlp_coteaching",
            "mlp_jocor",
            "mlp_elr",
            "mlp_sce",
            "frozengrace_gb_mlp_softdist",
            "frozengrace_gb_mlp_softdist_shuffled",
            "frozengrace_gb_mlp_hard",
            "featuregb_mlp_softdist",
        ]:
            with self.subTest(method=method):
                conf = load_conf(None, method, "coauthorcs")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.training["n_epochs"], 200)

    def test_edge_drop_is_deterministic_and_removes_edges(self):
        indices = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ]
        )
        adj = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (4, 4)).coalesce()

        perturbed_a = perturb_test_adj(adj, mode="edge_drop", p=0.5, seed=7)
        perturbed_b = perturb_test_adj(adj, mode="edge_drop", p=0.5, seed=7)

        self.assertTrue(torch.equal(perturbed_a.indices(), perturbed_b.indices()))
        self.assertLess(perturbed_a._nnz(), adj._nnz())

    def test_isolated_removes_edges_incident_to_test_nodes(self):
        indices = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ]
        )
        adj = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (4, 4)).coalesce()
        perturbed = perturb_test_adj(adj, mode="isolated", p=1.0, seed=0, test_mask=torch.tensor([1, 2]))
        rows, cols = perturbed.indices()

        self.assertFalse(torch.isin(rows, torch.tensor([1, 2])).any())
        self.assertFalse(torch.isin(cols, torch.tensor([1, 2])).any())

    def test_mlp_forward_is_independent_of_adjacency(self):
        torch.manual_seed(0)
        model = MLP(in_channels=3, hidden_channels=4, out_channels=2, n_layers=2, dropout=0.0)
        model.eval()
        x = torch.randn(5, 3)
        adj_a = torch.eye(5).to_sparse()
        adj_b = torch.zeros(5, 5).to_sparse()

        self.assertTrue(torch.allclose(model(x, adj_a), model(x, adj_b)))

    def test_summary_expected_label_grid_has_600_rows(self):
        self.assertEqual(len(expected_label_keys()), 600)
        self.assertIn(("coauthorcs", "instance", 0.4, "mlp_jocor", 4), expected_label_keys())

    def test_summary_accepts_ssl_swap_rows_and_judges_c1(self):
        rows = []
        for seed in range(5):
            for method, value in [
                ("frozengrace_gb_mlp_softdist", 0.70 if seed < 3 else 0.60),
                ("mlp_small_loss", 0.65),
                ("mlp_coteaching", 0.65),
                ("mlp_jocor", 0.80),
                ("mlp_elr", 0.65),
                ("mlp_sce", 0.65),
            ]:
                rows.append(
                    {
                        "dataset": "CoauthorCS",
                        "base_method": method,
                        "seed": str(seed),
                        "noise_type": "instance",
                        "noise_rate": "0.4",
                        "test": str(value),
                        "status": "ok",
                    }
                )
        normalized = [normalize_label_row(row, "fixture.csv") for row in rows]
        deltas = paired_label_deltas(normalized, tuple({row["method"] for row in normalized}))
        c1 = {
            (row["dataset"], row["noise_type"], float(row["noise_rate"])): row
            for row in judge_c1(deltas)
        }

        self.assertEqual(c1[("coauthorcs", "instance", 0.4)]["setting_pass"], 1)

    def test_runner_uses_rocm_and_reuses_legacy_label_dir(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_mlp_dualrobust_20260701.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn(".venv-rocm", script)
        self.assertIn("runs\\mlp_noiserobust_20260701", script)
        self.assertIn("--noise_scope train", script)
        self.assertIn("instance0p4_coauthorcs", script)


if __name__ == "__main__":
    unittest.main()
