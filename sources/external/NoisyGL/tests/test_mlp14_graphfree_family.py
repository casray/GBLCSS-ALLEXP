import importlib
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from utils.tools import load_conf


DATASETS = ["cora", "citeseer", "pubmed", "coauthorcs", "amazonpho"]
METHODS = ["gsdn", "graphmlp"]


class TinyData:
    def __init__(self):
        self.n_nodes = 6
        self.n_classes = 3
        self.dim_feats = 4
        self.feats = torch.randn(self.n_nodes, self.dim_feats)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])
        self.noisy_label = torch.tensor([0, 2, 2, 0, 1, 1])
        self.train_masks = torch.tensor([0, 1, 2])
        self.val_masks = torch.tensor([3])
        self.test_masks = torch.tensor([4, 5])
        # Includes train->*, val->*, and test->* sources. GSDN label-level
        # propagation must keep only sources in train_masks.
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 0],
            ],
            dtype=torch.long,
        )
        values = torch.ones(edge_index.shape[1])
        self.adj = torch.sparse_coo_tensor(edge_index, values, (self.n_nodes, self.n_nodes)).coalesce()


def tiny_conf(method):
    return Namespace(
        model={
            "method": method,
            "n_feat": 4,
            "n_classes": 3,
            "n_hidden": 8,
            "n_layer": 2,
            "dropout": 0.0,
            "graph_batch_size": 4,
            "lambda_feature": 0.5,
            "lambda_label": 0.5,
            "lambda_negative": 0.1,
            "lambda_contrast": 0.5,
            "tau": 0.5,
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


class MLP14GraphFreeFamilyTests(unittest.TestCase):
    def test_methods_are_registered_to_runner_entrypoints(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")

        for method in METHODS:
            with self.subTest(method=method):
                self.assertEqual(single_exp.PREDICTOR_MODULES[method], "predictor.GraphFreeFamily_Predictor")
                self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.GraphFreeFamily_Predictor")
                self.assertEqual(ssl_swap.PREDICTOR_MODULES[method], "predictor.GraphFreeFamily_Predictor")

        self.assertEqual(
            standard.METHOD_CAPABILITIES["gsdn"]["implementation_status"],
            "provisional adapted implementation",
        )

    def test_configs_load_and_match_mlp_backbone(self):
        for dataset in DATASETS:
            for method in METHODS:
                with self.subTest(dataset=dataset, method=method):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.model["dropout"], 0.5)
                    self.assertEqual(conf.training["lr"], 1e-2)
                    self.assertEqual(conf.training["weight_decay"], 5e-4)
                    self.assertEqual(conf.training["n_epochs"], 200)

    def test_gsdn_has_two_heads_and_predicts_with_main_head_only(self):
        from predictor.GraphFreeFamily_Predictor import gsdn_Predictor

        data = TinyData()
        predictor = gsdn_Predictor(tiny_conf("gsdn"), data, device="cpu")

        self.assertIsNot(predictor.model, predictor.aux_model)
        out_with_adj, _, _ = predictor.get_prediction(data.feats, data.adj)
        out_without_adj, _, _ = predictor.get_prediction(data.feats, None)
        self.assertTrue(torch.allclose(out_with_adj, out_without_adj))

    def test_gsdn_mixup_coefficient_is_open_interval(self):
        from predictor.GraphFreeFamily_Predictor import gsdn_Predictor

        predictor = gsdn_Predictor(tiny_conf("gsdn"), TinyData(), device="cpu")
        coeff = predictor._mixup_coeff()
        self.assertGreater(float(coeff.detach()), 0.0)
        self.assertLess(float(coeff.detach()), 1.0)

    def test_gsdn_label_edges_use_train_sources_only(self):
        from predictor.GraphFreeFamily_Predictor import gsdn_Predictor

        data = TinyData()
        predictor = gsdn_Predictor(tiny_conf("gsdn"), data, device="cpu")
        src, _ = predictor._label_edge_indices()
        self.assertTrue(src.numel() > 0)
        self.assertTrue(set(src.cpu().tolist()).issubset(set(data.train_masks.cpu().tolist())))

    def test_gsdn_model_selection_reads_noisy_validation_labels(self):
        from predictor.GraphFreeFamily_Predictor import gsdn_Predictor

        data = TinyData()
        data.noisy_label[data.val_masks] = 2
        predictor = gsdn_Predictor(tiny_conf("gsdn"), data, device="cpu")
        observed = []
        original_evaluate = predictor.evaluate

        def record_evaluate(label, mask):
            observed.append((label.detach().clone(), torch.as_tensor(mask).clone()))
            return original_evaluate(label, mask)

        predictor.evaluate = record_evaluate
        predictor.train()

        self.assertGreaterEqual(len(observed), 2)
        validation_label, validation_mask = observed[0]
        self.assertTrue(torch.equal(validation_mask, data.val_masks))
        self.assertTrue(torch.equal(validation_label, data.noisy_label))
        self.assertFalse(torch.equal(
            validation_label[data.val_masks],
            data.labels[data.val_masks],
        ))

    def test_graphmlp_inference_is_graph_free_and_contrast_loss_is_finite(self):
        from predictor.GraphFreeFamily_Predictor import graphmlp_Predictor

        data = TinyData()
        predictor = graphmlp_Predictor(tiny_conf("graphmlp"), data, device="cpu")
        out_with_adj, _, _ = predictor.get_prediction(data.feats, data.adj)
        out_without_adj, _, _ = predictor.get_prediction(data.feats, None)
        self.assertTrue(torch.allclose(out_with_adj, out_without_adj))

        _, hidden = predictor._forward_with_hidden(data.feats)
        loss = predictor._graph_contrast_loss(hidden)
        self.assertTrue(torch.isfinite(loss))

    def test_reference_rows_use_main_train_percent_for_cora_citeseer_uniform04(self):
        from scripts.summarize_mlp14_graphfree_family import (
            MLP10_DIR,
            MLP12_DIR,
            MLP9_DIR,
            MLP_MAIN_DIR,
            GCN_REF,
            GCN_XH_REF,
            OURS,
            aggregate,
            load_reference_rows,
        )

        rows = load_reference_rows(Path(MLP_MAIN_DIR), Path(MLP10_DIR), Path(MLP12_DIR), Path(MLP9_DIR))
        summary = {
            (row["dataset"], row["method"], row["noise_type"], float(row["noise_rate"])): row
            for row in aggregate(rows)
        }

        cora = summary[("cora", OURS, "uniform", 0.4)]
        citeseer = summary[("citeseer", OURS, "uniform", 0.4)]
        self.assertEqual(cora["n"], 5)
        self.assertEqual(citeseer["n"], 5)
        self.assertAlmostEqual(float(cora["mean"]), 0.5559, delta=0.0001)
        self.assertAlmostEqual(float(citeseer["mean"]), 0.5383, delta=0.0001)

        gcn_cora = summary[("cora", GCN_REF, "uniform", 0.4)]
        gcn_xh_cora = summary[("cora", GCN_XH_REF, "uniform", 0.4)]
        self.assertEqual(gcn_cora["n"], 5)
        self.assertEqual(gcn_xh_cora["n"], 5)
        self.assertAlmostEqual(float(gcn_cora["mean"]), 0.7877, delta=0.0001)
        self.assertAlmostEqual(float(gcn_xh_cora["mean"]), 0.8005, delta=0.0001)


if __name__ == "__main__":
    unittest.main()
