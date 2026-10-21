import importlib
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch


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
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 2],
                [1, 2, 3, 4, 5, 0, 2, 0],
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
            "teacher_hidden": 8,
            "teacher_n_layer": 2,
            "teacher_dropout": 0.0,
            "teacher_epochs": 2,
            "distill_temperature": 1.0,
            "lambda_kd": 1.0,
            "lambda_ce": 0.0,
            "lambda_low": 0.5,
            "lambda_high": 0.5,
            "reliability_perturbations": 2,
            "reliability_noise_std": 0.01,
            "reliable_fraction": 0.5,
            "codebook_size": 4,
            "token_temperature": 1.0,
            "lambda_token": 1.0,
            "lambda_node": 0.001,
            "lambda_edge": 0.03,
            "vq_edge_sample": 8,
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


class MLP19G2MBaselineTests(unittest.TestCase):
    def test_ffg2m_reimplemented_is_registered_with_frozen_configs(self):
        import sys
        from unittest.mock import patch
        from utils.tools import load_conf

        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        module = importlib.import_module("predictor.G2MBaselines_Predictor")
        method = "ffg2m_reimplemented"

        self.assertEqual(single_exp.PREDICTOR_MODULES[method], "predictor.G2MBaselines_Predictor")
        self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.G2MBaselines_Predictor")
        self.assertEqual(ssl_swap.PREDICTOR_MODULES[method], "predictor.G2MBaselines_Predictor")
        self.assertEqual(getattr(module, f"{method}_Predictor").strategy, method)

        for dataset in ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]:
            with self.subTest(dataset=dataset):
                conf = load_conf(None, method, dataset)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.model["teacher_hidden"], 64)
                self.assertEqual(conf.model["teacher_n_layer"], 2)
                self.assertEqual(conf.model["ffg2m_lambda"], 0.4)
                self.assertEqual(conf.model["ffg2m_tau"], 1.5)

    def test_ffg2m_reimplemented_total_loss_matches_official_weighting(self):
        from predictor.G2MBaselines_Predictor import (
            ffg2m_high_frequency_loss,
            ffg2m_low_frequency_loss,
            ffg2m_reimplemented_Predictor,
        )

        data = TinyData()
        conf = tiny_conf("ffg2m_reimplemented")
        conf.model["ffg2m_lambda"] = 0.4
        conf.model["ffg2m_tau"] = 1.5
        predictor = ffg2m_reimplemented_Predictor(conf, data, device="cpu")
        student = torch.randn(data.n_nodes, data.n_classes, requires_grad=True)
        teacher = torch.randn(data.n_nodes, data.n_classes)
        predictor.teacher_logits = teacher

        actual = predictor._student_loss(
            student,
            torch.softmax(teacher, dim=1),
            torch.ones(data.n_nodes),
            temperature=1.0,
        )
        ce = torch.nn.functional.cross_entropy(
            student[data.train_masks], data.noisy_label[data.train_masks]
        )
        low = ffg2m_low_frequency_loss(student, teacher, data.adj.indices(), data.n_nodes)
        high = ffg2m_high_frequency_loss(student, teacher, data.adj.indices(), tau=1.5)
        expected = 0.4 * ce + 0.6 * (low + high)

        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.all(torch.isfinite(student.grad)))

    def test_teacher_model_selection_reads_test_mask_once(self):
        from predictor.G2MBaselines_Predictor import ffg2m_adapted_Predictor

        data = TinyData()
        predictor = ffg2m_adapted_Predictor(tiny_conf("ffg2m_adapted"), data, device="cpu")
        call_sizes = []

        def counted_metric(labels, logits):
            call_sizes.append(len(labels))
            return 0.5

        predictor.metric = counted_metric
        predictor._train_teacher()

        self.assertEqual(call_sizes.count(len(data.test_masks)), 1)
        self.assertEqual(call_sizes.count(len(data.val_masks)), 2)

    def test_ffg2m_low_frequency_loss_matches_official_neighbor_kl(self):
        from predictor.G2MBaselines_Predictor import ffg2m_low_frequency_loss

        student = torch.tensor([[2.0, 0.1], [0.2, 1.5], [1.0, 0.8]])
        teacher = torch.tensor([[1.7, 0.3], [0.1, 1.8], [0.9, 0.7]])
        edge_index = torch.tensor([[0, 1], [1, 2]])

        actual = ffg2m_low_frequency_loss(student, teacher, edge_index, num_nodes=3)
        loop_nodes = torch.arange(3)
        edges = torch.cat([edge_index, torch.stack([loop_nodes, loop_nodes])], dim=1)
        expected = torch.nn.functional.kl_div(
            torch.log_softmax(student[edges[0]], dim=1),
            torch.log_softmax(teacher[edges[1]], dim=1),
            reduction="batchmean",
            log_target=True,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_ffg2m_high_frequency_loss_matches_official_edge_difference_kl(self):
        from predictor.G2MBaselines_Predictor import ffg2m_high_frequency_loss

        student = torch.tensor([[2.0, 0.1, -0.2], [0.2, 1.5, 0.4], [1.0, 0.8, 0.1]])
        teacher = torch.tensor([[1.7, 0.3, 0.0], [0.1, 1.8, 0.2], [0.9, 0.7, 0.4]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]])
        tau = 1.5

        actual = ffg2m_high_frequency_loss(student, teacher, edge_index, tau=tau)
        keep = edge_index[0] != edge_index[1]
        src, dst = edge_index[:, keep]
        expected = torch.nn.functional.kl_div(
            torch.log_softmax((student[src] - student[dst]).abs() / tau, dim=1),
            torch.log_softmax((teacher[src] - teacher[dst]).abs() / tau, dim=1),
            reduction="batchmean",
            log_target=True,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_methods_are_registered_to_standard_runner(self):
        standard = importlib.import_module("scripts.run_standard_downstream")
        for method in ["ffg2m_adapted", "krd_adapted", "vqgraph_adapted"]:
            with self.subTest(method=method):
                self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.G2MBaselines_Predictor")

    def test_standard_runner_formats_not_runnable_metric(self):
        standard = importlib.import_module("scripts.run_standard_downstream")
        self.assertEqual(standard._format_metric_for_log(""), "NA")
        self.assertEqual(standard._format_metric_for_log(0.12345), "0.1235")

    def test_student_inference_is_graph_free_for_runnable_methods(self):
        from predictor.G2MBaselines_Predictor import (
            ffg2m_adapted_Predictor,
            krd_adapted_Predictor,
            vqgraph_adapted_Predictor,
        )

        data = TinyData()
        for cls in [ffg2m_adapted_Predictor, krd_adapted_Predictor, vqgraph_adapted_Predictor]:
            with self.subTest(cls=cls.__name__):
                predictor = cls(tiny_conf(cls.__name__.replace("_Predictor", "")), data, device="cpu")
                out_with_adj, _, _ = predictor.get_prediction(data.feats, data.adj)
                out_without_adj, _, _ = predictor.get_prediction(data.feats, None)
                self.assertTrue(torch.allclose(out_with_adj, out_without_adj))

    def test_ffg2m_frequency_loss_is_finite_and_label_free(self):
        from predictor.G2MBaselines_Predictor import ffg2m_adapted_Predictor

        data = TinyData()
        predictor = ffg2m_adapted_Predictor(tiny_conf("ffg2m_adapted"), data, device="cpu")
        student = torch.randn(data.n_nodes, data.n_classes)
        teacher = torch.softmax(torch.randn(data.n_nodes, data.n_classes), dim=1)
        loss = predictor._ffg2m_frequency_loss(student, teacher)
        self.assertTrue(torch.isfinite(loss))

    def test_krd_reliability_does_not_read_clean_labels(self):
        from predictor.G2MBaselines_Predictor import krd_adapted_Predictor

        data = TinyData()
        predictor = krd_adapted_Predictor(tiny_conf("krd_adapted"), data, device="cpu")
        teacher = predictor._teacher_model()

        torch.manual_seed(123)
        reliability_a = predictor._estimate_teacher_reliability(teacher)
        predictor.clean_label = torch.tensor([2, 2, 2, 2, 2, 2])
        torch.manual_seed(123)
        reliability_b = predictor._estimate_teacher_reliability(teacher)

        self.assertTrue(torch.allclose(reliability_a, reliability_b))
        self.assertEqual(reliability_a.shape[0], data.n_nodes)
        self.assertGreaterEqual(float(reliability_a.min()), 0.0)
        self.assertLessEqual(float(reliability_a.max()), 1.0)

    def test_vqgraph_token_loss_is_finite_and_clean_label_free(self):
        from predictor.G2MBaselines_Predictor import vqgraph_adapted_Predictor

        data = TinyData()
        predictor = vqgraph_adapted_Predictor(tiny_conf("vqgraph_adapted"), data, device="cpu")
        teacher = predictor._teacher_model()
        with torch.no_grad():
            _, _, assignments, _ = teacher(data.feats, data.adj)
            predictor.teacher_token_assignments = assignments
            predictor.teacher_codebook = torch.nn.functional.normalize(teacher.codebook.detach(), p=2, dim=1)
        _, hidden = predictor.model.forward_with_hidden(data.feats)
        loss_a = predictor._student_token_loss(hidden)
        predictor.clean_label = torch.tensor([2, 2, 2, 2, 2, 2])
        loss_b = predictor._student_token_loss(hidden)

        self.assertEqual(predictor.run_status, "ok")
        self.assertTrue(torch.isfinite(loss_a))
        self.assertTrue(torch.allclose(loss_a, loss_b))

    def test_summarizer_preserves_not_runnable_rows_but_excludes_from_aggregate(self):
        from scripts.summarize_mlp19_g2m_baselines_20260709 import aggregate, load_mlp19_rows, write_csv

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_csv(
                run_dir / "raw.csv",
                [
                    {
                        "dataset": "cora",
                        "method": "ffg2m_adapted",
                        "seed": 0,
                        "noise_type": "uniform",
                        "noise_rate": 0.4,
                        "status": "ok",
                        "test": 0.5,
                    },
                    {
                        "dataset": "cora",
                        "method": "vqgraph_adapted",
                        "seed": 0,
                        "noise_type": "uniform",
                        "noise_rate": 0.4,
                        "status": "not_runnable",
                        "error": "tokenizer path missing",
                        "test": "",
                    },
                ],
            )
            rows = load_mlp19_rows(run_dir)
            summary = aggregate(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["status"], "not_runnable")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["method"], "ffg2m_adapted")


if __name__ == "__main__":
    unittest.main()
