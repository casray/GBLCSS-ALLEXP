import csv
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.dataloader import Dataset
from utils.dataloader import pyg_load_dataset
from utils.tools import load_conf


METHODS = (
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_ct",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_logitclip",
    "mlp_anl",
    "mlp_fine",
    "mlp_dividemix_lite",
)


class MLP21DatasetReplacementTests(unittest.TestCase):
    def test_corafull_routes_to_citationfull_cora(self):
        sentinel = object()
        with patch("utils.dataloader.CitationFull", return_value=sentinel) as loader:
            result = pyg_load_dataset("corafull", path="test-data")

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(root="test-data", name="cora")

    def test_dataset_prepares_corafull_as_pyg_graph(self):
        graph = SimpleNamespace(
            x=torch.eye(4),
            y=torch.tensor([0, 0, 1, 1]),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            num_edges=4,
        )

        class FakeDataset:
            num_classes = 2

            def __getitem__(self, index):
                return graph

        with (
            patch("utils.dataloader.pyg_load_dataset", return_value=FakeDataset()),
            patch(
                "utils.dataloader.get_split",
                return_value=(np.array([0, 2]), np.array([1]), np.array([3])),
            ),
        ):
            data = Dataset(
                "corafull",
                path="test-data",
                split_type="percent",
                train_percent=0.5,
                val_percent=0.25,
                test_percent=0.25,
                add_self_loop=False,
                device="cpu",
                verbose=False,
            )

        self.assertEqual(data.n_nodes, 4)
        self.assertEqual(data.n_classes, 2)

    def test_cli_lists_corafull(self):
        repo_root = Path(__file__).resolve().parents[1]
        for script in ["single_exp.py", "total_exp.py"]:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("corafull", result.stdout)

    def test_gate_configs_are_complete_and_frozen(self):
        for dataset, embedding_dir in [
            ("amazoncom", "amazoncom"),
            ("corafull", "CoraFull"),
        ]:
            for method in METHODS:
                with self.subTest(dataset=dataset, method=method):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertAlmostEqual(float(conf.model["dropout"]), 0.5)
                    if method == "mlp_logitclip":
                        self.assertAlmostEqual(float(conf.model["logit_clip_tau"]), 0.5)
                    if method.startswith("frozengrace_gb"):
                        path = conf.gb["embedding_path"].replace("\\", "/")
                        self.assertIn(f"embeddings_v2/{embedding_dir}/grace/H.pt", path)
                        self.assertEqual(int(conf.gb["max_balls"]), 64)
                        self.assertEqual(int(conf.gb["min_ball_size"]), 32)
                        self.assertAlmostEqual(float(conf.gb["split_improvement"]), 0.05)
                        self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_gate_requires_structure_and_best_baseline_margin(self):
        summary = importlib.import_module("scripts.summarize_mlp21_dataset_replacement_20260710")
        rows = []
        for seed in range(5):
            rows.extend(
                [
                    {"dataset": "amazoncom", "method": summary.OURS, "seed": seed, "test": 0.80},
                    {"dataset": "amazoncom", "method": summary.SHUFFLED, "seed": seed, "test": 0.78},
                    {"dataset": "amazoncom", "method": "mlp_jocor", "seed": seed, "test": 0.785},
                ]
            )

        decision = summary.decide_gate(rows, "amazoncom")
        self.assertEqual(decision["decision"], "pass")
        self.assertAlmostEqual(decision["vs_shuffled_mean_delta"], 0.02)
        self.assertAlmostEqual(decision["vs_best_baseline_mean_delta"], 0.015)

        for row in rows:
            if row["method"] == "mlp_jocor":
                row["test"] = 0.795
        decision = summary.decide_gate(rows, "amazoncom")
        self.assertEqual(decision["decision"], "boundary")

    def test_loader_excludes_wrong_protocol_rows(self):
        summary = importlib.import_module("scripts.summarize_mlp21_dataset_replacement_20260710")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            fieldnames = [
                "dataset", "method", "seed", "noise_type", "noise_rate", "noise_scope",
                "split_type", "train_percent", "val_percent", "test_percent", "status", "test",
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "amazoncom", "method": "mlp_ce", "seed": 0,
                        "noise_type": "uniform", "noise_rate": 0.4, "noise_scope": "train",
                        "split_type": "percent", "train_percent": 0.2, "val_percent": 0.1,
                        "test_percent": 0.7, "status": "ok", "test": 0.7,
                    }
                )
                writer.writerow(
                    {
                        "dataset": "amazoncom", "method": "mlp_ce", "seed": 1,
                        "noise_type": "uniform", "noise_rate": 0.4, "noise_scope": "all",
                        "split_type": "percent", "train_percent": 0.2, "val_percent": 0.1,
                        "test_percent": 0.7, "status": "ok", "test": 0.9,
                    }
                )

            rows = summary.load_protocol_rows([path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 0)


if __name__ == "__main__":
    unittest.main()
