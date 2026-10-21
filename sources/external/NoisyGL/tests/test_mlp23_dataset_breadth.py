import subprocess
import sys
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

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


class MLP23DatasetBreadthTests(unittest.TestCase):
    def test_coraml_routes_to_citationfull_cora_ml(self):
        sentinel = object()
        with patch("utils.dataloader.CitationFull", return_value=sentinel) as loader:
            result = pyg_load_dataset("coraml", path="test-data")

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(root="test-data", name="cora_ml")

    def test_attribwiki_routes_to_attributed_graph_wiki(self):
        sentinel = object()
        with patch("utils.dataloader.AttributedGraphDataset", return_value=sentinel) as loader:
            result = pyg_load_dataset("attribwiki", path="test-data")

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(root="test-data", name="Wiki")

    def test_citeseerfull_routes_to_citationfull_citeseer(self):
        sentinel = object()
        with patch("utils.dataloader.CitationFull", return_value=sentinel) as loader:
            result = pyg_load_dataset("citeseerfull", path="test-data")

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(root="test-data", name="citeseer")

    def test_cli_lists_coraml(self):
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
                self.assertIn("coraml", result.stdout)
                self.assertIn("attribwiki", result.stdout)
                self.assertIn("citeseerfull", result.stdout)

    def test_coraml_gate_configs_keep_frozen_protocol(self):
        for method in METHODS:
            with self.subTest(method=method):
                conf = load_conf(None, method, "coraml")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertAlmostEqual(float(conf.model["dropout"]), 0.5)
                if method.startswith("frozengrace_gb"):
                    path = conf.gb["embedding_path"].replace("\\", "/")
                    self.assertIn("embeddings_v2/CoraML/grace/H.pt", path)
                    self.assertEqual(int(conf.gb["max_balls"]), 64)
                    self.assertEqual(int(conf.gb["min_ball_size"]), 32)
                    self.assertAlmostEqual(float(conf.gb["split_improvement"]), 0.05)
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_attribwiki_gate_configs_keep_frozen_protocol(self):
        for method in METHODS:
            with self.subTest(method=method):
                conf = load_conf(None, method, "attribwiki")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertAlmostEqual(float(conf.model["dropout"]), 0.5)
                if method.startswith("frozengrace_gb"):
                    path = conf.gb["embedding_path"].replace("\\", "/")
                    self.assertIn("embeddings_v2/attribwiki/grace/H.pt", path)
                    self.assertEqual(int(conf.gb["max_balls"]), 64)
                    self.assertEqual(int(conf.gb["min_ball_size"]), 32)
                    self.assertAlmostEqual(float(conf.gb["split_improvement"]), 0.05)
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_citeseerfull_gate_configs_keep_frozen_protocol(self):
        for method in METHODS:
            with self.subTest(method=method):
                conf = load_conf(None, method, "citeseerfull")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertAlmostEqual(float(conf.model["dropout"]), 0.5)
                if method.startswith("frozengrace_gb"):
                    path = conf.gb["embedding_path"].replace("\\", "/")
                    self.assertIn("embeddings_v2/citeseerfull/grace/H.pt", path)
                    self.assertEqual(int(conf.gb["max_balls"]), 64)
                    self.assertEqual(int(conf.gb["min_ball_size"]), 32)
                    self.assertAlmostEqual(float(conf.gb["split_improvement"]), 0.05)
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_coauthorph_gate_configs_keep_frozen_protocol(self):
        for method in METHODS:
            with self.subTest(method=method):
                conf = load_conf(None, method, "coauthorph")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertAlmostEqual(float(conf.model["dropout"]), 0.5)
                if method.startswith("frozengrace_gb"):
                    path = conf.gb["embedding_path"].replace("\\", "/")
                    self.assertIn("embeddings_v2/coauthorph/grace/H.pt", path)
                    self.assertEqual(int(conf.gb["max_balls"]), 64)
                    self.assertEqual(int(conf.gb["min_ball_size"]), 32)
                    self.assertAlmostEqual(float(conf.gb["split_improvement"]), 0.05)
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)

    def test_summary_gate_requires_structure_and_best_baseline_margin(self):
        summary = importlib.import_module("scripts.summarize_mlp23_dataset_breadth_20260710")
        rows = []
        for seed in range(5):
            rows.append({"dataset": "toy", "method": summary.OURS, "seed": seed, "test": 0.80})
            rows.append({"dataset": "toy", "method": summary.SHUFFLED, "seed": seed, "test": 0.78})
            for baseline in summary.BASELINES:
                value = 0.785 if baseline == "mlp_jocor" else 0.78
                rows.append({"dataset": "toy", "method": baseline, "seed": seed, "test": value})

        decision = summary.decide_gate(rows, "toy")
        self.assertEqual(decision["decision"], "pass")

        for row in rows:
            if row["method"] == "mlp_jocor":
                row["test"] = 0.795
        decision = summary.decide_gate(rows, "toy")
        self.assertEqual(decision["decision"], "boundary")


if __name__ == "__main__":
    unittest.main()
