import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.diagnose_biggraph_orthogonality import BIG_BALL_PARAMS, _edge_homophily
from utils.dataloader import Dataset, pyg_load_dataset
from utils.tools import load_conf


class LastFMAsiaSupportTest(unittest.TestCase):
    def test_pyg_loader_routes_lastfmasia_to_native_dataset(self):
        sentinel = object()
        with patch("utils.dataloader.LastFMAsia", return_value=sentinel) as loader:
            result = pyg_load_dataset("lastfmasia", path="test-data")

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(root="test-data")

    def test_dataset_accepts_lastfmasia_with_percent_split(self):
        graph = SimpleNamespace(
            x=torch.eye(4),
            y=torch.tensor([0, 0, 1, 1]),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            num_edges=4,
        )

        class FakeDataset:
            num_classes = 2

            def __getitem__(self, index):
                self.last_index = index
                return graph

        with (
            patch("utils.dataloader.pyg_load_dataset", return_value=FakeDataset()),
            patch(
                "utils.dataloader.get_split",
                return_value=(
                    np.array([0, 2]),
                    np.array([1]),
                    np.array([3]),
                ),
            ),
        ):
            data = Dataset(
                "lastfmasia",
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

    def test_single_exp_help_lists_lastfmasia(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "single_exp.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lastfmasia", result.stdout)

    def test_total_exp_help_lists_lastfmasia(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "total_exp.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lastfmasia", result.stdout)

    def test_lastfmasia_configs_use_percent_split_and_shared_embedding(self):
        repo_root = Path(__file__).resolve().parents[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_conf = load_conf(str(repo_root / "config" / "_dataset" / "lastfmasia.yaml"))
        self.assertEqual(data_conf.split["split_type"], "percent")
        self.assertEqual(data_conf.split["train_percent"], 0.2)
        self.assertEqual(data_conf.split["val_percent"], 0.1)
        self.assertEqual(data_conf.split["test_percent"], 0.7)

        embedding_suffix = "embeddings_v2/LastFMAsia/grace/H.pt"
        for method in [
            "gcn_xh_ce",
            "frozengrace_gb_gcn_softdist_xh",
            "frozengrace_gb_gcn_softdist_xh_shuffled",
            "nrgnn_xh",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conf = load_conf(None, method, "lastfmasia")
            self.assertTrue(conf.gb["embedding_path"].replace("\\", "/").endswith(embedding_suffix))

        for method in ["gcn", "nrgnn"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                load_conf(None, method, "lastfmasia")

        for method in [
            "frozengrace_gb_gcn_softdist_xh",
            "frozengrace_gb_gcn_softdist_xh_shuffled",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conf = load_conf(None, method, "lastfmasia")
            self.assertEqual(conf.gb["max_balls"], 256)
            self.assertEqual(conf.gb["min_ball_size"], 32)
            self.assertAlmostEqual(conf.gb["split_improvement"], 0.001)

    def test_lastfmasia_uses_amazon_photo_ball_tier(self):
        self.assertEqual(BIG_BALL_PARAMS["lastfmasia"], BIG_BALL_PARAMS["amazonpho"])

    def test_edge_homophily_excludes_self_loops(self):
        labels = torch.tensor([0, 0, 1])
        edge_index = torch.tensor([
            [0, 1, 1, 2, 0, 1, 2],
            [1, 0, 2, 1, 0, 1, 2],
        ])
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (3, 3)).coalesce()

        self.assertAlmostEqual(_edge_homophily(labels, adj), 0.5)


if __name__ == "__main__":
    unittest.main()
