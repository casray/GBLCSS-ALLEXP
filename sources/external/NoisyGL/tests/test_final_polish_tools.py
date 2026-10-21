import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plot_cdr_distance import summarize_distance_bins
from plot_tsne_embeddings import class_distance_summary
from run_ball_count_sensitivity import summarize_rows
from predictor.module.TSS import Personalized_PageRank


class FinalPolishToolsTest(unittest.TestCase):
    def test_personalized_pagerank_respects_cpu_device(self):
        adj = torch.eye(3).to_sparse()

        pi = Personalized_PageRank(0.85, adj)

        self.assertEqual(pi.device.type, "cpu")
        self.assertEqual(tuple(pi.shape), (3, 3))

    def test_ball_count_summary_groups_by_max_balls(self):
        rows = [
            {"dataset": "cora", "method": "softdist", "max_balls": 16, "test": 0.8, "num_balls": 16},
            {"dataset": "cora", "method": "softdist", "max_balls": 16, "test": 0.82, "num_balls": 15},
            {"dataset": "cora", "method": "softdist", "max_balls": 32, "test": 0.81, "num_balls": 30},
        ]

        summary = summarize_rows(rows)

        by_key = {(row["dataset"], row["method"], row["max_balls"]): row for row in summary}
        self.assertAlmostEqual(by_key[("cora", "softdist", 16)]["test_mean"], 0.81)
        self.assertAlmostEqual(by_key[("cora", "softdist", 16)]["num_balls_mean"], 15.5)
        self.assertEqual(by_key[("cora", "softdist", 32)]["n"], 1)

    def test_cdr_distance_summary_filters_variant_and_all_bin(self):
        rows = [
            {"dataset": "cora", "seed": 0, "variant": "baseline", "bin": "0", "cdr": 0.1, "ncr": 0.2},
            {"dataset": "cora", "seed": 1, "variant": "baseline", "bin": "0", "cdr": 0.3, "ncr": 0.4},
            {"dataset": "cora", "seed": 0, "variant": "baseline", "bin": "all", "cdr": 0.9, "ncr": 0.9},
            {"dataset": "cora", "seed": 0, "variant": "other", "bin": "0", "cdr": 0.9, "ncr": 0.9},
        ]

        summary = summarize_distance_bins(rows, variant="baseline")

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["dataset"], "cora")
        self.assertEqual(summary[0]["bin"], 0)
        self.assertAlmostEqual(summary[0]["cdr_mean"], 0.2)
        self.assertAlmostEqual(summary[0]["ncr_mean"], 0.3)

    def test_class_distance_summary_reports_class_separation(self):
        embeddings = torch.tensor([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]])
        labels = torch.tensor([0, 0, 1, 1])

        summary = class_distance_summary(embeddings, labels)

        self.assertLess(summary["mean_within_class_distance"], summary["mean_between_class_distance"])


if __name__ == "__main__":
    unittest.main()
