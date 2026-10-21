import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_embedding_geometry import gate_candidates, summarize_rows


class EmbeddingGeometryDiagnosticTest(unittest.TestCase):
    def test_summarize_rows_computes_mean_metrics(self):
        rows = [
            {"dataset": "cora", "ssl_method": "grace", "num_balls": 50, "true_all_purity_weighted": 0.7},
            {"dataset": "cora", "ssl_method": "grace", "num_balls": 52, "true_all_purity_weighted": 0.8},
        ]

        summary = summarize_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["dataset"], "cora")
        self.assertEqual(summary[0]["ssl_method"], "grace")
        self.assertAlmostEqual(summary[0]["num_balls_mean"], 51.0)
        self.assertAlmostEqual(summary[0]["true_all_purity_weighted_mean"], 0.75)

    def test_gate_candidates_ignores_probe_accuracy(self):
        summary = [
            {
                "dataset": "cora",
                "ssl_method": "grace",
                "num_balls_mean": 52.0,
                "true_all_purity_weighted_mean": 0.78,
                "clean_probe_mean": 0.80,
            },
            {
                "dataset": "cora",
                "ssl_method": "grace_unif0p1",
                "num_balls_mean": 40.0,
                "true_all_purity_weighted_mean": 0.79,
                "clean_probe_mean": 0.10,
            },
            {
                "dataset": "cora",
                "ssl_method": "grace_unif0p5",
                "num_balls_mean": 60.0,
                "true_all_purity_weighted_mean": 0.77,
                "clean_probe_mean": 0.99,
            },
        ]

        decisions = gate_candidates(summary, anchor_method="grace")

        by_method = {(row["dataset"], row["ssl_method"]): row for row in decisions}
        self.assertTrue(by_method[("cora", "grace_unif0p1")]["passes_geometry_gate"])
        self.assertFalse(by_method[("cora", "grace_unif0p5")]["passes_geometry_gate"])

    def test_gate_candidates_allows_configured_purity_tolerance(self):
        summary = [
            {
                "dataset": "cora",
                "ssl_method": "grace",
                "num_balls_mean": 100.0,
                "true_all_purity_weighted_mean": 0.8000,
            },
            {
                "dataset": "cora",
                "ssl_method": "grace_angproj_knn10_anchor1",
                "num_balls_mean": 50.0,
                "true_all_purity_weighted_mean": 0.7960,
            },
        ]

        decisions = gate_candidates(summary, anchor_method="grace", purity_tolerance=0.005)

        self.assertTrue(decisions[0]["purity_ok"])
        self.assertTrue(decisions[0]["passes_geometry_gate"])


if __name__ == "__main__":
    unittest.main()
