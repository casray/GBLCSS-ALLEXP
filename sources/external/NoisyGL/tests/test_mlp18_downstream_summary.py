import unittest

from scripts.summarize_mlp18_ellipsoid_feasibility import aggregate_rows, paired_delta_rows


class MLP18DownstreamSummaryTest(unittest.TestCase):
    def test_aggregate_rows_reports_mean_std(self):
        rows = [
            {"dataset": "cora", "method": "a", "seed": 0, "test": 0.5},
            {"dataset": "cora", "method": "a", "seed": 1, "test": 0.7},
            {"dataset": "cora", "method": "b", "seed": 0, "test": 0.4},
        ]

        aggregate = aggregate_rows(rows)
        by_method = {row["method"]: row for row in aggregate}

        self.assertAlmostEqual(by_method["a"]["test_mean"], 0.6)
        self.assertAlmostEqual(by_method["a"]["test_std"], 0.1414213562)
        self.assertEqual(by_method["a"]["n"], 2)

    def test_paired_delta_rows_align_by_dataset_and_seed(self):
        rows = [
            {"dataset": "cora", "method": "candidate", "seed": 0, "test": 0.55},
            {"dataset": "cora", "method": "current", "seed": 0, "test": 0.50},
            {"dataset": "cora", "method": "candidate", "seed": 1, "test": 0.60},
            {"dataset": "citeseer", "method": "current", "seed": 1, "test": 0.40},
        ]

        deltas = paired_delta_rows(rows, "candidate", "current", "candidate_vs_current")

        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["dataset"], "cora")
        self.assertEqual(deltas[0]["seed"], 0)
        self.assertAlmostEqual(deltas[0]["test_delta"], 0.05)


if __name__ == "__main__":
    unittest.main()
