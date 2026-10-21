import math
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_noise_robustness_completion import aggregate_rows, normalize_row, paired_deltas


class NoiseRobustnessSummaryTest(unittest.TestCase):
    def test_normalize_supports_standard_and_xh_rows(self):
        standard = normalize_row(
            {
                "dataset": "cora",
                "method": "gcn",
                "seed": "0",
                "noise_type": "uniform",
                "noise_rate": "0.5",
                "actual_train_noise_rate": "0.49",
                "status": "ok",
                "test": "0.8",
            },
            Path(__file__),
        )
        xh = normalize_row(
            {
                "dataset": "cora",
                "base_method": "gcn_xh_ce",
                "seed": "0",
                "noise_type": "uniform",
                "noise_rate": "0.5",
                "test": "0.81",
            },
            Path(__file__),
        )
        self.assertEqual(standard["method"], "gcn")
        self.assertEqual(xh["method"], "gcn_xh_ce")
        self.assertEqual(standard["actual_noise_rate"], 0.49)

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = [
            {
                "dataset": "cora",
                "noise_type": "uniform",
                "noise_rate": 0.5,
                "method": "gcn",
                "method_label": "GCN+CE",
                "test_acc": value,
            }
            for value in (0.7, 0.8, 0.9)
        ]
        summary = aggregate_rows(rows)[0]
        self.assertTrue(math.isclose(summary["mean"], 0.8))
        self.assertTrue(math.isclose(summary["std"], 0.1))

    def test_paired_delta_counts_seedwise_wins(self):
        rows = []
        for seed, left, right in ((0, 0.8, 0.7), (1, 0.6, 0.7)):
            for method, value in (
                ("frozengrace_gb_gcn_softdist_xh", left),
                ("gcn_xh_ce", right),
            ):
                rows.append(
                    {
                        "dataset": "cora",
                        "noise_type": "uniform",
                        "noise_rate": 0.5,
                        "method": method,
                        "seed": seed,
                        "test_acc": value,
                    }
                )
        delta = next(row for row in paired_deltas(rows) if row["comparison"] == "softdist_vs_gcn_xh")
        self.assertEqual(delta["n"], 2)
        self.assertEqual(delta["positive_count"], 1)
        self.assertTrue(math.isclose(delta["mean_delta"], 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
