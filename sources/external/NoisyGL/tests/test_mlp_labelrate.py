import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mlp_labelrate import (
    METHODS,
    aggregate,
    load_rows,
    paired_deltas,
    trend_rows,
)


class TestMLPLabelRateSummary(unittest.TestCase):
    def test_summary_parses_train_percent_from_filename_and_trends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for alias, soft_acc, ce_acc in [
                ("0p05", 0.70, 0.55),
                ("0p10", 0.68, 0.56),
                ("0p50", 0.60, 0.57),
                ("0p65", 0.58, 0.56),
            ]:
                path = run_dir / f"labelrate_train{alias}_seeds_0.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["dataset", "method", "seed", "noise_type", "noise_rate", "status", "test"],
                    )
                    writer.writeheader()
                    for method in METHODS:
                        if method == "frozengrace_gb_mlp_softdist_beta05":
                            value = soft_acc
                        elif method == "mlp_ce":
                            value = ce_acc
                        else:
                            value = ce_acc - 0.01
                        writer.writerow(
                            {
                                "dataset": "cora",
                                "method": method,
                                "seed": 0,
                                "noise_type": "uniform",
                                "noise_rate": 0.4,
                                "status": "ok",
                                "test": value,
                            }
                        )

            rows = load_rows(run_dir)
            self.assertEqual(len(rows), 20)
            agg = aggregate(rows)
            self.assertEqual(len(agg), 20)
            deltas = paired_deltas(rows)
            lookup = {(row["train_percent"], row["comparison"]): row for row in deltas}
            self.assertAlmostEqual(lookup[(0.05, "beta05_vs_ce")]["mean_delta"], 0.15)
            trends = trend_rows(deltas)
            trend_lookup = {(row["dataset"], row["comparison"]): row for row in trends}
            self.assertEqual(trend_lookup[("cora", "beta05_vs_ce")]["trend_direction"], "low_label_advantage")


if __name__ == "__main__":
    unittest.main()
