import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_mlp_beta_hardness as summary


class TestMLPBetaHardnessSummary(unittest.TestCase):
    def test_summary_outputs_decision_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = []
            for dataset in ["cora"]:
                for rate in [0.2, 0.4]:
                    for method, acc in [
                        ("frozengrace_gb_mlp_softdist", 0.70),
                        ("frozengrace_gb_mlp_softdist_beta05", 0.72),
                        ("frozengrace_gb_mlp_softdist_beta1", 0.74),
                        ("frozengrace_gb_mlp_softdist_beta1_shuffled", 0.71),
                    ]:
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "base_method": method,
                                "seed": 0,
                                "noise_type": "uniform",
                                "noise_rate": rate,
                                "status": "ok",
                                "test": acc,
                            }
                        )
            path = run_dir / "beta_uniform.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            loaded = summary.load_rows(run_dir)
            summary.validate_complete(loaded, ["cora"], [0])
            agg = summary.aggregate(loaded)
            deltas = summary.paired_deltas(loaded)
            decisions = summary.decision_rows(agg, ["cora"])
            summary.write_report(run_dir / "report.md", agg, deltas, decisions, ["cora"])

            self.assertEqual(len(decisions), 2)
            self.assertTrue(all(row["decision"] == "hardening_positive_seed0" for row in decisions))
            hard_delta = [
                row for row in deltas
                if row["comparison"] == "beta1_minus_soft" and float(row["noise_rate"]) == 0.2
            ][0]
            self.assertAlmostEqual(float(hard_delta["mean_delta"]), 0.04)
            self.assertIn("beta=1", (run_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
