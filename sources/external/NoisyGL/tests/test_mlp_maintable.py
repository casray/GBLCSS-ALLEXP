import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mlp_maintable import (
    MAIN_METHODS,
    aggregate_rows,
    expected_keys,
    load_rows,
    missing_keys,
    paired_deltas,
)


class MLPMainTableTests(unittest.TestCase):
    def test_expected_grid_counts_main_and_labelrate(self):
        self.assertEqual(len(expected_keys()), 850)
        self.assertIn(("main", "pubmed", "instance", 0.4, 0.2, "mlp_fine", 4), expected_keys())
        self.assertIn(
            ("labelrate", "cora", "uniform", 0.4, 0.05, "frozengrace_gb_mlp_softdist_beta05", 0),
            expected_keys(),
        )

    def test_summary_parses_main_and_labelrate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            main_path = run_dir / "main_uniform0p4_seeds_0.csv"
            with main_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["dataset", "method", "seed", "noise_type", "noise_rate", "status", "test"],
                )
                writer.writeheader()
                for method in MAIN_METHODS:
                    writer.writerow(
                        {
                            "dataset": "cora",
                            "method": method,
                            "seed": 0,
                            "noise_type": "uniform",
                            "noise_rate": 0.4,
                            "status": "ok",
                            "test": 0.7 if method == "frozengrace_gb_mlp_softdist_beta05" else 0.6,
                        }
                    )
            label_path = run_dir / "labelrate_train0p05_seeds_0.csv"
            with label_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["dataset", "method", "seed", "noise_type", "noise_rate", "status", "test"],
                )
                writer.writeheader()
                for method in [
                    "mlp_small_loss",
                    "mlp_jocor",
                    "mlp_fine",
                    "frozengrace_gb_mlp_softdist_beta05",
                    "frozengrace_gb_mlp_softdist_beta05_shuffled",
                ]:
                    writer.writerow(
                        {
                            "dataset": "cora",
                            "method": method,
                            "seed": 0,
                            "noise_type": "uniform",
                            "noise_rate": 0.4,
                            "status": "ok",
                            "test": 0.7 if method == "frozengrace_gb_mlp_softdist_beta05" else 0.6,
                        }
                    )

            rows = load_rows(run_dir)
            self.assertEqual(len(rows), 15)
            self.assertEqual(len(aggregate_rows(rows)), 15)
            deltas = paired_deltas(rows)
            self.assertTrue(any(row["comparison"] == "beta05_vs_mlp_fine" for row in deltas))
            self.assertGreater(len(missing_keys(rows)), 0)

    def test_runner_mentions_rocm_and_required_methods(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_mlp_maintable_20260701.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn(".venv-rocm", script)
        self.assertIn("mlp_fine", script)
        self.assertIn("mlp_dividemix_lite", script)
        self.assertIn("--noise_scope train", script)


if __name__ == "__main__":
    unittest.main()
