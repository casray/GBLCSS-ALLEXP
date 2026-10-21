import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class LinearAblationTest(unittest.TestCase):
    def test_standard_runner_registers_linear_ablation_methods(self):
        from scripts.run_standard_downstream import predictor_class_for

        self.assertEqual(predictor_class_for("linear_ce").__name__, "linear_ce_Predictor")
        self.assertEqual(
            predictor_class_for("frozengrace_gb_linear_softdist_beta05").__name__,
            "frozengrace_gb_linear_softdist_beta05_Predictor",
        )
        self.assertEqual(
            predictor_class_for("frozengrace_gb_linear_softdist_beta05_shuffled").__name__,
            "frozengrace_gb_linear_softdist_beta05_shuffled_Predictor",
        )

    def test_summary_reports_linear_softdist_and_mlp_gap(self):
        from scripts.summarize_linear_ablation_20260704 import summarize

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs"
            run_dir.mkdir()
            linear_rows = [
                {"dataset": "cora", "method": "linear_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.30},
                {"dataset": "cora", "method": "frozengrace_gb_linear_softdist_beta05", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.50},
                {"dataset": "cora", "method": "frozengrace_gb_linear_softdist_beta05_shuffled", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.35},
            ]
            maintable_rows = [
                {"table": "labelrate", "train_percent": "0.05", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.99},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "mlp_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.40},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.60},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05_shuffled", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.45},
            ]
            self._write_csv(run_dir / "linear_ablation_uniform0p4_seeds_0_1_2_3_4.csv", linear_rows)
            self._write_csv(root / "maintable.csv", maintable_rows)

            summarize(run_dir=run_dir, maintable_path=root / "maintable.csv")

            deltas = self._read_csv(run_dir / "linear_ablation_paired_deltas.csv")
            lookup = {row["comparison"]: row for row in deltas}
            self.assertIn("softdist_linear_vs_linear_ce", lookup)
            self.assertIn("softdist_linear_vs_linear_shuffled", lookup)
            self.assertIn("softdist_mlp_vs_softdist_linear", lookup)
            self.assertIn("softdist_linear_vs_mlp_ce", lookup)
            self.assertAlmostEqual(float(lookup["softdist_linear_vs_linear_ce"]["delta"]), 0.20)
            self.assertAlmostEqual(float(lookup["softdist_mlp_vs_softdist_linear"]["delta"]), 0.10)

            aggregate = self._read_csv(run_dir / "linear_ablation_aggregate.csv")
            softdist_mlp = [
                row for row in aggregate
                if row["dataset"] == "cora" and row["method"] == "frozengrace_gb_mlp_softdist_beta05"
            ][0]
            self.assertAlmostEqual(float(softdist_mlp["mean"]), 0.60)

    def _write_csv(self, path, rows):
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path):
        with path.open(newline="") as f:
            return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
