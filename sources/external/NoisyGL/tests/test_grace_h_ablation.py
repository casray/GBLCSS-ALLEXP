import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class GraceHAblationTest(unittest.TestCase):
    def test_standard_runner_registers_grace_h_methods(self):
        from scripts.run_standard_downstream import predictor_class_for

        self.assertEqual(
            predictor_class_for("grace_h_linear_ce").__name__,
            "grace_h_linear_ce_Predictor",
        )
        self.assertEqual(
            predictor_class_for("grace_h_mlp_ce").__name__,
            "grace_h_mlp_ce_Predictor",
        )

    def test_summary_filters_main_train_percent_and_reports_required_deltas(self):
        from scripts.summarize_grace_h_ablation_20260704 import summarize

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs"
            run_dir.mkdir()
            new_rows = [
                {"dataset": "cora", "method": "grace_h_linear_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.50},
                {"dataset": "cora", "method": "grace_h_mlp_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.52},
                {"dataset": "cora", "method": "mlp_xh_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "status": "ok", "test": 0.54},
            ]
            maintable_rows = [
                {"table": "labelrate", "train_percent": "0.05", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.99},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "mlp_ce", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.40},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.60},
                {"table": "main", "train_percent": "0.2", "dataset": "cora", "method": "frozengrace_gb_mlp_softdist_beta05_shuffled", "seed": 0, "noise_type": "uniform", "noise_rate": 0.4, "test_acc": 0.45},
            ]
            self._write_csv(run_dir / "grace_h_ablation_uniform0p4_seeds_0_1_2_3_4.csv", new_rows)
            self._write_csv(root / "maintable.csv", maintable_rows)

            summarize(run_dir=run_dir, maintable_path=root / "maintable.csv")

            deltas = self._read_csv(run_dir / "grace_h_ablation_paired_deltas.csv")
            lookup = {row["comparison"]: row for row in deltas}
            self.assertIn("softdist_vs_grace_h_linear", lookup)
            self.assertIn("softdist_vs_grace_h_mlp", lookup)
            self.assertIn("softdist_vs_mlp_xh", lookup)
            self.assertIn("grace_h_mlp_vs_mlp_ce", lookup)
            self.assertIn("softdist_vs_shuffled", lookup)
            self.assertAlmostEqual(float(lookup["softdist_vs_grace_h_mlp"]["delta"]), 0.08)
            self.assertAlmostEqual(float(lookup["softdist_vs_shuffled"]["delta"]), 0.15)

            aggregate = self._read_csv(run_dir / "grace_h_ablation_aggregate.csv")
            softdist = [
                row for row in aggregate
                if row["dataset"] == "cora" and row["method"] == "frozengrace_gb_mlp_softdist_beta05"
            ][0]
            self.assertAlmostEqual(float(softdist["mean"]), 0.60)

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
