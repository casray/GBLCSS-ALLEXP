import importlib
import tempfile
import unittest
from pathlib import Path


class TestMLP32ResourceTable(unittest.TestCase):
    def test_complete_resource_table(self):
        summary = importlib.import_module("scripts.summarize_mlp32_resource_table")
        rows = []
        for dataset_index, dataset in enumerate(summary.DATASETS):
            for seed in range(5):
                common = {
                    "dataset": dataset,
                    "seed": str(seed),
                    "noise_type": "uniform",
                    "noise_rate": "0.4",
                    "noise_scope": "train",
                    "status": "ok",
                    "split_hash": f"split-{dataset}-{seed}",
                    "noise_hash": f"noise-{dataset}-{seed}",
                    "modified_count": "10",
                    "modified_train_count": "10",
                    "modified_val_count": "0",
                    "modified_test_count": "0",
                }
                for method, test in (
                    (summary.DIRECT, 0.60),
                    (summary.PLAIN, 0.80),
                    (summary.NRGNN, 0.79),
                    (summary.NRGNN_XH, 0.81),
                ):
                    rows.append({**common, "method": method, "test": str(test)})
        combined, aggregate, paired = summary.build_summary(rows)
        self.assertEqual(len(combined), 120)
        self.assertEqual(len(aggregate), 24)
        self.assertEqual(len(paired), 6)
        self.assertTrue(all(abs(row["plain_minus_nrgnn"] - 0.01) < 1e-12 for row in paired))

    def test_failed_historical_attempt_is_ignored_when_complete_ok_row_exists(self):
        summary = importlib.import_module("scripts.summarize_mlp32_resource_table")
        rows = []
        for dataset in summary.DATASETS:
            for seed in range(5):
                common = {
                    "dataset": dataset,
                    "seed": str(seed),
                    "noise_type": "uniform",
                    "noise_rate": "0.4",
                    "noise_scope": "train",
                    "split_hash": f"s-{dataset}-{seed}",
                    "noise_hash": f"n-{dataset}-{seed}",
                    "modified_count": "1",
                    "modified_train_count": "1",
                    "modified_val_count": "0",
                    "modified_test_count": "0",
                }
                for method in summary.METHODS:
                    rows.append({**common, "method": method, "status": "ok", "test": "0.5"})
        rows.append(
            {
                "dataset": "cora",
                "method": summary.NRGNN_XH,
                "seed": "0",
                "noise_type": "uniform",
                "noise_rate": "0.4",
                "noise_scope": "train",
                "status": "failed",
                "test": "",
            }
        )
        combined, _, _ = summary.build_summary(rows)
        self.assertEqual(len(combined), 120)

    def test_write_accepts_heterogeneous_diagnostic_columns(self):
        summary = importlib.import_module("scripts.summarize_mlp32_resource_table")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            summary._write(path, [{"a": 1}, {"a": 2, "teacher_best_test": 0.8}])
            text = path.read_text()
        self.assertIn("teacher_best_test", text)


if __name__ == "__main__":
    unittest.main()
