import importlib
import unittest


class TestMLP30NoiseRateGrid(unittest.TestCase):
    def test_complete_grid_and_rate_gates(self):
        summary = importlib.import_module("scripts.summarize_mlp30_noise_rate_grid")
        rows = []
        for dataset_index, dataset in enumerate(summary.DATASETS):
            for rate in summary.RATES:
                lcss = 0.50 + 0.01 * dataset_index
                small = lcss - 0.01
                gain = {0.2: 0.005, 0.4: 0.015, 0.6: 0.03}[rate]
                combo = lcss + gain
                shuffled = combo - 0.02
                for method, test in (
                    (summary.LCSS, lcss),
                    (summary.SMALL_ONLY, small),
                    (summary.SMALL_COMBO, combo),
                    (summary.SMALL_SHUFFLED, shuffled),
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": "0",
                            "noise_type": "uniform",
                            "noise_rate": str(rate),
                            "noise_scope": "train",
                            "status": "ok",
                            "test": str(test),
                            "split_hash": f"split-{dataset}",
                            "noise_hash": f"noise-{dataset}-{rate}",
                            "modified_count": "10",
                            "modified_train_count": "10",
                            "modified_val_count": "0",
                            "modified_test_count": "0",
                        }
                    )

        combined, deltas, gates, trends = summary.build_summary(rows)

        self.assertEqual(len(combined), 108)
        self.assertEqual(len(deltas), 27)
        self.assertEqual(len(trends), 9)
        gate_by_rate = {row["noise_rate"]: row for row in gates}
        self.assertEqual(gate_by_rate[0.2]["pass"], 0)
        self.assertEqual(gate_by_rate[0.4]["pass"], 1)
        self.assertEqual(gate_by_rate[0.6]["pass"], 1)

    def test_duplicate_grid_cell_is_rejected(self):
        summary = importlib.import_module("scripts.summarize_mlp30_noise_rate_grid")
        row = {
            "dataset": "cora",
            "method": summary.LCSS,
            "seed": "0",
            "noise_type": "uniform",
            "noise_rate": "0.2",
            "noise_scope": "train",
            "status": "ok",
            "test": "0.5",
            "split_hash": "s",
            "noise_hash": "n",
            "modified_count": "1",
            "modified_train_count": "1",
            "modified_val_count": "0",
            "modified_test_count": "0",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.build_summary([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
