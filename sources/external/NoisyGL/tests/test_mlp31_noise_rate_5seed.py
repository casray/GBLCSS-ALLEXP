import importlib
import unittest


class TestMLP31NoiseRateFiveSeed(unittest.TestCase):
    def _rows(self):
        summary = importlib.import_module("scripts.summarize_mlp31_noise_rate_5seed")
        rows = []
        for rate, datasets in summary.DATASETS_BY_RATE.items():
            for dataset_index, dataset in enumerate(datasets):
                for seed in summary.SEEDS:
                    lcss = 0.50 + 0.01 * dataset_index + 0.001 * seed
                    small = lcss - 0.01
                    combo = lcss + 0.02
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
                                "seed": str(seed),
                                "noise_type": "uniform",
                                "noise_rate": str(rate),
                                "noise_scope": "train",
                                "status": "ok",
                                "test": str(test),
                                "split_hash": f"split-{dataset}-{seed}",
                                "noise_hash": f"noise-{dataset}-{rate}-{seed}",
                                "modified_count": "10",
                                "modified_train_count": "10",
                                "modified_val_count": "0",
                                "modified_test_count": "0",
                            }
                        )
        return rows

    def test_complete_matrix_and_gates(self):
        summary = importlib.import_module("scripts.summarize_mlp31_noise_rate_5seed")
        combined, aggregate, seed_deltas, paired, gates = summary.build_summary(self._rows())
        self.assertEqual(len(combined), 240)
        self.assertEqual(len(aggregate), 48)
        self.assertEqual(len(seed_deltas), 60)
        self.assertEqual(len(paired), 12)
        self.assertEqual(len(gates), 2)
        self.assertTrue(all(row["pass"] == 1 for row in gates))
        self.assertTrue(all(row["synergy_positive_seeds"] == 5 for row in paired))

    def test_missing_seed_is_rejected(self):
        summary = importlib.import_module("scripts.summarize_mlp31_noise_rate_5seed")
        rows = self._rows()
        rows = [
            row
            for row in rows
            if not (
                row["dataset"] == "cora"
                and row["noise_rate"] == "0.4"
                and row["seed"] == "4"
                and row["method"] == summary.SMALL_COMBO
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing"):
            summary.build_summary(rows)


if __name__ == "__main__":
    unittest.main()
