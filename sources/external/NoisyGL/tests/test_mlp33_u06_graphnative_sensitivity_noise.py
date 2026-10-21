import importlib
import unittest


class TestMLP33Summary(unittest.TestCase):
    def _noise_rows(self):
        summary = importlib.import_module(
            "scripts.summarize_mlp33_u06_graphnative_sensitivity_noise"
        )
        rows = []
        for noise_type in summary.NOISE_TYPES:
            for dataset_index, dataset in enumerate(summary.DATASETS):
                for seed in summary.SEEDS:
                    lcss = 0.50 + 0.01 * dataset_index + 0.001 * seed
                    small = lcss - 0.01
                    combo = lcss + 0.02
                    shuffled = combo - 0.03
                    for method, test in (
                        (summary.LCSS, lcss),
                        (summary.SMALL_ONLY, small),
                        (summary.COMBO, combo),
                        (summary.SHUFFLED_COMBO, shuffled),
                    ):
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "seed": str(seed),
                                "noise_type": noise_type,
                                "noise_rate": "0.4",
                                "noise_scope": "train",
                                "status": "ok",
                                "test": str(test),
                                "split_hash": f"split-{dataset}-{seed}",
                                "noise_hash": f"noise-{noise_type}-{dataset}-{seed}",
                                "modified_count": "10",
                                "modified_train_count": "10",
                                "modified_val_count": "0",
                                "modified_test_count": "0",
                            }
                        )
        return rows

    def test_noise_type_matrix_and_frozen_gate(self):
        summary = importlib.import_module(
            "scripts.summarize_mlp33_u06_graphnative_sensitivity_noise"
        )
        raw, aggregate, seed_deltas, paired, gates = summary.build_noise_type_summary(
            self._noise_rows()
        )
        self.assertEqual(len(raw), 200)
        self.assertEqual(len(aggregate), 40)
        self.assertEqual(len(seed_deltas), 50)
        self.assertEqual(len(paired), 10)
        self.assertEqual(len(gates), 2)
        self.assertTrue(all(row["pass"] == 1 for row in gates))
        self.assertTrue(all(row["synergy_positive_seeds"] == 5 for row in paired))

    def test_noise_type_missing_seed_is_rejected(self):
        summary = importlib.import_module(
            "scripts.summarize_mlp33_u06_graphnative_sensitivity_noise"
        )
        rows = self._noise_rows()
        rows = [
            row
            for row in rows
            if not (
                row["noise_type"] == "pair"
                and row["dataset"] == "cora"
                and row["seed"] == "4"
                and row["method"] == summary.COMBO
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing"):
            summary.build_noise_type_summary(rows)

    def test_graphnative_gate_requires_three_of_five(self):
        summary = importlib.import_module(
            "scripts.summarize_mlp33_u06_graphnative_sensitivity_noise"
        )
        comparisons = [
            {"dataset": dataset, "best_xonly_minus_best_graph": delta}
            for dataset, delta in zip(summary.DATASETS, (-0.01, 0.00, -0.02, -0.03, 0.01))
        ]
        gate = summary.evaluate_graphnative_gate(comparisons)
        self.assertEqual(gate["qualifying_datasets"], 4)
        self.assertEqual(gate["pass"], 1)
        comparisons[1]["best_xonly_minus_best_graph"] = -0.021
        comparisons[4]["best_xonly_minus_best_graph"] = -0.021
        gate = summary.evaluate_graphnative_gate(comparisons)
        self.assertEqual(gate["qualifying_datasets"], 2)
        self.assertEqual(gate["pass"], 0)

    def test_misspec_summary_uses_correct_anchor(self):
        summary = importlib.import_module(
            "scripts.summarize_mlp33_u06_graphnative_sensitivity_noise"
        )
        scores = {
            0.4: {dataset: 0.50 for dataset in summary.SENSITIVITY_DATASETS},
            0.5: {dataset: 0.51 for dataset in summary.SENSITIVITY_DATASETS},
            0.6: {dataset: 0.52 for dataset in summary.SENSITIVITY_DATASETS},
        }
        rows = summary.build_misspec_summary(scores)
        self.assertEqual(len(rows), 9)
        wrong = next(row for row in rows if row["assumed_noise_rate"] == 0.4)
        self.assertAlmostEqual(wrong["delta_vs_correct"], -0.02)


if __name__ == "__main__":
    unittest.main()
