import importlib
import unittest


class TestMLP32RobustStudentSummary(unittest.TestCase):
    def test_gate_passes_for_strong_synthetic_student(self):
        summary = importlib.import_module("scripts.summarize_mlp32_robust_student_distill")
        direct_rows = []
        distill_rows = []
        for index, dataset in enumerate(summary.DATASETS):
            common = {
                "dataset": dataset,
                "seed": "0",
                "noise_type": "uniform",
                "noise_rate": "0.4",
                "noise_scope": "train",
                "status": "ok",
                "split_hash": f"split-{dataset}",
                "noise_hash": f"noise-{dataset}",
                "modified_count": "10",
                "modified_train_count": "10",
                "modified_val_count": "0",
                "modified_test_count": "0",
            }
            direct_rows.append({**common, "method": summary.DIRECT, "test": "0.60"})
            distill_rows.extend(
                [
                    {**common, "method": summary.PLAIN, "test": "0.75"},
                    {
                        **common,
                        "method": summary.ROBUST,
                        "test": "0.78",
                        "teacher_best_test": "0.80",
                    },
                    {
                        **common,
                        "method": summary.SHUFFLED,
                        "test": "0.75",
                        "teacher_best_test": "0.80",
                    },
                ]
            )
        deltas, gate = summary.build_summary(direct_rows, distill_rows)
        self.assertEqual(len(deltas), 6)
        self.assertEqual(gate["pass"], 1)
        self.assertEqual(gate["above_plain_datasets"], 6)
        self.assertEqual(gate["structure_1pp_datasets"], 6)


if __name__ == "__main__":
    unittest.main()
