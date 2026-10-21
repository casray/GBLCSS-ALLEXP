import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_v32_alpha_calibration.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_v32_alpha_calibration_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AlphaCalibrationTest(unittest.TestCase):
    def test_summarize_bins_reports_target_quality(self):
        module = load_module()
        records = {
            "alpha": np.array([0.0, 0.1, 0.5, 0.9]),
            "target_correct": np.array([0.0, 1.0, 1.0, 1.0]),
            "clean_mass": np.array([0.2, 0.8, 0.7, 0.9]),
            "brier": np.array([1.0, 0.1, 0.2, 0.02]),
            "train_votes": np.array([2.0, 3.0, 8.0, 12.0]),
            "tie": np.array([1.0, 0.0, 0.0, 0.0]),
        }

        rows = module.summarize_bins("toy", records)

        self.assertEqual(sum(row["count"] for row in rows), 4)
        first = rows[0]
        self.assertEqual(first["alpha_bin"], "[0.0,0.2)")
        self.assertEqual(first["count"], 2)
        self.assertAlmostEqual(first["mean_alpha"], 0.05)
        self.assertAlmostEqual(first["target_argmax_accuracy"], 0.5)
        self.assertAlmostEqual(first["mean_train_votes"], 2.5)
        self.assertAlmostEqual(first["tie_rate"], 0.5)

    def test_empty_bins_are_not_emitted(self):
        module = load_module()
        records = {
            "alpha": np.array([0.45]),
            "target_correct": np.array([1.0]),
            "clean_mass": np.array([0.8]),
            "brier": np.array([0.05]),
            "train_votes": np.array([6.0]),
            "tie": np.array([0.0]),
        }

        rows = module.summarize_bins("toy", records)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alpha_bin"], "[0.4,0.6)")


if __name__ == "__main__":
    unittest.main()
