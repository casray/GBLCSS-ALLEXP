import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "scripts"
    / "verify_tie_safe_full_grid_repeat.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("verify_repeat", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(seed=0):
    return {
        "dataset": "cora",
        "regime": "uniform0.2",
        "method": "frozengrace_gb_mlp_softdist_beta05",
        "seed": str(seed),
        "train": "0.8",
        "valid": "0.7",
        "test": "0.6",
        "split_hash": "split",
        "noise_hash": "noise",
        "modified_count": "100",
        "modified_train_count": "100",
        "modified_val_count": "0",
        "modified_test_count": "0",
        "feature_embedding_hash": "feature",
        "ball_embedding_hash": "ball",
        "ball_assignment_hash": "assignment",
        "tie_mode": "distribution_only",
        "tie_ball_count": "2",
        "tie_active_ball_count": "2",
        "tie_affected_node_count": "20",
        "tie_affected_alpha_sum": "3.5",
    }


class VerifyTieSafeFullGridRepeatTest(unittest.TestCase):
    def test_exact_repeat_passes(self):
        module = load_module()
        first = [row(0), row(1)]
        second = [dict(item) for item in first]
        audit, summary = module.compare_rows(first, second)
        self.assertEqual(len(audit), 2)
        self.assertEqual(summary["matched_rows"], 2)
        self.assertEqual(summary["mismatched_rows"], 0)
        self.assertEqual(summary["status"], "PASS")

    def test_metric_or_fingerprint_drift_fails(self):
        module = load_module()
        first = [row()]
        second = [dict(first[0])]
        second[0]["test"] = "0.61"
        second[0]["ball_assignment_hash"] = "drifted"
        audit, summary = module.compare_rows(first, second)
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["mismatched_rows"], 1)
        self.assertIn("test", audit[0]["mismatched_fields"])
        self.assertIn(
            "ball_assignment_hash", audit[0]["mismatched_fields"]
        )

    def test_missing_key_fails(self):
        module = load_module()
        first = [row(0), row(1)]
        second = [dict(first[0])]
        audit, summary = module.compare_rows(first, second)
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["missing_in_second"], 1)
        missing = next(item for item in audit if item["row_status"] == "MISSING_SECOND")
        self.assertEqual(int(missing["seed"]), 1)


if __name__ == "__main__":
    unittest.main()
