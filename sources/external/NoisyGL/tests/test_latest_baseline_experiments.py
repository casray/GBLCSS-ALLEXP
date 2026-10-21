import unittest
import multiprocessing
import tempfile
import time
from pathlib import Path


class LatestBaselineExperimentsTest(unittest.TestCase):
    def test_each_latest_method_has_exactly_twelve_unique_configs(self):
        from scripts.run_latest_baseline_experiments import build_grid

        for method in ("tfr", "procon", "gd2"):
            grid = build_grid(method)
            frozen = {tuple(sorted(config.items())) for config in grid}
            self.assertEqual(len(grid), 12)
            self.assertEqual(len(frozen), 12)

    def test_best_config_selection_uses_validation_not_test(self):
        from scripts.run_latest_baseline_experiments import select_best_config

        rows = [
            {"status": "ok", "valid": 0.7, "test": 0.99, "config_index": 0},
            {"status": "ok", "valid": 0.8, "test": 0.1, "config_index": 1},
        ]

        selected = select_best_config(rows)

        self.assertEqual(selected["config_index"], 1)

    def test_protocol_audit_rejects_mismatched_hashes(self):
        from scripts.run_latest_baseline_experiments import audit_protocol_hashes

        rows = [
            {"dataset": "cora", "seed": 0, "split_hash": "a", "noise_hash": "n"},
            {"dataset": "cora", "seed": 0, "split_hash": "b", "noise_hash": "n"},
        ]

        with self.assertRaises(ValueError):
            audit_protocol_hashes(rows)

    def test_summary_requires_all_five_seeds(self):
        from scripts.run_latest_baseline_experiments import summarize_rows

        rows = [
            {"dataset": "cora", "method": "gd2", "seed": seed, "status": "ok", "test": 0.7}
            for seed in range(4)
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary[0]["status"], "incomplete")
        self.assertEqual(summary[0]["mean"], "")

    def test_pending_combinations_skip_existing_keys(self):
        from scripts.run_latest_baseline_experiments import pending_combinations

        combinations = [
            ("cora", "unionnet", 0),
            ("cora", "unionnet", 1),
            ("cora", "crgnn", 0),
        ]
        rows = [{"dataset": "cora", "method": "unionnet", "seed": "0"}]

        self.assertEqual(
            pending_combinations(combinations, rows),
            [("cora", "unionnet", 1), ("cora", "crgnn", 0)],
        )

    def test_timeout_row_keeps_protocol_fingerprints(self):
        from scripts.run_latest_baseline_experiments import make_timeout_row

        template = {
            "dataset": "cora",
            "method": "gcn",
            "seed": "2",
            "noise_type": "uniform",
            "noise_rate": "0.4",
            "noise_scope": "train",
            "split_hash": "split-2",
            "noise_hash": "noise-2",
            "modified_count": "216",
            "modified_train_count": "216",
            "modified_val_count": "0",
            "modified_test_count": "0",
        }

        row = make_timeout_row(template, "unionnet", 1800.0)

        self.assertEqual(row["method"], "unionnet")
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["split_hash"], "split-2")
        self.assertEqual(row["noise_hash"], "noise-2")
        self.assertEqual(row["wall_time_seconds"], 1800.0)
        self.assertEqual(row["valid"], "")
        self.assertIn("1800", row["error"])

    def test_upsert_replaces_matching_key_without_duplicates(self):
        from scripts.run_latest_baseline_experiments import upsert_row

        rows = [
            {"dataset": "cora", "method": "unionnet", "seed": "0", "status": "timeout"},
            {"dataset": "cora", "method": "unionnet", "seed": "1", "status": "ok"},
        ]
        replacement = {
            "dataset": "cora",
            "method": "unionnet",
            "seed": 0,
            "status": "ok",
        }

        updated = upsert_row(rows, replacement)

        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0]["status"], "ok")
        self.assertEqual(updated[1]["seed"], "1")

    def test_run_with_timeout_terminates_slow_child(self):
        from scripts.run_latest_baseline_experiments import run_with_timeout

        completed, result = run_with_timeout(time.sleep, (2.0,), 0.05)

        self.assertFalse(completed)
        self.assertIsNone(result)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_run_appendix_resumes_after_existing_row(self):
        from scripts.run_latest_baseline_experiments import read_csv, run_appendix
        from scripts.run_standard_downstream import write_csv

        existing = self._protocol_row("cora", "unionnet", 0)
        completed = self._protocol_row("cora", "unionnet", 1)
        calls = []

        def fake_runner(target, target_args, timeout_seconds):
            calls.append((target_args, timeout_seconds))
            return True, completed

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_csv(output_dir / "appendix_raw.csv", [existing])
            write_csv(output_dir / "formal_main_raw.csv", [
                self._protocol_row("cora", "gcn", 0),
                self._protocol_row("cora", "gcn", 1),
            ])

            rows = run_appendix(
                output_dir,
                "cpu",
                timeout_seconds=12.0,
                combinations=[
                    ("cora", "unionnet", 0),
                    ("cora", "unionnet", 1),
                ],
                timed_runner=fake_runner,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0][2], 1)
            self.assertEqual(len(read_csv(output_dir / "appendix_raw.csv")), 2)

    def test_run_appendix_records_timeout_with_formal_protocol(self):
        from scripts.run_latest_baseline_experiments import run_appendix
        from scripts.run_standard_downstream import write_csv

        template = self._protocol_row("cora", "gcn", 2)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_csv(output_dir / "formal_main_raw.csv", [template])

            rows = run_appendix(
                output_dir,
                "cpu",
                timeout_seconds=30.0,
                combinations=[("cora", "unionnet", 2)],
                timed_runner=lambda *args: (False, None),
            )

            self.assertEqual(rows[0]["status"], "timeout")
            self.assertEqual(rows[0]["split_hash"], template["split_hash"])
            self.assertEqual(rows[0]["noise_hash"], template["noise_hash"])

    def test_replace_with_retry_handles_transient_windows_lock(self):
        from scripts.run_latest_baseline_experiments import replace_with_retry

        class FlakySource:
            def __init__(self):
                self.calls = 0

            def replace(self, target):
                self.calls += 1
                if self.calls == 1:
                    raise PermissionError("temporarily locked")

        source = FlakySource()

        replace_with_retry(source, object(), attempts=2, delay_seconds=0)

        self.assertEqual(source.calls, 2)

    @staticmethod
    def _protocol_row(dataset, method, seed):
        return {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "noise_scope": "train",
            "split_hash": f"split-{seed}",
            "noise_hash": f"noise-{seed}",
            "config_id": "config",
            "modified_count": 10,
            "actual_noise_rate": 0.1,
            "modified_train_count": 10,
            "actual_train_noise_rate": 0.4,
            "modified_val_count": 0,
            "modified_test_count": 0,
            "wall_time_seconds": 1.0,
            "parameter_count": 5,
            "status": "ok",
            "error": "",
            "train": 0.7,
            "valid": 0.6,
            "test": 0.5,
        }


if __name__ == "__main__":
    unittest.main()
