import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_validation_scope_sensitivity.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "validation_scope_sensitivity_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_noisy_rows(runner):
    rows = []
    for dataset_index, dataset in enumerate(runner.DATASETS):
        for seed in runner.SEEDS:
            split_hash = f"split-{dataset}-{seed}"
            train_hash = f"train-{dataset}-{seed}"
            for method_index, method in enumerate(runner.METHODS):
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": str(seed),
                    "noise_type": "uniform",
                    "noise_rate": "0.4",
                    "noise_scope": "train_val",
                    "split_type": "percent",
                    "train_percent": "0.2",
                    "val_percent": "0.1",
                    "test_percent": "0.7",
                    "split_hash": split_hash,
                    "noise_hash": f"full-{dataset}-{seed}",
                    "train_noise_hash": train_hash,
                    "val_noise_hash": f"val-{dataset}-{seed}",
                    "modified_count": "100",
                    "modified_train_count": "60",
                    "modified_val_count": "40",
                    "modified_test_count": "0",
                    "status": "ok",
                    "valid": "0.5",
                    "test": str(
                        0.50
                        + 0.01 * dataset_index
                        + 0.005 * method_index
                        + 0.001 * seed
                    ),
                    "tie_mode": (
                        "distribution_only"
                        if method in runner.SOFTDIST_METHODS
                        else ""
                    ),
                })
    return rows


class ValidationScopeSensitivityTest(unittest.TestCase):
    def test_frozen_grid_has_45_commands_and_90_rows(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = runner.build_command_specs(
                python=Path("python.exe"),
                noisygl_root=Path("NoisyGL"),
                output_dir=root,
            )

        self.assertEqual(len(specs), 45)
        self.assertEqual(sum(spec.expected_rows for spec in specs), 90)
        self.assertEqual(runner.DATASETS, ("cora", "citeseer", "corafull"))
        self.assertEqual(runner.SEEDS, (0, 1, 2, 3, 4))
        self.assertIn("gsdn", runner.METHODS)
        self.assertNotIn("gsdn_faithful_core", runner.METHODS)

        for spec in specs:
            command = list(spec.command)
            self.assertEqual(
                command[command.index("--noise_scope") + 1],
                "train_val",
            )
            self.assertEqual(
                command[command.index("--noise_type") + 1],
                "uniform",
            )
            self.assertEqual(
                command[command.index("--noise_rate") + 1],
                "0.4",
            )
            overrides = json.loads(
                command[command.index("--config_overrides") + 1]
            )
            if spec.group == "softdist":
                self.assertEqual(overrides, {
                    "gb.tie_mode": "distribution_only",
                })
            else:
                self.assertEqual(overrides, {})

    def test_audit_requires_hash_pairing_and_clean_test_labels(self):
        runner = load_runner()
        rows = fake_noisy_rows(runner)
        audit = runner.audit_noisy_rows(rows)
        self.assertEqual(len(audit), 15)
        self.assertTrue(all(row["status"] == "PASS" for row in audit))

        broken = [dict(row) for row in rows]
        broken[0]["train_noise_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "train_noise_hash"):
            runner.audit_noisy_rows(broken)

        broken = [dict(row) for row in rows]
        broken[0]["modified_test_count"] = "1"
        with self.assertRaisesRegex(ValueError, "test labels"):
            runner.audit_noisy_rows(broken)

    def test_exact_complete_requires_two_expected_method_rows(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = runner.build_command_specs(
                python=Path("python.exe"),
                noisygl_root=Path("NoisyGL"),
                output_dir=root,
                seeds=(0,),
            )[0]
            rows = [
                row
                for row in fake_noisy_rows(runner)
                if row["dataset"] == spec.dataset
                and int(row["seed"]) == spec.seed
                and row["method"] in spec.methods
            ]
            runner.write_csv(spec.raw_path, rows)
            self.assertTrue(runner.is_exact_complete(spec))

            rows.pop()
            runner.write_csv(spec.raw_path, rows)
            self.assertFalse(runner.is_exact_complete(spec))


if __name__ == "__main__":
    unittest.main()
