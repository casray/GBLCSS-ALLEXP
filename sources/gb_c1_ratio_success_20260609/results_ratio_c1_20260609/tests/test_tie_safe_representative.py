import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper_pipeline"
    / "scripts"
    / "run_tie_safe_representative.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("tie_safe_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_rows():
    rows = []
    mode_offsets = {
        "min_index": [0.0] * 5,
        "alpha_zero": [-0.002, -0.001, 0.0, 0.001, 0.002],
        "distribution_only": [0.012, 0.011, 0.013, 0.012, 0.014],
    }
    for dataset_index, dataset in enumerate(("cora", "corafull")):
        for mode, offsets in mode_offsets.items():
            for seed, offset in enumerate(offsets):
                base = 0.40 - 0.10 * dataset_index + 0.001 * seed
                rows.append(
                    {
                        "dataset": dataset,
                        "method": "frozengrace_gb_mlp_softdist_beta05",
                        "base_method": "frozengrace_gb_mlp_softdist_beta05",
                        "ssl_method": "grace",
                        "seed": str(seed),
                        "noise_type": "uniform",
                        "noise_rate": "0.6",
                        "noise_scope": "train",
                        "split_type": "percent",
                        "train_percent": "0.2",
                        "val_percent": "0.1",
                        "test_percent": "0.7",
                        "split_hash": f"split-{dataset}-{seed}",
                        "noise_hash": f"noise-{dataset}-{seed}",
                        "modified_count": str(100 + seed),
                        "modified_train_count": str(100 + seed),
                        "modified_val_count": "0",
                        "modified_test_count": "0",
                        "feature_embedding_path": f"C:/{dataset}/grace/H.pt",
                        "ball_embedding_path": f"C:/{dataset}/grace/H.pt",
                        "status": "ok",
                        "valid": str(base - 0.02),
                        "test": str(base + offset),
                        "tie_mode": mode,
                        "tie_ball_count": "3",
                        "tie_active_ball_count": "3",
                        "tie_affected_node_count": "25",
                        "tie_affected_alpha_sum": "4.5",
                        "ball_assignment_hash": f"assignment-{dataset}-{seed}",
                    }
                )
    return rows


class TieSafeRepresentativeTest(unittest.TestCase):
    def test_training_subprocess_environment_forces_deterministic_thread_counts(self):
        module = load_module()
        env = module.deterministic_training_env({"KEEP_ME": "yes", "OMP_NUM_THREADS": "8"})
        self.assertEqual(env["KEEP_ME"], "yes")
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(env[key], "1")
        self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(env["MKL_DYNAMIC"], "FALSE")
        self.assertEqual(env["PYTHONHASHSEED"], "0")

    def test_read_csv_accepts_windows_cp936_output_with_chinese_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp936.csv"
            path.write_bytes(
                "dataset,embedding_path,test\r\n"
                "cora,C:\\实验\\H.pt,0.5\r\n".encode("cp936")
            )
            rows = module.read_csv(path)
            self.assertEqual(rows[0]["embedding_path"], "C:\\实验\\H.pt")

    def test_raw_file_complete_requires_exact_five_ok_rows_and_fields(self):
        module = load_module()
        rows = [r for r in fake_rows() if r["dataset"] == "cora" and r["tie_mode"] == "min_index"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            module.write_csv(path, rows)
            self.assertTrue(module.raw_file_complete(path, dataset="cora", tie_mode="min_index"))

            rows[0]["ball_assignment_hash"] = ""
            module.write_csv(path, rows)
            self.assertFalse(module.raw_file_complete(path, dataset="cora", tie_mode="min_index"))

            module.write_csv(path, rows[:4])
            self.assertFalse(module.raw_file_complete(path, dataset="cora", tie_mode="min_index"))

    def test_protocol_drift_is_rejected(self):
        module = load_module()
        rows = fake_rows()
        target = next(
            row
            for row in rows
            if row["dataset"] == "cora"
            and row["tie_mode"] == "distribution_only"
            and row["seed"] == "2"
        )
        target["noise_hash"] = "drifted"
        with self.assertRaisesRegex(ValueError, "protocol drift"):
            module.validate_protocol_consistency(rows)

    def test_summary_uses_paired_seed_deltas_and_frozen_rules(self):
        module = load_module()
        summaries, paired = module.summarize_rows(fake_rows())

        target = next(
            row
            for row in summaries
            if row["dataset"] == "cora" and row["tie_mode"] == "distribution_only"
        )
        self.assertAlmostEqual(float(target["test_mean"]), 0.4144)
        self.assertAlmostEqual(float(target["paired_delta_mean"]), 0.0124)
        self.assertEqual(int(target["positive_seed_count"]), 5)
        self.assertEqual(int(target["negative_seed_count"]), 0)
        self.assertEqual(int(target["zero_seed_count"]), 0)
        self.assertEqual(int(target["practical_materiality"]), 1)
        self.assertEqual(int(target["statistical_detectability"]), 1)
        self.assertEqual(target["role"], "fixed_primary_repair")
        self.assertEqual(int(target["full_grid_required_after_integrity_pass"]), 1)

        ablation = next(
            row
            for row in summaries
            if row["dataset"] == "cora" and row["tie_mode"] == "alpha_zero"
        )
        self.assertEqual(ablation["role"], "mechanism_ablation")
        self.assertEqual(int(ablation["practical_materiality"]), 0)

        target_deltas = [
            float(row["delta"])
            for row in paired
            if row["dataset"] == "cora" and row["tie_mode"] == "distribution_only"
        ]
        self.assertEqual(len(target_deltas), 5)
        self.assertTrue(math.isclose(sum(target_deltas) / 5, 0.0124))

    def test_dry_run_writes_manifest_without_starting_training(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            noisygl = root / "NoisyGL"
            output = root / "out"
            python = root / "python.exe"
            runner = noisygl / "scripts" / "run_ssl_swap_downstream.py"
            target_builder = noisygl / "utils" / "frozen_grace_targets.py"
            predictor = noisygl / "predictor" / "FrozenGraceGB_GCN_Predictor.py"
            embedding = root / "embeddings_v2"
            plan = root / "plan.md"
            config_dir = (
                noisygl
                / "config"
                / "frozengrace_gb_mlp_softdist_beta05"
            )
            runner.parent.mkdir(parents=True)
            target_builder.parent.mkdir(parents=True)
            predictor.parent.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            embedding.mkdir(parents=True)
            python.write_bytes(b"")
            runner.write_text("# placeholder\n", encoding="utf-8")
            target_builder.write_text("# placeholder\n", encoding="utf-8")
            predictor.write_text("# placeholder\n", encoding="utf-8")
            plan.write_text("# frozen\n", encoding="utf-8")
            for dataset in ("cora", "corafull"):
                (config_dir / f"frozengrace_gb_mlp_softdist_beta05_{dataset}.yaml").write_text(
                    "gb:\n  tie_mode: min_index\n",
                    encoding="utf-8",
                )

            with mock.patch.object(module.subprocess, "run") as run_mock:
                rc = module.orchestrate(
                    python=python,
                    noisygl_root=noisygl,
                    embedding_root=embedding,
                    output_dir=output,
                    plan_path=plan,
                    dry_run=True,
                    summarize_only=False,
                )
            self.assertEqual(rc, 0)
            run_mock.assert_not_called()
            manifest = output / "command_manifest.csv"
            self.assertTrue(manifest.exists())
            with manifest.open(newline="", encoding="utf-8") as handle:
                commands = list(csv.DictReader(handle))
            self.assertEqual(len(commands), 6)
            self.assertEqual(
                {row["tie_mode"] for row in commands},
                {"min_index", "alpha_zero", "distribution_only"},
            )


if __name__ == "__main__":
    unittest.main()
