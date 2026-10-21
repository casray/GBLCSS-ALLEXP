import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "scripts"
    / "run_tie_safe_full_grid.py"
)
FROZEN_LEDGER = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "00_evidence"
    / "protocol_ledger.csv"
)


def load_module():
    spec = importlib.util.spec_from_file_location("tie_safe_full_grid", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_grid_rows():
    module = load_module()
    rows = []
    for dataset_index, dataset in enumerate(module.DATASETS):
        for setting_index, regime in enumerate(module.REGIMES):
            for seed in module.SEEDS:
                base = 0.50 + 0.002 * dataset_index + 0.001 * setting_index + 0.0001 * seed
                for method in module.METHODS:
                    test = base
                    if (
                        dataset_index == 0
                        and setting_index == 0
                        and method == module.OURS
                    ):
                        test += 0.01
                    is_softdist = method in module.SOFTDIST_METHODS
                    rows.append(
                        {
                            "dataset": dataset,
                            "regime": regime.key,
                            "method": method,
                            "base_method": method,
                            "ssl_method": "grace",
                            "seed": str(seed),
                            "noise_type": regime.noise_type,
                            "noise_rate": str(regime.noise_rate),
                            "noise_scope": "train",
                            "split_type": "percent",
                            "train_percent": "0.2",
                            "val_percent": "0.1",
                            "test_percent": "0.7",
                            "split_hash": f"split-{dataset}-{regime.key}-{seed}",
                            "noise_hash": f"noise-{dataset}-{regime.key}-{seed}",
                            "config_id": f"config-{method}-{dataset}",
                            "modified_count": str(100 + seed),
                            "modified_train_count": str(100 + seed),
                            "modified_val_count": "0",
                            "modified_test_count": "0",
                            "feature_embedding_path": (
                                f"C:/{dataset}/grace/H.pt" if is_softdist else ""
                            ),
                            "feature_embedding_hash": (
                                f"feature-hash-{dataset}" if is_softdist else ""
                            ),
                            "ball_embedding_path": (
                                f"C:/{dataset}/grace/H.pt" if is_softdist else ""
                            ),
                            "ball_embedding_hash": (
                                f"ball-hash-{dataset}" if is_softdist else ""
                            ),
                            "ball_assignment_hash": (
                                f"assignment-{method}-{dataset}-{regime.key}-{seed}"
                                if is_softdist
                                else ""
                            ),
                            "tie_mode": "distribution_only" if is_softdist else "",
                            "tie_ball_count": "2" if is_softdist else "",
                            "tie_active_ball_count": "2" if is_softdist else "",
                            "tie_affected_node_count": "20" if is_softdist else "",
                            "tie_affected_alpha_sum": "3.5" if is_softdist else "",
                            "status": "ok",
                            "valid": str(test - 0.02),
                            "test": str(test),
                        }
                    )
    return rows


def create_fake_runtime(root):
    module = load_module()
    python = root / "python.exe"
    noisygl = root / "NoisyGL"
    embedding = root / "embeddings"
    plan = root / "plan.md"
    amendment = root / "amendment.md"
    ledger = root / "protocol_ledger.csv"
    runner = noisygl / "scripts" / "run_ssl_swap_downstream.py"
    target_builder = noisygl / "utils" / "frozen_grace_targets.py"
    predictor = noisygl / "predictor" / "FrozenGraceGB_GCN_Predictor.py"
    for path in (runner, target_builder, predictor):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    python.write_bytes(b"")
    embedding.mkdir(parents=True)
    plan.write_text("# frozen plan\n", encoding="utf-8")
    amendment.write_text("# frozen amendment\n", encoding="utf-8")
    ledger.write_bytes(FROZEN_LEDGER.read_bytes())
    for method in module.METHODS:
        config_root = noisygl / "config" / method
        config_root.mkdir(parents=True)
        for dataset in module.DATASETS:
            config = config_root / f"{method}_{dataset}.yaml"
            text = "gb:\n  tie_mode: min_index\n" if method in module.SOFTDIST_METHODS else "model:\n  name: baseline\n"
            config.write_text(text, encoding="utf-8")
    return {
        "python": python,
        "noisygl": noisygl,
        "embedding": embedding,
        "plan": plan,
        "amendment": amendment,
        "ledger": ledger,
    }


class TieSafeFullGridTest(unittest.TestCase):
    def test_command_specs_split_softdist_and_baselines_into_50_commands(self):
        module = load_module()
        specs = module.build_command_specs(
            python=Path("python.exe"),
            noisygl_root=Path("NoisyGL"),
            embedding_root=Path("embeddings"),
            output_dir=Path("out"),
        )
        self.assertEqual(len(specs), 50)
        self.assertEqual(sum(spec.group == "softdist" for spec in specs), 25)
        self.assertEqual(sum(spec.group == "baseline" for spec in specs), 25)
        for spec in specs:
            command = list(spec.command)
            methods = command[command.index("--base_methods") + 1]
            overrides = json.loads(command[command.index("--config_overrides") + 1])
            if spec.group == "softdist":
                self.assertEqual(
                    methods,
                    "frozengrace_gb_mlp_softdist_beta05,"
                    "frozengrace_gb_mlp_softdist_beta05_shuffled",
                )
                self.assertEqual(overrides, {"gb.tie_mode": "distribution_only"})
            else:
                self.assertEqual(methods, "mlp_ce,mlp_jocor")
                self.assertEqual(overrides, {})

    def test_read_csv_accepts_cp936_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            path.write_bytes(
                "dataset,feature_embedding_path,test\r\n"
                "cora,C:\\实验\\H.pt,0.5\r\n".encode("cp936")
            )
            self.assertEqual(
                module.read_csv(path)[0]["feature_embedding_path"],
                "C:\\实验\\H.pt",
            )

    def test_resume_requires_exact_ten_ok_rows_with_method_seed_coverage(self):
        module = load_module()
        regime = module.REGIMES[0]
        rows = [
            row
            for row in fake_grid_rows()
            if row["dataset"] == "cora"
            and row["regime"] == regime.key
            and row["method"] in module.SOFTDIST_METHODS
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            module.write_csv(path, rows)
            self.assertTrue(
                module.raw_file_complete(path, "cora", regime, "softdist")
            )
            rows[-1]["seed"] = "3"
            module.write_csv(path, rows)
            self.assertFalse(
                module.raw_file_complete(path, "cora", regime, "softdist")
            )

    def test_resume_rejects_nonfinite_metric_or_missing_tie_diagnostics(self):
        module = load_module()
        regime = module.REGIMES[0]
        rows = [
            row
            for row in fake_grid_rows()
            if row["dataset"] == "cora"
            and row["regime"] == regime.key
            and row["method"] in module.SOFTDIST_METHODS
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            rows[0]["test"] = "nan"
            module.write_csv(path, rows)
            self.assertFalse(
                module.raw_file_complete(path, "cora", regime, "softdist")
            )
            rows[0]["test"] = "0.5"
            rows[0]["tie_affected_node_count"] = ""
            module.write_csv(path, rows)
            self.assertFalse(
                module.raw_file_complete(path, "cora", regime, "softdist")
            )

    def test_fixed_field_injection_rejects_raw_protocol_drift(self):
        module = load_module()
        regime = module.REGIMES[0]
        rows = [
            row
            for row in fake_grid_rows()
            if row["dataset"] == "cora"
            and row["regime"] == regime.key
            and row["method"] in module.BASELINE_METHODS
        ]
        rows[0]["noise_scope"] = "all"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            module.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "protocol drift"):
                module._inject_fixed_fields(
                    path, "cora", regime, "baseline"
                )

    def test_fixed_field_injection_preserves_immutable_raw_bytes(self):
        module = load_module()
        regime = module.REGIMES[0]
        rows = [
            row
            for row in fake_grid_rows()
            if row["dataset"] == "cora"
            and row["regime"] == regime.key
            and row["method"] in module.BASELINE_METHODS
        ]
        for row in rows:
            row.pop("regime")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            immutable = Path(tmp) / "immutable" / "raw.csv"
            module.write_csv(path, rows)
            original_bytes = path.read_bytes()
            module._inject_fixed_fields(
                path,
                "cora",
                regime,
                "baseline",
                immutable_path=immutable,
            )
            self.assertEqual(immutable.read_bytes(), original_bytes)
            enriched = module.read_csv(path)
            self.assertEqual({row["regime"] for row in enriched}, {regime.key})

    def test_protocol_drift_is_rejected(self):
        module = load_module()
        rows = fake_grid_rows()
        target = next(
            row
            for row in rows
            if row["dataset"] == "cora"
            and row["regime"] == "uniform0.2"
            and row["seed"] == "2"
            and row["method"] == "mlp_ce"
        )
        target["noise_hash"] = "drifted"
        with self.assertRaisesRegex(ValueError, "protocol drift"):
            module.validate_protocol_consistency(rows)

    def test_softdist_embedding_drift_is_rejected(self):
        module = load_module()
        rows = fake_grid_rows()
        targets = [
            row
            for row in rows
            if row["dataset"] == "cora"
            and row["regime"] == "uniform0.2"
            and row["seed"] == "2"
            and row["method"] in module.SOFTDIST_METHODS
        ]
        self.assertEqual(len(targets), 2)
        for target in targets:
            target["feature_embedding_hash"] = "drifted-feature-hash"
        with self.assertRaisesRegex(ValueError, "embedding"):
            module.validate_protocol_consistency(rows)

    def test_completeness_always_contains_all_500_planned_keys(self):
        module = load_module()
        rows = fake_grid_rows()
        missing = rows.pop()
        completeness = module.build_completeness(rows)
        self.assertEqual(len(completeness), 500)
        missing_status = next(
            row
            for row in completeness
            if row["dataset"] == missing["dataset"]
            and row["regime"] == missing["regime"]
            and row["method"] == missing["method"]
            and int(row["seed"]) == int(missing["seed"])
        )
        self.assertEqual(missing_status["completion_status"], "missing")
        self.assertEqual(
            sum(row["completion_status"] == "ok" for row in completeness),
            499,
        )

    def test_statistics_and_holm_are_paired_within_each_comparator_family(self):
        module = load_module()
        rows = fake_grid_rows()
        module.validate_protocol_consistency(rows)
        aggregate, deltas, tests = module.summarize_rows(rows)
        self.assertEqual(len(aggregate), 100)
        self.assertEqual(len(deltas), 375)
        self.assertEqual(len(tests), 75)
        for comparator in module.COMPARATORS:
            family = [row for row in tests if row["comparator"] == comparator]
            self.assertEqual(len(family), 25)
            self.assertEqual(
                sum(int(row["holm_significant_0p05"]) for row in family),
                1,
            )
            first = next(
                row
                for row in family
                if row["dataset"] == "cora" and row["regime"] == "uniform0.2"
            )
            self.assertAlmostEqual(float(first["paired_delta_mean"]), 0.01)
            self.assertAlmostEqual(float(first["paired_delta_ci95_low"]), 0.01)
            self.assertAlmostEqual(float(first["paired_delta_ci95_high"]), 0.01)
            self.assertEqual(float(first["p_value_two_sided"]), 0.0)
            self.assertEqual(float(first["holm_adjusted_p"]), 0.0)

    def test_dry_run_writes_50_command_manifest_without_subprocess(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = create_fake_runtime(Path(tmp))
            output = Path(tmp) / "out"
            with mock.patch.object(module.subprocess, "run") as run_mock:
                rc = module.orchestrate(
                    python=paths["python"],
                    noisygl_root=paths["noisygl"],
                    embedding_root=paths["embedding"],
                    output_dir=output,
                    plan_path=paths["plan"],
                    amendment_path=paths["amendment"],
                    ledger_path=paths["ledger"],
                    dry_run=True,
                    summarize_only=False,
                )
            self.assertEqual(rc, 0)
            run_mock.assert_not_called()
            with (output / "command_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 50)


if __name__ == "__main__":
    unittest.main()
