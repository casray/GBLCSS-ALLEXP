import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_nearest_work_full_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "nearest_work_full_grid_test",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def native_rows_for_spec(runner, spec):
    rows = []
    for method_index, method in enumerate(runner.METHODS):
        for seed in runner.SEEDS:
            value = 0.50 + 0.01 * method_index + 0.001 * seed
            rows.append(
                {
                    "dataset": spec.dataset,
                    "method": method,
                    "seed": str(seed),
                    "noise_type": spec.regime.noise_type,
                    "noise_rate": str(spec.regime.noise_rate),
                    "noise_scope": "train",
                    "split_hash": (
                        f"split-{spec.dataset}-{spec.regime.key}-{seed}"
                    ),
                    "noise_hash": (
                        f"noise-{spec.dataset}-{spec.regime.key}-{seed}"
                    ),
                    "config_id": f"config-{method}-{spec.dataset}",
                    "modified_count": str(10 + seed),
                    "modified_train_count": str(10 + seed),
                    "modified_val_count": "0",
                    "modified_test_count": "0",
                    "status": "ok",
                    "train": str(value + 0.03),
                    "valid": str(value + 0.01),
                    "test": str(value),
                }
            )
    return rows


def fake_grid_rows(runner):
    rows = []
    for dataset_index, dataset in enumerate(runner.DATASETS):
        for regime_index, regime in enumerate(runner.REGIMES):
            spec = runner.build_command_specs(
                python=Path("python"),
                noisygl_root=Path("NoisyGL"),
                output_dir=Path("out"),
            )[dataset_index * len(runner.REGIMES) + regime_index]
            for row in native_rows_for_spec(runner, spec):
                enriched = dict(row)
                enriched.update(
                    {
                        "regime": regime.key,
                        "split_type": "percent",
                        "train_percent": "0.2",
                        "val_percent": "0.1",
                        "test_percent": "0.7",
                    }
                )
                rows.append(enriched)
    return rows


def fake_frozen_ours_rows(runner, nearest_rows):
    rows = []
    for row in nearest_rows:
        if row["method"] != runner.METHODS[0]:
            continue
        ours = dict(row)
        ours["method"] = runner.OURS_V2_METHOD
        ours["base_method"] = runner.OURS_V2_METHOD
        ours["test"] = str(float(row["test"]) + 0.10)
        rows.append(ours)
    return rows


def fake_v2_combined_rows(runner, nearest_rows):
    v2_methods = (
        runner.OURS_V2_METHOD,
        "frozengrace_gb_mlp_softdist_beta05_shuffled",
        "mlp_ce",
        "mlp_jocor",
    )
    mapping = dict(zip(runner.METHODS, v2_methods))
    rows = []
    for row in nearest_rows:
        converted = dict(row)
        converted["method"] = mapping[row["method"]]
        converted["base_method"] = mapping[row["method"]]
        if converted["method"] == runner.OURS_V2_METHOD:
            converted["test"] = str(float(converted["test"]) + 0.10)
        rows.append(converted)
    return rows


def create_fake_runtime(runner, root):
    python = root / "python.exe"
    python.write_bytes(b"")
    noisygl = root / "NoisyGL"
    standard = noisygl / "scripts" / "run_standard_downstream.py"
    standard.parent.mkdir(parents=True)
    standard.write_text(
        "METHOD_CAPABILITIES = " + repr(runner.CAPABILITY_REGISTRY) + "\n",
        encoding="utf-8",
    )
    for relative in (
        "predictor/GSDNFaithful_Predictor.py",
        "predictor/NearestWorkSoftTargets_Predictor.py",
        "utils/dataloader.py",
        "utils/labelnoise.py",
        "utils/tools.py",
    ):
        path = noisygl / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen input\n", encoding="utf-8")
    for method in runner.METHODS:
        for dataset in runner.DATASETS:
            config = (
                noisygl
                / "config"
                / method
                / f"{method}_{dataset}.yaml"
            )
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("training:\n  n_epochs: 2\n", encoding="utf-8")
    for dataset in runner.DATASETS:
        config = noisygl / "config" / "_dataset" / f"{dataset}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("split:\n  split_type: percent\n", encoding="utf-8")
    amendment = root / "amendment.md"
    amendment.write_text("# frozen amendment\n", encoding="utf-8")
    ours_v2_dir = root / "tie_safe_full_grid_v2_immutable"
    v2_rows = fake_v2_combined_rows(
        runner,
        fake_grid_rows(runner),
    )
    runner.write_csv(ours_v2_dir / "raw_combined.csv", v2_rows)
    return {
        "python": python,
        "noisygl": noisygl,
        "amendment": amendment,
        "ours_v2_dir": ours_v2_dir,
    }


class NearestWorkFullGridTest(unittest.TestCase):
    def test_runner_exists_at_frozen_pipeline_path(self):
        self.assertTrue(
            RUNNER_PATH.is_file(),
            f"missing frozen nearest-work runner: {RUNNER_PATH}",
        )

    def test_frozen_grid_has_exactly_500_rows_and_x_only_capabilities(self):
        runner = load_runner()
        self.assertTrue(hasattr(runner, "DATASETS"), "missing frozen grid constants")

        self.assertEqual(
            runner.DATASETS,
            ("cora", "citeseer", "coauthorcs", "corafull", "amazonpho"),
        )
        self.assertEqual(
            [regime.key for regime in runner.REGIMES],
            [
                "uniform0.2",
                "uniform0.4",
                "uniform0.6",
                "pair0.4",
                "instance0.4",
            ],
        )
        self.assertEqual(runner.SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(
            runner.METHODS,
            (
                "gsdn_faithful_core",
                "postel_mlp_noipl_rm",
                "als_mlp_rm",
                "sals_mlp_adapted",
            ),
        )
        self.assertEqual(len(runner._planned_keys()), 500)
        self.assertEqual(set(runner.CAPABILITY_REGISTRY), set(runner.METHODS))
        expected_status = {
            "gsdn_faithful_core": (
                "equation-grounded common-protocol adaptation"
            ),
            "postel_mlp_noipl_rm": "official_code_faithful_target_project_mlp",
            "als_mlp_rm": "mechanism_faithful_project_mlp_adaptation",
            "sals_mlp_adapted": "mechanism_faithful_project_mlp_adaptation",
        }
        for method in runner.METHODS:
            self.assertEqual(
                runner.CAPABILITY_REGISTRY[method],
                {
                    "requires_embedding": False,
                    "training_uses_graph": True,
                    "inference_uses_graph": False,
                    "comparison_tier": "same_resource_x_only",
                    "implementation_status": expected_status[method],
                },
            )

    def test_builds_25_standard_downstream_commands_without_embeddings(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "build_command_specs"),
            "missing standard-downstream command builder",
        )
        specs = runner.build_command_specs(
            python=Path("python.exe"),
            noisygl_root=Path("NoisyGL"),
            output_dir=Path("out"),
        )

        self.assertEqual(len(specs), 25)
        self.assertEqual(sum(spec.expected_rows for spec in specs), 500)
        for spec in specs:
            command = list(spec.command)
            command_text = json.dumps(command).lower()
            self.assertTrue(
                command[1].replace("\\", "/").endswith(
                    "scripts/run_standard_downstream.py"
                )
            )
            self.assertEqual(
                command[command.index("--methods") + 1],
                ",".join(runner.METHODS),
            )
            self.assertEqual(
                command[command.index("--seeds") + 1],
                "0,1,2,3,4",
            )
            self.assertEqual(
                command[command.index("--noise_scope") + 1],
                "train",
            )
            self.assertEqual(
                command[command.index("--split_type") + 1],
                "percent",
            )
            self.assertEqual(
                command[command.index("--train_percent") + 1],
                "0.2",
            )
            self.assertEqual(
                command[command.index("--val_percent") + 1],
                "0.1",
            )
            self.assertEqual(
                command[command.index("--test_percent") + 1],
                "0.7",
            )
            self.assertNotIn("embedding", command_text)

    def test_capability_registry_is_statically_verified_against_standard_runner(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "validate_capability_registry"),
            "missing static capability-registry verifier",
        )
        with tempfile.TemporaryDirectory() as tmp:
            noisygl_root = Path(tmp) / "NoisyGL"
            standard_runner = (
                noisygl_root / "scripts" / "run_standard_downstream.py"
            )
            standard_runner.parent.mkdir(parents=True)
            standard_runner.write_text(
                "METHOD_CAPABILITIES = "
                + repr(runner.CAPABILITY_REGISTRY)
                + "\n",
                encoding="utf-8",
            )

            observed = runner.validate_capability_registry(noisygl_root)
            self.assertEqual(observed, runner.CAPABILITY_REGISTRY)

            incompatible = {
                method: dict(capability)
                for method, capability in runner.CAPABILITY_REGISTRY.items()
            }
            incompatible["gsdn_faithful_core"]["requires_embedding"] = True
            standard_runner.write_text(
                "METHOD_CAPABILITIES = " + repr(incompatible) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "capability registry mismatch",
            ):
                runner.validate_capability_registry(noisygl_root)

            legacy_identity = {
                method: dict(capability)
                for method, capability in runner.CAPABILITY_REGISTRY.items()
            }
            legacy_identity["gsdn_faithful_core"][
                "implementation_status"
            ] = "equation_faithful_reimplementation"
            standard_runner.write_text(
                "METHOD_CAPABILITIES = " + repr(legacy_identity) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "capability registry mismatch",
            ):
                runner.validate_capability_registry(noisygl_root)

    def test_input_snapshots_are_created_once_then_only_verified(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_write_or_verify_input_snapshots"),
            "missing resume-safe input snapshot function",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [root / "a.py", root / "b.yaml"]
            sources[0].write_text("alpha\n", encoding="utf-8")
            sources[1].write_text("beta\n", encoding="utf-8")
            output_dir = root / "run"

            first = runner._write_or_verify_input_snapshots(
                sources,
                output_dir,
            )
            manifest = output_dir / "input_snapshot_manifest.csv"
            snapshots = sorted((output_dir / "input_snapshots").iterdir())
            original_manifest = manifest.read_bytes()
            original_payloads = {
                path.name: path.read_bytes() for path in snapshots
            }
            original_mtimes = {
                path.name: path.stat().st_mtime_ns for path in snapshots
            }

            second = runner._write_or_verify_input_snapshots(
                sources,
                output_dir,
            )

            self.assertEqual(first, "created")
            self.assertEqual(second, "verified_existing")
            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertEqual(
                {path.name: path.read_bytes() for path in snapshots},
                original_payloads,
            )
            self.assertEqual(
                {path.name: path.stat().st_mtime_ns for path in snapshots},
                original_mtimes,
            )

    def test_changed_input_aborts_resume_without_overwriting_snapshot(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_write_or_verify_input_snapshots"),
            "missing resume-safe input snapshot function",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.py"
            source.write_text("frozen\n", encoding="utf-8")
            output_dir = root / "run"
            runner._write_or_verify_input_snapshots([source], output_dir)
            snapshot = next((output_dir / "input_snapshots").iterdir())
            frozen_bytes = snapshot.read_bytes()

            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "input snapshot mismatch",
            ):
                runner._write_or_verify_input_snapshots([source], output_dir)

            self.assertEqual(snapshot.read_bytes(), frozen_bytes)

    def test_environment_fingerprint_preserves_initial_and_appends_resume(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_write_environment_fingerprint"),
            "missing resume-safe environment fingerprint writer",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch.object(
                runner,
                "_collect_environment_fingerprint",
                side_effect=[
                    {"captured_at_utc": "initial"},
                    {"captured_at_utc": "resume"},
                ],
            ):
                first = runner._write_environment_fingerprint(
                    Path("python"),
                    Path("NoisyGL"),
                    output_dir,
                )
                frozen_bytes = first.read_bytes()
                second = runner._write_environment_fingerprint(
                    Path("python"),
                    Path("NoisyGL"),
                    output_dir,
                )

            self.assertEqual(first.name, "environment_fingerprint.json")
            self.assertEqual(first.read_bytes(), frozen_bytes)
            self.assertNotEqual(second, first)
            self.assertTrue(
                second.name.startswith("environment_fingerprint.resume_")
            )
            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8"))["lineage_role"],
                "initial",
            )
            self.assertEqual(
                json.loads(second.read_text(encoding="utf-8"))["lineage_role"],
                "resume",
            )

    def test_immutable_raw_is_never_overwritten(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_preserve_immutable_raw"),
            "missing immutable raw preservation function",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "working.csv"
            immutable = root / "raw_immutable" / "frozen.csv"
            source.write_bytes(b"dataset,test\r\ncora,0.5\r\n")

            first = runner._preserve_immutable_raw(source, immutable)
            frozen_bytes = immutable.read_bytes()
            frozen_mtime = immutable.stat().st_mtime_ns
            second = runner._preserve_immutable_raw(source, immutable)

            self.assertEqual(first, "created")
            self.assertEqual(second, "verified_existing")
            self.assertEqual(immutable.read_bytes(), frozen_bytes)
            self.assertEqual(immutable.stat().st_mtime_ns, frozen_mtime)

            source.write_bytes(b"dataset,test\r\ncora,0.6\r\n")
            with self.assertRaisesRegex(
                ValueError,
                "immutable raw mismatch",
            ):
                runner._preserve_immutable_raw(source, immutable)
            self.assertEqual(immutable.read_bytes(), frozen_bytes)

    def test_finalize_preserves_native_bytes_then_enriches_only_working_copy(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_finalize_command_output"),
            "missing immutable/enriched output finalizer",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = runner.build_command_specs(
                python=Path("python"),
                noisygl_root=Path("NoisyGL"),
                output_dir=root,
            )[0]
            runner.write_csv(spec.raw_path, native_rows_for_spec(runner, spec))
            native_bytes = spec.raw_path.read_bytes()

            status = runner._finalize_command_output(spec)

            self.assertEqual(status, "created")
            self.assertEqual(spec.immutable_path.read_bytes(), native_bytes)
            self.assertNotIn(b"regime", spec.immutable_path.read_bytes())
            enriched = runner.read_csv(spec.raw_path)
            self.assertEqual(len(enriched), 20)
            self.assertEqual({row["regime"] for row in enriched}, {spec.regime.key})
            self.assertEqual({row["split_type"] for row in enriched}, {"percent"})
            self.assertEqual({row["train_percent"] for row in enriched}, {"0.2"})
            self.assertTrue(runner.raw_file_complete(spec))
            self.assertTrue(runner.immutable_raw_file_complete(spec))

    def test_finalize_rejects_non_train_noise_or_validation_test_mutation(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "_finalize_command_output"),
            "missing immutable/enriched output finalizer",
        )
        mutations = [
            ("noise_scope", "all"),
            ("modified_val_count", "1"),
            ("modified_test_count", "1"),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    spec = runner.build_command_specs(
                        python=Path("python"),
                        noisygl_root=Path("NoisyGL"),
                        output_dir=root,
                    )[0]
                    rows = native_rows_for_spec(runner, spec)
                    rows[0][field] = value
                    runner.write_csv(spec.raw_path, rows)
                    with self.assertRaisesRegex(
                        ValueError,
                        "protocol drift",
                    ):
                        runner._finalize_command_output(spec)
                    self.assertTrue(spec.immutable_path.exists())

    def test_completeness_and_protocol_audit_require_exact_500_rows(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "protocol_audit"),
            "missing protocol audit",
        )
        rows = fake_grid_rows(runner)

        completeness = runner.build_completeness(rows)
        audit = runner.protocol_audit(rows)
        runner.validate_protocol_consistency(rows)

        self.assertEqual(len(rows), 500)
        self.assertEqual(len(completeness), 500)
        self.assertEqual(
            {row["completion_status"] for row in completeness},
            {"ok"},
        )
        self.assertEqual(len(audit), 125)
        self.assertEqual(
            {row["protocol_status"] for row in audit},
            {"PASS"},
        )

        rows[0]["noise_hash"] = "drift"
        with self.assertRaisesRegex(ValueError, "protocol drift"):
            runner.validate_protocol_consistency(rows)

    def test_aggregate_and_cross_directory_pairing_stay_separate_from_v2(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "summarize_rows"),
            "missing within-run aggregate summarizer",
        )
        self.assertTrue(
            hasattr(runner, "summarize_vs_frozen_ours"),
            "missing cross-directory LCSS pairing",
        )
        rows = fake_grid_rows(runner)
        ours_rows = fake_frozen_ours_rows(runner, rows)

        aggregate = runner.summarize_rows(rows)
        deltas, tests = runner.summarize_vs_frozen_ours(rows, ours_rows)

        self.assertEqual(len(aggregate), 100)
        self.assertEqual(len(deltas), 500)
        self.assertEqual(len(tests), 100)
        self.assertEqual(
            {row["ours_method"] for row in deltas},
            {runner.OURS_V2_METHOD},
        )
        self.assertEqual(
            {row["comparison_source"] for row in deltas},
            {"tie_safe_full_grid_v2_immutable"},
        )
        self.assertTrue(all(float(row["delta"]) > 0 for row in deltas))
        self.assertEqual(
            {int(row["holm_family_size"]) for row in tests},
            {25},
        )
        for row in tests:
            self.assertIn("paired_delta_ci95_low", row)
            self.assertIn("paired_delta_ci95_high", row)

        ours_rows[0]["noise_hash"] = "wrong-noise-lineage"
        with self.assertRaisesRegex(
            ValueError,
            "cross-directory lineage mismatch",
        ):
            runner.summarize_vs_frozen_ours(rows, ours_rows)

    def test_dry_run_freezes_hashes_without_starting_training(self):
        runner = load_runner()
        self.assertTrue(
            hasattr(runner, "orchestrate"),
            "missing dry-run orchestrator",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = create_fake_runtime(runner, root)
            output_dir = root / "nearest_grid"
            with patch.object(runner.subprocess, "run") as run_mock:
                result = runner.orchestrate(
                    python=runtime["python"],
                    noisygl_root=runtime["noisygl"],
                    output_dir=output_dir,
                    amendment_path=runtime["amendment"],
                    ours_v2_dir=runtime["ours_v2_dir"],
                    device="cuda:0",
                    dry_run=True,
                    summarize_only=False,
                )
            run_mock.assert_not_called()
            self.assertEqual(result, 0)

            manifest = runner.read_csv(output_dir / "command_manifest.csv")
            self.assertEqual(len(manifest), 25)
            self.assertEqual({row["status"] for row in manifest}, {"planned"})
            self.assertTrue(
                all(row["command_sha256"] for row in manifest)
            )
            self.assertTrue(
                all(row["deterministic_env_sha256"] for row in manifest)
            )
            self.assertTrue(
                all(
                    row["input_snapshot_manifest_sha256"]
                    for row in manifest
                )
            )
            self.assertTrue(
                all(row["ours_v2_raw_sha256"] for row in manifest)
            )
            snapshot_manifest = (
                output_dir / "input_snapshot_manifest.csv"
            )
            frozen_manifest = snapshot_manifest.read_bytes()
            frozen_snapshots = {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (output_dir / "input_snapshots").iterdir()
            }

            with patch.object(runner.subprocess, "run") as run_mock:
                second = runner.orchestrate(
                    python=runtime["python"],
                    noisygl_root=runtime["noisygl"],
                    output_dir=output_dir,
                    amendment_path=runtime["amendment"],
                    ours_v2_dir=runtime["ours_v2_dir"],
                    device="cuda:0",
                    dry_run=True,
                    summarize_only=False,
                )
            run_mock.assert_not_called()
            self.assertEqual(second, 0)
            self.assertEqual(snapshot_manifest.read_bytes(), frozen_manifest)
            self.assertEqual(
                {
                    path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in (output_dir / "input_snapshots").iterdir()
                },
                frozen_snapshots,
            )

    def test_mocked_execution_writes_complete_separate_evidence_and_resumes(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = create_fake_runtime(runner, root)
            output_dir = root / "nearest_grid"
            v2_raw = runtime["ours_v2_dir"] / "raw_combined.csv"
            frozen_v2_bytes = v2_raw.read_bytes()

            def fake_training(command, **kwargs):
                command = list(command)
                dataset = command[command.index("--datasets") + 1]
                noise_type = command[command.index("--noise_type") + 1]
                noise_rate = float(command[command.index("--noise_rate") + 1])
                output = Path(command[command.index("--output") + 1])
                regime = next(
                    candidate
                    for candidate in runner.REGIMES
                    if candidate.noise_type == noise_type
                    and candidate.noise_rate == noise_rate
                )
                spec = next(
                    item
                    for item in runner.build_command_specs(
                        python=runtime["python"],
                        noisygl_root=runtime["noisygl"],
                        output_dir=output_dir,
                    )
                    if item.dataset == dataset
                    and item.regime == regime
                )
                runner.write_csv(
                    output,
                    native_rows_for_spec(runner, spec),
                )
                return SimpleNamespace(returncode=0)

            with patch.object(
                runner,
                "_collect_environment_fingerprint",
                return_value={"captured_at_utc": "initial"},
            ), patch.object(
                runner.subprocess,
                "run",
                side_effect=fake_training,
            ) as run_mock:
                try:
                    result = runner.orchestrate(
                        python=runtime["python"],
                        noisygl_root=runtime["noisygl"],
                        output_dir=output_dir,
                        amendment_path=runtime["amendment"],
                        ours_v2_dir=runtime["ours_v2_dir"],
                        device="cuda:0",
                        dry_run=False,
                        summarize_only=False,
                    )
                except NotImplementedError as exc:
                    self.fail(str(exc))

            self.assertEqual(result, 0)
            self.assertEqual(run_mock.call_count, 25)
            self.assertEqual(v2_raw.read_bytes(), frozen_v2_bytes)
            self.assertEqual(
                len(runner.read_csv(output_dir / "raw_combined.csv")),
                500,
            )
            self.assertEqual(
                len(runner.read_csv(output_dir / "completeness.csv")),
                500,
            )
            self.assertEqual(
                len(runner.read_csv(output_dir / "protocol_audit.csv")),
                125,
            )
            self.assertEqual(
                len(runner.read_csv(output_dir / "aggregate.csv")),
                100,
            )
            self.assertEqual(
                len(
                    runner.read_csv(
                        output_dir / "five_method_raw_combined.csv"
                    )
                ),
                625,
            )
            self.assertEqual(
                len(
                    runner.read_csv(
                        output_dir / "five_method_aggregate.csv"
                    )
                ),
                125,
            )
            self.assertEqual(
                {
                    row["method"]
                    for row in runner.read_csv(
                        output_dir / "five_method_aggregate.csv"
                    )
                },
                {
                    runner.OURS_V2_METHOD,
                    "gsdn_faithful_core",
                    "postel_mlp_noipl_rm",
                    "als_mlp_rm",
                    "sals_mlp_adapted",
                },
            )
            self.assertEqual(
                len(
                    runner.read_csv(
                        output_dir / "cross_vs_lcss_paired_deltas.csv"
                    )
                ),
                500,
            )
            self.assertEqual(
                len(
                    runner.read_csv(
                        output_dir / "cross_vs_lcss_paired_tests.csv"
                    )
                ),
                100,
            )
            immutable_manifest = runner.read_csv(
                output_dir / "immutable_raw_manifest.csv"
            )
            self.assertEqual(len(immutable_manifest), 25)
            self.assertEqual(
                {row["complete"] for row in immutable_manifest},
                {"1"},
            )
            self.assertTrue((output_dir / "report.md").is_file())

            with patch.object(
                runner,
                "_collect_environment_fingerprint",
                return_value={"captured_at_utc": "resume"},
            ), patch.object(runner.subprocess, "run") as resume_run_mock:
                resumed = runner.orchestrate(
                    python=runtime["python"],
                    noisygl_root=runtime["noisygl"],
                    output_dir=output_dir,
                    amendment_path=runtime["amendment"],
                    ours_v2_dir=runtime["ours_v2_dir"],
                    device="cuda:0",
                    dry_run=False,
                    summarize_only=False,
                )
            resume_run_mock.assert_not_called()
            self.assertEqual(resumed, 0)
            self.assertEqual(
                {
                    row["status"]
                    for row in runner.read_csv(
                        output_dir / "command_manifest.csv"
                    )
                },
                {"skipped_exact_complete"},
            )
            self.assertEqual(v2_raw.read_bytes(), frozen_v2_bytes)


if __name__ == "__main__":
    unittest.main()
