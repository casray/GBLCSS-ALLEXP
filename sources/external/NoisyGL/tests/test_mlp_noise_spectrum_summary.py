import math
import unittest
from pathlib import Path

from scripts.summarize_softdist_mlp_noise import (
    aggregate_rows,
    build_report,
    cross_noise_robust,
    deduplicate,
    evaluate_regimes,
    expected_new_keys,
    normalize_row,
    paired_deltas,
)


class SoftDistMLPNoiseSummaryTests(unittest.TestCase):
    def test_powershell_runner_contains_only_ascii_paths(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_softdist_mlp_noise_spectrum_20260618.ps1"
        ).read_text(encoding="utf-8")

        self.assertTrue(script.isascii())
        self.assertIn("NOISYGL_ROCM_PYTHON must point", script)

    def test_normalize_accepts_all_preregistered_noise_types(self):
        for noise_type, noise_rate in (
            ("clean", 0.0),
            ("uniform", 0.2),
            ("pair", 0.4),
            ("instance", 0.4),
        ):
            with self.subTest(noise_type=noise_type):
                row = normalize_row(
                    {
                        "dataset": "Cora",
                        "base_method": "frozengrace_gb_mlp_softdist",
                        "seed": "2",
                        "noise_type": noise_type,
                        "noise_rate": str(noise_rate),
                        "actual_train_noise_rate": "0.33",
                        "test": "0.72",
                    },
                    "fixture.csv",
                )
                self.assertEqual(row["dataset"], "cora")
                self.assertEqual(row["method"], "frozengrace_gb_mlp_softdist")
                self.assertEqual(row["noise_type"], noise_type)
                self.assertEqual(row["seed"], 2)
                self.assertTrue(math.isclose(row["test_acc"], 0.72))

    def test_deduplicate_rejects_conflicting_results(self):
        base = {
            "dataset": "cora",
            "method": "mlp_ce",
            "noise_type": "uniform",
            "noise_rate": 0.2,
            "seed": 0,
            "test_acc": 0.5,
            "actual_noise_rate": 0.2,
            "source": "a.csv",
        }
        conflict = {**base, "test_acc": 0.6, "source": "b.csv"}

        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            deduplicate([base, conflict])

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = [
            {
                "dataset": "cora",
                "method": "mlp_ce",
                "noise_type": "uniform",
                "noise_rate": 0.2,
                "seed": seed,
                "test_acc": value,
            }
            for seed, value in enumerate((0.4, 0.5, 0.6))
        ]

        summary = aggregate_rows(rows)[0]

        self.assertTrue(math.isclose(summary["mean"], 0.5))
        self.assertTrue(math.isclose(summary["std"], 0.1))

    def test_paired_deltas_count_seedwise_wins(self):
        rows = []
        for seed, real, ce, shuffled in (
            (0, 0.6, 0.5, 0.55),
            (1, 0.4, 0.5, 0.45),
        ):
            for method, value in (
                ("frozengrace_gb_mlp_softdist", real),
                ("mlp_ce", ce),
                ("frozengrace_gb_mlp_softdist_shuffled", shuffled),
            ):
                rows.append(
                    {
                        "dataset": "cora",
                        "method": method,
                        "noise_type": "uniform",
                        "noise_rate": 0.2,
                        "seed": seed,
                        "test_acc": value,
                    }
                )

        deltas = paired_deltas(rows)
        vs_ce = next(row for row in deltas if row["comparison"] == "softdist_vs_ce")
        vs_shuffled = next(row for row in deltas if row["comparison"] == "softdist_vs_shuffled")

        self.assertEqual(vs_ce["positive_count"], 1)
        self.assertEqual(vs_shuffled["positive_count"], 1)
        self.assertTrue(math.isclose(vs_ce["mean_delta"], 0.0, abs_tol=1e-12))

    def test_expected_new_grid_contains_180_rows(self):
        keys = expected_new_keys()

        self.assertEqual(len(keys), 180)
        self.assertIn(("cora", "clean", 0.0, "mlp_ce", 0), keys)
        self.assertIn(
            (
                "citeseer",
                "instance",
                0.4,
                "frozengrace_gb_mlp_softdist_shuffled",
                4,
            ),
            keys,
        )

    def test_regime_decision_and_clean_safety_follow_preregistered_rules(self):
        rows = []
        for seed in range(5):
            for method, value in (
                ("mlp_ce", 0.50),
                ("frozengrace_gb_mlp_softdist", 0.51 if seed < 4 else 0.49),
                ("frozengrace_gb_mlp_softdist_shuffled", 0.48),
            ):
                rows.append(
                    {
                        "dataset": "cora",
                        "method": method,
                        "noise_type": "clean",
                        "noise_rate": 0.0,
                        "seed": seed,
                        "test_acc": value,
                    }
                )

        decision = evaluate_regimes(paired_deltas(rows))[0]

        self.assertEqual(decision["stable_increment"], 1)
        self.assertEqual(decision["clean_safe"], 1)

    def test_cross_noise_claim_excludes_clean_and_requires_four_of_six(self):
        decisions = []
        for dataset in ("cora", "citeseer"):
            decisions.append(
                {
                    "dataset": dataset,
                    "noise_type": "clean",
                    "stable_increment": 0,
                }
            )
            for index in range(6):
                decisions.append(
                    {
                        "dataset": dataset,
                        "noise_type": "uniform" if index < 4 else "pair",
                        "stable_increment": int(index < 4),
                    }
                )

        robust, counts = cross_noise_robust(decisions)

        self.assertTrue(robust)
        self.assertEqual(counts, {"cora": 4, "citeseer": 4})

    def test_report_labels_clean_regime_by_safety(self):
        aggregate = []
        for method in (
            "mlp_ce",
            "frozengrace_gb_mlp_softdist",
            "frozengrace_gb_mlp_softdist_shuffled",
        ):
            aggregate.append(
                {
                    "dataset": "cora",
                    "noise_type": "clean",
                    "noise_rate": 0.0,
                    "method": method,
                    "mean": 0.5,
                    "std": 0.01,
                }
            )
        decisions = [
            {
                "dataset": "cora",
                "noise_type": "clean",
                "noise_rate": 0.0,
                "complete": 1,
                "stable_increment": 0,
                "clean_safe": 1,
                "vs_ce_mean_delta": 0.0,
                "vs_ce_positive_count": 2,
                "vs_shuffled_mean_delta": 0.0,
                "vs_shuffled_positive_count": 2,
            }
        ]

        report = build_report(aggregate, decisions, set())

        self.assertIn("| clean safe |", report)


if __name__ == "__main__":
    unittest.main()
