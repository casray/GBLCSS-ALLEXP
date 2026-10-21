import math
import unittest
from pathlib import Path

from scripts.summarize_mlp_noiserobust import (
    aggregate_rows,
    build_report,
    expected_keys,
    evaluate_gates,
    normalize_row,
    paired_deltas,
)


class MLPNoiseRobustSummaryTests(unittest.TestCase):
    def test_runner_uses_rocm_and_train_only_noise_scope(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_mlp_noiserobust_20260701.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn(".venv-rocm", script)
        self.assertIn("--noise_scope train", script)
        self.assertIn("mlp_coteaching", script)

    def test_normalize_row_accepts_run_ssl_swap_output(self):
        row = normalize_row(
            {
                "dataset": "Cora",
                "base_method": "mlp_elr",
                "seed": "3",
                "noise_type": "uniform",
                "noise_rate": "0.6",
                "actual_train_noise_rate": "0.59",
                "test": "0.42",
                "status": "ok",
            },
            "fixture.csv",
        )

        self.assertEqual(row["dataset"], "cora")
        self.assertEqual(row["method"], "mlp_elr")
        self.assertEqual(row["seed"], 3)
        self.assertTrue(math.isclose(row["test_acc"], 0.42))

    def test_expected_grid_contains_320_rows(self):
        self.assertEqual(len(expected_keys()), 320)
        self.assertIn(("cora", "uniform", 0.4, "mlp_coteaching", 4), expected_keys())

    def test_gate_requires_seedwise_wins_against_g1_baselines(self):
        rows = []
        for seed in range(5):
            for method, value in (
                ("frozengrace_gb_mlp_softdist", 0.60 if seed < 3 else 0.40),
                ("mlp_ce", 0.50),
                ("mlp_small_loss", 0.50),
                ("mlp_coteaching", 0.50),
                ("mlp_elr", 0.50),
                ("frozengrace_gb_mlp_softdist_shuffled", 0.50),
                ("mlp_jocor", 0.70),
                ("mlp_sce", 0.50),
            ):
                rows.append(
                    {
                        "dataset": "cora",
                        "noise_type": "uniform",
                        "noise_rate": 0.4,
                        "method": method,
                        "method_label": method,
                        "seed": seed,
                        "test_acc": value,
                    }
                )

        aggregate = aggregate_rows(rows)
        deltas = paired_deltas(rows)
        gate = evaluate_gates(deltas)[0]
        report = build_report(aggregate, deltas, [gate], set())

        self.assertEqual(gate["g1_pass"], 1)
        self.assertEqual(gate["strict_pass"], 0)
        self.assertIn("G1 overall", report)


if __name__ == "__main__":
    unittest.main()
