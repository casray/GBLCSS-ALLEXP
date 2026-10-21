import unittest

from scripts.diagnose_ellipsoid_softdist import (
    aggregate_diagnostic_rows,
    evaluate_diagnostic_gate,
    paired_delta_rows,
)


class EllipsoidSoftDistDiagnosticTest(unittest.TestCase):
    def test_diagnostic_gate_requires_two_datasets_with_paired_seed_direction_and_shuffled_control(self):
        rows = []
        methods = [
            "frozengrace_gb_gcn_softdist_xh",
            "frozengrace_gb_gcn_softdist_xh_ellipsoid",
            "frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled",
        ]
        for dataset in ("cora", "citeseer", "coauthorcs"):
            for seed in range(5):
                for method in methods:
                    mass = -0.02
                    cdr = 0.20
                    ncr = 0.50
                    fallback = 0.0
                    if method.endswith("_ellipsoid"):
                        if dataset in {"cora", "citeseer"} and seed < 4:
                            mass = 0.00
                            cdr = 0.16
                        elif dataset in {"cora", "citeseer"}:
                            mass = -0.015
                            cdr = 0.19
                        else:
                            mass = -0.018
                            cdr = 0.19
                    elif method.endswith("_ellipsoid_shuffled"):
                        mass = -0.025
                        cdr = 0.22
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "soft_clean_mass_delta": mass,
                            "soft_pseudo_cdr": cdr,
                            "soft_pseudo_ncr": ncr,
                            "ellipsoid_fallback_ball_ratio": fallback,
                        }
                    )

        passed, decisions = evaluate_diagnostic_gate(rows)
        by_dataset = {row["dataset"]: row for row in decisions}

        self.assertTrue(passed)
        self.assertTrue(by_dataset["cora"]["passed"])
        self.assertTrue(by_dataset["citeseer"]["passed"])
        self.assertFalse(by_dataset["coauthorcs"]["passed"])
        self.assertEqual(by_dataset["cora"]["positive_seed_count"], 4)

    def test_diagnostic_gate_rejects_when_ellipsoid_does_not_beat_shuffled(self):
        rows = []
        for seed in range(5):
            rows.extend(
                [
                    {
                        "dataset": "cora",
                        "seed": seed,
                        "method": "frozengrace_gb_gcn_softdist_xh",
                        "soft_clean_mass_delta": -0.02,
                        "soft_pseudo_cdr": 0.20,
                        "soft_pseudo_ncr": 0.50,
                    },
                    {
                        "dataset": "cora",
                        "seed": seed,
                        "method": "frozengrace_gb_gcn_softdist_xh_ellipsoid",
                        "soft_clean_mass_delta": 0.00,
                        "soft_pseudo_cdr": 0.16,
                        "soft_pseudo_ncr": 0.50,
                        "ellipsoid_fallback_ball_ratio": 0.0,
                    },
                    {
                        "dataset": "cora",
                        "seed": seed,
                        "method": "frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled",
                        "soft_clean_mass_delta": 0.01,
                        "soft_pseudo_cdr": 0.15,
                        "soft_pseudo_ncr": 0.50,
                        "ellipsoid_fallback_ball_ratio": 0.0,
                    },
                ]
            )

        passed, decisions = evaluate_diagnostic_gate(rows, min_passing_datasets=1)

        self.assertFalse(passed)
        self.assertFalse(decisions[0]["passed"])
        self.assertFalse(decisions[0]["beats_shuffled"])

    def test_aggregate_and_paired_delta_rows_report_seed_aligned_differences(self):
        rows = [
            {
                "dataset": "cora",
                "seed": 0,
                "method": "frozengrace_gb_gcn_softdist_xh",
                "soft_clean_mass_delta": -0.02,
                "soft_pseudo_cdr": 0.20,
                "soft_pseudo_ncr": 0.50,
            },
            {
                "dataset": "cora",
                "seed": 0,
                "method": "frozengrace_gb_gcn_softdist_xh_ellipsoid",
                "soft_clean_mass_delta": 0.01,
                "soft_pseudo_cdr": 0.15,
                "soft_pseudo_ncr": 0.52,
            },
        ]

        aggregate = aggregate_diagnostic_rows(rows)
        deltas = paired_delta_rows(rows)

        self.assertEqual(len(aggregate), 2)
        self.assertEqual(len(deltas), 1)
        self.assertAlmostEqual(deltas[0]["soft_clean_mass_delta_delta"], 0.03)
        self.assertAlmostEqual(deltas[0]["soft_pseudo_cdr_delta"], -0.05)
        self.assertAlmostEqual(deltas[0]["soft_pseudo_ncr_delta"], 0.02)


if __name__ == "__main__":
    unittest.main()
