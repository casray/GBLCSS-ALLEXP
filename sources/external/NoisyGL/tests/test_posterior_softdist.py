import unittest

import numpy as np
import torch

from scripts.diagnose_posterior_softdist import (
    aggregate_support_rows,
    evaluate_support_gate,
)
from utils.frozen_grace_targets import support_bin_diagnostics


class PosteriorSoftDistTest(unittest.TestCase):
    def test_support_bin_diagnostics_groups_train_vote_counts(self):
        noisy = torch.tensor(
            [
                0, 0, 1,
                0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 1, 1,
                2, 2,
            ]
        )
        clean = torch.tensor(
            [
                0, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 1, 1, 1,
                2, 2,
            ]
        )
        train_mask = np.arange(16)
        assignment = np.array(
            [
                0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2,
                0, 1,
            ]
        )

        rows = support_bin_diagnostics(
            noisy_label=noisy,
            clean_label=clean,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        by_bin = {row["support_bin"]: row for row in rows}
        self.assertEqual(set(by_bin), {"2-3", "4-7", "8+"})
        self.assertEqual(by_bin["2-3"]["ball_count"], 1)
        self.assertEqual(by_bin["2-3"]["train_count"], 3)
        self.assertEqual(by_bin["4-7"]["train_count"], 5)
        self.assertEqual(by_bin["8+"]["train_count"], 8)
        self.assertAlmostEqual(by_bin["2-3"]["mean_observed_margin"], 2.0 / 3.0)
        self.assertAlmostEqual(by_bin["4-7"]["mean_observed_margin"], 3.0 / 5.0)
        self.assertAlmostEqual(by_bin["8+"]["mean_observed_margin"], 6.0 / 8.0)
        for row in rows:
            for key in [
                "ncr",
                "cdr",
                "mean_alpha",
                "clean_mass_delta_total",
                "clean_mass_delta_per_node",
                "changed_clean_mass_delta_per_node",
                "majority_clean_accuracy",
            ]:
                self.assertIn(key, row)

    def test_support_bin_diagnostics_uses_train_nodes_only(self):
        noisy_a = torch.tensor([0, 0, 1, 2, 2])
        noisy_b = torch.tensor([0, 0, 1, 1, 0])
        clean_a = torch.tensor([0, 1, 1, 2, 2])
        clean_b = torch.tensor([0, 1, 1, 0, 0])
        train_mask = np.array([0, 1, 2])
        assignment = np.array([0, 0, 0, 0, 0])

        rows_a = support_bin_diagnostics(
            noisy_label=noisy_a,
            clean_label=clean_a,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )
        rows_b = support_bin_diagnostics(
            noisy_label=noisy_b,
            clean_label=clean_b,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertEqual(rows_a, rows_b)
        self.assertEqual(rows_a[0]["support_bin"], "2-3")
        self.assertEqual(rows_a[0]["train_count"], 3)

    def test_support_gate_requires_two_datasets_and_four_paired_seeds(self):
        rows = []
        for dataset in ("cora", "citeseer", "coauthorcs"):
            for seed in range(5):
                if dataset == "cora":
                    low_cdr, high_cdr = 0.18, 0.10
                    low_mass, high_mass = -0.04, 0.00
                elif dataset == "citeseer":
                    low_cdr, high_cdr = (0.17, 0.10) if seed < 4 else (0.10, 0.10)
                    low_mass, high_mass = -0.03, 0.00
                else:
                    low_cdr, high_cdr = 0.12, 0.10
                    low_mass, high_mass = -0.005, 0.00
                rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "support_bin": "2-3",
                            "cdr": low_cdr,
                            "clean_mass_delta_per_node": low_mass,
                            "train_count": 10,
                        },
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "support_bin": "8+",
                            "cdr": high_cdr,
                            "clean_mass_delta_per_node": high_mass,
                            "train_count": 20,
                        },
                    ]
                )

        passed, decisions = evaluate_support_gate(rows)
        by_dataset = {row["dataset"]: row for row in decisions}

        self.assertTrue(passed)
        self.assertTrue(by_dataset["cora"]["passed"])
        self.assertTrue(by_dataset["citeseer"]["passed"])
        self.assertFalse(by_dataset["coauthorcs"]["passed"])
        self.assertEqual(by_dataset["citeseer"]["cdr_positive_seed_count"], 4)

    def test_support_gate_falls_back_to_mid_support_bin(self):
        rows = []
        for seed in range(5):
            rows.extend(
                [
                    {
                        "dataset": "cora",
                        "seed": seed,
                        "support_bin": "2-3",
                        "cdr": 0.20,
                        "clean_mass_delta_per_node": -0.05,
                        "train_count": 10,
                    },
                    {
                        "dataset": "cora",
                        "seed": seed,
                        "support_bin": "4-7",
                        "cdr": 0.10,
                        "clean_mass_delta_per_node": 0.00,
                        "train_count": 20,
                    },
                ]
            )

        passed, decisions = evaluate_support_gate(rows, min_passing_datasets=1)

        self.assertTrue(passed)
        self.assertEqual(decisions[0]["comparison_bin"], "4-7")

    def test_support_aggregate_reports_mean_and_seed_count(self):
        rows = [
            {
                "dataset": "cora",
                "seed": 0,
                "support_bin": "2-3",
                "cdr": 0.2,
                "ncr": 0.1,
                "clean_mass_delta_per_node": -0.04,
                "train_count": 10,
            },
            {
                "dataset": "cora",
                "seed": 1,
                "support_bin": "2-3",
                "cdr": 0.4,
                "ncr": 0.3,
                "clean_mass_delta_per_node": -0.02,
                "train_count": 12,
            },
        ]

        aggregate = aggregate_support_rows(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["n"], 2)
        self.assertAlmostEqual(aggregate[0]["cdr_mean"], 0.3)
        self.assertAlmostEqual(aggregate[0]["ncr_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
