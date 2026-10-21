import unittest

import numpy as np
import torch

from scripts.diagnose_multigranularity import (
    coarse_mid_agreement_summary,
    diagnose_multigranularity_rules,
    majority_votes_for_assignment,
)


class MultigranularityDiagnosticTest(unittest.TestCase):
    def test_majority_votes_use_train_labels_only_and_abstain_when_sparse(self):
        noisy_label = torch.tensor([0, 0, 1, 1, 2, 2])
        assignment = np.array([0, 0, 0, 1, 1, 2])
        train_mask = torch.tensor([True, True, False, True, False, True])

        vote, train_count, margin = majority_votes_for_assignment(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertEqual(vote.tolist(), [0, 0, 0, -1, -1, -1])
        self.assertEqual(train_count.tolist(), [2, 2, 2, 1, 1, 1])
        self.assertTrue(torch.allclose(margin, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))

    def test_hard_rules_reduce_cdr_by_protecting_clean_mid_corrections(self):
        noisy_label = torch.tensor([1, 1, 1, 1, 2])
        clean_label = torch.tensor([0, 0, 1, 1, 2])
        train_mask = torch.ones(5, dtype=torch.bool)
        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        mid_pseudo = torch.tensor([0, 0, 0, 0, 2])
        votes_by_scale = {
            "fine": torch.tensor([0, 0, 1, -1, 2]),
            "mid": torch.tensor([0, 0, 0, 0, 2]),
            "coarse": torch.tensor([0, 0, 1, 0, 2]),
        }

        rows = diagnose_multigranularity_rules(
            noisy_label=noisy_label,
            clean_label=clean_label,
            train_mask=train_mask,
            alpha=alpha,
            mid_pseudo_label=mid_pseudo,
            votes_by_scale=votes_by_scale,
        )
        all_consistent = rows["all_consistent"]
        coarse_mid = rows["coarse_mid_consistent"]

        self.assertEqual(all_consistent["baseline_changed_train_count"], 4)
        self.assertEqual(all_consistent["protected_count"], 1)
        self.assertAlmostEqual(all_consistent["baseline_ncr"], 1.0)
        self.assertAlmostEqual(all_consistent["rule_ncr"], 1.0)
        self.assertAlmostEqual(all_consistent["ncr_relative_loss"], 0.0)
        self.assertAlmostEqual(all_consistent["baseline_cdr"], 2 / 3)
        self.assertAlmostEqual(all_consistent["rule_cdr"], 1 / 3)
        self.assertAlmostEqual(all_consistent["cdr_relative_reduction"], 0.5)
        self.assertAlmostEqual(all_consistent["protection_precision"], 1.0)

        self.assertEqual(coarse_mid["protected_count"], 1)
        self.assertAlmostEqual(coarse_mid["rule_cdr"], 1 / 3)

    def test_weighted_rule_uses_agreement_count_as_continuous_alpha_scale(self):
        noisy_label = torch.tensor([1, 1, 1, 1, 2])
        clean_label = torch.tensor([0, 0, 1, 1, 2])
        train_mask = torch.ones(5, dtype=torch.bool)
        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        mid_pseudo = torch.tensor([0, 0, 0, 0, 2])
        votes_by_scale = {
            "fine": torch.tensor([0, 0, 1, -1, 2]),
            "mid": torch.tensor([0, 0, 0, 0, 2]),
            "coarse": torch.tensor([0, -1, 1, 0, 2]),
        }

        rows = diagnose_multigranularity_rules(
            noisy_label=noisy_label,
            clean_label=clean_label,
            train_mask=train_mask,
            alpha=alpha,
            mid_pseudo_label=mid_pseudo,
            votes_by_scale=votes_by_scale,
        )
        weighted = rows["weighted_agreement"]

        self.assertAlmostEqual(weighted["execute_mass"], 1.0 + 2 / 3 + 1 / 3 + 2 / 3)
        self.assertAlmostEqual(weighted["protected_mass"], 1 + 1 / 3)
        self.assertAlmostEqual(weighted["rule_ncr"], (1.0 + 2 / 3) / 2)
        self.assertAlmostEqual(weighted["rule_cdr"], (1 / 3 + 2 / 3) / 3)
        self.assertAlmostEqual(weighted["cdr_relative_reduction"], 0.5)
        self.assertAlmostEqual(weighted["protection_precision"], 0.75)

    def test_coarse_mid_agreement_summary_counts_only_nodes_with_both_votes(self):
        candidate_mask = torch.tensor([True, True, True, False])
        mid_vote = torch.tensor([0, 1, -1, 1])
        coarse_vote = torch.tensor([0, 0, 1, 1])

        summary = coarse_mid_agreement_summary(candidate_mask, mid_vote, coarse_vote)

        self.assertEqual(summary["coarse_mid_both_vote_count"], 2)
        self.assertEqual(summary["coarse_mid_agree_count"], 1)
        self.assertAlmostEqual(summary["coarse_mid_agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
