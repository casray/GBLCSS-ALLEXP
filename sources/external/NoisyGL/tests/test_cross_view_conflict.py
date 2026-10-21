import math
import unittest

import numpy as np
import torch

from scripts.diagnose_cross_view_conflict import (
    conditional_auc_with_clean_subset,
    cross_view_conflict_scores,
    leave_one_out_label_views,
)


class CrossViewConflictTest(unittest.TestCase):
    def test_cross_view_scores_match_manual_majority_conflict(self):
        noisy = torch.tensor([0, 0, 1, 1, 1, 0])
        train_idx = np.arange(6, dtype=np.int64)
        assignment_x = np.array([0, 0, 0, 1, 1, 1])
        assignment_h = np.array([0, 1, 1, 0, 0, 1])

        scores = cross_view_conflict_scores(
            noisy_label=noisy,
            assignment_x=assignment_x,
            assignment_h=assignment_h,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )

        expected_hard = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(scores["s_xview"], expected_hard))
        self.assertTrue(np.isfinite(scores["s_xview_soft"]).all())
        self.assertTrue(np.all(scores["eligible"]))

    def test_identical_assignments_produce_no_hard_conflict(self):
        noisy = torch.tensor([0, 0, 1, 1, 0, 1])
        train_idx = np.arange(6, dtype=np.int64)
        assignment = np.array([0, 0, 0, 1, 1, 1])

        scores = cross_view_conflict_scores(
            noisy_label=noisy,
            assignment_x=assignment,
            assignment_h=assignment,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )

        self.assertTrue(np.allclose(scores["s_xview"], np.zeros(6)))
        self.assertTrue(np.allclose(scores["s_xview_soft"], np.zeros(6)))

    def test_clean_labels_do_not_change_cross_view_scores(self):
        noisy = torch.tensor([0, 0, 1, 1, 1, 0])
        clean_a = torch.tensor([0, 0, 1, 1, 1, 0])
        clean_b = torch.tensor([1, 1, 0, 0, 0, 1])
        train_idx = np.arange(6, dtype=np.int64)
        assignment_x = np.array([0, 0, 0, 1, 1, 1])
        assignment_h = np.array([0, 1, 1, 0, 0, 1])

        del clean_a, clean_b
        a = cross_view_conflict_scores(
            noisy_label=noisy,
            assignment_x=assignment_x,
            assignment_h=assignment_h,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )
        b = cross_view_conflict_scores(
            noisy_label=noisy,
            assignment_x=assignment_x,
            assignment_h=assignment_h,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )

        self.assertTrue(np.allclose(a["s_xview"], b["s_xview"]))
        self.assertTrue(np.allclose(a["s_xview_soft"], b["s_xview_soft"]))

    def test_undefined_views_are_neutral_and_ineligible(self):
        noisy = torch.tensor([0, 1, 1, 0])
        train_idx = np.arange(4, dtype=np.int64)
        assignment_x = np.array([0, 1, 1, 1])
        assignment_h = np.array([0, 0, 0, 0])

        views = leave_one_out_label_views(
            noisy_label=noisy,
            assignment=assignment_x,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )
        scores = cross_view_conflict_scores(
            noisy_label=noisy,
            assignment_x=assignment_x,
            assignment_h=assignment_h,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
        )

        self.assertFalse(bool(views["eligible"][0]))
        self.assertAlmostEqual(float(scores["s_xview"][0]), 0.5)
        self.assertFalse(bool(scores["eligible"][0]))

    def test_conditional_auc_small_noisy_guard_returns_nan(self):
        labels = np.array([1, 0, 0, 0, 0], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.2, 0.1, 0.3], dtype=np.float64)
        subset = np.array([True, True, True, False, False])

        value = conditional_auc_with_clean_subset(labels, scores, subset, min_positive=2)

        self.assertTrue(math.isnan(value))


if __name__ == "__main__":
    unittest.main()
