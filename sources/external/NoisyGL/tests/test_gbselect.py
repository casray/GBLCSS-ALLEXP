import unittest

import torch


class GBSelectTest(unittest.TestCase):
    def test_ball_disagreement_scores_use_train_noisy_distribution(self):
        from utils.gbselect import ball_disagreement_scores

        noisy = torch.tensor([0, 0, 1, 1, 2, 2])
        assignment = [0, 0, 0, 1, 1, 1]
        train_mask = torch.tensor([0, 1, 2, 3])

        scores, eligible = ball_disagreement_scores(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertTrue(torch.allclose(scores[:3], torch.tensor([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0])))
        self.assertEqual(float(scores[3]), 0.0)
        self.assertFalse(bool(eligible[3]))
        self.assertFalse(bool(eligible[4]))

    def test_selection_is_train_only_and_supports_gcn_only_control(self):
        from utils.gbselect import select_clean_train_mask

        train_mask = torch.tensor([0, 1, 2, 3])
        s_ball = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.0, 0.0])
        s_gcn = torch.tensor([0.1, 0.7, 0.2, 0.8, 0.0, 0.0])

        selected = select_clean_train_mask(s_ball, s_gcn, train_mask, keep_ratio=0.5, mode="intersection")
        gcn_only = select_clean_train_mask(s_ball, s_gcn, train_mask, keep_ratio=0.5, mode="gcn_only")

        self.assertEqual(selected.nonzero(as_tuple=False).flatten().tolist(), [0])
        self.assertEqual(gcn_only.nonzero(as_tuple=False).flatten().tolist(), [0, 2])
        self.assertFalse(bool(selected[4]))
        self.assertFalse(bool(gcn_only[4]))

    def test_clean_labels_do_not_affect_selection_and_keep_one_keeps_all_train(self):
        from utils.gbselect import build_gbselect_selection

        noisy = torch.tensor([0, 0, 1, 1, 2, 2])
        clean_a = torch.tensor([0, 0, 1, 1, 2, 2])
        clean_b = torch.tensor([2, 1, 0, 2, 1, 0])
        assignment = [0, 0, 0, 1, 1, 1]
        train_mask = torch.tensor([0, 1, 2, 3])
        s_gcn = torch.tensor([0.1, 0.7, 0.2, 0.8, 0.0, 0.0])

        selected_a, _ = build_gbselect_selection(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            s_gcn=s_gcn,
            keep_ratio=1.0,
            mode="intersection",
            clean_label=clean_a,
        )
        selected_b, _ = build_gbselect_selection(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            s_gcn=s_gcn,
            keep_ratio=1.0,
            mode="intersection",
            clean_label=clean_b,
        )

        self.assertTrue(torch.equal(selected_a, selected_b))
        self.assertEqual(selected_a.nonzero(as_tuple=False).flatten().tolist(), [0, 1, 2, 3])

    def test_predictor_variants_encode_selection_controls(self):
        from predictor.GBSelect_Predictor import (
            gbselect_Predictor,
            gbselect_gcn_only_Predictor,
            gbselect_shuffled_Predictor,
        )

        self.assertEqual(gbselect_Predictor.selection_mode, "intersection")
        self.assertFalse(gbselect_Predictor.random_control)
        self.assertEqual(gbselect_gcn_only_Predictor.selection_mode, "gcn_only")
        self.assertFalse(gbselect_gcn_only_Predictor.random_control)
        self.assertEqual(gbselect_shuffled_Predictor.selection_mode, "intersection")
        self.assertTrue(gbselect_shuffled_Predictor.random_control)

    def test_ssl_swap_runner_can_load_gbselect_variants(self):
        from scripts.run_ssl_swap_downstream import predictor_class_for

        self.assertEqual(predictor_class_for("gbselect").__name__, "gbselect_Predictor")
        self.assertEqual(predictor_class_for("gbselect_gcn_only").__name__, "gbselect_gcn_only_Predictor")
        self.assertEqual(predictor_class_for("gbselect_shuffled").__name__, "gbselect_shuffled_Predictor")


if __name__ == "__main__":
    unittest.main()
