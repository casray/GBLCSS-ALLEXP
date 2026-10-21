import unittest

import torch


class TestXHFeatures(unittest.TestCase):
    def test_concat_l2_normalized_embeddings_appends_row_normalized_h(self):
        from predictor.FrozenGraceGB_GCN_Predictor import concat_l2_normalized_embeddings

        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

        out = concat_l2_normalized_embeddings(features, embeddings)

        expected = torch.tensor([[1.0, 2.0, 0.6, 0.8], [3.0, 4.0, 0.0, 0.0]])
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertTrue(torch.allclose(out, expected))

    def test_concat_l2_normalized_embeddings_rejects_node_count_mismatch(self):
        from predictor.FrozenGraceGB_GCN_Predictor import concat_l2_normalized_embeddings

        features = torch.ones(2, 3)
        embeddings = torch.ones(3, 4)

        with self.assertRaises(ValueError):
            concat_l2_normalized_embeddings(features, embeddings)

    def test_xh_predictor_variants_use_expected_switches(self):
        from predictor.FrozenGraceGB_GCN_Predictor import (
            frozengrace_gb_gcn_softdist_xh_Predictor,
            frozengrace_gb_gcn_softdist_xh_shuffled_Predictor,
            gcn_xh_ce_Predictor,
        )

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_Predictor.use_soft_targets)
        self.assertFalse(frozengrace_gb_gcn_softdist_xh_Predictor.random_control)

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.use_soft_targets)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.random_control)

        self.assertTrue(gcn_xh_ce_Predictor.use_xh_features)
        self.assertFalse(gcn_xh_ce_Predictor.use_soft_targets)
        self.assertTrue(gcn_xh_ce_Predictor.skip_ball_targets)


if __name__ == "__main__":
    unittest.main()
