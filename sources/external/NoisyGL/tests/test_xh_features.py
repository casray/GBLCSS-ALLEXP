import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory
from pathlib import Path

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

    def test_h_only_normalized_embeddings_replaces_sparse_features_with_h(self):
        from predictor.FrozenGraceGB_GCN_Predictor import l2_normalized_embeddings_like_features

        indices = torch.tensor([[0, 1], [1, 2]])
        values = torch.tensor([1.0, 2.0])
        features = torch.sparse_coo_tensor(indices, values, size=(2, 3))
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

        out = l2_normalized_embeddings_like_features(features, embeddings)

        expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(torch.allclose(out, expected))

    def test_xh_predictor_variants_use_expected_switches(self):
        from predictor.FrozenGraceGB_GCN_Predictor import (
            frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor,
            frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor,
            frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor,
            frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor,
            frozengrace_gb_gcn_softdist_xh_Predictor,
            frozengrace_gb_gcn_softdist_xh_shuffled_Predictor,
            grace_h_linear_ce_Predictor,
            grace_h_mlp_ce_Predictor,
            linear_ce_Predictor,
            gcn_xh_ce_Predictor,
            mlp_xh_ce_Predictor,
            frozengrace_gb_linear_softdist_beta05_Predictor,
            frozengrace_gb_linear_softdist_beta05_shuffled_Predictor,
        )

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_Predictor.use_soft_targets)
        self.assertFalse(frozengrace_gb_gcn_softdist_xh_Predictor.random_control)

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.use_soft_targets)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_shuffled_Predictor.random_control)

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor.use_soft_targets)
        self.assertFalse(frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor.random_control)
        self.assertEqual(frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor.softdist_weighting, "ellipsoid")

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor.use_soft_targets)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor.random_control)
        self.assertEqual(frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor.softdist_weighting, "ellipsoid")

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor.use_soft_targets)
        self.assertFalse(frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor.random_control)
        self.assertEqual(frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor.softdist_weighting, "gmm_resp")

        self.assertTrue(frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor.use_xh_features)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor.use_soft_targets)
        self.assertTrue(frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor.random_control)
        self.assertEqual(frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor.softdist_weighting, "gmm_resp")

        self.assertTrue(gcn_xh_ce_Predictor.use_xh_features)
        self.assertFalse(gcn_xh_ce_Predictor.use_soft_targets)
        self.assertTrue(gcn_xh_ce_Predictor.skip_ball_targets)

        self.assertTrue(mlp_xh_ce_Predictor.use_xh_features)
        self.assertFalse(mlp_xh_ce_Predictor.use_soft_targets)
        self.assertTrue(mlp_xh_ce_Predictor.skip_ball_targets)
        self.assertEqual(mlp_xh_ce_Predictor.backbone, "mlp")

        self.assertTrue(grace_h_linear_ce_Predictor.use_h_only_features)
        self.assertFalse(grace_h_linear_ce_Predictor.use_xh_features)
        self.assertTrue(grace_h_linear_ce_Predictor.skip_ball_targets)
        self.assertEqual(grace_h_linear_ce_Predictor.backbone, "mlp")

        self.assertTrue(grace_h_mlp_ce_Predictor.use_h_only_features)
        self.assertFalse(grace_h_mlp_ce_Predictor.use_xh_features)
        self.assertTrue(grace_h_mlp_ce_Predictor.skip_ball_targets)
        self.assertEqual(grace_h_mlp_ce_Predictor.backbone, "mlp")

        self.assertEqual(linear_ce_Predictor.backbone, "mlp")
        self.assertTrue(linear_ce_Predictor.skip_ball_targets)
        self.assertFalse(linear_ce_Predictor.use_soft_targets)

        self.assertEqual(frozengrace_gb_linear_softdist_beta05_Predictor.backbone, "mlp")
        self.assertTrue(frozengrace_gb_linear_softdist_beta05_Predictor.use_soft_targets)
        self.assertFalse(frozengrace_gb_linear_softdist_beta05_Predictor.random_control)

        self.assertEqual(frozengrace_gb_linear_softdist_beta05_shuffled_Predictor.backbone, "mlp")
        self.assertTrue(frozengrace_gb_linear_softdist_beta05_shuffled_Predictor.use_soft_targets)
        self.assertTrue(frozengrace_gb_linear_softdist_beta05_shuffled_Predictor.random_control)

    def test_standard_downstream_runner_knows_ellipsoid_and_gmmresp_methods(self):
        from scripts.run_standard_downstream import predictor_class_for

        self.assertEqual(
            predictor_class_for("frozengrace_gb_gcn_softdist_xh_ellipsoid").__name__,
            "frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor",
        )
        self.assertEqual(
            predictor_class_for("frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled").__name__,
            "frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor",
        )
        self.assertEqual(
            predictor_class_for("frozengrace_gb_gcn_softdist_xh_gmmresp").__name__,
            "frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor",
        )
        self.assertEqual(
            predictor_class_for("frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled").__name__,
            "frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor",
        )

    def test_nrgnn_xh_prepares_concatenated_feature_dimension(self):
        from predictor.NRGNN_Predictor import nrgnn_xh_Predictor

        with TemporaryDirectory() as tmp:
            embedding_path = Path(tmp) / "H.pt"
            torch.save(torch.tensor([[3.0, 4.0], [0.0, 2.0]]), embedding_path)
            conf = Namespace(
                gb={"embedding_path": str(embedding_path)},
                model={"n_feat": 3},
            )
            predictor = nrgnn_xh_Predictor.__new__(nrgnn_xh_Predictor)
            predictor.conf = conf
            predictor.n_nodes = 2
            predictor.device = torch.device("cpu")
            predictor.feats = torch.ones(2, 3)

            in_channels = predictor._prepare_xh_features(conf)

        self.assertEqual(in_channels, 5)
        self.assertEqual(conf.model["n_feat"], 5)
        self.assertEqual(tuple(predictor.feats.shape), (2, 5))
        self.assertTrue(torch.allclose(predictor.feats[0, 3:], torch.tensor([0.6, 0.8])))

    def test_softdist_xh_uses_separate_feature_and_ball_embedding_paths(self):
        from predictor.FrozenGraceGB_GCN_Predictor import frozengrace_gb_gcn_softdist_xh_Predictor

        predictor = frozengrace_gb_gcn_softdist_xh_Predictor.__new__(
            frozengrace_gb_gcn_softdist_xh_Predictor
        )
        predictor.conf = Namespace(
            gb={
                "embedding_path": "feature-H.pt",
                "ball_embedding_path": "ball-H-prime.pt",
            }
        )

        self.assertEqual(predictor._feature_embedding_path(), "feature-H.pt")
        self.assertEqual(predictor._ball_embedding_path(), "ball-H-prime.pt")

    def test_softdist_xh_ball_embedding_path_falls_back_to_feature_embedding_path(self):
        from predictor.FrozenGraceGB_GCN_Predictor import frozengrace_gb_gcn_softdist_xh_Predictor

        predictor = frozengrace_gb_gcn_softdist_xh_Predictor.__new__(
            frozengrace_gb_gcn_softdist_xh_Predictor
        )
        predictor.conf = Namespace(gb={"embedding_path": "feature-H.pt"})

        self.assertEqual(predictor._feature_embedding_path(), "feature-H.pt")
        self.assertEqual(predictor._ball_embedding_path(), "feature-H.pt")

    def test_h_only_and_xh_features_are_mutually_exclusive(self):
        from predictor.FrozenGraceGB_GCN_Predictor import _FrozenGraceGBGCNBase

        class BadPredictor(_FrozenGraceGBGCNBase):
            use_h_only_features = True
            use_xh_features = True

        predictor = BadPredictor.__new__(BadPredictor)
        predictor.conf = Namespace(gb={"embedding_path": "H.pt"})

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            predictor._validate_feature_mode()


if __name__ == "__main__":
    unittest.main()
