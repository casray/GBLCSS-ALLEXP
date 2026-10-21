import importlib
import unittest

import torch

from predictor.module.GNNs import MLP
from utils.tools import load_conf


class TestMLPSoftDist(unittest.TestCase):
    def test_mlp_accepts_and_ignores_adjacency(self):
        torch.manual_seed(0)
        model = MLP(3, 4, 2, 2, dropout=0.0, use_bn=False).eval()
        features = torch.randn(5, 3)
        adjacency = torch.tensor([[0, 1], [1, 0]])

        expected = model(features)
        actual = model(features, adjacency)

        self.assertTrue(torch.allclose(actual, expected))

    def test_predictor_variants_isolate_soft_targets(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        class_names = [
            "mlp_ce_Predictor",
            "frozengrace_gb_mlp_softdist_Predictor",
            "frozengrace_gb_mlp_softdist_shuffled_Predictor",
            "frozengrace_gb_mlp_softdist_beta05_Predictor",
            "frozengrace_gb_mlp_softdist_beta05_shuffled_Predictor",
            "frozengrace_gb_mlp_softdist_beta1_Predictor",
            "frozengrace_gb_mlp_softdist_beta1_shuffled_Predictor",
            "frozengrace_gb_mlp_softdist_gmmresp_Predictor",
            "frozengrace_gb_mlp_softdist_gmmresp_beta05_Predictor",
            "frozengrace_gb_mlp_softdist_gmmresp_shuffled_Predictor",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_Predictor",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled_Predictor",
            "lp_mlp_softdist_beta05_Predictor",
        ]
        for class_name in class_names:
            self.assertTrue(hasattr(module, class_name), class_name)

        ce = module.mlp_ce_Predictor
        softdist = module.frozengrace_gb_mlp_softdist_Predictor
        shuffled = module.frozengrace_gb_mlp_softdist_shuffled_Predictor
        beta05 = module.frozengrace_gb_mlp_softdist_beta05_Predictor
        beta05_shuffled = module.frozengrace_gb_mlp_softdist_beta05_shuffled_Predictor
        beta1 = module.frozengrace_gb_mlp_softdist_beta1_Predictor
        beta1_shuffled = module.frozengrace_gb_mlp_softdist_beta1_shuffled_Predictor
        gmmresp = module.frozengrace_gb_mlp_softdist_gmmresp_Predictor
        gmmresp_beta05 = module.frozengrace_gb_mlp_softdist_gmmresp_beta05_Predictor
        gmmresp_shuffled = module.frozengrace_gb_mlp_softdist_gmmresp_shuffled_Predictor
        ellipsoid_beta05 = module.frozengrace_gb_mlp_softdist_ellipsoid_beta05_Predictor
        ellipsoid_beta05_shuffled = module.frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled_Predictor
        lp_target = module.lp_mlp_softdist_beta05_Predictor

        self.assertEqual(ce.backbone, "mlp")
        self.assertFalse(ce.use_xh_features)
        self.assertFalse(ce.use_soft_targets)
        self.assertTrue(ce.skip_ball_targets)

        self.assertEqual(softdist.backbone, "mlp")
        self.assertFalse(softdist.use_xh_features)
        self.assertTrue(softdist.use_soft_targets)
        self.assertFalse(softdist.random_control)

        self.assertEqual(shuffled.backbone, "mlp")
        self.assertFalse(shuffled.use_xh_features)
        self.assertTrue(shuffled.use_soft_targets)
        self.assertTrue(shuffled.random_control)

        self.assertEqual(beta05.backbone, "mlp")
        self.assertTrue(beta05.use_soft_targets)
        self.assertFalse(beta05.random_control)

        self.assertEqual(beta05_shuffled.backbone, "mlp")
        self.assertTrue(beta05_shuffled.use_soft_targets)
        self.assertTrue(beta05_shuffled.random_control)

        self.assertEqual(beta1.backbone, "mlp")
        self.assertTrue(beta1.use_soft_targets)
        self.assertFalse(beta1.random_control)

        self.assertEqual(beta1_shuffled.backbone, "mlp")
        self.assertTrue(beta1_shuffled.use_soft_targets)
        self.assertTrue(beta1_shuffled.random_control)

        self.assertEqual(gmmresp.backbone, "mlp")
        self.assertFalse(gmmresp.use_xh_features)
        self.assertTrue(gmmresp.use_soft_targets)
        self.assertFalse(gmmresp.random_control)
        self.assertEqual(gmmresp.softdist_weighting, "gmm_resp")

        self.assertEqual(gmmresp_beta05.backbone, "mlp")
        self.assertTrue(gmmresp_beta05.use_soft_targets)
        self.assertFalse(gmmresp_beta05.random_control)
        self.assertEqual(gmmresp_beta05.softdist_weighting, "gmm_resp")

        self.assertEqual(gmmresp_shuffled.backbone, "mlp")
        self.assertFalse(gmmresp_shuffled.use_xh_features)
        self.assertTrue(gmmresp_shuffled.use_soft_targets)
        self.assertTrue(gmmresp_shuffled.random_control)
        self.assertEqual(gmmresp_shuffled.softdist_weighting, "gmm_resp")

        self.assertEqual(ellipsoid_beta05.backbone, "mlp")
        self.assertTrue(ellipsoid_beta05.use_soft_targets)
        self.assertFalse(ellipsoid_beta05.random_control)
        self.assertEqual(ellipsoid_beta05.softdist_weighting, "ellipsoid")

        self.assertEqual(ellipsoid_beta05_shuffled.backbone, "mlp")
        self.assertTrue(ellipsoid_beta05_shuffled.use_soft_targets)
        self.assertTrue(ellipsoid_beta05_shuffled.random_control)
        self.assertEqual(ellipsoid_beta05_shuffled.softdist_weighting, "ellipsoid")

        self.assertEqual(lp_target.backbone, "mlp")
        self.assertTrue(lp_target.use_soft_targets)
        self.assertEqual(lp_target.soft_target_builder, "lp")

    def test_all_smoke_configs_load_with_matched_mlp_hyperparameters(self):
        methods = [
            "mlp_ce",
            "frozengrace_gb_mlp_softdist",
            "frozengrace_gb_mlp_softdist_shuffled",
            "frozengrace_gb_mlp_softdist_beta05",
            "frozengrace_gb_mlp_softdist_beta05_shuffled",
            "frozengrace_gb_mlp_softdist_beta1",
            "frozengrace_gb_mlp_softdist_beta1_shuffled",
            "frozengrace_gb_mlp_softdist_gmmresp",
            "frozengrace_gb_mlp_softdist_gmmresp_beta05",
            "frozengrace_gb_mlp_softdist_gmmresp_shuffled",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled",
            "lp_mlp_softdist_beta05",
        ]
        expected_beta = {
            "frozengrace_gb_mlp_softdist": 0.0,
            "frozengrace_gb_mlp_softdist_shuffled": 0.0,
            "frozengrace_gb_mlp_softdist_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_beta05_shuffled": 0.5,
            "frozengrace_gb_mlp_softdist_beta1": 1.0,
            "frozengrace_gb_mlp_softdist_beta1_shuffled": 1.0,
            "frozengrace_gb_mlp_softdist_gmmresp_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled": 0.5,
            "lp_mlp_softdist_beta05": 0.5,
        }
        for dataset in ["cora", "citeseer"]:
            for method in methods:
                with self.subTest(dataset=dataset, method=method):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.model["dropout"], 0.5)
                    self.assertEqual(conf.training["n_epochs"], 200)
                    if method != "mlp_ce":
                        self.assertEqual(conf.gb["target_mode"], "distribution")
                    if method == "lp_mlp_softdist_beta05":
                        self.assertEqual(conf.gb["soft_target_builder"], "lp")
                        self.assertEqual(conf.gb["lp_steps"], 2)
                    elif method != "mlp_ce":
                        self.assertEqual(conf.gb["max_balls"], 64)
                        self.assertEqual(conf.gb["min_ball_size"], 32)
                    if method in expected_beta:
                        self.assertAlmostEqual(float(conf.gb.get("target_beta", 0.0)), expected_beta[method])
                    if method.endswith("gmmresp") or method.endswith("gmmresp_shuffled"):
                        self.assertEqual(conf.gb["softdist_weighting"], "gmm_resp")
                        self.assertEqual(conf.gb["gmm_covariance_type"], "diag")
                        self.assertEqual(conf.gb["gmm_prior"], "jeffreys")
                    if "gmmresp" in method:
                        self.assertEqual(conf.gb["softdist_weighting"], "gmm_resp")
                        self.assertEqual(conf.gb["gmm_covariance_type"], "diag")
                        self.assertEqual(conf.gb["gmm_prior"], "jeffreys")
                    if "ellipsoid" in method:
                        self.assertEqual(conf.gb["softdist_weighting"], "ellipsoid")
                        self.assertEqual(conf.gb["ellipsoid_ridge_scale"], 0.05)
                        self.assertEqual(conf.gb["ellipsoid_weight_floor"], 0.1)

    def test_beta_smoke_configs_load_for_pubmed(self):
        methods = [
            "frozengrace_gb_mlp_softdist",
            "frozengrace_gb_mlp_softdist_shuffled",
            "frozengrace_gb_mlp_softdist_beta05",
            "frozengrace_gb_mlp_softdist_beta05_shuffled",
            "frozengrace_gb_mlp_softdist_beta1",
            "frozengrace_gb_mlp_softdist_beta1_shuffled",
            "frozengrace_gb_mlp_softdist_gmmresp_beta05",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05",
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled",
            "lp_mlp_softdist_beta05",
        ]
        expected_beta = {
            "frozengrace_gb_mlp_softdist": 0.0,
            "frozengrace_gb_mlp_softdist_shuffled": 0.0,
            "frozengrace_gb_mlp_softdist_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_beta05_shuffled": 0.5,
            "frozengrace_gb_mlp_softdist_beta1": 1.0,
            "frozengrace_gb_mlp_softdist_beta1_shuffled": 1.0,
            "frozengrace_gb_mlp_softdist_gmmresp_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05": 0.5,
            "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled": 0.5,
            "lp_mlp_softdist_beta05": 0.5,
        }
        for method in methods:
            with self.subTest(method=method):
                conf = load_conf(None, method, "pubmed")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.gb["target_mode"], "distribution")
                self.assertAlmostEqual(float(conf.gb.get("target_beta", 0.0)), expected_beta[method])

    def test_labelrate_baseline_configs_load_for_pubmed(self):
        for method in ["mlp_ce", "mlp_small_loss", "mlp_jocor"]:
            with self.subTest(method=method):
                conf = load_conf(None, method, "pubmed")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.training["n_epochs"], 200)


if __name__ == "__main__":
    unittest.main()
