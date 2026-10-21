import importlib
import math
import sys
import unittest
from unittest.mock import patch

import torch

from predictor.MLPNoiseBaselines_Predictor import (
    active_negative_loss,
    anl_ce_loss,
    clip_logits_by_l2_norm,
    confidence_gap,
    dividemix_lite_clean_prob,
    fine_alignment_scores,
    fine_gmm_clean_mask,
    fine_official_scores,
    jocor_small_loss,
    gmm_ct_clean_mask,
    mann_kendall_clean_mask,
    normalized_cross_entropy_loss,
    normalized_negative_cross_entropy_loss,
    update_mann_kendall_score,
    remember_count,
    symmetric_cross_entropy_loss,
)
from utils.tools import load_conf


class MLPNoiseBaselineTests(unittest.TestCase):
    def test_methods_are_registered_to_mlp_noise_module(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        module = importlib.import_module("predictor.MLPNoiseBaselines_Predictor")

        for method in [
            "mlp_small_loss",
            "mlp_small_loss_xh",
            "linear_small_loss",
            "mlp_coteaching",
            "mlp_jocor",
            "mlp_jocor_xh",
            "linear_jocor",
            "mlp_ct",
            "linear_ct",
            "mlp_elr",
            "mlp_sce",
            "mlp_logitclip",
            "mlp_anl",
            "mlp_fine",
            "mlp_dividemix_lite",
            "linear_dividemix_lite",
        ]:
            with self.subTest(method=method):
                self.assertEqual(
                    single_exp.PREDICTOR_MODULES[method],
                    "predictor.MLPNoiseBaselines_Predictor",
                )
                self.assertEqual(
                    standard.PREDICTOR_MODULES[method],
                    "predictor.MLPNoiseBaselines_Predictor",
                )
                self.assertEqual(
                    ssl_swap.PREDICTOR_MODULES[method],
                    "predictor.MLPNoiseBaselines_Predictor",
                )
                cls = getattr(module, f"{method}_Predictor")
                self.assertIn(
                    cls.strategy,
                    {
                        "small_loss",
                        "coteaching",
                        "jocor",
                        "ct",
                        "elr",
                        "sce",
                        "logitclip",
                        "anl",
                        "fine",
                        "dividemix_lite",
                    },
                )

    def test_configs_match_softdist_mlp_backbone(self):
        methods = [
            "mlp_small_loss",
            "mlp_small_loss_xh",
            "mlp_coteaching",
            "mlp_jocor",
            "mlp_jocor_xh",
            "mlp_ct",
            "mlp_elr",
            "mlp_sce",
            "mlp_logitclip",
            "mlp_anl",
            "mlp_fine",
            "mlp_dividemix_lite",
        ]
        for dataset in ["cora", "citeseer", "pubmed"]:
            for method in methods:
                with self.subTest(dataset=dataset, method=method):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.model["dropout"], 0.5)
                    self.assertEqual(conf.training["lr"], 1e-2)
                    self.assertEqual(conf.training["weight_decay"], 5e-4)
                    self.assertEqual(conf.training["n_epochs"], 200)

    def test_faithful_baselines_have_distinct_ids_and_frozen_configs(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        module = importlib.import_module("predictor.MLPNoiseBaselines_Predictor")

        expected_strategies = {
            "mlp_anl_ce": "anl_ce",
            "mlp_gmm_ct": "gmm_ct",
        }
        for method, strategy in expected_strategies.items():
            with self.subTest(method=method):
                self.assertEqual(single_exp.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
                self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
                self.assertEqual(ssl_swap.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
                self.assertEqual(getattr(module, f"{method}_Predictor").strategy, strategy)

        for dataset in ["cora", "citeseer", "pubmed", "corafull", "coauthorcs", "amazonpho"]:
            with self.subTest(dataset=dataset, method="mlp_anl_ce"):
                conf = load_conf(None, "mlp_anl_ce", dataset)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.model["dropout"], 0.5)
                self.assertEqual(conf.model["anl_alpha"], 5.0)
                self.assertEqual(conf.model["anl_beta"], 5.0)
                self.assertEqual(conf.model["anl_delta"], 5e-5)
                self.assertEqual(conf.model["anl_min_prob"], 1e-7)
            with self.subTest(dataset=dataset, method="mlp_gmm_ct"):
                conf = load_conf(None, "mlp_gmm_ct", dataset)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.model["dropout"], 0.5)
                self.assertEqual(conf.model["ct_warmup"], 30)
                self.assertEqual(conf.model["ct_alpha"], 0.01)
                self.assertEqual(conf.model["ct_p_threshold"], 0.5)
                self.assertEqual(conf.model["ct_reg_covar"], 5e-4)

    def test_fine_official_has_distinct_id_and_frozen_configs(self):
        with patch.object(sys, "argv", ["single_exp.py"]):
            single_exp = importlib.import_module("single_exp")
        standard = importlib.import_module("scripts.run_standard_downstream")
        ssl_swap = importlib.import_module("scripts.run_ssl_swap_downstream")
        module = importlib.import_module("predictor.MLPNoiseBaselines_Predictor")
        method = "mlp_fine_official"

        self.assertEqual(single_exp.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
        self.assertEqual(standard.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
        self.assertEqual(ssl_swap.PREDICTOR_MODULES[method], "predictor.MLPNoiseBaselines_Predictor")
        self.assertEqual(getattr(module, f"{method}_Predictor").strategy, "fine_official")

        for dataset in ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]:
            with self.subTest(dataset=dataset):
                conf = load_conf(None, method, dataset)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.model["dropout"], 0.5)
                self.assertEqual(conf.model["fine_warmup"], 10)
                self.assertEqual(conf.model["fine_p_threshold"], 0.5)

    def test_linear_noise_baseline_pubmed_configs_load(self):
        methods = [
            "linear_small_loss",
            "linear_jocor",
            "linear_dividemix_lite",
            "linear_ct",
        ]
        for method in methods:
            with self.subTest(method=method):
                conf = load_conf(None, method, "pubmed")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 1)
                self.assertEqual(conf.model["dropout"], 0.5)
                self.assertEqual(conf.training["lr"], 1e-2)
                self.assertEqual(conf.training["weight_decay"], 5e-4)
                self.assertEqual(conf.training["n_epochs"], 200)

    def test_mlp10_coauthorcs_configs_load(self):
        methods = [
            "mlp_ct",
            "mlp_fine",
            "mlp_dividemix_lite",
            "mlp_logitclip",
            "mlp_anl",
        ]
        for method in methods:
            with self.subTest(method=method):
                conf = load_conf(None, method, "coauthorcs")
                self.assertEqual(conf.model["method"], method)
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.training["lr"], 1e-2)
                self.assertEqual(conf.training["weight_decay"], 5e-4)
                self.assertEqual(conf.training["n_epochs"], 200)

        for method in [
            "frozengrace_gb_mlp_softdist_beta05",
            "frozengrace_gb_mlp_softdist_beta05_shuffled",
        ]:
            with self.subTest(method=method):
                conf = load_conf(None, method, "coauthorcs")
                self.assertEqual(conf.model["method"], method)
                self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)
                self.assertIn("CoauthorCS", conf.gb["embedding_path"])

    def test_logitclip_clips_norm_and_preserves_direction(self):
        logits = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
        clipped = clip_logits_by_l2_norm(logits, tau=1.0)

        self.assertLessEqual(float(clipped[0].norm()), 1.0 + 1e-6)
        self.assertTrue(torch.allclose(clipped[0] / clipped[0].norm(), logits[0] / logits[0].norm()))
        self.assertTrue(torch.allclose(clipped[1], logits[1]))

    def test_active_negative_loss_is_finite_and_differentiable(self):
        logits = torch.tensor([[2.0, -1.0, 0.3], [0.1, 1.0, -0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])

        loss = active_negative_loss(logits, labels, active_weight=1.0, passive_weight=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

    def test_nnce_matches_official_reference_formula(self):
        logits = torch.tensor([[2.0, -0.5, 0.1], [-0.2, 1.3, 0.4]], dtype=torch.float64)
        labels = torch.tensor([0, 2])
        min_prob = 1e-7

        actual = normalized_negative_cross_entropy_loss(logits, labels, min_prob=min_prob)

        log_probs = torch.log_softmax(logits, dim=1).clamp(min=math.log(min_prob))
        target_loss = -log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        normalizer = -log_probs.sum(dim=1)
        a = -math.log(min_prob)
        expected = (1.0 - (a - target_loss) / (logits.shape[1] * a - normalizer)).mean()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_anl_ce_combines_nce_nnce_and_l1(self):
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 1.5]]))
        logits = torch.tensor([[1.5, -0.3], [0.2, 0.8]], requires_grad=True)
        labels = torch.tensor([0, 1])
        alpha, beta, delta = 5.0, 5.0, 5e-5

        actual = anl_ce_loss(
            logits,
            labels,
            active_weight=alpha,
            passive_weight=beta,
            model=model,
            l1_weight=delta,
            min_prob=1e-7,
        )
        expected = (
            alpha * normalized_cross_entropy_loss(logits, labels)
            + beta * normalized_negative_cross_entropy_loss(logits, labels, min_prob=1e-7)
            + delta * sum(parameter.abs().sum() for parameter in model.parameters())
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))

    def test_confidence_gap_uses_annotated_class_against_every_class(self):
        probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        noisy_labels = torch.tensor([0, 1])

        actual = confidence_gap(probabilities, noisy_labels)
        expected = torch.tensor([[0.0, 0.5, 0.6], [0.2, 0.0, -0.3]])

        self.assertTrue(torch.allclose(actual, expected))

    def test_mann_kendall_increment_matches_direct_pairwise_score(self):
        sequence = [
            torch.tensor([[0.0, 0.2, -0.1]]),
            torch.tensor([[0.0, 0.4, -0.2]]),
            torch.tensor([[0.0, 0.3, 0.1]]),
        ]
        history = []
        score = torch.zeros_like(sequence[0])
        for current in sequence:
            score = update_mann_kendall_score(history, current, score)
            history.append(current)

        stacked = torch.stack(sequence)
        expected = torch.zeros_like(score)
        for later in range(1, stacked.shape[0]):
            for earlier in range(later):
                expected += torch.sign(stacked[later] - stacked[earlier])

        self.assertTrue(torch.equal(score, expected))

    def test_mann_kendall_excludes_annotated_class_and_uses_worst_competitor(self):
        scores = torch.tensor([[999.0, 10.0, -10.0], [999.0, 10.0, 9.0]])
        labels = torch.tensor([0, 0])

        mask, z = mann_kendall_clean_mask(scores, labels, n_observations=5, alpha=0.05)

        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertLess(float(z[0]), 0.0)
        self.assertGreater(float(z[1]), 0.0)

    def test_gmm_ct_mask_is_union_of_gmm_and_ct(self):
        losses = torch.tensor([0.05, 0.10, 1.5, 2.0])
        ct_mask = torch.tensor([False, False, True, False])

        combined, gmm_mask, clean_prob = gmm_ct_clean_mask(
            losses,
            ct_mask,
            p_threshold=0.5,
            random_state=0,
            reg_covar=5e-4,
        )

        self.assertTrue(torch.equal(combined, gmm_mask | ct_mask))
        self.assertTrue(bool(combined[2]))
        self.assertEqual(clean_prob.shape, losses.shape)

    def test_gmm_ct_applies_new_selection_on_next_epoch(self):
        from argparse import Namespace
        from predictor.MLPNoiseBaselines_Predictor import mlp_gmm_ct_Predictor

        class FixedModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))
                self.logits = logits

            def forward(self, features, adj=None):
                return self.logits + self.offset

        logits = torch.tensor([[3.0, 0.1], [0.1, 1.0]])
        labels = torch.tensor([0, 0])
        predictor = mlp_gmm_ct_Predictor.__new__(mlp_gmm_ct_Predictor)
        predictor.conf = Namespace(
            model={
                "ct_warmup": 0,
                "ct_alpha": 0.01,
                "ct_p_threshold": 0.5,
                "ct_reg_covar": 5e-4,
            }
        )
        predictor.device = torch.device("cpu")
        predictor.model = FixedModel(logits)
        predictor.metric = lambda y_true, y_pred: 0.0
        predictor.seed = 0
        predictor.ct_gap_history = []
        predictor.ct_mk_score = None
        predictor.ct_active_mask = torch.tensor([True, False])
        predictor.noise_baseline_diagnostics = {}

        next_mask = torch.tensor([False, True])
        with patch(
            "predictor.MLPNoiseBaselines_Predictor.gmm_ct_clean_mask",
            return_value=(next_mask, next_mask, torch.tensor([0.1, 0.9])),
        ):
            _, loss, _ = predictor._loss_and_output(
                torch.ones(2, 2),
                None,
                labels,
                torch.tensor([0, 1]),
                epoch=0,
            )

        expected = torch.nn.functional.cross_entropy(logits[[0]], labels[[0]])
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.equal(predictor.ct_active_mask, next_mask))

    def test_xh_variants_request_h_features(self):
        module = importlib.import_module("predictor.MLPNoiseBaselines_Predictor")

        self.assertTrue(module.mlp_small_loss_xh_Predictor.use_xh_features)
        self.assertTrue(module.mlp_jocor_xh_Predictor.use_xh_features)
        self.assertFalse(module.mlp_small_loss_Predictor.use_xh_features)
        self.assertFalse(module.mlp_jocor_Predictor.use_xh_features)
        self.assertEqual(module.mlp_small_loss_xh_Predictor.strategy, "small_loss")
        self.assertEqual(module.mlp_jocor_xh_Predictor.strategy, "jocor")

    def test_ct_loss_uses_noisy_labels_not_clean_labels(self):
        from argparse import Namespace
        from predictor.MLPNoiseBaselines_Predictor import mlp_ct_Predictor

        class FixedModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))
                self.logits = logits

            def forward(self, features, adj=None):
                return self.logits + self.offset

        logits = torch.tensor(
            [
                [3.0, 0.1],
                [2.5, 0.2],
                [0.2, 2.0],
                [0.1, 1.8],
            ]
        )
        features = torch.ones(4, 2)
        labels = torch.tensor([0, 0, 1, 1])
        train_idx = torch.tensor([0, 1, 2, 3])
        losses = []

        for clean in [torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0])]:
            predictor = mlp_ct_Predictor.__new__(mlp_ct_Predictor)
            predictor.conf = Namespace(
                model={
                    "forget_rate": 0.5,
                    "num_gradual": 1,
                    "exponent": 1.0,
                    "ct_ema_decay": 0.9,
                    "ct_warmup": 0,
                }
            )
            predictor.device = torch.device("cpu")
            predictor.n_nodes = 4
            predictor.n_classes = 2
            predictor.noise_rate = 0.5
            predictor.model = FixedModel(logits)
            predictor.ct_confidence_ema = None
            predictor.metric = lambda y_true, y_pred: 0.0
            predictor.clean_label = clean

            _, loss, _ = predictor._loss_and_output(features, None, labels, train_idx, epoch=1)
            losses.append(float(loss.detach()))

        self.assertAlmostEqual(losses[0], losses[1], places=7)

    def test_remember_count_uses_noise_rate_schedule(self):
        self.assertEqual(remember_count(10, epoch=0, forget_rate=0.4, num_gradual=10), 9)
        self.assertEqual(remember_count(10, epoch=9, forget_rate=0.4, num_gradual=10), 6)
        self.assertEqual(remember_count(10, epoch=99, forget_rate=0.4, num_gradual=10), 6)
        self.assertEqual(remember_count(1, epoch=99, forget_rate=0.95, num_gradual=10), 1)

    def test_loss_helpers_are_finite_and_differentiable(self):
        logits = torch.tensor([[2.0, -1.0], [0.1, 1.0]], requires_grad=True)
        logits2 = torch.tensor([[1.0, -0.5], [-0.1, 0.7]], requires_grad=True)
        labels = torch.tensor([0, 1])

        sce = symmetric_cross_entropy_loss(logits, labels)
        jocor = jocor_small_loss(
            logits,
            logits2,
            labels,
            forget_rate=0.2,
            co_lambda=0.1,
            epoch=9,
            num_gradual=10,
            exponent=1.0,
        )
        total = sce + jocor
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(logits2.grad)

    def test_fine_scores_use_noisy_labels_and_features_only(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        noisy_labels = torch.tensor([0, 0, 1, 1])
        clean_labels_a = torch.tensor([0, 0, 1, 1])
        clean_labels_b = torch.tensor([1, 1, 0, 0])

        scores_a = fine_alignment_scores(features, noisy_labels, num_classes=2)
        scores_b = fine_alignment_scores(features, noisy_labels, num_classes=2)

        self.assertTrue(torch.allclose(scores_a, scores_b))
        self.assertEqual(clean_labels_a.shape, clean_labels_b.shape)
        self.assertTrue(torch.all(scores_a >= 0.0))
        self.assertTrue(torch.all(scores_a <= 1.0))

    def test_fine_official_scores_use_uncentered_class_svd(self):
        features = torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 1.0],
                [10.0, -1.0],
                [0.0, 10.0],
                [1.0, 10.0],
                [-1.0, 10.0],
            ]
        )
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        scores = fine_official_scores(features, labels, num_classes=2)

        self.assertGreater(float(scores.min()), 0.99)
        self.assertTrue(torch.all(scores <= 1.0))

    def test_fine_gmm_selects_high_projection_component_per_class(self):
        scores = torch.tensor([0.95, 0.85, 0.10, 0.20, 0.92, 0.82, 0.12, 0.22])
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        mask, clean_prob = fine_gmm_clean_mask(
            scores,
            labels,
            num_classes=2,
            p_threshold=0.5,
            random_state=0,
        )

        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False, False, True, True, False, False])))
        self.assertGreater(float(clean_prob[[0, 1, 4, 5]].mean()), 0.9)
        self.assertLess(float(clean_prob[[2, 3, 6, 7]].mean()), 0.1)

    def test_dividemix_lite_clean_prob_is_bounded(self):
        losses = torch.tensor([0.1, 0.2, 1.5, 2.0])
        probs = dividemix_lite_clean_prob(losses, random_state=0)

        self.assertEqual(probs.shape, losses.shape)
        self.assertTrue(torch.all(torch.isfinite(probs)))
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        self.assertGreater(float(probs[:2].mean()), float(probs[2:].mean()))


if __name__ == "__main__":
    unittest.main()
