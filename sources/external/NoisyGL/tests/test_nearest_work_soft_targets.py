import hashlib
import importlib
import unittest
import warnings
from argparse import Namespace

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"ast\.Str is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Sparse invariant checks are implicitly disabled.*",
)

import torch
import torch.nn.functional as F

from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
METHODS = ("postel_mlp_noipl_rm", "als_mlp_rm", "sals_mlp_adapted")


def nearest_work_module():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module(
            "predictor.NearestWorkSoftTargets_Predictor"
        )


class TinyData:
    def __init__(self, clean_labels=None, edge_index=None):
        self.n_nodes = 8
        self.n_classes = 3
        self.dim_feats = 5
        generator = torch.Generator().manual_seed(19)
        self.feats = torch.randn(self.n_nodes, self.dim_feats, generator=generator)
        self.labels = (
            torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
            if clean_labels is None
            else clean_labels.clone()
        )
        self.noisy_label = torch.tensor([0, 1, 0, 1, 1, 2, 2, 0])
        self.train_masks = torch.tensor([0, 1, 2, 3])
        self.val_masks = torch.tensor([4, 5])
        self.test_masks = torch.tensor([6, 7])
        if edge_index is None:
            edge_index = torch.tensor(
                [
                    [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 4, 0, 3, 5, 0, 1, 0, 1, 2, 3],
                ],
                dtype=torch.long,
            )
        values = torch.ones(edge_index.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.adj = torch.sparse_coo_tensor(
                edge_index,
                values,
                (self.n_nodes, self.n_nodes),
                check_invariants=False,
            ).coalesce()


def tiny_conf(method):
    model = {
        "method": method,
        "n_feat": 5,
        "n_classes": 3,
        "n_hidden": 8,
        "n_layer": 2,
        "dropout": 0.0,
    }
    if method == "postel_mlp_noipl_rm":
        model.update(
            {
                "soft_label_ratio": 0.8,
                "smoothing_ratio": 0.4,
                "degree_cutoff": 1,
                "temperature": 1.0,
            }
        )
    elif method == "als_mlp_rm":
        model.update(
            {
                "lp_steps": 2,
                "lp_residual": 0.5,
                "kl_weight": 1e-3,
                "pacing": "exponential",
                "pacing_initial": 0.1,
                "pacing_rate": -0.001,
                "alpha_max": 0.1,
            }
        )
    elif method == "sals_mlp_adapted":
        model.update({"epsilon": 0.4, "gamma": 0.8})
    return Namespace(
        model=model,
        training={
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "n_epochs": 2,
            "patience": None,
            "criterion": "metric",
            "debug": False,
        },
        dataset={"sparse": True},
        analysis={"flag": False},
    )


def shuffled_edges():
    return torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7],
            [3, 1, 4, 2, 0, 5, 1, 0, 0, 1, 2, 3],
        ],
        dtype=torch.long,
    )


class NearestWorkSoftTargetTests(unittest.TestCase):
    def test_postel_probability_step_matches_official_log_sequence(self):
        from predictor.NearestWorkSoftTargets_Predictor import (
            postel_prob_label_from_likelihood,
        )

        likelihood = torch.tensor(
            [[1e-200, 1e-210], [1e-220, 1e-230]], dtype=torch.float64
        )
        prior = torch.tensor([0.8, 0.2], dtype=torch.float64)

        actual, invalid = postel_prob_label_from_likelihood(
            likelihood, prior, temperature=2.0
        )
        log_prob = torch.log(likelihood) + torch.log(prior)
        log_prob = log_prob - log_prob.max()
        expected = torch.exp(log_prob).pow(2.0)
        expected = expected / expected.sum(dim=-1, keepdim=True)

        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(invalid.any())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_postel_matches_official_no_ipl_target_equations(self):
        from predictor.NearestWorkSoftTargets_Predictor import (
            build_postel_targets,
        )

        labels = torch.tensor([0, 1, 0, 1, 2, 2])
        train_idx = torch.tensor([0, 1, 2, 3])
        edges = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 3],
                [1, 2, 4, 0, 3, 5, 0, 1],
            ]
        )

        targets, diagnostics = build_postel_targets(
            labels,
            train_idx,
            edges,
            num_nodes=6,
            n_classes=3,
            degree_cutoff=1,
            temperature=1.0,
            smoothing_ratio=0.0,
            soft_label_ratio=1.0,
        )

        self.assertTrue(
            torch.allclose(
                targets[torch.tensor([0, 1])],
                torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                targets[2],
                torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0]),
                atol=1e-6,
            )
        )
        self.assertEqual(diagnostics["postel_degree_cutoff"], 1)
        self.assertEqual(diagnostics["postel_temperature"], 1.0)

    def test_postel_applies_smoothing_before_hard_label_mixing(self):
        from predictor.NearestWorkSoftTargets_Predictor import (
            build_postel_targets,
        )

        labels = torch.tensor([0, 1, 0, 1, 2, 2])
        train_idx = torch.tensor([0, 1, 2, 3])
        edges = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 3],
                [1, 2, 4, 0, 3, 5, 0, 1],
            ]
        )
        posterior, _ = build_postel_targets(
            labels,
            train_idx,
            edges,
            num_nodes=6,
            n_classes=3,
            smoothing_ratio=0.0,
            soft_label_ratio=1.0,
        )
        actual, _ = build_postel_targets(
            labels,
            train_idx,
            edges,
            num_nodes=6,
            n_classes=3,
            smoothing_ratio=0.4,
            soft_label_ratio=0.8,
        )
        hard = F.one_hot(labels[train_idx], 3).to(torch.float32)
        smoothed = 0.6 * posterior[train_idx] + 0.2 * (~hard.bool()).float()
        expected = 0.8 * smoothed + 0.2 * hard

        self.assertTrue(
            torch.allclose(actual[train_idx], expected, atol=1e-6, rtol=1e-6)
        )

    def test_sals_uses_only_train_labeled_neighbors_and_uniform_fallback(self):
        from predictor.NearestWorkSoftTargets_Predictor import (
            build_sals_targets,
        )

        labels = torch.tensor([0, 1, 2, 2, 0])
        train_idx = torch.tensor([0, 1, 2])
        edges = torch.tensor([[0, 0, 1, 2], [1, 3, 4, 3]])

        targets, diagnostics = build_sals_targets(
            labels,
            train_idx,
            edges,
            num_nodes=5,
            n_classes=3,
            epsilon=0.4,
            gamma=0.8,
        )

        expected_node0 = (
            0.6 * F.one_hot(torch.tensor(0), 3).float()
            + 0.4
            * (
                0.8 * F.one_hot(torch.tensor(1), 3).float()
                + 0.2 * torch.full((3,), 1.0 / 3.0)
            )
        )
        expected_fallback = (
            0.6 * F.one_hot(torch.tensor(1), 3).float()
            + 0.4 * torch.full((3,), 1.0 / 3.0)
        )
        self.assertTrue(torch.allclose(targets[0], expected_node0))
        self.assertTrue(torch.allclose(targets[1], expected_fallback))
        self.assertEqual(diagnostics["fallback_count"], 2)

    def test_als_label_propagation_preserves_train_only_initial_state(self):
        from predictor.NearestWorkSoftTargets_Predictor import (
            als_label_propagation,
        )

        labels_a = torch.tensor([0, 1, 2, 0, 1, 2])
        labels_b = torch.tensor([0, 1, 2, 0, 2, 0])
        train_idx = torch.tensor([0, 1, 2, 3])
        edges = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 0, 0, 1]], dtype=torch.long
        )

        propagated_a, _ = als_label_propagation(
            labels_a,
            train_idx,
            edges,
            num_nodes=6,
            n_classes=3,
            steps=2,
            residual=0.5,
        )
        propagated_b, _ = als_label_propagation(
            labels_b,
            train_idx,
            edges,
            num_nodes=6,
            n_classes=3,
            steps=2,
            residual=0.5,
        )

        self.assertTrue(torch.allclose(propagated_a, propagated_b))
        self.assertTrue(torch.isfinite(propagated_a).all())

    def test_all_targets_are_finite_normalized_and_graph_sensitive(self):
        module = nearest_work_module()
        data = TinyData()
        shuffled = shuffled_edges()

        for method in METHODS:
            cls = getattr(module, f"{method}_Predictor")
            torch.manual_seed(123)
            predictor = cls(tiny_conf(method), data, device="cpu")
            changed_data = TinyData(edge_index=shuffled)
            torch.manual_seed(123)
            changed = cls(tiny_conf(method), changed_data, device="cpu")
            with self.subTest(method=method):
                targets = predictor.current_soft_targets().detach()
                changed_targets = changed.current_soft_targets().detach()
                train_idx = data.train_masks
                self.assertTrue(torch.isfinite(targets[train_idx]).all())
                self.assertTrue(
                    torch.allclose(
                        targets[train_idx].sum(dim=1),
                        torch.ones(train_idx.numel()),
                        atol=1e-6,
                    )
                )
                self.assertFalse(
                    torch.allclose(targets[train_idx], changed_targets[train_idx])
                )

    def test_clean_validation_and_test_labels_do_not_change_targets(self):
        module = nearest_work_module()
        clean_a = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        clean_b = torch.tensor([0, 1, 2, 0, 2, 0, 1, 2])

        for method in METHODS:
            cls = getattr(module, f"{method}_Predictor")
            first_data = TinyData(clean_a)
            second_data = TinyData(clean_b)
            second_data.noisy_label = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2])
            torch.manual_seed(123)
            first = cls(tiny_conf(method), first_data, device="cpu")
            torch.manual_seed(123)
            second = cls(tiny_conf(method), second_data, device="cpu")
            with self.subTest(method=method):
                self.assertTrue(
                    torch.allclose(
                        first.current_soft_targets(),
                        second.current_soft_targets(),
                    )
                )
                self.assertEqual(
                    first.method_diagnostics["target_sha256"],
                    second.method_diagnostics["target_sha256"],
                )

    def test_inference_is_adjacency_invariant_and_training_has_gradients(self):
        module = nearest_work_module()
        data = TinyData()

        for method in METHODS:
            cls = getattr(module, f"{method}_Predictor")
            predictor = cls(tiny_conf(method), data, device="cpu")
            predictor.model.eval()
            with self.subTest(method=method):
                logits_a, _, _ = predictor.get_prediction(data.feats, data.adj)
                logits_b, _, _ = predictor.get_prediction(
                    data.feats,
                    torch.eye(data.n_nodes),
                )
                self.assertTrue(torch.allclose(logits_a, logits_b))

                predictor.model.train()
                predictor.optim.zero_grad()
                logits, _, _ = predictor.get_prediction(data.feats, None)
                loss = predictor.training_loss(logits, epoch=1)
                loss.backward()
                model_grad = sum(
                    float(parameter.grad.abs().sum())
                    for parameter in predictor.model.parameters()
                    if parameter.grad is not None
                )
                self.assertGreater(model_grad, 0.0)
                if method == "als_mlp_rm":
                    self.assertIsNotNone(predictor.class_relevance.grad)
                    self.assertGreater(
                        float(predictor.class_relevance.grad.abs().sum()),
                        0.0,
                    )

    def test_als_eq5_matches_published_objective_and_keeps_gradients(self):
        from predictor.NearestWorkSoftTargets_Predictor import als_eq5_loss

        logits = torch.tensor(
            [[1.2, -0.4, 0.1], [-0.2, 0.7, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        hard_labels = torch.tensor([0, 2])
        soft_targets = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64
        )
        alpha = 0.07
        gamma = 1e-3

        actual = als_eq5_loss(
            logits,
            hard_labels,
            soft_targets,
            alpha=alpha,
            kl_weight=gamma,
        )
        hard_ce = F.cross_entropy(logits, hard_labels)
        soft_ce = -(soft_targets * F.log_softmax(logits, dim=1)).sum(1).mean()
        kl_to_uniform = (
            soft_targets
            * (
                soft_targets.log()
                - torch.full_like(soft_targets, 1.0 / 3.0).log()
            )
        ).sum(1).mean()
        expected = (1.0 - alpha) * hard_ce + alpha * soft_ce + gamma * kl_to_uniform

        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        actual.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_als_mlp_uses_published_exponential_decreasing_pacing_family(self):
        from predictor.NearestWorkSoftTargets_Predictor import als_pacing_alpha

        values = [
            als_pacing_alpha(
                epoch,
                pacing="exponential",
                alpha_max=0.1,
                rate=-1e-3,
                initial=0.1,
            )
            for epoch in (0, 100, 200)
        ]

        self.assertAlmostEqual(values[0], 0.1)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_hashes_are_stable_and_sensitive(self):
        from predictor.NearestWorkSoftTargets_Predictor import tensor_sha256

        tensor = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        same = tensor.clone()
        changed = tensor.clone()
        changed[0, 0] += 1e-4

        self.assertEqual(tensor_sha256(tensor), tensor_sha256(same))
        self.assertNotEqual(tensor_sha256(tensor), tensor_sha256(changed))
        self.assertEqual(len(tensor_sha256(tensor)), hashlib.sha256().digest_size * 2)

    def test_diagnostics_publish_protocol_and_hashes(self):
        module = nearest_work_module()
        for method in METHODS:
            cls = getattr(module, f"{method}_Predictor")
            predictor = cls(tiny_conf(method), TinyData(), device="cpu")
            diagnostics = predictor.method_diagnostics
            with self.subTest(method=method):
                self.assertEqual(diagnostics["training_uses_graph"], 1)
                self.assertEqual(diagnostics["inference_uses_graph"], 0)
                self.assertEqual(diagnostics["clean_label_target_reads"], 0)
                for key in (
                    "target_sha256",
                    "edge_sha256",
                    "fallback_sha256",
                ):
                    self.assertEqual(len(diagnostics[key]), 64)

    def test_configs_match_frozen_same_resource_protocol(self):
        for dataset in DATASETS:
            for method in METHODS:
                with self.subTest(dataset=dataset, method=method):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.model["dropout"], 0.5)
                    self.assertEqual(conf.training["lr"], 1e-2)
                    self.assertEqual(conf.training["weight_decay"], 5e-4)
                    self.assertEqual(conf.training["n_epochs"], 200)
                    if method == "postel_mlp_noipl_rm":
                        self.assertEqual(conf.model["degree_cutoff"], 1)
                        self.assertEqual(conf.model["temperature"], 1.0)
                        self.assertEqual(conf.model["soft_label_ratio"], 0.8)
                        self.assertEqual(conf.model["smoothing_ratio"], 0.4)
                        self.assertFalse(conf.model["iterative_pseudo_labeling"])
                    elif method == "als_mlp_rm":
                        self.assertEqual(conf.model["lp_steps"], 2)
                        self.assertEqual(conf.model["lp_residual"], 0.5)
                        self.assertEqual(conf.model["kl_weight"], 1e-3)
                        self.assertEqual(conf.model["pacing"], "exponential")
                        self.assertEqual(conf.model["pacing_initial"], 0.1)
                        self.assertEqual(conf.model["pacing_rate"], -1e-3)
                    else:
                        self.assertEqual(conf.model["epsilon"], 0.4)
                        self.assertEqual(conf.model["gamma"], 0.8)
                        self.assertEqual(
                            conf.model["unlabeled_neighbor_fallback"],
                            "uniform",
                        )


if __name__ == "__main__":
    unittest.main()
