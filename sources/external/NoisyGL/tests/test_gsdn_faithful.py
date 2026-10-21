import math
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.tools import load_conf


DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]


def _eq7_v5_reference(
    target_logits,
    neighbor_target_logits,
    augmented_neighbor_logits,
    augmented_reverse_logits,
    negative_neighbor_logits,
):
    """Independent tensor oracle for arXiv:2210.02097 v5 Eq. (7)."""
    # Eq. (7) positive direction i -> j: ||y_i - z'_j||^2.
    forward_positive = (
        (target_logits - augmented_neighbor_logits).square().sum(dim=-1)
    )
    # Eq. (7) positive direction j -> i: ||y_j - z'_i||^2.
    reverse_positive = (
        (neighbor_target_logits - augmented_reverse_logits)
        .square()
        .sum(dim=-1)
    )
    # Eq. (7) hats apply only in the negative terms.
    negative_probs = torch.softmax(negative_neighbor_logits, dim=-1)
    forward_negative = (
        (
            torch.softmax(target_logits, dim=-1)
            - negative_probs
        )
        .square()
        .sum(dim=-1)
    )
    reverse_negative = (
        (
            torch.softmax(neighbor_target_logits, dim=-1)
            - negative_probs
        )
        .square()
        .sum(dim=-1)
    )
    return (
        forward_positive
        + reverse_positive
        - forward_negative
        - reverse_negative
    ).mean()


def _eq8_v5_reference(
    endpoint_nodes,
    endpoint_target_logits,
    neighbor_logits_at_edge_dst,
    neighbor_logits_at_edge_src,
    edge_src,
    edge_dst,
    noisy_labels,
    train_idx,
):
    """Independent tensor oracle for arXiv:2210.02097 v5 Eq. (8)."""
    train_nodes = set(torch.as_tensor(train_idx).tolist())

    # Eq. (8) first sum: one CE(s_i, yhat_i) per unique i in V_L ∩ V_b.
    target_terms = []
    for row, node in enumerate(endpoint_nodes.tolist()):
        if node in train_nodes:
            target_terms.append(
                F.cross_entropy(
                    endpoint_target_logits[row : row + 1],
                    noisy_labels[node : node + 1],
                    reduction="sum",
                )
            )

    # Eq. (8) nested edge sum: one CE(s_i, zhat_j) for every incident
    # relation of labeled i. Each canonical relation therefore admits both
    # i -> j and j -> i when the corresponding endpoint is labeled.
    neighbor_terms = []
    for edge_id, (source, destination) in enumerate(
        zip(edge_src.tolist(), edge_dst.tolist())
    ):
        if source in train_nodes:
            neighbor_terms.append(
                F.cross_entropy(
                    neighbor_logits_at_edge_dst[
                        edge_id : edge_id + 1
                    ],
                    noisy_labels[source : source + 1],
                    reduction="sum",
                )
            )
        if destination in train_nodes:
            neighbor_terms.append(
                F.cross_entropy(
                    neighbor_logits_at_edge_src[
                        edge_id : edge_id + 1
                    ],
                    noisy_labels[
                        destination : destination + 1
                    ],
                    reduction="sum",
                )
            )

    zero = endpoint_target_logits.sum() * 0.0
    numerator = sum(target_terms, zero) + sum(neighbor_terms, zero)
    return numerator / max(int(edge_src.numel()), 1)


class TinyData:
    def __init__(
        self,
        clean_labels=None,
        edge_index=None,
        train_masks=None,
    ):
        self.n_nodes = 7
        self.n_classes = 3
        self.dim_feats = 4
        self.feats = torch.tensor(
            [
                [1.0, 0.0, 0.5, -0.5],
                [0.5, 1.0, -0.5, 0.0],
                [0.0, 0.5, 1.0, -0.5],
                [-0.5, 0.0, 0.5, 1.0],
                [1.0, -0.5, 0.0, 0.5],
                [0.5, 1.0, 0.0, -0.5],
                [-0.5, 0.5, 1.0, 0.0],
            ]
        )
        default_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        self.labels = default_labels if clean_labels is None else clean_labels
        self.noisy_label = torch.tensor([0, 2, 1, 2, 0, 1, 2])
        self.train_masks = (
            torch.tensor([0, 1, 2])
            if train_masks is None
            else torch.as_tensor(train_masks).long()
        )
        self.val_masks = torch.tensor([3, 4])
        self.test_masks = torch.tensor([5, 6])
        if edge_index is None:
            edge_index = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 0],
                    [1, 2, 3, 4, 5, 6, 0, 2],
                ],
                dtype=torch.long,
            )
        values = torch.ones(edge_index.shape[1])
        self.adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            (self.n_nodes, self.n_nodes),
            check_invariants=False,
        ).coalesce()


def tiny_conf():
    return Namespace(
        model={
            "method": "gsdn_faithful_core",
            "n_feat": 4,
            "n_classes": 3,
            "n_hidden": 8,
            "n_layer": 2,
            "dropout": 0.0,
            "graph_batch_size": 3,
            "lambda": 0.8,
        },
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


class GSDNEquationTests(unittest.TestCase):
    def test_eq3_matches_manual_edge_dependent_mixup(self):
        from predictor.GSDNFaithful_Predictor import gsdn_eq3

        x_i = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        x_j = torch.tensor([[3.0, 1.0], [2.0, -1.0]])
        h_i = torch.tensor([[0.2, 0.8], [0.7, -0.1]])
        h_j = torch.tensor([[0.6, -0.2], [0.1, 0.9]])
        mix_weight = torch.tensor([[0.5, -0.25], [1.0, 0.75]])
        attention = torch.tensor([0.4, -0.2, 0.1, 0.3])
        head_weight = torch.tensor([[0.5, 1.0], [-0.75, 0.25]])
        head_bias = torch.tensor([0.1, -0.2])

        logits, beta = gsdn_eq3(
            x_i,
            x_j,
            h_i,
            h_j,
            mix_weight,
            attention,
            head_weight,
            head_bias,
        )

        manual_beta = torch.sigmoid(
            torch.cat([x_i @ mix_weight, x_j @ mix_weight], dim=1) @ attention
        )
        manual_hidden = (
            manual_beta[:, None] * h_j
            + (1.0 - manual_beta[:, None]) * h_i
        )
        manual_logits = F.linear(manual_hidden, head_weight, head_bias)
        self.assertTrue(torch.allclose(beta, manual_beta, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(logits, manual_logits, atol=1e-6, rtol=0.0))
        self.assertNotAlmostEqual(float(beta[0]), float(beta[1]), places=6)

    def test_eq7_matches_manual_feature_distillation(self):
        from predictor.GSDNFaithful_Predictor import gsdn_eq7

        y_i = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
        y_j = torch.tensor([[0.0, 1.0], [-0.25, 0.75]])
        z_ij = torch.tensor([[0.8, 0.2], [0.3, -0.3]])
        z_ji = torch.tensor([[0.1, 0.9], [-0.1, 0.6]])
        z_k = torch.tensor([[-0.5, 0.5], [0.75, -0.25]])

        actual = gsdn_eq7(y_i, y_j, z_ij, z_ji, z_k)
        expected = _eq7_v5_reference(
            y_i,
            y_j,
            z_ij,
            z_ji,
            z_k,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_eq8_matches_unique_endpoint_and_incident_relation_oracle(self):
        from predictor.GSDNFaithful_Predictor import gsdn_eq8

        endpoint_nodes = torch.tensor([0, 1, 2])
        target_logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
        )
        neighbor_logits_dst = torch.tensor(
            [[0.5, 1.5], [1.25, 0.25], [0.1, 1.9]]
        )
        neighbor_logits_src = torch.tensor(
            [[1.1, 0.3], [0.4, 1.6], [1.7, 0.2]]
        )
        edge_src = torch.tensor([0, 0, 1])
        edge_dst = torch.tensor([1, 2, 2])
        noisy_labels = torch.tensor([1, 0, 0])
        train_mask = torch.tensor([0, 2])

        actual = gsdn_eq8(
            endpoint_nodes,
            target_logits,
            neighbor_logits_dst,
            neighbor_logits_src,
            edge_src,
            edge_dst,
            noisy_labels,
            train_mask,
        )
        expected = _eq8_v5_reference(
            endpoint_nodes,
            target_logits,
            neighbor_logits_dst,
            neighbor_logits_src,
            edge_src,
            edge_dst,
            noisy_labels,
            train_mask,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)


class GSDNArchitectureAndProtocolTests(unittest.TestCase):
    def test_eq8_propagates_from_trained_high_endpoint_of_canonical_relation(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        data = TinyData(
            edge_index=torch.tensor([[0], [1]]),
            train_masks=torch.tensor([1]),
        )
        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), data, device="cpu"
        )
        predictor.model.eval()
        relation = predictor.training_edge_index
        predictor._training_batch_loss(
            relation, negative_nodes=torch.tensor([0])
        )

        with torch.no_grad():
            hidden = predictor.model.encode(predictor.feats[[0, 1]])
            target_logits = predictor.model.target_head(hidden)
            neighbor_logits = predictor.model.neighbor_head(hidden)
            target_once = F.cross_entropy(
                target_logits[1:2],
                predictor.noisy_label[1:2],
                reduction="sum",
            )
            reverse_neighbor = F.cross_entropy(
                neighbor_logits[0:1],
                predictor.noisy_label[1:2],
                reduction="sum",
            )
            expected = target_once + reverse_neighbor

        self.assertAlmostEqual(
            predictor.gsdn_faithful_diagnostics["last_label_loss"],
            float(expected),
            places=6,
        )

    def test_eq8_uses_both_neighbor_directions_when_both_endpoints_train(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        data = TinyData(
            edge_index=torch.tensor([[0], [1]]),
            train_masks=torch.tensor([0, 1]),
        )
        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), data, device="cpu"
        )
        predictor.model.eval()
        relation = predictor.training_edge_index
        predictor._training_batch_loss(
            relation, negative_nodes=torch.tensor([0])
        )

        with torch.no_grad():
            hidden = predictor.model.encode(predictor.feats[[0, 1]])
            target_logits = predictor.model.target_head(hidden)
            neighbor_logits = predictor.model.neighbor_head(hidden)
            target_once_each = F.cross_entropy(
                target_logits,
                predictor.noisy_label[[0, 1]],
                reduction="sum",
            )
            forward_neighbor = F.cross_entropy(
                neighbor_logits[1:2],
                predictor.noisy_label[0:1],
                reduction="sum",
            )
            reverse_neighbor = F.cross_entropy(
                neighbor_logits[0:1],
                predictor.noisy_label[1:2],
                reduction="sum",
            )
            expected = (
                target_once_each
                + forward_neighbor
                + reverse_neighbor
            )

        self.assertAlmostEqual(
            predictor.gsdn_faithful_diagnostics["last_label_loss"],
            float(expected),
            places=6,
        )

    def test_training_relations_canonicalize_reverse_arcs_and_remove_loops(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        raw_arcs = torch.tensor(
            [
                [0, 1, 1, 2, 3, 4],
                [1, 0, 2, 1, 4, 4],
            ],
            dtype=torch.long,
        )
        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(edge_index=raw_arcs), device="cpu"
        )

        expected_relations = torch.tensor(
            [[0, 1, 3], [1, 2, 4]], dtype=torch.long
        )
        self.assertTrue(
            torch.equal(predictor.training_edge_index, expected_relations)
        )

    def test_canonicalization_retains_a_single_directed_arc_as_one_relation(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(),
            TinyData(edge_index=torch.tensor([[4], [2]])),
            device="cpu",
        )

        self.assertTrue(torch.equal(
            predictor.training_edge_index,
            torch.tensor([[2], [4]], dtype=torch.long),
        ))

    def test_canonicalization_is_deterministic_across_arc_order(self):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        arcs_a = torch.tensor(
            [[3, 0, 2, 1, 4], [4, 1, 1, 0, 3]], dtype=torch.long
        )
        arcs_b = arcs_a[:, torch.tensor([4, 2, 0, 3, 1])]
        predictor_a = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(edge_index=arcs_a), device="cpu"
        )
        predictor_b = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(edge_index=arcs_b), device="cpu"
        )

        self.assertTrue(torch.equal(
            predictor_a.training_edge_index,
            predictor_b.training_edge_index,
        ))

    def test_relation_diagnostics_report_raw_unique_and_reverse_dedup_counts(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        raw_arcs = torch.tensor(
            [
                [0, 1, 1, 2, 3, 4],
                [1, 0, 2, 1, 4, 4],
            ],
            dtype=torch.long,
        )
        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(edge_index=raw_arcs), device="cpu"
        )
        diagnostics = predictor.gsdn_faithful_diagnostics

        self.assertEqual(diagnostics.get("raw_arc_count"), 6)
        self.assertEqual(
            diagnostics.get("unique_undirected_relation_count"), 3
        )
        self.assertEqual(
            diagnostics.get("deduplicated_reverse_arc_count"), 2
        )
        self.assertEqual(diagnostics["self_loops_removed"], 1)

    def test_identity_is_equation_grounded_adaptation_not_equation_faithful(
        self,
    ):
        from predictor.GSDNFaithful_Predictor import (
            gsdn_faithful_core_Predictor,
        )

        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(), device="cpu"
        )
        identity = predictor.gsdn_faithful_diagnostics[
            "baseline_identity"
        ].lower()

        self.assertIn(
            "equation-grounded common-protocol adaptation", identity
        )
        self.assertNotIn("equation-faithful", identity)
        self.assertIn("not official code", identity)

    def test_one_shared_backbone_and_two_linear_heads(self):
        from predictor.GSDNFaithful_Predictor import GSDNFaithfulModel

        model = GSDNFaithfulModel(4, 8, 3, n_layers=2, dropout=0.0)
        backbone_instances = [
            module
            for module in model.modules()
            if module.__class__.__name__ == "GSDNBackbone"
        ]
        self.assertEqual(len(backbone_instances), 1)
        self.assertIsNot(model.target_head, model.neighbor_head)
        self.assertEqual(model.target_head.in_features, 8)
        self.assertEqual(model.neighbor_head.in_features, 8)
        self.assertEqual(len(model.backbone.layers), 2)

        training_ids = {id(p) for p in model.training_parameters()}
        deployed_ids = {id(p) for p in model.deployed_parameters()}
        self.assertTrue(deployed_ids < training_ids)
        self.assertTrue(
            {id(p) for p in model.backbone.parameters()}.issubset(deployed_ids)
        )
        self.assertTrue(
            {id(p) for p in model.target_head.parameters()}.issubset(deployed_ids)
        )
        self.assertTrue(
            {id(p) for p in model.neighbor_head.parameters()}.isdisjoint(deployed_ids)
        )

    def test_edge_batches_cover_every_real_edge_once(self):
        from predictor.GSDNFaithful_Predictor import iter_edge_batches

        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]]
        )
        batches = list(
            iter_edge_batches(
                edge_index,
                batch_size=3,
                generator=torch.Generator().manual_seed(7),
            )
        )
        self.assertEqual(len(batches), math.ceil(edge_index.shape[1] / 3))
        ids = torch.cat([edge_ids for _, edge_ids in batches])
        self.assertEqual(ids.numel(), edge_index.shape[1])
        self.assertEqual(torch.unique(ids).numel(), edge_index.shape[1])
        self.assertEqual(
            sorted(ids.tolist()), list(range(edge_index.shape[1]))
        )

    def test_empty_graph_is_rejected_instead_of_fabricating_ring(self):
        from predictor.GSDNFaithful_Predictor import gsdn_faithful_core_Predictor

        empty_edges = torch.empty((2, 0), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "at least one non-self edge"):
            gsdn_faithful_core_Predictor(
                tiny_conf(), TinyData(edge_index=empty_edges), device="cpu"
            )

    def test_inference_uses_features_only(self):
        from predictor.GSDNFaithful_Predictor import gsdn_faithful_core_Predictor

        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(), device="cpu"
        )
        predictor.model.eval()
        adj_a = TinyData().adj
        adj_b = torch.eye(7).to_sparse()
        with torch.no_grad():
            out_a, _, _ = predictor.get_prediction(predictor.feats, adj_a)
            out_b, _, _ = predictor.get_prediction(predictor.feats, adj_b)
        self.assertTrue(torch.allclose(out_a, out_b, atol=0.0, rtol=0.0))

    def test_batch_objective_has_gradients_for_all_training_components(self):
        from predictor.GSDNFaithful_Predictor import gsdn_faithful_core_Predictor

        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(), device="cpu"
        )
        batch = predictor.training_edge_index[:, :3]
        negatives = torch.tensor([4, 5, 6])
        loss = predictor._training_batch_loss(batch, negatives)
        loss.backward()

        groups = {
            "backbone": list(predictor.model.backbone.parameters()),
            "target_head": list(predictor.model.target_head.parameters()),
            "neighbor_head": list(predictor.model.neighbor_head.parameters()),
            "mix_projection": list(predictor.model.mix_projection.parameters()),
            "attention": [predictor.model.attention],
        }
        for name, params in groups.items():
            with self.subTest(name=name):
                grads = [p.grad for p in params]
                self.assertTrue(all(g is not None for g in grads))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(
                    sum(float(g.abs().sum()) for g in grads), 0.0
                )

    def test_training_objective_ignores_clean_validation_and_test_labels(self):
        from predictor.GSDNFaithful_Predictor import gsdn_faithful_core_Predictor

        labels_a = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        labels_b = torch.tensor([0, 1, 2, 2, 2, 0, 1])
        torch.manual_seed(11)
        predictor_a = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(clean_labels=labels_a), device="cpu"
        )
        torch.manual_seed(11)
        predictor_b = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(clean_labels=labels_b), device="cpu"
        )
        predictor_a.model.eval()
        predictor_b.model.eval()
        batch = predictor_a.training_edge_index[:, :3]
        negatives = torch.tensor([4, 5, 6])
        loss_a = predictor_a._training_batch_loss(batch, negatives)
        loss_b = predictor_b._training_batch_loss(batch, negatives)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=0.0, rtol=0.0))

    def test_train_reports_exact_epoch_edge_coverage_and_parameter_counts(self):
        from predictor.GSDNFaithful_Predictor import gsdn_faithful_core_Predictor

        torch.manual_seed(5)
        predictor = gsdn_faithful_core_Predictor(
            tiny_conf(), TinyData(), device="cpu"
        )
        result = predictor.train()
        diagnostics = predictor.gsdn_faithful_diagnostics
        self.assertIn("test", result)
        self.assertEqual(diagnostics["edge_coverage_exact"], 1)
        self.assertEqual(
            diagnostics["edge_batches_per_epoch"],
            math.ceil(
                predictor.training_edge_index.shape[1]
                / tiny_conf().model["graph_batch_size"]
            ),
        )
        self.assertEqual(
            diagnostics["edges_seen_last_epoch"],
            predictor.training_edge_index.shape[1],
        )
        self.assertGreater(
            diagnostics["training_parameter_count"],
            diagnostics["deployed_parameter_count"],
        )
        self.assertEqual(diagnostics["graph_free_inference"], 1)

    def test_frozen_configs_match_common_resource_protocol(self):
        config_root = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "gsdn_faithful_core"
        )
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                conf = load_conf(None, "gsdn_faithful_core", dataset)
                self.assertEqual(conf.model["method"], "gsdn_faithful_core")
                self.assertEqual(conf.model["n_hidden"], 64)
                self.assertEqual(conf.model["n_layer"], 2)
                self.assertEqual(conf.model["dropout"], 0.5)
                self.assertEqual(conf.model["graph_batch_size"], 512)
                self.assertEqual(conf.model["lambda"], 0.8)
                self.assertEqual(conf.training["lr"], 1e-2)
                self.assertEqual(conf.training["weight_decay"], 5e-4)
                self.assertEqual(conf.training["n_epochs"], 200)
                self.assertTrue(
                    (config_root / f"gsdn_faithful_core_{dataset}.yaml").exists()
                )


if __name__ == "__main__":
    unittest.main()
