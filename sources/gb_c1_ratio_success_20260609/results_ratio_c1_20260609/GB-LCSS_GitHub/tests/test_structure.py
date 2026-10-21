from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import GraphData


@pytest.fixture
def structure():
    return import_module("gblcss.baselines.structure")


@pytest.fixture
def graph():
    train = torch.tensor([3, 0, 2, 1])
    validation = torch.tensor([5, 4])
    test = torch.tensor([7, 6])

    def mask(indices):
        return torch.zeros(8, dtype=torch.bool).scatter_(0, indices, True)

    return GraphData(
        name="structure-fixture",
        x=torch.randn(8, 5, generator=torch.Generator().manual_seed(19)),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 0],
            [1, 2, 4, 0, 3, 5, 0, 1, 0, 1, 2, 3, 1],
        ]),
        y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        noisy_y=torch.tensor([0, 1, 0, 1, 1, 2, 2, 0]),
        train_mask=mask(train), val_mask=mask(validation), test_mask=mask(test),
        train_index=train, val_index=validation, test_index=test,
    )


def test_postel_product_posterior_then_smoothing_and_mixing(structure, graph):
    posterior, _ = structure.build_postel_targets(
        graph.noisy_y, graph.train_index, graph.edge_index,
        num_nodes=8, n_classes=3, smoothing_ratio=0.0, soft_label_ratio=1.0,
    )
    expected = torch.tensor([[0.5, 0.5, 0], [0.5, 0.5, 0],
                             [2 / 3, 1 / 3, 0], [1 / 3, 2 / 3, 0]])
    torch.testing.assert_close(posterior[:4], expected)
    actual, _ = structure.build_postel_targets(
        graph.noisy_y, graph.train_index, graph.edge_index,
        num_nodes=8, n_classes=3,
    )
    hard = F.one_hot(graph.noisy_y[:4], 3).float()
    torch.testing.assert_close(actual[:4], 0.8 * (0.6 * expected + 0.2 * (1 - hard)) + 0.2 * hard)
    torch.testing.assert_close(actual[4:], torch.full((4, 3), 1 / 3))


def test_postel_keeps_global_log_max_and_invalid_fallback(structure):
    likelihood = torch.tensor([[1.0, 0.5], [1e-300, 2e-300]], dtype=torch.float64)
    posterior, invalid = structure.postel_prob_label_from_likelihood(
        likelihood, torch.tensor([0.5, 0.5]), temperature=2.0,
    )
    torch.testing.assert_close(posterior[0], torch.tensor([0.8, 0.2], dtype=torch.float64))
    assert invalid.tolist() == [False, True]
    targets, diagnostics = structure.build_postel_targets(
        torch.tensor([0, 1, 2]), torch.tensor([1, 0]),
        torch.empty(2, 0, dtype=torch.long), num_nodes=3, n_classes=3,
    )
    torch.testing.assert_close(targets[0], torch.tensor([0.68, 0.16, 0.16]))
    assert diagnostics["fallback_count"] == 2


def test_sals_train_neighbors_and_uniform_fallback(structure):
    labels = torch.tensor([0, 1, 2, 2, 0])
    edges = torch.tensor([[0, 0, 1, 2], [1, 3, 4, 3]])
    targets, diagnostics = structure.build_sals_targets(
        labels, torch.tensor([2, 0, 1]), edges, num_nodes=5, n_classes=3,
    )
    expected = torch.tensor([[0.6, 0.32, 0], [0, 0.6, 0], [0, 0, 0.6]])
    expected[0] += 0.08 / 3
    expected[1:] += 0.4 / 3
    torch.testing.assert_close(targets[:3], expected)
    assert diagnostics["fallback_count"] == 2


@pytest.mark.parametrize("builder", ["build_postel_targets", "build_sals_targets", "als_label_propagation"])
def test_targets_ignore_nontrain_labels_and_duplicate_order(structure, graph, builder):
    build = getattr(structure, builder)
    labels = graph.noisy_y.clone()
    labels[~graph.train_mask] = -999
    args = dict(num_nodes=graph.num_nodes, n_classes=graph.num_classes)
    expected, _ = build(graph.noisy_y, graph.train_index, graph.edge_index, **args)
    changed, _ = build(labels, graph.train_mask, graph.edge_index.flip(1), **args)
    torch.testing.assert_close(expected, changed, rtol=0, atol=0)


def test_als_propagation_uses_symmetric_outdegree_normalization(structure):
    labels = torch.tensor([0, 1, 1])
    edges = torch.tensor([[0, 0, 1], [1, 2, 0]])
    propagated, _ = structure.als_label_propagation(
        labels, torch.tensor([1, 0]), edges, num_nodes=3, n_classes=2,
        steps=2, residual=0.3,
    )
    initial = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    adj = torch.tensor([[0., 1 / math.sqrt(2), 1 / math.sqrt(2)],
                        [1 / math.sqrt(2), 0., 0.], [0., 0., 0.]])
    expected = 0.7 * (adj @ (0.7 * adj @ initial + 0.3 * initial)) + 0.3 * initial
    torch.testing.assert_close(propagated, expected)


def test_als_eq5_has_trainable_target_gradients_and_decaying_pacing(structure):
    logits = torch.tensor([[0.4, -0.2], [-0.6, 0.8]], requires_grad=True)
    relevance = torch.tensor([[0.2, -0.5], [0.7, 0.1]], requires_grad=True)
    propagated = torch.tensor([[0.8, 0.1], [0.3, 0.4]])
    soft = F.softmax(propagated @ relevance.t(), dim=1)
    labels = torch.tensor([0, 1])
    loss = structure.als_eq5_loss(logits, labels, soft, alpha=0.1, kl_weight=0.001)
    expected = (0.9 * F.cross_entropy(logits, labels)
                - 0.1 * (soft * F.log_softmax(logits, dim=1)).sum(1).mean()
                + 0.001 * (soft * (soft.log() + math.log(2))).sum(1).mean())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(relevance.grad) == 4
    assert torch.isfinite(logits.grad).all()
    alpha = structure.als_pacing_alpha(199, pacing="exponential", alpha_max=0.1, rate=-0.001)
    assert alpha == pytest.approx(0.1 * math.exp(-0.199))
    assert structure.als_pacing_alpha(5, pacing="linear", alpha_max=0.1, rate=0.03) == 0.1


def test_edge_preparation_keeps_loops_except_for_gsdn(structure):
    edges = torch.tensor([[1, 0, 0, 1], [1, 1, 1, 0]])
    canonical = structure._canonical_edge_index(edges, 3, "cpu")
    assert canonical.tolist() == [[0, 1, 1], [1, 0, 1]]
    assert structure._clean_gsdn_edges(canonical, 3).tolist() == [[0, 1], [1, 0]]
    loops = torch.tensor([[0, 1, 2], [0, 1, 2]])
    assert structure._clean_gsdn_edges(loops, 3).tolist() == [[0, 1, 2], [1, 2, 0]]
    assert structure._clean_gsdn_edges(torch.empty(2, 0, dtype=torch.long), 3).shape == (2, 0)


@pytest.mark.parametrize("method", ["postel", "als", "sals", "gsdn"])
def test_two_epoch_training_repeatable_with_protocol_settings(structure, graph, method):
    config = yaml.safe_load((ROOT / "configs/main/protocol_adaptations.yaml").read_text())
    settings = config["structure_assisted"][method]
    original = deepcopy(settings)
    torch.manual_seed(17)
    first = structure.train_structure_baseline(graph, method, settings, epochs=2, hidden=8)
    state = torch.get_rng_state().clone()
    torch.manual_seed(17)
    second = structure.train_structure_baseline(graph, method, settings, epochs=2, hidden=8)
    assert settings == original
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    assert first.logits.shape == (8, 3)
    assert torch.isfinite(first.logits).all()
    assert first.logits.device.type == "cpu" and not first.logits.requires_grad
    assert first.best_epoch in (1, 2)


@pytest.mark.parametrize("method", ["postel", "als", "sals", "gsdn"])
def test_first_validation_tie_restores_model_and_auxiliary_state(structure, graph, monkeypatch, method):
    created = []
    original = structure.MLP

    def record(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(structure, "MLP", record)
    monkeypatch.setattr(structure, "accuracy", lambda logits, labels: 0.5)
    settings = {"graph_batch_size": 5} if method == "gsdn" else {}
    torch.manual_seed(17)
    first = structure.train_structure_baseline(graph, method, settings, epochs=1, hidden=8)
    states = [deepcopy(model.state_dict()) for model in created]
    created.clear()
    torch.manual_seed(17)
    later = structure.train_structure_baseline(graph, method, settings, epochs=2, hidden=8)
    assert later.best_epoch == 1
    torch.testing.assert_close(first.logits, later.logits, rtol=0, atol=0)
    for actual, expected in zip(created, states, strict=True):
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_training_respects_ordered_indices_and_does_not_warm_up(structure, graph, monkeypatch):
    seen = []
    original = structure.soft_cross_entropy

    def capture(logits, targets):
        seen.append(targets.detach().clone())
        return original(logits, targets)

    monkeypatch.setattr(structure, "soft_cross_entropy", capture)
    targets, _ = structure.build_sals_targets(
        graph.noisy_y, graph.train_index, graph.edge_index, num_nodes=8, n_classes=3,
    )
    structure.train_structure_baseline(graph, "sals", {}, epochs=2, hidden=8)
    assert len(seen) == 2
    for actual in seen:
        torch.testing.assert_close(actual, targets[graph.train_index], rtol=0, atol=0)


@pytest.mark.parametrize("method,settings,match", [
    ("not-a-method", {}, "unsupported"),
    ("postel", {"iterative_pseudo_labeling": True}, "pseudo"),
    ("sals", {"unlabeled_neighbor_fallback": "hard"}, "fallback"),
    ("als", {"pacing": "cosine"}, "pacing"),
    ("als", {"propagation_steps": -1}, "steps"),
    ("sals", {"epsilon": 1.1}, "epsilon"),
    ("postel", {"temperature": 0}, "temperature"),
    ("gsdn", {"graph_batch_size": 0}, "batch"),
    ("als", {"warmup_epochs": 5}, "unsupported"),
])
def test_invalid_configuration_is_not_silently_approximated(structure, graph, method, settings, match):
    with pytest.raises(ValueError, match=match):
        structure.train_structure_baseline(graph, method, settings, epochs=2, hidden=8)


def test_zero_epochs_and_truly_edgeless_gsdn_rejected(structure, graph):
    with pytest.raises(ValueError, match="epochs"):
        structure.train_structure_baseline(graph, "als", {}, epochs=0)
    graph.edge_index = torch.empty(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="edge"):
        structure.train_structure_baseline(graph, "gsdn", {}, epochs=2)
