from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import GraphData
from gblcss.models import GCN, MLP
from gblcss.training import TrainResult


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def transfer():
    assert importlib.util.find_spec("gblcss.baselines.graph_transfer") is not None, (
        "missing graph_transfer module and train_transfer_baseline API"
    )
    return importlib.import_module("gblcss.baselines.graph_transfer")


def synthetic_graph() -> GraphData:
    generator = torch.Generator().manual_seed(19)
    n = 12
    source = torch.arange(n)
    target = source.roll(1)
    edges = torch.stack([torch.cat([source, target]), torch.cat([target, source])])
    train = torch.tensor([5, 0, 8, 2, 10, 3])
    val = torch.tensor([7, 1, 6])
    test = torch.tensor([11, 4, 9])
    masks = []
    for index in (train, val, test):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    labels = source % 3
    noisy = labels.clone()
    noisy[train[:2]] = (noisy[train[:2]] + 1) % 3
    return GraphData(
        "synthetic", torch.randn(n, 5, generator=generator), edges, labels,
        *masks, noisy_y=noisy, train_index=train, val_index=val, test_index=test,
    )


@pytest.mark.parametrize("method", ["glnn_style", "ffg2m", "krd", "vqgraph"])
def test_two_epoch_routes_are_repeatable_and_do_not_mutate_inputs(transfer, method):
    graph = synthetic_graph()
    settings = yaml.safe_load((ROOT / "configs/main/protocol_adaptations.yaml").read_text())[
        "graph_transfer"
    ][method]
    before = deepcopy(graph)
    settings_before = deepcopy(settings)
    torch.manual_seed(31)
    result = transfer.train_transfer_baseline(graph, method, settings, epochs=2, hidden=8)
    final_rng = torch.get_rng_state()
    torch.manual_seed(31)
    again = transfer.train_transfer_baseline(graph, method, settings, epochs=2, hidden=8)
    assert isinstance(result, TrainResult)
    assert result.logits.shape == (12, 3)
    assert result.logits.device.type == "cpu"
    assert not result.logits.requires_grad
    assert torch.isfinite(result.logits).all()
    assert result.best_epoch in (1, 2)
    assert all(0 <= value <= 1 for value in (
        result.train_accuracy, result.validation_accuracy, result.test_accuracy,
    ))
    torch.testing.assert_close(result.logits, again.logits, rtol=0, atol=0)
    assert torch.equal(final_rng, torch.get_rng_state())
    assert settings == settings_before
    for name, value in vars(before).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(graph, name))


def test_ffg2m_uses_logit_kl_and_detaches_teacher(transfer):
    torch.manual_seed(3)
    student = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 1, 3, 2]])
    loop_free = edges[:, edges[0] != edges[1]]
    with_loops = torch.cat([loop_free, torch.arange(4).repeat(2, 1)], dim=1)
    src, dst = with_loops
    expected_low = F.kl_div(
        F.log_softmax(student[src], dim=1), F.log_softmax(teacher.detach()[dst], dim=1),
        reduction="batchmean", log_target=True,
    )
    src, dst = loop_free
    expected_high = F.kl_div(
        F.log_softmax((student[src] - student[dst]).abs() / 1.5, dim=1),
        F.log_softmax((teacher.detach()[src] - teacher.detach()[dst]).abs() / 1.5, dim=1),
        reduction="batchmean", log_target=True,
    )
    loss = transfer.ffg2m_low_frequency_loss(student, teacher, edges, 4)
    loss += transfer.ffg2m_high_frequency_loss(student, teacher, edges, tau=1.5)
    expected = expected_low + expected_high
    torch.testing.assert_close(loss, expected)
    actual_gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    loss.backward()
    assert teacher.grad is None


def test_ffg2m_formal_weighting_and_empty_edges(transfer):
    graph = synthetic_graph()
    student = torch.randn(12, 3, requires_grad=True)
    teacher = torch.randn(12, 3)
    edges = torch.empty(2, 0, dtype=torch.long)
    high = transfer.ffg2m_high_frequency_loss(student, teacher, edges, tau=1.5)
    assert high.item() == 0
    low = transfer.ffg2m_low_frequency_loss(student, teacher, edges, 12)
    expected = F.kl_div(F.log_softmax(student, dim=1), F.log_softmax(teacher, dim=1),
                        reduction="batchmean", log_target=True)
    torch.testing.assert_close(low, expected)
    result = transfer._ffg2m_loss(student, teacher, edges, graph.noisy_y, graph.train_index, {})
    torch.testing.assert_close(result, 0.4 * F.cross_entropy(
        student[graph.train_index], graph.noisy_y[graph.train_index],
    ) + 0.6 * low)


def test_distillation_uses_all_nodes_and_normalized_detached_weights(transfer):
    student = torch.randn(6, 3, requires_grad=True)
    teacher = F.softmax(torch.randn(6, 3), dim=1)
    weights = torch.tensor([0., 2., 0., 1., 0., 0.], requires_grad=True)
    expected = -(teacher * F.log_softmax(student / 2, dim=1)).sum(dim=1) * 4
    loss = transfer._distillation_loss(student, teacher, 2., weights)
    torch.testing.assert_close(loss, (expected * weights.detach() / weights.detach().mean()).mean())
    loss.backward()
    assert weights.grad is None
    assert student.grad[weights == 0].abs().sum() == 0
    assert student.grad[3].abs().sum() > 0
    unweighted = transfer._distillation_loss(student, teacher, 2.)
    torch.testing.assert_close(unweighted, expected.mean())
    torch.testing.assert_close(
        transfer._distillation_loss(student, teacher, 2., torch.zeros(6)), expected.mean(),
    )


def test_krd_uses_four_perturbations_and_top_half(transfer):
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x, adj):
            self.inputs.append(x.clone())
            return x[:, :3]

    graph = synthetic_graph()
    teacher = Teacher()
    torch.manual_seed(41)
    weights = transfer._krd_weights(teacher, graph.x, graph.edge_index, {})
    assert not teacher.training
    assert len(teacher.inputs) == 5
    torch.manual_seed(41)
    for observed in teacher.inputs[1:]:
        torch.testing.assert_close(observed, graph.x + torch.randn_like(graph.x) * 0.01)
    assert (weights > 0).sum() == 6
    assert not weights.requires_grad
    constant = transfer._krd_weights(teacher, torch.zeros_like(graph.x), graph.edge_index,
                                    {"perturbation_std": 0.})
    assert (constant > 0).sum() == 6
    assert torch.all(constant[constant > 0] == 1)


@pytest.mark.parametrize("model_name,base", [("_MLPWithHidden", MLP), ("_GCNWithHidden", GCN)])
def test_hidden_wrappers_preserve_initialization_and_pre_dropout_hidden(transfer, model_name, base):
    graph = synthetic_graph()
    torch.manual_seed(12)
    plain = base(5, 8, 3, 2, 0.5)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(12)
    model = getattr(transfer, model_name)(5, 8, 3, 2, 0.5)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for a, b in zip(plain.parameters(), model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.train()
    if base is MLP:
        expected_hidden = F.relu(model.layers[0](graph.x))
        _, hidden = model.forward_with_hidden(graph.x)
    else:
        expected_hidden = F.relu(model.layers[0](graph.x, graph.edge_index))
        _, hidden = model.forward_with_hidden(graph.x, graph.edge_index)
    torch.testing.assert_close(hidden, expected_hidden)


def test_vq_codebook_node_edge_and_token_losses_have_gradients(transfer):
    graph = synthetic_graph()
    teacher = transfer._VQGraphTeacher(transfer._GCNWithHidden(5, 8, 3, 2, 0.), 8, 7)
    logits, hidden, assignments, quantized = teacher(graph.x, graph.edge_index)
    torch.testing.assert_close(assignments.sum(dim=1), torch.ones(12))
    node = F.mse_loss(F.normalize(hidden, dim=1), quantized)
    torch.manual_seed(9)
    edge = transfer._vq_edge_loss(quantized, graph.edge_index, 5)
    assert node > 0 and edge > 0
    loss = F.cross_entropy(logits[graph.train_index], graph.noisy_y[graph.train_index])
    loss = loss + 0.001 * node + 0.03 * edge
    loss.backward()
    assert teacher.codebook.grad.abs().sum() > 0
    assert teacher.base_gcn.layers[0].lin.weight.grad.abs().sum() > 0
    student = torch.randn(12, 8, requires_grad=True)
    target = assignments.detach()
    codebook = F.normalize(teacher.codebook.detach(), dim=1)
    token = transfer._vq_token_loss(student, target, codebook, 1.)
    expected = F.kl_div(F.log_softmax(F.normalize(student, dim=1) @ codebook.t(), dim=1),
                        target, reduction="batchmean")
    torch.testing.assert_close(token, expected)
    token.backward()
    assert student.grad.abs().sum() > 0
    assert transfer._vq_edge_loss(quantized, torch.empty(2, 0, dtype=torch.long), 5) == 0


def test_teacher_keeps_latest_tie_student_keeps_first_tie(transfer, monkeypatch):
    graph = synthetic_graph()
    teacher_epochs = []
    fit = transfer._fit_teacher

    def capture(*args, **kwargs):
        model, epoch = fit(*args, **kwargs)
        teacher_epochs.append(epoch)
        return model, epoch

    monkeypatch.setattr(transfer, "_fit_teacher", capture)
    monkeypatch.setattr(transfer, "accuracy", lambda logits, labels: 0.5)
    result = transfer.train_transfer_baseline(graph, "glnn_style", {}, epochs=2, hidden=8)
    assert teacher_epochs == [2]
    assert result.best_epoch == 1


@pytest.mark.parametrize("method,settings,kwargs,match", [
    ("unknown", {}, {}, "unsupported"),
    ("glnn_style", {}, {"epochs": 0}, "epochs"),
    ("glnn_style", {"teacher_epochs": 0}, {}, "teacher_epochs"),
    ("vqgraph", {"sampled_edges_per_epoch": 0}, {}, "sampled_edges_per_epoch"),
    ("vqgraph", {"teacher_hidden": 9}, {"hidden": 8}, "hidden"),
])
def test_invalid_configuration_fails_before_rng_changes(transfer, method, settings, kwargs, match):
    graph = synthetic_graph()
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=match):
        transfer.train_transfer_baseline(graph, method, settings, **kwargs)
    assert torch.equal(state, torch.get_rng_state())
