from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from gblcss import distillation, training
from gblcss.baselines import noisy_label
from gblcss.experiment import make_smoke_graph


@pytest.mark.parametrize('method', ['mlp_ce', 'small_loss', 'coteaching', 'jocor', 'sce', 'elr', 'logitclip'])
def test_baseline_keeps_first_validation_tie(monkeypatch, method):
    graph, _ = make_smoke_graph(0)
    monkeypatch.setattr(noisy_label, 'accuracy', lambda logits, labels: 0.5)
    result = noisy_label.train_noisy_label_baseline(graph, method, noise_rate=0.4, epochs=3)
    assert result.best_epoch == 1


def test_direct_mlp_keeps_first_validation_tie(monkeypatch):
    graph, _ = make_smoke_graph(0)
    targets = F.one_hot(graph.noisy_y, graph.num_classes).float()
    monkeypatch.setattr(training, 'accuracy', lambda logits, labels: 0.5)
    result = training.train_x_only_mlp(graph, targets, epochs=3)
    assert result.best_epoch == 1


def test_consensus_student_precedes_teacher_and_keeps_first_tie(monkeypatch):
    graph, _ = make_smoke_graph(0)
    targets = F.one_hot(graph.noisy_y, graph.num_classes).float()
    constructors = []
    mlp, gcn = distillation.MLP, distillation.GCN

    def student(*args, **kwargs):
        constructors.append('student')
        return mlp(*args, **kwargs)

    def teacher(*args, **kwargs):
        constructors.append('teacher')
        return gcn(*args, **kwargs)

    monkeypatch.setattr(distillation, 'MLP', student)
    monkeypatch.setattr(distillation, 'GCN', teacher)
    monkeypatch.setattr(distillation, 'accuracy', lambda logits, labels: 0.5)
    result = distillation.train_consensus_teacher_student(
        graph, targets, teacher_epochs=2, student_epochs=3)
    assert constructors == ['student', 'teacher']
    assert result.best_epoch == 1


def test_zero_epochs_rejected():
    graph, _ = make_smoke_graph(0)
    targets = F.one_hot(graph.noisy_y, graph.num_classes).float()
    with pytest.raises(ValueError, match='epochs'):
        training.train_x_only_mlp(graph, targets, epochs=0)


@pytest.mark.parametrize('route', ['direct', 'teacher'])
def test_preserves_training_index_order_and_sparse_teacher(monkeypatch, route):
    graph, _ = make_smoke_graph(0)
    graph.train_index = torch.where(graph.train_mask)[0].flip(0)
    targets = F.one_hot(graph.noisy_y, graph.num_classes).float()
    module = training if route == 'direct' else distillation
    original = module.soft_cross_entropy
    seen = []

    def loss(logits, target):
        seen.append(target.detach().clone())
        return original(logits, target)

    monkeypatch.setattr(module, 'soft_cross_entropy', loss)
    if route == 'direct':
        training.train_x_only_mlp(graph, targets, epochs=1)
    else:
        forward = distillation.GCN.forward

        def sparse_forward(model, x, adjacency):
            assert adjacency.layout == torch.sparse_coo
            return forward(model, x, adjacency)

        monkeypatch.setattr(distillation.GCN, 'forward', sparse_forward)
        distillation.train_consensus_teacher_student(graph, targets, teacher_epochs=1, student_epochs=1)
    assert torch.equal(seen[0], targets[graph.train_index])


def test_baseline_uses_its_method_configuration(monkeypatch):
    graph, _ = make_smoke_graph(0)
    seen = []
    original = noisy_label._symmetric_cross_entropy

    def capture(logits, labels, alpha, beta):
        seen.append((alpha, beta))
        return original(logits, labels, alpha, beta)

    monkeypatch.setattr(noisy_label, '_symmetric_cross_entropy', capture)
    noisy_label.train_noisy_label_baseline(
        graph, 'sce', noise_rate=.4, epochs=1,
        settings={'cross_entropy_weight': .2, 'reverse_cross_entropy_weight': .8})
    assert seen == [(.2, .8)]


def test_matched_baseline_preserves_training_order(monkeypatch):
    graph, _ = make_smoke_graph(0)
    graph.train_index = torch.where(graph.train_mask)[0].flip(0)
    seen = []
    original = noisy_label._symmetric_cross_entropy

    def capture(logits, labels, alpha, beta):
        seen.append(labels.clone())
        return original(logits, labels, alpha, beta)

    monkeypatch.setattr(noisy_label, '_symmetric_cross_entropy', capture)
    noisy_label.train_noisy_label_baseline(graph, 'sce', noise_rate=.4, epochs=1)
    assert torch.equal(seen[0], graph.noisy_y[graph.train_index])
