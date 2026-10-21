import torch
import torch.nn.functional as F
import pytest

from gblcss.baselines.graph_mlp import contrastive_loss, train_graph_mlp
from gblcss.data import set_seed
from gblcss.experiment import make_smoke_graph


def test_sampled_contrast_matches_original_objective():
    hidden = torch.randn(8, 4, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    set_seed(4)
    actual = contrastive_loss(hidden, edges, 9, .5)
    set_seed(4)
    source, target = edges[:, torch.randint(4, (9,))]
    negative = torch.randint(8, (9,))
    normalized = F.normalize(hidden, dim=1)
    scores = torch.stack([(normalized[source] * normalized[target]).sum(1),
                          (normalized[source] * normalized[negative]).sum(1)], dim=1) / .5
    expected = F.cross_entropy(scores, torch.zeros(9, dtype=torch.long))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(hidden.grad).all()


def test_training_repeatable_and_attribute_only():
    graph, _ = make_smoke_graph(0)
    set_seed(2)
    first = train_graph_mlp(graph, {}, epochs=2)
    set_seed(2)
    second = train_graph_mlp(graph, {}, epochs=2)
    torch.testing.assert_close(first.logits, second.logits, rtol=1e-5, atol=1e-6)
    assert first.test_accuracy == second.test_accuracy
    assert first.logits.shape == (graph.num_nodes, graph.num_classes)
    assert 0 <= first.test_accuracy <= 1


def test_empty_edges_fail_clearly():
    with pytest.raises(ValueError, match='edges'):
        contrastive_loss(torch.randn(5, 2), torch.empty(2, 0, dtype=torch.long), 4, .5)
