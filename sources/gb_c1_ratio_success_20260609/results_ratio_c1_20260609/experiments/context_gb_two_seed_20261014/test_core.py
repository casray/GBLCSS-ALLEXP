"""Numerical and scientific invariants for the experimental context pipeline."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_block_contrastive_matches_dense_value_and_gradient():
    import core
    torch.manual_seed(1)
    first = torch.randn(11, 7, dtype=torch.double, requires_grad=True)
    second = torch.randn(11, 7, dtype=torch.double, requires_grad=True)
    z1, z2 = F.normalize(first, dim=1), F.normalize(second, dim=1)
    a, b, cross = (z1 @ z1.T / .7).exp(), (z2 @ z2.T / .7).exp(), (z1 @ z2.T / .7).exp()
    expected = .5 * (-torch.log(cross.diag() / (a.sum(1) + cross.sum(1) - a.diag()))
                      -torch.log(cross.diag() / (b.sum(1) + cross.sum(0) - b.diag()))).mean()
    grads = torch.autograd.grad(expected, (first, second))
    actual = core.contrastive_loss(first, second, .7, chunk_size=4)
    actual_grads = torch.autograd.grad(actual, (first, second))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    for a, b in zip(actual_grads, grads):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_context_excludes_root_and_isolates_empty_nodes():
    import core
    h = torch.tensor([[1., 3.], [4., 5.], [8., 9.], [11., 12.]], requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 0, 3], [1, 0, 0, 0, 3]])
    mean, valid = core.neighbor_mean(h, edges)
    torch.testing.assert_close(mean[0], (h[1]+h[2])/2)
    torch.testing.assert_close(mean[1], h[0])
    assert valid.tolist() == [True, True, False, False]
    mean[0].sum().backward()
    assert h.grad[0].abs().sum() == 0
    torch.testing.assert_close(h.grad[1], torch.ones(2)/2)


def test_vicreg_masks_same_roots_and_does_not_normalize():
    import core
    torch.manual_seed(3)
    c1 = torch.randn(8, 4, requires_grad=True)
    c2 = torch.randn(8, 4, requires_grad=True)
    valid1 = torch.tensor([1, 1, 1, 1, 0, 0, 1, 0], dtype=torch.bool)
    valid2 = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
    mask = valid1 & valid2
    loss, stats = core.vicreg(c1, c2, mask)
    assert stats['valid_roots'] == 4
    direct, _ = core.vicreg(c1[mask], c2[mask], torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(loss, direct)
    loss.backward()
    assert c1.grad[~mask].abs().sum() == 0
    assert c2.grad[~mask].abs().sum() == 0
    skipped, _ = core.vicreg(c1, c2, torch.zeros(8, dtype=torch.bool))
    assert skipped.item() == 0


def test_combined_space_has_weighted_distance_and_validity():
    import core
    h = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    c = torch.tensor([[0., 2.], [3., 0.], [0., 0.]])
    result, valid = core.combine_space(h, c, .25, torch.ones(3, dtype=torch.bool))
    assert valid.tolist() == [True, True, False]
    expected = .75 * (F.normalize(h, dim=1)[0]-F.normalize(h, dim=1)[1]).square().sum()
    expected += .25 * (F.normalize(c, dim=1)[0]-F.normalize(c, dim=1)[1]).square().sum()
    torch.testing.assert_close((result[0]-result[1]).square().sum(), expected)
    with pytest.raises(ValueError):
        core.combine_space(h, c, 1.1, valid)


def test_sampled_context_caps_degree_and_keeps_one_neighbor():
    import core
    edges = torch.tensor([[1, 2, 3, 4, 0], [0, 0, 0, 0, 1]])
    torch.manual_seed(1)
    sampled = core.sample_context_edges(edges, num_nodes=5, max_neighbors=2)
    counts = torch.bincount(sampled[1], minlength=5)
    assert counts.tolist() == [2, 1, 0, 0, 0]


def test_backbone_accepts_edge_weight_gradient():
    import core
    torch.manual_seed(5)
    model = core.WeightedGRACE(4, 6, 6, .7, 'relu')
    x = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    weights = torch.ones(6, requires_grad=True)
    model(x, edges, weights).square().mean().backward()
    assert weights.grad is not None and weights.grad.abs().sum() > 0


def test_context_gradient_can_be_added_without_updating_generator():
    import core
    torch.manual_seed(6)
    encoder = torch.nn.Linear(3, 4)
    generator = torch.nn.Linear(3, 3)
    head = torch.nn.Linear(4, 2)
    x = torch.randn(7, 3)
    h1, h2 = encoder(x+generator(x)), encoder(x-generator(x))
    node = core.contrastive_loss(h1, h2, .7, 3)
    expected = torch.autograd.grad(node, list(generator.parameters()), retain_graph=True)
    ctx, _ = core.vicreg(head(h1), head(h2), torch.ones(7,dtype=torch.bool))
    parameters = list(encoder.parameters())+list(head.parameters())
    grads = torch.autograd.grad(.1*ctx/25,parameters,retain_graph=True)
    node.backward()
    for p,g in zip(parameters,grads):p.grad=g if p.grad is None else p.grad+g
    for p,g in zip(generator.parameters(),expected):torch.testing.assert_close(p.grad,g)


def test_weighted_encoder_matches_existing_grace_without_weights():
    import core
    project = Path(__file__).resolve().parents[2]/'GB-LCSS_code/src'
    sys.path.insert(0,str(project))
    from gblcss.grace import GRACE, Encoder
    torch.manual_seed(0)
    old = GRACE(Encoder(4,6,torch.nn.ReLU()),6,6,.7)
    new = core.WeightedGRACE(4,6,6,.7,'relu')
    new.convolutions.load_state_dict(old.encoder.convolutions.state_dict())
    new.projection_1.load_state_dict(old.projection_1.state_dict())
    new.projection_2.load_state_dict(old.projection_2.state_dict())
    x=torch.randn(5,4);edges=torch.tensor([[0,1,2,3],[1,2,3,0]])
    torch.testing.assert_close(old(x,edges),new(x,edges))
    torch.testing.assert_close(old.projection(old(x,edges)),new.project(new(x,edges)))
