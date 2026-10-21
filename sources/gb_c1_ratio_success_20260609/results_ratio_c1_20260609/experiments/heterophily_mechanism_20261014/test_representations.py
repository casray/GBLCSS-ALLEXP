"""Numerical and protocol tests for label-free representation preparation."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def implementation():
    path = HERE / "prepare_representations.py"
    assert path.exists(), "representation implementation is not written yet"
    spec = importlib.util.spec_from_file_location("representations_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_chunked_grace_loss_and_gradients_match_original_dense(chunk):
    impl = implementation()
    torch.manual_seed(7)
    first = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    second = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    def semi(a, b):
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        within, between = torch.exp(a @ a.T / .7), torch.exp(a @ b.T / .7)
        return -torch.log(between.diag() / (within.sum(1) + between.sum(1) - within.diag()))
    dense = .5 * (semi(first, second) + semi(second, first)).mean()
    grads = torch.autograd.grad(dense, (first, second))
    actual = impl.chunked_grace_loss(first, second, .7, chunk)
    actual_grads = torch.autograd.grad(actual, (first, second))
    torch.testing.assert_close(actual, dense, atol=1e-12, rtol=1e-12)
    for a, b in zip(actual_grads, grads):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_graphacl_loss_matches_official_function_and_gradients():
    impl = implementation()
    source = ast.parse((HERE / "vendor/graphacl_hete_model.py").read_text())
    func = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "contrastive_loss")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "official_GraphACL", "exec"), scope)
    torch.manual_seed(11)
    projected, online = [F.normalize(torch.randn(6, 4, dtype=torch.float64), dim=1).requires_grad_() for _ in range(2)]
    target = F.normalize(torch.randn(6, 4, dtype=torch.float64), dim=1)
    positive = torch.tensor([[0, 1, 2], [1, 3, 4], [3, 2, 5], [0, 2, 3], [1, 2, 4], [3, 4, 5]])
    negative = positive.flip(0)
    expected = scope['contrastive_loss'](projected[:, None], online[:, None], target[positive], online[negative], .5, 1.2)
    actual = impl.graphacl_loss(projected, online, target, positive, negative, .5, 1.2)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, (projected, online), retain_graph=True)
    actual_grad = torch.autograd.grad(actual, (projected, online))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_graphconv_matches_dgl_symmetric_normalization_definition():
    impl = implementation()
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    layer = impl.SymmetricGraphConv(2, 3).double()
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
    inv = adjacency.sum(1).rsqrt()
    expected = (inv[:, None] * adjacency * inv[None, :]) @ x @ layer.weight + layer.bias
    torch.testing.assert_close(layer(x, edges), expected)


def test_roman_loader_preserves_node_order_and_never_loads_labels():
    impl = implementation()
    x, edges = impl.load_roman_inputs(HERE / "data/roman_empire.npz")
    with np.load(HERE / "data/roman_empire.npz") as data:
        torch.testing.assert_close(x, torch.from_numpy(data['node_features']))
        assert edges.shape[0] == 2
        pairs = set(map(tuple, edges.T.tolist()))
        assert all((b, a) in pairs for a, b in pairs)
        assert x.shape == (22662, 300)
    assert "node_labels" not in __import__('inspect').getsource(impl.load_roman_inputs)


def test_graphacl_target_is_frozen_and_ema_uses_online_parameters():
    impl = implementation()
    model = impl.GraphACL(3, 4, 2, .95)
    assert all(not p.requires_grad for p in model.target.parameters())
    before = [p.detach().clone() for p in model.target.parameters()]
    with torch.no_grad():
        for p in model.online.parameters(): p.add_(2)
    model.update_target()
    for old, target in zip(before, model.target.parameters()):
        torch.testing.assert_close(target, old + .1)


def test_sampling_preserves_official_padding_and_rejects_isolates():
    impl = implementation()
    neighbors = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]]
    positive, negative = impl.sample_graphacl_indices(neighbors, 3)
    assert positive.shape == negative.shape == (6, 3)
    assert positive[0].tolist() == [1, 1, 1]
    assert neighbors[0] == [1, 1, 1]
    assert all(len(set(row.tolist())) == 3 for row in negative)
    with pytest.raises(ValueError, match="isolated"):
        impl.sample_graphacl_indices([[] for _ in range(6)], 3)


def test_graphacl_full_training_step_is_finite_and_leaves_target_without_gradient():
    impl = implementation()
    torch.manual_seed(22)
    model = impl.GraphACL(3, 4, 1, .95)
    x = torch.randn(6, 3)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    positive, negative = impl.sample_graphacl_indices([[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]], 3)
    projected, online, target = model(x, edges)
    loss = impl.graphacl_loss(projected, online, target, positive, negative, .5)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.target.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.online.parameters())


def test_chunked_backward_matches_dense_through_encoder_and_projection():
    impl = implementation()
    import copy
    torch.manual_seed(37)
    dense_model = impl.GRACE(impl.Encoder(3, 4, torch.nn.ReLU(), 2), 4, 5, .7).double()
    chunk_model = copy.deepcopy(dense_model)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    first_x, second_x = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    dense_loss = dense_model.loss(dense_model(first_x, edges), dense_model(second_x, edges))
    chunk_loss = impl.chunked_grace_loss(chunk_model.projection(chunk_model(first_x, edges)),
                                         chunk_model.projection(chunk_model(second_x, edges)), .7, 2)
    dense_loss.backward()
    chunk_loss.backward()
    torch.testing.assert_close(dense_loss, chunk_loss, atol=1e-12, rtol=1e-12)
    for (name, dense_parameter), (_, chunk_parameter) in zip(dense_model.named_parameters(), chunk_model.named_parameters()):
        assert dense_parameter.grad is not None, name
        assert chunk_parameter.grad is not None, name
        torch.testing.assert_close(dense_parameter.grad, chunk_parameter.grad, atol=1e-12, rtol=1e-12)
