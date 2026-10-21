"""CPU contracts for the bounded extension of the unchanged TFR reference."""
import ast
from copy import deepcopy
from pathlib import Path
import types
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

try:
    import tfr_core
except ModuleNotFoundError as error:
    if error.name != "tfr_core":
        raise
    tfr_core = None


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'robust' if (HERE / 'robust').is_dir() else HERE.parent / "robust_graph_seed0_20261006"
SETTINGS = dict(beta=.5, beta_dropout_ratio=.581229595176513,
                pseudo_ratio=.3156206408879477, use_recon_softmax=False)
DROPOUT = .6886554578492039


def graph():
    undirected = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (4, 5), (3, 5), (5, 6)]
    edges = torch.tensor(undirected + [(j, i) for i, j in undirected]).T.contiguous()
    assignment = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    torch.manual_seed(20261020)
    x = torch.rand(7, 5, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    sparse = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], dtype=x.dtype), (7, 7)).coalesce()
    return x, edges, sparse, assignment, labels, torch.tensor([0, 2, 4])


def original_objective(decoder, x, labels, settings, coefficient):
    """Execute the actual nested objective AST, rather than rewriting its math."""
    path = REFERENCE / "run_screen.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_tfr")
    objective = next(n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == "objective")
    _, edges, _, _, _, _ = graph()
    namespace = dict(F=F, dropout_edge=dropout_edge, decoder=decoder, loop_edges=edges,
                     model_conf=settings, coefficient=coefficient,
                     data=types.SimpleNamespace(x=x, noisy_y=labels, num_nodes=x.shape[0]))
    exec(compile(ast.Module(body=[objective], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["objective"]


class TFRCoreTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(tfr_core, "the bounded TFR adapter has not been implemented")
        # These warnings describe the unchanged reference's native COO path.
        warnings.filterwarnings("ignore", message="Converting sparse tensor to CSR format.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")

    def models(self, rho=0., mode="independent", dropout=DROPOUT):
        x, edges, sparse, assignment, labels, ids = graph()
        GCN, Decoder = tfr_core.load_reference_classes()
        reference = GCN(5, 16, 3, n_layers=2, dropout=dropout, norm_info=None,
                        act="F.relu", input_layer=False, output_layer=False).double()
        model = tfr_core.TFRClassifier(5, 16, 3, dropout, assignment, rho, mode).double()
        model.base.load_state_dict(reference.state_dict())
        decoder = Decoder(3, 16, 5, .3907730489369916, False).double()
        return reference, model, decoder, (x, edges, sparse, assignment, labels, ids)

    def test_reference_parameterization_and_assignment_snapshot(self):
        reference, model, _, (_, _, _, assignment, _, _) = self.models()
        self.assertEqual(list(dict(model.base.named_parameters())), list(dict(reference.named_parameters())))
        self.assertEqual(sum(p.numel() for p in reference.parameters()), sum(p.numel() for p in model.parameters()))
        self.assertEqual(model.base.n_layers, 2)
        self.assertFalse(model.base.is_norm)
        self.assertEqual(model.base.dropout, DROPOUT)
        assignment.fill_(0)
        self.assertEqual(model.assignment.tolist(), [0, 0, 0, 0, 1, 1, 2])
        self.assertIn("assignment", dict(model.named_buffers()))

    def test_rho_zero_native_logits_loss_gradients_and_three_adam_steps(self):
        for adjacency_kind in ("edge", "sparse"):
            with self.subTest(adjacency=adjacency_kind):
                reference, model, decoder, data = self.models()
                x, edges, sparse, _, labels, ids = data
                adj = edges if adjacency_kind == "edge" else sparse
                other_decoder = deepcopy(decoder)
                opt_ref = torch.optim.Adam(list(reference.parameters()) + list(decoder.parameters()), lr=.1, weight_decay=.001)
                opt_new = torch.optim.Adam(list(model.parameters()) + list(other_decoder.parameters()), lr=.1, weight_decay=.001)
                coefficient = 0.
                for step in range(3):
                    reference.train()
                    model.train()
                    decoder.train(step == 0)
                    other_decoder.train(step == 0)
                    opt_ref.zero_grad()
                    opt_new.zero_grad()
                    objective = original_objective(decoder, x, labels, SETTINGS, coefficient)
                    rng = torch.get_rng_state()
                    expected_logits = reference(x, adj)
                    expected_loss = objective(expected_logits, ids, True)
                    expected_rng = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    logits = model(x, adj)
                    loss, parts = tfr_core.tfr_loss(logits, x, labels, ids, other_decoder, edges, coefficient, SETTINGS)
                    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(set(parts), {"recon", "pseudo", "supervised"})
                    self.assertTrue(all(value.requires_grad for value in parts.values()))
                    self.assertEqual(other_decoder.training, step == 0)
                    expected_loss.backward()
                    loss.backward()
                    for p, q in zip(list(reference.parameters()) + list(decoder.parameters()),
                                    list(model.parameters()) + list(other_decoder.parameters())):
                        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    opt_ref.step()
                    opt_new.step()
                    for p, q in zip(list(reference.parameters()) + list(decoder.parameters()),
                                    list(model.parameters()) + list(other_decoder.parameters())):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                    coefficient = float((logits[ids].argmax(1) == labels[ids]).float().mean())
                    # The qualified loop consumes two augmented eval objectives.
                    reference.eval()
                    model.eval()
                    decoder.eval()
                    other_decoder.eval()
                    objective = original_objective(decoder, x, labels, SETTINGS, coefficient)
                    rng = torch.get_rng_state()
                    with torch.no_grad():
                        expected_logits = reference(x, adj)
                        expected_eval = [objective(expected_logits, ids, False) for _ in range(2)]
                    expected_rng = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    with torch.no_grad():
                        logits = model(x, adj, original=True)
                        for expected_value in expected_eval:
                            actual_value, _ = tfr_core.tfr_loss(logits, x, labels, ids, other_decoder,
                                                             edges, coefficient, SETTINGS)
                            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertFalse(other_decoder.training)

    def test_operator_endpoints_and_mixture_match_dense_equations_with_bias(self):
        for rho in (0., .5, 1.):
            for adjacency_kind in ("edge", "sparse"):
                with self.subTest(rho=rho, adjacency=adjacency_kind):
                    _, model, _, (x, edges, sparse, assignment, _, _) = self.models(rho=rho, dropout=0.)
                    model.eval()
                    adjacency = sparse.to_dense() + torch.eye(7, dtype=x.dtype)
                    degree = adjacency.sum(1).rsqrt()
                    p = degree[:, None] * adjacency * degree[None, :]
                    membership = F.one_hot(assignment).to(x.dtype)
                    pi = membership @ torch.diag(1 / membership.sum(0)) @ membership.T
                    mixed = (1 - rho) * p + rho * pi @ p @ pi
                    expected = x
                    with torch.no_grad():
                        for layer in model.base.convs:
                            layer.bias.copy_(torch.linspace(.2, .8, layer.bias.numel(), dtype=x.dtype))
                    for index, layer in enumerate(model.base.convs):
                        expected = mixed @ expected @ layer.lin.weight.T + layer.bias
                        if index == 0:
                            expected = F.relu(expected)
                    adj = edges if adjacency_kind == "edge" else sparse
                    actual = model(x, adj)
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    expected_grads = torch.autograd.grad(expected.square().sum(), tuple(model.parameters()))
                    actual_grads = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_shared_mask_matches_group_dropout_and_original_eval_uses_reference(self):
        reference, model, _, (x, edges, _, assignment, _, _) = self.models(rho=1., mode="shared")
        def project(z):
            return torch.stack([z[assignment == group].mean(0) for group in assignment])[...]
        model.train()
        torch.manual_seed(42)
        hidden = F.relu(project(model.base.convs[0](project(x), edges)))
        group_mask = F.dropout(hidden.new_ones((3, 16)), DROPOUT, training=True)
        expected = project(model.base.convs[1](project(hidden * group_mask[assignment]), edges))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        torch.testing.assert_close(model(x, edges), expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        model.eval()
        reference.eval()
        torch.testing.assert_close(model(x, edges, original=True), reference(x, edges), rtol=0, atol=0)
        first = model(x, edges)
        torch.testing.assert_close(first, model(x, edges), rtol=0, atol=0)

    def test_loss_softmax_option_matches_original_and_canonical_components(self):
        for use_softmax in (False, True):
            _, _, decoder, (x, edges, _, _, labels, ids) = self.models()
            settings = dict(SETTINGS, use_recon_softmax=use_softmax)
            logits = torch.randn(7, 3, dtype=x.dtype, requires_grad=True)
            reference_decoder = deepcopy(decoder)
            component_decoder = deepcopy(decoder)
            objective = original_objective(reference_decoder, x, labels, settings, .4)
            rng = torch.get_rng_state()
            expected = objective(logits, ids, True)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            loss, parts = tfr_core.tfr_loss(logits, x, labels, ids, decoder, edges, .4, settings)
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            torch.set_rng_state(rng)
            pseudo = logits.softmax(1)
            dropped = dropout_edge(edges, p=settings["beta_dropout_ratio"])[0]
            reconstructed, confidence = component_decoder(pseudo, dropped)
            selected = confidence.squeeze().argsort(descending=True)[:int(x.shape[0] * settings["pseudo_ratio"])]
            if use_softmax:
                reconstructed = reconstructed.softmax(1)
            canonical = dict(recon=F.cross_entropy(reconstructed, x),
                             pseudo=F.cross_entropy(logits[selected], pseudo[selected].detach()),
                             supervised=F.cross_entropy(logits[ids], labels[ids]))
            for key, value in canonical.items():
                torch.testing.assert_close(parts[key], value, rtol=0, atol=0)
            expected.backward(retain_graph=True)
            loss.backward()
            for p, q in zip(reference_decoder.parameters(), decoder.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_invalid_rho_assignment_and_mask_mode(self):
        _, _, _, assignment, _, _ = graph()
        for rho in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tfr_core.TFRClassifier(5, 16, 3, DROPOUT, assignment, rho)
        for invalid in (torch.tensor([]).long(), torch.tensor([0, -1]), torch.tensor([0, 2]), torch.tensor([[0, 1]])):
            with self.assertRaises(ValueError):
                tfr_core.TFRClassifier(5, 16, 3, DROPOUT, invalid)
        with self.assertRaises(TypeError):
            tfr_core.TFRClassifier(5, 16, 3, DROPOUT, assignment.float())
        with self.assertRaises(ValueError):
            tfr_core.TFRClassifier(5, 16, 3, DROPOUT, assignment, mask_mode="unknown")
        _, model, _, (x, edges, _, _, _, _) = self.models()
        with self.assertRaises(ValueError):
            model(x[:-1], edges)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
