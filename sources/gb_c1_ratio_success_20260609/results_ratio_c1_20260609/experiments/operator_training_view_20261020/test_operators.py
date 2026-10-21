"""Small exact fixtures for projection, risk, and coarse/node equivalence."""
import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from operators import OperatorGCN, apply_operator, per_example_loss, project


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.a = torch.tensor([0, 0, 1, 2, 2, 2])
        self.h = F.one_hot(self.a, 3).to(torch.float64)
        self.r = self.h.T / self.h.sum(0)[:, None]
        self.pi = self.h @ self.r
        adjacency = torch.eye(6, dtype=torch.float64)
        for i, j in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 5), (4, 5)):
            adjacency[i, j] = adjacency[j, i] = 1
        inv_sqrt_degree = adjacency.sum(1).rsqrt()
        self.p = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        self.x = torch.randn(6, 4, dtype=torch.float64)

    def assertClose(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    def test_projection_and_gradient_match_explicit_unequal_groups(self):
        z = self.x.clone().requires_grad_()
        actual = project(z, self.a, 3)
        expected = self.pi @ z
        probe = torch.randn_like(z)
        self.assertClose(actual, expected)
        self.assertClose(torch.autograd.grad((actual * probe).sum(), z)[0],
                         torch.autograd.grad((expected * probe).sum(), z)[0])

    def test_singleton_projection_is_identity(self):
        self.assertClose(project(self.x, torch.arange(6)), self.x)

    def test_invalid_projection_inputs_are_rejected(self):
        cases = [
            (torch.tensor(1.), torch.tensor([0]), None),
            (torch.empty(6, 2, 2), self.a, None),
            (torch.empty(0, 4), torch.empty(0, dtype=torch.long), None),
            (torch.empty(6, 0), self.a, None),
            (self.x, self.a.float(), None),
            (self.x, self.a[:-1], None),
            (self.x, torch.tensor([0, 0, -1, 2, 2, 2]), None),
            (self.x, torch.tensor([0, 0, 2, 2, 2, 2]), None),
            (self.x, self.a, 4),
            (self.x, self.a, 2),
            (self.x, self.a[:, None], None),
        ]
        for z, a, m in cases:
            with self.subTest(shape=z.shape, assignment=a.tolist(), groups=m):
                with self.assertRaises((ValueError, TypeError)):
                    project(z, a, m)

    def test_four_operators_dense_and_sparse_match_matrix_and_gradient(self):
        matrices = {"P": self.p, "PPI": self.p @ self.pi,
                    "PIP": self.pi @ self.p,
                    "PIPPI": self.pi @ self.p @ self.pi}
        # CSR is checked as well as COO; its PyTorch beta warning is expected.
        for layout in ("dense", "coo", "csr"):
            p = self.p if layout == "dense" else getattr(self.p, f"to_sparse_{layout}")()
            for mode, matrix in matrices.items():
                with self.subTest(layout=layout, mode=mode):
                    z = self.x.clone().requires_grad_()
                    probe = torch.randn_like(z)
                    actual = apply_operator(z, p, self.a, mode)
                    expected = matrix @ z
                    self.assertClose(actual, expected)
                    self.assertClose(torch.autograd.grad((actual * probe).sum(), z)[0],
                                     torch.autograd.grad((expected * probe).sum(), z)[0])

    def test_invalid_operator_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_operator(self.x, self.p, self.a, "S")

    def make_model(self, mode="PIPPI", mask_mode="shared", dropout=.5):
        model = OperatorGCN(4, 9, 3, dropout, self.a, mode, mask_mode).double()
        with torch.no_grad():
            model.l1.bias.fill_(.4)
            model.l2.bias.copy_(torch.tensor([.1, -.3, .7]))
        return model

    def test_node_coarse_logits_loss_and_parameter_gradients_with_bias(self):
        self.assertClose(self.p, self.p.T)
        self.assertGreater((self.p.sum(1) - 1).abs().max().item(), .1)
        node = self.make_model()
        coarse = OperatorGCN(4, 9, 3, .5, torch.arange(3), "P").double()
        coarse.load_state_dict(node.state_dict())
        group_mask = torch.tensor([[0, 2, 2, 0, 2, 0, 2, 2, 0],
                                   [2, 0, 2, 2, 0, 2, 0, 2, 0],
                                   [0, 2, 0, 2, 0, 2, 2, 0, 2]], dtype=torch.float64)
        selected = torch.tensor([0, 2, 3, 5])
        labels = torch.tensor([0, 1, 2, 0])
        own = F.one_hot(labels, 3).double()
        counts = torch.zeros(3, 3, dtype=torch.float64)
        counts.index_add_(0, self.a[selected], own)
        sizes = counts.sum(1)
        soft = counts / sizes[:, None]
        output = node(self.x, self.p, group_mask[self.a])
        coarse_output = coarse(self.r @ self.x, self.r @ self.p @ self.h, group_mask)
        self.assertClose(output, self.h @ coarse_output)
        own_loss = per_example_loss(output[selected], own).mean()
        soft_loss = per_example_loss(output[selected], soft[self.a[selected]]).mean()
        coarse_loss = (per_example_loss(coarse_output, soft) * sizes).sum() / sizes.sum()
        self.assertClose(own_loss, soft_loss)
        self.assertClose(own_loss, coarse_loss)
        own_loss.backward()
        coarse_loss.backward()
        for a, b in zip(node.parameters(), coarse.parameters()):
            self.assertClose(a.grad, b.grad)

    def test_coarse_and_node_remain_equivalent_over_adam_steps(self):
        node = self.make_model(dropout=0)
        coarse = OperatorGCN(4, 9, 3, 0, torch.arange(3), "P").double()
        coarse.load_state_dict(node.state_dict())
        optimizers = [torch.optim.Adam(m.parameters(), lr=.01, weight_decay=.0005)
                      for m in (node, coarse)]
        labels = F.one_hot(torch.tensor([0, 1, 1, 2, 0, 2]), 3).double()
        soft = self.r @ labels
        for _ in range(3):
            outputs = [node(self.x, self.p), coarse(self.r @ self.x, self.r @ self.p @ self.h)]
            losses = [per_example_loss(outputs[0], labels).mean(),
                      (per_example_loss(outputs[1], soft) * self.h.sum(0)).sum() / 6]
            self.assertClose(outputs[0], self.h @ outputs[1])
            self.assertClose(losses[0], losses[1])
            for loss, optimizer in zip(losses, optimizers):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for a, b in zip(node.parameters(), coarse.parameters()):
                self.assertClose(a, b)

    def test_assignment_is_nonpersistent_and_parameters_match_old_gcn(self):
        model = self.make_model()
        self.assertIn("assignment", dict(model.named_buffers()))
        self.assertEqual(set(model.state_dict()), {"l1.weight", "l1.bias", "l2.weight", "l2.bias"})

    def test_original_gcn_and_new_p_match_three_adam_steps_for_a_and_b(self):
        here = Path(__file__).resolve().parent
        candidates = (here.parent / "budget_gb_coarsening_20261019" / "v2" / "run_arm.py",
                      here / "legacy" / "run_arm.py")
        source_path = next((path for path in candidates if path.is_file()), None)
        self.assertIsNotNone(source_path, "original or frozen legacy/run_arm.py is required")
        source_tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        original_class = next(node for node in source_tree.body
                              if isinstance(node, ast.ClassDef) and node.name == "GCN")
        isolated_class = ast.Module(body=[original_class], type_ignores=[])
        namespace = {"torch": torch, "nn": torch.nn, "F": F}
        exec(compile(isolated_class, str(source_path), "exec"), namespace)
        original_gcn = namespace["GCN"]
        initial = {name: value.detach().clone() for name, value in
                   self.make_model(mode="P", mask_mode="independent").state_dict().items()}
        selected = torch.tensor([0, 2, 3, 5])
        labels = torch.tensor([0, 1, 2, 0])
        own_targets = F.one_hot(labels, 3).double()
        counts = torch.zeros(3, 3, dtype=torch.float64)
        counts.index_add_(0, self.a[selected], own_targets)
        ball_targets = (counts / counts.sum(1, keepdim=True))[self.a[selected]]
        p = self.p.to_sparse_coo()
        for arm, targets in (("A", own_targets), ("B", ball_targets)):
            with self.subTest(arm=arm):
                old = original_gcn(4, 9, 3, .5).double()
                new = self.make_model(mode="P", mask_mode="independent")
                old.load_state_dict(initial)
                new.load_state_dict(initial)
                optimizers = [torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
                              for model in (old, new)]
                for step in range(3):
                    torch.manual_seed(741 + step)
                    old_logits = old(self.x, p)
                    torch.manual_seed(741 + step)
                    new_logits = new(self.x, p)
                    self.assertClose(old_logits, new_logits, rtol=1e-10, atol=1e-12)
                    old_loss = (F.cross_entropy(old_logits[selected], labels) if arm == "A"
                                else -(targets * old_logits[selected].log_softmax(1)).sum(1).mean())
                    new_loss = per_example_loss(new_logits[selected], targets).mean()
                    self.assertClose(old_loss, new_loss, rtol=1e-10, atol=1e-12)
                    for loss, optimizer in zip((old_loss, new_loss), optimizers):
                        optimizer.zero_grad()
                        loss.backward()
                    for old_parameter, new_parameter in zip(old.parameters(), new.parameters()):
                        self.assertClose(old_parameter.grad, new_parameter.grad, rtol=1e-10, atol=1e-12)
                    for optimizer in optimizers:
                        optimizer.step()
                    for old_parameter, new_parameter in zip(old.parameters(), new.parameters()):
                        self.assertClose(old_parameter, new_parameter, rtol=1e-10, atol=1e-12)

    def test_mixup_joint_propagated_features_and_bias_mass_matches_gcn_logits(self):
        model = self.make_model(mode="P", dropout=0).eval()
        output = model(self.x, self.p)
        hidden = F.relu(self.p @ model.l1(self.x))
        propagated_hidden = self.p @ hidden
        bias_mass = self.p @ torch.ones(6, 1, dtype=torch.float64)
        joint = torch.cat((propagated_hidden, bias_mass), dim=1)
        joint_weight = torch.cat((model.l2.weight.T, model.l2.bias[None, :]), dim=0)
        self.assertClose(joint @ joint_weight, output)
        selected = torch.tensor([0, 2, 3, 5])
        pairing = torch.tensor([3, 2, 0, 1])
        lam = .37
        targets = F.one_hot(torch.tensor([0, 1, 2, 0]), 3).double()
        mixed_joint = lam * joint[selected] + (1 - lam) * joint[selected[pairing]]
        mixed_targets = lam * targets + (1 - lam) * targets[pairing]
        mixed_logits = mixed_joint @ joint_weight
        expected_logits = lam * output[selected] + (1 - lam) * output[selected[pairing]]
        self.assertClose(mixed_logits, expected_logits)
        self.assertClose(mixed_targets.sum(1), torch.ones(4, dtype=torch.float64))
        self.assertGreater((mixed_targets.max(1).values < 1).sum().item(), 0)
        actual_loss = per_example_loss(mixed_logits, mixed_targets).mean()
        expected_loss = -(mixed_targets * expected_logits.log_softmax(1)).sum(1).mean()
        self.assertClose(actual_loss, expected_loss)
        actual_grads = torch.autograd.grad(actual_loss, tuple(model.parameters()), retain_graph=True)
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        for actual, expected in zip(actual_grads, expected_grads):
            self.assertClose(actual, expected)
        # A bias added after propagation silently assumes P1=1 and is a different classifier.
        wrong_logits = F.linear(mixed_joint[:, :-1], model.l2.weight, model.l2.bias)
        self.assertGreater((mixed_logits - wrong_logits).abs().max().item(), .01)

    def test_shared_dropout_matches_native_coarse_rng_and_group_correlation(self):
        model = self.make_model(mode="P")
        model.l1.weight.data.zero_()
        model.l1.bias.data.fill_(1)
        model.l2.weight.data.zero_()
        model.l2.weight.data[:, :3] = torch.eye(3, dtype=torch.float64)
        model.l2.bias.data.zero_()
        torch.manual_seed(71)
        group_mask = F.dropout(torch.ones(3, 9, dtype=torch.float64), .5, True)
        torch.manual_seed(71)
        actual = model(self.x, torch.eye(6, dtype=torch.float64))
        self.assertClose(actual, group_mask[self.a, :3])
        for group in range(3):
            values = actual[self.a == group]
            self.assertClose(values, values[0].expand_as(values))

    def test_independent_dropout_matches_native_rng(self):
        model = self.make_model(mode="P", mask_mode="independent")
        hidden = F.relu(self.p @ model.l1(self.x))
        torch.manual_seed(76)
        expected = self.p @ model.l2(F.dropout(hidden, .5, True))
        torch.manual_seed(76)
        self.assertClose(model(self.x, self.p), expected)

    def test_explicit_mask_is_reproducible_and_eval_disables_all_dropout(self):
        for mask_mode in ("shared", "independent"):
            model = self.make_model(mask_mode=mask_mode)
            mask = torch.randint(0, 2, (6, 9)).double() * 2
            first = model(self.x, self.p, mask)
            torch.manual_seed(999)
            self.assertClose(model(self.x, self.p, mask), first)
            model.eval()
            expected = apply_operator(model.l2(F.relu(apply_operator(model.l1(self.x), self.p, self.a, "PIPPI"))), self.p, self.a, "PIPPI")
            self.assertClose(model(self.x, self.p), expected)
            self.assertClose(model(self.x, self.p, torch.zeros_like(mask)), expected)

    def test_gce_counts_equal_individual_risk_and_reject_wrong_soft_formula(self):
        logits = torch.tensor([[3., -.4, .7], [-1., 2., .4]], dtype=torch.float64, requires_grad=True)
        counts = torch.tensor([[2., 1., 3.], [0., 2., 1.]], dtype=torch.float64)
        soft = counts / counts.sum(1, keepdim=True)
        q = .7
        risk = per_example_loss(logits, soft, "gce", q=q)
        for i in range(2):
            labels = torch.repeat_interleave(torch.arange(3), counts[i].long())
            individual = (1 - logits[i].softmax(0)[labels].pow(q)) / q
            self.assertClose(risk[i], individual.mean())
        wrong = (1 - (soft * logits.softmax(1)).sum(1).pow(q)) / q
        self.assertGreater((risk - wrong).abs().max().item(), .01)
        risk.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_losses_extreme_logits_tiny_q_and_label_smoothing(self):
        logits = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[.2, .3, .5], [0., 1., 0.]], dtype=torch.float64)
        ce = -(targets * logits.log_softmax(1)).sum(1)
        self.assertClose(per_example_loss(logits, targets), ce)
        self.assertClose(per_example_loss(logits, targets, "gce", q=1e-12), ce, rtol=1e-8, atol=1e-8)
        expected = -((.9 * targets + .1 / 3) * logits.log_softmax(1)).sum(1)
        self.assertClose(per_example_loss(logits, targets, "label_smoothing"), expected)
        self.assertTrue(torch.isfinite(per_example_loss(logits, targets, "gce")).all())

    def test_loss_and_model_invalid_options_are_rejected(self):
        logits = torch.zeros(2, 3)
        targets = torch.ones(2, 3) / 3
        for kwargs in ({"kind": "other"}, {"kind": "gce", "q": 0}, {"kind": "gce", "q": 1.1},
                       {"kind": "label_smoothing", "epsilon": -.1}, {"kind": "label_smoothing", "epsilon": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                per_example_loss(logits, targets, **kwargs)
        with self.assertRaises(ValueError):
            per_example_loss(logits, targets[:, :2])
        with self.assertRaises(ValueError):
            self.make_model(mask_mode="unknown")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU unavailable")
    def test_gpu_float32_projection_operator_and_coarse_parameter_gradients(self):
        device = torch.device("cuda")
        a, p, x, pi = [v.to(device) for v in (self.a, self.p.float(), self.x.float(), self.pi.float())]
        z = x.detach().requires_grad_()
        output = apply_operator(z, p.to_sparse(), a, "PIPPI")
        self.assertClose(output, pi @ p @ pi @ z, rtol=2e-5, atol=2e-6)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        node = self.make_model().float().to(device)
        coarse = OperatorGCN(4, 9, 3, .5, torch.arange(3), "P").to(device)
        coarse.load_state_dict(node.state_dict())
        h, r = self.h.float().to(device), self.r.float().to(device)
        torch.manual_seed(31)
        mask = F.dropout(torch.ones(3, 9, device=device), .5, True)
        out_n = node(x, p.to_sparse(), mask[a])
        out_c = coarse(r @ x, (r @ p @ h).to_sparse(), mask)
        self.assertClose(out_n, h @ out_c, rtol=2e-5, atol=2e-6)
        out_n.square().mean().backward()
        ((out_c.square().mean(1) * h.sum(0)).sum() / 6).backward()
        for n, c in zip(node.parameters(), coarse.parameters()):
            self.assertClose(n.grad, c.grad, rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
