import unittest
import torch
from simmlp_adapter import SparseConv, author_objects, effective_config


class SimMLPTests(unittest.TestCase):
    def test_aggregation_and_gradient(self):
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        conv = SparseConv(4, 4)
        actual = conv(x, edges, None)
        a = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]], dtype=torch.double)
        inv = a.sum(1).rsqrt()
        reference = (inv[:, None] * a * inv[None, :]) @ x
        torch.testing.assert_close(actual, reference)
        grad = torch.autograd.grad(actual.square().sum(), x)[0]
        expected = torch.autograd.grad(reference.square().sum(), x)[0]
        torch.testing.assert_close(grad, expected)

    def test_official_core(self):
        obj = author_objects()
        mlp = obj['MLP'](5, 4, 4, torch.nn.PReLU, 2)
        z = mlp(torch.randn(6, 5))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(6))
        loss = obj['Bootstrap'](eta=1)(z, z.detach())
        self.assertLess(abs(loss.item()), 1e-6)
        self.assertEqual(obj['LogisticRegression'](4, 3)(z).shape, (6, 3))

    def test_fixed_protocol_budget(self):
        for name in ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho']:
            cfg = effective_config(name, False)
            self.assertEqual(cfg['epochs'], 200)
            self.assertEqual(cfg['lr_epochs'], 200)
            self.assertEqual(cfg['hidden_dim'], 64)
            self.assertEqual(cfg['pretrain_selection'], 'fixed final epoch; no label-based selection')
            self.assertEqual(cfg['probe_selection'], 'clean validation accuracy; last tie')
        self.assertEqual(effective_config('corafull', False)['author_preset'], 'cora')


if __name__ == '__main__':
    unittest.main()
