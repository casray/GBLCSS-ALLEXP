"""CPU contracts for the robustness worker's native CE operator and diagnostics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('robust_extension_worker', HERE / 'run_job.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class DoseContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(129)
        self.assignment = torch.tensor([0, 0, 1, 2, 2])
        self.x = torch.randn(5, 3, dtype=torch.double)
        # Deliberately irregular rows: a bias before P cannot be moved after P.
        self.p = torch.tensor([[.5, .2, 0, 0, 0], [.2, .8, .1, 0, 0],
                               [0, .1, .4, .3, 0], [0, 0, .3, .6, .2],
                               [0, 0, 0, .2, .9]], dtype=torch.double)
        self.pi = torch.tensor([[.5, .5, 0, 0, 0], [.5, .5, 0, 0, 0],
                                [0, 0, 1, 0, 0], [0, 0, 0, .5, .5],
                                [0, 0, 0, .5, .5]], dtype=torch.double)

    def test_each_layer_dose_matches_dense_endpoints_and_midpoint_gradients(self):
        for rho in (0., .5, 1.):
            with self.subTest(rho=rho):
                model = worker.DoseGCN(3, 4, 2, .5, self.assignment, rho=rho).double().eval()
                x = self.x.clone().requires_grad_()
                actual = model(x, self.p)
                operator = (1-rho)*self.p + rho*(self.pi @ self.p @ self.pi)
                expected = operator @ model.l2(F.relu(operator @ model.l1(x)))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                args = (x, *model.parameters())
                grad_actual = torch.autograd.grad(actual.square().sum(), args, retain_graph=True)
                grad_expected = torch.autograd.grad(expected.square().sum(), args)
                for a, b in zip(grad_actual, grad_expected):
                    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_rho_zero_matches_native_initialization_and_dropout_rng(self):
        torch.manual_seed(54)
        native = worker.OperatorGCN(3, 4, 2, .5, self.assignment).double().train()
        torch.manual_seed(54)
        dose = worker.DoseGCN(3, 4, 2, .5, self.assignment, rho=0).double().train()
        for a, b in zip(native.parameters(), dose.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.manual_seed(99)
        a = native(self.x, self.p)
        state = torch.random.get_rng_state()
        torch.manual_seed(99)
        b = dose(self.x, self.p)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_original_bypasses_dose_and_training_dropout(self):
        model = worker.DoseGCN(3, 4, 2, 1., self.assignment, rho=1,
                               mask_mode='shared').double().train()
        expected = self.p @ model.l2(F.relu(self.p @ model.l1(self.x)))
        torch.testing.assert_close(model(self.x, self.p, original=True), expected)

    def test_trace_clean_empty_is_none_and_only_train_nodes_count(self):
        clean = np.array([0, 1, 2, 0, 1])
        ids = np.array([0, 2, 4])
        pred = np.array([0, 2, 1, 1, 1])
        clean_metrics = worker.trace_metrics(pred, clean, clean, ids)
        self.assertIsNone(clean_metrics['fit_wrong_label'])
        self.assertIsNone(clean_metrics['recover_true_label'])
        self.assertEqual(clean_metrics['flipped_count'], 0)
        self.assertEqual(clean_metrics['unflipped_count'], 3)
        noisy = np.array([2, 0, 1, 2, 1])
        metrics = worker.trace_metrics(pred, clean, noisy, ids)
        self.assertEqual(metrics['flipped_count'], 2)
        self.assertEqual(metrics['fit_wrong_label'], .5)
        self.assertEqual(metrics['recover_true_label'], .5)
        self.assertEqual(metrics['unflipped_accuracy'], 1.)

    def test_trace_preserves_rng_and_model_mode(self):
        model = worker.DoseGCN(3, 4, 2, .5, self.assignment, rho=.5,
                               mask_mode='shared').double().train()
        before = torch.random.get_rng_state()
        rows = worker.collect_trace(model, self.x, self.p, np.zeros(5, dtype=int),
                                    np.zeros(5, dtype=int), np.array([0, 2]), 5)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertEqual([r['view'] for r in rows], ['P', 'train'])
        self.assertTrue(all(r['epoch'] == 5 for r in rows))


if __name__ == '__main__':
    unittest.main()
