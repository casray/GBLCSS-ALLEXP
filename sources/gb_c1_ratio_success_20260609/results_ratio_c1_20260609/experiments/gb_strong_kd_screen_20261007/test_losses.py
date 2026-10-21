import importlib.util
from pathlib import Path
import unittest
import warnings

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def source(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'vendor' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LossTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'losses.py').exists(), 'loss wrappers not implemented')
        import losses
        self.impl = losses
        torch.manual_seed(1907)

    def compare_official(self, device):
        for family, temperature in [('DIST', 1.), ('DIST', 4.), ('DGKD-T', .7), ('DGKD-T', 1.)]:
            a = torch.randn(13, 7, dtype=torch.float64, device=device, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            teacher_logits = torch.randn_like(a, requires_grad=True)
            teacher = teacher_logits.softmax(1)
            actual = self.impl.KDLoss(family, teacher, temperature)(a)
            if family == 'DIST':
                expected = source('dist_official').DIST(tau=temperature)(b, teacher_logits.detach())
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    expected = source('dgkd_official').dgkd_loss(
                        b, teacher_logits.detach(), teacher.argmax(1), 15, 3, temperature)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)
            self.assertIsNone(teacher_logits.grad)

    def test_official_cpu(self):
        self.compare_official('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU unavailable')
    def test_official_gpu(self):
        self.compare_official('cuda')

    def test_plain_ce_and_temperature(self):
        teacher_logits = torch.randn(9, 4, dtype=torch.float64)
        teacher = teacher_logits.softmax(1)
        student = torch.randn_like(teacher, requires_grad=True)
        for temperature in (1., 4.):
            expected = -(F.softmax(teacher_logits / temperature, 1) *
                         F.log_softmax(student / temperature, 1)).sum(1).mean() * temperature**2
            actual = self.impl.KDLoss('KD', teacher, temperature)(student)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_class_permutation(self):
        teacher = torch.randn(8, 5, dtype=torch.float64).softmax(1)
        student = torch.randn_like(teacher)
        order = torch.tensor([3, 0, 4, 1, 2])
        for family in ('KD', 'DIST', 'DGKD-T'):
            a = self.impl.KDLoss(family, teacher, 1)(student)
            b = self.impl.KDLoss(family, teacher[:, order], 1)(student[:, order])
            torch.testing.assert_close(a, b)

    def test_extreme_and_invalid_input(self):
        teacher = torch.tensor([[1., 0., 0.], [.2, .3, .5]])
        for family in ('KD', 'DIST', 'DGKD-T'):
            student = torch.tensor([[10., 0., -10.], [1., -2., 3.]], requires_grad=True)
            value = self.impl.KDLoss(family, teacher, 1)(student)
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertTrue(torch.isfinite(student.grad).all())
        with self.assertRaises(ValueError):
            self.impl.KDLoss('unknown', teacher, 1)
        with self.assertRaises(ValueError):
            self.impl.KDLoss('KD', teacher, 0)
        with self.assertRaises(ValueError):
            self.impl.KDLoss('KD', teacher * 2, 1)


if __name__ == '__main__':
    unittest.main()
