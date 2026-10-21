import unittest

import torch
from torch.nn import functional as F

from regional_loss import RegionalKL, residual_components


class RegionalLossTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.tensor([[.8, .2], [.6, .4], [.1, .9]], dtype=torch.float64)
        self.student = torch.tensor([[.4, .6], [.7, .3], [.3, .7]], dtype=torch.float64)

    def test_size_weighting_matches_explicit_two_group_calculation(self):
        actual = RegionalKL(self.teacher, torch.tensor([5, 5, 12]))(self.student)
        teacher = torch.stack([self.teacher[:2].mean(0), self.teacher[2]])
        student = torch.stack([self.student[:2].mean(0), self.student[2]])
        expected = (F.kl_div(student.log(), teacher, reduction='none').sum(1)
                    * torch.tensor([2/3, 1/3], dtype=torch.float64)).sum()
        torch.testing.assert_close(actual, expected)

    def test_singletons_equal_pointwise_kl_and_global_is_mean_kl(self):
        singleton = RegionalKL(self.teacher, torch.arange(3))(self.student)
        pointwise = F.kl_div(self.student.log(), self.teacher, reduction='batchmean')
        global_loss = RegionalKL(self.teacher, torch.zeros(3, dtype=torch.long))(self.student)
        expected = F.kl_div(self.student.mean(0).log(), self.teacher.mean(0), reduction='sum')
        torch.testing.assert_close(singleton, pointwise)
        torch.testing.assert_close(global_loss, expected)
        self.assertLessEqual(float(global_loss), float(pointwise) + 1e-12)

    def test_teacher_is_detached_and_student_gradient_matches_loop(self):
        teacher = self.teacher.clone().requires_grad_()
        logits = self.student.log().clone().requires_grad_()
        loss = RegionalKL(teacher, torch.tensor([0, 0, 1]))(logits.softmax(1))
        gradient, = torch.autograd.grad(loss, logits)
        reference_logits = logits.detach().clone().requires_grad_()
        probabilities = reference_logits.softmax(1)
        reference = (2/3 * F.kl_div(probabilities[:2].mean(0).log(),
                                   teacher[:2].detach().mean(0), reduction='sum')
                     + 1/3 * F.kl_div(probabilities[2].log(), teacher[2].detach(), reduction='sum'))
        reference_gradient, = torch.autograd.grad(reference, reference_logits)
        torch.testing.assert_close(gradient, reference_gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertIsNone(teacher.grad)

    def test_permutation_invariance_and_residual_decomposition(self):
        groups = torch.tensor([0, 0, 1])
        order = torch.tensor([2, 0, 1])
        a = RegionalKL(self.teacher, groups)(self.student)
        b = RegionalKL(self.teacher[order], groups[order])(self.student[order])
        torch.testing.assert_close(a, b)
        values = residual_components(self.student, self.teacher, groups)
        self.assertLess(values['decomposition_error'], 1e-12)

    def test_invalid_inputs_and_zero_teacher_probabilities(self):
        with self.assertRaises(ValueError):
            RegionalKL(self.teacher, torch.tensor([0, -1, 1]))
        with self.assertRaises(ValueError):
            RegionalKL(self.teacher * 2, torch.arange(3))
        hard_teacher = torch.tensor([[1., 0.], [0., 1.]])
        loss = RegionalKL(hard_teacher, torch.arange(2))(torch.full((2, 2), .5))
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
