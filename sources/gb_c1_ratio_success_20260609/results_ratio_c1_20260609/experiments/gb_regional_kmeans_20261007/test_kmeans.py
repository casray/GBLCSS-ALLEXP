import unittest
import torch
import run_kmeans as run


class RegionalControlTests(unittest.TestCase):
    def test_partition_relabeling_and_explicit_means(self):
        torch.manual_seed(0)
        teacher = torch.randn(8, 3, dtype=torch.float64).softmax(1)
        logits = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
        probabilities = logits.softmax(1)
        actual = run.stable.RegionalKL(teacher, groups)(probabilities)
        expected = logits.new_zeros(())
        for j in range(3):
            mask = groups == j
            a, b = teacher[mask].mean(0), probabilities[mask].mean(0)
            expected = expected + mask.double().mean() * (a * (a.log()-b.log())).sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits, retain_graph=True)[0])
        permuted = torch.tensor([2, 0, 1])[groups]
        torch.testing.assert_close(actual, run.stable.RegionalKL(teacher, permuted)(probabilities))


if __name__ == '__main__':
    unittest.main()
