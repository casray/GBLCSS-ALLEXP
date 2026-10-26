from pathlib import Path
import sys
import unittest
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StaticFeatureRepair(unittest.TestCase):
    def test_static_observed_class_frequency_is_scaled(self):
        from targets import evidence
        from static_repair import gate_features
        z = torch.tensor([0, 0, 0, 0, 1, 2])
        e = evidence(z, torch.zeros(6).long(), 3)
        p = torch.full((6, 3), 1/3)
        f = gate_features(e, p)
        self.assertLess(abs(float(f[:, 8].mean())), 1e-6)
        self.assertAlmostEqual(float(f[:, 8].std(unbiased=False)), 1., places=6)
        self.assertFalse(f.requires_grad)
        self.assertEqual(tuple(f.shape), (6, 9))
        self.assertTrue(bool(((f[:, 5:8] >= 0) & (f[:, 5:8] <= 1)).all()))


if __name__ == '__main__':
    unittest.main()
