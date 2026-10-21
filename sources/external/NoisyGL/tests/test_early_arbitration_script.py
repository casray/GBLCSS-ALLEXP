import unittest

import torch

from scripts.diagnose_early_arbitration import _prepare_warmup_features


class EarlyArbitrationScriptTest(unittest.TestCase):
    def test_prepare_warmup_features_x_uses_original_features_only(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

        out = _prepare_warmup_features(features, embeddings, "x", torch.device("cpu"))

        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(torch.allclose(out, features))

    def test_prepare_warmup_features_xh_appends_l2_normalized_embeddings(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

        out = _prepare_warmup_features(features, embeddings, "xh", torch.device("cpu"))

        expected = torch.tensor([[1.0, 2.0, 0.6, 0.8], [3.0, 4.0, 0.0, 0.0]])
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
