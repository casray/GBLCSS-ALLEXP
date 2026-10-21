import unittest
import inspect
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import torch
import yaml
from torch.nn import functional as F

from adapters import (scatter, semantic_labels_gpu, fit_gpu_logistic,
                      probe_probabilities, erase_classes, normalized_adjacency)
from run_screen import load_data, DATASETS, train_tfr


class AdapterTests(unittest.TestCase):
    def test_tfr_graph_loop_configuration(self):
        root = Path(__file__).resolve().parent
        config = yaml.safe_load((root / "vendor/tfr/config/_dataset/dblp.yaml").read_text())
        self.assertFalse(config["modify"]["add_self_loop"])
        source = inspect.getsource(train_tfr)
        self.assertIn("loop_edges = data.edge_index", source)
        self.assertNotIn("add_self_loops(", source)

    def test_protocol_hashes_and_clean_validation(self):
        for name in DATASETS:
            for rate in (.4, .6):
                _, audit = load_data(name, rate)
                self.assertEqual(audit["modified_val_count"], 0)

    def test_scatter_mean(self):
        src = torch.tensor([[2., 4.], [4., 8.], [9., 3.]], device="cuda")
        out = scatter(src, torch.tensor([0, 0, 2], device="cuda"), 0, 4, "mean")
        torch.testing.assert_close(out.cpu(), torch.tensor([[3., 6.], [0., 0.], [9., 3.], [0., 0.]]))

    def test_semantics_against_author_numpy(self):
        torch.manual_seed(0)
        x = torch.randn(20, 6, device="cuda")
        mask = torch.arange(20, device="cuda") < 12
        labels = torch.arange(20, device="cuda") % 3
        y = F.one_hot(labels, 3).float()
        y[~mask] = 0
        prototype = np.vstack([x[mask][labels[mask] == i].cpu().numpy().mean(0) for i in range(3)])
        cos = torch.tensor(np.abs(cosine_similarity(x.cpu().numpy(), prototype)), device="cuda")
        expected = F.softmax(.6 * y + .4 * cos, 1)
        torch.testing.assert_close(semantic_labels_gpu(x[mask], y, x, mask, .6, 3), expected, atol=2e-6, rtol=2e-6)

    def test_probe_matches_sklearn(self):
        torch.manual_seed(2)
        x = torch.randn(120, 8, device="cuda")
        y = (x[:, :3].argmax(1) + 1).long()
        probe = fit_gpu_logistic(x, y)
        actual = probe_probabilities(x, probe, 4).cpu().numpy()[:, 1:]
        normalized = F.normalize(x, dim=1).cpu().numpy()
        clf = LogisticRegression(max_iter=1000, tol=1e-4).fit(normalized, y.cpu().numpy())
        expected = clf.predict_proba(normalized)
        self.assertLess(np.max(np.abs(actual - expected)), .003)
        self.assertTrue(np.array_equal(actual.argmax(1), expected.argmax(1)))

    def test_sparse_column_normalization(self):
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device="cuda")
        actual = normalized_adjacency(edges, 3).to_dense()
        expected = torch.tensor([[0, .5, 0], [1, 0, 1], [0, .5, 0]], device="cuda")
        torch.testing.assert_close(actual, expected)

    def test_compression_formula_matches_dense_memberships(self):
        _, Loss = erase_classes()
        objective = Loss(1, 2, .05, corafull=True)
        torch.manual_seed(0)
        w = torch.randn(5, 18, device="cuda", requires_grad=True)
        labels = torch.arange(18, device="cuda") % 3
        membership = torch.stack([torch.diag((labels == i).float()) for i in range(3)])
        dense = objective.compute_compress_loss_empirical(w, membership)
        masked = objective.compute_compress_loss_empirical_manyclass(w, labels)
        torch.testing.assert_close(dense, masked, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(dense, w)[0], torch.autograd.grad(masked, w)[0], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
