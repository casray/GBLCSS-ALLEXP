import tempfile
import unittest
from pathlib import Path

import torch

from scripts.linear_probe_embeddings import embedding_path_for, train_linear_probe


class LinearProbeEmbeddingsTest(unittest.TestCase):
    def test_embedding_path_for_uses_dataset_method_hpt_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "Cora" / "grace" / "H.pt"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"placeholder")

            path = embedding_path_for(root, "Cora", "grace")

            self.assertEqual(path, expected)

    def test_embedding_path_for_grace_falls_back_to_ratio_pretrain0_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "Cora" / "ratio_train0p2_val0p1_split0_pretrain0" / "H.pt"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"placeholder")

            path = embedding_path_for(root, "cora", "grace")

            self.assertEqual(path, expected)

    def test_train_linear_probe_learns_linearly_separable_embeddings(self):
        embeddings = torch.tensor([
            [2.0, 0.0],
            [1.8, 0.1],
            [0.0, 2.0],
            [0.1, 1.8],
            [2.2, 0.0],
            [0.0, 2.2],
        ])
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        train_mask = torch.tensor([True, True, True, True, False, False])
        val_mask = torch.tensor([False, True, False, True, False, False])
        test_mask = torch.tensor([False, False, False, False, True, True])

        result = train_linear_probe(
            embeddings=embeddings,
            train_labels=labels,
            eval_labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            n_classes=2,
            seed=0,
            epochs=200,
            lr=0.1,
            weight_decay=0.0,
            patience=50,
            device=torch.device("cpu"),
        )

        self.assertGreaterEqual(result["test_acc_at_best_val"], 0.99)
        self.assertGreaterEqual(result["best_val_acc"], 0.99)
        self.assertGreater(result["best_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
