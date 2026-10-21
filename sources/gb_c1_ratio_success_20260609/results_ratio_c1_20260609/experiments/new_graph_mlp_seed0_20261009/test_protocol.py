import unittest

import torch

from common import DATASETS, load_data


class ProtocolTests(unittest.TestCase):
    def test_five_historical_protocols_and_hidden_test_labels(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                data, audit = load_data(name, "cpu")
                self.assertTrue((data.y[~data.train_mask] == -1).all())
                self.assertTrue((data.y[data.train_mask] >= 0).all())
                self.assertEqual(len(data.val_y), len(data.val_index))
                self.assertEqual(audit["modified_val_count"], 0)
                self.assertEqual(audit["modified_test_count"], 0)
                self.assertFalse((data.train_mask & data.val_mask).any())
                self.assertFalse((data.train_mask & data.test_mask).any())
                self.assertFalse((data.val_mask & data.test_mask).any())
                self.assertEqual(data.x.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
