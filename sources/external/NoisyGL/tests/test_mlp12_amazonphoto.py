import unittest

import numpy as np

from utils.dataloader import Dataset
from utils.labelnoise import label_process
from utils.tools import load_conf


METHODS = [
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_fine",
    "mlp_dividemix_lite",
    "mlp_ct",
    "mlp_logitclip",
    "mlp_anl",
]


class MLP12AmazonPhotoTests(unittest.TestCase):
    def test_amazonphoto_configs_load_and_match_coauthor_backbone(self):
        for method in METHODS:
            with self.subTest(method=method):
                photo = load_conf(None, method, "amazonpho")
                coauthor = load_conf(None, method, "coauthorcs")
                self.assertEqual(photo.model["method"], method)
                for key in ["n_hidden", "n_layer", "dropout", "input_layer", "output_layer"]:
                    self.assertEqual(photo.model.get(key), coauthor.model.get(key))
                self.assertEqual(photo.training["n_epochs"], coauthor.training["n_epochs"])
                self.assertEqual(photo.training["lr"], coauthor.training["lr"])
                if "gb" in photo.__dict__:
                    self.assertIn("AmazonPhoto", photo.gb["embedding_path"])
                    self.assertAlmostEqual(float(photo.gb["target_beta"]), float(coauthor.gb["target_beta"]))
                    self.assertEqual(photo.gb["max_balls"], coauthor.gb["max_balls"])
                    self.assertEqual(photo.gb["split_improvement"], coauthor.gb["split_improvement"])

    def test_amazonphoto_percent_split_and_train_only_noise(self):
        data = Dataset(
            "amazonpho",
            feat_norm=False,
            adj_norm=False,
            verbose=False,
            split_type="percent",
            train_percent=0.2,
            val_percent=0.1,
            test_percent=0.7,
            add_self_loop=False,
            device="cpu",
        )
        self.assertEqual(data.n_nodes, 7650)
        self.assertEqual(len(data.train_masks), 1530)
        self.assertEqual(len(data.val_masks), 765)
        self.assertEqual(len(data.test_masks), 5355)

        noisy, modified = label_process(
            data.labels,
            data.feats,
            data.n_classes,
            noise_type="uniform",
            noise_rate=0.4,
            random_seed=0,
            eligible_mask=data.train_masks,
            debug=False,
        )
        modified = set(np.asarray(modified, dtype=np.int64).tolist())
        self.assertTrue(modified)
        self.assertTrue(modified.issubset(set(np.asarray(data.train_masks, dtype=np.int64).tolist())))
        self.assertFalse(modified.intersection(set(np.asarray(data.val_masks, dtype=np.int64).tolist())))
        self.assertFalse(modified.intersection(set(np.asarray(data.test_masks, dtype=np.int64).tolist())))


if __name__ == "__main__":
    unittest.main()
