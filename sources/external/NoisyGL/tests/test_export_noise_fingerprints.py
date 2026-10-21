import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TinyProtocolData:
    def __init__(self):
        self.n_classes = 3
        self.feats = torch.eye(9)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.train_masks = torch.tensor([0, 1, 2])
        self.val_masks = torch.tensor([3, 4, 5])
        self.test_masks = torch.tensor([6, 7, 8])


class ExportNoiseFingerprintsTest(unittest.TestCase):
    def test_script_is_directly_executable_from_repo_root(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "export_noise_fingerprints.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(script.parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_train_only_manifest_has_stable_subset_hashes(self):
        from scripts import export_noise_fingerprints as exporter

        with patch.object(
            exporter,
            "_load_dataset",
            return_value=TinyProtocolData(),
        ):
            rows = exporter.build_rows(
                datasets=("cora",),
                seeds=(7,),
                device="cpu",
                noise_type="uniform",
                noise_rate=1.0,
                split_type="percent",
                train_percent=0.2,
                val_percent=0.1,
                test_percent=0.7,
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["noise_scope"], "train")
        self.assertEqual(row["modified_train_count"], 3)
        self.assertEqual(row["modified_val_count"], 0)
        self.assertEqual(row["modified_test_count"], 0)
        self.assertTrue(row["train_noise_hash"])
        self.assertTrue(row["val_noise_hash"])
        self.assertTrue(row["noise_hash"])

    def test_writer_round_trips_manifest(self):
        from scripts import export_noise_fingerprints as exporter

        rows = [{
            "dataset": "cora",
            "seed": 0,
            "noise_scope": "train",
            "split_hash": "split",
            "noise_hash": "noise",
            "train_noise_hash": "train",
            "val_noise_hash": "val",
            "modified_count": 1,
            "modified_train_count": 1,
            "modified_val_count": 0,
            "modified_test_count": 0,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            exporter.write_csv(path, rows)
            loaded = exporter.read_csv(path)
        self.assertEqual(loaded[0]["train_noise_hash"], "train")


if __name__ == "__main__":
    unittest.main()
