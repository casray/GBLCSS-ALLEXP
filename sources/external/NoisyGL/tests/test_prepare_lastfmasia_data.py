import json
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_lastfmasia_data import build_lastfmasia_arrays, prepare_lastfmasia_data


class PrepareLastFMAsiaDataTest(unittest.TestCase):
    @staticmethod
    def _write_raw_fixture(raw_dir: Path) -> None:
        (raw_dir / "lastfm_asia_edges.csv").write_text(
            "node_1,node_2\n0,1\n1,2\n2,3\n",
            encoding="utf-8",
        )
        (raw_dir / "lastfm_asia_target.csv").write_text(
            "id,target\n2,1\n0,0\n3,1\n1,0\n",
            encoding="utf-8",
        )
        (raw_dir / "lastfm_asia_features.json").write_text(
            json.dumps(
                {
                    "0": [10, 11],
                    "1": [11, 12],
                    "2": [12, 13],
                    "3": [10, 13],
                }
            ),
            encoding="utf-8",
        )

    def test_builds_deterministic_pyg_arrays_from_official_raw_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            self._write_raw_fixture(raw_dir)

            first, first_meta = build_lastfmasia_arrays(
                raw_dir,
                n_components=2,
                random_state=7,
            )
            second, second_meta = build_lastfmasia_arrays(
                raw_dir,
                n_components=2,
                random_state=7,
            )

        self.assertEqual(first["features"].shape, (4, 2))
        self.assertEqual(first["features"].dtype, np.float32)
        np.testing.assert_array_equal(first["target"], np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(
            first["edges"],
            np.array(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                    [2, 3],
                    [3, 2],
                ],
                dtype=np.int64,
            ),
        )
        np.testing.assert_allclose(first["features"], second["features"])
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(first_meta["num_nodes"], 4)
        self.assertEqual(first_meta["num_classes"], 2)
        self.assertEqual(first_meta["num_unique_raw_features"], 4)
        self.assertEqual(first_meta["svd_components"], 2)
        self.assertEqual(first_meta["svd_random_state"], 7)

    def test_writes_pyg_npz_and_label_free_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "fixture"
            fixture_dir.mkdir()
            self._write_raw_fixture(fixture_dir)
            source_zip = tmp_path / "lastfm_asia.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                for path in fixture_dir.iterdir():
                    archive.write(path, f"lasftm_asia/{path.name}")

            npz_path, metadata_path = prepare_lastfmasia_data(
                source_zip,
                tmp_path / "pyg-root",
                n_components=2,
                random_state=7,
                validate_official_counts=False,
            )

            with np.load(npz_path) as data:
                self.assertEqual(set(data.files), {"features", "target", "edges"})
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            actual_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()

        self.assertEqual(metadata["dataset_name"], "LastFMAsia")
        self.assertFalse(
            metadata["reconstruction"]["labels_used_for_feature_construction"]
        )
        self.assertEqual(metadata["npz_sha256"], actual_hash)

    def test_refuses_to_overwrite_raw_data_when_pyg_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "fixture"
            fixture_dir.mkdir()
            self._write_raw_fixture(fixture_dir)
            source_zip = tmp_path / "lastfm_asia.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                for path in fixture_dir.iterdir():
                    archive.write(path, f"lasftm_asia/{path.name}")
            output_root = tmp_path / "pyg-root"
            processed_dir = output_root / "processed"
            processed_dir.mkdir(parents=True)
            (processed_dir / "data.pt").write_bytes(b"stale")

            with self.assertRaisesRegex(RuntimeError, "processed/data.pt"):
                prepare_lastfmasia_data(
                    source_zip,
                    output_root,
                    n_components=2,
                    random_state=7,
                    validate_official_counts=False,
                )


if __name__ == "__main__":
    unittest.main()
