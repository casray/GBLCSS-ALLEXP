import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_tie_safe_full_grid.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_tie_safe_full_grid_lineage_test",
    RUNNER_PATH,
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class TieSafeFullGridLineageTest(unittest.TestCase):
    def test_existing_input_snapshots_are_verified_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "source_a.txt"
            source_b = root / "source_b.txt"
            source_a.write_text("alpha", encoding="utf-8")
            source_b.write_text("beta", encoding="utf-8")
            output_dir = root / "run"

            status = RUNNER._write_or_verify_input_snapshots(
                [source_a, source_b],
                output_dir,
            )
            manifest_path = output_dir / "input_snapshot_manifest.csv"
            snapshot_paths = sorted((output_dir / "input_snapshots").iterdir())
            original_manifest = manifest_path.read_bytes()
            original_snapshots = {
                path.name: path.read_bytes() for path in snapshot_paths
            }
            original_mtimes = {
                path.name: path.stat().st_mtime_ns for path in snapshot_paths
            }

            second_status = RUNNER._write_or_verify_input_snapshots(
                [source_a, source_b],
                output_dir,
            )

            self.assertEqual(status, "created")
            self.assertEqual(second_status, "verified_existing")
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in snapshot_paths
                },
                original_snapshots,
            )
            self.assertEqual(
                {
                    path.name: path.stat().st_mtime_ns
                    for path in snapshot_paths
                },
                original_mtimes,
            )

    def test_changed_source_aborts_resume_without_overwriting_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            output_dir = root / "run"
            RUNNER._write_or_verify_input_snapshots([source], output_dir)
            snapshot = next((output_dir / "input_snapshots").iterdir())
            original_snapshot = snapshot.read_bytes()

            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "input snapshot mismatch",
            ):
                RUNNER._write_or_verify_input_snapshots([source], output_dir)

            self.assertEqual(snapshot.read_bytes(), original_snapshot)

    def test_existing_environment_fingerprint_is_preserved_on_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            initial_path = output_dir / "environment_fingerprint.json"
            initial_path.write_text(
                json.dumps({"captured_at_utc": "initial"}),
                encoding="utf-8",
            )
            original = initial_path.read_bytes()

            with patch.object(
                RUNNER,
                "_collect_environment_fingerprint",
                return_value={"captured_at_utc": "resume"},
            ):
                written = RUNNER._write_environment_fingerprint(
                    Path("python"),
                    Path("noisygl"),
                    output_dir,
                )

            self.assertEqual(initial_path.read_bytes(), original)
            self.assertNotEqual(written, initial_path)
            self.assertTrue(written.name.startswith("environment_fingerprint.resume_"))
            self.assertEqual(
                json.loads(written.read_text(encoding="utf-8"))[
                    "captured_at_utc"
                ],
                "resume",
            )

    def test_manifest_records_exact_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            output_dir = root / "run"

            RUNNER._write_or_verify_input_snapshots([source], output_dir)

            with (output_dir / "input_snapshot_manifest.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sha256"], RUNNER.sha256_file(source))


if __name__ == "__main__":
    unittest.main()
