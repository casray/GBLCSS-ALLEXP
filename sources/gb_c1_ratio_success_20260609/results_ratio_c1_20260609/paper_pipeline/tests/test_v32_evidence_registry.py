import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_v32_evidence_registry.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_v32_evidence_registry_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tie_safe_rows(tie_mode="distribution_only"):
    rows = []
    methods = (
        "frozengrace_gb_mlp_softdist_beta05",
        "frozengrace_gb_mlp_softdist_beta05_shuffled",
        "mlp_ce",
        "mlp_jocor",
    )
    for method in methods:
        rows.append(
            {
                "dataset": "cora",
                "regime": "uniform0.4",
                "method": method,
                "seed": "0",
                "noise_scope": "train",
                "tie_mode": (
                    tie_mode if method.startswith("frozengrace_gb_") else ""
                ),
                "status": "ok",
            }
        )
    return rows


def nearest_rows():
    return [
        {
            "dataset": "cora",
            "regime": "uniform0.4",
            "method": method,
            "seed": "0",
            "noise_scope": "train",
            "status": "ok",
        }
        for method in (
            "frozengrace_gb_mlp_softdist_beta05",
            "gsdn_faithful_core",
            "postel_mlp_noipl_rm",
            "als_mlp_rm",
            "sals_mlp_adapted",
        )
    ]


def old_gsdn_rows():
    return [
        {
            "dataset": "cora",
            "method": "gsdn",
            "seed": "0",
            "noise_type": "uniform",
            "noise_rate": "0.4",
            "status": "ok",
        }
    ]


class V32EvidenceRegistryTest(unittest.TestCase):
    def test_rejects_legacy_tie_mode_from_current_evidence(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tie_path = root / "tie.csv"
            nearest_path = root / "nearest.csv"
            write_csv(tie_path, tie_safe_rows(tie_mode="min_index"))
            write_csv(nearest_path, nearest_rows())

            with self.assertRaisesRegex(
                ValueError,
                "legacy tie_mode cannot enter current evidence registry",
            ):
                builder.build_registry(
                    tie_safe_path=tie_path,
                    nearest_path=nearest_path,
                )

    def test_separates_current_recent_work_from_provisional_gsdn(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tie_path = root / "tie.csv"
            nearest_path = root / "nearest.csv"
            old_gsdn_path = root / "old_gsdn.csv"
            write_csv(tie_path, tie_safe_rows())
            write_csv(nearest_path, nearest_rows())
            write_csv(old_gsdn_path, old_gsdn_rows())

            rows = builder.build_registry(
                tie_safe_path=tie_path,
                nearest_path=nearest_path,
                old_gsdn_paths=(old_gsdn_path,),
            )

            by_method = {row["method"]: row for row in rows}
            self.assertEqual(
                by_method["postel_mlp_noipl_rm"]["headline_allowed"],
                "1",
            )
            self.assertEqual(
                by_method["gsdn_faithful_core"]["headline_allowed"],
                "0",
            )
            self.assertIn(
                "pending user reimplementation",
                by_method["gsdn_faithful_core"]["limitations"],
            )
            self.assertEqual(by_method["gsdn"]["headline_allowed"], "0")
            self.assertEqual(
                by_method["gsdn"]["implementation_status"],
                "provisional adapted implementation",
            )

    def test_real_sources_have_expected_current_row_counts(self):
        builder = load_builder()
        rows = builder.build_registry(
            tie_safe_path=builder.DEFAULT_TIE_SAFE_PATH,
            nearest_path=builder.DEFAULT_NEAREST_PATH,
            old_gsdn_paths=builder.DEFAULT_OLD_GSDN_PATHS,
        )
        by_method = {row["method"]: row for row in rows}

        for method in (
            "frozengrace_gb_mlp_softdist_beta05",
            "frozengrace_gb_mlp_softdist_beta05_shuffled",
            "mlp_ce",
            "mlp_jocor",
        ):
            self.assertEqual(by_method[method]["row_count"], "125")

        for method in (
            "postel_mlp_noipl_rm",
            "als_mlp_rm",
            "sals_mlp_adapted",
            "gsdn_faithful_core",
        ):
            self.assertEqual(by_method[method]["row_count"], "125")


if __name__ == "__main__":
    unittest.main()
