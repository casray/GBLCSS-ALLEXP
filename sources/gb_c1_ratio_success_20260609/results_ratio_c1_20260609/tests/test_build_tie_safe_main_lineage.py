import csv
import hashlib
import importlib.util
import json
import statistics
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "scripts"
    / "build_tie_safe_main_lineage.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("tie_safe_lineage", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root):
    datasets = ("cora", "citeseer", "coauthorcs", "corafull", "amazonpho")
    methods = (
        "frozengrace_gb_mlp_softdist_beta05",
        "frozengrace_gb_mlp_softdist_beta05_shuffled",
        "mlp_ce",
        "mlp_jocor",
    )
    softdist = set(methods[:2])
    raw_rows = []
    aggregate_rows = []
    protocol_rows = []
    for dataset_index, dataset in enumerate(datasets):
        for seed in range(5):
            protocol_rows.append(
                {
                    "dataset": dataset,
                    "regime": "uniform0.4",
                    "seed": seed,
                    "protocol_status": "PASS",
                    "error": "",
                }
            )
        for method_index, method in enumerate(methods):
            values = [
                0.50
                + 0.01 * dataset_index
                + 0.001 * method_index
                + 0.0001 * seed
                for seed in range(5)
            ]
            for seed, value in enumerate(values):
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "regime": "uniform0.4",
                        "method": method,
                        "seed": seed,
                        "status": "ok",
                        "test": value,
                        "tie_mode": (
                            "distribution_only" if method in softdist else ""
                        ),
                    }
                )
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "regime": "uniform0.4",
                    "method": method,
                    "n": 5,
                    "seeds": "0,1,2,3,4",
                    "test_mean": statistics.mean(values),
                    "test_sample_std": statistics.stdev(values),
                }
            )
    raw = root / "raw_combined.csv"
    aggregate = root / "aggregate.csv"
    protocol = root / "protocol_audit.csv"
    environment = root / "environment_fingerprint.json"
    output = root / "tie_safe_main_table_lineage.csv"
    write_csv(raw, raw_rows)
    write_csv(aggregate, aggregate_rows)
    write_csv(protocol, protocol_rows)
    environment.write_text(
        json.dumps({"git_commit": "abc123", "python": "3.x"}) + "\n",
        encoding="utf-8",
    )
    return {
        "raw": raw,
        "aggregate": aggregate,
        "protocol": protocol,
        "environment": environment,
        "output": output,
    }


class TieSafeMainLineageTest(unittest.TestCase):
    def test_builds_20_closed_cells_with_real_hashes_and_minimum_schema(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            rows = module.build_lineage(
                raw_path=paths["raw"],
                aggregate_path=paths["aggregate"],
                protocol_path=paths["protocol"],
                environment_path=paths["environment"],
                output_path=paths["output"],
            )
            self.assertEqual(len(rows), 20)
            self.assertTrue(paths["output"].exists())
            self.assertEqual({row["lineage_status"] for row in rows}, {"CLOSED"})
            self.assertEqual({row["n"] for row in rows}, {5})
            self.assertEqual({row["seeds"] for row in rows}, {"0,1,2,3,4"})
            required = {
                "generation",
                "generation_id",
                "version",
                "generator_version",
                "table",
                "manuscript_table_id",
                "cell",
                "cell_id",
                "claim",
                "claim_id",
                "supersedes",
                "dataset",
                "regime",
                "method",
                "tie_mode",
                "seeds",
                "n",
                "metric",
                "unit",
                "displayed_value",
                "computed_mean",
                "computed_sample_std",
                "raw_path",
                "raw_sha256",
                "raw_filter_expression",
                "raw_aggregation_formula",
                "raw_status",
                "aggregate_path",
                "aggregate_sha256",
                "aggregate_filter_expression",
                "aggregate_formula",
                "aggregate_status",
                "protocol_path",
                "protocol_sha256",
                "protocol_filter_expression",
                "protocol_formula",
                "protocol_status",
                "environment_path",
                "environment_sha256",
                "environment_filter_expression",
                "environment_formula",
                "environment_status",
                "lineage_status",
                "boundary",
                "claim_boundary",
            }
            self.assertTrue(required.issubset(rows[0]))
            expected_raw_hash = hashlib.sha256(paths["raw"].read_bytes()).hexdigest().upper()
            self.assertEqual({row["raw_sha256"] for row in rows}, {expected_raw_hash})
            soft_rows = [
                row for row in rows if row["method"].startswith("frozengrace_")
            ]
            baseline_rows = [
                row for row in rows if row["method"] in {"mlp_ce", "mlp_jocor"}
            ]
            self.assertEqual(
                {row["tie_mode"] for row in soft_rows},
                {"distribution_only"},
            )
            self.assertEqual(
                {row["tie_mode"] for row in baseline_rows},
                {"NOT_APPLICABLE"},
            )
            first = rows[0]
            self.assertEqual(first["displayed_value"], "50.02±0.02")
            self.assertAlmostEqual(float(first["computed_mean"]), 0.5002)
            self.assertAlmostEqual(
                float(first["computed_sample_std"]),
                statistics.stdev([0.5, 0.5001, 0.5002, 0.5003, 0.5004]),
            )

    def test_rejects_missing_or_duplicate_seed_rows(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            with paths["raw"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows.pop()
            write_csv(paths["raw"], rows)
            with self.assertRaisesRegex(ValueError, "exactly five unique seeds"):
                module.build_lineage(
                    raw_path=paths["raw"],
                    aggregate_path=paths["aggregate"],
                    protocol_path=paths["protocol"],
                    environment_path=paths["environment"],
                    output_path=paths["output"],
                )

    def test_rejects_aggregate_mean_or_sample_std_drift(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            with paths["aggregate"].open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["test_mean"] = str(float(rows[0]["test_mean"]) + 1e-6)
            write_csv(paths["aggregate"], rows)
            with self.assertRaisesRegex(ValueError, "aggregate mismatch"):
                module.build_lineage(
                    raw_path=paths["raw"],
                    aggregate_path=paths["aggregate"],
                    protocol_path=paths["protocol"],
                    environment_path=paths["environment"],
                    output_path=paths["output"],
                )

    def test_rejects_wrong_softdist_tie_mode(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            with paths["raw"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["tie_mode"] = "min_index"
            write_csv(paths["raw"], rows)
            with self.assertRaisesRegex(ValueError, "tie_mode"):
                module.build_lineage(
                    raw_path=paths["raw"],
                    aggregate_path=paths["aggregate"],
                    protocol_path=paths["protocol"],
                    environment_path=paths["environment"],
                    output_path=paths["output"],
                )

    def test_rejects_non_pass_protocol_row(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            with paths["protocol"].open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["protocol_status"] = "FAIL"
            write_csv(paths["protocol"], rows)
            with self.assertRaisesRegex(ValueError, "protocol"):
                module.build_lineage(
                    raw_path=paths["raw"],
                    aggregate_path=paths["aggregate"],
                    protocol_path=paths["protocol"],
                    environment_path=paths["environment"],
                    output_path=paths["output"],
                )

    def test_cli_accepts_four_inputs_and_output(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_fixture(Path(tmp))
            rc = module.main(
                [
                    "--raw",
                    str(paths["raw"]),
                    "--aggregate",
                    str(paths["aggregate"]),
                    "--protocol-audit",
                    str(paths["protocol"]),
                    "--environment",
                    str(paths["environment"]),
                    "--output",
                    str(paths["output"]),
                ]
            )
            self.assertEqual(rc, 0)
            with paths["output"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 20)


if __name__ == "__main__":
    unittest.main()
