import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mlp_weighting_revalidate import (
    BASE,
    ELLIPSOID,
    ELLIPSOID_SHUFFLED,
    GMM_RESP,
    aggregate,
    decide,
    load_rows,
    paired_deltas,
)


class TestMLPWeightingRevalidateSummary(unittest.TestCase):
    def test_summary_computes_seed0_weighting_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.csv"
            rows = [
                ("cora", BASE, 0.55),
                ("cora", ELLIPSOID, 0.57),
                ("cora", GMM_RESP, 0.56),
                ("cora", ELLIPSOID_SHUFFLED, 0.54),
                ("citeseer", BASE, 0.53),
                ("citeseer", ELLIPSOID, 0.52),
                ("citeseer", GMM_RESP, 0.54),
                ("citeseer", ELLIPSOID_SHUFFLED, 0.51),
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["dataset", "method", "seed", "noise_type", "noise_rate", "status", "test"],
                )
                writer.writeheader()
                for dataset, method, test_acc in rows:
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": 0,
                            "noise_type": "uniform",
                            "noise_rate": 0.4,
                            "status": "ok",
                            "test": test_acc,
                        }
                    )

            loaded = load_rows(Path(tmpdir))
            self.assertEqual(len(loaded), 8)
            agg = aggregate(loaded)
            self.assertEqual(len(agg), 8)
            deltas = paired_deltas(loaded)
            delta_lookup = {(row["dataset"], row["comparison"]): row for row in deltas}
            self.assertAlmostEqual(delta_lookup[("cora", "ellipsoid_minus_beta05")]["mean_delta"], 0.02)
            self.assertAlmostEqual(delta_lookup[("citeseer", "gmmresp_minus_beta05")]["mean_delta"], 0.01)
            decisions = decide(deltas, ["cora", "citeseer"])
            decision_lookup = {row["method_family"]: row for row in decisions}
            self.assertEqual(decision_lookup["ellipsoid+beta05"]["seed0_signal"], 0)
            self.assertEqual(decision_lookup["gmm_resp+beta05"]["seed0_signal"], 1)


if __name__ == "__main__":
    unittest.main()
