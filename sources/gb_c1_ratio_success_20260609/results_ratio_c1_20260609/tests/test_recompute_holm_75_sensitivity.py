import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_pipeline" / "scripts" / "recompute_holm_75_sensitivity.py"
FIELDNAMES = [
    "dataset",
    "noise_type",
    "noise_rate",
    "comparison",
    "n",
    "mean_delta",
    "std_delta",
    "ci95_low",
    "ci95_high",
    "positive_seeds",
    "negative_seeds",
    "paired_t_p",
    "holm_p",
]


def make_rows():
    rows = []
    comparisons = ("lcss_vs_ce", "lcss_vs_jocor", "lcss_vs_shuffled")
    for family_index, comparison in enumerate(comparisons):
        for setting_index in range(25):
            raw_p = 0.0001 if family_index == 0 and setting_index == 0 else 1.0
            family_holm = 0.0025 if raw_p == 0.0001 else 1.0
            rows.append(
                {
                    "dataset": f"dataset_{setting_index}",
                    "noise_type": "uniform",
                    "noise_rate": "0.4",
                    "comparison": comparison,
                    "n": "5",
                    "mean_delta": "0.1",
                    "std_delta": "0.01",
                    "ci95_low": "0.08",
                    "ci95_high": "0.12",
                    "positive_seeds": "5",
                    "negative_seeds": "0",
                    "paired_t_p": str(raw_p),
                    "holm_p": str(family_holm),
                }
            )
    return rows


class Holm75SensitivityCliTests(unittest.TestCase):
    def run_script(self, rows):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        directory = Path(temporary_directory.name)
        source = directory / "source.csv"
        output = directory / "output.csv"
        report = directory / "report.md"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::SyntaxWarning",
                str(SCRIPT),
                "--input",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result, output, report

    def test_recomputes_one_holm_family_across_all_75_rows(self):
        result, output, report = self.run_script(make_rows())

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stderr, "")
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        first = next(row for row in rows if float(row["paired_t_p"]) == 0.0001)
        self.assertEqual(len(rows), 75)
        self.assertAlmostEqual(float(first["holm_p_75_family"]), 0.0075)
        self.assertEqual(first["holm_rank_75"], "1")
        self.assertEqual(first["significant_75_family_alpha_0_05"], "1")
        self.assertIn("75", report.read_text(encoding="utf-8"))

    def test_rejects_a_duplicate_four_column_test_key(self):
        rows = make_rows()
        rows[-1] = dict(rows[0])

        result, _, _ = self.run_script(rows)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
