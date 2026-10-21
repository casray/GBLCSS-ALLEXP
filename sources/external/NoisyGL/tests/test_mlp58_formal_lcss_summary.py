import unittest

import pandas as pd

from scripts.summarize_mlp58_formal_lcss_teacher import (
    aggregate_results,
    paired_comparisons,
    evaluate_gates,
    validate_control_pairing,
    validate_new_runs,
)


REAL = "mlp_distill_formal_lcss_x"
SHUFFLED = "mlp_distill_formal_lcss_x_shuffled"


def _new_rows():
    rows = []
    for rate in [0.4, 0.6]:
        for dataset in ["cora", "citeseer", "corafull"]:
            for seed in range(5):
                for method, random_control, test in [
                    (REAL, 0, 0.80 + seed * 0.001),
                    (SHUFFLED, 1, 0.76 + seed * 0.001),
                ]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "noise_type": "uniform",
                            "noise_rate": rate,
                            "noise_scope": "train",
                            "status": "ok",
                            "test": test,
                            "teacher_use_xh": 0,
                            "target_beta": 0.5,
                            "leave_one_out": 1,
                            "tie_mode": "distribution_only",
                            "random_control": random_control,
                            "modified_val_count": 0,
                            "modified_test_count": 0,
                            "split_hash": f"split-{dataset}-{seed}",
                            "noise_hash": f"noise-{rate}-{dataset}-{seed}",
                        }
                    )
    return pd.DataFrame(rows)


def _controls():
    rows = []
    for rate in [0.4, 0.6]:
        for dataset in ["cora", "citeseer", "corafull"]:
            for seed in range(5):
                rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "method": "mlp_distill_gcn",
                            "seed": seed,
                            "noise_rate": rate,
                            "test": 0.77 + seed * 0.001,
                        },
                        {
                            "dataset": dataset,
                            "method": "mlp_distill_softdist_x",
                            "seed": seed,
                            "noise_rate": rate,
                            "test": 0.79 + seed * 0.001,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def _hashed_controls(raw):
    controls = _controls()
    hashes = raw[raw["method"] == REAL][
        ["dataset", "noise_rate", "seed", "split_hash", "noise_hash"]
    ]
    return controls.merge(hashes, on=["dataset", "noise_rate", "seed"])


class MLP58SummaryTests(unittest.TestCase):
    def test_protocol_validation_and_aggregation(self):
        raw = _new_rows()
        audit = validate_new_runs(raw)
        self.assertTrue(audit["passed"].all())
        aggregate = aggregate_results(raw)
        self.assertEqual(len(aggregate), 12)
        self.assertTrue((aggregate["count"] == 5).all())

    def test_paired_comparisons_and_gates(self):
        raw = _new_rows()
        controls = _controls()
        paired, summary = paired_comparisons(raw, controls)
        self.assertEqual(len(paired), 90)
        self.assertEqual(len(summary), 18)

        gates = evaluate_gates(summary)
        self.assertEqual(set(gates["gate"]), {"vs_gcn", "vs_beta0", "vs_shuffled"})
        self.assertTrue(gates["passed"].all())

    def test_control_hash_pairing(self):
        raw = _new_rows()
        audit = validate_control_pairing(raw, _hashed_controls(raw))
        self.assertEqual(len(audit), 60)
        self.assertTrue(audit["passed"].all())


if __name__ == "__main__":
    unittest.main()
