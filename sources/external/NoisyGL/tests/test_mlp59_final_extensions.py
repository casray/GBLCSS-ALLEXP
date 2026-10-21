import unittest
import importlib

import pandas as pd

from utils.tools import load_conf


TEACHER_REAL = "mlp_distill_formal_lcss_x"
TEACHER_SHUFFLED = "mlp_distill_formal_lcss_x_shuffled"
BOUNDARY_REAL = "frozengrace_gb_mlp_softdist_beta05_loo"
BOUNDARY_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"


class MLP59ConfigTests(unittest.TestCase):
    def test_teacher_configs_cover_the_two_new_datasets(self):
        for dataset in ["coauthorcs", "amazonpho"]:
            real = load_conf(None, TEACHER_REAL, dataset)
            shuffled = load_conf(None, TEACHER_SHUFFLED, dataset)

            for conf, method, random_control in [
                (real, TEACHER_REAL, False),
                (shuffled, TEACHER_SHUFFLED, True),
            ]:
                self.assertEqual(conf.model["method"], method)
                self.assertFalse(conf.gb["teacher_use_xh"])
                self.assertEqual(conf.gb["teacher_target"], "soft_target")
                self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)
                self.assertTrue(conf.gb["leave_one_out"])
                self.assertEqual(conf.gb["tie_mode"], "distribution_only")
                self.assertEqual(bool(conf.gb["random_control"]), random_control)

            for key, value in real.gb.items():
                if key != "random_control":
                    self.assertEqual(value, shuffled.gb[key])

    def test_boundary_configs_use_the_final_target(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        for dataset in [
            "pubmed",
            "amazoncom",
            "coraml",
            "attribwiki",
            "citeseerfull",
            "coauthorph",
        ]:
            real = load_conf(None, BOUNDARY_REAL, dataset)
            shuffled = load_conf(None, BOUNDARY_SHUFFLED, dataset)

            for conf, method, random_control in [
                (real, BOUNDARY_REAL, False),
                (shuffled, BOUNDARY_SHUFFLED, True),
            ]:
                self.assertEqual(conf.model["method"], method)
                self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)
                self.assertTrue(conf.gb["leave_one_out"])
                self.assertEqual(conf.gb["tie_mode"], "distribution_only")
                predictor = getattr(module, f"{method}_Predictor")
                self.assertEqual(bool(predictor.random_control), random_control)


def _rows(datasets, rates, methods):
    rows = []
    offsets = {
        TEACHER_REAL: 0.04,
        TEACHER_SHUFFLED: 0.00,
        "mlp_distill_gcn": 0.02,
        BOUNDARY_REAL: 0.03,
        BOUNDARY_SHUFFLED: 0.00,
        "mlp_jocor": 0.02,
    }
    for dataset in datasets:
        for rate in rates:
            for seed in range(5):
                for method in methods:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "noise_type": "uniform",
                            "noise_rate": rate,
                            "noise_scope": "train",
                            "status": "ok",
                            "test": 0.70 + offsets[method] + 0.001 * seed,
                            "split_hash": f"split-{dataset}-{seed}",
                            "noise_hash": f"noise-{dataset}-{rate}-{seed}",
                            "modified_train_count": 10,
                            "modified_count": 10,
                            "modified_val_count": 0,
                            "modified_test_count": 0,
                            "target_beta": 0.5,
                            "leave_one_out": 1,
                            "tie_mode": "distribution_only",
                            "teacher_use_xh": 0,
                            "random_control": int("shuffled" in method),
                        }
                    )
    return pd.DataFrame(rows)


class MLP59SummaryTests(unittest.TestCase):
    def test_protocol_pairing_and_teacher_gate(self):
        from scripts.summarize_mlp59_final_extensions import (
            evaluate_teacher_gate,
            paired_comparisons,
            validate_protocol,
        )

        raw = _rows(
            ["coauthorcs", "amazonpho"],
            [0.4, 0.6],
            [TEACHER_REAL, TEACHER_SHUFFLED, "mlp_distill_gcn"],
        )
        audit = validate_protocol(
            raw,
            expected_datasets={"coauthorcs", "amazonpho"},
            expected_methods={TEACHER_REAL, TEACHER_SHUFFLED, "mlp_distill_gcn"},
            expected_settings={("uniform", 0.4), ("uniform", 0.6)},
            require_formal_target={TEACHER_REAL, TEACHER_SHUFFLED},
        )
        self.assertTrue(audit["passed"].all())

        paired, summary = paired_comparisons(
            raw,
            lhs=TEACHER_REAL,
            rhs_methods=[TEACHER_SHUFFLED, "mlp_distill_gcn"],
        )
        self.assertEqual(len(paired), 40)
        decisions = evaluate_teacher_gate(summary, base_positive_cells=4)
        self.assertTrue(bool(decisions.loc[0, "passed"]))

    def test_boundary_entry_gate(self):
        from scripts.summarize_mlp59_final_extensions import (
            evaluate_boundary_entry,
            paired_comparisons,
        )

        raw = _rows(
            ["coraml"],
            [0.4],
            [BOUNDARY_REAL, BOUNDARY_SHUFFLED, "mlp_jocor"],
        )
        _, summary = paired_comparisons(
            raw,
            lhs=BOUNDARY_REAL,
            rhs_methods=[BOUNDARY_SHUFFLED, "mlp_jocor"],
        )
        decision = evaluate_boundary_entry(
            summary,
            shuffled_method=BOUNDARY_SHUFFLED,
            baseline_method="mlp_jocor",
        )
        self.assertTrue(bool(decision.loc[0, "passed"]))

    def test_hash_mismatch_is_rejected(self):
        from scripts.summarize_mlp59_final_extensions import paired_comparisons

        raw = _rows(
            ["coraml"],
            [0.4],
            [BOUNDARY_REAL, BOUNDARY_SHUFFLED],
        )
        mask = (raw["method"] == BOUNDARY_SHUFFLED) & (raw["seed"] == 0)
        raw.loc[mask, "noise_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            paired_comparisons(
                raw,
                lhs=BOUNDARY_REAL,
                rhs_methods=[BOUNDARY_SHUFFLED],
            )

    def test_baseline_selection_can_be_scoped_to_boundary_datasets(self):
        from scripts.summarize_mlp59_final_extensions import strongest_baseline_methods

        controls = pd.DataFrame(
            {
                "dataset": ["pubmed", "cora"],
                "method": ["mlp_jocor", "mlp_jocor"],
                "test": [0.78, 0.52],
            }
        )
        controls = controls[controls["dataset"].isin({"pubmed"})]

        selected = strongest_baseline_methods(
            controls,
            allowed_methods={"mlp_jocor"},
        )

        self.assertEqual(selected, {"pubmed": "mlp_jocor"})

    def test_full_spectrum_selects_best_baseline_per_setting(self):
        from scripts.summarize_mlp59_final_extensions import select_best_baseline_per_setting

        controls = _rows(
            ["citeseerfull"],
            [0.2, 0.6],
            ["mlp_jocor"],
        )
        alternative = controls.copy()
        alternative["method"] = "mlp_small_loss"
        alternative.loc[alternative["noise_rate"] == 0.2, "test"] += 0.02
        alternative.loc[alternative["noise_rate"] == 0.6, "test"] -= 0.02
        controls = pd.concat([controls, alternative], ignore_index=True)

        rows, selected = select_best_baseline_per_setting(
            controls,
            allowed_methods={"mlp_jocor", "mlp_small_loss"},
        )

        self.assertEqual(len(rows), 10)
        chosen = selected.set_index("noise_rate")["baseline_source_method"].to_dict()
        self.assertEqual(chosen, {0.2: "mlp_small_loss", 0.6: "mlp_jocor"})


if __name__ == "__main__":
    unittest.main()
