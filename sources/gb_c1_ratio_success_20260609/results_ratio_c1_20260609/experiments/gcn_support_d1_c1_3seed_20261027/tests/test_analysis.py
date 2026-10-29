"""CPU-only statistics and adversarial postprocessing contract tests."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = module("gcn_extension_analysis_test", "analysis.py")

    def rows(self):
        return [dict(dataset=d, seed=s, condition=c, stage="student", method="JOINT",
                     score=100*s+offset+noise)
                for d, offset in (("a", 0), ("b", 20)) for s in range(3)
                for c, noise in (("clean", 1000), ("uniform0.4", 0), ("uniform0.6", 10))]

    def test_macro_is_seed_first_with_sample_sd(self):
        out = self.a.summarize(self.rows(), ("dataset", "condition", "stage", "method"), ("score",))
        macro = next(r for r in out if r["dataset"] == "ALL" and r["condition"] == "NOISY_MACRO")
        self.assertEqual(macro["n_seeds"], 3)
        self.assertEqual(macro["score_mean"], 115)
        self.assertEqual(macro["score_std"], 100)
        self.assertEqual([macro[f"score_seed{s}"] for s in range(3)], [15, 115, 215])
        self.assertEqual(macro["cells_per_seed"], 4)

    def test_clean_is_separate(self):
        out = self.a.summarize(self.rows(), ("dataset", "condition", "stage", "method"), ("score",))
        clean = next(r for r in out if r["dataset"] == "ALL" and r["condition"] == "clean")
        self.assertEqual(clean["score_mean"], 1110)
        self.assertEqual(clean["score_std"], statistics.stdev([1010, 1110, 1210]))

    def test_missing_or_duplicate_seed_cell_fails(self):
        ids = ("dataset", "condition", "stage", "method")
        with self.assertRaises(ValueError):
            self.a.summarize(self.rows()[:-1], ids, ("score",))
        with self.assertRaises(ValueError):
            self.a.summarize(self.rows()+[self.rows()[0]], ids, ("score",))

    def test_paired_values_are_matched_within_seed(self):
        rows = [dict(dataset="x", seed=s, condition="clean", stage=stage, method=m,
                     test_accuracy=v/100, test_macro_f1=v/100, validation=v/100)
                for s in range(3) for stage in ("teacher", "student")
                for m, v in (("JOINT", 70+s), ("CE", 73-s))]
        pairs, gaps = self.a.comparisons(rows, controls=("CE",), methods=("JOINT", "CE"))
        self.assertEqual(len(pairs), 6)
        np.testing.assert_allclose([r["test_accuracy_diff_pp"] for r in pairs[:3]], [-3, -1, 1])
        self.assertTrue(all(abs(r["student_minus_teacher_accuracy_pp"]) < 1e-12 for r in gaps))
        with self.assertRaises(ValueError):
            self.a.comparisons(rows[:-1], controls=("CE",), methods=("JOINT", "CE"))

    def test_metrics_include_absent_classes(self):
        result, recalls = self.a.metrics(np.array([[.9, .1, 0], [.8, .2, 0], [.1, .9, 0]]),
                                         np.array([0, 1, 1]), np.arange(3), 3)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["macro_f1"], 4/9)
        self.assertEqual(recalls[2], dict(class_id=2, count=0, recall=None))
        with self.assertRaises(ValueError):
            self.a.metrics(np.array([[float("nan"), 0, 0]]), np.array([0]), np.array([0]), 3)

    def test_manifest_mutation_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"x").write_text("before", encoding="utf-8")
            reader = self.a.Reader()
            reader.hash(root/"x")
            (root/"x").write_text("after", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                reader.verify()
            with self.assertRaises(ValueError):
                self.a.relative_input(root, "../escape")

    def test_config_exact_scope(self):
        config = json.loads((HERE/"config.json").read_text(encoding="utf-8"))
        self.a.validate_config(config)
        config["seeds"] = [0]
        with self.assertRaises(ValueError):
            self.a.validate_config(config)

    def test_sources_import_without_training_module(self):
        source = (HERE/"analysis.py").read_text(encoding="utf-8")
        self.assertNotIn("from run import", source)
        self.assertNotIn("import protocol", source)

    def test_full_freeze_schema_and_wrong_selection_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.a.save(root/"teacher_refs_frozen.json", {})
            self.a.save(root/"refs.json", {})
            teacher_paths, all_paths = {}, {}
            for dataset in self.a.DATASETS:
                for seed in range(3):
                    for condition in self.a.CONDITIONS:
                        for stage in ("teacher", "student"):
                            path = self.a.selection_path(root, stage, dataset, seed, condition)
                            self.a.save(path, {})
                            name = path.relative_to(root).as_posix()
                            all_paths[name] = self.a.sha(path)
                            if stage == "teacher":
                                teacher_paths[name] = all_paths[name]
            self.a.save(root/"teachers_frozen.json", dict(selections=teacher_paths, races={},
                        refs_sha256=self.a.sha(root/"teacher_refs_frozen.json")))
            frozen = dict(choices=all_paths, refs_sha256=self.a.sha(root/"refs.json"),
                          teacher_freeze_sha256=self.a.sha(root/"teachers_frozen.json"))
            self.a.save(root/"selections_frozen.json", frozen)
            self.a.check_freeze(root, self.a.Reader())
            name, digest = next(iter(frozen["choices"].items()))
            del frozen["choices"][name]
            self.a.save(root/"arbitrary.json", {})
            frozen["choices"]["arbitrary.json"] = self.a.sha(root/"arbitrary.json")
            self.a.save(root/"selections_frozen.json", frozen)
            with self.assertRaises(ValueError):
                self.a.check_freeze(root, self.a.Reader())

    def test_tex_tables_are_valid_and_escape_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"table.tex"
            self.a.tex_table(path, [dict(dataset="a_b", test_accuracy_mean=.5, test_accuracy_std=.01)],
                             ("dataset",), ("test_accuracy",))
            text = path.read_text(encoding="utf-8")
            self.assertIn(r"a\_b", text)
            self.assertIn(r"50.000 \pm 1.000", text)
            self.assertIn(r"\begin{longtable}{lr}", text)

    def test_aliases_keep_all_540_logical_rows_and_reject_cross_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, refs = Path(tmp), {}
            for dataset in self.a.DATASETS:
                for seed in range(3):
                    for condition in self.a.CONDITIONS:
                        choices = {}
                        for stage in ("teacher", "student"):
                            key = self.a.cell_id(dataset, seed, condition)+f"/{stage}/CE"
                            refs[key] = dict(dataset=dataset, seed=seed, condition=condition, kind=stage,
                                             arm="CE", folder=str(root/"fits"/key), target_path=str(root/"t.pt"),
                                             historical=self.a.historical(dataset, seed))
                            choices[stage] = dict.fromkeys(self.a.METHODS, key)
                        for stage in ("teacher", "student"):
                            selection = dict(dataset=dataset, seed=seed, condition=condition, choices=choices[stage])
                            if stage == "student":
                                selection["teacher_choices"] = choices["teacher"]
                            self.a.save(self.a.selection_path(root, stage, dataset, seed, condition), selection)
            rows = self.a.check_choices(root, self.a.Reader(), refs)
            self.assertEqual(len(rows), 540)
            path = self.a.selection_path(root, "student", "cora", 1, "clean")
            selection = self.a.read(path)
            selection["choices"]["JOINT"] = "cora/seed0/clean/student/CE"
            self.a.save(path, selection)
            with self.assertRaises(ValueError):
                self.a.check_choices(root, self.a.Reader(), refs)

    def test_requested_support_and_group_strengths(self):
        spec = self.a.selected_spec("S128_G_A0.25_C1.0")
        self.assertEqual(spec["requested_support"], 128)
        self.assertEqual(spec["agree_strength"], .25)
        self.assertEqual(spec["conflict_strength"], 1)
        self.assertEqual(spec["tie_strength"], 1)
        self.assertEqual(self.a.selected_spec("LEGACY_SUPPORT32")["requested_support"], 32)
        self.assertIsNone(self.a.selected_spec("OLD64")["requested_support"])
        self.assertIsNone(self.a.selected_spec("CE")["requested_support"])
        self.assertEqual(self.a.selected_spec("OLD64")["ball_count_cap"], 64)


class AuditMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = module("gcn_extension_audit_test", "audit.py")

    def test_loo_ties_low_support_and_d1_weights(self):
        e = self.a.evidence(np.array([0, 1, 1, 0, 1, 2]), np.array([4, 4, 4, 8, 8, 9]), 3)
        np.testing.assert_array_equal(e["counts"][0], [0, 2, 0])
        self.assertTrue(e["tied"][1])
        np.testing.assert_allclose(e["q"][1], [.5, .5, 0])
        np.testing.assert_array_equal(e["alpha"][3:], 0)
        spec = dict(kind="group", support=32, agree=0., conflict=1.)
        self.assertEqual(self.a.weights(e, spec)[1], 1.)
        target = self.a.shrink(e, self.a.weights(e, spec))
        np.testing.assert_allclose(target.sum(1), 1)
        np.testing.assert_allclose(target[0], [0, 1, 0])
        np.testing.assert_allclose(target[5], [0, 0, 1])

    def test_prefix_strict_accuracy_first_not_nll(self):
        logs = [dict(epoch=1, validation_correct=5, validation_nll=2.),
                dict(epoch=2, validation_correct=5, validation_nll=.1),
                dict(epoch=3, validation_correct=6, validation_nll=.2)]
        score = self.a.prefix_score(logs, 2, "CE", 0, 0., 10)
        self.assertEqual((score["best_epoch"], score["valid_nll"]), (1, 2.))
        with self.assertRaises(ValueError):
            self.a.prefix_score(logs[1:], 2, "CE", 0, 0., 10)

    def score(self, arm, **kw):
        return dict(arm=arm, valid_correct=5, valid_total=10, valid_nll=.5, mean_lambda=.5,
                    support=32, best_epoch=1, epoch=25, **kw)

    def test_anchor_within_quota_and_rank_field_allowlist(self):
        rows = [self.score(f"S{i}_L0.5") for i in range(16)]
        anchor = self.score("S32_L1.0")
        anchor["valid_correct"] = 0
        chosen = self.a.survivors(rows+[anchor], 14, 25)
        self.assertEqual(len(chosen), 14)
        self.assertIn("S32_L1.0", chosen)
        with self.assertRaises(ValueError):
            self.a.rank(dict(rows[0], test_accuracy=1))
        with self.assertRaises(ValueError):
            self.a.survivors(rows+[dict(anchor, epoch=75)], 14, 25)

    def test_grid_exact_42_and_fixed_8(self):
        grid = self.a.candidates()
        self.assertEqual(len(grid), 42)
        self.assertEqual(1+sum(x["support"] == 32 for x in grid.values()), 8)

    def test_target_diagnostics_are_observed_only_and_ce_has_zero_mass(self):
        e = self.a.evidence(np.array([0, 1, 1, 2]), np.array([0, 0, 0, 4]), 3)
        es = {"LEGACY": e, "32": e}
        ref = dict(arm="CE", partition="LEGACY", historical=False)
        row = self.a.target_diagnostic("cora", 1, "clean", "CE", ref, es, self.a.candidates())
        self.assertEqual(row["actual_loo_support_mean"], 1.5)
        self.assertEqual(row["actual_loo_support_p50"], 2)
        self.assertEqual((row["actual_loo_support_min"], row["actual_loo_support_max"]), (0, 2))
        self.assertEqual(row["support_ge2_coverage"], .75)
        self.assertEqual(row["mean_lambda"], 0)
        self.assertEqual(row["mean_effective_m"], 0)
        self.assertEqual(row["modified_rate"], 0)
        ref.update(arm="S32_G_A0.0_C1.0", partition="32")
        row = self.a.target_diagnostic("cora", 1, "clean", "FIXED_D1", ref, es, self.a.candidates())
        self.assertGreater(row["mean_effective_m"], 0)
        self.assertEqual(row["selected_arm"], "S32_G_A0.0_C1.0")
        # Both tied LOO votes retain lambda=1, hence all three supported rows change.
        self.assertAlmostEqual(row["modified_rate"], .75)

    def test_tie_break_order(self):
        base = self.score("S32_L0.5")
        alternatives = [dict(base, valid_nll=.6), dict(base, mean_lambda=.6),
                        dict(base, support=16), dict(base, best_epoch=2)]
        self.assertTrue(all(self.a.rank(base) < self.a.rank(r) for r in alternatives))

    def test_frozen_check_rejects_unfrozen_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.a.check_freeze(Path(tmp), self.a.analysis.Reader(), historical=False)


if __name__ == "__main__":
    unittest.main()
