import importlib
import tempfile
import unittest
from pathlib import Path

from utils.tools import load_conf


class MLP9DenoisedTeacherTests(unittest.TestCase):
    def test_predictor_variants_registered(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        self.assertTrue(module.mlp_distill_gcn_xh_Predictor.teacher_use_xh)
        self.assertEqual(module.mlp_distill_gcn_xh_Predictor.teacher_target, "ce")
        self.assertTrue(module.mlp_distill_softdist_xh_Predictor.teacher_use_xh)
        self.assertEqual(module.mlp_distill_softdist_xh_Predictor.teacher_target, "soft_target")
        self.assertTrue(module.mlp_distill_softdist_xh_Predictor.use_soft_targets)
        self.assertFalse(module.mlp_distill_softdist_xh_Predictor.skip_ball_targets)
        self.assertEqual(module.mlp_distill_knn_xh_Predictor.soft_target_builder, "knn")
        self.assertTrue(module.mlp_distill_softdist_xh_shuffled_Predictor.random_control)

        ssl_runner = importlib.import_module("scripts.run_ssl_swap_downstream")
        for method in [
            "mlp_distill_gcn_xh",
            "mlp_distill_softdist_xh",
            "mlp_distill_knn_xh",
            "mlp_distill_softdist_xh_shuffled",
        ]:
            with self.subTest(method=method):
                cls = ssl_runner.predictor_class_for(method)
                self.assertIs(cls, getattr(module, f"{method}_Predictor"))

    def test_configs_load_and_keep_student_x_only(self):
        for method in [
            "mlp_distill_gcn_xh",
            "mlp_distill_softdist_xh",
            "mlp_distill_knn_xh",
            "mlp_distill_softdist_xh_shuffled",
        ]:
            for dataset in ["cora", "citeseer", "pubmed"]:
                with self.subTest(method=method, dataset=dataset):
                    conf = load_conf(None, method, dataset)
                    self.assertEqual(conf.model["method"], method)
                    self.assertEqual(conf.model["n_hidden"], 64)
                    self.assertEqual(conf.model["n_layer"], 2)
                    self.assertEqual(conf.training["n_epochs"], 200)
                    self.assertTrue(conf.gb["teacher_use_xh"])
                    self.assertEqual(conf.gb["teacher_hidden"], 64)
                    self.assertEqual(conf.gb["teacher_n_layer"], 2)
                    self.assertAlmostEqual(float(conf.gb["distill_temperature"]), 1.0)
                    self.assertNotIn("use_xh_features", conf.model)

    def test_teacher_target_modes_are_pre_registered(self):
        ce_conf = load_conf(None, "mlp_distill_gcn_xh", "cora")
        ball_conf = load_conf(None, "mlp_distill_softdist_xh", "cora")
        knn_conf = load_conf(None, "mlp_distill_knn_xh", "cora")
        shuf_conf = load_conf(None, "mlp_distill_softdist_xh_shuffled", "cora")
        self.assertEqual(ce_conf.gb["teacher_target"], "ce")
        self.assertEqual(ball_conf.gb["teacher_target"], "soft_target")
        self.assertEqual(knn_conf.gb["teacher_target"], "soft_target")
        self.assertEqual(shuf_conf.gb["teacher_target"], "soft_target")
        self.assertEqual(knn_conf.gb["soft_target_builder"], "knn")
        self.assertAlmostEqual(float(ball_conf.gb["target_beta"]), 0.0)
        self.assertAlmostEqual(float(knn_conf.gb["target_beta"]), 0.0)
        self.assertAlmostEqual(float(shuf_conf.gb["target_beta"]), 0.0)

    def test_summarizer_uses_5seed_gate_logic(self):
        summary = importlib.import_module("scripts.summarize_mlp9_denoised_teacher")
        rows = []
        for dataset in ["cora", "citeseer", "pubmed"]:
            for seed in range(5):
                base = 0.50 + 0.01 * seed
                rows.extend([
                    {"dataset": dataset, "noise_rate": 0.4, "method": summary.T1, "seed": seed, "student_test": base},
                    {"dataset": dataset, "noise_rate": 0.4, "method": summary.T2, "seed": seed, "student_test": base + 0.02},
                    {"dataset": dataset, "noise_rate": 0.4, "method": summary.T3, "seed": seed, "student_test": base + 0.04},
                    {"dataset": dataset, "noise_rate": 0.4, "method": summary.T4, "seed": seed, "student_test": base + 0.02},
                    {"dataset": dataset, "noise_rate": 0.4, "method": summary.T5, "seed": seed, "student_test": base + 0.01},
                ])
        paired = summary.paired_delta_rows(rows)
        gates = summary.gate_rows(paired)
        overall = {
            row["gate"]: row
            for row in gates
            if row["scope"] == "overall" and float(row["noise_rate"]) == 0.4
        }
        self.assertEqual(overall["G1_win_glnn"]["decision"], "pass")
        self.assertEqual(overall["G2_denoising_attribution"]["decision"], "pass")
        self.assertEqual(overall["G3_ball_attribution"]["decision"], "pass")
        self.assertEqual(overall["G4_structure_sanity"]["decision"], "pass")
        self.assertEqual(overall["G3_ball_attribution"]["dataset_pass_count"], 3)

    def test_summarizer_loads_multiseed_new_rows_including_t1(self):
        summary = importlib.import_module("scripts.summarize_mlp9_denoised_teacher")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp9_uniform0p4_seeds_1_2_3_4.csv"
            path.write_text(
                "dataset,method,seed,noise_type,noise_rate,status,test,teacher_best_test,teacher_use_xh,teacher_input_dim,student_input_dim,teacher_target\n"
                "cora,mlp_distill_gcn,1,uniform,0.4,ok,0.70,0.71,0,1433,1433,ce\n"
                "cora,mlp_distill_softdist_xh,1,uniform,0.4,ok,0.72,0.73,1,1561,1433,soft_target\n",
                encoding="utf-8",
            )
            rows = summary._load_new_rows(Path(tmp))
        methods = {row["method"] for row in rows}
        self.assertIn(summary.T1, methods)
        self.assertIn(summary.T3, methods)

    def test_knn_matched_k_audit_keeps_seed_level_values(self):
        summary = importlib.import_module("scripts.summarize_mlp9_denoised_teacher")
        rows = [
            {
                "dataset": "cora",
                "noise_rate": 0.4,
                "method": summary.T4,
                "seed": 2,
                "matched_k": 10.0,
                "matched_num_balls": 52.0,
                "student_test": 0.81,
                "teacher_test": 0.82,
            },
            {
                "dataset": "cora",
                "noise_rate": 0.4,
                "method": summary.T3,
                "seed": 2,
                "matched_k": "",
                "matched_num_balls": "",
                "student_test": 0.80,
                "teacher_test": 0.81,
            },
        ]
        audit = summary.knn_matched_k_rows(rows)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["matched_k"], 10.0)
        self.assertEqual(audit[0]["matched_num_balls"], 52.0)


if __name__ == "__main__":
    unittest.main()
