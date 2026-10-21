import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_v32_tables.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_v32_tables_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateV32TablesTest(unittest.TestCase):
    def test_real_sources_generate_resource_separated_tables(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            generator.generate_all(output_dir=output_dir)

            table_a = (output_dir / "v32_table_a_target_only.md").read_text(
                encoding="utf-8"
            )
            table_b = (
                output_dir / "v32_table_b_graph_assisted_xonly.md"
            ).read_text(encoding="utf-8")
            table_c = (output_dir / "v32_table_c_distillation.md").read_text(
                encoding="utf-8"
            )
            boundary = (output_dir / "v32_boundary_datasets.md").read_text(
                encoding="utf-8"
            )

            self.assertIn("GB-LCSS", table_a)
            self.assertIn("JoCoR", table_a)
            self.assertNotIn("GSDN", table_a)
            self.assertNotIn("PosteL", table_a)
            self.assertNotIn("PubMed", table_a)

            for method in ("PosteL", "ALS", "SALS", "GSDN"):
                self.assertIn(method, table_b)
            self.assertIn("临时适配实现", table_b)
            self.assertNotIn("gsdn_faithful_core", table_b)
            self.assertNotIn("待用户", table_b)
            self.assertNotIn("headline", table_b)
            self.assertNotIn("；", table_b)
            self.assertNotIn("PubMed", table_b)

            self.assertIn("历史扩展证据", table_c)
            self.assertNotIn("GSDN", table_c)
            self.assertNotIn("headline", table_c)
            self.assertNotIn("；", table_c)
            self.assertNotIn("PubMed", table_c)

            self.assertIn("PubMed", boundary)
            self.assertIn("仅附录", boundary)

    def test_headline_metrics_are_current_tie_safe_values(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            generator.generate_all(output_dir=output_dir)
            metrics = json.loads(
                (output_dir / "v32_headline_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metrics["vs_shuffled"]["positive_settings"], 25)
            self.assertAlmostEqual(
                metrics["vs_shuffled"]["mean_delta_pp"],
                7.8410,
                places=3,
            )
            self.assertEqual(metrics["vs_ce"]["positive_settings"], 24)
            self.assertAlmostEqual(
                metrics["vs_ce"]["mean_delta_pp"],
                7.2517,
                places=3,
            )
            self.assertEqual(metrics["vs_jocor"]["positive_settings"], 23)
            self.assertAlmostEqual(
                metrics["vs_jocor"]["mean_delta_pp"],
                2.9638,
                places=3,
            )
            self.assertAlmostEqual(
                metrics["vs_postel"]["mean_delta_pp"],
                5.202,
                places=3,
            )
            self.assertAlmostEqual(
                metrics["vs_als"]["mean_delta_pp"],
                7.215,
                places=3,
            )
            self.assertAlmostEqual(
                metrics["vs_sals"]["mean_delta_pp"],
                4.202,
                places=3,
            )
            self.assertNotIn("vs_gsdn", metrics)

    def test_body_tables_exclude_legacy_tie_rule_language(self):
        generator = load_generator()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            generator.generate_all(output_dir=output_dir)
            body = "\n".join(
                (output_dir / name).read_text(encoding="utf-8")
                for name in (
                    "v32_table_a_target_only.md",
                    "v32_table_b_graph_assisted_xonly.md",
                    "v32_table_c_distillation.md",
                )
            )
            self.assertNotIn("min_index", body)
            self.assertNotIn("最小索引并列", body)


if __name__ == "__main__":
    unittest.main()
