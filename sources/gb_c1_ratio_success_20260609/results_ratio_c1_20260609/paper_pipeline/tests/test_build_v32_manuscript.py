import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_v32_manuscript.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_v32_manuscript_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildV32ManuscriptTest(unittest.TestCase):
    def test_builds_v32_with_current_claims_and_resource_layers(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "v32.md"
            builder.build_manuscript(output_path=output)
            text = output.read_text(encoding="utf-8")

            self.assertIn("25/25", text)
            self.assertIn("24/25", text)
            self.assertIn("23/25", text)
            self.assertIn("7.84", text)
            self.assertIn("7.25", text)
            self.assertIn("2.96", text)
            self.assertIn("5.20", text)
            self.assertIn("7.22", text)
            self.assertIn("4.20", text)
            self.assertIn("临时适配实现", text)
            self.assertIn("distribution_only", text)
            self.assertIn("验证标签噪声敏感性", text)
            self.assertIn("transductive", text)

            self.assertNotIn("14/15", text)
            self.assertNotIn("+7.16", text)
            self.assertNotIn("+7.83", text)
            self.assertNotIn("+2.87", text)
            self.assertNotIn("全面优于 GSDN", text)
            self.assertNotIn("待用户", text)
            self.assertNotIn("headline", text)
            self.assertNotIn("；", text)

    def test_pubmed_is_appendix_only(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "v32.md"
            builder.build_manuscript(output_path=output)
            text = output.read_text(encoding="utf-8")
            body, supplement = text.split("# 补充材料", 1)

            self.assertNotIn("PubMed", body)
            self.assertIn("PubMed 边界案例", supplement)
            self.assertIn("71.90±0.88", supplement)
            self.assertIn("77.84±1.10", supplement)

    def test_tie_safe_target_and_theory_are_present(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "v32.md"
            builder.build_manuscript(output_path=output)
            text = output.read_text(encoding="utf-8")

            self.assertIn("|M_j|=1", text)
            self.assertIn("|M_j|>1", text)
            self.assertIn("类别置换", text)
            self.assertIn(r"\rho<(C-1)/C", text)
            self.assertIn("Hoeffding", text)
            self.assertIn("不覆盖 pair", text)


if __name__ == "__main__":
    unittest.main()
