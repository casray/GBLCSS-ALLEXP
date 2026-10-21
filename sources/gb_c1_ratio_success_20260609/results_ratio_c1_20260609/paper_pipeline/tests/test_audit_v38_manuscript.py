import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_v38_manuscript.py"
MANUSCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper_final_zh_v38_loo_main_20260730.md"
)
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "04_experiments"
    / "v38_evidence_registry_20260730.csv"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_v38_manuscript_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AuditV38ManuscriptTest(unittest.TestCase):
    def test_real_v38_manuscript_has_zero_errors(self):
        auditor = load_auditor()
        self.assertEqual(auditor.audit_manuscript(MANUSCRIPT_PATH), [])

    def test_rejects_legacy_headline_and_non_node_specific_formula(self):
        auditor = load_auditor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(
                "# 粒球局部共识软监督\n"
                "旧目标规则。7.841。C:\\Users\\Name\\raw.csv\n"
                "\\tag{1}\n# 补充材料\n",
                encoding="utf-8",
            )
            errors = auditor.audit_manuscript(path)
            self.assertTrue(any("7.841" in item for item in errors))
            self.assertTrue(any("旧目标规则" in item for item in errors))
            self.assertTrue(any("C:\\Users" in item for item in errors))

    def test_real_v38_registry_has_zero_errors(self):
        auditor = load_auditor()
        self.assertEqual(auditor.audit_registry(REGISTRY_PATH), [])

    def test_registry_rejects_hash_and_row_count_mismatch(self):
        auditor = load_auditor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            source.write_text("x\n1\n2\n", encoding="utf-8")
            registry = root / "registry.csv"
            registry.write_text(
                '"evidence_id","manuscript_role","manuscript_location",'
                '"source_path","source_sha256","method_or_bundle",'
                '"dataset_scope","noise_scope","tie_mode","protocol",'
                '"row_count","headline_allowed","implementation_status",'
                '"limitations"\n'
                '"V38_LOO_FULLSPECTRUM","role","Table","source.csv","WRONG",'
                '"method","data","train","distribution_only","protocol",'
                '"3","1","current","none"\n',
                encoding="utf-8",
            )
            original_project_root = auditor.PROJECT_ROOT
            auditor.PROJECT_ROOT = root
            try:
                errors = auditor.audit_registry(registry, workspace_root=root)
            finally:
                auditor.PROJECT_ROOT = original_project_root
            self.assertTrue(any("SHA256 mismatch" in item for item in errors))
            self.assertTrue(any("row_count mismatch" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
