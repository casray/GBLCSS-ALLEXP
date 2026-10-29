import importlib.util
from pathlib import Path
import types
import unittest

CODE = Path(__file__).resolve().parents[1]


class AuditRecoveryTests(unittest.TestCase):
    def test_binds_original_path_validator_without_replacing_it(self):
        spec = importlib.util.spec_from_file_location('audit_wrapper_test', CODE/'audit_recovery_v1.py')
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        spec = importlib.util.spec_from_file_location('frozen_audit_test', CODE/'audit.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = (CODE/'audit.py').read_bytes()
        helper = module.analysis.relative_input
        self.assertIs(wrapper.bind_helper(module).relative_input, helper)
        self.assertEqual(module.relative_input(CODE,'audit.py'),(CODE/'audit.py').resolve())
        with self.assertRaises(ValueError):
            module.relative_input(CODE,'../escape')
        self.assertEqual(before,(CODE/'audit.py').read_bytes())
