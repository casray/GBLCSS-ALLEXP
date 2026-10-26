from pathlib import Path
import sys
import unittest
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class WindowsPathContract(unittest.TestCase):
    def test_scoring_keys_are_separator_independent(self):
        import analyze_windows as module
        self.assertTrue(hasattr(module, 'PathKeyDict'), 'Windows score-path normalization not implemented')
        values = module.PathKeyDict()
        values['fits\\B1\\cora\\R0\\clean\\CE'] = 7
        self.assertEqual(values['fits/B1/cora/R0/clean/CE'], 7)
        self.assertEqual(values['fits\\B1\\cora\\R0\\clean\\CE'], 7)
        r = module.canonical_record(dict(folder='fits\\B1\\cora\\R0\\clean\\CE', stage='B1', alias=True))
        self.assertFalse(r['alias'])
        a = module.canonical_record(dict(folder='fits\\C1_ANCHOR\\citeseer\\R0\\clean\\CE', stage='C1_REF', alias=True))
        self.assertEqual(a['stage'], 'C1_ANCHOR')
        self.assertFalse(a['alias'])


if __name__ == '__main__':
    unittest.main()
