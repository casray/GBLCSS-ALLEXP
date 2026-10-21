import importlib
from pathlib import Path
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class GridTests(unittest.TestCase):
    def runner(self):
        self.assertTrue((ROOT / 'run_grid.py').is_file(), 'five-seed runner not implemented')
        return importlib.import_module('run_grid')

    def test_distinct_paths_and_actual_seed_configs(self):
        r = self.runner()
        paths = set()
        for dataset in r.DATASETS:
            base = r.adapter.effective_config(dataset, False, 0)
            for seed in range(5):
                paths.add(r.location(dataset, seed))
                cfg = r.adapter.effective_config(dataset, False, seed)
                self.assertEqual(cfg, dict(base, seed=seed))
                self.assertEqual(cfg['supervised_epochs_total'], 200)
        self.assertEqual(len(paths), 25)
        self.assertNotEqual(r.location('cora', 1, True), r.location('cora', 1, False))

    def test_frozen_training_function_identity(self):
        r = self.runner()
        self.assertIs(r.adapter.run, r.previous.adapter.run)
        self.assertEqual(r.previous.ROOT, ROOT.parent / 'gssc_matched_seed0_20261009')

    def test_summary_uses_sample_std_and_all_five_seeds(self):
        r = self.runner()
        records = [dict(seed=s, test=(s+1)/10) for s in range(5)]
        result = r.summarize(records)
        self.assertAlmostEqual(result['mean_percent'], 30)
        self.assertAlmostEqual(result['std_percent'], math.sqrt(250))
        with self.assertRaises(ValueError):
            r.summarize(records[:-1])
        with self.assertRaises(ValueError):
            r.summarize(records[:-1] + [records[0]])


if __name__ == '__main__':
    unittest.main()
