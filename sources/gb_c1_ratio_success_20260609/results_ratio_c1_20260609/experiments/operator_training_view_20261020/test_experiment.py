import copy
import importlib.util
import json
from pathlib import Path
import unittest


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiment'), 'experiment module is not implemented')
        import experiment
        self.module = experiment
        self.config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf8'))

    def test_manifest_complete_and_seeds_paired(self):
        jobs = self.module.build_jobs(self.config)
        self.assertEqual(len(jobs), 180)
        self.assertEqual(len({j['id'] for j in jobs}), 180)
        self.assertEqual({j['seed'] for j in jobs}, {0, 1, 2})
        self.assertEqual(len(self.module.VARIANTS), 15)
        self.assertEqual(jobs, self.module.build_jobs(self.config))

    def test_decimal_noises_and_label_sources_have_distinct_keys(self):
        f = self.module.cache_key
        self.assertNotEqual(f({'noise':'u0.4'}), f({'noise':'u0.6'}))
        self.assertNotEqual(f({'labels':'a'}), f({'labels':'b'}))
        self.assertEqual(f({'b':2,'a':1}), f({'a':1,'b':2}))

    def test_shared_plugin_baselines_are_target_and_mask_matched(self):
        v = self.module.VARIANTS
        self.assertEqual(v['P_shared_GCE']['target'], v['S_shared_GCE']['target'])
        self.assertEqual(v['P_shared_GCE']['mask'], v['S_shared_GCE']['mask'])
        self.assertEqual(v['P_shared_GCE']['loss'], 'gce')
        self.assertEqual(v['S_shared_GCE']['mode'], 'PIPPI')

    def test_pairs_never_cross_seed_or_partition(self):
        rows = []
        for seed, variant, acc, assignment in [(0,'S_shared_B',.6,'a'),(0,'P_shared_B',.7,'a'),
                    (1,'S_shared_B',.9,'a'),(1,'P_shared_B',.1,'b')]:
            rows.append({'job':{'dataset':'citeseer','noise':'u0.6','seed':seed,'variant':variant},
                'test_accuracy':acc,'split_sha256':'s','noise_sha256':'n','assignment_sha256':assignment})
        pairs = self.module.paired_summary(rows)
        r = next(r for r in pairs if r['comparison']=='S_shared_B-P_shared_B')
        self.assertEqual(r['n_pairs'], 1)
        self.assertAlmostEqual(r['delta_pp_mean'], -10)

    def test_loss_controls_not_mislabeled_as_official_graph_mixup(self):
        self.assertEqual(self.module.VARIANTS['P_ind_mixup']['loss'], 'postprop_mixup')
        self.assertEqual(self.module.VARIANTS['P_ind_LS']['loss'], 'label_smoothing')


if __name__ == '__main__':
    unittest.main()
