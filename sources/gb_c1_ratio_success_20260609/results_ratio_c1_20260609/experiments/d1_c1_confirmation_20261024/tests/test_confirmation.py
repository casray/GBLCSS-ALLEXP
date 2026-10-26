"""Protocol tests written before the confirmation implementation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ConfirmationTests(unittest.TestCase):
    def module(self, name):
        self.assertTrue((ROOT / (name + '.py')).exists(), name + ' implementation missing')
        return __import__(name)

    def test_targets_loo_endpoints_ties_and_groups(self):
        t = self.module('targets')
        z = torch.tensor([0, 0, 1, 1, 2, 2, 0])
        e = t.evidence(z, torch.tensor([0, 0, 0, 1, 1, 1, 2]), 3)
        self.assertTrue(torch.equal(t.shrink(e, 0), e['own']))
        torch.testing.assert_close(t.shrink(e, 1), e['native'], atol=1e-7, rtol=0)
        self.assertEqual(e['counts'][0].tolist(), [1, 1, 0])
        self.assertEqual(t.groups(e).tolist(), [2, 2, 1, 1, 2, 2, 2])
        self.assertEqual(float(e['m'][-1]), 0)

    def test_exact_mass_controls(self):
        t = self.module('targets')
        m = torch.tensor([.2, .8, 0, .4], dtype=torch.double)
        w = torch.tensor([0, 1, .25, .5], dtype=torch.double)
        r = t.match_mass(m, w, w.flip(0))
        self.assertLess(abs(float((m*r).sum()-(m*w).sum())), 1e-12)
        self.assertAlmostEqual(float((m*t.global_mass(m, w)).sum()), float((m*w).sum()))

    def test_seeded_initialization_and_rng_restore(self):
        p = self.module('protocol')
        def sample(seed):
            p.seed_all(seed)
            m = p.AlignedMLP(4, 3, 5, 64, .5)
            return p.state_hash(m.state_dict())
        self.assertEqual(sample(1), sample(1))
        self.assertNotEqual(sample(1), sample(2))
        p.seed_all(2)
        state = p.rng_get()
        first = (torch.rand(3), np.random.rand(3), p.random.random())
        p.rng_set(state)
        torch.testing.assert_close(first[0], torch.rand(3))
        np.testing.assert_array_equal(first[1], np.random.rand(3))
        self.assertEqual(first[2], p.random.random())

    def test_label_and_write_whitelists(self):
        p = self.module('protocol')
        for forbidden in ['test_y', 'clean_train', 'noise_rate', 'noise_type']:
            with self.assertRaises(ValueError):
                p.training_data({'x': torch.zeros(1), forbidden: 1})
        with self.assertRaises(ValueError):
            p.output(ROOT.parent/'lcss_strength_support_20261024'/'forbidden')
        self.assertEqual(p.canonical(ROOT/'fits'/'a'), 'fits/a')

    def test_selector_epoch_and_causal_prefix(self):
        s = self.module('selection')
        base = dict(support=0, strength=0, valid_correct=8, valid_total=10,
                    valid_nll=.4, best_epoch=10, epoch=25)
        a, b = dict(base, arm='a'), dict(base, arm='b', best_epoch=9)
        self.assertEqual(min([a,b], key=s.rank_key)['arm'], 'b')
        b['best_epoch'] = 10
        self.assertEqual(min([a,b], key=s.rank_key)['arm'], 'a')
        anchor = dict(base, arm='S32_L1.0', support=32, strength=1, valid_correct=0)
        self.assertIn('S32_L1.0', s.probe_survivors([a,b,anchor], 2, 25))
        with self.assertRaises(ValueError):
            s.probe_survivors([dict(a, epoch=200),anchor], 1, 25)
        with self.assertRaises(ValueError):
            s.rank_key(dict(a, test_accuracy=1))

    def test_ddof1_and_seed_first_macro(self):
        a = self.module('analysis_stats')
        self.assertEqual(a.mean_std([1,2,3]), (2.,1.))
        rows = [dict(seed=s, value=s+c) for s in [0,1,2] for c in [0,10]]
        self.assertEqual(a.seed_macro(rows, 'value'), [5.,6.,7.])

    def test_storage_cleanup_is_guarded(self):
        f = self.module('fit')
        with self.assertRaises(ValueError):
            f.cleanup_resume(ROOT.parent/'lcss_strength_support_20261024'/'fits')

    def test_queue_contains_only_authorized_120_cells(self):
        r = self.module('run')
        cells = list(r.cells())
        self.assertEqual(len(cells), 120)
        self.assertEqual(len(set(cells)), 120)
        self.assertEqual({x[1] for x in cells}, {0,1,2})
        self.assertEqual(len(r.candidates()), 18)
        self.assertEqual(r.candidates()['S32_L1.0'], (32,1.0))

    def test_fit_input_hash_cache_has_seed_and_condition(self):
        import ast
        source = (ROOT/'fit.py').read_text()
        self.assertIn("metadata.get('dataset'), seed, metadata.get('condition'), backbone", source)
        self.assertNotIn('seed_all(0)', source)

    def test_c1_full_grid_requires_frozen_probe(self):
        r = self.module('run')
        with self.assertRaises(RuntimeError):
            r.require_probe(ROOT/'nonexistent_probe.json')

    def test_analysis_requires_global_freeze_before_label_reader(self):
        a = self.module('analyze')
        with self.assertRaises(RuntimeError):
            a.require_global_freeze(ROOT/'nonexistent_freeze.json')

    def test_paired_diff_is_seed_matched(self):
        a = self.module('analyze')
        rows = [dict(dataset='g',seed=s,backbone='R0',condition='clean',stage='D1',
                arm=arm,test_accuracy=value,validation=value,test_macro_f1=value)
                for s in [0,1,2] for arm,value in [('GROUP_SELECTED',s+1),('CE',s)]]
        diffs = a.make_pairs(rows)
        ce = [r for r in diffs if r['reference']=='CE']
        self.assertEqual([r['test_accuracy_diff_pp'] for r in ce],[100.,100.,100.])

    def test_audit_rejects_last_tie_checkpoint(self):
        a = self.module('audit')
        log = [dict(epoch=1,validation_correct=5,validation_nll=.5),
               dict(epoch=2,validation_correct=6,validation_nll=.4),
               dict(epoch=3,validation_correct=6,validation_nll=.3)]
        self.assertEqual(a.first_best(log)['epoch'],2)

    def test_audit_prefix_rank_from_raw_logs(self):
        a = self.module('audit')
        logs = [dict(epoch=i,validation_correct=i%3,validation_nll=1/i) for i in range(1,201)]
        self.assertEqual(a.first_best(logs[:25])['epoch'],2)

    def test_historical_h_normalization_and_container(self):
        p = self.module('protocol')
        raw = torch.tensor([[3.,4.],[0.,2.]])
        expected = torch.tensor([[.6,.8],[0.,1.]])
        torch.testing.assert_close(p.normalize_h(raw),expected)
        torch.testing.assert_close(p.normalize_h({'H':raw}),expected)

    def test_clean_cache_verification_rejects_mismatch(self):
        p = self.module('protocol')
        self.assertTrue(hasattr(p,'verify_clean_labels'))
        labels = torch.tensor([0,1,2,0])
        geo = {'train':torch.tensor([1,3]),'valid':torch.tensor([2])}
        pack = {'observed':torch.tensor([1,0]),'valid_y':torch.tensor([2])}
        p.verify_clean_labels(pack,geo,labels)
        with self.assertRaises(RuntimeError):
            p.verify_clean_labels(dict(pack,observed=torch.tensor([0,0])),geo,labels)

    def test_missing_historical_clean_uses_e1_cache(self):
        p = self.module('protocol')
        for name in ('coauthorcs','amazonpho'):
            pack = p.cell_inputs(name,0,'clean')
            self.assertIn('observed',pack)
            self.assertNotIn('test_y',pack)


if __name__ == '__main__':
    unittest.main()
