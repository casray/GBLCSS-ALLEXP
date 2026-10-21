import unittest
import tempfile
from pathlib import Path
import numpy as np
import common as c


class ProtocolTests(unittest.TestCase):
    def test_binary_conditions_not_duplicate_pair(self):
        self.assertEqual(set(c.conditions('tolokers')), {'clean','symmetric02','symmetric04'})
        self.assertEqual(c.conditions('tolokers')['symmetric04'], ['uniform',.4])

    def test_multiclass_legacy_conditions(self):
        self.assertEqual(c.conditions('roman')['pair'], ['pair',.4])

    def test_gate_requires_both_references_and_noise_conditions(self):
        rows=[]
        for seed in range(5):
            for noise in ('uniform','pair'):
                for arm,b in [('R_TREE',.1),('X_TREE',.2),('SHUFFLE_TREE',.15)]:
                    rows.append(dict(dataset='a',seed=seed,noise=noise,arm=arm,target_brier=b))
        self.assertTrue(c.compute_gate(rows)['a']['passed'])
        for r in rows:
            if r['noise']=='pair' and r['arm']=='SHUFFLE_TREE':r['target_brier']=.09
        self.assertFalse(c.compute_gate(rows)['a']['passed'])

    def test_gate_rejects_missing_seed(self):
        with self.assertRaises(ValueError):
            c.compute_gate([dict(dataset='a',seed=0,noise='uniform',arm='R_TREE',target_brier=.1)])

    def test_manifest_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('one');m=Path(tmp)/'manifest.json'
            c.freeze_manifest(m,[p]);c.freeze_manifest(m,[p]);p.write_text('two')
            with self.assertRaises(ValueError):c.freeze_manifest(m,[p])


if __name__=='__main__':unittest.main()
