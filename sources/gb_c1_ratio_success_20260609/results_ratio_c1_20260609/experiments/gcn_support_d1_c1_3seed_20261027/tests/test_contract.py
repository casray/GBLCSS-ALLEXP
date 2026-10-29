import importlib.util
from pathlib import Path
import sys
import unittest

CODE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gcn_expand_test',CODE/'run.py')
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)


class Contracts(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(len(run.cells()),45)
        self.assertEqual(sum(run.historical(n,s,c) for n,s,c in run.cells()),9)
        self.assertEqual(len(run.GRID),42)

    def test_write_scope(self):
        self.assertEqual(run.fitmodule.ROOT,run.ROOT)
        with self.assertRaises(ValueError):run.p.output(run.OLD/'forbidden.txt')

    def test_seed_identity(self):
        self.assertNotEqual(run.cid('cora',0,'clean'),run.cid('cora',1,'clean'))
        self.assertNotEqual(run.cid('cora',1,'uniform0.4'),run.cid('cora',1,'uniform0.6'))

    def test_no_test_fields(self):
        with self.assertRaises(ValueError):run.p.training_data({'test_y':None})
        row=dict(arm='CE',valid_correct=1,valid_total=2,valid_nll=.2,mean_lambda=0.,support=0,best_epoch=1,epoch=25)
        with self.assertRaises(ValueError):run.rules.rank_key(dict(row,test_accuracy=1.))

    def test_prefix_first_best(self):
        row=run.rules.prefix_score([dict(epoch=1,validation_correct=2,validation_nll=.8),
            dict(epoch=2,validation_correct=2,validation_nll=.1)],2,'CE',0,0.,3)
        self.assertEqual(row['best_epoch'],1);self.assertEqual(row['valid_nll'],.8)

    def test_target_tie_and_zero_strength(self):
        import torch
        e=run.targets.evidence(torch.tensor([0,1,2]),torch.zeros(3,dtype=torch.long),3)
        self.assertTrue(torch.equal(run.targets.shrink(e,0),e['own']))
        self.assertTrue(torch.equal(run.rules.weights(e,dict(kind='group',agree=0.,conflict=0.)),torch.ones(3,dtype=torch.float64)))


if __name__=='__main__':unittest.main()
