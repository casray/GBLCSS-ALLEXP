import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rules


class RulesTests(unittest.TestCase):
    def row(self, arm='S32_L1.0', correct=3, nll=.5, mean=1., support=32, epoch=4):
        return dict(arm=arm, valid_correct=correct, valid_total=10, valid_nll=nll,
                    mean_lambda=mean, support=support, best_epoch=epoch, epoch=25)

    def test_grid(self):
        grid = rules.candidates()
        self.assertEqual(len(grid), 42)
        self.assertEqual(sum(v['kind']=='uniform' for v in grid.values()), 18)

    def test_prefix_ignores_future(self):
        logs = [dict(epoch=i,validation_correct=2 if i<26 else 9,
                     validation_nll=.4,validation_macro_f1=.2) for i in range(1,201)]
        r=rules.prefix_score(logs,25,'S32_L1.0',32,1.,10)
        self.assertEqual((r['valid_correct'],r['best_epoch'],r['epoch']), (2,1,25))

    def test_missing_prefix_rejected(self):
        with self.assertRaises(ValueError):
            rules.prefix_score([],25,'S32_L1.0',32,1.,10)

    def test_rank_order(self):
        a=self.row(); b=dict(a,valid_correct=4)
        self.assertLess(rules.rank_key(b),rules.rank_key(a))
        for change in [dict(valid_nll=.4),dict(mean_lambda=.5),dict(support=64),dict(best_epoch=2)]:
            self.assertLess(rules.rank_key(dict(a,**change)),rules.rank_key(a))

    def test_anchor_and_causal_epoch(self):
        rows=[self.row(arm=f'S4_G{i}',correct=7) for i in range(5)]+[self.row()]
        self.assertIn('S32_L1.0',rules.survivors(rows,2,25))
        self.assertEqual(len(rules.survivors(rows,2,25)),2)
        with self.assertRaises(ValueError):
            rules.survivors([dict(r,epoch=200) for r in rows],2,25)

    def test_label_whitelist(self):
        with self.assertRaises(ValueError):
            rules.rank_key(dict(self.row(),test_accuracy=.99))

    def test_fixed_controls_gate(self):
        with self.assertRaises(RuntimeError):
            rules.require_joint_frozen(Path(__file__).parent/'absent.json')

    def test_seed_std(self):
        self.assertEqual(rules.mean_std([1,2,3]),(2.,1.))

    def test_loo_groups_and_new_ties(self):
        import torch
        from targets import evidence, groups, shrink
        e=evidence(torch.tensor([0,0,1,2]),torch.zeros(4).long(),3)
        self.assertTrue(torch.equal(e['counts'][0],torch.tensor([1.,1.,1.]).double()))
        self.assertEqual(groups(e).tolist(),[2,2,1,1])
        w=rules.weights(e,dict(kind='group',agree=0.,conflict=.25))
        self.assertEqual(w.tolist(),[1.,1.,.25,.25])
        self.assertTrue(torch.equal(shrink(e,0),e['own']))
        self.assertTrue(torch.allclose(shrink(e,1),e['native'],atol=3e-7))
        self.assertTrue(torch.allclose(shrink(e,w),e['own']+w.float()[:,None]*(e['native']-e['own']),atol=3e-7))

    def test_own_readonly_and_training_fields(self):
        from protocol import ROOT, output, training_data
        with self.assertRaises(ValueError):
            output(ROOT.parent/'forbidden.txt')
        with self.assertRaises(ValueError):
            training_data(dict(test_y=1))

    def test_global_order_on_complete_tie(self):
        a=self.row(arm='S32_G_A0.0_C1.0')
        b=self.row(arm='S32_L1.0')
        self.assertLess(rules.rank_key(b),rules.rank_key(a))


if __name__=='__main__':
    unittest.main()
