import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rules import choose, support_from_probe, distinct_supports, refinement_cost, freeze_required


def result(acc=.8, f1=.7, nll=.5, epoch=10):
    return dict(validation=acc, valid_macro_f1=f1, valid_nll=nll, best_epoch=epoch)


class RulesTests(unittest.TestCase):
    def test_accuracy_then_nll_then_epoch_then_order(self):
        pool = [('CE', result()), ('G', result(epoch=5))]
        self.assertEqual(choose(pool, 'accuracy'), 'G')
        pool[1] = ('G', result())
        self.assertEqual(choose(pool, 'accuracy'), 'CE')
        pool[1] = ('G', result(nll=.4, epoch=20))
        self.assertEqual(choose(pool, 'accuracy'), 'G')

    def test_balanced_applies_to_baseline_and_expanded(self):
        pool = [('CE', result(.8,.8)), ('L1', result(.85,.6))]
        self.assertEqual(choose(pool, 'accuracy'), 'L1')
        self.assertEqual(choose(pool, 'balanced'), 'CE')
        self.assertEqual(choose(pool+[('G', result(.9,.5))], 'balanced'), 'CE')

    def test_no_test_score_selection(self):
        pool = [('CE', dict(result(), test_accuracy=1)), ('G', dict(result(), test_accuracy=0))]
        self.assertEqual(choose(pool, 'accuracy'), 'CE')
        with self.assertRaises(ValueError):
            choose(pool, 'test_accuracy')

    def test_ce_branch_and_support_dedup(self):
        self.assertIsNone(support_from_probe('CE'))
        self.assertEqual(support_from_probe('S128_L0.5'), 128)
        self.assertEqual(distinct_supports('CE'), [32])
        self.assertEqual(distinct_supports('S32_L1.0'), [32])
        self.assertEqual(distinct_supports('S8_L1.0'), [32,8])

    def test_refinement_accounts_for_unfinished_candidates(self):
        probe = dict(rounds=[dict(survivors=['S8_L1.0','S8_L0.5','S32_L1.0']),
                             dict(survivors=['S8_L1.0','S32_L1.0'])])
        cost = refinement_cost(probe, 8)
        self.assertEqual(cost['additional_global_updates'], 175+125)
        self.assertEqual(cost['upstream_probe_with_ce'], 1200)
        self.assertEqual(cost['expanded_total_updates'], 1200+300+800)
        self.assertEqual(refinement_cost(probe, None)['expanded_total_updates'],1200)

    def test_freeze_required(self):
        with self.assertRaises(RuntimeError):
            freeze_required(Path(__file__).parent/'missing_freeze.json')

    def test_output_and_label_boundaries(self):
        from protocol import ROOT,output,training_data
        with self.assertRaises(ValueError):
            output(ROOT.parent/'d1_c1_confirmation_20261024/illegal.json')
        with self.assertRaises(ValueError):
            training_data(dict(test_y=1))

    def test_target_endpoints_and_same_tree_groups(self):
        import torch
        from targets import evidence,shrink,groups
        z=torch.tensor([0,0,1,1,1,0])
        e=evidence(z,torch.tensor([0,0,0,1,1,1]),2)
        self.assertTrue(torch.equal(shrink(e,0),e['own']))
        self.assertTrue(torch.allclose(shrink(e,1),e['native']))
        self.assertEqual(e['support'].tolist(),[2.]*6)
        self.assertEqual(groups(e).tolist(),[2,2,1,2,2,1])

    def test_seed_first_sample_std(self):
        from analyze import summarize
        rows=[dict(seed=s,dataset=d,condition='uniform0.4',metric=v) for s,vs in [(0,[0.,2.]),(1,[2.,4.]),(2,[4.,6.])] for d,v in zip(['a','b'],vs)]
        out=summarize(rows,['dataset','condition'],['metric'],[0,1,2])
        macro=next(r for r in out if r['dataset']=='ALL')
        self.assertEqual(macro['metric_mean'],3.)
        self.assertEqual(macro['metric_std_ddof1'],2.)

    def test_matched_global_and_expanded_pool_sizes(self):
        from run import GLOBALS,GROUP_ARMS
        self.assertEqual(len(GLOBALS),4)
        self.assertEqual(len(GLOBALS+GROUP_ARMS),8)
        self.assertEqual(GLOBALS[0],'CE')

    def test_atomic_save_retries_transient_permission_error(self):
        from unittest.mock import patch
        from protocol import ROOT,save,read
        original=Path.replace
        calls=[]
        def transient(path,target):
            calls.append(1)
            if len(calls)==1:
                raise PermissionError('temporary Windows reader lock')
            return original(path,target)
        path=ROOT/'tests/runtime/atomic.json'
        with patch.object(Path,'replace',transient):
            save(path,dict(ok=True))
        self.assertEqual(read(path),dict(ok=True))
        self.assertEqual(len(calls),2)


if __name__ == '__main__':
    unittest.main()
