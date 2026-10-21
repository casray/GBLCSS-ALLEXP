import unittest
import numpy as np
from split_diagnostic import diagnose_tree


class SplitDiagnosticTests(unittest.TestCase):
    def test_local_gains_telescope_and_oracle_can_skip_bad_early_split(self):
        final=np.repeat(np.arange(4),16)
        truth=np.r_[np.zeros(48,dtype=int),np.ones(16,dtype=int)]
        observed=truth.copy();observed[16:28]=1
        trace=[dict(accepted=True,parent_id=0,child_ids=[1,2],parent_sse=64,sse_gain=10,actual_count_after=2),
               dict(accepted=True,parent_id=1,child_ids=[3,4],parent_sse=20,sse_gain=2,actual_count_after=3),
               dict(accepted=True,parent_id=2,child_ids=[5,6],parent_sse=20,sse_gain=2,actual_count_after=4)]
        result=diagnose_tree(final,trace,np.arange(64),observed,truth,2)
        self.assertAlmostEqual(sum(x['global_brier_gain'] for x in result['splits']),result['root_brier']-result['final_brier'])
        self.assertLess(result['oracle_brier'],result['best_prefix_brier'])
        self.assertEqual(result['oracle_balls'],3)
        self.assertLess(result['splits'][1]['global_brier_gain'],0)

    def test_single_ball_has_no_splits_or_invented_gain(self):
        result=diagnose_tree(np.zeros(20,dtype=int),[],np.arange(20),np.r_[np.zeros(10,dtype=int),np.ones(10,dtype=int)],np.zeros(20,dtype=int),2)
        self.assertEqual(result['splits'],[])
        self.assertEqual(result['oracle_balls'],1)
        self.assertEqual(result['oracle_brier'],result['root_brier'])


if __name__=='__main__':unittest.main()
