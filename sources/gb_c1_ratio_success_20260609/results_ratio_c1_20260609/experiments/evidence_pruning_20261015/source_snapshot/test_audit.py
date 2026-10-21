import unittest
import numpy as np
from test_pruning import toy_tree
from gblcss.granular_ball_pruning import crossfit_targets
from pruning_audit import reference_vote, verify_output


class AuditTests(unittest.TestCase):
    def test_independent_recomputation_accepts_full_output(self):
        tree=toy_tree();z=np.repeat([0,1,0,1],24)
        out=crossfit_targets(tree,z,2,seed=1,min_voters=2)
        stats=verify_output(tree,z,2,out,min_voters=2,delta=.1)
        self.assertEqual(stats['components_checked'],36)

    def test_corrupted_target_is_rejected(self):
        tree=toy_tree();z=np.arange(96)%2
        out=crossfit_targets(tree,z,2,seed=1,min_voters=2)
        out['bundles']['PRUNE']['target'][0,0]+=.01
        with self.assertRaises(AssertionError):verify_output(tree,z,2,out,min_voters=2,delta=.1)

    def test_reference_tie_no_sharpening_and_all_vote_leave_one_out(self):
        tree=toy_tree();z=np.arange(96)%2
        ref=reference_vote(tree,z,2,np.array([0]),np.array([1,2]),[0],2,False)
        np.testing.assert_allclose(ref['q'],[[.5,.5]])
        self.assertEqual(ref['alpha'][0],0.)
        ref=reference_vote(tree,z,2,np.array([0]),np.arange(1,96),[0],2,True)
        np.testing.assert_allclose(ref['r'],[[47/95,48/95]])


if __name__=='__main__':unittest.main()
