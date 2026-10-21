import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'GB-LCSS_code/src'))
from gblcss.granular_ball_pruning import EvidenceTree, crossfit_targets, optimal_pruning


def toy_tree():
    return EvidenceTree.from_trace(np.repeat(np.arange(4),24), [
        {'accepted':True,'parent_id':0,'child_ids':[1,2]},
        {'accepted':True,'parent_id':1,'child_ids':[3,4]},
        {'accepted':True,'parent_id':2,'child_ids':[5,6]}])


class PruningTests(unittest.TestCase):
    def test_subtree_lookahead_and_tie_parent(self):
        children={0:(1,2),1:(3,4),2:(5,6)}
        cost={0:10.,1:8.,2:8.,3:1.,4:1.,5:1.,6:1.}
        selected,value=optimal_pruning(children,cost)
        self.assertEqual(selected,[3,4,5,6]); self.assertEqual(value,4.)
        self.assertEqual(optimal_pruning({0:(1,2)},{0:2.,1:1.,2:1.})[0],[0])

    def test_invalid_children_never_leave_uncovered_nodes(self):
        self.assertEqual(optimal_pruning({0:(1,2)},{0:4.,1:float('inf'),2:0.})[0],[0])

    def test_tree_exact_membership_and_rejected_steps(self):
        tree=toy_tree()
        np.testing.assert_array_equal(tree.members[0],np.arange(96))
        np.testing.assert_array_equal(tree.members[1],np.arange(48))
        self.assertEqual(tree.parent[3],1)

    def test_self_label_never_changes_own_evidence(self):
        tree=toy_tree(); z=np.repeat([0,1,0,1],24)
        a=crossfit_targets(tree,z,2,seed=8,min_voters=2)
        for i in [0,25,50,95]:
            other=z.copy(); other[i]=1-other[i]
            b=crossfit_targets(tree,other,2,seed=8,min_voters=2)
            for arm in a['bundles']:
                for key in ['q','alpha','r','support']:
                    np.testing.assert_allclose(a['bundles'][arm][key][i],b['bundles'][arm][key][i],atol=1e-12)
                np.testing.assert_array_equal(a['components'][arm]['region'][:,i],b['components'][arm]['region'][:,i])

    def test_average_targets_and_support(self):
        out=crossfit_targets(toy_tree(),np.repeat([0,1,0,1],24),2,seed=9,min_voters=2)
        for arm,bundle in out['bundles'].items():
            np.testing.assert_allclose(bundle['target'].sum(1),1.)
            self.assertTrue(np.all(bundle['target']>=0))
            one=np.eye(2)[np.repeat([0,1,0,1],24)]
            np.testing.assert_allclose(bundle['target'],(1-bundle['alpha'][:,None])*one+bundle['alpha'][:,None]*bundle['q'])
            c=out['components'][arm]
            np.testing.assert_allclose(bundle['target'],c['target'].mean(0))
            self.assertTrue(np.all(c['support'][c['alpha']>0]>=2))

    def test_root_tie_and_insufficient_voters_fall_back(self):
        tree=EvidenceTree.from_trace(np.zeros(6,dtype=int),[])
        z=np.array([0,1,0,1,0,1])
        out=crossfit_targets(tree,z,2,seed=0,min_voters=16)
        for b in out['bundles'].values():
            np.testing.assert_array_equal(b['target'],np.eye(2)[z])
            self.assertTrue(np.all(b['alpha']==0))

    def test_folds_are_label_free_and_deterministic(self):
        tree=toy_tree(); z=np.arange(96)%2
        a=crossfit_targets(tree,z,2,seed=15,min_voters=2)
        b=crossfit_targets(tree,1-z,2,seed=15,min_voters=2)
        np.testing.assert_array_equal(a['folds'],b['folds'])
        for fold in range(3): self.assertEqual(int((a['folds']==fold).sum()),32)
        for decision in a['decisions']:
            q,s,v=[np.array(decision[k]) for k in ['query','selector','voters']]
            self.assertFalse(np.intersect1d(q,s).size or np.intersect1d(q,v).size or np.intersect1d(s,v).size)
            self.assertEqual(len(np.unique(np.r_[q,s,v])),96)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): EvidenceTree.from_trace(np.array([0,2]),[])
        with self.assertRaises(ValueError): crossfit_targets(toy_tree(),np.zeros(95,dtype=int),2,seed=0)
        with self.assertRaises(ValueError): crossfit_targets(toy_tree(),np.full(96,2),2,seed=0)
        with self.assertRaises(ValueError): crossfit_targets(toy_tree(),np.zeros(96,dtype=int),2,seed=0,delta=0)


if __name__=='__main__': unittest.main()
