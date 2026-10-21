import unittest
import numpy as np
import torch
from target_tools import ball_bundle, gated_bundle
from gblcss.targets import build_leave_one_out_targets


class TargetTests(unittest.TestCase):
    def test_targets_match_formal_ties_and_insufficient_support(self):
        labels=np.array([0,1,2,0,0,2,1]); groups=np.array([0,0,0,1,1,1,2])
        actual=ball_bundle(groups,labels,3)
        formal=build_leave_one_out_targets(torch.tensor(labels),groups,np.arange(7),3,min_train_per_ball=2,beta=.5)
        np.testing.assert_allclose(actual['target'],formal.soft_target.numpy(),atol=1e-7)
        np.testing.assert_allclose(actual['alpha'],formal.alpha.numpy(),atol=1e-7)
        self.assertEqual(actual['alpha'][-1],0)

    def test_own_vote_does_not_enter_q(self):
        labels=np.array([0,1,1,2]); groups=np.zeros(4,dtype=int)
        first=ball_bundle(groups,labels,3); labels[0]=2
        second=ball_bundle(groups,labels,3)
        np.testing.assert_array_equal(first['q'][0],second['q'][0])
        self.assertEqual(first['alpha'][0],second['alpha'][0])

    def test_gate_keeps_q_and_probability_normalization(self):
        labels=np.array([0,1,1,2]); base=ball_bundle(np.zeros(4,dtype=int),labels,3)
        new=gated_bundle(base,labels,np.array([1.,2.,3.,4.]))
        np.testing.assert_array_equal(base['q'],new['q'])
        self.assertTrue(np.all(new['alpha']<=base['alpha']))
        np.testing.assert_allclose(new['target'].sum(1),1.)


if __name__=='__main__':unittest.main()
