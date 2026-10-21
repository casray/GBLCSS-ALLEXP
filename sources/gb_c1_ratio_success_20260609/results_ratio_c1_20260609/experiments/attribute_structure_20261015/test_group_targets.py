import unittest
import numpy as np
import torch
import common as c
from group_targets import group_vote
from gblcss.targets import build_leave_one_out_targets


class VoteTests(unittest.TestCase):
    def test_exact_existing_diagnostic_and_formal_target(self):
        rng=np.random.default_rng(4)
        for n in (9,40):
            a=rng.integers(0,4,n);z=rng.integers(0,3,n)
            actual=group_vote(a,z,3)
            expected=c.d.vote(c.d.group_weights(a),z,3)
            for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
            t=c.d.mix(actual[1],actual[2],z).astype(np.float32)
            formal=build_leave_one_out_targets(torch.from_numpy(z),a,np.arange(n),3).soft_target.numpy()
            np.testing.assert_allclose(t,formal,rtol=0,atol=1.2e-7)

    def test_self_vote_exclusion_and_low_support(self):
        r,q,alpha,support,eff=group_vote(np.array([0,0,1]),np.array([0,1,0]),2)
        np.testing.assert_array_equal(support,[1,1,0]);np.testing.assert_array_equal(alpha,[0,0,0])
        np.testing.assert_array_equal(r,[[0,1],[1,0],[.5,.5]])

    def test_tie_uses_distribution(self):
        r,q,alpha,_,_=group_vote(np.zeros(5,dtype=int),np.array([0,0,1,1,2]),3)
        np.testing.assert_array_equal(q[-1],r[-1])


if __name__=='__main__':unittest.main()
