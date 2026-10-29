import importlib
import unittest
import numpy as np
import torch


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module('mechanisms')

    def test_loo_counts_and_own_noise(self):
        z=np.array([0,1,0,2]);a=np.array([0,0,0,1])
        out=self.m.vote_details(z,a,3)
        np.testing.assert_array_equal(out['counts'],[[1,1,0],[2,0,0],[1,1,0],[0,0,0]])
        z[0]=2;other=self.m.vote_details(z,a,3)
        np.testing.assert_array_equal(out['counts'][0],other['counts'][0])
        np.testing.assert_array_equal(out['q'][0],other['q'][0])
        self.assertEqual(out['alpha'][0],other['alpha'][0])
        self.assertTrue(np.isfinite(out['target']).all())
        np.testing.assert_allclose(out['target'].sum(1),1)

    def test_folds_and_outside_label_invariance(self):
        train=np.arange(12);a,b=self.m.split_folds(train,4)
        self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(train))
        group=np.arange(12)%2;z=np.arange(12)%3
        p=self.m.fold_package(train[a],z[a],group,3)
        z[b]=(z[b]+1)%3;q=self.m.fold_package(train[a],z[a],group,3)
        for k in p:np.testing.assert_array_equal(p[k],q[k])

    def test_soft_pairs_exclude_self_and_preserve_shuffle_rows(self):
        t=np.array([[.9,.1],[.7,.3],[.1,.9],[.2,.8]],np.float32)
        p=self.m.positive_weights(t,np.array([0,0,1,1]))
        np.testing.assert_allclose(p.sum(1),1);self.assertTrue((p.diagonal()==0).all())
        np.testing.assert_array_equal(p[:2,2:],0)
        rng=np.random.default_rng(7);s=t[rng.permutation(4)]
        self.assertEqual(sorted(map(tuple,t)),sorted(map(tuple,s)))

    def test_aux_loss_manual_and_gradients(self):
        h=torch.tensor([[1.,.2],[.5,1.],[-1.,.2],[.2,-1.]],requires_grad=True)
        w=torch.tensor([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
        loss=self.m.soft_contrastive(h,w,torch.arange(4),.5)
        z=torch.nn.functional.normalize(h,dim=1);s=z@z.T/.5
        expected=torch.stack([-s[i,(i^1)]+torch.logsumexp(torch.cat((s[i,:i],s[i,i+1:])),0) for i in range(4)]).mean()
        torch.testing.assert_close(loss,expected)
        g=torch.autograd.grad(loss,h,retain_graph=True)[0];e=torch.autograd.grad(expected,h)[0]
        torch.testing.assert_close(g,e)
        self.assertTrue(torch.isfinite(g).all())

    def test_zero_positive_loss_is_finite_zero(self):
        h=torch.randn(3,4,requires_grad=True)
        loss=self.m.soft_contrastive(h,torch.zeros(3,3),torch.arange(3),.5)
        self.assertEqual(loss.item(),0);loss.backward();self.assertEqual(h.grad.abs().sum().item(),0)

    def test_region_relation_loo_and_normalized(self):
        h=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        a=np.array([0,0,1,1]);p=self.m.region_distribution(h,a,np.arange(4),.5)
        np.testing.assert_allclose(p.sum(1),1)
        own=0.;other=-1/np.sqrt(2)
        expected=np.exp(own/.5)/(np.exp(own/.5)+np.exp(other/.5))
        self.assertAlmostEqual(p[0,0],expected,places=6)

    def test_support_change_hand_calculation(self):
        old=np.array([0,0,1,1]);new=np.array([0,1,0,1])
        x=self.m.support_changes(old,new)
        np.testing.assert_array_equal(x['intersection'],0)
        np.testing.assert_array_equal(x['changed'],1)
        np.testing.assert_array_equal(x['jaccard'],0)


if __name__=='__main__':unittest.main()
