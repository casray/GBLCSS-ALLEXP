import importlib.util, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from test_family import dense,official

class ProTest(unittest.TestCase):
    def test_implementation_exists(self): self.assertIsNotNone(importlib.util.find_spec('progcl'))
    def test_weighted_tail_gradient(self):
        from contrastive_family import backward_loss
        for n,k in [(7,3),(8,4)]:
            for dtype in [torch.float32,torch.float64]:
                torch.manual_seed(3)
                a=torch.randn(n,4,dtype=dtype,requires_grad=True); b=torch.randn_like(a,requires_grad=True)
                aa=a.detach().clone().requires_grad_(); bb=b.detach().clone().requires_grad_()
                B=torch.rand(n,n-1,dtype=torch.float64)+.1
                l=dense(a,b,.7,B);l.backward(); got=backward_loss(aa,bb,.7,k,B.numpy())
                tol=1e-7 if dtype==torch.float64 else 1e-4
                self.assertAlmostEqual(l.item(),got,delta=tol/10)
                torch.testing.assert_close(a.grad,aa.grad,atol=tol,rtol=tol);torch.testing.assert_close(b.grad,bb.grad,atol=tol,rtol=tol)
    def test_official_fit_and_resume(self):
        from progcl import build_weights,load_weights,weights_for_epoch
        up=official('ProGCL/pGRACE/model.py'); torch.manual_seed(4)
        a=torch.randn(8,5,dtype=torch.float64); b=torch.randn_like(a)
        norm=lambda t:torch.nn.functional.normalize(t,dim=1)
        sim=(norm(a)@norm(b).T+1)/2
        np.random.seed(9); ix=np.random.randint(0,8,100)
        mix=up.BetaMixture1D(10,torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([2.,1.],dtype=torch.float64),torch.tensor([.95,.05],dtype=torch.float64))
        mix.fit(sim[:,ix].flatten().clone())
        neg=sim[~torch.eye(8,dtype=torch.bool)].reshape(8,7)
        expected=(mix.posterior(neg,0)*neg).numpy()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'B.npy'; np.random.seed(9)
            self.assertIsNone(weights_for_epoch(399,a,b,path))
            actual=weights_for_epoch(400,a,b,path,chunk=3)
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)
            state=np.random.get_state(); restored=weights_for_epoch(401,a,b,path)
            np.testing.assert_array_equal(actual,restored);np.testing.assert_array_equal(state[1],np.random.get_state()[1])
            np.testing.assert_array_equal(actual,load_weights(path))
            with self.assertRaises(FileExistsError):build_weights(a,b,path)
            del actual,restored
    def test_missing_frozen_weights_fail(self):
        from progcl import weights_for_epoch
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):weights_for_epoch(401,torch.ones(2,3),torch.ones(2,3),Path(td)/'B.npy')


class FullReferenceTest(unittest.TestCase):
    def test_official_full_weighted_parameter_gradients(self):
        from types import SimpleNamespace
        from contrastive_family import Model,backward_loss
        import copy
        up=official('ProGCL/pGRACE/model.py');torch.manual_seed(21)
        m=Model(5,dict(hidden=4,projection=6,activation='prelu')).double()
        ref=up.GRACE(up.Encoder(5,4,torch.nn.PReLU()),4,6,.7).double();ref.load_state_dict(m.state_dict())
        x=torch.randn(7,5,dtype=torch.float64);e=torch.tensor([[0,1,2,2,3,4],[1,2,0,3,4,5]])
        up.B=torch.rand(7,6,dtype=torch.float64)+.1
        loss=ref.loss(ref(x,e),ref(x*.8,e),401,SimpleNamespace(epoch_start=400,mode='weight'),None);loss.backward()
        actual=backward_loss(m.project(m(x,e)),m.project(m(x*.8,e)),.7,3,up.B)
        self.assertAlmostEqual(actual,loss.item(),delta=1e-8)
        for a,b in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-7)
    def test_hash_corruption_and_zero_B_fail(self):
        from progcl import build_weights,load_weights
        from contrastive_family import backward_loss
        a=torch.randn(7,4,requires_grad=True);b=torch.randn(7,4,requires_grad=True)
        with self.assertRaises(FloatingPointError):backward_loss(a,b,.7,3,np.zeros((7,6)))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'B.npy';arr=build_weights(a,b,p);del arr
            with p.open('r+b') as f:f.seek(-1,2);f.write(b'X')
            with self.assertRaises(ValueError):load_weights(p)

if __name__=='__main__': unittest.main()

