import unittest
import torch
import torch.nn.functional as F
from run_job import DoseGCN
from tfr_core import TFRClassifier, load_reference_classes, tfr_loss


@unittest.skipUnless(torch.cuda.is_available(),'GPU required')
class GPUContracts(unittest.TestCase):
    def test_tfr_native_forward_loss_and_gradients_on_gpu(self):
        GCN,Decoder=load_reference_classes()
        torch.manual_seed(41)
        x=torch.rand(7,5,device='cuda');a=torch.tensor([0,0,0,1,1,2,2],device='cuda')
        e=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6],[1,0,2,1,3,2,4,3,5,4,6,5]],device='cuda')
        z=torch.tensor([0,1,2,0,1,2,0],device='cuda');tr=torch.tensor([0,2,4],device='cuda')
        native=GCN(5,16,3,n_layers=2,dropout=.5,norm_info=None).cuda()
        adapter=TFRClassifier(5,16,3,.5,a,0,'independent').cuda();adapter.base.load_state_dict(native.state_dict())
        torch.manual_seed(13);left=native(x,e)
        torch.manual_seed(13);right=adapter(x,e)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        decoder=Decoder(3,16,5,.3,False).cuda()
        settings=dict(beta=.5,beta_dropout_ratio=.3,pseudo_ratio=.5,use_recon_softmax=False)
        loss,_=tfr_loss(right,x,z,tr,decoder,e,.2,settings);loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in [*adapter.parameters(),*decoder.parameters()]))

    def test_mid_dose_matches_dense_float32_and_backward(self):
        x=torch.randn(6,4,device='cuda');a=torch.tensor([0,0,0,1,1,2],device='cuda')
        p=torch.tensor([[.5,.1,0,0,0,0],[.1,.6,.1,0,0,0],[0,.1,.4,.3,0,0],[0,0,.3,.5,.2,0],[0,0,0,.2,.4,.1],[0,0,0,0,.1,.9]],device='cuda')
        h=F.one_hot(a,3).float();pi=h@torch.diag(1/h.sum(0))@h.T;s=.5*p+.5*pi@p@pi
        model=DoseGCN(4,8,3,.5,a,.5,'shared').cuda().eval()
        output=model(x,p.to_sparse());expected=s@model.l2(F.relu(s@model.l1(x)))
        torch.testing.assert_close(output,expected,rtol=2e-5,atol=2e-6)
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(q.grad).all() for q in model.parameters()))


if __name__=='__main__':unittest.main()
