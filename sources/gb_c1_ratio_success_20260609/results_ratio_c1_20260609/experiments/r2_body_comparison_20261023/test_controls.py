import unittest
import torch
from run import loss,st

class Controls(unittest.TestCase):
    def test_gce_extreme_gradient_finite(self):
        x=torch.tensor([[1000.,-1000.]],requires_grad=True)
        value=loss(x,torch.tensor([1]),torch.tensor([[0.,1.]]),'GCE');value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(x.grad).all())
    def test_sals_nontraining_labels_hidden(self):
        z=torch.tensor([0,1,-1,-1]);train=torch.tensor([0,1]);edges=torch.tensor([[0,1,2,3],[1,0,0,1]])
        a=st.build_sals_targets(z,train,edges,num_nodes=4,n_classes=2)[0]
        z[2:]=torch.tensor([1,0]);b=st.build_sals_targets(z,train,edges,num_nodes=4,n_classes=2)[0]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.allclose(a.sum(1),torch.ones(4)))
    def test_als_relevance_has_gradient(self):
        rel=torch.eye(2,requires_grad=True);logits=torch.tensor([[2.,-1.],[-1.,2.]],requires_grad=True)
        value=st.als_eq5_loss(logits,torch.tensor([0,1]),rel.softmax(1),alpha=.1,kl_weight=.001)
        value.backward();self.assertGreater(float(rel.grad.abs().sum()),0.)
    def test_freeze_and_classifier_initialization(self):
        if not torch.cuda.is_available():self.skipTest('GPU required')
        from run import AlignedMLP,model,state_hash
        x=torch.randn(12,5,device='cuda');base=AlignedMLP(5,3,4,64,.5).cuda();state=base.state_dict()
        a=model(x,3,4,0,state);b=model(x,3,4,0,state)
        self.assertEqual(state_hash(a.deployment_state()),state_hash(b.deployment_state()))
        before=state_hash(a.encoder_state());a.train_head();a(x)[0].sum().backward()
        self.assertEqual(before,state_hash(a.encoder_state()))
        self.assertTrue(all(p.grad is None for p in a.encoder_parameters()))

if __name__=='__main__':unittest.main()
