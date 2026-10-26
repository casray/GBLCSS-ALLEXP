import unittest
import torch
from fit_pipeline import AlignedMLP, copy_encoder, loss_value, state_hash, set_seed


class FitTests(unittest.TestCase):
    def test_encoder_copy_preserves_fresh_head(self):
        set_seed(0)
        a=AlignedMLP(5,3,7)
        initial={k:v.clone() for k,v in a.state_dict().items()}
        b=AlignedMLP(5,3,7)
        copy_encoder(a,b.state_dict())
        self.assertTrue(torch.equal(a.state_dict()['backbone.layers.1.weight'],initial['backbone.layers.1.weight']))
        self.assertTrue(torch.equal(a.state_dict()['backbone.layers.0.weight'],b.state_dict()['backbone.layers.0.weight']))

    def test_frozen_encoder_bn(self):
        model=AlignedMLP(5,3,7)
        model.freeze_encoder()
        before=state_hash(model.encoder_state())
        model.train_head()
        x=torch.randn(12,5)
        opt=torch.optim.Adam(model.classifier_parameters())
        loss_value(model(x)[0],torch.ones(12,3)/3).backward()
        opt.step()
        self.assertEqual(before,state_hash(model.encoder_state()))
        self.assertFalse(model.backbone.training)

    def test_gce_finite_extreme(self):
        logits=torch.tensor([[1000.,-1000.,0.],[-1000.,1000.,0.]],requires_grad=True)
        z=torch.tensor([1,0])
        loss=loss_value(logits,None,z,True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_soft_ce_equivalent(self):
        logits=torch.randn(8,3)
        z=torch.randint(3,(8,))
        target=torch.nn.functional.one_hot(z,3).float()
        self.assertTrue(torch.allclose(loss_value(logits,target),torch.nn.functional.cross_entropy(logits,z)))


if __name__=='__main__':
    unittest.main()
