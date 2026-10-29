import copy, importlib.util, os, unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
ROOT = Path(os.environ.get('SSL_PROJECT_ROOT', r'C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609'))
def official(name):
    p=ROOT/'research/ssl_backbone_selection_20261026/upstream'/name
    spec=importlib.util.spec_from_file_location('official_'+p.parent.name, p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def dense(a,b,tau,B=None):
    a,b=F.normalize(a,dim=1),F.normalize(b,dim=1)
    mask=~torch.eye(len(a),dtype=torch.bool)
    out=[]
    for u,v in [(a,b),(b,a)]:
        s=(u@u.T/tau).exp(); c=(u@v.T/tau).exp(); pos=c.diag()
        den=s.sum(1)+c.sum(1)-s.diag() if B is None else pos+((s[mask].view(len(a),-1)+c[mask].view(len(a),-1))*B).sum(1)/B.mean(1)
        out.append(-(pos/den).log())
    return torch.stack(out).mean()

class FamilyTest(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('contrastive_family'))
    def test_loss_and_all_parameter_gradients(self):
        from contrastive_family import Model,backward_loss
        for n,chunk in [(7,3),(8,4)]:
            for dtype in [torch.float64,torch.float32]:
                torch.manual_seed(8)
                m=Model(5,dict(hidden=4,projection=6,activation='prelu')).to(dtype)
                ref=copy.deepcopy(m); x=torch.randn(n,5,dtype=dtype)
                e=torch.stack([torch.arange(n),torch.arange(n).roll(1)])
                l=dense(ref.project(ref(x,e)),ref.project(ref(x*.7,e)),.4); l.backward()
                got=backward_loss(m.project(m(x,e)),m.project(m(x*.7,e)),.4,chunk)
                tol=1e-7 if dtype==torch.float64 else 1e-4
                self.assertAlmostEqual(got,l.item(),delta=tol/10)
                for p,q in zip(m.parameters(),ref.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=tol,rtol=tol)
    def test_official_structure_and_forward(self):
        from contrastive_family import Model
        up=official('GRACE/model.py')
        torch.manual_seed(3); a=Model(5,dict(hidden=4,projection=6,activation='prelu'))
        torch.manual_seed(3); b=up.Model(up.Encoder(5,4,torch.nn.PReLU()),4,6)
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)
        x=torch.randn(7,5); e=torch.tensor([[0,1,3],[1,2,4]])
        torch.testing.assert_close(a(x,e),b(x,e)); torch.testing.assert_close(a.project(a(x,e)),b.projection(b(x,e)))
    def test_degenerate_weights_and_zero_probability(self):
        from contrastive_family import prepare_weights,augment
        x=torch.tensor([[1.,0.,1.],[1.,0.,1.],[0.,0.,0.]])
        e=torch.tensor([[0,1],[1,0]])
        w=prepare_weights(x,e)
        self.assertTrue(all(torch.isfinite(v).all() for v in w.values()))
        self.assertEqual(w['feature'][1],0)
        state=torch.get_rng_state(); xx,ee=augment(x,e,0,0,w)
        torch.testing.assert_close(state,torch.get_rng_state()); torch.testing.assert_close(xx,x);torch.testing.assert_close(ee,e)
    def test_official_degree_weights(self):
        from contrastive_family import prepare_weights
        from torch_geometric.utils import degree,to_undirected
        x=torch.tensor([[1.,0.,1.],[1.,1.,0.],[0.,1.,1.],[0.,1.,0.]])
        e=torch.tensor([[0,1,1,2,1,3],[1,0,2,1,3,1]])
        deg=degree(to_undirected(e)[1],num_nodes=4); w=prepare_weights(x,e)
        for key,s in [('edge',deg[e[1]].log()),('feature',(x.bool().float().T@deg).log())]:
            torch.testing.assert_close(w[key],(s.max()-s)/(s.max()-s.mean()))


class TrajectoryTest(unittest.TestCase):
    def test_official_degree_augmentation_rng_and_warmup(self):
        from contrastive_family import Model,prepare_weights,augment_views,backward_loss
        torch.manual_seed(13)
        x=torch.randint(0,2,(7,5)).float();e=torch.tensor([[0,1,1,2,1,3,3,4,4,5,5,6],[1,0,2,1,3,1,4,3,5,4,6,5]])
        w=prepare_weights(x,e);p=[.2,.4];f=[.3,.2]
        torch.manual_seed(91)
        expected_e=[e[:,torch.bernoulli(1-(w['edge']/w['edge'].mean()*q).clamp(max=.7)).bool()] for q in p]
        for q in f:torch.empty(x.shape[1]).uniform_(0,1)<q
        expected_x=[]
        for q in f:
            mask=torch.bernoulli((w['feature']/w['feature'].mean()*q).clamp(max=.7)).bool()
            xx=x.clone();xx[:,mask]=0;expected_x.append(xx)
        expected_rng=torch.get_rng_state()
        torch.manual_seed(91);views=augment_views(x,e,p,f,w)
        for i,(xx,ee) in enumerate(views):
            torch.testing.assert_close(xx,expected_x[i]);torch.testing.assert_close(ee,expected_e[i])
        torch.testing.assert_close(expected_rng,torch.get_rng_state())
        states=[]
        for method in ['gca_degree','progcl_weight_degree']:
            torch.manual_seed(5);model=Model(5,dict(hidden=4,projection=4,activation='relu'));opt=torch.optim.Adam(model.parameters(),lr=.001)
            torch.manual_seed(91)
            for epoch in [397,398,399]:
                (x1,e1),(x2,e2)=augment_views(x,e,p,f,w);opt.zero_grad()
                backward_loss(model.project(model(x1,e1)),model.project(model(x2,e2)),.4,3);opt.step()
            states.append(([t.detach().clone() for t in model.parameters()],torch.get_rng_state()))
        for a,b in zip(states[0][0],states[1][0]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(states[0][1],states[1][1])
    def test_export_does_not_consume_rng(self):
        from contrastive_family import Model
        model=Model(3,dict(hidden=4,projection=4,activation='relu'));x=torch.ones(4,3);e=torch.tensor([[0,1],[1,2]])
        model.eval();before=torch.get_rng_state()
        with torch.no_grad():
            a=F.normalize(model(x,e),dim=1);b=F.normalize(model(x,e),dim=1)
        torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(before,torch.get_rng_state())

if __name__=='__main__': unittest.main()

