import copy,unittest
import numpy as np
import torch
from common import ge,state_hash
from mechanisms import fold_package,positive_weights,soft_contrastive
from crossfit import one_step


class CrossFitTests(unittest.TestCase):
    def test_combined_chunked_gradient_matches_dense_objective(self):
        torch.manual_seed(3)
        cfg=dict(hidden=4,layers=2,num_mlp=1,temperature=.5,lam=.1,lambda_loss=1.)
        x=torch.randn(6,3);edge=ge.with_loops(torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]]),6);adj=ge.adjacency(edge,6,x.dtype)
        a=ge.Model(3,cfg);b=copy.deepcopy(a);ids=torch.tensor([0,1,3,4]);q=torch.arange(4)
        target=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]])
        w=torch.from_numpy(positive_weights(target,np.array([0,0,1,1])))
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        one_step(a,oa,x,adj,edge,cfg,ids,w,q,.2)
        b.train();ob.zero_grad();g,t=b(x,adj)
        loss=ge.dense_loss(g,t,edge,tau=.5,lam=.1,lambda_loss=1.)+.2*soft_contrastive(t[ids],w,q,.5)
        loss.backward();ob.step()
        for ka,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[ka],rtol=2e-5,atol=1e-7)

    def test_outside_label_change_leaves_full_update_identical(self):
        torch.manual_seed(6);x=torch.randn(8,3);ids=np.array([0,2,4,6]);labels=np.array([0,1,0,1,1,0,1,0]);a=np.array([0,0,0,0,1,1,1,1])
        before=fold_package(ids,labels[ids],a,2);labels[[1,3,5,7]]^=1;after=fold_package(ids,labels[ids],a,2)
        cfg=dict(hidden=4,layers=1,num_mlp=1,temperature=.5,lam=.1,lambda_loss=1.)
        edge=ge.with_loops(torch.tensor([[0,2,4,6],[2,4,6,0]]),8);adj=ge.adjacency(edge,8,x.dtype)
        initial=ge.Model(3,cfg).state_dict();hashes=[]
        for p in [before,after]:
            model=ge.Model(3,cfg);model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=.001)
            w=torch.from_numpy(positive_weights(p['target'],p['group']))
            one_step(model,opt,x,adj,edge,cfg,torch.from_numpy(ids),w,torch.arange(4),.1)
            hashes.append(state_hash(model.state_dict()))
        self.assertEqual(*hashes)


if __name__=='__main__':unittest.main()
