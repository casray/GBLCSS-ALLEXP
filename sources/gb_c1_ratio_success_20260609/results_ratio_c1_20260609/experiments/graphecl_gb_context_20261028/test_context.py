import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

def module():
    spec=importlib.util.find_spec('context')
    assert spec is not None, 'context implementation is required'
    import context
    return context

def fixture():
    edges=[]
    for i in range(8):
        for j in range(8,16):
            if (i+j)%3: edges.extend([(i,j),(j,i)])
    edges.extend([(0,1),(1,0),(16,0),(0,16)])
    edge=np.array(edges,dtype=np.int64).T
    ball=np.array([0]*8+[1]*8+[2,2])
    return edge,ball

def test_leave_one_out_against_independent_reference():
    c=module();e,b=fixture();out=c.context_scores(e,b,seed=3)
    n=len(b);k=b.max()+1;p=np.zeros((n,k));degree=np.bincount(e[1],minlength=n)
    for j,i in e.T:p[i,b[j]]+=1
    p/=np.maximum(degree[:,None],1)
    for idx,(j,i) in enumerate(e.T):
        groups=[np.flatnonzero((b==b[i])&(out['fold']==f)&(np.arange(n)!=i)&(degree>0)) for f in [0,1]]
        active=all(len(g)>=2 for g in groups) and degree[i]>=2
        assert bool(out['active'][i])==active
        if active:
            bg=p[(np.arange(n)!=i)&(degree>0)].mean(0)[b[j]]
            expected=min(p[g].mean(0)[b[j]]/(p[g].mean(0)[b[j]]+bg+1e-12) for g in groups)
            assert out['scores'][idx]==pytest.approx(expected,abs=1e-12)

def test_weight_mass_self_isolates_degree_one_and_bounds():
    c=module();e,b=fixture();out=c.context_scores(e,b,0)
    d=c.center_scores(e,out['scores'],len(b),out['active'])
    degree=np.bincount(e[1],minlength=len(b))
    for eta in [0,.25,1]:
        w=(1+eta*d)/(degree[e[1]]+1)
        total=np.bincount(e[1],weights=w,minlength=len(b))+1/(degree+1)
        np.testing.assert_allclose(total,1,atol=1e-14)
        assert (1+eta*d>=1-eta-1e-14).all() and (1+eta*d<=1+eta+1e-14).all()
    assert (d[e[1]==16]==0).all() and degree[17]==0

def test_shuffle_multiset_cosine_range():
    c=module();e,b=fixture();out=c.context_scores(e,b,4)
    d=c.center_scores(e,out['scores'],len(b),out['active']);s=c.shuffle_delta(e,d,9)
    for i in range(len(b)):np.testing.assert_array_equal(np.sort(s[e[1]==i]),np.sort(d[e[1]==i]))
    h=np.random.default_rng(5).normal(size=(len(b),4));r=c.cosine_scores(h,e)
    assert (r>=0).all() and (r<=1).all()
    np.testing.assert_allclose(c.cosine_scores(np.array([[1.,0],[-1.,0]]),np.array([[0,1],[1,0]])),0)

def test_nontrivial_intervention_exists():
    c=module();e,b=fixture();out=c.context_scores(e,b,2)
    delta=c.center_scores(e,out['scores'],len(b),out['active'])
    assert np.max(abs(delta))>1e-3

def test_chunked_gradients_match_dense_and_zero_dose():
    c=module();torch.manual_seed(7);n=7;e=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
    e=torch.cat([e,torch.arange(n).repeat(2,1)],1)
    rel=torch.zeros(e.shape[1],dtype=torch.float64);rel[0]=.2;rel[3]=-.2
    for eta in [0,.25,1]:
        h=torch.randn(n,5,dtype=torch.float64,requires_grad=True);t=torch.randn(n,5,dtype=torch.float64,requires_grad=True)
        loss=c.dense_loss(h,t,e,rel,eta,tau=.5,lam=.001);loss.backward();expected=[h.grad.clone(),t.grad.clone()]
        hh=h.detach().clone().requires_grad_();tt=t.detach().clone().requires_grad_()
        actual=c.backward_loss(hh,tt,e,rel,eta,chunk=3,tau=.5,lam=.001)
        assert actual==pytest.approx(float(loss.detach()),abs=1e-12)
        torch.testing.assert_close(hh.grad,expected[0],atol=1e-12,rtol=1e-10);torch.testing.assert_close(tt.grad,expected[1],atol=1e-12,rtol=1e-10)
        if eta==0:
            z=torch.nn.functional.normalize(h,dim=1);q=torch.nn.functional.normalize(t,dim=1);src,dst=e
            deg=torch.bincount(dst,minlength=n);a=torch.exp(z@z.T/.5).sum(1);bb=.001*torch.exp(z@q.T/.5).sum(1)
            reference=((-((z[src]*q[dst]).sum(1)/.5)+torch.log(a[dst]+bb[src]))/deg[dst]).sum()/n
            assert float(loss.detach())==float(reference.detach())

def test_invalid_inputs_rejected():
    c=module();e,b=fixture()
    with pytest.raises(ValueError):c.context_scores(np.array([[0],[99]]),b,0)
    with pytest.raises(ValueError):c.context_scores(np.array([[0],[0]]),b,0)

def test_plan_counts_and_matched_doses():
    import json
    cfg=json.loads((Path(__file__).parent/'config.json').read_text())
    assert len(cfg['datasets'])*len(cfg['seeds'])*len(cfg['arms'])==84
    assert len(cfg['datasets'])*len(cfg['seeds'])*(6+6*9)==720
    for suffix in ['025','100']:
        assert cfg['arms']['gb'+suffix]==cfg['arms']['cos'+suffix]==cfg['arms']['shuffle'+suffix]
    assert cfg['clean_gate'] is False

def test_independent_target_singleton_and_tie():
    from analyze import independent_target
    y=np.array([0,1,0,1,2]);ass=np.array([0,0,0,0,1])
    target=independent_target(y,ass,3)
    np.testing.assert_allclose(target.sum(1),1)
    np.testing.assert_array_equal(target[-1],np.array([0,0,1]))
    # For node0: three other votes [1,2,0], alpha=.5, sharpened=[1/6,5/6,0].
    np.testing.assert_allclose(target[0],np.array([7/12,5/12,0]))

def test_optimizer_snapshot_is_immutable_across_replays():
    import copy,runtime
    restore=getattr(runtime,'restore_snapshot',None)
    assert restore is not None, 'restore must isolate mutable optimizer state'
    torch.manual_seed(8);model=torch.nn.Linear(2,2);opt=torch.optim.Adam(model.parameters(),lr=.01);x=torch.ones(3,2)
    model(x).sum().backward();opt.step()
    ck=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=runtime.rng_state()))
    key=next(iter(ck['optimizer']['state']));step=ck['optimizer']['state'][key]['step'].clone();hashes=[]
    for _ in range(2):
        restore(model,opt,ck);opt.zero_grad();model(x).sum().backward();opt.step();hashes.append(runtime.state_hash(model.state_dict()))
    assert torch.equal(ck['optimizer']['state'][key]['step'],step)
    assert hashes[0]==hashes[1]
