import json
import pytest
import torch
from artifacts import inside, ROOT
import graphecl as ge

@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
@pytest.mark.parametrize('n,chunk',[(7,3),(8,4)])
def test_ecl_chunk_gradient(dtype,n,chunk):
    torch.manual_seed(18)
    edge=ge.with_loops(torch.tensor([[0,1,2,3,0,4,1],[1,2,0,4,5,5,6]]),n)
    h=torch.randn(n,5,dtype=dtype,requires_grad=True)
    q=torch.randn(n,5,dtype=dtype,requires_grad=True)
    ref=ge.dense_loss(h,q,edge);ref.backward()
    gh,gq=h.grad.clone(),q.grad.clone();h.grad=None;q.grad=None
    value=ge.backward_loss(h,q,edge,chunk)
    tol=1e-7 if dtype==torch.float64 else 1e-4
    assert abs(value-float(ref.detach()))<(1e-8 if dtype==torch.float64 else 1e-5)
    assert torch.allclose(h.grad,gh,atol=tol,rtol=0)
    assert torch.allclose(q.grad,gq,atol=tol,rtol=0)

def test_ecl_directed_conv():
    edge=ge.with_loops(torch.tensor([[0,0,1,2,3],[1,2,2,0,1]]),5)
    adj=ge.adjacency(edge,5,torch.float64)
    dense=torch.zeros(5,5,dtype=torch.float64)
    src,dst=edge
    for s,d in zip(src,dst):
        dense[d,s]+=1/((src==s).sum()*(dst==d).sum()).double().sqrt()
    assert torch.allclose(adj.to_dense(),dense,atol=1e-12)
    model=ge.Conv(3,4).double();x=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    out=model(x,adj);ref=dense@x@model.weight+model.bias
    assert torch.allclose(out,ref,atol=1e-12)
    g1=torch.autograd.grad(out.square().sum(),(x,model.weight,model.bias),retain_graph=True)
    g2=torch.autograd.grad(ref.square().sum(),(x,model.weight,model.bias))
    assert all(torch.allclose(a,b,atol=1e-10) for a,b in zip(g1,g2))

def test_ecl_both_branches_and_export():
    torch.manual_seed(9);model=ge.Model(3,{'hidden':8,'layers':2})
    x=torch.randn(7,3);edge=ge.with_loops(torch.tensor([[0,1,2],[1,2,0]]),7)
    adj=ge.adjacency(edge,7,x.dtype)
    h,q=model(x,adj);ge.backward_loss(h,q,edge,3)
    for group in [model.convs,model.encoder_target,model.projector]:
        assert sum(float(p.grad.abs().sum()) for p in group.parameters())>0
    model.eval()
    assert torch.equal(model.embedding(x),model.embedding(x))

def test_output_confined():
    assert inside(ROOT/'test.json')==ROOT.resolve()/'test.json'
    with pytest.raises(ValueError): inside(ROOT/'../outside.json')

def test_downstream_loose_width_and_ties():
    from downstream import select_best_epoch,build_support32_targets
    rows=[{'epoch':1,'validation_correct':2},{'epoch':2,'validation_correct':2}]
    assert select_best_epoch(rows)['epoch']==1
    obs=torch.tensor([0,1,1]);assign=torch.tensor([0,1,1])
    targets=build_support32_targets(obs,assign,classes=2)
    assert torch.equal(targets,torch.nn.functional.one_hot(obs,2).float())

@pytest.mark.parametrize('name',['coauthorcs','amazonpho'])
@pytest.mark.parametrize('seed',[0,1])
def test_data(name,seed):
    from data_bridge import load_ssl,load_features,_old
    ssl=load_ssl(name,seed)
    assert set(ssl)=={'x','edge_index','metadata'}
    raw=load_features(name,seed)
    assert torch.equal(raw['train'],_old._validate_split(name,seed,len(ssl['x']))[0][0])
    sums=ssl['x'].abs().sum(1)
    assert torch.all((sums==0)|torch.isclose(sums,torch.ones_like(sums),atol=1e-6))
