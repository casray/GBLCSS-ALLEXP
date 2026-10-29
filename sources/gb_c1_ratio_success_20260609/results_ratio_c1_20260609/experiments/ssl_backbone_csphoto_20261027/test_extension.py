import torch
import torch.nn.functional as F
from artifacts import PROTOCOL as P
import graphecl as ge
from run import ecl_loss_kwargs

def test_photo_projector_matches_official_two_linear_relu_order():
    torch.manual_seed(7)
    m=ge.Model(5,dict(hidden=8,layers=1,num_mlp=2)).eval()
    h=torch.randn(9,8)
    reference=F.linear(F.relu(F.linear(h,m.projector[0].weight,m.projector[0].bias)),m.projector[2].weight,m.projector[2].bias)
    assert torch.equal(m.projector(h),reference)

def test_photo_loss_parameters_are_used_and_exact_gradient():
    cfg=P['graphecl']['dataset_configs']['amazonpho']
    kw=ecl_loss_kwargs(cfg)
    assert kw==dict(tau=.5,lam=.1,lambda_loss=1.)
    torch.manual_seed(13)
    edge=ge.with_loops(torch.tensor([[0,1,2,3],[1,2,3,0]]),6)
    h=torch.randn(6,4,dtype=torch.double,requires_grad=True)
    q=torch.randn(6,4,dtype=torch.double,requires_grad=True)
    ref=ge.dense_loss(h,q,edge,**kw);ref.backward()
    gh,gq=h.grad.clone(),q.grad.clone();h.grad=None;q.grad=None
    value=ge.backward_loss(h,q,edge,chunk=2,**kw)
    assert abs(value-ref.item())<1e-10
    assert torch.allclose(h.grad,gh,atol=1e-10,rtol=0)
    assert torch.allclose(q.grad,gq,atol=1e-10,rtol=0)

def test_two_layer_projection_reaches_both_layers():
    torch.manual_seed(15)
    m=ge.Model(5,dict(hidden=8,layers=1,num_mlp=2))
    x=torch.randn(9,5);edge=ge.with_loops(torch.tensor([[0,1,2],[1,2,3]]),9)
    h,q=m(x,ge.adjacency(edge,9,x.dtype));ge.backward_loss(h,q,edge,chunk=3,lam=.1)
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.projector.parameters())

def test_extension_counts_and_unchanged_supervision():
    assert P['datasets']==['coauthorcs','amazonpho'] and P['seeds']==[0,1]
    methods=[m for stage in P['stage_order'] for m in stage]
    assert len(methods)*len(P['datasets'])*len(P['seeds'])==16
    assert P['expected']['total_prediction_cells']==16*8+16+4*7
    assert P['downstream']['support']==32 and P['downstream']['beta']==.5
    assert P['downstream']['knn_kmeans_controls'] is False

def test_single_projection_state_layout_retained():
    torch.manual_seed(21);a=ge.Model(5,dict(hidden=8,layers=2))
    torch.manual_seed(21);b=ge.Model(5,dict(hidden=8,layers=2,num_mlp=1))
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
