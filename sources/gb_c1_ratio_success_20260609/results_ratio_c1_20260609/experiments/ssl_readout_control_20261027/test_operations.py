import json
from pathlib import Path
import torch
import pytest
from operations import tasks,normalize_rows,fit_pca,project_pca

def test_64_new_heads_and_no_native_training():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    rows=tasks(cfg)
    assert len(rows)==64
    assert len({tuple(r.values()) for r in rows})==64
    assert {r['representation'] for r in rows}=={'pca64','r2_unit64'}

def test_normalization_preserves_zero_and_direction():
    x=torch.tensor([[0.,0.],[3.,4.]])
    assert torch.equal(normalize_rows(x),torch.tensor([[0.,0.],[.6,.8]]))

def test_pca_known_spectrum_and_no_whitening():
    h=torch.tensor([[2.,0.],[-2.,0.],[0.,1.],[0.,-1.]],dtype=torch.double)+torch.tensor([5.,7.])
    state=fit_pca(h,1);z=project_pca(h,state)
    assert torch.allclose(z.flatten(),torch.tensor([2.,-2.,0.,0.],dtype=torch.double),atol=1e-12)
    assert abs(state['retained_fraction']-.8)<1e-12
    assert torch.allclose(state['mean'],torch.tensor([5.,7.],dtype=torch.double))

def test_full_pca_reconstruction_and_sign_repeat():
    torch.manual_seed(8);h=torch.randn(30,5,dtype=torch.double)
    a=fit_pca(h,5);b=fit_pca(h,5);z=project_pca(h,a)
    assert torch.allclose(z@a['components'].T+a['mean'],h,atol=1e-10,rtol=0)
    assert torch.equal(a['components'],b['components'])
    assert abs(a['retained_fraction']-1)<1e-12

@pytest.mark.parametrize('k',[0,6])
def test_invalid_dimension_rejected(k):
    with pytest.raises(ValueError):fit_pca(torch.ones(8,5),k)

def test_nonfinite_rejected():
    with pytest.raises(ValueError):fit_pca(torch.tensor([[float('nan'),0.],[0.,1.]]),1)
