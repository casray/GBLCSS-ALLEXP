import pytest
import torch
import artifacts as a

def test_hash_gated_reuse(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    folder=tmp_path/'fit'
    a.save(folder/'config.json',{'seed':0})
    a.save_tensor(folder/'model.pt',{'w':torch.ones(2)})
    assert not a.reusable(folder,{'seed':0})
    a.complete(folder,{'seed':0},['config.json','model.pt'])
    assert a.reusable(folder,{'seed':0})
    with pytest.raises(RuntimeError):a.reusable(folder,{'seed':1})
    (folder/'config.json').write_text('{}')
    with pytest.raises(RuntimeError):a.reusable(folder,{'seed':0})

def test_r2_native_widths():
    from downstream import _CORE
    for width in (128,256,1024,2048):
        model=_CORE.AlignedMLP(9,3,width,64,.5).eval()
        x=torch.randn(8,9)
        assert model.projection(model.encode(x)).shape==(8,width)
        model.freeze_encoder();before={k:v.clone() for k,v in model.encoder_state().items()}
        model.train_head();loss=model.backbone.layers[-1](model.encode(x)).sum();loss.backward()
        assert all(torch.equal(v,model.encoder_state()[k]) for k,v in before.items())
        assert all(p.grad is None for p in model.encoder_parameters())
