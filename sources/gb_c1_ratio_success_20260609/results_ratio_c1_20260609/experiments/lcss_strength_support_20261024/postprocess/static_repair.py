"""Specification repair: all six static gate quantities use train-only scaling."""
import torch
from meta_gate import static_features as original_static, gate_features as original_features


def static_features(e):
    z = e['z']
    freq = torch.bincount(z, minlength=e['own'].shape[1]).double()/len(z)
    value = freq[z]
    scaled = (value-value.mean())/value.std(unbiased=False).clamp_min(1e-8)
    return torch.cat([original_static(e), scaled[:, None]], 1)


def gate_features(e, p, static=None):
    static = static_features(e).to(p) if static is None else static.to(p)
    f = original_features(e, p, static=static[:, :5])
    f[:, 8] = static[:, 5]
    return f.detach()
