"""Ball-coupling losses for the minimal encoder-coupling smoke (S3 Phase 2).

Goal: let the (trainable) encoder representation be shaped by granular-ball structure, the way CGBC /
GBmooer do — instead of fitting balls on a frozen, ball-agnostic representation.

Key design points (match the approved plan):
- Ball MEMBERSHIP (`group_ids`) is computed periodically from a DETACHED H (so build_dcgb_balls runs on
  a fixed snapshot), but the intra/inter losses are computed from the CURRENT differentiable H, with the
  ball centers RE-COMPUTED here as the mean of differentiable member embeddings. This is what makes the
  gradient flow back into the encoder — `build_dcgb_balls`' own `centers`/`node_distances` come from the
  detached input and are NOT differentiable.
- Only HIGH-QUALITY balls (label-free gate) contribute. Soft, confidence-weighted. NO hard same-ball
  positive, NO view generation.
- Clean labels NEVER enter. noisy labels only weakly *select* inter-ball pairs (different noisy-majority),
  never as a target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def label_free_quality_mask(
    group_ids: torch.Tensor,
    h_detached: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    min_train: int = 3,
    compactness_quantile: float = 0.5,
    entropy_quantile: float = 0.5,
) -> torch.Tensor:
    """Return a per-BALL boolean quality mask (length = num balls = group_ids.max()+1).

    A ball is high-quality iff it jointly satisfies:
      - train-count >= min_train,
      - compactness >= compactness_quantile (mean cosine to ball center; higher = tighter),
      - noisy-label entropy <= entropy_quantile (lower = more label-consistent).
    Compactness uses geometry only; entropy uses NOISY labels only. Clean labels never enter.
    """
    gid = group_ids.detach().cpu().long()
    h = F.normalize(h_detached.detach().cpu().float(), p=2, dim=1)
    labels = noisy_labels.detach().cpu().long()
    tm = train_mask.detach().cpu().bool()
    n_balls = int(gid.max().item()) + 1 if gid.numel() and gid.max() >= 0 else 0
    if n_balls == 0:
        return torch.zeros(0, dtype=torch.bool)

    train_count = torch.zeros(n_balls)
    compactness = torch.zeros(n_balls)
    entropy = torch.full((n_balls,), float("nan"))
    for b in range(n_balls):
        members = (gid == b).nonzero(as_tuple=False).view(-1)
        if members.numel() == 0:
            continue
        center = h[members].mean(dim=0)
        center = center / center.norm().clamp_min(1e-8)
        compactness[b] = float(((h[members] @ center).clamp(-1, 1).mean() + 1.0) / 2.0)
        local_train = members[tm[members]]
        train_count[b] = float(local_train.numel())
        if local_train.numel() > 0:
            counts = torch.bincount(labels[local_train], minlength=num_classes).float()
            p = counts / counts.sum().clamp_min(1.0)
            nz = p[p > 0]
            ent = float(-(nz * nz.log()).sum() / torch.log(torch.tensor(float(num_classes)))) if num_classes > 1 else 0.0
            entropy[b] = ent

    has_train = train_count >= float(min_train)
    if has_train.sum() == 0:
        return torch.zeros(n_balls, dtype=torch.bool)
    comp_thr = torch.quantile(compactness[has_train], float(compactness_quantile))
    ent_vals = entropy[has_train]
    ent_vals = ent_vals[~torch.isnan(ent_vals)]
    ent_thr = torch.quantile(ent_vals, float(entropy_quantile)) if ent_vals.numel() else torch.tensor(1.0)
    quality = has_train & (compactness >= comp_thr) & (~torch.isnan(entropy)) & (entropy <= ent_thr)
    return quality


def ball_coupling_losses(
    h: torch.Tensor,
    group_ids: torch.Tensor,
    quality_mask: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable (L_intra, L_inter) from the CURRENT H and a fixed ball membership.

    L_intra: high-quality-ball train nodes pulled toward their ball center (mean of differentiable member
             embeddings), weighted by confidence (1+cos)/2. Soft, no hard positive.
    L_inter: high-quality ball-center pairs with DIFFERENT noisy-majority kept at cosine margin
             relu(cos(c_a,c_b) - margin). noisy-majority only *selects* pairs, never a target.

    h must be the differentiable encoder output (will be L2-normalized here). group_ids / quality_mask /
    noisy_labels are detached references. Returns zero tensors (grad-safe) if no high-quality ball.
    """
    device = h.device
    hn = F.normalize(h, p=2, dim=1)
    gid = group_ids.detach().to(device).long()
    labels = noisy_labels.detach().to(device).long()
    tm = train_mask.detach().to(device).bool()
    qmask = quality_mask.detach().to(device).bool()

    zero = h.sum() * 0.0  # grad-connected zero
    if qmask.numel() == 0 or qmask.sum() == 0:
        return zero, zero

    good_balls = qmask.nonzero(as_tuple=False).view(-1).tolist()
    centers = []          # differentiable centers (mean of member H)
    center_majorities = []
    intra_terms = []
    for b in good_balls:
        members = (gid == b).nonzero(as_tuple=False).view(-1)
        if members.numel() == 0:
            continue
        center = hn[members].mean(dim=0)
        center = center / center.norm().clamp_min(1e-8)
        centers.append(center)
        # intra: pull high-quality ball's TRAIN nodes to center, confidence-weighted
        local_train = members[tm[members]]
        if local_train.numel() > 0:
            cos = (hn[local_train] @ center).clamp(-1, 1)
            conf = ((cos + 1.0) / 2.0).detach()          # confidence weight, no grad through weight
            intra_terms.append((conf * (1.0 - cos)).sum())
            mc = torch.bincount(labels[local_train], minlength=num_classes)
            center_majorities.append(int(mc.argmax().item()))
        else:
            center_majorities.append(-1)

    if intra_terms:
        n_train_used = sum(int((tm[(gid == b)]).sum()) for b in good_balls)
        L_intra = torch.stack(intra_terms).sum() / max(1, n_train_used)
    else:
        L_intra = zero

    # inter: margin between centers of different-noisy-majority high-quality balls
    if len(centers) >= 2:
        C = torch.stack(centers)                          # [G, d] differentiable
        maj = torch.tensor(center_majorities, device=device)
        sim = C @ C.t()                                   # [G, G]
        G = C.size(0)
        iu = torch.triu_indices(G, G, offset=1, device=device)
        pair_i, pair_j = iu[0], iu[1]
        diff_major = (maj[pair_i] != maj[pair_j]) & (maj[pair_i] >= 0) & (maj[pair_j] >= 0)
        if diff_major.any():
            s = sim[pair_i, pair_j][diff_major]
            L_inter = F.relu(s - float(margin)).mean()
        else:
            L_inter = zero
    else:
        L_inter = zero

    return L_intra, L_inter
