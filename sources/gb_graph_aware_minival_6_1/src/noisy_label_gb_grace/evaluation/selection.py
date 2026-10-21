"""Commit-style ball-guided sample selection trainer (S2-redesign-v1).

Unlike the soft-target / reweight pipeline (`build_method_targets` + `train_frozen_classifier`),
this trainer trains a frozen-embedding linear head on HARD noisy labels but, each epoch, only the
*selected* (trusted) train nodes contribute to the loss. The selected set is the top-`k` train nodes
by a combined trust score `z(-loss) + z(rel)`, where:
  - `-loss`  = small-loss trick (co-teaching style), recomputed each epoch,
  - `rel`    = STATIC per-node ball-consensus reliability (fraction of same-ball other train nodes
               sharing this node's NOISY label) — built WITHOUT clean labels.

The keep-ratio schedule (warmup -> ramp -> hold at `rho = 1 - noise_rate`) is FIXED a priori and
IDENTICAL for the method and every control; only the `rel` source differs. So controls are
size-matched by construction and any gap is attributable to the reliability signal.

Clean labels are used ONLY for val/test accuracy (model selection on clean val, the benchmark
protocol) and for the eval-only selection-precision / noise-excluded-recall diagnostics. They never
enter the selection score or the training targets.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.evaluation.classifier import LinearClassifier
from noisy_label_gb_grace.utils import mask_to_index, set_seed


def ball_consensus_score(
    grouping: dict,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node ball consensus reliability (no clean labels).

    rel[i] = fraction of OTHER train nodes in i's ball whose NOISY label equals i's noisy label.
    Returns (rel[N] in [0,1], covered[N] bool). Uncovered nodes (empty ball, no other train node)
    get rel=0 and covered=False (treated as neutral downstream).
    """
    labels = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    group_ids = grouping["group_ids"].detach().cpu().long()
    n = int(labels.numel())
    rel = torch.zeros(n, dtype=torch.float32)
    covered = torch.zeros(n, dtype=torch.bool)
    for node in mask_to_index(train_mask).tolist():
        gid = int(group_ids[node])
        region = (group_ids == gid).nonzero(as_tuple=False).view(-1)
        local_train = region[train_mask[region]]
        local_train = local_train[local_train != node]
        if local_train.numel() == 0:
            continue
        counts = torch.bincount(labels[local_train], minlength=num_classes).float()
        probs = counts / counts.sum().clamp_min(1.0)
        rel[node] = float(probs[labels[node]].item())
        covered[node] = True
    return rel, covered


def knn_train_consensus_score(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    k: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """No-ball top-k consensus: rel[i] = fraction of i's k nearest TRAIN neighbors (cosine, excl self)
    sharing i's noisy label. Used for the size/coverage-matched no-ball `topk` selection control."""
    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    labels = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    n = int(labels.numel())
    rel = torch.zeros(n, dtype=torch.float32)
    covered = torch.zeros(n, dtype=torch.bool)
    if train_idx.numel() <= 1 or k <= 0:
        return rel, covered
    train_emb = emb[train_idx]
    sim = train_emb @ train_emb.t()
    sim.fill_diagonal_(-2.0)
    kk = min(int(k), int(train_idx.numel()) - 1)
    topk = sim.topk(kk, dim=1).indices
    for ti in range(int(train_idx.numel())):
        node = int(train_idx[ti])
        nbr = train_idx[topk[ti]]
        counts = torch.bincount(labels[nbr], minlength=num_classes).float()
        probs = counts / counts.sum().clamp_min(1.0)
        rel[node] = float(probs[labels[node]].item())
        covered[node] = True
    return rel, covered


def ball_density_reliability(
    grouping: dict,
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    knn_density_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GBmooer-style ball-level density structure reliability (NO clean labels).

    For a node i assigned to ball B:
        rel_i = density_score(B) * compactness(i, B) * noisy_label_consensus(B)
    where
        density_score(B)         = mean over B of each member's avg kNN cosine similarity
                                   (intra-ball local density on the unit hypersphere; geometry only),
        compactness(i, B)        = (1 + cos(h_i, mu_B)) / 2   (node closeness to ball center),
        noisy_label_consensus(B) = max class fraction of B's TRAIN nodes' NOISY labels.

    This is a *ball-level structural* signal that a per-node kNN consensus cannot express. Clean labels
    never enter. Nodes in empty/degenerate balls -> rel=0, covered=False (neutral).
    """
    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    labels = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    group_ids = grouping["group_ids"].detach().cpu().long()
    n = int(labels.numel())
    rel = torch.zeros(n, dtype=torch.float32)
    covered = torch.zeros(n, dtype=torch.bool)

    # global kNN cosine-density per node (for density_score aggregation), geometry only
    sim_all = emb @ emb.t()
    sim_all.fill_diagonal_(-2.0)
    kk = max(1, min(int(knn_density_k), n - 1))
    knn_density = sim_all.topk(kk, dim=1).values.clamp(-1.0, 1.0).mean(dim=1)  # [n] in [-1,1]
    knn_density = ((knn_density + 1.0) / 2.0)  # -> [0,1]

    for group in grouping["groups"]:
        members = torch.tensor(group, dtype=torch.long)
        if members.numel() == 0:
            continue
        local_train = members[train_mask[members]]
        # density_score(B): mean kNN density over ball members
        density_score = float(knn_density[members].mean().item())
        # noisy_label_consensus(B): max class fraction among ball's TRAIN noisy labels
        if local_train.numel() > 0:
            counts = torch.bincount(labels[local_train], minlength=num_classes).float()
            noisy_consensus = float((counts.max() / counts.sum().clamp_min(1.0)).item())
        else:
            noisy_consensus = 0.0
        center = emb[members].mean(dim=0)
        center = center / center.norm().clamp_min(1e-8)
        for node in local_train.tolist():
            compactness = float(((1.0 + torch.dot(emb[node], center).clamp(-1.0, 1.0)) / 2.0).item())
            rel[node] = density_score * compactness * noisy_consensus
            covered[node] = True
    return rel, covered


def _zscore(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Z-score `values` over `mask` (or all). Entries outside mask -> 0."""
    out = torch.zeros_like(values)
    sel = values if mask is None else values[mask]
    if sel.numel() == 0:
        return out
    mean = sel.mean()
    std = sel.std(unbiased=False).clamp_min(1e-8)
    if mask is None:
        return (values - mean) / std
    out[mask] = (values[mask] - mean) / std
    return out


def keep_ratio_at_epoch(epoch: int, warmup_epochs: int, ramp_epochs: int, keep_ratio_final: float) -> float:
    """Fixed co-teaching remember-rate schedule: 1.0 during warmup, linear ramp to `keep_ratio_final`,
    then hold. Deterministic; no tuning."""
    if epoch <= warmup_epochs:
        return 1.0
    if ramp_epochs <= 0:
        return float(keep_ratio_final)
    frac = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return float(1.0 - frac * (1.0 - keep_ratio_final))


def train_frozen_classifier_with_selection(
    embeddings: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
    reliability: torch.Tensor | None = None,
    reliability_covered: torch.Tensor | None = None,
    use_reliability: bool = True,
    keep_ratio_final: float = 0.6,
    warmup_epochs: int = 20,
    ramp_epochs: int = 20,
    epochs: int = 500,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: torch.device | None = None,
    clean_oracle: bool = False,
) -> dict:
    """Train a frozen-embedding linear head with per-epoch small-loss x ball-reliability selection.

    If `clean_oracle` is True, the selected set is fixed to the train nodes whose noisy label already
    equals the clean label (perfect selection upper bound; DIAGNOSTIC ONLY).
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = embeddings.detach().float().to(device)
    clean = clean_labels.detach().long().to(device)
    noisy = noisy_labels.detach().long().to(device)
    train_mask = train_mask.detach().bool().to(device)
    val_mask = val_mask.detach().bool().to(device)
    test_mask = test_mask.detach().bool().to(device)
    train_idx = mask_to_index(train_mask)
    val_idx = mask_to_index(val_mask)
    test_idx = mask_to_index(test_mask)
    n_train = int(train_idx.numel())

    # static z-scored reliability over train (covered nodes only; uncovered -> 0 neutral)
    rel_z = torch.zeros(n_train, dtype=torch.float32, device=device)
    if use_reliability and reliability is not None:
        rel = reliability.detach().float().to(device)[train_idx]
        cov = (
            reliability_covered.detach().bool().to(device)[train_idx]
            if reliability_covered is not None
            else torch.ones(n_train, dtype=torch.bool, device=device)
        )
        rel_z = _zscore(rel, cov)

    oracle_keep = (noisy[train_idx] == clean[train_idx]) if clean_oracle else None

    model = LinearClassifier(x.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_test = 0.0
    best_epoch = 0
    stale = 0
    selected_at_best = torch.ones(n_train, dtype=torch.bool, device=device)
    loss_at_best = torch.zeros(n_train, dtype=torch.float32, device=device)

    def _acc(logits, idx):
        if idx.numel() == 0:
            return 0.0
        return float((logits[idx].argmax(dim=-1) == clean[idx]).float().mean().item())

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        per_loss = F.cross_entropy(logits[train_idx], noisy[train_idx], reduction="none")

        if clean_oracle:
            keep_mask = oracle_keep
        elif epoch <= warmup_epochs:
            keep_mask = torch.ones(n_train, dtype=torch.bool, device=device)
        else:
            rho = keep_ratio_at_epoch(epoch, warmup_epochs, ramp_epochs, keep_ratio_final)
            k = max(1, int(round(rho * n_train)))
            neg = (-per_loss).detach()
            score = _zscore(neg) + (rel_z if use_reliability else 0.0)
            keep_idx = torch.topk(score, k).indices
            keep_mask = torch.zeros(n_train, dtype=torch.bool, device=device)
            keep_mask[keep_idx] = True

        if keep_mask.any():
            loss = per_loss[keep_mask].mean()
        else:
            loss = per_loss.mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(x)
            val_acc = _acc(eval_logits, val_idx)
            test_acc = _acc(eval_logits, test_idx)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            selected_at_best = keep_mask.detach().clone()
            loss_at_best = per_loss.detach().clone()
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        train_acc = _acc(final_logits, train_idx)
        val_acc = _acc(final_logits, val_idx)
        test_acc = _acc(final_logits, test_idx)

    # ---- selection diagnostics (clean labels eval-only) ----
    is_clean = (noisy[train_idx] == clean[train_idx])
    is_noisy = ~is_clean
    sel = selected_at_best
    selected_count = int(sel.sum().item())
    if selected_count > 0:
        selection_precision = float(is_clean[sel].float().mean().item())
    else:
        selection_precision = math.nan
    n_noisy = int(is_noisy.sum().item())
    if n_noisy > 0:
        noise_excluded_recall = float(((~sel) & is_noisy).float().sum().item() / n_noisy)
    else:
        noise_excluded_recall = math.nan

    # loss_mean over selected vs excluded train nodes (at best epoch)
    loss_mean_selected = float(loss_at_best[sel].mean().item()) if selected_count > 0 else math.nan
    excluded = ~sel
    loss_mean_excluded = float(loss_at_best[excluded].mean().item()) if int(excluded.sum()) > 0 else math.nan

    # per-class (by CLEAN class, eval-only): of clean train nodes in class c, fraction selected;
    # of noisy train nodes whose CLEAN class is c, fraction excluded.
    clean_tr = clean[train_idx]
    clean_selected_rate_by_class = []
    noisy_excluded_rate_by_class = []
    selected_clean_class_distribution = []
    selected_noisy_class_distribution = []
    for c in range(num_classes):
        in_c = (clean_tr == c)
        clean_c = in_c & is_clean
        noisy_c = in_c & is_noisy
        clean_selected_rate_by_class.append(
            float((clean_c & sel).float().sum().item() / max(1, int(clean_c.sum())))
            if int(clean_c.sum()) > 0 else math.nan
        )
        noisy_excluded_rate_by_class.append(
            float((noisy_c & excluded).float().sum().item() / max(1, int(noisy_c.sum())))
            if int(noisy_c.sum()) > 0 else math.nan
        )
        # class distribution within the SELECTED set, split clean vs noisy
        selected_clean_class_distribution.append(int((clean_c & sel).sum().item()))
        selected_noisy_class_distribution.append(int((noisy_c & sel).sum().item()))

    return {
        "best_val_acc": float(best_val),
        "test_acc_at_best_val": float(best_test),
        "best_epoch": int(best_epoch),
        "train_acc_at_best_state": float(train_acc),
        "val_acc_at_best_state": float(val_acc),
        "test_acc_at_best_state": float(test_acc),
        "selected_count": selected_count,
        "selected_coverage": float(selected_count / max(1, n_train)),
        "selection_precision": selection_precision,
        "noise_excluded_recall": noise_excluded_recall,
        "train_count": n_train,
        "loss_mean_selected": loss_mean_selected,
        "loss_mean_excluded": loss_mean_excluded,
        "clean_selected_rate_by_class": clean_selected_rate_by_class,
        "noisy_excluded_rate_by_class": noisy_excluded_rate_by_class,
        "selected_clean_class_distribution": selected_clean_class_distribution,
        "selected_noisy_class_distribution": selected_noisy_class_distribution,
    }
