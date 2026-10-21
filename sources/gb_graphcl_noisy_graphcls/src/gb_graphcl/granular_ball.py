from __future__ import annotations

from collections import Counter

import torch
from sklearn.cluster import KMeans
from torch import Tensor


def _purity(labels: Tensor, members: list[int]) -> float:
    vals = labels[torch.tensor(members, dtype=torch.long)].cpu().tolist()
    if not vals:
        return 0.0
    return max(Counter(vals).values()) / len(vals)


def build_granular_balls(
    embeddings: Tensor,
    noisy_labels: Tensor,
    train_indices: list[int],
    purity_threshold: float = 0.7,
    min_samples: int = 5,
    max_size: int = 100,
):
    """Build recursive purity-guided granular balls over train embeddings."""
    train_indices = list(map(int, train_indices))
    if not train_indices:
        raise ValueError("train_indices must not be empty")
    emb_cpu = embeddings.detach().cpu().float()
    labels_cpu = noisy_labels.detach().cpu().long()
    final_balls: list[list[int]] = []

    def split_ball(members: list[int]):
        purity = _purity(labels_cpu, members)
        if len(members) <= min_samples:
            final_balls.append(members)
            return
        if purity >= purity_threshold and len(members) <= max_size:
            final_balls.append(members)
            return
        if len(members) <= 2:
            final_balls.append(members)
            return

        points = emb_cpu[torch.tensor(members, dtype=torch.long)].numpy()
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            assign = km.fit_predict(points)
        except Exception:
            final_balls.append(members)
            return
        left = [m for m, a in zip(members, assign) if a == 0]
        right = [m for m, a in zip(members, assign) if a == 1]
        if not left or not right:
            final_balls.append(members)
            return
        split_ball(left)
        split_ball(right)

    split_ball(train_indices)

    pos_by_index = {idx: pos for pos, idx in enumerate(train_indices)}
    group_ids = torch.empty(len(train_indices), dtype=torch.long)
    centers = []
    sizes = []
    purities = []
    for gid, members in enumerate(final_balls):
        for idx in members:
            group_ids[pos_by_index[idx]] = gid
        member_tensor = torch.tensor(members, dtype=torch.long)
        centers.append(emb_cpu[member_tensor].mean(dim=0))
        sizes.append(len(members))
        purities.append(_purity(labels_cpu, members))

    centers_tensor = torch.stack(centers, dim=0) if centers else torch.empty((0, emb_cpu.size(1)))
    return group_ids, centers_tensor, sizes, purities


def shuffle_group_ids(group_ids: Tensor, seed: int) -> Tensor:
    """Shuffle group assignments while preserving group sizes."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ids = group_ids.detach().cpu().long()
    perm = torch.randperm(len(ids), generator=gen)
    return ids[perm]


def random_group_ids(num_samples: int, group_sizes: list[int], seed: int) -> Tensor:
    """Randomly assign samples into groups with the provided sizes."""
    if sum(group_sizes) != num_samples:
        raise ValueError("sum(group_sizes) must equal num_samples")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ordered = []
    for gid, size in enumerate(group_sizes):
        ordered.extend([gid] * int(size))
    ids = torch.tensor(ordered, dtype=torch.long)
    return ids[torch.randperm(num_samples, generator=gen)]

