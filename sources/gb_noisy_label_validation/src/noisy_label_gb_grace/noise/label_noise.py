from __future__ import annotations

from pathlib import Path

import torch

from noisy_label_gb_grace.utils import PROJECT_ROOT, ensure_parent, mask_to_index, set_seed


def label_noise_path(
    dataset: str,
    noise_type: str,
    noise_rate: float,
    split_seed: int,
    noise_seed: int,
    root: Path = PROJECT_ROOT,
    split_name: str | None = None,
) -> Path:
    rate_name = f"{noise_rate:g}".replace(".", "p")
    split_token = split_name or f"split{split_seed}"
    return (
        root
        / "results"
        / "noisy_labels"
        / dataset
        / f"{noise_type}_{rate_name}"
        / f"{split_token}_noise{noise_seed}.pt"
    )


def corrupt_symmetric_labels(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    noise_rate: float,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1]")
    set_seed(seed)
    labels_cpu = labels.detach().cpu().long()
    noisy = labels_cpu.clone()
    train_idx = mask_to_index(train_mask.detach().cpu())
    num_flip = int(round(float(noise_rate) * int(train_idx.numel())))
    if num_flip == 0:
        changed = torch.empty(0, dtype=torch.long)
    else:
        perm = torch.randperm(train_idx.numel())[:num_flip]
        changed = train_idx[perm]
        for idx in changed.tolist():
            old = int(labels_cpu[idx])
            candidates = [klass for klass in range(num_classes) if klass != old]
            sampled = candidates[int(torch.randint(len(candidates), (1,)).item())]
            noisy[idx] = sampled
    metadata = {
        "noise_type": "symmetric",
        "noise_rate": float(noise_rate),
        "noise_seed": int(seed),
        "num_train": int(train_idx.numel()),
        "num_flipped": int(changed.numel()),
        "changed_indices": changed.tolist(),
    }
    return noisy, metadata


def load_or_create_noisy_labels(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    dataset: str,
    noise_type: str,
    noise_rate: float,
    split_seed: int,
    noise_seed: int,
    root: Path = PROJECT_ROOT,
    split_name: str | None = None,
) -> tuple[torch.Tensor, Path, dict]:
    if noise_type != "symmetric":
        raise NotImplementedError("Gate experiment v1 only supports symmetric noise")
    path = label_noise_path(dataset, noise_type, noise_rate, split_seed, noise_seed, root, split_name)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        return payload["noisy_labels"].long(), path, dict(payload["metadata"])
    noisy, metadata = corrupt_symmetric_labels(
        labels,
        train_mask,
        num_classes,
        noise_rate,
        seed=noise_seed,
    )
    payload = {
        "clean_labels": labels.detach().cpu().long(),
        "noisy_labels": noisy.long(),
        "train_mask": train_mask.detach().cpu().bool(),
        "metadata": {
            **metadata,
            "dataset": dataset,
            "split_seed": int(split_seed),
            "split_name": split_name or f"split{split_seed}",
            "artifact_path": str(path),
        },
    }
    ensure_parent(path)
    torch.save(payload, path)
    return noisy.long(), path, dict(payload["metadata"])
