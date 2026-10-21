from __future__ import annotations

from pathlib import Path

import torch


def load_frozen_representation(path: str | Path, expected_nodes: int | None = None) -> torch.Tensor:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(artifact, dict):
        for key in ("H", "h", "embeddings", "embedding"):
            if key in artifact:
                artifact = artifact[key]
                break
    if not isinstance(artifact, torch.Tensor) or artifact.ndim != 2:
        raise TypeError("the frozen representation must be a two-dimensional tensor")
    if expected_nodes is not None and int(artifact.shape[0]) != int(expected_nodes):
        raise ValueError("the representation and graph contain different node counts")
    values = artifact.float()
    return values / values.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
