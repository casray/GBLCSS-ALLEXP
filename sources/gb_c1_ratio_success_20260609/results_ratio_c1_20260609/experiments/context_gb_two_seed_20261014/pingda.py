"""PiNGDA noise generators adapted for the local weighted-GRACE runner.

The generator blocks follow the public PiNGDA node-task implementation. The
edge head input is fixed to the actual concatenated embedding width (64 + 64),
and the relaxed edge weights retain a length-one dimension for one-edge graphs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EdgeNoiseGenerator(nn.Module):
    """Generate differentiable edge weights from endpoint feature embeddings."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.mlp_feat_model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        _validate_graph_inputs(x, edge_index, self.input_dim)
        node_embedding = self.mlp_feat_model(x)
        src, dst = edge_index[0].long(), edge_index[1].long()
        edge_embedding = torch.cat((node_embedding[src], node_embedding[dst]), dim=-1)
        return self.mlp_edge_model(edge_embedding)

    def apply_gumbel_softmax(self, logits: Tensor, device: torch.device | str | None = None) -> Tensor:
        """Apply PiNGDA's temperature-1 binary Concrete edge relaxation.

        The upstream implementation names this Gumbel softmax; for a binary
        gate it is implemented by adding logistic noise to the edge logits and
        applying sigmoid. No thresholding or hard edge deletion is performed.
        """
        if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.size(-1) != 1):
            raise ValueError("edge logits must have shape [E] or [E, 1]")
        if device is not None and torch.device(device) != logits.device:
            raise ValueError("device must match the edge-logit device")
        bias = 1e-4
        uniform = torch.rand_like(logits).mul_(1.0 - 2.0 * bias).add_(bias)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid(logits + logistic_noise).reshape(-1)


class FeatNoiseGenerator(nn.Module):
    """Generate a learned diagonal-Gaussian additive feature perturbation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc_mean = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.fc_variance = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def freeze(self, flag: bool = True) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = not flag

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        if feat.ndim != 2 or feat.size(-1) != self.input_dim:
            raise ValueError(f"feat must have shape [N, {self.input_dim}]")
        scale = self.fc_variance(feat).abs()
        mean = self.fc_mean(feat)
        return mean, scale

    def sampling(self, mean: Tensor, variance: Tensor) -> Tensor:
        """Sample mean + scale * N(0, I), matching upstream sampling code."""
        if mean.shape != variance.shape:
            raise ValueError("mean and variance tensors must have the same shape")
        if mean.device != variance.device or mean.dtype != variance.dtype:
            raise ValueError("mean and variance tensors must share device and dtype")
        return mean + variance * torch.randn_like(mean)


def build_generators(input_dim: int, device: torch.device | str) -> tuple[EdgeNoiseGenerator, FeatNoiseGenerator]:
    """Build PiNGDA's edge and feature generators on ``device``."""
    edge_gen = EdgeNoiseGenerator(input_dim).to(device)
    feat_gen = FeatNoiseGenerator(input_dim, hidden_dim=256).to(device)
    return edge_gen, feat_gen


def sample_augmentation(
    x: Tensor,
    edges: Tensor,
    edge_gen: EdgeNoiseGenerator | None,
    feat_gen: FeatNoiseGenerator,
    *,
    learnable_edge: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Return one feature-noised view and learned soft edge weights.

    This function preserves the computation graph so a node-level contrastive
    loss can update both generators. In the author's two-view training loop,
    the first view uses feature noise with the original topology; only the
    second view receives the learned edge weights returned here.
    """
    if feat_gen.input_dim != x.size(-1):
        raise ValueError("feature-generator input dimension must match x")
    mean, scale = feat_gen(x)
    x_aug = x + feat_gen.sampling(mean, scale)
    if not learnable_edge:
        return x_aug, None
    if edge_gen is None:
        raise ValueError("edge_gen is required when learnable_edge is True")
    if edge_gen.input_dim != x.size(-1):
        raise ValueError("edge-generator input dimension must match x")
    edge_logits = edge_gen(x, edges)
    edge_weight = edge_gen.apply_gumbel_softmax(edge_logits)
    return x_aug, edge_weight


def _validate_graph_inputs(x: Tensor, edge_index: Tensor, input_dim: int) -> None:
    if x.ndim != 2 or x.size(-1) != input_dim:
        raise ValueError(f"x must have shape [N, {input_dim}]")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if x.device != edge_index.device:
        raise ValueError("x and edge_index must be on the same device")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.size(0)):
        raise ValueError("edge_index contains a node index outside x")
