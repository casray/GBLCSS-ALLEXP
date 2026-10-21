"""Differentiable ball projections and fixed training-view GCN operators."""
import math

import torch
from torch import nn
import torch.nn.functional as F


MODES = ("P", "PPI", "PIP", "PIPPI")


def _validate_assignment(assignment, num_nodes, num_groups=None):
    if not isinstance(assignment, torch.Tensor) or assignment.dtype != torch.long:
        raise TypeError("assignment must be a LongTensor")
    if assignment.ndim != 1 or assignment.numel() != num_nodes or num_nodes == 0:
        raise ValueError("assignment must contain one group per node of a nonempty input")
    if assignment.min().item() < 0:
        raise ValueError("assignment must be nonnegative")
    actual_groups = assignment.max().item() + 1
    if actual_groups > num_nodes:
        raise ValueError("the number of nonempty groups cannot exceed the node count")
    if num_groups is None:
        num_groups = actual_groups
    if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups != actual_groups:
        raise ValueError("num_groups must equal the number of contiguous nonempty groups")
    if (torch.bincount(assignment, minlength=num_groups) == 0).any().item():
        raise ValueError("assignment groups must be contiguous and nonempty")
    return num_groups


def _validate_features(z, assignment):
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("features must be a nonempty node-by-feature matrix")
    if not z.is_floating_point():
        raise TypeError("features must have floating point dtype")
    if not isinstance(assignment, torch.Tensor):
        raise TypeError("assignment must be a LongTensor")
    if z.device != assignment.device:
        raise ValueError("features and assignment must be on the same device")


def _project(z, assignment, num_groups):
    sums = z.new_zeros((num_groups, z.shape[1])).index_add(0, assignment, z)
    counts = torch.bincount(assignment, minlength=num_groups).to(z.dtype)
    return (sums / counts[:, None])[assignment]


def project(z, assignment, num_groups=None):
    """Average within each nonempty group and broadcast to the original nodes."""
    _validate_features(z, assignment)
    num_groups = _validate_assignment(assignment, z.shape[0], num_groups)
    return _project(z, assignment, num_groups)


def _validate_p(p, z):
    if p.ndim != 2 or p.shape != (z.shape[0], z.shape[0]):
        raise ValueError("p must be a square node-by-node matrix")
    if p.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr):
        raise ValueError("p must be dense, sparse COO, or sparse CSR")
    if p.device != z.device or p.dtype != z.dtype:
        raise ValueError("p and features must have the same device and dtype")


def _apply_operator(z, p, assignment, mode, num_groups):
    if mode in ("PPI", "PIPPI"):
        z = _project(z, assignment, num_groups)
    z = p @ z if p.layout == torch.strided else torch.sparse.mm(p, z)
    if mode in ("PIP", "PIPPI"):
        z = _project(z, assignment, num_groups)
    return z


def apply_operator(z, p, assignment, mode):
    """Apply P, P Pi, Pi P, or Pi P Pi without forming a projection matrix."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    _validate_features(z, assignment)
    num_groups = _validate_assignment(assignment, z.shape[0])
    _validate_p(p, z)
    return _apply_operator(z, p, assignment, mode, num_groups)


class OperatorGCN(nn.Module):
    """Old GCN parameterization with a selected operator at both layers."""

    def __init__(self, din, hidden, classes, dropout, assignment, mode="P", mask_mode="independent"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mask_mode not in ("independent", "shared"):
            raise ValueError("mask_mode must be independent or shared")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.num_groups = _validate_assignment(assignment, assignment.numel())
        self.register_buffer("assignment", assignment.detach().clone(), persistent=False)
        self.l1, self.l2 = nn.Linear(din, hidden), nn.Linear(hidden, classes)
        self.dropout = dropout
        self.mode = mode
        self.mask_mode = mask_mode

    def forward(self, x, p, mask=None):
        _validate_features(x, self.assignment)
        if x.shape[0] != self.assignment.numel():
            raise ValueError("input node count must match assignment")
        _validate_p(p, x)
        # The immutable assignment was validated at construction; avoid repeated GPU syncs.
        z = F.relu(_apply_operator(self.l1(x), p, self.assignment, self.mode, self.num_groups))
        if self.training:
            if mask is not None:
                if mask.shape != z.shape or mask.device != z.device or mask.dtype != z.dtype:
                    raise ValueError("explicit mask must match the hidden features' shape, device and dtype")
                z = z * mask
            elif self.mask_mode == "independent":
                z = F.dropout(z, self.dropout, training=True)
            else:
                group_mask = F.dropout(z.new_ones((self.num_groups, z.shape[1])), self.dropout, training=True)
                z = z * group_mask[self.assignment]
        return _apply_operator(self.l2(z), p, self.assignment, self.mode, self.num_groups)


def per_example_loss(logits, targets, kind="ce", q=.7, epsilon=.1):
    """Return soft-target per-node CE, classwise GCE, or smoothed CE risk."""
    if logits.ndim != 2 or logits.shape != targets.shape or 0 in logits.shape:
        raise ValueError("logits and probability targets must have the same nonempty matrix shape")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must have floating point dtype")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    if kind not in ("ce", "gce", "label_smoothing"):
        raise ValueError("kind must be ce, gce, or label_smoothing")
    if not math.isfinite(q) or not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    log_probs = logits.log_softmax(dim=1)
    if kind == "gce":
        return (targets * (-torch.expm1(q * log_probs) / q)).sum(dim=1)
    if kind == "label_smoothing":
        targets = (1 - epsilon) * targets + epsilon / logits.shape[1]
    return -(targets * log_probs).sum(dim=1)
