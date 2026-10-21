from __future__ import annotations

import torch
from torch.nn import functional as F


def regional_mean(probabilities: torch.Tensor, groups: torch.Tensor,
                  sizes: torch.Tensor) -> torch.Tensor:
    sums = probabilities.new_zeros((sizes.numel(), probabilities.shape[1]))
    sums.index_add_(0, groups, probabilities)
    return sums / sizes[:, None]


class RegionalKL:
    def __init__(self, teacher: torch.Tensor, assignment: torch.Tensor) -> None:
        if teacher.ndim != 2 or teacher.shape[0] < 1:
            raise ValueError('teacher must be a nonempty node-by-class matrix')
        assignment = torch.as_tensor(assignment, device=teacher.device, dtype=torch.long)
        if assignment.ndim != 1 or assignment.numel() != teacher.shape[0]:
            raise ValueError('assignment must contain one group id per node')
        if (assignment < 0).any():
            raise ValueError('every node must belong to a group')
        if not torch.isfinite(teacher).all() or (teacher < 0).any():
            raise ValueError('teacher probabilities must be finite and nonnegative')
        if (teacher.sum(1) - 1).abs().max() > 1e-5:
            raise ValueError('teacher probability rows must sum to one')
        _, self.groups = torch.unique(assignment, sorted=True, return_inverse=True)
        self.sizes = torch.bincount(self.groups).to(dtype=teacher.dtype)
        self.weights = self.sizes / teacher.shape[0]
        self.teacher_mean = regional_mean(teacher.detach(), self.groups, self.sizes).detach()
        self.shape = teacher.shape

    def __call__(self, student: torch.Tensor) -> torch.Tensor:
        if student.shape != self.shape or student.device != self.groups.device:
            raise ValueError('student and teacher must have matching shape and device')
        means = regional_mean(student, self.groups, self.sizes)
        log_means = means.clamp_min(torch.finfo(means.dtype).tiny).log()
        divergence = F.kl_div(log_means, self.teacher_mean, reduction='none').sum(1)
        return (self.weights * divergence).sum()


def residual_components(student: torch.Tensor, teacher: torch.Tensor,
                        assignment: torch.Tensor) -> dict[str, float]:
    residual = student.double() - teacher.double()
    _, groups = torch.unique(assignment.to(residual.device), return_inverse=True)
    sizes = torch.bincount(groups).to(dtype=residual.dtype)
    means = regional_mean(residual, groups, sizes)
    total = residual.square().sum(1).mean()
    between = (sizes * means.square().sum(1)).sum() / residual.shape[0]
    within = (residual - means[groups]).square().sum(1).mean()
    return dict(total=float(total), between=float(between), within=float(within),
                decomposition_error=float((total - between - within).abs()))
