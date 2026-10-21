from __future__ import annotations

import torch
from torch.nn import functional as F

from vendor.dist_official import DIST
from vendor.dgkd_official import dgkd_loss


class KDLoss:
    """Teacher-only adaptation; no graph labels are accepted by this interface."""

    def __init__(self, family, teacher, temperature):
        if family not in ('KD', 'DIST', 'DGKD-T') or temperature <= 0:
            raise ValueError('invalid family or temperature')
        if teacher.ndim != 2 or teacher.shape[1] < 2:
            raise ValueError('teacher must contain multiclass probability rows')
        if not torch.isfinite(teacher).all() or (teacher < 0).any():
            raise ValueError('invalid teacher probabilities')
        if not torch.allclose(teacher.sum(1), torch.ones_like(teacher[:, 0]), atol=1e-5, rtol=1e-5):
            raise ValueError('teacher rows must sum to one')
        self.family = family
        self.temperature = float(temperature)
        self.teacher = teacher.detach()
        self.logits = self.teacher.clamp_min(torch.finfo(teacher.dtype).tiny).log()
        self.targets = self.teacher.argmax(1)
        self.soft = (self.logits / temperature).softmax(1)
        self.dist = DIST(beta=1., gamma=1., tau=temperature)

    def __call__(self, student):
        if student.shape != self.teacher.shape:
            raise ValueError('student/teacher shape mismatch')
        if self.family == 'KD':
            if self.temperature == 1:
                return -(self.teacher * F.log_softmax(student, dim=1)).sum(dim=1).mean()
            return -(self.soft * F.log_softmax(student / self.temperature, dim=1)).sum(1).mean() * self.temperature**2
        if self.family == 'DIST':
            return self.dist(student, self.logits)
        return dgkd_loss(student, self.logits, self.targets, 15., 3., self.temperature)
