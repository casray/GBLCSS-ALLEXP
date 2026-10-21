from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_code/src'))
from gblcss.models import MLP


def state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((key, value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_targets(z, assignment, classes, *, leave_one_out=True, tie_safe=True, beta=.5, min_votes=2):
    if z.ndim != 1 or assignment.shape != z.shape or len(z) == 0:
        raise ValueError('Expected observed training labels and matching assignments')
    if (z < 0).any() or (z >= classes).any() or not 0 <= beta <= 1:
        raise ValueError('Invalid training labels or beta')
    one_hot = F.one_hot(z.long(), classes).float()
    _, inverse = torch.unique(assignment.long(), sorted=True, return_inverse=True)
    counts = one_hot.new_zeros((int(inverse.max())+1, classes))
    counts.index_add_(0, inverse, one_hot)
    local = counts[inverse] - one_hot if leave_one_out else counts[inverse]
    support = local.sum(1)
    r = local / support.clamp_min(1).unsqueeze(1)
    maximum, winner = local.max(1)
    alpha = ((maximum/support.clamp_min(1) - 1/classes)/(1-1/classes)).clamp(0, 1)
    alpha = torch.where(support >= min_votes, alpha, 0.)
    tied = (local == maximum.unsqueeze(1)).sum(1) > 1
    strength = torch.where(tied, 0., beta) if tie_safe else torch.full_like(alpha, beta)
    q = (1-strength.unsqueeze(1))*r + strength.unsqueeze(1)*F.one_hot(winner, classes)
    targets = (1-alpha.unsqueeze(1))*one_hot + alpha.unsqueeze(1)*q
    if not torch.isfinite(targets).all() or not torch.allclose(targets.sum(1), torch.ones_like(alpha), atol=1e-6):
        raise ValueError('Invalid constructed targets')
    return targets, alpha


def keep_count(n, epoch, rate, gradual=10):
    if n <= 0:
        return 0
    rate = max(0., min(float(rate), .95))
    progress = min(1., (int(epoch)+1)/max(1, gradual))
    return max(1, min(n, int((1-rate*progress)*n)))


def supervised_loss(logits, targets, rule, epoch, rate):
    if rule not in ('T0', 'T1', 'T2', 'T3'):
        raise ValueError(rule)
    losses = -(targets*F.log_softmax(logits, dim=1)).sum(1)
    selected = (torch.topk(losses.detach(), keep_count(len(losses), epoch, rate), largest=False).indices
                if rule in ('T2', 'T3') else torch.arange(len(losses), device=logits.device))
    return losses[selected].mean(), selected


def alignment_loss(projected, target):
    return (1-F.cosine_similarity(projected, target.detach(), dim=1, eps=1e-12)).mean()


def select_weight(rows):
    weights = sorted({float(row['weight']) for row in rows})
    if not weights:
        raise ValueError('No validation results')
    scores = {w: np.mean([row['validation'] for row in rows if row['weight'] == w]) for w in weights}
    return max(weights, key=lambda w: (scores[w], -w))


class AlignedMLP(nn.Module):
    def __init__(self, features, classes, representation_dim, hidden=64, dropout=.5):
        super().__init__()
        self.backbone = MLP(features, hidden, classes, 2, dropout)
        # Projection initialization must not change the baseline dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(hidden, representation_dim)

    def encode(self, x):
        return self.backbone.batch_norms[0](F.relu(self.backbone.layers[0](x)))

    def forward(self, x):
        hidden = self.encode(x)
        logits = self.backbone.layers[-1](F.dropout(hidden, self.backbone.dropout, self.backbone.training))
        return logits, hidden

    def encoder_parameters(self):
        return list(self.backbone.layers[0].parameters()) + list(self.backbone.batch_norms[0].parameters())

    def classifier_parameters(self):
        return list(self.backbone.layers[-1].parameters())

    def encoder_state(self):
        return {k: v for k, v in self.backbone.state_dict().items() if not k.startswith('layers.1.')}

    def freeze_encoder(self):
        for p in self.encoder_parameters() + list(self.projection.parameters()):
            p.requires_grad_(False)
            p.grad = None
        self.eval()

    def train_head(self):
        self.eval()
        self.backbone.layers[-1].train()

    def deployment_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.backbone.state_dict().items()}


def reference_remember_count():
    path = PROJECT / 'external/NoisyGL/predictor/MLPNoiseBaselines_Predictor.py'
    nodes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name == 'remember_count']
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['remember_count']


def reference_target_functions():
    path = PROJECT/'external/NoisyGL/utils/frozen_grace_targets.py'
    names = {'apply_ball_soft_targets', '_mask_to_index_array', '_alpha_from_margin', '_ball_assignment_hash'}
    nodes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, np=np, F=F, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace
