"""Bounded propagation extension around the unchanged official TFR classes.

Only the classifier propagation and hidden dropout mask can change. The decoder,
raw-feature reconstruction target, confidence selection, and loss RNG sequence
retain the qualified reference behavior.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge


HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_reference_classes():
    """Load unchanged GCN/MPNN definitions through the qualified adapter.

    Keeping the adapter at its real path also keeps its vendor-relative ROOT
    intact. A frozen copy takes priority once the extension is packaged.
    """
    candidates = (HERE / "robust" / "adapters.py",
                  HERE.parent / "robust_graph_seed0_20261006" / "adapters.py")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("TFR reference adapters.py was not found in the frozen or source location")
    name = "_operator_robust_extension_tfr_reference_adapters"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the TFR reference adapter at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.tfr_classes()


def _assignment_groups(assignment):
    if not isinstance(assignment, torch.Tensor) or assignment.dtype != torch.long:
        raise TypeError("assignment must be a LongTensor")
    if assignment.ndim != 1 or assignment.numel() == 0:
        raise ValueError("assignment must contain one group per node in a nonempty vector")
    if assignment.min().item() < 0:
        raise ValueError("assignment must be nonnegative")
    num_groups = int(assignment.max().item()) + 1
    if num_groups > assignment.numel():
        raise ValueError("assignment groups must be contiguous and nonempty")
    if (torch.bincount(assignment, minlength=num_groups) == 0).any().item():
        raise ValueError("assignment groups must be contiguous and nonempty")
    return num_groups


class TFRClassifier(nn.Module):
    """Reference two-layer GCN with M_rho=(1-rho)P+rho Pi P Pi.

    The assignment is snapshotted into a buffer; no extra trainable parameter or
    node-by-node projection matrix is introduced. Native GCNConv adds its bias
    after propagation. Projecting its output preserves that bias since Pi 1=1.
    """

    def __init__(self, din, hidden, classes, dropout, assignment, rho=0., mask_mode="independent"):
        super().__init__()
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("rho must be finite and in [0, 1]")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and in [0, 1]")
        if mask_mode not in ("independent", "shared"):
            raise ValueError("mask_mode must be independent or shared")
        self.num_groups = _assignment_groups(assignment)
        self.register_buffer("assignment", assignment.detach().clone(), persistent=False)
        self.rho = float(rho)
        self.mask_mode = mask_mode
        GCN, _ = load_reference_classes()
        self.base = GCN(din, hidden, classes, n_layers=2, dropout=dropout,
                        norm_info=None, act="F.relu", input_layer=False, output_layer=False)

    def _project(self, features):
        sums = features.new_zeros((self.num_groups, features.shape[1]))
        sums = sums.index_add(0, self.assignment, features)
        counts = torch.bincount(self.assignment, minlength=self.num_groups).to(features.dtype)
        return (sums / counts[:, None])[self.assignment]

    def forward(self, x, adj, original=False):
        if x.ndim != 2 or x.shape[0] != self.assignment.numel() or x.shape[1] == 0:
            raise ValueError("features must be a nonempty matrix with one row per assignment")
        if not x.is_floating_point():
            raise TypeError("features must have floating point dtype")
        if x.device != self.assignment.device:
            raise ValueError("features and assignment must share a device")
        # Delegation preserves the native rho=0 path bit for bit, including RNG.
        if original or (self.rho == 0 and self.mask_mode == "independent"):
            return self.base(x, adj)
        for index, layer in enumerate(self.base.convs):
            if self.rho == 0:
                x = layer(x, adj)
            elif self.rho == 1:
                x = self._project(layer(self._project(x), adj))
            else:
                ordinary = layer(x, adj)
                projected = self._project(layer(self._project(x), adj))
                x = (1 - self.rho) * ordinary + self.rho * projected
            if index < self.base.n_layers - 1:
                x = self.base.act(x)
                if self.mask_mode == "independent":
                    x = F.dropout(x, p=self.base.dropout, training=self.training)
                elif self.training:
                    group_mask = F.dropout(x.new_ones((self.num_groups, x.shape[1])),
                                           p=self.base.dropout, training=True)
                    x = x * group_mask[self.assignment]
        return x.squeeze(1)


def tfr_loss(logits, x, noisy_y, train_ids, decoder, edge_index, coefficient, settings):
    """Return the original TFR objective plus its differentiable components.

    The caller controls decoder train/eval mode and coefficient updates. Edge
    dropout intentionally retains its default training=True even during the
    caller's validation/test passes, as in the qualified reference objective.
    """
    pseudo = logits.softmax(1)
    dropped = dropout_edge(edge_index, p=settings["beta_dropout_ratio"])[0]
    reconstructed, confidence = decoder(pseudo, dropped)
    selected = confidence.squeeze().argsort(descending=True)[:int(x.shape[0] * settings["pseudo_ratio"])]
    if settings["use_recon_softmax"]:
        reconstructed = reconstructed.softmax(1)
    recon = F.cross_entropy(reconstructed, x)
    pseudo_loss = F.cross_entropy(logits[selected], pseudo[selected].detach())
    supervised = F.cross_entropy(logits[train_ids], noisy_y[train_ids])
    total = (1 - coefficient) * supervised + settings["beta"] * recon + coefficient * pseudo_loss
    return total, dict(recon=recon, pseudo=pseudo_loss, supervised=supervised)
