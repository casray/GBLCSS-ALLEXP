"""PosteL no-IPL, ALS, SALS and GSDN adaptations with an attribute-only MLP.

PosteL target construction follows ml-postech/PosteL/relabel.py::postel;
see THIRD_PARTY_NOTICES.md. Existing graph self-loops are retained in target
construction and removed in GSDN's edge sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from ..data import GraphData
from ..models import MLP
from ..training import TrainResult, accuracy, soft_cross_entropy


_DEFAULTS = {
    "postel": dict(soft_label_ratio=0.8, smoothing_ratio=0.4, degree_cutoff=1,
                   temperature=1.0, iterative_pseudo_labeling=False),
    "als": dict(propagation_steps=2, residual=0.5, kl_weight=0.001,
                pacing="exponential", pacing_initial=0.1, pacing_rate=-0.001,
                max_smoothing_weight=0.1),
    "sals": dict(epsilon=0.4, gamma=0.8, unlabeled_neighbor_fallback="uniform"),
    "gsdn": dict(graph_batch_size=512, feature_weight=0.8, label_weight=0.8,
                 negative_weight=0.1, initial_mixup=0.5),
}
_ALIASES = {
    "als": {"lp_steps": "propagation_steps", "lp_residual": "residual",
            "alpha_max": "max_smoothing_weight"},
    "gsdn": {"lambda_feature": "feature_weight", "lambda_label": "label_weight",
             "lambda_negative": "negative_weight", "mixup_init": "initial_mixup"},
}


def _method_settings(method: str, settings: Mapping[str, object]) -> dict:
    if method not in _DEFAULTS:
        raise ValueError(f"unsupported structure-assisted baseline: {method}")
    supplied = dict(settings)
    supplied.pop("implementation", None)  # Descriptive packaging metadata.
    for source, target in _ALIASES.get(method, {}).items():
        if source in supplied:
            value = supplied.pop(source)
            if target in supplied and supplied[target] != value:
                raise ValueError(f"conflicting settings: {source} and {target}")
            supplied[target] = value
    unknown = supplied.keys() - _DEFAULTS[method].keys()
    if unknown:
        raise ValueError(f"unsupported {method} settings: {sorted(unknown)}")
    result = {**_DEFAULTS[method], **supplied}
    if method == "postel" and result["iterative_pseudo_labeling"] is not False:
        raise ValueError("PosteL no-IPL requires iterative_pseudo_labeling=false")
    if method == "sals" and result["unlabeled_neighbor_fallback"] != "uniform":
        raise ValueError("SALS requires unlabeled_neighbor_fallback=uniform")
    if method == "gsdn" and int(result["graph_batch_size"]) < 1:
        raise ValueError("graph_batch_size must be positive")
    return result


def _as_index(mask, device) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


def _split_index(data: GraphData, split: str, device) -> torch.Tensor:
    ordered = getattr(data, f"{split}_index")
    index = _as_index(ordered if ordered is not None else getattr(data, f"{split}_mask"), device)
    if index.numel() == 0:
        raise ValueError(f"{split} split must not be empty")
    return index


def _canonical_edge_index(edge_index, num_nodes, device) -> torch.Tensor:
    edges = torch.as_tensor(edge_index, device=device).long()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edges.numel() == 0:
        return edges.reshape(2, 0)
    if edges.min() < 0 or edges.max() >= int(num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")
    linear = torch.unique(edges[0] * int(num_nodes) + edges[1], sorted=True)
    return torch.stack([torch.div(linear, int(num_nodes), rounding_mode="floor"),
                        linear % int(num_nodes)], dim=0)


def _neighbor_label_histograms(labels, train_idx, edge_index, num_nodes, n_classes):
    edges = _canonical_edge_index(edge_index, num_nodes, labels.device)
    lookup = torch.zeros(int(num_nodes), dtype=torch.bool, device=labels.device)
    lookup[train_idx] = True
    src, dst = edges
    keep = lookup[src] & lookup[dst]
    src, dst = src[keep], dst[keep]
    histograms = torch.zeros((int(num_nodes), int(n_classes)),
                             dtype=torch.float64, device=labels.device)
    if src.numel():
        histograms.index_put_(
            (src, labels[dst].long()),
            torch.ones(src.numel(), dtype=torch.float64, device=labels.device),
            accumulate=True,
        )
    return histograms


def _target_result(train_targets, train_idx, fallback, num_nodes, n_classes):
    targets = torch.full((int(num_nodes), int(n_classes)), 1.0 / int(n_classes),
                         dtype=torch.float32, device=train_targets.device)
    targets[train_idx] = train_targets.to(torch.float32)
    fallback_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=targets.device)
    fallback_mask[train_idx] = fallback
    return targets, {"fallback_count": int(fallback_mask.sum().item()),
                     "_fallback_mask": fallback_mask}


def postel_prob_label_from_likelihood(likelihood, prior, temperature=1.0):
    """Keep the original global maximum and underflow/invalid-posterior behavior."""
    likelihood = torch.as_tensor(likelihood)
    prior = torch.as_tensor(prior, dtype=likelihood.dtype, device=likelihood.device)
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_prob = torch.log(likelihood) + torch.log(prior)
    log_prob = log_prob - log_prob.max()
    probability = torch.exp(log_prob).pow(temperature)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    return probability, (~torch.isfinite(probability)).any(dim=1)


def build_postel_targets(
    noisy_labels, train_mask, edge_index, *, num_nodes, n_classes,
    degree_cutoff=1, temperature=1.0, smoothing_ratio=0.4, soft_label_ratio=0.8,
):
    """Original product posterior, degree fallback, smoothing, then hard mixing."""
    labels = torch.as_tensor(noisy_labels).long()
    train_idx = _as_index(train_mask, labels.device)
    n_classes = int(n_classes)
    if n_classes < 2 or train_idx.numel() == 0:
        raise ValueError("PosteL requires two classes and at least one training node")
    hard = F.one_hot(labels[train_idx], n_classes).to(torch.float64)
    prior = hard.mean(dim=0)
    histograms = _neighbor_label_histograms(labels, train_idx, edge_index, num_nodes, n_classes)
    conditional = torch.empty((n_classes, n_classes), dtype=torch.float64, device=labels.device)
    uniform = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float64, device=labels.device)
    for class_id in range(n_classes):
        class_nodes = train_idx[labels[train_idx] == class_id]
        class_hist = histograms[class_nodes].sum(dim=0)
        total = class_hist.sum()
        conditional[class_id] = class_hist / total if total > 0 else uniform
    train_hist = histograms[train_idx]
    likelihood = conditional.unsqueeze(0).pow(train_hist.unsqueeze(1)).prod(dim=2)
    posterior, invalid = postel_prob_label_from_likelihood(likelihood, prior, temperature)
    fallback = invalid | (train_hist.sum(dim=1) < int(degree_cutoff))
    posterior[fallback] = hard[fallback]
    smoothing_ratio, soft_label_ratio = float(smoothing_ratio), float(soft_label_ratio)
    if not 0 <= smoothing_ratio <= 1:
        raise ValueError("smoothing_ratio must lie in [0, 1]")
    if not 0 <= soft_label_ratio <= 1:
        raise ValueError("soft_label_ratio must lie in [0, 1]")
    smoothed = ((1.0 - smoothing_ratio) * posterior
                + smoothing_ratio / (n_classes - 1) * (~hard.bool()).to(torch.float64))
    train_targets = soft_label_ratio * smoothed + (1.0 - soft_label_ratio) * hard
    return _target_result(train_targets, train_idx, fallback, num_nodes, n_classes)


def build_sals_targets(
    noisy_labels, train_mask, edge_index, *, num_nodes, n_classes, epsilon=0.4, gamma=0.8,
):
    """SALS equation with train-labeled neighbors and uniform empty-neighbor fallback."""
    labels = torch.as_tensor(noisy_labels).long()
    train_idx = _as_index(train_mask, labels.device)
    epsilon, gamma = float(epsilon), float(gamma)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1]")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    histograms = _neighbor_label_histograms(labels, train_idx, edge_index, num_nodes, n_classes)
    train_hist = histograms[train_idx]
    counts = train_hist.sum(dim=1, keepdim=True)
    fallback = counts.squeeze(1) <= 0
    uniform = torch.full((train_idx.numel(), int(n_classes)), 1.0 / int(n_classes),
                         dtype=torch.float64, device=labels.device)
    neighborhood = train_hist / counts.clamp_min(1.0)
    neighborhood[fallback] = uniform[fallback]
    hard = F.one_hot(labels[train_idx], int(n_classes)).to(torch.float64)
    train_targets = (1.0 - epsilon) * hard + epsilon * (gamma * neighborhood + (1.0 - gamma) * uniform)
    return _target_result(train_targets, train_idx, fallback, num_nodes, n_classes)


def als_label_propagation(
    noisy_labels, train_mask, edge_index, *, num_nodes, n_classes, steps=2, residual=0.5,
):
    """Residual propagation with the original symmetric outdegree normalization."""
    labels = torch.as_tensor(noisy_labels).long()
    train_idx = _as_index(train_mask, labels.device)
    edges = _canonical_edge_index(edge_index, num_nodes, labels.device)
    steps, residual = int(steps), float(residual)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0 <= residual <= 1:
        raise ValueError("residual must lie in [0, 1]")
    initial = torch.zeros((int(num_nodes), int(n_classes)), dtype=torch.float32, device=labels.device)
    initial[train_idx] = F.one_hot(labels[train_idx], int(n_classes)).to(torch.float32)
    src, dst = edges
    degree = torch.bincount(src, minlength=int(num_nodes)).to(torch.float32)
    values = degree[src].clamp_min(1.0).rsqrt() * degree[dst].clamp_min(1.0).rsqrt()
    normalized_adj = torch.sparse_coo_tensor(
        edges, values, (int(num_nodes), int(num_nodes)), device=labels.device,
    ).coalesce()
    propagated = initial
    for _ in range(steps):
        propagated = (1.0 - residual) * torch.sparse.mm(normalized_adj, propagated) + residual * initial
    fallback_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=labels.device)
    fallback_mask[train_idx] = propagated[train_idx].sum(dim=1) <= 0
    return propagated, {"fallback_count": int(fallback_mask.sum().item()),
                        "_fallback_mask": fallback_mask, "_initial_labels": initial}


def als_eq5_loss(logits, hard_labels, soft_targets, *, alpha, kl_weight):
    """Hard/soft CE mixture plus KL(soft || uniform), including target gradients."""
    alpha = float(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    soft_targets = torch.as_tensor(soft_targets, dtype=logits.dtype, device=logits.device)
    hard_labels = torch.as_tensor(hard_labels, dtype=torch.long, device=logits.device)
    hard_loss = F.cross_entropy(logits, hard_labels)
    soft_loss = soft_cross_entropy(logits, soft_targets)
    uniform_log = -math.log(float(soft_targets.shape[1]))
    kl_uniform = (soft_targets * (soft_targets.clamp_min(1e-12).log() - uniform_log)).sum(dim=1).mean()
    return (1.0 - alpha) * hard_loss + alpha * soft_loss + float(kl_weight) * kl_uniform


def als_pacing_alpha(epoch, *, pacing, alpha_max, rate, initial=0.1):
    epoch, alpha_max, rate, initial = float(epoch), float(alpha_max), float(rate), float(initial)
    if pacing == "linear":
        value = rate * epoch
    elif pacing == "exponential":
        value = initial * math.exp(rate * epoch)
    else:
        raise ValueError(f"unsupported ALS pacing family: {pacing}")
    return min(max(value, 0.0), alpha_max)


def _clean_gsdn_edges(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edges.numel() == 0:
        return edges.reshape(2, 0)
    edges = edges[:, edges[0] != edges[1]]
    if edges.shape[1] == 0:
        src = torch.arange(num_nodes, device=edges.device)
        edges = torch.stack([src, torch.roll(src, shifts=-1)], dim=0)
    return edges


def _gsdn_forward(model: MLP, features: torch.Tensor) -> torch.Tensor:
    # Full-X training uses the non-inplace hidden-forward path for full X.
    hidden = features
    for index, linear in enumerate(model.layers[:-1]):
        hidden = F.relu(linear(hidden))
        hidden = model.batch_norms[index](hidden)
        hidden = F.dropout(hidden, p=model.dropout, training=model.training)
    return model.layers[-1](hidden)


def gsdn_extra_loss(model, aux_model, mixup_logit, features, logits, labels, train_idx, edges, settings):
    """Old GSDN's ordered sampling, detached mixup target and live negative branch."""
    batch_size = int(settings["graph_batch_size"])
    if edges.shape[1] == 0:
        raise ValueError("GSDN requires at least one edge for training-time regularization")
    sampled = torch.randint(edges.shape[1], (batch_size,), device=features.device)
    src, dst = edges[0, sampled], edges[1, sampled]
    neg = torch.randint(int(features.shape[0]), (src.shape[0],), device=features.device)
    coeff = torch.sigmoid(mixup_logit).clamp(1e-6, 1.0 - 1e-6)
    mixed_x = coeff * features[src] + (1.0 - coeff) * features[dst]
    mixed_logits = model(mixed_x)
    aux_logits = aux_model(features)
    with torch.no_grad():
        soft_src = F.softmax(aux_logits[src], dim=1)
        soft_dst = F.softmax(aux_logits[dst], dim=1)
        mixed_target = coeff * soft_src + (1.0 - coeff) * soft_dst
    feature_loss = F.kl_div(F.log_softmax(mixed_logits, dim=1), mixed_target, reduction="batchmean")
    main_probs = F.softmax(logits[src], dim=1)
    neg_probs = F.softmax(aux_logits[neg], dim=1)
    negative_loss = (main_probs * neg_probs).sum(dim=1).mean()
    lookup = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    lookup[train_idx] = True
    keep = lookup[edges[0]]
    label_src, label_dst = edges[0, keep], edges[1, keep]
    if label_src.numel() == 0:
        label_loss = logits.new_zeros(())
    else:
        sample_idx = torch.randint(label_src.shape[0], (batch_size,), device=features.device)
        label_loss = F.cross_entropy(aux_logits[label_dst[sample_idx]], labels[label_src[sample_idx]])
    return (float(settings["feature_weight"]) * feature_loss
            + float(settings["label_weight"]) * label_loss
            + float(settings["negative_weight"]) * negative_loss)


def train_structure_baseline(
    data: GraphData,
    method: str,
    settings: Mapping[str, object],
    *,
    epochs: int = 200,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    device: str = "cpu",
) -> TrainResult:
    """Train postel/als/sals/gsdn using a per-method protocol_adaptations mapping.

    Defaults and accepted YAML keys are in _DEFAULTS; original NoisyGL names
    in _ALIASES are also accepted. The caller owns seeding and split creation.
    No RNG is reset here. Checkpoint selection is strict validation accuracy
    improvement, with no early stopping, as in the original method configs.
    train_accuracy is the selected epoch's pre-update, training-mode accuracy
    (the original recorder convention); logits are from the restored eval MLP.
    """
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    if int(layers) < 1:
        raise ValueError("layers must be positive")
    method = method.lower()
    options = _method_settings(method, settings)
    x = data.x.to(device)
    clean = data.y.to(device)
    noisy = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train_idx, val_idx, test_idx = (_split_index(data, split, device) for split in ("train", "val", "test"))
    edges = _canonical_edge_index(data.edge_index, data.num_nodes, device)
    model = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    parameters = list(model.parameters())
    targets = propagated = relevance = aux_model = mixup_logit = None
    target_args = dict(num_nodes=data.num_nodes, n_classes=data.num_classes)
    if method == "postel":
        targets, _ = build_postel_targets(
            noisy, train_idx, edges, **target_args,
            **{key: options[key] for key in ("soft_label_ratio", "smoothing_ratio", "degree_cutoff", "temperature")},
        )
    elif method == "sals":
        targets, _ = build_sals_targets(noisy, train_idx, edges, **target_args,
                                        epsilon=options["epsilon"], gamma=options["gamma"])
    elif method == "als":
        propagated, _ = als_label_propagation(
            noisy, train_idx, edges, **target_args,
            steps=options["propagation_steps"], residual=options["residual"],
        )
        relevance = torch.empty((data.num_classes, data.num_classes), dtype=torch.float32, device=device)
        torch.nn.init.xavier_uniform_(relevance)
        relevance = torch.nn.Parameter(relevance)
        parameters.append(relevance)
    else:
        edges = _clean_gsdn_edges(edges, data.num_nodes)
        aux_model = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
        initial = min(max(float(options["initial_mixup"]), 1e-4), 1.0 - 1e-4)
        mixup_logit = torch.nn.Parameter(torch.logit(torch.tensor(initial, device=device)))
        parameters.extend(list(aux_model.parameters()) + [mixup_logit])

    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    best_validation, best_train, best_epoch, best_state = -1.0, -1.0, -1, None
    forward = _gsdn_forward if method == "gsdn" else lambda network, features: network(features)
    for epoch in range(int(epochs)):
        model.train()
        if aux_model is not None:
            aux_model.train()
        optimizer.zero_grad()
        logits = forward(model, x)
        if method == "als":
            soft = F.softmax(propagated[train_idx] @ relevance.t(), dim=1)
            alpha = als_pacing_alpha(epoch, pacing=options["pacing"],
                                     alpha_max=options["max_smoothing_weight"],
                                     rate=options["pacing_rate"], initial=options["pacing_initial"])
            loss = als_eq5_loss(logits[train_idx], noisy[train_idx], soft,
                               alpha=alpha, kl_weight=options["kl_weight"])
        elif method == "gsdn":
            loss = F.cross_entropy(logits[train_idx], noisy[train_idx])
            loss = loss + gsdn_extra_loss(model, aux_model, mixup_logit, x, logits,
                                         noisy, train_idx, edges, options)
        else:
            loss = soft_cross_entropy(logits[train_idx], targets[train_idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {method} loss at epoch {epoch + 1}")
        train_accuracy = accuracy(logits[train_idx].detach(), noisy[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        if aux_model is not None:
            aux_model.eval()
        with torch.no_grad():
            validation = accuracy(forward(model, x)[val_idx], noisy[val_idx])
        if validation > best_validation:
            best_validation, best_train, best_epoch = validation, train_accuracy, epoch + 1
            best_state = {"model": deepcopy(model.state_dict())}
            if relevance is not None:
                best_state["relevance"] = relevance.detach().clone()
            if aux_model is not None:
                best_state["aux_model"] = deepcopy(aux_model.state_dict())
                best_state["mixup_logit"] = mixup_logit.detach().clone()

    model.load_state_dict(best_state["model"])
    with torch.no_grad():
        if relevance is not None:
            relevance.copy_(best_state["relevance"])
        if aux_model is not None:
            aux_model.load_state_dict(best_state["aux_model"])
            mixup_logit.copy_(best_state["mixup_logit"])
        logits = forward(model, x)
        test_accuracy = accuracy(logits[test_idx], clean[test_idx])
    return TrainResult(train_accuracy=best_train, validation_accuracy=best_validation,
                       test_accuracy=test_accuracy, best_epoch=best_epoch, logits=logits.cpu())
