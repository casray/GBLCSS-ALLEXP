"""Equation-grounded GSDN adaptation under the project's unified protocol.

This is an independent reimplementation of arXiv:2210.02097 (v5), not the
authors' official code.  Eqs. (2), (3), (7), and (8) ground the core
components: one shared MLP backbone, a target head, a neighborhood head,
training-time relation batches, and feature-only inference with the target
head.  Per-batch optimization follows the project's common protocol and is
not claimed to be equation-identical to the paper.
"""

import math
import time
from copy import deepcopy

import nni
import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor


def _index_tensor(mask, device):
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.flatten().long()


def canonicalize_undirected_relations(edge_index, num_nodes):
    """Return deterministic unique non-self ``(min(u,v), max(u,v))`` pairs."""
    edge_index = torch.as_tensor(edge_index).long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")

    raw_arc_count = int(edge_index.shape[1])
    if raw_arc_count == 0:
        empty = edge_index.reshape(2, 0)
        return empty, {
            "raw_arc_count": 0,
            "self_loops_removed": 0,
            "unique_undirected_relation_count": 0,
            "deduplicated_reverse_arc_count": 0,
        }

    src, dst = edge_index
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    self_loops_removed = raw_arc_count - int(src.numel())

    directed_ids = torch.unique(
        src * int(num_nodes) + dst,
        sorted=True,
    )
    directed_src = torch.div(
        directed_ids, int(num_nodes), rounding_mode="floor"
    )
    directed_dst = torch.remainder(directed_ids, int(num_nodes))
    low = torch.minimum(directed_src, directed_dst)
    high = torch.maximum(directed_src, directed_dst)
    relation_ids = torch.unique(
        low * int(num_nodes) + high,
        sorted=True,
    )
    canonical = torch.stack(
        [
            torch.div(
                relation_ids, int(num_nodes), rounding_mode="floor"
            ),
            torch.remainder(relation_ids, int(num_nodes)),
        ],
        dim=0,
    )
    unique_relation_count = int(relation_ids.numel())
    diagnostics = {
        "raw_arc_count": raw_arc_count,
        "self_loops_removed": self_loops_removed,
        "unique_undirected_relation_count": unique_relation_count,
        "deduplicated_reverse_arc_count": (
            int(directed_ids.numel()) - unique_relation_count
        ),
    }
    return canonical, diagnostics


def gsdn_eq3(
    target_features,
    neighbor_features,
    target_hidden,
    neighbor_hidden,
    mix_weight,
    attention,
    neighbor_head_weight,
    neighbor_head_bias=None,
):
    """Eq. (3): edge-specific mixup followed by the neighborhood head."""
    target_projected = target_features @ mix_weight
    neighbor_projected = neighbor_features @ mix_weight
    attention_input = torch.cat(
        [target_projected, neighbor_projected], dim=-1
    )
    beta = torch.sigmoid(attention_input @ attention)
    mixed_hidden = (
        beta.unsqueeze(-1) * neighbor_hidden
        + (1.0 - beta.unsqueeze(-1)) * target_hidden
    )
    augmented_logits = F.linear(
        mixed_hidden, neighbor_head_weight, neighbor_head_bias
    )
    return augmented_logits, beta


def gsdn_eq7(
    target_logits,
    neighbor_target_logits,
    augmented_neighbor_logits,
    augmented_reverse_logits,
    negative_neighbor_logits,
):
    """Eq. (7): batch feature-level neighborhood self-distillation."""
    positive = (
        (target_logits - augmented_neighbor_logits).pow(2).sum(dim=-1)
        + (neighbor_target_logits - augmented_reverse_logits)
        .pow(2)
        .sum(dim=-1)
    )
    negative_probs = F.softmax(negative_neighbor_logits, dim=-1)
    negative = (
        (F.softmax(target_logits, dim=-1) - negative_probs)
        .pow(2)
        .sum(dim=-1)
        + (F.softmax(neighbor_target_logits, dim=-1) - negative_probs)
        .pow(2)
        .sum(dim=-1)
    )
    return (positive - negative).mean()


def gsdn_eq8(
    endpoint_nodes,
    endpoint_target_logits,
    neighbor_logits_at_edge_dst,
    neighbor_logits_at_edge_src,
    edge_src,
    edge_dst,
    noisy_labels,
    train_mask,
):
    """Eq. (8), restricted to project-noisy labels in the training split.

    The first CE term is evaluated once for every labeled endpoint in V_b.
    The second CE term propagates each labeled endpoint's noisy label across
    every incident unique undirected relation, in either direction.  Dividing
    each batch expression by the unique-relation count |E_b| is the project's
    common-protocol optimization adaptation; the paper writes a global
    ``1/B`` where B is the number of batches, so the optimization is not
    claimed to be equation-identical.
    """
    device = endpoint_target_logits.device
    endpoint_nodes = torch.as_tensor(
        endpoint_nodes, device=device, dtype=torch.long
    )
    edge_src = torch.as_tensor(edge_src, device=device, dtype=torch.long)
    edge_dst = torch.as_tensor(edge_dst, device=device, dtype=torch.long)
    noisy_labels = torch.as_tensor(noisy_labels, device=device).long()
    train_idx = _index_tensor(train_mask, device)

    endpoint_is_train = torch.isin(endpoint_nodes, train_idx)
    source_is_train = torch.isin(edge_src, train_idx)
    destination_is_train = torch.isin(edge_dst, train_idx)
    zero = endpoint_target_logits.sum() * 0.0

    if endpoint_is_train.any():
        labeled_endpoints = endpoint_nodes[endpoint_is_train]
        target_loss = F.cross_entropy(
            endpoint_target_logits[endpoint_is_train],
            noisy_labels[labeled_endpoints],
            reduction="sum",
        )
    else:
        target_loss = zero

    if source_is_train.any():
        neighbor_loss = F.cross_entropy(
            neighbor_logits_at_edge_dst[source_is_train],
            noisy_labels[edge_src[source_is_train]],
            reduction="sum",
        )
    else:
        neighbor_loss = zero

    if destination_is_train.any():
        neighbor_loss = neighbor_loss + F.cross_entropy(
            neighbor_logits_at_edge_src[destination_is_train],
            noisy_labels[edge_dst[destination_is_train]],
            reduction="sum",
        )

    normalizer = max(int(edge_src.numel()), 1)
    return (target_loss + neighbor_loss) / normalizer


def iter_edge_batches(edge_index, batch_size, generator=None):
    """Yield a shuffled partition in which every supplied edge occurs once."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    edge_count = int(edge_index.shape[1])
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    permutation = torch.randperm(
        edge_count, device=edge_index.device, generator=generator
    )
    for start in range(0, edge_count, batch_size):
        edge_ids = permutation[start : start + batch_size]
        yield edge_index[:, edge_ids], edge_ids


class GSDNBackbone(nn.Module):
    """The paper's Eq. (2) MLP: Linear -> ReLU -> BN -> Dropout."""

    def __init__(self, in_features, hidden_features, n_layers, dropout):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError("GSDN backbone requires at least one layer")
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(int(n_layers)):
            layer_in = in_features if layer == 0 else hidden_features
            self.layers.append(nn.Linear(layer_in, hidden_features))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        self.dropout = float(dropout)

    def forward(self, features):
        hidden = features
        for linear, batch_norm in zip(self.layers, self.batch_norms):
            hidden = linear(hidden)
            hidden = F.relu(hidden)
            hidden = batch_norm(hidden)
            hidden = F.dropout(
                hidden, p=self.dropout, training=self.training
            )
        return hidden


class GSDNFaithfulModel(nn.Module):
    """Shared backbone plus the paper's two linear inference heads."""

    def __init__(
        self,
        in_features,
        hidden_features,
        n_classes,
        n_layers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.backbone = GSDNBackbone(
            in_features, hidden_features, n_layers, dropout
        )
        self.target_head = nn.Linear(hidden_features, n_classes)
        self.neighbor_head = nn.Linear(hidden_features, n_classes)
        self.mix_projection = nn.Linear(
            in_features, hidden_features, bias=False
        )
        self.attention = nn.Parameter(torch.empty(2 * hidden_features))
        nn.init.xavier_uniform_(self.mix_projection.weight)
        nn.init.normal_(self.attention, mean=0.0, std=0.02)

    def encode(self, features):
        return self.backbone(features)

    def forward(self, features):
        return self.target_head(self.encode(features))

    def training_parameters(self):
        return list(self.parameters())

    def deployed_parameters(self):
        return list(self.backbone.parameters()) + list(
            self.target_head.parameters()
        )

    def deployed_state_dict(self):
        return {
            "backbone": deepcopy(self.backbone.state_dict()),
            "target_head": deepcopy(self.target_head.state_dict()),
        }


class gsdn_faithful_core_Predictor(Predictor):
    """Equation-grounded GSDN common-protocol adaptation."""

    def method_init(self, conf, data):
        self.model = GSDNFaithfulModel(
            in_features=conf.model["n_feat"],
            hidden_features=conf.model["n_hidden"],
            n_classes=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.train_idx = _index_tensor(self.train_mask, self.device)
        edge_index = self.edge_index.to(self.device).long()
        (
            self.training_edge_index,
            relation_diagnostics,
        ) = canonicalize_undirected_relations(edge_index, self.n_nodes)
        if self.training_edge_index.shape[1] == 0:
            raise ValueError(
                "GSDN faithful training requires at least one non-self edge"
            )

        self.graph_batch_size = int(
            conf.model.get("graph_batch_size", 512)
        )
        self.lambda_feature = float(conf.model.get("lambda", 0.8))
        self.optim = torch.optim.Adam(
            self.model.training_parameters(),
            lr=conf.training["lr"],
            weight_decay=conf.training["weight_decay"],
        )
        edge_count = int(self.training_edge_index.shape[1])
        training_count = sum(
            parameter.numel()
            for parameter in self.model.training_parameters()
        )
        deployed_count = sum(
            parameter.numel()
            for parameter in self.model.deployed_parameters()
        )
        self.gsdn_faithful_diagnostics = {
            "baseline_identity": (
                "GSDN equation-grounded common-protocol adaptation; "
                "independent reimplementation, not official code"
            ),
            "equations": "Eq.3,Eq.7,Eq.8",
            "graph_used_during_training": 1,
            "graph_free_inference": 1,
            "negative_distribution": "uniform_all_nodes",
            "label_source": "project_noisy_train_labels_only",
            "validation_source": "clean_validation_model_selection_only",
            "test_source": "clean_test_final_evaluation_only",
            "raw_edge_count": int(edge_index.shape[1]),
            "training_edge_count": edge_count,
            **relation_diagnostics,
            "edge_batches_expected": math.ceil(
                edge_count / self.graph_batch_size
            ),
            "training_parameter_count": training_count,
            "deployed_parameter_count": deployed_count,
            "training_parameter_names": [
                name for name, _ in self.model.named_parameters()
            ],
            "deployed_parameter_names": [
                "backbone." + name
                for name, _ in self.model.backbone.named_parameters()
            ]
            + [
                "target_head." + name
                for name, _ in self.model.target_head.named_parameters()
            ],
        }
        # Existing runners inspect this conventional attribute.
        self.graphfree_diagnostics = self.gsdn_faithful_diagnostics
        self.best_epoch = -1

    def get_prediction(self, features, adj, label=None, mask=None):
        del adj
        output = self.model(features)
        loss, metric = None, None
        if label is not None and mask is not None:
            idx = _index_tensor(mask, self.device)
            loss = self.loss_fn(output[idx], label[idx])
            metric = self.metric(
                label[idx].detach().cpu().numpy(),
                output[idx].detach().cpu().numpy(),
            )
        return output, loss, metric

    def _training_batch_loss(self, batch_edges, negative_nodes):
        src, dst = batch_edges
        endpoint_nodes = torch.unique(
            torch.cat([src, dst], dim=0), sorted=True
        )
        all_nodes = torch.unique(
            torch.cat([endpoint_nodes, negative_nodes], dim=0), sorted=True
        )
        hidden = self.model.encode(self.feats[all_nodes])

        src_hidden = hidden[torch.searchsorted(all_nodes, src)]
        dst_hidden = hidden[torch.searchsorted(all_nodes, dst)]
        negative_hidden = hidden[
            torch.searchsorted(all_nodes, negative_nodes)
        ]
        endpoint_hidden = hidden[
            torch.searchsorted(all_nodes, endpoint_nodes)
        ]

        target_src = self.model.target_head(src_hidden)
        target_dst = self.model.target_head(dst_hidden)
        target_endpoints = self.model.target_head(endpoint_hidden)
        neighbor_src = self.model.neighbor_head(src_hidden)
        neighbor_dst = self.model.neighbor_head(dst_hidden)
        neighbor_negative = self.model.neighbor_head(negative_hidden)

        augmented_forward, beta_forward = gsdn_eq3(
            self.feats[src],
            self.feats[dst],
            src_hidden,
            dst_hidden,
            self.model.mix_projection.weight.t(),
            self.model.attention,
            self.model.neighbor_head.weight,
            self.model.neighbor_head.bias,
        )
        augmented_reverse, beta_reverse = gsdn_eq3(
            self.feats[dst],
            self.feats[src],
            dst_hidden,
            src_hidden,
            self.model.mix_projection.weight.t(),
            self.model.attention,
            self.model.neighbor_head.weight,
            self.model.neighbor_head.bias,
        )
        feature_loss = gsdn_eq7(
            target_src,
            target_dst,
            augmented_forward,
            augmented_reverse,
            neighbor_negative,
        )
        label_loss = gsdn_eq8(
            endpoint_nodes,
            target_endpoints,
            neighbor_dst,
            neighbor_src,
            src,
            dst,
            self.noisy_label,
            self.train_idx,
        )
        self.gsdn_faithful_diagnostics.update(
            {
                "last_beta_min": float(
                    torch.minimum(
                        beta_forward.min(), beta_reverse.min()
                    )
                    .detach()
                    .cpu()
                ),
                "last_beta_max": float(
                    torch.maximum(
                        beta_forward.max(), beta_reverse.max()
                    )
                    .detach()
                    .cpu()
                ),
                "last_feature_loss": float(
                    feature_loss.detach().cpu()
                ),
                "last_label_loss": float(label_loss.detach().cpu()),
            }
        )
        return label_loss + self.lambda_feature * feature_loss

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            self.model.train()
            seen_edge_ids = []
            epoch_loss_sum = 0.0
            epoch_edges = 0
            batch_count = 0
            t0 = time.time()

            for batch_edges, edge_ids in iter_edge_batches(
                self.training_edge_index, self.graph_batch_size
            ):
                negative_nodes = torch.randint(
                    self.n_nodes,
                    (batch_edges.shape[1],),
                    device=self.device,
                )
                self.optim.zero_grad()
                loss_train = self._training_batch_loss(
                    batch_edges, negative_nodes
                )
                loss_train.backward()
                self.optim.step()

                batch_edges_count = int(batch_edges.shape[1])
                epoch_loss_sum += (
                    float(loss_train.detach().cpu()) * batch_edges_count
                )
                epoch_edges += batch_edges_count
                batch_count += 1
                seen_edge_ids.append(edge_ids.detach())

            seen = torch.cat(seen_edge_ids)
            unique_seen = int(torch.unique(seen).numel())
            expected_edges = int(self.training_edge_index.shape[1])
            exact_coverage = int(
                seen.numel() == expected_edges
                and unique_seen == expected_edges
            )
            self.gsdn_faithful_diagnostics.update(
                {
                    "edge_batches_per_epoch": batch_count,
                    "edges_seen_last_epoch": int(seen.numel()),
                    "unique_edges_seen_last_epoch": unique_seen,
                    "edge_coverage_fraction": (
                        unique_seen / expected_edges
                    ),
                    "edge_coverage_exact": exact_coverage,
                    "last_completed_epoch": epoch + 1,
                }
            )
            if not exact_coverage:
                raise RuntimeError(
                    "GSDN edge coverage invariant failed within an epoch"
                )

            self.model.eval()
            with torch.no_grad():
                full_output = self.model(self.feats)
                acc_train = self.metric(
                    self.noisy_label[self.train_idx].cpu().numpy(),
                    full_output[self.train_idx].cpu().numpy(),
                )
            loss_val, acc_val = self.evaluate(
                self.clean_label, self.val_mask
            )
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            improve = ""
            if flag:
                improve = "*"
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.best_epoch = epoch + 1
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | "
                    "Acc(val) {:.4f} | EdgeBatches {} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        epoch_loss_sum / max(epoch_edges, 1),
                        acc_train,
                        float(loss_val),
                        acc_val,
                        batch_count,
                        improve,
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self.gsdn_faithful_diagnostics.update(
            {
                "best_epoch": self.best_epoch,
                "deployed_state_components": "backbone,target_head",
                "discarded_after_training": (
                    "neighbor_head,mix_projection,attention"
                ),
            }
        )
        return self.result
