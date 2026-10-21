"""Graph-derived soft-target baselines with an X-only MLP at inference.

The three predictors in this module share the project's two-layer MLP but do
not share a target construction rule:

* ``postel_mlp_noipl_rm`` follows the official PosteL ``postel`` code path,
  with iterative pseudo-labeling disabled.
* ``als_mlp_rm`` implements ALS label propagation, its trainable global class
  relevance matrix, KL regularizer, and the published full-batch MLP
  exponential pacing family.
* ``sals_mlp_adapted`` applies the published SALS target equation to the
  project's MLP.  It is an adaptation because the original paper did not
  report this exact MLP backbone.

Only noisy labels indexed by ``train_mask`` participate in target creation.
Validation and test labels are never read by the target builders.
"""

import hashlib
import math
import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import MLP


def _as_index(mask, device):
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


def tensor_sha256(tensor):
    """Return a stable SHA-256 over tensor metadata and canonical CPU bytes."""
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_edge_index(edge_index, num_nodes, device):
    edge_index = torch.as_tensor(edge_index, device=device).long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    if edge_index.min() < 0 or edge_index.max() >= int(num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")
    linear = edge_index[0] * int(num_nodes) + edge_index[1]
    linear = torch.unique(linear, sorted=True)
    return torch.stack(
        [torch.div(linear, int(num_nodes), rounding_mode="floor"), linear % int(num_nodes)],
        dim=0,
    )


def _train_lookup(train_idx, num_nodes, device):
    lookup = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    lookup[train_idx] = True
    return lookup


def _neighbor_label_histograms(
    noisy_labels,
    train_idx,
    edge_index,
    num_nodes,
    n_classes,
):
    device = noisy_labels.device
    edge_index = _canonical_edge_index(edge_index, num_nodes, device)
    train_lookup = _train_lookup(train_idx, num_nodes, device)
    src, dst = edge_index
    keep = train_lookup[src] & train_lookup[dst]
    src = src[keep]
    dst = dst[keep]
    histograms = torch.zeros(
        (int(num_nodes), int(n_classes)),
        dtype=torch.float64,
        device=device,
    )
    if src.numel():
        histograms.index_put_(
            (src, noisy_labels[dst].long()),
            torch.ones(src.numel(), dtype=torch.float64, device=device),
            accumulate=True,
        )
    return histograms, edge_index


def _base_diagnostics(edge_index, fallback_mask):
    return {
        "edge_sha256": tensor_sha256(edge_index),
        "fallback_sha256": tensor_sha256(fallback_mask),
        "fallback_count": int(fallback_mask.sum().item()),
        "_fallback_mask": fallback_mask,
    }


def postel_prob_label_from_likelihood(likelihood, prior, temperature=1.0):
    """Mirror PosteL ``prob_label_from_log_likelihood`` after its product step."""
    likelihood = torch.as_tensor(likelihood)
    prior = torch.as_tensor(
        prior, dtype=likelihood.dtype, device=likelihood.device
    )
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_prob = torch.log(likelihood) + torch.log(prior)
    log_prob = log_prob - log_prob.max()
    probability = torch.exp(log_prob).pow(temperature)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    invalid = (~torch.isfinite(probability)).any(dim=1)
    return probability, invalid


def build_postel_targets(
    noisy_labels,
    train_mask,
    edge_index,
    *,
    num_nodes,
    n_classes,
    degree_cutoff=1,
    temperature=1.0,
    smoothing_ratio=0.4,
    soft_label_ratio=0.8,
):
    """Build official-code-faithful PosteL labels without pseudo-labeling.

    This mirrors ``ml-postech/PosteL/relabel.py::postel``: estimate empirical
    class-conditional neighbor-label distributions and priors from training
    nodes, compute a posterior by a product likelihood, apply the official
    degree/invalid-posterior fallback, then apply its smoothing and hard-label
    interpolation in the same order.
    """
    labels = torch.as_tensor(noisy_labels).long()
    device = labels.device
    train_idx = _as_index(train_mask, device)
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("PosteL requires at least two classes")
    if train_idx.numel() == 0:
        raise ValueError("PosteL requires at least one training node")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    hard = F.one_hot(labels[train_idx], n_classes).to(torch.float64)
    prior = hard.mean(dim=0)
    histograms, canonical_edges = _neighbor_label_histograms(
        labels, train_idx, edge_index, num_nodes, n_classes
    )

    conditional = torch.empty(
        (n_classes, n_classes), dtype=torch.float64, device=device
    )
    uniform = torch.full(
        (n_classes,), 1.0 / n_classes, dtype=torch.float64, device=device
    )
    for class_id in range(n_classes):
        class_nodes = train_idx[labels[train_idx] == class_id]
        class_hist = histograms[class_nodes].sum(dim=0)
        total = class_hist.sum()
        conditional[class_id] = class_hist / total if total > 0 else uniform

    train_hist = histograms[train_idx]
    likelihood = conditional.unsqueeze(0).pow(train_hist.unsqueeze(1)).prod(dim=2)
    posterior, invalid = postel_prob_label_from_likelihood(
        likelihood, prior, temperature
    )
    low_degree = train_hist.sum(dim=1) < int(degree_cutoff)
    fallback_train = invalid | low_degree
    posterior[fallback_train] = hard[fallback_train]

    smoothing_ratio = float(smoothing_ratio)
    soft_label_ratio = float(soft_label_ratio)
    if not 0.0 <= smoothing_ratio <= 1.0:
        raise ValueError("smoothing_ratio must lie in [0, 1]")
    if not 0.0 <= soft_label_ratio <= 1.0:
        raise ValueError("soft_label_ratio must lie in [0, 1]")
    smoothed = (
        (1.0 - smoothing_ratio) * posterior
        + smoothing_ratio / (n_classes - 1) * (~hard.bool()).to(torch.float64)
    )
    train_targets = soft_label_ratio * smoothed + (1.0 - soft_label_ratio) * hard

    targets = torch.full(
        (int(num_nodes), n_classes),
        1.0 / n_classes,
        dtype=torch.float32,
        device=device,
    )
    targets[train_idx] = train_targets.to(torch.float32)
    fallback_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    fallback_mask[train_idx] = fallback_train
    diagnostics = _base_diagnostics(canonical_edges, fallback_mask)
    diagnostics.update(
        {
            "postel_degree_cutoff": int(degree_cutoff),
            "postel_temperature": temperature,
            "postel_smoothing_ratio": smoothing_ratio,
            "postel_soft_label_ratio": soft_label_ratio,
            "postel_iterative_pseudo_labeling": 0,
            "postel_conditional_sha256": tensor_sha256(conditional),
            "postel_prior_sha256": tensor_sha256(prior),
        }
    )
    return targets, diagnostics


def als_label_propagation(
    noisy_labels,
    train_mask,
    edge_index,
    *,
    num_nodes,
    n_classes,
    steps=2,
    residual=0.5,
):
    """Compute ALS residual label propagation from a train-only label state."""
    labels = torch.as_tensor(noisy_labels).long()
    device = labels.device
    train_idx = _as_index(train_mask, device)
    edge_index = _canonical_edge_index(edge_index, num_nodes, device)
    steps = int(steps)
    residual = float(residual)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 <= residual <= 1.0:
        raise ValueError("residual must lie in [0, 1]")

    initial = torch.zeros(
        (int(num_nodes), int(n_classes)), dtype=torch.float32, device=device
    )
    initial[train_idx] = F.one_hot(
        labels[train_idx], int(n_classes)
    ).to(torch.float32)
    src, dst = edge_index
    degree = torch.bincount(src, minlength=int(num_nodes)).to(torch.float32)
    if src.numel():
        values = degree[src].clamp_min(1.0).rsqrt() * degree[dst].clamp_min(1.0).rsqrt()
        normalized_adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            (int(num_nodes), int(num_nodes)),
            device=device,
        ).coalesce()
    else:
        normalized_adj = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            (int(num_nodes), int(num_nodes)),
            device=device,
        ).coalesce()

    propagated = initial
    for _ in range(steps):
        propagated = (
            (1.0 - residual) * torch.sparse.mm(normalized_adj, propagated)
            + residual * initial
        )
    fallback_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    fallback_mask[train_idx] = propagated[train_idx].sum(dim=1) <= 0
    diagnostics = _base_diagnostics(edge_index, fallback_mask)
    diagnostics.update(
        {
            "als_lp_steps": steps,
            "als_lp_residual": residual,
            "als_propagated_sha256": tensor_sha256(propagated),
            "_initial_labels": initial,
        }
    )
    return propagated, diagnostics


def als_eq5_loss(logits, hard_labels, soft_targets, *, alpha, kl_weight):
    """ALS Eq. (5): hard/soft CE mixture plus KL(soft || uniform)."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    soft_targets = torch.as_tensor(
        soft_targets, dtype=logits.dtype, device=logits.device
    )
    hard_labels = torch.as_tensor(
        hard_labels, dtype=torch.long, device=logits.device
    )
    hard_loss = F.cross_entropy(logits, hard_labels)
    soft_loss = _soft_cross_entropy(logits, soft_targets)
    n_classes = int(soft_targets.shape[1])
    uniform_log = -math.log(float(n_classes))
    kl_uniform = (
        soft_targets
        * (soft_targets.clamp_min(1e-12).log() - uniform_log)
    ).sum(dim=1).mean()
    return (
        (1.0 - alpha) * hard_loss
        + alpha * soft_loss
        + float(kl_weight) * kl_uniform
    )


def als_pacing_alpha(epoch, *, pacing, alpha_max, rate, initial=0.1):
    """Published ALS pacing families; full-batch MLP uses exponential pacing."""
    epoch = float(epoch)
    alpha_max = float(alpha_max)
    rate = float(rate)
    initial = float(initial)
    if pacing == "linear":
        value = rate * epoch
    elif pacing == "exponential":
        value = initial * math.exp(rate * epoch)
    else:
        raise ValueError(f"unsupported ALS pacing family: {pacing}")
    return min(max(value, 0.0), alpha_max)


def build_sals_targets(
    noisy_labels,
    train_mask,
    edge_index,
    *,
    num_nodes,
    n_classes,
    epsilon=0.4,
    gamma=0.8,
):
    """Apply the published SALS equation using train-labeled neighbors only."""
    labels = torch.as_tensor(noisy_labels).long()
    device = labels.device
    train_idx = _as_index(train_mask, device)
    epsilon = float(epsilon)
    gamma = float(gamma)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    histograms, canonical_edges = _neighbor_label_histograms(
        labels, train_idx, edge_index, num_nodes, n_classes
    )
    train_hist = histograms[train_idx]
    counts = train_hist.sum(dim=1, keepdim=True)
    fallback_train = counts.squeeze(1) <= 0
    uniform = torch.full(
        (train_idx.numel(), int(n_classes)),
        1.0 / int(n_classes),
        dtype=torch.float64,
        device=device,
    )
    neighborhood = train_hist / counts.clamp_min(1.0)
    neighborhood[fallback_train] = uniform[fallback_train]
    hard = F.one_hot(labels[train_idx], int(n_classes)).to(torch.float64)
    train_targets = (
        (1.0 - epsilon) * hard
        + epsilon * (gamma * neighborhood + (1.0 - gamma) * uniform)
    )

    targets = torch.full(
        (int(num_nodes), int(n_classes)),
        1.0 / int(n_classes),
        dtype=torch.float32,
        device=device,
    )
    targets[train_idx] = train_targets.to(torch.float32)
    fallback_mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    fallback_mask[train_idx] = fallback_train
    diagnostics = _base_diagnostics(canonical_edges, fallback_mask)
    diagnostics.update(
        {
            "sals_epsilon": epsilon,
            "sals_gamma": gamma,
            "sals_unlabeled_neighbor_fallback": "uniform",
        }
    )
    return targets, diagnostics


def _soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class _NearestWorkSoftTargetBase(Predictor):
    method_kind = None

    def method_init(self, conf, data):
        self.train_idx = _as_index(self.train_mask, self.device)
        self.canonical_edges = _canonical_edge_index(
            self.edge_index, self.n_nodes, self.device
        )
        self.model = MLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.class_relevance = None
        self._init_targets(conf)
        parameters = list(self.model.parameters())
        if self.class_relevance is not None:
            parameters.append(self.class_relevance)
        self.optim = torch.optim.Adam(
            parameters,
            lr=float(conf.training["lr"]),
            weight_decay=float(conf.training["weight_decay"]),
        )
        self.best_epoch = -1
        self._refresh_target_diagnostics()

    def _conf(self, key, default):
        return self.conf.model.get(key, default)

    def _init_targets(self, conf):
        raise NotImplementedError

    def current_soft_targets(self):
        return self.soft_targets

    def _refresh_target_diagnostics(self):
        public = {
            key: value
            for key, value in self._target_diagnostics.items()
            if not key.startswith("_")
        }
        public.update(
            {
                "method_kind": self.method_kind,
                "training_uses_graph": 1,
                "inference_uses_graph": 0,
                "clean_label_target_reads": 0,
                "target_sha256": tensor_sha256(self.current_soft_targets()),
            }
        )
        self.method_diagnostics = public

    def get_prediction(self, features, adj, label=None, mask=None):
        # The graph is intentionally absent from deployed inference.
        output = self.model(features, None)
        loss, acc = None, None
        if label is not None and mask is not None:
            idx = _as_index(mask, self.device)
            loss = self.loss_fn(output[idx], label[idx])
            acc = self.metric(
                label[idx].detach().cpu().numpy(),
                output[idx].detach().cpu().numpy(),
            )
        return output, loss, acc

    def training_loss(self, logits, epoch):
        return _soft_cross_entropy(
            logits[self.train_idx], self.current_soft_targets()[self.train_idx]
        )

    def _state_snapshot(self):
        snapshot = {"model": deepcopy(self.model.state_dict())}
        if self.class_relevance is not None:
            snapshot["class_relevance"] = self.class_relevance.detach().clone()
        if "als_last_alpha" in self._target_diagnostics:
            snapshot["als_last_alpha"] = self._target_diagnostics["als_last_alpha"]
        return snapshot

    def _load_snapshot(self):
        if self.weights is None:
            return
        self.model.load_state_dict(self.weights["model"])
        if self.class_relevance is not None:
            with torch.no_grad():
                self.class_relevance.copy_(self.weights["class_relevance"])
        if "als_last_alpha" in self.weights:
            self._target_diagnostics["als_last_alpha"] = self.weights["als_last_alpha"]

    def train(self):
        for epoch in range(int(self.conf.training["n_epochs"])):
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            logits = self.model(self.feats, None)
            loss_train = self.training_loss(logits, epoch)
            loss_train.backward()
            self.optim.step()
            acc_train = self.metric(
                self.noisy_label[self.train_idx].detach().cpu().numpy(),
                logits[self.train_idx].detach().cpu().numpy(),
            )

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, early_stop = self.recoder.add(loss_val, acc_val)
            if flag:
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.best_epoch = epoch + 1
                self.weights = self._state_snapshot()
            elif early_stop:
                break
            if self.conf.training["debug"]:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
                        acc_train,
                        loss_val,
                        acc_val,
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self._refresh_target_diagnostics()
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, None, label, mask)
        return loss, acc

    def test(self, mask):
        self._load_snapshot()
        return self.evaluate(self.clean_label, mask)


class postel_mlp_noipl_rm_Predictor(_NearestWorkSoftTargetBase):
    method_kind = "postel_official_code_no_ipl_project_mlp"

    def _init_targets(self, conf):
        self.soft_targets, self._target_diagnostics = build_postel_targets(
            self.noisy_label,
            self.train_idx,
            self.canonical_edges,
            num_nodes=self.n_nodes,
            n_classes=self.n_classes,
            degree_cutoff=self._conf("degree_cutoff", 1),
            temperature=self._conf("temperature", 1.0),
            smoothing_ratio=self._conf("smoothing_ratio", 0.4),
            soft_label_ratio=self._conf("soft_label_ratio", 0.8),
        )


class als_mlp_rm_Predictor(_NearestWorkSoftTargetBase):
    method_kind = "als_complete_mechanism_project_mlp"

    def _init_targets(self, conf):
        self.propagated_targets, self._target_diagnostics = als_label_propagation(
            self.noisy_label,
            self.train_idx,
            self.canonical_edges,
            num_nodes=self.n_nodes,
            n_classes=self.n_classes,
            steps=self._conf("lp_steps", 2),
            residual=self._conf("lp_residual", 0.5),
        )
        relevance = torch.empty(
            (self.n_classes, self.n_classes),
            dtype=torch.float32,
            device=self.device,
        )
        torch.nn.init.xavier_uniform_(relevance)
        self.class_relevance = torch.nn.Parameter(relevance)
        self._target_diagnostics.update(
            {
                "als_kl_weight": float(self._conf("kl_weight", 1e-3)),
                "als_pacing": str(self._conf("pacing", "exponential")),
                "als_pacing_initial": float(
                    self._conf("pacing_initial", 0.1)
                ),
                "als_pacing_rate": float(self._conf("pacing_rate", -1e-3)),
                "als_alpha_max": float(self._conf("alpha_max", 0.1)),
            }
        )

    def _alpha(self, epoch):
        return als_pacing_alpha(
            epoch,
            pacing=str(self._conf("pacing", "exponential")),
            alpha_max=self._conf("alpha_max", 0.1),
            rate=self._conf("pacing_rate", -1e-3),
            initial=self._conf("pacing_initial", 0.1),
        )

    def _als_soft_train_targets(self):
        propagated = self.propagated_targets[self.train_idx]
        # Paper notation is W y; row-major batches therefore multiply by W^T.
        return F.softmax(propagated @ self.class_relevance.t(), dim=1)

    def current_soft_targets(self):
        targets = torch.full(
            (self.n_nodes, self.n_classes),
            1.0 / self.n_classes,
            dtype=torch.float32,
            device=self.device,
        )
        targets[self.train_idx] = self._als_soft_train_targets()
        return targets

    def training_loss(self, logits, epoch):
        alpha = self._alpha(epoch)
        soft_targets = self._als_soft_train_targets()
        self._target_diagnostics["als_last_alpha"] = alpha
        return als_eq5_loss(
            logits[self.train_idx],
            self.noisy_label[self.train_idx],
            soft_targets,
            alpha=alpha,
            kl_weight=self._conf("kl_weight", 1e-3),
        )


class sals_mlp_adapted_Predictor(_NearestWorkSoftTargetBase):
    method_kind = "sals_equation_faithful_project_mlp_adaptation"

    def _init_targets(self, conf):
        self.soft_targets, self._target_diagnostics = build_sals_targets(
            self.noisy_label,
            self.train_idx,
            self.canonical_edges,
            num_nodes=self.n_nodes,
            n_classes=self.n_classes,
            epsilon=self._conf("epsilon", 0.4),
            gamma=self._conf("gamma", 0.8),
        )
