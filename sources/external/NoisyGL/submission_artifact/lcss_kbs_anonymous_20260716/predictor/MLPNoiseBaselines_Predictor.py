import time
import math
import warnings
from statistics import NormalDist
from copy import deepcopy

import nni
import numpy as np
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.FrozenGraceGB_GCN_Predictor import concat_l2_normalized_embeddings
from predictor.module.GNNs import MLP
from utils.frozen_grace_targets import load_frozen_embeddings


def remember_count(num_examples, epoch, forget_rate, num_gradual=10, exponent=1.0):
    if num_examples <= 0:
        return 0
    forget_rate = max(0.0, min(float(forget_rate), 0.95))
    num_gradual = max(1, int(num_gradual))
    progress = min(1.0, float(epoch + 1) / float(num_gradual))
    current_forget = forget_rate * (progress ** float(exponent))
    return max(1, min(num_examples, int((1.0 - current_forget) * num_examples)))


def symmetric_cross_entropy_loss(logits, labels, alpha=0.1, beta=1.0):
    ce = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    one_hot = one_hot.clamp(min=1e-4, max=1.0)
    rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
    return float(alpha) * ce + float(beta) * rce


def clip_logits_by_l2_norm(logits, tau):
    if tau is None or str(tau).lower() in {"inf", "infinity", "none"}:
        return logits
    tau = float(tau)
    if tau <= 0:
        raise ValueError("logit clipping tau must be positive")
    norms = logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(tau / norms, max=1.0)
    return logits * scale


def normalized_cross_entropy_loss(logits, labels):
    log_probs = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(log_probs, labels, reduction="none")
    normalizer = (-log_probs.sum(dim=1)).clamp_min(1e-12)
    return (ce / normalizer).mean()


def normalized_negative_cross_entropy_loss(logits, labels, min_prob=1e-7):
    min_prob = float(min_prob)
    if not 0.0 < min_prob < 1.0:
        raise ValueError("ANL min_prob must be between 0 and 1")
    log_min_prob = math.log(min_prob)
    log_probs = F.log_softmax(logits, dim=1).clamp(min=log_min_prob)
    target_loss = -log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    normalizer = -log_probs.sum(dim=1)
    a = -log_min_prob
    denominator = (logits.shape[1] * a - normalizer).clamp_min(1e-12)
    return (1.0 - (a - target_loss) / denominator).mean()


def anl_ce_loss(
    logits,
    labels,
    active_weight=1.0,
    passive_weight=1.0,
    model=None,
    l1_weight=0.0,
    min_prob=1e-7,
):
    loss = (
        float(active_weight) * normalized_cross_entropy_loss(logits, labels)
        + float(passive_weight)
        * normalized_negative_cross_entropy_loss(logits, labels, min_prob=min_prob)
    )
    if model is not None and float(l1_weight) != 0.0:
        loss = loss + float(l1_weight) * sum(parameter.abs().sum() for parameter in model.parameters())
    return loss


def negative_normalized_negative_ce_loss(logits, labels):
    probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
    negative_log = -torch.log((1.0 - probs).clamp_min(1e-7))
    label_mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    passive = negative_log.masked_fill(label_mask, 0.0).sum(dim=1)
    normalizer = negative_log.sum(dim=1).clamp_min(1e-12)
    return (passive / normalizer).mean()


def active_negative_loss(logits, labels, active_weight=1.0, passive_weight=1.0):
    active = normalized_cross_entropy_loss(logits, labels)
    passive = negative_normalized_negative_ce_loss(logits, labels)
    return float(active_weight) * active + float(passive_weight) * passive


def jocor_small_loss(logits1, logits2, labels, forget_rate, co_lambda, epoch, num_gradual, exponent):
    ce1 = F.cross_entropy(logits1, labels, reduction="none")
    ce2 = F.cross_entropy(logits2, labels, reduction="none")
    prob1 = F.softmax(logits1, dim=1)
    prob2 = F.softmax(logits2, dim=1)
    kl12 = F.kl_div(F.log_softmax(logits1, dim=1), prob2, reduction="none").sum(dim=1)
    kl21 = F.kl_div(F.log_softmax(logits2, dim=1), prob1, reduction="none").sum(dim=1)
    per_sample = (1.0 - float(co_lambda)) * (ce1 + ce2) + float(co_lambda) * (kl12 + kl21)
    k = remember_count(labels.shape[0], epoch, forget_rate, num_gradual, exponent)
    selected = torch.topk(per_sample.detach(), k, largest=False).indices
    return per_sample[selected].mean()


def mlp_forward_with_hidden(model, features, adj=None):
    hidden = features
    for layer_idx, linear in enumerate(model.lins[:-1]):
        hidden = linear(hidden)
        hidden = F.relu(hidden)
        if model.use_bn:
            hidden = model.bns[layer_idx](hidden)
        hidden = F.dropout(hidden, p=model.dropout, training=model.training)
    logits = model.lins[-1](hidden)
    return logits, hidden


def fine_alignment_scores(features, labels, num_classes, eps=1e-12):
    scores = torch.ones(labels.shape[0], device=features.device, dtype=features.dtype)
    for class_id in range(int(num_classes)):
        class_idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_idx.numel() < 2:
            continue
        class_features = features[class_idx].detach().float()
        centered = class_features - class_features.mean(dim=0, keepdim=True)
        if float(centered.norm().detach().cpu()) <= eps:
            continue
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        except RuntimeError:
            covariance = centered.t().matmul(centered)
            _, eigenvectors = torch.linalg.eigh(covariance)
            direction = eigenvectors[:, -1]
        class_scores = F.cosine_similarity(
            centered,
            direction.unsqueeze(0).expand_as(centered),
            dim=1,
            eps=eps,
        ).abs()
        scores[class_idx] = class_scores.to(scores.dtype).clamp(0.0, 1.0)
    return scores


def fine_official_scores(features, labels, num_classes, eps=1e-12):
    scores = torch.ones(labels.shape[0], device=features.device, dtype=features.dtype)
    for class_id in range(int(num_classes)):
        class_idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_idx.numel() < 2:
            continue
        class_features = features[class_idx].detach().float()
        try:
            _, _, vh = torch.linalg.svd(class_features, full_matrices=False)
            direction = vh[0]
        except RuntimeError:
            gram = class_features.t().matmul(class_features)
            _, eigenvectors = torch.linalg.eigh(gram)
            direction = eigenvectors[:, -1]
        normalized = F.normalize(class_features, p=2, dim=1, eps=eps)
        class_scores = normalized.matmul(direction).abs()
        scores[class_idx] = class_scores.to(scores.dtype).clamp(0.0, 1.0)
    return scores


def fine_gmm_clean_mask(
    scores,
    labels,
    num_classes,
    p_threshold=0.5,
    random_state=0,
):
    clean_prob = torch.ones_like(scores)
    mask = torch.ones(labels.shape[0], dtype=torch.bool, device=scores.device)
    from sklearn.mixture import GaussianMixture

    for class_id in range(int(num_classes)):
        class_idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_idx.numel() < 2:
            continue
        values = scores[class_idx].detach().cpu().numpy().reshape(-1, 1)
        if float(values.max() - values.min()) <= 1e-12:
            continue
        try:
            gmm = GaussianMixture(
                n_components=2,
                covariance_type="full",
                tol=1e-6,
                max_iter=100,
                random_state=int(random_state),
                reg_covar=1e-6,
            )
            from sklearn.exceptions import ConvergenceWarning

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gmm.fit(values)
            clean_component = int(np.argmax(gmm.means_.reshape(-1)))
            probabilities = gmm.predict_proba(values)[:, clean_component]
            class_prob = torch.as_tensor(
                probabilities,
                device=scores.device,
                dtype=scores.dtype,
            )
            clean_prob[class_idx] = class_prob
            mask[class_idx] = class_prob > float(p_threshold)
            if not bool(mask[class_idx].any()):
                mask[class_idx[torch.argmax(class_prob)]] = True
        except Exception:
            mask[class_idx] = True
            clean_prob[class_idx] = 1.0
    return mask, clean_prob


def dividemix_lite_clean_prob(losses, random_state=0):
    detached = losses.detach()
    if detached.numel() < 2:
        return torch.ones_like(detached)
    low = detached.min()
    high = detached.max()
    span = high - low
    if float(span.detach().cpu()) <= 1e-12:
        return torch.ones_like(detached)
    normalized = ((detached - low) / span).cpu().numpy().reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=2, random_state=int(random_state), reg_covar=1e-6)
        gmm.fit(normalized)
        clean_component = int(np.argmin(gmm.means_.reshape(-1)))
        probabilities = gmm.predict_proba(normalized)[:, clean_component]
    except Exception:
        probabilities = 1.0 - normalized.reshape(-1)
    return torch.as_tensor(probabilities, device=losses.device, dtype=losses.dtype).clamp(0.0, 1.0)


def confidence_gap(probabilities, labels):
    annotated = probabilities.gather(1, labels.view(-1, 1))
    return annotated - probabilities


def update_mann_kendall_score(history, current, score=None):
    current = current.detach().cpu()
    if score is None:
        score = torch.zeros_like(current)
    else:
        score = score.detach().cpu()
    if history:
        previous = torch.stack([item.detach().cpu() for item in history], dim=0)
        score = score + (current.unsqueeze(0) > previous).sum(dim=0)
        score = score - (current.unsqueeze(0) < previous).sum(dim=0)
    return score


def mann_kendall_clean_mask(scores, labels, n_observations, alpha=0.01):
    n_observations = int(n_observations)
    if n_observations < 2:
        empty = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        return empty, torch.zeros(scores.shape[0], dtype=scores.dtype, device=scores.device)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("CT alpha must be between 0 and 1")

    competing_scores = scores.clone()
    row_index = torch.arange(labels.shape[0], device=scores.device)
    competing_scores[row_index, labels.to(scores.device)] = torch.inf
    minimum_score = competing_scores.min(dim=1).values
    variance = n_observations * (n_observations - 1) * (2 * n_observations + 5) / 18.0
    sqrt_variance = math.sqrt(variance)
    z = torch.where(
        minimum_score > 0,
        (minimum_score - 1.0) / sqrt_variance,
        (minimum_score + 1.0) / sqrt_variance,
    )
    z = torch.where(minimum_score == 0, torch.zeros_like(z), z)
    threshold = NormalDist().inv_cdf(1.0 - float(alpha))
    return z > threshold, z


def gmm_ct_clean_mask(
    losses,
    ct_mask,
    p_threshold=0.5,
    random_state=0,
    reg_covar=5e-4,
):
    detached = losses.detach()
    if detached.numel() < 2:
        clean_prob = torch.ones_like(detached)
    else:
        low = detached.min()
        span = detached.max() - low
        if float(span.detach().cpu()) <= 1e-12:
            clean_prob = torch.ones_like(detached)
        else:
            normalized = ((detached - low) / span).cpu().numpy().reshape(-1, 1)
            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(
                    n_components=2,
                    max_iter=10,
                    tol=1e-2,
                    reg_covar=float(reg_covar),
                    random_state=int(random_state),
                )
                gmm.fit(normalized)
                clean_component = int(np.argmin(gmm.means_.reshape(-1)))
                probabilities = gmm.predict_proba(normalized)[:, clean_component]
                clean_prob = torch.as_tensor(
                    probabilities,
                    device=losses.device,
                    dtype=losses.dtype,
                )
            except Exception:
                clean_prob = 1.0 - torch.as_tensor(
                    normalized.reshape(-1),
                    device=losses.device,
                    dtype=losses.dtype,
                )
    clean_prob = clean_prob.clamp(0.0, 1.0)
    gmm_mask = clean_prob > float(p_threshold)
    ct_mask = ct_mask.to(device=losses.device, dtype=torch.bool)
    return gmm_mask | ct_mask, gmm_mask, clean_prob


class _MLPNoiseBaselineBase(Predictor):
    strategy = "small_loss"
    use_xh_features = False

    def method_init(self, conf, data):
        in_channels = conf.model["n_feat"]
        self.xh_embedding_dim = 0
        if self.use_xh_features:
            embedding_path = self._feature_embedding_path()
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
            self.xh_embedding_dim = int(embeddings.shape[1])
            self.feats = concat_l2_normalized_embeddings(self.feats, embeddings).to(self.device)
            in_channels = int(self.feats.shape[1])
            conf.model["n_feat"] = in_channels
        self.model = MLP(
            in_channels=in_channels,
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.model2 = None
        if self.strategy in {"coteaching", "jocor"}:
            self.model2 = MLP(
                in_channels=in_channels,
                hidden_channels=conf.model["n_hidden"],
                out_channels=conf.model["n_classes"],
                n_layers=conf.model["n_layer"],
                dropout=conf.model["dropout"],
            ).to(self.device)
        parameters = list(self.model.parameters())
        if self.model2 is not None:
            parameters += list(self.model2.parameters())
        self.optim = torch.optim.Adam(
            parameters,
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        self.best_epoch = -1
        self.target_history = None
        self.ct_confidence_ema = None
        self.ct_gap_history = []
        self.ct_mk_score = None
        self.ct_active_mask = None
        self.noise_baseline_diagnostics = {}

    def _conf_value(self, key, default):
        return self.conf.model.get(key, default)

    def _gb_conf(self):
        return getattr(self.conf, "gb", {})

    def _gb_value(self, key, default=None):
        gb_conf = self._gb_conf()
        if isinstance(gb_conf, dict):
            value = gb_conf.get(key, default)
        else:
            value = getattr(gb_conf, key, default)
        return default if value is None else value

    def _feature_embedding_path(self):
        embedding_path = self._gb_value("embedding_path", None)
        if embedding_path is None:
            raise ValueError("gb.embedding_path is required for XH MLP noise baselines")
        return embedding_path

    def _train_index(self, mask):
        return torch.as_tensor(mask, device=self.device, dtype=torch.long)

    def _active_noise_rate(self):
        configured = self._conf_value("forget_rate", None)
        if configured is not None and str(configured).lower() != "auto":
            return float(configured)
        return float(getattr(self, "noise_rate", self._conf_value("noise_rate", 0.4)))

    def _remember_count(self, num_examples, epoch):
        return remember_count(
            num_examples,
            epoch,
            self._active_noise_rate(),
            self._conf_value("num_gradual", 10),
            self._conf_value("exponent", 1.0),
        )

    def _loss_and_output(self, features, adj, label, mask, epoch):
        idx = self._train_index(mask)
        labels = label[idx]
        if self.strategy in {"fine", "fine_official", "dividemix_lite"}:
            output, hidden = mlp_forward_with_hidden(self.model, features, adj)
        else:
            output = self.model(features, adj)
            hidden = None
        logits = output[idx]

        if self.strategy == "small_loss":
            losses = F.cross_entropy(logits, labels, reduction="none")
            k = self._remember_count(labels.shape[0], epoch)
            selected = torch.topk(losses.detach(), k, largest=False).indices
            selection_hook = getattr(self, "_record_selection", None)
            if selection_hook is not None:
                selection_hook(epoch, idx, selected)
            loss = losses[selected].mean()
        elif self.strategy == "coteaching":
            output2 = self.model2(features, adj)
            logits2 = output2[idx]
            losses1 = F.cross_entropy(logits, labels, reduction="none")
            losses2 = F.cross_entropy(logits2, labels, reduction="none")
            k = self._remember_count(labels.shape[0], epoch)
            selected1 = torch.topk(losses1.detach(), k, largest=False).indices
            selected2 = torch.topk(losses2.detach(), k, largest=False).indices
            loss = losses1[selected2].mean() + losses2[selected1].mean()
        elif self.strategy == "jocor":
            output2 = self.model2(features, adj)
            logits2 = output2[idx]
            loss = jocor_small_loss(
                logits,
                logits2,
                labels,
                self._active_noise_rate(),
                self._conf_value("co_lambda", 0.1),
                epoch,
                self._conf_value("num_gradual", 10),
                self._conf_value("exponent", 1.0),
            )
        elif self.strategy == "ct":
            losses = F.cross_entropy(logits, labels, reduction="none")
            probs = F.softmax(logits, dim=1)
            label_conf = probs.gather(1, labels.view(-1, 1)).squeeze(1).detach()
            if self.ct_confidence_ema is None:
                self.ct_confidence_ema = torch.zeros(self.n_nodes, device=self.device)
            decay = float(self._conf_value("ct_ema_decay", 0.9))
            decay = max(0.0, min(decay, 0.999))
            with torch.no_grad():
                previous = self.ct_confidence_ema[idx]
                updated = label_conf if epoch == 0 else decay * previous + (1.0 - decay) * label_conf
                self.ct_confidence_ema[idx] = updated
            warmup = int(self._conf_value("ct_warmup", 5))
            if epoch < warmup:
                loss = losses.mean()
            else:
                k = self._remember_count(labels.shape[0], epoch)
                scores = self.ct_confidence_ema[idx]
                selected = torch.topk(scores.detach(), k, largest=True).indices
                loss = losses[selected].mean()
        elif self.strategy == "gmm_ct":
            losses = F.cross_entropy(logits, labels, reduction="none")
            if self.ct_active_mask is None:
                self.ct_active_mask = torch.ones(
                    labels.shape[0], dtype=torch.bool, device=self.device
                )
            active_mask = self.ct_active_mask.to(self.device)
            if not bool(active_mask.any()):
                active_mask = torch.ones_like(active_mask)
            loss = losses[active_mask].mean()

            probabilities = F.softmax(logits.detach(), dim=1)
            gaps = confidence_gap(probabilities, labels).cpu()
            self.ct_mk_score = update_mann_kendall_score(
                self.ct_gap_history,
                gaps,
                self.ct_mk_score,
            )
            self.ct_gap_history.append(gaps)

            warmup = int(self._conf_value("ct_warmup", 30))
            if (epoch + 1) < warmup:
                next_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=self.device)
                ct_mask = torch.zeros_like(next_mask)
                gmm_mask = next_mask.clone()
                clean_prob = torch.ones_like(losses)
            else:
                ct_mask_cpu, _ = mann_kendall_clean_mask(
                    self.ct_mk_score,
                    labels.detach().cpu(),
                    n_observations=len(self.ct_gap_history),
                    alpha=self._conf_value("ct_alpha", 0.01),
                )
                next_mask, gmm_mask, clean_prob = gmm_ct_clean_mask(
                    losses,
                    ct_mask_cpu,
                    p_threshold=self._conf_value("ct_p_threshold", 0.5),
                    random_state=int(getattr(self, "seed", 0)),
                    reg_covar=self._conf_value("ct_reg_covar", 5e-4),
                )
                ct_mask = ct_mask_cpu.to(self.device)
                if not bool(next_mask.any()):
                    next_mask[torch.argmin(losses.detach())] = True
            self.ct_active_mask = next_mask
            history_bytes = sum(item.numel() * item.element_size() for item in self.ct_gap_history)
            self.noise_baseline_diagnostics.update(
                {
                    "ct_observations": len(self.ct_gap_history),
                    "ct_selected_count": int(active_mask.sum().item()),
                    "ct_selected_ratio": float(active_mask.float().mean().item()),
                    "ct_next_selected_count": int(next_mask.sum().item()),
                    "ct_next_selected_ratio": float(next_mask.float().mean().item()),
                    "ct_only_count": int(ct_mask.sum().item()),
                    "ct_gmm_count": int(gmm_mask.sum().item()),
                    "ct_clean_prob_mean": float(clean_prob.mean().item()),
                    "ct_history_bytes": int(history_bytes),
                }
            )
        elif self.strategy == "elr":
            if self.target_history is None:
                self.target_history = torch.zeros(
                    self.n_nodes, self.n_classes, device=self.device
                )
            probs = F.softmax(logits, dim=1)
            beta = float(self._conf_value("elr_beta", 0.7))
            with torch.no_grad():
                self.target_history[idx] = (
                    beta * self.target_history[idx] + (1.0 - beta) * probs.detach()
                )
                normalizer = self.target_history[idx].sum(dim=1, keepdim=True).clamp_min(1e-12)
                self.target_history[idx] = self.target_history[idx] / normalizer
            ce = F.cross_entropy(logits, labels)
            agreement = (self.target_history[idx] * probs).sum(dim=1).clamp(
                min=1e-4, max=1.0 - 1e-4
            )
            elr = torch.log(1.0 - agreement).mean()
            loss = ce + float(self._conf_value("elr_lambda", 1.0)) * elr
        elif self.strategy == "sce":
            loss = symmetric_cross_entropy_loss(
                logits,
                labels,
                alpha=self._conf_value("alpha", 0.1),
                beta=self._conf_value("beta", 1.0),
            )
        elif self.strategy == "logitclip":
            clipped_logits = clip_logits_by_l2_norm(
                logits,
                self._conf_value("logit_clip_tau", 1.0),
            )
            loss = F.cross_entropy(clipped_logits, labels)
        elif self.strategy == "anl":
            loss = active_negative_loss(
                logits,
                labels,
                active_weight=self._conf_value("anl_active_weight", 1.0),
                passive_weight=self._conf_value("anl_passive_weight", 1.0),
            )
        elif self.strategy == "anl_ce":
            loss = anl_ce_loss(
                logits,
                labels,
                active_weight=self._conf_value("anl_alpha", 5.0),
                passive_weight=self._conf_value("anl_beta", 5.0),
                model=self.model,
                l1_weight=self._conf_value("anl_delta", 5e-5),
                min_prob=self._conf_value("anl_min_prob", 1e-7),
            )
        elif self.strategy == "fine":
            scores = fine_alignment_scores(hidden[idx], labels, self.n_classes)
            k = self._remember_count(labels.shape[0], epoch)
            selected = torch.topk(scores.detach(), k, largest=True).indices
            loss = F.cross_entropy(logits[selected], labels[selected])
        elif self.strategy == "fine_official":
            warmup = int(self._conf_value("fine_warmup", 10))
            if epoch < warmup:
                selected_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=self.device)
                clean_prob = torch.ones(labels.shape[0], dtype=logits.dtype, device=self.device)
            else:
                scores = fine_official_scores(hidden[idx], labels, self.n_classes)
                selected_mask, clean_prob = fine_gmm_clean_mask(
                    scores,
                    labels,
                    self.n_classes,
                    p_threshold=self._conf_value("fine_p_threshold", 0.5),
                    random_state=int(getattr(self, "seed", 0)),
                )
            loss = F.cross_entropy(logits[selected_mask], labels[selected_mask])
            selected_classes = torch.unique(labels[selected_mask]).numel()
            self.noise_baseline_diagnostics.update(
                {
                    "fine_selected_count": int(selected_mask.sum().item()),
                    "fine_selected_ratio": float(selected_mask.float().mean().item()),
                    "fine_selected_classes": int(selected_classes),
                    "fine_clean_prob_mean": float(clean_prob.mean().item()),
                }
            )
        elif self.strategy == "dividemix_lite":
            losses = F.cross_entropy(logits, labels, reduction="none")
            warmup = int(self._conf_value("dm_warmup", 10))
            if epoch < warmup:
                loss = losses.mean()
            else:
                clean_prob = dividemix_lite_clean_prob(
                    losses,
                    random_state=int(getattr(self, "seed", 0)),
                )
                pseudo = F.softmax(logits.detach(), dim=1)
                pseudo_loss = -(pseudo * F.log_softmax(logits, dim=1)).sum(dim=1)
                loss = (clean_prob * losses + (1.0 - clean_prob) * pseudo_loss).mean()
        else:
            raise ValueError(f"unsupported MLP noise strategy: {self.strategy}")

        acc = self.metric(labels.detach().cpu().numpy(), logits.detach().cpu().numpy())
        return output, loss, acc

    def get_prediction(self, features, adj, label=None, mask=None, epoch=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            idx = self._train_index(mask)
            loss = F.cross_entropy(output[idx], label[idx])
            acc = self.metric(label[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            if self.model2 is not None:
                self.model2.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            _, loss_train, acc_train = self._loss_and_output(
                features, adj, self.noisy_label, self.train_mask, epoch
            )
            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = "*"
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.best_epoch = epoch + 1
                self.weights = {
                    "model": deepcopy(self.model.state_dict()),
                    "model2": deepcopy(self.model2.state_dict()) if self.model2 is not None else None,
                }
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
                        acc_train,
                        loss_val,
                        acc_val,
                        improve,
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        if self.conf.training["debug"]:
            print("Optimization Finished!")
            print("Time(s): {:.4f}".format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights["model"])
            if self.model2 is not None and self.weights["model2"] is not None:
                self.model2.load_state_dict(self.weights["model2"])
        return self.evaluate(self.clean_label, mask)


class mlp_small_loss_Predictor(_MLPNoiseBaselineBase):
    strategy = "small_loss"


class mlp_coteaching_Predictor(_MLPNoiseBaselineBase):
    strategy = "coteaching"


class mlp_jocor_Predictor(_MLPNoiseBaselineBase):
    strategy = "jocor"


class mlp_small_loss_xh_Predictor(mlp_small_loss_Predictor):
    use_xh_features = True


class mlp_jocor_xh_Predictor(mlp_jocor_Predictor):
    use_xh_features = True


class mlp_ct_Predictor(_MLPNoiseBaselineBase):
    strategy = "ct"


class mlp_elr_Predictor(_MLPNoiseBaselineBase):
    strategy = "elr"


class mlp_sce_Predictor(_MLPNoiseBaselineBase):
    strategy = "sce"


class mlp_logitclip_Predictor(_MLPNoiseBaselineBase):
    strategy = "logitclip"


class mlp_anl_Predictor(_MLPNoiseBaselineBase):
    strategy = "anl"


class mlp_anl_ce_Predictor(_MLPNoiseBaselineBase):
    strategy = "anl_ce"


class mlp_gmm_ct_Predictor(_MLPNoiseBaselineBase):
    strategy = "gmm_ct"


class mlp_fine_Predictor(_MLPNoiseBaselineBase):
    strategy = "fine"


class mlp_fine_official_Predictor(_MLPNoiseBaselineBase):
    strategy = "fine_official"


class mlp_dividemix_lite_Predictor(_MLPNoiseBaselineBase):
    strategy = "dividemix_lite"


class linear_small_loss_Predictor(mlp_small_loss_Predictor):
    pass


class linear_jocor_Predictor(mlp_jocor_Predictor):
    pass


class linear_ct_Predictor(mlp_ct_Predictor):
    pass


class linear_dividemix_lite_Predictor(mlp_dividemix_lite_Predictor):
    pass
