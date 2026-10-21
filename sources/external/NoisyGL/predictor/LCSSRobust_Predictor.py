import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.FrozenGraceGB_GCN_Predictor import _FrozenGraceGBGCNBase
from predictor.MLPNoiseBaselines_Predictor import (
    clip_logits_by_l2_norm,
    remember_count,
)
from predictor.module.GNNs import MLP


def soft_cross_entropy_per_sample(logits, targets):
    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError("logits and targets must both be 2-D")
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}")
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)


def select_soft_small_loss_indices(
    losses,
    epoch,
    noise_rate,
    num_gradual=10,
    exponent=1.0,
):
    if losses.ndim != 1:
        raise ValueError("losses must be a 1-D tensor")
    keep = remember_count(
        int(losses.numel()),
        int(epoch),
        float(noise_rate),
        int(num_gradual),
        float(exponent),
    )
    return torch.topk(losses.detach(), keep, largest=False).indices


def soft_jocor_per_sample(logits1, logits2, targets, co_lambda=0.1):
    co_lambda = float(co_lambda)
    if not 0.0 <= co_lambda <= 1.0:
        raise ValueError("co_lambda must be between 0 and 1")
    ce1 = soft_cross_entropy_per_sample(logits1, targets)
    ce2 = soft_cross_entropy_per_sample(logits2, targets)
    prob1 = F.softmax(logits1, dim=1)
    prob2 = F.softmax(logits2, dim=1)
    kl12 = F.kl_div(F.log_softmax(logits1, dim=1), prob2, reduction="none").sum(dim=1)
    kl21 = F.kl_div(F.log_softmax(logits2, dim=1), prob1, reduction="none").sum(dim=1)
    return (1.0 - co_lambda) * (ce1 + ce2) + co_lambda * (kl12 + kl21)


class _LCSSRobustBase(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    robust_strategy = "logitclip"
    strategy = "lcss_logitclip"

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.robust_epoch = 0
        self.noise_baseline_diagnostics = {
            "robust_strategy": self.robust_strategy,
        }

    def _robust_conf_value(self, key, default):
        value = self.conf.model.get(key, default)
        return default if value is None else value

    def _active_noise_rate(self):
        configured = self._robust_conf_value("forget_rate", "auto")
        if str(configured).lower() != "auto":
            return float(configured)
        return float(getattr(self, "noise_rate", 0.4))

    def _refresh_dynamic_soft_targets(self, epoch):
        self.robust_epoch = int(epoch)
        return super()._refresh_dynamic_soft_targets(epoch)

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if label is None or mask is None:
            return output, loss, acc

        if not isinstance(label, torch.Tensor) or label.ndim != 2:
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc

        logits = output[mask]
        targets = label[mask]
        if self.robust_strategy == "logitclip":
            tau = self._robust_conf_value("logit_clip_tau", 1.0)
            train_logits = clip_logits_by_l2_norm(logits, tau)
            losses = soft_cross_entropy_per_sample(train_logits, targets)
            loss = losses.mean()
            self.noise_baseline_diagnostics.update(
                {
                    "logit_clip_tau": float(tau),
                    "robust_selected_count": int(losses.numel()),
                    "robust_selected_ratio": 1.0,
                }
            )
        elif self.robust_strategy == "soft_small_loss":
            losses = soft_cross_entropy_per_sample(logits, targets)
            selected = select_soft_small_loss_indices(
                losses,
                epoch=self.robust_epoch,
                noise_rate=self._active_noise_rate(),
                num_gradual=self._robust_conf_value("num_gradual", 10),
                exponent=self._robust_conf_value("exponent", 1.0),
            )
            selection_hook = getattr(self, "_record_selection", None)
            if selection_hook is not None:
                if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                    global_train_idx = torch.nonzero(mask, as_tuple=False).flatten()
                else:
                    global_train_idx = torch.as_tensor(mask, device=self.device, dtype=torch.long)
                selection_hook(self.robust_epoch, global_train_idx, selected)
            loss = losses[selected].mean()
            self.noise_baseline_diagnostics.update(
                {
                    "forget_rate": self._active_noise_rate(),
                    "robust_selected_count": int(selected.numel()),
                    "robust_selected_ratio": float(selected.numel() / max(1, losses.numel())),
                }
            )
        else:
            raise ValueError(f"unsupported LCSS robust strategy: {self.robust_strategy}")

        hard_label = targets.argmax(dim=1)
        acc = self.metric(hard_label.detach().cpu().numpy(), logits.detach().cpu().numpy())
        return output, loss, acc


class frozengrace_gb_mlp_softdist_beta05_logitclip_Predictor(_LCSSRobustBase):
    robust_strategy = "logitclip"
    strategy = "lcss_logitclip"


class frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled_Predictor(_LCSSRobustBase):
    robust_strategy = "logitclip"
    strategy = "lcss_logitclip_shuffled"
    random_control = True


class frozengrace_gb_mlp_softdist_beta05_small_loss_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_soft_small_loss"


class frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_soft_small_loss_shuffled"
    random_control = True


class frozengrace_gb_mlp_softdist_beta05_small_loss_loo_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_loo_soft_small_loss"


class frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_loo_soft_small_loss_shuffled"
    random_control = True


class frozengrace_gb_mlp_softdist_beta05_small_loss_loo_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_loo_soft_small_loss"


class frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled_Predictor(_LCSSRobustBase):
    robust_strategy = "soft_small_loss"
    strategy = "lcss_loo_soft_small_loss_shuffled"
    random_control = True


class _LCSSSoftJoCoRBase(_LCSSRobustBase):
    robust_strategy = "soft_jocor"
    strategy = "lcss_soft_jocor"
    uses_two_models = True

    def method_init(self, conf, data):
        super().method_init(conf, data)
        in_channels = int(self.feats.shape[1])
        self.model2 = MLP(
            in_channels=in_channels,
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.model2.parameters()),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )

    def get_prediction(self, features, adj, label=None, mask=None):
        output1 = self.model(features, adj)
        output2 = self.model2(features, adj)
        output = 0.5 * (output1 + output2)
        loss, acc = None, None
        if label is None or mask is None:
            return output, loss, acc

        if not isinstance(label, torch.Tensor) or label.ndim != 2:
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc

        targets = label[mask]
        per_sample = soft_jocor_per_sample(
            output1[mask],
            output2[mask],
            targets,
            co_lambda=self._robust_conf_value("co_lambda", 0.1),
        )
        selected = select_soft_small_loss_indices(
            per_sample,
            epoch=self.robust_epoch,
            noise_rate=self._active_noise_rate(),
            num_gradual=self._robust_conf_value("num_gradual", 10),
            exponent=self._robust_conf_value("exponent", 1.0),
        )
        loss = per_sample[selected].mean()
        self.noise_baseline_diagnostics.update(
            {
                "co_lambda": float(self._robust_conf_value("co_lambda", 0.1)),
                "forget_rate": self._active_noise_rate(),
                "robust_selected_count": int(selected.numel()),
                "robust_selected_ratio": float(selected.numel() / max(1, per_sample.numel())),
            }
        )
        hard_label = targets.argmax(dim=1)
        acc = self.metric(hard_label.detach().cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        self._make_corrected_labels()
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self._refresh_dynamic_soft_targets(epoch)
            self.model.train()
            self.model2.train()
            self.optim.zero_grad()
            _, loss_train, acc_train = self.get_prediction(
                self.feats,
                self.adj,
                self.soft_target,
                self.train_mask,
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
                    "model2": deepcopy(self.model2.state_dict()),
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
        self._append_diagnostics()
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.model2.eval()
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, self.adj, label, mask)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights["model"])
            self.model2.load_state_dict(self.weights["model2"])
        return self.evaluate(self.clean_label, mask)


class frozengrace_gb_mlp_softdist_beta05_soft_jocor_Predictor(_LCSSSoftJoCoRBase):
    pass


class frozengrace_gb_mlp_softdist_beta05_soft_jocor_shuffled_Predictor(_LCSSSoftJoCoRBase):
    random_control = True
    strategy = "lcss_soft_jocor_shuffled"
