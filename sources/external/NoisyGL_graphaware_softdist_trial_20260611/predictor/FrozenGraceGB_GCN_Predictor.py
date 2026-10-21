import csv
import os
import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    ball_consistency_loss,
    build_ball_expansion_soft_targets,
    build_frozen_grace_soft_targets,
    build_frozen_grace_targets,
    diagnostics_for_soft_targets,
    expansion_soft_target_loss,
    load_frozen_embeddings,
)


def concat_l2_normalized_embeddings(features, embeddings):
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D tensor")
    if features.shape[0] != embeddings.shape[0]:
        raise ValueError(f"feature node count {features.shape[0]} != embedding node count {embeddings.shape[0]}")
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    features = features.float()
    embeddings = embeddings.to(device=features.device, dtype=features.dtype)
    norm = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    embeddings = embeddings / norm
    return torch.cat([features, embeddings], dim=1)


class _FrozenGraceGBGCNBase(Predictor):
    random_control = False
    use_soft_targets = False
    use_dynamic_alpha = False
    use_xh_features = False
    skip_ball_targets = False
    use_ball_expansion = False
    use_topology_alpha = False

    def method_init(self, conf, data):
        in_channels = conf.model['n_feat']
        self.xh_embedding_dim = 0
        if self.use_xh_features:
            embedding_path = self._conf_value("embedding_path", None)
            if embedding_path is None:
                raise ValueError("gb.embedding_path is required for XH feature concatenation")
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
            self.xh_embedding_dim = int(embeddings.shape[1])
            self.feats = concat_l2_normalized_embeddings(self.feats, embeddings).to(self.device)
            in_channels = int(self.feats.shape[1])
            conf.model['n_feat'] = in_channels
        self.model = GCN(in_channels=in_channels, hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                         dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.dataset_name = data.name
        self.corrected_label = None
        self.soft_target = None
        self.soft_alpha = None
        self.soft_pseudo_label = None
        self.gb_diagnostics = {}
        self.best_epoch = -1
        self.dynamic_refresh_count = 0
        self.lambda_bc = 0.0
        self.last_bc_loss = 0.0
        self.lambda_u = 0.0
        self.expansion_target = None
        self.expansion_mask = None
        self.last_expansion_loss = 0.0

    def _gb_conf(self):
        return getattr(self.conf, "gb", {})

    def _conf_value(self, key, default):
        gb_conf = self._gb_conf()
        if isinstance(gb_conf, dict):
            value = gb_conf.get(key, default)
        else:
            value = getattr(gb_conf, key, default)
        return default if value is None else value

    def _ball_consistency_weight(self):
        return float(self._conf_value("lambda_bc", 0.0))

    def _make_corrected_labels(self):
        if self.skip_ball_targets:
            self.corrected_label = self.noisy_label.detach().clone().to(self.device)
            self.soft_target = None
            self.soft_alpha = None
            self.soft_pseudo_label = None
            self.gb_assignment = None
            self.gb_diagnostics = {
                "skip_ball_targets": 1,
                "use_xh_features": int(self.use_xh_features),
                "xh_feature_dim": int(self.feats.shape[1]),
                "xh_embedding_dim": int(self.xh_embedding_dim),
            }
            if self.conf.training['debug']:
                print("Frozen-GRACE GB diagnostics:", self.gb_diagnostics)
            self.lambda_bc = 0.0
            return
        embedding_path = self._conf_value("embedding_path", None)
        if embedding_path is None:
            raise ValueError("gb.embedding_path is required for frozen GRACE GB-GCN")
        seed = int(getattr(self, "seed", torch.initial_seed() % (2 ** 32 - 1)))
        common_kwargs = {
            "embedding_path": embedding_path,
            "noisy_label": self.noisy_label,
            "clean_label": self.clean_label,
            "train_mask": self.train_mask,
            "n_classes": self.n_classes,
            "max_balls": self._conf_value("max_balls", 64),
            "min_ball_size": self._conf_value("min_ball_size", 32),
            "split_improvement": self._conf_value("split_improvement", 0.05),
            "min_train_per_ball": self._conf_value("min_train_per_ball", 2),
            "random_state": seed,
            "random_control": self.random_control,
        }
        if self.use_soft_targets:
            result = build_frozen_grace_soft_targets(
                **common_kwargs,
                alpha_baseline=self._conf_value("alpha_baseline", "binary"),
                target_mode=self._conf_value("target_mode", "majority"),
                ball_mode=self._conf_value("ball_mode", "geometric"),
                purity_threshold=self._conf_value("purity_threshold", 0.6),
                min_train_per_child=self._conf_value("min_train_per_child", 4),
                min_train_to_split=self._conf_value("min_train_to_split", 16),
                adaptive_min_ball_size=self._conf_value("adaptive_min_ball_size", 8),
                adaptive_max_balls=self._conf_value("adaptive_max_balls", 128),
                min_purity_gain=self._conf_value("min_purity_gain", 0.05),
                split_attempts=self._conf_value("split_attempts", 1),
                adj=self.adj,
                topology_alpha=self.use_topology_alpha or bool(self._conf_value("topology_alpha", False)),
                topology_alpha_floor=self._conf_value("topology_alpha_floor", 0.5),
            )
        else:
            result = build_frozen_grace_targets(
                **common_kwargs,
                min_majority_confidence=self._conf_value("min_majority_confidence", 0.0),
            )
        self.corrected_label = result.corrected_label.to(self.device)
        self.soft_target = result.soft_target.to(self.device) if result.soft_target is not None else None
        self.soft_alpha = result.alpha.to(self.device) if result.alpha is not None else None
        self.soft_pseudo_label = result.pseudo_label.to(self.device) if result.pseudo_label is not None else None
        self.gb_assignment = result.assignment
        self.gb_diagnostics = result.diagnostics
        self.gb_diagnostics["use_xh_features"] = int(self.use_xh_features)
        self.gb_diagnostics["skip_ball_targets"] = int(self.skip_ball_targets)
        self.gb_diagnostics["xh_feature_dim"] = int(self.feats.shape[1])
        self.gb_diagnostics["xh_embedding_dim"] = int(self.xh_embedding_dim)
        if self.conf.training['debug']:
            print("Frozen-GRACE GB diagnostics:", self.gb_diagnostics)
        self.lambda_bc = self._ball_consistency_weight()
        if self.lambda_bc > 0:
            self.gb_diagnostics["lambda_bc"] = self.lambda_bc
            self.gb_diagnostics["bc_min_ball_size"] = int(self._conf_value("bc_min_ball_size", 2))
        self.lambda_u = float(self._conf_value("lambda_u", 0.0)) if self.use_ball_expansion else 0.0
        if self.use_ball_expansion:
            expansion = build_ball_expansion_soft_targets(
                noisy_label=self.noisy_label,
                assignment=self.gb_assignment,
                train_mask=self.train_mask,
                n_classes=self.n_classes,
                min_train_per_ball=self._conf_value("expansion_min_train_per_ball", 10),
                margin_threshold=self._conf_value("expansion_margin_threshold", 0.5),
            )
            self.expansion_target = expansion.soft_target.to(self.device)
            self.expansion_mask = expansion.candidate_mask.to(self.device)
            self.gb_diagnostics.update({
                "lambda_u": self.lambda_u,
                **expansion.diagnostics,
            })

    def _refresh_dynamic_soft_targets(self, epoch):
        if not self.use_dynamic_alpha or self.soft_target is None:
            return
        warmup_epochs = int(self._conf_value("dynamic_warmup_epochs", 20))
        refresh_interval = int(self._conf_value("dynamic_refresh_interval", 20))
        if epoch < warmup_epochs:
            return
        if refresh_interval <= 0 or (epoch - warmup_epochs) % refresh_interval != 0:
            return
        beta = float(self._conf_value("dynamic_agreement_beta", 0.5))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.feats, self.adj)
            pred_label = logits.argmax(dim=1)
            soft_target, alpha, pseudo_label = apply_ball_soft_targets(
                noisy_label=self.noisy_label,
                assignment=self.gb_assignment,
                train_mask=self.train_mask,
                n_classes=self.n_classes,
                min_train_per_ball=self._conf_value("min_train_per_ball", 2),
                alpha_baseline=self._conf_value("alpha_baseline", "binary"),
                target_mode=self._conf_value("target_mode", "majority"),
                prediction_label=pred_label,
                agreement_beta=beta,
            )
        self.soft_target = soft_target.to(self.device)
        self.soft_alpha = alpha.to(self.device)
        self.soft_pseudo_label = pseudo_label.to(self.device)
        self.corrected_label = pseudo_label.to(self.device)
        self.dynamic_refresh_count += 1
        dynamic_diag = diagnostics_for_soft_targets(
            noisy_label=self.noisy_label,
            clean_label=self.clean_label,
            soft_target=self.soft_target,
            alpha=self.soft_alpha,
            pseudo_label=self.soft_pseudo_label,
            train_mask=self.train_mask,
        )
        self.gb_diagnostics.update({
            "dynamic_alpha": 1,
            "dynamic_refresh_count": self.dynamic_refresh_count,
            "dynamic_last_epoch": int(epoch),
            "dynamic_agreement_beta": beta,
            **{f"dynamic_{key}": value for key, value in dynamic_diag.items()},
        })
        if self.conf.training['debug']:
            print("Dynamic alpha refresh:", {
                "epoch": int(epoch),
                "refresh_count": self.dynamic_refresh_count,
                "mean_alpha_train": dynamic_diag.get("mean_alpha_train"),
                "soft_shift_train_mass": dynamic_diag.get("soft_shift_train_mass"),
            })

    def _append_diagnostics(self):
        if not self.gb_diagnostics:
            return
        os.makedirs("./runs", exist_ok=True)
        path = "./runs/frozengrace_gb_gcn_diagnostics_v2.csv"
        row = {
            "method": getattr(self, "method_name", self.conf.model.get("method", "frozengrace_gb_gcn")),
            "dataset": self.dataset_name,
            "seed": getattr(self, "seed", ""),
            "noise_type": getattr(self, "noise_type", ""),
            "noise_rate": getattr(self, "noise_rate", ""),
            "train": self.result.get("train", ""),
            "valid": self.result.get("valid", ""),
            "test": self.result.get("test", ""),
            "total_time": self.total_time,
            "best_epoch": self.best_epoch,
            **self.gb_diagnostics,
        }
        fieldnames = [
            "method", "dataset", "seed", "noise_type", "noise_rate",
            "train", "valid", "test", "total_time", "best_epoch", "random_control",
            "use_xh_features", "skip_ball_targets", "xh_feature_dim", "xh_embedding_dim",
            "num_balls", "mean_ball_size", "max_ball_size", "singleton_ratio", "mean_radius",
            "min_train_per_ball", "min_majority_confidence",
            "eligible_balls", "confidence_pass_balls", "eligible_train_nodes",
            "confidence_pass_train_nodes", "confidence_pass_train_ratio",
            "changed_train_count", "changed_train_ratio", "ncr", "cdr",
            "noisy_train_count", "clean_train_count",
            "alpha_positive_balls", "alpha_positive_train_nodes", "alpha_positive_train_ratio_by_ball",
            "alpha_positive_train_count", "alpha_positive_train_ratio", "mean_alpha_train",
            "soft_shift_train_mass", "soft_pseudo_changed_train_count",
            "soft_pseudo_ncr", "soft_pseudo_cdr", "soft_clean_mass_delta",
            "soft_alpha_baseline", "soft_target_mode",
            "ball_mode", "purity_threshold", "min_purity_gain", "split_attempts",
            "min_train_per_child", "min_train_to_split",
            "adaptive_num_splits", "adaptive_num_rejected_splits",
            "mean_train_per_ball", "zero_train_balls",
            "noisy_train_purity_weighted", "true_train_purity_weighted",
            "true_all_purity_weighted", "purity_eligible_train_total",
            "dynamic_alpha", "dynamic_refresh_count", "dynamic_last_epoch", "dynamic_agreement_beta",
            "dynamic_alpha_positive_train_count", "dynamic_alpha_positive_train_ratio",
            "dynamic_mean_alpha_train", "dynamic_soft_shift_train_mass",
            "dynamic_soft_pseudo_changed_train_count", "dynamic_soft_pseudo_ncr",
            "dynamic_soft_pseudo_cdr", "dynamic_soft_clean_mass_delta",
            "lambda_bc", "bc_min_ball_size", "last_bc_loss",
            "lambda_u", "expansion_min_train_per_ball", "expansion_margin_threshold",
            "expansion_eligible_balls", "expansion_selected_balls",
            "expansion_candidate_count", "expansion_coverage_unlabeled",
            "expansion_coverage_total", "expansion_mean_selected_margin",
            "expansion_mean_selected_train_votes", "last_expansion_loss",
            "topology_alpha", "topology_alpha_floor",
            "topology_conf_mean_raw", "topology_conf_min_raw", "topology_conf_max_raw",
            "topology_scale_mean", "topology_scale_min", "topology_scale_max",
        ]
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        self._make_corrected_labels()
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self._refresh_dynamic_soft_targets(epoch)
            self.model.train()
            self.optim.zero_grad()
            train_label = self.soft_target if self.use_soft_targets else self.corrected_label
            output, loss_train, acc_train = self.get_prediction(
                self.feats, self.adj, train_label, self.train_mask
            )
            if self.lambda_bc > 0:
                bc_loss = ball_consistency_loss(
                    output,
                    self.gb_assignment,
                    min_ball_size=self._conf_value("bc_min_ball_size", 2),
                )
                self.last_bc_loss = float(bc_loss.detach().item())
                loss_train = loss_train + self.lambda_bc * bc_loss
            if self.lambda_u > 0:
                expansion_loss = expansion_soft_target_loss(
                    output,
                    self.expansion_target,
                    self.expansion_mask,
                    self.lambda_u,
                )
                self.last_expansion_loss = float(expansion_loss.detach().item())
                loss_train = loss_train + expansion_loss
            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.best_epoch = epoch + 1
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.lambda_bc > 0:
            self.gb_diagnostics["last_bc_loss"] = self.last_bc_loss
        if self.lambda_u > 0:
            self.gb_diagnostics["last_expansion_loss"] = self.last_expansion_loss
        self._append_diagnostics()
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            if isinstance(label, torch.Tensor) and label.ndim == 2:
                log_prob = F.log_softmax(output[mask], dim=1)
                loss = -(label[mask] * log_prob).sum(dim=1).mean()
                hard_label = label[mask].argmax(dim=1)
                acc = self.metric(hard_label.detach().cpu().numpy(), output[mask].detach().cpu().numpy())
            else:
                loss = self.loss_fn(output[mask], label[mask])
                acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc


class frozengrace_gb_gcn_Predictor(_FrozenGraceGBGCNBase):
    random_control = False


class frozengrace_gb_gcn_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True


class frozengrace_gb_gcn_gated_Predictor(_FrozenGraceGBGCNBase):
    random_control = False


class frozengrace_gb_gcn_gated_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True


class frozengrace_gb_gcn_soft_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True


class frozengrace_gb_gcn_soft_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True


class frozengrace_gb_gcn_softdist_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True


class frozengrace_gb_gcn_softdist_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True


class frozengrace_gb_gcn_softdist_xh_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_xh_features = True


class frozengrace_gb_gcn_softdist_xh_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_xh_features = True


class frozengrace_gb_gcn_softdist_xh_topoalpha_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_xh_features = True
    use_topology_alpha = True


class frozengrace_gb_gcn_softdist_xh_topoalpha_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_xh_features = True
    use_topology_alpha = True


class gcn_xh_ce_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = False
    use_xh_features = True
    skip_ball_targets = True


class frozengrace_gb_gcn_softdist_dynalpha_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_dynamic_alpha = True


class frozengrace_gb_gcn_softdist_dynalpha_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_dynamic_alpha = True


class frozengrace_gb_gcn_softdist_dynalpha_fast_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_dynamic_alpha = True


class frozengrace_gb_gcn_softdist_dynalpha_fast_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_dynamic_alpha = True


class frozengrace_gb_gcn_softdist_bc_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True


class frozengrace_gb_gcn_softdist_bc_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True


class frozengrace_gb_gcn_softdist_expand_l005_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_ball_expansion = True


class frozengrace_gb_gcn_softdist_expand_l01_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_ball_expansion = True


class frozengrace_gb_gcn_adaptive_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True


class frozengrace_gb_gcn_adaptive_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
