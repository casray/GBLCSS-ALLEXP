import csv
import os
import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import APPNPNet, GATNet, GCN, MLP
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    ball_consistency_loss,
    build_ball_expansion_soft_targets,
    build_frozen_grace_knn_expansion_soft_targets,
    build_frozen_grace_soft_targets,
    build_frozen_grace_kmeans_soft_targets,
    build_frozen_grace_knn_soft_targets,
    build_global_smoothing_target_result,
    build_label_propagation_soft_targets,
    build_frozen_grace_targets,
    build_kmeans_soft_targets_from_features,
    build_knn_expansion_soft_targets_from_features,
    build_knn_soft_targets_from_features,
    build_soft_targets_from_features,
    diagnostics_for_soft_targets,
    expansion_soft_target_loss,
    load_frozen_embeddings,
)
from utils.ordinary_gb import resolve_effective_max_balls


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


def l2_normalized_embeddings_like_features(features, embeddings):
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D tensor")
    if features.shape[0] != embeddings.shape[0]:
        raise ValueError(f"feature node count {features.shape[0]} != embedding node count {embeddings.shape[0]}")
    feature_dtype = features.dtype if not getattr(features, "is_sparse", False) else features.dtype
    embeddings = embeddings.to(device=features.device, dtype=feature_dtype).float()
    norm = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return embeddings / norm


class _FrozenGraceGBGCNBase(Predictor):
    random_control = False
    use_soft_targets = False
    use_dynamic_alpha = False
    use_xh_features = False
    use_h_only_features = False
    skip_ball_targets = False
    use_ball_expansion = False
    backbone = "gcn"
    ball_feature_source = "grace"
    softdist_weighting = "uniform"
    soft_target_builder = "ball"

    def method_init(self, conf, data):
        self._validate_feature_mode()
        in_channels = conf.model['n_feat']
        self.xh_embedding_dim = 0
        self.h_only_embedding_dim = 0
        if self.use_h_only_features:
            embedding_path = self._feature_embedding_path()
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
            self.h_only_embedding_dim = int(embeddings.shape[1])
            self.feats = l2_normalized_embeddings_like_features(self.feats, embeddings).to(self.device)
            in_channels = int(self.feats.shape[1])
            conf.model['n_feat'] = in_channels
        if self.use_xh_features:
            embedding_path = self._feature_embedding_path()
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
            self.xh_embedding_dim = int(embeddings.shape[1])
            self.feats = concat_l2_normalized_embeddings(self.feats, embeddings).to(self.device)
            in_channels = int(self.feats.shape[1])
            conf.model['n_feat'] = in_channels
        backbone = str(conf.model.get("backbone", self.backbone)).lower()
        if backbone == "gcn":
            self.model = GCN(in_channels=in_channels, hidden_channels=conf.model['n_hidden'],
                             out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                             dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer']).to(self.device)
        elif backbone == "gat":
            self.model = GATNet(in_channels=in_channels, hidden_channels=conf.model['n_hidden'],
                                out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                                dropout=conf.model['dropout'],
                                heads=conf.model.get("heads", 1)).to(self.device)
        elif backbone == "appnp":
            self.model = APPNPNet(in_channels=in_channels, hidden_channels=conf.model['n_hidden'],
                                  out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                                  dropout=conf.model['dropout'],
                                  k=conf.model.get("appnp_k", 10),
                                  alpha=conf.model.get("appnp_alpha", 0.1)).to(self.device)
        elif backbone == "mlp":
            self.model = MLP(in_channels=in_channels, hidden_channels=conf.model['n_hidden'],
                             out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                             dropout=conf.model['dropout']).to(self.device)
        else:
            raise ValueError(f"unsupported backbone: {backbone}")
        self.backbone = backbone
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

    def _validate_feature_mode(self):
        if self.use_xh_features and self.use_h_only_features:
            raise ValueError("use_xh_features and use_h_only_features are mutually exclusive")

    def _feature_embedding_path(self, required=True):
        embedding_path = self._conf_value("embedding_path", None)
        if embedding_path is None and required:
            raise ValueError("gb.embedding_path is required for XH feature concatenation")
        return embedding_path

    def _ball_embedding_path(self):
        feature_embedding_path = self._feature_embedding_path(required=False)
        return self._conf_value("ball_embedding_path", feature_embedding_path)

    def _ball_consistency_weight(self):
        return float(self._conf_value("lambda_bc", 0.0))

    def _resolve_ball_capacity(self):
        configured = int(self._conf_value("max_balls", 64))
        policy = str(self._conf_value("max_balls_policy", "fixed")).strip().lower()
        effective = resolve_effective_max_balls(
            n_nodes=int(self.noisy_label.shape[0]),
            configured_max_balls=configured,
            policy=policy,
        )
        return {
            "max_balls_policy": policy,
            "configured_max_balls": configured,
            "effective_max_balls": effective,
        }

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
                "use_h_only_features": int(self.use_h_only_features),
                "xh_feature_dim": int(self.feats.shape[1]),
                "xh_embedding_dim": int(self.xh_embedding_dim),
                "h_only_embedding_dim": int(self.h_only_embedding_dim),
            }
            if self.conf.training['debug']:
                print("Frozen-GRACE GB diagnostics:", self.gb_diagnostics)
            self.lambda_bc = 0.0
            return
        ball_feature_source = str(
            self._conf_value("ball_feature_source", self.ball_feature_source)
        ).lower()
        seed = int(getattr(self, "seed", torch.initial_seed() % (2 ** 32 - 1)))
        capacity = self._resolve_ball_capacity()
        common_kwargs = {
            "noisy_label": self.noisy_label,
            "clean_label": self.clean_label,
            "train_mask": self.train_mask,
            "n_classes": self.n_classes,
            "max_balls": capacity["effective_max_balls"],
            "min_ball_size": self._conf_value("min_ball_size", 32),
            "split_improvement": self._conf_value("split_improvement", 0.05),
            "min_train_per_ball": self._conf_value("min_train_per_ball", 2),
            "random_state": seed,
            "random_control": self.random_control,
        }
        ball_build_started = time.perf_counter()
        if self.use_soft_targets:
            soft_target_builder = str(self._conf_value("soft_target_builder", self.soft_target_builder)).lower()
            soft_kwargs = {
                **common_kwargs,
                "alpha_baseline": self._conf_value("alpha_baseline", "binary"),
                "target_mode": self._conf_value("target_mode", "majority"),
                "target_beta": self._conf_value("target_beta", 0.0),
                "leave_one_out": self._conf_value("leave_one_out", False),
                "softdist_weighting": self._conf_value("softdist_weighting", self.softdist_weighting),
                "ellipsoid_ridge_scale": self._conf_value("ellipsoid_ridge_scale", 0.05),
                "ellipsoid_weight_floor": self._conf_value("ellipsoid_weight_floor", 0.1),
                "ellipsoid_feature_space": self._conf_value("ellipsoid_feature_space", "fullH"),
                "ellipsoid_covariance_estimator": self._conf_value("ellipsoid_covariance_estimator", "ridge"),
                "ellipsoid_distance_temperature": self._conf_value("ellipsoid_distance_temperature", 1.0),
                "gmm_n_components": self._conf_value("gmm_n_components", None),
                "gmm_covariance_type": self._conf_value("gmm_covariance_type", "diag"),
                "gmm_reg_covar": self._conf_value("gmm_reg_covar", 1e-5),
                "gmm_max_iter": self._conf_value("gmm_max_iter", 100),
                "gmm_n_init": self._conf_value("gmm_n_init", 1),
                "gmm_prior": self._conf_value("gmm_prior", "jeffreys"),
                "gmm_geometry_confidence": self._conf_value("gmm_geometry_confidence", "max_resp"),
                "ball_mode": self._conf_value("ball_mode", "geometric"),
                "purity_threshold": self._conf_value("purity_threshold", 0.6),
                "min_train_per_child": self._conf_value("min_train_per_child", 4),
                "min_train_to_split": self._conf_value("min_train_to_split", 16),
                "adaptive_min_ball_size": self._conf_value("adaptive_min_ball_size", 8),
                "adaptive_max_balls": self._conf_value("adaptive_max_balls", 128),
                "min_purity_gain": self._conf_value("min_purity_gain", 0.05),
                "split_attempts": self._conf_value("split_attempts", 1),
            }
            if soft_target_builder in {"uniform_smoothing", "prior_smoothing"}:
                result = build_global_smoothing_target_result(
                    noisy_label=self.noisy_label,
                    clean_label=self.clean_label,
                    train_mask=self.train_mask,
                    n_classes=self.n_classes,
                    epsilon=self._conf_value("smoothing_epsilon", 0.1),
                    distribution=soft_target_builder.removesuffix("_smoothing"),
                )
            elif soft_target_builder == "lp":
                result = build_label_propagation_soft_targets(
                    noisy_label=self.noisy_label,
                    clean_label=self.clean_label,
                    train_mask=self.train_mask,
                    adj=self.adj,
                    n_classes=self.n_classes,
                    num_steps=self._conf_value("lp_steps", 2),
                    alpha_baseline=self._conf_value("alpha_baseline", "multiclass"),
                    target_mode=self._conf_value("target_mode", "distribution"),
                    target_beta=self._conf_value("target_beta", 0.5),
                )
            elif soft_target_builder in {"knn", "kmeans"}:
                replacement_kwargs = {
                    "noisy_label": self.noisy_label,
                    "clean_label": self.clean_label,
                    "train_mask": self.train_mask,
                    "n_classes": self.n_classes,
                    "max_balls": self._conf_value("max_balls", 64),
                    "min_ball_size": self._conf_value("min_ball_size", 32),
                    "split_improvement": self._conf_value("split_improvement", 0.05),
                    "alpha_baseline": self._conf_value("alpha_baseline", "multiclass"),
                    "target_mode": self._conf_value("target_mode", "distribution"),
                    "target_beta": self._conf_value("target_beta", 0.5),
                    "random_state": seed,
                }
                if soft_target_builder == "knn":
                    replacement_kwargs["matched_k"] = self._conf_value("matched_k", None)
                    if ball_feature_source == "grace":
                        embedding_path = self._ball_embedding_path()
                        if embedding_path is None:
                            raise ValueError("gb.embedding_path or gb.ball_embedding_path is required for kNN targets")
                        result = build_frozen_grace_knn_soft_targets(
                            embedding_path=embedding_path,
                            **replacement_kwargs,
                        )
                    elif ball_feature_source == "x":
                        result = build_knn_soft_targets_from_features(
                            ball_features=self.feats,
                            **replacement_kwargs,
                        )
                    else:
                        raise ValueError("ball_feature_source must be 'grace' or 'x'")
                else:
                    replacement_kwargs.update({
                        "min_train_per_ball": self._conf_value("min_train_per_ball", 2),
                        "matched_kmeans_clusters": self._conf_value("matched_kmeans_clusters", None),
                    })
                    if ball_feature_source == "grace":
                        embedding_path = self._ball_embedding_path()
                        if embedding_path is None:
                            raise ValueError("gb.embedding_path or gb.ball_embedding_path is required for kmeans targets")
                        result = build_frozen_grace_kmeans_soft_targets(
                            embedding_path=embedding_path,
                            **replacement_kwargs,
                        )
                    elif ball_feature_source == "x":
                        result = build_kmeans_soft_targets_from_features(
                            ball_features=self.feats,
                            **replacement_kwargs,
                        )
                    else:
                        raise ValueError("ball_feature_source must be 'grace' or 'x'")
            elif ball_feature_source == "grace":
                embedding_path = self._ball_embedding_path()
                if embedding_path is None:
                    raise ValueError("gb.embedding_path or gb.ball_embedding_path is required for GRACE ball targets")
                result = build_frozen_grace_soft_targets(
                    embedding_path=embedding_path,
                    **soft_kwargs,
                )
            elif ball_feature_source == "x":
                result = build_soft_targets_from_features(
                    ball_features=self.feats,
                    **soft_kwargs,
                )
            else:
                raise ValueError("ball_feature_source must be 'grace' or 'x'")
        else:
            if ball_feature_source != "grace":
                raise ValueError(
                    "hard corrected-label targets only support ball_feature_source=grace"
                )
            embedding_path = self._ball_embedding_path()
            if embedding_path is None:
                raise ValueError("gb.embedding_path or gb.ball_embedding_path is required for GRACE ball targets")
            result = build_frozen_grace_targets(
                embedding_path=embedding_path,
                **common_kwargs,
                min_majority_confidence=self._conf_value("min_majority_confidence", 0.0),
                ball_mode=self._conf_value("ball_mode", "geometric"),
            )
        self.corrected_label = result.corrected_label.to(self.device)
        self.soft_target = result.soft_target.to(self.device) if result.soft_target is not None else None
        self.soft_alpha = result.alpha.to(self.device) if result.alpha is not None else None
        self.soft_pseudo_label = result.pseudo_label.to(self.device) if result.pseudo_label is not None else None
        self.gb_assignment = result.assignment
        self.gb_diagnostics = result.diagnostics
        self.gb_diagnostics.update(capacity)
        self.gb_diagnostics["max_balls_cap_hit"] = int(
            int(self.gb_diagnostics.get("num_balls", 0)) >= int(capacity["effective_max_balls"])
        )
        self.gb_diagnostics["ball_build_time_seconds"] = float(
            time.perf_counter() - ball_build_started
        )
        self.gb_diagnostics["ball_feature_source"] = ball_feature_source
        if self.use_soft_targets:
            self.gb_diagnostics["soft_target_builder"] = str(
                self._conf_value("soft_target_builder", self.soft_target_builder)
            ).lower()
        feature_embedding_path = self._feature_embedding_path(required=False)
        ball_embedding_path = self._ball_embedding_path()
        if feature_embedding_path is not None:
            self.gb_diagnostics["feature_embedding_path"] = str(feature_embedding_path)
        if ball_embedding_path is not None:
            self.gb_diagnostics["ball_embedding_path"] = str(ball_embedding_path)
        self.gb_diagnostics["use_xh_features"] = int(self.use_xh_features)
        self.gb_diagnostics["use_h_only_features"] = int(self.use_h_only_features)
        self.gb_diagnostics["skip_ball_targets"] = int(self.skip_ball_targets)
        self.gb_diagnostics["xh_feature_dim"] = int(self.feats.shape[1])
        self.gb_diagnostics["xh_embedding_dim"] = int(self.xh_embedding_dim)
        self.gb_diagnostics["h_only_embedding_dim"] = int(self.h_only_embedding_dim)
        if self.conf.training['debug']:
            print("Frozen-GRACE GB diagnostics:", self.gb_diagnostics)
        self.lambda_bc = self._ball_consistency_weight()
        if self.lambda_bc > 0:
            self.gb_diagnostics["lambda_bc"] = self.lambda_bc
            self.gb_diagnostics["bc_min_ball_size"] = int(self._conf_value("bc_min_ball_size", 2))
        self.lambda_u = float(self._conf_value("lambda_u", 0.0)) if self.use_ball_expansion else 0.0
        if self.use_ball_expansion:
            expansion_builder = str(self._conf_value("expansion_builder", "ball")).lower()
            if expansion_builder == "ball":
                expansion = build_ball_expansion_soft_targets(
                    noisy_label=self.noisy_label,
                    assignment=self.gb_assignment,
                    train_mask=self.train_mask,
                    n_classes=self.n_classes,
                    min_train_per_ball=self._conf_value("expansion_min_train_per_ball", 10),
                    margin_threshold=self._conf_value("expansion_margin_threshold", 0.5),
                    target_beta=self._conf_value("expansion_target_beta", 0.0),
                )
                expansion.diagnostics["expansion_builder"] = "ball"
            elif expansion_builder == "knn":
                expansion_feature_source = str(self._conf_value("expansion_feature_source", ball_feature_source)).lower()
                expansion_kwargs = {
                    "noisy_label": self.noisy_label,
                    "train_mask": self.train_mask,
                    "n_classes": self.n_classes,
                    "max_balls": self._conf_value("max_balls", 64),
                    "min_ball_size": self._conf_value("min_ball_size", 32),
                    "split_improvement": self._conf_value("split_improvement", 0.05),
                    "min_train_per_ball": self._conf_value("expansion_min_train_per_ball", 10),
                    "margin_threshold": self._conf_value("expansion_margin_threshold", 0.5),
                    "target_beta": self._conf_value("expansion_target_beta", 0.0),
                    "random_state": seed,
                    "matched_k": self._conf_value("expansion_matched_k", None),
                }
                if expansion_feature_source == "grace":
                    embedding_path = self._ball_embedding_path()
                    if embedding_path is None:
                        raise ValueError("gb.embedding_path or gb.ball_embedding_path is required for kNN expansion")
                    expansion = build_frozen_grace_knn_expansion_soft_targets(
                        embedding_path=embedding_path,
                        **expansion_kwargs,
                    )
                elif expansion_feature_source == "x":
                    expansion = build_knn_expansion_soft_targets_from_features(
                        ball_features=self.feats,
                        **expansion_kwargs,
                    )
                else:
                    raise ValueError("expansion_feature_source must be 'grace' or 'x'")
            else:
                raise ValueError("expansion_builder must be 'ball' or 'knn'")
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
                target_beta=self._conf_value("target_beta", 0.0),
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
            "use_xh_features", "use_h_only_features", "skip_ball_targets",
            "xh_feature_dim", "xh_embedding_dim", "h_only_embedding_dim",
            "num_balls", "mean_ball_size", "max_ball_size", "singleton_ratio", "mean_radius",
            "max_balls_policy", "configured_max_balls", "effective_max_balls",
            "max_balls_cap_hit", "ball_build_time_seconds",
            "min_train_per_ball", "min_majority_confidence",
            "eligible_balls", "confidence_pass_balls", "eligible_train_nodes",
            "confidence_pass_train_nodes", "confidence_pass_train_ratio",
            "changed_train_count", "changed_train_ratio", "ncr", "cdr",
            "noisy_train_count", "clean_train_count",
            "alpha_positive_balls", "alpha_positive_train_nodes", "alpha_positive_train_ratio_by_ball",
            "alpha_positive_train_count", "alpha_positive_train_ratio", "mean_alpha_train",
            "soft_shift_train_mass", "soft_pseudo_changed_train_count",
            "soft_pseudo_ncr", "soft_pseudo_cdr", "soft_clean_mass_delta",
            "soft_clean_mass_delta_mean", "soft_target_brier", "noisy_onehot_brier",
            "soft_brier_gain", "soft_target_argmax_accuracy",
            "soft_alpha_baseline", "soft_target_mode", "target_beta", "softdist_weighting",
            "leave_one_out", "smoothing_distribution", "smoothing_epsilon",
            "soft_target_builder", "target_builder", "matched_k", "matched_num_balls",
            "matched_reference_num_balls", "matched_mean_train_per_ball",
            "matched_zero_train_balls", "kmeans_n_clusters",
            "ellipsoid_ridge_scale", "ellipsoid_weight_floor",
            "ellipsoid_feature_space", "ellipsoid_covariance_estimator",
            "ellipsoid_distance_temperature", "ellipsoid_requested_feature_space",
            "ellipsoid_eligible_balls", "ellipsoid_fallback_balls",
            "ellipsoid_covariance_failure_balls", "ellipsoid_covariance_failure_ratio",
            "ellipsoid_fallback_ball_ratio", "ellipsoid_weighted_train_nodes",
            "ellipsoid_mean_train_weight", "ellipsoid_median_train_weight",
            "ellipsoid_min_train_weight", "ellipsoid_effective_support_mean",
            "ellipsoid_effective_support_ratio_mean", "ellipsoid_pca_effective_dim_mean",
            "ellipsoid_pca_effective_dim_max",
            "gmm_fallback", "gmm_fit_error", "gmm_n_components",
            "gmm_covariance_type", "gmm_reg_covar", "gmm_max_iter", "gmm_n_init",
            "gmm_prior", "gmm_geometry_confidence", "gmm_converged", "gmm_n_iter",
            "gmm_component_support_min", "gmm_component_support_median",
            "gmm_component_support_mean", "gmm_low_support_component_ratio",
            "gmm_max_resp_mean", "gmm_resp_entropy_mean", "gmm_noise_auc",
            "gmm_support_bin_lt2_train_count", "gmm_support_bin_lt2_changed_train_count",
            "gmm_support_bin_lt2_ncr", "gmm_support_bin_lt2_cdr",
            "gmm_support_bin_lt2_mean_alpha", "gmm_support_bin_lt2_clean_mass_delta",
            "gmm_support_bin_2-3_train_count", "gmm_support_bin_2-3_changed_train_count",
            "gmm_support_bin_2-3_ncr", "gmm_support_bin_2-3_cdr",
            "gmm_support_bin_2-3_mean_alpha", "gmm_support_bin_2-3_clean_mass_delta",
            "gmm_support_bin_4-7_train_count", "gmm_support_bin_4-7_changed_train_count",
            "gmm_support_bin_4-7_ncr", "gmm_support_bin_4-7_cdr",
            "gmm_support_bin_4-7_mean_alpha", "gmm_support_bin_4-7_clean_mass_delta",
            "gmm_support_bin_8+_train_count", "gmm_support_bin_8+_changed_train_count",
            "gmm_support_bin_8+_ncr", "gmm_support_bin_8+_cdr",
            "gmm_support_bin_8+_mean_alpha", "gmm_support_bin_8+_clean_mass_delta",
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
            "expansion_target_beta", "expansion_builder", "expansion_matched_k",
            "expansion_eligible_balls", "expansion_selected_balls",
            "expansion_eligible_queries", "expansion_selected_queries",
            "expansion_candidate_count", "expansion_coverage_unlabeled",
            "expansion_coverage_total", "expansion_mean_selected_margin",
            "expansion_mean_selected_train_votes", "last_expansion_loss",
            "teacher_epochs", "teacher_best_epoch", "teacher_best_val",
            "teacher_best_test", "teacher_use_xh", "teacher_input_dim",
            "student_input_dim", "teacher_target", "distill_temperature",
            "distill_student_best_epoch",
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


class frozengrace_geaf_gcn_xh_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_xh_features = True


class frozengrace_geaf_gcn_xh_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_xh_features = True


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


class frozengrace_gb_gcn_softdist_xh_ellipsoid_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_xh_features = True
    softdist_weighting = "ellipsoid"


class frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_xh_features = True
    softdist_weighting = "ellipsoid"


class frozengrace_gb_gcn_softdist_xh_gmmresp_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = True
    use_xh_features = True
    softdist_weighting = "gmm_resp"


class frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled_Predictor(_FrozenGraceGBGCNBase):
    random_control = True
    use_soft_targets = True
    use_xh_features = True
    softdist_weighting = "gmm_resp"


class gcn_xh_ce_Predictor(_FrozenGraceGBGCNBase):
    random_control = False
    use_soft_targets = False
    use_xh_features = True
    skip_ball_targets = True


class mlp_ce_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    skip_ball_targets = True


class linear_ce_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    skip_ball_targets = True


class lp_mlp_softdist_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    soft_target_builder = "lp"


class mlp_xh_ce_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_xh_features = True
    skip_ball_targets = True


class grace_h_linear_ce_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_h_only_features = True
    skip_ball_targets = True


class grace_h_mlp_ce_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_h_only_features = True
    skip_ball_targets = True


class frozengrace_gb_mlp_softdist_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta05_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta05_loo_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta05_loo_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True


class mlp_uniform_smoothing_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    soft_target_builder = "uniform_smoothing"


class mlp_prior_smoothing_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    soft_target_builder = "prior_smoothing"


class frozengrace_gb_linear_softdist_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True


class frozengrace_gb_linear_softdist_beta05_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta1_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True


class frozengrace_gb_mlp_softdist_beta1_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True


class frozengrace_gb_mlp_hard_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"


class frozengrace_gb_mlp_softdist_gmmresp_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    softdist_weighting = "gmm_resp"


class frozengrace_gb_mlp_softdist_gmmresp_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    softdist_weighting = "gmm_resp"


class frozengrace_gb_mlp_softdist_gmmresp_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True
    softdist_weighting = "gmm_resp"


class frozengrace_gb_mlp_softdist_ellipsoid_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    softdist_weighting = "ellipsoid"


class frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True
    softdist_weighting = "ellipsoid"


class frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    softdist_weighting = "ellipsoid_pca_shrinkage"


class frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True
    softdist_weighting = "ellipsoid_pca_shrinkage"


class knn_mlp_softdist_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    soft_target_builder = "knn"


class kmeans_mlp_softdist_beta05_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    soft_target_builder = "kmeans"


class frozengrace_gb_mlp_softdist_beta05_expand_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    use_ball_expansion = True


class frozengrace_gb_mlp_softdist_beta05_expand_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True
    use_ball_expansion = True


class knn_mlp_softdist_beta05_expand_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    use_ball_expansion = True
    soft_target_builder = "knn"


class mlp_distill_gcn_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    skip_ball_targets = True
    teacher_use_xh = False
    teacher_target = "ce"

    def _teacher_use_xh_features(self):
        return bool(self._conf_value("teacher_use_xh", self.teacher_use_xh))

    def _teacher_target_mode(self):
        return str(self._conf_value("teacher_target", self.teacher_target)).lower()

    def _teacher_input(self):
        if hasattr(self, "_teacher_feats"):
            return self._teacher_feats
        if self._teacher_use_xh_features():
            embedding_path = self._feature_embedding_path()
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
            self._teacher_feats = concat_l2_normalized_embeddings(self.feats, embeddings).to(self.device)
        else:
            self._teacher_feats = self.feats
        return self._teacher_feats

    def _teacher_model(self):
        teacher_feats = self._teacher_input()
        return GCN(
            in_channels=int(teacher_feats.shape[1]),
            hidden_channels=int(self._conf_value("teacher_hidden", self.conf.model["n_hidden"])),
            out_channels=self.n_classes,
            n_layers=int(self._conf_value("teacher_n_layer", self.conf.model["n_layer"])),
            dropout=float(self._conf_value("teacher_dropout", self.conf.model["dropout"])),
            norm_info=self.conf.model["norm_info"],
            act=self.conf.model["act"],
            input_layer=self.conf.model["input_layer"],
            output_layer=self.conf.model["output_layer"],
        ).to(self.device)

    def _teacher_loss(self, logits):
        mode = self._teacher_target_mode()
        if mode in {"soft", "soft_target", "softdist"}:
            if self.soft_target is None:
                raise ValueError("teacher_target=soft_target requires a soft target builder")
            log_prob = F.log_softmax(logits[self.train_mask], dim=1)
            return -(self.soft_target[self.train_mask] * log_prob).sum(dim=1).mean()
        if mode == "ce":
            return self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
        raise ValueError(f"unsupported teacher_target: {mode}")

    def _train_teacher_probs(self):
        teacher = self._teacher_model()
        teacher_feats = self._teacher_input()
        teacher_optim = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", self.conf.training["lr"])),
            weight_decay=float(self._conf_value("teacher_weight_decay", self.conf.training["weight_decay"])),
        )
        teacher_epochs = int(self._conf_value("teacher_epochs", self.conf.training["n_epochs"]))
        best_val = -1.0
        best_state = None
        best_test = -1.0
        best_epoch = -1
        for epoch in range(teacher_epochs):
            self.teacher_epoch = int(epoch)
            teacher.train()
            teacher_optim.zero_grad()
            logits = teacher(teacher_feats, self.adj)
            loss = self._teacher_loss(logits)
            loss.backward()
            teacher_optim.step()

            teacher.eval()
            with torch.no_grad():
                logits = teacher(teacher_feats, self.adj)
                val_acc = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
                test_acc = self.metric(
                    self.clean_label[self.test_mask].cpu().numpy(),
                    logits[self.test_mask].detach().cpu().numpy(),
                )
            if val_acc >= best_val:
                best_val = float(val_acc)
                best_test = float(test_acc)
                best_epoch = epoch + 1
                best_state = deepcopy(teacher.state_dict())
        if best_state is not None:
            teacher.load_state_dict(best_state)
        teacher.eval()
        temperature = float(self._conf_value("distill_temperature", 1.0))
        with torch.no_grad():
            teacher_logits = teacher(teacher_feats, self.adj)
            teacher_probs = F.softmax(teacher_logits / max(1e-6, temperature), dim=1).detach()
        self.gb_diagnostics.update({
            "teacher_epochs": int(teacher_epochs),
            "teacher_best_epoch": int(best_epoch),
            "teacher_best_val": float(best_val),
            "teacher_best_test": float(best_test),
            "teacher_use_xh": int(self._teacher_use_xh_features()),
            "teacher_input_dim": int(teacher_feats.shape[1]),
            "student_input_dim": int(self.feats.shape[1]),
            "teacher_target": self._teacher_target_mode(),
            "distill_temperature": float(temperature),
        })
        return teacher_probs, temperature

    def train(self):
        self._make_corrected_labels()
        teacher_probs, temperature = self._train_teacher_probs()
        all_mask = torch.ones(self.n_nodes, dtype=torch.bool, device=self.device)
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output = self.model(self.feats, self.adj)
            log_prob = F.log_softmax(output[all_mask] / max(1e-6, temperature), dim=1)
            loss_train = -(teacher_probs[all_mask] * log_prob).sum(dim=1).mean() * (temperature ** 2)
            acc_train = self.metric(
                self.noisy_label[self.train_mask].cpu().numpy(),
                output[self.train_mask].detach().cpu().numpy(),
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
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self.gb_diagnostics["distill_student_best_epoch"] = int(self.best_epoch)
        self._append_diagnostics()
        if self.conf.training["debug"]:
            print("Optimization Finished!")
            print("Time(s): {:.4f}".format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result


class mlp_distill_gcn_xh_Predictor(mlp_distill_gcn_Predictor):
    teacher_use_xh = True


class mlp_distill_softdist_xh_Predictor(mlp_distill_gcn_Predictor):
    random_control = False
    use_soft_targets = True
    skip_ball_targets = False
    teacher_use_xh = True
    teacher_target = "soft_target"


class mlp_distill_knn_xh_Predictor(mlp_distill_gcn_Predictor):
    use_soft_targets = True
    skip_ball_targets = False
    soft_target_builder = "knn"
    teacher_use_xh = True
    teacher_target = "soft_target"


class mlp_distill_softdist_xh_shuffled_Predictor(mlp_distill_gcn_Predictor):
    random_control = True
    use_soft_targets = True
    skip_ball_targets = False
    teacher_use_xh = True
    teacher_target = "soft_target"


class featuregb_mlp_softdist_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    use_soft_targets = True
    ball_feature_source = "x"


class featuregb_mlp_softdist_shuffled_Predictor(_FrozenGraceGBGCNBase):
    backbone = "mlp"
    random_control = True
    use_soft_targets = True
    ball_feature_source = "x"


class frozengrace_gb_gcn_softdist_xh_gat_Predictor(frozengrace_gb_gcn_softdist_xh_Predictor):
    backbone = "gat"


class gcn_xh_ce_gat_Predictor(gcn_xh_ce_Predictor):
    backbone = "gat"


class frozengrace_gb_gcn_softdist_xh_appnp_Predictor(frozengrace_gb_gcn_softdist_xh_Predictor):
    backbone = "appnp"


class gcn_xh_ce_appnp_Predictor(gcn_xh_ce_Predictor):
    backbone = "appnp"


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
