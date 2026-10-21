from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from noisy_label_gb_grace.balls import build_dcgb_balls, build_grace_first_graph_balls, build_granular_balls, build_graph_aware_balls, shuffle_grouping
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets, train_noisy_linear_teacher
from noisy_label_gb_grace.smoothing.targets import CONFLICT_GATE_REPAIR_CONFIGS, REPAIR_CONTROL_CONFIGS
from noisy_label_gb_grace.utils import one_hot
from noisy_label_gb_grace.weighting import build_reliability_weights
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, save_csv, save_json
from scripts.extract_embeddings import embedding_path


WEIGHTED_METHODS = {"gb_weighted_ce", "gagb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"}
GRAPH_AWARE_METHODS = {"gagb", "gagb_ca_ce", "gagb_weighted_ce"}
V2_TARGET_METHODS = {"gagb_qb_pred_mix", "gagb_qb_teacher_prior", "gagb_qb_conflict_gate"} | set(CONFLICT_GATE_REPAIR_CONFIGS) | set(REPAIR_CONTROL_CONFIGS)
GRACE_GAGB_METHODS = {"grace_gagb"}
GRACE_GAGB_EMB_METHODS = {"grace_gagb_emb_only"}
GRACE_GAGB_TOPO_METHODS = {"grace_gagb_topo_only"}
GRACE_GAGB_MATCHED_METHODS = {"kmeans_grace_gagb_matched", "topk_grace_gagb_train_matched"}
GRACE_GAGB_QUALITY_METHODS = {
    "grace_gagb_qfilter_alpha07_q50",
    "grace_gagb_qfilter_alpha07_q70",
    "grace_gagb_qweight_alpha07",
    "grace_gagb_qsafe_boundary_alpha09_q50",
    "grace_gagb_qsafe_boundary_alpha09_q70",
    "grace_gagb_qsafe_boundary_alpha07_q50",
    "grace_gagb_qsafe_class_capped_alpha09_q50",
    "grace_gagb_qsafe_class_balanced_alpha09_q50",
}
GAGB_QUALITY_METHODS = {"gagb_qfilter_alpha07_q50", "gagb_qsafe_boundary_alpha09_q50"}
KHOP_QUALITY_METHODS = {"khop_only_qfilter_alpha07_q50", "khop_only_qsafe_boundary_alpha09_q50"}
GRACE_GAGB_SHUFFLED_METHODS = {
    "grace_gagb_shuffled_alpha_matched",
    "grace_gagb_shuffled_qfilter_alpha07_q50",
    "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
}
DCGB_METHODS = {"dcgb"}
DCGB_EMB_METHODS = {"dcgb_emb_only"}
DCGB_TOPO_METHODS = {"dcgb_topo_only"}
DCGB_MATCHED_METHODS = {"kmeans_dcgb_matched", "topk_dcgb_train_matched"}
DCGB_QUALITY_METHODS = {"dcgb_qsafe_boundary_alpha09_q50"}
DCGB_SHUFFLED_METHODS = {"dcgb_shuffled_alpha_matched"}
EMB_LOCAL_METHODS = {"emb_only_gagb_local"}
EMB_GLOBAL_METHODS = {"emb_only_gagb_global"}
GAGB_MATCHED_METHODS = {"kmeans_gagb_matched", "topk_gagb_matched", "topk_gagb_train_matched", "topk_gagb_allnode_matched"}
KHOP_METHODS = {"khop_only"}
SHUFFLED_SIZE_METHODS = {"gagb_shuffled", "gagb_shuffled_size_only"}
SHUFFLED_ALPHA_METHODS = {"gagb_shuffled_alpha_matched", "gagb_shuffled_alpha09_matched", "gagb_shuffled_replace_matched"}
DEFAULT_METHODS = [
    "ce",
    "gb_weighted_ce",
    "gagb_weighted_ce",
    "kmeans_weighted_ce",
    "topk_matched_weighted_ce",
    "global",
    "kmeans",
    "topk_matched",
    "gb",
    "gb_prior",
    "gagb",
    "gagb_ca_ce",
    "gagb_qb_pred_mix",
    "gagb_qb_teacher_prior",
    "gagb_qb_conflict_gate",
    "gagb_qb_conflict_gate_alpha03",
    "gagb_qb_conflict_gate_alpha06",
    "gagb_qb_conflict_gate_alpha07",
    "gagb_qb_conflict_gate_alpha08",
    "gagb_qb_conflict_gate_alpha09",
    "gagb_qb_conflict_gate_replace_qmix",
    "gagb_class_capped_alpha09",
    "gagb_class_balanced_alpha09",
    "gagb_boundary_safe_alpha09",
    "gagb_replay_random_same_count",
    "gagb_qb_conflict_gate_sharpen_qmix_T05",
    "gagb_qb_conflict_gate_margin_replace",
    "grace_gagb",
    "grace_gagb_emb_only",
    "grace_gagb_topo_only",
    "kmeans_grace_gagb_matched",
    "topk_grace_gagb_train_matched",
    "grace_gagb_shuffled_alpha_matched",
    "grace_gagb_qb_conflict_gate_alpha03",
    "grace_gagb_qb_conflict_gate_alpha07",
    "grace_gagb_qb_conflict_gate_alpha09",
    "grace_gagb_qfilter_alpha07_q50",
    "grace_gagb_qfilter_alpha07_q70",
    "grace_gagb_qweight_alpha07",
    "grace_gagb_qsafe_boundary_alpha09_q50",
    "grace_gagb_qsafe_boundary_alpha09_q70",
    "grace_gagb_qsafe_boundary_alpha07_q50",
    "grace_gagb_qsafe_class_capped_alpha09_q50",
    "grace_gagb_qsafe_class_balanced_alpha09_q50",
    "gagb_qsafe_boundary_alpha09_q50",
    "khop_only_qsafe_boundary_alpha09_q50",
    "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
    "gagb_qfilter_alpha07_q50",
    "khop_only_qfilter_alpha07_q50",
    "grace_gagb_shuffled_qfilter_alpha07_q50",
    "dcgb",
    "dcgb_emb_only",
    "dcgb_topo_only",
    "kmeans_dcgb_matched",
    "topk_dcgb_train_matched",
    "dcgb_shuffled_alpha_matched",
    "dcgb_qsafe_boundary_alpha09_q50",
    "emb_only_gagb_local",
    "emb_only_gagb_global",
    "khop_only",
    "kmeans_gagb_matched",
    "topk_gagb_train_matched",
    "topk_gagb_allnode_matched",
    "gagb_shuffled_size_only",
    "gagb_shuffled_alpha_matched",
    "gagb_shuffled_alpha09_matched",
    "gagb_shuffled_replace_matched",
    "kmeans_gagb_matched_alpha09",
    "kmeans_gagb_matched_replace",
    "topk_gagb_train_matched_alpha09",
    "topk_gagb_train_matched_replace",
    "khop_only_alpha09",
    "khop_only_replace",
]


def _normalised_entropy(probs: torch.Tensor) -> float:
    if probs.numel() <= 1 or float(probs.sum().item()) <= 0.0:
        return 0.0
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    return entropy / math.log(float(probs.numel()))


def _group_label_diagnostics(
    grouping: dict,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
) -> dict:
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    total_train_in_groups = 0
    clean_majority_sum = 0
    entropies = []
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        local = idx[train_mask[idx]]
        if local.numel() == 0:
            continue
        total_train_in_groups += int(local.numel())
        clean_counts = torch.bincount(clean[local], minlength=num_classes).float()
        noisy_counts = torch.bincount(noisy[local], minlength=num_classes).float()
        clean_majority_sum += int(clean_counts.max().item())
        probs = noisy_counts / noisy_counts.sum().clamp_min(1.0)
        entropies.append(_normalised_entropy(probs))
    return {
        "clean_purity": float(clean_majority_sum / max(1, total_train_in_groups)),
        "noisy_label_entropy": float(sum(entropies) / len(entropies)) if entropies else math.nan,
        "effective_coverage": float(total_train_in_groups / max(1, int(train_mask.sum().item()))),
    }


def _target_noise_diagnostics(
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor | None,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict:
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    if train_idx.numel() == 0:
        return {
            "noise_detection_auroc": math.nan,
            "noise_correction_rate": math.nan,
            "clean_damage_rate": math.nan,
        }

    is_noisy = clean[train_idx] != noisy[train_idx]
    targets = soft_targets.detach().cpu().float()
    target_pred = targets.argmax(dim=1)
    if sample_weights is not None:
        suspicion = 1.0 - sample_weights.detach().cpu().float()[train_idx]
    else:
        suspicion = 1.0 - targets[train_idx, noisy[train_idx]]
    if bool(is_noisy.any().item()) and bool((~is_noisy).any().item()):
        auroc = float(roc_auc_score(is_noisy.numpy().astype(int), suspicion.numpy()))
    else:
        auroc = math.nan
    noisy_idx = train_idx[is_noisy]
    clean_idx = train_idx[~is_noisy]
    correction = (
        float((target_pred[noisy_idx] == clean[noisy_idx]).float().mean().item())
        if noisy_idx.numel()
        else math.nan
    )
    damage = (
        float((target_pred[clean_idx] != clean[clean_idx]).float().mean().item())
        if clean_idx.numel()
        else math.nan
    )
    return {
        "noise_detection_auroc": auroc,
        "noise_correction_rate": correction,
        "clean_damage_rate": damage,
    }


def _target_repair_diagnostics(
    soft_targets: torch.Tensor,
    smoothing_stats: dict,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    teacher_probs: torch.Tensor | None,
) -> dict:
    gate_mask_raw = smoothing_stats.get("target_update_gate_mask")
    candidate_raw = smoothing_stats.get("target_update_candidate_argmax")
    train_mask = train_mask.detach().cpu().bool()
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    empty = {
        "gated_train_ratio": 0.0,
        "gated_noisy_ratio": math.nan,
        "possible_correction_before_alpha": math.nan,
        "lost_by_alpha_rate": math.nan,
        "target_argmax_change_rate": math.nan,
        "teacher_gated_noisy_clean_acc": math.nan,
        "qmix_argmax_eq_clean_rate": math.nan,
        "target_argmax_eq_clean_rate": math.nan,
    }
    if gate_mask_raw is None or candidate_raw is None or train_idx.numel() == 0:
        return empty
    gate_mask = torch.tensor(gate_mask_raw, dtype=torch.bool)
    candidate = torch.tensor(candidate_raw, dtype=torch.long)
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    target_pred = soft_targets.detach().cpu().float().argmax(dim=1)
    gated_train = gate_mask & train_mask
    gated_idx = gated_train.nonzero(as_tuple=False).view(-1)
    changed_train = target_pred[train_idx] != noisy[train_idx]
    out = {
        **empty,
        "gated_train_ratio": float(gated_idx.numel() / max(1, train_idx.numel())),
        "target_argmax_change_rate": float(changed_train.float().mean().item()),
    }
    if gated_idx.numel() == 0:
        return out
    gated_noisy = gated_train & (clean != noisy)
    gated_noisy_idx = gated_noisy.nonzero(as_tuple=False).view(-1)
    out["gated_noisy_ratio"] = float(gated_noisy_idx.numel() / max(1, gated_idx.numel()))
    if gated_noisy_idx.numel() == 0:
        return out
    candidate_clean = candidate[gated_noisy_idx] == clean[gated_noisy_idx]
    target_clean = target_pred[gated_noisy_idx] == clean[gated_noisy_idx]
    out["possible_correction_before_alpha"] = float(candidate_clean.float().mean().item())
    out["qmix_argmax_eq_clean_rate"] = out["possible_correction_before_alpha"]
    out["target_argmax_eq_clean_rate"] = float(target_clean.float().mean().item())
    out["lost_by_alpha_rate"] = float((candidate_clean & ~target_clean).float().mean().item())
    if teacher_probs is not None:
        teacher_pred = teacher_probs.detach().cpu().float().argmax(dim=1)
        out["teacher_gated_noisy_clean_acc"] = float((teacher_pred[gated_noisy_idx] == clean[gated_noisy_idx]).float().mean().item())
    return out


def _empty_group_label_diagnostics() -> dict:
    return {
        "clean_purity": math.nan,
        "noisy_label_entropy": math.nan,
        "effective_coverage": 0.0,
    }


def _method_grouping(
    method: str,
    gb_grouping: dict,
    gagb_grouping: dict,
    grace_gagb_grouping: dict,
    grace_gagb_emb_grouping: dict,
    grace_gagb_topo_grouping: dict,
    emb_local_grouping: dict,
    emb_global_grouping: dict,
    khop_grouping: dict,
    shuffled_size_grouping: dict,
    shuffled_alpha_grouping: dict,
    grace_gagb_shuffled_grouping: dict,
    dcgb_grouping: dict,
    dcgb_emb_grouping: dict,
    dcgb_topo_grouping: dict,
    dcgb_shuffled_grouping: dict,
) -> dict:
    method_key = method.lower()
    if (
        method_key in DCGB_METHODS
        or method_key in DCGB_MATCHED_METHODS
        or method_key in DCGB_QUALITY_METHODS
        or method_key.startswith("dcgb_qb_conflict_gate")
    ):
        return dcgb_grouping
    if method_key in DCGB_EMB_METHODS:
        return dcgb_emb_grouping
    if method_key in DCGB_TOPO_METHODS:
        return dcgb_topo_grouping
    if method_key in DCGB_SHUFFLED_METHODS:
        return dcgb_shuffled_grouping
    if (
        method_key in GRACE_GAGB_METHODS
        or method_key in GRACE_GAGB_MATCHED_METHODS
        or method_key in GRACE_GAGB_QUALITY_METHODS
        or method_key.startswith("grace_gagb_qb_conflict_gate")
    ):
        return grace_gagb_grouping
    if method_key in GRACE_GAGB_EMB_METHODS:
        return grace_gagb_emb_grouping
    if method_key in GRACE_GAGB_TOPO_METHODS:
        return grace_gagb_topo_grouping
    if method_key in GRACE_GAGB_SHUFFLED_METHODS:
        return grace_gagb_shuffled_grouping
    if method_key in EMB_LOCAL_METHODS:
        return emb_local_grouping
    if method_key in EMB_GLOBAL_METHODS:
        return emb_global_grouping
    if method_key in KHOP_METHODS or method_key in KHOP_QUALITY_METHODS or method_key in {"khop_only_alpha09", "khop_only_replace"}:
        return khop_grouping
    if method_key in SHUFFLED_SIZE_METHODS:
        return shuffled_size_grouping
    if method_key in SHUFFLED_ALPHA_METHODS:
        return shuffled_alpha_grouping
    if method_key in GRAPH_AWARE_METHODS or method_key in GAGB_QUALITY_METHODS or method_key in V2_TARGET_METHODS or method_key in GAGB_MATCHED_METHODS:
        return gagb_grouping
    return gb_grouping


def _method_alpha_values(method: str, requested_alphas: list[float]) -> list[float]:
    config = CONFLICT_GATE_REPAIR_CONFIGS.get(method.lower())
    if config is not None:
        return [float(config["alpha_max"])]
    return requested_alphas


def result_dir(args: argparse.Namespace) -> Path:
    rate_name = f"{args.noise_rate:g}".replace(".", "p")
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    return (
        PROJECT_ROOT
        / "results"
        / "gate_runs"
        / args.dataset
        / f"{args.noise_type}_{rate_name}"
        / f"{split_token}_pretrain{args.pretrain_seed}_noise{args.noise_seed}_clf{args.classifier_seed}"
    )


def run(args: argparse.Namespace) -> list[dict]:
    device = get_device(args.gpu_id)
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    artifact_path = args.embedding or embedding_path(
        args.dataset,
        args.split_seed,
        args.pretrain_seed,
        args.split_mode,
        args.train_ratio,
        args.val_ratio,
    )
    payload = torch.load(artifact_path, map_location="cpu")
    h = payload["H"].float()
    h_regions = F.normalize(h, p=2, dim=1)
    edge_index = payload.get("edge_index")
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    noisy_labels, noisy_path, noise_meta = load_or_create_noisy_labels(
        clean_labels,
        masks["train_mask"],
        num_classes,
        dataset=args.dataset,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        split_seed=args.split_seed,
        noise_seed=args.noise_seed,
        split_name=split_token,
    )
    gb_grouping = build_granular_balls(
        h_regions,
        min_ball_size=args.gb_min_ball_size,
        min_radius_gain=args.gb_min_radius_gain,
        seed=args.pretrain_seed,
    )
    if edge_index is None:
        _, data = load_dataset(
            args.dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index.detach().cpu().long()
    gagb_grouping = build_graph_aware_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=args.gagb_lambda_h,
        lambda_t=args.gagb_lambda_t,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
    )
    emb_only_local_grouping = build_graph_aware_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=1.0,
        lambda_t=0.0,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        candidate_scope="khop",
    )
    emb_only_local_grouping["diagnostics"]["grouping_kind"] = "emb_only_gagb_local"
    emb_only_local_grouping["diagnostics"]["graph_aware_lambda_h"] = 1.0
    emb_only_local_grouping["diagnostics"]["graph_aware_lambda_t"] = 0.0
    emb_only_global_grouping = build_graph_aware_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=1.0,
        lambda_t=0.0,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        candidate_scope="global",
    )
    emb_only_global_grouping["diagnostics"]["grouping_kind"] = "emb_only_gagb_global"
    emb_only_global_grouping["diagnostics"]["graph_aware_lambda_h"] = 1.0
    emb_only_global_grouping["diagnostics"]["graph_aware_lambda_t"] = 0.0
    khop_grouping = build_graph_aware_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=0.0,
        lambda_t=1.0,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
    )
    khop_grouping["diagnostics"]["grouping_kind"] = "khop_only"
    khop_grouping["diagnostics"]["graph_aware_lambda_h"] = 0.0
    khop_grouping["diagnostics"]["graph_aware_lambda_t"] = 1.0
    gagb_shuffled_size_grouping = shuffle_grouping(gagb_grouping, seed=args.noise_seed, embeddings=h_regions)
    gagb_shuffled_size_grouping["diagnostics"]["grouping_kind"] = "gagb_shuffled_size_only"
    gagb_shuffled_alpha_grouping = shuffle_grouping(gagb_grouping, seed=args.noise_seed, embeddings=h_regions)
    gagb_shuffled_alpha_grouping["diagnostics"]["grouping_kind"] = "gagb_shuffled_alpha_matched"
    grace_gagb_grouping = build_grace_first_graph_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="grace_first",
    )
    grace_gagb_emb_grouping = build_grace_first_graph_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="emb_only",
    )
    grace_gagb_emb_grouping["diagnostics"]["grouping_kind"] = "grace_gagb_emb_only"
    grace_gagb_topo_grouping = build_grace_first_graph_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="topo_only",
    )
    grace_gagb_topo_grouping["diagnostics"]["grouping_kind"] = "grace_gagb_topo_only"
    grace_gagb_shuffled_grouping = shuffle_grouping(grace_gagb_grouping, seed=args.noise_seed, embeddings=h_regions)
    grace_gagb_shuffled_grouping["diagnostics"]["grouping_kind"] = "grace_gagb_shuffled_alpha_matched"
    dcgb_grouping = build_dcgb_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="dcgb",
    )
    dcgb_emb_grouping = build_dcgb_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="emb_only",
    )
    dcgb_emb_grouping["diagnostics"]["grouping_kind"] = "dcgb_emb_only"
    dcgb_topo_grouping = build_dcgb_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="topo_only",
    )
    dcgb_topo_grouping["diagnostics"]["grouping_kind"] = "dcgb_topo_only"
    dcgb_shuffled_grouping = shuffle_grouping(dcgb_grouping, seed=args.noise_seed, embeddings=h_regions)
    dcgb_shuffled_grouping["diagnostics"]["grouping_kind"] = "dcgb_shuffled_alpha_matched"

    rows = []
    methods = args.methods or DEFAULT_METHODS
    teacher_probs = None
    teacher_stats = {}
    if any(method.lower() in V2_TARGET_METHODS for method in methods):
        teacher_probs, teacher_stats = train_noisy_linear_teacher(
            h_regions,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            seed=args.noise_seed,
            epochs=args.teacher_epochs,
            learning_rate=args.teacher_lr,
            weight_decay=args.teacher_weight_decay,
            temperature=args.teacher_temperature,
            device=device,
        )
    for method in methods:
        method_alphas = [0.0] if method == "ce" or method in WEIGHTED_METHODS else _method_alpha_values(method, args.alpha_max)
        for alpha in method_alphas:
            sample_weights = None
            grouping = _method_grouping(
                method,
                gb_grouping,
                gagb_grouping,
                grace_gagb_grouping,
                grace_gagb_emb_grouping,
                grace_gagb_topo_grouping,
                emb_only_local_grouping,
                emb_only_global_grouping,
                khop_grouping,
                gagb_shuffled_size_grouping,
                gagb_shuffled_alpha_grouping,
                grace_gagb_shuffled_grouping,
                dcgb_grouping,
                dcgb_emb_grouping,
                dcgb_topo_grouping,
                dcgb_shuffled_grouping,
            )
            if method in WEIGHTED_METHODS:
                soft_targets = one_hot(noisy_labels, num_classes)
                smoothing_stats = {
                    "effective_smoothing_coverage": 0.0,
                    "effective_train_nodes_smoothed": 0,
                    "mean_alpha_train": 0.0,
                    "max_alpha_train": 0.0,
                    "alpha_mean_all_train": 0.0,
                    "alpha_mean_nonzero": 0.0,
                    "alpha_max_actual": 0.0,
                    "alpha_max": 0.0,
                    "alpha_std_train": 0.0,
                    "alpha_q25_train": 0.0,
                    "alpha_q50_train": 0.0,
                    "alpha_q75_train": 0.0,
                }
                sample_weights, weight_stats = build_reliability_weights(
                    method,
                    h_regions,
                    noisy_labels,
                    masks["train_mask"],
                    num_classes,
                    gb_grouping=grouping,
                    min_train_per_region=args.min_train_per_ball,
                    raw_weight_min=args.raw_weight_min,
                    final_weight_low=args.final_weight_low,
                    final_weight_high=args.final_weight_high,
                    eps=args.eps,
                    seed=args.noise_seed,
                )
                smoothing_stats.update(weight_stats)
            else:
                alpha_override = None
                alpha_match_reference = None
                if method == "gagb_shuffled_alpha_matched":
                    _, alpha_match_reference = build_method_targets(
                        "gagb",
                        h_regions,
                        noisy_labels,
                        masks["train_mask"],
                        num_classes,
                        alpha_max=float(alpha),
                        gb_grouping=gagb_grouping,
                        min_train_per_ball=args.min_train_per_ball,
                        eps=args.eps,
                        seed=args.noise_seed,
                        prior_gamma=args.prior_gamma,
                    )
                    alpha_override = alpha_match_reference["node_alpha"]
                if method == "grace_gagb_shuffled_alpha_matched":
                    _, alpha_match_reference = build_method_targets(
                        "grace_gagb",
                        h_regions,
                        noisy_labels,
                        masks["train_mask"],
                        num_classes,
                        alpha_max=float(alpha),
                        gb_grouping=grace_gagb_grouping,
                        min_train_per_ball=args.min_train_per_ball,
                        eps=args.eps,
                        seed=args.noise_seed,
                        prior_gamma=args.prior_gamma,
                    )
                    alpha_override = alpha_match_reference["node_alpha"]
                if method == "dcgb_shuffled_alpha_matched":
                    _, alpha_match_reference = build_method_targets(
                        "dcgb",
                        h_regions,
                        noisy_labels,
                        masks["train_mask"],
                        num_classes,
                        alpha_max=float(alpha),
                        gb_grouping=dcgb_grouping,
                        min_train_per_ball=args.min_train_per_ball,
                        eps=args.eps,
                        seed=args.noise_seed,
                        prior_gamma=args.prior_gamma,
                    )
                    alpha_override = alpha_match_reference["node_alpha"]
                soft_targets, smoothing_stats = build_method_targets(
                    method,
                    h_regions,
                    noisy_labels,
                    masks["train_mask"],
                    num_classes,
                    alpha_max=float(alpha),
                    gb_grouping=grouping,
                    min_train_per_ball=args.min_train_per_ball,
                    eps=args.eps,
                    seed=args.noise_seed,
                    prior_gamma=args.prior_gamma,
                    teacher_probs=teacher_probs,
                    qb_pred_beta=args.qb_pred_beta,
                    qb_teacher_gamma=args.qb_teacher_gamma,
                    conflict_noisy_support_max=args.conflict_noisy_support_max,
                    conflict_target_conf_min=args.conflict_target_conf_min,
                    alpha_override=alpha_override,
                    alpha_match_reference=alpha_match_reference,
                    edge_index=edge_index,
                )
                if method.lower() in V2_TARGET_METHODS:
                    smoothing_stats.update(teacher_stats)
            if method == "ce":
                smoothing_stats.update(_empty_group_label_diagnostics())
            else:
                smoothing_stats.update(_group_label_diagnostics(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes))
            smoothing_stats.update(
                _target_noise_diagnostics(
                    soft_targets,
                    sample_weights,
                    clean_labels,
                    noisy_labels,
                    masks["train_mask"],
                )
            )
            smoothing_stats.update(
                _target_repair_diagnostics(
                    soft_targets,
                    smoothing_stats,
                    clean_labels,
                    noisy_labels,
                    masks["train_mask"],
                    teacher_probs,
                )
            )
            row_smoothing_stats = dict(smoothing_stats)
            row_smoothing_stats.pop("node_alpha", None)
            row_smoothing_stats.pop("target_changed_mask", None)
            row_smoothing_stats.pop("target_l1_delta", None)
            row_smoothing_stats.pop("target_update_gate_mask", None)
            row_smoothing_stats.pop("target_update_candidate_argmax", None)
            eval_stats = train_frozen_classifier(
                h,
                clean_labels,
                soft_targets,
                masks["train_mask"],
                masks["val_mask"],
                masks["test_mask"],
                seed=args.classifier_seed,
                epochs=args.classifier_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                device=device,
                sample_weights=sample_weights,
            )
            row = {
                "dataset": args.dataset,
                "method": method,
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "split_mode": args.split_mode,
                "split_name": split_token,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "split_seed": int(args.split_seed),
                "pretrain_seed": int(args.pretrain_seed),
                "noise_seed": int(args.noise_seed),
                "classifier_seed": int(args.classifier_seed),
                "primary_alpha": bool(float(alpha) == float(args.primary_alpha)),
                "embedding_path": str(artifact_path),
                "corrupted_label_path": str(noisy_path),
                **noise_meta,
                **row_smoothing_stats,
                **eval_stats,
            }
            rows.append(row)
            print(
                "(R) | "
                f"dataset={args.dataset}, method={method}, alpha={alpha}, "
                f"val={row['best_val_acc']:.4f}, test={row['test_acc_at_best_val']:.4f}, "
                f"coverage={row.get('effective_smoothing_coverage', 0.0):.4f}"
            )

    out_dir = result_dir(args)
    save_csv(out_dir / "result_rows.csv", rows)
    save_json(out_dir / "result_rows.json", rows)
    save_json(out_dir / "gb_diagnostics.json", gb_grouping)
    save_json(out_dir / "gagb_diagnostics.json", gagb_grouping)
    save_json(out_dir / "emb_only_gagb_local_diagnostics.json", emb_only_local_grouping)
    save_json(out_dir / "emb_only_gagb_global_diagnostics.json", emb_only_global_grouping)
    save_json(out_dir / "khop_diagnostics.json", khop_grouping)
    save_json(out_dir / "gagb_shuffled_size_only_diagnostics.json", gagb_shuffled_size_grouping)
    save_json(out_dir / "gagb_shuffled_alpha_matched_diagnostics.json", gagb_shuffled_alpha_grouping)
    save_json(out_dir / "grace_gagb_diagnostics.json", grace_gagb_grouping)
    save_json(out_dir / "grace_gagb_emb_only_diagnostics.json", grace_gagb_emb_grouping)
    save_json(out_dir / "grace_gagb_topo_only_diagnostics.json", grace_gagb_topo_grouping)
    save_json(out_dir / "grace_gagb_shuffled_alpha_matched_diagnostics.json", grace_gagb_shuffled_grouping)
    save_json(out_dir / "dcgb_diagnostics.json", dcgb_grouping)
    save_json(out_dir / "dcgb_emb_only_diagnostics.json", dcgb_emb_grouping)
    save_json(out_dir / "dcgb_topo_only_diagnostics.json", dcgb_topo_grouping)
    save_json(out_dir / "dcgb_shuffled_alpha_matched_diagnostics.json", dcgb_shuffled_grouping)
    save_json(out_dir / "run_config.json", vars(args))
    print(f"(S) | Saved noisy classifier results to {out_dir}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer"], default="Cora")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["planetoid", "ratio"], default="planetoid")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--pretrain-seed", type=int, default=39788)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--alpha-max", "--alphas", dest="alpha_max", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--primary-alpha", type=float, default=0.3)
    parser.add_argument("--min-train-per-ball", type=int, default=3)
    parser.add_argument("--gb-min-ball-size", type=int, default=5)
    parser.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    parser.add_argument("--gagb-max-hops", type=int, default=2)
    parser.add_argument("--gagb-tau-add", type=float, default=0.55)
    parser.add_argument("--gagb-lambda-h", type=float, default=0.7)
    parser.add_argument("--gagb-lambda-t", type=float, default=0.3)
    parser.add_argument("--gagb-max-ball-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--prior-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--qb-pred-beta", type=float, default=0.5)
    parser.add_argument("--qb-teacher-gamma", type=float, default=2.0)
    parser.add_argument("--conflict-noisy-support-max", type=float, default=0.35)
    parser.add_argument("--conflict-target-conf-min", type=float, default=0.55)
    parser.add_argument("--raw-weight-min", type=float, default=0.3)
    parser.add_argument("--final-weight-low", type=float, default=0.5)
    parser.add_argument("--final-weight-high", type=float, default=1.5)
    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "ce",
            "global",
            "kmeans",
            "topk",
            "topk_matched",
            "gb",
            "gb_prior",
            "gb_ca_ce",
            "gb_ca_global",
            "gagb",
            "gagb_ca_ce",
            "emb_only_gagb_local",
            "emb_only_gagb_global",
            "gagb_weighted_ce",
            "khop_only",
            "gagb_shuffled",
            "gagb_shuffled_size_only",
            "gagb_shuffled_alpha_matched",
            "gagb_shuffled_alpha09_matched",
            "gagb_shuffled_replace_matched",
            "kmeans_gagb_matched",
            "topk_gagb_matched",
            "topk_gagb_train_matched",
            "topk_gagb_allnode_matched",
            "kmeans_gagb_matched_alpha09",
            "kmeans_gagb_matched_replace",
            "topk_gagb_train_matched_alpha09",
            "topk_gagb_train_matched_replace",
            "khop_only_alpha09",
            "khop_only_replace",
            "gagb_qb_pred_mix",
            "gagb_qb_teacher_prior",
            "gagb_qb_conflict_gate",
            "gagb_qb_conflict_gate_alpha03",
            "gagb_qb_conflict_gate_alpha06",
            "gagb_qb_conflict_gate_alpha07",
            "gagb_qb_conflict_gate_alpha08",
            "gagb_qb_conflict_gate_alpha09",
            "gagb_qb_conflict_gate_replace_qmix",
            "gagb_class_capped_alpha09",
            "gagb_class_balanced_alpha09",
            "gagb_boundary_safe_alpha09",
            "gagb_replay_random_same_count",
            "gagb_qb_conflict_gate_sharpen_qmix_T05",
            "gagb_qb_conflict_gate_margin_replace",
            "grace_gagb",
            "grace_gagb_emb_only",
            "grace_gagb_topo_only",
            "kmeans_grace_gagb_matched",
            "topk_grace_gagb_train_matched",
            "grace_gagb_shuffled_alpha_matched",
            "grace_gagb_qb_conflict_gate_alpha03",
            "grace_gagb_qb_conflict_gate_alpha07",
            "grace_gagb_qb_conflict_gate_alpha09",
            "grace_gagb_qfilter_alpha07_q50",
            "grace_gagb_qfilter_alpha07_q70",
            "grace_gagb_qweight_alpha07",
            "grace_gagb_qsafe_boundary_alpha09_q50",
            "grace_gagb_qsafe_boundary_alpha09_q70",
            "grace_gagb_qsafe_boundary_alpha07_q50",
            "grace_gagb_qsafe_class_capped_alpha09_q50",
            "grace_gagb_qsafe_class_balanced_alpha09_q50",
            "gagb_qsafe_boundary_alpha09_q50",
            "khop_only_qsafe_boundary_alpha09_q50",
            "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
            "gagb_qfilter_alpha07_q50",
            "khop_only_qfilter_alpha07_q50",
            "grace_gagb_shuffled_qfilter_alpha07_q50",
            "dcgb",
            "dcgb_emb_only",
            "dcgb_topo_only",
            "kmeans_dcgb_matched",
            "topk_dcgb_train_matched",
            "dcgb_shuffled_alpha_matched",
            "dcgb_qsafe_boundary_alpha09_q50",
            "gb_weighted_ce",
            "kmeans_weighted_ce",
            "topk_matched_weighted_ce",
        ],
        default=None,
    )
    parser.add_argument("--embedding", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
