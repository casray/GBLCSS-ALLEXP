from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_grace_first_gagb_grouping import build_groupings, embedding_path, summarize_rows
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets, compute_group_quality_scores, train_noisy_linear_teacher
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, save_csv, save_json
from run_gate_experiment import METHOD_CHOICES
from run_noisy_classifier import (
    V2_TARGET_METHODS,
    _empty_group_label_diagnostics,
    _group_label_diagnostics,
    _method_grouping,
    _target_noise_diagnostics,
    _target_repair_diagnostics,
)


DEFAULT_METHODS = [
    "grace_gagb_qb_conflict_gate_alpha07",
    "grace_gagb_qfilter_alpha07_q50",
    "grace_gagb_qfilter_alpha07_q70",
    "grace_gagb_qweight_alpha07",
    "gagb_qfilter_alpha07_q50",
    "khop_only_qfilter_alpha07_q50",
    "grace_gagb_shuffled_qfilter_alpha07_q50",
]


def _method_alpha(method: str, default_alpha: float) -> float:
    from noisy_label_gb_grace.smoothing.targets import CONFLICT_GATE_REPAIR_CONFIGS, REPAIR_CONTROL_CONFIGS

    key = method.lower()
    config = CONFLICT_GATE_REPAIR_CONFIGS.get(key)
    if config is None and key in REPAIR_CONTROL_CONFIGS:
        config = CONFLICT_GATE_REPAIR_CONFIGS[REPAIR_CONTROL_CONFIGS[key]["base_method"]]
    return float(config["alpha_max"]) if config is not None else float(default_alpha)


def _normalised_entropy(probs: torch.Tensor) -> float:
    if probs.numel() <= 1 or float(probs.sum().item()) <= 0.0:
        return 0.0
    probs = probs / probs.sum().clamp_min(1e-12)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    return entropy / math.log(float(probs.numel()))


def _ball_label_quality(
    grouping: dict,
    selected: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    prefix: str,
) -> dict:
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    total = 0
    majority = 0
    entropies = []
    for group_id, group in enumerate(grouping["groups"]):
        if group_id >= selected.numel() or not bool(selected[group_id].item()):
            continue
        idx = torch.tensor(group, dtype=torch.long)
        train_nodes = idx[train_mask[idx]]
        if train_nodes.numel() == 0:
            continue
        total += int(train_nodes.numel())
        clean_counts = torch.bincount(clean[train_nodes], minlength=num_classes).float()
        noisy_counts = torch.bincount(noisy[train_nodes], minlength=num_classes).float()
        majority += int(clean_counts.max().item())
        entropies.append(_normalised_entropy(noisy_counts))
    return {
        f"{prefix}_ball_train_count": int(total),
        f"{prefix}_ball_clean_purity": float(majority / total) if total else math.nan,
        f"{prefix}_ball_noisy_entropy": float(sum(entropies) / len(entropies)) if entropies else math.nan,
    }


def _quality_selection_diagnostics(
    method: str,
    h_norm: torch.Tensor,
    edge_index: torch.Tensor,
    noisy_labels: torch.Tensor,
    clean_labels: torch.Tensor,
    train_mask: torch.Tensor,
    grouping: dict,
    num_classes: int,
    min_train_per_ball: int,
    stats: dict,
) -> dict:
    if "quality_threshold" not in stats:
        return {}
    quality = compute_group_quality_scores(
        h_norm,
        edge_index,
        noisy_labels,
        train_mask,
        grouping,
        num_classes,
        min_train_per_ball=min_train_per_ball,
    )
    q = quality["group_quality"]
    train_count = quality["group_train_count"]
    eligible = train_count >= int(min_train_per_ball)
    if method.endswith("qweight_alpha07"):
        selected = eligible & (q > 0)
    else:
        threshold = float(stats["quality_threshold"])
        selected = eligible & (q >= threshold)
    rejected = eligible & ~selected
    return {
        **_ball_label_quality(grouping, selected, clean_labels, noisy_labels, train_mask, num_classes, "quality_selected"),
        **_ball_label_quality(grouping, rejected, clean_labels, noisy_labels, train_mask, num_classes, "quality_rejected"),
    }


def _sanitize_stats(stats: dict) -> dict:
    drop_keys = {
        "node_alpha",
        "target_changed_mask",
        "target_l1_delta",
        "target_update_gate_mask",
        "target_update_candidate_argmax",
        "target_update_selected_mask",
    }
    return {key: value for key, value in stats.items() if key not in drop_keys}


def diagnose_case(args: argparse.Namespace, dataset: str, train_ratio: float, noise_rate: float, seed: int) -> list[dict]:
    artifact = embedding_path(dataset, args.split_seed, seed, args.split_mode, train_ratio, args.val_ratio)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {artifact}")
    payload = torch.load(artifact, map_location="cpu")
    h = payload["H"].float()
    h_norm = F.normalize(h, p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index.detach().cpu().long()
    split_token = split_name(args.split_mode, args.split_seed, train_ratio, args.val_ratio)
    noisy_labels, _, noise_meta = load_or_create_noisy_labels(
        clean_labels,
        masks["train_mask"],
        num_classes,
        dataset=dataset,
        noise_type=args.noise_type,
        noise_rate=noise_rate,
        split_seed=args.split_seed,
        noise_seed=seed,
        split_name=split_token,
    )
    args.pretrain_seed = seed
    args.noise_seed = seed
    groupings = build_groupings(h_norm, edge_index, args)
    methods = args.methods or DEFAULT_METHODS
    teacher_probs = None
    teacher_stats = {}
    if any(method.lower() in V2_TARGET_METHODS for method in methods):
        teacher_probs, teacher_stats = train_noisy_linear_teacher(
            h_norm,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            seed=seed,
            epochs=args.teacher_epochs,
            learning_rate=args.teacher_lr,
            weight_decay=args.teacher_weight_decay,
            temperature=args.teacher_temperature,
            device=get_device(args.gpu_id),
        )

    rows = []
    for method in methods:
        alpha = 0.0 if method == "ce" else _method_alpha(method, args.alpha_max)
        grouping = _method_grouping(
            method,
            groupings["gb"],
            groupings["gagb"],
            groupings["grace_gagb"],
            groupings["grace_gagb_emb"],
            groupings["grace_gagb_topo"],
            groupings["emb_local"],
            groupings["emb_global"],
            groupings["khop"],
            groupings["shuffled_size"],
            groupings["shuffled_alpha"],
            groupings["grace_gagb_shuffled"],
        )
        soft_targets, stats = build_method_targets(
            method,
            h_norm,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            alpha_max=float(alpha),
            gb_grouping=grouping,
            edge_index=edge_index,
            min_train_per_ball=args.min_train_per_ball,
            eps=args.eps,
            seed=seed,
            teacher_probs=teacher_probs,
            qb_pred_beta=args.qb_pred_beta,
            qb_teacher_gamma=args.qb_teacher_gamma,
            conflict_noisy_support_max=args.conflict_noisy_support_max,
            conflict_target_conf_min=args.conflict_target_conf_min,
        )
        if method.lower() in V2_TARGET_METHODS:
            stats.update(teacher_stats)
        if method == "ce":
            stats.update(_empty_group_label_diagnostics())
        else:
            stats.update(_group_label_diagnostics(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes))
        stats.update(_target_noise_diagnostics(soft_targets, None, clean_labels, noisy_labels, masks["train_mask"]))
        stats.update(_target_repair_diagnostics(soft_targets, stats, clean_labels, noisy_labels, masks["train_mask"], teacher_probs))
        stats.update(
            _quality_selection_diagnostics(
                method,
                h_norm,
                edge_index,
                noisy_labels,
                clean_labels,
                masks["train_mask"],
                grouping,
                num_classes,
                args.min_train_per_ball,
                stats,
            )
        )
        row = {
            "dataset": dataset,
            "split_mode": args.split_mode,
            "split_name": split_token,
            "train_ratio": train_ratio,
            "val_ratio": args.val_ratio,
            "noise_type": args.noise_type,
            "noise_rate": noise_rate,
            "split_seed": args.split_seed,
            "pretrain_seed": seed,
            "noise_seed": seed,
            "seed": seed,
            "method": method,
            "alpha_max": float(alpha),
            **noise_meta,
            **_sanitize_stats(stats),
        }
        rows.append(row)
        print(
            "(Q) | "
            f"dataset={dataset}, noise={noise_rate:g}, seed={seed}, method={method}, "
            f"ncr={row.get('noise_correction_rate', math.nan):.4f}, cdr={row.get('clean_damage_rate', math.nan):.4f}, "
            f"qsel={row.get('quality_selected_ball_count', math.nan)}"
        )
    return rows


def run(args: argparse.Namespace) -> None:
    rows = []
    for dataset in args.datasets:
        for train_ratio in args.train_ratios:
            for noise_rate in args.noise_rates:
                for seed in args.seeds:
                    rows.extend(diagnose_case(args, dataset, train_ratio, noise_rate, seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    save_csv(args.output_dir / "per_seed_quality_diagnostics.csv", rows)
    save_csv(args.output_dir / "quality_filter_summary.csv", summary_rows)
    save_json(args.output_dir / "run_config.json", vars(args))
    report = [
        "# GAGB v2.6 Q(B) Quality Filtering Diagnostic",
        "",
        f"Rows: {len(rows)}",
        f"Summary rows: {len(summary_rows)}",
        "",
        "This script does not train a classifier. It rebuilds groupings, trains the noisy-label-only teacher when needed, and uses build_method_targets for all target construction.",
        "",
        "Clean labels are used only after target construction for diagnostics such as NCR, CDR, selected-ball clean purity, and rejected-ball clean purity.",
        "",
        "Primary methods: " + ", ".join(args.methods or DEFAULT_METHODS),
    ]
    (args.output_dir / "quality_filter_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"(S) | Wrote quality diagnostics to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["planetoid", "ratio"], default="ratio")
    parser.add_argument("--train-ratios", nargs="+", type=float, default=[0.2])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rates", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=DEFAULT_METHODS)
    parser.add_argument("--alpha-max", type=float, default=0.3)
    parser.add_argument("--min-train-per-ball", type=int, default=3)
    parser.add_argument("--gb-min-ball-size", type=int, default=5)
    parser.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    parser.add_argument("--gagb-max-hops", type=int, default=2)
    parser.add_argument("--gagb-tau-add", type=float, default=0.55)
    parser.add_argument("--gagb-lambda-h", type=float, default=0.7)
    parser.add_argument("--gagb-lambda-t", type=float, default=0.3)
    parser.add_argument("--gagb-max-ball-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--qb-pred-beta", type=float, default=0.5)
    parser.add_argument("--qb-teacher-gamma", type=float, default=2.0)
    parser.add_argument("--conflict-noisy-support-max", type=float, default=0.35)
    parser.add_argument("--conflict-target-conf-min", type=float, default=0.55)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "gagb_v2_6_quality_filter_diag_20260604")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
