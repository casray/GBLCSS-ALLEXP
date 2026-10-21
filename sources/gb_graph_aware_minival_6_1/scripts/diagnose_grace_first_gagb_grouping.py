from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from noisy_label_gb_grace.balls import build_dcgb_balls, build_grace_first_graph_balls, build_granular_balls, build_graph_aware_balls, shuffle_grouping
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets, train_noisy_linear_teacher
from noisy_label_gb_grace.smoothing.targets import CONFLICT_GATE_REPAIR_CONFIGS, REPAIR_CONTROL_CONFIGS
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
    "gb",
    "gagb",
    "khop_only",
    "grace_gagb",
    "grace_gagb_emb_only",
    "grace_gagb_topo_only",
    "kmeans_grace_gagb_matched",
    "topk_grace_gagb_train_matched",
    "grace_gagb_shuffled_alpha_matched",
    "grace_gagb_qb_conflict_gate_alpha03",
    "grace_gagb_qb_conflict_gate_alpha07",
    "dcgb",
    "dcgb_emb_only",
    "dcgb_topo_only",
    "kmeans_dcgb_matched",
    "topk_dcgb_train_matched",
    "dcgb_shuffled_alpha_matched",
    "dcgb_qsafe_boundary_alpha09_q50",
]


def embedding_path(dataset: str, split_seed: int, pretrain_seed: int, split_mode: str, train_ratio: float | None, val_ratio: float | None) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "embeddings" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "H.pt"


def _method_alpha(method: str, default_alpha: float) -> float:
    key = method.lower()
    config = CONFLICT_GATE_REPAIR_CONFIGS.get(key)
    if config is None and key in REPAIR_CONTROL_CONFIGS:
        config = CONFLICT_GATE_REPAIR_CONFIGS[REPAIR_CONTROL_CONFIGS[key]["base_method"]]
    return float(config["alpha_max"]) if config is not None else float(default_alpha)


def _sanitize_stats(stats: dict) -> dict:
    drop_keys = {
        "node_alpha",
        "target_changed_mask",
        "target_l1_delta",
        "target_update_gate_mask",
        "target_update_candidate_argmax",
    }
    return {key: value for key, value in stats.items() if key not in drop_keys}


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not finite:
        return math.nan, math.nan
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return float(mean), float(math.sqrt(var))


def summarize_rows(rows: list[dict]) -> list[dict]:
    numeric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"seed", "split_seed", "pretrain_seed", "noise_seed"}
    })
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["split_mode"],
            row["split_name"],
            row["train_ratio"],
            row["val_ratio"],
            row["noise_type"],
            row["noise_rate"],
            row["method"],
            row["alpha_max"],
        )
        grouped[key].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        summary = {
            "dataset": key[0],
            "split_mode": key[1],
            "split_name": key[2],
            "train_ratio": key[3],
            "val_ratio": key[4],
            "noise_type": key[5],
            "noise_rate": key[6],
            "method": key[7],
            "alpha_max": key[8],
            "n": len(group),
        }
        for field in numeric_keys:
            values = [row.get(field) for row in group if row.get(field) not in (None, "")]
            mean, std = _mean_std(values) if values else (math.nan, math.nan)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
        summaries.append(summary)
    return summaries


def build_groupings(h_norm: torch.Tensor, edge_index: torch.Tensor, args: argparse.Namespace) -> dict[str, dict]:
    gb_grouping = build_granular_balls(
        h_norm,
        min_ball_size=args.gb_min_ball_size,
        min_radius_gain=args.gb_min_radius_gain,
        seed=args.pretrain_seed,
    )
    gagb_grouping = build_graph_aware_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=args.gagb_lambda_h,
        lambda_t=args.gagb_lambda_t,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
    )
    emb_local = build_graph_aware_balls(
        h_norm,
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
    emb_global = build_graph_aware_balls(
        h_norm,
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
    khop = build_graph_aware_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=0.0,
        lambda_t=1.0,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
    )
    shuffled_size = shuffle_grouping(gagb_grouping, seed=args.noise_seed, embeddings=h_norm)
    shuffled_alpha = shuffle_grouping(gagb_grouping, seed=args.noise_seed, embeddings=h_norm)
    grace_gagb = build_grace_first_graph_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="grace_first",
    )
    grace_emb = build_grace_first_graph_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="emb_only",
    )
    grace_topo = build_grace_first_graph_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="topo_only",
    )
    grace_shuffled = shuffle_grouping(grace_gagb, seed=args.noise_seed, embeddings=h_norm)
    dcgb = build_dcgb_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="dcgb",
    )
    dcgb_emb = build_dcgb_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="emb_only",
    )
    dcgb_emb["diagnostics"]["grouping_kind"] = "dcgb_emb_only"
    dcgb_topo = build_dcgb_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=args.pretrain_seed,
        mode="topo_only",
    )
    dcgb_topo["diagnostics"]["grouping_kind"] = "dcgb_topo_only"
    dcgb_shuffled = shuffle_grouping(dcgb, seed=args.noise_seed, embeddings=h_norm)
    dcgb_shuffled["diagnostics"]["grouping_kind"] = "dcgb_shuffled_alpha_matched"
    return {
        "gb": gb_grouping,
        "gagb": gagb_grouping,
        "grace_gagb": grace_gagb,
        "grace_gagb_emb": grace_emb,
        "grace_gagb_topo": grace_topo,
        "emb_local": emb_local,
        "emb_global": emb_global,
        "khop": khop,
        "shuffled_size": shuffled_size,
        "shuffled_alpha": shuffled_alpha,
        "grace_gagb_shuffled": grace_shuffled,
        "dcgb": dcgb,
        "dcgb_emb": dcgb_emb,
        "dcgb_topo": dcgb_topo,
        "dcgb_shuffled": dcgb_shuffled,
    }


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
            groupings["dcgb"],
            groupings["dcgb_emb"],
            groupings["dcgb_topo"],
            groupings["dcgb_shuffled"],
        )
        alpha_override = None
        alpha_match_reference = None
        if method == "grace_gagb_shuffled_alpha_matched":
            _, alpha_match_reference = build_method_targets(
                "grace_gagb",
                h_norm,
                noisy_labels,
                masks["train_mask"],
                num_classes,
                alpha_max=float(alpha),
                gb_grouping=groupings["grace_gagb"],
                min_train_per_ball=args.min_train_per_ball,
                eps=args.eps,
                seed=seed,
            )
            alpha_override = alpha_match_reference["node_alpha"]
        if method == "dcgb_shuffled_alpha_matched":
            _, alpha_match_reference = build_method_targets(
                "dcgb",
                h_norm,
                noisy_labels,
                masks["train_mask"],
                num_classes,
                alpha_max=float(alpha),
                gb_grouping=groupings["dcgb"],
                min_train_per_ball=args.min_train_per_ball,
                eps=args.eps,
                seed=seed,
            )
            alpha_override = alpha_match_reference["node_alpha"]
        soft_targets, stats = build_method_targets(
            method,
            h_norm,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            alpha_max=float(alpha),
            gb_grouping=grouping,
            min_train_per_ball=args.min_train_per_ball,
            eps=args.eps,
            seed=seed,
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
            stats.update(teacher_stats)
        if method == "ce":
            stats.update(_empty_group_label_diagnostics())
        else:
            stats.update(_group_label_diagnostics(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes))
        stats.update(_target_noise_diagnostics(soft_targets, None, clean_labels, noisy_labels, masks["train_mask"]))
        stats.update(_target_repair_diagnostics(soft_targets, stats, clean_labels, noisy_labels, masks["train_mask"], teacher_probs))
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
            "(G) | "
            f"dataset={dataset}, noise={noise_rate:g}, seed={seed}, method={method}, "
            f"purity={row.get('clean_purity', math.nan):.4f}, auroc={row.get('noise_detection_auroc', math.nan):.4f}"
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
    save_csv(args.output_dir / "per_seed_grouping_diagnostics.csv", rows)
    save_csv(args.output_dir / "grouping_summary.csv", summary_rows)
    save_json(args.output_dir / "run_config.json", vars(args))
    report = [
        "# GRACE-first GAGB Grouping Diagnostic",
        "",
        f"Rows: {len(rows)}",
        f"Summary rows: {len(summary_rows)}",
        "",
        "This script does not train a classifier. It rebuilds groupings and uses build_method_targets for target diagnostics.",
    ]
    (args.output_dir / "grouping_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"(S) | Wrote grouping diagnostics to {args.output_dir}")


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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "gagb_v2_5_grace_first_grouping_diag_20260604")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
