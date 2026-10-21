from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.balls import build_granular_balls
from noisy_label_gb_grace.data import split_name
from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets
from noisy_label_gb_grace.utils import one_hot
from noisy_label_gb_grace.weighting import build_reliability_weights
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, save_csv, save_json
from scripts.extract_embeddings import embedding_path


WEIGHTED_METHODS = {"gb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"}
DEFAULT_METHODS = [
    "ce",
    "gb_weighted_ce",
    "kmeans_weighted_ce",
    "topk_matched_weighted_ce",
    "global",
    "kmeans",
    "topk_matched",
    "gb",
    "gb_prior",
]


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

    rows = []
    methods = args.methods or DEFAULT_METHODS
    for method in methods:
        method_alphas = [0.0] if method == "ce" or method in WEIGHTED_METHODS else args.alpha_max
        for alpha in method_alphas:
            sample_weights = None
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
                }
                sample_weights, weight_stats = build_reliability_weights(
                    method,
                    h_regions,
                    noisy_labels,
                    masks["train_mask"],
                    num_classes,
                    gb_grouping=gb_grouping,
                    min_train_per_region=args.min_train_per_ball,
                    raw_weight_min=args.raw_weight_min,
                    final_weight_low=args.final_weight_low,
                    final_weight_high=args.final_weight_high,
                    eps=args.eps,
                    seed=args.noise_seed,
                )
                smoothing_stats.update(weight_stats)
            else:
                soft_targets, smoothing_stats = build_method_targets(
                    method,
                    h_regions,
                    noisy_labels,
                    masks["train_mask"],
                    num_classes,
                    alpha_max=float(alpha),
                    gb_grouping=gb_grouping,
                    min_train_per_ball=args.min_train_per_ball,
                    eps=args.eps,
                    seed=args.noise_seed,
                    prior_gamma=args.prior_gamma,
                )
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
                **smoothing_stats,
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
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--prior-gamma", type=float, default=1.0)
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
