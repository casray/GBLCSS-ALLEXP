from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls import build_granular_balls
from noisy_label_gb_grace.data import split_name
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.utils import PROJECT_ROOT, mask_to_index, save_csv, save_json
from noisy_label_gb_grace.weighting import build_reliability_weights
from scripts.extract_embeddings import embedding_path


METHODS = ["gb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"]


def result_dir(args: argparse.Namespace) -> Path:
    rate_name = f"{args.noise_rate:g}".replace(".", "p")
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    return (
        PROJECT_ROOT
        / "results"
        / "weight_preflight"
        / args.dataset
        / f"{args.noise_type}_{rate_name}"
        / f"{split_token}_pretrain{args.pretrain_seed}_noise{args.noise_seed}"
    )


def _corr(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    left = left.float()
    right = right.float()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 0.0:
        return 1.0
    return float((left_centered @ right_centered / denom).item())


def _diagnose_clean_noisy(
    weights: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict:
    train_mask = train_mask.detach().cpu().bool()
    clean_labels = clean_labels.detach().cpu().long()
    noisy_labels = noisy_labels.detach().cpu().long()
    weights = weights.detach().cpu().float()
    train_idx = mask_to_index(train_mask)
    flipped = train_mask & (clean_labels != noisy_labels)
    clean = train_mask & ~flipped
    low_weight = train_mask & (weights < 1.0)
    return {
        "mean_weight_clean": float(weights[clean].mean().item()) if clean.any() else 0.0,
        "mean_weight_noisy": float(weights[flipped].mean().item()) if flipped.any() else 0.0,
        "num_train_nodes": int(train_idx.numel()),
        "num_flipped_train_nodes": int(flipped.sum().item()),
        "downweight_precision": float((low_weight & flipped).sum().item() / max(1, int(low_weight.sum().item()))),
        "downweight_recall": float((low_weight & flipped).sum().item() / max(1, int(flipped.sum().item()))),
    }


def _gate_summary(rows: list[dict], weights_by_method: dict[str, torch.Tensor], train_mask: torch.Tensor) -> dict:
    by_method = {row["method"]: row for row in rows}
    gb = by_method.get("gb_weighted_ce", {})
    train_idx = mask_to_index(train_mask.detach().cpu().bool())
    gb_weights = weights_by_method.get("gb_weighted_ce", torch.empty(0))[train_idx]
    kmeans_weights = weights_by_method.get("kmeans_weighted_ce", torch.empty(0))[train_idx]
    topk_weights = weights_by_method.get("topk_matched_weighted_ce", torch.empty(0))[train_idx]
    gb_kmeans_corr = _corr(gb_weights, kmeans_weights)
    gb_topk_corr = _corr(gb_weights, topk_weights)
    clean_gap = float(gb.get("mean_weight_clean", 0.0)) - float(gb.get("mean_weight_noisy", 0.0))
    clip_total = float(gb.get("clip_low_ratio", 0.0)) + float(gb.get("clip_high_ratio", 0.0))
    passed = (
        float(gb.get("weight_std", 0.0)) > 0.03
        and clip_total < 0.5
        and float(gb.get("mean_weight_noisy", 0.0)) < float(gb.get("mean_weight_clean", 0.0))
        and clean_gap >= 0.02
        and gb_kmeans_corr < 0.95
        and gb_topk_corr < 0.95
    )
    return {
        "preflight_pass": bool(passed),
        "gb_kmeans_weight_corr": gb_kmeans_corr,
        "gb_topk_weight_corr": gb_topk_corr,
        "gb_clean_noisy_weight_gap": clean_gap,
        "gb_clip_total_ratio": clip_total,
        "required_weight_std_gt": 0.03,
        "required_clip_total_lt": 0.5,
        "required_clean_noisy_gap_ge": 0.02,
        "required_weight_corr_lt": 0.95,
    }


def run(args: argparse.Namespace) -> list[dict]:
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
    weights_by_method = {}
    for method in METHODS:
        weights, stats = build_reliability_weights(
            method,
            h_regions,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            gb_grouping=gb_grouping,
            min_train_per_region=args.min_train_per_region,
            raw_weight_min=args.raw_weight_min,
            final_weight_low=args.final_weight_low,
            final_weight_high=args.final_weight_high,
            eps=args.eps,
            seed=args.noise_seed,
        )
        weights_by_method[method] = weights
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
            "embedding_path": str(artifact_path),
            "corrupted_label_path": str(noisy_path),
            **noise_meta,
            **stats,
            **_diagnose_clean_noisy(weights, clean_labels, noisy_labels, masks["train_mask"]),
        }
        rows.append(row)
    summary = {
        **_gate_summary(rows, weights_by_method, masks["train_mask"]),
        "dataset": args.dataset,
        "split_name": split_token,
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
    }
    out_dir = result_dir(args)
    save_csv(out_dir / "weight_rows.csv", rows)
    save_json(out_dir / "weight_summary.json", summary)
    save_json(out_dir / "run_config.json", vars(args))
    print(f"(S) | Saved weight preflight results to {out_dir}")
    print(f"(G) | preflight_pass={summary['preflight_pass']}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer"], default="Cora")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["ratio"], default="ratio")
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rate", type=float, default=0.4)
    parser.add_argument("--min-train-per-region", type=int, default=3)
    parser.add_argument("--raw-weight-min", type=float, default=0.3)
    parser.add_argument("--final-weight-low", type=float, default=0.5)
    parser.add_argument("--final-weight-high", type=float, default=1.5)
    parser.add_argument("--gb-min-ball-size", type=int, default=5)
    parser.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--embedding", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
